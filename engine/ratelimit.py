"""Rate limiting and retry discipline.

Three mechanisms, all bounded and fail-closed:

- A per-client token bucket isolates one abusive client from the
  rest; the clock is injectable so behavior is deterministic in tests.
- A bounded exponential backoff with jitter governs client retries:
  the delay doubles per attempt, never exceeds the cap, and the
  jitter fraction is bounded.
- A priority gate protects the control class: under overload, batch
  traffic is shed first, interactive traffic is degraded, and control
  traffic is always served.
"""
from __future__ import annotations

import random
import time
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Callable

MAX_COST = 1_000
MAX_CAPACITY_RPS = 1_000_000
MAX_BACKOFF_S = 3_600.0

PRIORITY = {"control": 0, "interactive": 1, "batch": 2}
DEFAULT_PRIORITY = "batch"  # unknown classes fall to the lowest priority
CONTROL_TOOL = "monitoring.health"  # the control plane must survive overload

Verdict = namedtuple("Verdict", ["verdict", "reason"])


class RateLimitError(Exception):
    """Fail-closed rejection of an invalid bucket, policy, or request."""


def _now_default() -> float:
    return time.monotonic()


@dataclass
class TokenBucket:
    """Deterministic per-client token bucket."""

    capacity: int
    refill_per_sec: float
    now: Callable[[], float] = field(default=_now_default)
    _tokens: float = field(init=False)
    _last: float = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.capacity, int) or self.capacity <= 0:
            raise RateLimitError("bucket_capacity_rejected")
        if not isinstance(self.refill_per_sec, (int, float)) or not (
                0 < self.refill_per_sec <= MAX_CAPACITY_RPS):
            raise RateLimitError("bucket_refill_rejected")
        self._tokens = float(self.capacity)
        self._last = self.now()

    def take(self, tokens: int = 1) -> bool:
        if not isinstance(tokens, int) or not (0 < tokens <= MAX_COST):
            raise RateLimitError("bucket_take_rejected")
        now = self.now()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(float(self.capacity),
                           self._tokens + elapsed * self.refill_per_sec)
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff with jitter."""

    base_s: float
    max_s: float
    jitter: float  # 0.0..1.0 fraction of the computed delay

    def __post_init__(self) -> None:
        if not (0 < self.base_s <= MAX_BACKOFF_S):
            raise RateLimitError("retry_base_rejected")
        if not (self.base_s <= self.max_s <= MAX_BACKOFF_S):
            raise RateLimitError("retry_max_rejected")
        if not (0.0 <= self.jitter <= 1.0):
            raise RateLimitError("retry_jitter_rejected")

    def delay(self, attempt: int, rng: random.Random | None = None) -> float:
        """Delay before retry number `attempt` (attempt >= 1)."""
        if not isinstance(attempt, int) or attempt < 1:
            raise RateLimitError("retry_attempt_rejected")
        raw = min(self.max_s, self.base_s * (2.0 ** (attempt - 1)))
        if self.jitter == 0.0:
            return raw
        source = rng if rng is not None else random
        spread = raw * self.jitter
        return max(0.0, raw - spread + source.random() * 2.0 * spread)


class PriorityGate:
    """Priority-aware admission over a one-second capacity window."""

    def __init__(self, capacity_rps: int, per_client_rps: int,
                 now: Callable[[], float] | None = None) -> None:
        if not isinstance(capacity_rps, int) or not (
                0 < capacity_rps <= MAX_CAPACITY_RPS):
            raise RateLimitError("capacity_rejected")
        if not isinstance(per_client_rps, int) or not (
                0 < per_client_rps <= MAX_CAPACITY_RPS):
            raise RateLimitError("per_client_capacity_rejected")
        self.capacity_rps = capacity_rps
        self.per_client_rps = per_client_rps
        self._now = now or _now_default
        self._buckets: dict[str, TokenBucket] = {}
        self._window_start = self._now()
        self._window_used = 0
        self.counters = {"allow": 0, "degrade": 0, "shed": 0}

    def _bucket(self, client: str) -> TokenBucket:
        bucket = self._buckets.get(client)
        if bucket is None:
            bucket = TokenBucket(self.per_client_rps,
                                 float(self.per_client_rps), now=self._now)
            self._buckets[client] = bucket
        return bucket

    def _overloaded(self) -> bool:
        now = self._now()
        if now - self._window_start >= 1.0:
            self._window_start = now
            self._window_used = 0
        return self._window_used >= self.capacity_rps

    def decide(self, client: str, cost: int = 1, cls: str = DEFAULT_PRIORITY,
               tool: str = "") -> Verdict:
        if cls not in PRIORITY:
            cls = DEFAULT_PRIORITY  # fail closed to the lowest priority
        if not isinstance(cost, int) or not (0 < cost <= MAX_COST):
            raise RateLimitError("event_cost_rejected")
        verdict, reason = "allow", "within_capacity"
        if tool == CONTROL_TOOL or cls == "control":
            verdict, reason = "allow", "control_plane_protected"
        elif not self._bucket(client).take(cost):
            verdict, reason = "shed", "per_client_rate_exceeded"
        elif self._overloaded():
            if cls == "batch":
                verdict, reason = "shed", "overload_shed_low_priority"
            else:
                verdict, reason = "degrade", "overload_graceful_degradation"
        if verdict == "allow":
            self._window_used += cost
        self.counters[verdict] += 1
        return Verdict(verdict, reason)
