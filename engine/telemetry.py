"""Bounded telemetry: a capped in-memory ring of search events.

Events are operational evidence (mode, result count, latency), never
query text. The ring is bounded: old events fall off at CAP entries.
"""

import time
from collections import deque

CAP = 256
_EVENTS = deque(maxlen=CAP)


def record(mode, results_count, latency_ms):
    _EVENTS.append({
        "mode": mode,
        "results": results_count,
        "latency_ms": round(latency_ms, 3),
        "at": time.time(),
    })


def events():
    return list(_EVENTS)


def clear():
    _EVENTS.clear()


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.start) * 1000.0
        return False
