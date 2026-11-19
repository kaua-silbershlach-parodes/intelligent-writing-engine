"""Disaster risk assessment matrix — gate template.

A standardized matrix that ranks each risk facing an organization by
its probability of occurrence within a year multiplied by its
potential impact to the organization.

Standard scales (exact values):
  probability: almost never 0.0, unlikely 0.2, somewhat unlikely 0.4,
               likely 0.6, highly likely 0.8, inevitable 1.0
  impact:      negligible 0.0, minimal 0.2, moderate 0.5,
               severe 0.8, critical 1.0

Ranking = probability x impact. Ranked values order the risks from
highest to lowest as a guide for prioritization and preparation; a
risk ranking 0.8 needs more immediate attention than risks ranking
0.5 or 0.3, and response plans must be developed for the most
critical risks.

This module turns the matrix into a release/change gate template:
a risk register is assessed, and the gate fails closed when any risk
is unassessed or when a risk at/above the plan threshold has no
response plan. The bundled sample register carries the standard
themes and risks as an unassessed template — values must be tailored
per organization, because they depend on what the organization does,
its infrastructure, and where it is located.

Everything is local and synthetic: registers are caller-supplied JSON
processed in memory. No network, no production system is touched.
This is a rehearsal/planning harness, not a compliance certification.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Exact standard scales.
PROBABILITY = {
    "almost never": 0.0,
    "unlikely": 0.2,
    "somewhat unlikely": 0.4,
    "likely": 0.6,
    "highly likely": 0.8,
    "inevitable": 1.0,
}
IMPACT = {
    "negligible": 0.0,
    "minimal": 0.2,
    "moderate": 0.5,
    "severe": 0.8,
    "critical": 1.0,
}

MAX_RISKS = 10_000
MAX_TEXT = 500
MAX_SYSTEMS = 100

# Guidance: risks at/above this ranking need a response plan
# (contrast 0.8 vs 0.5 vs 0.3). Configurable per organization.
DEFAULT_PLAN_THRESHOLD = 0.5


class MatrixError(Exception):
    """Fail-closed rejection of an invalid register, label, or value."""


def _norm_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def _resolve(value: Any, scale: dict[str, float], name: str) -> float:
    if isinstance(value, str):
        key = _norm_label(value)
        if key in scale:
            return scale[key]
        raise MatrixError(
            f"unknown {name} label {value!r}; "
            f"expected one of {sorted(scale)} or an exact scale value"
        )
    if isinstance(value, bool):
        raise MatrixError(f"{name} must be a label or a scale value, got bool")
    if isinstance(value, (int, float)):
        for v in scale.values():
            if abs(float(value) - v) < 1e-12:
                return v
        raise MatrixError(
            f"{name} value {value!r} is not on the standard scale "
            f"{sorted(scale.values())}"
        )
    raise MatrixError(f"{name} must be a label or a scale value, got {type(value).__name__}")


def resolve_probability(value: Any) -> float:
    """Map a probability label (or exact scale value) to its number."""
    return _resolve(value, PROBABILITY, "probability")


def resolve_impact(value: Any) -> float:
    """Map an impact label (or exact scale value) to its number."""
    return _resolve(value, IMPACT, "impact")


def _bounded_text(value: Any, name: str, required: bool = True) -> str | None:
    if value is None:
        if required:
            raise MatrixError(f"{name} is required")
        return None
    if not isinstance(value, str) or not value.strip():
        raise MatrixError(f"{name} must be a non-empty string")
    text = value.strip()
    if len(text) > MAX_TEXT:
        raise MatrixError(f"{name} exceeds {MAX_TEXT} characters")
    return text


@dataclass
class RiskAssessment:
    """One row of the disaster risk assessment matrix."""

    theme: str
    risk: str
    probability: Any = None  # label, exact scale value, or None (unassessed)
    impact: Any = None       # label, exact scale value, or None (unassessed)
    systems: tuple[str, ...] = ()
    response_plan: str | None = None

    def rank(self) -> float | None:
        """probability x impact; None when the risk is not yet assessed."""
        if self.probability is None or self.impact is None:
            return None
        return round(resolve_probability(self.probability)
                     * resolve_impact(self.impact), 2)


def parse_register(obj: Any) -> list[RiskAssessment]:
    """Validate a caller-supplied register dict into assessed rows.

    Expected shape: {"risks": [{"theme", "risk", "probability"?,
    "impact"?, "systems"?, "response_plan"?}, ...]}. Fail-closed on any
    malformed input.
    """
    if not isinstance(obj, dict):
        raise MatrixError("register must be a JSON object")
    risks = obj.get("risks")
    if not isinstance(risks, list):
        raise MatrixError('register must contain a "risks" list')
    if len(risks) > MAX_RISKS:
        raise MatrixError(f"register exceeds {MAX_RISKS} risks")
    out: list[RiskAssessment] = []
    for i, raw in enumerate(risks):
        if not isinstance(raw, dict):
            raise MatrixError(f"risk #{i} must be an object")
        systems: tuple[str, ...] = ()
        if "systems" in raw and raw["systems"] is not None:
            if not isinstance(raw["systems"], list):
                raise MatrixError(f"risk #{i}: systems must be a list")
            if len(raw["systems"]) > MAX_SYSTEMS:
                raise MatrixError(f"risk #{i}: too many systems")
            systems = tuple(
                _bounded_text(s, f"risk #{i} system") for s in raw["systems"]  # type: ignore[misc]
            )
        out.append(RiskAssessment(
            theme=_bounded_text(raw.get("theme"), f"risk #{i} theme"),  # type: ignore[arg-type]
            risk=_bounded_text(raw.get("risk"), f"risk #{i} risk"),  # type: ignore[arg-type]
            probability=raw.get("probability"),
            impact=raw.get("impact"),
            systems=systems,
            response_plan=_bounded_text(raw.get("response_plan"),
                                        f"risk #{i} response_plan",
                                        required=False),
        ))
    # Eager validation: unknown labels / off-scale values fail here.
    for entry in out:
        if entry.probability is not None:
            resolve_probability(entry.probability)
        if entry.impact is not None:
            resolve_impact(entry.impact)
    return out


@dataclass
class AssessmentResult:
    ranked: list[tuple[RiskAssessment, float]] = field(default_factory=list)
    unassessed: list[RiskAssessment] = field(default_factory=list)


def assess(register: list[RiskAssessment]) -> AssessmentResult:
    """Rank every assessed risk highest -> lowest; collect unassessed rows."""
    result = AssessmentResult()
    for entry in register:
        rank = entry.rank()
        if rank is None:
            result.unassessed.append(entry)
        else:
            result.ranked.append((entry, rank))
    result.ranked.sort(key=lambda pair: pair[1], reverse=True)
    return result


@dataclass
class GateResult:
    passed: bool
    blockers: list[tuple[RiskAssessment, str]] = field(default_factory=list)


def gate(register: list[RiskAssessment],
         plan_threshold: float = DEFAULT_PLAN_THRESHOLD) -> GateResult:
    """Fail-closed gate: every risk assessed, critical risks need a plan.

    A risk blocks the gate when it is unassessed, or when its ranking is
    at/above plan_threshold and no response plan is recorded — the
    "develop response plans for the most critical risks" rule, enforced
    as a template gate.
    """
    if not isinstance(plan_threshold, (int, float)) or isinstance(plan_threshold, bool):
        raise MatrixError("plan_threshold must be a number")
    if not 0.0 <= float(plan_threshold) <= 1.0:
        raise MatrixError("plan_threshold must be within [0.0, 1.0]")
    result = GateResult(passed=True)
    for entry in register:
        rank = entry.rank()
        if rank is None:
            result.blockers.append((entry, "unassessed"))
        elif rank >= float(plan_threshold) and not entry.response_plan:
            result.blockers.append((entry, f"no response plan (rank {rank})"))
    result.passed = not result.blockers
    return result


# Standard sample themes/risks, shipped unassessed: the values depend
# on the organization, so the template forces tailoring.
SAMPLE_REGISTER: list[dict[str, Any]] = [
    {"theme": "Environmental", "risk": name}
    for name in ("Earthquake", "Flood", "Fire", "Hurricane")
] + [
    {"theme": "Infrastructure reliability", "risk": name}
    for name in ("Power outage",
                 "Loss of internet connectivity",
                 "Authentication system down",
                 "High system latency/infrastructure slowdown")
] + [
    {"theme": "Security", "risk": name}
    for name in ("System compromise",
                 "Insider theft of intellectual property",
                 "DDoS/DoS attack",
                 "Misuse of system resources—e.g., cryptocurrency mining",
                 "Vandalism/website defacement",
                 "Phishing attack",
                 "Software security bug",
                 "Hardware security bug",
                 "Emerging serious vulnerability, e.g., Meltdown/Spectre, Heartbleed")
]


def sample_register_json() -> str:
    """The template register as JSON, ready to tailor per organization."""
    return json.dumps({"risks": SAMPLE_REGISTER}, indent=2, ensure_ascii=False)


def render_table(result: AssessmentResult) -> str:
    """Fixed-width ranked table in the matrix's columns."""
    lines = [
        f"{'Rank':>6}  {'Prob':>5}  {'Impact':>6}  Theme / Risk",
        "-" * 72,
    ]
    for entry, rank in result.ranked:
        prob = resolve_probability(entry.probability)
        impact = resolve_impact(entry.impact)
        plan = " [plan]" if entry.response_plan else " [NO PLAN]"
        lines.append(
            f"{rank:>6.2f}  {prob:>5.1f}  {impact:>6.1f}  "
            f"{entry.theme} — {entry.risk}{plan}"
        )
    if result.unassessed:
        lines.append("-" * 72)
        lines.append(f"UNASSESSED ({len(result.unassessed)}):")
        for entry in result.unassessed:
            lines.append(f"   ---   ---    ---   {entry.theme} — {entry.risk}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Disaster risk assessment matrix gate.")
    parser.add_argument("--register", type=Path,
                        help="JSON risk register file to assess")
    parser.add_argument("--plan-threshold", type=float,
                        default=DEFAULT_PLAN_THRESHOLD,
                        help="ranking at/above which a response plan is "
                             "required (default %(default)s)")
    parser.add_argument("--sample", action="store_true",
                        help="print the template register as JSON")
    parser.add_argument("--json", action="store_true",
                        help="emit the assessment as JSON instead of a table")
    args = parser.parse_args(argv)

    if args.sample:
        print(sample_register_json())
        return 0
    if args.register is None:
        parser.error("--register is required unless --sample is given")
    try:
        raw = json.loads(args.register.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read register: {exc}", file=sys.stderr)
        return 1
    try:
        register = parse_register(raw)
        assessed = assess(register)
        gated = gate(register, args.plan_threshold)
    except MatrixError as exc:
        print(f"error: invalid register: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            "ranked": [
                {"theme": e.theme, "risk": e.risk, "rank": r,
                 "has_plan": bool(e.response_plan),
                 "systems": list(e.systems)}
                for e, r in assessed.ranked
            ],
            "unassessed": [
                {"theme": e.theme, "risk": e.risk} for e in assessed.unassessed
            ],
            "gate_passed": gated.passed,
            "blockers": [
                {"theme": e.theme, "risk": e.risk, "reason": reason}
                for e, reason in gated.blockers
            ],
        }, indent=2, ensure_ascii=False))
    else:
        print(render_table(assessed))
        print("-" * 72)
        if gated.passed:
            print("GATE: PASS — all risks assessed, critical risks have plans.")
        else:
            print(f"GATE: BLOCKED — {len(gated.blockers)} blocker(s):")
            for entry, reason in gated.blockers:
                print(f"  - {entry.theme} — {entry.risk}: {reason}")
    return 0 if gated.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
