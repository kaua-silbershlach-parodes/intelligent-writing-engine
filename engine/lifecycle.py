"""Assessment lifecycle gate.

An assessment advances through fixed phases, and every transition
requires its own evidence. The gate checks, in order: the caller's
role, the tenant binding, the approval, the telemetry, that the
transition exists, and that the evidence matches the transition.
"""
from __future__ import annotations

from collections import namedtuple

Decision = namedtuple("Decision", ["allowed", "failed_gate"])

# (current phase, next phase, evidence the transition requires)
TRANSITIONS = (
    ("scoping", "threat_modeling", "scope_verified"),
    ("threat_modeling", "adversarial_preview", "threats_reviewed"),
    ("adversarial_preview", "analysis_reporting", "tests_reproduced"),
    ("analysis_reporting", "remediation", "findings_correlated"),
    ("remediation", "independent_retest", "fix_reviewed"),
    ("independent_retest", "closed", "retest_passed"),
)

# (scenario id, gate that must block it)
SCENARIOS = (
    ("skipped_scope", "transition"),
    ("unverified_test", "evidence"),
    ("premature_closure", "transition"),
    ("missing_approval", "approval"),
    ("untrusted_reviewer", "role"),
    ("cross_tenant_assessment", "tenant"),
    ("unreviewed_report", "evidence"),
    ("missing_telemetry", "telemetry"),
)

# The violating call for each scenario, as (role, tenant, current, next,
# evidence, approval, telemetry).
_SCENARIO_CALLS = {
    "skipped_scope": ("reviewer", "same_tenant", "scoping", "analysis_reporting",
                      "scope_verified", "approved_read_only", "redacted_trace"),
    "unverified_test": ("reviewer", "same_tenant", "adversarial_preview",
                        "analysis_reporting", "unverified",
                        "approved_read_only", "redacted_trace"),
    "premature_closure": ("reviewer", "same_tenant", "analysis_reporting",
                          "closed", "findings_correlated",
                          "approved_read_only", "redacted_trace"),
    "missing_approval": ("reviewer", "same_tenant", "scoping", "threat_modeling",
                         "scope_verified", "missing", "redacted_trace"),
    "untrusted_reviewer": ("viewer", "same_tenant", "scoping", "threat_modeling",
                           "scope_verified", "approved_read_only", "redacted_trace"),
    "cross_tenant_assessment": ("reviewer", "other_tenant", "scoping",
                                "threat_modeling", "scope_verified",
                                "approved_read_only", "redacted_trace"),
    "unreviewed_report": ("reviewer", "same_tenant", "analysis_reporting",
                          "remediation", "unreviewed",
                          "approved_read_only", "redacted_trace"),
    "missing_telemetry": ("reviewer", "same_tenant", "scoping", "threat_modeling",
                          "scope_verified", "approved_read_only", "missing"),
}


def check(role: str, tenant: str, current: str, next_phase: str,
          evidence: str, approval: str, telemetry: str) -> Decision:
    """Decide whether a phase transition may advance."""
    if role != "reviewer":
        return Decision(False, "role")
    if tenant != "same_tenant":
        return Decision(False, "tenant")
    if approval != "approved_read_only":
        return Decision(False, "approval")
    if telemetry != "redacted_trace":
        return Decision(False, "telemetry")
    required = None
    for cur, nxt, req in TRANSITIONS:
        if cur == current and nxt == next_phase:
            required = req
            break
    if required is None:
        return Decision(False, "transition")
    if evidence != required:
        return Decision(False, "evidence")
    return Decision(True, "none")


def simulate(scenario_id: str) -> Decision:
    """Run a negative scenario; it must be blocked at its named gate."""
    call = _SCENARIO_CALLS.get(scenario_id)
    if call is None:
        return Decision(False, "unknown")
    return check(*call)


def baseline() -> Decision:
    """The one clean advance: scoping -> threat_modeling, fully evidenced."""
    return check("reviewer", "same_tenant", "scoping", "threat_modeling",
                 "scope_verified", "approved_read_only", "redacted_trace")
