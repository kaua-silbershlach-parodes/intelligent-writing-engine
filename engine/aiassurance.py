"""AI assurance gate.

Twenty-four gates must hold in order before an AI system's decision
is treated as assured: an approved mission and scope, corroborated
evidence distinguished from assumption, bounded autonomy under a
trusted authority, reviewed memory and current knowledge, tenant
isolation, redacted privacy, reviewed fairness and safety,
attributable transparency, a redacted telemetry trail, assigned
response and ownership, documented consumers, bounded volume, an
honest financial field, reviewed regulatory context, a documented
rollback, and independent verification before closure.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

Decision = namedtuple("Decision", ["allowed", "failed_gate"])

GATE_NAMES = (
    "mission", "scope", "evidence", "confidence", "autonomy", "authority",
    "memory", "knowledge", "recovery", "fanout", "tenant", "privacy",
    "fairness", "safety", "transparency", "telemetry", "response",
    "consumers", "volume", "financial", "regulatory", "rollback",
    "owner", "verification",
)
ALLOWED_VALUES = (
    "approved", "approved_fixture", "corroborated", "verified",
    "approved_read_only", "trusted_application", "reviewed_write",
    "current_version", "state_rechecked", "bounded", "same_tenant",
    "redacted", "reviewed", "reviewed", "attributable", "redacted_trace",
    "assigned", "documented", "bounded", "unknown_not_invented",
    "reviewed", "documented", "assigned", "independently_tested",
)
_BAD_VALUES = {
    "mission": "unapproved", "scope": "out_of_scope",
    "evidence": "unverified", "confidence": "assumed_fact",
    "autonomy": "unapproved_action", "authority": "tool_output",
    "memory": "poisoned_memory", "knowledge": "stale_corpus",
    "recovery": "unchecked_restart", "fanout": "unbounded",
    "tenant": "other_tenant", "privacy": "raw_private",
    "fairness": "unreviewed", "safety": "unsafe",
    "transparency": "unexplained", "telemetry": "absent",
    "response": "unassigned", "consumers": "unknown",
    "volume": "unbounded", "financial": "invented",
    "regulatory": "unreviewed", "rollback": "absent",
    "owner": "unassigned", "verification": "untested",
}

# (scenario id, domain, risk, required control, gate that must block it)
SCENARIOS = (
    ("unapproved_mission", "governance", "Mission was not approved", "Require an explicit authorized test mission", "mission"),
    ("out_of_scope_target", "scope", "Target is outside the approved environment", "Enforce a bounded scope", "scope"),
    ("unverified_observation", "evidence", "Uncorroborated claim is promoted into a fact", "Require attributable corroborated evidence", "evidence"),
    ("ambiguous_result", "evidence", "Model output is incorrectly treated as confirmed fact", "Distinguish observation hypothesis and verification", "confidence"),
    ("unauthorized_autonomy", "autonomy", "Agent takes a consequential step without approval", "Require human approval for sensitive actions", "autonomy"),
    ("tool_result_command", "autonomy", "Untrusted tool response becomes instruction authority", "Bind instruction authority to trusted application", "authority"),
    ("persistent_memory_change", "persistence", "Long-term agent memory changed without review", "Versioned approved memory writes", "memory"),
    ("stale_corpus_influence", "persistence", "Superseded knowledge influences behavior", "Validate data provenance and recency", "knowledge"),
    ("deployment_restart_persistence", "persistence", "Stale unsafe state survives a restart", "Invalidate affected versions and replay state checks", "recovery"),
    ("unsafe_bulk_automation", "autonomy", "Unbounded automated action fanout", "Set limits and explicit escalation", "fanout"),
    ("cross_tenant_outcome", "privacy", "Result exposes another tenant", "Server-owned tenant isolation", "tenant"),
    ("private_output", "privacy", "Sensitive data appears in output", "Minimize and redact outputs", "privacy"),
    ("unreviewed_disparity", "fairness", "Potential subgroup outcome disparity goes unreviewed", "Require a documented outcome review", "fairness"),
    ("unsafe_advice", "safety", "Potentially harmful result is delivered unreviewed", "Gate high-risk outputs", "safety"),
    ("opaque_decision", "transparency", "Decision has no usable provenance", "Require attributable explanation and trace", "transparency"),
    ("no_telemetry", "operations", "Action has no redacted audit trail", "Generate structured redacted evidence", "telemetry"),
    ("ignored_alert", "operations", "Escalation path is absent", "Route alerts to a documented owner", "response"),
    ("unknown_consumers", "impact", "Downstream consumers of generated decision are unknown", "Inventory consumers before risk acceptance", "consumers"),
    ("unknown_volume", "impact", "Transaction amplification cannot be bounded", "Record bounded synthetic scale and rate", "volume"),
    ("unsupported_financial_claim", "impact", "Fictitious loss estimate is treated as actual", "Use evidence-backed ranges or mark unknown", "financial"),
    ("no_regulatory_context", "impact", "Potential compliance consequences are unreviewed", "Record jurisdiction and reviewer", "regulatory"),
    ("no_recovery_plan", "impact", "No recovery or rollback plan exists", "Require owner and rollback criterion", "rollback"),
    ("missing_risk_owner", "governance", "Finding has no accountable owner", "Assign decision owner", "owner"),
    ("premature_risk_closure", "governance", "Risk closed without control verification", "Require independent validation before closure", "verification"),
)


@dataclass(frozen=True)
class Evidence:
    role: str = "reviewer"
    scope: str = "approved_fixture"
    session: str = "same_session"
    offline: bool = True
    gates: tuple = ()  # ((name, value), ...) in GATE_NAMES order


def baseline_fixture() -> Evidence:
    return Evidence(gates=tuple(zip(GATE_NAMES, ALLOWED_VALUES)))


def bad_fixture(gate: str) -> Evidence:
    """The baseline with exactly one gate broken."""
    if gate == "no_live_access":
        return Evidence(offline=False, gates=baseline_fixture().gates)
    gates = tuple(
        (name, _BAD_VALUES.get(name, value) if name == gate else value)
        for name, value in baseline_fixture().gates
    )
    return Evidence(gates=gates)


def check(evidence: Evidence) -> Decision:
    if evidence.role != "reviewer":
        return Decision(False, "role")
    if evidence.scope != "approved_fixture":
        return Decision(False, "scope")
    if evidence.session != "same_session":
        return Decision(False, "session")
    if not evidence.offline:
        return Decision(False, "no_live_access")
    if len(evidence.gates) != len(GATE_NAMES):
        return Decision(False, "evidence_shape")
    for (name, value), want_name, want_value in zip(
            evidence.gates, GATE_NAMES, ALLOWED_VALUES):
        if name != want_name or value != want_value:
            return Decision(False, want_name)
    return Decision(True, "none")


def simulate(scenario_id: str) -> Decision:
    for row in SCENARIOS:
        if row[0] == scenario_id:
            return check(bad_fixture(row[4]))
    return Decision(False, "unknown")


def decide(role: str, scope: str, evidence: str, approval: str,
           telemetry: str) -> Decision:
    """A single decision is previewable only on corroborated evidence."""
    if (role != "reviewer" or scope != "approved_fixture"
            or evidence != "corroborated" or approval != "approved_read_only"
            or telemetry != "redacted_trace"):
        return Decision(False, "decide")
    return Decision(True, "none")


def impact(role: str, scope: str, volume: str, consumer: str,
           financial: str) -> Decision:
    """An impact claim is reviewable only bounded and documented."""
    if (role != "reviewer" or scope != "approved_fixture"
            or volume != "bounded" or consumer != "documented"
            or financial != "unknown_not_invented"):
        return Decision(False, "impact")
    return Decision(True, "none")
