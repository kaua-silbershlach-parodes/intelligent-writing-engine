"""Change security gate.

A change is reviewable only when twelve gates hold in order: trusted
origin, corroborated evidence, a reviewed diff, passing tests,
redacted secrets, documented consumers, bounded volume, reviewed
regulatory context, an honest financial field (unknown, never
invented), human approval, a documented rollback, and a read-only
preview action. Any gate out of place fails the change at that gate.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field

Decision = namedtuple("Decision", ["allowed", "failed_gate"])

GATE_NAMES = (
    "origin", "evidence", "patch_review", "test_results",
    "secrets", "consumers", "transaction_volume", "regulatory",
    "financial", "human_approval", "rollback", "action",
)
SAFE_VALUES = (
    "trusted_application", "corroborated", "reviewed_diff", "passed",
    "redacted", "documented", "bounded", "reviewed",
    "unknown_not_invented", "approved", "documented", "read_only_preview",
)

# (scenario id, domain, gate that must block it)
SCENARIOS = (
    ("untrusted_change_instruction", "agent_input", "origin"),
    ("unverified_patch_justification", "evidence", "evidence"),
    ("unreviewed_code_diff", "change_review", "patch_review"),
    ("missing_test_results", "verification", "test_results"),
    ("private_data_in_patch", "privacy", "secrets"),
    ("unknown_downstream_consumers", "impact", "consumers"),
    ("unbounded_action_fanout", "impact", "transaction_volume"),
    ("regulatory_context_missing", "risk", "regulatory"),
    ("unsupported_loss_estimate", "risk", "financial"),
    ("autonomous_commit_without_approval", "authority", "human_approval"),
    ("no_backout_plan", "recovery", "rollback"),
    ("unapproved_remote_push", "action", "action"),
)


@dataclass(frozen=True)
class Evidence:
    role: str = "reviewer"
    scope: str = "approved_fixture"
    offline: bool = True
    gates: tuple = ()  # ((name, value), ...) in GATE_NAMES order


def baseline_fixture() -> Evidence:
    return Evidence(gates=tuple(zip(GATE_NAMES, SAFE_VALUES)))


def bad_fixture(gate: str) -> Evidence:
    """The baseline with exactly one gate broken."""
    if gate == "offline":
        return Evidence(offline=False, gates=baseline_fixture().gates)
    gates = tuple(
        (name, "not_verified" if name == gate else value)
        for name, value in baseline_fixture().gates
    )
    return Evidence(gates=gates)


def check(evidence: Evidence) -> Decision:
    if evidence.role != "reviewer":
        return Decision(False, "role")
    if evidence.scope != "approved_fixture":
        return Decision(False, "scope")
    if not evidence.offline:
        return Decision(False, "offline")
    if len(evidence.gates) != len(GATE_NAMES):
        return Decision(False, "shape")
    for (name, value), want_name, want_value in zip(
            evidence.gates, GATE_NAMES, SAFE_VALUES):
        if name != want_name or value != want_value:
            return Decision(False, want_name)
    return Decision(True, "none")


def simulate(scenario_id: str) -> Decision:
    for sid, _domain, gate in SCENARIOS:
        if sid == scenario_id:
            return check(bad_fixture(gate))
    return Decision(False, "unknown")


def impact(role: str, decisions: str, consumers: str, volume: str,
           financial: str, regulatory: str, recovery: str) -> Decision:
    """An impact statement is reviewable only fully documented."""
    if (role != "reviewer" or decisions != "documented"
            or consumers != "documented" or volume != "bounded"
            or financial != "unknown_not_invented" or regulatory != "reviewed"
            or recovery != "documented"):
        return Decision(False, "impact")
    return Decision(True, "none")
