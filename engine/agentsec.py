"""Agent action authorization.

Authorization is a structured decision, never inferred from output
text. Untrusted content — documents, web pages, tool output, shared
memory — is data: it can be read as a fixture, but it can never
authorize an action. Sensitive egress (revealing secrets, external
egress) is disabled outright; memory writes require review; tool
execution and private-memory reads require a sufficient role and an
approval, inside the same session.
"""
from __future__ import annotations

from collections import namedtuple

PolicyDecision = namedtuple("PolicyDecision", ["allowed", "reason"])

SOURCES = frozenset({
    "trusted_application", "user_input", "document", "document_batch",
    "web_page", "source_code", "tool_output", "shared_memory",
    "private_memory",
})
ACTIONS = frozenset({
    "read_fixture", "execute_tool", "write_shared_memory",
    "read_private_memory", "reveal_secret", "external_egress",
})
ROLES = frozenset({"viewer", "operator", "reviewer"})
SESSION_SCOPES = frozenset({"same_session", "cross_session"})
APPROVALS = frozenset({"none", "approved"})

# (scenario id, entry point, required control)
SCENARIOS = (
    ("agent_surface_inventory", "discovery", "tool_registry_and_contract_review"),
    ("direct_instruction_override", "user_input", "trusted_policy_precedence"),
    ("system_configuration_disclosure", "user_input", "no_sensitive_egress"),
    ("output_filter_bypass", "user_input", "data_classification_at_sink"),
    ("troubleshooting_pretext", "user_input", "per_action_authorization"),
    ("objective_reframing", "user_input", "per_action_authorization"),
    ("multi_turn_escalation", "conversation", "authorization_each_step"),
    ("multi_document_fragmentation", "document_batch", "aggregate_context_untrusted"),
    ("hidden_web_content", "web_page", "untrusted_extracted_content"),
    ("source_code_unicode_tricks", "source_code", "canonicalization_and_integrity_review"),
    ("tool_response_poisoning", "tool_output", "tool_data_never_authorizes"),
    ("shared_knowledge_poisoning", "shared_memory", "reviewed_provenance_and_write_gate"),
    ("predictable_session_id", "session", "opaque_unpredictable_session_tokens"),
    ("cross_session_memory_access", "session", "isolation_and_scope_binding"),
)


def authorize(source: str, action: str, role: str, session_scope: str,
              approval: str) -> PolicyDecision:
    """Decide whether `source` may perform `action` under this context."""
    if (source not in SOURCES or action not in ACTIONS or role not in ROLES
            or session_scope not in SESSION_SCOPES or approval not in APPROVALS):
        return PolicyDecision(False, "invalid_policy_input")
    if action in ("reveal_secret", "external_egress"):
        return PolicyDecision(False, "sensitive_egress_disabled")
    if session_scope != "same_session":
        return PolicyDecision(False, "session_boundary")
    if action == "read_fixture":
        return PolicyDecision(True, "synthetic_read_only")
    if source != "trusted_application":
        return PolicyDecision(False, "untrusted_data_cannot_authorize_actions")
    if action == "write_shared_memory" and (role != "reviewer" or approval != "approved"):
        return PolicyDecision(False, "memory_write_requires_review")
    if action in ("execute_tool", "read_private_memory") and role == "viewer":
        return PolicyDecision(False, "insufficient_role")
    if action in ("execute_tool", "read_private_memory") and approval != "approved":
        return PolicyDecision(False, "approval_required")
    return PolicyDecision(True, "authorized_preview_only")
