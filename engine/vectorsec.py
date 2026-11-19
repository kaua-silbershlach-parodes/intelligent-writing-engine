"""Vector retrieval security gate.

Retrieval over an embedding index is accepted only when the evidence
holds in order: a sufficient role bound to the same tenant and
session, the access list checked, a permitted classification (internal
material requires the reviewer role), a search purpose returning only
redacted summaries, an approved index at a pinned model revision,
bounded top-k and daily query budget, index isolation, encryption,
verified provenance, reviewed redaction, minimized input, suspicious
queries and repeated probes blocked, an audit event recorded, private
model metadata, raw vector and raw chunk exports disabled, and
approval required for writes. Raw vectors and raw chunks are never
returned: exports are how embeddings get inverted.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

Decision = namedtuple("Decision", ["accepted", "failed_gate"])


@dataclass(frozen=True)
class RetrievalEvidence:
    role: str = "reviewer"
    tenant_scope: str = "same_tenant"
    session_scope: str = "same_session"
    classification: str = "public"
    purpose: str = "search"
    result_kind: str = "redacted_summary"
    index_name: str = "approved_fixture_index"
    model_revision: str = "pinned_fixture_revision"
    top_k: int = 3
    daily_query_count: int = 1
    access_list_checked: bool = True
    index_isolated: bool = True
    encryption_enabled: bool = True
    provenance_verified: bool = True
    redaction_reviewed: bool = True
    input_data_minimized: bool = True
    suspicious_query_blocked: bool = True
    repeated_probe_blocked: bool = True
    audit_event_recorded: bool = True
    model_metadata_private: bool = True
    vector_export_disabled: bool = True
    raw_chunk_export_disabled: bool = True
    write_requires_approval: bool = True


_BOOL_GATES = (
    ("access_list_checked", "acl"), ("index_isolated", "index_isolation"),
    ("encryption_enabled", "encryption"), ("provenance_verified", "provenance"),
    ("redaction_reviewed", "redaction"),
    ("input_data_minimized", "data_minimization"),
    ("suspicious_query_blocked", "suspicious_query"),
    ("repeated_probe_blocked", "probe"), ("audit_event_recorded", "audit"),
    ("model_metadata_private", "metadata"),
    ("vector_export_disabled", "vector_export"),
    ("raw_chunk_export_disabled", "raw_chunk_export"),
    ("write_requires_approval", "approval"),
)

_VIOLATIONS = {
    "acl": ("access_list_checked", False),
    "tenant_scope": ("tenant_scope", "other_tenant"),
    "session_scope": ("session_scope", "other_session"),
    "role": ("role", "viewer"),
    "vector_export": ("vector_export_disabled", False),
    "raw_chunk_export": ("raw_chunk_export_disabled", False),
    "metadata": ("model_metadata_private", False),
    "model_revision": ("model_revision", "mutable_latest"),
    "suspicious_query": ("suspicious_query_blocked", False),
    "probe": ("repeated_probe_blocked", False),
    "encryption": ("encryption_enabled", False),
    "index": ("index_name", "unknown_index"),
    "top_k": ("top_k", 200),
    "query_budget": ("daily_query_count", 10001),
    "provenance": ("provenance_verified", False),
    "redaction": ("redaction_reviewed", False),
    "index_isolation": ("index_isolated", False),
    "approval": ("write_requires_approval", False),
    "audit": ("audit_event_recorded", False),
    "data_minimization": ("input_data_minimized", False),
}

# (scenario id, risk category, threat, required control, gate that must block it)
SCENARIOS = (
    ("unauthenticated_vector_database", "exposure", "vector_database_without_authenticated_access", "authenticate_before_any_query", "acl"),
    ("cross_tenant_vector_read", "isolation", "vector_search_spans_tenant_boundaries", "server_derived_tenant_filter", "tenant_scope"),
    ("cross_session_retrieval", "isolation", "retrieval_from_other_session", "bind_retrieval_to_session", "session_scope"),
    ("role_based_retrieval_bypass", "access", "unprivileged_sensitive_query", "document_acl_before_similarity", "role"),
    ("raw_vector_export", "inversion", "bulk_embedding_export", "disable_raw_vector_export", "vector_export"),
    ("raw_chunk_export", "inversion", "reconstruction_via_full_chunk_dump", "disable_unredacted_chunk_export", "raw_chunk_export"),
    ("model_dimension_leak", "reconnaissance", "embedding_dimension_fingerprint_exposure", "hide_index_and_model_metadata", "metadata"),
    ("model_revision_exposure", "reconnaissance", "model_configuration_disclosure", "pin_model_revision_and_hide_config", "model_revision"),
    ("sensitive_attribute_inference", "attribute_inference", "document_attributes_inferred_from_similarity", "restrict_sensitive_attribute_queries", "suspicious_query"),
    ("candidate_membership_probe", "membership_inference", "repeated_presence_absence_tests", "limit_sensitive_membership_oracles", "probe"),
    ("template_bank_inversion", "inversion", "synthetic_template_matching_risk", "deny_raw_vectors_and_enforce_minimization", "vector_export"),
    ("beam_search_inversion", "inversion", "candidate_reconstruction_from_vectors", "block_vector_and_chunk_exports", "raw_chunk_export"),
    ("pretrained_decoder_reconstruction", "inversion", "pretrained_embedding_decoder_exposure", "encrypt_and_acl_retrieval_storage", "encryption"),
    ("surrogate_model_comparison", "inversion", "untrusted_candidate_model_queries", "approved_index_and_model_revision", "index"),
    ("large_topk_exposure", "bulk_exposure", "excessive_nearest_neighbor_results", "hard_cap_top_k", "top_k"),
    ("high_query_volume_probe", "bulk_exposure", "many_similarity_queries", "per_principal_query_budget", "query_budget"),
    ("unreviewed_chunk_provenance", "poisoning", "unreviewed_retrieved_chunk", "signed_or_verified_ingest_provenance", "provenance"),
    ("unredacted_summary", "disclosure", "sensitive_data_in_retrieval_summary", "redacted_summary_only", "redaction"),
    ("cross_index_reuse", "isolation", "unsafe_mixed_index", "strict_index_isolation", "index_isolation"),
    ("unreviewed_vector_ingest", "poisoning", "unapproved_mutable_vector_entries", "approved_index_writes", "approval"),
    ("search_trace_missing", "observability", "unattributed_similarity_access", "structured_query_audit_event", "audit"),
    ("unminimized_secret_ingest", "data_minimization", "sensitive_secret_ingested_into_embeddings", "minimize_sensitive_input_before_embedding", "data_minimization"),
)


def verify(evidence: RetrievalEvidence) -> Decision:
    """Verify retrieval evidence; the first failed gate names the denial."""
    e = evidence
    if e.role not in ("reviewer", "operator"):
        return Decision(False, "role")
    if e.tenant_scope != "same_tenant":
        return Decision(False, "tenant_scope")
    if e.session_scope != "same_session":
        return Decision(False, "session_scope")
    if not e.access_list_checked:
        return Decision(False, "acl")
    if e.classification not in ("public", "internal"):
        return Decision(False, "classification")
    if e.classification == "internal" and e.role != "reviewer":
        return Decision(False, "role")
    if e.purpose != "search":
        return Decision(False, "purpose")
    if e.result_kind != "redacted_summary":
        return Decision(False, "result_kind")
    if e.index_name != "approved_fixture_index":
        return Decision(False, "index")
    if e.model_revision != "pinned_fixture_revision":
        return Decision(False, "model_revision")
    if not 1 <= e.top_k <= 5:
        return Decision(False, "top_k")
    if not 1 <= e.daily_query_count <= 100:
        return Decision(False, "query_budget")
    for field, gate in _BOOL_GATES[1:]:
        if not getattr(e, field):
            return Decision(False, gate)
    return Decision(True, "none")


def negative_fixture(gate: str) -> RetrievalEvidence:
    """The default evidence with exactly one gate broken."""
    field, bad = _VIOLATIONS[gate]
    values = {f: getattr(RetrievalEvidence(), f)
              for f in RetrievalEvidence.__dataclass_fields__}
    values[field] = bad
    return RetrievalEvidence(**values)


def simulate(scenario_id: str) -> Decision:
    for row in SCENARIOS:
        if row[0] == scenario_id:
            return verify(negative_fixture(row[4]))
    return Decision(False, "unknown")
