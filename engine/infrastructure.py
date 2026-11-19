"""Infrastructure policy gate.

A workload preview is allowed only when twenty-six policy fields
hold at once: a trusted reviewer in an approved scope, bound to the
same tenant and namespace, with least-privilege identity, private
authenticated endpoints, private scoped storage, short-lived
credentials, redacted secrets, a dedicated service account, minimal
read-only RBAC, restricted admission, unprivileged sandboxed runtime
with no host mounts, isolated accelerators, approved image provenance
with verified build attestation, denied egress, redacted audit, an
accountable approval, read-only activity, synthetic artifacts from a
trusted policy source, no external access, and resource limits.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

Decision = namedtuple("Decision", ["allowed", "failed_gate"])


@dataclass(frozen=True)
class PolicyEvidence:
    role: str = "reviewer"
    scope: str = "approved_fixture"
    tenant_scope: str = "same_tenant"
    cloud_identity: str = "least_privilege"
    namespace_scope: str = "same_namespace"
    endpoint_auth: str = "private_authenticated"
    storage_policy: str = "private_scoped"
    credentials: str = "short_lived"
    secrets: str = "redacted"
    service_account: str = "dedicated_no_automount"
    rbac: str = "minimal_read_only"
    admission: str = "restricted"
    pod_privilege: str = "unprivileged"
    host_mount: str = "none"
    runtime: str = "sandboxed"
    gpu_isolation: str = "isolated"
    image_provenance: str = "approved_digest"
    build_attestation: str = "verified_fixture"
    egress: str = "denied"
    audit: str = "redacted_audited"
    approval: str = "approved"
    activity: str = "read_only_preview"
    artifact_origin: str = "synthetic"
    policy_source: str = "trusted"
    no_external_access: bool = True
    resource_limits: bool = True


# Gate order is the field order of PolicyEvidence; each entry is
# (field, value that satisfies the gate).
_GATES = (
    ("role", "reviewer"), ("scope", "approved_fixture"),
    ("tenant_scope", "same_tenant"), ("cloud_identity", "least_privilege"),
    ("namespace_scope", "same_namespace"),
    ("endpoint_auth", "private_authenticated"),
    ("storage_policy", "private_scoped"), ("credentials", "short_lived"),
    ("secrets", "redacted"), ("service_account", "dedicated_no_automount"),
    ("rbac", "minimal_read_only"), ("admission", "restricted"),
    ("pod_privilege", "unprivileged"), ("host_mount", "none"),
    ("runtime", "sandboxed"), ("gpu_isolation", "isolated"),
    ("image_provenance", "approved_digest"),
    ("build_attestation", "verified_fixture"), ("egress", "denied"),
    ("audit", "redacted_audited"), ("approval", "approved"),
    ("activity", "read_only_preview"), ("artifact_origin", "synthetic"),
    ("policy_source", "trusted"),
)

_VIOLATIONS = {
    "role": ("role", "untrusted"), "scope": ("scope", "out_of_scope"),
    "tenant_scope": ("tenant_scope", "other_tenant"),
    "cloud_identity": ("cloud_identity", "admin"),
    "namespace_scope": ("namespace_scope", "other_namespace"),
    "endpoint_auth": ("endpoint_auth", "public_anonymous"),
    "storage_policy": ("storage_policy", "public"),
    "credentials": ("credentials", "long_lived"),
    "secrets": ("secrets", "in_plaintext"),
    "service_account": ("service_account", "default_auto_mount"),
    "rbac": ("rbac", "wildcard_write"), "admission": ("admission", "disabled"),
    "pod_privilege": ("pod_privilege", "privileged"),
    "host_mount": ("host_mount", "host_path"),
    "runtime": ("runtime", "host_shared"),
    "gpu_isolation": ("gpu_isolation", "unrestricted"),
    "image_provenance": ("image_provenance", "unverified"),
    "build_attestation": ("build_attestation", "missing"),
    "egress": ("egress", "public"), "audit": ("audit", "unredacted"),
    "approval": ("approval", "missing"),
    "activity": ("activity", "production_write"),
    "artifact_origin": ("artifact_origin", "unknown"),
    "policy_source": ("policy_source", "model_output"),
    "external_access": ("no_external_access", False),
    "resource_limits": ("resource_limits", False),
}

# (scenario id, category, risk, required control, gate that must block it)
SCENARIOS = (
    ("broad_ml_execution_role", "cloud_iam", "Overbroad inference role crosses workload scope", "Workload least privilege and explicit resource scoping", "cloud_identity"),
    ("unbounded_role_trust", "cloud_iam", "Role trust spans unapproved environment", "Bind the caller and trust policy to verified scope", "scope"),
    ("cross_tenant_assume_role", "cloud_iam", "Identity may traverse a tenant boundary", "Independent tenant binding before every tool action", "tenant_scope"),
    ("model_endpoint_without_auth", "inference", "Inference endpoint lacks independently verified identity", "Private authenticated inference endpoint", "endpoint_auth"),
    ("model_bucket_public_read", "storage", "Model artifact store is publicly readable", "Private storage scoped to required principal", "storage_policy"),
    ("training_dataset_cross_tenant", "storage", "Training dataset read across tenant boundary", "Tenant isolation enforced before storage read", "tenant_scope"),
    ("long_lived_cloud_credentials", "credentials", "Non-expiring credential used by workload", "Short-lived workload-bound credentials", "credentials"),
    ("secrets_in_model_metadata", "credentials", "Secrets included in model registry metadata", "Redact sensitive metadata before publishing", "secrets"),
    ("logs_with_identity_tokens", "telemetry", "Sensitive tokens appear in diagnostics", "Redacted structured logs and trace controls", "audit"),
    ("unapproved_external_callback", "network", "Workload contacts unapproved destination", "No outbound network access in offline fixture", "egress"),
    ("service_account_auto_token", "kubernetes", "Pod automatically receives API token without need", "Dedicated ServiceAccount and token automount disabled", "service_account"),
    ("namespace_cross_access", "kubernetes", "Workload accesses another namespace", "Bind namespace to trusted workload identity", "namespace_scope"),
    ("rolebinding_wildcard", "kubernetes", "RBAC grants wildcard rights on resources", "Restrict verbs and resources to minimal read-only scope", "rbac"),
    ("pod_admission_not_enforced", "kubernetes", "Pod admission admits unsafe specification", "Enforce restricted admission before scheduling", "admission"),
    ("privileged_container", "container", "Workload requests privileged container execution", "Deny privileged execution", "pod_privilege"),
    ("host_filesystem_mount", "container", "Workload requests access to host filesystem", "Disallow host path mounts", "host_mount"),
    ("host_namespace_sharing", "container", "Workload can reach host process or network namespace", "Enforce runtime isolation independent of app permissions", "runtime"),
    ("gpu_device_broad_mapping", "accelerator", "Accelerator device access exceeds authorized isolation", "Explicit GPU device allocation and isolation review", "gpu_isolation"),
    ("unsigned_inference_image", "supply_chain", "Serving image lacks approved digest", "Pinned and approved container image digest", "image_provenance"),
    ("unverified_training_build", "supply_chain", "Build output lacks required attestation evidence", "Require verified synthetic build evidence", "build_attestation"),
    ("policy_data_from_model_output", "policy", "Model response attempts to become deployment authority", "Only trusted service context can set policy source", "policy_source"),
    ("unapproved_deployment_change", "governance", "Deployment preview lacks an accountable approval", "Independent authorization for deployment activities", "approval"),
    ("production_mutation_attempt", "governance", "Unsafe production change requested", "Read-only synthetic previews only", "activity"),
    ("unapproved_artifact_origin", "supply_chain", "Model or runtime artifact originates outside lab", "Allow only synthetic authorized fixture metadata", "artifact_origin"),
    ("live_provider_api_access", "network", "Scenario would call a real provider API", "No external access under local simulation boundary", "external_access"),
    ("resource_limits_missing", "container", "Workload has no bounded resources", "Enforce resource requests and limits in deployment policy", "resource_limits"),
    ("untrusted_operator_role", "identity", "Unknown user role claims to approve deployments", "Reviewer role must be trusted service context", "role"),
    ("model_log_export", "telemetry", "Diagnostic data disclosed beyond allowed recipients", "Redacted auditable export policy", "audit"),
)


def check_policy(evidence: PolicyEvidence) -> Decision:
    for field, want in _GATES:
        if getattr(evidence, field) != want:
            return Decision(False, field)
    if not evidence.no_external_access:
        return Decision(False, "external_access")
    if not evidence.resource_limits:
        return Decision(False, "resource_limits")
    return Decision(True, "none")


def violate(gate: str) -> PolicyEvidence:
    """The default evidence with exactly one gate broken."""
    field, bad = _VIOLATIONS[gate]
    values = {f: getattr(PolicyEvidence(), f)
              for f in PolicyEvidence.__dataclass_fields__}
    values[field] = bad
    return PolicyEvidence(**values)


def simulate(scenario_id: str) -> Decision:
    for row in SCENARIOS:
        if row[0] == scenario_id:
            return check_policy(violate(row[4]))
    return Decision(False, "unknown")
