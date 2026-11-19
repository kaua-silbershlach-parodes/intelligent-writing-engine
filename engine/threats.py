"""Threat model: the risk register and the evidence-state machine.

Evidence states: observed -> hypothesis -> validated (only with
corroboration) ; any state -> invalidated. An invalidated assumption
stays invalidated: a fresh id and fresh evidence are required.
"""

STATES = ("observed", "hypothesis", "validated", "invalidated")
ACTIONS = ("hypothesize", "corroborate", "refute")

_TRANSITIONS = {
    ("observed", "hypothesize"): "hypothesis",
    ("hypothesis", "corroborate"): "validated",
    ("observed", "refute"): "invalidated",
    ("hypothesis", "refute"): "invalidated",
    ("validated", "refute"): "invalidated",
}


def transition(state, action):
    """Next evidence state, or None when the transition is illegal."""
    return _TRANSITIONS.get((state, action))


# Risk register: (id, asset, threat, likelihood, impact, mitigation).
RISK_MATRIX = (
    ("R1", "corpus", "indirect prompt injection planted in a document",
     "medium", "high", "raguard.screen on ingest and on query"),
    ("R2", "corpus", "poisoned or near-duplicate units skew rankings",
     "low", "medium", "raguard.near_duplicates audit"),
    ("R3", "index", "silent corpus change after indexing",
     "medium", "medium", "integrity manifest verified by assurance"),
    ("R4", "service", "query flooding / oversized inputs",
     "medium", "low", "policy bounds enforced by the gate"),
    ("R5", "supply chain", "non-stdlib or tampered engine code",
     "low", "high", "integrity stdlib audit + file hashes"),
    ("R6", "interface", "agent-to-agent delegation without scope",
     "medium", "medium", "a2a delegation gate (role + kind scope)"),
    ("R7", "errors", "error messages leaking corpus internals",
     "low", "low", "uniform error codes, no vocabulary echo"),
    ("R8", "results", "results without provenance",
     "low", "medium", "every result carries its unit id and kind"),
)

ASSUMPTIONS = (
    "The corpus is trusted after screening; screening is pattern-based, not proof.",
    "Roles are assigned by the embedding application, never self-claimed.",
    "Rankings are deterministic: same corpus + same query = same result.",
)

ZONES = ("query", "index", "corpus", "interface", "supply-chain")
