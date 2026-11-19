"""Service policy: the single source of operational limits and roles.

Every other module consults this policy instead of hard-coding its own
limits. A Decision names the gate that failed, or carries an empty
failed_gate when the operation is allowed.
"""

from collections import namedtuple

Decision = namedtuple("Decision", ["allowed", "failed_gate"])

# Retrieval surface.
MODES = ("bm25", "vector", "fusion", "hybrid")
MAX_QUERY_CHARS = 500
MAX_QUERY_TOKENS = 64
MAX_TOP = 50
MAX_THRESHOLD = 10.0

# Caller roles and the modes each role may use (offline authorization:
# roles are local caller labels, never a caller's claim of privilege —
# the embedding application assigns them).
ROLE_MODES = {
    "reader": ("bm25", "vector", "fusion", "hybrid"),
    "operator": ("bm25", "vector", "fusion", "hybrid"),
    "admin": ("bm25", "vector", "fusion", "hybrid"),
}

# Corpus classifications a caller may retrieve, by role.
ROLE_KINDS = {
    "reader": None,      # None = no kind restriction
    "operator": None,
    "admin": None,
}


def decide_mode(role, mode):
    if role not in ROLE_MODES:
        return Decision(False, "unknown_role")
    if mode not in MODES:
        return Decision(False, "unknown_mode")
    if mode not in ROLE_MODES[role]:
        return Decision(False, "mode_not_in_role")
    return Decision(True, "")


def decide_top(top):
    if not isinstance(top, int) or top < 1:
        return Decision(False, "top_below_one")
    if top > MAX_TOP:
        return Decision(False, "top_above_cap")
    return Decision(True, "")
