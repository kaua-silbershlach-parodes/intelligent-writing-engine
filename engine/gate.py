"""Complete mediation gate: every query passes here before any index
is touched, and every check runs on every call — no cached approvals.

Gates, in order: mode policy, query bounds, control characters, token
cap, and the vector-space check (a query whose tokens are all unknown
would be a zero vector — rejected here with a uniform error surface).
"""

from . import policy
from .corpus import tokenize


def mediate(query, mode, top, known_terms):
    """Return policy.Decision for a would-be search."""
    d = policy.decide_mode("reader", mode)
    if not d.allowed:
        return d
    d = policy.decide_top(top)
    if not d.allowed:
        return d
    if not isinstance(query, str) or not query.strip():
        return policy.Decision(False, "empty_query")
    if len(query) > policy.MAX_QUERY_CHARS:
        return policy.Decision(False, "query_too_long")
    if any(ord(c) < 32 and c not in "\t" for c in query):
        return policy.Decision(False, "control_characters")
    tokens = tokenize(query)
    if not tokens:
        return policy.Decision(False, "no_terms")
    if len(tokens) > policy.MAX_QUERY_TOKENS:
        return policy.Decision(False, "too_many_tokens")
    if not any(t in known_terms for t in tokens):
        return policy.Decision(False, "zero_vector")
    return policy.Decision(True, "")
