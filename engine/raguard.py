"""Retrieval security: screening and corpus hygiene for the RAG layer.

- screen(text): pattern screen for indirect-injection phrasing — text
  that tries to instruct the reader/agent instead of informing it.
  Screening is pattern-based evidence, never proof of intent.
- near_duplicates(units): corpus-poisoning audit — unit pairs whose
  token sets are almost identical (Jaccard >= 0.9).
- retrieval_decision(role, kind): the ACL gate for a retrieval —
  role must exist and the kind must be in the role's scope.
"""

import re

from . import policy

INJECTION_PATTERNS = (
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (the )?(system|previous|above)",
    r"you are now (a|an|in)\b",
    r"reveal (the )?(system prompt|instructions|prompt)",
    r"execute (the )?(following )?(command|code|script)",
    r"send (the )?(data|results|output) to\b",
    r"override (the )?(policy|rules|guardrails)",
)
_COMPILED = tuple(re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS)


def screen(text):
    """List of matched injection-pattern labels (empty = clean)."""
    return [p.pattern for p in _COMPILED if p.search(text or "")]


def corpus_screen(units):
    """{uid: [patterns]} for units whose text trips the screen."""
    flagged = {}
    for unit in units:
        hits = screen(unit.title + " " + unit.body)
        if hits:
            flagged[unit.uid] = hits
    return flagged


def near_duplicates(units, threshold=0.9):
    """Pairs of unit ids with token-set Jaccard >= threshold."""
    sets = [(u.uid, set(u.tokens)) for u in units]
    pairs = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i][1], sets[j][1]
            if not a or not b:
                continue
            jaccard = len(a & b) / len(a | b)
            if jaccard >= threshold:
                pairs.append((sets[i][0], sets[j][0]))
    return pairs


def retrieval_decision(role, kind):
    if role not in policy.ROLE_KINDS:
        return policy.Decision(False, "unknown_role")
    scope = policy.ROLE_KINDS[role]
    if scope is not None and kind is not None and kind not in scope:
        return policy.Decision(False, "kind_out_of_scope")
    return policy.Decision(True, "")
