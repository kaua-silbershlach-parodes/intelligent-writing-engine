"""Recovery rehearsal: the indexes are derived state. Rebuilding them
from the corpus must reproduce identical rankings — that is the whole
recovery story, rehearsed on demand.
"""

from .corpus import tokenize
from .search import build_indexes

PROBES = ("bottom line up front", "concept paper", "competing hypotheses")


def rehearse(probes=PROBES):
    """{probe: identical_bool} comparing two independent index builds."""
    bm1, vx1 = build_indexes()
    bm2, vx2 = build_indexes()
    report = {}
    for probe in probes:
        tokens = tokenize(probe)
        first = [u.uid for u, _s in bm1.search(tokens, top=10)]
        second = [u.uid for u, _s in bm2.search(tokens, top=10)]
        report[probe] = first == second and bool(first)
    return report
