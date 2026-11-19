"""Assurance: the capstone evidence run.

One call exercises the whole engine end to end and returns an evidence
report: integrity manifest self-consistency, the standard-library
audit, the contract matrix, the corpus injection screen, the recovery
rehearsal, and a fixed probe suite (query, mode, expected top-1).
Nothing here is callers' claims — every entry is measured.
"""

from . import audit, integrity, raguard, recovery
from .corpus import load_corpus
from .search import run

PROBES = (
    ("bottom line up front", "bm25", "analyst-01"),
    ("bottom line up front", "fusion", "analyst-01"),
    ("bottom line up front", "hybrid", "analyst-01"),
    ("use the active voice", "vector", "analytic-31"),
    ("why the US should care", "vector", "decision-09"),
    ("concept paper", "hybrid", "analytic-41"),
)


def run_probes():
    results = {}
    for query, mode, expected in PROBES:
        found = run(query, mode=mode, top=1)
        results[f"{mode}:{query}"] = bool(found) and found[0][0].uid == expected
    return results


def capstone():
    """The full evidence report. `ok` is True only if every check is."""
    probes = run_probes()
    checks = {
        "integrity": integrity.verify_manifest(integrity.build_manifest()).allowed,
        "stdlib_only": integrity.stdlib_audit().allowed,
        "contracts": audit.audit_contracts().allowed,
        "corpus_screen_clean": not raguard.corpus_screen(load_corpus()),
        "recovery": all(recovery.rehearse().values()),
        "probes": all(probes.values()),
    }
    return {
        "checks": checks,
        "probes": probes,
        "probes_passed": sum(probes.values()),
        "probes_total": len(probes),
        "ok": all(checks.values()),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(capstone(), indent=2))
