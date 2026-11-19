#!/usr/bin/env python3
"""Replay the project's probe and contract matrix through the engine.

Cases: the assurance probe suite, delegated through the A2A surface
(expected top-1 for each); negative delegation cases (an unknown
caller and an unknown mode must be refused at the gate); and the
contract matrix. A safe consumer may reject a valid preview — it must
never accept an expected-denied case. The report carries category
counts only: no query text, no file names.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import a2a, assurance, audit


def _probe_cases() -> list[tuple[str, str, str, str, bool]]:
    """(group, query, mode, caller, expected_allow) — probes plus negatives."""
    cases = [("probes", query, mode, "reader", True)
             for query, mode, _expected in assurance.PROBES]
    cases.append(("negatives", "concept paper", "hybrid", "nobody", False))
    cases.append(("negatives", "concept paper", "quantum", "reader", False))
    return cases


def audit_engine() -> dict:
    results: dict[str, Counter] = defaultdict(Counter)
    expected_top = {(q, m): e for q, m, e in assurance.PROBES}
    unsafe = 0
    rejected_valid = 0
    for group, query, mode, caller, expected_allow in _probe_cases():
        task = a2a.submit(caller, query, mode=mode, top=1)
        accepted = task.state == "completed"
        correct = accepted and bool(task.artifacts) and \
            task.artifacts[0]["unit"] == expected_top.get((query, mode))
        results[group]["cases"] += 1
        results[group]["expected_allowed" if expected_allow else "expected_denied"] += 1
        results[group]["accepted" if accepted else "refused"] += 1
        if expected_allow and not correct:
            rejected_valid += 1
            results[group]["valid_preview_rejected"] += 1
        if not expected_allow and accepted:
            unsafe += 1
            results[group]["unsafe_acceptances"] += 1
    contracts_ok = audit.audit_contracts().allowed
    results["contracts"]["cases"] += 1
    results["contracts"]["accepted" if contracts_ok else "refused"] += 1
    counts = {k: dict(sorted(v.items())) for k, v in sorted(results.items())}
    total = sum(v["cases"] for v in results.values())
    return {
        "version": "engine-matrix-consumer-v1",
        "cases": total,
        "groups": counts,
        "unsafe_acceptances": unsafe,
        "valid_preview_rejected": rejected_valid,
        "security_gate_go": unsafe == 0 and rejected_valid == 0 and contracts_ok,
        "production_authorized": False,
        "network_connections_attempted": False,
        "repo_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the engine matrix; counts only")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    out = audit_engine()
    text = json.dumps(out, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.write_text(text, encoding="utf-8")
    print(text)
    return 0 if out["security_gate_go"] else 2


if __name__ == "__main__":
    sys.exit(main())
