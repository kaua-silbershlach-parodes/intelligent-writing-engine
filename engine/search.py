"""Command-line entry point: search the technique corpus.

    python -m engine "bottom line up front" --mode hybrid --top 5
    python -m engine "active voice" --mode bm25
    python -m engine "competing hypotheses" --mode vector
    python -m engine "concept paper" --mode fusion
    python -m engine "briefing" --mode hybrid --graph --expand
    python -m engine "judgements" --mode hybrid --kind "Analyst writing techniques"
"""

import argparse

from .bm25 import BM25Index
from .corpus import load_corpus, tokenize
from .fusion import fusion_search
from .graph import GraphIndex
from .recon import EngineError
from .rerank import hybrid_search
from .vector import VectorIndex


def build_indexes(units=None):
    units = load_corpus() if units is None else units
    return BM25Index(units), VectorIndex(units)


def run(query, mode="hybrid", top=5, threshold=0.0, graph=False,
        expand=False, kind=None):
    from . import gate, recon, telemetry
    bm25_index, vector_index = build_indexes()
    decision = gate.mediate(query, mode, top, set(bm25_index.idf))
    if not decision.allowed:
        raise recon.EngineError(decision.failed_gate)
    with telemetry.Timer() as timer:
        results = _run(query, mode, top, threshold, graph, expand, kind,
                       bm25_index, vector_index)
    telemetry.record(mode, len(results), timer.ms)
    return results


def _run(query, mode, top, threshold, graph, expand, kind,
         bm25_index, vector_index):
    q_tokens = tokenize(query)
    if expand and mode in ("bm25", "vector", "fusion"):
        from .expand import expand_tokens
        q_tokens, _added = expand_tokens(q_tokens, vector_index)
    if mode == "bm25":
        return bm25_index.search(q_tokens, top=top)
    if mode == "vector":
        return vector_index.search(q_tokens, top=top)
    if mode == "fusion":
        results = fusion_search(q_tokens, bm25_index, vector_index,
                                top=len(bm25_index.units), threshold=threshold)
        if kind is not None:
            results = [pair for pair in results if pair[0].kind == kind]
        return results[:top]
    graph_index = GraphIndex(bm25_index.units, vector_index) if graph else None
    return hybrid_search(query, bm25_index, vector_index, top=top,
                         threshold=threshold, graph_index=graph_index,
                         expand=expand, kind=kind)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="engine", description=__doc__)
    parser.add_argument("query", help="what to look for in the corpus")
    parser.add_argument("--mode", choices=["bm25", "vector", "fusion", "hybrid"],
                        default="hybrid")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="drop results below this final score")
    parser.add_argument("--graph", action="store_true",
                        help="weight hybrid results by PageRank authority")
    parser.add_argument("--expand", action="store_true",
                        help="expand the query by pseudo-relevance feedback")
    parser.add_argument("--kind", default=None,
                        help="metadata filter: only this corpus kind")
    args = parser.parse_args(argv)
    try:
        results = run(args.query, mode=args.mode, top=args.top,
                      threshold=args.threshold, graph=args.graph,
                      expand=args.expand, kind=args.kind)
    except EngineError as error:
        print(str(error))
        return 2
    if not results:
        print("No matching techniques.")
        return 1
    for rank, (unit, score) in enumerate(results, start=1):
        print(f"{rank}. [{unit.uid}] {unit.title} — {score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
