"""Command-line entry point: search the technique corpus.

    python -m engine "bottom line up front" --mode hybrid --top 5
    python -m engine "active voice" --mode bm25
    python -m engine "competing hypotheses" --mode vector
"""

import argparse

from .bm25 import BM25Index
from .corpus import load_corpus, tokenize
from .rerank import hybrid_search
from .vector import VectorIndex


def build_indexes(units=None):
    units = load_corpus() if units is None else units
    return BM25Index(units), VectorIndex(units)


def run(query, mode="hybrid", top=5):
    bm25_index, vector_index = build_indexes()
    if mode == "bm25":
        return bm25_index.search(tokenize(query), top=top)
    if mode == "vector":
        return vector_index.search(tokenize(query), top=top)
    return hybrid_search(query, bm25_index, vector_index, top=top)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="engine", description=__doc__)
    parser.add_argument("query", help="what to look for in the corpus")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"],
                        default="hybrid")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)
    results = run(args.query, mode=args.mode, top=args.top)
    if not results:
        print("No matching techniques.")
        return 1
    for rank, (unit, score) in enumerate(results, start=1):
        print(f"{rank}. [{unit.uid}] {unit.title} — {score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
