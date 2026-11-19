"""Intelligence Writing Engine — retrieval over the technique corpus.

Retrieval modes over the same corpus (the README's technique items plus
the archived documents, page by page):

- bm25   — lexical ranking (Okapi BM25).
- vector — vector-space ranking (TF-IDF vectors, cosine similarity).
- fusion — weighted linear fusion (0.7 keyword + 0.3 vector).
- hybrid — multi-way recall, reciprocal rank fusion, feature reranking.

Options: PageRank graph authority (--graph), pseudo-relevance-feedback
query expansion (--expand), a similarity threshold (--threshold) and a
metadata filter (--kind).

A security and assurance layer wraps the rankers: policy, a mediation
gate, an agent-to-agent surface, retrieval screening, a threat model,
integrity manifests, contract audits, bounded telemetry, a recovery
rehearsal, a publication guard, a uniform error surface, and a capstone
assurance run (python -m engine.assurance).

Standard library only. Run from the repository root:

    python -m engine "bottom line up front" --mode hybrid --top 5
"""
