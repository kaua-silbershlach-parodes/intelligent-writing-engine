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

Standard library only. Run from the repository root:

    python -m engine "bottom line up front" --mode hybrid --top 5
"""
