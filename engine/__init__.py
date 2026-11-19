"""Intelligence Writing Engine — retrieval over the technique corpus.

Three retrieval modes over the same corpus (the README's technique items
plus the archived documents, page by page):

- BM25   — lexical ranking (Okapi BM25).
- vector — vector-space ranking (TF-IDF vectors, cosine similarity).
- hybrid — reciprocal rank fusion of the two, plus a feature reranker.

Standard library only. Run from the repository root:

    python -m engine "bottom line up front" --mode hybrid --top 5
"""
