"""Query expansion by pseudo-relevance feedback (Rocchio-style).

First pass: run the vector index and take the top documents. Their
highest-weight terms — corpus vocabulary the user did not type — are
added to the query, so a second pass can match documents that express
the same idea in different words. The original query terms are kept at
double weight; expansion terms enter once.
"""

from collections import Counter

TOP_DOCS = 3
TOP_TERMS = 5


def expand_tokens(query_tokens, vector_index, top_docs=TOP_DOCS, top_terms=TOP_TERMS):
    """Return (expanded_tokens, added_terms). Deterministic."""
    if not query_tokens:
        return list(query_tokens), []
    first_pass = vector_index.search(query_tokens, top=top_docs)
    if not first_pass:
        return list(query_tokens), []
    query_set = set(query_tokens)
    weights = Counter()
    position = {u.uid: i for i, u in enumerate(vector_index.units)}
    for unit, _score in first_pass:
        vec = vector_index.vectors[position[unit.uid]]
        for term, weight in vec.items():
            if term not in query_set:
                weights[term] += weight
    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    added = [term for term, _w in ranked[:top_terms]]
    return list(query_tokens) * 2 + added, added
