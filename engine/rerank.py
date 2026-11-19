"""Hybrid ranking: reciprocal rank fusion + feature reranking.

Pipeline:
1. BM25 and the vector index each return their top `pool` units.
2. Reciprocal rank fusion (k = 60) merges the two rankings.
3. A reranker rescores the fused pool with surface features the base
   rankers cannot see: exact phrase match (title, then body) and
   query-term coverage of the title and of the body.

Deterministic: ties break on unit id.
"""

from .corpus import tokenize

RRF_K = 60
RRF_SCALE = 10.0
PHRASE_IN_TITLE = 0.30
PHRASE_IN_BODY = 0.15
TITLE_COVERAGE = 0.20
BODY_COVERAGE = 0.10


def _norm(text):
    return " ".join(text.lower().split())


def rerank(query, fused, pool_units):
    """fused: {uid: rrf_score}; pool_units: {uid: Unit}. Ranked list out."""
    q_terms = set(tokenize(query))
    q_phrase = _norm(query)
    results = []
    for uid, rrf in fused.items():
        unit = pool_units[uid]
        score = rrf * RRF_SCALE
        if q_phrase and q_phrase in _norm(unit.title):
            score += PHRASE_IN_TITLE
        elif q_phrase and q_phrase in _norm(unit.body):
            score += PHRASE_IN_BODY
        if q_terms:
            title_hits = len(q_terms & set(unit.title_tokens))
            body_terms = set(tokenize(unit.body))
            score += TITLE_COVERAGE * (title_hits / len(q_terms))
            score += BODY_COVERAGE * (len(q_terms & body_terms) / len(q_terms))
        results.append((unit, score))
    results.sort(key=lambda pair: (-pair[1], pair[0].uid))
    return results


def hybrid_search(query, bm25_index, vector_index, top=10, pool=30):
    q_tokens = tokenize(query)
    fused = {}
    pool_units = {}
    for ranking in (bm25_index.search(q_tokens, top=pool),
                    vector_index.search(q_tokens, top=pool)):
        for rank, (unit, _score) in enumerate(ranking, start=1):
            fused[unit.uid] = fused.get(unit.uid, 0.0) + 1.0 / (RRF_K + rank)
            pool_units[unit.uid] = unit
    return rerank(query, fused, pool_units)[:top]
