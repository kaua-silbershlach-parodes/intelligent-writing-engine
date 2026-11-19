"""Weighted linear fusion of keyword and vector similarity.

Each ranker's scores are min-max normalized over the candidate pool,
then combined linearly:

    fused = keyword_weight * bm25_norm + vector_weight * vector_norm

Defaults: keyword 0.7, vector 0.3. A similarity threshold can drop
every result below a fused-score floor.
"""

KEYWORD_WEIGHT = 0.7
VECTOR_WEIGHT = 0.3


def _normalize(pairs):
    if not pairs:
        return {}
    scores = dict(pairs)
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {uid: 1.0 for uid in scores}
    return {uid: (s - lo) / (hi - lo) for uid, s in scores.items()}


def fusion_search(query_tokens, bm25_index, vector_index, top=10,
                  keyword_weight=KEYWORD_WEIGHT,
                  vector_weight=VECTOR_WEIGHT, threshold=0.0):
    n = len(bm25_index.units)
    bm = _normalize([(u.uid, s) for u, s in bm25_index.search(query_tokens, top=n)])
    vx = _normalize([(u.uid, s) for u, s in vector_index.search(query_tokens, top=n)])
    by_uid = {u.uid: u for u in bm25_index.units}
    fused = []
    for uid in set(bm) | set(vx):
        score = keyword_weight * bm.get(uid, 0.0) + vector_weight * vx.get(uid, 0.0)
        if score >= threshold and score > 0:
            fused.append((by_uid[uid], score))
    fused.sort(key=lambda pair: (-pair[1], pair[0].uid))
    return fused[:top]
