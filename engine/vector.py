"""Vector-space ranking: TF-IDF vectors with cosine similarity.

Each unit is a sparse vector over the vocabulary; term weight is
(1 + ln tf) * idf. The query is embedded in the same space and ranked
by cosine similarity to every unit.
"""

import math
from collections import Counter


class VectorIndex:
    def __init__(self, units):
        self.units = units
        n = len(units)
        df = Counter()
        counters = [Counter(u.tokens) for u in units]
        for counts in counters:
            for term in counts:
                df[term] += 1
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self.vectors = []
        self.norms = []
        for counts in counters:
            vec = {
                term: (1 + math.log(f)) * self.idf[term]
                for term, f in counts.items()
            }
            self.vectors.append(vec)
            self.norms.append(math.sqrt(sum(w * w for w in vec.values())) or 1.0)

    def search(self, query_tokens, top=10):
        qcounts = Counter(query_tokens)
        qvec = {
            term: (1 + math.log(f)) * self.idf[term]
            for term, f in qcounts.items()
            if term in self.idf
        }
        qnorm = math.sqrt(sum(w * w for w in qvec.values()))
        if qnorm == 0:
            return []
        scores = []
        for i, vec in enumerate(self.vectors):
            dot = sum(w * vec.get(term, 0.0) for term, w in qvec.items())
            scores.append(dot / (qnorm * self.norms[i]))
        ranked = sorted(range(len(self.units)), key=lambda i: (-scores[i], self.units[i].uid))
        return [(self.units[i], scores[i]) for i in ranked[:top] if scores[i] > 0]
