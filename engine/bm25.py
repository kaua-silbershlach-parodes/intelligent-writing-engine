"""Okapi BM25 ranking over the corpus (lexical retrieval).

idf uses the Lucene form, ln(1 + (N - n + 0.5) / (n + 0.5)), which stays
positive for terms that occur in more than half the corpus.
"""

import math
from collections import Counter

K1 = 1.5
B = 0.75


class BM25Index:
    def __init__(self, units):
        self.units = units
        self.n = len(units)
        self.doc_len = [len(u.tokens) for u in units]
        self.avgdl = sum(self.doc_len) / max(self.n, 1)
        self.tf = [Counter(u.tokens) for u in units]
        df = Counter()
        for counts in self.tf:
            for term in counts:
                df[term] += 1
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query_tokens, top=10):
        scores = [0.0] * self.n
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.tf):
                f = counts.get(term, 0)
                if not f:
                    continue
                denom = f + K1 * (1 - B + B * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (K1 + 1)) / denom
        ranked = sorted(range(self.n), key=lambda i: (-scores[i], self.units[i].uid))
        return [(self.units[i], scores[i]) for i in ranked[:top] if scores[i] > 0]
