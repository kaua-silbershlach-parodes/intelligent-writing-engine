"""Graph authority over the corpus: PageRank on the similarity graph.

Units are nodes; an edge links each unit to its nearest neighbours by
TF-IDF cosine similarity. PageRank (damping 0.85) gives every unit a
static authority score: units that sit at the centre of the corpus —
the ones many other units point to — rank as authorities. Retrieval
can then rank by similarity x authority, the same shape used for
knowledge-graph chunks: PageRank value multiplied by similarity.
"""

import math

DAMPING = 0.85
NEIGHBORS = 5
ITERATIONS = 100
TOLERANCE = 1e-8


class GraphIndex:
    def __init__(self, units, vector_index):
        self.units = units
        n = len(units)
        # Cosine similarity between every pair of unit vectors.
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            sims = []
            vi, ni = vector_index.vectors[i], vector_index.norms[i]
            for j in range(n):
                if i == j:
                    continue
                vj = vector_index.vectors[j]
                dot = sum(w * vj.get(t, 0.0) for t, w in vi.items())
                sims.append((dot / (ni * vector_index.norms[j]), j))
            sims.sort(key=lambda pair: (-pair[0], units[pair[1]].uid))
            neighbors[i] = [j for s, j in sims[:NEIGHBORS] if s > 0]
        # Symmetrize: an edge counts if either side lists the other.
        edges = [set() for _ in range(n)]
        for i in range(n):
            for j in neighbors[i]:
                edges[i].add(j)
                edges[j].add(i)
        # PageRank power iteration.
        pr = [1.0 / n] * n
        for _ in range(ITERATIONS):
            new = [(1 - DAMPING) / n] * n
            for i in range(n):
                if not edges[i]:
                    share = DAMPING * pr[i] / n
                    for j in range(n):
                        new[j] += share
                    continue
                share = DAMPING * pr[i] / len(edges[i])
                for j in edges[i]:
                    new[j] += share
            delta = sum(abs(a - b) for a, b in zip(new, pr))
            pr = new
            if delta < TOLERANCE:
                break
        self.pagerank = {units[i].uid: pr[i] for i in range(n)}
        top = max(pr) if pr else 1.0
        self.pagerank_norm = {uid: v / top for uid, v in self.pagerank.items()}

    def authority(self, uid):
        """PageRank normalized to [0, 1] by the corpus maximum."""
        return self.pagerank_norm.get(uid, 0.0)
