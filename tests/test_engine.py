"""Real retrieval tests for the engine.

Every assertion is a behavior the engine must keep: a query that names
a technique must surface that technique, the corpus must stay complete,
and rankings must be deterministic. Run from the repository root:

    python -m unittest discover -s tests -t . -v
"""

import unittest

from engine.bm25 import BM25Index
from engine.corpus import load_corpus, load_readme_items, tokenize
from engine.rerank import hybrid_search
from engine.vector import VectorIndex

UNITS = load_corpus()
BM25 = BM25Index(UNITS)
VECTOR = VectorIndex(UNITS)


def uids(results):
    return [unit.uid for unit, _score in results]


class CorpusTests(unittest.TestCase):
    def test_readme_items_complete(self):
        self.assertEqual(len(load_readme_items()), 111)

    def test_total_units(self):
        # 111 README items + 85 content pages (9 cover/divider/blank
        # pages of the handbook fall below the token floor).
        self.assertEqual(len(UNITS), 196)


class BM25Tests(unittest.TestCase):
    def test_bluf_is_top1(self):
        results = BM25.search(tokenize("bottom line up front"), top=5)
        self.assertEqual(uids(results)[0], "analyst-01")

    def test_ach_material_in_top5(self):
        results = BM25.search(tokenize("analysis of competing hypotheses"), top=5)
        ach = {"analytic-62", "analytic-63", "analytic-64", "analytic-65",
               "doc-analytic_thinking_and_presentation-p63",
               "doc-analytic_thinking_and_presentation-p64",
               "doc-analytic_thinking_and_presentation-p65",
               "doc-analytic_thinking_and_presentation-p66"}
        self.assertTrue(ach & set(uids(results)))

    def test_unknown_terms_match_nothing(self):
        self.assertEqual(BM25.search(tokenize("zxqw vbnm"), top=5), [])


class VectorTests(unittest.TestCase):
    def test_active_voice_is_top1(self):
        results = VECTOR.search(tokenize("use the active voice"), top=5)
        self.assertEqual(uids(results)[0], "analytic-31")

    def test_us_care_is_top1(self):
        results = VECTOR.search(tokenize("why the US should care"), top=5)
        self.assertEqual(uids(results)[0], "decision-09")

    def test_unknown_terms_match_nothing(self):
        self.assertEqual(VECTOR.search(tokenize("zxqw vbnm"), top=5), [])


class HybridTests(unittest.TestCase):
    def test_bluf_is_top1(self):
        results = hybrid_search("bottom line up front", BM25, VECTOR, top=5)
        self.assertEqual(uids(results)[0], "analyst-01")

    def test_concept_paper_is_top1(self):
        results = hybrid_search("concept paper", BM25, VECTOR, top=5)
        self.assertEqual(uids(results)[0], "analytic-41")

    def test_seven_steps_is_top1(self):
        results = hybrid_search("how to review a draft in seven steps",
                                BM25, VECTOR, top=5)
        self.assertEqual(uids(results)[0], "decision-14")

    def test_briefing_material_in_top5(self):
        results = hybrid_search("giving an intelligence briefing",
                                BM25, VECTOR, top=5)
        briefing = {f"analytic-{n:02d}" for n in range(73, 81)} | {
            f"doc-analytic_thinking_and_presentation-p{n:02d}"
            for n in range(79, 89)
        }
        self.assertTrue(briefing & set(uids(results)))

    def test_deterministic(self):
        first = uids(hybrid_search("mind-set and bias", BM25, VECTOR, top=10))
        second = uids(hybrid_search("mind-set and bias", BM25, VECTOR, top=10))
        self.assertEqual(first, second)
        self.assertTrue(first)

    def test_scores_descending(self):
        results = hybrid_search("objectivity and bias", BM25, VECTOR, top=10)
        scores = [score for _unit, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
