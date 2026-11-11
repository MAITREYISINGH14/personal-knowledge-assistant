"""Run with: python -m unittest discover tests"""
import json
import tempfile
import unittest
from pathlib import Path

from eval.metrics import score_retrieval
from eval.schema import EvalSetError, load_eval_set
from pka.interfaces import RetrievedChunk
from scripts.inventory_corpus import parse_name

ROOT = Path(__file__).resolve().parent.parent


def write_set(questions):
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump({"questions": questions}, f)
    f.close()
    return Path(f.name)


class TestSchema(unittest.TestCase):
    def test_bundled_eval_set_is_valid(self):
        self.assertEqual(len(load_eval_set(ROOT / "tests" / "fixtures" / "example_eval_set.json")), 6)

    def test_temporal_requires_version(self):
        path = write_set([{"id": "t", "type": "temporal", "temporal_kind": "current",
                           "question": "q", "expected_answer": "a",
                           "sources": [{"doc_id": "d"}]}])
        with self.assertRaises(EvalSetError):
            load_eval_set(path)

    def test_duplicate_ids_rejected(self):
        q = {"id": "x", "type": "fact", "question": "q", "expected_answer": "a",
             "sources": [{"doc_id": "d"}]}
        with self.assertRaises(EvalSetError):
            load_eval_set(write_set([q, q]))


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.q = load_eval_set(ROOT / "tests" / "fixtures" / "example_eval_set.json")

    def by_id(self, qid):
        return next(q for q in self.q if q.id == qid)

    def test_fact_hit_needs_section_and_version(self):
        q = self.by_id("fact-001")
        good = RetrievedChunk("rental_agreement", "", section="Agreement > Termination", version=2)
        old = RetrievedChunk("rental_agreement", "", section="Agreement > Termination", version=1)
        self.assertTrue(score_retrieval(q, [good], 5)["hit"])
        self.assertFalse(score_retrieval(q, [old], 5)["hit"])

    def test_rank_and_cutoff(self):
        q = self.by_id("fact-002")
        noise = RetrievedChunk("other", "")
        good = RetrievedChunk("rag_research_notes", "", section="Notes > Embeddings")
        self.assertAlmostEqual(score_retrieval(q, [noise, good], 5)["rr"], 0.5)
        self.assertFalse(score_retrieval(q, [noise, good], 1)["hit"])

    def test_change_question_full_hit(self):
        q = self.by_id("temporal-001")
        v1 = RetrievedChunk("rental_agreement", "", section="Rent", version=1)
        v2 = RetrievedChunk("rental_agreement", "", section="Rent", version=2)
        self.assertFalse(score_retrieval(q, [v2], 5)["full_hit"])
        self.assertTrue(score_retrieval(q, [v1, v2], 5)["full_hit"])

    def test_overview_matches_summary_node(self):
        q = self.by_id("overview-001")
        node = RetrievedChunk("market_research_report", "", level=3)
        self.assertTrue(score_retrieval(q, [node], 5)["hit"])


class TestCorpusNames(unittest.TestCase):
    def test_parse_name(self):
        self.assertEqual(parse_name("rental_agreement__v2"), ("rental_agreement", 2))
        self.assertEqual(parse_name("notes"), ("notes", 1))


if __name__ == "__main__":
    unittest.main()


class TestRealEvalSet(unittest.TestCase):
    def test_project_eval_set_is_valid(self):
        self.assertGreaterEqual(len(load_eval_set(ROOT / "eval" / "eval_set.json")), 1)
