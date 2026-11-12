"""Phase 3: BM25, rank fusion, diversity and the hybrid/rerank retrieval flow (with fakes)."""
import unittest
from types import SimpleNamespace
from unittest import mock

from pka.retrieval.bm25 import BM25, tokenize
from pka.retrieval.fusion import diversify, rrf


class TestBM25(unittest.TestCase):
    def test_tokenize_keeps_money_and_hyphenated_terms(self):
        self.assertEqual(tokenize("Fee is ₹84,000; RAG-Sequence"),
                         ["fee", "₹84,000", "rag-sequence", "rag", "sequence"])

    def test_rare_exact_term_wins(self):
        docs = ["chunk boundaries and thresholds in general",
                "Rohan said the 92nd percentile threshold was fine",
                "a recipe for dal with three whistles"]
        top = BM25(docs).search("what percentile did Rohan suggest", 3)
        self.assertEqual(top[0][0], 1)

    def test_no_match_returns_nothing(self):
        self.assertEqual(BM25(["alpha beta"]).search("gamma"), [])


class TestFusion(unittest.TestCase):
    def test_rrf_rewards_agreement(self):
        fused = [x for x, _ in rrf([["a", "b", "c"], ["c", "a", "d"]])]
        self.assertEqual(fused[0], "a")            # ranked well by both
        self.assertEqual(set(fused), {"a", "b", "c", "d"})

    def test_diversify_caps_per_group(self):
        items = [("a1", 1), ("a2", 1), ("a3", 1), ("b1", 1), ("c1", 1)]
        out = diversify(items, key=lambda x: x[0][0], k=4, max_per_group=2)
        self.assertEqual([x[0] for x in out], ["a1", "a2", "b1", "c1"])

    def test_diversify_backfills_when_short(self):
        items = [("a1", 1), ("a2", 1), ("a3", 1)]
        out = diversify(items, key=lambda x: x[0][0], k=3, max_per_group=1)
        self.assertEqual(len(out), 3)


class TestHybridFlow(unittest.TestCase):
    def make(self, rerank_scores=None):
        from pka.hybrid import HybridSystem
        payloads = {
            "p1": {"doc_id": "survey", "version": 1, "text": "RAG overview one", "section": ""},
            "p2": {"doc_id": "survey", "version": 1, "text": "RAG overview two", "section": ""},
            "p3": {"doc_id": "survey", "version": 1, "text": "RAG overview three", "section": ""},
            "p4": {"doc_id": "survey", "version": 1, "text": "RAG overview four", "section": ""},
            "p5": {"doc_id": "rag_paper", "version": 1, "text": "RAG-Sequence and RAG-Token", "section": ""},
        }
        s = HybridSystem.__new__(HybridSystem)
        s.k, s.pool, s.max_per_doc, s.query_prefix = 3, 10, 2, ""
        s.payloads, s.ids = payloads, list(payloads)
        s.bm25 = BM25([p["text"] for p in payloads.values()])
        s.embedder = mock.Mock(embed=lambda texts, prefix="": [[0.0]])
        s.store = mock.Mock(search=lambda vec, limit: [SimpleNamespace(id=i) for i in ["p1", "p2", "p3", "p4"]])
        s.reranker = mock.Mock(score=lambda q, ps: rerank_scores(ps)) if rerank_scores else None
        return s

    def test_keyword_hit_enters_and_one_document_is_capped(self):
        out = self.make().retrieve("RAG-Sequence variants")
        self.assertIn("rag_paper", [c.doc_id for c in out])
        self.assertLessEqual(sum(c.doc_id == "survey" for c in out), 2)

    def test_reranker_decides_order(self):
        out = self.make(lambda ps: [5.0 if "RAG-Token" in p else 0.0 for p in ps]).retrieve("RAG-Token")
        self.assertEqual(out[0].doc_id, "rag_paper")


if __name__ == "__main__":
    unittest.main()
