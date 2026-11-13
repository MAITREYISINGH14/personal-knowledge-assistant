"""Phase 4: tree planning, prompts, caching/incremental reuse, and merged retrieval."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pka.summarize import (SummaryBuilder, build_prompt, group_consecutive, plan_tree,
                           span_label)


def rows(n, tokens=300, prefix="chunk", section=lambda i: f"Doc > {i + 1}. Part"):
    return [{"id": f"{prefix}{i}", "text": f"{prefix} text {i}", "section": section(i),
             "n_tokens": tokens} for i in range(n)]


class TestPlanning(unittest.TestCase):
    def test_single_chunk_documents_need_no_summaries(self):
        self.assertEqual(plan_tree("d", 1, "T", rows(1), 1800, 8), [])

    def test_tree_always_ends_in_one_root(self):
        for n in (2, 5, 9, 40, 400):
            nodes = plan_tree("d", 1, "T", rows(n), 1800, 8)
            roots = [x for x in nodes if x.is_root]
            self.assertEqual(len(roots), 1, n)
            self.assertIs(nodes[-1], roots[0])
            self.assertEqual(roots[0].level, max(x.level for x in nodes))

    def test_tiny_budget_still_terminates(self):
        nodes = plan_tree("d", 1, "T", rows(30), 100, 8)
        self.assertTrue(nodes[-1].is_root)

    def test_groups_are_consecutive_and_within_budget(self):
        groups = group_consecutive([SimpleNamespace(tokens=500)] * 7, 1800, 8)
        self.assertEqual([len(g) for g in groups], [3, 4])   # lone last node merged

    def test_same_content_same_hash_across_versions(self):
        a = plan_tree("d", 1, "T", rows(9), 1800, 8)
        changed = rows(9)
        changed[8]["text"] = "edited"                 # only the last chunk changes
        b = plan_tree("d", 2, "T", changed, 1800, 8)
        self.assertEqual(a[0].hash, b[0].hash)        # first part unchanged: reusable
        self.assertNotEqual(a[-1].hash, b[-1].hash)   # its ancestors must be redone

    def test_span_labels(self):
        self.assertEqual(span_label(["p.3", "p.4 | p.5"]), "pages 3–5")
        self.assertEqual(span_label(["Doc > 1. Parties", "Doc > 4. Fee"]), "1. Parties … 4. Fee")

    def test_root_prompt_asks_for_structure(self):
        nodes = plan_tree("d", 1, "My Doc", rows(3), 1800, 8)
        p = build_prompt(nodes[-1], "My Doc")
        self.assertIn("how it is organised", p)
        self.assertIn("[1] chunk text 0", p)


class TestBuilder(unittest.TestCase):
    def setUp(self):
        from pka.storage.metadata import MetadataStore
        from pka.versioning import SCHEMA as VSCHEMA
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "m.db"
        meta = MetadataStore(self.db)
        doc = SimpleNamespace(doc_id="d", version=1, path="p", sha256="h", title="T",
                              doc_date=None, file_mtime="2026-11-10T00:00:00+00:00")
        chunk_rows = [{"chunk_id": r["id"], "chunk_index": i, "section": r["section"],
                       "text": r["text"], "n_words": 3, "n_tokens": r["n_tokens"]}
                      for i, r in enumerate(rows(9))]
        meta.save_document(doc, "pka_semantic", chunk_rows)
        meta.conn.executescript(VSCHEMA)
        meta.conn.execute("INSERT INTO versions VALUES ('pka_semantic','d',1,'2026-11-10',NULL,'file date')")
        meta.conn.commit()
        meta.conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def builder(self):
        b = SummaryBuilder(db_path=self.db)
        b._store = mock.Mock()
        return b

    def test_build_then_rebuild_is_free(self):
        calls = []
        llm = lambda system, user: calls.append(user) or f"summary {len(calls)}"
        b = self.builder()
        with mock.patch("builtins.print"):
            done, n, stopped = b.build(llm=llm)
        self.assertEqual((done, stopped), (1, None))
        self.assertEqual(n, len(calls))
        self.assertGreater(n, 1)
        with mock.patch("builtins.print"):
            _, n2, _ = b.build(llm=llm)                   # everything cached now
        self.assertEqual(n2, 0)
        b.close()

    def test_daily_limit_stops_and_resumes(self):
        from pka.llm import RateLimitExhausted
        state = {"n": 0}

        def flaky(system, user):
            state["n"] += 1
            if state["n"] == 2:
                raise RateLimitExhausted("daily")
            return "s"
        b = self.builder()
        with mock.patch("builtins.print"):
            done, calls, stopped = b.build(llm=flaky)
        self.assertEqual(done, 0)
        self.assertIn("limit", stopped)
        with mock.patch("builtins.print"):
            done, calls2, stopped2 = b.build(llm=lambda s, u: "s")
        self.assertEqual((done, stopped2), (1, None))
        total = len(plan_tree("d", 1, "T", rows(9), 1800, 8))
        self.assertEqual(calls + calls2, total)            # nothing generated twice
        b.close()


class TestMergedRetrieval(unittest.TestCase):
    def test_dense_merges_both_indexes_by_score(self):
        from pka.summary_system import SummarySystem
        s = SummarySystem.__new__(SummarySystem)
        s.store = mock.Mock(search=lambda vec, limit, query_filter=None:
                            [SimpleNamespace(id="c1", score=0.6), SimpleNamespace(id="c2", score=0.3)])
        s.summary_store = mock.Mock(search=lambda vec, limit, query_filter=None:
                                    [SimpleNamespace(id="root", score=0.8)])
        self.assertEqual(s.dense_ids([0.0], 2), ["root", "c1"])


if __name__ == "__main__":
    unittest.main()
