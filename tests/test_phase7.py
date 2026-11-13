"""Phase 7: sync history, query rewriting, summary prompt wording, scanned-page warnings, results tables."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.test_phase6 import COLL, ServiceCase, SyncCase


class TestSyncHistory(SyncCase):
    def events(self, **kw):
        return self.meta.events(COLL, **kw)

    def test_every_action_is_recorded(self):
        for i in range(5):
            self.write(f"keep{i}.md", f"# K\n\nText {i}.")
        p = self.write("notes.md", "# N\n\nOne.")
        self.sync()
        p.write_text("# N\n\nTwo.", encoding="utf-8")
        self.sync()
        p.unlink()
        self.sync()
        kinds = [(e["kind"], e["version"]) for e in self.events(doc_id="notes")]
        self.assertEqual(kinds, [("removed", 2), ("removed", 1), ("new_version", 2), ("added", 1)])  # newest first
        self.assertTrue(all(e["at"] for e in self.events()))

    def test_removed_documents_keep_their_history(self):
        for i in range(5):
            self.write(f"keep{i}.md", f"# K\n\nText {i}.")
        gone = self.write("gone.md", "# G\n\nBye.")
        self.sync()
        gone.unlink()
        self.sync()
        self.assertEqual(self.meta.latest_version("gone", COLL), None)      # out of the index
        self.assertEqual([e["kind"] for e in self.events(doc_id="gone")], ["removed", "added"])

    def test_the_same_failure_is_not_logged_twice(self):
        import pka.sync as sync_mod
        self.write("good.md", "# G\n\nFine.")
        bad = self.write("bad.md", "# B\n\nBroken.")
        real = sync_mod.load_file

        def flaky(path):
            if Path(path).name == "bad.md":
                raise ValueError("corrupt")
            return real(path)
        with mock.patch.object(sync_mod, "load_file", flaky):
            self.sync()
            bad.write_text("# B\n\nStill broken.", encoding="utf-8")
            self.sync()
        failed = [e for e in self.events() if e["kind"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["file"], "bad.md")

    def test_scanned_pages_are_reported(self):
        import pka.sync as sync_mod
        self.write("report.md", "# R\n\nText.")
        real = sync_mod.load_file

        def with_empty_pages(path):
            doc = real(path)
            doc.pages, doc.empty_pages = 40, 3
            return doc
        with mock.patch.object(sync_mod, "load_file", with_empty_pages):
            rep = self.sync()
        self.assertTrue(any("3 of 40 pages" in w for w in rep.warnings))

    def test_fully_scanned_file_explains_what_to_do(self):
        self.write("scan.md", "")
        rep = self.sync()
        self.assertIn("OCR", rep.failed[0][1])


class TestServiceHistory(ServiceCase):
    def log(self, events):
        from pka.storage.metadata import MetadataStore
        meta = MetadataStore(self.db)
        meta.log_events(COLL, events, at="2026-11-13T11:00:00+00:00")
        meta.conn.close()

    def test_activity_and_last_change(self):
        self.log([("added", "rental_agreement", 2, None, ""), ("failed", None, None, "scan.pdf", "no text")])
        a = self.make()
        kinds = [e["kind"] for e in a.activity()]
        self.assertEqual(kinds, ["failed", "added"])
        self.assertEqual(a.status()["last_change"]["kind"], "added")      # failures aren't "changes"

    def test_document_history_and_removed_documents(self):
        self.log([("added", "demo_trip", 1, None, ""), ("removed", "demo_trip", 1, None, "file no longer on disk"),
                  ("new_version", "rental_agreement", 2, None, "edited")])
        a = self.make()
        self.assertEqual(a.document("rental_agreement")["history"][0]["kind"], "new_version")
        gone = a.document("demo_trip")
        self.assertTrue(gone["removed"])
        self.assertEqual([e["kind"] for e in gone["history"]], ["removed", "added"])
        self.assertIsNone(a.document("never_existed"))


class TestQueryRewrite(unittest.TestCase):
    def rewriter(self, reply):
        from pka.query_rewrite import QueryRewriter
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []

        def llm(q):
            self.calls.append(q)
            if isinstance(reply, Exception):
                raise reply
            return reply
        rw = QueryRewriter(db_path=Path(self.tmp.name) / "q.db", llm=llm)
        self.addCleanup(rw.conn.close)
        return rw

    def test_parse_cleans_numbering_duplicates_and_the_original(self):
        from pka.query_rewrite import parse
        out = parse("1. monthly licence fee\n- Licence fee amount\n\nmonthly licence fee\nWhat was my rent?\n"
                    "\"rent payable\"\nfourth one", "What was my rent?")
        self.assertEqual(out, ["monthly licence fee", "Licence fee amount", "rent payable"])

    def test_rewrites_are_cached(self):
        rw = self.rewriter("monthly licence fee\nrent amount")
        self.assertEqual(rw.rewrite("What was my rent?"), ["monthly licence fee", "rent amount"])
        self.assertEqual(rw.rewrite("  what was my   rent?"), ["monthly licence fee", "rent amount"])
        self.assertEqual(len(self.calls), 1)

    def test_failure_means_no_rewrites(self):
        from pka.llm import LLMError
        self.assertEqual(self.rewriter(LLMError("down")).rewrite("q"), [])

    def test_rewrites_find_the_differently_worded_passage(self):
        from tests.test_phase5 import TODAY, TestTemporalRetrieval
        s = TestTemporalRetrieval().make()
        s.payloads["r1"]["text"] = "Monthly licence fee is 28,000"
        s.payloads["x1"]["text"] = "Notes about rent and the society"
        from pka.retrieval.bm25 import BM25
        s.bm25 = BM25([p["text"] for p in s.payloads.values()])
        s.store = mock.Mock(search=lambda vec, limit, query_filter=None: [])     # dense finds nothing
        s.embedder = mock.Mock(embed=lambda texts, prefix="": [[0.0]] * len(texts))
        seen = []
        s.reranker = mock.Mock(score=lambda q, ps: seen.append(q) or [1.0] * len(ps))
        s.rewriter = mock.Mock(rewrite=lambda q: ["monthly licence fee"])
        with mock.patch("pka.temporal.today", return_value=TODAY):
            out = s.retrieve("What was my rent in June 2026?")
        self.assertIn("r1", [pid for pid in s.payloads if s.payloads[pid]["text"] in [c.text for c in out]])
        self.assertEqual(seen, ["What was my rent ?"])     # the reranker sees the question, not the rewrite

    def test_no_rewriter_means_unchanged_behaviour(self):
        from tests.test_phase5 import TODAY, TestTemporalRetrieval
        s = TestTemporalRetrieval().make()
        with mock.patch("pka.temporal.today", return_value=TODAY):
            out = s.retrieve("What is my licence fee now?")
        self.assertEqual({c.version for c in out if c.doc_id == "rent"}, {2})


class TestSummaryWording(unittest.TestCase):
    def test_ranges_are_not_mistaken_for_titles(self):
        from pka.summarize import build_prompt, plan_tree, span_phrase
        self.assertEqual(span_phrase(["Doc > 1. Parties", "Doc > 8. Use of Premises"]),
                         'the sections from "1. Parties" to "8. Use of Premises"')
        self.assertEqual(span_phrase(["p.3", "p.9"]), "pages 3–9")
        self.assertEqual(span_phrase(["Doc > 9. Repairs"]), 'the section "9. Repairs"')
        rows = [{"id": f"c{i}", "text": f"t{i}", "section": f"Doc > {i + 1}. Part", "n_tokens": 600}
                for i in range(12)]
        root = plan_tree("d", 1, "Doc", rows, 1800, 8)[-1]
        self.assertIn('Part 1, covering the sections from "1. Part"', build_prompt(root, "Doc"))


class TestResults(unittest.TestCase):
    def report(self, folder, name, system, n=35, partial=False, corr=0.8, hit=0.9):
        t = {"n": 10, "hit_rate": hit, "full_hit_rate": hit, "mrr": 0.7, "correctness": corr,
             "version_clean": 1.0, "avg_latency_s": 1}
        summary = {"fact": dict(t), "overview": dict(t), "temporal": dict(t),
                   "all": dict(t, n=n)}
        (folder / name).write_text(json.dumps({"system": system, "llm": "groq/m", "partial": partial,
                                               "summary": summary, "rows": []}))

    def test_tables_prefer_complete_runs_and_flag_partial_ones(self):
        from eval.results import build, update_readme
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.report(d, "20261112-010000_baseline.json", "baseline", corr=0.70)
            self.report(d, "20261113-010000_temporal.json", "temporal", corr=0.84)
            self.report(d, "20261113-020000_temporal.json", "temporal", n=10, corr=0.80)  # --only temporal
            self.report(d, "20261113-030000_summary.json", "summary", n=34, partial=True, corr=0.89)
            self.report(d, "20261113-040000_hybrid_retrieval.json", "hybrid_retrieval", corr=None)
            md = build(d)
            self.assertIn("**0.70**", md)
            self.assertIn("**0.84**", md)                 # complete temporal run, not the 10-question one
            self.assertNotIn("**0.80**", md)
            self.assertIn("partial: 34 of 35 questions", md)
            self.assertIn("+ BM25 + fusion + cap", md)
            readme = d / "README.md"
            readme.write_text("intro\n<!-- results:start -->\nold\n<!-- results:end -->\noutro\n")
            self.assertTrue(update_readme(readme, md))
            text = readme.read_text()
            self.assertIn("intro", text)
            self.assertIn("outro", text)
            self.assertNotIn("old", text)
            self.assertIn("**0.84**", text)


if __name__ == "__main__":
    unittest.main()
