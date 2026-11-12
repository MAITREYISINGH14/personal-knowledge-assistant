"""Phase 5: time scopes, version validity, chunk diffs, time-aware retrieval, metric, judge."""
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pka.query_time import parse_time_scope
from pka.versioning import ALWAYS, VersionRegistry, diff_chunks, word_diff

TODAY = date(2026, 11, 13)


class TestTimeScope(unittest.TestCase):
    def kind(self, q):
        return parse_time_scope(q, TODAY).kind

    def test_kinds(self):
        self.assertEqual(self.kind("What is my monthly rent now?"), "current")
        self.assertEqual(self.kind("What was my rent in June 2026?"), "as_of")
        self.assertEqual(self.kind("What changed when the agreement was renewed?"), "change")
        self.assertEqual(self.kind("How and why did the plan change mid-quarter?"), "change")
        self.assertEqual(self.kind("What happened to the Friday rule?"), "change")

    def test_dates(self):
        self.assertEqual(parse_time_scope("rent in June 2026?", TODAY).on, date(2026, 6, 30))
        self.assertEqual(parse_time_scope("budget as of the end of July?", TODAY).on, date(2026, 7, 31))
        self.assertEqual(parse_time_scope("plan on 5 August 2026?", TODAY).on, date(2026, 8, 5))
        self.assertEqual(parse_time_scope("rent as of 2026-03-01?", TODAY).on, date(2026, 3, 1))
        self.assertEqual(parse_time_scope("budget in December?", TODAY).on, date(2025, 12, 31))

    def test_may_the_verb_is_not_a_month(self):
        self.assertEqual(self.kind("May I ask what my rent is?"), "current")

    def test_future_date_means_current(self):
        self.assertEqual(self.kind("What will the rent be in March 2027?"), "current")

    def test_as_of_parameter_only_without_date_in_question(self):
        s = parse_time_scope("What was my rent?", TODAY, as_of="2026-02-01")
        self.assertEqual((s.kind, s.on), ("as_of", date(2026, 2, 1)))


def chunk(cid, text, vec, section="S"):
    return {"id": cid, "text": text, "section": section, "vector": vec}


class TestDiff(unittest.TestCase):
    def test_word_diff_names_the_numbers(self):
        d = word_diff("The monthly fee is ₹28,000 per month.", "The monthly fee is ₹30,240 per month.")
        self.assertIn('"₹28,000" → "₹30,240"', d)

    def test_kinds(self):
        old = [chunk("o1", "same text", [1, 0, 0]), chunk("o2", "fee is 100", [0, 1, 0]),
               chunk("o3", "parking slot", [0, 0, 1])]
        new = [chunk("n1", "same text", [1, 0, 0]), chunk("n2", "fee is 120", [0, 1, 0.05]),
               chunk("n3", "pets allowed", [0.6, -0.8, 0])]
        kinds = sorted(r["kind"] for r in diff_chunks(old, new))
        self.assertEqual(kinds, ["added", "modified", "removed"])      # identical text not listed


class FakeQdrant:
    """Just enough of the Qdrant client for the registry."""
    def __init__(self, points):
        self.points, self.payload_updates = points, []

    def scroll(self, collection, scroll_filter=None, limit=256, offset=None, **_):
        want = {c.key: c.match.value for c in scroll_filter.must}
        hits = [p for p in self.points if all(p.payload.get(k) == v for k, v in want.items())]
        return hits, None

    def set_payload(self, collection, payload, points):
        want = {c.key: c.match.value for c in points.filter.must}
        for p in self.points:
            if all(p.payload.get(k) == v for k, v in want.items()):
                p.payload.update(payload)


def qdrant_models_available():
    try:
        import qdrant_client.models  # noqa: F401
        return True
    except ImportError:
        return False


def stub_qdrant_models():
    """Minimal stand-in for qdrant_client.models when the library isn't installed."""
    import sys
    import types
    if qdrant_models_available():
        return
    m = types.ModuleType("qdrant_client.models")
    m.Filter = lambda must: SimpleNamespace(must=must)
    m.FieldCondition = lambda key, match=None, range=None: SimpleNamespace(key=key, match=match, range=range)
    m.MatchValue = lambda value: SimpleNamespace(value=value)
    m.FilterSelector = lambda filter: SimpleNamespace(filter=filter)
    pkg = types.ModuleType("qdrant_client")
    pkg.models = m
    sys.modules.setdefault("qdrant_client", pkg)
    sys.modules.setdefault("qdrant_client.models", m)


class TestRegistry(unittest.TestCase):
    def setUp(self):
        stub_qdrant_models()

    def test_validity_and_change_log(self):
        from pka.storage.metadata import MetadataStore
        pts = [SimpleNamespace(id="a1", vector=[1, 0], payload={"doc_id": "rent", "version": 1,
                               "chunk_index": 0, "text": "Fee is ₹28,000.", "section": "Fee"}),
               SimpleNamespace(id="a2", vector=[1, 0.05], payload={"doc_id": "rent", "version": 2,
                               "chunk_index": 0, "text": "Fee is ₹30,240.", "section": "Fee"}),
               SimpleNamespace(id="n1", vector=[0, 1], payload={"doc_id": "note", "version": 1,
                               "chunk_index": 0, "text": "A note.", "section": ""})]
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "m.db"
            meta = MetadataStore(db)
            for doc_id, v, dated in (("rent", 1, "2025-12-15"), ("rent", 2, "2026-11-01"), ("note", 1, None)):
                meta.save_document(SimpleNamespace(doc_id=doc_id, version=v, path="p", sha256="h",
                                                   title="t", doc_date=dated,
                                                   file_mtime="2026-11-10T00:00:00+00:00"), "c", [])
            reg = VersionRegistry("c", db_path=db, client=FakeQdrant(pts))
            with mock.patch("builtins.print"):
                reg.rebuild()
            v1, v2, note = (p.payload for p in pts)
            self.assertEqual((v1["valid_from_day"], v1["valid_to_day"]), (0, date(2026, 11, 1).toordinal()))
            self.assertEqual((v2["valid_from_day"], v2["valid_to_day"]),
                             (date(2026, 11, 1).toordinal(), ALWAYS))
            self.assertEqual(v2["version_status"], "current")
            self.assertEqual(note["version_status"], "only")
            self.assertEqual(reg.versioned_docs(), {"rent"})
            log = reg.change_log("rent")
            self.assertIn("v1 (dated 2025-12-15) to v2 (dated 2026-11-01)", log)
            self.assertIn('"₹28,000." → "₹30,240."', log)


class TestTemporalRetrieval(unittest.TestCase):
    def make(self):
        from pka.retrieval.bm25 import BM25
        from pka.temporal import TemporalSystem
        nov1, aug = date(2026, 11, 1).toordinal(), date(2026, 8, 1).toordinal()
        payloads = {
            "r1": {"doc_id": "rent", "version": 1, "text": "Licence fee ₹28,000", "valid_from_day": 0,
                   "valid_to_day": nov1, "version_status": "superseded", "doc_date": "2025-12-15"},
            "r2": {"doc_id": "rent", "version": 2, "text": "Licence fee ₹30,240", "valid_from_day": nov1,
                   "valid_to_day": ALWAYS, "version_status": "current", "doc_date": "2026-11-01"},
            "x1": {"doc_id": "note", "version": 1, "text": "Fee notes", "valid_from_day": 0,
                   "valid_to_day": ALWAYS, "version_status": "only"},
        }
        s = TemporalSystem.__new__(TemporalSystem)
        s.k, s.pool, s.max_per_doc, s.query_prefix = 3, 10, 3, ""
        s.payloads, s.ids = payloads, list(payloads)
        s.bm25 = BM25([p["text"] for p in payloads.values()])
        s.embedder = mock.Mock(embed=lambda texts, prefix="": [[0.0]])

        def search(vec, limit, query_filter=None):
            ids = list(payloads)
            if query_filter is not None:
                d = query_filter
                ids = [i for i in ids if payloads[i]["valid_from_day"] <= d < payloads[i]["valid_to_day"]]
            return [SimpleNamespace(id=i) for i in ids]
        s.store = mock.Mock(search=search)
        s._date_filter = lambda d: d          # the fake search takes the day directly
        s.reranker = None
        s.registry = mock.Mock(changed_chunk_ids=lambda d: ["r1", "r2"],
                               change_log=lambda d: "Changes in rent ...")
        s.versioned, s._scope, s._log = {"rent"}, {}, {}
        return s

    def versions(self, chunks):
        return {c.version for c in chunks if c.doc_id == "rent"}

    def test_current_sees_only_new_version(self):
        with mock.patch("pka.temporal.today", return_value=TODAY):
            out = self.make().retrieve("What is my licence fee now?")
        self.assertEqual(self.versions(out), {2})
        self.assertIn("current version", out[0].meta["in_force"] if out[0].doc_id == "rent" else
                      next(c for c in out if c.doc_id == "rent").meta["in_force"])

    def test_as_of_sees_only_old_version(self):
        with mock.patch("pka.temporal.today", return_value=TODAY):
            out = self.make().retrieve("What was my licence fee in June 2026?")
        self.assertEqual(self.versions(out), {1})

    def test_change_gets_both_versions_and_log(self):
        s = self.make()
        with mock.patch("pka.temporal.today", return_value=TODAY):
            out = s.retrieve("What changed in the licence fee?")
        self.assertEqual(self.versions(out), {1, 2})
        self.assertIn("What changed in the licence fee?", s._log)


class TestMetricAndJudge(unittest.TestCase):
    def test_version_clean(self):
        from eval.metrics import score_retrieval
        from eval.schema import Question, Source
        from pka.interfaces import RetrievedChunk
        q = Question("t", "temporal", "rent now?", "a", [Source("rent", "", 2, [])], temporal_kind="current")
        self.assertTrue(score_retrieval(q, [RetrievedChunk("rent", "x", version=2)], 5)["version_clean"])
        self.assertFalse(score_retrieval(q, [RetrievedChunk("rent", "x", version=2),
                                             RetrievedChunk("rent", "y", version=1)], 5)["version_clean"])

    def test_empty_judge_output_is_ungraded_not_incorrect(self):
        from eval import judge
        with mock.patch.object(judge, "settings", mock.Mock(llm_provider="groq")), \
             mock.patch.object(judge, "chat", return_value=""):
            g = judge.Judge().grade("q", "a", "an answer")
        self.assertEqual((g["verdict"], g["score"]), ("ungraded", None))

    def test_citation_tidying(self):
        from pka.llm import tidy_citations
        self.assertEqual(tidy_citations("It is ₹30,240【2†L1-L2】."), "It is ₹30,240[2].")


if __name__ == "__main__":
    unittest.main()


class TestPhase5Fixes(unittest.TestCase):
    def test_different_clauses_are_not_a_modification(self):
        # Same document, similar vectors (shared header), but a different clause entirely.
        old = [chunk("o", "One covered parking slot B-12 is included in the fee.", [1, 0.1], "Doc > 7. Parking")]
        new = [chunk("n", "The Licensee may keep one small pet with a deposit.", [1, 0.12], "Doc > 7. Pets")]
        self.assertEqual(sorted(r["kind"] for r in diff_chunks(old, new)), ["added", "removed"])

    def test_rewritten_clause_with_same_heading_is_modified(self):
        old = [chunk("o", "Minor repairs are the Licensee's job; major ones the Licensor's.", [1, 0], "Doc > 9. Repairs")]
        new = [chunk("n", "Repairs under 2,000 rupees are paid by the Licensee.", [1, 0.1], "Doc > 9. Repairs")]
        self.assertEqual([r["kind"] for r in diff_chunks(old, new)], ["modified"])

    def test_date_phrase_removed_from_search_query(self):
        s = TestTemporalRetrieval().make()
        seen = []
        s.embedder = mock.Mock(embed=lambda texts, prefix="": seen.append(texts[0]) or [[0.0]])
        with mock.patch("pka.temporal.today", return_value=TODAY):
            s.retrieve("What was my licence fee in June 2026?")
        self.assertNotIn("June", seen[0])
        self.assertIn("licence fee", seen[0])

    def test_cache_context_changes_with_change_log(self):
        s = TestTemporalRetrieval().make()
        a = s.cache_context("q")
        s._log["q"] = "Changes in rent ..."
        self.assertNotEqual(a, s.cache_context("q"))
