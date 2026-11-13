"""Phase 6: file discovery, sync (auto-versioning, pruning, safety rails), watcher, service, API."""
import json
import os
import shutil
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pka import index_state
from pka.ingestion.loaders import discover
from pka.storage.metadata import MetadataStore
from pka.sync import SyncBusy, Syncer

COLL = "pka_semantic"


class FakeEmbedder:
    dim, max_tokens = 4, 512

    def embed(self, texts, prefix=""):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def count_tokens_batch(self, texts):
        return [len(t.split()) for t in texts]


class FakeStore:
    """Stands in for the Qdrant wrapper: remembers which (doc, version) have points."""
    def __init__(self):
        self.points = {}
        self.client = mock.Mock(collection_exists=lambda name: False)

    def delete_document(self, doc_id, version):
        self.points.pop((doc_id, version), None)

    def upsert(self, ids, vectors, payloads):
        for p in payloads:
            self.points.setdefault((p["doc_id"], p["version"]), []).append(p)


def paragraph_chunker(doc):
    """One chunk per block: enough to exercise syncing without models."""
    pairs = [(SimpleNamespace(index=i, text=b.text, section=b.section, n_words=len(b.text.split())),
              b.text) for i, b in enumerate(doc.blocks)]
    return pairs, None


class SyncCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.corpus = root / "corpus"
        self.corpus.mkdir()
        self.state = root / "generation"
        self.meta = MetadataStore(root / "m.db")
        self.store = FakeStore()
        self.rebuilds = []
        self.syncer = Syncer(plan=(COLL, FakeEmbedder(), paragraph_chunker), store=self.store,
                             meta=self.meta, corpus_dir=self.corpus, state_path=self.state,
                             rebuild_versions=lambda: self.rebuilds.append(1),
                             summary_status=lambda: {"docs_total": 0, "docs_pending": 0, "calls_needed": 0})

    def tearDown(self):
        self.meta.conn.close()
        self.tmp.cleanup()

    def write(self, rel, text):
        p = self.corpus / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def sync(self, **kw):
        return self.syncer.sync(log=lambda *_: None, **kw)

    def stored(self):
        return self.meta.ingested_rows(COLL)


class TestDiscover(unittest.TestCase):
    def test_skips_hidden_and_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in ("a.md", "sub/b.txt", ".hidden.md", "~$c.pdf", "~tmp.md",
                        ".git/x.md", "README.md", "note.docx"):
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("x")
            self.assertEqual([p.name for p in discover(root)], ["a.md", "b.txt"])


class TestSyncBasics(SyncCase):
    def test_first_sync_adds_everything_and_unchanged_after(self):
        self.write("notes.md", "# Notes\n\nFirst draft.")
        self.write("policy__v1.md", "# Policy\n\nOld rule.")
        self.write("policy__v2.md", "# Policy\n\nNew rule.")
        rep = self.sync()
        self.assertEqual(len(rep.added), 3)
        self.assertEqual(self.stored(), [("notes", 1), ("policy", 1), ("policy", 2)])
        self.assertEqual(len(self.rebuilds), 1)
        self.assertTrue(index_state.current(self.state))
        gen = index_state.current(self.state)

        rep2 = self.sync()
        self.assertFalse(rep2.changed)
        self.assertEqual(rep2.unchanged, 3)
        self.assertEqual(len(self.rebuilds), 1)                  # nothing to rebuild
        self.assertEqual(index_state.current(self.state), gen)   # nothing for the app to reload

    def test_new_file_is_picked_up(self):
        self.write("a.md", "# A\n\nText.")
        self.sync()
        self.write("sub/b.md", "# B\n\nMore text.")
        rep = self.sync()
        self.assertEqual([(d, v) for d, v, _ in rep.added], [("b", 1)])
        self.assertEqual(rep.unchanged, 1)


class TestAutoVersioning(SyncCase):
    def test_editing_a_plain_file_creates_a_new_version_and_keeps_history(self):
        p = self.write("notes.md", "# Notes\n\nBudget is 100.")
        self.sync()
        p.write_text("# Notes\n\nBudget is 120.", encoding="utf-8")
        rep = self.sync()
        self.assertEqual([(d, v) for d, v, _ in rep.versioned], [("notes", 2)])
        self.assertEqual(self.stored(), [("notes", 1), ("notes", 2)])
        self.assertIn(("notes", 1), self.store.points)            # the old text is still indexed
        self.assertTrue(self.meta.effective_of("notes", 2))
        self.assertEqual(self.meta.auto_versions(COLL), [("notes", 2)])
        self.assertEqual(self.sync().unchanged, 1)                # stable afterwards

    def test_second_edit_makes_version_three(self):
        p = self.write("notes.md", "# N\n\nOne.")
        self.sync()
        for text in ("# N\n\nTwo.", "# N\n\nThree."):
            p.write_text(text, encoding="utf-8")
            self.sync()
        self.assertEqual(self.stored(), [("notes", 1), ("notes", 2), ("notes", 3)])

    def test_edit_date_is_never_before_the_previous_version(self):
        doc = SimpleNamespace(doc_date="2026-01-05", file_mtime="2026-03-01T12:00:00+00:00")
        self.assertEqual(Syncer._edit_date(doc, "2026-02-01"), "2026-03-01")
        doc.doc_date = "2026-04-01"                               # header moved forward: trust it
        self.assertEqual(Syncer._edit_date(doc, "2026-02-01"), "2026-04-01")
        doc.doc_date = None
        self.assertEqual(Syncer._edit_date(doc, None), "2026-03-01")

    def test_editing_an_explicit_version_overwrites_it(self):
        p = self.write("policy__v2.md", "# P\n\nRule A.")
        self.sync()
        p.write_text("# P\n\nRule B.", encoding="utf-8")
        with mock.patch.object(self.syncer, "_drop_summaries") as drop:
            rep = self.sync()
        self.assertEqual([(d, v) for d, v, _ in rep.updated], [("policy", 2)])
        self.assertEqual(self.stored(), [("policy", 2)])
        drop.assert_called_once_with("policy", 2)                 # stale summaries must go
        self.assertEqual(self.meta.auto_versions(COLL), [])


class TestPruning(SyncCase):
    def test_deleted_files_are_removed(self):
        keep = [self.write(f"keep{i}.md", f"# K{i}\n\nText {i}.") for i in range(6)]
        gone = self.write("policy__v1.md", "# P\n\nOld.")
        self.write("policy__v2.md", "# P\n\nNew.")
        self.sync()
        gone.unlink()
        rep = self.sync()
        self.assertEqual([(d, v) for d, v, _ in rep.removed], [("policy", 1)])
        self.assertNotIn(("policy", 1), self.store.points)
        self.assertIn(("policy", 2), self.store.points)
        self.assertEqual(len(self.rebuilds), 2)

    def test_deleting_a_plain_file_removes_all_its_versions(self):
        for i in range(5):
            self.write(f"keep{i}.md", f"# K\n\nText {i}.")
        p = self.write("notes.md", "# N\n\nOne.")
        self.sync()
        p.write_text("# N\n\nTwo.", encoding="utf-8")
        self.sync()
        p.unlink()
        rep = self.sync()
        self.assertEqual(sorted((d, v) for d, v, _ in rep.removed), [("notes", 1), ("notes", 2)])
        self.assertFalse([s for s in self.stored() if s[0] == "notes"])

    def test_empty_folder_removes_nothing(self):
        p = self.write("a.md", "# A\n\nText.")
        self.sync()
        p.unlink()
        rep = self.sync()
        self.assertEqual(rep.removed, [])
        self.assertTrue(any("looks empty" in w for w in rep.warnings))
        self.assertEqual(self.stored(), [("a", 1)])

    def test_mass_deletion_needs_force(self):
        paths = [self.write(f"f{i}.md", f"# F{i}\n\nText {i}.") for i in range(10)]
        self.sync()
        for p in paths[:8]:
            p.unlink()
        rep = self.sync()
        self.assertEqual(rep.removed, [])
        self.assertTrue(any("--force-prune" in w for w in rep.warnings))
        rep = self.sync(force_prune=True)
        self.assertEqual(len(rep.removed), 8)

    def test_no_prune_option(self):
        a, b = self.write("a.md", "# A\n\nT."), self.write("b.md", "# B\n\nT.")
        for i in range(4):
            self.write(f"k{i}.md", f"# K\n\nT{i}.")
        self.sync()
        a.unlink()
        self.assertEqual(self.sync(prune=False).removed, [])


class TestOddCases(SyncCase):
    def test_plain_file_clashing_with_versioned_files_is_skipped_but_history_kept(self):
        self.write("notes__v1.md", "# N\n\nExplicit.")
        self.sync()
        self.write("notes.md", "# N\n\nPlain.")
        rep = self.sync()
        self.assertTrue(any("clashes" in w for w in rep.warnings))
        self.assertEqual(self.stored(), [("notes", 1)])

    def test_duplicate_names_in_different_folders(self):
        self.write("a/todo.md", "# T\n\nOne.")
        self.write("b/todo.md", "# T\n\nTwo.")
        rep = self.sync()
        self.assertEqual(len(rep.added), 1)
        self.assertTrue(any("same name" in w for w in rep.warnings))

    def test_a_broken_file_does_not_stop_the_others(self):
        self.write("good.md", "# G\n\nFine.")
        bad = self.write("bad.md", "# B\n\nBroken.")

        import pka.sync as sync_mod
        real_load = sync_mod.load_file

        def flaky(path):
            if Path(path).name == "bad.md":
                raise ValueError("corrupt")
            return real_load(path)
        with mock.patch.object(sync_mod, "load_file", flaky):
            rep = self.sync()
        self.assertEqual([d for d, _, _ in rep.added], ["good"])
        self.assertEqual(rep.failed[0][0], "bad.md")
        self.assertIn("corrupt", rep.failed[0][1])

    def test_file_with_no_text_is_reported_not_stored(self):
        self.write("empty.md", "")
        rep = self.sync()
        self.assertEqual(rep.added, [])
        self.assertEqual(rep.failed[0][0], "empty.md")

    def test_concurrent_sync_is_refused(self):
        self.syncer._lock.acquire()
        try:
            with self.assertRaises(SyncBusy):
                self.syncer.sync()
        finally:
            self.syncer._lock.release()

    def test_report_text_lists_changes(self):
        self.write("a.md", "# A\n\nText.")
        rep = self.sync()
        self.assertIn("1 added", rep.one_line())
        self.assertIn("a v1", rep.text())
        self.assertTrue(rep.to_dict()["changed"])


class TestMetadataMigration(unittest.TestCase):
    def test_old_database_gets_the_new_column(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "old.db"
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE documents (doc_id TEXT NOT NULL, version INTEGER NOT NULL, "
                         "path TEXT NOT NULL, sha256 TEXT NOT NULL, title TEXT, doc_date TEXT, "
                         "file_mtime TEXT, ingested_at TEXT NOT NULL, n_chunks INTEGER, "
                         "PRIMARY KEY (doc_id, version))")
            conn.execute("INSERT INTO documents VALUES ('d',1,'p','h','t',NULL,NULL,'now',3)")
            conn.commit()
            conn.close()
            meta = MetadataStore(db)
            doc = SimpleNamespace(doc_id="d", version=1, path="p", sha256="h2", title="t",
                                  doc_date=None, file_mtime="2026-11-10T00:00:00+00:00")
            meta.save_document(doc, "c", [], effective_date="2026-11-12")
            meta.save_document(doc, "c", [])                       # must not erase the edit date
            self.assertEqual(meta.effective_of("d", 1), "2026-11-12")
            meta.conn.close()


class TestIndexState(unittest.TestCase):
    def test_bump_changes_the_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "gen"
            self.assertEqual(index_state.current(p), "")
            a = index_state.bump(p)
            b = index_state.bump(p)
            self.assertEqual(index_state.current(p), b)
            self.assertNotEqual(a, b)


class TestWatcher(unittest.TestCase):
    def make(self, **kw):
        from pka.watch import Watcher
        self.now = 0.0
        self.snap = {"a": (1, 1)}
        self.syncs = 0

        def do_sync():
            self.syncs += 1
        w = Watcher(do_sync, corpus_dir=".", interval=1, settle=10, backoff=60, log=lambda *_: None,
                    clock=lambda: self.now, snapshot_fn=lambda: dict(self.snap), **kw)
        return w

    def test_waits_for_changes_to_settle_then_syncs_once(self):
        w = self.make()
        w.sync_now()
        self.assertEqual(w.tick(), "idle")
        self.snap["b"] = (5, 5)
        self.assertEqual(w.tick(), "waiting")
        self.now = 6
        self.snap["b"] = (9, 9)                    # file still being written
        self.assertEqual(w.tick(), "waiting")
        self.now = 12
        self.assertEqual(w.tick(), "waiting")      # only 6s since the last change
        self.now = 17
        self.assertEqual(w.tick(), "synced")
        self.assertEqual(self.syncs, 2)            # initial + this one
        self.assertEqual(w.tick(), "idle")

    def test_failure_backs_off_and_retries(self):
        from pka.watch import Watcher
        calls = []

        def failing():
            calls.append(1)
            raise RuntimeError("qdrant down")
        self.now = 0.0
        snap = {"a": (1, 1)}
        w = Watcher(failing, corpus_dir=".", interval=1, settle=0, backoff=60, log=lambda *_: None,
                    clock=lambda: self.now, snapshot_fn=lambda: dict(snap))
        self.assertEqual(w.tick(), "waiting")
        self.now = 1
        self.assertEqual(w.tick(), "error")
        self.now = 30
        self.assertEqual(w.tick(), "backoff")
        self.assertEqual(len(calls), 1)
        self.now = 62
        self.assertEqual(w.tick(), "error")
        self.assertEqual(len(calls), 2)

    def test_deleted_file_triggers_sync(self):
        w = self.make()
        w.sync_now()
        del self.snap["a"]
        w.tick()
        self.now = 11
        self.assertEqual(w.tick(), "synced")

    def test_busy_sync_is_retried_without_backoff(self):
        from pka.watch import Watcher
        state = {"busy": True}

        def sync():
            if state["busy"]:
                raise SyncBusy("busy")
        self.now = 0.0
        snap = {"a": (1, 1)}
        w = Watcher(sync, corpus_dir=".", interval=1, settle=0, log=lambda *_: None,
                    clock=lambda: self.now, snapshot_fn=lambda: dict(snap))
        w.tick()
        self.now = 1
        self.assertEqual(w.tick(), "busy")
        state["busy"] = False
        self.now = 2
        self.assertEqual(w.tick(), "synced")


# ---------------------------------------------------------------------- service
class FakeSystem:
    name = "summary"

    def __init__(self, answer="Your fee is ₹30,240 [1].", fail_generate=None, answerer=True):
        self.answerer = object() if answerer else None
        self.reloads = 0
        self._answer, self._fail = answer, fail_generate
        self.embedder = object()

    def retrieve(self, question, as_of=None):
        from pka.interfaces import RetrievedChunk
        return [RetrievedChunk("rental_agreement", "The fee is ₹30,240.", section="Doc > 4. Licence Fee",
                               version=2, level=0, score=5.0,
                               meta={"status": "current", "in_force": "current version"}),
                RetrievedChunk("rental_agreement", "Overview.", section="Doc", version=2, level=2, score=2.0)]

    def describe_scope(self, question):
        return "current (today, 2026-11-13)"

    def generate(self, question, chunks):
        if self._fail:
            raise self._fail
        return self._answer

    def reload(self):
        self.reloads += 1


class ServiceCase(unittest.TestCase):
    def setUp(self):
        from pka.service import Assistant
        from pka.summarize import SCHEMA as SUMMARY_SCHEMA
        from pka.versioning import SCHEMA as VERSION_SCHEMA
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.db, self.state = root / "m.db", root / "generation"
        meta = MetadataStore(self.db)
        meta.conn.executescript(VERSION_SCHEMA + SUMMARY_SCHEMA)
        for v, dated, edit in ((1, "2025-12-15", None), (2, "2026-11-01", "2026-11-01")):
            doc = SimpleNamespace(doc_id="rental_agreement", version=v, path="p", sha256=f"h{v}",
                                  title="Leave and Licence Agreement", doc_date=dated,
                                  file_mtime="2026-11-10T00:00:00+00:00")
            rows = [{"chunk_id": f"c{v}{i}", "chunk_index": i, "section": f"Doc > {i + 1}. Clause",
                     "text": f"Clause {i} v{v}", "n_words": 3, "n_tokens": 7} for i in range(2)]
            meta.save_document(doc, COLL, rows, effective_date=edit)
        note = SimpleNamespace(doc_id="notes", version=1, path="p", sha256="n", title="Notes",
                               doc_date=None, file_mtime="2026-11-10T00:00:00+00:00")
        meta.save_document(note, COLL, [{"chunk_id": "n0", "chunk_index": 0, "section": "", "text": "A note",
                                         "n_words": 2, "n_tokens": 3}])
        c = meta.conn
        c.execute("INSERT INTO versions VALUES (?,?,?,?,?,?)", (COLL, "rental_agreement", 1, "2025-12-15", "2026-11-01", "document"))
        c.execute("INSERT INTO versions VALUES (?,?,?,?,?,?)", (COLL, "rental_agreement", 2, "2026-11-01", None, "edit date"))
        c.execute("INSERT INTO versions VALUES (?,?,?,?,?,?)", (COLL, "notes", 1, "2026-11-10", None, "file date"))
        c.execute("INSERT INTO changes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (COLL, "rental_agreement", 1, 2, "modified", "a", "b", "Doc > 4. Licence Fee", "Doc > 4. Licence Fee", 0.9, "₹28,000 → ₹30,240"))
        c.execute("INSERT INTO changes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (COLL, "rental_agreement", 1, 2, "removed", "a", None, "Doc > 7. Parking", None, None, "parking slot"))
        c.execute("INSERT INTO summary_cache VALUES ('hash1','The overview text','m','now')")
        c.execute("INSERT INTO summary_nodes VALUES (?,?,?,?,?,?,?,?,?,?)",
                  ("pka_summaries", "n1", "rental_agreement", 2, 2, 0, 1, "[]", "hash1", "all clauses"))
        c.commit()
        meta.conn.close()
        self.make = lambda **kw: Assistant(db_path=self.db, state_path=self.state, **kw)

    def tearDown(self):
        self.tmp.cleanup()


class TestAsk(ServiceCase):
    def test_answer_with_sources(self):
        a = self.make(system_factory=FakeSystem)
        r = a.ask("What is my fee now?")
        self.assertEqual(r["error"], "")
        self.assertIn("₹30,240", r["answer"])
        self.assertEqual(r["time_frame"], "current (today, 2026-11-13)")
        s = r["sources"]
        self.assertEqual([x["n"] for x in s], [1, 2])
        self.assertEqual(s[0]["heading"], "4. Licence Fee")
        self.assertEqual(s[0]["status"], "current")
        self.assertEqual(s[1]["level"], 2)

    def test_llm_failure_still_returns_sources(self):
        from pka.llm import RateLimitExhausted
        a = self.make(system_factory=lambda: FakeSystem(fail_generate=RateLimitExhausted("daily limit")))
        r = a.ask("What is my fee?")
        self.assertEqual(r["answer"], "")
        self.assertIn("answer model isn't available", r["error"])
        self.assertEqual(len(r["sources"]), 2)

    def test_no_llm_configured(self):
        r = self.make(system_factory=lambda: FakeSystem(answerer=False)).ask("q")
        self.assertIn("only the retrieved passages", r["error"])
        self.assertEqual(len(r["sources"]), 2)

    def test_bad_questions(self):
        a = self.make(system_factory=FakeSystem)
        self.assertIn("Type a question", a.ask("   ")["error"])
        self.assertIn("limited to", a.ask("x" * 2001)["error"])

    def test_empty_index_gives_a_friendly_message(self):
        def empty():
            raise RuntimeError("The 'pka_semantic' index is empty. Run: python -m pka.ingest")
        r = self.make(system_factory=empty).ask("hello")
        self.assertIn("The index is empty", r["error"])
        self.assertIn("corpus folder", r["error"])

    def test_qdrant_down_message(self):
        def down():
            raise RuntimeError("Can't reach Qdrant. Open Docker Desktop")
        r = self.make(system_factory=down).ask("hello")
        self.assertIn("docker compose up -d", r["error"])

    def test_failed_load_is_not_retried_until_the_index_changes(self):
        calls = []

        def factory():
            calls.append(1)
            raise RuntimeError("index is empty")
        a = self.make(system_factory=factory)
        a.ask("one")
        a.ask("two")
        self.assertEqual(len(calls), 1)
        index_state.bump(self.state)                  # a sync finished: try again
        a.ask("three")
        self.assertEqual(len(calls), 2)

    def test_system_reloads_when_the_index_changes(self):
        system = FakeSystem()
        a = self.make(system_factory=lambda: system)
        a.ask("first")
        self.assertEqual(system.reloads, 0)
        a.ask("again")
        self.assertEqual(system.reloads, 0)           # nothing changed
        index_state.bump(self.state)
        a.ask("after a sync")
        self.assertEqual(system.reloads, 1)
        a.ask("and again")
        self.assertEqual(system.reloads, 1)

    def test_reload_failure_clears_the_system(self):
        class Breaking(FakeSystem):
            def reload(self):
                raise RuntimeError("The 'pka_semantic' index is empty.")
        a = self.make(system_factory=Breaking)
        a.ask("first")
        index_state.bump(self.state)
        r = a.ask("second")
        self.assertIn("The index is empty", r["error"])
        self.assertFalse(a.ready)


class TestLibrary(ServiceCase):
    def test_documents(self):
        docs = {d["doc_id"]: d for d in self.make().documents()}
        self.assertEqual(set(docs), {"rental_agreement", "notes"})
        r = docs["rental_agreement"]
        self.assertEqual((r["versions"], r["latest_version"], r["changes"], r["summarised"]), (2, 2, 2, True))
        self.assertEqual(r["latest_date"], "2026-11-01")
        self.assertEqual(docs["notes"]["versions"], 1)
        self.assertFalse(docs["notes"]["summarised"])

    def test_document_detail(self):
        d = self.make().document("rental_agreement")
        v1, v2 = d["versions"]
        self.assertEqual((v1["version"], v1["status"], v1["until"]), (1, "superseded", "2026-11-01"))
        self.assertEqual((v2["version"], v2["status"], v2["until"], v2["auto"]), (2, "current", None, True))
        self.assertEqual(v2["summary"], {"nodes": 1, "has_overview": True})
        kinds = [(c["kind"], c["section"]) for c in d["changes"]]
        self.assertEqual(kinds, [("modified", "4. Licence Fee"), ("removed", "7. Parking")])
        self.assertEqual(self.make().document("notes")["versions"][0]["status"], "only")
        self.assertIsNone(self.make().document("missing"))

    def test_chunks_and_summaries(self):
        c = self.make().chunks("rental_agreement", 2)
        self.assertEqual([x["text"] for x in c["chunks"]], ["Clause 0 v2", "Clause 1 v2"])
        self.assertEqual(c["chunks"][0]["heading"], "1. Clause")
        self.assertEqual(c["summaries"][0]["text"], "The overview text")
        self.assertTrue(c["summaries"][0]["overview"])
        self.assertEqual(self.make().chunks("rental_agreement", 9)["chunks"], [])

    def test_empty_database_is_fine(self):
        from pka.service import Assistant
        a = Assistant(db_path=Path(self.tmp.name) / "none.db", state_path=self.state)
        self.assertEqual(a.documents(), [])
        self.assertEqual(a.status()["documents"], 0)

    def test_status_counts(self):
        s = self.make(system_factory=FakeSystem).status()
        self.assertEqual((s["documents"], s["versions"], s["chunks"], s["summaries"]), (2, 3, 5, 1))
        self.assertFalse(s["ready"])
        self.assertIsNone(s["last_sync"])


class TestSyncControl(ServiceCase):
    def fake_syncer(self, report):
        syncer = mock.Mock()
        syncer.sync.return_value = report
        return syncer

    def report(self):
        from pka.sync import SyncReport
        return SyncReport(started="2026-11-13T10:00:00", added=[("a", 1, "")])

    def test_run_sync_records_the_result(self):
        a = self.make(system_factory=FakeSystem, syncer_factory=lambda: self.fake_syncer(self.report()))
        a.run_sync()
        s = a.status()
        self.assertIn("1 added", s["last_sync"]["summary"])
        self.assertFalse(s["syncing"])
        self.assertEqual(s["last_sync_error"], "")

    def test_failed_sync_is_remembered_and_raised(self):
        from pka.sync import SyncError
        syncer = mock.Mock()
        syncer.sync.side_effect = SyncError("Can't reach Qdrant")
        a = self.make(syncer_factory=lambda: syncer)
        with self.assertRaises(SyncError):
            a.run_sync()
        self.assertIn("Qdrant", a.status()["last_sync_error"])
        self.assertFalse(a.status()["syncing"])

    def test_second_sync_while_running_is_refused(self):
        a = self.make(syncer_factory=lambda: self.fake_syncer(self.report()))
        a.syncing = True
        with self.assertRaises(SyncBusy):
            a.run_sync()
        self.assertFalse(a.start_sync())

    def test_auto_summaries_budget_is_passed_on(self):
        syncer = self.fake_syncer(self.report())
        a = self.make(syncer_factory=lambda: syncer, auto_summaries=25)
        a.run_sync()
        syncer.sync.assert_called_once_with(summaries_max_calls=25)


# ---------------------------------------------------------------------- web API and page
try:
    from fastapi.testclient import TestClient
    HAVE_FASTAPI = True
except Exception:          # fastapi or httpx not installed
    HAVE_FASTAPI = False


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx not installed")
class TestWebApi(ServiceCase):
    def client(self):
        from pka.api.main import create_app
        return TestClient(create_app(self.make(system_factory=FakeSystem)))

    def test_endpoints(self):
        c = self.client()
        self.assertEqual(c.get("/health").json()["status"], "ok")
        self.assertIn("Personal Knowledge Assistant", c.get("/").text)
        r = c.post("/api/ask", json={"question": "fee?"}).json()
        self.assertEqual(len(r["sources"]), 2)
        self.assertEqual(c.post("/ask", json={"question": "fee?"}).status_code, 200)
        self.assertEqual(c.post("/api/ask", json={"question": "x", "as_of": "nonsense"}).status_code, 422)
        self.assertEqual(len(c.get("/api/documents").json()), 2)
        self.assertEqual(c.get("/api/documents/rental_agreement").json()["versions"][1]["version"], 2)
        self.assertEqual(c.get("/api/documents/nope").status_code, 404)
        chunks = c.get("/api/documents/rental_agreement/chunks").json()      # latest version by default
        self.assertEqual(chunks["version"], 2)
        self.assertEqual(c.get("/api/documents/rental_agreement/chunks?version=1").json()["version"], 1)
        self.assertIn("documents", c.get("/api/status").json())


class TestWebPage(unittest.TestCase):
    PAGE = Path(__file__).resolve().parent.parent / "pka" / "web" / "index.html"

    def test_page_exists(self):
        self.assertIn("<title>Personal Knowledge Assistant</title>", self.PAGE.read_text(encoding="utf-8"))

    @unittest.skipUnless(shutil.which("node"), "node not installed")
    def test_interface_script_against_a_fake_browser(self):
        import subprocess
        smoke = Path(__file__).resolve().parent / "ui_smoke.js"
        out = subprocess.run(["node", str(smoke), str(self.PAGE)], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("ui smoke ok", out.stdout)


# ---------------------------------------------------------------------- reloading real systems
class TestReload(unittest.TestCase):
    def make(self):
        from pka.summary_system import SummarySystem
        s = SummarySystem.__new__(SummarySystem)
        s.collection = "pka_semantic"
        s.ingest_hint = "python -m pka.ingest --chunker semantic --reset"
        self.chunks = {"c1": {"doc_id": "a", "version": 1, "text": "alpha text", "valid_from_day": 0,
                              "valid_to_day": 99}}
        self.sums = {"s1": {"doc_id": "a", "version": 1, "text": "alpha overview", "level": 1}}
        self.summary_exists = True

        def scroll_from(source):
            return lambda coll, **kw: ([SimpleNamespace(id=i, payload=p) for i, p in source().items()], None)
        s.store = SimpleNamespace(client=SimpleNamespace(scroll=scroll_from(lambda: self.chunks)))
        s.summary_store = SimpleNamespace(client=SimpleNamespace(
            scroll=scroll_from(lambda: self.sums), collection_exists=lambda name: self.summary_exists))
        s.registry = SimpleNamespace(versioned_docs=lambda: {"a"})
        s._scope, s._log = {}, {}
        s._build_index()
        s._refresh_versions()
        return s

    def test_reload_sees_new_chunks_and_summaries(self):
        s = self.make()
        self.assertEqual(set(s.ids), {"c1", "s1"})
        self.assertEqual(s.n_summaries, 1)
        self.chunks["c2"] = {"doc_id": "b", "version": 1, "text": "beta text about parking",
                             "valid_from_day": 0, "valid_to_day": 99}
        self.sums["s2"] = {"doc_id": "b", "version": 1, "text": "beta overview", "level": 1}
        s.reload()
        self.assertEqual(set(s.ids), {"c1", "c2", "s1", "s2"})
        self.assertEqual(s.n_summaries, 2)
        top = s.bm25.search("parking", 1)
        self.assertEqual(s.ids[top[0][0]], "c2")                 # keyword index was rebuilt too

    def test_reload_drops_removed_documents(self):
        s = self.make()
        del self.chunks["c1"], self.sums["s1"]
        self.chunks["c9"] = {"doc_id": "z", "version": 1, "text": "only", "valid_from_day": 0, "valid_to_day": 99}
        s.reload()
        self.assertEqual(set(s.ids), {"c9"})

    def test_empty_index_raises_a_clear_error(self):
        s = self.make()
        self.chunks.clear()
        with self.assertRaisesRegex(RuntimeError, "index is empty"):
            s.reload()

    def test_summaries_collection_missing_raises(self):
        s = self.make()
        self.summary_exists = False
        with self.assertRaisesRegex(RuntimeError, "No summaries yet"):
            s.reload()


if __name__ == "__main__":
    unittest.main()
