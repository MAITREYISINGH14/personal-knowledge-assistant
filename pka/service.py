"""Everything the web app needs, as plain Python (so it can be tested without a web server).

The Assistant
  - loads the best available retrieval system (summaries if built, else time-aware search),
  - reloads it automatically when the index changes (a sync finished, summaries were built),
  - answers questions, returning the retrieved sources even if the LLM is unavailable,
  - reads documents, version history, change logs and chunks for the interface,
  - runs syncs and remembers the last result.
"""
import sqlite3
import threading
from datetime import datetime
from typing import Optional

from pka import index_state
from pka.config import settings

MAX_QUESTION_CHARS = 2000


def _heading(section: str) -> str:
    """Last part of the first section label: 'Doc > 4. Licence Fee | ...' -> '4. Licence Fee'."""
    first = (section or "").split(" | ")[0]
    return first.split(" > ")[-1].strip()


class Assistant:
    def __init__(self, system_factory=None, syncer_factory=None, db_path=None, state_path=None,
                 auto_summaries: Optional[int] = None):
        self._system_factory = system_factory
        self._syncer_factory = syncer_factory
        self.db_path = db_path or settings.sqlite_path
        self.state_path = state_path
        self.auto_summaries = auto_summaries
        self._lock = threading.RLock()          # guards the retrieval system
        self._system = None
        self._generation = None
        self._failed_generation = None
        self._system_error = ""
        self._syncer = None
        self._state = threading.Lock()          # guards the fields below
        self.syncing = False
        self.watching = False
        self.last_sync = None                   # SyncReport.to_dict() of the last good sync
        self.last_sync_error = ""
        self._pending = None
        self._pending_generation = None

    # ------------------------------------------------------------------ the retrieval system
    def _make_system(self):
        if self._system_factory:
            return self._system_factory()
        try:
            from pka.summary_system import SummarySystem
            return SummarySystem()
        except RuntimeError as e:
            if "No summaries" not in str(e):
                raise
            from pka.temporal import TemporalSystem
            return TemporalSystem()

    def _summaries_now_available(self, system) -> bool:
        try:
            client = system.store.client
            return bool(client.collection_exists(settings.summary_collection))
        except Exception:
            return False

    def system(self):
        """The loaded system, reloaded first if the index changed since it was loaded."""
        with self._lock:
            gen = index_state.current(self.state_path)
            if self._system is None:
                if self._failed_generation == gen and self._system_error:
                    raise RuntimeError(self._system_error)
                self._load(gen)
            elif gen != self._generation:
                self._refresh(gen)
            return self._system

    def _load(self, gen):
        try:
            self._system = self._make_system()
            self._generation, self._system_error, self._failed_generation = gen, "", None
        except Exception as e:
            self._system, self._system_error, self._failed_generation = None, str(e), gen
            raise RuntimeError(str(e)) from e

    def _refresh(self, gen):
        try:
            if getattr(self._system, "name", "") == "temporal" and self._summaries_now_available(self._system):
                self._system = self._make_system()           # upgrade: summaries now exist
            else:
                self._system.reload()
            self._generation = gen
        except Exception as e:                               # e.g. every document was removed
            self._system, self._system_error, self._failed_generation = None, str(e), gen
            raise RuntimeError(str(e)) from e

    def warm_up(self):
        """Load models ahead of the first question. Errors are kept for the status display."""
        try:
            self.system()
        except Exception:
            pass

    @property
    def ready(self) -> bool:
        return self._system is not None

    # ------------------------------------------------------------------ asking
    def ask(self, question: str, as_of: Optional[str] = None) -> dict:
        question = (question or "").strip()
        if not question:
            return self._reply(error="Type a question first.")
        if len(question) > MAX_QUESTION_CHARS:
            return self._reply(error=f"Questions are limited to {MAX_QUESTION_CHARS} characters.")
        try:
            system = self.system()
        except Exception as e:
            return self._reply(error=self._explain(e))
        from pka.llm import LLMError
        with self._lock:
            try:
                chunks = system.retrieve(question, as_of)
            except Exception as e:
                return self._reply(error=self._explain(e))
            time_frame = system.describe_scope(question) if hasattr(system, "describe_scope") else ""
            answer, error = "", ""
            if getattr(system, "answerer", None):
                try:
                    answer = system.generate(question, chunks)
                except LLMError as e:
                    error = f"The answer model isn't available right now ({e}). The passages found are shown below."
            else:
                error = "No answer model is configured, so only the retrieved passages are shown."
        return self._reply(answer=answer, error=error, time_frame=time_frame,
                           sources=[self._source(i, c) for i, c in enumerate(chunks, 1)])

    @staticmethod
    def _reply(answer="", error="", time_frame="", sources=None) -> dict:
        return {"answer": answer, "error": error, "time_frame": time_frame, "sources": sources or []}

    @staticmethod
    def _explain(e: Exception) -> str:
        text = str(e)
        if "index is empty" in text or "index found" in text:
            return ("The index is empty, so there is nothing to search yet. Put documents in the "
                    f"corpus folder ({settings.corpus_dir}); they are picked up automatically, or "
                    "press 'Sync now'.")
        if "Connection refused" in text or "Can't reach" in text:
            return ("Can't reach Qdrant. Open Docker Desktop, wait for 'Engine running', then run: "
                    "docker compose up -d")
        return f"{type(e).__name__}: {text}" if not isinstance(e, RuntimeError) else text

    @staticmethod
    def _source(n: int, c) -> dict:
        meta = getattr(c, "meta", None) or {}
        return {"n": n, "doc_id": c.doc_id, "version": c.version, "section": c.section,
                "heading": _heading(c.section), "level": getattr(c, "level", 0),
                "status": meta.get("status"), "in_force": meta.get("in_force", ""),
                "score": round(float(c.score), 4), "text": c.text}

    # ------------------------------------------------------------------ reading the library
    def _rows(self, sql, params=()):
        conn = sqlite3.connect(self.db_path)
        try:
            return conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []                    # a table that doesn't exist yet means "nothing yet"
        finally:
            conn.close()

    def documents(self) -> list:
        coll = settings.semantic_collection
        rows = self._rows(
            "SELECT d.doc_id, d.version, d.title, d.doc_date, d.n_chunks, d.effective_date "
            "FROM documents d JOIN ingested i ON i.doc_id=d.doc_id AND i.version=d.version "
            "AND i.collection=? ORDER BY d.doc_id, d.version", (coll,))
        roots = {(r[0], r[1]) for r in self._rows(
            "SELECT doc_id, version FROM summary_nodes WHERE collection=? AND is_root=1",
            (settings.summary_collection,))}
        changes = {}
        for doc_id, n in self._rows("SELECT doc_id, COUNT(*) FROM changes WHERE collection=? "
                                    "GROUP BY doc_id", (coll,)):
            changes[doc_id] = n
        docs = {}
        for doc_id, version, title, doc_date, n_chunks, edit_date in rows:
            d = docs.setdefault(doc_id, {"doc_id": doc_id, "versions": 0, "chunks": 0})
            d.update(title=title or doc_id, latest_version=version, latest_date=edit_date or doc_date,
                     chunks=n_chunks or 0, summarised=(doc_id, version) in roots)
            d["versions"] += 1
            d["changes"] = changes.get(doc_id, 0)
        return list(docs.values())

    def document(self, doc_id: str) -> Optional[dict]:
        coll = settings.semantic_collection
        vers = self._rows("SELECT version, effective, valid_to FROM versions WHERE collection=? "
                          "AND doc_id=? ORDER BY effective, version", (coll, doc_id))
        info = {r[0]: r for r in self._rows(
            "SELECT d.version, d.title, d.doc_date, d.n_chunks, d.effective_date, d.file_mtime "
            "FROM documents d JOIN ingested i ON i.doc_id=d.doc_id AND i.version=d.version "
            "AND i.collection=? WHERE d.doc_id=? ORDER BY d.version", (coll, doc_id))}
        if not info:
            return None
        timeline = {v: (eff, until) for v, eff, until in vers}
        summaries = {}
        for version, n, root in self._rows(
                "SELECT version, COUNT(*), MAX(is_root) FROM summary_nodes WHERE collection=? "
                "AND doc_id=? GROUP BY version", (settings.summary_collection, doc_id)):
            summaries[version] = {"nodes": n, "has_overview": bool(root)}
        many = len(info) > 1
        versions = []
        for v in sorted(info, key=lambda v: (timeline.get(v, ("", None))[0] or "", v)):
            _, title, doc_date, n_chunks, edit_date, mtime = info[v]
            eff, until = timeline.get(v, (edit_date or doc_date or (mtime or "")[:10], None))
            versions.append({
                "version": v, "title": title, "from": eff, "until": until, "chunks": n_chunks or 0,
                "status": ("current" if until is None else "superseded") if many else "only",
                "auto": bool(edit_date), "summary": summaries.get(v)})
        changes = [{"from": a, "to": b, "kind": kind, "section": _heading(new_sec if kind != "removed" else old_sec),
                    "detail": detail}
                   for a, b, kind, old_sec, new_sec, detail in self._rows(
                       "SELECT from_version, to_version, kind, old_section, new_section, detail "
                       "FROM changes WHERE collection=? AND doc_id=? ORDER BY from_version, rowid",
                       (coll, doc_id))]
        return {"doc_id": doc_id, "title": versions[-1]["title"], "versions": versions, "changes": changes}

    def chunks(self, doc_id: str, version: int) -> dict:
        rows = self._rows("SELECT chunk_index, section, n_tokens, text FROM chunks WHERE collection=? "
                          "AND doc_id=? AND version=? ORDER BY chunk_index",
                          (settings.semantic_collection, doc_id, version))
        summaries = self._rows(
            "SELECT n.level, n.idx, n.is_root, n.section, c.summary FROM summary_nodes n "
            "JOIN summary_cache c ON c.hash=n.hash WHERE n.collection=? AND n.doc_id=? AND n.version=? "
            "ORDER BY n.level DESC, n.idx", (settings.summary_collection, doc_id, version))
        return {"doc_id": doc_id, "version": version,
                "chunks": [{"index": i, "section": s or "", "heading": _heading(s), "tokens": t, "text": x}
                           for i, s, t, x in rows],
                "summaries": [{"level": lv, "part": ix + 1, "overview": bool(root), "span": sec, "text": txt}
                              for lv, ix, root, sec, txt in summaries]}

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        coll = settings.semantic_collection
        one = lambda sql, p=(): (self._rows(sql, p) or [(0,)])[0][0]
        gen = index_state.current(self.state_path)
        if self._pending_generation != gen:
            self._pending, self._pending_generation = self._compute_pending(), gen
        with self._state:
            last, err, syncing, watching = self.last_sync, self.last_sync_error, self.syncing, self.watching
        return {
            "ready": self.ready, "problem": self._system_error,
            "system": getattr(self._system, "name", None),
            "documents": one("SELECT COUNT(DISTINCT doc_id) FROM ingested WHERE collection=?", (coll,)),
            "versions": one("SELECT COUNT(*) FROM ingested WHERE collection=?", (coll,)),
            "chunks": one("SELECT COUNT(*) FROM chunks WHERE collection=?", (coll,)),
            "summaries": one("SELECT COUNT(*) FROM summary_nodes WHERE collection=?",
                             (settings.summary_collection,)),
            "pending": self._pending, "syncing": syncing, "watching": watching,
            "last_sync": ({"at": last["started"], "summary": last["one_line"], "warnings": last["warnings"],
                           "failed": last["failed"]} if last else None),
            "last_sync_error": err, "llm": settings.llm_provider, "corpus_dir": str(settings.corpus_dir),
            "server_time": datetime.now().isoformat(timespec="seconds"),
        }

    def _compute_pending(self):
        try:
            from pka.summarize import SummaryBuilder
            b = SummaryBuilder(db_path=self.db_path)
            try:
                return b.status()
            finally:
                b.close()
        except Exception:
            return None

    # ------------------------------------------------------------------ syncing
    def _get_syncer(self):
        with self._state:
            if self._syncer is None:
                if self._syncer_factory:
                    self._syncer = self._syncer_factory()
                else:
                    from pka.sync import Syncer
                    self._syncer = Syncer(embedder=getattr(self._system, "embedder", None))
            return self._syncer

    def run_sync(self):
        """Sync now, in the calling thread. Raises SyncBusy or SyncError on failure."""
        from pka.sync import SyncBusy
        syncer = self._get_syncer()
        with self._state:
            if self.syncing:
                raise SyncBusy("A sync is already running.")
            self.syncing = True
        try:
            report = syncer.sync(summaries_max_calls=self.auto_summaries)
        except SyncBusy:
            raise
        except Exception as e:
            with self._state:
                self.last_sync_error = str(e)
            raise
        else:
            with self._state:
                self.last_sync, self.last_sync_error = report.to_dict(), ""
            return report
        finally:
            with self._state:
                self.syncing = False

    def start_sync(self) -> bool:
        """Start a sync in the background (for the 'Sync now' button). False if one is running."""
        with self._state:
            if self.syncing:
                return False

        def work():
            try:
                self.run_sync()
            except Exception:
                pass                    # recorded in last_sync_error
        threading.Thread(target=work, daemon=True, name="pka-sync").start()
        return True
