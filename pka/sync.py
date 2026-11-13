"""Keep the semantic index in step with the corpus folder.

  python -m pka.sync                 # one pass: add new files, update changed ones, remove deleted ones
  python -m pka.sync --auto-summaries 30   # also build summaries for what changed (max 30 API calls)

What happens to each file
  new file                     -> added
  unchanged file               -> skipped (its hash matches)
  report.md, edited in place   -> NEW VERSION: the old one stays in the index as history, so
                                  "what changed?" and "as of last month" keep working
  report__v2.md, edited        -> that version is overwritten (you manage explicit versions yourself)
  file deleted                 -> its entries are removed (all versions, if it was a plain name)

Safety rails: nothing is removed if the corpus folder looks empty, or if a pass would remove more
than half of the index (pass --force-prune to override).
"""
import argparse
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from pka import index_state
from pka.config import settings
from pka.ingestion.loaders import VERSION_RE, discover, file_sha, load_file, parse_name
from pka.storage.metadata import MetadataStore


class SyncError(RuntimeError):
    """Sync can't run at all (e.g. Qdrant is down)."""


class SyncBusy(RuntimeError):
    """Another sync is already running."""


@dataclass
class SyncReport:
    started: str = ""
    seconds: float = 0.0
    added: list = field(default_factory=list)       # [(doc_id, version, note)]
    versioned: list = field(default_factory=list)   # new versions made from edits
    updated: list = field(default_factory=list)     # explicit versions overwritten in place
    removed: list = field(default_factory=list)
    unchanged: int = 0
    failed: list = field(default_factory=list)      # [(file name, reason)]
    warnings: list = field(default_factory=list)
    pending: dict = None                            # summaries still to build
    summaries: dict = None                          # summaries built during this sync

    @property
    def changed(self) -> bool:
        return bool(self.added or self.versioned or self.updated or self.removed)

    def one_line(self) -> str:
        parts = [f"{len(self.added)} added", f"{len(self.versioned)} new version(s)",
                 f"{len(self.updated)} updated", f"{len(self.removed)} removed",
                 f"{self.unchanged} unchanged"]
        if self.failed:
            parts.append(f"{len(self.failed)} failed")
        return ", ".join(parts)

    def text(self) -> str:
        lines = []
        for label, items in (("added", self.added), ("new version", self.versioned),
                             ("updated", self.updated), ("removed", self.removed)):
            for doc_id, version, note in items:
                lines.append(f"  {label:<12}{doc_id} v{version}" + (f"  ({note})" if note else ""))
        lines += [f"  FAILED      {name}: {why}" for name, why in self.failed]
        lines += [f"  note        {w}" for w in self.warnings]
        head = f"Sync finished in {self.seconds:.0f}s: {self.one_line()}."
        tail = []
        if self.summaries:
            s = self.summaries
            tail.append(f"Summaries: {s['documents']} document(s) built with {s['calls']} API call(s)"
                        + (f"; stopped early: {s['stopped']}" if s.get("stopped") else "") + ".")
        if self.pending and self.pending.get("docs_pending"):
            p = self.pending
            tail.append(f"{p['docs_pending']} document(s) still need summaries "
                        f"(~{p['calls_needed']} API calls): run python -m pka.summarize")
        return "\n".join([head] + lines + tail)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["changed"] = self.changed
        d["text"] = self.text()
        d["one_line"] = self.one_line()
        return d


class Syncer:
    def __init__(self, embedder=None, *, plan=None, store=None, meta=None, corpus_dir=None,
                 rebuild_versions=None, summary_status=None, state_path=None):
        """All keyword arguments exist so tests can substitute fakes."""
        self._embedder = embedder
        self._plan, self._store, self._meta = plan, store, meta
        self.corpus_dir = Path(corpus_dir or settings.corpus_dir)
        self._rebuild_versions = rebuild_versions
        self._summary_status = summary_status
        self._state_path = state_path
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def _ready(self):
        if self._plan is None:
            from pka.ingest import semantic_plan
            self._plan = semantic_plan(embedder=self._embedder)
        coll, embedder, _ = self._plan
        if self._store is None:
            from pka.storage.vectors import VectorStore
            store = VectorStore(settings.qdrant_url, coll, embedder.dim)
            try:
                store.ensure()
            except Exception as e:
                raise SyncError(
                    f"Can't reach Qdrant at {settings.qdrant_url} ({type(e).__name__}). Open Docker "
                    "Desktop, wait for 'Engine running', then run: docker compose up -d") from e
            self._store = store
        if self._meta is None:
            self._meta = MetadataStore(settings.sqlite_path)

    # ------------------------------------------------------------------ public
    def sync(self, prune=True, force_prune=False, summaries_max_calls=None, log=print) -> SyncReport:
        if not self._lock.acquire(blocking=False):
            raise SyncBusy("A sync is already running.")
        try:
            return self._sync(prune, force_prune, summaries_max_calls, log)
        finally:
            self._lock.release()

    # ------------------------------------------------------------------ the work
    def _sync(self, prune, force_prune, summaries_max_calls, log) -> SyncReport:
        from pka.ingest import ingest_document
        t0 = time.perf_counter()
        rep = SyncReport(started=datetime.now().isoformat(timespec="seconds"))
        self._ready()
        coll, embedder, chunk = self._plan
        meta, store = self._meta, self._store
        files = discover(self.corpus_dir)

        # 1. Work out what each file is. Files named name__vN.ext are explicit versions;
        #    plain names are one document whose edits become new versions.
        explicit, auto = {}, {}
        for path in files:
            doc_id, ver = parse_name(path.stem)
            if VERSION_RE.match(path.stem):
                if (doc_id, ver) in explicit:
                    rep.warnings.append(f"{path.name} duplicates {explicit[(doc_id, ver)].name}; skipped")
                else:
                    explicit[(doc_id, ver)] = path
            elif doc_id in auto:
                rep.warnings.append(f"{path.name} has the same name as {auto[doc_id].name}; skipped")
            else:
                auto[doc_id] = path
        present_explicit = {parse_name(p.stem) for p in files if VERSION_RE.match(p.stem)}
        present_plain_ids = {parse_name(p.stem)[0] for p in files if not VERSION_RE.match(p.stem)}
        explicit_ids = {d for d, _ in explicit}
        for doc_id in sorted(set(auto) & explicit_ids):
            rep.warnings.append(f"{auto[doc_id].name} clashes with versioned files named {doc_id}__vN; "
                                "skipped. Rename one of them.")
            del auto[doc_id]

        def ingest(doc, effective=None):
            n, _stats, _over = ingest_document(doc, chunk, embedder, store, meta, coll, effective)
            return n

        # 2a. Explicit versions: new -> add; changed -> overwrite that version.
        for (doc_id, ver), path in sorted(explicit.items()):
            try:
                sha = file_sha(path)
                stored = meta.stored_hash(doc_id, ver, coll)
                if stored == sha:
                    rep.unchanged += 1
                    continue
                if ingest(load_file(path)) == 0:
                    rep.failed.append((path.name, "no text could be extracted (scanned PDF?)"))
                    continue
                if stored is None:
                    rep.added.append((doc_id, ver, ""))
                else:
                    self._drop_summaries(doc_id, ver)
                    rep.updated.append((doc_id, ver, "edited in place; this version was overwritten"))
            except Exception as e:      # one bad file must not stop the rest
                rep.failed.append((path.name, f"{type(e).__name__}: {e}"))

        # 2b. Plain names: new -> v1; changed -> a NEW version, keeping the old as history.
        for doc_id, path in sorted(auto.items()):
            try:
                sha = file_sha(path)
                latest = meta.latest_version(doc_id, coll)
                if latest is not None and meta.stored_hash(doc_id, latest, coll) == sha:
                    rep.unchanged += 1
                    continue
                doc = load_file(path)
                if latest is None:
                    if ingest(doc) == 0:
                        rep.failed.append((path.name, "no text could be extracted (scanned PDF?)"))
                    else:
                        rep.added.append((doc_id, doc.version, ""))
                    continue
                doc.version, doc.versioned = latest + 1, True
                effective = self._edit_date(doc, meta.effective_of(doc_id, latest))
                if ingest(doc, effective) == 0:
                    rep.failed.append((path.name, "no text could be extracted (scanned PDF?)"))
                else:
                    rep.versioned.append((doc_id, doc.version,
                                          f"edited; v{latest} kept as history, v{doc.version} dated {effective}"))
            except Exception as e:
                rep.failed.append((path.name, f"{type(e).__name__}: {e}"))

        # 3. Remove what no longer exists on disk (with safety rails).
        if prune:
            stored = meta.ingested_rows(coll)
            orphans = [(d, v) for d, v in stored
                       if (d, v) not in present_explicit and d not in present_plain_ids]
            if orphans and not files:
                rep.warnings.append("The corpus folder looks empty, so nothing was removed "
                                    f"({len(orphans)} entries kept). Is {self.corpus_dir} the right folder?")
                orphans = []
            elif orphans and not force_prune and len(orphans) > max(3, len(stored) // 2):
                rep.warnings.append(f"This would remove {len(orphans)} of {len(stored)} indexed versions, "
                                    "which looks like a mistake, so nothing was removed. If it is "
                                    "intended, run: python -m pka.sync --force-prune")
                orphans = []
            for doc_id, ver in orphans:
                store.delete_document(doc_id, ver)
                self._drop_summaries(doc_id, ver)
                meta.remove_document(doc_id, ver, coll)
                rep.removed.append((doc_id, ver, "file no longer on disk"))

        # 4. Refresh version dates and change logs.
        if rep.changed:
            log("Updating version information...")
            self._do_rebuild()

        # 5. Optional summaries for what changed, then report what's still pending.
        if summaries_max_calls and rep.changed:
            rep.summaries = self._build_summaries(summaries_max_calls, rep)
        rep.pending = self._pending(rep)

        if rep.changed or rep.summaries:
            index_state.bump(self._state_path)       # tell a running app to reload
        rep.seconds = time.perf_counter() - t0
        return rep

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _edit_date(doc, previous: str):
        """Date from which an edited version applies. Uses the 'Last updated' date inside the
        document if it moved forward, otherwise the day the file was saved; never earlier than
        the previous version."""
        saved = datetime.fromisoformat(doc.file_mtime).astimezone().date().isoformat()
        header = doc.doc_date
        candidate = header if header and (not previous or header > previous) else saved
        return max(candidate, previous) if previous else candidate

    def _client(self):
        return getattr(self._store, "client", None)

    def _drop_summaries(self, doc_id, version):
        client = self._client()
        if client is None:
            return
        from pka.summarize import drop_summaries
        try:
            drop_summaries(self._meta.conn, client, doc_id, version)
        except Exception:      # summaries are optional; never let this break a sync
            pass

    def _do_rebuild(self):
        if self._rebuild_versions:
            self._rebuild_versions()
            return
        from pka.versioning import VersionRegistry
        reg = VersionRegistry(self._plan[0], client=self._store.client)
        try:
            reg.rebuild(verbose=False)
        finally:
            reg.conn.close()

    def _build_summaries(self, max_calls, rep):
        try:
            from pka.summarize import SummaryBuilder
            builder = SummaryBuilder(embedder=self._plan[1])
            done, calls, stopped = builder.build(max_calls=max_calls)
            return {"documents": done, "calls": calls, "stopped": stopped}
        except Exception as e:
            rep.warnings.append(f"Summaries were not updated: {e}")
            return None

    def _pending(self, rep):
        try:
            if self._summary_status:
                return self._summary_status()
            from pka.summarize import SummaryBuilder
            builder = SummaryBuilder()
            try:
                return builder.status()
            finally:
                builder.close()
        except Exception:
            return None


def main():
    ap = argparse.ArgumentParser(description="Sync the index with the corpus folder.")
    ap.add_argument("--no-prune", action="store_true", help="don't remove entries for deleted files")
    ap.add_argument("--force-prune", action="store_true", help="allow removing more than half the index")
    ap.add_argument("--auto-summaries", type=int, metavar="N",
                    help="also build summaries for changed documents, using at most N API calls")
    args = ap.parse_args()
    try:
        report = Syncer().sync(prune=not args.no_prune, force_prune=args.force_prune,
                               summaries_max_calls=args.auto_summaries)
    except (SyncError, SyncBusy) as e:
        raise SystemExit(str(e))
    print(report.text())


if __name__ == "__main__":
    main()
