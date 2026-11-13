"""SQLite metadata store: the source of truth for documents and chunk text."""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT NOT NULL,
    version     INTEGER NOT NULL,
    path        TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    title       TEXT,
    doc_date    TEXT,          -- date stated inside the document (used for versioning in Phase 5)
    file_mtime  TEXT,
    ingested_at TEXT NOT NULL,
    n_chunks    INTEGER,
    effective_date TEXT,       -- set for versions created automatically from an edit (Phase 6)
    PRIMARY KEY (doc_id, version)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    collection  TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    version     INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    section     TEXT,
    text        TEXT NOT NULL,
    n_words     INTEGER,
    n_tokens    INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (collection, doc_id, version);
CREATE TABLE IF NOT EXISTS sync_events (  -- permanent history of what each sync did
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    at         TEXT NOT NULL,
    collection TEXT NOT NULL,
    kind       TEXT NOT NULL,           -- added | new_version | updated | removed | failed
    doc_id     TEXT,
    version    INTEGER,
    file       TEXT,
    note       TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_doc ON sync_events (collection, doc_id);
CREATE TABLE IF NOT EXISTS ingested (     -- what each index was built from
    collection TEXT NOT NULL,
    doc_id     TEXT NOT NULL,
    version    INTEGER NOT NULL,
    sha256     TEXT NOT NULL,
    PRIMARY KEY (collection, doc_id, version)
);
"""


def ensure_effective_date(conn):
    """Older databases (before Phase 6) lack the effective_date column: add it."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
    if cols and "effective_date" not in cols:
        conn.execute("ALTER TABLE documents ADD COLUMN effective_date TEXT")
        conn.commit()


class MetadataStore:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the web app, the folder watcher and sync use it from
        # different threads (never at the same instant for writes: one sync runs at a time).
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        ensure_effective_date(self.conn)

    def reset(self, collection: str):
        self.conn.execute("DELETE FROM chunks WHERE collection = ?", (collection,))
        self.conn.execute("DELETE FROM ingested WHERE collection = ?", (collection,))
        self.conn.commit()

    def stored_hash(self, doc_id: str, version: int, collection: str):
        row = self.conn.execute(
            "SELECT sha256 FROM ingested WHERE collection=? AND doc_id=? AND version=?",
            (collection, doc_id, version)).fetchone()
        return row["sha256"] if row else None

    def save_document(self, doc, collection: str, chunk_rows: list, effective_date=None):
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE collection=? AND doc_id=? AND version=?",
                              (collection, doc.doc_id, doc.version))
            self.conn.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
                [(r["chunk_id"], collection, doc.doc_id, doc.version, r["chunk_index"],
                  r["section"], r["text"], r["n_words"], r["n_tokens"]) for r in chunk_rows])
            self.conn.execute("INSERT OR REPLACE INTO ingested VALUES (?,?,?,?)",
                              (collection, doc.doc_id, doc.version, doc.sha256))
            self.conn.execute(
                "INSERT INTO documents (doc_id, version, path, sha256, title, doc_date, file_mtime, "
                "ingested_at, n_chunks, effective_date) VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(doc_id, version) DO UPDATE SET path=excluded.path, sha256=excluded.sha256, "
                "title=excluded.title, doc_date=excluded.doc_date, file_mtime=excluded.file_mtime, "
                "ingested_at=excluded.ingested_at, n_chunks=excluded.n_chunks, "
                "effective_date=COALESCE(excluded.effective_date, documents.effective_date)",
                (doc.doc_id, doc.version, str(doc.path), doc.sha256, doc.title,
                 doc.doc_date, doc.file_mtime, now, len(chunk_rows), effective_date))

    def counts(self, collection: str):
        docs = self.conn.execute("SELECT COUNT(*) FROM ingested WHERE collection=?",
                                 (collection,)).fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks WHERE collection=?",
                                   (collection,)).fetchone()[0]
        return docs, chunks

    # ------------------------------------------------------------------ Phase 6 helpers
    def latest_version(self, doc_id: str, collection: str):
        row = self.conn.execute("SELECT MAX(version) FROM ingested WHERE collection=? AND doc_id=?",
                                (collection, doc_id)).fetchone()
        return row[0]

    def ingested_rows(self, collection: str):
        return [(r[0], r[1]) for r in self.conn.execute(
            "SELECT doc_id, version FROM ingested WHERE collection=? ORDER BY doc_id, version",
            (collection,))]

    def effective_of(self, doc_id: str, version: int):
        """ISO date from which a version applies: its edit date, the date inside it, or its file date."""
        row = self.conn.execute(
            "SELECT effective_date, doc_date, file_mtime FROM documents WHERE doc_id=? AND version=?",
            (doc_id, version)).fetchone()
        if not row:
            return None
        if row[0] or row[1]:
            return row[0] or row[1]
        return datetime.fromisoformat(row[2]).astimezone().date().isoformat() if row[2] else None

    def auto_versions(self, collection: str):
        """Versions created automatically from edits. Their history exists only in the index."""
        return [(r[0], r[1]) for r in self.conn.execute(
            "SELECT d.doc_id, d.version FROM documents d JOIN ingested i ON i.doc_id=d.doc_id "
            "AND i.version=d.version AND i.collection=? WHERE d.effective_date IS NOT NULL",
            (collection,))]

    def remove_document(self, doc_id: str, version: int, collection: str):
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE collection=? AND doc_id=? AND version=?",
                              (collection, doc_id, version))
            self.conn.execute("DELETE FROM ingested WHERE collection=? AND doc_id=? AND version=?",
                              (collection, doc_id, version))
            left = self.conn.execute("SELECT 1 FROM ingested WHERE doc_id=? AND version=? LIMIT 1",
                                     (doc_id, version)).fetchone()
            if not left:      # no other index (e.g. the baseline) still uses this document
                self.conn.execute("DELETE FROM documents WHERE doc_id=? AND version=?",
                                  (doc_id, version))

    # ------------------------------------------------------------------ sync history (Phase 7)
    def log_events(self, collection: str, events: list, at: str = None):
        """events: [(kind, doc_id, version, file, note)]. Kept forever, so removals stay explainable."""
        if not events:
            return
        at = at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.conn:
            self.conn.executemany(
                "INSERT INTO sync_events (at, collection, kind, doc_id, version, file, note) "
                "VALUES (?,?,?,?,?,?,?)", [(at, collection, *e) for e in events])

    def events(self, collection: str, doc_id: str = None, limit: int = 50):
        sql = "SELECT at, kind, doc_id, version, file, note FROM sync_events WHERE collection=?"
        params = [collection]
        if doc_id is not None:
            sql += " AND doc_id=?"
            params.append(doc_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [dict(zip(("at", "kind", "doc_id", "version", "file", "note"), r))
                for r in self.conn.execute(sql, params)]
