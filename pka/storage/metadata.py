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
"""


class MetadataStore:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def reset(self, collection: str):
        self.conn.execute("DELETE FROM chunks WHERE collection = ?", (collection,))
        self.conn.execute("DELETE FROM documents")
        self.conn.commit()

    def stored_hash(self, doc_id: str, version: int):
        row = self.conn.execute("SELECT sha256 FROM documents WHERE doc_id=? AND version=?",
                                (doc_id, version)).fetchone()
        return row["sha256"] if row else None

    def save_document(self, doc, collection: str, chunk_rows: list):
        now = datetime.now(timezone.utc).isoformat()
        with self.conn:
            self.conn.execute("DELETE FROM chunks WHERE collection=? AND doc_id=? AND version=?",
                              (collection, doc.doc_id, doc.version))
            self.conn.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
                [(r["chunk_id"], collection, doc.doc_id, doc.version, r["chunk_index"],
                  r["section"], r["text"], r["n_words"], r["n_tokens"]) for r in chunk_rows])
            self.conn.execute(
                "INSERT OR REPLACE INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
                (doc.doc_id, doc.version, str(doc.path), doc.sha256, doc.title,
                 doc.doc_date, doc.file_mtime, now, len(chunk_rows)))

    def counts(self, collection: str):
        docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks WHERE collection=?",
                                   (collection,)).fetchone()[0]
        return docs, chunks
