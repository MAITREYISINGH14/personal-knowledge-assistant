"""Phase 5: document versions over time.

For every document with several versions this module
  1. orders the versions by the date written inside them (falling back to file date),
  2. gives every chunk a validity period: valid_from = its version's date,
     valid_to = the next version's date (or "still current"),
  3. diffs consecutive versions chunk by chunk: unchanged / modified / added / removed,
     and stores a readable change log in SQLite.

Runs automatically after `python -m pka.ingest --chunker semantic`, or by hand:
  python -m pka.versioning                       # rebuild validity + change logs
  python -m pka.versioning --show rental_agreement
"""
import argparse
import difflib
import hashlib
import re
import sqlite3
from collections import defaultdict
from datetime import date, datetime

import numpy as np

from pka.config import settings

ALWAYS = 99_999_999          # "valid forever" as an ordinal day
MODIFIED_SIMILARITY = 0.80   # chunks at least this similar count as the same passage, edited

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    collection TEXT, doc_id TEXT, version INTEGER, effective TEXT, valid_to TEXT,
    date_source TEXT, PRIMARY KEY (collection, doc_id, version));
CREATE TABLE IF NOT EXISTS changes (
    collection TEXT, doc_id TEXT, from_version INTEGER, to_version INTEGER, kind TEXT,
    old_chunk TEXT, new_chunk TEXT, old_section TEXT, new_section TEXT,
    similarity REAL, detail TEXT);
CREATE INDEX IF NOT EXISTS idx_changes ON changes (collection, doc_id);
"""


def day(d: date) -> int:
    return d.toordinal()


def _short(section: str) -> str:
    first = (section or "").split(" | ")[0]
    return first.split(" > ")[-1] if first else "(no heading)"


def _clip(text: str, words: int = 30) -> str:
    w = text.split()
    return " ".join(w[:words]) + (" …" if len(w) > words else "")


def word_diff(old: str, new: str, max_ops: int = 5) -> str:
    """Compact description of what changed between two passages."""
    a, b = old.split(), new.split()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    if sm.ratio() < 0.5:
        return f"rewritten. Before: \"{_clip(old)}\" After: \"{_clip(new)}\""
    parts = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        before, after = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        ctx = " ".join(a[max(0, i1 - 4):i1])
        if op == "replace":
            parts.append(f"…{ctx} \"{_clip(before, 15)}\" → \"{_clip(after, 15)}\"")
        elif op == "delete":
            parts.append(f"…{ctx} removed \"{_clip(before, 15)}\"")
        else:
            parts.append(f"…{ctx} added \"{_clip(after, 15)}\"")
        if len(parts) == max_ops:
            parts.append("(more edits)")
            break
    return "; ".join(parts)


def _section_names(section: str):
    names = set()
    for part in (section or "").split(" | "):
        name = part.split(" > ")[-1].strip().lower()
        name = re.sub(r"^\d+[.)]\s*", "", name)          # "7. Parking" -> "parking"
        if name and not name.startswith("p."):
            names.add(name)
    return names


def same_passage(a, b) -> bool:
    """Embeddings alone over-match: chunks of one document share a header and vocabulary.
    Require a shared section name, or genuine word overlap between the two texts."""
    if _section_names(a["section"]) & _section_names(b["section"]):
        return True
    ratio = difflib.SequenceMatcher(a=a["text"].split(), b=b["text"].split(), autojunk=False).ratio()
    return ratio >= 0.5


def diff_chunks(old, new):
    """old/new: lists of dicts with id, text, section, vector. Returns change records."""
    out, used_old, used_new = [], set(), set()
    by_hash = defaultdict(list)
    for i, c in enumerate(old):
        by_hash[hashlib.sha1(c["text"].encode()).hexdigest()].append(i)
    for j, c in enumerate(new):                                   # 1. identical text
        h = hashlib.sha1(c["text"].encode()).hexdigest()
        if by_hash[h]:
            i = by_hash[h].pop(0)
            used_old.add(i), used_new.add(j)
    if old and new:                                               # 2. best similar pairs
        A = np.array([c["vector"] for c in old], dtype=np.float32)
        B = np.array([c["vector"] for c in new], dtype=np.float32)
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
        B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
        sim = A @ B.T
        pairs = sorted(((sim[i, j], i, j) for i in range(len(old)) for j in range(len(new))
                        if i not in used_old and j not in used_new), reverse=True)
        for s, i, j in pairs:
            if s < MODIFIED_SIMILARITY:
                break
            if i in used_old or j in used_new:
                continue
            if not same_passage(old[i], new[j]):
                continue
            used_old.add(i), used_new.add(j)
            out.append({"kind": "modified", "old": old[i], "new": new[j], "similarity": float(s),
                        "detail": word_diff(old[i]["text"], new[j]["text"])})
    for i, c in enumerate(old):                                   # 3. leftovers
        if i not in used_old:
            out.append({"kind": "removed", "old": c, "new": None, "similarity": None,
                        "detail": _clip(c["text"], 40)})
    for j, c in enumerate(new):
        if j not in used_new:
            out.append({"kind": "added", "old": None, "new": c, "similarity": None,
                        "detail": _clip(c["text"], 40)})
    return out


class VersionRegistry:
    def __init__(self, collection: str, db_path=None, client=None):
        self.collection = collection
        self.conn = sqlite3.connect(db_path or settings.sqlite_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        from pka.storage.metadata import ensure_effective_date
        ensure_effective_date(self.conn)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=settings.qdrant_url)
        return self._client

    # ------------------------------------------------------------------ building
    def _documents(self):
        rows = self.conn.execute(
            "SELECT d.doc_id, d.version, d.doc_date, d.file_mtime, d.effective_date FROM documents d "
            "JOIN ingested i ON i.doc_id=d.doc_id AND i.version=d.version AND i.collection=?",
            (self.collection,)).fetchall()
        docs = defaultdict(list)
        for doc_id, version, doc_date, mtime, edit_date in rows:
            if edit_date:                      # a version created automatically from an edit
                eff, src = date.fromisoformat(edit_date), "edit date"
            elif doc_date:
                eff, src = date.fromisoformat(doc_date), "document"
            else:
                eff, src = datetime.fromisoformat(mtime).date(), "file date"
            docs[doc_id].append((version, eff, src))
        return docs

    def _chunks(self, doc_id: str, version: int):
        from qdrant_client import models
        points, offset = [], None
        flt = models.Filter(must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(key="version", match=models.MatchValue(value=version))])
        while True:
            batch, offset = self.client.scroll(self.collection, scroll_filter=flt, limit=256,
                                               offset=offset, with_payload=True, with_vectors=True)
            points.extend(batch)
            if offset is None:
                break
        points.sort(key=lambda p: p.payload.get("chunk_index", 0))
        return [{"id": str(p.id), "text": p.payload["text"], "section": p.payload.get("section", ""),
                 "vector": p.vector} for p in points]

    def _set_validity(self, doc_id, version, valid_from: int, valid_to: int, status: str):
        from qdrant_client import models
        flt = models.Filter(must=[
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
            models.FieldCondition(key="version", match=models.MatchValue(value=version))])
        self.client.set_payload(self.collection, payload={
            "valid_from_day": valid_from, "valid_to_day": valid_to, "version_status": status},
            points=models.FilterSelector(filter=flt))

    def rebuild(self, verbose: bool = True):
        docs = self._documents()
        with self.conn:
            self.conn.execute("DELETE FROM versions WHERE collection=?", (self.collection,))
            self.conn.execute("DELETE FROM changes WHERE collection=?", (self.collection,))
        n_versioned = n_changes = 0
        for doc_id, versions in sorted(docs.items()):
            versions.sort(key=lambda v: (v[1], v[0]))          # by effective date, then number
            for idx, (version, eff, src) in enumerate(versions):
                nxt = versions[idx + 1][1] if idx + 1 < len(versions) else None
                # The first version also covers the time before its date (nothing older exists).
                vf = 0 if idx == 0 else day(eff)
                vt = day(nxt) if nxt else ALWAYS
                status = "current" if nxt is None else "superseded"
                self._set_validity(doc_id, version, vf, vt, status if len(versions) > 1 else "only")
                with self.conn:
                    self.conn.execute("INSERT INTO versions VALUES (?,?,?,?,?,?)",
                                      (self.collection, doc_id, version, eff.isoformat(),
                                       nxt.isoformat() if nxt else None, src))
            if len(versions) < 2:
                continue
            n_versioned += 1
            for (v_old, d_old, _), (v_new, d_new, _) in zip(versions, versions[1:]):
                records = diff_chunks(self._chunks(doc_id, v_old), self._chunks(doc_id, v_new))
                with self.conn:
                    for r in records:
                        o, n = r["old"], r["new"]
                        self.conn.execute(
                            "INSERT INTO changes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (self.collection, doc_id, v_old, v_new, r["kind"],
                             o and o["id"], n and n["id"], o and o["section"], n and n["section"],
                             r["similarity"], r["detail"]))
                n_changes += len(records)
                if verbose:
                    kinds = defaultdict(int)
                    for r in records:
                        kinds[r["kind"]] += 1
                    print(f"  {doc_id}: v{v_old} ({d_old}) → v{v_new} ({d_new}): "
                          f"{kinds['modified']} modified, {kinds['added']} added, "
                          f"{kinds['removed']} removed")
        if verbose:
            print(f"Versioning: {len(docs)} documents, {n_versioned} with several versions, "
                  f"{n_changes} chunk changes recorded.")

    # ------------------------------------------------------------------ reading
    def versions(self, doc_id: str):
        return self.conn.execute(
            "SELECT version, effective, valid_to FROM versions WHERE collection=? AND doc_id=? "
            "ORDER BY effective, version", (self.collection, doc_id)).fetchall()

    def changes(self, doc_id: str):
        return self.conn.execute(
            "SELECT from_version, to_version, kind, old_chunk, new_chunk, old_section, new_section, "
            "detail FROM changes WHERE collection=? AND doc_id=? ORDER BY from_version, rowid",
            (self.collection, doc_id)).fetchall()

    def changed_chunk_ids(self, doc_id: str):
        ids = []
        for *_, kind, old, new, _os, _ns, _d in self.changes(doc_id):
            ids.extend(x for x in (old, new) if x)
        return ids

    def change_log(self, doc_id: str) -> str:
        """Readable change log for the LLM and for people."""
        vers = {v: eff for v, eff, _ in self.versions(doc_id)}
        lines, current = [], None
        for v_old, v_new, kind, _o, _n, old_sec, new_sec, detail in self.changes(doc_id):
            if (v_old, v_new) != current:
                current = (v_old, v_new)
                lines.append(f"Changes in {doc_id} from v{v_old} (dated {vers.get(v_old)}) "
                             f"to v{v_new} (dated {vers.get(v_new)}):")
            if kind == "modified":
                lines.append(f"- MODIFIED [{_short(new_sec)}]: {detail}")
            elif kind == "added":
                lines.append(f"- ADDED [{_short(new_sec)}]: {detail}")
            else:
                lines.append(f"- REMOVED [{_short(old_sec)}]: {detail}")
        return "\n".join(lines)

    def versioned_docs(self):
        return {r[0] for r in self.conn.execute(
            "SELECT doc_id FROM versions WHERE collection=? GROUP BY doc_id HAVING COUNT(*) > 1",
            (self.collection,))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=settings.semantic_collection)
    ap.add_argument("--show", metavar="DOC_ID", help="print a document's versions and change log")
    args = ap.parse_args()
    reg = VersionRegistry(args.collection)
    if args.show:
        for v, eff, until in reg.versions(args.show):
            print(f"v{v}: in force from {eff}" + (f" until {until}" if until else " (current)"))
        print()
        print(reg.change_log(args.show) or "No changes recorded (only one version?).")
        return
    try:
        reg.rebuild()
    except Exception as e:
        raise SystemExit(f"Versioning failed: {e}. Is Qdrant running (docker compose up -d)?")


if __name__ == "__main__":
    main()
