"""Phase 4: hierarchical summaries (a summary tree per document version).

  level 0   the semantic chunks from Phase 2
  level 1   summaries of groups of consecutive chunks ("parts")
  level 2+  summaries of groups of parts, until one ROOT overview of the whole document

Summaries are cached by a hash of exactly what they summarise. So:
  - an unchanged part of a new version reuses its old summary (only changed parts and their
    ancestors cost API calls - incremental re-summarisation),
  - a stopped run (free-tier daily limit) resumes where it left off,
  - re-ingesting the corpus costs nothing for content that didn't change.

  python -m pka.summarize --estimate          # how many API calls / tokens it will take
  python -m pka.summarize                     # build (smallest documents first)
  python -m pka.summarize --max-calls 40      # stop after 40 API calls today
  python -m pka.summarize --docs rental_agreement,raptor_tree_retrieval_2024
  python -m pka.summarize --show rental_agreement
"""
import argparse
import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from types import SimpleNamespace

from pka.config import settings

PROMPT_VERSION = "summary-v1"
SUMMARY_TOKENS_EST = 260            # typical length of a generated summary, for planning
NAMESPACE = uuid.UUID("2b6f7a10-4e1d-4c55-9a0e-5d3c2b1a0f99")

SUMMARY_SYSTEM = (
    "You write faithful, information-dense summaries of the user's own documents for a search "
    "index. Use only the text given. Keep important names, numbers, amounts, dates and defined "
    "terms exactly as written. Write plain prose: no headings, no bullet points, no preamble.")

SCHEMA = """
CREATE TABLE IF NOT EXISTS summary_cache (
    hash TEXT PRIMARY KEY, summary TEXT NOT NULL, model TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS summary_nodes (
    collection TEXT, node_id TEXT, doc_id TEXT, version INTEGER, level INTEGER, idx INTEGER,
    is_root INTEGER, children TEXT, hash TEXT, section TEXT,
    PRIMARY KEY (collection, node_id));
"""


@dataclass
class Node:
    node_id: str
    level: int
    idx: int
    hash: str
    tokens: int
    sections: list                     # section labels of all chunks underneath
    children: list = field(default_factory=list)   # child Nodes (empty for chunks)
    text: str = ""                     # chunk text, or the summary once generated
    is_root: bool = False

    @property
    def role(self) -> str:
        if self.is_root:
            return "root"
        return "part" if self.level == 1 else "section"


# ------------------------------------------------------------------ planning (no API calls)
def group_consecutive(nodes, max_tokens: int, max_children: int, min_children: int = 1):
    """Split nodes into consecutive groups under a token budget, avoiding a lone last node.
    min_children >= 2 guarantees each level shrinks, so the tree always reaches one root."""
    groups, cur, cur_tok = [], [], 0
    for n in nodes:
        full = cur_tok + n.tokens > max_tokens or len(cur) >= max_children
        if cur and full and len(cur) >= min_children:
            groups.append(cur)
            cur, cur_tok = [], 0
        cur.append(n)
        cur_tok += n.tokens
    if cur:
        if len(cur) < max(2, min_children) and groups and (
                min_children > 1 or sum(x.tokens for x in groups[-1]) + cur[0].tokens <= max_tokens * 1.25):
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return groups


def _hash(*parts) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]


def span_label(sections) -> str:
    names = []
    for sec in sections:
        for part in (sec or "").split(" | "):
            name = part.split(" > ")[-1].strip()
            if name and name not in names:
                names.append(name)
    pages = [n for n in names if n.startswith("p.") and n[2:].isdigit()]
    if pages and len(pages) == len(names):
        nums = sorted(int(p[2:]) for p in pages)
        return f"pages {nums[0]}–{nums[-1]}" if nums[0] != nums[-1] else f"page {nums[0]}"
    if not names:
        return "whole text"
    return names[0] if len(names) == 1 else f"{names[0]} … {names[-1]}"


def plan_tree(doc_id: str, version: int, title: str, chunks, max_tokens: int, max_children: int):
    """chunks: dicts with id, text, section, n_tokens (in document order).
    Returns the summary nodes, bottom-up (level 1 first). Empty for single-chunk documents,
    which are already whole-document passages."""
    if len(chunks) < 2:
        return []
    level_nodes = [Node(c["id"], 0, i, _hash("chunk", c["text"]), int(c["n_tokens"] or 0) + 10,
                        [c["section"]], text=c["text"]) for i, c in enumerate(chunks)]
    out, level = [], 1
    while True:
        groups = group_consecutive(level_nodes, max_tokens, max_children,
                                   min_children=1 if level == 1 else 2)
        is_top = len(groups) == 1
        new_nodes = []
        for i, g in enumerate(groups):
            role = "root" if is_top else ("part" if level == 1 else "section")
            h = _hash(PROMPT_VERSION, role, title, *[c.hash for c in g])
            node_id = str(uuid.uuid5(NAMESPACE, f"{settings.summary_collection}/{doc_id}/v{version}/L{level}/{i}"))
            sections = [s for c in g for s in c.sections]
            new_nodes.append(Node(node_id, level, i, h, SUMMARY_TOKENS_EST, sections, g, is_root=is_top))
        out.extend(new_nodes)
        if is_top:
            return out
        level_nodes, level = new_nodes, level + 1


def build_prompt(node: Node, title: str) -> str:
    span = span_label(node.sections)
    if node.level == 1:
        items = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(node.children, 1))
        kind = "Passages"
    else:
        items = "\n\n".join(f"Part {i} ({span_label(c.sections)}): {c.text}"
                            for i, c in enumerate(node.children, 1))
        kind = "Summaries of its parts"
    if node.role == "root":
        return (f"Document: {title}\n{kind}, in document order:\n\n{items}\n\n"
                "Write an overview of the whole document in 150-250 words: what it is and its "
                "purpose, how it is organised (its main parts in order), and its key points or "
                "conclusions.")
    if node.role == "part":
        return (f"Document: {title}\nOne part of the document ({span}). {kind}, in order:\n\n{items}\n\n"
                "Summarise what this part covers and its key points in 80-150 words.")
    return (f"Document: {title}\nA larger section of the document ({span}). {kind}, in order:\n\n"
            f"{items}\n\nCombine them into one summary of this section in 120-180 words, keeping "
            "its structure and key points.")


def input_tokens(node: Node) -> int:
    return sum(c.tokens for c in node.children) + 120


# ------------------------------------------------------------------ building
class SummaryBuilder:
    def __init__(self, db_path=None, embedder=None):
        self.conn = sqlite3.connect(db_path or settings.sqlite_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self._embedder = embedder
        self.source = settings.semantic_collection

    def close(self):
        self.conn.close()

    @property
    def embedder(self):
        if self._embedder is None:
            from pka.embeddings import Embedder
            self._embedder = Embedder(settings.semantic_embed_model)
        return self._embedder

    # -- inputs
    def documents(self):
        rows = self.conn.execute(
            "SELECT d.doc_id, d.version, d.title, d.doc_date FROM documents d JOIN ingested i "
            "ON i.doc_id=d.doc_id AND i.version=d.version AND i.collection=? "
            "ORDER BY d.doc_id, d.version", (self.source,)).fetchall()
        return [SimpleNamespace(doc_id=r[0], version=r[1], title=r[2] or r[0], doc_date=r[3]) for r in rows]

    def chunks(self, doc_id, version):
        rows = self.conn.execute(
            "SELECT chunk_id, text, section, n_tokens FROM chunks WHERE collection=? AND doc_id=? "
            "AND version=? ORDER BY chunk_index", (self.source, doc_id, version)).fetchall()
        return [{"id": r[0], "text": r[1], "section": r[2] or "", "n_tokens": r[3]} for r in rows]

    def cached(self, h):
        row = self.conn.execute("SELECT summary FROM summary_cache WHERE hash=?", (h,)).fetchone()
        return row[0] if row else None

    def validity(self, doc_id):
        """Per-version validity from the Phase 5 registry: {version: (from_day, to_day, status)}."""
        from pka.versioning import ALWAYS
        rows = self.conn.execute(
            "SELECT version, effective, valid_to FROM versions WHERE collection=? AND doc_id=? "
            "ORDER BY effective, version", (self.source, doc_id)).fetchall()
        out = {}
        for i, (v, eff, until) in enumerate(rows):
            status = "only" if len(rows) == 1 else ("current" if until is None else "superseded")
            out[v] = (0 if i == 0 else date.fromisoformat(eff).toordinal(),
                      date.fromisoformat(until).toordinal() if until else ALWAYS, status)
        return out

    def status(self):
        """How much summarising is left: documents and API calls (no API use)."""
        stored = {(r[0], r[1]) for r in self.conn.execute(
            "SELECT DISTINCT doc_id, version FROM summary_nodes WHERE collection=?",
            (settings.summary_collection,))}
        plans = self.plan()
        pending = [(d, todo) for d, _, todo in plans if todo or (d.doc_id, d.version) not in stored]
        return {"docs_total": len(plans), "docs_pending": len(pending),
                "calls_needed": sum(len(t) for _, t in pending)}

    def plan(self, doc_filter=None):
        plans = []
        for d in self.documents():
            if doc_filter and d.doc_id not in doc_filter:
                continue
            nodes = plan_tree(d.doc_id, d.version, d.title, self.chunks(d.doc_id, d.version),
                              settings.group_max_tokens, settings.group_max_children)
            if not nodes:
                continue
            todo = [n for n in nodes if self.cached(n.hash) is None]
            plans.append((d, nodes, todo))
        plans.sort(key=lambda p: sum(input_tokens(n) for n in p[2]))   # cheapest first
        return plans

    # -- outputs
    def _store(self, d, nodes):
        from qdrant_client import models
        from pka.chunking.semantic import context_header
        from pka.storage.vectors import VectorStore
        store = VectorStore(settings.qdrant_url, settings.summary_collection, self.embedder.dim)
        store.ensure()
        vf, vt, status = self.validity(d.doc_id).get(d.version, (0, 99_999_999, "only"))
        versioned = len(self.validity(d.doc_id)) > 1
        meta = SimpleNamespace(title=d.title, doc_id=d.doc_id, doc_date=d.doc_date,
                               versioned=versioned, version=d.version)
        texts, payloads, ids = [], [], []
        for n in nodes:
            header = context_header(meta, "") + f" | Summary of: {span_label(n.sections)}"
            texts.append(f"{header}\n\nSummary: {n.text}")
            ids.append(n.node_id)
            payloads.append({
                "doc_id": d.doc_id, "version": d.version, "level": n.level, "is_root": n.is_root,
                "text": n.text, "embed_text": texts[-1], "title": d.title, "doc_date": d.doc_date,
                "section": " | ".join(dict.fromkeys(s for s in n.sections if s)),
                "children": [c.node_id for c in n.children], "valid_from_day": vf,
                "valid_to_day": vt, "version_status": status})
        store.client.delete(settings.summary_collection, points_selector=models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=d.doc_id)),
                models.FieldCondition(key="version", match=models.MatchValue(value=d.version))])))
        store.upsert(ids, self.embedder.embed(texts), payloads)
        with self.conn:
            self.conn.execute("DELETE FROM summary_nodes WHERE collection=? AND doc_id=? AND version=?",
                              (settings.summary_collection, d.doc_id, d.version))
            self.conn.executemany("INSERT INTO summary_nodes VALUES (?,?,?,?,?,?,?,?,?,?)", [
                (settings.summary_collection, n.node_id, d.doc_id, d.version, n.level, n.idx,
                 int(n.is_root), json.dumps([c.node_id for c in n.children]), n.hash,
                 span_label(n.sections)) for n in nodes])
        from pka import index_state
        index_state.bump()                  # a running app should reload and use these summaries

    def build(self, doc_filter=None, max_calls=None, llm=None):
        """llm(system, user) -> text. Returns (docs_done, calls_made, stopped_reason)."""
        from pka.llm import LLMError, RateLimitExhausted, active_model, chat, tidy_citations
        model = "custom"
        if llm is None:                      # real run: summaries use their own model/quota
            model = active_model("summary")
            llm = lambda system, user: chat(system, user, max_tokens=500, purpose="summary")
        calls, done = 0, 0
        for d, nodes, todo in self.plan(doc_filter):
            if max_calls is not None and calls + len(todo) > max_calls and calls > 0:
                return done, calls, f"reached --max-calls {max_calls}"
            reused = len(nodes) - len(todo)
            try:
                for n in nodes:                       # bottom-up: children are ready first
                    text = self.cached(n.hash)
                    if text is None:
                        text = tidy_citations(llm(SUMMARY_SYSTEM, build_prompt(n, d.title)).strip())
                        if not text:
                            raise RuntimeError(f"empty summary for {d.doc_id} level {n.level}")
                        calls += 1
                        with self.conn:
                            self.conn.execute("INSERT OR REPLACE INTO summary_cache VALUES (?,?,?,?)",
                                              (n.hash, text, model,
                                               datetime.now(timezone.utc).isoformat()))
                    n.text = text
            except RateLimitExhausted as e:
                return done, calls, f"free-tier limit reached ({e})"
            except LLMError as e:          # e.g. wrong model name or key: stop cleanly, keep progress
                return done, calls, f"API error ({e})"
            self._store(d, nodes)
            done += 1
            levels = max(n.level for n in nodes)
            note = f", reused {reused}" if reused else ""
            print(f"  ok    {d.doc_id} v{d.version}: {len(nodes)} summaries over {levels} "
                  f"level(s), {len(todo)} new{note}")
        return done, calls, None


def drop_summaries(conn, client, doc_id: str, version: int):
    """Remove a document version's summary tree from the index (its content changed or it was
    deleted). Cached summary texts stay, so unchanged parts are reused when it is rebuilt."""
    try:
        with conn:
            conn.execute("DELETE FROM summary_nodes WHERE collection=? AND doc_id=? AND version=?",
                         (settings.summary_collection, doc_id, version))
    except sqlite3.OperationalError:
        pass                                        # summaries were never set up
    if client.collection_exists(settings.summary_collection):
        from qdrant_client import models
        client.delete(settings.summary_collection, points_selector=models.FilterSelector(
            filter=models.Filter(must=[
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                models.FieldCondition(key="version", match=models.MatchValue(value=version))])))


def check_qdrant(client_factory=None):
    """Fail early and clearly if Qdrant isn't running, before any API calls are spent."""
    try:
        if client_factory is None:
            from qdrant_client import QdrantClient
            client_factory = lambda: QdrantClient(url=settings.qdrant_url)
        client_factory().get_collections()
    except Exception as e:
        raise SystemExit(
            f"Can't reach Qdrant at {settings.qdrant_url} ({type(e).__name__}).\n"
            "Open Docker Desktop, wait for 'Engine running', then run: docker compose up -d\n"
            "Nothing was lost: finished summaries are saved and will be reused.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true", help="show the cost, make no API calls")
    ap.add_argument("--docs", help="comma-separated doc_ids to summarise (default: all)")
    ap.add_argument("--max-calls", type=int, help="stop after this many API calls")
    ap.add_argument("--show", metavar="DOC_ID", help="print a document's summary tree")
    args = ap.parse_args()
    b = SummaryBuilder()
    docs = set(args.docs.split(",")) if args.docs else None

    if args.show:
        rows = b.conn.execute(
            "SELECT version, level, idx, is_root, section, hash FROM summary_nodes WHERE doc_id=? "
            "ORDER BY version, level DESC, idx", (args.show,)).fetchall()
        if not rows:
            print(f"No summaries for '{args.show}' yet.")
        for v, level, idx, root, section, h in rows:
            label = "ROOT overview" if root else f"level {level}, part {idx + 1}"
            print(f"\n━━ v{v} · {label} · {section}\n{b.cached(h)}")
        return

    if not b.conn.execute("SELECT 1 FROM versions WHERE collection=? LIMIT 1", (b.source,)).fetchone():
        raise SystemExit("Run python -m pka.versioning first (summaries inherit version dates).")

    plans = b.plan(docs)
    total_nodes = sum(len(n) for _, n, _ in plans)
    todo_nodes = sum(len(t) for _, _, t in plans)
    tokens = sum(input_tokens(n) for _, _, t in plans for n in t)
    print(f"{len(plans)} documents need a summary tree: {total_nodes} summaries, "
          f"{total_nodes - todo_nodes} already cached, {todo_nodes} to generate.")
    print(f"Estimated input: ~{tokens:,} tokens (+ ~{todo_nodes * 400:,} output/reasoning tokens).")
    if args.estimate:
        print("\nCheapest first:")
        for d, nodes, todo in plans:
            if todo:
                print(f"  {d.doc_id} v{d.version}: {len(todo)} calls, ~{sum(input_tokens(n) for n in todo):,} tokens")
        print("\nCompare with your daily token limit for the summary model (Groq console → Limits).")
        return
    check_qdrant()
    if not todo_nodes:
        print("Everything is already summarised; refreshing the summary index.")
    try:
        done, calls, stopped = b.build(docs, args.max_calls)
    except RuntimeError as e:
        raise SystemExit(f"Summarising stopped: {e}")
    print(f"\nDone: {done} documents stored, {calls} API calls made.")
    if stopped:
        print(f"Stopped early: {stopped}.\nRun the same command again later; finished summaries "
              "are cached and won't be regenerated.")


if __name__ == "__main__":
    main()
