"""Optional query rewriting: bridge the gap between how you ask and how documents are written.

"What was my rent?" never matches a contract that only says "licence fee". With QUERY_REWRITE=1 in
.env, each question is also searched in a few alternative wordings suggested by the LLM (using the
summary model, so it doesn't use the answer quota). Results from all wordings are merged, and the
reranker still judges every passage against your ORIGINAL question.

Costs one small API call per new question; rewrites are cached, so repeats are free.
"""
import json
import re
import sqlite3
from datetime import datetime, timezone

from pka.config import settings

PROMPT = """Rewrite this question as up to 3 short search queries that use words the source documents
are likely to use: formal, legal, technical or domain terms and close synonyms. Keep names, numbers
and dates. One query per line, no numbering, no explanations.

Question: {question}"""

SCHEMA = """CREATE TABLE IF NOT EXISTS query_rewrites (
    question TEXT PRIMARY KEY, rewrites TEXT NOT NULL, created TEXT)"""


def parse(text: str, question: str, limit: int = 3) -> list:
    out = []
    for line in (text or "").splitlines():
        q = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
        if 3 <= len(q) <= 200 and q.lower() != question.lower() and q.lower() not in (x.lower() for x in out):
            out.append(q)
    return out[:limit]


class QueryRewriter:
    def __init__(self, db_path=None, llm=None):
        self.conn = sqlite3.connect(db_path or settings.sqlite_path, check_same_thread=False)
        self.conn.execute(SCHEMA)
        self.conn.commit()
        self._llm = llm

    def _ask(self, question: str) -> str:
        if self._llm:
            return self._llm(question)
        from pka.llm import chat
        return chat("You rewrite search queries. Output only the queries.",
                    PROMPT.format(question=question), max_tokens=200, purpose="summary")

    def rewrite(self, question: str) -> list:
        key = " ".join(question.split()).lower()
        row = self.conn.execute("SELECT rewrites FROM query_rewrites WHERE question=?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        try:
            rewrites = parse(self._ask(question), question)
        except Exception:
            return []                    # rewriting is a bonus: never let it break a question
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO query_rewrites VALUES (?,?,?)",
                              (key, json.dumps(rewrites), datetime.now(timezone.utc).isoformat()))
        return rewrites
