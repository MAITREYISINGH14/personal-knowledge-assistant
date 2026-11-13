"""Phase 5 system: hybrid + rerank retrieval that knows about time.

  "What is my rent now?"            -> only chunks in force today
  "What was my rent in June 2026?"  -> only chunks in force on 2026-06-30
  "What changed in the agreement?"  -> the chunks that changed between versions, plus the
                                       full change log for the answer
Requires the version registry: it runs after `python -m pka.ingest --chunker semantic`,
or on its own with `python -m pka.versioning`.
"""
import os
from datetime import date
from typing import Optional

from pka.config import settings
from pka.hybrid import RerankSystem
from pka.interfaces import RetrievedChunk
from pka.llm import SYSTEM
from pka.query_time import parse_time_scope
from pka.retrieval.fusion import diversify, rrf
from pka.versioning import ALWAYS, VersionRegistry, day

TEMPORAL_RULES = """
Time rules:
- Each passage label says which version it is and whether it is current or superseded.
- For questions about now or about a past date, only passages in force at that time are given:
  answer from them directly and don't bring up other versions unless the question asks.
- For questions about what changed, the change log is complete and authoritative. If the
  question asks what changed in general or "overall", list every substantive change in the log
  (skip pure date or signature updates); if it asks about one topic, cover that topic fully.
  Say what it was before and after, and cite passages where they help."""


def today() -> date:
    """PKA_TODAY=YYYY-MM-DD in .env pins 'today', so eval results don't drift with the calendar."""
    pinned = os.getenv("PKA_TODAY")
    return date.fromisoformat(pinned) if pinned else date.today()


class TemporalSystem(RerankSystem):
    name = "temporal"

    def __init__(self, k: Optional[int] = None, use_llm: Optional[bool] = None):
        super().__init__(k, use_llm)
        if not any("valid_from_day" in p for p in self.payloads.values()):
            raise RuntimeError("No version information in the index yet. Run: python -m pka.versioning")
        self.registry = VersionRegistry(self.collection, client=self.store.client)
        self.versioned = self.registry.versioned_docs()
        self._scope, self._log = {}, {}

    # ---------------------------------------------------------------- helpers
    def _date_filter(self, d: int):
        from pka.storage.vectors import valid_on_filter      # needs qdrant-client
        return valid_on_filter(d)

    def _valid(self, pid: str, d: int) -> bool:
        p = self.payloads[pid]
        return p.get("valid_from_day", 0) <= d < p.get("valid_to_day", ALWAYS)

    def _meta(self, pid: str) -> dict:
        p = self.payloads[pid]
        status, dated = p.get("version_status"), p.get("doc_date")
        if status == "current":
            text = "current version" + (f", dated {dated}" if dated else "")
        elif status == "superseded":
            until = date.fromordinal(p["valid_to_day"]).isoformat()
            text = "superseded version" + (f", dated {dated}" if dated else "") + f", replaced on {until}"
        else:
            text = ""
        return {"in_force": text, "status": status}

    def _to_chunks(self, ranked):
        return [RetrievedChunk(doc_id=self.payloads[pid]["doc_id"], text=self.payloads[pid]["text"],
                               section=self.payloads[pid].get("section", ""),
                               version=self.payloads[pid].get("version"),
                               level=self.payloads[pid].get("level", 0), score=score,
                               meta=self._meta(pid))
                for pid, score in ranked]

    def _rank(self, question: str, ids_with_scores):
        """Rerank (id, score) candidates, cap per document version, keep the top k."""
        if self.reranker and ids_with_scores:
            ids = [pid for pid, _ in ids_with_scores]
            scores = self.reranker.score(question, [self._passage(pid) for pid in ids])
            ids_with_scores = sorted(zip(ids, scores), key=lambda x: -x[1])
        return diversify(ids_with_scores, key=lambda x: (self.payloads[x[0]]["doc_id"],
                                                         self.payloads[x[0]].get("version")),
                         k=self.k, max_per_group=self.max_per_doc)

    # ---------------------------------------------------------------- retrieval
    def retrieve(self, question: str, as_of: Optional[str] = None):
        scope = parse_time_scope(question, today(), as_of)
        self._scope[question] = scope
        if scope.kind == "change":
            changed = self._retrieve_changes(question)
            if changed:
                return changed
        return self._retrieve_on(question, day(scope.on))

    def _retrieve_on(self, question: str, d: int):
        # The date is now a filter, so it shouldn't also pull the search towards chunks that
        # merely mention dates ("June 2026" matching "Last updated: 20 June 2026").
        scope = self._scope.get(question)
        if scope and scope.kind == "as_of" and scope.phrase and scope.phrase != "as_of parameter":
            question = " ".join(question.replace(scope.phrase, " ").split())
        vec = self.embedder.embed([question], prefix=self.query_prefix)[0]
        dense = self.dense_ids(vec, self.pool, query_filter=self._date_filter(d))
        keyword = [self.ids[i] for i, _ in self.bm25.search(question, self.pool * 4)
                   if self._valid(self.ids[i], d)][:self.pool]
        return self._to_chunks(self._rank(question, rrf([dense, keyword])[:self.pool]))

    def _retrieve_changes(self, question: str):
        ranked = self.candidates(question)               # unfiltered: find the document first
        target = next((self.payloads[pid]["doc_id"] for pid, _ in ranked
                       if self.payloads[pid]["doc_id"] in self.versioned), None)
        if not target:
            return None
        changed = [pid for pid in dict.fromkeys(self.registry.changed_chunk_ids(target))
                   if pid in self.payloads]
        if not changed:
            return None
        self._log[question] = self.registry.change_log(target)
        top = self._rank(question, [(pid, 0.0) for pid in changed])
        if len(top) < self.k:                            # few changes: top up from the document
            extra = [(pid, s) for pid, s in ranked if self.payloads[pid]["doc_id"] == target
                     and pid not in {p for p, _ in top}]
            top += extra[:self.k - len(top)]
        return self._to_chunks(top)

    # ---------------------------------------------------------------- answering
    def cache_context(self, question: str) -> str:
        """Everything besides the retrieved chunks that shapes the answer, for the eval cache."""
        return SYSTEM + TEMPORAL_RULES + self._log.get(question, "")

    def describe_scope(self, question: str) -> str:
        scope = self._scope.get(question) or parse_time_scope(question, today())
        return scope.describe()

    def generate(self, question: str, chunks) -> str:
        if not self.answerer:
            return ""
        scope = self._scope.get(question) or parse_time_scope(question, today())
        if scope.kind == "change" and self._log.get(question):
            extra = "Change log (computed by comparing the document versions):\n" + self._log[question]
        else:
            extra = (f"Time frame of the question: {scope.describe()}. "
                     "Only passages in force at that time are included.")
        return self.answerer.answer(question, chunks, system=SYSTEM + TEMPORAL_RULES, extra=extra)
