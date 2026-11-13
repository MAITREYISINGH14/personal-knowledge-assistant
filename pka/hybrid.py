"""Phase 3 systems, built on the Phase 2 semantic index (no re-ingest needed).

  HybridSystem   dense search + BM25 keyword search, merged with reciprocal rank fusion,
                 then a per-document cap so one long document can't fill every slot
  RerankSystem   the same, plus a cross-encoder that re-scores the merged candidates

Evaluate both with --retrieval-only to see what each step adds, for free.
"""
from typing import Optional

from pka.config import settings
from pka.interfaces import RetrievedChunk
from pka.retrieval.bm25 import BM25
from pka.retrieval.fusion import diversify, rrf
from pka.semantic import SemanticSystem


class HybridSystem(SemanticSystem):
    name = "hybrid"
    use_rerank = False

    def __init__(self, k: Optional[int] = None, use_llm: Optional[bool] = None):
        super().__init__(k, use_llm)
        self.pool = settings.candidate_pool
        self.max_per_doc = settings.max_per_doc
        self.payloads, self.ids = self._load_payloads()
        self.bm25 = BM25([self._passage(pid) for pid in self.ids])
        self.reranker = None
        if self.use_rerank:
            from pka.retrieval.rerank import Reranker
            self.reranker = Reranker(settings.reranker_model)

    def _load_payloads(self):
        payloads, offset = {}, None
        while True:
            points, offset = self.store.client.scroll(self.collection, limit=512, offset=offset,
                                                      with_payload=True, with_vectors=False)
            for p in points:
                payloads[str(p.id)] = p.payload
            if offset is None:
                break
        if not payloads:
            raise RuntimeError(f"The '{self.collection}' index is empty. Run: {self.ingest_hint}")
        return payloads, list(payloads)

    def _passage(self, pid: str) -> str:
        p = self.payloads[pid]
        return p.get("embed_text") or p["text"]

    def dense_ids(self, vec, limit: int, query_filter=None):
        """Ids of the nearest chunks by embedding. Overridden to search several indexes."""
        return [str(p.id) for p in self.store.search(vec, limit=limit, query_filter=query_filter)]

    def candidates(self, question: str):
        """Dense and keyword results merged with RRF: [(id, fused score)], best first."""
        vec = self.embedder.embed([question], prefix=self.query_prefix)[0]
        dense = self.dense_ids(vec, self.pool)
        keyword = [self.ids[i] for i, _ in self.bm25.search(question, self.pool)]
        return rrf([dense, keyword])[:self.pool]

    def retrieve(self, question: str, as_of: Optional[str] = None):
        ranked = self.candidates(question)
        if self.reranker:
            ids = [pid for pid, _ in ranked]
            scores = self.reranker.score(question, [self._passage(pid) for pid in ids])
            ranked = sorted(zip(ids, scores), key=lambda x: -x[1])
        top = diversify(ranked, key=lambda x: (self.payloads[x[0]]["doc_id"],
                                               self.payloads[x[0]].get("version")),
                        k=self.k, max_per_group=self.max_per_doc)
        return [RetrievedChunk(doc_id=self.payloads[pid]["doc_id"], text=self.payloads[pid]["text"],
                               section=self.payloads[pid].get("section", ""),
                               version=self.payloads[pid].get("version"),
                               level=self.payloads[pid].get("level", 0), score=score)
                for pid, score in top]


class RerankSystem(HybridSystem):
    name = "hybrid_rerank"
    use_rerank = True
