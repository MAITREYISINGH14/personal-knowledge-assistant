"""Phase 4 system: Phase 5's time-aware hybrid retrieval over chunks AND summaries.

All levels compete in one ranking ("collapsed tree", as in the RAPTOR paper): a detailed
question is usually won by a chunk, a broad one by a part summary or a document overview.
The reranker makes that call per question, so no separate question classifier is needed.
"""
from typing import Optional

from pka.config import settings
from pka.retrieval.bm25 import BM25
from pka.temporal import TemporalSystem


class SummarySystem(TemporalSystem):
    name = "summary"

    def __init__(self, k: Optional[int] = None, use_llm: Optional[bool] = None):
        from pka.storage.vectors import VectorStore
        self.summary_store = VectorStore(settings.qdrant_url, settings.summary_collection, dim=0)
        super().__init__(k, use_llm)

    def _build_index(self):
        super()._build_index()                      # chunks, then add the summaries
        client = self.summary_store.client
        if not client.collection_exists(settings.summary_collection):
            raise RuntimeError("No summaries yet. Run: python -m pka.summarize")
        summaries, offset = {}, None
        while True:
            points, offset = client.scroll(settings.summary_collection, limit=512, offset=offset,
                                           with_payload=True, with_vectors=False)
            summaries.update({str(p.id): p.payload for p in points})
            if offset is None:
                break
        self.n_summaries = len(summaries)
        self.payloads.update(summaries)
        self.ids = list(self.payloads)
        self.bm25 = BM25([self._passage(pid) for pid in self.ids])

    def dense_ids(self, vec, limit: int, query_filter=None):
        hits = (self.store.search(vec, limit=limit, query_filter=query_filter)
                + self.summary_store.search(vec, limit=limit, query_filter=query_filter))
        hits.sort(key=lambda p: -p.score)          # same embedding model, so scores compare
        return [str(p.id) for p in hits[:limit]]
