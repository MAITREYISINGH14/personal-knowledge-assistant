"""Phase 1 baseline: fixed-size chunks, dense retrieval, top-k straight to the LLM.

Deliberately simple. Every later phase must beat its scores on the eval set.
"""
from typing import Optional

from pka.config import settings
from pka.interfaces import QAResult, RetrievedChunk


class BaselineSystem:
    name = "baseline"

    def __init__(self, k: Optional[int] = None, use_llm: Optional[bool] = None):
        from pka.embeddings import Embedder
        from pka.storage.vectors import VectorStore
        self.k = k or settings.top_k
        self.embedder = Embedder(settings.embed_model)
        self.store = VectorStore(settings.qdrant_url, settings.baseline_collection, dim=0)
        try:
            exists = self.store.client.collection_exists(settings.baseline_collection)
        except Exception as e:
            raise RuntimeError("Can't reach Qdrant. Open Docker Desktop, wait for 'Engine running', "
                               "then run: docker compose up -d") from e
        if not exists:
            raise RuntimeError("No baseline index found. Run: python -m pka.ingest")
        self.use_llm = (settings.llm_provider != "none") if use_llm is None else use_llm
        self.answerer = None
        if self.use_llm:
            from pka.llm import Answerer
            self.answerer = Answerer()

    def retrieve(self, question: str, as_of: Optional[str] = None):
        vec = self.embedder.embed([question])[0]
        return [RetrievedChunk(doc_id=p.payload["doc_id"], text=p.payload["text"],
                               section=p.payload.get("section", ""),
                               version=p.payload.get("version"), level=0, score=p.score)
                for p in self.store.search(vec, limit=self.k)]

    def generate(self, question: str, chunks) -> str:
        return self.answerer.answer(question, chunks) if self.answerer else ""

    def answer(self, question: str, as_of: Optional[str] = None) -> QAResult:
        # The baseline ignores as_of on purpose: time awareness arrives in Phase 5.
        chunks = self.retrieve(question, as_of)
        return QAResult(answer=self.generate(question, chunks), retrieved=chunks)
