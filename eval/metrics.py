"""Retrieval metrics. A retrieved chunk 'matches' a source if doc, section and version agree."""
from pka.interfaces import RetrievedChunk
from eval.schema import Question, Source


def chunk_matches(chunk: RetrievedChunk, src: Source, qtype: str) -> bool:
    if chunk.doc_id != src.doc_id:
        return False
    if src.version is not None and chunk.version != src.version:
        return False
    # Overview questions are satisfied by any chunk or summary node from the doc.
    if qtype == "overview" or not src.section:
        return True
    return src.section.lower() in (chunk.section or "").lower()


def score_retrieval(q: Question, retrieved: list, k: int) -> dict:
    top = retrieved[:k]
    first_rank = None
    covered = set()
    for rank, chunk in enumerate(top, 1):
        for i, src in enumerate(q.sources):
            if chunk_matches(chunk, src, q.type):
                covered.add(i)
                if first_rank is None:
                    first_rank = rank
    return {
        "hit": first_rank is not None,                   # any expected source in top-k
        "full_hit": len(covered) == len(q.sources),      # all expected sources in top-k
        "rr": 1.0 / first_rank if first_rank else 0.0,   # reciprocal rank, for MRR
    }
