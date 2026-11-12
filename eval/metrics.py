"""Retrieval metrics.

A retrieved chunk 'matches' a source if doc, version and section agree and, when the source
lists evidence phrases, the chunk's text contains at least one of them. Evidence makes the
hit mean "the chunk with the answer was retrieved", not just "some chunk of that document".
"""
import re
import unicodedata

from pka.interfaces import RetrievedChunk
from eval.schema import Question, Source


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)      # unify dashes
    text = re.sub(r"[\u2018\u2019]", "'", text)
    return re.sub(r"\s+", " ", text).strip()


def has_evidence(text: str, phrases) -> bool:
    if not phrases:
        return True
    norm = normalize(text)
    return any(normalize(p) in norm for p in phrases)


def chunk_matches(chunk: RetrievedChunk, src: Source, qtype: str) -> bool:
    if chunk.doc_id != src.doc_id:
        return False
    if src.version is not None and chunk.version != src.version:
        return False
    if not has_evidence(chunk.text, src.evidence):
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
        "version_clean": version_clean(q, top),
    }


def version_clean(q: Question, top: list):
    """For 'current' and 'as of' questions: True if no retrieved chunk comes from a version of
    the asked-about document other than the expected one(s). None for other questions."""
    if q.type != "temporal" or q.temporal_kind not in ("current", "as_of"):
        return None
    expected = {}
    for src in q.sources:
        if src.version is not None:
            expected.setdefault(src.doc_id, set()).add(src.version)
    return not any(c.doc_id in expected and c.version not in expected[c.doc_id] for c in top)
