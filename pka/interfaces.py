"""The contract every version of the system implements.

Phase 1 (baseline), Phase 2 (semantic chunking) and later phases each provide
a class with this shape, so the same eval harness can score all of them.
"""
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class RetrievedChunk:
    doc_id: str                    # stable id, e.g. "lender_agreement"
    text: str
    section: str = ""              # breadcrumb, e.g. "Fees > Late payment"
    version: Optional[int] = None  # document version this chunk belongs to
    level: int = 0                 # 0 = leaf chunk, >0 = summary node (Phase 4)
    score: float = 0.0
    meta: dict = field(default_factory=dict)   # e.g. dates and version status (Phase 5)


@dataclass
class QAResult:
    answer: str
    retrieved: list = field(default_factory=list)  # list[RetrievedChunk]


class QASystem(Protocol):
    name: str

    def answer(self, question: str, as_of: Optional[str] = None) -> QAResult:
        """Answer a question. `as_of` is an ISO date for time-scoped questions."""
        ...


class NullSystem:
    """Placeholder that answers nothing. Lets the eval harness run in Phase 0."""
    name = "null"

    def answer(self, question: str, as_of: Optional[str] = None) -> QAResult:
        return QAResult(answer="", retrieved=[])
