"""Eval-set format and validation. Standard library only."""
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

QUESTION_TYPES = {"fact", "overview", "temporal"}
TEMPORAL_KINDS = {"current", "as_of", "change"}


@dataclass
class Source:
    doc_id: str
    section: str = ""              # substring of the expected breadcrumb; "" = any
    version: Optional[int] = None  # required for temporal questions


@dataclass
class Question:
    id: str
    type: str
    question: str
    expected_answer: str
    sources: list = field(default_factory=list)  # list[Source]
    as_of: Optional[str] = None                  # ISO date, for as_of questions
    temporal_kind: Optional[str] = None
    notes: str = ""


class EvalSetError(ValueError):
    pass


def _parse_question(raw: dict, idx: int) -> Question:
    where = f"question #{idx} ({raw.get('id', '?')})"
    for key in ("id", "type", "question", "expected_answer", "sources"):
        if key not in raw or raw[key] in ("", None, []):
            raise EvalSetError(f"{where}: missing '{key}'")
    if raw["type"] not in QUESTION_TYPES:
        raise EvalSetError(f"{where}: type must be one of {sorted(QUESTION_TYPES)}")

    sources = []
    for s in raw["sources"]:
        if not s.get("doc_id"):
            raise EvalSetError(f"{where}: every source needs a doc_id")
        sources.append(Source(s["doc_id"], s.get("section", ""), s.get("version")))

    q = Question(
        id=raw["id"], type=raw["type"], question=raw["question"],
        expected_answer=raw["expected_answer"], sources=sources,
        as_of=raw.get("as_of"), temporal_kind=raw.get("temporal_kind"),
        notes=raw.get("notes", ""),
    )
    if q.type == "temporal":
        if q.temporal_kind not in TEMPORAL_KINDS:
            raise EvalSetError(f"{where}: temporal_kind must be one of {sorted(TEMPORAL_KINDS)}")
        if any(s.version is None for s in q.sources):
            raise EvalSetError(f"{where}: temporal sources must each give a version")
        if q.temporal_kind == "as_of" and not q.as_of:
            raise EvalSetError(f"{where}: as_of questions need an 'as_of' date")
        if q.temporal_kind == "change" and len({s.version for s in q.sources}) < 2:
            raise EvalSetError(f"{where}: change questions should cite both versions")
    return q


def load_eval_set(path: Path) -> list:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_questions = data["questions"] if isinstance(data, dict) else data
    questions = [_parse_question(r, i) for i, r in enumerate(raw_questions, 1)]
    dupes = [k for k, n in Counter(q.id for q in questions).items() if n > 1]
    if dupes:
        raise EvalSetError(f"duplicate ids: {dupes}")
    return questions


def coverage_warnings(questions: list) -> list:
    """Soft checks against the Phase 0 targets. Returns human-readable warnings."""
    warnings = []
    n = len(questions)
    if n < 30:
        warnings.append(f"Only {n} questions; aim for 30-50.")
    counts = Counter(q.type for q in questions)
    for t in sorted(QUESTION_TYPES):
        share = counts.get(t, 0) / n if n else 0
        if share < 0.2:
            warnings.append(f"'{t}' questions are {share:.0%} of the set; aim for at least 20%.")
    versioned_docs = {s.doc_id for q in questions if q.type == "temporal" for s in q.sources}
    if len(versioned_docs) < 3:
        warnings.append(f"Temporal questions cover {len(versioned_docs)} versioned docs; aim for 3+.")
    return warnings
