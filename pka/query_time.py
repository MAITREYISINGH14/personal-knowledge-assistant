"""Work out which point in time a question is about.

  current  "What is my rent now?"                      -> versions valid today
  as_of    "What was my rent in June 2026?"            -> versions valid on 2026-06-30
  change   "What changed in the renewed agreement?"    -> compare versions

The date always comes from the question's own words, so the eval can't lean on its labels.
"""
import calendar
import re
from dataclasses import dataclass
from datetime import date
from typing import Optional

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9
MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))

CHANGE_RE = re.compile(
    r"\b(what('s| has| have)? changed|what changes|(how|why)\b.{0,40}\b(did|has|have|does)\b.{0,60}\bchange|"
    r"changed? (between|from|since)|differen(ce|ces|t) between|compare[ds]? (to|with)|"
    r"what happened to|what was (added|removed|dropped)|(added|removed|dropped) in the|"
    r"(old|previous|earlier|first) (and|vs\.?|versus) (new|latest|current|second)|"
    r"\bvs\.? the\b|draft and the|was renewed|revised|revision)\b",
    re.IGNORECASE)
ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DMY_RE = re.compile(rf"\b(\d{{1,2}})\s+({MONTH_RE})\.?,?\s+(\d{{4}})\b", re.IGNORECASE)
# A month counts only with a year or after a time preposition, so "may" the verb is ignored.
MONTH_YEAR_RE = re.compile(rf"\b({MONTH_RE})\.?,?\s+(\d{{4}})\b", re.IGNORECASE)
MONTH_PREP_RE = re.compile(
    rf"\b(?:in|of|during|by|since|before|after|until|end of|start of|beginning of)\s+"
    rf"(?:the\s+)?(?:(end|start|beginning) of\s+)?({MONTH_RE})\b(?:\s+(\d{{4}}))?",
    re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:in|during|as of|for)\s+(20\d{2}|19\d{2})\b", re.IGNORECASE)


@dataclass
class TimeScope:
    kind: str                      # current | as_of | change
    on: date                       # the date whose versions apply (today for current/change)
    phrase: str = ""               # the words that set it, for explanations

    def describe(self) -> str:
        if self.kind == "as_of":
            return f"as of {self.on.isoformat()} (from \"{self.phrase}\")"
        if self.kind == "change":
            return "comparing versions"
        return f"current (today, {self.on.isoformat()})"


def _month_date(month: int, year: Optional[int], today: date, edge: str = "end") -> date:
    if year is None:                                   # no year: the most recent such month
        year = today.year if month <= today.month else today.year - 1
    last = calendar.monthrange(year, month)[1]
    return date(year, month, 1 if edge in ("start", "beginning") else last)


def find_date(question: str, today: date):
    """Return (date, matched phrase) for the first date expression in the question, or None."""
    m = ISO_RE.search(question)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3])), m[0]
    m = DMY_RE.search(question)
    if m:
        return date(int(m[3]), MONTHS[m[2].lower()], int(m[1])), m[0]
    m = MONTH_PREP_RE.search(question)
    if m:
        edge, month, year = m[1] or "end", MONTHS[m[2].lower()], int(m[3]) if m[3] else None
        return _month_date(month, year, today, edge.lower()), m[0]
    m = MONTH_YEAR_RE.search(question)
    if m:
        return _month_date(MONTHS[m[1].lower()], int(m[2]), today), m[0]
    m = YEAR_RE.search(question)
    if m:
        return date(int(m[1]), 12, 31), m[0]
    return None


def parse_time_scope(question: str, today: Optional[date] = None,
                     as_of: Optional[str] = None) -> TimeScope:
    """as_of (ISO date) is only used when the question itself names no date."""
    today = today or date.today()
    if CHANGE_RE.search(question):
        return TimeScope("change", today, CHANGE_RE.search(question)[0])
    found = find_date(question, today)
    if found:
        on, phrase = found
        if on < today:
            return TimeScope("as_of", on, phrase)
        return TimeScope("current", today, phrase)    # a future date: answer with today's state
    if as_of:
        return TimeScope("as_of", date.fromisoformat(as_of), "as_of parameter")
    return TimeScope("current", today)
