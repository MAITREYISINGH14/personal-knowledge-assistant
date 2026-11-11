"""Find the date a document says it was written, e.g. '**Last updated:** 15 December 2025'."""
import re
from datetime import datetime
from typing import Optional

LABELS = r"(?:last updated|effective date|updated|date)"
DATE_RE = re.compile(
    rf"{LABELS}\s*:?\**\s*:?\s*(\d{{1,2}}\s+[A-Za-z]+\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})",
    re.IGNORECASE,
)


def parse_doc_date(text: str, max_lines: int = 15) -> Optional[str]:
    """Return an ISO date (YYYY-MM-DD) from the document header, or None."""
    head = "\n".join(text.splitlines()[:max_lines])
    m = DATE_RE.search(head)
    if not m:
        return None
    raw = m.group(1)
    for fmt in ("%d %B %Y", "%d %b %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None
