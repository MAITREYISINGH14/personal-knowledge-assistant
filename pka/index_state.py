"""A tiny marker file that changes whenever the index changes.

The web app keeps the search index in memory for speed. Whoever changes the index (sync,
summarising) calls bump(); the app compares current() with what it loaded and reloads.
"""
import time
from pathlib import Path

from pka.config import ROOT

STATE_FILE = ROOT / "data" / "index_generation"


def bump(path=None) -> str:
    p = Path(path or STATE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    value = str(time.time_ns())
    p.write_text(value)
    return value


def current(path=None) -> str:
    try:
        return Path(path or STATE_FILE).read_text().strip()
    except OSError:
        return ""
