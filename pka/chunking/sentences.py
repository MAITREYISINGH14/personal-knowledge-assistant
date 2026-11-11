"""Split text blocks into units for semantic chunking: sentences for prose, and whole
tables / code blocks / list items kept intact. Regex-based, so no extra model is needed."""
import re
from dataclasses import dataclass

ABBREVIATIONS = {"e.g.", "i.e.", "mr.", "ms.", "mrs.", "dr.", "prof.", "st.", "no.", "vs.",
                 "fig.", "figs.", "eq.", "sec.", "al.", "approx.", "dept.", "inc.", "ltd.", "co."}
SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9₹\"'(\[])")
LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)]|\[[ x~]\])\s+")


@dataclass
class Unit:
    text: str
    section: str
    para_start: bool       # first unit of a paragraph / page / list item
    kind: str = "prose"    # prose | list | table | code


def split_sentences(text: str):
    pieces, start = [], 0
    for m in SENT_END.finditer(text):
        candidate = text[start:m.start()].strip()
        last_word = candidate.split()[-1].lower() if candidate.split() else ""
        if last_word in ABBREVIATIONS or re.fullmatch(r"[A-Z]\.", candidate.split()[-1] if candidate.split() else ""):
            continue          # "e.g. The", "Dr. Rao", initials: not a sentence end
        pieces.append(candidate)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return [p for p in pieces if p]


def to_units(blocks):
    units = []
    for block in blocks:
        lines = block.text.split("\n")
        i, first = 0, True
        while i < len(lines):
            line = lines[i]
            if line.strip().startswith("```"):                     # code block: keep whole
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("```"):
                    j += 1
                units.append(Unit("\n".join(lines[i:j + 1]), block.section, first, "code"))
                i = j + 1
            elif line.strip().startswith("|"):                     # table: keep whole
                j = i
                while j < len(lines) and lines[j].strip().startswith("|"):
                    j += 1
                units.append(Unit("\n".join(lines[i:j]), block.section, first, "table"))
                i = j
            elif LIST_ITEM.match(line):                            # list item: one unit
                units.append(Unit(line.strip(), block.section, True, "list"))
                i += 1
            else:
                for k, sent in enumerate(split_sentences(line)):
                    units.append(Unit(sent, block.section, first and k == 0))
                i += 1
            first = False
    return [u for u in units if u.text.strip()]
