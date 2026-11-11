"""Turn files into Documents: text blocks tagged with their heading path.

doc_id and version come from the file name: rental_agreement__v2.md -> ("rental_agreement", 2).
"""
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pka.ingestion.dates import parse_doc_date

SUPPORTED = {".pdf", ".md", ".txt"}
VERSION_RE = re.compile(r"^(?P<doc_id>.+?)__v(?P<version>\d+)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Block:
    text: str
    section: str = ""   # heading path, e.g. "Title > 5. Security Deposit"; "p.3" for PDFs


@dataclass
class Document:
    doc_id: str
    version: int
    path: Path
    sha256: str
    title: str
    doc_date: Optional[str]        # date stated inside the document, if any
    file_mtime: str
    blocks: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def parse_name(stem: str):
    m = VERSION_RE.match(stem)
    return (m["doc_id"], int(m["version"])) if m else (stem, 1)


def _clean(text: str) -> str:
    text = text.replace("**", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_markdown(raw: str):
    """Split Markdown into paragraph blocks, each tagged with its heading path."""
    stack: list = []          # [(level, title)]
    blocks, para, in_code = [], [], False

    def flush():
        if para:
            text = _clean("\n".join(para))
            if text:
                blocks.append(Block(text, " > ".join(t for _, t in stack)))
            para.clear()

    for line in raw.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            para.append(line)
            continue
        m = None if in_code else HEADING_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), _clean(m.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        elif not line.strip() and not in_code:
            flush()
        else:
            para.append(line)
    flush()
    title = ""
    for line in raw.splitlines():
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) == 1:
            title = m.group(2)
            break
    return blocks, _clean(title)


def parse_text(raw: str):
    paras = [p for p in re.split(r"\n\s*\n", raw) if p.strip()]
    return [Block(_clean(p)) for p in paras], ""


def parse_pdf(path: Path):
    try:
        import pymupdf
    except ImportError:          # older PyMuPDF releases
        import fitz as pymupdf
    blocks, title = [], ""
    with pymupdf.open(path) as pdf:
        title = (pdf.metadata or {}).get("title") or ""
        for i, page in enumerate(pdf, 1):
            text = page.get_text("text")
            text = re.sub(r"-\n(?=[a-z])", "", text)           # re-join hyphenated words
            text = re.sub(r"\s*\n\s*", " ", text)
            text = _clean(text)
            if text:
                blocks.append(Block(text, f"p.{i}"))
    return blocks, title


def load_file(path: Path) -> Document:
    path = Path(path)
    data = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        blocks, title = parse_pdf(path)
        header = blocks[0].text if blocks else ""
    else:
        raw = data.decode("utf-8", errors="replace")
        blocks, title = parse_markdown(raw) if suffix == ".md" else parse_text(raw)
        header = raw
    doc_id, version = parse_name(path.stem)
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return Document(
        doc_id=doc_id, version=version, path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        title=title or doc_id.replace("_", " "),
        doc_date=parse_doc_date(header), file_mtime=mtime, blocks=blocks,
    )


def discover(corpus_dir: Path):
    return sorted(p for p in Path(corpus_dir).rglob("*")
                  if p.is_file() and p.suffix.lower() in SUPPORTED and p.name != "README.md")
