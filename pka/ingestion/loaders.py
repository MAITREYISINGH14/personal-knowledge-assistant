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
    versioned: bool = False        # file name carries a __vN suffix

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
        if suffix == ".txt" and not title:           # a short first line works as a title
            first = raw.strip().splitlines()[0].strip() if raw.strip() else ""
            if 0 < len(first) <= 80:
                title = first
    doc_id, version = parse_name(path.stem)
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return Document(
        doc_id=doc_id, version=version, path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        title=title or doc_id.replace("_", " "),
        doc_date=parse_doc_date(header), file_mtime=mtime, blocks=blocks,
        versioned=bool(VERSION_RE.match(path.stem)),
    )


def file_sha(path) -> str:
    """SHA-256 of a file's bytes (same value as Document.sha256) without parsing it."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def discover(corpus_dir: Path):
    """Supported files in the corpus folder, skipping hidden files/folders (.git, .DS_Store)
    and editor or Office temp files (~$report.pdf, ~notes.md)."""
    root = Path(corpus_dir)
    out = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED or p.name == "README.md":
            continue
        if p.name.startswith((".", "~")):
            continue
        if any(part.startswith(".") for part in p.relative_to(root).parts[:-1]):
            continue
        out.append(p)
    return sorted(out)
