"""Phase 2: semantic chunking with context-aware boundaries.

1. Text becomes units (sentences, list items, whole tables/code).
2. Headings are hard boundaries; PDF page breaks are only soft hints.
3. Inside a section, every gap between units gets a boundary score:
       score = semantic distance between the WINDOW of units before the gap and the window after
             + structural bonus if a new paragraph / list item / page starts
             - continuity penalty if the next unit leans on the previous one ("However", "This"...)
4. Gaps scoring above the document's own percentile threshold become cuts.
5. Pieces that are too small merge across their weakest boundary; pieces that are too big split
   at their strongest internal gap. Tiny sections merge into a neighbour.
6. A short document stays one chunk.
"""
import re
from dataclasses import dataclass, field

import numpy as np

from pka.chunking.sentences import to_units

CONTINUITY = re.compile(
    r"^(however|this|these|that|those|it|its|they|their|them|therefore|thus|hence|so|also|but|and|"
    r"for example|for instance|in addition|moreover|furthermore|as a result|such|which|then|here)\b",
    re.IGNORECASE)


@dataclass
class SemanticChunk:
    index: int
    text: str
    section: str
    n_words: int
    n_tokens: int


@dataclass
class ChunkStats:
    units: int = 0
    semantic_cuts: int = 0     # cuts chosen because meaning shifted
    structural_cuts: int = 0   # cuts at headings
    forced_cuts: int = 0       # cuts made only because a piece was too long
    single_chunk_doc: bool = False
    threshold: float = 0.0
    extra: dict = field(default_factory=dict)


def _join(units):
    out = ""
    for u in units:
        sep = "\n" if (u.kind != "prose" or (out and out.endswith("\n"))) else " "
        out = (out + sep + u.text) if out else u.text
    return out.strip()


def _section_label(units):
    return " | ".join(dict.fromkeys(u.section for u in units if u.section))


def _hard_split_text(text, count_tokens, max_tokens):
    """Last resort for a single unit longer than the limit (e.g. a huge table)."""
    words, parts, cur = text.split(), [], []
    for w in words:
        cur.append(w)
        if count_tokens([" ".join(cur)])[0] > max_tokens:
            cur.pop()
            parts.append(" ".join(cur))
            cur = [w]
    if cur:
        parts.append(" ".join(cur))
    return parts


def semantic_chunks(blocks, *, embed_fn, count_tokens, is_pdf=False, max_tokens=400,
                    min_tokens=80, tiny_tokens=40, single_doc_tokens=320, percentile=92.0,
                    window=2, w_struct=0.05, w_cont=0.15):
    """embed_fn(list[str]) -> list of unit-normalised vectors.
    count_tokens(list[str]) -> list[int] token counts for the retrieval model."""
    stats = ChunkStats()
    units = to_units(blocks)
    stats.units = len(units)
    if not units:
        return [], stats
    tokens = count_tokens([u.text for u in units])

    def make(chunk_units, chunk_tokens, idx):
        text = _join(chunk_units)
        return SemanticChunk(idx, text, _section_label(chunk_units), len(text.split()),
                             int(sum(chunk_tokens)))

    # Short document: one chunk keeps all its context together.
    if sum(tokens) <= single_doc_tokens:
        stats.single_chunk_doc = True
        return [make(units, tokens, 0)], stats

    emb = np.asarray(embed_fn([u.text for u in units]), dtype=np.float32)

    # Segments separated by hard boundaries (headings). PDFs: one segment.
    seg_bounds, start = [], 0
    for i in range(1, len(units)):
        if not is_pdf and units[i].section != units[i - 1].section:
            seg_bounds.append((start, i))
            start = i
    seg_bounds.append((start, len(units)))
    forced_gaps = set()

    # Score every gap inside a segment. gap g sits between unit g and unit g+1.
    gap_score = {}
    for s, e in seg_bounds:
        for g in range(s, e - 1):
            left = emb[max(s, g - window + 1):g + 1].mean(axis=0)
            right = emb[g + 1:min(e, g + 1 + window)].mean(axis=0)
            cos = float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-9))
            score = (1.0 - cos)
            score += w_struct * units[g + 1].para_start
            score -= w_cont * bool(CONTINUITY.match(units[g + 1].text))
            gap_score[g] = score
    threshold = float(np.percentile(list(gap_score.values()), percentile)) if gap_score else 1e9
    stats.threshold = threshold

    pieces = []   # (start, end) unit ranges, in order, each inside one segment
    for s, e in seg_bounds:
        cuts = [g for g in range(s, e - 1) if gap_score[g] > threshold]
        bounds = [s] + [g + 1 for g in cuts] + [e]
        seg_pieces = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        size = lambda p: sum(tokens[p[0]:p[1]])

        # Merge pieces that are too small across their weaker neighbouring boundary.
        merged = True
        while merged and len(seg_pieces) > 1:
            merged = False
            for i, p in enumerate(seg_pieces):
                if size(p) >= min_tokens:
                    continue
                left_gap = gap_score.get(p[0] - 1) if i > 0 else None
                right_gap = gap_score.get(p[1] - 1) if i < len(seg_pieces) - 1 else None
                options = [(sc, j) for sc, j in ((left_gap, i - 1), (right_gap, i + 1)) if sc is not None]
                if not options:
                    continue
                _, j = min(options)
                a, b = sorted((i, j))
                if size((seg_pieces[a][0], seg_pieces[b][1])) > max_tokens:
                    continue
                seg_pieces[a:b + 1] = [(seg_pieces[a][0], seg_pieces[b][1])]
                merged = True
                break

        # Split pieces that are too big at their strongest internal gap.
        out, stack, forced = [], list(reversed(seg_pieces)), 0
        while stack:
            p = stack.pop()
            if size(p) <= max_tokens or p[1] - p[0] == 1:
                out.append(p)
                continue
            # Strongest gap wins; among near-equal gaps prefer one near the middle,
            # so forced splits give balanced pieces instead of slicing off one sentence.
            total = size(p)
            def split_key(x, p=p, total=total):
                before = sum(tokens[p[0]:x + 1]) / total
                return gap_score[x] - 0.05 * abs(before - 0.5)
            g = max(range(p[0], p[1] - 1), key=split_key)
            forced += 1
            forced_gaps.add(g)
            stack.extend([(g + 1, p[1]), (p[0], g + 1)])
        pieces.extend(out)

    # Merge tiny pieces (e.g. a two-line section) into the next piece, else the previous one.
    i = 0
    while i < len(pieces) and len(pieces) > 1:
        p = pieces[i]
        if sum(tokens[p[0]:p[1]]) < tiny_tokens:
            j = i + 1 if i + 1 < len(pieces) else i - 1
            a, b = sorted((i, j))
            if sum(tokens[pieces[a][0]:pieces[b][1]]) <= max_tokens + tiny_tokens:
                pieces[a:b + 1] = [(pieces[a][0], pieces[b][1])]
                continue
        i += 1

    # Count the cuts that survived, by reason.
    for s, _ in pieces[1:]:
        if not is_pdf and units[s].section != units[s - 1].section:
            stats.structural_cuts += 1
        elif (s - 1) in forced_gaps:
            stats.forced_cuts += 1
        else:
            stats.semantic_cuts += 1

    chunks = []
    for s, e in pieces:
        if e - s == 1 and tokens[s] > max_tokens:      # one enormous unit
            for part in _hard_split_text(units[s].text, count_tokens, max_tokens):
                chunks.append(SemanticChunk(len(chunks), part, units[s].section,
                                            len(part.split()), count_tokens([part])[0]))
            continue
        c = make(units[s:e], tokens[s:e], len(chunks))
        chunks.append(c)
    return chunks, stats


def context_header(doc, section: str) -> str:
    """Short header prepended to a chunk before embedding, so a chunk 'knows' where it is from."""
    parts = [f"Document: {doc.title}"]
    # The file name often says what a document is ("call_transcript_design_review") when the
    # title doesn't, so add it unless the title already covers it.
    name = str(getattr(doc, "doc_id", "") or "").replace("_", " ").strip()
    if name and name.lower() not in doc.title.lower():
        parts[0] += f" (file: {name})"
    names = []
    for sec in section.split(" | "):
        if not sec or sec.startswith("p."):
            continue
        trail = [x.strip() for x in sec.split(" > ") if x.strip() and x.strip() != doc.title.strip()]
        if trail and " > ".join(trail[-2:]) not in names:
            names.append(" > ".join(trail[-2:]))
    if names:
        parts.append("Section: " + "; ".join(names[:3]))
    if doc.doc_date:
        parts.append(f"Dated: {doc.doc_date}")
    if getattr(doc, "versioned", False):
        parts.append(f"Version {doc.version}")
    return " | ".join(parts)
