"""Phase 1 baseline: fixed-size word windows with overlap, ignoring meaning and structure.

Boundaries fall every N words regardless of headings. Each chunk still records every
section its words came from, so the eval can tell where the text originated.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    index: int
    text: str
    section: str        # all sections the chunk touches, joined with " | "
    n_words: int


def fixed_size_chunks(blocks, max_words: int = 180, overlap: int = 20):
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    words, tags = [], []
    for b in blocks:
        for w in b.text.split():
            words.append(w)
            tags.append(b.section)

    chunks, start, step = [], 0, max_words - overlap
    while start < len(words):
        end = min(start + max_words, len(words))
        sections = list(dict.fromkeys(t for t in tags[start:end] if t))
        chunks.append(Chunk(len(chunks), " ".join(words[start:end]), " | ".join(sections), end - start))
        if end == len(words):
            break
        start += step
    return chunks
