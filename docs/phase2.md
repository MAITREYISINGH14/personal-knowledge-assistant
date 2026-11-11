# Phase 2: Semantic chunking

**Goal:** cut documents where the meaning changes, and give every chunk enough context to be found.

## What changed from the baseline
| | Baseline (Phase 1) | Semantic (Phase 2) |
|---|---|---|
| Where cuts go | every 180 words | headings, then where the topic shifts |
| Chunk size | fixed | 80–400 tokens, varies with content |
| Short notes | split if over 180 words | kept whole (up to ~320 tokens) |
| Tables, lists, code | can be cut mid-way | kept intact |
| What gets embedded | bare chunk text | header (document, section, date, version) + text |
| Retrieval model | all-MiniLM-L6-v2 (384-d, 256 tokens) | bge-base-en-v1.5 (768-d, 512 tokens) |

## How a boundary is chosen (pka/chunking/semantic.py)
1. Text becomes units: sentences, list items, whole tables and code blocks.
2. Headings are hard boundaries. PDF page breaks are only a small hint.
3. Every gap between units gets a score, using MiniLM sentence embeddings:
   `distance(average of the 2 units before, average of the 2 after) + 0.05 if a new paragraph starts − 0.15 if the next unit starts with "However", "This", "It"...`
4. Gaps above the document's own 92nd-percentile score become cuts.
5. Pieces under 80 tokens merge across their weaker boundary; pieces over 400 split at their
   strongest gap (balanced if gaps are similar); sections under 40 tokens join a neighbour.

## Commands
```bash
python -m pka.ingest --chunker semantic --reset          # build the semantic index
python -m pka.chunks rental_agreement --version 2        # see how a document was cut
python -m eval.run_eval --system pka.semantic:SemanticSystem --judge
python -m eval.compare                                   # baseline vs semantic
```

## Tuning (optional)
Set in `.env`, then rebuild with `--reset`: `CHUNK_PERCENTILE` (try 90 and 95),
`CHUNK_MAX_TOKENS`, `CHUNK_MIN_TOKENS`, `CHUNK_WINDOW`. Change one thing at a time and
compare Hit@5 and MRR, which don't depend on the LLM.

## Note on attribution
Phase 2 changes three things at once (boundaries, headers, embedding model). To see which
helped most, you can switch `SEMANTIC_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2`
(and remove the query prefix in config) to isolate chunking from the model upgrade.
