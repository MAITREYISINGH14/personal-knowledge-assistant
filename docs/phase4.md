# Phase 4: Hierarchical summaries

## The tree
Each document version gets a summary tree on top of its Phase 2 chunks:
- **Level 1 "parts"**: one summary per group of consecutive chunks (up to ~1,800 tokens / 8 chunks)
- **Level 2+**: summaries of groups of parts, repeated until one **root overview**
- Single-chunk documents (most short notes) need no tree: the chunk already is the whole note.

The root prompt asks for the document's purpose, its main parts *in order*, and key points, which
is what questions like "how is this survey organised?" need.

## Retrieval: collapsed tree
Chunks and summaries go into one ranking (dense + BM25 → rerank → per-version cap), with the
Phase 5 date filter applied to both. Detailed questions are usually won by a chunk, broad ones
by a summary. This follows the RAPTOR paper's finding that searching all levels together works
better than walking the tree top-down.

## Cost control (free tier)
- Summaries use their **own model** (Groq: `openai/gpt-oss-20b` by default), so they don't use
  the answer model's daily quota. Override with `SUMMARY_MODEL`, or move summaries to another
  provider with `SUMMARY_PROVIDER=gemini` (+ `GEMINI_API_KEY`).
- Every summary is cached by a hash of exactly what it summarises. A run stopped by a daily limit
  resumes where it left off; unchanged parts of a new version reuse old summaries, so only
  changed parts and their ancestors cost calls (incremental re-summarisation).
- Documents are processed cheapest first; `--max-calls` and `--docs` give finer control.

## Commands
```bash
python -m pka.summarize --estimate
python -m pka.summarize                       # repeat on later days until it says "Done"
python -m pka.summarize --show raptor_tree_retrieval_2024
python -m pka.ask "How is the big survey of large language models organised?"
python -m eval.run_eval --system pka.summary_system:SummarySystem --retrieval-only
python -m eval.run_eval --system pka.summary_system:SummarySystem --judge
python -m eval.compare temporal summary
```

## Known limits
- Trees are per document. Questions spanning many notes (recurring themes) still rely on
  retrieving several notes; a corpus-level layer that clusters documents is a possible extension.
- Summaries can lose details or, rarely, misstate them. Facts are still answered from chunks when
  the reranker prefers them, and every summary is traceable to its chunks (`summary_nodes` table).
