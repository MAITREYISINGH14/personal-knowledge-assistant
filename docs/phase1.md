# Phase 1: Baseline RAG

**Goal:** a deliberately simple system to beat later.

## How it works
1. `pka/ingestion/loaders.py` reads `.md` (heading-aware), `.txt` and `.pdf` (per page) into text
   blocks tagged with their section, and gets `doc_id`, version and the document's own date.
2. `pka/chunking/fixed.py` cuts every 180 words with a 20-word overlap, ignoring meaning and headings.
3. `pka/embeddings.py` embeds chunks with all-MiniLM-L6-v2 (384 dimensions).
4. Vectors go to Qdrant (`pka_baseline` collection); chunk text and document records go to SQLite (`data/pka.db`).
5. `pka/baseline.py` embeds the question, takes the top 5 chunks and asks Claude to answer with citations.

**Why 180 words, not 500 tokens:** all-MiniLM-L6-v2 only reads the first 256 tokens of any input,
so 500-token chunks would have their second half silently ignored. 180 words fits inside the limit,
and the ingest report tells you how many chunks still went over. Phase 2 moves to bge-base-en (512 tokens).

**What the baseline ignores on purpose:** document structure when cutting, version and date
(it retrieves from v1 and v2 alike), and broad questions. Those gaps are what Phases 2–5 fix.

## Commands
```bash
python -m pka.ingest --reset                 # build the index
python -m pka.ask "What is my current rent?"  # try a question
python -m eval.run_eval --system pka.baseline:BaselineSystem            # retrieval scores
python -m eval.run_eval --system pka.baseline:BaselineSystem --judge    # + answer correctness
uvicorn pka.api.main:app --reload            # API at http://127.0.0.1:8000/docs
```
