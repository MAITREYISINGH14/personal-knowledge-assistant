# Personal Knowledge Assistant

Ask questions about your own documents (PDFs, notes, agreements, plans) and get answers that cite
their sources, understand **when** something was true, and know **what changed** between versions.

Built in phases, each measured against the same evaluation set before moving on:
semantic chunking with context-aware boundaries, hybrid search with reranking, temporal awareness
for document versions, hierarchical summaries, and a self-updating web interface.

## What it does

- **Answers with citations**: every claim links to the passage it came from.
- **Understands time**: "What is my rent *now*?", "What was it *in June 2026*?", "What *changed* when it
  was renewed?" each search only what was in force at that time, or compare the versions directly.
- **Keeps history automatically**: edit a note and the old text is kept as the previous version; delete a
  file and it disappears from the index, with a permanent record of what happened.
- **Handles broad questions**: summaries at several levels let it answer "how is this report organised?"
  as well as "what is the late fee?".
- **Runs on free tiers**: local embedding models and Qdrant; Groq's free tier (or Gemini, Anthropic,
  or a local Ollama model) for answers and summaries, with rate-limit handling and resumable jobs.

## Results

35 questions written before any system was built: 15 factual, 10 overview, 10 about time and versions.
A retrieval hit only counts if the passage containing the answer was retrieved (checked with evidence
phrases); answers are graded by an LLM judge against reference answers.

<!-- results:start -->
### Answer quality by phase

| Phase | System | Hit@5 | MRR | Version clean | Correct: fact | overview | temporal | **all** | Run |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Baseline: fixed 180-word chunks, MiniLM, dense top-5 | 0.86 | 0.69 | — | 0.80 | 0.60 | 0.65 | **0.70** | 2026-11-12 |
| 2 | + semantic chunks, contextual headers, bge-base | 0.94 | 0.76 | — | 0.93 | 0.70 | 0.55 | **0.76** | 2026-11-12 |
| 3 | + BM25 hybrid search, cross-encoder reranker, per-document cap | 0.97 | 0.88 | — | 1.00 | 0.65 | 0.25 | **0.69** | 2026-11-12 |
| 5 | + version validity dates, time-aware retrieval, change logs | 0.97 | 0.92 | 1.00 | 1.00 | 0.70 | 0.75 | **0.84** | 2026-11-12 |
| 4 | + hierarchical summaries (collapsed tree) | 0.94 | 0.91 | 1.00 | 1.00 | 0.75 | 0.85 | **0.89** | 2026-11-13 ⚠ |

### Retrieval only (exact and repeatable: no LLM involved)

| Step | Hit@5 | Full hit@5 | MRR | Version clean |
|---|---|---|---|---|
| Semantic chunks | 0.94 | 0.86 | 0.80 | — |
| + BM25 + fusion + cap | 0.97 | 0.91 | 0.83 | — |
| + reranker | 0.97 | 0.94 | 0.88 | — |
| + time-aware retrieval | 0.97 | 0.94 | 0.92 | 1.00 |
| + summaries | 0.94 | 0.91 | 0.91 | 1.00 |

_Answers and grades by groq/openai/gpt-oss-120b. With 10–15 questions per type, differences under ~0.10
in the correctness columns are within run-to-run noise; Hit@5 and MRR are exact._

⚠ Phase 4 was measured with 18 of 19 summary trees built (the 100-page survey was still pending).
Re-run the eval and `python -m eval.results` when it is complete; this block is regenerated from the reports.
<!-- results:end -->

## What we learned

1. **Measure the measuring first.** The first baseline counted "some chunk of the right document" as a
   hit and graded answers against every detail of the reference. Requiring evidence phrases removed
   false hits, and grading only what the question asked removed false failures, before any system work.
2. **Semantic chunking pays off most in ranking.** Fact MRR rose from 0.59 to 0.79. Phase 2 changed
   three things at once (boundaries, chunk headers with titles and file names, a stronger embedding
   model), so the gain is shared between them; headers alone added about 0.04 MRR in a later check.
3. **Smaller, sharper chunks made version confusion worse.** Old and new versions became many
   near-identical chunks competing for the same five slots: temporal correctness fell while facts rose.
4. **Better retrieval scores can mean worse answers.** Phase 3 reached Hit@5 0.97, yet temporal
   correctness dropped to 0.25: the per-document cap pulled old versions in next to new ones and the
   model mixed them up. Hit@k couldn't see this; grading answers did, and a "version clean" metric now does.
5. **Time is a filter, not a ranking signal.** Giving every chunk a validity period and filtering by the
   date in the question took temporal correctness from 0.25 to 0.75–0.85 with no wrong-version passages,
   and answering "what changed" from a stored change log made those answers complete.
6. **Summaries help broad questions and cost precision elsewhere.** Overview answers improved, but
   summaries sometimes take the slot of the exact passage (one fact question was still answered
   correctly, from a summary, but its source passage was no longer retrieved).

**Known gaps.** "What was my rent in June?" fails: the agreement only says "licence fee", and nothing
in retrieval connects the two words (optional query rewriting targets this). Questions about themes
across many separate notes rely on retrieving several notes, since summaries are per document. The
eval set is small and the time-question parser was checked against it, so try your own phrasings too.

## How it works

```
  data/corpus/  ──►  sync (on change)  ──►  load: PDF pages, Markdown headings, text
                                              │
                                              ▼
                       semantic chunks: sentence windows + structure + continuity
                       contextual header (title, file, section, date, version)
                                              │
            ┌─────────────────────────────────┼──────────────────────────────┐
            ▼                                 ▼                              ▼
   Qdrant: chunk vectors            SQLite: documents, chunks,       summary trees per version
   (bge-base) + validity dates      versions, change logs, history   (parts → sections → overview)

  question ─► time scope (now / as of date / what changed) ─► dense + BM25 (+ rewrites)
           ─► rank fusion ─► date filter ─► cross-encoder rerank ─► per-version cap
           ─► top 5 chunks and/or summaries (+ change log) ─► LLM answer with [n] citations
```

| Phase | Adds | Main code |
|---|---|---|
| 0 | Eval set, harness, metrics | `eval/` |
| 1 | Baseline RAG | `pka/baseline.py`, `pka/ingest.py` |
| 2 | Semantic chunking, contextual headers | `pka/chunking/semantic.py` |
| 3 | BM25, rank fusion, reranker, diversity cap | `pka/hybrid.py`, `pka/retrieval/` |
| 5 | Validity periods, chunk diffs, time-aware retrieval | `pka/versioning.py`, `pka/temporal.py`, `pka/query_time.py` |
| 4 | Hierarchical summaries, incremental and resumable | `pka/summarize.py`, `pka/summary_system.py` |
| 6 | Folder sync, edit versioning, web interface | `pka/sync.py`, `pka/watch.py`, `pka/app.py`, `pka/web/` |
| 7 | Sync history, query rewriting, results, write-up | `pka/query_rewrite.py`, `eval/results.py` |

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env              # add GROQ_API_KEY (free) or another provider
docker compose up -d              # Qdrant
python -m unittest discover tests
python -m pka.app                 # http://127.0.0.1:8000, syncs data/corpus automatically
```

Other entry points: `python -m pka.ask "question"`, `python -m pka.sync`, `python -m pka.summarize`,
`python -m eval.run_eval --system pka.summary_system:SummarySystem --judge`, `python -m eval.results`.
Phase-by-phase notes are in `docs/`.

**Your data stays local** except the passages sent to the answer model. Versions created from edits
exist only in `data/pka.db` and `.qdrant/`, so back those up.

## Limitations and next steps

- Scanned PDFs need text recognition (OCR) first; pages without text are reported, not read.
- Dates come from a "Last updated" / "Effective date" line in the document, else the file's date.
- The time parser handles common English phrasing ("in June 2026", "as of the end of July", ISO dates),
  not relative phrases like "last month" or "before the renewal".
- No login: the app only listens on your own computer.
- Next: cross-document summaries for "themes across my notes", relative-date understanding,
  OCR for scanned pages, and a larger eval set written by someone other than the builder.
