# Phase 7: Hardening and write-up

## Sync history
Every sync records what it did in a permanent table (`sync_events` in `data/pka.db`): added,
new version (from an edit), updated in place, removed, and files it couldn't read (each problem once).
- **Header:** "Last change 3 min ago: new version of trip (v2)", plus when the folder was last checked.
- **Documents tab:** opens on *Recent activity*; each document shows its *History*; removed documents
  still open and show when and why they were removed.
- **API:** `GET /api/activity`.

## Query rewriting (optional)
`QUERY_REWRITE=1` in `.env`: each new question is also searched in up to 3 alternative wordings that use
the documents' likely vocabulary ("rent" → "monthly licence fee"). Results are merged; the reranker still
judges every passage against your original question. One small call per new question, on the summary
model (separate quota on Groq); rewrites are cached. With it on, the system's name gets a `_rewrite`
suffix so its eval reports stay separate:
```bash
QUERY_REWRITE=1 python -m eval.run_eval --system pka.summary_system:SummarySystem --retrieval-only
python -m eval.compare summary_retrieval summary_rewrite_retrieval
```
(Here even `--retrieval-only` makes up to 35 small API calls, once; later runs reuse the cache.)

## Smaller fixes
- Summary prompts describe part ranges as `the sections from "A" to "B"`, so ranges aren't read as titles.
  Existing summaries keep their wording (they're cached by content); new ones use the clearer form.
- Current Gemini model names are tried first.
- PDFs: pages without text are counted and reported ("3 of 40 pages have no readable text"); fully
  scanned files explain how to fix them (export with OCR).

## Results and write-up
`python -m eval.results` builds the cross-phase tables from your saved reports, writes `docs/results.md`
and refreshes the block in `README.md`. It picks each system's latest complete run (all questions, not
stopped early) and flags anything partial.
