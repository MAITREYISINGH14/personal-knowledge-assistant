# Phase 5: Temporal awareness and versioning

## What it adds
1. **Validity periods.** Versions of a document are ordered by the date written inside them
   ("Last updated", "Effective date"), falling back to the file date. Each chunk gets
   `valid_from` = its version's date and `valid_to` = the next version's date (or "current").
   The first version also covers earlier dates, since nothing older exists.
2. **Chunk diffs.** Consecutive versions are compared chunk by chunk: identical text →
   unchanged; ≥ 0.80 embedding similarity → modified (with a word-level "before → after");
   leftovers → removed / added. Stored in SQLite as a readable change log.
3. **Time-aware questions** (`pka/query_time.py`), using only the question's own words:
   - *current* ("now", or no date): only chunks in force today
   - *as of* ("in June 2026", "as of the end of July", "on 5 August 2026"): chunks in force then
   - *change* ("what changed", "how did … change", "what happened to"): the changed chunks of
     the document, plus its full change log for the answer
4. **Version-clean metric.** For current / as-of questions: did retrieval avoid every chunk from
   the wrong version? Hit@5 couldn't see this contamination.
5. **Judge fix.** An empty or unclear grade is retried once, then marked *ungraded* and left out
   of the score instead of silently counting as incorrect. Odd citations like 【2】 become [2].

## Commands
```bash
python -m pka.versioning                          # add versions to the existing semantic index
python -m pka.versioning --show rental_agreement  # versions + change log of one document
python -m pka.ask "What was my rent in June 2026?"
python -m eval.run_eval --system pka.temporal:TemporalSystem --retrieval-only
python -m eval.compare hybrid_rerank_retrieval temporal_retrieval
python -m eval.run_eval --system pka.temporal:TemporalSystem --judge
python -m eval.compare hybrid_rerank temporal
```
After any re-ingest with `--chunker semantic`, versioning runs automatically.

## Pinning "today"
"Current" means today's date. To keep results reproducible, you can pin it in `.env`:
`PKA_TODAY=2026-11-13`.

## Known limits
- Dates come from the document's own header line; a document without one uses its file date.
- The time parser covers common English phrasings (month names, ISO dates, "as of", "end of").
  Relative phrases like "last month" or "before the renewal" aren't understood yet.
- The parser was checked against this eval set, so also try your own phrasings with pka.ask.
