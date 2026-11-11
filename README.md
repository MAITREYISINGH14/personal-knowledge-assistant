# Personal Knowledge Assistant: Phase 0

Phase 0 sets up the project and, more importantly, the **evaluation harness**, so every
later phase (semantic chunking, reranking, summaries, versioning) can be measured
against the same questions.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY if you want the LLM judge
docker compose up -d            # Qdrant on :6333 (needed from Phase 1)
python -m unittest discover tests
```

## Layout

```
pka/
  interfaces.py     QASystem contract every phase implements (+ NullSystem)
  config.py         settings from .env
  api/main.py       FastAPI stub: /health works, /ask arrives in Phase 1
  ingestion/ chunking/ retrieval/ storage/   filled in by later phases
data/corpus/        your test documents (git-ignored); see its README
eval/
  eval_set.json     your questions + expected answers + where they live
  schema.py         format + validation + coverage warnings
  metrics.py        hit@k, full hit@k (all sources found), MRR
  judge.py          optional LLM grader for answer correctness
  run_eval.py       runs the set against a system, writes reports/
scripts/inventory_corpus.py   checks the corpus against Phase 0 targets
```

## Your Phase 0 tasks

1. **Build the corpus.** Drop 20–40 real documents into `data/corpus/`, including 3–4
   in two versions named `name__v1.pdf` / `name__v2.pdf`. Check with
   `python -m scripts.inventory_corpus`.
2. **Write the eval set.** Replace the 6 examples in `eval/eval_set.json` with 30–50
   questions about your corpus, at least 20% each of `fact`, `overview` and `temporal`.
   For each, record the expected answer and the `doc_id` + section (+ version) holding it.
3. **Run the harness.** `python -m eval.run_eval` should loop over all questions and
   write a report (all zeros with the NullSystem; that's expected).

**Exit criterion:** the inventory shows all targets ticked, the eval set loads with no
warnings, and the harness runs end to end.

## Writing good eval questions

- Write questions the way you'd actually ask them, not copied from the document text,
  or retrieval will look better than it is.
- Make some `fact` questions hinge on exact terms (names, numbers, IDs). They test BM25 in Phase 3.
- Make `overview` questions impossible to answer from one paragraph. They test Phase 4.
- For `temporal`, mix `current` (should prefer the newest version), `as_of` (a past date)
  and `change` (cite both versions) questions. They test Phase 5.
- Keep expected answers short and factual so the judge can grade them reliably.

## Plugging in a system (Phase 1 onward)

Any class with a `name` and an `answer(question, as_of=None) -> QAResult` method works:

```bash
python -m eval.run_eval --system pka.baseline:BaselineSystem --k 5 --judge
```

Each run writes `eval/reports/<timestamp>_<system>.md` and `.json`, so phases can be compared side by side.
