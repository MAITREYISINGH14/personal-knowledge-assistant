# Phase 6: Automation and interface

## Run it
```bash
docker compose up -d            # Qdrant
python -m pka.app               # web interface + automatic folder syncing
```
Open http://127.0.0.1:8000 (it opens for you). Add, edit or delete files in `data/corpus/`; within
about half a minute the header shows "Last sync … 1 added" and you can ask about them.

Useful flags: `--no-watch` (no automatic sync; the Sync button still works), `--port 9000`,
`--no-browser`, `--interval 20 --settle 8` (how often to look, and how long a change must stay quiet),
`--auto-summaries 30` (also build summaries after a sync, at most 30 API calls each time).

Without the web page: `python -m pka.sync` (one pass), `python -m pka.watch` (keep watching).

## What happens to your files
| You do | The assistant does |
|---|---|
| add `trip.md` | indexes it as version 1 |
| edit `trip.md` (plain name) | **keeps the old text as v1 and adds v2**; "what changed in trip?" and "as of last month" work |
| edit `policy__v2.md` | overwrites that version (you manage `__vN` files yourself) |
| delete a file | removes it (all versions, for a plain name) |
| rename a file | old name removed, new name added as a new document |

Safety rails: nothing is removed if the folder looks empty or if one pass would remove more than half
the index (`python -m pka.sync --force-prune` overrides). Hidden files and editor temp files are ignored.

**History lives in the index.** Versions made from edits exist only in `data/pka.db` and the Qdrant
volume (`.qdrant/`), not as files. Back both up. `python -m pka.ingest --chunker semantic --reset`
would delete that history, so it refuses unless you add `--force-reset`. Use `pka.sync` for updates.

## The interface
- **Ask**: answers with clickable [1] citations; the right-hand panel shows the passages, their
  version (current / superseded) and whether they are summaries. "Show chunk details" adds scores and
  full text. A "Time frame" tag shows how the question's timing was understood.
- **Documents**: version history with dates, the change log between versions, and the chunks a
  document was split into ("Show chunk boundaries" outlines each chunk), plus its summaries.
- **Header**: index size, summary progress, last sync, whether the folder is being watched.

If the answer model is unavailable (daily limit), you still get the retrieved passages.

## How it stays current
A change marker (`data/index_generation`) is updated after each sync or summary build. The running app
checks it before each question and reloads its in-memory search index when it changed, so no restart is
needed. When the first summaries appear, the app also upgrades itself from plain time-aware search.

## Summaries
They are not built automatically (they cost API calls). The header shows how many documents still need
them; run `python -m pka.summarize` whenever you like. Edited documents get their old summaries dropped
and rebuilt; unchanged parts reuse cached summaries, so only changed parts cost calls.

## Notes
- The app has no login and listens only on your own computer. Don't use `--allow-network` on a shared network.
- `pka_baseline` (the Phase 1 index used for comparisons) is not touched by sync.
- Keep demo files out of `data/corpus/` when you run evals, or delete them afterwards (sync removes them
  from the index), so your scores stay comparable.
