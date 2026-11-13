"""Build the cross-phase comparison tables for the write-up from saved eval reports.

  python -m eval.results          # print the tables, write docs/results.md, update README.md

For each system it uses the latest COMPLETE judged report (all questions, not stopped early);
if there is none, it uses the latest one and marks it. Nothing is re-run: run the evals first.
"""
import json
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
START, END = "<!-- results:start -->", "<!-- results:end -->"

# In the order the phases were built (Phase 5 came before Phase 4).
JUDGED = [
    ("baseline", "1", "Baseline: fixed 180-word chunks, MiniLM, dense top-5"),
    ("semantic", "2", "+ semantic chunks, contextual headers, bge-base"),
    ("hybrid_rerank", "3", "+ BM25 hybrid search, cross-encoder reranker, per-document cap"),
    ("temporal", "5", "+ version validity dates, time-aware retrieval, change logs"),
    ("summary", "4", "+ hierarchical summaries (collapsed tree)"),
    ("summary_rewrite", "7", "+ query rewriting (optional)"),
]
RETRIEVAL = [
    ("semantic_retrieval", "Semantic chunks"),
    ("hybrid_retrieval", "+ BM25 + fusion + cap"),
    ("hybrid_rerank_retrieval", "+ reranker"),
    ("temporal_retrieval", "+ time-aware retrieval"),
    ("summary_retrieval", "+ summaries"),
    ("summary_rewrite_retrieval", "+ query rewriting"),
]


def load_reports(reports_dir: Path):
    out = []
    for p in sorted(Path(reports_dir).glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        r["_file"] = p.name
        out.append(r)
    return out


def pick(reports, system: str):
    """Latest complete report for a system; else the latest at all (flagged)."""
    mine = [r for r in reports if r.get("system") == system and "all" in r.get("summary", {})]
    if not mine:
        return None, ""
    full_n = max(r["summary"]["all"]["n"] for r in reports if "all" in r.get("summary", {}))
    complete = [r for r in mine if not r.get("partial") and r["summary"]["all"]["n"] == full_n]
    if complete:
        return complete[-1], ""
    r = mine[-1]
    return r, f"partial: {r['summary']['all']['n']} of {full_n} questions"


def fmt(x):
    return "—" if x is None else f"{x:.2f}"


def stamp(report) -> str:
    m = re.match(r"(\d{4})(\d{2})(\d{2})-", report["_file"])
    return f"{m[1]}-{m[2]}-{m[3]}" if m else ""


def build(reports_dir: Path) -> str:
    reports = load_reports(reports_dir)
    lines = ["### Answer quality by phase", "",
             "| Phase | System | Hit@5 | MRR | Version clean | Correct: fact | overview | temporal | **all** | Run |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    notes, llms = [], set()
    for system, phase, label in JUDGED:
        r, flag = pick(reports, system)
        if not r:
            continue
        s = r["summary"]
        llms.add(r.get("llm") or "?")
        cell = lambda t, k: fmt(s.get(t, {}).get(k)) if t in s else "—"
        lines.append(f"| {phase} | {label} | {cell('all', 'hit_rate')} | {cell('all', 'mrr')} | "
                     f"{cell('temporal', 'version_clean')} | {cell('fact', 'correctness')} | "
                     f"{cell('overview', 'correctness')} | {cell('temporal', 'correctness')} | "
                     f"**{cell('all', 'correctness')}** | {stamp(r)}{' ⚠' if flag else ''} |")
        if flag:
            notes.append(f"⚠ `{system}`: {flag} ({r['_file']}).")
    if len(lines) == 4:
        lines.append("| | (no judged reports found) | | | | | | | | |")

    lines += ["", "### Retrieval only (exact and repeatable: no LLM involved)", "",
              "| Step | Hit@5 | Full hit@5 | MRR | Version clean |", "|---|---|---|---|---|"]
    found = False
    for system, label in RETRIEVAL:
        r, flag = pick(reports, system)
        if not r:
            continue
        found = True
        a = r["summary"]["all"]
        vc = r["summary"].get("temporal", {}).get("version_clean")
        lines.append(f"| {label} | {fmt(a['hit_rate'])} | {fmt(a['full_hit_rate'])} | {fmt(a['mrr'])} | "
                     f"{fmt(vc)} |{' ⚠ ' + flag if flag else ''}")
    if not found:
        lines.append("| (no retrieval-only reports found) | | | | |")

    lines += ["", f"_Answers and grades by {', '.join(sorted(llms)) or '—'}. With 10–15 questions per type, "
              "differences under ~0.10 in the correctness columns are within run-to-run noise; "
              "Hit@5 and MRR are exact._"]
    if notes:
        lines += [""] + notes
    lines += ["", f"_Generated {datetime.now():%Y-%m-%d %H:%M} by `python -m eval.results`._"]
    return "\n".join(lines) + "\n"


def update_readme(readme: Path, block: str) -> bool:
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{block}{END}{tail}", encoding="utf-8")
    return True


def main():
    block = build(HERE / "reports")
    print(block)
    out = ROOT / "docs" / "results.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("# Results\n\n" + block, encoding="utf-8")
    print(f"Wrote {out.relative_to(ROOT)}"
          + (" and updated README.md" if update_readme(ROOT / "README.md", block) else ""))


if __name__ == "__main__":
    main()
