"""Run the eval set against a QA system and write a report.

Usage:
  python -m eval.run_eval                          # NullSystem, just exercises the harness
  python -m eval.run_eval --system pka.baseline:BaselineSystem --k 5 --judge
"""
import argparse
import importlib
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from eval.metrics import score_retrieval
from eval.schema import QUESTION_TYPES, coverage_warnings, load_eval_set

HERE = Path(__file__).resolve().parent


def load_system(spec: str):
    if spec == "null":
        from pka.interfaces import NullSystem
        return NullSystem()
    module, _, cls = spec.partition(":")
    return getattr(importlib.import_module(module), cls)()


def summarize(rows: list) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r["type"]].append(r)
        groups["all"].append(r)
    out = {}
    for name, items in groups.items():
        n = len(items)
        judged = [r["judge"]["score"] for r in items if r.get("judge")]
        out[name] = {
            "n": n,
            "hit_rate": sum(r["hit"] for r in items) / n,
            "full_hit_rate": sum(r["full_hit"] for r in items) / n,
            "mrr": sum(r["rr"] for r in items) / n,
            "correctness": sum(judged) / len(judged) if judged else None,
            "avg_latency_s": sum(r["latency_s"] for r in items) / n,
        }
    return out


def to_markdown(system_name: str, k: int, summary: dict, warnings: list) -> str:
    lines = [f"# Eval report: `{system_name}` (k={k})", "",
             f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", ""]
    if warnings:
        lines += ["**Eval-set warnings:**", ""] + [f"- {w}" for w in warnings] + [""]
    lines += ["| Type | N | Hit@k | Full hit@k | MRR | Correctness | Latency (s) |",
              "|---|---|---|---|---|---|---|"]
    for t in sorted(QUESTION_TYPES) + ["all"]:
        if t not in summary:
            continue
        s = summary[t]
        corr = f"{s['correctness']:.2f}" if s["correctness"] is not None else "—"
        lines.append(f"| {t} | {s['n']} | {s['hit_rate']:.2f} | {s['full_hit_rate']:.2f} | "
                     f"{s['mrr']:.2f} | {corr} | {s['avg_latency_s']:.2f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default=str(HERE / "eval_set.json"))
    ap.add_argument("--system", default="null", help="'null' or module.path:ClassName")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--judge", action="store_true", help="grade answers with the LLM judge")
    ap.add_argument("--only", choices=sorted(QUESTION_TYPES), help="run one question type")
    args = ap.parse_args()

    questions = load_eval_set(Path(args.eval_set))
    warnings = coverage_warnings(questions)
    if args.only:
        questions = [q for q in questions if q.type == args.only]

    system = load_system(args.system)
    judge = None
    if args.judge:
        from eval.judge import Judge
        judge = Judge()

    rows = []
    for q in questions:
        t0 = time.perf_counter()
        result = system.answer(q.question, as_of=q.as_of)
        latency = time.perf_counter() - t0
        row = {"id": q.id, "type": q.type, "question": q.question,
               "answer": result.answer, "latency_s": latency,
               "retrieved": [{"doc_id": c.doc_id, "section": c.section,
                              "version": c.version, "level": c.level}
                             for c in result.retrieved[:args.k]],
               **score_retrieval(q, result.retrieved, args.k)}
        if judge:
            row["judge"] = judge.grade(q.question, q.expected_answer, result.answer)
        rows.append(row)
        print(f"[{'HIT ' if row['hit'] else 'MISS'}] {q.id}: {q.question[:70]}")

    summary = summarize(rows)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = getattr(system, "name", args.system)
    out_dir = HERE / "reports"
    out_dir.mkdir(exist_ok=True)
    report = {"system": name, "k": args.k, "summary": summary,
              "warnings": warnings, "rows": rows}
    (out_dir / f"{stamp}_{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md = to_markdown(name, args.k, summary, warnings)
    (out_dir / f"{stamp}_{name}.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Reports written to {out_dir}/{stamp}_{name}.{{json,md}}")


if __name__ == "__main__":
    main()
