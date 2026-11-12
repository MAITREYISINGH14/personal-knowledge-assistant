"""Run the eval set against a QA system and write a report.

Usage:
  python -m eval.run_eval                          # NullSystem, just exercises the harness
  python -m eval.run_eval --system pka.baseline:BaselineSystem --judge

LLM answers and judge grades are cached in eval/cache/, so if a free-tier limit stops a run,
running the same command again continues where it left off instead of starting over.
"""
import argparse
import hashlib
import importlib
import json
import re
import sys
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
        judged = [r["judge"]["score"] for r in items
                  if r.get("judge") and r["judge"].get("score") is not None]
        clean = [r["version_clean"] for r in items if r.get("version_clean") is not None]
        out[name] = {
            "n": n,
            "hit_rate": sum(r["hit"] for r in items) / n,
            "full_hit_rate": sum(r["full_hit"] for r in items) / n,
            "mrr": sum(r["rr"] for r in items) / n,
            "correctness": sum(judged) / len(judged) if judged else None,
            "version_clean": sum(clean) / len(clean) if clean else None,
            "ungraded": sum(1 for r in items if r.get("judge", {}).get("verdict") == "ungraded"),
            "avg_latency_s": sum(r["latency_s"] for r in items) / n,
        }
    return out


def to_markdown(system_name, k, summary, warnings, llm_label="", note="") -> str:
    lines = [f"# Eval report: `{system_name}` (k={k})", "",
             f"_Generated {datetime.now():%Y-%m-%d %H:%M}_" + (f" · LLM: {llm_label}" if llm_label else ""), ""]
    if note:
        lines += [f"**{note}**", ""]
    if warnings:
        lines += ["**Eval-set warnings:**", ""] + [f"- {w}" for w in warnings] + [""]
    lines += ["| Type | N | Hit@k | Full hit@k | MRR | Version clean | Correctness | Latency (s) |",
              "|---|---|---|---|---|---|---|---|"]
    for t in sorted(QUESTION_TYPES) + ["all"]:
        if t not in summary:
            continue
        s = summary[t]
        corr = f"{s['correctness']:.2f}" if s["correctness"] is not None else "—"
        if s.get("ungraded"):
            corr += f" ({s['ungraded']} ungraded)"
        vc = f"{s['version_clean']:.2f}" if s.get("version_clean") is not None else "—"
        lines.append(f"| {t} | {s['n']} | {s['hit_rate']:.2f} | {s['full_hit_rate']:.2f} | "
                     f"{s['mrr']:.2f} | {vc} | {corr} | {s['avg_latency_s']:.2f} |")
    return "\n".join(lines) + "\n"


class Cache:
    """Answers and grades keyed by question + retrieved text, saved after every question."""

    def __init__(self, name: str):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        self.path = HERE / "cache" / f"{safe}.json"
        self.path.parent.mkdir(exist_ok=True)
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:20]

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default=str(HERE / "eval_set.json"))
    ap.add_argument("--system", default="null", help="'null' or module.path:ClassName")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--judge", action="store_true", help="grade answers with the LLM judge")
    ap.add_argument("--only", choices=sorted(QUESTION_TYPES), help="run one question type")
    ap.add_argument("--fresh", action="store_true", help="ignore cached answers and grades")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="score retrieval only: no answers, no judge, no API usage")
    args = ap.parse_args()

    questions = load_eval_set(Path(args.eval_set))
    warnings = coverage_warnings(questions)
    if args.only:
        questions = [q for q in questions if q.type == args.only]

    from pka.llm import LLMError, RateLimitExhausted
    try:
        system = load_system(args.system)
    except RuntimeError as e:
        sys.exit(str(e))
    name = getattr(system, "name", args.system)

    if args.retrieval_only:
        args.judge = False
        system.answerer = None
        name = f"{name}_retrieval"
    uses_llm = args.judge or bool(getattr(system, "answerer", None))
    llm_label = ""
    if uses_llm:
        from pka.config import settings
        from pka.llm import active_model
        try:
            llm_label = f"{settings.llm_provider}/{active_model()}"
        except LLMError as e:
            sys.exit(f"LLM setup problem: {e}")
    cache = Cache(f"{name}__{llm_label or 'nollm'}")
    if args.fresh:
        cache.data = {}

    judge = None
    if args.judge:
        from eval.judge import Judge
        judge = Judge()

    can_cache = hasattr(system, "retrieve") and hasattr(system, "generate")
    # Changing a prompt must invalidate cached answers/grades made with the old one.
    from pka.llm import SYSTEM as ANSWER_PROMPT
    judge_prompt = ""
    if judge:
        from eval.judge import PROMPT as judge_prompt
    rows, errors, stopped = [], 0, None
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        try:
            if can_cache:
                retrieved = system.retrieve(q.question, q.as_of)
                akey = Cache.key("answer", ANSWER_PROMPT, q.question, *[c.text for c in retrieved[:args.k]])
                cached = "" if args.retrieval_only else cache.get(akey)
                if cached is None:
                    answer = system.generate(q.question, retrieved)
                    cache.put(akey, answer)
                else:
                    answer = cached
            else:
                result = system.answer(q.question, as_of=q.as_of)
                retrieved, answer = result.retrieved, result.answer
            latency = time.perf_counter() - t0
            row = {"id": q.id, "type": q.type, "question": q.question, "answer": answer,
                   "latency_s": latency,
                   "retrieved": [{"doc_id": c.doc_id, "section": c.section, "version": c.version,
                                  "level": c.level} for c in retrieved[:args.k]],
                   **score_retrieval(q, retrieved, args.k)}
            if judge:
                jkey = Cache.key("judge", judge_prompt, q.question, q.expected_answer, answer)
                grade = cache.get(jkey)
                if grade is None:
                    grade = judge.grade(q.question, q.expected_answer, answer)
                    if grade["verdict"] != "ungraded":      # retry ungraded ones next run
                        cache.put(jkey, grade)
                row["judge"] = grade
        except RateLimitExhausted as e:
            stopped = str(e)
            break
        except LLMError as e:
            errors += 1
            print(f"[ERR ] {q.id}: {e}")
            continue
        rows.append(row)
        mark = "HIT " if row["hit"] else "MISS"
        verdict = f" | judge: {row['judge']['verdict']}" if row.get("judge") else ""
        print(f"[{mark}] {i:>2}/{len(questions)} {q.id}: {q.question[:60]}{verdict}")

    if not rows:
        sys.exit(stopped or "No questions completed.")

    note = ""
    if stopped:
        note = f"PARTIAL RUN: {len(rows)}/{len(questions)} questions. {stopped}"
    elif errors:
        note = f"{errors} question(s) failed with LLM errors and are not included."
    summary = summarize(rows)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = HERE / "reports"
    out_dir.mkdir(exist_ok=True)
    base, n = out_dir / f"{stamp}_{name}", 1
    while base.with_suffix(".json").exists():
        n += 1
        base = out_dir / f"{stamp}_{name}_{n}"
    report = {"system": name, "llm": llm_label, "k": args.k, "partial": bool(stopped),
              "summary": summary, "warnings": warnings, "rows": rows}
    base.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    md = to_markdown(name, args.k, summary, warnings, llm_label, note)
    base.with_suffix(".md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"Reports written to {base}.{{json,md}}")
    if stopped:
        print("\nStopped early because of a free-tier limit. Run the same command again later; "
              "finished questions are cached and won't be repeated.")
        sys.exit(2)


if __name__ == "__main__":
    main()
