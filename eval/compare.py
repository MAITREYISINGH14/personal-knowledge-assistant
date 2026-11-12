"""Compare two eval reports side by side.

  python -m eval.compare                    # latest baseline vs latest semantic report
  python -m eval.compare A.json B.json      # any two reports
"""
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = [("hit_rate", "Hit@k"), ("mrr", "MRR"), ("correctness", "Correct")]


def latest(system: str) -> str:
    import re
    pattern = re.compile(rf"_{re.escape(system)}(_\d+)?$")    # skips e.g. *_semantic_retrieval
    files = sorted(f for f in glob.glob(str(HERE / "reports" / "*.json"))
                   if pattern.search(Path(f).stem))
    if not files:
        sys.exit(f"No report found for '{system}'. Run the eval for it first.")
    return files[-1]


def main():
    a, b = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else (latest("baseline"), latest("semantic"))
    ra, rb = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    print(f"A: {Path(a).name}  ({ra['system']}, {ra.get('llm', '')})")
    print(f"B: {Path(b).name}  ({rb['system']}, {rb.get('llm', '')})\n")
    head = f"{'type':<10}" + "".join(f"{label:>22}" for _, label in METRICS)
    print(head)
    print("-" * len(head))
    for t in ["fact", "overview", "temporal", "all"]:
        if t not in ra["summary"] or t not in rb["summary"]:
            continue
        cells = []
        for key, _ in METRICS:
            x, y = ra["summary"][t][key], rb["summary"][t][key]
            if x is None or y is None:
                cells.append(f"{'—':>22}")
                continue
            d = y - x
            flag = "  " if abs(d) < 0.10 else (" ▲" if d > 0 else " ▼")
            cells.append(f"{x:>6.2f} → {y:.2f} ({d:+.2f}){flag}")
        print(f"{t:<10}" + "".join(f"{c:>22}" for c in cells))
    print("\n▲/▼ marks changes of 0.10 or more; smaller changes are within run-to-run noise.")

    rows_a = {r["id"]: r for r in ra["rows"]}
    changed = []
    for r in rb["rows"]:
        old = rows_a.get(r["id"])
        if old and old["hit"] != r["hit"]:
            changed.append(f"  {r['id']}: {'MISS → HIT' if r['hit'] else 'HIT → MISS'}")
    if changed:
        print("\nRetrieval changes per question:")
        print("\n".join(changed))


if __name__ == "__main__":
    main()
