"""Add evidence phrases to eval/eval_set.json (safe to run more than once).

Evidence makes a retrieval hit require the chunk that actually contains the answer,
instead of any chunk from the right document. Your notes and answers are left unchanged.
"""
import json
from pathlib import Path

EVIDENCE = {  # question id -> one list of phrases per source, in order
    "fact-001": [["₹84,000"]], "fact-002": [["10:30 pm"]], "fact-003": [["under ₹330"]],
    "fact-004": [["company VPN"]], "fact-005": [["outside India"]],
    "fact-006": [["boundary detection"]], "fact-007": [["768"]], "fact-008": [["3 whistles"]],
    "fact-009": [["₹500 per day"]], "fact-010": [["92nd percentile"]], "fact-011": [["65 hours"]],
    "fact-012": [["h = 8", "h=8", "8 parallel attention"]],
    "fact-013": [["govern, map, measure", "map, measure, and manage", "map, measure and manage"]],
    "fact-014": [["beginning or end"]], "fact-015": [["RAG-Sequence"]],
    "temporal-001": [["₹30,240"]], "temporal-002": [["₹28,000"]],
    "temporal-003": [["Minor day-to-day repairs"], ["less than ₹2,000"]],
    "temporal-005": [["up to three days"]], "temporal-006": [["₹1,000 per month"]],
    "temporal-007": [["Fridays"], ["11:00 and 16:00"]], "temporal-008": [["₹4,00,000"]],
}


def main():
    path = Path(__file__).resolve().parent.parent / "eval" / "eval_set.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    updated = 0
    for q in data["questions"]:
        for src, phrases in zip(q["sources"], EVIDENCE.get(q["id"], [])):
            if src.get("evidence") != phrases:
                src["evidence"] = phrases
                updated += 1
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Evidence set on {updated} sources across {len(EVIDENCE)} questions.")
    print("Next: python -m scripts.check_evidence")


if __name__ == "__main__":
    main()
