"""Check that every evidence phrase in the eval set really appears in its document.

Reads the documents the same way ingestion does (so PDF text is extracted identically),
and reports any phrase that can't be found, e.g. because a PDF writes it differently.
"""
from eval.metrics import has_evidence, normalize
from eval.schema import load_eval_set
from pka.config import ROOT, settings
from pka.ingestion.loaders import discover, load_file


def main():
    docs = {}
    for path in discover(settings.corpus_dir):
        try:
            d = load_file(path)
        except Exception as e:
            print(f"  could not read {path.name}: {e}")
            continue
        docs[(d.doc_id, d.version)] = normalize(d.text)

    problems = checked = 0
    for q in load_eval_set(ROOT / "eval" / "eval_set.json"):
        for src in q.sources:
            if not src.evidence:
                continue
            checked += 1
            texts = [t for (doc_id, v), t in docs.items()
                     if doc_id == src.doc_id and (src.version is None or v == src.version)]
            if not texts:
                print(f"  {q.id}: document {src.doc_id} v{src.version or '*'} not found in corpus")
                problems += 1
            elif not any(has_evidence(t, src.evidence) for t in texts):
                print(f"  {q.id}: none of {src.evidence} found in {src.doc_id}")
                problems += 1
    print(f"\nChecked {checked} evidence entries: "
          + ("all found." if not problems else f"{problems} problem(s) above."))
    if problems:
        print("Fix: open the PDF, search for the fact, and put the exact words the PDF uses into "
              "that question's 'evidence' list in eval/eval_set.json.")


if __name__ == "__main__":
    main()
