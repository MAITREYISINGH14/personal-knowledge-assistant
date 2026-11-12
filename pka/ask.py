"""Ask a question from the terminal.

  python -m pka.ask "What is my current rent?"                  # best system so far
  python -m pka.ask --system baseline "What is my current rent?"
"""
import argparse
import sys

SYSTEMS = {
    "baseline": ("pka.baseline", "BaselineSystem"),
    "semantic": ("pka.semantic", "SemanticSystem"),
    "hybrid": ("pka.hybrid", "HybridSystem"),
    "rerank": ("pka.hybrid", "RerankSystem"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--system", choices=list(SYSTEMS), default="rerank")
    args = ap.parse_args()
    question = " ".join(args.question)

    import importlib
    from pka.llm import LLMError
    module, cls = SYSTEMS[args.system]
    try:
        system = getattr(importlib.import_module(module), cls)()
        result = system.answer(question)
    except LLMError as e:
        print(f"\nLLM error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n{e}")
        sys.exit(1)
    print("\n" + (result.answer or "(No LLM configured, so showing retrieved chunks only.)") + "\n")
    print("Sources:")
    for i, c in enumerate(result.retrieved, 1):
        where = f" | {c.section.split(' | ')[0].split(' > ')[-1][:50]}" if c.section else ""
        print(f"  [{i}] {c.doc_id} v{c.version}{where}  (score {c.score:.3f})")


if __name__ == "__main__":
    main()
