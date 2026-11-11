"""Ask a question from the terminal:  python -m pka.ask "What is my current rent?" """
import sys

from pka.baseline import BaselineSystem


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m pka.ask "your question"')
        return
    question = " ".join(sys.argv[1:])
    from pka.llm import LLMError
    try:
        result = BaselineSystem().answer(question)
    except LLMError as e:
        print(f"\nLLM error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n{e}")
        sys.exit(1)
    print("\n" + (result.answer or "(No LLM configured, so showing retrieved chunks only.)") + "\n")
    print("Sources:")
    for i, c in enumerate(result.retrieved, 1):
        where = f" | {c.section[:60]}" if c.section else ""
        print(f"  [{i}] {c.doc_id} v{c.version}{where}  (score {c.score:.3f})")


if __name__ == "__main__":
    main()
