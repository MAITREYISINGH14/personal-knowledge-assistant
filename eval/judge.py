"""Optional LLM judge: grades an answer against the expected answer.

Uses whichever LLM_PROVIDER is configured. A small local model is a rougher judge than
Claude, so compare phases using the same judge rather than reading scores as absolute.
"""
import json
import re

from pka.config import settings
from pka.llm import chat

PROMPT = """You are grading a question-answering system.

Question: {question}
Reference answer: {expected}
System answer: {answer}

Grade whether the system answer correctly answers THE QUESTION, using the reference answer as the
source of truth for the facts.
- "correct": it states the facts the question asks for, and nothing it states contradicts the
  reference. Extra details in the reference that the question did not ask for are optional.
- "partial": it gets some of what the question asks for but misses or gets wrong another part.
- "incorrect": the key facts are wrong or missing, or it says it couldn't find the answer.
Ignore wording, formatting and citation style.
Reply with JSON only: {{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}"""

SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class Judge:
    def __init__(self):
        if settings.llm_provider == "none":
            raise RuntimeError("No LLM configured (LLM_PROVIDER=none); run without --judge.")

    def _ask(self, question, expected, answer):
        text = chat("You are a strict, fair grader. Output JSON only.",
                    PROMPT.format(question=question, expected=expected, answer=answer), max_tokens=200)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            result = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            result = {}
        verdict = str(result.get("verdict", "")).lower()
        if verdict not in SCORES:      # no clean JSON: look for an explicit verdict word
            low = text.lower()
            found = re.findall(r"\b(incorrect|partial|correct)\b", low)
            verdict = found[0] if len(set(found)) == 1 else ""
        return verdict, result.get("reason", text[:120]), text

    def grade(self, question: str, expected: str, answer: str) -> dict:
        if not answer.strip():
            return {"verdict": "incorrect", "score": 0.0, "reason": "Empty answer."}
        for _ in range(2):                 # retry once: reasoning models sometimes return nothing
            verdict, reason, raw = self._ask(question, expected, answer)
            if verdict in SCORES:
                return {"verdict": verdict, "score": SCORES[verdict], "reason": reason}
        # Never count a grading failure as a wrong answer: leave it out of the score instead.
        return {"verdict": "ungraded", "score": None, "reason": f"No usable verdict: {raw[:120]!r}"}
