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
Expected answer: {expected}
System answer: {answer}

Grade the system answer on factual agreement with the expected answer, ignoring wording and citations.
Reply with JSON only: {{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}"""

SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class Judge:
    def __init__(self):
        if settings.llm_provider == "none":
            raise RuntimeError("No LLM configured (LLM_PROVIDER=none); run without --judge.")

    def grade(self, question: str, expected: str, answer: str) -> dict:
        if not answer.strip():
            return {"verdict": "incorrect", "score": 0.0, "reason": "Empty answer."}
        text = chat("You are a strict, fair grader. Output JSON only.",
                    PROMPT.format(question=question, expected=expected, answer=answer), max_tokens=200)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            result = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            result = {}
        verdict = str(result.get("verdict", "")).lower()
        if verdict not in SCORES:  # small models sometimes skip the JSON; fall back to keywords
            low = text.lower()
            verdict = next((v for v in ("incorrect", "partial", "correct") if v in low), "incorrect")
        return {"verdict": verdict, "score": SCORES[verdict], "reason": result.get("reason", text[:120])}
