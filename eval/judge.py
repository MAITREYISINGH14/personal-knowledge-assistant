"""Optional LLM judge: grades an answer against the expected answer."""
import json

from pka.config import settings

PROMPT = """You are grading a question-answering system.

Question: {question}
Expected answer: {expected}
System answer: {answer}

Grade the system answer on factual agreement with the expected answer, ignoring wording.
Reply with JSON only: {{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}"""

SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class Judge:
    def __init__(self):
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; run without --judge.")
        import anthropic  # imported lazily so the harness runs without it
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.judge_model

    def grade(self, question: str, expected: str, answer: str) -> dict:
        if not answer.strip():
            return {"verdict": "incorrect", "score": 0.0, "reason": "Empty answer."}
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            messages=[{"role": "user", "content": PROMPT.format(
                question=question, expected=expected, answer=answer)}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        text = text.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            result = json.loads(text)
            verdict = result.get("verdict", "incorrect")
        except json.JSONDecodeError:
            verdict, result = "incorrect", {"reason": f"Unparseable judge output: {text[:80]}"}
        return {"verdict": verdict, "score": SCORES.get(verdict, 0.0),
                "reason": result.get("reason", "")}
