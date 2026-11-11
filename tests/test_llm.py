"""LLM layer and eval resilience, tested with a fake HTTP layer (no network, no model)."""
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import pka.llm as llm


class FakeResp:
    def __init__(self, payload): self.payload = payload
    def read(self): return json.dumps(self.payload).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False


def ok(text="ok"):
    return FakeResp({"choices": [{"message": {"content": text}}]})


def http_error(code, body=b"", headers=None):
    return urllib.error.HTTPError("u", code, "err", headers or {}, io.BytesIO(body))


class LLMTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patches = [
            mock.patch.object(llm, "MODEL_CACHE", Path(self.tmp.name) / "llm_model.json"),
            mock.patch.object(llm, "settings", mock.Mock(
                llm_provider="groq", llm_model="", llm_min_interval=0,
                ollama_url="http://x", ollama_model="m")),
            mock.patch.dict("os.environ", {"GROQ_API_KEY": "k"}),
            mock.patch("time.sleep"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        llm._resolved.clear()
        self.addCleanup(llm._resolved.clear)
        self.addCleanup(self.tmp.cleanup)


class TestGroq(LLMTestCase):
    def test_headers_and_reply(self):
        with mock.patch("urllib.request.urlopen", return_value=ok("hi")) as op:
            self.assertEqual(llm.chat("s", "u"), "hi")
            req = op.call_args[0][0]
            self.assertTrue(req.full_url.startswith("https://api.groq.com/openai/v1"))
            self.assertEqual(req.headers["Authorization"], "Bearer k")
            self.assertNotIn("urllib", req.get_header("User-agent").lower())

    def test_falls_back_to_available_model_and_pins_it(self):
        responses = [http_error(404, b'{"error":{"code":"model_not_found"}}'), ok(), ok("answer")]
        with mock.patch("urllib.request.urlopen", side_effect=responses) as op:
            self.assertEqual(llm.chat("s", "u"), "answer")
            used = json.loads(op.call_args[0][0].data)["model"]
        self.assertEqual(used, "openai/gpt-oss-120b")
        self.assertEqual(json.loads(llm.MODEL_CACHE.read_text())["groq"], "openai/gpt-oss-120b")

    def test_pinned_model_is_reused_without_probing(self):
        llm.MODEL_CACHE.write_text(json.dumps({"groq": "openai/gpt-oss-20b"}))
        with mock.patch("urllib.request.urlopen", return_value=ok()) as op:
            llm.chat("s", "u")
            self.assertEqual(op.call_count, 1)

    def test_reasoning_model_settings_and_think_stripping(self):
        llm.MODEL_CACHE.write_text(json.dumps({"groq": "openai/gpt-oss-120b"}))
        with mock.patch("urllib.request.urlopen", return_value=ok("<think>hmm</think> Final.")) as op:
            self.assertEqual(llm.chat("s", "u", max_tokens=200), "Final.")
            body = json.loads(op.call_args[0][0].data)
        self.assertGreaterEqual(body["max_tokens"], 2048)
        self.assertEqual(body["reasoning_effort"], "low")

    def test_retries_short_rate_limit(self):
        llm.MODEL_CACHE.write_text(json.dumps({"groq": "llama-3.3-70b-versatile"}))
        with mock.patch("urllib.request.urlopen",
                        side_effect=[http_error(429, b"slow", {"Retry-After": "3"}), ok("done")]):
            self.assertEqual(llm.chat("s", "u"), "done")

    def test_daily_limit_stops(self):
        llm.MODEL_CACHE.write_text(json.dumps({"groq": "llama-3.3-70b-versatile"}))
        with mock.patch("urllib.request.urlopen",
                        side_effect=http_error(429, b"Rate limit reached: tokens per day (TPD)")):
            with self.assertRaises(llm.RateLimitExhausted):
                llm.chat("s", "u")

    def test_bad_key_message(self):
        llm.MODEL_CACHE.write_text(json.dumps({"groq": "x"}))
        with mock.patch("urllib.request.urlopen", side_effect=http_error(401, b"invalid_api_key")):
            with self.assertRaisesRegex(llm.LLMError, "API key"):
                llm.chat("s", "u")

    def test_missing_key(self):
        with mock.patch.dict("os.environ", {"GROQ_API_KEY": ""}):
            with self.assertRaises(llm.LLMError):
                llm.chat("s", "u")


class TestOllama(LLMTestCase):
    def test_chat(self):
        llm.settings.llm_provider = "ollama"
        with mock.patch("urllib.request.urlopen", return_value=FakeResp({"message": {"content": " hi "}})):
            self.assertEqual(llm.chat("s", "u"), "hi")


class TestJudgeParsing(unittest.TestCase):
    def test_messy_judge_output(self):
        from eval import judge
        with mock.patch.object(judge, "settings", mock.Mock(llm_provider="groq")), \
             mock.patch.object(judge, "chat", return_value='Sure! {"verdict": "partial", "reason": "x"}'):
            self.assertEqual(judge.Judge().grade("q", "a", "b")["score"], 0.5)
        with mock.patch.object(judge, "settings", mock.Mock(llm_provider="groq")), \
             mock.patch.object(judge, "chat", return_value="This answer is correct."):
            self.assertEqual(judge.Judge().grade("q", "a", "b")["verdict"], "correct")


class TestEvalResume(unittest.TestCase):
    """A run stopped by a daily limit resumes from the cache without repeating LLM calls."""

    def test_resume(self):
        from eval import run_eval
        from pka.interfaces import RetrievedChunk

        calls = {"n": 0}

        class FakeSystem:
            name = "fake"
            answerer = object()
            def retrieve(self, q, as_of=None):
                return [RetrievedChunk("rental_agreement", "t", section="Security Deposit", version=1)]
            def generate(self, q, chunks):
                calls["n"] += 1
                if calls["n"] == 3:
                    raise llm.RateLimitExhausted("daily limit")
                return "an answer"

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(run_eval, "HERE", Path(tmp)), \
             mock.patch.object(run_eval, "load_system", return_value=FakeSystem()), \
             mock.patch("pka.llm.active_model", return_value="m"), \
             mock.patch("sys.argv", ["run_eval", "--system", "x",
                                     "--eval-set", "tests/fixtures/example_eval_set.json"]), \
             mock.patch("builtins.print"):
            with self.assertRaises(SystemExit) as stop:
                run_eval.main()
            self.assertEqual(stop.exception.code, 2)          # partial run
            first = calls["n"]
            run_eval.main()                                   # resumes and finishes
            self.assertEqual(calls["n"] - first, 4)           # only the 4 unfinished questions
            reports = list((Path(tmp) / "reports").glob("*.json"))
            self.assertEqual(len(reports), 2)


if __name__ == "__main__":
    unittest.main()


class TestEvidence(unittest.TestCase):
    def test_evidence_required_for_hit(self):
        from eval.metrics import score_retrieval
        from eval.schema import Question, Source
        from pka.interfaces import RetrievedChunk
        q = Question("x", "fact", "q", "a", [Source("doc", "", None, ["92nd percentile"])])
        wrong = RetrievedChunk("doc", "we used the 90th to 95th range")
        right = RetrievedChunk("doc", "ME: 92nd   Percentile right now")
        self.assertFalse(score_retrieval(q, [wrong], 5)["hit"])
        self.assertEqual(score_retrieval(q, [wrong, right], 5)["rr"], 0.5)

    def test_dash_normalisation(self):
        from eval.metrics import has_evidence
        self.assertTrue(has_evidence("RAG\u2011Sequence model", ["RAG-Sequence"]))
