"""LLM access for answers and the eval judge.

Set LLM_PROVIDER in .env:
  groq               free tier, cloud (GROQ_API_KEY from console.groq.com)
  gemini             free tier, cloud (GEMINI_API_KEY from aistudio.google.com)
  anthropic          Claude (ANTHROPIC_API_KEY)
  ollama             free, local
  openai_compatible  any OpenAI-style API (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL)
  none               retrieval only

For groq and gemini the model is chosen automatically: the first model in the preference
list that your key can actually use is picked once, saved to data/llm_model.json, and
reused on every later run so eval scores stay comparable. Set LLM_MODEL to override.
"""
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from functools import lru_cache

from pka.config import ROOT, settings

SYSTEM = """You answer questions using only the numbered context passages from the user's own documents.
Rules:
- Use only the context. If it does not contain the answer, say you couldn't find it in the documents.
- Cite passages with plain square-bracket numbers only, like [1] or [2][3], right after the
  claims they support. Never use any other citation style.
- If passages from different versions of the same document disagree, say which version says what.
- Be concise: a direct answer in one to four sentences. No headings or bullet lists unless the
  question asks for a list."""

PRESETS = {
    # provider: (base URL, key variable, models in order of preference)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", [
        "llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b", "llama-3.1-8b-instant"]),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", [
        "gemini-2.5-flash", "gemini-2.5-flash-lite"]),
}
MODEL_CACHE = ROOT / "data" / "llm_model.json"
USER_AGENT = "pka-knowledge-assistant/1.0"   # Cloudflare blocks Python's default agent (error 1010)
LONG_WAIT = 120                              # a Retry-After longer than this means "stop for now"


class LLMError(RuntimeError):
    pass


class RateLimitExhausted(LLMError):
    """A daily quota (or a very long wait) was hit. Stop and resume later."""


class ModelUnavailable(LLMError):
    pass


# ---------------------------------------------------------------- HTTP plumbing
@lru_cache(maxsize=1)
def _ssl_context():
    """certifi's certificates: python.org Pythons on macOS ship without system ones."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_last_call = 0.0


def _throttle():
    global _last_call
    wait = settings.llm_min_interval - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def _request(url, payload=None, headers=None, timeout=180):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET", headers={
        "Content-Type": "application/json", "User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode())


def _post_json(url: str, payload: dict, headers: dict = None, retries: int = 8) -> dict:
    for attempt in range(retries + 1):
        try:
            return _request(url, payload, headers)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:400]
            low = body.lower()
            if e.code == 429:
                retry_after = float(e.headers.get("Retry-After") or 0)
                if "per day" in low or "daily" in low or retry_after > LONG_WAIT:
                    raise RateLimitExhausted(
                        f"Free-tier daily limit reached ({body[:160]}). Progress is saved; "
                        "run the same command again later to continue.") from e
                if attempt < retries:
                    delay = retry_after or min(60, 5 * 2 ** attempt)
                    print(f"  (rate limited, waiting {delay:.0f}s)")
                    time.sleep(delay)
                    continue
            if e.code in (500, 502, 503) and attempt < retries:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            if e.code == 404 or "model_not_found" in low or "does not exist" in low:
                raise ModelUnavailable(f"Model not available to this key: {body}") from e
            if "1010" in body:
                raise LLMError("Blocked by the provider's firewall (Cloudflare 1010).") from e
            if e.code in (401, 403):
                raise LLMError(f"HTTP {e.code}: the API key was rejected. Check it in .env. {body}") from e
            raise LLMError(f"HTTP {e.code} from {url}: {body}") from e
        except urllib.error.URLError as e:
            if attempt < 2:
                time.sleep(3)
                continue
            raise LLMError(f"Could not reach {url}: {e.reason}. Check your internet connection.") from e
    raise LLMError(f"Gave up on {url} after {retries} retries.")


# ---------------------------------------------------------------- model handling
def _is_reasoning(model: str) -> bool:
    return "gpt-oss" in model or "qwen3" in model


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _openai_chat(base_url, api_key, model, system, user, max_tokens):
    if not api_key:
        raise LLMError("No API key found for this provider. Add it to .env.")
    payload = {"model": model, "temperature": 0,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": user}]}
    if _is_reasoning(model):
        # Reasoning models spend tokens "thinking" first; give them room and keep it short.
        payload["max_tokens"] = max(max_tokens, 2048)
        if "gpt-oss" in model:
            payload["reasoning_effort"] = "low"
    else:
        payload["max_tokens"] = max_tokens
    _throttle()
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        data = _post_json(url, payload, headers)
    except LLMError as e:
        if "reasoning_effort" in payload and "reasoning" in str(e).lower():
            payload.pop("reasoning_effort")          # provider didn't accept the option
            data = _post_json(url, payload, headers)
        else:
            raise
    content = (data["choices"][0]["message"].get("content") or "")
    return _THINK.sub("", content).strip()


def _load_pinned(provider):
    try:
        return json.loads(MODEL_CACHE.read_text()).get(provider)
    except (OSError, ValueError):
        return None


def _save_pinned(provider, model):
    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(MODEL_CACHE.read_text())
    except (OSError, ValueError):
        data = {}
    data[provider] = model
    MODEL_CACHE.write_text(json.dumps(data, indent=2))


_resolved: dict = {}


def resolve_model(provider: str) -> str:
    """The model to use for a preset provider: LLM_MODEL, else the pinned one, else probe."""
    if settings.llm_model:
        return settings.llm_model
    if provider in _resolved:
        return _resolved[provider]
    base_url, key_var, candidates = PRESETS[provider]
    pinned = _load_pinned(provider)
    if pinned:
        _resolved[provider] = pinned
        return pinned
    api_key = os.getenv(key_var, "")
    tried = []
    for model in candidates:
        try:
            _openai_chat(base_url, api_key, model, "Reply with OK.", "OK?", 16)
        except ModelUnavailable:
            tried.append(model)
            continue
        print(f"  Using {provider} model: {model} (remembered in {MODEL_CACHE.name})")
        _save_pinned(provider, model)
        _resolved[provider] = model
        return model
    raise LLMError(f"None of these {provider} models are available to your key: {tried}. "
                   "Set LLM_MODEL in .env to one listed in your provider console.")


def active_model() -> str:
    p = settings.llm_provider
    if p in PRESETS:
        return resolve_model(p)
    return {"anthropic": settings.answer_model, "ollama": settings.ollama_model,
            "openai_compatible": settings.llm_model}.get(p, "none")


# ---------------------------------------------------------------- public API
def chat(system: str, user: str, max_tokens: int = 600) -> str:
    provider = settings.llm_provider
    if provider in PRESETS:
        base_url, key_var, _ = PRESETS[provider]
        model = resolve_model(provider)
        try:
            return _openai_chat(base_url, os.getenv(key_var, ""), model, system, user, max_tokens)
        except ModelUnavailable:
            if settings.llm_model:
                raise LLMError(f"LLM_MODEL={model} isn't available to your key. "
                               "Pick another from your provider console.")
            _resolved.pop(provider, None)               # pinned model withdrawn: pick again
            try:
                MODEL_CACHE.unlink()
            except OSError:
                pass
            print(f"  {model} is no longer available; choosing another model.")
            model = resolve_model(provider)
            return _openai_chat(base_url, os.getenv(key_var, ""), model, system, user, max_tokens)

    if provider == "openai_compatible":
        return _openai_chat(settings.llm_base_url, settings.llm_api_key, settings.llm_model,
                            system, user, max_tokens)

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        msg = client.messages.create(model=settings.answer_model, max_tokens=max_tokens,
                                     system=system, messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    if provider == "ollama":
        try:
            data = _request(f"{settings.ollama_url}/api/chat", {
                "model": settings.ollama_model, "stream": False,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "options": {"temperature": 0, "num_predict": max_tokens}})
        except urllib.error.URLError as e:
            raise LLMError("Could not reach Ollama. Open the Ollama app or run 'ollama serve'.") from e
        if "error" in data:
            raise LLMError(f"Ollama: {data['error']}")
        return _THINK.sub("", data["message"]["content"]).strip()

    raise LLMError(f"No LLM configured (LLM_PROVIDER={provider!r}).")


def format_context(chunks) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        label = f"{c.doc_id} v{c.version}" + (f" | {c.section}" if c.section else "")
        parts.append(f"[{i}] ({label})\n{c.text}")
    return "\n\n".join(parts)


class Answerer:
    def answer(self, question: str, chunks) -> str:
        return chat(SYSTEM, f"Context:\n\n{format_context(chunks)}\n\nQuestion: {question}")
