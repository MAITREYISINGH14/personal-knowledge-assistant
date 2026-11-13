"""Central settings, read from environment / .env."""
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent


def _default_provider() -> str:
    if os.getenv("LLM_PROVIDER"):
        return os.getenv("LLM_PROVIDER").strip().lower()
    for provider, key in (("anthropic", "ANTHROPIC_API_KEY"), ("groq", "GROQ_API_KEY"),
                          ("gemini", "GEMINI_API_KEY")):
        if os.getenv(key):
            return provider
    return "none"


@dataclass(frozen=True)
class Settings:
    llm_provider: str = _default_provider()      # groq | gemini | anthropic | ollama | openai_compatible | none
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    llm_min_interval: float = float(os.getenv("LLM_MIN_INTERVAL", "0"))  # seconds between calls
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    answer_model: str = os.getenv("ANSWER_MODEL", "claude-sonnet-5-5")
    judge_model: str = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    sqlite_path: Path = ROOT / os.getenv("SQLITE_PATH", "data/pka.db")
    corpus_dir: Path = ROOT / os.getenv("CORPUS_DIR", "data/corpus")

    # Phase 1 baseline chunking. all-MiniLM-L6-v2 truncates input at 256 word-piece
    # tokens, so chunks are sized to fit (~180 words is roughly 230-250 tokens).
    baseline_collection: str = "pka_baseline"
    baseline_chunk_words: int = 180
    baseline_overlap_words: int = 20
    top_k: int = 5

    # Phase 2 semantic chunking
    semantic_collection: str = "pka_semantic"
    semantic_embed_model: str = os.getenv("SEMANTIC_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
    semantic_query_prefix: str = "Represent this sentence for searching relevant passages: "
    boundary_model: str = os.getenv("BOUNDARY_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_max_tokens: int = int(os.getenv("CHUNK_MAX_TOKENS", "400"))
    chunk_min_tokens: int = int(os.getenv("CHUNK_MIN_TOKENS", "80"))
    chunk_percentile: float = float(os.getenv("CHUNK_PERCENTILE", "92"))
    chunk_window: int = int(os.getenv("CHUNK_WINDOW", "2"))

    # Phase 3 hybrid search and reranking (uses the semantic index; no re-ingest needed)
    candidate_pool: int = int(os.getenv("CANDIDATE_POOL", "30"))     # from each of dense and BM25
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    max_per_doc: int = int(os.getenv("MAX_PER_DOC", "3"))           # per document version

    # Phase 4 hierarchical summaries
    summary_collection: str = "pka_summaries"
    summary_provider: str = os.getenv("SUMMARY_PROVIDER", "").strip().lower()  # default: LLM_PROVIDER
    summary_model: str = os.getenv("SUMMARY_MODEL", "")
    group_max_tokens: int = int(os.getenv("GROUP_MAX_TOKENS", "1800"))   # input per summary call
    group_max_children: int = int(os.getenv("GROUP_MAX_CHILDREN", "8"))

    # Phase 7: optional query rewriting (one small API call per new question; see pka/query_rewrite.py)
    query_rewrite: bool = os.getenv("QUERY_REWRITE", "").strip().lower() in ("1", "true", "on", "yes")


settings = Settings()
