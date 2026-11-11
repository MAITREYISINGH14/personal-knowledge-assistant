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


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    judge_model: str = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    sqlite_path: Path = ROOT / os.getenv("SQLITE_PATH", "data/pka.db")
    corpus_dir: Path = ROOT / os.getenv("CORPUS_DIR", "data/corpus")


settings = Settings()
