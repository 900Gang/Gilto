"""Settings read from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str
    fast_model: str
    fallback_model: str
    chroma_path: str
    embedding_model: str
    max_file_mb: int
    max_pages_per_file: int
    max_docs_per_user: int
    max_total_pages: int
    chunk_size: int
    chunk_overlap: int
    similarity_threshold: float
    min_page_chars: int
    history_turns: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""),
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            fast_model=os.getenv("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite"),
            fallback_model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash"),
            chroma_path=os.getenv("CHROMA_PATH", "./chroma_db"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            max_file_mb=_int("MAX_FILE_MB", 20),
            max_pages_per_file=_int("MAX_PAGES_PER_FILE", 100),
            max_docs_per_user=_int("MAX_DOCS_PER_USER", 15),
            max_total_pages=_int("MAX_TOTAL_PAGES", 500),
            chunk_size=_int("CHUNK_SIZE", 800),
            chunk_overlap=_int("CHUNK_OVERLAP", 150),
            similarity_threshold=_float("SIMILARITY_THRESHOLD", 0.80),
            min_page_chars=_int("MIN_PAGE_CHARS", 50),
            history_turns=_int("HISTORY_TURNS", 4),
        )