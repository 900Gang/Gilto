"""Settings read from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return raw.strip().lower() in ("1", "true", "yes", "on") if raw else default


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)  # keep the key out of tracebacks and test output
    model: str
    fast_model: str
    fallback_model: str
    chroma_path: str
    embedding_provider: str
    embedding_model: str  # the local model, used when embedding_provider is "local"
    gemini_embedding_model: str
    embedding_dimensions: int
    max_file_mb: int
    max_pages_per_file: int
    max_docs_per_user: int
    max_total_pages: int
    chunk_size: int
    chunk_overlap: int
    similarity_threshold: float
    min_page_chars: int
    history_turns: int
    ocr_confidence_threshold: float
    ocr_dpi: int
    ocr_cache_path: str
    tesseract_cmd: str
    ocr_langs: str
    allowed_emails: str
    auth_disabled: bool
    sqlite_path: str
    hourly_question_limit: int
    daily_token_limit: int
    monthly_token_cap: int
    inactive_days: int
    hybrid_search: bool

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("EMBEDDING_PROVIDER", "gemini").strip().lower()
        if provider not in ("gemini", "local"):
            raise ValueError("EMBEDDING_PROVIDER must be 'gemini' or 'local'.")
        return cls(
            api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""),
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            fast_model=os.getenv("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite"),
            fallback_model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash"),
            chroma_path=os.getenv("CHROMA_PATH", "./chroma_db"),
            embedding_provider=provider,
            embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            embedding_dimensions=_int("EMBEDDING_DIMENSIONS", 768),
            max_file_mb=_int("MAX_FILE_MB", 20),
            max_pages_per_file=_int("MAX_PAGES_PER_FILE", 100),
            max_docs_per_user=_int("MAX_DOCS_PER_USER", 15),
            max_total_pages=_int("MAX_TOTAL_PAGES", 500),
            chunk_size=_int("CHUNK_SIZE", 800),
            chunk_overlap=_int("CHUNK_OVERLAP", 150),
            # Each embedding model spreads its similarity scores differently, so the
            # default depends on the provider. The Gemini value was measured on a small sample (relevant >= 0.65, unrelated <= 0.54).
            similarity_threshold=_float("SIMILARITY_THRESHOLD", 0.80 if provider == "local" else 0.60),
            min_page_chars=_int("MIN_PAGE_CHARS", 50),
            history_turns=_int("HISTORY_TURNS", 4),
            ocr_confidence_threshold=_float("OCR_CONFIDENCE_THRESHOLD", 60.0),
            ocr_dpi=_int("OCR_DPI", 200),
            ocr_cache_path=os.getenv("OCR_CACHE_PATH", "./ocr_cache"),
            tesseract_cmd=os.getenv("TESSERACT_CMD", ""),
            ocr_langs=os.getenv("OCR_LANGS", "eng+hin"),
            allowed_emails=os.getenv("ALLOWED_EMAILS", ""),
            auth_disabled=_bool("AUTH_DISABLED", False),
            sqlite_path=os.getenv("SQLITE_PATH", "./gilto.db"),
            hourly_question_limit=_int("HOURLY_QUESTION_LIMIT", 30),
            daily_token_limit=_int("DAILY_TOKEN_LIMIT", 200000),
            monthly_token_cap=_int("MONTHLY_TOKEN_CAP", 5000000),
            inactive_days=_int("INACTIVE_DAYS", 90),
            hybrid_search=_bool("HYBRID_SEARCH", True),
        )