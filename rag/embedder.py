"""Embedding interface, the Gemini embedding API, and the optional local model."""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod

from google import genai
from google.genai import types

from .config import Settings
from .errors import ConfigError, LLMError
from .llm import TRANSIENT_CODES, is_transient, translate_error

RETRYABLE_CODES = TRANSIENT_CODES | {429}


class Embedder(ABC):
    @abstractmethod
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...


def _normalize(vector: list[float]) -> list[float]:
    """gemini-embedding-001 only returns unit-length vectors at its full size, so shorter ones are scaled here."""
    norm = math.sqrt(sum(x * x for x in vector))
    return vector if norm == 0 else [x / norm for x in vector]


class GeminiEmbedder(Embedder):
    """Embeddings from the Gemini API. The text of every chunk and question is sent to Google."""

    BATCH_SIZE = 50
    ATTEMPTS = 4

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        client=None,
        sleep=time.sleep,
    ) -> None:
        self._client = client or genai.Client(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        self._sleep = sleep

    def _embed_batch(self, batch: list[str], task_type: str) -> list[list[float]]:
        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self._dimensions)
        response = None
        for attempt in range(self.ATTEMPTS):
            try:
                response = self._client.models.embed_content(model=self._model, contents=batch, config=config)
                break
            except Exception as exc:
                if attempt == self.ATTEMPTS - 1 or not is_transient(exc, RETRYABLE_CODES):
                    raise translate_error(exc) from exc
                self._sleep(min(2 ** (attempt + 1), 16))
        vectors = [_normalize(list(item.values)) for item in response.embeddings]
        if len(vectors) != len(batch):
            raise LLMError("The embedding service returned an unexpected result. Try again.")
        return vectors

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start : start + self.BATCH_SIZE], task_type))
        return vectors

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class E5Embedder(Embedder):
    """multilingual-e5 models need a `query: ` or `passage: ` prefix on every input.
    Needs `pip install -r requirements-local.txt` and about 1.2 GB of free memory."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"query: {text}"])[0]


def embedding_tag(settings: Settings) -> str:
    """Names the vector space, so vectors from different models never share a collection.
    Empty for the original local model, which keeps existing local collections valid."""
    if settings.embedding_provider == "local":
        return ""
    return f"{settings.gemini_embedding_model}-{settings.embedding_dimensions}"


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "local":
        return E5Embedder(settings.embedding_model)
    if not settings.api_key:
        raise ConfigError("GEMINI_API_KEY is not set. Add it to your .env file and restart the app.")
    return GeminiEmbedder(settings.api_key, settings.gemini_embedding_model, settings.embedding_dimensions)

