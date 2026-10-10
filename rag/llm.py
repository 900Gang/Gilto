"""Gemini wrapper. The rest of the app depends on the LLM interface, not on the SDK."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator

import httpx
from google import genai
from google.genai import errors, types

from .config import Settings
from .errors import ConfigError, LLMError

# Gemini counts internal "thinking" tokens against this limit, so keep it generous.
ANSWER_MAX_TOKENS = 8192
FAST_MAX_TOKENS = 1024
TRANSIENT_CODES = {500, 503, 504}
MAIN_ATTEMPTS = 3


class LLM(ABC):
    @abstractmethod
    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        """Yield answer text as it is generated. messages use roles 'user' and 'assistant'."""

    @abstractmethod
    def complete_fast(self, system: str, prompt: str) -> str:
        """One short completion with the cheap model."""


def is_transient(exc: Exception) -> bool:
    if isinstance(exc, errors.APIError):
        return getattr(exc, "code", None) in TRANSIENT_CODES
    return isinstance(exc, httpx.TransportError)


def translate_error(exc: Exception) -> LLMError:
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        text = str(getattr(exc, "message", "") or exc).lower()
        if code in (401, 403) or "api key" in text:
            return LLMError("The Gemini API key was rejected. Check GEMINI_API_KEY.")
        if code == 429:
            return LLMError("The Gemini API rate limit or quota was reached. Wait a minute and try again.")
        if code in TRANSIENT_CODES:
            return LLMError("Gemini is busy right now. Wait a few seconds and try again.")
        return LLMError(f"The Gemini API returned an error ({code}). Try again in a moment.")
    if isinstance(exc, httpx.TransportError):
        return LLMError("Could not reach the Gemini API. Check your internet connection and try again.")
    return LLMError("Something went wrong while generating the answer. Try again.")


class GeminiLLM(LLM):
    def __init__(
        self,
        api_key: str,
        model: str,
        fast_model: str,
        fallback_model: str = "",
        client=None,
        sleep=time.sleep,
    ) -> None:
        self._client = client or genai.Client(api_key=api_key)
        self._model = model
        self._fast_model = fast_model
        self._fallback_model = fallback_model
        self._sleep = sleep

    @staticmethod
    def _contents(messages: list[dict]) -> list:
        return [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
        ]

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        """Retry transient failures with backoff, then try the fallback model once.
        Retrying only happens before any text has been yielded."""
        models = [self._model] * MAIN_ATTEMPTS
        if self._fallback_model and self._fallback_model != self._model:
            models.append(self._fallback_model)
        config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=ANSWER_MAX_TOKENS)
        contents = self._contents(messages)

        for attempt, model in enumerate(models):
            started = False
            try:
                for chunk in self._client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                ):
                    if chunk.text:
                        started = True
                        yield chunk.text
                return
            except Exception as exc:
                last_attempt = attempt == len(models) - 1
                if started or last_attempt or not is_transient(exc):
                    raise translate_error(exc) from exc
                self._sleep(min(2**attempt, 4))

    def complete_fast(self, system: str, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._fast_model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=FAST_MAX_TOKENS),
            )
            return (response.text or "").strip()
        except Exception as exc:
            raise translate_error(exc) from exc


def make_llm(settings: Settings) -> LLM:
    if not settings.api_key:
        raise ConfigError("GEMINI_API_KEY is not set. Add it to your .env file and restart the app.")
    return GeminiLLM(settings.api_key, settings.model, settings.fast_model, settings.fallback_model)