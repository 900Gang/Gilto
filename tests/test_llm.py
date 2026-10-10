import httpx
import pytest
from google.genai import errors

from rag.config import Settings
from rag.errors import ConfigError, LLMError
from rag.llm import GeminiLLM, make_llm, translate_error


class FakeAPIError(errors.APIError):
    """Builds an SDK error without depending on its constructor signature."""

    def __init__(self, code, message):
        Exception.__init__(self, message)
        self.code = code
        self.message = message


class Chunk:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, chunks=None, error=None, text="rewritten", fail_first=0, fail_error=None):
        self.chunks = chunks or []
        self.error = error
        self.text = text
        self.fail_first = fail_first
        self.fail_error = fail_error
        self.stream_calls = []
        self.generate_calls = []

    def generate_content_stream(self, model, contents, config):
        self.stream_calls.append({"model": model, "contents": contents, "config": config})
        if self.error:
            raise self.error
        if len(self.stream_calls) <= self.fail_first:
            raise self.fail_error
        yield from self.chunks

    def generate_content(self, model, contents, config):
        self.generate_calls.append({"model": model, "contents": contents, "config": config})
        if self.error:
            raise self.error

        class Response:
            text = self.text

        return Response()
    
class FakeClient:
    def __init__(self, models):
        self.models = models


def make(models, fallback=""):
    return GeminiLLM("key", "main-model", "fast-model", fallback, client=FakeClient(models), sleep=lambda s: None)

def test_stream_yields_text_and_skips_empty_chunks():
    llm = make(FakeModels(chunks=[Chunk("Hello "), Chunk(None), Chunk("world")]))
    assert list(llm.stream("sys", [{"role": "user", "content": "hi"}])) == ["Hello ", "world"]


def test_stream_uses_main_model_system_prompt_and_maps_roles():
    models = FakeModels(chunks=[Chunk("x")])
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    list(make(models).stream("be careful", messages))
    call = models.stream_calls[0]
    assert call["model"] == "main-model"
    assert call["config"].system_instruction == "be careful"
    assert [c.role for c in call["contents"]] == ["user", "model", "user"]


def test_complete_fast_uses_fast_model():
    models = FakeModels(text="  standalone question  ")
    assert make(models).complete_fast("sys", "prompt") == "standalone question"
    assert models.generate_calls[0]["model"] == "fast-model"


@pytest.mark.parametrize(
    "error, expected",
    [
        (FakeAPIError(429, "quota exceeded"), "rate limit"),
        (FakeAPIError(403, "permission denied"), "API key"),
        (FakeAPIError(400, "API key not valid. Please pass a valid API key."), "API key"),
        (FakeAPIError(500, "internal"), "busy"),
        (httpx.ConnectError("offline"), "Could not reach"),
        (RuntimeError("weird"), "Something went wrong"),
    ],
)
def test_errors_become_friendly_messages(error, expected):
    assert expected in str(translate_error(error))


def test_stream_errors_are_translated():
    llm = make(FakeModels(error=FakeAPIError(429, "quota")))
    with pytest.raises(LLMError, match="rate limit"):
        list(llm.stream("sys", [{"role": "user", "content": "hi"}]))


def test_complete_fast_errors_are_translated():
    llm = make(FakeModels(error=FakeAPIError(403, "denied")))
    with pytest.raises(LLMError, match="API key"):
        llm.complete_fast("sys", "prompt")


def test_make_llm_without_key_raises_config_error():
    import dataclasses

    settings = dataclasses.replace(Settings.from_env(), api_key="")
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        make_llm(settings)
        
def test_transient_error_is_retried_then_succeeds():
    models = FakeModels(chunks=[Chunk("ok")], fail_first=2, fail_error=FakeAPIError(503, "high demand"))
    assert list(make(models).stream("sys", [{"role": "user", "content": "hi"}])) == ["ok"]
    assert [c["model"] for c in models.stream_calls] == ["main-model"] * 3


def test_fallback_model_used_after_main_attempts_fail():
    models = FakeModels(chunks=[Chunk("ok")], fail_first=3, fail_error=FakeAPIError(503, "high demand"))
    llm = make(models, fallback="backup-model")
    assert list(llm.stream("sys", [{"role": "user", "content": "hi"}])) == ["ok"]
    assert models.stream_calls[-1]["model"] == "backup-model"


def test_persistent_503_ends_in_friendly_error():
    models = FakeModels(error=FakeAPIError(503, "high demand"))
    with pytest.raises(LLMError, match="busy"):
        list(make(models, fallback="backup-model").stream("sys", [{"role": "user", "content": "hi"}]))
    assert len(models.stream_calls) == 4


def test_non_transient_error_is_not_retried():
    models = FakeModels(error=FakeAPIError(403, "denied"))
    with pytest.raises(LLMError, match="API key"):
        list(make(models, fallback="backup-model").stream("sys", [{"role": "user", "content": "hi"}]))
    assert len(models.stream_calls) == 1