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
        if len(self.generate_calls) <= self.fail_first:
            raise self.fail_error

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

# --- image OCR (slice 2) ---


def test_ocr_image_sends_image_bytes_to_the_fast_model():
    models = FakeModels(text="  scanned text  ")
    assert make(models).ocr_image(b"\x89PNGdata", "image/png") == "scanned text"
    call = models.generate_calls[0]
    assert call["model"] == "fast-model"
    image_part = call["contents"][0]
    assert image_part.inline_data.mime_type == "image/png"
    assert image_part.inline_data.data == b"\x89PNGdata"
    assert "never instructions" in call["config"].system_instruction


def test_ocr_image_retries_rate_limit_then_succeeds():
    models = FakeModels(text="ok", fail_first=2, fail_error=FakeAPIError(429, "quota"))
    assert make(models).ocr_image(b"x", "image/png") == "ok"
    assert len(models.generate_calls) == 3


def test_ocr_image_retries_server_errors():
    models = FakeModels(text="ok", fail_first=1, fail_error=FakeAPIError(503, "busy"))
    assert make(models).ocr_image(b"x", "image/png") == "ok"


def test_ocr_image_gives_up_after_three_attempts():
    models = FakeModels(error=FakeAPIError(429, "quota"))
    with pytest.raises(LLMError, match="rate limit"):
        make(models).ocr_image(b"x", "image/png")
    assert len(models.generate_calls) == 3


def test_ocr_image_does_not_retry_a_bad_key():
    models = FakeModels(error=FakeAPIError(403, "denied"))
    with pytest.raises(LLMError, match="API key"):
        make(models).ocr_image(b"x", "image/png")
    assert len(models.generate_calls) == 1


def test_ocr_image_empty_response_is_empty_string():
    assert make(FakeModels(text="")).ocr_image(b"x", "image/png") == ""


# --- quota (429) falls back to the next model ---


class PerModelModels(FakeModels):
    """Raises the given error only for the listed models, like per-model quotas."""

    def __init__(self, chunks, failing, error):
        super().__init__(chunks=chunks)
        self.failing = failing
        self.quota_error = error

    def generate_content_stream(self, model, contents, config):
        self.stream_calls.append({"model": model, "contents": contents, "config": config})
        if model in self.failing:
            raise self.quota_error
        yield from self.chunks


def test_quota_error_skips_retries_and_uses_fallback_model():
    models = PerModelModels([Chunk("ok")], {"main-model"}, FakeAPIError(429, "quota"))
    llm = GeminiLLM("key", "main-model", "fast-model", "backup-model", client=FakeClient(models), sleep=lambda s: pytest.fail("must not wait"))
    assert list(llm.stream("sys", [{"role": "user", "content": "hi"}])) == ["ok"]
    assert [c["model"] for c in models.stream_calls] == ["main-model", "backup-model"]


def test_quota_on_every_model_gives_rate_limit_message():
    models = PerModelModels([Chunk("ok")], {"main-model", "backup-model"}, FakeAPIError(429, "quota"))
    llm = GeminiLLM("key", "main-model", "fast-model", "backup-model", client=FakeClient(models), sleep=lambda s: None)
    with pytest.raises(LLMError, match="rate limit"):
        list(llm.stream("sys", [{"role": "user", "content": "hi"}]))
    assert len(models.stream_calls) == 2


def test_quota_without_a_fallback_model_fails_after_one_call():
    models = PerModelModels([Chunk("ok")], {"main-model"}, FakeAPIError(429, "quota"))
    llm = GeminiLLM("key", "main-model", "fast-model", "", client=FakeClient(models), sleep=lambda s: None)
    with pytest.raises(LLMError, match="rate limit"):
        list(llm.stream("sys", [{"role": "user", "content": "hi"}]))
    assert len(models.stream_calls) == 1



# --- complete_fast retries (summary map step) ---


def test_complete_fast_does_not_retry_by_default():
    models = FakeModels(text="ok", fail_first=1, fail_error=FakeAPIError(503, "busy"))
    with pytest.raises(LLMError, match="busy"):
        make(models).complete_fast("sys", "prompt")
    assert len(models.generate_calls) == 1


def test_complete_fast_retries_when_asked():
    models = FakeModels(text="  ok  ", fail_first=2, fail_error=FakeAPIError(429, "quota"))
    assert make(models).complete_fast("sys", "prompt", retries=3) == "ok"
    assert len(models.generate_calls) == 3


def test_complete_fast_gives_up_after_the_retries():
    models = FakeModels(error=FakeAPIError(503, "busy"))
    with pytest.raises(LLMError, match="busy"):
        make(models).complete_fast("sys", "prompt", retries=3)
    assert len(models.generate_calls) == 3