import dataclasses

import pytest
from google.genai import errors

from rag.config import Settings
from rag.embedder import E5Embedder, GeminiEmbedder, embedding_tag, make_embedder
from rag.errors import ConfigError, LLMError


class FakeAPIError(errors.APIError):
    """Builds an SDK error without depending on its constructor signature."""

    def __init__(self, code, message):
        Exception.__init__(self, message)
        self.code = code
        self.message = message


class Item:
    def __init__(self, values):
        self.values = values


class Response:
    def __init__(self, embeddings):
        self.embeddings = embeddings


class FakeModels:
    def __init__(self, fail_first=0, fail_code=503, error=None, short=False):
        self.calls = []
        self.fail_first = fail_first
        self.fail_code = fail_code
        self.error = error
        self.short = short

    def embed_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": list(contents), "config": config})
        if self.error:
            raise self.error
        if len(self.calls) <= self.fail_first:
            raise FakeAPIError(self.fail_code, "try later")
        count = len(contents) - 1 if self.short else len(contents)
        return Response([Item([3.0, 4.0, 0.0, 0.0]) for _ in range(count)])


class FakeClient:
    def __init__(self, models):
        self.models = models


def make(models):
    return GeminiEmbedder("key", dimensions=4, client=FakeClient(models), sleep=lambda s: None)


def test_passages_use_the_document_task_and_requested_size():
    models = FakeModels()
    make(models).embed_passages(["a", "b"])
    call = models.calls[0]
    assert call["model"] == "gemini-embedding-001"
    assert call["contents"] == ["a", "b"]
    assert call["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert call["config"].output_dimensionality == 4


def test_query_uses_the_query_task():
    models = FakeModels()
    make(models).embed_query("question")
    assert models.calls[0]["config"].task_type == "RETRIEVAL_QUERY"


def test_vectors_are_normalized_to_unit_length():
    vector = make(FakeModels()).embed_query("q")
    assert vector == pytest.approx([0.6, 0.8, 0.0, 0.0])


def test_a_zero_vector_is_left_alone():
    class ZeroModels(FakeModels):
        def embed_content(self, model, contents, config):
            return Response([Item([0.0, 0.0]) for _ in contents])

    assert make(ZeroModels()).embed_query("q") == [0.0, 0.0]


def test_large_inputs_are_sent_in_batches():
    models = FakeModels()
    vectors = make(models).embed_passages([f"t{i}" for i in range(120)])
    assert [len(c["contents"]) for c in models.calls] == [50, 50, 20]
    assert len(vectors) == 120


def test_no_texts_makes_no_call():
    models = FakeModels()
    assert make(models).embed_passages([]) == []
    assert models.calls == []


def test_server_errors_are_retried():
    models = FakeModels(fail_first=2, fail_code=503)
    assert len(make(models).embed_passages(["a"])) == 1
    assert len(models.calls) == 3


def test_rate_limits_are_retried():
    models = FakeModels(fail_first=1, fail_code=429)
    assert len(make(models).embed_passages(["a"])) == 1
    assert len(models.calls) == 2


def test_persistent_failure_gives_a_friendly_error():
    models = FakeModels(error=FakeAPIError(503, "busy"))
    with pytest.raises(LLMError, match="busy"):
        make(models).embed_passages(["a"])
    assert len(models.calls) == 4


def test_a_bad_key_is_not_retried():
    models = FakeModels(error=FakeAPIError(403, "denied"))
    with pytest.raises(LLMError, match="API key"):
        make(models).embed_passages(["a"])
    assert len(models.calls) == 1


def test_wrong_number_of_vectors_is_an_error():
    with pytest.raises(LLMError, match="unexpected"):
        make(FakeModels(short=True)).embed_passages(["a", "b"])


def settings_for(provider, **overrides):
    return dataclasses.replace(Settings.from_env(), embedding_provider=provider, **overrides)


def test_tag_names_the_gemini_vector_space_and_is_empty_for_local():
    assert embedding_tag(settings_for("gemini", gemini_embedding_model="m", embedding_dimensions=768)) == "m-768"
    assert embedding_tag(settings_for("local")) == ""


def test_make_embedder_picks_the_provider():
    assert isinstance(make_embedder(settings_for("gemini", api_key="key")), GeminiEmbedder)
    assert isinstance(make_embedder(settings_for("local")), E5Embedder)


def test_gemini_embeddings_need_a_key():
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        make_embedder(settings_for("gemini", api_key=""))


def test_default_threshold_depends_on_the_provider(monkeypatch):
    monkeypatch.delenv("SIMILARITY_THRESHOLD", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    assert Settings.from_env().similarity_threshold == 0.80
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    assert Settings.from_env().similarity_threshold == 0.60
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.7")
    assert Settings.from_env().similarity_threshold == 0.7


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        Settings.from_env()