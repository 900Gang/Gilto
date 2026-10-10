import dataclasses

import chromadb
import pytest

from rag.config import Settings
from rag.errors import LimitError, LoaderError
from rag.indexer import BlockingIndexer
from rag.store import VectorStore

PAGE_A = "The warranty period is twelve months from the date of purchase. " * 3
PAGE_B = "Returns are accepted within thirty days if the item is unused. " * 3


class FakeEmbedder:
    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def embed_passages(self, texts):
        self.calls += 1
        if self.fail_on_call and self.calls == self.fail_on_call:
            raise RuntimeError("embedding failed")
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.fixture
def store(tmp_path):
    return VectorStore(client=chromadb.PersistentClient(path=str(tmp_path / "chroma")))


def make_indexer(store, embedder=None, **overrides):
    settings = dataclasses.replace(Settings.from_env(), **overrides)
    return BlockingIndexer(store, embedder or FakeEmbedder(), settings)


def test_index_stores_chunks_and_lists_document(store, pdf_factory):
    result = make_indexer(store).index("u", "policy.pdf", pdf_factory([PAGE_A, PAGE_B]))
    assert not result.duplicate
    assert result.pages_indexed == 2
    assert result.chunks >= 2
    docs = store.list_docs("u")
    assert len(docs) == 1
    assert docs[0].filename == "policy.pdf"
    assert docs[0].page_count == 2


def test_progress_reported_once_per_page(store, pdf_factory):
    seen = []
    make_indexer(store).index("u", "p.pdf", pdf_factory([PAGE_A, PAGE_B]), progress=lambda d, t, m: seen.append((d, t)))
    assert seen == [(1, 2), (2, 2)]


def test_same_content_is_not_indexed_twice(store, pdf_factory):
    data = pdf_factory([PAGE_A])
    indexer = make_indexer(store)
    indexer.index("u", "a.pdf", data)
    again = indexer.index("u", "renamed.pdf", data)
    assert again.duplicate
    assert len(store.list_docs("u")) == 1


def test_same_file_for_two_users_is_indexed_for_both(store, pdf_factory):
    data = pdf_factory([PAGE_A])
    indexer = make_indexer(store)
    indexer.index("alice", "a.pdf", data)
    assert not indexer.index("bob", "a.pdf", data).duplicate


def test_skipped_pages_are_reported(store, pdf_factory):
    result = make_indexer(store).index("u", "a.pdf", pdf_factory([PAGE_A, ""]))
    assert result.skipped_pages == [2]
    assert result.pages_indexed == 1


def test_scanned_pdf_is_rejected(store, pdf_factory):
    with pytest.raises(LoaderError):
        make_indexer(store).index("u", "scan.pdf", pdf_factory(["", ""]))
    assert store.list_docs("u") == []


def test_oversize_file_rejected(store, pdf_factory):
    data = pdf_factory([PAGE_A])
    with pytest.raises(LimitError, match="larger than"):
        make_indexer(store, max_file_mb=0).index("u", "a.pdf", data)


def test_too_many_pages_rejected(store, pdf_factory):
    with pytest.raises(LimitError, match="pages per file"):
        make_indexer(store, max_pages_per_file=1).index("u", "a.pdf", pdf_factory([PAGE_A, PAGE_B]))


def test_document_count_limit(store, pdf_factory):
    indexer = make_indexer(store, max_docs_per_user=1)
    indexer.index("u", "a.pdf", pdf_factory([PAGE_A]))
    with pytest.raises(LimitError, match="Delete one"):
        indexer.index("u", "b.pdf", pdf_factory([PAGE_B]))


def test_total_page_limit(store, pdf_factory):
    indexer = make_indexer(store, max_total_pages=2)
    indexer.index("u", "a.pdf", pdf_factory([PAGE_A, PAGE_B]))
    with pytest.raises(LimitError, match="in total"):
        indexer.index("u", "b.pdf", pdf_factory([PAGE_A]))


def test_failure_midway_leaves_no_partial_document(store, pdf_factory):
    indexer = make_indexer(store, embedder=FakeEmbedder(fail_on_call=2))
    data = pdf_factory([PAGE_A, PAGE_B])
    with pytest.raises(RuntimeError):
        indexer.index("u", "a.pdf", data)
    assert store.list_docs("u") == []
    # a retry must work, not be treated as a duplicate
    retry = make_indexer(store).index("u", "a.pdf", data)
    assert not retry.duplicate
