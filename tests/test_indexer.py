import dataclasses

import chromadb
import pytest

from rag.config import Settings
from rag.errors import LimitError, LLMError, LoaderError
from rag.indexer import BlockingIndexer
from rag.ocr import OCRCache, OCRPage
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


class FakeOCR:
    """Stands in for OCRPipeline. Returns one canned page for every image."""

    def __init__(self, page="default", error=None):
        if page == "default":
            page = OCRPage("Scanned text about the warranty period of twelve months.", "vision")
        self.page = page
        self.error = error
        self.calls = []

    def process(self, image, mime):
        self.calls.append(mime)
        if self.error:
            raise self.error
        return self.page


def make_indexer(store, embedder=None, ocr=None, cache=None, **overrides):
    settings = dataclasses.replace(Settings.from_env(), **overrides)
    return BlockingIndexer(store, embedder or FakeEmbedder(), settings, ocr=ocr, cache=cache)


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


# --- scanned pages and images (slice 2) ---


def test_scanned_pdf_is_read_with_ocr(store, scan_pdf_factory):
    ocr = FakeOCR()
    result = make_indexer(store, ocr=ocr).index("u", "scan.pdf", scan_pdf_factory([PAGE_A, PAGE_B]))
    assert result.vision_pages == 2
    assert result.ocr_pages == 0
    assert result.pages_indexed == 2
    assert ocr.calls == ["image/png", "image/png"]
    hits = store.query("u", [1.0, 0.0], k=5)
    assert {h.metadata["source_type"] for h in hits} == {"vision"}


def test_image_file_is_indexed_with_its_own_mime_type(store, png_factory):
    ocr = FakeOCR(OCRPage("Text read from a screenshot of the policy page.", "ocr", 90.0))
    result = make_indexer(store, ocr=ocr).index("u", "shot.png", png_factory(PAGE_A))
    assert result.ocr_pages == 1
    assert ocr.calls == ["image/png"]
    assert store.list_docs("u")[0].page_count == 1


def test_mixed_pdf_keeps_page_order_and_source_types(store, pdf_factory):
    ocr = FakeOCR(OCRPage("Scanned page two text, long enough to be stored as a chunk.", "ocr", 88.0))
    result = make_indexer(store, ocr=ocr).index("u", "mixed.pdf", pdf_factory([PAGE_A, "", PAGE_B]))
    assert result.pages_indexed == 3
    hits = store.query("u", [1.0, 0.0], k=10)
    by_page = {h.metadata["page"]: h.metadata["source_type"] for h in hits}
    assert by_page == {1: "text", 2: "ocr", 3: "text"}
    assert sorted(h.metadata["chunk_index"] for h in hits) == [0, 1, 2]


def test_ocr_result_is_cached_so_a_retry_does_not_ocr_again(store, scan_pdf_factory, tmp_path):
    cache = OCRCache(str(tmp_path / "ocr"))
    data = scan_pdf_factory([PAGE_A])
    first = FakeOCR()
    failing = make_indexer(store, embedder=FakeEmbedder(fail_on_call=1), ocr=first, cache=cache)
    with pytest.raises(RuntimeError):
        failing.index("u", "s.pdf", data)
    assert len(first.calls) == 1
    second = FakeOCR(error=AssertionError("must come from the cache"))
    result = make_indexer(store, ocr=second, cache=cache).index("u", "s.pdf", data)
    assert second.calls == []
    assert result.pages_indexed == 1


def test_low_confidence_pages_are_reported(store, png_factory):
    ocr = FakeOCR(OCRPage("Blurry text that tesseract was not sure about at all.", "ocr", 35.0, low_confidence=True))
    result = make_indexer(store, ocr=ocr).index("u", "blurry.png", png_factory(PAGE_A))
    assert result.low_confidence_pages == [1]


def test_unreadable_scanned_pages_are_skipped_not_fatal(store, pdf_factory):
    result = make_indexer(store, ocr=FakeOCR(page=None)).index("u", "a.pdf", pdf_factory([PAGE_A, ""]))
    assert result.skipped_pages == [2]
    assert result.pages_indexed == 1


def test_image_with_no_readable_text_is_rejected(store, png_factory):
    with pytest.raises(LoaderError, match="No readable text"):
        make_indexer(store, ocr=FakeOCR(page=None)).index("u", "blank.png", png_factory(PAGE_A))
    assert store.list_docs("u") == []


def test_scan_without_any_ocr_explains_what_is_missing(store, scan_pdf_factory):
    with pytest.raises(LoaderError, match="OCR is not set up"):
        make_indexer(store).index("u", "scan.pdf", scan_pdf_factory([PAGE_A]))


def test_ocr_failure_leaves_no_partial_document(store, scan_pdf_factory):
    ocr = FakeOCR(error=LLMError("Gemini is busy"))
    with pytest.raises(LLMError):
        make_indexer(store, ocr=ocr).index("u", "scan.pdf", scan_pdf_factory([PAGE_A]))
    assert store.list_docs("u") == []


def test_unsupported_file_type_is_rejected(store):
    with pytest.raises(LoaderError, match="Unsupported"):
        make_indexer(store).index("u", "notes.docx", b"PK\x03\x04 not a pdf")


def test_forget_removes_chunks_and_ocr_cache(store, scan_pdf_factory, tmp_path):
    cache = OCRCache(str(tmp_path / "ocr"))
    indexer = make_indexer(store, ocr=FakeOCR(), cache=cache)
    result = indexer.index("u", "scan.pdf", scan_pdf_factory([PAGE_A]))
    assert cache.get(result.doc_id, 1) is not None
    indexer.forget("u", result.doc_id)
    assert store.list_docs("u") == []
    assert cache.get(result.doc_id, 1) is None
