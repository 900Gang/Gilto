"""Turn an uploaded file into stored, searchable chunks.

`Indexer` is the interface the UI depends on. The blocking implementation reads
text-layer pages directly and sends scanned pages and images through OCR.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from .chunker import chunk_pages
from .config import Settings
from .embedder import Embedder
from .errors import LimitError, LoaderError
from .loader import MIME_TYPES, LoadedPDF, Page, file_kind, load_image, load_pdf, render_page
from .ocr import OCRCache, OCRPipeline
from .store import VectorStore

ProgressCallback = Callable[[int, int, str], None]  # done, total, message


@dataclass
class IndexResult:
    doc_id: str
    filename: str
    chunks: int = 0
    pages_indexed: int = 0
    ocr_pages: int = 0  # pages read with Tesseract
    vision_pages: int = 0  # pages read with Gemini vision
    low_confidence_pages: list[int] = field(default_factory=list)
    skipped_pages: list[int] = field(default_factory=list)  # pages with no readable text
    duplicate: bool = False


class Indexer(ABC):
    @abstractmethod
    def index(
        self,
        user_id: str,
        filename: str,
        data: bytes,
        progress: ProgressCallback | None = None,
    ) -> IndexResult: ...

    @abstractmethod
    def forget(self, user_id: str, doc_id: str) -> None:
        """Remove a document and everything derived from it."""


class BlockingIndexer(Indexer):
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        settings: Settings,
        ocr: OCRPipeline | None = None,
        cache: OCRCache | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings
        self._ocr = ocr
        self._cache = cache

    def _check_limits(self, user_id: str, page_count: int) -> None:
        s = self._settings
        if page_count > s.max_pages_per_file:
            raise LimitError(f"This file has {page_count} pages. The limit is {s.max_pages_per_file} pages per file.")
        docs = self._store.list_docs(user_id)
        if len(docs) >= s.max_docs_per_user:
            raise LimitError(f"You already have {len(docs)} documents. Delete one to add another.")
        used = sum(d.page_count for d in docs)
        if used + page_count > s.max_total_pages:
            raise LimitError(
                f"Adding {page_count} pages would pass your limit of {s.max_total_pages} pages in total "
                f"({used} used)."
            )

    def _read_scanned(self, doc_id: str, kind: str, data: bytes, number: int, result: IndexResult) -> Page | None:
        """OCR one page (or take it from the cache). None means nothing readable."""
        if self._ocr is None:
            result.skipped_pages.append(number)
            return None

        cached = self._cache.get(doc_id, number) if self._cache else None
        if cached is None:
            if kind == "pdf":
                image, mime = render_page(data, number, self._settings.ocr_dpi), "image/png"
            else:
                image, mime = data, MIME_TYPES[kind]
            cached = self._ocr.process(image, mime)  # may raise LLMError
            if cached is None:
                result.skipped_pages.append(number)
                return None
            if self._cache:
                self._cache.put(doc_id, number, cached)

        if cached.source_type == "vision":
            result.vision_pages += 1
        else:
            result.ocr_pages += 1
        if cached.low_confidence:
            result.low_confidence_pages.append(number)
        return Page(number=number, text=cached.text, source_type=cached.source_type)

    def index(
        self,
        user_id: str,
        filename: str,
        data: bytes,
        progress: ProgressCallback | None = None,
    ) -> IndexResult:
        s = self._settings
        if len(data) > s.max_file_mb * 1024 * 1024:
            raise LimitError(f"This file is larger than {s.max_file_mb} MB.")
        kind = file_kind(data)

        doc_id = hashlib.sha256(data).hexdigest()[:32]
        if self._store.has_doc(user_id, doc_id):
            return IndexResult(doc_id=doc_id, filename=filename, duplicate=True)

        loaded: LoadedPDF = load_pdf(data, min_chars=s.min_page_chars) if kind == "pdf" else load_image(data)
        self._check_limits(user_id, loaded.page_count)

        text_pages = {p.number: p for p in loaded.pages}
        order = sorted([*text_pages, *loaded.ocr_pages])
        result = IndexResult(doc_id=doc_id, filename=filename)
        next_index = 0

        try:
            for done, number in enumerate(order, 1):
                page = text_pages.get(number) or self._read_scanned(doc_id, kind, data, number, result)
                if page is not None:
                    chunks = chunk_pages(
                        [page], doc_id, filename, s.chunk_size, s.chunk_overlap, start_index=next_index
                    )
                    for chunk in chunks:
                        chunk.metadata["page_count"] = loaded.page_count
                    if chunks:
                        embeddings = self._embedder.embed_passages([c.text for c in chunks])
                        self._store.add_chunks(user_id, chunks, embeddings)
                        next_index += len(chunks)
                        result.pages_indexed += 1
                if progress:
                    progress(done, len(order), f"Read page {number} of {loaded.page_count}")

            if next_index == 0:
                if loaded.ocr_pages and self._ocr is None:
                    raise LoaderError(
                        "No text could be extracted. This looks like a scan, and OCR is not set up. "
                        "Install Tesseract or set GEMINI_API_KEY."
                    )
                raise LoaderError("No readable text was found in this file.")
        except Exception:
            self._store.delete_doc(user_id, doc_id)  # never leave a half-indexed document behind
            raise

        result.chunks = next_index
        return result

    def forget(self, user_id: str, doc_id: str) -> None:
        self._store.delete_doc(user_id, doc_id)
        if self._cache:
            self._cache.delete(doc_id)
