"""Turn an uploaded file into stored, searchable chunks.

`Indexer` is the interface the UI depends on. Slice 1 ships a blocking
implementation; a background-worker version can replace it later.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import groupby

from .chunker import chunk_pages
from .config import Settings
from .embedder import Embedder
from .errors import LimitError
from .loader import load_pdf
from .store import VectorStore

ProgressCallback = Callable[[int, int, str], None]  # done, total, message


@dataclass
class IndexResult:
    doc_id: str
    filename: str
    chunks: int = 0
    pages_indexed: int = 0
    skipped_pages: list[int] = field(default_factory=list)
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


class BlockingIndexer(Indexer):
    def __init__(self, store: VectorStore, embedder: Embedder, settings: Settings) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings

    def _check_limits(self, user_id: str, data: bytes, page_count: int) -> None:
        s = self._settings
        if len(data) > s.max_file_mb * 1024 * 1024:
            raise LimitError(f"This file is larger than {s.max_file_mb} MB.")
        if page_count > s.max_pages_per_file:
            raise LimitError(f"This PDF has {page_count} pages. The limit is {s.max_pages_per_file} pages per file.")
        docs = self._store.list_docs(user_id)
        if len(docs) >= s.max_docs_per_user:
            raise LimitError(f"You already have {len(docs)} documents. Delete one to add another.")
        used = sum(d.page_count for d in docs)
        if used + page_count > s.max_total_pages:
            raise LimitError(
                f"Adding {page_count} pages would pass your limit of {s.max_total_pages} pages in total "
                f"({used} used)."
            )

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

        doc_id = hashlib.sha256(data).hexdigest()[:32]
        if self._store.has_doc(user_id, doc_id):
            return IndexResult(doc_id=doc_id, filename=filename, duplicate=True)

        loaded = load_pdf(data, min_chars=s.min_page_chars)
        self._check_limits(user_id, data, loaded.page_count)

        chunks = chunk_pages(loaded.pages, doc_id, filename, size=s.chunk_size, overlap=s.chunk_overlap)
        for chunk in chunks:
            chunk.metadata["page_count"] = loaded.page_count

        total = len(loaded.pages)
        try:
            for done, (page_number, group) in enumerate(groupby(chunks, key=lambda c: c.metadata["page"]), 1):
                page_chunks = list(group)
                embeddings = self._embedder.embed_passages([c.text for c in page_chunks])
                self._store.add_chunks(user_id, page_chunks, embeddings)
                if progress:
                    progress(done, total, f"Indexed page {page_number} of {loaded.page_count}")
        except Exception:
            self._store.delete_doc(user_id, doc_id)  # never leave a half-indexed document behind
            raise

        return IndexResult(
            doc_id=doc_id,
            filename=filename,
            chunks=len(chunks),
            pages_indexed=total,
            skipped_pages=loaded.skipped_pages,
        )
