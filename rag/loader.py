"""PDF -> pages of text. Slice 1 reads the text layer only (no OCR)."""
from __future__ import annotations

from dataclasses import dataclass, field

import fitz  # PyMuPDF

from .errors import LoaderError


@dataclass(frozen=True)
class Page:
    number: int  # 1-based
    text: str


@dataclass
class LoadedPDF:
    pages: list[Page]
    page_count: int
    skipped_pages: list[int] = field(default_factory=list)


def count_pages(data: bytes) -> int:
    return _open(data).page_count


def _open(data: bytes) -> "fitz.Document":
    if not data:
        raise LoaderError("This file is empty.")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several error types for bad input
        raise LoaderError("This file could not be read as a PDF.") from exc
    if doc.needs_pass:
        doc.close()
        raise LoaderError("This PDF is password-protected. Remove the password and try again.")
    return doc


def load_pdf(data: bytes, min_chars: int = 50) -> LoadedPDF:
    """Extract text per page. Pages with fewer than `min_chars` characters are
    reported in `skipped_pages` (most likely scans without a text layer)."""
    doc = _open(data)
    try:
        pages: list[Page] = []
        skipped: list[int] = []
        for index in range(doc.page_count):
            text = doc.load_page(index).get_text("text").strip()
            if len(text) < min_chars:
                skipped.append(index + 1)
            else:
                pages.append(Page(number=index + 1, text=text))
        page_count = doc.page_count
    finally:
        doc.close()

    if not pages:
        raise LoaderError(
            "No text could be extracted. This PDF looks like a scan; "
            "scanned pages need OCR, which is not available yet."
        )
    return LoadedPDF(pages=pages, page_count=page_count, skipped_pages=skipped)
