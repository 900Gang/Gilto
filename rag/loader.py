"""Documents -> pages of text.

PDF pages with a text layer are read directly. Pages without one, and every
image file, are marked for OCR (see ocr.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

from .errors import LoaderError

MIME_TYPES = {"pdf": "application/pdf", "png": "image/png", "jpeg": "image/jpeg"}


@dataclass(frozen=True)
class Page:
    number: int  # 1-based
    text: str
    source_type: str = "text"  # text | ocr | vision


@dataclass
class LoadedPDF:
    pages: list[Page]  # pages that have a usable text layer
    page_count: int
    ocr_pages: list[int] = field(default_factory=list)  # pages that need OCR


def file_kind(data: bytes) -> str:
    """Identify the file by its first bytes, not its name: 'pdf', 'png' or 'jpeg'."""
    if not data:
        raise LoaderError("This file is empty.")
    if b"%PDF-" in data[:1024]:
        return "pdf"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    raise LoaderError("Unsupported file. Upload a PDF, PNG or JPG.")


def _open(data: bytes) -> "pymupdf.Document":
    if not data:
        raise LoaderError("This file is empty.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several error types for bad input
        raise LoaderError("This file could not be read as a PDF.") from exc
    if doc.needs_pass:
        doc.close()
        raise LoaderError("This PDF is password-protected. Remove the password and try again.")
    return doc


def count_pages(data: bytes) -> int:
    doc = _open(data)
    try:
        return doc.page_count
    finally:
        doc.close()


def load_pdf(data: bytes, min_chars: int = 50) -> LoadedPDF:
    """Extract the text layer per page. Pages with fewer than `min_chars`
    characters are reported in `ocr_pages` (most likely scans)."""
    doc = _open(data)
    try:
        pages: list[Page] = []
        ocr_pages: list[int] = []
        for index in range(doc.page_count):
            text = doc.load_page(index).get_text("text").strip()
            if len(text) < min_chars:
                ocr_pages.append(index + 1)
            else:
                pages.append(Page(number=index + 1, text=text))
        page_count = doc.page_count
    finally:
        doc.close()
    return LoadedPDF(pages=pages, page_count=page_count, ocr_pages=ocr_pages)


def load_image(data: bytes) -> LoadedPDF:
    """An image is one page with no text layer, so it always goes to OCR."""
    try:
        pymupdf.Pixmap(data)  # raises if the bytes are not a decodable image
    except Exception as exc:
        raise LoaderError("This image could not be read.") from exc
    return LoadedPDF(pages=[], page_count=1, ocr_pages=[1])


def render_page(data: bytes, page_number: int, dpi: int = 200) -> bytes:
    """Render one PDF page to PNG bytes for OCR."""
    doc = _open(data)
    try:
        return doc.load_page(page_number - 1).get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()
