import pymupdf
import pytest


def make_pdf(page_texts: list[str]) -> bytes:
    """Build a small text-layer PDF in memory. An empty string gives a page with no text."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def make_png(text: str, dpi: int = 150) -> bytes:
    """A PNG picture of a page of text. It has no text layer, like a screenshot or scan."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=14)
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    doc.close()
    return png


def make_scan_pdf(page_texts: list[str]) -> bytes:
    """A PDF whose pages are pictures only, so it has no text layer."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_image(page.rect, stream=make_png(text))
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def pdf_factory():
    return make_pdf


@pytest.fixture
def png_factory():
    return make_png


@pytest.fixture
def scan_pdf_factory():
    return make_scan_pdf
