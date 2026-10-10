import fitz
import pytest


def make_pdf(page_texts: list[str]) -> bytes:
    """Build a small PDF in memory. An empty string gives a page with no text."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_textbox(fitz.Rect(50, 50, 550, 800), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def pdf_factory():
    return make_pdf
