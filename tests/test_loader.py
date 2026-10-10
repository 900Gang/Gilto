import pytest

from rag.errors import LoaderError
from rag.loader import count_pages, file_kind, load_image, load_pdf, render_page

PAGE_A = "The warranty period is twelve months from the date of purchase."
PAGE_B = "Returns are accepted within thirty days if the item is unused."


def test_text_pdf_loads_all_pages(pdf_factory):
    result = load_pdf(pdf_factory([PAGE_A, PAGE_B]))
    assert result.page_count == 2
    assert [p.number for p in result.pages] == [1, 2]
    assert "warranty" in result.pages[0].text
    assert result.ocr_pages == []
    assert result.pages[0].source_type == "text"


def test_page_without_text_is_marked_for_ocr(pdf_factory):
    result = load_pdf(pdf_factory([PAGE_A, ""]))
    assert [p.number for p in result.pages] == [1]
    assert result.ocr_pages == [2]
    assert result.page_count == 2


def test_scanned_pdf_has_every_page_marked_for_ocr(scan_pdf_factory):
    result = load_pdf(scan_pdf_factory([PAGE_A, PAGE_B]))
    assert result.pages == []
    assert result.ocr_pages == [1, 2]


def test_empty_bytes_raises():
    with pytest.raises(LoaderError, match="empty"):
        load_pdf(b"")


def test_garbage_bytes_raises():
    with pytest.raises(LoaderError, match="could not be read"):
        load_pdf(b"this is not a pdf at all")


def test_count_pages(pdf_factory):
    assert count_pages(pdf_factory([PAGE_A, PAGE_B, PAGE_A])) == 3


def test_file_kind_uses_content_not_name(pdf_factory, png_factory):
    assert file_kind(pdf_factory([PAGE_A])) == "pdf"
    assert file_kind(png_factory(PAGE_A)) == "png"
    assert file_kind(b"\xff\xd8\xff\xe0" + b"0" * 20) == "jpeg"


def test_file_kind_rejects_other_files():
    with pytest.raises(LoaderError, match="Unsupported"):
        file_kind(b"PK\x03\x04 a zip or docx file")
    with pytest.raises(LoaderError, match="empty"):
        file_kind(b"")


def test_load_image_is_one_page_for_ocr(png_factory):
    result = load_image(png_factory(PAGE_A))
    assert result.page_count == 1
    assert result.pages == []
    assert result.ocr_pages == [1]


def test_load_image_rejects_corrupt_image():
    with pytest.raises(LoaderError, match="could not be read"):
        load_image(b"\x89PNG\r\n\x1a\n not really a png")


def test_render_page_returns_png(scan_pdf_factory):
    png = render_page(scan_pdf_factory([PAGE_A]), 1, dpi=72)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
