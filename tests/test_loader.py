import pytest

from rag.errors import LoaderError
from rag.loader import count_pages, load_pdf


def test_text_pdf_loads_all_pages(pdf_factory):
    data = pdf_factory(
        [
            "The warranty period is twelve months from the date of purchase.",
            "Returns are accepted within thirty days if the item is unused.",
        ]
    )
    result = load_pdf(data)
    assert result.page_count == 2
    assert [p.number for p in result.pages] == [1, 2]
    assert "warranty" in result.pages[0].text
    assert result.skipped_pages == []


def test_page_without_text_is_skipped(pdf_factory):
    data = pdf_factory(["This page has plenty of real text on it, enough to count.", ""])
    result = load_pdf(data)
    assert [p.number for p in result.pages] == [1]
    assert result.skipped_pages == [2]
    assert result.page_count == 2


def test_pdf_with_no_text_at_all_raises(pdf_factory):
    data = pdf_factory(["", ""])
    with pytest.raises(LoaderError, match="scan"):
        load_pdf(data)


def test_empty_bytes_raises():
    with pytest.raises(LoaderError, match="empty"):
        load_pdf(b"")


def test_garbage_bytes_raises():
    with pytest.raises(LoaderError, match="could not be read"):
        load_pdf(b"this is not a pdf at all")


def test_count_pages(pdf_factory):
    assert count_pages(pdf_factory(["a" * 60, "b" * 60, "c" * 60])) == 3
