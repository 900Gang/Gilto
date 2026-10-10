from rag.chunker import chunk_pages, split_text
from rag.loader import Page


def test_short_text_is_one_chunk():
    assert split_text("hello world", size=800, overlap=150) == ["hello world"]


def test_chunks_respect_size():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = split_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"w{i:04d}" for i in range(400))
    chunks = split_text(text, size=200, overlap=50)
    for prev, nxt in zip(chunks, chunks[1:]):
        tail = prev[-30:]
        assert tail in nxt, "next chunk must repeat the end of the previous one"


def test_no_text_is_lost():
    words = [f"w{i:04d}" for i in range(300)]
    chunks = split_text(" ".join(words), size=150, overlap=30)
    joined = " ".join(chunks)
    assert all(w in joined for w in words)


def test_prefers_paragraph_breaks():
    text = ("a" * 90) + "\n\n" + ("b" * 90)
    chunks = split_text(text, size=100, overlap=10)
    assert chunks[0] == "a" * 90
    assert chunks[1].startswith("b")


def test_chunk_pages_metadata():
    pages = [Page(number=1, text="first page " * 10), Page(number=3, text="third page " * 10)]
    chunks = chunk_pages(pages, doc_id="abc", filename="f.pdf", size=800, overlap=150)
    assert [c.metadata["page"] for c in chunks] == [1, 3]
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1]
    for c in chunks:
        assert c.metadata["filename"] == "f.pdf"
        assert c.metadata["doc_id"] == "abc"
        assert c.metadata["source_type"] == "text"


def test_chunk_index_is_continuous_across_pages():
    long_text = " ".join(f"w{i:04d}" for i in range(400))
    pages = [Page(number=1, text=long_text), Page(number=2, text=long_text)]
    chunks = chunk_pages(pages, doc_id="d", filename="f.pdf", size=200, overlap=40)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_invalid_overlap_rejected():
    import pytest

    with pytest.raises(ValueError):
        split_text("x" * 500, size=100, overlap=100)
