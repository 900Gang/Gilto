import chromadb
import pytest

from rag.chunker import Chunk
from rag.store import VectorStore


@pytest.fixture
def store(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return VectorStore(client=client)


def make_chunk(doc_id, idx, text, page=1, filename="a.pdf", page_count=3):
    return Chunk(
        text=text,
        metadata={
            "doc_id": doc_id,
            "filename": filename,
            "page": page,
            "chunk_index": idx,
            "source_type": "text",
            "page_count": page_count,
        },
    )


def test_add_and_query_returns_nearest_first(store):
    chunks = [make_chunk("d1", 0, "alpha"), make_chunk("d1", 1, "beta"), make_chunk("d1", 2, "gamma")]
    store.add_chunks("alice", chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    hits = store.query("alice", [0.0, 1.0, 0.0], k=3)
    assert hits[0].text == "beta"
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert hits[0].score > hits[1].score
    assert hits[0].metadata["page"] == 1


def test_query_on_empty_store_returns_nothing(store):
    assert store.query("alice", [1.0, 0.0, 0.0], k=5) == []


def test_users_cannot_see_each_others_chunks(store):
    store.add_chunks("alice", [make_chunk("d1", 0, "alice secret")], [[1.0, 0.0]])
    store.add_chunks("bob", [make_chunk("d2", 0, "bob note")], [[1.0, 0.0]])
    texts = [h.text for h in store.query("bob", [1.0, 0.0], k=5)]
    assert texts == ["bob note"]
    assert store.list_docs("bob")[0].doc_id == "d2"
    assert not store.has_doc("bob", "d1")


def test_delete_doc_removes_only_that_doc(store):
    store.add_chunks(
        "alice",
        [make_chunk("d1", 0, "one"), make_chunk("d2", 0, "two", filename="b.pdf")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    store.delete_doc("alice", "d1")
    assert not store.has_doc("alice", "d1")
    assert store.has_doc("alice", "d2")
    assert [h.text for h in store.query("alice", [1.0, 0.0], k=5)] == ["two"]


def test_has_doc_detects_duplicates(store):
    store.add_chunks("alice", [make_chunk("d1", 0, "one")], [[1.0, 0.0]])
    assert store.has_doc("alice", "d1")
    assert not store.has_doc("alice", "nope")


def test_list_docs_counts_chunks(store):
    store.add_chunks(
        "alice",
        [make_chunk("d1", 0, "one"), make_chunk("d1", 1, "two"), make_chunk("d2", 0, "x", filename="b.pdf")],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )
    docs = {d.doc_id: d for d in store.list_docs("alice")}
    assert docs["d1"].chunks == 2
    assert docs["d1"].filename == "a.pdf"
    assert docs["d1"].page_count == 3
    assert docs["d2"].chunks == 1


def test_query_can_be_limited_to_documents(store):
    store.add_chunks(
        "alice",
        [make_chunk("d1", 0, "one"), make_chunk("d2", 0, "two", filename="b.pdf")],
        [[1.0, 0.0], [1.0, 0.0]],
    )
    hits = store.query("alice", [1.0, 0.0], k=5, doc_ids=["d2"])
    assert [h.text for h in hits] == ["two"]


def test_mismatched_lengths_rejected(store):
    with pytest.raises(ValueError):
        store.add_chunks("alice", [make_chunk("d1", 0, "one")], [])
