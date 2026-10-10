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



def test_delete_user_drops_only_that_users_data(store):
    store.add_chunks("alice", [make_chunk("d1", 0, "one")], [[1.0, 0.0]])
    store.add_chunks("bob", [make_chunk("d2", 0, "two")], [[1.0, 0.0]])
    store.delete_user("alice")
    assert store.list_docs("alice") == []
    assert [d.doc_id for d in store.list_docs("bob")] == ["d2"]


def test_delete_user_with_no_data_is_harmless(store):
    store.delete_user("nobody")


def test_retrieved_round_trips_through_a_dict():
    from rag.store import Retrieved

    hit = Retrieved(text="t", metadata={"filename": "a.pdf", "page": 2}, score=0.9)
    assert Retrieved.from_dict(hit.to_dict()) == hit
    


def test_all_chunks_returns_everything_in_reading_order_without_scores(store):
    store.add_chunks(
        "alice",
        [make_chunk("d2", 1, "b second", filename="b.pdf"), make_chunk("d1", 1, "a second"), make_chunk("d1", 0, "a first")],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    )
    chunks = store.all_chunks("alice")
    assert [c.text for c in chunks] == ["a first", "a second", "b second"]
    assert all(c.score is None for c in chunks)


def test_all_chunks_can_be_limited_to_documents_and_never_crosses_users(store):
    store.add_chunks("alice", [make_chunk("d1", 0, "one"), make_chunk("d2", 0, "two", filename="b.pdf")], [[1.0, 0.0], [0.0, 1.0]])
    store.add_chunks("bob", [make_chunk("d3", 0, "bobs")], [[1.0, 0.0]])
    assert [c.text for c in store.all_chunks("alice", ["d2"])] == ["two"]
    assert [c.text for c in store.all_chunks("bob")] == ["bobs"]


def test_all_chunks_of_an_empty_library_is_empty(store):
    assert store.all_chunks("nobody") == []