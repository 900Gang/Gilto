import chromadb
import pytest

from rag.chunker import Chunk
from rag.db import Database
from rag.ocr import OCRCache, OCRPage
from rag.retention import DAY, delete_user_data, purge_inactive
from rag.store import VectorStore


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def chunk(doc_id):
    return Chunk(
        text="some text",
        metadata={
            "doc_id": doc_id,
            "filename": f"{doc_id}.pdf",
            "page": 1,
            "chunk_index": 0,
            "source_type": "text",
            "page_count": 1,
        },
    )


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    db = Database(str(tmp_path / "g.db"), clock=clock)
    store = VectorStore(client=chromadb.PersistentClient(path=str(tmp_path / "chroma")))
    cache = OCRCache(str(tmp_path / "ocr"))
    return clock, db, store, cache


def fill(db, store, cache, user, doc_id):
    db.upsert_user(user)
    store.add_chunks(user, [chunk(doc_id)], [[1.0, 0.0]])
    cache.put(doc_id, 1, OCRPage("text", "ocr"))
    conv = db.create_conversation(user, "chat")
    db.add_message(user, conv, "user", "hi")
    db.record_usage(user, 100)


def test_delete_user_data_removes_everything_for_that_user_only(env):
    _, db, store, cache = env
    fill(db, store, cache, "alice@x.com", "d1")
    fill(db, store, cache, "bob@x.com", "d2")
    delete_user_data("alice@x.com", store, cache, db)
    assert store.list_docs("alice@x.com") == []
    assert cache.get("d1", 1) is None
    assert db.list_conversations("alice@x.com") == []
    assert [d.doc_id for d in store.list_docs("bob@x.com")] == ["d2"]
    assert cache.get("d2", 1) is not None
    assert len(db.list_conversations("bob@x.com")) == 1
    assert db.app_tokens_since(0) == 200  # usage stays, anonymized


def test_delete_user_data_for_unknown_user_is_harmless(env):
    _, db, store, cache = env
    delete_user_data("nobody@x.com", store, cache, db)


def test_purge_inactive_removes_only_old_accounts(env):
    clock, db, store, cache = env
    fill(db, store, cache, "old@x.com", "d1")
    clock.t += 100 * DAY
    fill(db, store, cache, "new@x.com", "d2")
    purged = purge_inactive(90, store, cache, db, now=clock.t)
    assert purged == ["old@x.com"]
    assert store.list_docs("old@x.com") == []
    assert [d.doc_id for d in store.list_docs("new@x.com")] == ["d2"]


def test_purge_inactive_is_off_when_days_is_zero(env):
    clock, db, store, cache = env
    fill(db, store, cache, "old@x.com", "d1")
    clock.t += 1000 * DAY
    assert purge_inactive(0, store, cache, db, now=clock.t) == []
    assert len(store.list_docs("old@x.com")) == 1