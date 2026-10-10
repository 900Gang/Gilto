import pytest

from rag.db import Database


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def db(tmp_path, clock):
    database = Database(str(tmp_path / "gilto.db"), clock=clock)
    database.upsert_user("alice@x.com")
    database.upsert_user("bob@x.com")
    return database


def test_consent_is_off_until_given(db):
    assert not db.has_consent("alice@x.com")
    db.give_consent("alice@x.com")
    assert db.has_consent("alice@x.com")
    assert not db.has_consent("bob@x.com")
    assert not db.has_consent("nobody@x.com")


def test_upsert_user_updates_last_seen(db, clock):
    clock.t += 500
    db.upsert_user("alice@x.com")
    assert db.inactive_users(clock.t - 100) == ["bob@x.com"]


def test_messages_round_trip_with_sources(db):
    conv = db.create_conversation("alice@x.com", "Refunds")
    sources = [{"text": "t", "metadata": {"filename": "a.pdf", "page": 1}, "score": 0.9}]
    db.add_message("alice@x.com", conv, "user", "How long?")
    db.add_message("alice@x.com", conv, "assistant", "14 days [a.pdf, p.1]", sources)
    messages = db.get_messages("alice@x.com", conv)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["sources"] == []
    assert messages[1]["sources"] == sources


def test_conversations_are_listed_newest_first_and_per_user(db, clock):
    first = db.create_conversation("alice@x.com", "first")
    clock.t += 10
    second = db.create_conversation("alice@x.com", "second")
    db.create_conversation("bob@x.com", "bobs")
    assert [c.id for c in db.list_conversations("alice@x.com")] == [second, first]
    clock.t += 10
    db.add_message("alice@x.com", first, "user", "bump")
    assert [c.id for c in db.list_conversations("alice@x.com")] == [first, second]


def test_user_cannot_touch_another_users_conversation(db):
    conv = db.create_conversation("alice@x.com", "private")
    db.add_message("alice@x.com", conv, "user", "secret")
    assert db.get_messages("bob@x.com", conv) == []
    assert not db.conversation_exists("bob@x.com", conv)
    with pytest.raises(LookupError):
        db.add_message("bob@x.com", conv, "user", "intruder")
    assert db.delete_conversation("bob@x.com", conv) is False
    assert len(db.get_messages("alice@x.com", conv)) == 1


def test_delete_conversation_removes_its_messages(db):
    conv = db.create_conversation("alice@x.com", "x")
    db.add_message("alice@x.com", conv, "user", "hi")
    assert db.delete_conversation("alice@x.com", conv) is True
    assert db.get_messages("alice@x.com", conv) == []
    assert db.list_conversations("alice@x.com") == []


def test_usage_windows(db, clock):
    db.record_usage("alice@x.com", 100)
    clock.t += 3000
    db.record_usage("alice@x.com", 50)
    db.record_usage("bob@x.com", 7)
    assert db.usage_since("alice@x.com", clock.t - 4000) == [(1_000_000.0, 100), (1_003_000.0, 50)]
    assert db.usage_since("alice@x.com", clock.t - 100) == [(1_003_000.0, 50)]
    assert db.app_tokens_since(0) == 157


def test_delete_user_removes_data_but_keeps_anonymous_usage(db):
    conv = db.create_conversation("alice@x.com", "x")
    db.add_message("alice@x.com", conv, "user", "hi")
    db.give_consent("alice@x.com")
    db.record_usage("alice@x.com", 100)
    db.delete_user("alice@x.com")
    assert db.list_conversations("alice@x.com") == []
    assert db.get_messages("alice@x.com", conv) == []
    assert not db.has_consent("alice@x.com")
    assert db.usage_since("alice@x.com", 0) == []
    assert db.app_tokens_since(0) == 100
    assert db.list_conversations("bob@x.com") == []