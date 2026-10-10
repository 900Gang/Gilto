"""SQLite storage for users, chats and usage.

Each call opens a short-lived connection, so Streamlit sessions running in
different threads never share one. Every chat query also checks the owner,
so one user can never read or change another user's conversation.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    last_seen REAL NOT NULL,
    consented_at REAL
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(email) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    ts REAL NOT NULL,
    tokens INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_user ON usage(user_id, ts);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts);
"""


@dataclass
class Conversation:
    id: int
    title: str
    updated_at: float


class Database:
    def __init__(self, path: str, clock: Callable[[], float] = time.time) -> None:
        self._path = path
        self._clock = clock
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(SCHEMA)
            con.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self._path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # --- users ---

    def upsert_user(self, email: str) -> None:
        now = self._clock()
        with self._conn() as con:
            con.execute(
                "INSERT INTO users(email, created_at, last_seen) VALUES (?, ?, ?) "
                "ON CONFLICT(email) DO UPDATE SET last_seen = excluded.last_seen",
                (email, now, now),
            )

    def has_consent(self, email: str) -> bool:
        with self._conn() as con:
            row = con.execute("SELECT consented_at FROM users WHERE email = ?", (email,)).fetchone()
        return row is not None and row["consented_at"] is not None

    def give_consent(self, email: str) -> None:
        with self._conn() as con:
            con.execute(
                "UPDATE users SET consented_at = ? WHERE email = ? AND consented_at IS NULL",
                (self._clock(), email),
            )

    def inactive_users(self, cutoff: float) -> list[str]:
        with self._conn() as con:
            rows = con.execute("SELECT email FROM users WHERE last_seen < ?", (cutoff,)).fetchall()
        return [row["email"] for row in rows]

    def delete_user(self, email: str) -> None:
        """Remove the user, their chats and messages. Usage rows stay, anonymized,
        so the app-wide monthly count does not drop when someone leaves."""
        with self._conn() as con:
            con.execute("DELETE FROM conversations WHERE user_id = ?", (email,))
            con.execute("UPDATE usage SET user_id = 'deleted' WHERE user_id = ?", (email,))
            con.execute("DELETE FROM users WHERE email = ?", (email,))

    # --- conversations ---

    def create_conversation(self, user_id: str, title: str = "New chat") -> int:
        now = self._clock()
        with self._conn() as con:
            cursor = con.execute(
                "INSERT INTO conversations(user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (user_id, title, now, now),
            )
            return int(cursor.lastrowid)

    def conversation_exists(self, user_id: str, conversation_id: int) -> bool:
        with self._conn() as con:
            row = con.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id)
            ).fetchone()
        return row is not None

    def list_conversations(self, user_id: str, limit: int = 30) -> list[Conversation]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT id, title, updated_at FROM conversations WHERE user_id = ? "
                "ORDER BY updated_at DESC, id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [Conversation(r["id"], r["title"], r["updated_at"]) for r in rows]

    def delete_conversation(self, user_id: str, conversation_id: int) -> bool:
        with self._conn() as con:
            cursor = con.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id)
            )
            return cursor.rowcount > 0

    # --- messages ---

    def add_message(
        self,
        user_id: str,
        conversation_id: int,
        role: str,
        content: str,
        sources: list[dict] | None = None,
    ) -> None:
        now = self._clock()
        with self._conn() as con:
            owned = con.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id)
            ).fetchone()
            if owned is None:
                raise LookupError("Conversation not found.")
            con.execute(
                "INSERT INTO messages(conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, json.dumps(sources) if sources else None, now),
            )
            con.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))

    def get_messages(self, user_id: str, conversation_id: int) -> list[dict]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT m.role, m.content, m.sources FROM messages m "
                "JOIN conversations c ON c.id = m.conversation_id "
                "WHERE m.conversation_id = ? AND c.user_id = ? ORDER BY m.id",
                (conversation_id, user_id),
            ).fetchall()
        return [
            {"role": r["role"], "content": r["content"], "sources": json.loads(r["sources"]) if r["sources"] else []}
            for r in rows
        ]

    # --- usage ---

    def record_usage(self, user_id: str, tokens: int) -> None:
        with self._conn() as con:
            con.execute("INSERT INTO usage(user_id, ts, tokens) VALUES (?, ?, ?)", (user_id, self._clock(), tokens))

    def usage_since(self, user_id: str, since: float) -> list[tuple[float, int]]:
        with self._conn() as con:
            rows = con.execute(
                "SELECT ts, tokens FROM usage WHERE user_id = ? AND ts >= ? ORDER BY ts", (user_id, since)
            ).fetchall()
        return [(r["ts"], r["tokens"]) for r in rows]

    def app_tokens_since(self, since: float) -> int:
        with self._conn() as con:
            row = con.execute("SELECT COALESCE(SUM(tokens), 0) AS total FROM usage WHERE ts >= ?", (since,)).fetchone()
        return int(row["total"])