"""Deleting a user's data, on request or after long inactivity."""
from __future__ import annotations

import time

from .db import Database
from .ocr import OCRCache
from .store import VectorStore

DAY = 86400


def delete_user_data(user_id: str, store: VectorStore, cache: OCRCache | None, db: Database) -> None:
    """Remove documents (chunks and OCR cache), chats and the account row."""
    if cache is not None:
        for doc in store.list_docs(user_id):
            cache.delete(doc.doc_id)
    store.delete_user(user_id)
    db.delete_user(user_id)


def purge_inactive(
    days: int,
    store: VectorStore,
    cache: OCRCache | None,
    db: Database,
    now: float | None = None,
) -> list[str]:
    """Delete every account not seen for `days` days. days <= 0 turns this off."""
    if days <= 0:
        return []
    cutoff = (time.time() if now is None else now) - days * DAY
    purged = db.inactive_users(cutoff)
    for user_id in purged:
        delete_user_data(user_id, store, cache, db)
    return purged