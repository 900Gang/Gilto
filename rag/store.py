"""Chroma wrapper. One collection per user, plus a user_id filter on every read."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import chromadb

from .chunker import Chunk


@dataclass
class Retrieved:
    text: str
    metadata: dict
    score: float | None  # cosine similarity (higher is closer); None for a keyword-only match

    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": self.metadata, "score": self.score}

    @classmethod
    def from_dict(cls, data: dict) -> "Retrieved":
        return cls(text=data["text"], metadata=data["metadata"], score=data["score"])


@dataclass
class DocInfo:
    doc_id: str
    filename: str
    chunks: int
    page_count: int


class VectorStore:
    def __init__(self, path: str | None = None, client=None) -> None:
        if client is None:
            if path is None:
                raise ValueError("VectorStore needs a path or a client")
            client = chromadb.PersistentClient(path=path)
        self._client = client

    @staticmethod
    def _name(user_id: str) -> str:
        return "u_" + hashlib.sha256(user_id.encode()).hexdigest()[:32]

    def _collection(self, user_id: str):
        return self._client.get_or_create_collection(name=self._name(user_id), metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _where(user_id: str, doc_ids: list[str] | None = None) -> dict:
        user = {"user_id": user_id}
        if doc_ids is None:
            return user
        return {"$and": [user, {"doc_id": {"$in": doc_ids}}]}

    def add_chunks(self, user_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self._collection(user_id).upsert(
            ids=[f"{c.metadata['doc_id']}:{c.metadata['chunk_index']}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{**c.metadata, "user_id": user_id} for c in chunks],
        )

    def delete_user(self, user_id: str) -> None:
        """Drop everything stored for a user. Does nothing if they have no data."""
        name = self._name(user_id)
        existing = {c if isinstance(c, str) else c.name for c in self._client.list_collections()}
        if name in existing:
            self._client.delete_collection(name)

    def query(
        self,
        user_id: str,
        embedding: list[float],
        k: int = 5,
        doc_ids: list[str] | None = None,
    ) -> list[Retrieved]:
        collection = self._collection(user_id)
        total = collection.count()
        if total == 0:
            return []
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(k, total),
            where=self._where(user_id, doc_ids),
        )
        return [
            Retrieved(text=text, metadata=meta, score=1.0 - dist)
            for text, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def all_chunks(self, user_id: str, doc_ids: list[str] | None = None) -> list[Retrieved]:
        """Every chunk the user has, in reading order (file name, then position).
        Used for keyword search and for summaries. Scores are None."""
        found = self._collection(user_id).get(
            where=self._where(user_id, doc_ids), include=["documents", "metadatas"]
        )
        chunks = [
            Retrieved(text=text, metadata=meta, score=None)
            for text, meta in zip(found["documents"], found["metadatas"])
        ]
        chunks.sort(key=lambda c: (c.metadata["filename"].lower(), c.metadata["doc_id"], c.metadata["chunk_index"]))
        return chunks

    def has_doc(self, user_id: str, doc_id: str) -> bool:
        found = self._collection(user_id).get(where=self._where(user_id, [doc_id]), limit=1)
        return len(found["ids"]) > 0

    def list_docs(self, user_id: str) -> list[DocInfo]:
        found = self._collection(user_id).get(where={"user_id": user_id}, include=["metadatas"])
        docs: dict[str, DocInfo] = {}
        for meta in found["metadatas"]:
            info = docs.get(meta["doc_id"])
            if info is None:
                docs[meta["doc_id"]] = DocInfo(
                    doc_id=meta["doc_id"],
                    filename=meta["filename"],
                    chunks=1,
                    page_count=int(meta.get("page_count", 0)),
                )
            else:
                info.chunks += 1
        return sorted(docs.values(), key=lambda d: d.filename.lower())

    def delete_doc(self, user_id: str, doc_id: str) -> None:
        self._collection(user_id).delete(where=self._where(user_id, [doc_id]))