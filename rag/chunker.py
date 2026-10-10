"""Recursive text splitter with overlap, plus per-page chunk metadata."""
from __future__ import annotations

from dataclasses import dataclass, field

from .loader import Page

_SEPARATORS = ["\n\n", "\n", ". ", " "]


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _split_pieces(text: str, size: int, separators: list[str]) -> list[str]:
    """Break `text` into pieces no longer than `size`, keeping separators
    attached to the piece they end, so joining pieces restores the text."""
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(separators):
        if sep not in text:
            continue
        parts = text.split(sep)
        parts = [p + sep for p in parts[:-1]] + [parts[-1]]
        pieces: list[str] = []
        for part in parts:
            if not part:
                continue
            if len(part) <= size:
                pieces.append(part)
            else:
                pieces.extend(_split_pieces(part, size, separators[i + 1 :]))
        return pieces
    return [text[j : j + size] for j in range(0, len(text), size)]


def split_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    pieces = _split_pieces(text, size, _SEPARATORS)
    chunks: list[str] = []
    current: list[str] = []
    length = 0

    def emit() -> None:
        chunk = "".join(current).strip()
        if chunk:
            chunks.append(chunk)

    for piece in pieces:
        if current and length + len(piece) > size:
            emit()
            # carry the tail of the finished chunk into the next one
            carry: list[str] = []
            carried = 0
            for prev in reversed(current):
                if carried + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carried += len(prev)
            while carry and carried + len(piece) > size:
                carried -= len(carry.pop(0))
            current, length = carry, carried
        current.append(piece)
        length += len(piece)
    emit()
    return chunks


def chunk_pages(
    pages: list[Page],
    doc_id: str,
    filename: str,
    size: int = 800,
    overlap: int = 150,
    source_type: str = "text",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for text in split_text(page.text, size, overlap):
            chunks.append(
                Chunk(
                    text=text,
                    metadata={
                        "doc_id": doc_id,
                        "filename": filename,
                        "page": page.number,
                        "chunk_index": len(chunks),
                        "source_type": source_type,
                    },
                )
            )
    return chunks
