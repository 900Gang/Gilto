"""Helpers for whole-document requests such as "summarize this". No model calls here."""
from __future__ import annotations

import re

from .store import Retrieved

MAX_SUMMARY_DOCS = 4
MAX_SUMMARY_CHUNKS = 400  # about 100 pages
SUMMARY_BATCH_CHARS = 20000

_INTENT = re.compile(
    "|".join(
        [
            r"\bsummari[sz]\w*",
            r"\bsummary\b",
            r"\bsummaries\b",
            r"\boverview\b",
            r"\btl;?dr\b",
            r"\bgist\b",
            r"\boutline\b",
            r"\b(?:main|key) points?\b",
            r"\bwhat(?:'s| is| are)? (?:this|these|the) (?:document|documents|file|files|pdf|pdfs|paper|report)s? about\b",
            r"\bcompar(?:e|es|ed|ing|ison|isons)\b",
        ]
    ),
    re.IGNORECASE,
)


def is_summary_request(question: str) -> bool:
    return bool(_INTENT.search(question))


def batch_chunks(chunks: list[Retrieved], max_chars: int) -> list[list[Retrieved]]:
    """Group chunks, in order, into batches of at most max_chars characters.
    A single chunk longer than max_chars gets a batch of its own."""
    batches: list[list[Retrieved]] = []
    current: list[Retrieved] = []
    size = 0
    for chunk in chunks:
        if current and size + len(chunk.text) > max_chars:
            batches.append(current)
            current, size = [], 0
        current.append(chunk)
        size += len(chunk.text)
    if current:
        batches.append(current)
    return batches