"""Keyword search (BM25) over a user's chunks.

Runs in memory. A user has at most a few thousand chunks, so this is fast
enough and needs no extra index to keep in sync.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)  # the second range keeps Hindi vowel signs inside words

STOPWORDS = frozenset(
    (
        "a an and are as at be but by for from has have how i if in is it its of on or that the their there "
        "this to was were what when where which who why will with you your do does did can could should "
        "would about into than then so not no "
        "का की के है हैं में से और को पर यह वह था थी थे क्या कि"
    ).split()
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_terms(text: str) -> list[str]:
    """The query words that carry meaning, each once, in order."""
    seen: list[str] = []
    for token in tokenize(text):
        if token in STOPWORDS or (len(token) < 2 and not token.isdigit()):
            continue
        if token not in seen:
            seen.append(token)
    return seen


def keyword_rank(query: str, texts: list[str], k1: float = 1.5, b: float = 0.75) -> list[tuple[int, float]]:
    """Return (index, score) for texts that contain at least half of the query's
    meaningful words, best first. The half rule stops a single common word from
    pulling in unrelated passages."""
    terms = content_terms(query)
    if not terms or not texts:
        return []

    counts = [Counter(tokenize(text)) for text in texts]
    lengths = [sum(c.values()) for c in counts]
    average = (sum(lengths) / len(lengths)) or 1.0
    total = len(texts)
    doc_freq = {t: sum(1 for c in counts if t in c) for t in terms}
    needed = max(1, math.ceil(len(terms) / 2))

    ranked: list[tuple[int, float]] = []
    for index, bag in enumerate(counts):
        matched = [t for t in terms if bag.get(t)]
        if len(matched) < needed:
            continue
        score = 0.0
        for term in matched:
            idf = math.log(1 + (total - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            tf = bag[term]
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[index] / average))
        ranked.append((index, score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked