"""Merge several ranked lists into one (reciprocal rank fusion)."""
from __future__ import annotations

from collections.abc import Hashable


def rrf(rankings: list[list[Hashable]], k: int = 60) -> list[Hashable]:
    """An item near the top of any list scores well, and an item in several lists scores best.
    Ties are broken by key so the order is stable."""
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda key: (-scores[key], key))