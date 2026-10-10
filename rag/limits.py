"""Per-user and app-wide usage limits. A limit of 0 turns that limit off."""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import Settings
from .db import Database
from .errors import LimitError

HOUR = 3600
DAY = 86400


def month_start(now: float) -> float:
    moment = datetime.fromtimestamp(now, tz=timezone.utc)
    return moment.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()


@dataclass
class QuotaStatus:
    questions_left: int | None  # None means no limit
    tokens_left: int | None


class QuotaGuard:
    def __init__(self, db: Database, settings: Settings, clock: Callable[[], float] = time.time) -> None:
        self._db = db
        self._settings = settings
        self._clock = clock

    def check(self, user_id: str) -> None:
        """Raise LimitError with a plain message if this question is not allowed."""
        now = self._clock()
        s = self._settings

        if s.monthly_token_cap > 0 and self._db.app_tokens_since(month_start(now)) >= s.monthly_token_cap:
            raise LimitError("Gilto has used its monthly allowance. It resets at the start of next month.")

        if s.hourly_question_limit > 0:
            rows = self._db.usage_since(user_id, now - HOUR)
            if len(rows) >= s.hourly_question_limit:
                minutes = max(1, math.ceil((rows[0][0] + HOUR - now) / 60))
                raise LimitError(
                    f"You have reached the limit of {s.hourly_question_limit} questions per hour. "
                    f"Try again in about {minutes} minute(s)."
                )

        if s.daily_token_limit > 0:
            used = sum(tokens for _, tokens in self._db.usage_since(user_id, now - DAY))
            if used >= s.daily_token_limit:
                raise LimitError("You have used your daily allowance. It frees up gradually over the next 24 hours.")

    def status(self, user_id: str) -> QuotaStatus:
        now = self._clock()
        s = self._settings
        questions_left = tokens_left = None
        if s.hourly_question_limit > 0:
            used = len(self._db.usage_since(user_id, now - HOUR))
            questions_left = max(0, s.hourly_question_limit - used)
        if s.daily_token_limit > 0:
            used = sum(tokens for _, tokens in self._db.usage_since(user_id, now - DAY))
            tokens_left = max(0, s.daily_token_limit - used)
        return QuotaStatus(questions_left, tokens_left)

    def record(self, user_id: str, tokens: int) -> None:
        self._db.record_usage(user_id, max(0, int(tokens)))