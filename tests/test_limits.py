import dataclasses

import pytest

from rag.config import Settings
from rag.db import Database
from rag.errors import LimitError
from rag.limits import DAY, HOUR, QuotaGuard, month_start


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def make(tmp_path, **overrides):
    clock = Clock()
    db = Database(str(tmp_path / "g.db"), clock=clock)
    db.upsert_user("a@x.com")
    db.upsert_user("b@x.com")
    base = dict(hourly_question_limit=3, daily_token_limit=1000, monthly_token_cap=5000)
    settings = dataclasses.replace(Settings.from_env(), **{**base, **overrides})
    return QuotaGuard(db, settings, clock=clock), clock


def test_under_every_limit_is_allowed(tmp_path):
    guard, _ = make(tmp_path)
    guard.record("a@x.com", 100)
    guard.check("a@x.com")


def test_hourly_question_limit_blocks_and_reports_wait(tmp_path):
    guard, clock = make(tmp_path)
    for _ in range(3):
        guard.record("a@x.com", 1)
        clock.t += 60
    with pytest.raises(LimitError, match=r"3 questions per hour.*57 minute"):
        guard.check("a@x.com")
    guard.check("b@x.com")  # other users are unaffected


def test_hourly_window_slides(tmp_path):
    guard, clock = make(tmp_path)
    for _ in range(3):
        guard.record("a@x.com", 1)
    clock.t += HOUR + 1
    guard.check("a@x.com")


def test_daily_token_limit_blocks(tmp_path):
    guard, clock = make(tmp_path, hourly_question_limit=0)
    guard.record("a@x.com", 1000)
    with pytest.raises(LimitError, match="daily allowance"):
        guard.check("a@x.com")
    clock.t += DAY + 1
    guard.check("a@x.com")


def test_monthly_cap_blocks_everyone_until_next_month(tmp_path):
    guard, clock = make(tmp_path, hourly_question_limit=0, daily_token_limit=0)
    guard.record("a@x.com", 5000)
    with pytest.raises(LimitError, match="monthly allowance"):
        guard.check("b@x.com")
    clock.t = month_start(clock.t) + 32 * DAY
    guard.check("b@x.com")


def test_zero_turns_a_limit_off(tmp_path):
    guard, _ = make(tmp_path, hourly_question_limit=0, daily_token_limit=0, monthly_token_cap=0)
    for _ in range(50):
        guard.record("a@x.com", 10**6)
    guard.check("a@x.com")
    status = guard.status("a@x.com")
    assert status.questions_left is None
    assert status.tokens_left is None


def test_status_counts_down(tmp_path):
    guard, _ = make(tmp_path)
    guard.record("a@x.com", 400)
    status = guard.status("a@x.com")
    assert status.questions_left == 2
    assert status.tokens_left == 600


def test_negative_usage_is_recorded_as_zero(tmp_path):
    guard, _ = make(tmp_path)
    guard.record("a@x.com", -50)
    assert guard.status("a@x.com").tokens_left == 1000