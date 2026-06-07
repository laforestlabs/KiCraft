"""Tests for kicraft.server.spend_guard per-project cost attribution.

Pure stdlib + sqlite. SpendGuard only touches `settings.ledger_path` for these
paths, so a SimpleNamespace stands in for a full Settings object.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from kicraft.server.spend_guard import SpendGuard


@pytest.fixture
def guard(tmp_path):
    return SpendGuard(SimpleNamespace(ledger_path=str(tmp_path / "ledger.db")))


def _rec(guard, run_id, cost, stage="intent"):
    guard.record("deepseek/deepseek-v4-flash", 100, 50, cost,
                 meta={"run_id": run_id, "stage": stage})


def test_spent_for_project_sums_only_that_project(guard):
    _rec(guard, "p5-1000", 0.01)
    _rec(guard, "p5-1000", 0.02)     # same run
    _rec(guard, "p5-2000", 0.03)     # a later run of the SAME project (reopen/continue)
    _rec(guard, "p6-1000", 0.10)     # a different project
    _rec(guard, "p50-1000", 0.99)    # 'p5-%' must NOT match 'p50-...' (the '-' guards it)
    assert guard.spent_for_project(5) == pytest.approx(0.06)   # 0.01 + 0.02 + 0.03
    assert guard.spent_for_project(6) == pytest.approx(0.10)
    assert guard.spent_for_project(50) == pytest.approx(0.99)
    assert guard.spent_for_project(999) == 0.0                 # no calls -> 0, not None
    # the whole point: a project's spend is NOT the global running total (the old bug)
    assert guard.spent_total() == pytest.approx(1.15)


def test_spent_for_project_ignores_legacy_bare_meta(guard):
    guard.record("m", 1, 1, 0.05, meta="stream")   # legacy bare-string meta, no run_id
    _rec(guard, "p7-1", 0.02)
    assert guard.spent_for_project(7) == pytest.approx(0.02)   # bare row skipped, no crash


def test_spent_for_project_none_is_zero(guard):
    assert guard.spent_for_project(None) == 0.0
