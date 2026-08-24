"""Tests for kicraft.server.spend_guard per-project cost attribution.

Pure stdlib + sqlite. SpendGuard only touches `settings.ledger_path` for these
paths, so a SimpleNamespace stands in for a full Settings object.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
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


def test_spent_by_day_counts_all_calls(guard):
    _rec(guard, "p1-1", 0.02)
    _rec(guard, "p1-1", 0.03)                       # two project calls today
    guard.record("m", 1, 1, 0.05, meta="eval")      # a NON-project call counts too
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)).isoformat()
    with sqlite3.connect(guard.path) as c:          # backdate one row 10 days
        c.execute("INSERT INTO spend (ts, model, input_tokens, output_tokens, "
                  "cost_usd, meta) VALUES (?, 'm', 0, 0, 0.07, 'x')", (old,))
    series = dict(guard.spent_by_day(30))
    assert series[dt.date.today().isoformat()] == pytest.approx(0.10)  # incl. non-project
    assert sum(series.values()) == pytest.approx(0.17)                 # + backdated 0.07
    # ...and the all-day total equals the ledger total (matches OpenRouter)
    assert sum(series.values()) == pytest.approx(guard.spent_total())
    assert sum(dict(guard.spent_by_day(5)).values()) == pytest.approx(0.10)  # window


def test_record_stage_writes_resource_row(guard):
    guard.record_stage(run_id="p1-1", stage="bom", ok=True, attempts=1,
                       rounds=4, tool_calls=12, wall_s=33.7, cpu_s=1.8, cost_usd=0.04)
    with sqlite3.connect(guard.path) as c:
        row = c.execute(
            "SELECT run_id, stage, ok, attempts, rounds, tool_calls, wall_s, cpu_s, "
            "cost_usd, failure_kind FROM stage_runs").fetchone()
    assert row == ("p1-1", "bom", 1, 1, 4, 12, 33.7, 1.8, 0.04, None)


def test_record_stage_classifies_failures(guard):
    guard.record_stage(run_id="p1-1", stage="bom", ok=False, attempts=2,
                       rounds=6, tool_calls=8, wall_s=20.0, cpu_s=0.5,
                       cost_usd=0.02, failure_kind="invalid_json")
    guard.record_stage(run_id="p1-1", stage="wiring", ok=False, attempts=5,
                       rounds=None, tool_calls=None, wall_s=30.0, cpu_s=0.9,
                       cost_usd=0.03, failure_kind="commit_rejected")
    with sqlite3.connect(guard.path) as c:
        kinds = c.execute(
            "SELECT stage, failure_kind FROM stage_runs ORDER BY stage").fetchall()
    assert kinds == [("bom", "invalid_json"), ("wiring", "commit_rejected")]


def test_stage_runs_schema_migrates_legacy_ledger_and_keeps_rows(tmp_path):
    # A ledger created BEFORE the failure_kind column must gain it via ALTER
    # TABLE (CREATE TABLE IF NOT EXISTS alone cannot), keep its old rows, and
    # accept classified rows alongside them — all readable by the cost report.
    db = tmp_path / "ledger.db"
    with sqlite3.connect(db) as c:
        c.execute(
            "CREATE TABLE stage_runs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, run_id TEXT, "
            "stage TEXT NOT NULL, ok INTEGER, attempts INTEGER, rounds INTEGER, "
            "tool_calls INTEGER, wall_s REAL, cpu_s REAL, cost_usd REAL)")
        c.execute(
            "INSERT INTO stage_runs (ts, run_id, stage, ok, attempts, wall_s, "
            "cost_usd) VALUES ('2026-07-01T00:00:00+00:00', 'p9-1', 'bom', 0, 4, "
            "10.0, 0.01)")
    SpendGuard(SimpleNamespace(ledger_path=str(db)))  # init runs the migration
    guard = SpendGuard(SimpleNamespace(ledger_path=str(db)))
    guard.record_stage(run_id="p9-2", stage="bom", ok=False, attempts=2,
                       rounds=6, tool_calls=8, wall_s=20.0, cpu_s=0.5,
                       cost_usd=0.02, failure_kind="invalid_json")
    from kicraft.cli.web_cost_report import load_stage_runs
    rows = load_stage_runs(str(db))
    assert len(rows) == 2
    legacy = next(r for r in rows if r["run_id"] == "p9-1")
    classified = next(r for r in rows if r["run_id"] == "p9-2")
    assert legacy["failure_kind"] is None        # old row stays readable, unclassified
    assert classified["failure_kind"] == "invalid_json"
    assert classified["attempts"] == 2


def test_record_stage_nulls_rounds_for_single_shot_stages(guard):
    guard.record_stage(run_id="p1-1", stage="intent", ok=True, attempts=2,
                       rounds=None, tool_calls=None, wall_s=2.1, cpu_s=0.05,
                       cost_usd=0.01)
    with sqlite3.connect(guard.path) as c:
        row = c.execute("SELECT rounds, tool_calls FROM stage_runs").fetchone()
    assert row == (None, None)


def test_record_stage_does_not_inflate_spend_ceiling(guard):
    # stage_runs cost mirrors LLM spend for the report but must NOT be summed
    # into the spend ceiling (the per-call `spend` rows own that).
    guard.record_stage(run_id="p1-1", stage="bom", ok=True, attempts=1,
                       rounds=1, tool_calls=1, wall_s=1.0, cpu_s=0.0, cost_usd=5.0)
    assert guard.spent_total() == 0.0
    assert guard.spent_today() == 0.0
