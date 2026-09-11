"""Tests for kicraft.server.spend_guard per-project cost attribution.

Pure stdlib + sqlite. SpendGuard only touches `settings.ledger_path` for these
paths, so a SimpleNamespace stands in for a full Settings object.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from types import SimpleNamespace

import pytest

from kicraft.server.spend_guard import BudgetExceeded, SpendGuard


@pytest.fixture
def guard(tmp_path):
    return SpendGuard(SimpleNamespace(ledger_path=str(tmp_path / "ledger.db")))


def _rec(guard, run_id, cost, stage="intent"):
    guard.record(
        "deepseek/deepseek-v4-flash", 100, 50, cost, meta={"run_id": run_id, "stage": stage}
    )


def test_spent_for_project_sums_only_that_project(guard):
    _rec(guard, "p5-1000", 0.01)
    _rec(guard, "p5-1000", 0.02)  # same run
    _rec(guard, "p5-2000", 0.03)  # a later run of the SAME project (reopen/continue)
    _rec(guard, "p6-1000", 0.10)  # a different project
    _rec(guard, "p50-1000", 0.99)  # 'p5-%' must NOT match 'p50-...' (the '-' guards it)
    assert guard.spent_for_project(5) == pytest.approx(0.06)  # 0.01 + 0.02 + 0.03
    assert guard.spent_for_project(6) == pytest.approx(0.10)
    assert guard.spent_for_project(50) == pytest.approx(0.99)
    assert guard.spent_for_project(999) == 0.0  # no calls -> 0, not None
    # the whole point: a project's spend is NOT the global running total (the old bug)
    assert guard.spent_total() == pytest.approx(1.15)


def test_spent_for_project_ignores_legacy_bare_meta(guard):
    guard.record("m", 1, 1, 0.05, meta="stream")  # legacy bare-string meta, no run_id
    _rec(guard, "p7-1", 0.02)
    assert guard.spent_for_project(7) == pytest.approx(0.02)  # bare row skipped, no crash


def test_spent_for_project_none_is_zero(guard):
    assert guard.spent_for_project(None) == 0.0


def test_preflight_reserves_call_against_project_run_budget(tmp_path):
    settings = SimpleNamespace(
        ledger_path=str(tmp_path / "ledger.db"),
        kill_switch=False,
        daily_usd_ceiling=10.0,
        total_usd_ceiling=10.0,
        project_llm_budget_usd=0.10,
    )
    guarded = SpendGuard(settings)
    _rec(guarded, "p7-run", 0.07)
    guarded.preflight(call_ceiling_usd=0.03, run_id="p7-run")
    with pytest.raises(BudgetExceeded, match="project run p7-run remaining budget"):
        guarded.preflight(call_ceiling_usd=0.031, run_id="p7-run")
    guarded.preflight(call_ceiling_usd=0.09, run_id="another-run")


def test_configured_project_cap_admits_only_fully_covered_calls(tmp_path, monkeypatch):
    from dataclasses import replace

    from kicraft.server.config import Settings

    settings = Settings(
        api_key="test",
        ledger_path=tmp_path / "ledger.db",
        kill_switch=False,
        daily_usd_ceiling=10.0,
        total_usd_ceiling=10.0,
    )
    guard = SpendGuard(settings)
    _rec(guard, "run", 0.0648, stage="bom")
    with pytest.raises(BudgetExceeded) as refused:
        guard.preflight(call_ceiling_usd=0.0384, run_id="run")
    assert refused.value.scope == "project"
    assert refused.value.spent_usd == pytest.approx(0.0648)
    assert refused.value.call_ceiling_usd == pytest.approx(0.0384)
    assert refused.value.limit_usd == pytest.approx(0.10)

    monkeypatch.setenv("KICRAFT_PROJECT_LLM_BUDGET_USD", "0.15")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    configured = replace(
        settings, project_llm_budget_usd=Settings.from_env(dotenv=False).project_llm_budget_usd
    )
    larger_guard = SpendGuard(configured)
    larger_guard.preflight(call_ceiling_usd=0.0384, run_id="run")
    # Admission is not a charge. Only the billed call reduces later headroom.
    assert larger_guard.spent_for_run("run") == pytest.approx(0.0648)
    _rec(larger_guard, "run", 0.0384, stage="bom")
    with pytest.raises(BudgetExceeded):
        larger_guard.preflight(call_ceiling_usd=0.046801, run_id="run")
    assert larger_guard.spent_by_stage_for_run("run") == {"bom": pytest.approx(0.1032)}


def test_spent_by_day_counts_all_calls(guard):
    _rec(guard, "p1-1", 0.02)
    _rec(guard, "p1-1", 0.03)  # two project calls today
    guard.record("m", 1, 1, 0.05, meta="eval")  # a NON-project call counts too
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=10)).isoformat()
    with sqlite3.connect(guard.path) as c:  # backdate one row 10 days
        c.execute(
            "INSERT INTO spend (ts, model, input_tokens, output_tokens, "
            "cost_usd, meta) VALUES (?, 'm', 0, 0, 0.07, 'x')",
            (old,),
        )
    series = dict(guard.spent_by_day(30))
    assert series[dt.date.today().isoformat()] == pytest.approx(0.10)  # incl. non-project
    assert sum(series.values()) == pytest.approx(0.17)  # + backdated 0.07
    # ...and the all-day total equals the ledger total (matches OpenRouter)
    assert sum(series.values()) == pytest.approx(guard.spent_total())
    assert sum(dict(guard.spent_by_day(5)).values()) == pytest.approx(0.10)  # window


def test_record_stage_writes_resource_row(guard):
    guard.record_stage(
        run_id="p1-1",
        stage="bom",
        ok=True,
        attempts=1,
        rounds=4,
        tool_calls=12,
        wall_s=33.7,
        cpu_s=1.8,
        cost_usd=0.04,
    )
    with sqlite3.connect(guard.path) as c:
        row = c.execute(
            "SELECT run_id, stage, ok, attempts, rounds, tool_calls, wall_s, cpu_s, "
            "cost_usd, failure_kind FROM stage_runs"
        ).fetchone()
    assert row == ("p1-1", "bom", 1, 1, 4, 12, 33.7, 1.8, 0.04, None)


def test_record_stage_classifies_failures(guard):
    guard.record_stage(
        run_id="p1-1",
        stage="bom",
        ok=False,
        attempts=2,
        rounds=6,
        tool_calls=8,
        wall_s=20.0,
        cpu_s=0.5,
        cost_usd=0.02,
        failure_kind="invalid_json",
    )
    guard.record_stage(
        run_id="p1-1",
        stage="wiring",
        ok=False,
        attempts=5,
        rounds=None,
        tool_calls=None,
        wall_s=30.0,
        cpu_s=0.9,
        cost_usd=0.03,
        failure_kind="commit_rejected",
    )
    with sqlite3.connect(guard.path) as c:
        kinds = c.execute("SELECT stage, failure_kind FROM stage_runs ORDER BY stage").fetchall()
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
            "tool_calls INTEGER, wall_s REAL, cpu_s REAL, cost_usd REAL)"
        )
        c.execute(
            "INSERT INTO stage_runs (ts, run_id, stage, ok, attempts, wall_s, "
            "cost_usd) VALUES ('2026-07-01T00:00:00+00:00', 'p9-1', 'bom', 0, 4, "
            "10.0, 0.01)"
        )
    SpendGuard(SimpleNamespace(ledger_path=str(db)))  # init runs the migration
    guard = SpendGuard(SimpleNamespace(ledger_path=str(db)))
    guard.record_stage(
        run_id="p9-2",
        stage="bom",
        ok=False,
        attempts=2,
        rounds=6,
        tool_calls=8,
        wall_s=20.0,
        cpu_s=0.5,
        cost_usd=0.02,
        failure_kind="invalid_json",
    )
    from kicraft.cli.web_cost_report import load_stage_runs

    rows = load_stage_runs(str(db))
    assert len(rows) == 2
    legacy = next(r for r in rows if r["run_id"] == "p9-1")
    classified = next(r for r in rows if r["run_id"] == "p9-2")
    assert legacy["failure_kind"] is None  # old row stays readable, unclassified
    assert classified["failure_kind"] == "invalid_json"
    assert classified["attempts"] == 2


def test_record_stage_nulls_rounds_for_single_shot_stages(guard):
    guard.record_stage(
        run_id="p1-1",
        stage="intent",
        ok=True,
        attempts=2,
        rounds=None,
        tool_calls=None,
        wall_s=2.1,
        cpu_s=0.05,
        cost_usd=0.01,
    )
    with sqlite3.connect(guard.path) as c:
        row = c.execute("SELECT rounds, tool_calls FROM stage_runs").fetchone()
    assert row == (None, None)


def test_record_stage_does_not_inflate_spend_ceiling(guard):
    # stage_runs cost mirrors LLM spend for the report but must NOT be summed
    # into the spend ceiling (the per-call `spend` rows own that).
    guard.record_stage(
        run_id="p1-1",
        stage="bom",
        ok=True,
        attempts=1,
        rounds=1,
        tool_calls=1,
        wall_s=1.0,
        cpu_s=0.0,
        cost_usd=5.0,
    )
    assert guard.spent_total() == 0.0
    assert guard.spent_today() == 0.0


def test_work_unit_stage_and_attempt_metrics_are_redacted_and_persisted(guard):
    guard.record_stage(
        run_id="p1-1",
        stage="wiring",
        ok=True,
        attempts=3,
        rounds=None,
        tool_calls=None,
        wall_s=2.0,
        cpu_s=0.1,
        cost_usd=0.01,
        work_units=2,
        reused_work_units=1,
        aggregate_repair_rounds=1,
    )
    guard.record_stage_attempt(
        run_id="p1-1",
        stage="wiring",
        attempt=3,
        call_mode="normal",
        outcome="candidate",
        unit_id="wiring-u001",
        unit_attempt=2,
        aggregate_round=1,
        commit_gate_codes=["9.11", "9.29"],
        offender_count=3,
        rejection_signature="0123456789abcdef",
        provider_profile="flash",
        fallback_reason="provider_rate_limited",
        candidate_retained=True,
        reasoning_failure_kind="reasoning_token_exhaustion",
        schema_error="parts.0.mpn: string_type",
        failure_detail="invalid_schema",
        response_chars=417,
        response_format_mode="json_schema",
    )
    with sqlite3.connect(guard.path) as connection:
        run = connection.execute(
            "SELECT work_units,reused_work_units,aggregate_repair_rounds FROM stage_runs"
        ).fetchone()
        attempt = connection.execute(
            "SELECT unit_id,unit_attempt,aggregate_round,commit_gate_codes,"
            "offender_count,rejection_signature,provider_profile,fallback_reason,"
            "candidate_retained,reasoning_failure_kind,schema_error,failure_detail,"
            "response_chars,response_format_mode FROM stage_attempts"
        ).fetchone()
        attempt_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(stage_attempts)")
        }
    assert run == (2, 1, 1)
    assert attempt == (
        "wiring-u001",
        2,
        1,
        '["9.11","9.29"]',
        3,
        "0123456789abcdef",
        "flash",
        "provider_rate_limited",
        1,
        "reasoning_token_exhaustion",
        "parts.0.mpn: string_type",
        "invalid_schema",
        417,
        "json_schema",
    )
    assert {
        "unit_id",
        "unit_attempt",
        "aggregate_round",
        "commit_gate_codes",
        "offender_count",
        "rejection_signature",
        "provider_profile",
        "fallback_reason",
        "candidate_retained",
        "reasoning_failure_kind",
        "schema_error",
        "failure_detail",
        "response_chars",
        "response_format_mode",
    } <= attempt_columns
    aggregate = guard.stage_attempt_aggregates()[0]
    assert aggregate["unit_id"] == "wiring-u001"
    assert not ({"prompt", "response", "brief"} & aggregate.keys())


def test_cost_report_attributes_gate_signature_profile_and_work_unit(tmp_path):
    from kicraft.cli.web_cost_report import load_stage_attempts, summarize_stage_attempts

    guarded = SpendGuard(SimpleNamespace(ledger_path=str(tmp_path / "ledger.db")))
    common = {
        "run_id": "p4-run",
        "stage": "wiring",
        "unit_id": "wiring-u001",
        "provider_profile": "flash",
        "commit_gate_codes": ["9.15"],
        "offender_count": 1,
        "rejection_signature": "deadbeef",
    }
    guarded.record_stage_attempt(
        **common,
        attempt=1,
        call_mode="deterministic_commit",
        outcome="commit_rejected",
        cost_usd=0.0,
        candidate_retained=True,
    )
    guarded.record_stage_attempt(
        **common,
        attempt=2,
        call_mode="normal",
        outcome="candidate",
        unit_attempt=2,
        aggregate_round=1,
        cost_usd=0.012,
        candidate_retained=True,
    )

    summary = summarize_stage_attempts(load_stage_attempts(str(tmp_path / "ledger.db")))
    assert summary["provider_cost_usd"] == pytest.approx(0.012)
    assert summary["by_stage_profile"]["wiring|flash"] == {
        "calls": 1,
        "cost": pytest.approx(0.012),
    }
    assert summary["commit_gates"]["9.15"] == {
        "rejections": 1,
        "repair_calls": 1,
        "cost": pytest.approx(0.012),
    }
    assert summary["repeated_rejection_signatures"][0]["occurrences"] == 2
    assert summary["work_unit_passes"]["wiring|aggregate_repair"]["success_rate"] == 1.0


def test_work_unit_columns_migrate_legacy_stage_tables(tmp_path):
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as connection:
        connection.execute(
            "CREATE TABLE stage_runs (id INTEGER PRIMARY KEY,ts TEXT NOT NULL,"
            "run_id TEXT,stage TEXT NOT NULL,ok INTEGER,attempts INTEGER,rounds INTEGER,"
            "tool_calls INTEGER,wall_s REAL,cpu_s REAL,cost_usd REAL)"
        )
        connection.execute(
            "CREATE TABLE stage_attempts (id INTEGER PRIMARY KEY,ts TEXT NOT NULL,"
            "run_id TEXT,stage TEXT NOT NULL,attempt INTEGER NOT NULL,"
            "call_mode TEXT NOT NULL,outcome TEXT NOT NULL)"
        )
    SpendGuard(SimpleNamespace(ledger_path=str(db)))
    with sqlite3.connect(db) as connection:
        run_columns = {row[1] for row in connection.execute("PRAGMA table_info(stage_runs)")}
        attempt_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(stage_attempts)")
        }
    assert {"work_units", "reused_work_units", "aggregate_repair_rounds"} <= run_columns
    assert {"unit_id", "unit_attempt", "aggregate_round"} <= attempt_columns
