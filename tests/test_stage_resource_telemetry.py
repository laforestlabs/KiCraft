"""Per-stage resource telemetry: wall_s / cpu_s / rounds / tool_calls.

Exercises the three sinks together: the durable StageStatus fields in
state.json (GUI restore), the spend ledger's stage_runs table (cross-run
report), and the values returned to the caller. The fake client returns
garbage so every stage ends on the fail path, which is exactly where the
driver records its final timing; a real SpendGuard is attached so the
ledger write is exercised (not the spend ceiling).
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

from kicraft.server.spend_guard import SpendGuard
from kicraft.server.stage_driver import drive_stage


class _FailingClient:
    """Returns invalid JSON so the stage exhausts retries and hits the
    final-fail recording path. Records nothing itself; timing is the subject."""

    def __init__(self, guard):
        self.guard = guard

    def chat(
        self,
        messages,
        max_tokens=4096,
        temperature=0.2,
        progress=None,
        meta_ctx=None,
        reasoning=None,
        reasoning_guard=None,
        collection_bounds=(),
        response_format=None,
    ):
        return {"text": "not json", "cost_usd": 0.0, "reasoning": "", "finish_reason": "stop"}

    def chat_with_tools(
        self,
        messages,
        tools,
        executor,
        max_tokens=4096,
        temperature=0.2,
        max_rounds=6,
        progress=None,
        meta_ctx=None,
        reasoning=None,
        reasoning_guard=None,
        collection_bounds=(),
        response_format=None,
    ):
        return {
            "text": "not json",
            "cost_usd": 0.0,
            "rounds": 1,
            "tool_calls": 0,
            "finish_reason": "stop",
        }


def _guard(tmp_path) -> SpendGuard:
    return SpendGuard(SimpleNamespace(ledger_path=str(tmp_path / "ledger.db")))


def _drive(tmp_path, stage):
    (tmp_path / ".kicraft").mkdir(parents=True, exist_ok=True)
    sp = tmp_path / ".kicraft" / "state.json"
    state = {}
    if stage == "bom":
        state["architecture"] = {
            "sheets": [{"name": "POWER", "stem": "POWER", "function": "Power"}],
            "power_nets": [],
            "inter_sheet_nets": [],
        }
    sp.write_text(json.dumps(state), encoding="utf-8")
    g = _guard(tmp_path)
    client = _FailingClient(g)
    r = drive_stage(
        client, stage, "a USB-powered LED", sp, tmp_path, max_retries=0, meta_ctx={"run_id": "p7-1"}
    )
    return r, g, sp


def test_stage_status_model_carries_resource_fields():
    from kicraft.design.models import StageStatus

    ss = StageStatus(ok=True, wall_s=33.7, cpu_s=1.8, rounds=4, tool_calls=12)
    d = ss.model_dump()
    assert d["wall_s"] == 33.7 and d["cpu_s"] == 1.8
    assert d["rounds"] == 4 and d["tool_calls"] == 12


def test_failed_bom_records_wall_cpu_rounds_to_all_three_sinks(tmp_path):
    r, g, sp = _drive(tmp_path, "bom")
    assert r["commit_ok"] is False
    # caller-visible
    assert r["wall_s"] is not None and r["wall_s"] >= 0.0
    assert r["cpu_s"] is not None and r["cpu_s"] >= 0.0
    assert r["rounds"] == 1 and r["tool_calls"] == 0
    # the terminal classification and ACTUAL call count (normal + one
    # serialization call) reach the returned result
    assert r["failure_kind"] == "invalid_json"
    assert r["attempts"] == 2
    # durable state.json stage_status
    sj = json.loads(sp.read_text(encoding="utf-8"))
    e = sj["stage_status"]["bom"]
    assert e["ok"] is False
    assert e["wall_s"] == r["wall_s"] and e["cpu_s"] == r["cpu_s"]
    assert e["rounds"] == 1 and e["tool_calls"] == 0
    assert e["failure_kind"] == "invalid_json" and e["attempts"] == 2
    # ledger stage_runs
    with sqlite3.connect(g.path) as c:
        row = c.execute(
            "SELECT stage, ok, rounds, tool_calls, wall_s, cpu_s, "
            "cost_usd, run_id, failure_kind, emitted_collection_count, "
            "expanded_component_count FROM stage_runs"
        ).fetchone()
    assert row[0] == "bom" and row[1] == 0 and row[2] == 1 and row[3] == 0
    assert row[4] == r["wall_s"] and row[6] == 0.0 and row[7] == "p7-1"
    assert row[8] == "invalid_json"
    assert row[9:] == (0, 0)


def test_failed_intent_records_wall_cpu_but_null_rounds(tmp_path):
    r, g, sp = _drive(tmp_path, "intent")
    assert r["commit_ok"] is False
    assert r["wall_s"] is not None and r["cpu_s"] is not None
    assert r["rounds"] is None and r["tool_calls"] is None  # single-shot stage
    with sqlite3.connect(g.path) as c:
        row = c.execute("SELECT rounds, tool_calls FROM stage_runs").fetchone()
    assert row == (None, None)


def test_web_cost_report_aggregates_stage_resources():
    from kicraft.cli.web_cost_report import format_stage_runs, summarize_stage_runs

    rows = [
        {
            "stage": "bom",
            "ok": True,
            "wall_s": 40.0,
            "cpu_s": 1.0,
            "cost_usd": 0.05,
            "rounds": 6,
            "tool_calls": 12,
            "attempts": 1,
        },
        {
            "stage": "bom",
            "ok": False,
            "wall_s": 20.0,
            "cpu_s": 0.5,
            "cost_usd": 0.02,
            "rounds": 6,
            "tool_calls": 8,
            "attempts": 5,
        },
        {
            "stage": "wiring",
            "ok": True,
            "wall_s": 5.0,
            "cpu_s": 0.1,
            "cost_usd": 0.01,
            "rounds": None,
            "tool_calls": None,
            "attempts": 1,
        },
    ]
    agg = summarize_stage_runs(rows)
    assert agg["bom"]["n"] == 2
    assert agg["bom"]["wall_s"] == 60.0 and agg["bom"]["cpu_s"] == 1.5
    assert agg["bom"]["rounds"] == 12 and agg["bom"]["tool_calls"] == 20
    assert agg["bom"]["fails"] == 1
    assert agg["wiring"]["rounds"] == 0  # None contributes 0
    txt = format_stage_runs(rows)
    assert "By stage" in txt and "bom" in txt and "wiring" in txt
    assert "cpu/wall" in txt  # the wall/cpu ratio line
