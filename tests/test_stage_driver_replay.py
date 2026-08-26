"""Tests for the LLM-stage replay / full-pipeline harness (stage_driver).

Pure units, no network: the per-run budget guard, the mock-mode budget client,
and the frozen-state seeding + brief recovery of ``drive_replay`` (via a
monkeypatched ``drive_chain`` so no stage-prep subprocess or model call runs).
"""
from __future__ import annotations

import json

import pytest

from kicraft.server import stage_pipeline as sd
from kicraft.server.spend_guard import BudgetExceeded
from kicraft.server.stage_pipeline import (
    _BudgetGuard,
    drive_replay,
    make_budget_client,
)


class _FakeBase:
    """A minimal SpendGuard stand-in with a mutable spend total."""

    def __init__(self, spent: float = 0.0):
        self.spent = spent

    def spent_total(self) -> float:
        return self.spent

    def preflight(self) -> None:  # no-op base ceilings
        return None

    def status(self) -> dict:
        return {"spent_total_usd": self.spent, "daily_remaining_usd": 1.0}

    def record(self, *a, **k) -> None:
        return None

    def record_stage(self, *a, **k) -> None:
        return None


def test_budget_guard_refuses_once_budget_exhausted():
    base = _FakeBase(0.0)
    g = _BudgetGuard(base, 0.25)
    g.preflight()  # under budget -> ok
    base.spent = 0.25
    with pytest.raises(BudgetExceeded):
        g.preflight()


def test_budget_guard_measures_only_its_own_run():
    # The snapshot is taken at construction, so a pre-existing global total
    # does not count against this run's budget.
    base = _FakeBase(spent=12.40)
    g = _BudgetGuard(base, 0.25)
    g.preflight()  # delta 0.0 -> ok
    base.spent = 12.40 + 0.24
    g.preflight()  # delta 0.24 -> ok
    base.spent = 12.40 + 0.25
    with pytest.raises(BudgetExceeded):
        g.preflight()


def test_budget_guard_delegates_other_methods():
    base = _FakeBase(0.0)
    g = _BudgetGuard(base, 0.25)
    assert g.status()["spent_total_usd"] == 0.0
    g.record("m", 1, 1, 0.0)
    g.record_stage(run_id=None, stage="intent", ok=True, attempts=1,
                   rounds=None, tool_calls=None, wall_s=0.0, cpu_s=0.0,
                   cost_usd=0.0)


def test_make_budget_client_mock_mode_skips_wrapper(monkeypatch):
    monkeypatch.setenv("KICRAFT_LLM_MODE", "mock")
    client = make_budget_client(0.25)
    # mock client carries a NullGuard, not the budget wrapper
    assert not isinstance(client.guard, _BudgetGuard)


def test_drive_replay_bad_state_path():
    out = drive_replay("/nonexistent/state.json", "wiring")
    assert "error" in out
    assert "not found" in out["error"]


def test_drive_replay_bad_stage(tmp_path):
    state = {"intent": {"goal": "Build a thing"}, "bom": {"parts": []}}
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps(state))
    out = drive_replay(str(sp), "not_a_stage")
    assert "error" in out
    assert "unsupported stage" in out["error"]


def test_drive_replay_seeds_temp_workspace_and_recovers_brief(tmp_path, monkeypatch):
    state = {"intent": {"goal": "Build a CAN bus node"},
             "bom": {"parts": [], "connections": []}}
    sp = tmp_path / "state.json"
    sp.write_text(json.dumps(state))

    captured = {}

    def _fake_drive_chain(stages, brief, workspace, **kw):
        captured["stages"] = list(stages)
        captured["brief"] = brief
        captured["workspace"] = str(workspace)
        captured["client_is_fake"] = kw.get("client") is fake_client
        return ([{"stage": stages[0], "commit_ok": True}],
                {"spent_total_usd": 0.0}, str(workspace / ".kicraft" / "state.json"))

    monkeypatch.setattr(sd, "drive_chain", _fake_drive_chain)
    fake_client = object()

    out = drive_replay(str(sp), "wiring", client=fake_client)

    assert out["all_committed"] is True
    assert out["brief"] == "Build a CAN bus node"
    assert captured["stages"] == ["wiring"]
    assert captured["brief"] == "Build a CAN bus node"
    assert captured["client_is_fake"] is True
    # source untouched, replay workspace holds a copy
    assert sp.read_text() == json.dumps(state)
    assert captured["workspace"] != str(tmp_path)
