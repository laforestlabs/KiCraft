"""R3 driver: run_post_wiring_review — the web-flow electrical review between
the wiring commit and the build.

Covers the regressions found on KC-T6ERHM: the review ran for minutes with zero
events (GUI looked stalled), emitted a `[elec-review]` line no classifier
matched, and never persisted findings anywhere the reopened GUI reads
(state.review_findings / stage_status["electrical_review"]).
"""
from __future__ import annotations

import json
from pathlib import Path

import kicraft.design.cli_app as cli_app
from kicraft.design.cli_app import run_post_wiring_review
from kicraft.design.models import (
    BOM,
    BomPart,
    ConversationState,
    NetConnection,
    PinEndpoint,
)
from kicraft.server.stagetabs import _build_substage
from kicraft.server.web import _REVIEW_FINDING_RE


def _state() -> ConversationState:
    return ConversationState(bom=BOM(
        parts=[BomPart(ref="U1", value="x", symbol="Fake:X",
                       footprint="Package_SO:SOIC-8", sheet="A")],
        connections=[NetConnection(net_name="+3V3", sheet="A",
                                   endpoints=[PinEndpoint(ref="U1", pin="1")])],
    ))


def _write_state(tmp_path: Path) -> Path:
    sp = tmp_path / ".kicraft" / "state.json"
    sp.parent.mkdir(parents=True)
    sp.write_text(_state().model_dump_json(indent=2) + "\n")
    return sp


def _events_of(kind: str, events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("kind") == kind]


BLOCKER = {"severity": "blocker", "area": "power",
           "issue": "VSENSE divider swapped", "suggestion": "swap R1/R2"}
WARNING = {"severity": "warning", "area": "esd",
           "issue": "no TVS on USB", "suggestion": "add one"}


def test_clean_review_emits_stage_events_and_persists(tmp_path, monkeypatch):
    monkeypatch.delenv("KICRAFT_ELECTRICAL_REVIEW", raising=False)
    sp = _write_state(tmp_path)
    monkeypatch.setattr(cli_app, "_maybe_electrical_review", lambda st, pd: {
        "ran": True, "blocked": False, "findings": [WARNING], "cost_usd": 0.02})
    events: list[dict] = []
    res = run_post_wiring_review(sp, tmp_path, events.append, rewire=None)
    assert res["ran"] is True

    starts = _events_of("stage_start", events)
    dones = _events_of("stage_done", events)
    assert [e["stage"] for e in starts] == ["electrical_review"]
    assert [e["stage"] for e in dones] == ["electrical_review"]
    assert dones[0]["ok"] is True

    # Findings lines are classifier- and reopen-parser-compatible.
    logs = [e["text"] for e in _events_of("build_log", events)]
    finding_lines = [ln for ln in logs if _REVIEW_FINDING_RE.search(ln)]
    assert len(finding_lines) == 1 and "no TVS" in finding_lines[0]
    assert all(_build_substage(ln) == "electrical_review"
               for ln in logs if "review" in ln.lower())

    # Durable outcome: findings + stage_status survive to the reopened GUI.
    sj = json.loads(sp.read_text())
    assert [f["issue"] for f in sj["review_findings"]] == ["no TVS on USB"]
    er = sj["stage_status"]["electrical_review"]
    assert er["ok"] is True and er["cost_usd"] == 0.02
    # _surface_review_findings ran too: the warning reached bom.assumptions.
    assert any("no TVS" in a for a in sj["bom"]["assumptions"])


def test_blocker_re_drives_wiring_once_and_persists_second_pass(tmp_path, monkeypatch):
    monkeypatch.delenv("KICRAFT_ELECTRICAL_REVIEW", raising=False)
    sp = _write_state(tmp_path)
    passes = iter([
        {"ran": True, "blocked": True, "findings": [BLOCKER], "cost_usd": 0.02},
        {"ran": True, "blocked": False, "findings": [WARNING], "cost_usd": 0.03},
    ])
    monkeypatch.setattr(cli_app, "_maybe_electrical_review",
                        lambda st, pd: next(passes))
    rewires: list[str] = []
    events: list[dict] = []
    run_post_wiring_review(sp, tmp_path, events.append, rewires.append)

    assert len(rewires) == 1 and "VSENSE divider swapped" in rewires[0]
    # Two review segments: one per pass, each start paired with a done.
    assert len(_events_of("stage_start", events)) == 2
    assert len(_events_of("stage_done", events)) == 2
    # The pass-2 findings are what persists (they describe the fixed wiring).
    sj = json.loads(sp.read_text())
    assert [f["issue"] for f in sj["review_findings"]] == ["no TVS on USB"]
    assert sj["stage_status"]["electrical_review"]["cost_usd"] == 0.05


def test_skipped_review_closes_the_tab_without_durable_status(tmp_path, monkeypatch):
    monkeypatch.delenv("KICRAFT_ELECTRICAL_REVIEW", raising=False)
    sp = _write_state(tmp_path)
    monkeypatch.setattr(cli_app, "_maybe_electrical_review", lambda st, pd: {
        "ran": False, "blocked": False, "findings": [], "cost_usd": 0.0})
    events: list[dict] = []
    res = run_post_wiring_review(sp, tmp_path, events.append)
    assert res["ran"] is False
    # The tab opened and closed (no spinner left behind) ...
    assert len(_events_of("stage_start", events)) == 1
    assert len(_events_of("stage_done", events)) == 1
    # ... but no durable outcome is claimed for a review that never ran.
    sj = json.loads(sp.read_text())
    assert "electrical_review" not in (sj.get("stage_status") or {})


def test_env_opt_out_emits_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_ELECTRICAL_REVIEW", "0")
    sp = _write_state(tmp_path)
    events: list[dict] = []
    res = run_post_wiring_review(sp, tmp_path, events.append)
    assert res["ran"] is False and events == []


def test_review_crash_is_fail_soft(tmp_path, monkeypatch):
    monkeypatch.delenv("KICRAFT_ELECTRICAL_REVIEW", raising=False)
    sp = _write_state(tmp_path)

    def _boom(st, pd):
        raise RuntimeError("network down")

    monkeypatch.setattr(cli_app, "_maybe_electrical_review", _boom)
    events: list[dict] = []
    res = run_post_wiring_review(sp, tmp_path, events.append)  # must not raise
    assert res["ran"] is False
    # The open segment still closes so the live tab never spins forever.
    assert len(_events_of("stage_done", events)) == 1


def test_missing_state_is_a_silent_skip(tmp_path, monkeypatch):
    monkeypatch.delenv("KICRAFT_ELECTRICAL_REVIEW", raising=False)
    events: list[dict] = []
    res = run_post_wiring_review(tmp_path / "nope" / "state.json", tmp_path,
                                 events.append)
    assert res["ran"] is False and events == []
