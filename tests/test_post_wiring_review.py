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
    return ConversationState.model_validate({
        "review_coverage": {
            "source_fingerprint": "0" * 64, "expected_refs": ["U1"], "included_refs": ["U1"],
            "expected_nets": ["+3V3"], "included_nets": ["+3V3"],
            "omitted_mandatory": [], "complete": True,
        },
        "bom": {
            "parts": [{"ref": "U1", "value": "x", "symbol": "Fake:X",
                       "footprint": "Package_SO:SOIC-8", "sheet": "A"}],
            "connections": [{"net_name": "+3V3", "sheet": "A",
                             "endpoints": [{"ref": "U1", "pin": "1"}]}],
        },
    })


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
    monkeypatch.setattr(cli_app, "_maybe_electrical_review", lambda st, pd, **_: {
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
                        lambda st, pd, **_: next(passes))
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


def test_skipped_review_persists_incomplete_status(tmp_path, monkeypatch):
    monkeypatch.delenv("KICRAFT_ELECTRICAL_REVIEW", raising=False)
    sp = _write_state(tmp_path)
    monkeypatch.setattr(cli_app, "_maybe_electrical_review", lambda st, pd, **_: {
        "ran": False, "blocked": False, "findings": [], "cost_usd": 0.0})
    events: list[dict] = []
    res = run_post_wiring_review(sp, tmp_path, events.append)
    assert res["ran"] is False
    # The tab opened and closed (no spinner left behind) ...
    assert len(_events_of("stage_start", events)) == 1
    assert len(_events_of("stage_done", events)) == 1
    sj = json.loads(sp.read_text())
    status = sj["stage_status"]["electrical_review"]
    assert status["ok"] is False and status["semantic_clean"] is None
    assert status["diagnostics"][0]["code"] == "electrical_review_incomplete"


def test_env_opt_out_persists_unreviewed(tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_ELECTRICAL_REVIEW", "0")
    sp = _write_state(tmp_path)
    events: list[dict] = []
    res = run_post_wiring_review(sp, tmp_path, events.append)
    assert res["ran"] is False
    saved = json.loads(sp.read_text())
    assert saved["stage_status"]["electrical_review"]["ok"] is False
    assert saved["stage_status"]["electrical_review"]["semantic_clean"] is None
    assert _events_of("stage_done", events)[0]["ok"] is False


def test_review_crash_is_fail_soft(tmp_path, monkeypatch):
    monkeypatch.delenv("KICRAFT_ELECTRICAL_REVIEW", raising=False)
    sp = _write_state(tmp_path)

    def _boom(st, pd, **_):
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


def test_review_blocker_scope_follows_the_findings_owner(tmp_path, monkeypatch):
    """A review blocker is repaired at the owning stage, not always at wiring."""
    monkeypatch.delenv("KICRAFT_ELECTRICAL_REVIEW", raising=False)

    def _run(blocker, name):
        case_dir = tmp_path / name
        case_dir.mkdir()
        sp = _write_state(case_dir)
        passes = iter([
            {"ran": True, "blocked": True, "findings": [blocker], "cost_usd": 0.01},
            {"ran": True, "blocked": False, "findings": [], "cost_usd": 0.01},
        ])
        monkeypatch.setattr(cli_app, "_maybe_electrical_review", lambda st, pd, **_: next(passes))
        calls: list[tuple[str, list[str]]] = []

        def rewire(instruction: str, stages=None) -> None:
            calls.append((instruction, list(stages or [])))

        run_post_wiring_review(sp, case_dir, lambda event: None, rewire)
        return calls[0]

    instruction, stages = _run(
        {"severity": "blocker", "area": "bom", "issue": "C1 value too small",
         "suggestion": "use 10uF"},
        "bom_case",
    )
    assert stages == ["bom", "wiring"]
    assert "bom" in instruction

    instruction, stages = _run(
        {"severity": "blocker", "area": "wiring", "issue": "VBUS net missing",
         "suggestion": "connect it"},
        "wiring_case",
    )
    assert stages == ["wiring"]

    instruction, stages = _run(
        {"severity": "blocker", "area": "topology", "issue": "power tree broken",
         "suggestion": "re-derive"},
        "topology_case",
    )
    assert stages == ["architecture", "bom", "wiring"]


def test_legacy_one_argument_rewire_still_works(tmp_path, monkeypatch):
    monkeypatch.delenv("KICRAFT_ELECTRICAL_REVIEW", raising=False)
    sp = _write_state(tmp_path)
    passes = iter([
        {"ran": True, "blocked": True, "findings": [BLOCKER], "cost_usd": 0.01},
        {"ran": True, "blocked": False, "findings": [], "cost_usd": 0.01},
    ])
    monkeypatch.setattr(cli_app, "_maybe_electrical_review", lambda st, pd, **_: next(passes))
    calls: list[str] = []
    run_post_wiring_review(sp, tmp_path, lambda event: None, calls.append)
    assert len(calls) == 1 and "VSENSE divider swapped" in calls[0]


def test_lifecycle_records_execution_mode_and_buckets_spend_separately(tmp_path, monkeypatch):
    """Batch and web share one lifecycle; the mode is durable provenance and each
    phase's spend is its own bucket, so review/silk cost can never be read as
    designer spend or as fulfillment evidence."""
    sp = _write_state(tmp_path)
    monkeypatch.setattr(cli_app, "_maybe_electrical_review", lambda st, pd, **_: {
        "ran": True, "blocked": False, "findings": [WARNING], "cost_usd": 0.02})
    monkeypatch.setattr(cli_app, "run_silk_plan_authoring", lambda *a, **kw: {
        "ran": True, "cost_usd": 0.03})

    lifecycle = cli_app.run_post_wiring_lifecycle(
        sp, tmp_path, lambda event: None, execution_mode="batch"
    )
    assert lifecycle["execution_mode"] == "batch"
    assert lifecycle["post_wiring_review"] == {"status": "completed", "ran": True, "cost_usd": 0.02}
    assert lifecycle["silkscreen"] == {"status": "completed", "ran": True, "cost_usd": 0.03}
    # Fail-soft: an opt-out review still reports its phase separately.
    monkeypatch.setattr(cli_app, "_maybe_electrical_review", lambda st, pd, **_: {
        "ran": False, "blocked": False, "findings": [], "cost_usd": 0.0})
    lifecycle = cli_app.run_post_wiring_lifecycle(sp, tmp_path, lambda event: None)
    assert lifecycle["execution_mode"] == "web"
    assert lifecycle["post_wiring_review"]["ran"] is False
    assert lifecycle["post_wiring_review"]["cost_usd"] == 0.0


def test_wiring_replacement_invalidates_previous_clean_review():
    state = _state()
    state.stage_status["electrical_review"] = cli_app.StageStatus(
        ok=True, provider_ok=True, schema_ok=True, semantic_clean=True,
    )
    cli_app._apply_slot(state, "wiring", {
        "connections": [{"net_name": "GND", "sheet": "A",
                         "endpoints": [{"ref": "U1", "pin": "1"}]}],
    }, project_stem=None)
    assert state.review_coverage is None
    assert "electrical_review" not in state.stage_status
