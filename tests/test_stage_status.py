"""Tests for durable per-stage outcomes (state.json's stage_status block) and
the status derivation that restores the GUI's stage tabs on a reopened project.

Mostly pure functions + local file ops; one test drives the real stage-commit
CLI subprocess (offline, deterministic) to prove the load/validate/dump
round-trip preserves the block.
"""
from __future__ import annotations

import json
from pathlib import Path

from kicraft.design.models import ConversationState
from kicraft.server.session import (
    BUILD_PHASES,
    derive_stage_statuses,
    null_downstream,
    read_state,
)
from kicraft.server.stage_driver import DESIGN_STAGES, _commit, _stamp_stage_status


def _write_state(ws: Path, data: dict) -> None:
    (ws / ".kicraft").mkdir(parents=True, exist_ok=True)
    (ws / ".kicraft" / "state.json").write_text(json.dumps(data), encoding="utf-8")


def _complete_design() -> dict:
    return {"stage_status": {s: {"ok": True} for s in DESIGN_STAGES}}


# ---- derive_stage_statuses: design stages -----------------------------------

def test_empty_state_is_all_pending():
    out = derive_stage_statuses({})
    assert set(out) == {*DESIGN_STAGES, *BUILD_PHASES}
    assert set(out.values()) == {"pending"}


def test_stage_status_block_wins():
    st = {"stage_status": {"intent": {"ok": True, "cost_usd": 0.01},
                           "functional_spec": {"ok": False}}}
    out = derive_stage_statuses(st)
    assert out["intent"] == "done"
    assert out["functional_spec"] == "failed"
    assert out["architecture"] == "pending"


def test_legacy_slot_presence_fallback():
    # Projects persisted before stage_status existed: slot presence marks done,
    # wiring via the bom connections it populates.
    st = {"intent": {}, "functional_spec": {}, "architecture": {},
          "bom": {"parts": [{}], "connections": [{"net_name": "V"}]}}
    out = derive_stage_statuses(st)
    assert all(out[s] == "done" for s in DESIGN_STAGES)


def test_unanswered_question_parks_its_stage():
    st = {"intent": {"goal": "x"},
          "open_questions": [{"text": "?", "stage": "bom", "answer": None}]}
    out = derive_stage_statuses(st)
    assert out["bom"] == "parked"
    assert out["intent"] == "done"


def test_answered_question_does_not_park():
    st = {"open_questions": [{"text": "?", "stage": "bom", "answer": "usb"}]}
    assert derive_stage_statuses(st)["bom"] == "pending"


# ---- derive_stage_statuses: build phases ------------------------------------

def test_build_phases_done_from_artifacts():
    out = derive_stage_statuses(_complete_design(), project_status="ok",
                                sheets_exist=True, pcb_ready=True, zip_ok=True)
    assert all(out[p] == "done" for p in BUILD_PHASES)


def test_failure_localizes_to_place_route():
    out = derive_stage_statuses(_complete_design(), project_status="failed",
                                sheets_exist=True)
    assert out["synthesize"] == "done"
    assert out["place_route"] == "failed"
    assert out["fab"] == "pending"


def test_failure_localizes_to_synthesize_on_failed_checks():
    out = derive_stage_statuses(_complete_design(), project_status="failed",
                                sheets_exist=True, synth_checks_failed=True)
    assert out["synthesize"] == "failed"
    assert out["place_route"] == "pending"
    assert out["fab"] == "pending"


def test_failure_localizes_to_fab():
    out = derive_stage_statuses(_complete_design(), project_status="failed",
                                sheets_exist=True, pcb_ready=True)
    assert out["synthesize"] == "done"
    assert out["place_route"] == "done"
    assert out["fab"] == "failed"


def test_failed_build_outranks_stale_zip():
    # A failed (re)build keeps the failed candidate board; a zip surviving
    # from an earlier successful build is stale and must NOT turn fab green.
    out = derive_stage_statuses(_complete_design(), project_status="failed",
                                sheets_exist=True, pcb_ready=True, zip_ok=True)
    assert out["fab"] == "failed"


def test_stale_artifacts_ignored_when_design_incomplete():
    # An edit nulled downstream slots; the generated tree from the earlier
    # build still exists but must not paint the build phases done.
    st = {"stage_status": {"intent": {"ok": True}}}
    out = derive_stage_statuses(st, project_status="ok", sheets_exist=True,
                                pcb_ready=True, zip_ok=True)
    assert all(out[p] == "pending" for p in BUILD_PHASES)


# ---- _stamp_stage_status -----------------------------------------------------

def test_stamp_creates_missing_state(tmp_path):
    sp = tmp_path / ".kicraft" / "state.json"
    _stamp_stage_status(sp, "intent", False)
    sj = json.loads(sp.read_text())
    assert sj["stage_status"]["intent"]["ok"] is False
    assert sj["stage_status"]["intent"]["finished_at"]


def test_stamp_preserves_other_state_and_entries(tmp_path):
    sp = tmp_path / ".kicraft" / "state.json"
    sp.parent.mkdir(parents=True)
    sp.write_text(json.dumps({"project_stem": "X",
                              "stage_status": {"intent": {"ok": True}}}))
    _stamp_stage_status(sp, "bom", True, cost_usd=0.123456789, attempts=2)
    sj = json.loads(sp.read_text())
    assert sj["project_stem"] == "X"
    assert sj["stage_status"]["intent"]["ok"] is True
    entry = sj["stage_status"]["bom"]
    assert entry["ok"] is True
    assert entry["attempts"] == 2
    assert entry["cost_usd"] == 0.123457  # rounded to 6 places


# ---- persistence round-trips ---------------------------------------------------

def test_conversation_state_roundtrips_stage_status():
    cs = ConversationState.model_validate(
        {"stage_status": {"intent": {"ok": True, "cost_usd": 0.01, "attempts": 1}}})
    dumped = json.loads(cs.model_dump_json())
    assert dumped["stage_status"]["intent"]["ok"] is True


def test_cli_commit_preserves_stage_status(tmp_path):
    # The stage-commit CLI loads/validates/dumps the whole state; a block
    # stamped before the commit must survive that rewrite.
    sp = tmp_path / ".kicraft" / "state.json"
    _stamp_stage_status(sp, "functional_spec", False)
    slot = {"goal": "a USB-powered LED", "constraints": [], "named_parts": [],
            "inferred_expertise": "intermediate", "assumptions": []}
    ok, out = _commit("intent", slot, sp, "a USB-powered LED", "USB_LED", tmp_path)
    assert ok, out
    sj = json.loads(sp.read_text())
    assert sj["intent"]["goal"] == "a USB-powered LED"
    assert sj["stage_status"]["functional_spec"]["ok"] is False  # preserved


# ---- null_downstream drops stale outcomes ------------------------------------

def test_null_downstream_drops_stale_stage_status(tmp_path):
    _write_state(tmp_path, {
        "intent": {}, "functional_spec": {}, "architecture": {},
        "bom": {"parts": [], "connections": []},
        "stage_status": {s: {"ok": True} for s in DESIGN_STAGES}})
    null_downstream(tmp_path, "functional_spec")
    ss = read_state(tmp_path)["stage_status"]
    assert set(ss) == {"intent", "functional_spec"}
