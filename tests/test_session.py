"""Tests for the resumable-session helpers: stage math and state.json edits.

Pure functions + local state.json file ops (no OpenRouter, no pcbnew), so fast.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.server.session import (
    downstream_stages,
    null_downstream,
    read_state,
    record_answers,
    remaining_stages,
    run_session,
)


def _write_state(ws: Path, data: dict) -> None:
    (ws / ".kicraft").mkdir(parents=True, exist_ok=True)
    (ws / ".kicraft" / "state.json").write_text(json.dumps(data), encoding="utf-8")


# ---- remaining_stages (resume math) ---------------------------------------

def test_remaining_stages_from_scratch():
    assert remaining_stages({}) == ["intent", "functional_spec", "architecture", "bom", "wiring"]


def test_remaining_stages_after_intent():
    assert remaining_stages({"intent": {"goal": "x"}}) == \
        ["functional_spec", "architecture", "bom", "wiring"]


def test_remaining_stages_wiring_only_when_bom_has_no_connections():
    st = {"intent": {}, "functional_spec": {}, "architecture": {}, "bom": {"parts": [{}]}}
    assert remaining_stages(st) == ["wiring"]  # bom present but wiring not done


def test_remaining_stages_complete_when_bom_wired():
    st = {"intent": {}, "functional_spec": {}, "architecture": {},
          "bom": {"parts": [{}], "connections": [{"net_name": "VBUS"}]}}
    assert remaining_stages(st) == []


def test_remaining_stages_hole_reruns_the_tail():
    st = {"intent": {}, "functional_spec": {}, "bom": {"parts": []}}  # architecture missing
    assert remaining_stages(st) == ["architecture", "bom", "wiring"]


# ---- downstream_stages (edit invalidation) --------------------------------

def test_downstream_stages():
    assert downstream_stages("architecture") == ["bom", "wiring"]
    assert downstream_stages("intent") == ["functional_spec", "architecture", "bom", "wiring"]
    assert downstream_stages("wiring") == []
    assert downstream_stages("nope") == []


# ---- null_downstream (edit-and-rerun consistency) -------------------------

def test_null_downstream_clears_slots_and_questions(tmp_path):
    _write_state(tmp_path, {
        "intent": {"goal": "x"}, "functional_spec": {"blocks": []},
        "architecture": {"sheets": []}, "bom": {"parts": [{"ref": "R1"}]},
        "open_questions": [{"text": "q", "stage": "bom"}, {"text": "k", "stage": "intent"}],
    })
    cleared = null_downstream(tmp_path, "functional_spec")
    assert cleared == ["architecture", "bom", "wiring"]
    sj = read_state(tmp_path)
    assert sj["architecture"] is None and sj["bom"] is None  # downstream nulled
    assert sj["intent"] == {"goal": "x"}                     # upstream untouched
    assert [q["stage"] for q in sj["open_questions"]] == ["intent"]  # downstream qs dropped


def test_null_downstream_of_bom_clears_only_wiring(tmp_path):
    _write_state(tmp_path, {
        "bom": {"parts": [{"ref": "R1"}], "connections": [{"net_name": "N"}],
                "no_connect_pins": [{"ref": "R1", "pin": "1"}]}})
    assert null_downstream(tmp_path, "bom") == ["wiring"]
    sj = read_state(tmp_path)
    assert sj["bom"]["parts"] == [{"ref": "R1"}]          # bom parts kept
    assert sj["bom"]["connections"] == []                # wiring data cleared
    assert sj["bom"]["no_connect_pins"] == []


# ---- record_answers + read_state ------------------------------------------

def test_record_answers_stamps_matching_stage(tmp_path):
    _write_state(tmp_path, {"open_questions": [
        {"text": "Battery?", "stage": "intent", "answer": None},
        {"text": "Other", "stage": "architecture", "answer": None}]})
    record_answers(tmp_path, "intent", [{"text": "Battery?", "answer": "LiPo 1S"}])
    qs = {q["text"]: q.get("answer") for q in read_state(tmp_path)["open_questions"]}
    assert qs["Battery?"] == "LiPo 1S"  # stamped
    assert qs["Other"] is None          # different stage untouched


def test_read_state_missing_is_empty(tmp_path):
    assert read_state(tmp_path) == {}




# ---- park -> answer -> resume, end to end with a fake LLM (no network) ------
#
# Drives the real stage-prep / stage-commit CLI subprocesses (offline,
# deterministic) but swaps the OpenRouter client for a canned-reply fake, so the
# whole question-park-and-resume loop is exercised without spending tokens.

class _FakeGuard:
    def status(self):
        return {"spent_total_usd": 0.0, "daily_remaining_usd": 5.0, "daily_ceiling_usd": 5.0}


class _FakeClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self.guard = _FakeGuard()

    def chat(self, messages, max_tokens=4096, temperature=0.2, progress=None, meta_ctx=None,
             reasoning=None):
        return {"text": self._replies.pop(0), "cost_usd": 0.0, "reasoning": "",
                "finish_reason": "stop"}


def test_run_session_parks_on_question_then_resumes(tmp_path):
    brief = "a USB-powered LED"
    q_reply = json.dumps({"questions": [
        {"text": "Battery?", "options": ["LiPo"], "blocking": True}]})
    intent_slot = json.dumps({
        "goal": "a USB-powered LED", "constraints": [], "named_parts": [],
        "inferred_expertise": "intermediate", "assumptions": [], "project_stem": "USB_LED"})

    # 1) the model asks a blocking question -> the session parks
    res = run_session(tmp_path, brief, ["intent"], client=_FakeClient([q_reply]))
    assert res["status"] == "awaiting_input"
    assert res["questions"][0]["text"] == "Battery?"
    assert read_state(tmp_path)["open_questions"][0]["text"] == "Battery?"  # persisted

    # 2) the user answers -> the stage commits (answers suppress re-asking)
    res2 = run_session(tmp_path, brief, ["intent"],
                       answers=[{"text": "Battery?", "answer": "LiPo 1S"}],
                       client=_FakeClient([intent_slot]))
    assert res2["status"] == "ok"
    assert read_state(tmp_path)["intent"]["goal"] == "a USB-powered LED"
    # The driver stamps the durable outcome the GUI restores stage tabs from.
    assert read_state(tmp_path)["stage_status"]["intent"]["ok"] is True


# ---- derive_stage_statuses: electrical_review tab -----------------------------

from kicraft.server.session import derive_stage_statuses


def _er_state(ok=True, findings=None, where="top"):
    st = {"stage_status": {"electrical_review": {"ok": ok, "cost_usd": 0.05}}}
    if findings is not None:
        if where == "top":
            st["review_findings"] = findings
        else:  # legacy build-tail location
            st["artifacts"] = {"review_findings": findings}
    return st


def test_electrical_review_done_when_ran_clean():
    out = derive_stage_statuses(_er_state(findings=[]))
    assert out["electrical_review"] == "done"


def test_electrical_review_warning_on_uncleared_blocker():
    out = derive_stage_statuses(_er_state(findings=[
        {"severity": "blocker", "issue": "VSENSE divider swapped"}]))
    assert out["electrical_review"] == "warning"


def test_electrical_review_done_with_only_warnings():
    out = derive_stage_statuses(_er_state(findings=[
        {"severity": "warning", "issue": "no TVS"}]))
    assert out["electrical_review"] == "done"


def test_electrical_review_legacy_artifacts_findings():
    out = derive_stage_statuses(_er_state(findings=[
        {"severity": "blocker", "issue": "x"}], where="artifacts"))
    assert out["electrical_review"] == "warning"


def test_electrical_review_pending_without_stage_status():
    # Pre-R3 projects / skipped reviews claim nothing.
    assert derive_stage_statuses({})["electrical_review"] == "pending"


def test_electrical_review_failed_status():
    assert derive_stage_statuses(_er_state(ok=False))["electrical_review"] == "failed"
