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
    assert remaining_stages({"intent": {"goal": "x"}}) == [
        "functional_spec",
        "architecture",
        "bom",
        "wiring",
    ]


def test_remaining_stages_wiring_only_when_bom_has_no_connections():
    st = {"intent": {}, "functional_spec": {}, "architecture": {}, "bom": {"parts": [{}]}}
    assert remaining_stages(st) == ["wiring"]  # bom present but wiring not done


def test_remaining_stages_complete_when_bom_wired():
    st = {
        "intent": {},
        "functional_spec": {},
        "architecture": {},
        "bom": {"parts": [{}], "connections": [{"net_name": "VBUS"}]},
    }
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
    _write_state(
        tmp_path,
        {
            "intent": {"goal": "x"},
            "functional_spec": {"blocks": []},
            "architecture": {"sheets": []},
            "bom": {"parts": [{"ref": "R1"}]},
            "open_questions": [{"text": "q", "stage": "bom"}, {"text": "k", "stage": "intent"}],
        },
    )
    cleared = null_downstream(tmp_path, "functional_spec")
    assert cleared == ["architecture", "bom", "wiring"]
    sj = read_state(tmp_path)
    assert sj["architecture"] is None and sj["bom"] is None  # downstream nulled
    assert sj["intent"] == {"goal": "x"}  # upstream untouched
    assert [q["stage"] for q in sj["open_questions"]] == ["intent"]  # downstream qs dropped


def test_null_downstream_of_bom_clears_only_wiring(tmp_path):
    _write_state(
        tmp_path,
        {
            "bom": {
                "parts": [{"ref": "R1"}],
                "connections": [{"net_name": "N"}],
                "no_connect_pins": [{"ref": "R1", "pin": "1"}],
            }
        },
    )
    assert null_downstream(tmp_path, "bom") == ["wiring"]
    sj = read_state(tmp_path)
    assert sj["bom"]["parts"] == [{"ref": "R1"}]  # bom parts kept
    assert sj["bom"]["connections"] == []  # wiring data cleared
    assert sj["bom"]["no_connect_pins"] == []


def test_bom_invalidation_preserves_compiler_wires_but_requires_wiring(tmp_path):
    state = {
        "intent": {}, "functional_spec": {}, "architecture": {},
        "bom": {
            "parts": [{"ref": "U1"}, {"ref": "R1"}],
            "connections": [
                {"net_name": "VCC", "endpoints": [
                    {"ref": "U1", "pin": "1"}, {"ref": "R1", "pin": "1"}
                ]},
            ],
            "lowerer_ownership": [{"ref": "U1", "pin": "1", "net": "VCC"}],
        },
        "stage_status": {"wiring": {"ok": True}},
    }
    _write_state(tmp_path, state)
    assert null_downstream(tmp_path, "bom") == ["wiring"]
    saved = read_state(tmp_path)
    assert saved["bom"]["connections"] == [
        {"net_name": "VCC", "endpoints": [{"ref": "U1", "pin": "1"}]}
    ]
    assert saved["wiring_complete"] is False
    assert remaining_stages(saved) == ["wiring"]
    assert derive_stage_statuses(saved)["wiring"] == "pending"
    assert derive_stage_statuses(saved, sheets_exist=True, pcb_ready=True, zip_ok=True)["fab"] == "pending"


def test_legacy_and_explicit_wiring_completion():
    state = {
        "intent": {}, "functional_spec": {}, "architecture": {},
        "bom": {"connections": [{"net_name": "VCC", "endpoints": [{"ref": "U1", "pin": "1"}]}]},
    }
    # Legacy projects without a marker or status retain their historical meaning.
    assert remaining_stages(state) == []
    assert derive_stage_statuses(state)["wiring"] == "done"
    state["wiring_complete"] = False
    state["stage_status"] = {"wiring": {"ok": True}}
    assert remaining_stages(state) == ["wiring"]
    assert derive_stage_statuses(state)["wiring"] == "pending"
    state["wiring_complete"] = True
    assert remaining_stages(state) == []
    assert derive_stage_statuses(state)["wiring"] == "done"
    state["stage_status"]["wiring"] = {"ok": False}
    assert remaining_stages(state) == ["wiring"]
    assert derive_stage_statuses(state)["wiring"] == "failed"


def test_legacy_compiler_only_graph_needs_committed_wiring_status():
    state = {
        "intent": {}, "functional_spec": {}, "architecture": {},
        "bom": {
            "parts": [{"ref": "U1"}, {"ref": "R1"}],
            "connections": [{"net_name": "VCC", "endpoints": [{"ref": "U1", "pin": "1"}]}],
            "lowerer_ownership": [{"ref": "U1", "pin": "1", "net": "VCC"}],
        },
    }
    assert remaining_stages(state) == ["wiring"]
    assert derive_stage_statuses(state)["wiring"] == "pending"
    # The same compiler-only graph may be complete; only its durable historical
    # commit status distinguishes it from a partial graph in a legacy file.
    state["stage_status"] = {"wiring": {"ok": True}}
    assert remaining_stages(state) == []
    assert derive_stage_statuses(state)["wiring"] == "done"


@pytest.mark.parametrize("recipe", [
    "rp2040-minimal@1", "rp2040-minimal@2", "drv8833-dual-motor@1",
    "tps54331-adjustable@1", "tps54331-12v-1a@1",
])
@pytest.mark.parametrize("ownership", ["manifest", "part"])
def test_legacy_recipe_bom_requires_regeneration(recipe, ownership):
    bom = {"parts": [{"ref": "U1"}], "connections": [{"net_name": "V"}]}
    if ownership == "manifest":
        bom["recipe_ownership"] = [{"recipe": recipe}]
    else:
        bom["parts"][0]["recipe_id"] = recipe
    state = {
        "intent": {}, "functional_spec": {}, "architecture": {}, "bom": bom,
        "wiring_complete": True, "stage_status": {"bom": {"ok": True}, "wiring": {"ok": True}},
    }
    assert remaining_stages(state) == ["bom", "wiring"]
    statuses = derive_stage_statuses(state, sheets_exist=True, pcb_ready=True, zip_ok=True)
    assert statuses["architecture"] == "done"
    assert statuses["bom"] == statuses["wiring"] == statuses["fab"] == "pending"


@pytest.mark.parametrize("ambiguous", [False, True])
def test_resume_discards_legacy_derived_state_before_driver(tmp_path, monkeypatch, ambiguous):
    from kicraft.server import session

    selection = {
        "recipe": "rp2040-minimal@2", "instance": "core", "sheets": {},
        "parameters": {} if ambiguous else {"native_usb": True},
    }
    original = {
        "intent": {"goal": "keep"}, "functional_spec": {"blocks": []},
        "architecture": {"recipe_selections": [selection]},
        "bom": {"parts": [{"ref": "U1", "recipe_id": "rp2040-minimal@2"}]},
        "wiring_complete": True,
        "stage_status": {"architecture": {"ok": True}, "bom": {"ok": True}, "wiring": {"ok": True}},
    }
    _write_state(tmp_path, original)
    expected = ["architecture", "bom", "wiring"] if ambiguous else ["bom", "wiring"]
    assert remaining_stages(original) == expected
    seen = []

    def stop_at_driver(stages, brief, workspace, **kwargs):
        seen.extend(stages)
        return [], None, workspace / ".kicraft" / "state.json"

    monkeypatch.setattr(session, "drive_chain", stop_at_driver)
    run_session(tmp_path, "", ["wiring"])
    saved = read_state(tmp_path)
    assert seen == expected
    assert saved["intent"] == original["intent"]
    assert saved["functional_spec"] == original["functional_spec"]
    assert saved["bom"] is None
    assert saved["wiring_complete"] is False
    assert saved["architecture"] == (None if ambiguous else original["architecture"])



@pytest.mark.parametrize("contents", ["{broken", "[]"])
@pytest.mark.parametrize("update", ["status", "questions"])
def test_state_updates_refuse_to_overwrite_unreadable_existing_state(tmp_path, contents, update):
    from kicraft.server.stage_state_io import attach_questions, stamp_stage_status

    path = tmp_path / "state.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(RuntimeError):
        if update == "status":
            stamp_stage_status(path, "bom", False)
        else:
            attach_questions(path, "bom", [{"stage": "bom", "text": "Question?"}])
    assert path.read_text(encoding="utf-8") == contents


@pytest.mark.parametrize("contents", ["{broken", "[]"])
def test_build_recovery_refuses_to_overwrite_unreadable_state(tmp_path, contents):
    from kicraft.server.session import write_build_recovery

    path = tmp_path / ".kicraft" / "state.json"
    path.parent.mkdir()
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(RuntimeError):
        write_build_recovery(tmp_path, ok=False)
    assert path.read_text(encoding="utf-8") == contents

# ---- record_answers + read_state ------------------------------------------


def test_record_answers_stamps_matching_stage(tmp_path):
    _write_state(
        tmp_path,
        {
            "open_questions": [
                {"text": "Battery?", "stage": "intent", "answer": None},
                {"text": "Other", "stage": "architecture", "answer": None},
            ]
        },
    )
    record_answers(tmp_path, "intent", [{"text": "Battery?", "answer": "LiPo 1S"}])
    qs = {q["text"]: q.get("answer") for q in read_state(tmp_path)["open_questions"]}
    assert qs["Battery?"] == "LiPo 1S"  # stamped
    assert qs["Other"] is None  # different stage untouched


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
        return {
            "text": self._replies.pop(0),
            "cost_usd": 0.0,
            "reasoning": "",
            "finish_reason": "stop",
        }


def test_run_session_defaults_direct_question_without_parking(tmp_path):
    brief = "a USB-powered LED"
    question = json.dumps(
        {
            "questions": [
                {
                    "text": "Which battery chemistry?",
                    "stage": "intent",
                    "options": ["LiPo", "Alkaline"],
                    "blocking": True,
                }
            ]
        }
    )
    intent_slot = json.dumps(
        {
            "goal": "a USB-powered LED",
            "constraints": [],
            "named_parts": [],
            "inferred_expertise": "intermediate",
            "assumptions": ["Alkaline battery selected (defaulted)"],
            "project_stem": "USB_LED",
        }
    )
    client = _FakeClient([question, intent_slot])

    result = run_session(tmp_path, brief, ["intent"], client=client)

    assert result["status"] == "ok"
    state = read_state(tmp_path)
    assert state["intent"]["goal"] == brief
    assert state["open_questions"] == []
    assert state["stage_status"]["intent"]["ok"] is True


def test_run_session_preserves_explicit_interactive_question_policy(tmp_path):
    question = json.dumps(
        {
            "questions": [
                {
                    "text": "Which battery chemistry?",
                    "stage": "intent",
                    "options": ["LiPo", "Alkaline"],
                    "blocking": True,
                }
            ]
        }
    )

    result = run_session(
        tmp_path,
        "a USB-powered LED",
        ["intent"],
        client=_FakeClient([question]),
        auto_default_questions=False,
    )

    assert result["status"] == "awaiting_input"
    assert result["questions"][0]["options"] == ["LiPo", "Alkaline"]


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
    out = derive_stage_statuses(
        _er_state(findings=[{"severity": "blocker", "issue": "VSENSE divider swapped"}])
    )
    assert out["electrical_review"] == "warning"


def test_electrical_review_done_with_only_warnings():
    out = derive_stage_statuses(_er_state(findings=[{"severity": "warning", "issue": "no TVS"}]))
    assert out["electrical_review"] == "done"


def test_electrical_review_legacy_artifacts_findings():
    out = derive_stage_statuses(
        _er_state(findings=[{"severity": "blocker", "issue": "x"}], where="artifacts")
    )
    assert out["electrical_review"] == "warning"


def test_electrical_review_pending_without_stage_status():
    # Pre-R3 projects / skipped reviews claim nothing.
    assert derive_stage_statuses({})["electrical_review"] == "pending"


def test_electrical_review_failed_status():
    assert derive_stage_statuses(_er_state(ok=False))["electrical_review"] == "failed"
