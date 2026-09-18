"""Tests for the resumable, non-mutating stage debug CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from kicraft.server import stage_driver


def _intent_candidate() -> dict:
    return {
        "goal": "USB-powered status LED.",
        "constraints": ["3.3 V logic"],
        "named_parts": [],
        "inferred_expertise": "intermediate",
        "assumptions": ["Rectangular board (defaulted)"],
        "project_stem": "STATUS_LED",
    }


def _pending_artifact(workspace: Path, slot: dict, basis_sha256: str = "absent") -> Path:
    path = workspace / ".kicraft" / "debug" / "intent.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "status": "needs_review",
                "stage": "intent",
                "basis_sha256": basis_sha256,
                "brief": "USB-powered status LED.\n",
                "instruction": None,
                "answers": [],
                "result": {
                    "stage": "intent",
                    "needs_review": True,
                    "commit_ok": False,
                    "slot": slot,
                    "diagnostics": [],
                    "cost_usd": 0.001,
                    "attempts": 1,
                    "rounds": None,
                    "tool_calls": None,
                    "wall_s": 0.1,
                    "cpu_s": 0.0,
                    "provider_ok": True,
                    "schema_ok": True,
                    "semantic_clean": True,
                    "repair_required": False,
                    "fab_safe": True,
                    "debug_context": {"raw_response": json.dumps(slot)},
                },
                "events": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_debug_draft_writes_complete_artifact_without_state(tmp_path, monkeypatch, capsys):
    brief = tmp_path / "brief.txt"
    brief.write_text("USB-powered status LED.\n", encoding="utf-8")
    slot = _intent_candidate()

    monkeypatch.setattr(stage_driver, "make_budget_client", lambda budget: object())

    def fake_drive(*args, progress, **kwargs):
        progress({"kind": "stage_start", "stage": "intent"})
        progress({"kind": "candidate_review", "stage": "intent", "attempt": 1})
        return {
            "stage": "intent",
            "needs_review": True,
            "commit_ok": False,
            "slot": slot,
            "diagnostics": [],
            "cost_usd": 0.001,
            "attempts": 1,
            "rounds": None,
            "tool_calls": None,
            "wall_s": 0.1,
            "cpu_s": 0.0,
            "provider_ok": True,
            "schema_ok": True,
            "semantic_clean": True,
            "repair_required": False,
            "fab_safe": True,
            "debug_context": {"raw_response": json.dumps(slot)},
        }

    monkeypatch.setattr(stage_driver, "drive_stage", fake_drive)
    rc = stage_driver.main(
        [
            "debug-draft",
            "--workspace",
            str(tmp_path),
            "--stage",
            "intent",
            "--brief-file",
            str(brief),
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "needs_review"
    artifact = json.loads(Path(summary["artifact"]).read_text(encoding="utf-8"))
    assert artifact["version"] == 1
    assert artifact["basis_sha256"] == "absent"
    assert artifact["brief"] == "USB-powered status LED.\n"
    assert artifact["result"]["slot"] == slot
    assert [event["kind"] for event in artifact["events"]] == [
        "stage_start",
        "candidate_review",
    ]
    assert not (tmp_path / ".kicraft" / "state.json").exists()


def test_debug_draft_requests_production_question_behaviour(tmp_path, monkeypatch):
    """debug-draft must keep production's auto-default while pausing before commit.

    Regression guard for the divergence where a debug walkthrough parked on a
    blocking question that a live kicraft.io run answers itself.
    """
    brief = tmp_path / "brief.txt"
    brief.write_text("A LoRa node.\n", encoding="utf-8")
    captured: dict = {}

    monkeypatch.setattr(stage_driver, "make_budget_client", lambda budget: object())

    def fake_drive(*args, progress, **kwargs):
        captured.update(kwargs)
        return {
            "stage": "intent",
            "needs_review": True,
            "commit_ok": False,
            "slot": _intent_candidate(),
            "diagnostics": [],
            "cost_usd": 0.0,
            "attempts": 1,
            "rounds": None,
            "tool_calls": None,
            "wall_s": 0.0,
            "cpu_s": 0.0,
            "provider_ok": True,
            "schema_ok": True,
        }

    monkeypatch.setattr(stage_driver, "drive_stage", fake_drive)
    rc = stage_driver.main(
        [
            "debug-draft",
            "--workspace",
            str(tmp_path),
            "--stage",
            "intent",
            "--brief-file",
            str(brief),
        ]
    )

    assert rc == 0
    assert captured["review_before_commit"] is True  # still pauses before commit
    assert captured["auto_default_questions"] is True  # but mirrors a live run


def test_debug_commit_rejects_stale_basis_without_writes(tmp_path, capsys):
    artifact_path = _pending_artifact(tmp_path, _intent_candidate())
    artifact_before = artifact_path.read_bytes()
    state_path = tmp_path / ".kicraft" / "state.json"
    state_path.write_text("{}\n", encoding="utf-8")
    state_before = state_path.read_bytes()
    history = tmp_path / "history.txt"
    history.write_text("Accepted the reviewed intent.\n", encoding="utf-8")

    rc = stage_driver.main(
        [
            "debug-commit",
            "--workspace",
            str(tmp_path),
            "--stage",
            "intent",
            "--history-message-file",
            str(history),
        ]
    )

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.err.strip() == "state changed since draft; re-run debug-draft"
    assert state_path.read_bytes() == state_before
    assert artifact_path.read_bytes() == artifact_before


def test_debug_commit_rejection_preserves_state_and_artifact(tmp_path, capsys):
    bad_slot = _intent_candidate() | {"inferred_expertise": "wizard"}
    artifact_path = _pending_artifact(tmp_path, bad_slot)
    artifact_before = artifact_path.read_bytes()
    history = tmp_path / "history.txt"
    history.write_text("Accepted the reviewed intent.\n", encoding="utf-8")

    rc = stage_driver.main(
        [
            "debug-commit",
            "--workspace",
            str(tmp_path),
            "--stage",
            "intent",
            "--history-message-file",
            str(history),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False
    assert payload["errors"]
    assert not (tmp_path / ".kicraft" / "state.json").exists()
    assert artifact_path.read_bytes() == artifact_before


def test_debug_commit_accepts_exact_candidate_and_finalizes_trace(tmp_path, capsys):
    slot = _intent_candidate()
    artifact_path = _pending_artifact(tmp_path, slot)
    history = tmp_path / "history.txt"
    history.write_text("Accepted the reviewed USB status LED intent.\n", encoding="utf-8")

    rc = stage_driver.main(
        [
            "debug-commit",
            "--workspace",
            str(tmp_path),
            "--stage",
            "intent",
            "--history-message-file",
            str(history),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0, payload
    assert payload["invalidated_stages"] == [
        "functional_spec",
        "architecture",
        "bom",
        "wiring",
    ]
    state_path = tmp_path / ".kicraft" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert all(state["intent"][key] == value for key, value in slot.items() if key != "project_stem")
    assert state["project_stem"] == slot["project_stem"]
    assert state["functional_spec"] is None
    assert state["architecture"] is None
    assert state["bom"] is None
    assert state["history"][-1]["content"] == "Accepted the reviewed USB status LED intent."
    assert state["stage_status"]["intent"]["cost_usd"] == 0.001

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["status"] == "accepted"
    assert artifact["commit"]["invalidated_stages"] == payload["invalidated_stages"]
    assert artifact["accepted_state_sha256"] == hashlib.sha256(state_path.read_bytes()).hexdigest()


def test_debug_draft_walks_the_architecture_section_rung_without_a_campaign(tmp_path, monkeypatch, capsys):
    """`debug-draft --stage architecture` drives the same sub-step sequence as a live run.

    A single brief is stepped through the refused whole-slot draft and the section call that repairs
    it, so the operator sees the section the rung asked for in the artifact's own event stream.
    """
    from test_architecture_intent import _half_usb_pair_intent, _hub75_intent
    from test_stage_driver_retry import _OK_INTENT, _ScriptedClient

    from kicraft.server.config import Settings
    from kicraft.server.session import run_session

    def reply(payload: dict) -> dict:
        return {
            "text": json.dumps(payload),
            "reasoning": "",
            "finish_reason": "stop",
            "cost_usd": 0.0,
        }

    brief_text = "a USB-C ESP32-S3 HUB75 controller"
    brief = tmp_path / "brief.txt"
    brief.write_text(brief_text + "\n", encoding="utf-8")

    intent_client = _ScriptedClient([reply(json.loads(_OK_INTENT))])
    intent_client.s = Settings(api_key="test")
    assert run_session(tmp_path, brief_text, ["intent"], client=intent_client)["status"] == "ok"

    signals = [dict(row) for row in _hub75_intent()["signals"]]
    signals.append({"name": "USB_DATA_TEST", "from": "esp32.usb_dm", "to": "usb_test.pin1"})
    client = _ScriptedClient([reply(_half_usb_pair_intent()), reply({"signals": signals})])
    client.s = Settings(api_key="test", architecture_sections=True)
    monkeypatch.setattr(stage_driver, "make_budget_client", lambda budget: client)

    rc = stage_driver.main(
        [
            "debug-draft",
            "--workspace",
            str(tmp_path),
            "--stage",
            "architecture",
            "--brief-file",
            str(brief),
        ]
    )

    assert rc == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["status"] == "needs_review"
    artifact = json.loads(Path(summary["artifact"]).read_text(encoding="utf-8"))
    assert artifact["result"]["attempts"] == 2
    sections = [event for event in artifact["events"] if event["kind"] == "architecture_section"]
    assert [(event["section"], event["accepted"]) for event in sections] == [("signals", True)]
    refusals = [
        event
        for event in artifact["events"]
        if event["kind"] == "retry" and event.get("call_mode") == "architecture_section"
    ]
    assert refusals and refusals[0]["diagnostic"]["code"] == "incomplete_usb_edge"
    state = json.loads((tmp_path / ".kicraft" / "state.json").read_text(encoding="utf-8"))
    assert state["intent"] is not None and state["architecture"] is None  # reviews before commit
