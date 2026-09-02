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
