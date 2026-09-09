"""The reopen path loads events.jsonl back into state['events'] (web._load_events).

events.jsonl is persisted at finalize but was historically never read back, so a
reopened project showed a blank build timeline / reasoning panel. Pure-data helper:
no NiceGUI, no pcbnew.
"""
from __future__ import annotations

import json

from kicraft.server import web


def test_load_events_reads_and_tolerates_corrupt_lines(tmp_path):
    (tmp_path / "events.jsonl").write_text(
        '{"kind": "build_log", "text": "a"}\n'
        "not json -- skipped\n"
        "\n"  # blank -- skipped
        '{"kind": "stage", "text": "b"}\n'
        "[1, 2, 3]\n",  # non-dict JSON -- skipped
        encoding="utf-8",
    )
    evs = web._load_events(tmp_path)
    assert [e.get("text") for e in evs] == ["a", "b"]


def test_provenance_is_crash_durable_and_deduplicated_on_reopen(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "_project_dir", lambda _state: tmp_path)
    state = {
        "events": [],
        "run_id": "p7-100",
        "_provenance_seq": 0,
    }
    recorded = web._record_progress_event(
        state,
        {
            "kind": "work_unit_plan",
            "stage": "bom",
            "unit_id": "bom-s000",
            "unit_sheet": "POWER",
            "source": "deterministic_architecture_lowering",
        },
    )

    assert recorded["event_id"] == "p7-100:1"
    assert recorded["ts"]
    durable = web._load_events(tmp_path)
    assert durable == [recorded]

    (tmp_path / "events.jsonl").write_text(
        json.dumps(recorded) + "\n",
    )
    assert web._load_events(tmp_path) == [recorded]


def test_reasoning_delta_stays_memory_only(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "_project_dir", lambda _state: tmp_path)
    state = {"events": [], "run_id": "p7-100", "_provenance_seq": 0}

    web._record_progress_event(state, {"kind": "reasoning_delta", "text": "private draft"})

    assert state["events"] == [{"kind": "reasoning_delta", "text": "private draft"}]
    assert not (tmp_path / "provenance.jsonl").exists()


def test_load_events_missing_or_none_is_empty(tmp_path):
    assert web._load_events(None) == []
    assert web._load_events(tmp_path) == []  # no events.jsonl in this dir
