"""Manual-route round trip: persisted outcome + status line + events.

The manual-route job writes .experiments/manual/last_route_result.json
(kicraft.design.cli_app._write_manual_route_result); the editor and the
place/route tab read it back via the runner helpers so a failed attempt
carries its diagnosis instead of dead-ending in the generic failed view.
"""

from __future__ import annotations

import json
from pathlib import Path

from kicraft.design.cli_app import _write_manual_route_result
from kicraft.layout_editor.runner import (
    load_last_route_result,
    log_manual_event,
    manual_layout_status,
)


def test_write_and_load_route_result(tmp_path: Path):
    verify = {"shorts": 0, "unconnected": 3, "unconnected_nets": ["SDA", "SCL"]}
    _write_manual_route_result(tmp_path, rc=7, stage="verify", verify=verify)

    got = load_last_route_result(tmp_path / ".experiments")
    assert got["rc"] == 7
    assert got["stage"] == "verify"
    assert got["verify"]["unconnected_nets"] == ["SDA", "SCL"]
    assert got["finished_at"]


def test_status_line_states(tmp_path: Path):
    # No saved layout -> nothing to report.
    assert manual_layout_status(tmp_path) is None

    manual = tmp_path / ".experiments" / "manual"
    manual.mkdir(parents=True)
    (manual / "manual_layout.json").write_text("{}", encoding="utf-8")
    assert manual_layout_status(tmp_path) == "Manual layout saved"

    _write_manual_route_result(tmp_path, rc=6, stage="route")
    assert "routing failed" in manual_layout_status(tmp_path)

    _write_manual_route_result(tmp_path, rc=7, stage="verify")
    assert "failed verification" in manual_layout_status(tmp_path)

    _write_manual_route_result(tmp_path, rc=0, stage="ok")
    assert "fab-ready" in manual_layout_status(tmp_path)


def test_log_manual_event_appends_jsonl(tmp_path: Path):
    exp = tmp_path / ".experiments"
    log_manual_event(exp, "editor_opened", leaves=9)
    log_manual_event(exp, "stamp_result", rc=0, shorts=0)

    lines = (exp / "manual" / "usage.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event"] == "editor_opened" and first["leaves"] == 9
    assert first["ts"]


def test_load_parent_local_components(tmp_path: Path):
    from kicraft.layout_editor.runner import load_parent_local_components

    exp = tmp_path / ".experiments"
    snap_dir = exp / "hierarchical_autoexperiment" / "round_0001"
    snap_dir.mkdir(parents=True)
    (snap_dir / "parent_pipeline.json").write_text(json.dumps({
        "state": {
            "parent_local": [
                {"ref": "H1", "x": 5.0, "y": 5.0, "width_mm": 6.4,
                 "height_mm": 6.4, "rotation": 0.0, "kind": "mounting_hole"},
                {"ref": "bogus", "x": "not-a-number"},
            ],
        },
    }), encoding="utf-8")

    got = load_parent_local_components(exp)
    assert len(got) == 1
    assert got[0]["ref"] == "H1" and got[0]["kind"] == "mounting_hole"

    # Old-schema snapshot (no parent_local key) -> empty, no crash.
    (snap_dir / "parent_pipeline.json").write_text(
        json.dumps({"state": {"entries": []}}), encoding="utf-8")
    assert load_parent_local_components(exp) == []
