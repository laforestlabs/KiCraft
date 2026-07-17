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
