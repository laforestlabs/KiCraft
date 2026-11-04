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


def test_seeded_grid_places_offset_leaf_bounds_inside_outline(tmp_path: Path):
    from kicraft.layout_editor.leaves import LeafInfo
    from kicraft.layout_editor.runner import load_initial_layout

    leaves = [
        LeafInfo(
            instance_path="/offset", sheet_name="offset", width_mm=10, height_mm=10,
            artifact_dir=tmp_path, edge_min_x=100, edge_min_y=200,
            edge_max_x=110, edge_max_y=210,
        ),
        LeafInfo(
            instance_path="/wide", sheet_name="wide", width_mm=10, height_mm=10,
            artifact_dir=tmp_path, edge_min_x=-8, edge_min_y=-5,
            edge_max_x=12, edge_max_y=7,
        ),
    ]
    layout = load_initial_layout(tmp_path, leaves)
    boxes = []
    for leaf, placement in zip(leaves, layout["placements"]):
        origin = placement["origin"]
        box = (
            origin["x"] + leaf.edge_min_x, origin["y"] + leaf.edge_min_y,
            origin["x"] + leaf.edge_max_x, origin["y"] + leaf.edge_max_y,
        )
        outline = layout["board_outline"]
        assert outline["min"]["x"] <= box[0] < box[2] <= outline["max"]["x"]
        assert outline["min"]["y"] <= box[1] < box[3] <= outline["max"]["y"]
        boxes.append(box)
    assert boxes[0][2] < boxes[1][0]


def test_old_canvas_extent_version_is_not_reused(tmp_path: Path):
    import os

    from kicraft.layout_editor.render import RENDERER_VERSION, _read_sidecar

    pcb = tmp_path / "leaf_routed.kicad_pcb"
    png = tmp_path / "leaf_canvas.png"
    sidecar = tmp_path / "leaf_canvas.png.extent.json"
    pcb.write_text("(kicad_pcb)", encoding="utf-8")
    png.write_bytes(b"cached")
    payload = {
        "renderer_version": 4, "x_mm": 10, "y_mm": 10,
        "width_mm": 3, "height_mm": 4,
    }
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    os.utime(pcb, (1, 1))
    os.utime(png, (2, 2))
    os.utime(sidecar, (2, 2))
    assert _read_sidecar(png, pcb) is None

    payload.update(renderer_version=RENDERER_VERSION, x_mm=5, y_mm=5,
                   width_mm=10, height_mm=10)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    assert _read_sidecar(png, pcb) == (5, 5, 10, 10)
