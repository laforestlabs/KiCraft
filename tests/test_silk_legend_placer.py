"""Integration tests for the silk-legend placer subprocess.

Runs the real pcbnew subprocess on the routed USB_PD_TRIGGER fixture board
and checks the placement contract: everything lands inside the outline with
edge margin, nothing overlaps courtyards/pads, and what cannot be placed is
dropped with a reason (never silently squeezed).
"""
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pcbnew")

from kicraft.autoplacer.freerouting_runner import _run_pcbnew_script_file
from kicraft.autoplacer.hardware.silk_geometry import bbox_inside_poly, boxes_overlap

FIXTURE = (Path(__file__).parent / "fixtures" / "replay_workspace"
           / "USB_PD_TRIGGER" / "USB_PD_TRIGGER.kicad_pcb")
SCRIPT = (Path(__file__).parents[1] / "kicraft" / "autoplacer" / "hardware"
          / "_silk_legend_subprocess.py")


def _run(tmp_path: Path, labels: list[dict], legend_lines=None) -> tuple[dict, Path]:
    board = tmp_path / "board.kicad_pcb"
    shutil.copy(FIXTURE, board)
    result_path = tmp_path / "result.json"
    if legend_lines is None:
        legend_lines = [
            {"text": "USB-C PD Trigger", "height_mm": 1.2},
            {"text": "KiCraft KC-TEST rev 1.0 2026-01-01", "height_mm": 0.8},
        ]
    payload = {
        "pcb_path": str(board),
        "output_path": str(board),
        "result_path": str(result_path),
        "clearance_mm": 0.25,
        "edge_margin_mm": 0.5,
        "legend": {"lines": legend_lines, "gap_mm": 0.3},
        "labels": labels,
    }
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload))
    _run_pcbnew_script_file(str(SCRIPT), str(payload_path))
    return json.loads(result_path.read_text()), board


def _board_texts_and_geometry(board_path: Path):
    import pcbnew

    from kicraft.autoplacer.hardware._silk_legend_subprocess import (
        _collect_obstacles,
        _mm_box,
        _outline_poly,
    )

    board = pcbnew.LoadBoard(str(board_path))
    texts = [
        d for d in board.GetDrawings()
        if isinstance(d, pcbnew.PCB_TEXT)
        and d.GetLayer() in (pcbnew.F_SilkS, pcbnew.B_SilkS)
    ]
    return board, texts, _outline_poly(board), _mm_box


def test_legend_and_anchored_label_placed_inside_outline(tmp_path):
    result, board_path = _run(tmp_path, [
        {"id": "out-rating", "text": "OUT 9/12/20V", "ref": "J2",
         "prefer": "below", "priority": 1, "heights_mm": [1.0, 0.9, 0.8]},
    ])
    placed_ids = {p["id"] for p in result["placed"]}
    assert {"legend:0", "legend:1", "out-rating"} <= placed_ids
    assert result["dropped"] == []

    board, texts, poly, mm_box = _board_texts_and_geometry(board_path)
    new_texts = [t for t in texts if "KiCraft" in t.GetText()
                 or "OUT 9/12/20V" in t.GetText()
                 or "PD Trigger" in t.GetText()]
    assert len(new_texts) == 3
    for t in new_texts:
        assert bbox_inside_poly(mm_box(t.GetBoundingBox()), poly, 0.4)


def test_placed_text_clears_courtyards_and_pads(tmp_path):
    result, board_path = _run(tmp_path, [
        {"id": "dip-table", "kind": "table",
         "text": "VOUT  1 2 3\n 9V   ON - -\n12V   - ON -\n20V   - - ON",
         "ref": "SW1", "prefer": "right", "priority": 1,
         "heights_mm": [1.0, 0.9, 0.8]},
    ])
    assert "dip-table" in {p["id"] for p in result["placed"]}

    import pcbnew

    from kicraft.autoplacer.hardware._silk_legend_subprocess import (
        _collect_obstacles,
    )

    board, texts, poly, mm_box = _board_texts_and_geometry(board_path)
    table = next(t for t in texts if "VOUT" in t.GetText())
    table_box = mm_box(table.GetBoundingBox())
    fresh = pcbnew.LoadBoard(str(FIXTURE))
    for ob in _collect_obstacles(fresh)["F"]:
        assert not boxes_overlap(table_box, ob, 0.1)


def test_unplaceable_and_unanchored_labels_drop_with_reason(tmp_path):
    result, _ = _run(tmp_path, [
        {"id": "ghost", "text": "GHOST", "ref": "Z99", "priority": 2,
         "heights_mm": [0.8]},
        # priority-2 monster block: no anchored spot and no fallback sweep
        {"id": "monster", "text": "\n".join(["X" * 30] * 5), "ref": "U1",
         "priority": 2, "heights_mm": [8.0]},
    ])
    dropped = {d["id"]: d["reason"] for d in result["dropped"]}
    assert "Z99" in dropped["ghost"]
    assert "monster" in dropped


def test_result_reports_positions_for_placed_labels(tmp_path):
    result, _ = _run(tmp_path, [])
    for p in result["placed"]:
        assert {"id", "x_mm", "y_mm", "height_mm", "layer"} <= set(p)
        assert p["layer"] in ("F.SilkS", "B.SilkS")
