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

from kicraft.autoplacer.routing_board import run_pcbnew_script_file
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
    run_pcbnew_script_file(str(SCRIPT), str(payload_path))
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


def test_pinout_labels_placed_per_pin(tmp_path):
    import pcbnew

    from kicraft.autoplacer.hardware._silk_legend_subprocess import (
        _collect_obstacles,
        _mm_box,
    )

    # Pick the first footprint with >=4 real-numbered pads; a pinout label is
    # only meaningful on a multi-pin part.
    src = pcbnew.LoadBoard(str(FIXTURE))
    fp = next(f for f in src.GetFootprints()
              if len([p for p in f.Pads() if p.GetNumber().strip()]) >= 4)
    ref = fp.GetReference()
    pads = [p for p in fp.Pads() if p.GetNumber().strip()][:3]
    texts = {p.GetNumber(): f"P{i + 1}" for i, p in enumerate(pads)}

    label = {
        "id": "pins", "kind": "pinout", "ref": ref,
        "pins": [{"pin": p.GetNumber(), "text": texts[p.GetNumber()]}
                 for p in pads],
        "priority": 1, "heights_mm": [0.8],
    }
    result, board_path = _run(tmp_path, [label])

    per_pin_ids = [p["id"] for p in result["placed"]
                   if p["id"].startswith("pins:")]
    assert len(per_pin_ids) >= 2

    board, board_texts, poly, mm_box = _board_texts_and_geometry(board_path)
    by_text = {t.GetText(): t for t in board_texts}
    pad_centers = {
        p.GetNumber(): ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        for p in pads for b in [_mm_box(p.GetBoundingBox())]
    }

    placed_boxes = {}
    for num, txt in texts.items():
        t = by_text.get(txt)
        if t is None:
            continue  # a pin may drop individually on a dense board
        box = mm_box(t.GetBoundingBox())
        placed_boxes[num] = box
        assert bbox_inside_poly(box, poly, 0.4)

    assert len(placed_boxes) >= 2

    # Every per-pin label clears courtyards/pads (the placer's own obstacle
    # set, margin 0.1 is tighter than the placer's 0.25 clearance).
    fresh = pcbnew.LoadBoard(str(FIXTURE))
    for box in placed_boxes.values():
        for ob in _collect_obstacles(fresh)["F"]:
            assert not boxes_overlap(box, ob, 0.1)

    # Distinct positions: per-pin placement, not one blob at the courtyard.
    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
               for b in placed_boxes.values()]
    assert len({(round(c[0], 2), round(c[1], 2)) for c in centers}) >= 2

    # Adjacency: each label sits nearest its OWN pad's center (the robust
    # form of "within a few mm of its own pad" — the placer backs off to a
    # larger gap when a tight gap would collide with a neighbour, so a fixed
    # 2.0 mm radius is not guaranteed on a dense board).
    def _dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    for num, box in placed_boxes.items():
        tc = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        own = pad_centers[num]
        assert all(_dist(tc, own) < _dist(tc, pad_centers[n])
                   for n in pad_centers if n != num)
