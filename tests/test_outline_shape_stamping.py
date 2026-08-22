"""Shaped board outlines must reach native KiCad Edge.Cuts geometry.

Drives the real parent stamp subprocess with polygonal and rectangular
outlines and checks the persisted board geometry.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.brain.types import Point  # noqa: E402
from kicraft.autoplacer.routing_board import run_pcbnew_script_file  # noqa: E402
from kicraft.cli.compose_subcircuits import _PARENT_STAMP_SCRIPT_PATH  # noqa: E402
from kicraft.layout_editor.outline import OutlineSpec  # noqa: E402


def _make_source_board(path: str) -> None:
    """Minimal source board; the stamper strips and rebuilds Edge.Cuts."""
    board = pcbnew.NewBoard(path)
    mm = pcbnew.FromMM
    for x1, y1, x2, y2 in [(0, 0, 10, 0), (10, 0, 10, 10), (10, 10, 0, 10), (0, 10, 0, 0)]:
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(mm(x1), mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(mm(x2), mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)
    board.Save(path)


def _stamp(payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="stamp_test_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        run_pcbnew_script_file(_PARENT_STAMP_SCRIPT_PATH, tmp)
    finally:
        os.unlink(tmp)


def _edge_cuts_segments(pcb_path: str) -> list[tuple[float, float, float, float]]:
    board = pcbnew.LoadBoard(pcb_path)
    segs = []
    for dwg in board.GetDrawings():
        if dwg.GetLayer() != pcbnew.Edge_Cuts:
            continue
        s, e = dwg.GetStart(), dwg.GetEnd()
        segs.append(
            (
                pcbnew.ToMM(s.x), pcbnew.ToMM(s.y),
                pcbnew.ToMM(e.x), pcbnew.ToMM(e.y),
            )
        )
    return segs


def _base_payload(src: str, out: str) -> dict:
    return {
        "pcb_path": src,
        "output_path": out,
        "components": [],
        "traces": [],
        "vias": [],
        "silkscreen": [],
        "keepouts": [],
    }


def _circle_spec() -> OutlineSpec:
    return OutlineSpec(
        shape="circle", min_pt=Point(0.0, 0.0), max_pt=Point(50.0, 50.0)
    )


def test_polyline_payload_stamps_exact_closed_loop():
    spec = _circle_spec()
    polyline = [[p.x, p.y] for p in spec.polyline()]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.kicad_pcb")
        out = os.path.join(d, "circle.kicad_pcb")
        _make_source_board(src)
        payload = _base_payload(src, out)
        payload["outline"] = {
            "tl_x": 0.0, "tl_y": 0.0, "br_x": 50.0, "br_y": 50.0,
            "polyline": polyline,
        }
        _stamp(payload)
        segs = _edge_cuts_segments(out)

    assert len(segs) == len(polyline), (
        f"expected one Edge.Cuts segment per polyline vertex "
        f"({len(polyline)}), got {len(segs)}"
    )
    # Segments chain: each start point is a polyline vertex and each end
    # is the next vertex (closed). Compare as rounded sets to be robust
    # to segment ordering in GetDrawings().
    want_edges = {
        (
            round(polyline[i][0], 3), round(polyline[i][1], 3),
            round(polyline[(i + 1) % len(polyline)][0], 3),
            round(polyline[(i + 1) % len(polyline)][1], 3),
        )
        for i in range(len(polyline))
    }
    got_edges = {tuple(round(v, 3) for v in s) for s in segs}
    assert got_edges == want_edges


def test_no_polyline_key_keeps_legacy_rectangle():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.kicad_pcb")
        out = os.path.join(d, "rect.kicad_pcb")
        _make_source_board(src)
        payload = _base_payload(src, out)
        payload["outline"] = {"tl_x": 0.0, "tl_y": 0.0, "br_x": 80.0, "br_y": 60.0}
        _stamp(payload)
        segs = _edge_cuts_segments(out)
    assert len(segs) == 4


