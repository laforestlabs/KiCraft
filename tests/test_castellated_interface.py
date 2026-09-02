from pathlib import Path
from types import SimpleNamespace

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.design.cli_app import _castellation_geometry_violations
from kicraft.design.models import EdgeInterface


_PRETTY = (
    Path(__file__).parents[1]
    / "kicraft"
    / "parts_library"
    / "castellated-pad-2p54"
    / "castellated-pad-2p54.pretty"
)


def _add_rect_outline(board, left=0.0, top=0.0, right=20.0, bottom=20.0):
    points = [(left, top), (right, top), (right, bottom), (left, bottom)]
    for start, end in zip(points, points[1:] + points[:1]):
        shape = pcbnew.PCB_SHAPE(board)
        shape.SetShape(pcbnew.SHAPE_T_SEGMENT)
        shape.SetLayer(pcbnew.Edge_Cuts)
        shape.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(start[0]), pcbnew.FromMM(start[1])))
        shape.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(end[0]), pcbnew.FromMM(end[1])))
        board.Add(shape)


def _board(tmp_path, positions):
    path = tmp_path / "castellated.kicad_pcb"
    board = pcbnew.NewBoard(str(path))
    _add_rect_outline(board)
    for ref, (x, y) in positions.items():
        footprint = pcbnew.FootprintLoad(str(_PRETTY), "Castellated_Pad_2.54mm")
        assert footprint is not None
        footprint.SetReference(ref)
        footprint.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y)))
        board.Add(footprint)
    board.Save(str(path))
    return path


def _state():
    interface = EdgeInterface(
        name="left-bank",
        refs=["TP1", "TP2", "TP3"],
        side="left",
        pitch_mm=2.54,
        behavior="castellated",
    )
    return SimpleNamespace(bom=SimpleNamespace(edge_interfaces=[interface]))


def test_castellation_gate_accepts_exact_edge_pitch_and_rejects_interior_pad(tmp_path):
    valid = _board(tmp_path, {"TP1": (0, 5), "TP2": (0, 7.54), "TP3": (0, 10.08)})
    assert _castellation_geometry_violations(_state(), valid) == []

    invalid = _board(tmp_path, {"TP1": (0, 5), "TP2": (2, 7.54), "TP3": (0, 10.08)})
    violations = _castellation_geometry_violations(_state(), invalid)
    assert any("TP2 touches no edge" in violation for violation in violations)
