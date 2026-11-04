from pathlib import Path
from types import SimpleNamespace

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.design.cli_app import _castellation_geometry_violations
from kicraft.design.models import EdgeInterface
from kicraft.autoplacer.brain.leaf_routing import _place_fabricated_edge_interfaces
from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.hardware.adapter import KiCadAdapter


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
        net = pcbnew.NETINFO_ITEM(board, f"NET_{ref}")
        board.Add(net)
        for pad in footprint.Pads():
            pad.SetNet(net)
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


def _mixed_interfaces():
    return [
        EdgeInterface(name="left", refs=[f"TP{i}" for i in range(1, 4)],
                      side="left", pitch_mm=2.54, behavior="castellated"),
        EdgeInterface(name="top", refs=[f"TP{i}" for i in range(4, 14)],
                      side="top", pitch_mm=2.54, behavior="castellated"),
    ]


def _interface_config(interfaces):
    return {
        "edge_interfaces": [
            {"refs": interface.refs, "side": interface.side,
             "pitch_mm": interface.pitch_mm, "exact_edge_datum": True,
             "behavior": interface.behavior}
            for interface in interfaces
        ],
        "edge_margin_mm": 2.0,
        "placement_clearance_mm": 0.0,
    }


def _native_mixed_board(tmp_path, interfaces):
    refs = [ref for interface in interfaces for ref in interface.refs]
    seed = _board(tmp_path, {ref: (10, 10) for ref in refs})
    cfg = _interface_config(interfaces)
    adapter = KiCadAdapter(str(seed), config=cfg)
    local = adapter.load()
    _place_fabricated_edge_interfaces(local, cfg)
    board = pcbnew.LoadBoard(str(seed))
    for footprint in board.Footprints():
        component = local.components[footprint.GetReferenceAsString()]
        footprint.SetPosition(pcbnew.VECTOR2I(
            pcbnew.FromMM(component.pos.x), pcbnew.FromMM(component.pos.y)
        ))
    for drawing in list(board.GetDrawings()):
        if drawing.GetLayer() == pcbnew.Edge_Cuts:
            board.Remove(drawing)
    tl, br = local.board_outline
    _add_rect_outline(board, tl.x, tl.y, br.x, br.y)
    result = tmp_path / "placed.kicad_pcb"
    board.Save(str(result))
    return result, cfg


def _assert_native_datums(path, interfaces):
    board = pcbnew.LoadBoard(str(path))
    segments = [shape for shape in board.GetDrawings() if shape.GetLayer() == pcbnew.Edge_Cuts]
    endpoints = [point for segment in segments for point in (segment.GetStart(), segment.GetEnd())]
    edges = {
        "left": pcbnew.ToMM(min(point.x for point in endpoints)),
        "right": pcbnew.ToMM(max(point.x for point in endpoints)),
        "top": pcbnew.ToMM(min(point.y for point in endpoints)),
        "bottom": pcbnew.ToMM(max(point.y for point in endpoints)),
    }
    coords = {}
    for interface in interfaces:
        along = []
        for ref in interface.refs:
            footprint = board.FindFootprintByReference(ref)
            pads = list(footprint.Pads())
            assert len(pads) == 1
            pad = pads[0]
            pos = pad.GetPosition()
            x, y = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
            # The vendored through-hole pad has a centered nonzero drill and
            # actual copper on both sides; F.Fab graphics are not copper.
            assert pad.IsOnLayer(pcbnew.F_Cu) and pad.IsOnLayer(pcbnew.B_Cu)
            assert pad.GetProperty() == pcbnew.PAD_PROP_CASTELLATED
            assert pcbnew.ToMM(pad.GetDrillSize().x) > 0
            offset = pad.GetOffset()
            assert offset.x == 0 and offset.y == 0
            assert pcbnew.ToMM(pad.GetSize().x) > pcbnew.ToMM(pad.GetDrillSize().x)
            assert (x if interface.side in {"left", "right"} else y) == pytest.approx(
                edges[interface.side], abs=0.001
            )
            along.append(y if interface.side in {"left", "right"} else x)
            coords[ref] = (x, y)
        assert all(b - a == pytest.approx(interface.pitch_mm, abs=0.001)
                   for a, b in zip(along, along[1:]))
    assert _castellation_geometry_violations(
        SimpleNamespace(bom=SimpleNamespace(edge_interfaces=interfaces)), path
    ) == []
    return edges, coords


def test_native_mixed_edges_reloaded_and_invalid_copper(tmp_path):
    interfaces = _mixed_interfaces()
    first = tmp_path / "first"
    reverse = tmp_path / "reverse"
    first.mkdir()
    reverse.mkdir()
    path, cfg = _native_mixed_board(first, interfaces)
    reverse_path, _ = _native_mixed_board(reverse, list(reversed(interfaces)))
    edges, positions = _assert_native_datums(path, interfaces)
    reverse_edges, reverse_positions = _assert_native_datums(reverse_path, interfaces)
    assert edges == pytest.approx(reverse_edges, abs=0.001)
    for ref in positions:
        assert positions[ref] == pytest.approx(reverse_positions[ref], abs=0.001)

    board = pcbnew.LoadBoard(str(path))
    fp = board.FindFootprintByReference("TP2")
    p = fp.GetPosition()
    fp.SetPosition(pcbnew.VECTOR2I(p.x + pcbnew.FromMM(2), p.y))
    moved = tmp_path / "moved.kicad_pcb"
    board.Save(str(moved))
    state = SimpleNamespace(bom=SimpleNamespace(edge_interfaces=interfaces))
    assert _castellation_geometry_violations(state, moved)

    # Put an ordinary footprint's copper directly over an edge pad.
    board = pcbnew.LoadBoard(str(path))
    ordinary = pcbnew.FootprintLoad(str(_PRETTY), "Castellated_Pad_2.54mm")
    ordinary.SetReference("R1")
    ordinary.SetPosition(board.FindFootprintByReference("TP2").GetPosition())
    board.Add(ordinary)
    collision = tmp_path / "collision.kicad_pcb"
    board.Save(str(collision))
    loaded = KiCadAdapter(str(collision), config=cfg).load()
    diagnostic = PlacementSolver(loaded, cfg, seed=0).legality_diagnostics(loaded.components)
    assert any({item["a"], item["b"]} == {"TP2", "R1"}
               for item in diagnostic["overlaps"])
