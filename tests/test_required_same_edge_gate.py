"""An explicit same-edge requirement is not an optional connector-access hint."""
from __future__ import annotations

import json

import pytest

from kicraft.design import cli_app


def _native_board(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    board = pcbnew.BOARD()
    for start, end in (((0, 0), (20, 0)), ((20, 0), (20, 20)),
                       ((20, 20), (0, 20)), ((0, 20), (0, 0))):
        edge = pcbnew.PCB_SHAPE()
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetLayer(pcbnew.Edge_Cuts)
        edge.SetStart(pcbnew.VECTOR2I(*(pcbnew.FromMM(v) for v in start)))
        edge.SetEnd(pcbnew.VECTOR2I(*(pcbnew.FromMM(v) for v in end)))
        edge.SetWidth(pcbnew.FromMM(0.05))
        board.Add(edge)
    footprints = {}
    for ref, x, y, value in (("J1", 5, 18.3, "HOST"), ("J2", 15, 10, "SWD")):
        footprint = pcbnew.FOOTPRINT(board)
        footprint.SetReference(ref)
        footprint.SetValue(value)
        footprint.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y)))
        pad = pcbnew.PAD(footprint)
        pad.SetNumber("1")
        pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
        pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
        pad.SetSize(pcbnew.VECTOR2I(pcbnew.FromMM(1.7), pcbnew.FromMM(1.7)))
        pad.SetDrillSize(pcbnew.VECTOR2I(pcbnew.FromMM(1), pcbnew.FromMM(1)))
        pad.SetPosition(footprint.GetPosition())
        footprint.Add(pad)
        board.Add(footprint)
        footprints[ref] = footprint
    path = tmp_path / "BOARD.kicad_pcb"
    pcbnew.SaveBoard(str(path), board)
    return pcbnew, board, footprints, path


def test_explicit_edge_includes_debug_header_and_blocks_fabrication(tmp_path, monkeypatch):
    pcbnew, board, footprints, pcb = _native_board(tmp_path)
    payload = {
        "required_same_edge_connectors": {"edge": "bottom", "refs": ["J1", "J2"]},
        "component_zones": {"J1": {"edge": "bottom"}, "J2": {"edge": "bottom"}},
    }
    (tmp_path / "BOARD_autoplacer.json").write_text(json.dumps(payload))
    # Isolate the independent electrical/DRC verdict, not the geometry under test.
    from kicraft.autoplacer import routing_board
    monkeypatch.setattr(routing_board, "validate_routed_board", lambda *a, **k: {
        "accepted": True, "drc": {}, "track_summary": {"footprint_refs": ["J1", "J2"]},
    })
    gate = cli_app._verify_routed_board(pcb, required_same_edge_refs=("J1", "J2"))
    assert gate["fab_acceptable"] is False
    assert any(reason.startswith("required_same_edge_gap:J2@")
               for reason in gate["required_same_edge_violations"])

    footprints["J2"].SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(15), pcbnew.FromMM(18.3)))
    pcbnew.SaveBoard(str(pcb), board)
    gate = cli_app._verify_routed_board(pcb, required_same_edge_refs=("J1", "J2"))
    assert gate["fab_acceptable"] is True
    assert gate["required_same_edge_violations"] == []
    demotion = tmp_path / ".experiments/subcircuits/leaf/edge_pins_demoted.json"
    demotion.parent.mkdir(parents=True)
    demotion.write_text(json.dumps({"refs": ["J2"]}))
    assert cli_app._effective_connector_zones(pcb)["J2"]["edge"] == "bottom"

    board.Remove(footprints["J2"])
    pcbnew.SaveBoard(str(pcb), board)
    assert cli_app._required_same_edge_violations(pcb, ("J1", "J2")) == [
        "required_same_edge_unmeasured:J2",
    ]


def test_original_contract_cannot_disappear_or_drop_a_connector(tmp_path):
    pcb = tmp_path / "BOARD.kicad_pcb"
    assert cli_app._required_same_edge_violations(pcb, ("J1", "J2")) == [
        "required_same_edge_contract_missing",
    ]
    config = tmp_path / "BOARD_autoplacer.json"
    config.write_text(json.dumps({
        "required_same_edge_connectors": {"edge": "bottom", "refs": ["J1"]},
        "component_zones": {"J1": {"edge": "bottom"}},
    }))
    assert cli_app._required_same_edge_violations(pcb, ("J1", "J2")) == [
        "required_same_edge_refs_mismatch",
    ]
    config.write_text(json.dumps({
        "required_same_edge_connectors": {"edge": "bottom", "refs": ["J1", "J2"]},
        "component_zones": {"J1": {"edge": "bottom"}, "J2": {"edge": "top"}},
    }))
    assert cli_app._required_same_edge_violations(pcb, ("J1", "J2")) == [
        "required_same_edge_zones_missing_or_conflicting",
    ]


def test_six_connector_fitted_outline_uses_rotated_header_end(tmp_path):
    """Exercise the producer's bank and fitted outline through the real gate."""
    from types import SimpleNamespace

    from kicraft.autoplacer.brain.types import (
        BoardState, Component, Layer, Pad, Point, TraceSegment, Via,
    )
    from kicraft.cli.compose_subcircuits import (
        _ensure_edge_blocks_extremal, _repair_parent_outline,
    )

    pcbnew = pytest.importorskip("pcbnew")
    components = {}
    zones = {}
    for index in range(6):
        ref = f"J{index + 1}"
        x = 5 + index * 12
        header = index == 0
        y = 10 if header else 16.815
        height = 15.33 if header else 4.0
        component = Component(
            ref=ref, value="", pos=Point(x, y), body_center=Point(x, y),
            rotation=90 if header else 0, layer=Layer.FRONT,
            width_mm=2.59 if header else 4.0, height_mm=height,
            kind="connector", mating_axis="board_normal" if header else "in_plane",
            pads=[Pad(ref=ref, pad_id="1", pos=Point(x, y + (6.35 if header else 0)),
                      net="", layer=Layer.FRONT, size_mm=Point(1.8, 1.8))],
        )
        components[ref] = component
        zones[ref] = {"edge": "bottom",
                      "anchor_offsets_by_rotation": {component.rotation: Point(0, height / 2)}}

    board_state = BoardState(
        components=components, board_outline=(Point(0, 0), Point(70, 18.94)),
        traces=[TraceSegment(start=Point(62, 18.4), end=Point(64, 18.4),
                             layer=Layer.FRONT, net="GND", width_mm=0.4)],
        vias=[Via(pos=Point(67, 18.4), net="GND", size_mm=0.6)],
    )
    state = SimpleNamespace(composition=SimpleNamespace(board_state=board_state),
                            manual_outline=None, edge_constrained_refs=set(components),
                            edge_zoned_outline_sides={"bottom"})
    refs = tuple(components)
    (tmp_path / "BOARD_autoplacer.json").write_text(json.dumps({
        "required_same_edge_connectors": {"edge": "bottom", "refs": list(refs)},
        "component_zones": {ref: {"edge": "bottom"} for ref in refs},
    }))
    pcb = tmp_path / "BOARD.kicad_pcb"

    def stamp():
        native = pcbnew.BOARD()
        tl, br = board_state.board_outline
        mm = pcbnew.FromMM
        for start, end in (((tl.x, tl.y), (br.x, tl.y)),
                           ((br.x, tl.y), (br.x, br.y)),
                           ((br.x, br.y), (tl.x, br.y)),
                           ((tl.x, br.y), (tl.x, tl.y))):
            edge = pcbnew.PCB_SHAPE()
            edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
            edge.SetLayer(pcbnew.Edge_Cuts)
            edge.SetStart(pcbnew.VECTOR2I(*map(mm, start)))
            edge.SetEnd(pcbnew.VECTOR2I(*map(mm, end)))
            native.Add(edge)
        for ref, comp in components.items():
            fp = pcbnew.FOOTPRINT(native)
            fp.SetReference(ref)
            fp.SetPosition(pcbnew.VECTOR2I(mm(comp.pos.x), mm(comp.pos.y)))
            low, high = comp.bbox()
            court = pcbnew.PCB_SHAPE(fp)
            court.SetShape(pcbnew.SHAPE_T_RECT)
            court.SetLayer(pcbnew.F_CrtYd)
            court.SetStart(pcbnew.VECTOR2I(mm(low.x), mm(low.y)))
            court.SetEnd(pcbnew.VECTOR2I(mm(high.x), mm(high.y)))
            court.SetWidth(mm(0.01))
            fp.Add(court)
            native.Add(fp)
        pcbnew.SaveBoard(str(pcb), native)

    stamp()
    before = cli_app._required_same_edge_violations(pcb, refs)
    assert len(before) == 1 and before[0].startswith("required_same_edge_gap:J1@")
    _ensure_edge_blocks_extremal(components, zones)
    _repair_parent_outline(state)
    stamp()
    assert cli_app._required_same_edge_violations(pcb, refs) == []
    assert board_state.board_outline[1].y >= 18.4 + 0.3 + 0.2
    assert board_state.board_outline[1].x <= 100
    assert board_state.board_outline[1].y <= 100
