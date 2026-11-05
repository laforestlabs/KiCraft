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
