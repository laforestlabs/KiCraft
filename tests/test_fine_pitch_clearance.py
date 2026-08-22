"""Native KiCad connector-shield DRC waiver contracts."""
from __future__ import annotations

import os

from kicraft.autoplacer import routing_board


def test_connector_shield_annular_padstack_waived(monkeypatch, tmp_path):
    board = os.path.join(tmp_path, "routed.kicad_pcb")
    open(board, "w").close()
    report = (
        "[annular_width]: Annular width too small @(50.0 mm, 50.0 mm)\n"
        "    Hole of J1\n"
        "[annular_width]: Annular width too small @(51.0 mm, 50.0 mm)\n"
        "    Hole of J1\n"
        "[padstack]: Pad issue @(50.0 mm, 51.0 mm)\n"
        "    Pad of J1\n"
        "[padstack]: Pad issue @(51.0 mm, 51.0 mm)\n"
        "    Pad of J1\n"
    )
    canned = {
        "shorts": 0, "unconnected": 0, "clearance": 0,
        "copper_edge_clearance": 0, "courtyard": 0,
        "solder_mask_bridge": 0, "annular_width": 2, "padstack": 2,
        "total": 4, "violations": [], "ran": True,
        "report_text": report, "timed_out": False, "missing_cli": False,
    }
    monkeypatch.setattr(routing_board, "run_kicad_cli_drc", lambda *_a, **_k: dict(canned))
    monkeypatch.setattr(
        routing_board,
        "count_board_tracks",
        lambda *_a, **_k: {"traces": 1, "vias": 0, "total_length_mm": 1.0},
    )
    v = routing_board.validate_routed_board(
        board, cfg={"component_zones": {"J1": {"edge": True}}}
    )
    assert v["footprint_internal_annular_count"] == 2
    assert v["footprint_internal_padstack_count"] == 2
    assert v["waived_connector_shield_refs"] == ["J1"]
    assert v["accepted"] is True


def test_connector_shield_not_waived_when_ref_unknown(monkeypatch, tmp_path):
    board = os.path.join(tmp_path, "routed.kicad_pcb")
    open(board, "w").close()
    report = "[annular_width]: too small @(1 mm, 1 mm)\n    Hole of U7\n"
    canned = {
        "shorts": 0, "unconnected": 0, "clearance": 0,
        "copper_edge_clearance": 0, "courtyard": 0,
        "solder_mask_bridge": 0, "annular_width": 1, "padstack": 0,
        "total": 1, "violations": [], "ran": True,
        "report_text": report, "timed_out": False, "missing_cli": False,
    }
    monkeypatch.setattr(routing_board, "run_kicad_cli_drc", lambda *_a, **_k: dict(canned))
    monkeypatch.setattr(
        routing_board,
        "count_board_tracks",
        lambda *_a, **_k: {"traces": 1, "vias": 0, "total_length_mm": 1.0},
    )
    v = routing_board.validate_routed_board(board, cfg={"component_zones": {}})
    assert "footprint_internal_annular_count" not in v
    assert v["accepted"] is True
