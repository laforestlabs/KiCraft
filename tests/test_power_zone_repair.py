"""Native regression for full-class repair of disconnected zone islands."""
import copy
import json
import os
import shutil
from pathlib import Path

import pytest


@pytest.mark.parametrize("zone_case", ["split_zone", "zone_to_pad"])
def test_zone_gap_uses_physical_anchors(tmp_path, zone_case):
    pcbnew = pytest.importorskip("pcbnew")
    router = os.environ.get("KICRAFT_KICAD_ROUTING_TOOLS_PATH")
    python = os.environ.get("KICRAFT_KICAD_ROUTING_TOOLS_PYTHON")
    if not shutil.which("kicad-cli") or not router or not python:
        pytest.skip("native DRC and explicitly configured pinned routing runtime are required")
    from kicraft.autoplacer.brain.unconnected_repair import (
        _parse_unconnected_edges, repair_unconnected_power,
    )
    from kicraft.autoplacer.routing_board import run_kicad_cli_drc
    from kicraft.design.synthesis.kicad_pro import DEFAULT_NETCLASS, POWER_NETCLASS

    mm = pcbnew.FromMM
    point = lambda x, y: pcbnew.VECTOR2I(mm(x), mm(y))
    path = tmp_path / "zone-gap.kicad_pcb"
    board = pcbnew.NewBoard(str(path))
    for start, end in zip(((0, 0), (20, 0), (20, 20), (0, 20)),
                          ((20, 0), (20, 20), (0, 20), (0, 0))):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetStart(point(*start))
        edge.SetEnd(point(*end))
        edge.SetLayer(pcbnew.Edge_Cuts)
        board.Add(edge)
    power = pcbnew.NETINFO_ITEM(board, "RAIL_A")
    ground = pcbnew.NETINFO_ITEM(board, "GND")
    board.Add(power)
    board.Add(ground)
    for ref, x in (("J1", 5), ("J2", 15.123456)):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(point(x, 10.123456 if ref == "J2" else 10))
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetNumber("1")
        pad.SetSize(point(1.2, 1.2))
        pad.SetPosition(fp.GetPosition())
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNet(power)
        fp.Add(pad)
    # A front-layer barrier splits one zone and forces a back-layer bridge.
    for start, end in (((10, .8), (10, 19.2)), ((2, 2), (10, 2))):
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(point(*start))
        track.SetEnd(point(*end))
        track.SetWidth(mm(.5))
        track.SetLayer(pcbnew.F_Cu)
        track.SetNet(ground)
        board.Add(track)
    zone = pcbnew.ZONE(board)
    zone.SetNet(power)
    zone.SetLayer(pcbnew.F_Cu)
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    zone.SetLocalClearance(mm(.3))
    zone.SetMinThickness(mm(.25))
    zone.Outline().NewOutline()
    right = 18 if zone_case == "split_zone" else 9
    for x, y in ((2, 2), (right, 2), (right, 18), (2, 18)):
        zone.Outline().Append(mm(x), mm(y))
    board.Add(zone)
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    fill = zone.GetFilledPolysList(pcbnew.F_Cu)
    assert fill.OutlineCount() == (2 if zone_case == "split_zone" else 1)
    original = {
        item.m_Uuid.AsString(): (
            item.GetNetname(), item.GetStart().x, item.GetStart().y,
            item.GetEnd().x, item.GetEnd().y, item.GetWidth(), item.GetLayer(),
        ) for item in board.GetTracks()
    }
    board.Save(str(path))
    pro = path.with_suffix(".kicad_pro")
    pro.write_text(json.dumps({
        "board": {"design_settings": {"meta": {"version": 2}}},
        "meta": {"filename": pro.name, "version": 1},
        "net_settings": {
            "classes": [dict(DEFAULT_NETCLASS), dict(POWER_NETCLASS, track_width=.5,
                                                        via_diameter=.8, via_drill=.4)],
            "meta": {"version": 3}, "net_colors": None,
            "netclass_assignments": None,
            "netclass_patterns": [{"netclass": "Power", "pattern": "RAIL_A"}],
        },
    }))
    dru = path.with_suffix(".kicad_dru")
    dru.write_text("(version 1)\n")
    cfg = {"power_nets": {"RAIL_A", "GND"},
           "kicad_routing_tools_path": str(Path(router)),
           "kicad_routing_tools_python": str(Path(python)),
           "fab_capability": {"min_track_mm": .17, "min_clearance_mm": .17},
           "escape_via_size_mm": .36, "escape_via_drill_mm": .15}
    original_cfg = copy.deepcopy(cfg)
    pro_bytes, dru_bytes = pro.read_bytes(), dru.read_bytes()
    initial = run_kicad_cli_drc(str(path))
    edges = _parse_unconnected_edges(initial["report_text"])
    zone_edges = [(a, b) for a, b in edges
                  if a.net == "RAIL_A" and "zone" in (a.kind, b.kind)]
    assert zone_edges, initial
    assert all(not fill.Contains(point(*end.xy))
               for edge in zone_edges for end in edge if end.kind == "zone")
    assert any("pad" in (a.kind, b.kind) for a, b in zone_edges) == (zone_case == "zone_to_pad")

    result = repair_unconnected_power(str(path), cfg)
    assert result["tied"] == 1, result
    assert cfg == original_cfg and pro.read_bytes() == pro_bytes and dru.read_bytes() == dru_bytes
    routed = pcbnew.LoadBoard(str(path))
    items = list(routed.GetTracks())
    retained = {
        item.m_Uuid.AsString(): (
            item.GetNetname(), item.GetStart().x, item.GetStart().y,
            item.GetEnd().x, item.GetEnd().y, item.GetWidth(), item.GetLayer(),
        ) for item in items if item.m_Uuid.AsString() in original
    }
    assert retained == original
    added = [item for item in items if item.m_Uuid.AsString() not in original]
    vias = [item for item in added if item.Type() == pcbnew.PCB_VIA_T]
    traces = [item for item in added if item.Type() == pcbnew.PCB_TRACE_T]
    assert vias and traces and {item.GetNetname() for item in added} == {"RAIL_A"}
    assert all(item.GetWidth() >= mm(.5) for item in traces)
    assert all(item.GetWidth(pcbnew.F_Cu) >= mm(.8) and item.GetDrillValue() >= mm(.4)
               for item in vias)
    final = run_kicad_cli_drc(str(path))
    for key in ("unconnected", "shorts", "clearance", "items_not_allowed",
                "tracks_crossing", "track_width", "annular_width", "padstack",
                "copper_edge_clearance"):
        assert final[key] == 0, (key, final)
