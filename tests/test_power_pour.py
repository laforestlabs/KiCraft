"""Tests for the power-plane pour (kicraft.autoplacer.brain.gnd_pour).

A power rail (e.g. VBUS) is poured as a solid plane on the layer opposite GND so
paired connector power pads the autorouter can't thread together connect through
copper. Solid (not thermal) pad connection is required -- thermal-relief spokes
need a gap wider than a dense connector's pad pitch and never form.
"""

from __future__ import annotations

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.brain.gnd_pour import (  # noqa: E402
    _detect_power_nets,
    pour_power_planes,
)

_mm = pcbnew.FromMM


def _board(path, pad_nets):
    """Build a board: rectangular outline + one footprint whose pads carry the
    given (number -> net name) mapping. Returns the saved+reloaded board path."""
    board = pcbnew.NewBoard(path)
    for name in {n for n in pad_nets.values() if n}:
        board.Add(pcbnew.NETINFO_ITEM(board, name))

    def net(n):
        return board.GetNetInfo().GetNetItem(n)

    # 20x20 mm Edge.Cuts rectangle so GetBoardEdgesBoundingBox is non-empty.
    corners = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)

    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("J1")
    board.Add(fp)
    x = 5
    for num, netname in pad_nets.items():
        pad = pcbnew.PAD(fp)
        pad.SetSize(pcbnew.VECTOR2I(_mm(1), _mm(1)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(10)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber(num)
        if netname:
            pad.SetNet(net(netname))
        fp.Add(pad)
        x += 3
    board.Save(path)
    return path


def test_detect_power_nets_ranks_by_pad_count_and_excludes_gnd(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, {"1": "VBUS", "2": "VBUS", "3": "GND", "4": "USB_D+"})
    board = pcbnew.LoadBoard(path)
    # GND excluded; VBUS (2 pads) ranks above any tie; max_nets default 1.
    assert _detect_power_nets(board, {"gnd_zone_net": "GND"}) == ["VBUS"]


def test_detect_power_nets_explicit_override(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, {"1": "VBUS", "2": "+3V3"})
    board = pcbnew.LoadBoard(path)
    cfg = {"gnd_zone_net": "GND", "power_plane_nets": ["+3V3"]}
    assert _detect_power_nets(board, cfg) == ["+3V3"]


def test_detect_power_nets_max_nets(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, {"1": "VBUS", "2": "VBUS", "3": "+3V3"})
    board = pcbnew.LoadBoard(path)
    cfg = {"gnd_zone_net": "GND", "power_plane_max_nets": 2}
    assert set(_detect_power_nets(board, cfg)) == {"VBUS", "+3V3"}


def test_pour_creates_solid_power_zone(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, {"1": "VBUS", "2": "VBUS", "3": "GND"})
    res = pour_power_planes(path, {"gnd_zone_net": "GND"}, layers=("F.Cu",))
    assert res["nets"] == ["VBUS"]
    assert res["zones"] == 1

    board = pcbnew.LoadBoard(path)
    vbus_zones = [
        z
        for z in board.Zones()
        if z.GetNetname() == "VBUS" and z.GetLayer() == pcbnew.F_Cu
    ]
    assert len(vbus_zones) == 1
    z = vbus_zones[0]
    assert z.GetPadConnection() == pcbnew.ZONE_CONNECTION_FULL
    assert z.GetAssignedPriority() == 1
    # No GND zone is created by the power pour (that is the GND pour's job).
    assert not [zz for zz in board.Zones() if zz.GetNetname() == "GND"]


def test_pour_disabled_is_noop(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, {"1": "VBUS", "2": "VBUS"})
    res = pour_power_planes(path, {"power_plane_enabled": False})
    assert res == {"nets": [], "zones": 0}
