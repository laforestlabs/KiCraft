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


# ---------------------------------------------------------------------------
# GND thermal-via stitcher: collision guard + small-pad escape stitching
# ---------------------------------------------------------------------------

from kicraft.autoplacer.brain.gnd_pour import add_gnd_pour_and_thermal_vias  # noqa: E402


def _sot23_board(path, *, blocker_track=False):
    """A 3-pad SOT-23-shaped U1 whose GND pad (0.5 mm -- too small for an
    in-pad 0.6 mm via) sits at (10,10), plus a 1.3 mm via-fitting GND pad on a
    6-pad U2 at (15,15). Optionally a foreign B.Cu track right under U2's pad
    (the IP2368-bank shorts shape: a thermal via stamped blind through it)."""
    board = pcbnew.NewBoard(path)
    for name in ("GND", "VIN", "VOUT", "SIG"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))

    def net(n):
        return board.GetNetInfo().GetNetItem(n)

    corners = [(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)

    def add_fp(ref, pads):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        board.Add(fp)
        for num, (netname, x, y, w, h) in pads.items():
            pad = pcbnew.PAD(fp)
            pad.SetSize(pcbnew.VECTOR2I(_mm(w), _mm(h)))
            pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pcbnew.PAD.SMDMask())
            pad.SetNumber(num)
            if netname:
                pad.SetNet(net(netname))
            fp.Add(pad)

    add_fp("U1", {
        "1": ("VIN", 8.5, 10.0, 0.5, 0.5),
        "2": ("GND", 10.0, 10.0, 0.5, 0.5),   # too small for an in-pad via
        "3": ("VOUT", 11.5, 10.0, 0.5, 0.5),
    })
    add_fp("U2", {str(i): (("GND" if i == 1 else "SIG"), 13.0 + 2 * i, 15.0, 1.3, 1.3)
                  for i in range(1, 7)})
    if blocker_track:
        t = pcbnew.PCB_TRACK(board)
        t.SetStart(pcbnew.VECTOR2I(_mm(13.0), _mm(15.0)))
        t.SetEnd(pcbnew.VECTOR2I(_mm(19.0), _mm(15.0)))
        t.SetWidth(_mm(0.3))
        t.SetLayer(pcbnew.B_Cu)
        t.SetNet(net("SIG"))
        board.Add(t)
    board.Save(path)
    return path


def _gnd_vias(path):
    board = pcbnew.LoadBoard(path)
    return [t for t in board.GetTracks()
            if isinstance(t, pcbnew.PCB_VIA) and t.GetNetname() == "GND"]


def test_small_ic_gnd_pad_is_escape_stitched(tmp_path):
    # The post-connector-fix rc7 signature (run_03 U1.5 / run_05 U2.2): a
    # SOT-23-class regulator's lone GND pad floats as an F.Cu pour island
    # because the old stitcher required >= 6 pads AND an in-pad via fit.
    # It must now get a short escape stub with a via at the tip.
    path = str(tmp_path / "b.kicad_pcb")
    _sot23_board(path)
    res = add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND"})
    assert res["escape_stitched"] >= 1, res
    # U2's via-fitting GND pad still gets its in-pad via.
    assert res["thermal_vias_added"] >= 1, res
    vias = _gnd_vias(path)
    assert len(vias) >= 2
    # The escape via is OFF the small pad (10,10), not drilled through it.
    assert all(
        (pcbnew.ToMM(v.GetPosition().x), pcbnew.ToMM(v.GetPosition().y))
        != (pytest.approx(10.0), pytest.approx(10.0))
        for v in vias
    )


def test_thermal_via_not_stamped_through_foreign_track(tmp_path):
    # Regression for the IP2368-bank incident: _add_via had no collision check
    # and stamped GND vias straight through routed B.Cu tracks (7 shorts). A
    # foreign B.Cu track under U2's GND pad must block that via.
    path = str(tmp_path / "b.kicad_pcb")
    _sot23_board(path, blocker_track=True)
    res = add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND"})
    assert res["thermal_vias_blocked"] >= 1, res
    blocked_at = (15.0, 15.0)  # U2 pad 1 centre, right on the SIG track
    for v in _gnd_vias(path):
        pos = (pcbnew.ToMM(v.GetPosition().x), pcbnew.ToMM(v.GetPosition().y))
        assert pos != (pytest.approx(blocked_at[0]), pytest.approx(blocked_at[1]))
