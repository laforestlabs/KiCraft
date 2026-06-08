"""Tests for the reusable footprint breakout-stub tool."""

from __future__ import annotations

import math

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.brain.breakout_stubs import (  # noqa: E402
    BreakoutSpec,
    add_breakout_stubs,
    auto_power_tie_specs,
    auto_signal_escape_specs,
    perimeter_tie_specs,
    radial_breakout_specs,
    radial_escape_point,
)

_mm = pcbnew.FromMM


def _board(path, fp_center, pad_nets):
    """Board with a 30x30 outline + one footprint at *fp_center* whose pads
    (num -> (net, x, y)) carry the given nets/positions (mm)."""
    board = pcbnew.NewBoard(path)
    for name in {v[0] for v in pad_nets.values() if v[0]}:
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

    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("J1")
    fp.SetPosition(pcbnew.VECTOR2I(_mm(fp_center[0]), _mm(fp_center[1])))
    board.Add(fp)
    for num, (netname, x, y) in pad_nets.items():
        pad = pcbnew.PAD(fp)
        pad.SetSize(pcbnew.VECTOR2I(_mm(0.3), _mm(1.0)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber(num)
        if netname:
            pad.SetNet(net(netname))
        fp.Add(pad)
    board.Save(path)
    return path


def test_radial_escape_point_extends_along_center_to_pad_ray():
    # Centre (5,5), pad (8,5): escape is +x by length.
    assert radial_escape_point((5, 5), (8, 5), 2.0) == pytest.approx((10.0, 5.0))
    # Degenerate pad-at-centre falls back to +x.
    assert radial_escape_point((5, 5), (5, 5), 1.5) == pytest.approx((6.5, 5.0))
    # Diagonal direction preserved, length honoured.
    ex, ey = radial_escape_point((0, 0), (3, 4), 5.0)  # unit (0.6,0.8)
    assert math.hypot(ex - 3, ey - 4) == pytest.approx(5.0)
    assert (ex, ey) == pytest.approx((6.0, 8.0))


def test_radial_stub_is_single_locked_segment_outward(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    # pad to the right of centre -> escape +x.
    _board(path, (5.0, 10.0), {"B5": ("CC2", 8.0, 10.0)})
    res = add_breakout_stubs(
        path, [BreakoutSpec(ref="J1", pad="B5", length_mm=1.5)]
    )
    assert res["stubs"] == 1 and res["segments"] == 1 and res["vias"] == 0

    board = pcbnew.LoadBoard(path)
    tracks = list(board.GetTracks())
    segs = [t for t in tracks if isinstance(t, pcbnew.PCB_TRACK) and not isinstance(t, pcbnew.PCB_VIA)]
    assert len(segs) == 1
    t = segs[0]
    assert t.IsLocked()
    assert t.GetNetname() == "CC2"
    assert pcbnew.ToMM(t.GetStart().x) == pytest.approx(8.0)
    assert pcbnew.ToMM(t.GetEnd().x) == pytest.approx(9.5)  # 8 + 1.5 outward


def test_waypoint_path_lays_multiple_segments(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, (5.0, 10.0), {"B5": ("CC2", 8.0, 10.0)})
    spec = BreakoutSpec(
        ref="J1", pad="B5", waypoints=[(10.0, 10.0), (10.0, 3.0), (12.0, 3.0)]
    )
    res = add_breakout_stubs(path, [spec])
    # pad -> wp1 -> wp2 -> wp3 == 3 segments.
    assert res["segments"] == 3


def test_via_at_end_adds_locked_via(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, (5.0, 10.0), {"B5": ("CC2", 8.0, 10.0)})
    res = add_breakout_stubs(
        path, [BreakoutSpec(ref="J1", pad="B5", via_at_end=True)]
    )
    assert res["vias"] == 1
    board = pcbnew.LoadBoard(path)
    vias = [t for t in board.GetTracks() if isinstance(t, pcbnew.PCB_VIA)]
    assert len(vias) == 1 and vias[0].GetNetname() == "CC2"


def test_missing_pad_and_unnetted_pad_are_skipped(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, (5.0, 10.0), {"B5": ("CC2", 8.0, 10.0), "B6": ("", 8.0, 11.0)})
    res = add_breakout_stubs(
        path,
        [
            BreakoutSpec(ref="J1", pad="NOPE"),
            BreakoutSpec(ref="J1", pad="B6"),  # no net
        ],
    )
    assert res["stubs"] == 0
    assert "J1.NOPE:pad_not_found" in res["skipped"]
    assert "J1.B6:no_net" in res["skipped"]


def test_radial_escape_is_clipped_before_a_neighbour_pad(tmp_path):
    # Blocker pad of a different net sits in the +x escape path; the stub must
    # stop before it (no short), not run through it.
    path = str(tmp_path / "b.kicad_pcb")
    _board(
        path,
        (5.0, 10.0),
        {"B5": ("CC2", 8.0, 10.0), "BLK": ("GND", 9.2, 10.0)},
    )
    res = add_breakout_stubs(
        path, [BreakoutSpec(ref="J1", pad="B5", length_mm=3.0)]
    )
    assert res["stubs"] == 1
    board = pcbnew.LoadBoard(path)
    seg = next(
        t
        for t in board.GetTracks()
        if isinstance(t, pcbnew.PCB_TRACK) and not isinstance(t, pcbnew.PCB_VIA)
    )
    # Blocker left edge ~9.05; the clipped stub must end short of it.
    assert pcbnew.ToMM(seg.GetEnd().x) < 9.0


def test_radial_escape_skipped_when_no_safe_room(tmp_path):
    # Blocker hard against the pad -> no safe escape -> stub skipped, not shorted.
    path = str(tmp_path / "b.kicad_pcb")
    _board(
        path,
        (5.0, 10.0),
        {"B5": ("CC2", 8.0, 10.0), "BLK": ("GND", 8.35, 10.0)},
    )
    res = add_breakout_stubs(
        path, [BreakoutSpec(ref="J1", pad="B5", length_mm=2.0)]
    )
    assert res["stubs"] == 0
    assert any("no_safe_radial_escape" in s for s in res["skipped"])


def test_perimeter_tie_routes_around_bbox(tmp_path):
    # Two VBUS pads at opposite ends of the connector + an obstacle pad between
    # them. The tie must route around the bbox (not straight through the middle).
    path = str(tmp_path / "b.kicad_pcb")
    _board(
        path,
        (10.0, 10.0),
        {
            "L": ("VBUS", 6.0, 10.0),
            "MID": ("CC2", 10.0, 10.0),
            "R": ("VBUS", 14.0, 10.0),
        },
    )
    board = pcbnew.LoadBoard(path)
    specs = perimeter_tie_specs(board, "J1", net_names=["VBUS"], margin_mm=1.0)
    assert len(specs) == 1
    s = specs[0]
    # Ends on the far VBUS pad.
    assert s.waypoints[-1] == pytest.approx((14.0, 10.0))
    # Detours off the pad row (corner waypoints well above/below y=10)...
    assert any(abs(y - 10.0) > 0.8 for _, y in s.waypoints)
    # ...and never lands on the obstacle (CC2 pad at (10,10)).
    assert not any(
        abs(x - 10.0) < 0.5 and abs(y - 10.0) < 0.5 for x, y in s.waypoints
    )


def test_auto_power_tie_detects_connector_and_skips_gnd(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(
        path,
        (10.0, 10.0),
        {
            "1": ("VBUS", 6.0, 10.0),
            "2": ("VBUS", 14.0, 10.0),
            "3": ("GND", 6.0, 12.0),
            "4": ("GND", 14.0, 12.0),
            "5": ("CC2", 10.0, 10.0),
        },
    )
    board = pcbnew.LoadBoard(path)
    specs = auto_power_tie_specs(board, {"gnd_zone_net": "GND"})
    # VBUS (2 pads) tied; GND excluded (handled by GND plane); CC2 (1 pad) skipped.
    nets = {board.GetFootprints()[0].FindPadByNumber(s.pad).GetNetname() for s in specs}
    assert nets == {"VBUS"}


def test_auto_power_tie_respects_exclude(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, (10.0, 10.0), {"1": ("VBUS", 6.0, 10.0), "2": ("VBUS", 14.0, 10.0)})
    board = pcbnew.LoadBoard(path)
    assert auto_power_tie_specs(board, {"power_tie_exclude_refs": ["J1"]}) == []


def _conn_resistor_board(path, conn_pads, res_pads):
    """A 30x30 board with a J1 connector + an R1 resistor (each num -> (net,x,y))
    so a signal net can span two pads (connector pin -> resistor)."""
    board = pcbnew.NewBoard(path)
    all_pads = {**conn_pads, **res_pads}
    for name in {v[0] for v in all_pads.values() if v[0]}:
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

    def add_fp(ref, center, pads):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(_mm(center[0]), _mm(center[1])))
        board.Add(fp)
        for num, (netname, x, y) in pads.items():
            pad = pcbnew.PAD(fp)
            pad.SetSize(pcbnew.VECTOR2I(_mm(0.3), _mm(1.0)))
            pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pcbnew.PAD.SMDMask())
            pad.SetNumber(num)
            if netname:
                pad.SetNet(net(netname))
            fp.Add(pad)

    add_fp("J1", (5.0, 10.0), conn_pads)
    add_fp("R1", (20.0, 10.0), res_pads)
    board.Save(path)
    return path


def _j1_escaped_nets(board, specs):
    fp = next(f for f in board.GetFootprints() if f.GetReferenceAsString() == "J1")
    return {fp.FindPadByNumber(s.pad).GetNetname() for s in specs}


def test_auto_signal_escape_escapes_connector_signal_pads(tmp_path):
    # USB-C-like connector: spread VBUS (2 pads) marks it dense; its CC1/CC2
    # signal pins each reach a resistor pad (2-pad nets) and must escape; VBUS/GND
    # (power) and a single-pad interface pin (SBU1) are left alone.
    path = str(tmp_path / "b.kicad_pcb")
    _conn_resistor_board(
        path,
        conn_pads={
            "A4": ("VBUS", 6.0, 9.0), "B9": ("VBUS", 6.0, 11.0),
            "A1": ("GND", 7.0, 9.0), "B12": ("GND", 7.0, 11.0),
            "A5": ("CC1", 5.0, 9.0), "B5": ("CC2", 5.0, 11.0),
            "A8": ("SBU1", 4.0, 9.0),  # single pad on the board -> skipped
        },
        res_pads={"1": ("CC1", 20.0, 9.0), "2": ("CC2", 20.0, 11.0)},
    )
    board = pcbnew.LoadBoard(path)
    specs = auto_signal_escape_specs(board, {"gnd_zone_net": "GND"})
    assert all(s.ref == "J1" for s in specs)  # resistor (no spread power) untouched
    assert _j1_escaped_nets(board, specs) == {"CC1", "CC2"}


def test_auto_signal_escape_skips_footprint_without_spread_power(tmp_path):
    # Only one VBUS pad -> not a dense connector -> no signal escapes.
    path = str(tmp_path / "b.kicad_pcb")
    _conn_resistor_board(
        path,
        conn_pads={"1": ("VBUS", 6.0, 10.0), "5": ("CC2", 5.0, 11.0)},
        res_pads={"1": ("CC2", 20.0, 11.0)},
    )
    board = pcbnew.LoadBoard(path)
    assert auto_signal_escape_specs(board, {"gnd_zone_net": "GND"}) == []


def test_auto_signal_escape_disable_and_exclude(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _conn_resistor_board(
        path,
        conn_pads={"A4": ("VBUS", 6.0, 9.0), "B9": ("VBUS", 6.0, 11.0), "B5": ("CC2", 5.0, 11.0)},
        res_pads={"1": ("CC2", 20.0, 11.0)},
    )
    board = pcbnew.LoadBoard(path)
    assert auto_signal_escape_specs(board, {"auto_signal_escape": False}) == []
    assert auto_signal_escape_specs(board, {"signal_escape_exclude_refs": ["J1"]}) == []


def test_radial_breakout_specs_filters(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(
        path,
        (5.0, 10.0),
        {
            "B5": ("CC2", 8.0, 10.0),
            "A5": ("CC1", 8.0, 11.0),
            "S1": ("", 8.0, 12.0),
        },
    )
    board = pcbnew.LoadBoard(path)
    # nets_only restricts to CC2; unnetted S1 always skipped.
    specs = radial_breakout_specs(board, "J1", nets_only=["CC2"])
    assert [s.pad for s in specs] == ["B5"]
    # default: all netted pads.
    specs_all = radial_breakout_specs(board, "J1")
    assert {s.pad for s in specs_all} == {"B5", "A5"}
