"""Tests for the reusable footprint breakout-stub tool."""

from __future__ import annotations

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.brain.breakout_stubs import (  # noqa: E402
    BreakoutSpec,
    _seg_seg_dist_mm,
    _segment_clears_pads,
    add_breakout_stubs,
    auto_power_tie_specs,
    auto_signal_escape_specs,
    perimeter_tie_specs,
    shield_tie_specs,
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
    # Blockers hemming the pad in on every quadrant (the _board test pads are
    # 0.3 mm circles): every candidate direction collides before a legal tip
    # exists -> stub skipped, not shorted.
    path = str(tmp_path / "b.kicad_pcb")
    _board(
        path,
        (5.0, 10.0),
        {
            "B5": ("CC2", 8.0, 10.0),
            "B1": ("GND", 8.25, 10.25),
            "B2": ("GND", 7.75, 10.25),
            "B3": ("GND", 8.25, 9.75),
            "B4": ("GND", 7.75, 9.75),
        },
    )
    res = add_breakout_stubs(
        path, [BreakoutSpec(ref="J1", pad="B5", length_mm=2.0)]
    )
    assert res["stubs"] == 0
    assert any("no_safe_radial_escape" in s for s in res["skipped"])


def test_radial_escape_falls_back_to_axis_direction(tmp_path):
    # Connector-row shape: the radial direction (footprint centre -> pad,
    # here (0.6, 0.8)) is hemmed in by a neighbour sitting just off the ray
    # (the USB-C CC2 signature). The stub must fall back to an axis direction
    # instead of being skipped. The strict (pair-clearance) margin round also
    # rejects +x -- it would pass only 0.147 mm from N0, inside the 0.153
    # clearance -- so the first STRICTLY clear axis is -x, at full length.
    path = str(tmp_path / "b.kicad_pcb")
    _board(
        path,
        (5.0, 6.0),
        {
            "B5": ("CC2", 8.0, 10.0),
            "N0": ("GND", 8.18, 10.30),  # kills the radial ray near the start
            "N3": ("GND", 8.0, 11.15),   # row neighbours block +/-y tips
            "N4": ("GND", 8.0, 8.85),
        },
    )
    res = add_breakout_stubs(
        path, [BreakoutSpec(ref="J1", pad="B5", length_mm=1.5)]
    )
    assert res["stubs"] == 1
    board = pcbnew.LoadBoard(path)
    seg = next(
        t
        for t in board.GetTracks()
        if isinstance(t, pcbnew.PCB_TRACK) and not isinstance(t, pcbnew.PCB_VIA)
    )
    end = (pcbnew.ToMM(seg.GetEnd().x), pcbnew.ToMM(seg.GetEnd().y))
    # The -x axis escape at the requested length, not the diagonal radial one.
    assert end == (pytest.approx(6.5), pytest.approx(10.0)), end


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


def test_perimeter_tie_uses_pad_field_not_inflated_bbox(tmp_path):
    # Regression for the U1 power-leaf short. An LDO ties VIN+EN (both VBUS) in a
    # left pad column; +3V3 and an NC pad sit in a right column. A courtyard/silk
    # graphic inflates fp.GetBoundingBox() on three sides (but not right), so the
    # nearest border for the VBUS pads lands on the RIGHT and a naive walk drives
    # the lead-in legs straight across the +3V3 / NC pads -- a short. Tying off the
    # pad field instead keeps every segment clear.
    path = str(tmp_path / "b.kicad_pcb")
    board = pcbnew.NewBoard(path)
    for name in ("VBUS", "GND", "P3V3"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("U1")
    fp.SetPosition(pcbnew.VECTOR2I(_mm(10.0), _mm(10.0)))
    board.Add(fp)
    pads = {
        "1": ("VBUS", 9.5, 9.0),   # VIN
        "2": ("GND", 9.5, 10.0),   # GND, between the two VBUS pads
        "3": ("VBUS", 9.5, 11.0),  # EN tied to VIN
        "4": ("", 10.5, 11.0),     # NC, no net
        "5": ("P3V3", 10.5, 9.0),  # VOUT
    }
    for num, (netname, x, y) in pads.items():
        pad = pcbnew.PAD(fp)
        pad.SetSize(pcbnew.VECTOR2I(_mm(0.6), _mm(1.1)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber(num)
        if netname:
            pad.SetNet(board.GetNetInfo().GetNetItem(netname))
        fp.Add(pad)
    # Tall silk line far to the left -> inflates the footprint bbox left/top/bottom
    # (as a generous courtyard does) without touching the pad field.
    sh = pcbnew.PCB_SHAPE(fp)
    sh.SetShape(pcbnew.SHAPE_T_SEGMENT)
    sh.SetStart(pcbnew.VECTOR2I(_mm(5.0), _mm(5.0)))
    sh.SetEnd(pcbnew.VECTOR2I(_mm(5.0), _mm(15.0)))
    sh.SetLayer(pcbnew.F_SilkS)
    fp.Add(sh)
    board.Save(path)

    board = pcbnew.LoadBoard(path)
    specs = perimeter_tie_specs(board, "U1", net_names=["VBUS"], margin_mm=1.0)
    assert len(specs) == 1
    s = specs[0]
    fp = next(f for f in board.GetFootprints() if f.GetReferenceAsString() == "U1")
    foreign = [p for p in fp.Pads() if p.GetNetname() != "VBUS"]
    start = fp.FindPadByNumber(s.pad).GetPosition()
    points = [(pcbnew.ToMM(start.x), pcbnew.ToMM(start.y)), *s.waypoints]
    assert all(
        _segment_clears_pads(foreign, a, b, 0.1)
        for a, b in zip(points, points[1:])
    )


def test_waypoint_spec_crossing_foreign_pad_is_dropped(tmp_path):
    # The hard invariant: add_breakout_stubs never stamps a waypoint path that
    # runs across a pad of another net. A straight tie from the CC2 pad to (12,10)
    # passes through the GND pad at (9,10) -> the whole spec is dropped, unshorted.
    path = str(tmp_path / "b.kicad_pcb")
    _board(
        path,
        (5.0, 10.0),
        {"B5": ("CC2", 6.0, 10.0), "BLK": ("GND", 9.0, 10.0)},
    )
    res = add_breakout_stubs(
        path, [BreakoutSpec(ref="J1", pad="B5", waypoints=[(12.0, 10.0)])]
    )
    assert res["stubs"] == 0
    assert any("waypoint_crosses_pad" in s for s in res["skipped"])
    board = pcbnew.LoadBoard(path)
    segs = [
        t
        for t in board.GetTracks()
        if isinstance(t, pcbnew.PCB_TRACK) and not isinstance(t, pcbnew.PCB_VIA)
    ]
    assert segs == []


def test_perimeter_tie_direct_when_only_same_net_pads_between(tmp_path):
    # The brief-2 DIP-switch shape: three adjacent commons on one power net,
    # nothing foreign between the farthest pair. A same-net pad is a landing,
    # not an obstacle, so the tie must be the straight pad-to-pad segment --
    # not a perimeter walk (which, with the pad row near the board edge, used
    # to stamp locked copper off the board and hang FreeRouting).
    path = str(tmp_path / "b.kicad_pcb")
    _board(
        path,
        (10.0, 2.0),
        {
            "4": ("VDD", 8.0, 1.4),
            "5": ("VDD", 10.0, 1.4),
            "6": ("VDD", 12.0, 1.4),
        },
    )
    board = pcbnew.LoadBoard(path)
    specs = perimeter_tie_specs(board, "J1", net_names=["VDD"], margin_mm=1.0)
    assert len(specs) == 1
    # Direct tie: a single waypoint -- the far pad -- with no detour corners.
    assert len(specs[0].waypoints) == 1
    assert specs[0].waypoints[0] == pytest.approx((12.0, 1.4))


def test_perimeter_tie_walk_is_clamped_onto_the_board(tmp_path):
    # Power pads near the top board edge with a foreign pad between them: the
    # unclamped walk rectangle (pad field + margin) pokes past the outline, so
    # the old walk stamped locked copper off the board -- which hangs
    # FreeRouting. The walk must instead be clamped inside the outline while
    # still clearing every pad.
    path = str(tmp_path / "b.kicad_pcb")
    _board(
        path,
        (10.0, 3.0),
        {
            "4": ("VDD", 8.0, 1.4),
            "5": ("CC2", 10.0, 1.4),  # foreign pad blocks the direct tie
            "6": ("VDD", 12.0, 1.4),
            "1": ("GND", 8.0, 9.0),
            "2": ("GND", 10.0, 9.0),
            "3": ("GND", 12.0, 9.0),
        },
    )
    board = pcbnew.LoadBoard(path)
    specs = perimeter_tie_specs(board, "J1", net_names=["VDD"], margin_mm=1.0)
    assert len(specs) == 1
    s = specs[0]
    # It is a walk (detour corners), not a direct segment...
    assert len(s.waypoints) > 1
    # ...every waypoint stays on the 30x30 board...
    assert all(0.0 <= x <= 30.0 and 0.0 <= y <= 30.0 for x, y in s.waypoints)
    # ...and the whole path still clears every foreign pad.
    fp = next(f for f in board.GetFootprints() if f.GetReferenceAsString() == "J1")
    foreign = [p for p in fp.Pads() if p.GetNetname() != "VDD"]
    start = fp.FindPadByNumber(s.pad).GetPosition()
    points = [(pcbnew.ToMM(start.x), pcbnew.ToMM(start.y)), *s.waypoints]
    assert all(
        _segment_clears_pads(foreign, a, b, 0.1)
        for a, b in zip(points, points[1:])
    )


def test_waypoint_spec_leaving_board_is_dropped(tmp_path):
    # Hard invariant at the stamp choke point: locked copper outside the board
    # outline hangs FreeRouting, so a spec with an off-board waypoint is
    # skipped entirely -- whatever generated it.
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, (5.0, 10.0), {"B5": ("CC2", 8.0, 10.0)})
    res = add_breakout_stubs(
        path, [BreakoutSpec(ref="J1", pad="B5", waypoints=[(8.0, -1.0)])]
    )
    assert res["stubs"] == 0
    assert any("off_board" in s for s in res["skipped"])
    board = pcbnew.LoadBoard(path)
    segs = [
        t
        for t in board.GetTracks()
        if isinstance(t, pcbnew.PCB_TRACK) and not isinstance(t, pcbnew.PCB_VIA)
    ]
    assert segs == []


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


# ---------------------------------------------------------------------------
# Netclass-aware stamping (the rc7 CC2 signature) + mutual stub clearance
# ---------------------------------------------------------------------------


def _write_pro_with_power_class(pcb_path, power_nets):
    """Sibling .kicad_pro so LoadBoard resolves the Power netclass (0.3 mm)."""
    import json
    from pathlib import Path

    from kicraft.design.synthesis.kicad_pro import DEFAULT_NETCLASS, POWER_NETCLASS

    p = Path(pcb_path).with_suffix(".kicad_pro")
    p.write_text(
        json.dumps(
            {
                "board": {"design_settings": {"meta": {"version": 2}}},
                "meta": {"filename": p.name, "version": 1},
                "net_settings": {
                    "classes": [dict(DEFAULT_NETCLASS), dict(POWER_NETCLASS)],
                    "meta": {"version": 3},
                    "net_colors": None,
                    "netclass_assignments": None,
                    "netclass_patterns": [
                        {"netclass": "Power", "pattern": n} for n in power_nets
                    ],
                },
            }
        )
    )
    return p


def _dense_row_board(path):
    """CC2 escape pad at (8,10) with a same-footprint VBUS pad alongside at
    (8,10.65) (1.0x0.6, spans x 7.5-8.5 / y 10.35-10.95) -- the USB-C shape:
    the escape path is collision-clear, but a tip at the requested 0.6 mm sits
    inside the VBUS pad's Power-netclass keep-out."""
    board = pcbnew.NewBoard(path)
    for name in ("CC2", "VBUS"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))
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
    fp.SetPosition(pcbnew.VECTOR2I(_mm(5.0), _mm(10.0)))
    board.Add(fp)
    for num, net, x, y, w, h in (
        ("B5", "CC2", 8.0, 10.0, 0.3, 1.0),
        ("B4A9", "VBUS", 8.0, 10.65, 1.0, 0.6),
    ):
        pad = pcbnew.PAD(fp)
        try:
            pad.SetShape(pcbnew.PAD_SHAPE_RECT)
        except TypeError:  # KiCad 9 padstack API wants the layer first
            pad.SetShape(pcbnew.F_Cu, pcbnew.PAD_SHAPE_RECT)
        pad.SetSize(pcbnew.VECTOR2I(_mm(w), _mm(h)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber(num)
        pad.SetNet(board.GetNetInfo().GetNetItem(net))
        fp.Add(pad)
    board.Save(path)
    return path


def _single_track_end_x(path):
    board = pcbnew.LoadBoard(path)
    segs = [
        t
        for t in board.GetTracks()
        if isinstance(t, pcbnew.PCB_TRACK) and not isinstance(t, pcbnew.PCB_VIA)
    ]
    assert len(segs) == 1
    return pcbnew.ToMM(segs[0].GetEnd().x)


def test_radial_tip_extends_past_power_netclass_keepout(tmp_path):
    # With the Power netclass resolved, a 0.6 mm escape's tip would sit inside
    # the VBUS pad's 0.3 mm pair-clearance keep-out -- a tip FreeRouting cannot
    # attach to, which abandons the net (the rc7 CC2 signature). The stub must
    # extend until its tip is legal (x >= ~8.64), not stamp the illegal tip.
    #
    # NewBoard() registers its (netclass-less) project with pcbnew's settings
    # manager, and a later LoadBoard of the same path reuses that stale project
    # instead of reading the sibling .kicad_pro -- so build the board at a
    # scratch path and copy the bytes to a path whose first load sees the pro.
    import shutil

    scratch = str(tmp_path / "scratch.kicad_pcb")
    _dense_row_board(scratch)
    path = str(tmp_path / "b.kicad_pcb")
    shutil.copyfile(scratch, path)
    _write_pro_with_power_class(path, ["VBUS"])
    board = pcbnew.LoadBoard(path)
    vbus = next(
        p for f in board.GetFootprints() for p in f.Pads() if p.GetNumber() == "B4A9"
    )
    assert pcbnew.ToMM(vbus.GetOwnClearance(pcbnew.F_Cu)) == pytest.approx(0.3)
    del board

    res = add_breakout_stubs(path, [BreakoutSpec(ref="J1", pad="B5", length_mm=0.6)])
    assert res["stubs"] == 1
    assert _single_track_end_x(path) > 8.62


def test_radial_tip_stays_at_requested_length_without_netclasses(tmp_path):
    # Control: same geometry, no project netclasses -> the flat floor applies
    # and the 0.6 mm tip is already legal, so the stub ends exactly there.
    path = str(tmp_path / "b.kicad_pcb")
    _dense_row_board(path)
    res = add_breakout_stubs(path, [BreakoutSpec(ref="J1", pad="B5", length_mm=0.6)])
    assert res["stubs"] == 1
    assert _single_track_end_x(path) == pytest.approx(8.6, abs=0.02)


def test_conflicting_spec_against_stamped_stub_is_skipped(tmp_path):
    # Two specs on different nets whose copper would land on top of each other:
    # the first stamps, the second must be dropped -- two locked tracks 0.05 mm
    # apart are a violation no router pass can repair.
    path = str(tmp_path / "b.kicad_pcb")
    _board(
        path,
        (5.0, 10.0),
        {"V1": ("VBUS", 8.0, 10.0), "C1": ("CC2", 9.0, 8.0)},
    )
    res = add_breakout_stubs(
        path,
        [
            BreakoutSpec(ref="J1", pad="V1", length_mm=1.5),  # (8,10)->(9.5,10)
            BreakoutSpec(ref="J1", pad="C1", waypoints=[(9.0, 12.0)]),  # crosses it
        ],
    )
    assert res["stubs"] == 1
    assert any("conflicts_with_stamped_stub" in s for s in res["skipped"])


def test_seg_seg_dist_crossing_and_parallel():
    assert _seg_seg_dist_mm((0, 0), (2, 0), (1, -1), (1, 1)) == 0.0
    assert _seg_seg_dist_mm((0, 0), (2, 0), (0, 1), (2, 1)) == pytest.approx(1.0)
    assert _seg_seg_dist_mm((0, 0), (1, 0), (3, 0), (4, 0)) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Shield ties: netted PTH legs -> nearest same-net pad
# ---------------------------------------------------------------------------


def _shield_board(path):
    """J2 with two GND through-hole shield legs ("3","4"), one GND SMD pad,
    one VBUS SMD pad and a no-net mounting post."""
    board = pcbnew.NewBoard(path)
    for name in ("GND", "VBUS"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))
    corners = [(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("J2")
    fp.SetPosition(pcbnew.VECTOR2I(_mm(10.0), _mm(10.0)))
    board.Add(fp)

    def add_pad(num, net, x, y, *, pth):
        pad = pcbnew.PAD(fp)
        pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
        pad.SetNumber(num)
        if pth:
            pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
            pad.SetLayerSet(pcbnew.PAD.PTHMask())
            pad.SetSize(pcbnew.VECTOR2I(_mm(1.2), _mm(1.2)))
            pad.SetDrillSize(pcbnew.VECTOR2I(_mm(0.8), _mm(0.8)))
        else:
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pcbnew.PAD.SMDMask())
            pad.SetSize(pcbnew.VECTOR2I(_mm(0.6), _mm(0.6)))
        if net:
            pad.SetNet(board.GetNetInfo().GetNetItem(net))
        fp.Add(pad)

    add_pad("4", "GND", 10.0, 10.0, pth=True)
    add_pad("3", "GND", 10.5, 10.0, pth=True)  # nearer to "4" than the SMD pad
    add_pad("A1", "GND", 11.2, 10.8, pth=False)
    add_pad("A4", "VBUS", 12.0, 9.0, pth=False)
    add_pad("M", "", 9.0, 9.0, pth=True)  # no net -> never tied
    board.Save(path)
    return path


def test_shield_tie_prefers_smd_same_net_pad(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _shield_board(path)
    board = pcbnew.LoadBoard(path)
    specs = shield_tie_specs(board)
    # Both GND legs tie to the SMD pad (the pour/router reach it), NOT to the
    # nearer sibling PTH leg -- two isolated legs tied together stay isolated.
    assert {(s.pad, s.waypoints[0]) for s in specs} == {
        ("4", (11.2, 10.8)),
        ("3", (11.2, 10.8)),
    }
    # A stitching via at the shared SMD end bonds the shield island to the
    # B.Cu GND plane (the parent pour cannot reach the connector area). Both
    # ties end on the SAME pad, so only ONE via lands -- the second would be
    # a coincident drill (a hole-to-hole violation) and is skipped as
    # redundant, while its track still stamps.
    assert all(s.via_at_end for s in specs)
    res = add_breakout_stubs(path, specs)
    assert res["stubs"] == 2 and res["vias"] == 1
    routed = pcbnew.LoadBoard(path)
    segs = [
        t
        for t in routed.GetTracks()
        if isinstance(t, pcbnew.PCB_TRACK) and not isinstance(t, pcbnew.PCB_VIA)
    ]
    assert all(t.GetNetname() == "GND" and t.IsLocked() for t in segs)


def test_shield_tie_respects_disable_and_max_distance(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _shield_board(path)
    board = pcbnew.LoadBoard(path)
    assert shield_tie_specs(board, {"shield_tie_enabled": False}) == []
    assert shield_tie_specs(board, {"shield_tie_max_mm": 1.0}) == []
    assert shield_tie_specs(board, {"shield_tie_exclude_refs": ["J2"]}) == []


# ---------------------------------------------------------------------------
# Strict same-footprint margins + tip-via guards
# ---------------------------------------------------------------------------

def test_foreign_pad_margins_strict_same_fp(tmp_path):
    from kicraft.autoplacer.brain.breakout_stubs import (
        _foreign_pad_margins,
        _own_clearance_mm,
    )

    path = str(tmp_path / "b.kicad_pcb")
    _board(path, (5.0, 10.0), {
        "B5": ("CC2", 8.0, 10.0),
        "B6": ("GND", 8.0, 11.5),
    })
    board = pcbnew.LoadBoard(path)
    pads = {p.GetNumber(): p for fp in board.GetFootprints() for p in fp.Pads()}
    kw = dict(floor_mm=0.153, half_width_mm=0.0765, layer_id=pcbnew.F_Cu)
    relaxed, _ = _foreign_pad_margins(board, pads["B5"], **kw)
    strict, _ = _foreign_pad_margins(board, pads["B5"], strict_same_fp=True, **kw)
    # Relaxed: collision-only (half_width + 0.05) vs the sibling pad.
    assert relaxed[0][1] == pcbnew.FromMM(0.0765 + 0.05)
    # Strict: the full pair clearance (the larger of the two pads' resolved
    # clearances) -- the verify DRC does not waive a stub grazing a
    # same-footprint pad.
    pair = max(0.153,
               _own_clearance_mm(pads["B5"], pcbnew.F_Cu, 0.153),
               _own_clearance_mm(pads["B6"], pcbnew.F_Cu, 0.153))
    assert strict[0][1] == pcbnew.FromMM(pair) and pair > 0.153


def test_tip_via_near_same_net_via_is_skipped_or_blocked(tmp_path):
    # via_at_end endgames: ON a same-net via -> redundant (track stamps, no
    # second drill); NEAR one (0.35 mm) -> hole-to-hole blocked, whole spec
    # dropped BEFORE any segment stamps (a stub with no plane via is dead
    # copper).
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, (5.0, 10.0), {"B5": ("GND", 8.0, 10.0)})
    board = pcbnew.LoadBoard(path)
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(10.0), _mm(10.0)))
    via.SetDrill(_mm(0.3))
    try:
        via.SetWidth(_mm(0.6))
    except TypeError:
        via.SetWidth(pcbnew.F_Cu, _mm(0.6))
    via.SetNet(board.GetNetInfo().GetNetItem("GND"))
    board.Add(via)
    board.Save(path)

    # End exactly on the existing via: redundant -> track yes, via no.
    res = add_breakout_stubs(path, [BreakoutSpec(
        ref="J1", pad="B5", waypoints=[(10.0, 10.0)], via_at_end=True)])
    assert res["stubs"] == 1 and res["segments"] == 1 and res["vias"] == 0

    # End 0.35 mm from the via: not touching, and the two drills would sit
    # inside the hole-to-hole minimum -> the spec must drop whole.
    path2 = str(tmp_path / "b2.kicad_pcb")
    _board(path2, (5.0, 10.0), {"B5": ("GND", 8.0, 10.0)})
    board = pcbnew.LoadBoard(path2)
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(10.35), _mm(10.0)))
    via.SetDrill(_mm(0.3))
    try:
        via.SetWidth(_mm(0.6))
    except TypeError:
        via.SetWidth(pcbnew.F_Cu, _mm(0.6))
    via.SetNet(board.GetNetInfo().GetNetItem("GND"))
    board.Add(via)
    board.Save(path2)

    res = add_breakout_stubs(path2, [BreakoutSpec(
        ref="J1", pad="B5", waypoints=[(10.0, 10.0)], via_at_end=True)])
    assert res["stubs"] == 0 and res["segments"] == 0
    assert any("via_blocked" in s for s in res["skipped"])
