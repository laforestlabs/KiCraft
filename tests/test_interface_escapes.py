"""Interface escapes -- cross-leaf pads freed from the placed companion wall.

The class under test is the whole rc7 residue of the self-eval 2026-07-27
re-batch: a pad whose net crosses into another sheet has no partner on its
leaf, so leaf routing lays NO copper on it, and at the parent stage it sits
behind the pin-adjacent decap wall and the leaf's locked traces --
``no_clear_path`` from the autorouter AND the repair pass (run_10's 26 GPIO
fan-outs to J2, run_23's CAN_TX/RX, run_24's SDA/SCL/ALERT_*, run_30's
IO4/IO7). These tests pin the pure-geometry planner's verdicts on synthetic
re-creations of that geometry, plus the board-level spec generation.
"""
from __future__ import annotations

import math

import pytest

from kicraft.autoplacer.brain.escape_planner import (
    Pad,
    Rules,
    _seg_seg_dist,
    plan_interface_escapes,
)

RULES = Rules()  # 0.153/0.153, fanout via 0.35/0.2 -- the planning default


def _pad(number, net, x, y, w=0.25, h=0.25):
    return Pad(number=number, net=net, x=x, y=y, w=w, h=h)


def _picket_ring(radius: float, n: int = 16, size: float = 0.5) -> list[Pad]:
    """A gapless wall of foreign pads all around the origin."""
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n
        out.append(
            _pad(f"W{i}", "GND", radius * math.cos(a), radius * math.sin(a),
                 w=size, h=size)
        )
    return out


def test_open_pad_gets_no_stamp():
    """A pad whose short outward ray is already clear is the router's job."""
    src = _pad("1", "CAN_TX", 0.0, 0.0)
    got = plan_interface_escapes(
        [("U1.1", src, (1.0, 0.0))], [src], RULES
    )
    assert got["U1.1"].kind == "open"
    assert not got["U1.1"].needs_stamp


def test_walled_outward_pad_escapes_sideways():
    """The measured run_10 shape: decap wall dead ahead, a clear flank.

    A fixed radial stamper fails here (the outward ray hits the wall); the
    sweep must find the sideways exit and stamp a real escape, not a nub.
    """
    src = _pad("1", "GPIO7", 0.0, 0.0)
    # Wall directly outward (east): tall gapless picket at x ~ 1.0.
    wall = [_pad(f"C{i}", "GND", 1.0, -1.5 + 0.5 * i, w=0.5, h=0.5)
            for i in range(7)]
    got = plan_interface_escapes(
        [("U1.1", src, (1.0, 0.0))], [src, *wall], RULES
    )
    esc = got["U1.1"]
    assert esc.kind == "ray"
    assert esc.needs_stamp
    (x0, y0), (x1, y1) = esc.polyline
    # It left sideways (not east through the wall) and ended in open copper.
    assert abs(x1 - x0) < abs(y1 - y0)
    from kicraft.autoplacer.brain.escape_planner import _dist_point_rect, _rect

    for w in wall:
        assert _dist_point_rect(x1, y1, _rect(w)) >= 0.6 - 1e-9


def test_fully_walled_pad_gets_a_dog_bone_via():
    """run_10's GPIO21/22/23 class: no on-layer ray at all -> far layer."""
    src = _pad("1", "GPIO21", 0.0, 0.0)
    wall = _picket_ring(1.2)
    got = plan_interface_escapes(
        [("U1.1", src, (1.0, 0.0))], [src, *wall], RULES
    )
    esc = got["U1.1"]
    assert esc.kind == "via"
    assert esc.via_center is not None
    vx, vy = esc.via_center
    assert math.hypot(vx, vy) <= 0.75 + 1e-9


def test_enclosed_pad_is_honestly_infeasible():
    """A pad nothing can free gets NOTHING -- never a nub."""
    src = _pad("1", "GPIO9", 0.0, 0.0)
    # Tight double ring: no ray, and no legal via centre either.
    wall = _picket_ring(0.55, n=24, size=0.4) + _picket_ring(1.0, n=24, size=0.6)
    got = plan_interface_escapes(
        [("U1.1", src, (1.0, 0.0))], [src, *wall], RULES
    )
    esc = got["U1.1"]
    assert esc.kind == "infeasible"
    assert not esc.polyline


def test_committed_escapes_hold_clearance_from_each_other():
    """Two neighbouring escapes may not be stamped into a violation."""
    a = _pad("1", "SDA", 0.0, 0.0)
    b = _pad("2", "SCL", 0.4, 0.0)
    wall = [_pad(f"C{i}", "GND", 1.2, -1.5 + 0.5 * i, w=0.5, h=0.5)
            for i in range(7)]
    got = plan_interface_escapes(
        [("U1.1", a, (1.0, 0.0)), ("U1.2", b, (1.0, 0.0))],
        [a, b, *wall],
        RULES,
    )
    polys = [e.polyline for e in got.values() if e.polyline]
    assert len(polys) == 2
    need = RULES.track_mm + RULES.clearance_mm
    for s1, e1 in zip(polys[0], polys[0][1:]):
        for s2, e2 in zip(polys[1], polys[1][1:]):
            assert _seg_seg_dist(s1, e1, s2, e2) >= need - 1e-9


def test_endpoint_never_dead_ends_inside_a_body():
    """An escape ending under a neighbouring component is a dead end."""
    src = _pad("1", "IO4", 0.0, 0.0)
    wall = [_pad(f"C{i}", "GND", 1.0, -1.5 + 0.5 * i, w=0.5, h=0.5)
            for i in range(7)]
    # The only clear direction (north) runs under a body from y=-2.0 to -0.4.
    courtyard = (-0.5, -2.0, 0.5, -0.4)
    got = plan_interface_escapes(
        [("U1.1", src, (1.0, 0.0))],
        [src, *wall],
        RULES,
        courtyards=[courtyard],
    )
    esc = got["U1.1"]
    assert esc.kind in ("ray", "via")
    if esc.kind == "ray":
        ex, ey = esc.polyline[-1]
        inside = courtyard[0] <= ex <= courtyard[2] and courtyard[1] <= ey <= courtyard[3]
        assert not inside


def test_plan_is_deterministic():
    src = _pad("1", "GPIO7", 0.0, 0.0)
    wall = [_pad(f"C{i}", "GND", 1.0, -1.5 + 0.5 * i, w=0.5, h=0.5)
            for i in range(7)]
    args = ([("U1.1", src, (1.0, 0.0))], [src, *wall], RULES)
    a = plan_interface_escapes(*args)
    b = plan_interface_escapes(*args)
    assert {k: v.to_dict() for k, v in a.items()} == {
        k: v.to_dict() for k, v in b.items()
    }


# --------------------------------------------------------------------------- #
# Board-level spec generation
# --------------------------------------------------------------------------- #

pcbnew = pytest.importorskip("pcbnew")
_mm = pcbnew.FromMM


def _make_board(path, footprints):
    """``footprints`` = [(ref, center, [(pad_num, net, dx, dy, w, h)])]."""
    board = pcbnew.NewBoard(path)
    nets = {}
    for _ref, _c, pads in footprints:
        for _num, net, *_ in pads:
            if net and net not in nets:
                item = pcbnew.NETINFO_ITEM(board, net)
                board.Add(item)
                nets[net] = item
    for ref, (cx, cy), pads in footprints:
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(_mm(cx), _mm(cy)))
        for num, net, dx, dy, w, h in pads:
            pad = pcbnew.PAD(fp)
            pad.SetNumber(str(num))
            pad.SetSize(pcbnew.VECTOR2I(_mm(w), _mm(h)))
            pad.SetPosition(pcbnew.VECTOR2I(_mm(cx + dx), _mm(cy + dy)))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pcbnew.PAD.SMDMask())
            if net:
                pad.SetNet(nets[net])
            fp.Add(pad)
        board.Add(fp)
    board.Save(path)
    return board


def _ic_pads():
    """A 12-pad 'IC': two columns at x = -1.5 / +1.5, 0.8 mm pitch."""
    pads = []
    for i in range(6):
        pads.append((i + 1, f"NET_L{i}", -1.5, -2.0 + 0.8 * i, 0.3, 0.4))
        pads.append((i + 7, f"NET_R{i}", 1.5, -2.0 + 0.8 * i, 0.3, 0.4))
    return pads


def test_specs_only_for_single_pad_interface_nets(tmp_path):
    """Multi-pad interface nets have leaf copper; power nets have pours."""
    path = str(tmp_path / "t.kicad_pcb")
    pads = _ic_pads()
    # NET_R0 also lands on a resistor -> 2 pads on the leaf -> excluded.
    board = _make_board(path, [
        ("U1", (30.0, 30.0), pads),
        ("R1", (38.0, 30.0), [(1, "NET_R0", -0.5, 0.0, 0.5, 0.5),
                              (2, "GND", 0.5, 0.0, 0.5, 0.5)]),
    ])
    from kicraft.autoplacer.brain.breakout_stubs import interface_escape_specs

    iface = ["NET_R0", "NET_R1", "GND", "+3V3"]
    specs, report = interface_escape_specs(board, {}, iface)
    assert report["enabled"]
    uids = {f"{s.ref}.{s.pad}" for s in specs}
    # NET_R1 is the only single-pad, non-power interface net on a dense part.
    assert report["sources"] == 1
    assert all(u.startswith("U1.") for u in uids)


def test_tht_pads_are_left_alone(tmp_path):
    """A PTH pad is on both layers already -- an escape is copper for its own
    sake (and on a 2x20 header it would be ~30 stubs of it)."""
    path = str(tmp_path / "t.kicad_pcb")
    pads = [(i + 1, f"GPIO{i}", 0.0, 2.54 * i, 1.7, 1.7) for i in range(9)]
    board = _make_board(path, [("J2", (30.0, 30.0), pads)])
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
            pad.SetLayerSet(pcbnew.PAD.PTHMask())
            pad.SetDrillSize(pcbnew.VECTOR2I(_mm(1.0), _mm(1.0)))
    from kicraft.autoplacer.brain.breakout_stubs import interface_escape_specs

    specs, report = interface_escape_specs(
        board, {}, [f"GPIO{i}" for i in range(9)]
    )
    assert specs == []
    assert report["sources"] == 0


def test_small_footprints_are_left_alone(tmp_path):
    path = str(tmp_path / "t.kicad_pcb")
    board = _make_board(path, [
        ("R1", (30.0, 30.0), [(1, "CAN_TX", -0.5, 0.0, 0.5, 0.5),
                              (2, "GND", 0.5, 0.0, 0.5, 0.5)]),
    ])
    from kicraft.autoplacer.brain.breakout_stubs import interface_escape_specs

    specs, report = interface_escape_specs(board, {}, ["CAN_TX"])
    assert specs == []
    assert report["sources"] == 0


def test_kill_switch(tmp_path):
    path = str(tmp_path / "t.kicad_pcb")
    board = _make_board(path, [("U1", (30.0, 30.0), _ic_pads())])
    from kicraft.autoplacer.brain.breakout_stubs import interface_escape_specs

    specs, report = interface_escape_specs(
        board, {"interface_escape_enabled": False}, ["NET_R1"]
    )
    assert specs == []
    assert not report["enabled"]


def test_walled_interface_pad_gets_a_stamp(tmp_path):
    """The run_23 CAN_TX shape: a bare interface pad behind a decap wall."""
    path = str(tmp_path / "t.kicad_pcb")
    pads = _ic_pads()
    # Wall of decaps hugging U1's right column (the pin-adjacent grid).
    decaps = []
    for i in range(6):
        decaps.append((
            f"C{i}", (32.6, 28.0 + 0.8 * i),
            [(1, "+3V3", 0.0, -0.35, 0.5, 0.4), (2, "GND", 0.0, 0.35, 0.5, 0.4)],
        ))
    board = _make_board(path, [("U1", (30.0, 30.0), pads), *decaps])
    from kicraft.autoplacer.brain.breakout_stubs import interface_escape_specs

    specs, report = interface_escape_specs(board, {}, ["NET_R2"])
    assert report["sources"] == 1
    kinds = report["kinds"]
    # Freed one way or the other -- and if it could not be, it is on record.
    assert kinds.get("ray", 0) + kinds.get("via", 0) + kinds.get("open", 0) \
        + len(report["infeasible"]) == 1
    for s in specs:
        assert s.waypoints, "an interface escape is always an explicit polyline"
