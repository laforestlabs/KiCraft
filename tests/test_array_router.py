"""Tests for the deterministic array daisy-chain router (turn-hop routing)."""
from __future__ import annotations

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.brain.array_router import (  # noqa: E402
    array_daisy_chain_specs,
)
from kicraft.autoplacer.brain.breakout_stubs import (  # noqa: E402
    BreakoutSpec,
    add_breakout_stubs,
)
from kicraft.autoplacer.brain.leaf_routing import (  # noqa: E402
    array_stamp_gate_tripped,
)

_mm = pcbnew.FromMM

# 1515-style 4-pad LED pinout (offsets from body centre, mm), rotation 0.
_PADS = {"1": ("VDD", -0.4, -0.4), "2": ("DOUT", -0.4, 0.4),
         "3": ("GND", 0.4, 0.4), "4": ("DIN", 0.4, -0.4)}


def _grid_board(path, centers, *, outline=14.0):
    """A board with a square outline + a serpentine-chained LED grid.

    *centers* is an ordered list of (ref, x, y) in data-chain order; consecutive
    members get a shared DOUT->DIN data net (Dn.DOUT == D(n+1).DIN) plus the
    global +5V / GND rails.
    """
    board = pcbnew.NewBoard(path)
    netnames = {"+5V", "GND"}
    for i in range(len(centers) - 1):
        netnames.add(f"DATA{i}")  # Dn.DOUT -> D(n+1).DIN

    for n in netnames:
        board.Add(pcbnew.NETINFO_ITEM(board, n))

    def net(n):
        return board.GetNetInfo().GetNetItem(n)

    corners = [(0, 0), (outline, 0), (outline, outline), (0, outline), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)

    for idx, (ref, cx, cy) in enumerate(centers):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(_mm(cx), _mm(cy)))
        board.Add(fp)
        for num, (role, dx, dy) in _PADS.items():
            pad = pcbnew.PAD(fp)
            pad.SetSize(pcbnew.VECTOR2I(_mm(0.3), _mm(0.3)))
            pad.SetPosition(pcbnew.VECTOR2I(_mm(cx + dx), _mm(cy + dy)))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pcbnew.PAD.SMDMask())
            pad.SetNumber(num)
            if role == "VDD":
                pad.SetNet(net("+5V"))
            elif role == "GND":
                pad.SetNet(net("GND"))
            elif role == "DOUT" and idx < len(centers) - 1:
                pad.SetNet(net(f"DATA{idx}"))
            elif role == "DIN" and idx > 0:
                pad.SetNet(net(f"DATA{idx - 1}"))
            fp.Add(pad)
    board.Save(path)
    return path


def test_in_row_hops_are_straight_turn_hops_are_l_routes(tmp_path):
    # 2x2 serpentine: D1(0,0) D2(0,1) | D3(1,1) D4(1,0) in chain order.
    # hops: D1->D2 in-row, D2->D3 TURN (row change), D3->D4 in-row.
    path = str(tmp_path / "g.kicad_pcb")
    _grid_board(path, [("D1", 4.0, 4.0), ("D2", 7.0, 4.0),
                       ("D3", 7.0, 7.0), ("D4", 4.0, 7.0)])
    cfg = {"arrays": [{"refs": ["D1", "D2", "D3", "D4"], "rows": 2, "cols": 2,
                       "serpentine": True}]}
    specs = array_daisy_chain_specs(pcbnew.LoadBoard(path), cfg)

    assert len(specs) == 3, "one tie per daisy-chain hop"
    by_ref = {s.ref: s for s in specs}
    # in-row hops: a single straight waypoint to the target pad
    assert len(by_ref["D1"].waypoints) == 1
    assert len(by_ref["D3"].waypoints) == 1
    # the turn hop (sourced from D2) routes an L/Z path -> multiple waypoints
    assert len(by_ref["D2"].waypoints) == 3, "turn hop is an edge-channel L route"
    # and its channel leg clears the LED column to the right (board edge side)
    chan_x = by_ref["D2"].waypoints[0][0]
    assert chan_x > 7.0, "channel runs out past the row-end column toward the edge"


def test_turn_routing_disabled_falls_back_to_straight(tmp_path):
    path = str(tmp_path / "g.kicad_pcb")
    _grid_board(path, [("D1", 4.0, 4.0), ("D2", 7.0, 4.0),
                       ("D3", 7.0, 7.0), ("D4", 4.0, 7.0)])
    cfg = {"arrays": [{"refs": ["D1", "D2", "D3", "D4"], "rows": 2, "cols": 2}],
           "array_turn_routing": False}
    specs = array_daisy_chain_specs(pcbnew.LoadBoard(path), cfg)
    by_ref = {s.ref: s for s in specs}
    assert len(by_ref["D2"].waypoints) == 1, "turn routing off -> straight tie"


def test_power_nets_not_tied(tmp_path):
    path = str(tmp_path / "g.kicad_pcb")
    _grid_board(path, [("D1", 4.0, 4.0), ("D2", 7.0, 4.0),
                       ("D3", 7.0, 7.0), ("D4", 4.0, 7.0)])
    cfg = {"arrays": [{"refs": ["D1", "D2", "D3", "D4"], "rows": 2, "cols": 2}]}
    specs = array_daisy_chain_specs(pcbnew.LoadBoard(path), cfg)
    # +5V and GND have 4 pads each (not 2) and are power-named -> never tied
    assert len(specs) == 3


# --- Layer 2: stamp-rate gate (array_data_channel_obstructed) ----------------

def _inrow(n):
    """n in-row (single-waypoint) daisy-chain ties D1..Dn."""
    return [BreakoutSpec(ref=f"D{i}", pad="2", waypoints=[(float(i), 0.0)])
            for i in range(1, n + 1)]


def _turn(ref):
    """A row-turn hop: an L/Z route -> multiple waypoints."""
    return BreakoutSpec(ref=ref, pad="2", waypoints=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])


def _skips(refs):
    return [f"{r}.2:waypoint_crosses_pad" for r in refs]


def test_stamp_gate_trips_on_collapsed_inrow_rate():
    cfg = {}
    specs = _inrow(45)  # the KC-NZXXEE in-row count (5x10 -> 5*9)
    # 0/45 in-row stamped -> obstructed -> gate trips.
    assert array_stamp_gate_tripped(specs, _skips([f"D{i}" for i in range(1, 46)]), cfg)
    # a couple of skips is healthy.
    assert not array_stamp_gate_tripped(specs, _skips(["D1", "D2"]), cfg)


def test_stamp_gate_excludes_row_turns():
    # The user's concern: a small array whose ONLY un-stamped hop is a row turn
    # must NOT false-fail. Build 6 in-row hops + one turn; skip only the turn.
    cfg = {"array_min_data_ties_for_gate": 4}
    specs = _inrow(6) + [_turn("D7")]
    assert not array_stamp_gate_tripped(specs, _skips(["D7"]), cfg), \
        "an un-stamped row turn alone must not trip the gate"
    # but if the in-row hops themselves are blocked, it trips.
    assert array_stamp_gate_tripped(
        specs, _skips([f"D{i}" for i in range(1, 7)]), cfg)


def test_stamp_gate_ignores_small_arrays():
    # A 2x2 has 2 in-row hops + 1 turn; even all-skipped it stays below the floor.
    cfg = {}  # default floor 6 in-row ties
    specs = _inrow(2) + [_turn("D3")]
    assert not array_stamp_gate_tripped(specs, _skips(["D1", "D2", "D3"]), cfg)
    assert not array_stamp_gate_tripped([], [], cfg)


def test_stamp_gate_is_configurable():
    specs = _inrow(20)
    allskip = _skips([f"D{i}" for i in range(1, 21)])
    assert not array_stamp_gate_tripped(specs, allskip, {"array_stamp_gate_enabled": False})
    # half stamped trips the default 0.5 floor when 11/20 are skipped (9 stamped).
    assert array_stamp_gate_tripped(specs, _skips([f"D{i}" for i in range(1, 12)]),
                                    {"array_min_stamp_rate": 0.5})
    assert not array_stamp_gate_tripped(specs, _skips([f"D{i}" for i in range(1, 10)]),
                                        {"array_min_stamp_rate": 0.5})


# A 4-pad LED pinout ORIENTED for a left-to-right chain: DOUT faces the next
# LED (top-right) and DIN faces the previous one (top-left), so an in-row hop is
# a clean straight segment across the open channel -- the placement the array
# orienter produces. (The shared _grid_board pinout is unrotated and crosses the
# body, which exercises the spec generator but not clean stamping.)
_ROW_PADS = {"1": ("VDD", -0.4, -0.4), "3": ("GND", 0.4, -0.4),
             "2": ("DOUT", 0.4, 0.4), "4": ("DIN", -0.4, 0.4)}


def _row_board(path, n, *, blockers=()):
    """A single row of *n* chained LEDs (4 mm pitch) with an oriented pinout,
    optionally with foreign +5V pads dropped onto the channel midlines."""
    board = pcbnew.NewBoard(path)
    nets = {"+5V", "GND"} | {f"DATA{i}" for i in range(n - 1)}
    for name in nets:
        board.Add(pcbnew.NETINFO_ITEM(board, name))

    def net(name):
        return board.GetNetInfo().GetNetItem(name)

    corners = [(0, 0), (4 * (n + 1), 0), (4 * (n + 1), 8), (0, 8), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)

    for idx in range(n):
        cx, cy = 4.0 * (idx + 1), 4.0
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(f"D{idx + 1}")
        fp.SetPosition(pcbnew.VECTOR2I(_mm(cx), _mm(cy)))
        board.Add(fp)
        for num, (role, dx, dy) in _ROW_PADS.items():
            pad = pcbnew.PAD(fp)
            pad.SetSize(pcbnew.VECTOR2I(_mm(0.3), _mm(0.3)))
            pad.SetPosition(pcbnew.VECTOR2I(_mm(cx + dx), _mm(cy + dy)))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pcbnew.PAD.SMDMask())
            pad.SetNumber(num)
            if role == "VDD":
                pad.SetNet(net("+5V"))
            elif role == "GND":
                pad.SetNet(net("GND"))
            elif role == "DOUT" and idx < n - 1:
                pad.SetNet(net(f"DATA{idx}"))
            elif role == "DIN" and idx > 0:
                pad.SetNet(net(f"DATA{idx - 1}"))
            fp.Add(pad)

    for j, (bx, by) in enumerate(blockers):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(f"C{j + 1}")
        fp.SetPosition(pcbnew.VECTOR2I(_mm(bx), _mm(by)))
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetSize(pcbnew.VECTOR2I(_mm(0.5), _mm(0.5)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(bx), _mm(by)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber("1")
        pad.SetNet(net("+5V"))
        fp.Add(pad)
    board.Save(path)
    return path


def test_obstructed_channel_collapses_stamp_rate(tmp_path):
    # A single 4-LED row. Without obstruction the in-row hops stamp cleanly;
    # drop a foreign +5V pad onto the data line of every channel and the guard
    # rejects them -- exactly the signal the Layer-2 gate keys on.
    cfg = {"arrays": [{"refs": ["D1", "D2", "D3", "D4"], "rows": 1, "cols": 4}]}
    gate_cfg = {"array_min_data_ties_for_gate": 3}

    clean = _row_board(str(tmp_path / "clean.kicad_pcb"), 4)
    specs = array_daisy_chain_specs(pcbnew.LoadBoard(clean), cfg)
    clean_skips = add_breakout_stubs(clean, specs, cfg=cfg).get("skipped", [])
    assert not array_stamp_gate_tripped(specs, clean_skips, gate_cfg)

    # A +5V pad on the DOUT->DIN line (y=4.4) at each channel midpoint.
    blocked = _row_board(str(tmp_path / "blocked.kicad_pcb"), 4,
                         blockers=[(6.0, 4.4), (10.0, 4.4), (14.0, 4.4)])
    specs2 = array_daisy_chain_specs(pcbnew.LoadBoard(blocked), cfg)
    blocked_skips = add_breakout_stubs(blocked, specs2, cfg=cfg).get("skipped", [])
    assert len(blocked_skips) > len(clean_skips), "the foreign pads block the data ties"
    assert array_stamp_gate_tripped(specs2, blocked_skips, gate_cfg)
