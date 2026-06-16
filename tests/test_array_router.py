"""Tests for the deterministic array daisy-chain router (turn-hop routing)."""
from __future__ import annotations

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.brain.array_router import (  # noqa: E402
    array_daisy_chain_specs,
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
