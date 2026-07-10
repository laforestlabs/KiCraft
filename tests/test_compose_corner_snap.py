"""WS4: corner/edge snap of parent-local mounting holes must be collision-aware.

A raw corner snap slid a mounting hole onto the exact cluster corner -- where a
corner leaf's header pads sit -- stamping the hole's PTH pad on leaf copper at
0.0 mm (encoder-oled-panel's ``candidate-search ... shorts=10..16`` abort).
"""

from __future__ import annotations

import pytest

pytest.importorskip("pcbnew", reason="KiCad Python bindings not available")

from kicraft.cli._compose_geometry import _bbox_disjoint
from kicraft.cli.compose_subcircuits import _collision_aware_corner_snap
from kicraft.autoplacer.brain.types import Component, Layer, Pad, Point


def _hole(ref: str, cx: float, cy: float, size: float = 2.0) -> Component:
    return Component(
        ref=ref, value=ref, pos=Point(cx, cy), rotation=0.0, layer=Layer.FRONT,
        width_mm=size, height_mm=size, kind="mounting_hole", pads=[],
        body_center=Point(cx, cy),
    )


def _connector(ref: str, cx: float, cy: float, w: float = 4.0, h: float = 4.0) -> Component:
    return Component(
        ref=ref, value=ref, pos=Point(cx, cy), rotation=0.0, layer=Layer.FRONT,
        width_mm=w, height_mm=h, kind="connector",
        pads=[Pad(ref=ref, pad_id="1", pos=Point(cx, cy), net="GND", layer=Layer.FRONT)],
        body_center=Point(cx, cy),
    )


def test_corner_snap_stops_before_leaf_pads():
    # J1 (leaf connector) sits at the corner target; H1 (mounting hole) snaps
    # toward it. The hole must stop before overlapping J1's courtyard.
    j1 = _connector("J1", 10.0, 10.0)  # courtyard (8,8)-(12,12)
    h1 = _hole("H1", 20.0, 20.0)       # courtyard (19,19)-(21,21)
    solved = {"J1": j1, "H1": h1}

    moved = _collision_aware_corner_snap(solved, "H1", dx=-10.0, dy=-10.0)

    assert moved  # it slid toward the corner ...
    # ... but its courtyard never overlaps J1's (no pad-on-pad short).
    assert _bbox_disjoint(solved["H1"].bbox(), solved["J1"].bbox())


def test_corner_snap_reaches_clear_target():
    # Nothing at the target -> the hole snaps all the way onto it.
    solved = {"H1": _hole("H1", 20.0, 20.0), "H2": _hole("H2", 100.0, 100.0)}

    assert _collision_aware_corner_snap(solved, "H1", dx=-10.0, dy=-10.0)
    assert solved["H1"].body_center.x == 10.0
    assert solved["H1"].body_center.y == 10.0
