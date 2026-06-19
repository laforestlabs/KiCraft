"""Unit tests for compose's parent-outline repair (E2E Finding 1b).

The constraint-aware outline can come out smaller than the placed-content
bbox, leaving footprints/pads/copper outside ``Edge.Cuts`` so FreeRouting
produces no SES. ``_repair_parent_outline`` grows the outline to enclose all
placed geometry (matching ``_validate_parent_geometry``'s rule) before
stamping, while preserving edge-connector body overhang and never shrinking
an outline that already encloses everything.
"""
from __future__ import annotations

from types import SimpleNamespace

from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Pad,
    Point,
    TraceSegment,
    Via,
)
from kicraft.cli.compose_subcircuits import (
    _repair_parent_outline,
    _validate_parent_geometry,
)


def _comp(ref, cx, cy, w, h, pads=None, kind=""):
    return Component(
        ref=ref,
        value="",
        pos=Point(cx, cy),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=w,
        height_mm=h,
        pads=pads or [],
        body_center=Point(cx, cy),
        kind=kind,
    )


def _pad(ref, x, y, w=1.0, h=1.0):
    return Pad(
        ref=ref, pad_id="1", pos=Point(x, y), net="", layer=Layer.FRONT,
        size_mm=Point(w, h),
    )


def _state(board_state, edge_refs=(), manual_outline=None, connector_sides=()):
    composition = SimpleNamespace(board_state=board_state)
    return SimpleNamespace(
        composition=composition,
        edge_constrained_refs=frozenset(edge_refs),
        edge_zoned_outline_sides=frozenset(connector_sides),
        geometry_validation=None,
        manual_outline=manual_outline,
    )


def test_repair_grows_undersized_outline_to_enclose_bodies():
    # Outline is 10x10 at origin, but a component body sits well outside it.
    bs = BoardState(
        components={"U1": _comp("U1", cx=20.0, cy=5.0, w=4.0, h=4.0)},
        board_outline=(Point(0.0, 0.0), Point(10.0, 10.0)),
    )
    state = _state(bs)
    # Pre-repair: component is outside -> validation rejects.
    assert _validate_parent_geometry(state)["accepted"] is False

    result = _repair_parent_outline(state, margin_mm=2.0)
    assert result["repaired"] is True
    tl, br = bs.board_outline
    # Right edge must clear the component body (22.0) + margin (2.0).
    assert br.x >= 22.0 + 2.0 - 1e-6
    # Only grows: left/top untouched (geometry never extended past them).
    assert tl.x <= 0.0 and tl.y <= 0.0
    # Post-repair the board now validates.
    assert _validate_parent_geometry(state)["accepted"] is True


def test_repair_is_noop_when_outline_already_encloses():
    bs = BoardState(
        components={"R1": _comp("R1", cx=10.0, cy=10.0, w=2.0, h=2.0,
                                pads=[_pad("R1", 10.0, 10.0)])},
        board_outline=(Point(0.0, 0.0), Point(20.0, 20.0)),
    )
    state = _state(bs)
    before = bs.board_outline
    result = _repair_parent_outline(state, margin_mm=2.0)
    assert result["repaired"] is False
    assert bs.board_outline == before


def test_edge_connector_body_overhang_does_not_grow_outline():
    # Edge connector body pokes left of the outline (flush mount), but its
    # pads sit inboard. The body is exempt, so the outline must NOT grow to
    # swallow the housing overhang.
    j1 = _comp(
        "J1", cx=-1.0, cy=10.0, w=6.0, h=6.0,
        pads=[_pad("J1", 3.0, 10.0)],  # pad inboard, inside outline
        kind="connector",
    )
    bs = BoardState(
        components={"J1": j1},
        board_outline=(Point(0.0, 0.0), Point(20.0, 20.0)),
    )
    state = _state(bs, edge_refs={"J1"})
    result = _repair_parent_outline(state, margin_mm=2.0)
    # Body overhang to x=-4 is exempt; pad at x=3 is already inside -> no grow.
    assert result["repaired"] is False
    assert bs.board_outline[0].x == 0.0


def test_connector_edge_not_buried_by_inboard_neighbor():
    # USB-C J1 mouth defines the LEFT edge at x=0 (outline already snapped
    # there). A neighbor cap C1 sits 0.5mm inboard of the mouth. Without the
    # connector-side rule, the repair would grow left to C1.body - 2mm margin
    # = -1.5, burying the port 1.5mm under FR4. With it, the left edge stays
    # at the mouth (x=0) because C1 is inboard and its copper is inside.
    j1 = _comp(
        "J1", cx=2.0, cy=10.0, w=4.0, h=6.0,   # body x in [0,4]; mouth at x=0
        pads=[_pad("J1", 3.0, 10.0)],          # pad well inboard
        kind="connector",
    )
    c1 = _comp(
        "C1", cx=1.5, cy=14.0, w=2.0, h=1.5,   # body x in [0.5, 2.5], inboard
        pads=[_pad("C1", 1.5, 14.0, w=1.0, h=1.0)],
    )
    bs = BoardState(
        components={"J1": j1, "C1": c1},
        board_outline=(Point(0.0, 0.0), Point(20.0, 20.0)),
    )
    state = _state(bs, edge_refs={"J1"}, connector_sides={"left"})
    result = _repair_parent_outline(state, margin_mm=2.0)
    # Left edge must NOT move out past the mouth (x=0); the port stays flush.
    assert bs.board_outline[0].x >= -1e-6, (
        f"left edge buried to {bs.board_outline[0].x}, port unmateable"
    )
    # And nothing is left outside the board.
    assert _validate_parent_geometry(state)["accepted"] is True


def test_connector_side_still_encloses_geometry_beyond_mouth():
    # A stray passive C1 placed slightly BEYOND the mouth (x=-1) must still be
    # enclosed (fabricable). The connector-side floor leaves
    # pad_edge_clearance_mm (0.2) of copper-to-edge clearance from the placed
    # copper -- so the leftmost copper (C1 body at x=-1.0) lands the edge at
    # -1.2, not the full breathing-room margin (-3.0) that would bury the port,
    # and not zero margin (-1.0) that puts C1's copper on the cut line.
    j1 = _comp("J1", cx=2.0, cy=10.0, w=4.0, h=6.0,
               pads=[_pad("J1", 3.0, 10.0)], kind="connector")
    c1 = _comp("C1", cx=-0.5, cy=14.0, w=1.0, h=1.0,  # body x in [-1.0, 0.0]
               pads=[_pad("C1", -0.5, 14.0, w=0.6, h=0.6)])
    bs = BoardState(
        components={"J1": j1, "C1": c1},
        board_outline=(Point(0.0, 0.0), Point(20.0, 20.0)),
    )
    state = _state(bs, edge_refs={"J1"}, connector_sides={"left"})
    _repair_parent_outline(state, margin_mm=2.0, pad_edge_clearance_mm=0.2)
    # Encloses C1 (x=-1.0) plus 0.2mm copper-to-edge clearance.
    assert bs.board_outline[0].x <= -1.2 + 1e-6
    assert bs.board_outline[0].x >= -1.2 - 1e-6
    assert _validate_parent_geometry(state)["accepted"] is True


def test_connector_side_clears_flush_pad_by_clearance():
    # The core fix: an edge-mount connector whose edge-facing PAD sits at its
    # body front (a BNC GND shield) -- pad flush with the constraint-aware
    # outline -- gets the board edge pulled pad_edge_clearance_mm outboard of the
    # pad, so the pad clears the cut line instead of landing on it.
    # J1 body left edge at x=0 (flush with the outline), GND pad also reaching x=0.
    j1 = _comp("J1", cx=2.0, cy=10.0, w=4.0, h=6.0,
               pads=[_pad("J1", 0.5, 10.0, w=1.0, h=1.0)],  # pad x in [0.0, 1.0]
               kind="connector")
    bs = BoardState(
        components={"J1": j1},
        board_outline=(Point(0.0, 0.0), Point(20.0, 20.0)),  # left edge at pad
    )
    state = _state(bs, edge_refs={"J1"}, connector_sides={"left"})
    _repair_parent_outline(state, margin_mm=2.0, pad_edge_clearance_mm=0.2)
    # Edge pulled to pad_left (0.0) - 0.2 = -0.2 so the pad clears by 0.2mm.
    assert abs(bs.board_outline[0].x - (-0.2)) < 1e-6, bs.board_outline[0].x
    assert _validate_parent_geometry(state)["accepted"] is True


def test_repair_encloses_stamped_traces_and_vias():
    bs = BoardState(
        components={"R1": _comp("R1", cx=5.0, cy=5.0, w=2.0, h=2.0)},
        traces=[TraceSegment(start=Point(5.0, 5.0), end=Point(40.0, 5.0),
                             layer=Layer.FRONT, net="N1", width_mm=0.15)],
        vias=[Via(pos=Point(40.0, 5.0), net="N1")],
        board_outline=(Point(0.0, 0.0), Point(10.0, 10.0)),
    )
    state = _state(bs)
    result = _repair_parent_outline(state, margin_mm=2.0)
    assert result["repaired"] is True
    assert bs.board_outline[1].x >= 40.0 + 2.0 - 1e-6
    assert _validate_parent_geometry(state)["accepted"] is True


def test_manual_outline_is_authoritative_no_grow_and_loud_validation():
    """Manual mode: the user's outline must never be silently grown.

    A component outside the user-drawn outline leaves the outline
    untouched and fails geometry validation instead (the editor
    surfaces the violation; routing stays gated)."""
    bs = BoardState(
        components={"U1": _comp("U1", cx=20.0, cy=5.0, w=4.0, h=4.0)},
        board_outline=(Point(0.0, 0.0), Point(10.0, 10.0)),
    )
    manual = {
        "shape": "rect",
        "min": {"x": 0.0, "y": 0.0},
        "max": {"x": 10.0, "y": 10.0},
    }
    state = _state(bs, manual_outline=manual)
    result = _repair_parent_outline(state, margin_mm=2.0)
    assert result["repaired"] is False
    assert "authoritative" in result["reason"]
    assert bs.board_outline == (Point(0.0, 0.0), Point(10.0, 10.0))
    assert _validate_parent_geometry(state)["accepted"] is False


def test_manual_circle_outline_flags_geometry_in_aabb_corner():
    """A part inside the AABB but outside the circle must be rejected."""
    bs = BoardState(
        components={"U1": _comp("U1", cx=4.0, cy=4.0, w=4.0, h=4.0)},
        board_outline=(Point(0.0, 0.0), Point(50.0, 50.0)),
    )
    manual = {
        "shape": "circle",
        "min": {"x": 0.0, "y": 0.0},
        "max": {"x": 50.0, "y": 50.0},
    }
    state = _state(bs, manual_outline=manual)
    validation = _validate_parent_geometry(state)
    assert validation["outline_shape"] == "circle"
    assert validation["accepted"] is False

    # The same part at the circle's center passes.
    bs_ok = BoardState(
        components={"U1": _comp("U1", cx=25.0, cy=25.0, w=4.0, h=4.0)},
        board_outline=(Point(0.0, 0.0), Point(50.0, 50.0)),
    )
    state_ok = _state(bs_ok, manual_outline=manual)
    assert _validate_parent_geometry(state_ok)["accepted"] is True
