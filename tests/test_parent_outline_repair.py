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


def _state(board_state, edge_refs=()):
    composition = SimpleNamespace(board_state=board_state)
    return SimpleNamespace(
        composition=composition,
        edge_constrained_refs=frozenset(edge_refs),
        geometry_validation=None,
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
