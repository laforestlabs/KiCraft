"""Edge-pin flush math must use the BODY bbox, not the footprint origin.

``width_mm``/``height_mm`` describe the body bbox centered on ``body_center``;
``pos`` is the footprint origin, which for a body-behind-mouth connector (BNC,
barrel jack, deep screw terminal) sits at the PADS, several mm from the body
center. ``_connector_edge_x/_y`` formerly flushed ``pos +/- half`` against the
canvas edge, leaving the physical mouth inboard by the body offset -- run_01's
BNC (offset 13.3mm) pinned "flush" with its mouth 13mm inside the canvas, the
packer filled the phantom gap with the trim pot, and the composed board
stranded the mouth behind it (self-eval 2026-07-19). The same mechanism at
1-2mm scale produced the KC-YXQ4EC inset-mouth family.
"""
from __future__ import annotations

from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Pad,
    Point,
)


def _bnc_like(ref: str, *, pos: Point, body_center: Point,
              w: float, h: float, opening: float) -> Component:
    return Component(
        ref=ref, value="BNC", pos=pos, rotation=0.0, layer=Layer.FRONT,
        width_mm=w, height_mm=h, kind="connector",
        body_center=body_center, opening_direction=opening,
        pads=[Pad(ref=ref, pad_id="1", pos=Point(pos.x, pos.y), net="N",
                  layer=Layer.FRONT, size_mm=Point(1, 1))],
    )


def _pin(comp: Component, edge: str, outline=(Point(0, 0), Point(60, 40))):
    state = BoardState(
        components={comp.ref: comp},
        board_outline=outline,
        keepout_rects=[],
    )
    cfg = {
        "component_zones": {comp.ref: {"edge": edge}},
        "connector_edge_inset_mm": 1.0,
        "edge_jitter_mm": 0.0,
        "placement_clearance_mm": 0.0,
    }
    solver = PlacementSolver(state, cfg, seed=0)
    solver._pin_edge_components(state.components)
    return comp


def test_right_edge_pin_flushes_body_not_origin():
    # Mouth points right (opening 0), pads at pos, body center 10mm BEHIND
    # the pads. Physical right face = body_center.x + w/2 must land at
    # br.x - inset; the old pos-based math left it 10mm inboard.
    comp = _bnc_like("J2", pos=Point(30, 20), body_center=Point(20, 20),
                     w=30.0, h=10.0, opening=0.0)
    _pin(comp, "right")
    body_right = comp.body_center.x + comp.width_mm / 2
    assert abs(body_right - (60 - 1.0)) < 0.2, (
        f"body right face at {body_right}, want flush at 59.0"
    )


def test_bottom_edge_pin_flushes_body_not_origin():
    comp = _bnc_like("J3", pos=Point(30, 20), body_center=Point(30, 14),
                     w=10.0, h=18.0, opening=90.0)
    _pin(comp, "bottom")
    body_bottom = comp.body_center.y + comp.height_mm / 2
    assert abs(body_bottom - (40 - 1.0)) < 0.2, (
        f"body bottom face at {body_bottom}, want flush at 39.0"
    )


def test_symmetric_connector_unchanged():
    # No body offset: the fix must be a no-op (pos-based == body-based).
    comp = _bnc_like("J4", pos=Point(30, 20), body_center=Point(30, 20),
                     w=10.0, h=8.0, opening=0.0)
    _pin(comp, "right")
    assert abs((comp.pos.x + 5.0) - 59.0) < 0.2
