"""Connector-bank orientation: short single-row pin headers on an edge sit with
their pin axis PERPENDICULAR to the edge, so a bank packs tight and its
shared-net (GND) pads line up into one uninterrupted strip.

Owning plan: pcb-area-compaction-plan Phase 6. Root case: KC-8A3US3, a 16x 1x3
servo-header board that shipped 197x30mm and failed fab on a fragmented GND pour
because pins-parallel interleaves each header's signal/power pads on the GND line.
"""
from __future__ import annotations

from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.types import Component, Layer, Pad, Point


def _pin_header(n_pins: int, pitch: float = 2.54, *, rows: int = 1) -> Component:
    """A vertical 1xN (or 2xN) header at rotation 0: pins run along +y.

    Body long axis is vertical (height > width), matching a real
    PinHeader_1xNN_Vertical footprint at 0 degrees.
    """
    pads: list[Pad] = []
    for r in range(rows):
        for i in range(n_pins):
            pads.append(
                Pad(
                    ref="J1",
                    pad_id=str(r * n_pins + i + 1),
                    pos=Point(r * pitch, i * pitch),
                    net="GND" if i == n_pins - 1 else f"SIG{i}",
                    layer=Layer.FRONT,
                )
            )
    span = (n_pins - 1) * pitch
    return Component(
        ref="J1",
        value=f"Conn_{rows}x{n_pins:02d}",
        pos=Point(0.0, 0.0),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=2.5 + (rows - 1) * pitch,
        height_mm=span + 2.54,
        kind="connector",
        pads=pads,
    )


CFG_ON = {"connector_perp_orientation": True}
CFG_OFF = {"connector_perp_orientation": False}


def test_short_header_goes_perpendicular_on_bottom_edge():
    # A 1x3 servo header: legacy orients pins PARALLEL (rot 90); the bank fix
    # orients them PERPENDICULAR (rot 0, pins pointing into the board).
    hdr = _pin_header(3)
    assert PlacementSolver._best_rotation_for_edge(hdr, "bottom", CFG_ON) == 0.0
    assert PlacementSolver._best_rotation_for_edge(hdr, "bottom", CFG_OFF) == 90.0
    # cfg=None is legacy (used by the static-call test path).
    assert PlacementSolver._best_rotation_for_edge(hdr, "bottom", None) == 90.0


def test_short_header_perpendicular_on_left_edge():
    # On a vertical edge, perpendicular means pins run horizontally (rot 90).
    hdr = _pin_header(3)
    assert PlacementSolver._best_rotation_for_edge(hdr, "left", CFG_ON) == 90.0
    assert PlacementSolver._best_rotation_for_edge(hdr, "left", CFG_OFF) == 0.0


def test_two_pin_header_left_alone():
    # 2-pin (screw terminal / power header): keep the default along-edge
    # orientation -- wire cages want to face off-board.
    hdr = _pin_header(2)
    assert PlacementSolver._best_rotation_for_edge(hdr, "bottom", CFG_ON) == 90.0
    assert not PlacementSolver._connector_wants_perp_axis(hdr, CFG_ON)


def test_long_header_left_alone():
    # A lone 1x20 GPIO header perpendicular would stab ~48mm into the board.
    hdr = _pin_header(20)
    assert PlacementSolver._best_rotation_for_edge(hdr, "bottom", CFG_ON) == 90.0
    assert not PlacementSolver._connector_wants_perp_axis(hdr, CFG_ON)


def test_multirow_header_left_alone():
    # 2xN IDC is not a single row; keep it along the edge.
    hdr = _pin_header(5, rows=2)
    assert not PlacementSolver._connector_wants_perp_axis(hdr, CFG_ON)


def test_non_connector_left_alone():
    hdr = _pin_header(3)
    hdr.kind = "ic"
    assert not PlacementSolver._connector_wants_perp_axis(hdr, CFG_ON)


def test_mouthed_connector_unaffected():
    # A connector with a detected opening still faces its mouth outward,
    # regardless of the bank setting.
    hdr = _pin_header(3)
    hdr.opening_direction = 90.0
    assert not PlacementSolver._connector_wants_perp_axis(hdr, CFG_ON)
    # bottom edge, front layer: mouth-out rotation is 0 either way.
    assert PlacementSolver._best_rotation_for_edge(hdr, "bottom", CFG_ON) == 0.0


def test_default_config_enables_bank_orientation():
    from kicraft.autoplacer.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.get("connector_perp_orientation") is True
    hdr = _pin_header(3)
    assert PlacementSolver._best_rotation_for_edge(hdr, "bottom", DEFAULT_CONFIG) == 0.0
