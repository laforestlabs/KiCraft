"""Regression tests for the pad-extent + physical_bbox refactor.

The composer previously tracked component extent as a mix of three
inconsistent representations -- courtyard only, courtyard + pad
centers, and (only when a leaf PCB was on disk) courtyard + pad
bboxes. This caused J3-style overhang failures: a pad with non-trivial
copper width whose center sat just inside the board outline could land
copper outside the board, and the validator's pad-center check missed
it.

These tests lock in the new contract:

* ``Pad.size_mm`` carries pad copper extent.
* ``Pad.bbox()`` is the AABB of that copper.
* ``Component.physical_bbox()`` is courtyard ∪ all pad bboxes -- the
  one true "where is this component physically" answer.
* Round-trip through solved_layout JSON preserves ``size_mm``.
* Rotation transforms swap pad width/height for orthogonal angles.
"""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.subcircuit_artifacts import serialize_component
from kicraft.autoplacer.brain.subcircuit_instances import _pad_from_dict
from kicraft.autoplacer.brain.types import Component, Layer, Pad, Point


def _pad(
    pos: tuple[float, float],
    *,
    size: tuple[float, float] | None = None,
    ref: str = "U1",
    pad_id: str = "1",
    net: str = "VBUS",
) -> Pad:
    return Pad(
        ref=ref,
        pad_id=pad_id,
        pos=Point(*pos),
        net=net,
        layer=Layer.FRONT,
        size_mm=Point(*size) if size else None,
    )


def _connector(
    *,
    pos: tuple[float, float] = (10.0, 0.0),
    body_w: float = 5.0,
    body_h: float = 4.0,
    pads: list[Pad] | None = None,
) -> Component:
    return Component(
        ref="J3",
        value="HEADER",
        pos=Point(*pos),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=body_w,
        height_mm=body_h,
        pads=pads or [],
        body_center=Point(*pos),
        kind="connector",
    )


# ---------------------------------------------------------------------------
# Pad.bbox
# ---------------------------------------------------------------------------


def test_pad_bbox_from_size():
    pad = _pad((10.0, 5.0), size=(2.0, 1.0))
    tl, br = pad.bbox()
    assert tl == Point(9.0, 4.5)
    assert br == Point(11.0, 5.5)


def test_pad_bbox_legacy_no_size_returns_point():
    """A legacy pad with no size_mm contributes its center only --
    backward compatible with old solved_layout.json artifacts."""
    pad = _pad((10.0, 5.0))
    tl, br = pad.bbox()
    assert tl == Point(10.0, 5.0)
    assert br == Point(10.0, 5.0)


# ---------------------------------------------------------------------------
# Component.physical_bbox
# ---------------------------------------------------------------------------


def test_physical_bbox_pads_inside_courtyard_match_body():
    """Pads inside the courtyard contribute nothing extra."""
    comp = _connector(
        body_w=10.0,
        body_h=10.0,
        pads=[_pad((10.0, 0.0), size=(1.0, 1.0))],
    )
    body_tl, body_br = comp.bbox()
    phys_tl, phys_br = comp.physical_bbox()
    assert phys_tl == body_tl
    assert phys_br == body_br


def test_physical_bbox_pad_extends_past_courtyard():
    """The exact J3-style failure: courtyard 5mm wide, but a pad 2mm
    wide whose center sits at the courtyard's right edge places copper
    1mm outside the courtyard. physical_bbox must include that copper.
    """
    comp = _connector(
        pos=(10.0, 0.0),
        body_w=5.0,  # courtyard 7.5..12.5
        body_h=4.0,
        pads=[
            # pad center at 12.5 (courtyard right edge), 2mm wide,
            # copper reaches x=13.5
            _pad((12.5, 0.0), size=(2.0, 1.0)),
        ],
    )
    body_tl, body_br = comp.bbox()
    phys_tl, phys_br = comp.physical_bbox()

    assert body_br.x == pytest.approx(12.5)
    assert phys_br.x == pytest.approx(13.5), (
        "physical_bbox must extend past the courtyard by half the pad width"
    )
    # Other axes unaffected by the right-side pad
    assert phys_tl.y == body_tl.y
    assert phys_br.y == body_br.y


def test_physical_bbox_includes_clearance_uniformly():
    comp = _connector(
        body_w=4.0, body_h=4.0,
        pads=[_pad((12.0, 0.0), size=(2.0, 1.0))],  # reaches 13.0
    )
    tl, br = comp.physical_bbox(clearance=0.5)
    # Right pad reach 13.0 + 0.5 clearance = 13.5
    assert br.x == pytest.approx(13.5)
    # Body extends to (10 + 2.0) on the right courtyard side; pad wins
    # Left side: courtyard 8.0 - 0.5 = 7.5
    assert tl.x == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_pad_size_round_trips_through_solved_layout():
    """Serialize+reconstruct must preserve pad size for fresh artifacts."""
    comp = _connector(
        pads=[_pad((12.5, 0.0), size=(2.0, 1.5))],
    )
    payload = serialize_component(comp)
    pad_payloads = payload["pads"]
    assert pad_payloads[0]["size_mm"] == {"x": 2.0, "y": 1.5}
    rebuilt = _pad_from_dict(pad_payloads[0])
    assert rebuilt.size_mm == Point(2.0, 1.5)


def test_pad_size_round_trip_legacy_artifact_has_none():
    """An old artifact missing the size_mm key reconstructs as size_mm=None."""
    legacy_payload = {
        "ref": "J3",
        "pad_id": "1",
        "pos": {"x": 12.5, "y": 0.0},
        "net": "VBUS",
        "layer": "F.Cu",
        # No size_mm field
    }
    pad = _pad_from_dict(legacy_payload)
    assert pad.size_mm is None
    tl, br = pad.bbox()
    assert tl == br == Point(12.5, 0.0)


def test_pad_size_explicit_null_round_trip():
    """size_mm=null in JSON also reconstructs as None."""
    payload = {
        "ref": "J3",
        "pad_id": "1",
        "pos": {"x": 12.5, "y": 0.0},
        "net": "VBUS",
        "layer": "F.Cu",
        "size_mm": None,
    }
    pad = _pad_from_dict(payload)
    assert pad.size_mm is None


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def test_pad_size_rotation_swaps_orthogonal():
    """A 2x1 pad rotated 90° has AABB 1x2."""
    from kicraft.autoplacer.brain.subcircuit_instances import _rotate_size

    rotated = _rotate_size(Point(2.0, 1.0), 90.0)
    assert rotated == Point(1.0, 2.0)

    rotated_180 = _rotate_size(Point(2.0, 1.0), 180.0)
    assert rotated_180 == Point(2.0, 1.0)

    rotated_270 = _rotate_size(Point(2.0, 1.0), 270.0)
    assert rotated_270 == Point(1.0, 2.0)


def test_pad_size_rotation_45_grows_aabb():
    """A 2x1 pad rotated 45° has an AABB of (2+1)/sqrt(2) x (2+1)/sqrt(2)."""
    import math

    from kicraft.autoplacer.brain.subcircuit_instances import _rotate_size

    rotated = _rotate_size(Point(2.0, 1.0), 45.0)
    expected = (2.0 + 1.0) / math.sqrt(2.0)
    assert rotated.x == pytest.approx(expected)
    assert rotated.y == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Comp.bbox stays as courtyard (regression test against accidental rename)
# ---------------------------------------------------------------------------


def test_comp_bbox_is_still_courtyard_only():
    """The two methods are intentionally distinct: ``bbox()`` is the
    courtyard / keep-out, ``physical_bbox()`` includes pad copper."""
    comp = _connector(
        body_w=4.0, body_h=4.0,
        pads=[_pad((12.5, 0.0), size=(2.0, 1.0))],
    )
    body = comp.bbox()
    phys = comp.physical_bbox()
    assert body != phys
    assert body[1].x == pytest.approx(12.0)  # courtyard right
    assert phys[1].x == pytest.approx(13.5)  # pad right
