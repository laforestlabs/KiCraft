"""kicraft.autoplacer.brain.geometry: the single rotation/frame convention.

Pins the KiCad-CW convention (`x·cos + y·sin; -x·sin + y·cos`) with:
- the documented empirical values for a unit pad across 0/90/180/270,
- agreement with pcbnew.SetOrientationDegrees (opt-in; needs KiCad),
- bbox-after-rotation orthogonal/diagonal cases,
- rotate_component_in_place (pads + AABB swap), and
- the inverse identity that the body-center-origin recovery relies on.
"""
from __future__ import annotations

import math

import pytest

from kicraft.autoplacer.brain import geometry
from kicraft.autoplacer.brain.types import Component, Layer, Pad, Point


def _close(a: Point, b: Point, tol: float = 1e-9) -> bool:
    return abs(a.x - b.x) <= tol and abs(a.y - b.y) <= tol


# ---- the documented empirical truth -----------------------------------------


@pytest.mark.parametrize(
    "deg, expected",
    [
        (0, (1.0, 0.0)),
        (90, (0.0, -1.0)),   # pcbnew places a (1,0) pad at (0,-1) when rot 90
        (180, (-1.0, 0.0)),
        (270, (0.0, 1.0)),
    ],
)
def test_rotate_vector_unit_pad(deg, expected):
    r = geometry.rotate_vector(Point(1.0, 0.0), deg)
    assert _close(r, Point(*expected))


def test_transform_point_adds_origin():
    r = geometry.transform_point(Point(1.0, 0.0), Point(10.0, 5.0), 90)
    assert _close(r, Point(10.0, 4.0))  # (0,-1) + (10,5)


def test_math_ccw_is_negative_angle():
    """The inverse-recovery sites use math-CCW == rotate_vector(v, -deg)."""
    for deg in (37.0, 90.0, 213.0):
        v = Point(2.3, -1.1)
        rad = math.radians(deg)
        ccw = Point(v.x * math.cos(rad) - v.y * math.sin(rad),
                    v.x * math.sin(rad) + v.y * math.cos(rad))
        assert _close(geometry.rotate_vector(v, -deg), ccw)


def test_rotate_vector_inverse_identity():
    v = Point(3.0, -7.0)
    for deg in (0, 17, 90, 180, 270, 359):
        back = geometry.rotate_vector(geometry.rotate_vector(v, deg), -deg)
        assert _close(back, v, tol=1e-9)


# ---- bbox after rotation -----------------------------------------------------


@pytest.mark.parametrize(
    "deg, expected",
    [(0, (2.0, 4.0)), (180, (2.0, 4.0)), (90, (4.0, 2.0)), (270, (4.0, 2.0))],
)
def test_bbox_orthogonal_exact(deg, expected):
    assert geometry.bbox_after_rotation(2.0, 4.0, deg) == expected


def test_bbox_diagonal_is_bounding_extent():
    w, h = geometry.bbox_after_rotation(2.0, 4.0, 45)
    assert w == pytest.approx((2 + 4) / math.sqrt(2))
    assert h == pytest.approx((2 + 4) / math.sqrt(2))


# ---- rotate_component_in_place ----------------------------------------------


def _comp() -> Component:
    return Component(
        ref="U1", value="X", pos=Point(10.0, 10.0), rotation=0.0,
        layer=Layer.FRONT, width_mm=2.0, height_mm=4.0,
        pads=[Pad(ref="U1", pad_id="1", pos=Point(11.0, 10.0), net="N",
                  layer=Layer.FRONT)],
        body_center=Point(10.0, 10.0),
    )


def test_rotate_component_in_place_90():
    c = _comp()
    geometry.rotate_component_in_place(c, 90)
    assert c.rotation == 90.0
    # pad was at +1 in x relative to pos -> rotates to -1 in y (KiCad CW)
    assert _close(c.pads[0].pos, Point(10.0, 9.0))
    # AABB w/h swap at 90
    assert (c.width_mm, c.height_mm) == (4.0, 2.0)


def test_rotate_component_in_place_rotates_pad_size_aabb():
    # pad.size_mm is a WORLD-axis-aligned AABB (types.Pad contract) and must
    # re-rotate with the pad -- at 90/270 the extents swap exactly.
    c = _comp()
    c.pads[0].size_mm = Point(1.5, 0.8)
    geometry.rotate_component_in_place(c, 90)
    assert _close(c.pads[0].size_mm, Point(0.8, 1.5))
    geometry.rotate_component_in_place(c, 90)  # now at 180 total
    assert _close(c.pads[0].size_mm, Point(1.5, 0.8))


def test_rotate_component_in_place_noop_for_zero():
    c = _comp()
    geometry.rotate_component_in_place(c, 0)
    assert c.rotation == 0.0
    assert _close(c.pads[0].pos, Point(11.0, 10.0))
    assert (c.width_mm, c.height_mm) == (2.0, 4.0)


# ---- agreement with pcbnew (opt-in: needs KiCad's pcbnew) --------------------


@pytest.mark.parametrize("deg", [0, 90, 180, 270])
def test_transform_point_agrees_with_pcbnew(deg):
    pcbnew = pytest.importorskip("pcbnew")
    board = pcbnew.BOARD()
    fp = pcbnew.FOOTPRINT(board)
    board.Add(fp)
    fp.SetPosition(pcbnew.VECTOR2I(0, 0))
    pad = pcbnew.PAD(fp)
    pad.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(1.0), pcbnew.FromMM(0.0)))
    fp.Add(pad)
    fp.SetOrientationDegrees(deg)

    pos = pad.GetPosition()
    pcb_pt = Point(round(pcbnew.ToMM(pos.x), 6), round(pcbnew.ToMM(pos.y), 6))
    ours = geometry.transform_point(Point(1.0, 0.0), Point(0.0, 0.0), deg)
    assert _close(ours, pcb_pt, tol=1e-4), f"deg={deg}: ours={ours} pcbnew={pcb_pt}"


@pytest.mark.parametrize("deg", [0, 90, 180, 270])
def test_flip_composition_is_local_y_mirror_then_rotate(deg):
    """The stamp sites compose ``fp.Flip(pos)`` + ``SetOrientationDegrees``;
    the net world transform is R_cw(deg) applied to the Y-MIRRORED local
    offset. _assign_layers' back-side flip models this with a Y mirror about
    pos (an X mirror -- the pre-B24 behavior -- only agrees at 180-deg-offset
    orientations and swaps 2-pad identities)."""
    pcbnew = pytest.importorskip("pcbnew")
    local = Point(3.0, 1.0)
    board = pcbnew.BOARD()
    fp = pcbnew.FOOTPRINT(board)
    board.Add(fp)
    fp.SetPosition(pcbnew.VECTOR2I(0, 0))
    pad = pcbnew.PAD(fp)
    pad.SetPosition(
        pcbnew.VECTOR2I(pcbnew.FromMM(local.x), pcbnew.FromMM(local.y))
    )
    fp.Add(pad)
    fp.Flip(fp.GetPosition(), False)
    fp.SetOrientationDegrees(deg)

    pos = pad.GetPosition()
    stamped = Point(round(pcbnew.ToMM(pos.x), 6), round(pcbnew.ToMM(pos.y), 6))
    model = geometry.rotate_vector(Point(local.x, -local.y), deg)
    assert _close(model, stamped, tol=1e-4), (
        f"deg={deg}: model={model} stamped={stamped}"
    )
