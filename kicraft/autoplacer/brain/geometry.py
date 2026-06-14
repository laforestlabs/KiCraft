"""Single source of truth for placement coordinate-frame + rotation math.

ONE convention lives here -- KiCad's footprint-orientation convention
(clockwise-positive), matching ``pcbnew.SetOrientationDegrees``. Rotating a
local point ``(x, y)`` by ``θ`` about the origin is::

    x' =  x·cos θ + y·sin θ
    y' = -x·sin θ + y·cos θ

Empirically verified (see ``tests/test_geometry.py``): a pad at library
``(1, 0)`` rotated 90° lands at world ``(0, -1)`` -- exactly where pcbnew
places it. The *math-CCW* convention (``x·cos θ - y·sin θ; x·sin θ + y·cos θ``)
puts it at ``(0, +1)`` -- the opposite side -- and mixing the two is the
documented root of intra-leaf shorts and mis-oriented edge connectors. Every
place that rotates placement geometry routes through these helpers so the
convention is chosen, documented, and tested exactly once.

Note on inverses: ``math-CCW(θ) ≡ rotate_vector(·, -θ)``. Code that "undoes" a
placement rotation therefore calls ``rotate_vector(v, -deg)`` -- the negative
angle makes the inverse explicit at the call site (and, where the inverse is
load-bearing, easy to audit).
"""
from __future__ import annotations

import math

from .types import Component, Point


def rotate_vector(v: Point, deg: float) -> Point:
    """Rotate ``v`` about the origin by ``deg`` (KiCad CW convention)."""
    theta = math.radians(deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return Point(v.x * cos_t + v.y * sin_t, -v.x * sin_t + v.y * cos_t)


def transform_point(point: Point, origin: Point, deg: float) -> Point:
    """Rigid local->world transform: rotate ``point`` about the origin by
    ``deg`` (KiCad CW), then translate by ``origin``. Matches
    ``pcbnew.SetOrientationDegrees`` (verified: a pad at (1, 0) rotated 90°
    lands at (0, -1))."""
    r = rotate_vector(point, deg)
    return Point(r.x + origin.x, r.y + origin.y)


def bbox_after_rotation(
    width: float, height: float, deg: float
) -> tuple[float, float]:
    """Axis-aligned ``(w, h)`` extent of a ``width`` x ``height`` rectangle
    rotated by ``deg``. Orthogonal angles are exact; others return the
    conservative bounding extent of the rotated rectangle."""
    rot = deg % 360.0
    if abs(rot) < 1e-3 or abs(rot - 180.0) < 1e-3:
        return (width, height)
    if abs(rot - 90.0) < 1e-3 or abs(rot - 270.0) < 1e-3:
        return (height, width)
    theta = math.radians(rot)
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))
    return (width * cos_t + height * sin_t, width * sin_t + height * cos_t)


def rotate_component_in_place(comp: Component, delta_deg: float) -> None:
    """Rotate a ``Component`` about its own ``pos`` by ``delta_deg`` -- its
    pads, ``body_center``, ``rotation``, and (for 90/270) the AABB
    ``width_mm``/``height_mm``. KiCad CW, so the result agrees with
    :func:`transform_point` / pcbnew / ``opening_board_angle``; math-CCW here
    would orient a connector mouth toward the opposite edge."""
    delta = delta_deg % 360.0
    if abs(delta) < 1e-6:
        return
    cx, cy = comp.pos.x, comp.pos.y

    def _rot(p: Point) -> Point:
        r = rotate_vector(Point(p.x - cx, p.y - cy), delta)
        return Point(cx + r.x, cy + r.y)

    if comp.body_center is not None:
        comp.body_center = _rot(comp.body_center)
    for pad in comp.pads:
        pad.pos = _rot(pad.pos)
    comp.rotation = (comp.rotation + delta) % 360.0
    if round(delta) % 180 == 90:
        comp.width_mm, comp.height_mm = comp.height_mm, comp.width_mm
