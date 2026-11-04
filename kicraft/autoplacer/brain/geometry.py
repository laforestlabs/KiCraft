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

Note on inverses: the inverse of ``rotate_vector(·, +θ)`` is
``rotate_vector(·, -θ)`` (and ``math-CCW(θ) ≡ rotate_vector(·, -θ)``). The
origin-recovery sites (parent_adapter._rotated, placement_utils._world_artifact_origin,
compose._block_artifact_origin) recover ``origin = pos - rotate_vector(offset,
+rot)`` -- a *subtraction* of the SAME-sign forward transform, not a negated
rotation; that is the true inverse of placing a block at ``pos`` via the
forward ``R_cw(+rot)`` body-center transform. (Negating the angle there was a
historical 90/270 stranding bug, now fixed.)
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
    """Rotate a component about its footprint origin in KiCad's CW convention.

    Body and pad dimensions are world AABBs. Retain their original rotation
    basis while they are unchanged, so repeated non-cardinal trial rotations
    neither shrink the bounds nor compound their conservative enlargement.
    """
    delta = delta_deg % 360.0
    if abs(delta) < 1e-6:
        return
    old_rot = comp.rotation
    new_rot = (old_rot + delta) % 360.0
    body_basis = getattr(comp, "_rotation_body_basis", None)
    if body_basis is None or body_basis[3:] != (old_rot, comp.width_mm, comp.height_mm):
        body_basis = (old_rot, comp.width_mm, comp.height_mm, old_rot, comp.width_mm, comp.height_mm)
    body_w, body_h = bbox_after_rotation(
        body_basis[1], body_basis[2], new_rot - body_basis[0]
    )
    pad_bases = getattr(comp, "_rotation_pad_bases", ())
    new_pad_bases = []
    cx, cy = comp.pos.x, comp.pos.y

    def _rot(p: Point) -> Point:
        r = rotate_vector(Point(p.x - cx, p.y - cy), delta)
        return Point(cx + r.x, cy + r.y)

    if comp.body_center is not None:
        comp.body_center = _rot(comp.body_center)
    for index, pad in enumerate(comp.pads):
        pad.pos = _rot(pad.pos)
        if pad.size_mm is None:
            new_pad_bases.append(None)
            continue
        current = (pad.ref, pad.pad_id, old_rot, pad.size_mm.x, pad.size_mm.y)
        basis = pad_bases[index] if index < len(pad_bases) else None
        if basis is None or basis[3:] != current:
            basis = (old_rot, pad.size_mm.x, pad.size_mm.y, *current)
        w, h = bbox_after_rotation(basis[1], basis[2], new_rot - basis[0])
        pad.size_mm = Point(w, h)
        new_pad_bases.append((basis[0], basis[1], basis[2], pad.ref, pad.pad_id, new_rot, w, h))
    comp.rotation = new_rot
    comp.width_mm, comp.height_mm = body_w, body_h
    comp._rotation_body_basis = (body_basis[0], body_basis[1], body_basis[2], new_rot, body_w, body_h)
    comp._rotation_pad_bases = new_pad_bases
