"""Shared geometry helpers for the placement pipeline.

Extracted from placement.py for modularity.  Import from
``placement`` (the re-export hub) for backward compatibility,
or directly from this module in new code.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from . import geometry
from .types import Component, Point


class PackingMetrics(NamedTuple):
    density: float  # 0-1, clamped (matches min(1.0, comp_area / placed_area))
    score: float  # 0-100, = min(100, density * fill_multiplier)


def packing_metrics(
    component_area: float,
    placed_bbox_area: float,
    *,
    fill_multiplier: float = 150.0,
) -> PackingMetrics:
    """Centralise the packing-score curve so the in-loop SA score and the
    post-compose round score agree on the meaning of "tightly packed".

    Pre-computed areas in mm²; callers own the bbox extraction. Returns
    density=0.0 / score=0.0 for degenerate placed_bbox_area (<= 0);
    callers wanting "no opinion" should guard before calling.
    """
    if placed_bbox_area <= 0.0:
        return PackingMetrics(0.0, 0.0)
    density = min(1.0, component_area / placed_bbox_area)
    score = max(0.0, min(100.0, density * fill_multiplier))
    return PackingMetrics(density, score)


def _world_artifact_origin(comp: Component) -> Point:
    """World-frame artifact origin for a synthetic block component.

    ``block_artifact_origin_offset`` is the local-frame vector
    (body_center - artifact_origin). When the block is rotated and placed
    at ``comp.pos`` (the body center), the artifact origin is the
    inverse-rotated offset subtracted from pos.
    """
    if comp.block_artifact_origin_offset is None:
        return Point(comp.pos.x, comp.pos.y)
    # rotate_vector(offset, -rotation) == the old math-CCW rotation. Same
    # inverse-recovery pattern (and same 90/270 caveat) flagged on
    # parent_adapter._rotated -- preserved exactly here.
    rotated = geometry.rotate_vector(
        comp.block_artifact_origin_offset, -comp.rotation
    )
    return Point(comp.pos.x - rotated.x, comp.pos.y - rotated.y)


def _blocker_pair_compatible(a: Component, b: Component) -> bool:
    """Return True if two blocks' sparse blocker sets permit bbox overlap.

    Used by the parent-side path: when both components carry block
    metadata, the unified solver consults ``can_overlap_sparse`` to
    decide if the courtyards may overlap (e.g. front-only vs back-only
    copper). Leaf placement components have ``block_blocker_set is None``
    and this short-circuits to False -- preserving today's bbox-only
    semantics for the leaf path.

    ``block_force_back_only`` rides on each Component as a project-level
    override for the layer-intent heuristic; passed through to
    ``can_overlap_sparse`` so its same-layer-outline check treats the
    flagged leaf as having no front-side copper intent regardless of
    its blocker geometry.
    """
    if a.block_blocker_set is None or b.block_blocker_set is None:
        return False
    # Local import to avoid a top-level cycle (subcircuit_composer
    # transitively pulls in this module's siblings via types).
    from .subcircuit_composer import can_overlap_sparse

    return can_overlap_sparse(
        a.block_blocker_set,
        _world_artifact_origin(a),
        a.rotation,
        b.block_blocker_set,
        _world_artifact_origin(b),
        b.rotation,
        force_back_only_a=bool(getattr(a, "block_force_back_only", False)),
        force_back_only_b=bool(getattr(b, "block_force_back_only", False)),
    )


def _pad_half_extents(comp: Component) -> tuple[float, float]:
    """Return pad-aware half-extents (max distance from pos to any pad or body edge).

    Battery holders and large THT components can have pads that extend beyond
    the body bounding box.  This function returns the effective half-width and
    half-height that covers both the body *and* all pads, ensuring clamping
    logic keeps all pads inside the board.

    When body_center is offset from pos, the courtyard extends further on the
    offset side.  The base half-extents include this offset so the entire
    courtyard is kept within board boundaries.
    """
    if comp.body_center:
        # The courtyard spans from body_center ± width/2.
        # Measured from pos, the furthest extent on each side is:
        #   body_center_offset + width/2
        hw = comp.width_mm / 2 + abs(comp.body_center.x - comp.pos.x)
        hh = comp.height_mm / 2 + abs(comp.body_center.y - comp.pos.y)
    else:
        hw = comp.width_mm / 2
        hh = comp.height_mm / 2
    for pad in comp.pads:
        dx = abs(pad.pos.x - comp.pos.x)
        dy = abs(pad.pos.y - comp.pos.y)
        hw = max(hw, dx)
        hh = max(hh, dy)
    return hw, hh


def _bbox_overlap(a: Component, b: Component, clearance: float = 0.5) -> bool:
    """Check if two component bounding boxes overlap with clearance."""
    a_tl, a_br = a.bbox(clearance / 2)
    b_tl, b_br = b.bbox(clearance / 2)
    return a_tl.x < b_br.x and a_br.x > b_tl.x and a_tl.y < b_br.y and a_br.y > b_tl.y


def _bbox_overlap_amount(a: Component, b: Component) -> float:
    """Return overlap area (0 if no overlap)."""
    a_tl, a_br = a.bbox()
    b_tl, b_br = b.bbox()
    ox = max(0, min(a_br.x, b_br.x) - max(a_tl.x, b_tl.x))
    oy = max(0, min(a_br.y, b_br.y) - max(a_tl.y, b_tl.y))
    return ox * oy


def _bbox_overlap_xy(
    a_tl: Point, a_br: Point, b_tl: Point, b_br: Point
) -> tuple[float, float]:
    """Return overlap distances on X/Y axes (0 if separated)."""
    ox = min(a_br.x, b_br.x) - max(a_tl.x, b_tl.x)
    oy = min(a_br.y, b_br.y) - max(a_tl.y, b_tl.y)
    return max(0.0, ox), max(0.0, oy)


def _effective_bbox(comp: Component, clearance: float = 0.0) -> tuple[Point, Point]:
    """Return a pad-aware bbox using true asymmetric body/pad extents.

    Unlike _pad_half_extents(), this preserves asymmetry when the footprint
    origin is offset from the physical body center, which is common for edge
    connectors.  Keeps legality overlap checks aligned to the real stamped
    footprint envelope instead of a symmetric box around comp.pos.
    """
    if comp.body_center is not None:
        min_x = comp.body_center.x - comp.width_mm / 2
        max_x = comp.body_center.x + comp.width_mm / 2
        min_y = comp.body_center.y - comp.height_mm / 2
        max_y = comp.body_center.y + comp.height_mm / 2
    else:
        min_x = comp.pos.x - comp.width_mm / 2
        max_x = comp.pos.x + comp.width_mm / 2
        min_y = comp.pos.y - comp.height_mm / 2
        max_y = comp.pos.y + comp.height_mm / 2

    for pad in comp.pads:
        min_x = min(min_x, pad.pos.x)
        max_x = max(max_x, pad.pos.x)
        min_y = min(min_y, pad.pos.y)
        max_y = max(max_y, pad.pos.y)

    return (
        Point(min_x - clearance, min_y - clearance),
        Point(max_x + clearance, max_y + clearance),
    )


def _swap_pad_positions(a: Component, b: Component):
    """After swapping a.pos and b.pos, update pad positions accordingly."""
    # Pads are at absolute positions. After swap, shift by the delta.
    # a's pads need to move by (a.pos - old_a_pos) = (b_old - a_old)
    # But .pos was already swapped so a.pos = b_old, b.pos = a_old
    # So a's old pos was b.pos (current), a's new pos is a.pos (current)
    delta_ax = a.pos.x - b.pos.x
    delta_ay = a.pos.y - b.pos.y
    for p in a.pads:
        p.pos = Point(p.pos.x + delta_ax, p.pos.y + delta_ay)
    for p in b.pads:
        p.pos = Point(p.pos.x - delta_ax, p.pos.y - delta_ay)
    if a.body_center is not None:
        a.body_center = Point(a.body_center.x + delta_ax, a.body_center.y + delta_ay)
    if b.body_center is not None:
        b.body_center = Point(b.body_center.x - delta_ax, b.body_center.y - delta_ay)


def _update_pad_positions(comp: Component, old_pos: Point, old_rot: float):
    """Update pad and body_center absolute positions after component move/rotate.

    Uses KiCad's rotation convention:
        x' = x·cos θ + y·sin θ
        y' = -x·sin θ + y·cos θ
    where θ is the rotation delta in radians.
    """
    dx = comp.pos.x - old_pos.x
    dy = comp.pos.y - old_pos.y
    rot_delta = math.radians(comp.rotation - old_rot)

    def _transform(pt: Point) -> Point:
        if abs(rot_delta) < 0.001:
            return Point(pt.x + dx, pt.y + dy)
        rx = pt.x - old_pos.x
        ry = pt.y - old_pos.y
        cos_r = math.cos(rot_delta)
        sin_r = math.sin(rot_delta)
        return Point(
            comp.pos.x + rx * cos_r + ry * sin_r,
            comp.pos.y - rx * sin_r + ry * cos_r,
        )

    for pad in comp.pads:
        pad.pos = _transform(pad.pos)
    if comp.body_center is not None:
        comp.body_center = _transform(comp.body_center)

