"""Shared geometry helpers for the placement pipeline.

Extracted from placement.py for modularity.  Import from
``placement`` (the re-export hub) for backward compatibility,
or directly from this module in new code.
"""

from __future__ import annotations

import math

from .types import BoardState, Component, Point

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


def compute_min_board_size(
    state: BoardState,
    overhead_factor: float = 2.5,
    group_blocks: list[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Estimate the minimum viable board dimensions from component area.

    Returns (min_width_mm, min_height_mm) based on total component area
    scaled by overhead_factor (to leave room for routing and clearances).

    Also ensures the board is large enough to contain the largest group
    block (if *group_blocks* is supplied) or the largest estimated
    component cluster.  The board aspect ratio is clamped to 1:1–2:1 and
    maximum dimensions are capped to avoid unnecessarily large boards.
    """
    total_area = sum(c.area for c in state.components.values())
    min_area = total_area * overhead_factor
    if min_area <= 0:
        return (40.0, 30.0)  # fallback

    # Start from board aspect ratio, clamped to 1:1 – 2:1
    bw = max(1.0, state.board_width)
    bh = max(1.0, state.board_height)
    aspect = max(1.0, min(2.0, bw / bh))

    # Area-based estimate
    min_w = math.sqrt(min_area * aspect)
    min_h = min_w / aspect

    # --- Ensure board can hold the largest block -----------------------
    largest_block_w = 0.0
    largest_block_h = 0.0

    # Explicit group blocks (provided by caller after solve_group)
    if group_blocks:
        for gw, gh in group_blocks:
            largest_block_w = max(largest_block_w, gw)
            largest_block_h = max(largest_block_h, gh)

    # Estimate from individual components: largest single component and
    # potential paired components (same kind, similar size) that the
    # group solver will likely merge into one block.
    comp_by_area = sorted(state.components.values(), key=lambda c: c.area, reverse=True)
    if comp_by_area:
        biggest = comp_by_area[0]
        largest_block_w = max(largest_block_w, biggest.width_mm)
        largest_block_h = max(largest_block_h, biggest.height_mm)

        # Look for a potential pair partner (same kind, similar size)
        for c2 in comp_by_area[1:5]:
            if (
                c2.kind == biggest.kind
                and c2.kind not in ("", "misc", "passive")
                and c2.area > 0
                and biggest.area > 0
                and min(c2.area, biggest.area) / max(c2.area, biggest.area) > 0.5
            ):
                gap = 2.0  # minimal gap between pair members
                # Horizontal arrangement
                horiz = (
                    biggest.width_mm + c2.width_mm + gap,
                    max(biggest.height_mm, c2.height_mm),
                )
                # Vertical arrangement
                vert = (
                    max(biggest.width_mm, c2.width_mm),
                    biggest.height_mm + c2.height_mm + gap,
                )
                # Pick the more compact arrangement (smaller area)
                if horiz[0] * horiz[1] <= vert[0] * vert[1]:
                    pair_w, pair_h = horiz
                else:
                    pair_w, pair_h = vert
                largest_block_w = max(largest_block_w, pair_w)
                largest_block_h = max(largest_block_h, pair_h)
                break  # only consider first matching pair

    # Board must fit largest block with edge margin on each side
    block_margin = 4.0
    min_w = max(min_w, largest_block_w + block_margin * 2)
    min_h = max(min_h, largest_block_h + block_margin * 2)

    # --- Re-check aspect ratio after block adjustment ------------------
    if min_w > 0 and min_h > 0:
        ratio = min_w / min_h
        if ratio > 2.0:
            min_h = min_w / 2.0
        elif ratio < 0.5:
            min_w = min_h / 2.0

    # --- Cap maximum dimensions to avoid runaway board sizes -----------
    min_w = min(min_w, 120.0)
    min_h = min(min_h, 100.0)

    # Round up to nearest 5mm
    min_w = math.ceil(min_w / 5.0) * 5.0
    min_h = math.ceil(min_h / 5.0) * 5.0
    return (max(30.0, min_w), max(20.0, min_h))
