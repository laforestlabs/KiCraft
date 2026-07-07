"""Pure geometry helpers for silkscreen text placement/legalization.

Shared by the parent-stamp subprocess (clamping leaf group labels into the
board outline) and the silk-legend subprocess (placing the board legend and
authored labels). Deliberately pcbnew-free: everything works on mm floats,
``(x, y)`` point tuples and ``(left, top, right, bottom)`` boxes, so the
logic is unit-testable outside a KiCad python.
"""
from __future__ import annotations

Box = tuple[float, float, float, float]  # (left, top, right, bottom), mm
Point = tuple[float, float]


def point_in_poly(x: float, y: float, poly: list[Point]) -> bool:
    """Ray-cast point-in-polygon (closed polygon, last->first edge implied)."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def poly_bbox(poly: list[Point]) -> Box:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def _box_probe_points(box: Box, margin: float) -> list[Point]:
    """Corners + edge midpoints of ``box`` inflated by ``margin``.

    Edge midpoints matter for wide/tall text near concave outline features:
    all four corners can sit inside while the middle crosses a notch.
    """
    left, top, right, bottom = box
    left -= margin
    top -= margin
    right += margin
    bottom += margin
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    return [
        (left, top), (right, top), (right, bottom), (left, bottom),
        (cx, top), (cx, bottom), (left, cy), (right, cy),
    ]


def bbox_inside_poly(box: Box, poly: list[Point], margin: float = 0.0) -> bool:
    """True when ``box`` (inflated by ``margin``) sits fully inside ``poly``.

    Probe-point approximation (corners + edge midpoints): exact for convex
    outlines and the rounded-rect family the pipeline emits, and errs on
    the strict side for anything reasonable-sized vs. the outline features.
    """
    return all(point_in_poly(px, py, poly) for px, py in _box_probe_points(box, margin))


def boxes_overlap(a: Box, b: Box, clearance: float = 0.0) -> bool:
    return (
        min(a[2], b[2]) - max(a[0], b[0]) > -clearance
        and min(a[3], b[3]) - max(a[1], b[1]) > -clearance
    )


def shift_box(box: Box, dx: float, dy: float) -> Box:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def clamp_shift_into_bbox(box: Box, bound: Box, margin: float = 0.0) -> tuple[float, float]:
    """The (dx, dy) translating ``box`` inside ``bound`` (minus margin).

    When the box is larger than the bound on an axis it is left-/top-aligned
    (the caller detects the no-fit case via ``bbox_inside_poly`` afterwards).
    """
    dx = 0.0
    dy = 0.0
    if box[0] < bound[0] + margin:
        dx = (bound[0] + margin) - box[0]
    elif box[2] > bound[2] - margin:
        dx = (bound[2] - margin) - box[2]
    if box[1] < bound[1] + margin:
        dy = (bound[1] + margin) - box[1]
    elif box[3] > bound[3] - margin:
        dy = (bound[3] - margin) - box[3]
    return dx, dy


def find_shift_into_poly(
    box: Box,
    poly: list[Point],
    margin: float = 0.2,
    max_shift: float = 5.0,
    step: float = 0.5,
) -> tuple[float, float] | None:
    """A small (dx, dy) that puts ``box`` fully inside ``poly``, or None.

    Deterministic: first clamps into the polygon bbox, then searches offsets
    around that clamp in rings of increasing L-inf radius (so the minimal
    move wins). Used to pull a clipped silk label back onto the board.
    """
    base_dx, base_dy = clamp_shift_into_bbox(box, poly_bbox(poly), margin)
    if bbox_inside_poly(shift_box(box, base_dx, base_dy), poly, margin):
        return base_dx, base_dy
    rings = int(round(max_shift / step))
    for r in range(1, rings + 1):
        d = r * step
        candidates: list[Point] = []
        steps = 2 * r
        for i in range(steps + 1):
            t = -d + i * step
            candidates.extend([(t, -d), (t, d), (-d, t), (d, t)])
        # De-dup preserving order; ring order keeps the search minimal-first.
        seen: set[Point] = set()
        for cdx, cdy in candidates:
            key = (round(cdx, 6), round(cdy, 6))
            if key in seen:
                continue
            seen.add(key)
            dx = base_dx + cdx
            dy = base_dy + cdy
            if bbox_inside_poly(shift_box(box, dx, dy), poly, margin):
                return dx, dy
    return None


__all__ = [
    "Box",
    "Point",
    "point_in_poly",
    "poly_bbox",
    "bbox_inside_poly",
    "boxes_overlap",
    "shift_box",
    "clamp_shift_into_bbox",
    "find_shift_into_poly",
]
