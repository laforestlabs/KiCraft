"""Pure bbox / rect / envelope geometry helpers for parent composition.

Split out of ``compose_subcircuits.py`` (Lever 2.5): these are leaf utilities
(they depend only on ``Point`` and a component's ``.bbox()``), shared by the
outline, slide, stamp and validation code. Re-exported from
``compose_subcircuits`` so existing references keep resolving.
"""
from __future__ import annotations

from kicraft.autoplacer.brain.types import Point


def _shift_envelope(
    envelope: tuple[Point, Point],
    origin: Point,
) -> tuple[Point, Point]:
    return (
        Point(envelope[0].x + origin.x, envelope[0].y + origin.y),
        Point(envelope[1].x + origin.x, envelope[1].y + origin.y),
    )


def _shift_rect(rect: tuple[Point, Point], origin: Point) -> tuple[Point, Point]:
    return (
        Point(rect[0].x + origin.x, rect[0].y + origin.y),
        Point(rect[1].x + origin.x, rect[1].y + origin.y),
    )


def _rect_area(rect: tuple[Point, Point]) -> float:
    return max(0.0, rect[1].x - rect[0].x) * max(0.0, rect[1].y - rect[0].y)


def _component_geometry_bbox(comp) -> tuple[Point, Point]:
    bbox_min, bbox_max = comp.bbox()
    min_x = bbox_min.x
    min_y = bbox_min.y
    max_x = bbox_max.x
    max_y = bbox_max.y
    for pad in comp.pads:
        min_x = min(min_x, pad.pos.x)
        min_y = min(min_y, pad.pos.y)
        max_x = max(max_x, pad.pos.x)
        max_y = max(max_y, pad.pos.y)
    return Point(min_x, min_y), Point(max_x, max_y)


def _bbox_disjoint(a: tuple[Point, Point], b: tuple[Point, Point]) -> bool:
    return a[1].x <= b[0].x or b[1].x <= a[0].x or a[1].y <= b[0].y or b[1].y <= a[0].y


def _rect_lists_disjoint(
    rects_a: list[tuple[Point, Point]],
    rects_b: list[tuple[Point, Point]],
) -> bool:
    for rect_a in rects_a:
        for rect_b in rects_b:
            if not _bbox_disjoint(rect_a, rect_b):
                return False
    return True
