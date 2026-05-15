"""Lock the leaf Edge.Cuts contour to the leaf silk-outline contour.

Edge.Cuts and the leaf F.SilkS poly both define the visible leaf boundary
in the canvas / monitor renders. If their shapes disagree (silk rounded
but Edge.Cuts sharp-cornered, or different radii / margins), the rendered
leaf shows the PCB substrate filling out past the yellow silk outline at
each corner -- the visible misalignment users see.

This test calls the two producers directly with realistic config and
asserts the contours trace each other within a tight tolerance.
"""

from __future__ import annotations

import math

from kicraft.autoplacer.brain.leaf_routing import _outline_around_geometry
from kicraft.autoplacer.brain.subcircuit_solver import _build_leaf_silkscreen
from kicraft.autoplacer.brain.types import Component, Layer, Point


CFG = {
    "silkscreen_margin_mm": 0.5,
    "silkscreen_corner_radius_mm": 1.0,
    "group_labels": {"U1": "FAKE LEAF"},
}

TOL_MM = 0.05  # well below the 0.15 mm silk stroke width


def _fake_components() -> dict[str, Component]:
    return {
        "U1": Component(
            ref="U1", value="IC", pos=Point(2.5, 5.0), rotation=0.0,
            layer=Layer.FRONT, width_mm=4.0, height_mm=8.0,
        ),
    }


def _component_bbox(components: dict[str, Component]) -> dict[str, float]:
    boxes = [c.physical_bbox() for c in components.values()]
    return {
        "min_x": min(tl.x for tl, _ in boxes),
        "min_y": min(tl.y for tl, _ in boxes),
        "max_x": max(br.x for _, br in boxes),
        "max_y": max(br.y for _, br in boxes),
    }


def _silk_poly_points(silk_elements) -> list[Point]:
    for el in silk_elements:
        if el.kind == "poly":
            return list(el.points)
    return []


def _point_to_segment_dist(p: Point, a: Point, b: Point) -> float:
    abx, aby = b.x - a.x, b.y - a.y
    L2 = abx * abx + aby * aby
    if L2 == 0.0:
        return math.hypot(p.x - a.x, p.y - a.y)
    t = ((p.x - a.x) * abx + (p.y - a.y) * aby) / L2
    t = max(0.0, min(1.0, t))
    fx = a.x + t * abx
    fy = a.y + t * aby
    return math.hypot(p.x - fx, p.y - fy)


def _polyline_segments(pts: list[Point], *, closed: bool) -> list[tuple[Point, Point]]:
    if len(pts) < 2:
        return []
    n = len(pts)
    pairs = [(pts[i], pts[i + 1]) for i in range(n - 1)]
    if closed:
        pairs.append((pts[-1], pts[0]))
    return pairs


def _max_dist_to_polyline(probe: list[Point], target_segments: list[tuple[Point, Point]]):
    worst = (0.0, None)
    for p in probe:
        d = min(_point_to_segment_dist(p, a, b) for a, b in target_segments)
        if d > worst[0]:
            worst = (d, p)
    return worst


def _edge_cuts_segments_from_outline(tl: Point, br: Point) -> list[tuple[Point, Point]]:
    """Edge.Cuts is currently stamped as a sharp 4-segment rectangle
    by ``hardware/adapter.py::_apply_board_outline``. This mirrors that
    shape so the test fails on the same geometric mismatch users see."""
    corners = [
        Point(tl.x, tl.y),
        Point(br.x, tl.y),
        Point(br.x, br.y),
        Point(tl.x, br.y),
    ]
    return [(corners[i], corners[(i + 1) % 4]) for i in range(4)]


def test_leaf_silk_vertices_lie_on_edge_cuts_contour():
    """Every silk-poly vertex must sit on the Edge.Cuts contour.

    The silk poly's corner-rounding vertices are the diagnostic: with a
    sharp-cornered Edge.Cuts rectangle, the silk vertices midway through
    each rounded corner sit ~radius*(1-cos(45 deg)) ~= 0.29 mm away
    from any Edge.Cuts segment at default radius=1.0 mm. So this assertion
    fails today and will continue to fail until Edge.Cuts traces the same
    rounded contour as the silk.
    """
    components = _fake_components()
    outline = _outline_around_geometry(components, CFG)
    assert outline is not None
    tl, br = outline

    silk_elements = _build_leaf_silkscreen(
        components, _component_bbox(components), extraction=None, config=CFG,
    )
    silk_pts = _silk_poly_points(silk_elements)
    assert silk_pts, "expected a silk poly element from _build_leaf_silkscreen"

    edge_segs = _edge_cuts_segments_from_outline(tl, br)
    worst_d, worst_pt = _max_dist_to_polyline(silk_pts, edge_segs)
    assert worst_d <= TOL_MM, (
        f"silk poly vertex {worst_pt} is {worst_d:.4f} mm from the Edge.Cuts "
        f"contour (tol={TOL_MM} mm). Edge.Cuts and the silk outline trace "
        f"different shapes -- the substrate corners poke out past the "
        f"rounded silk in the rendered leaf."
    )


def test_edge_cuts_corners_lie_on_silk_contour():
    """Every Edge.Cuts corner must sit on the silk-poly contour.

    The symmetric direction: if Edge.Cuts has a 90 deg sharp corner that
    the silk poly never visits (because silk is rounded), the corner
    point is at least radius*(sqrt(2)-1) ~= 0.41 mm from the silk
    perimeter at default radius=1.0 mm. Fails today; passes only when
    both layers agree on the same boundary shape.
    """
    components = _fake_components()
    outline = _outline_around_geometry(components, CFG)
    assert outline is not None
    tl, br = outline

    silk_elements = _build_leaf_silkscreen(
        components, _component_bbox(components), extraction=None, config=CFG,
    )
    silk_pts = _silk_poly_points(silk_elements)
    assert silk_pts

    edge_corners = [
        Point(tl.x, tl.y), Point(br.x, tl.y),
        Point(br.x, br.y), Point(tl.x, br.y),
    ]
    silk_segs = _polyline_segments(silk_pts, closed=True)
    worst_d, worst_pt = _max_dist_to_polyline(edge_corners, silk_segs)
    assert worst_d <= TOL_MM, (
        f"Edge.Cuts corner {worst_pt} is {worst_d:.4f} mm from the silk "
        f"contour (tol={TOL_MM} mm). Edge.Cuts steps into sharp corners "
        f"the silk never reaches -- the two outlines are different shapes."
    )
