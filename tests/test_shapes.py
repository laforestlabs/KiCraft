"""Phase 3b: the Shapely-backed named/compound shape library.

Pins the geometry the compose pipeline will consume: every named shape builds a
valid unit polygon, the snowman is a single connected silhouette (overlapping
circles), and circumscribe() grows any named shape to enclose placed content.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from shapely.geometry import Polygon

from kicraft.shapes import (
    KNOWN_SHAPES,
    PolygonOutline,
    UnknownShapeError,
    build_unit_polygon,
    circumscribe,
)


def _pt(x, y):
    return SimpleNamespace(x=float(x), y=float(y))


CONTENT = (_pt(0, 0), _pt(40, 20))


@pytest.mark.parametrize("name", sorted(KNOWN_SHAPES))
def test_unit_polygon_is_valid_and_normalized(name):
    poly = build_unit_polygon(name)
    assert isinstance(poly, Polygon)
    assert poly.is_valid and not poly.is_empty
    assert poly.area > 0
    minx, miny, maxx, maxy = poly.bounds
    # Normalized: centered on origin, longest side == 1.0.
    assert max(maxx - minx, maxy - miny) == pytest.approx(1.0, abs=1e-6)
    assert (minx + maxx) / 2 == pytest.approx(0.0, abs=1e-6)
    assert (miny + maxy) / 2 == pytest.approx(0.0, abs=1e-6)


def test_snowman_is_single_connected_silhouette():
    poly = build_unit_polygon("snowman")
    # Overlapping circles union to ONE polygon (not a MultiPolygon).
    assert poly.geom_type == "Polygon"
    # Taller than wide (stacked circles).
    minx, miny, maxx, maxy = poly.bounds
    assert (maxy - miny) > (maxx - minx)


def test_hexagon_has_six_sides():
    poly = build_unit_polygon("hexagon")
    # exterior coords include the duplicated closing point.
    assert len(poly.exterior.coords) - 1 == 6


def test_unknown_shape_raises():
    with pytest.raises(UnknownShapeError):
        build_unit_polygon("dodecahedron")


@pytest.mark.parametrize("name", sorted(KNOWN_SHAPES))
def test_circumscribe_contains_content(name):
    tl, br = CONTENT
    outline = circumscribe(name, tl, br)
    assert isinstance(outline, PolygonOutline)
    # The placed content rectangle is fully inside the grown shape.
    assert outline.contains_rect(tl.x, tl.y, br.x, br.y, tol=0.0)
    # Centered on the content.
    (minx, miny), (maxx, maxy) = outline.aabb()
    assert (minx + maxx) / 2 == pytest.approx((tl.x + br.x) / 2, abs=0.2)
    assert (miny + maxy) / 2 == pytest.approx((tl.y + br.y) / 2, abs=0.2)


def test_circumscribe_points_form_closed_ring():
    outline = circumscribe("star", *CONTENT)
    pts = outline.points()
    assert len(pts) >= 8  # a 5-point star has 10 vertices
    assert pts[0] != pts[-1]  # closing point not duplicated


def test_polygon_outline_containment_point_and_tol():
    outline = circumscribe("hexagon", *CONTENT)
    cx, cy = 20.0, 10.0  # content center -> inside
    assert outline.contains_point(cx, cy)
    # A point far outside is out, but tolerance can pull it in.
    (minx, miny), (maxx, maxy) = outline.aabb()
    just_out = (maxx + 1.0, cy)
    assert not outline.contains_point(*just_out)
    assert outline.contains_point(*just_out, tol=2.0)
