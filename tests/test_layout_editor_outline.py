"""OutlineSpec geometry + ManualLayout schema v2 tests.

Pins the contracts the shaped-board pipeline rests on:

- polyline generators emit closed convex loops whose sampling error
  (circle sagitta) stays under the geometry-validation margin;
- analytic containment agrees with the shape (corners of a circle's
  AABB are OUTSIDE);
- mounting-hole corner pegs reduce to the historical rect behaviour
  and stay on-board for every shape;
- v1 manual layouts load and migrate; v2 round-trips.
"""

from __future__ import annotations

import math

import pytest

from kicraft.autoplacer.brain.types import Point
from kicraft.layout_editor.model import ManualLayout
from kicraft.layout_editor.outline import (
    SHAPES,
    OutlineSpec,
    circle_segment_count,
)


def _spec(shape: str, w: float = 60.0, h: float = 40.0, **kw) -> OutlineSpec:
    if shape == "circle":
        h = w
    return OutlineSpec(
        shape=shape, min_pt=Point(0.0, 0.0), max_pt=Point(w, h), **kw
    )


def _polygon_is_convex_and_clockwise(pts) -> bool:
    """All turns the same sign (convex); sign positive for clockwise in
    KiCad's Y-down frame (cross z = dx1*dy2 - dy1*dx2 > 0)."""
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
        if cross < -1e-9:
            return False
    return True


@pytest.mark.parametrize("shape,kw", [
    ("rect", {}),
    ("rounded_rect", {"corner_radius_mm": 4.0}),
    ("chamfered_rect", {"chamfer_mm": 5.0}),
    ("circle", {}),
])
def test_polyline_closed_convex_clockwise_within_aabb(shape, kw):
    spec = _spec(shape, **kw)
    pts = spec.polyline()
    assert len(pts) >= 3
    # No duplicated closing point.
    first, last = pts[0], pts[-1]
    assert (first.x, first.y) != (last.x, last.y)
    assert _polygon_is_convex_and_clockwise(pts)
    for p in pts:
        assert spec.min_pt.x - 1e-9 <= p.x <= spec.max_pt.x + 1e-9
        assert spec.min_pt.y - 1e-9 <= p.y <= spec.max_pt.y + 1e-9


def test_rect_polyline_is_four_corners():
    pts = _spec("rect").polyline()
    assert [(p.x, p.y) for p in pts] == [
        (0.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)
    ]


def test_circle_polyline_sagitta_bound():
    spec = _spec("circle", w=50.0)
    r = 25.0
    pts = spec.polyline()
    n = len(pts)
    assert n == circle_segment_count(r)
    assert 32 <= n <= 128
    # Worst-case sagitta of the inscribed chord polygon.
    sagitta = r * (1.0 - math.cos(math.pi / n))
    assert sagitta <= 0.02 + 1e-9
    # Every vertex sits on the circle.
    cx = cy = 25.0
    for p in pts:
        assert math.hypot(p.x - cx, p.y - cy) == pytest.approx(r, abs=1e-9)


def test_rounded_rect_polyline_matches_leaf_generator():
    """Shaped parents and leaf outlines must sample identical geometry."""
    from kicraft.autoplacer.brain.subcircuit_solver import leaf_outline_polyline

    spec = _spec("rounded_rect", corner_radius_mm=3.0)
    via_spec = spec.polyline()
    via_leaf = leaf_outline_polyline(0.0, 0.0, 60.0, 40.0, 3.0)
    assert len(via_spec) == len(via_leaf)
    for a, b in zip(via_spec, via_leaf):
        assert a.x == pytest.approx(b.x, abs=1e-12)
        assert a.y == pytest.approx(b.y, abs=1e-12)


def test_oversized_radius_and_chamfer_clamp_instead_of_failing():
    spec = _spec("rounded_rect", corner_radius_mm=999.0)
    pts = spec.polyline()
    # Clamped to min(w,h)/2 = 20: degenerates to a stadium, still valid.
    assert _polygon_is_convex_and_clockwise(pts)
    spec_c = _spec("chamfered_rect", chamfer_mm=999.0)
    assert _polygon_is_convex_and_clockwise(spec_c.polyline())


def test_containment_rect_vs_shapes_at_aabb_corner():
    """The AABB corner is inside a rect but outside every other shape."""
    assert _spec("rect").contains_point(0.5, 0.5)
    assert not _spec("rounded_rect", corner_radius_mm=4.0).contains_point(0.5, 0.5)
    assert not _spec("chamfered_rect", chamfer_mm=5.0).contains_point(0.5, 0.5)
    assert not _spec("circle").contains_point(0.5, 0.5)
    # Center is inside everything.
    for shape, kw in [
        ("rect", {}),
        ("rounded_rect", {"corner_radius_mm": 4.0}),
        ("chamfered_rect", {"chamfer_mm": 5.0}),
        ("circle", {}),
    ]:
        spec = _spec(shape, **kw)
        cx = (spec.min_pt.x + spec.max_pt.x) / 2
        cy = (spec.min_pt.y + spec.max_pt.y) / 2
        assert spec.contains_point(cx, cy)


def test_containment_tolerance():
    spec = _spec("circle", w=50.0)
    # Just outside the circle along +x from center: r=25 at (50.04, 25).
    assert not spec.contains_point(50.04, 25.0)
    assert spec.contains_point(50.04, 25.0, tol=0.05)


def test_contains_rect_uses_all_corners():
    spec = _spec("circle", w=50.0)
    # A rect hugging the AABB corner: inside the AABB, outside the circle.
    assert not spec.contains_rect(1.0, 1.0, 8.0, 8.0)
    # A small centered rect is inside.
    assert spec.contains_rect(20.0, 20.0, 30.0, 30.0)


def test_polyline_vertices_pass_containment():
    """The stamped polyline must never be flagged by the validator that
    uses analytic containment (sampling sits exactly on the boundary)."""
    for shape, kw in [
        ("rounded_rect", {"corner_radius_mm": 4.0}),
        ("chamfered_rect", {"chamfer_mm": 5.0}),
        ("circle", {}),
    ]:
        spec = _spec(shape, **kw)
        for p in spec.polyline():
            assert spec.contains_point(p.x, p.y, tol=0.01), (shape, p.x, p.y)


def test_mounting_hole_rect_compat():
    """Plain rect keeps the historical corner + (inset, inset) peg."""
    spec = _spec("rect")
    pos = spec.mounting_hole_position("top-left", 5.0)
    assert (pos.x, pos.y) == (5.0, 5.0)
    pos = spec.mounting_hole_position("bottom-right", 5.0)
    assert (pos.x, pos.y) == (55.0, 35.0)


@pytest.mark.parametrize("shape,kw", [
    ("rounded_rect", {"corner_radius_mm": 6.0}),
    ("chamfered_rect", {"chamfer_mm": 8.0}),
    ("circle", {}),
])
@pytest.mark.parametrize(
    "corner", ["top-left", "top-right", "bottom-left", "bottom-right"]
)
def test_mounting_hole_pegs_stay_on_board_for_shapes(shape, kw, corner):
    spec = _spec(shape, **kw)
    pos = spec.mounting_hole_position(corner, 5.0)
    assert spec.contains_point(pos.x, pos.y), (shape, corner, pos.x, pos.y)


def test_from_dict_validation():
    with pytest.raises(ValueError, match="unknown outline shape"):
        OutlineSpec.from_dict(
            {"shape": "hexagon", "min": {"x": 0, "y": 0}, "max": {"x": 10, "y": 10}}
        )
    with pytest.raises(ValueError, match="square"):
        OutlineSpec.from_dict(
            {"shape": "circle", "min": {"x": 0, "y": 0}, "max": {"x": 60, "y": 40}}
        )
    with pytest.raises(ValueError, match="corner_radius_mm"):
        OutlineSpec.from_dict(
            {"shape": "rounded_rect", "min": {"x": 0, "y": 0}, "max": {"x": 60, "y": 40}}
        )
    with pytest.raises(ValueError, match="max>min"):
        OutlineSpec.from_dict(
            {"shape": "rect", "min": {"x": 10, "y": 0}, "max": {"x": 0, "y": 40}}
        )
    # Every declared shape round-trips through to_dict/from_dict.
    for shape in SHAPES:
        kw = {}
        if shape == "rounded_rect":
            kw["corner_radius_mm"] = 3.0
        if shape == "chamfered_rect":
            kw["chamfer_mm"] = 4.0
        spec = _spec(shape, **kw)
        again = OutlineSpec.from_dict(spec.to_dict())
        assert again == spec


# --- ManualLayout schema -----------------------------------------------------


def _v1_payload() -> dict:
    return {
        "schema_version": "manual_layout.v1",
        "board_outline": {
            "min": {"x": 0.0, "y": 0.0},
            "max": {"x": 80.0, "y": 60.0},
        },
        "placements": [
            {
                "instance_path": "/battery",
                "origin": {"x": 10.5, "y": 15.2},
                "rotation": 90.0,
            }
        ],
        "parent_local": [{"ref": "H1", "pos": {"x": 5.0, "y": 5.0}}],
        "mounting_holes": [
            {"index": 0, "corner": "top-left", "inset_mm": 5.0,
             "pos": {"x": 5.0, "y": 5.0}}
        ],
    }


def test_v1_payload_loads_and_migrates():
    layout = ManualLayout.from_dict(_v1_payload())
    assert layout.schema_version == "manual_layout.v2"
    assert layout.outline.shape == "rect"
    # AABB view preserved for pre-shape consumers.
    tl, br = layout.board_outline
    assert (tl.x, tl.y, br.x, br.y) == (0.0, 0.0, 80.0, 60.0)
    assert layout.placements[0].rotation == 90.0
    assert layout.mounting_holes[0].screw == "M3"  # default added


def test_v2_round_trip_preserves_shape_and_screw():
    layout = ManualLayout.from_dict(_v1_payload())
    layout.outline = OutlineSpec(
        shape="rounded_rect",
        min_pt=Point(0.0, 0.0),
        max_pt=Point(80.0, 60.0),
        corner_radius_mm=4.0,
    )
    layout.mounting_holes[0].screw = "M2.5"
    again = ManualLayout.from_dict(layout.to_dict())
    assert again.outline == layout.outline
    assert again.mounting_holes[0].screw == "M2.5"
    assert again.schema_version == "manual_layout.v2"


def test_unknown_schema_version_rejected():
    bad = _v1_payload()
    bad["schema_version"] = "manual_layout.v99"
    with pytest.raises(ValueError, match="schema_version"):
        ManualLayout.from_dict(bad)
