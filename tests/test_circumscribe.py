"""Phase 3a: circumscribe a parametric outline shape around placed content.

The auto (brief-driven) path places the circuit in a rectangular AABB, then
grows the requested shape around it so nothing lands outside Edge.Cuts. These
tests pin that the grown shape (a) actually contains the content, (b) round-
trips through OutlineSpec.from_dict (the validator's path), and (c) is wired
into _fit_requested_shape with the right no-op conditions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kicraft.autoplacer.brain.types import Point
from kicraft.cli._compose_validate import _fit_requested_shape
from kicraft.layout_editor.outline import OutlineSpec, circumscribe

CONTENT = (Point(0.0, 0.0), Point(40.0, 20.0))


# --------------------------------------------------------------------------- #
# circumscribe()
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shape", ["circle", "rounded_rect", "chamfered_rect"])
def test_circumscribe_contains_content(shape):
    tl, br = CONTENT
    spec = circumscribe(shape, tl, br)
    assert spec.shape == shape
    # The placed content must be fully inside the grown shape.
    assert spec.contains_rect(tl.x, tl.y, br.x, br.y, tol=0.0)
    # And it is centered on the content.
    cx = (spec.min_pt.x + spec.max_pt.x) / 2
    cy = (spec.min_pt.y + spec.max_pt.y) / 2
    assert cx == pytest.approx((tl.x + br.x) / 2)
    assert cy == pytest.approx((tl.y + br.y) / 2)


def test_circumscribe_circle_is_square_and_tight():
    tl, br = CONTENT
    spec = circumscribe("circle", tl, br)
    assert spec.width_mm == pytest.approx(spec.height_mm)  # square bbox
    # Tight-ish: diameter near the content diagonal, not wildly larger.
    diag = (40.0**2 + 20.0**2) ** 0.5
    assert diag <= spec.width_mm <= diag + 4.0


def test_circumscribe_roundtrips_through_from_dict():
    # The validator rebuilds the spec via OutlineSpec.from_dict(manual_outline);
    # the circumscribed spec must satisfy its stricter rules (square circle bbox,
    # corner_radius/chamfer > 0).
    for shape in ("circle", "rounded_rect", "chamfered_rect"):
        spec = circumscribe(shape, *CONTENT)
        rebuilt = OutlineSpec.from_dict(spec.to_dict())
        assert rebuilt.shape == shape
        assert rebuilt.contains_rect(0.0, 0.0, 40.0, 20.0, tol=0.0)


def test_circumscribe_honors_explicit_corner_radius():
    spec = circumscribe("rounded_rect", *CONTENT, corner_radius_mm=2.0)
    assert spec.corner_radius_mm == 2.0


def test_circumscribe_rejects_unknown_shape():
    with pytest.raises(ValueError):
        circumscribe("snowman", *CONTENT)


# --------------------------------------------------------------------------- #
# _fit_requested_shape()
# --------------------------------------------------------------------------- #

def _state(requested_shape, outline=CONTENT, manual_outline=None):
    board_state = SimpleNamespace(board_outline=outline)
    composition = SimpleNamespace(board_state=board_state)
    return SimpleNamespace(
        manual_outline=manual_outline,
        requested_shape=requested_shape,
        composition=composition,
    )


def test_fit_circumscribes_and_sets_manual_outline():
    st = _state({"shape": "circle", "size_mm": 50.0})
    result = _fit_requested_shape(st)
    assert result["fitted"] is True
    assert result["shape"] == "circle"
    # manual_outline now drives the stamp/validate/pour path.
    assert st.manual_outline["shape"] == "circle"
    # board_outline AABB tracks the spec; still encloses the content.
    spec = OutlineSpec.from_dict(st.manual_outline)
    assert spec.contains_rect(0.0, 0.0, 40.0, 20.0, tol=0.0)
    assert st.composition.board_state.board_outline == spec.aabb()


def test_fit_noop_for_rect():
    st = _state({"shape": "rect"})
    assert _fit_requested_shape(st)["fitted"] is False
    assert st.manual_outline is None


def test_fit_noop_when_no_requested_shape():
    st = _state(None)
    assert _fit_requested_shape(st)["fitted"] is False
    assert st.manual_outline is None


# --------------------------------------------------------------------------- #
# PR-N3: occupied-geometry fit (shaped-compose-leaf-nesting)
# --------------------------------------------------------------------------- #

def _ring_rects(cx=28.5, cy=28.5, r=24.0, n=12, half=2.5):
    import math

    rects = []
    for k in range(n):
        ang = 2.0 * math.pi * k / n
        x, y = cx + r * math.cos(ang), cy + r * math.sin(ang)
        rects.append((Point(x - half, y - half), Point(x + half, y + half)))
    return rects


def test_circumscribe_content_rects_fits_occupied_not_aabb_diagonal():
    # A 57x57 AABB whose occupied geometry is an annulus of 12 pads at r=24:
    # AABB mode must pay the diagonal (⌀ >= ~80); occupied mode fits ~⌀55.
    rects = _ring_rects()
    aabb = circumscribe("circle", Point(0, 0), Point(57, 57))
    occ = circumscribe(
        "circle", Point(0, 0), Point(57, 57), content_rects=rects
    )
    assert aabb.width_mm >= 79.0
    assert 52.0 <= occ.width_mm <= 58.0, occ.width_mm
    # Every occupied rect is inside the tighter fit.
    for r0, r1 in rects:
        assert occ.contains_rect(r0.x, r0.y, r1.x, r1.y, tol=0.0)


def test_circumscribe_content_rects_solid_matches_aabb():
    # Content rects that reach the AABB corners: identical fit to AABB mode.
    rects = [(Point(0, 0), Point(40, 20))]
    a = circumscribe("circle", Point(0, 0), Point(40, 20))
    b = circumscribe("circle", Point(0, 0), Point(40, 20), content_rects=rects)
    assert abs(a.width_mm - b.width_mm) < 0.1


def test_circumscribe_empty_content_rects_falls_back_to_aabb():
    a = circumscribe("circle", Point(0, 0), Point(40, 20))
    b = circumscribe("circle", Point(0, 0), Point(40, 20), content_rects=[])
    assert abs(a.width_mm - b.width_mm) < 1e-9


def test_polygon_circumscribe_content_rects_shrinks():
    from kicraft.shapes import circumscribe as circumscribe_polygon

    rects = _ring_rects()
    aabb_poly = circumscribe_polygon("hexagon", Point(0, 0), Point(57, 57))
    occ_poly = circumscribe_polygon(
        "hexagon", Point(0, 0), Point(57, 57), content_rects=rects
    )
    (ax0, _), (ax1, _) = aabb_poly.aabb()
    (ox0, _), (ox1, _) = occ_poly.aabb()
    assert (ox1 - ox0) < (ax1 - ax0) - 5.0
    for r0, r1 in rects:
        assert occ_poly.contains_rect(r0.x, r0.y, r1.x, r1.y, tol=0.0)


def test_fit_uses_occupied_geometry_when_components_present():
    # Hollow composition: component physical bboxes form the annulus; the
    # fitted circle must be sized to the occupied extent (grown to the 60mm
    # target), NOT the AABB diagonal (~80.4 -> guard rejection at cap).
    comps = {}
    for i, (r0, r1) in enumerate(_ring_rects()):
        rect = (r0, r1)
        comps[f"D{i+1}"] = SimpleNamespace(physical_bbox=lambda rect=rect: rect)
    board_state = SimpleNamespace(
        board_outline=(Point(0.0, 0.0), Point(57.0, 57.0)),
        components=comps, traces=[], vias=[],
    )
    st = SimpleNamespace(
        manual_outline=None,
        requested_shape={"shape": "circle", "size_mm": 60.0},
        composition=SimpleNamespace(board_state=board_state),
    )
    result = _fit_requested_shape(st)
    assert result["fitted"] is True, result
    assert result["size_mm"] == [60.0, 60.0]


def test_fit_noop_when_manual_outline_authoritative():
    st = _state({"shape": "circle"}, manual_outline={"shape": "rect"})
    res = _fit_requested_shape(st)
    assert res["fitted"] is False
    assert "manual" in res["reason"]


def test_fit_circle_uses_parametric_path():
    # circle is in both shape sets; the parametric (OutlineSpec) path wins.
    st = _state({"shape": "circle"})
    res = _fit_requested_shape(st)
    assert res["fitted"] is True
    assert res["kind"] == "parametric"
    assert st.manual_outline is not None
    assert getattr(st, "fitted_polygon", None) in (None, [])


@pytest.mark.parametrize("shape", ["hexagon", "octagon", "star", "heart", "snowman", "triangle"])
def test_fit_named_shape_sets_fitted_polygon(shape):
    st = _state({"shape": shape})
    st.fitted_polygon = None
    res = _fit_requested_shape(st)
    assert res["fitted"] is True, res
    assert res["kind"] == "polygon"
    # Polygon channel set; the JS-mirrored OutlineSpec channel left untouched.
    assert st.manual_outline is None
    assert st.fitted_polygon and len(st.fitted_polygon) >= 3
    # board_outline AABB now tracks the polygon bbox (as Points).
    tl, br = st.composition.board_state.board_outline
    assert br.x > tl.x and br.y > tl.y


def test_fit_named_shape_polygon_contains_content():
    # The validator rebuilds containment from fitted_polygon; it must enclose
    # the original placed content.
    from kicraft.shapes import polygon_outline_from_points

    st = _state({"shape": "snowman"})
    st.fitted_polygon = None
    _fit_requested_shape(st)
    checker = polygon_outline_from_points(st.fitted_polygon)
    assert checker.contains_rect(0.0, 0.0, 40.0, 20.0, tol=0.05)


def test_fit_skips_unsupported_named_shape():
    # A name with no generator (e.g. "gear") is a graceful no-op -> rect board,
    # not a crash.
    st = _state({"shape": "gear"})
    st.fitted_polygon = None
    res = _fit_requested_shape(st)
    assert res["fitted"] is False
    assert "not supported" in res["reason"]
    assert st.manual_outline is None
    assert st.fitted_polygon is None


# --------------------------------------------------------------------------- #
# Shape-aware placement (GAP 1a): inscribed_rect_bound + size_mm-as-target
# --------------------------------------------------------------------------- #

from kicraft.cli._compose_validate import (  # noqa: E402
    _SHAPE_SIZE_TOL,
    inscribed_rect_bound,
)


def test_inscribed_rect_bound_circle_matches_analytic():
    # Largest w x h rect (aspect a) whose circumscribed circle lands AT the
    # ⌀60 target: hypot(w, h) == 60 (up to the circumscribe margin). The
    # guard's 5% slack is deliberately NOT consumed by the aim — it stays
    # available to absorb packing overshoot past the seed.
    for aspect in (1.0, 1.39, 0.7):
        w, h = inscribed_rect_bound({"shape": "circle", "size_mm": 60.0}, aspect)
        assert w / h == pytest.approx(aspect, rel=1e-3)
        diag = (w**2 + h**2) ** 0.5
        assert diag == pytest.approx(60.0, abs=1.5)
        assert diag < 60.0 * (1.0 + _SHAPE_SIZE_TOL) - 1.0  # slack left over


def test_inscribed_rect_bound_none_without_shape_or_size():
    assert inscribed_rect_bound(None, 1.0) is None
    assert inscribed_rect_bound({"shape": "rect", "size_mm": 60.0}, 1.0) is None
    assert inscribed_rect_bound({"shape": "circle"}, 1.0) is None
    assert inscribed_rect_bound({"shape": "gear", "size_mm": 60.0}, 1.0) is None


def test_inscribed_rect_bound_roundtrips_through_fit():
    # The bound is DEFINED by the stamp-time guard: content at the bound must
    # circumscribe-fit; content 15% past it must be rejected. This is the
    # placement<->stamp contract that makes GAP 1a coherent end-to-end.
    req = {"shape": "circle", "size_mm": 60.0}
    w, h = inscribed_rect_bound(req, 1.0)
    st = _state(req, outline=(Point(0.0, 0.0), Point(w - 0.1, h - 0.1)))
    assert _fit_requested_shape(st)["fitted"] is True
    st = _state(req, outline=(Point(0.0, 0.0), Point(w * 1.15, h * 1.15)))
    res = _fit_requested_shape(st)
    assert res["fitted"] is False
    assert "exceeds requested size_mm" in res["reason"]


def test_fit_grows_parametric_shape_to_requested_size():
    # "round 60 mm" with small content must deliver ⌀60, not the minimal
    # circumscribed ⌀~23 — size_mm is a target, not only a cap.
    st = _state({"shape": "circle", "size_mm": 60.0}, outline=(Point(0.0, 0.0), Point(20.0, 10.0)))
    res = _fit_requested_shape(st)
    assert res["fitted"] is True
    spec = OutlineSpec.from_dict(st.manual_outline)
    assert spec.width_mm == pytest.approx(60.0)
    assert spec.height_mm == pytest.approx(60.0)
    # Content still inside, and centered on it.
    assert spec.contains_rect(0.0, 0.0, 20.0, 10.0, tol=0.0)


def test_fit_does_not_shrink_below_content_within_tolerance():
    # Content whose circumscribed circle lands in the (target, target*1.05]
    # slack band keeps its (larger) fitted size — grow-only, never shrink.
    st = _state({"shape": "circle", "size_mm": 44.0}, outline=(Point(0.0, 0.0), Point(40.0, 20.0)))
    res = _fit_requested_shape(st)
    assert res["fitted"] is True
    spec = OutlineSpec.from_dict(st.manual_outline)
    assert spec.width_mm >= (40.0**2 + 20.0**2) ** 0.5  # >= content diagonal
    assert spec.contains_rect(0.0, 0.0, 40.0, 20.0, tol=0.0)


def test_fit_grows_named_polygon_to_requested_size():
    st = _state({"shape": "hexagon", "size_mm": 60.0}, outline=(Point(0.0, 0.0), Point(15.0, 12.0)))
    st.fitted_polygon = None
    res = _fit_requested_shape(st)
    assert res["fitted"] is True, res
    w, h = res["size_mm"]
    # Uniform scale: the limiting axis reaches the target, neither exceeds it.
    assert max(w, h) == pytest.approx(60.0, abs=0.1)
    assert w <= 60.0 + 0.1 and h <= 60.0 + 0.1
