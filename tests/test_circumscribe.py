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
