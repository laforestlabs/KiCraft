"""Lock the leaf Edge.Cuts contour to the leaf silk-outline contour.

Edge.Cuts and the leaf F.SilkS poly both define the visible leaf boundary
in the canvas / monitor renders. Before the unification, the silk was a
rounded poly while Edge.Cuts was a sharp 4-segment rectangle, leaving a
visible ring of bare PCB substrate at each corner. These tests lock the
two layers to a single source of truth -- ``leaf_outline_polyline`` --
and verify the in-process stamper pulls the same point list onto
Edge.Cuts as the silk producer writes onto F.SilkS.
"""

from __future__ import annotations

import math

from kicraft.autoplacer.brain.leaf_routing import _outline_around_geometry
from kicraft.autoplacer.brain.subcircuit_solver import (
    _build_leaf_silkscreen,
    leaf_outline_polyline,
)
from kicraft.autoplacer.brain.types import Component, Layer, Point, SilkscreenElement
from kicraft.autoplacer.hardware.adapter import _extract_leaf_outline_polyline_mm


CFG = {
    "silkscreen_margin_mm": 0.5,
    "silkscreen_corner_radius_mm": 1.0,
    "group_labels": {"U1": "FAKE LEAF"},
}

TOL_MM = 1e-6


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


def _bbox_outer_rect(components: dict[str, Component], cfg: dict) -> tuple[float, float, float, float]:
    bbox = _component_bbox(components)
    m = float(cfg["silkscreen_margin_mm"])
    return (bbox["min_x"] - m, bbox["min_y"] - m, bbox["max_x"] + m, bbox["max_y"] + m)


def test_silk_poly_uses_canonical_leaf_outline_polyline():
    """The silk producer must emit exactly the canonical leaf-outline
    polyline -- the single source of truth that Edge.Cuts also consumes.

    If this fails the silk producer has been forked off from
    ``leaf_outline_polyline`` (or its inputs differ from what
    ``_build_leaf_silkscreen`` actually feeds in). When silk and the
    canonical helper drift, Edge.Cuts (which traces the canonical
    helper's output via the stamper) will visually diverge from silk
    in every leaf render.
    """
    components = _fake_components()
    bbox = _component_bbox(components)
    x0, y0, x1, y1 = _bbox_outer_rect(components, CFG)

    expected = leaf_outline_polyline(x0, y0, x1, y1, radius_mm=CFG["silkscreen_corner_radius_mm"])
    silk_pts = _silk_poly_points(
        _build_leaf_silkscreen(components, bbox, extraction=None, config=CFG)
    )

    assert silk_pts, "expected a silk poly element from _build_leaf_silkscreen"
    assert len(silk_pts) == len(expected), (
        f"silk has {len(silk_pts)} vertices but canonical outline has "
        f"{len(expected)}; the two pipelines have diverged"
    )
    for s, e in zip(silk_pts, expected, strict=True):
        assert math.isclose(s.x, e.x, abs_tol=TOL_MM), f"silk x diverges: {s} vs {e}"
        assert math.isclose(s.y, e.y, abs_tol=TOL_MM), f"silk y diverges: {s} vs {e}"


def test_extract_leaf_outline_polyline_returns_silk_poly_points():
    """The stamper hands the silk poly's exact ``.points`` to Edge.Cuts.

    ``_extract_leaf_outline_polyline_mm`` is the shim that pulls those
    points off the BoardState; this test pins it to that contract so a
    future refactor that introduces a separate Edge.Cuts polyline source
    (and thus reintroduces drift) fails loudly here.
    """
    components = _fake_components()
    silk_elements = _build_leaf_silkscreen(
        components, _component_bbox(components), extraction=None, config=CFG,
    )
    silk_pts = _silk_poly_points(silk_elements)
    assert silk_pts

    extracted = _extract_leaf_outline_polyline_mm(silk_elements)
    assert extracted is not None, "stamper extractor returned None for a labeled leaf"
    assert len(extracted) == len(silk_pts)
    for (ex, ey), p in zip(extracted, silk_pts, strict=True):
        assert math.isclose(ex, p.x, abs_tol=TOL_MM)
        assert math.isclose(ey, p.y, abs_tol=TOL_MM)


def test_outline_aabb_matches_silk_aabb():
    """The leaf bbox stored in ``state.board_outline`` and the silk poly's
    AABB must agree so any consumer that uses the bbox (parent composer,
    placement scorers) sees the same rectangle the rounded silk fits in.
    """
    components = _fake_components()
    outline = _outline_around_geometry(components, CFG)
    assert outline is not None
    tl, br = outline

    silk_pts = _silk_poly_points(
        _build_leaf_silkscreen(components, _component_bbox(components), extraction=None, config=CFG)
    )
    assert silk_pts
    sx0 = min(p.x for p in silk_pts)
    sy0 = min(p.y for p in silk_pts)
    sx1 = max(p.x for p in silk_pts)
    sy1 = max(p.y for p in silk_pts)

    assert math.isclose(tl.x, sx0, abs_tol=TOL_MM)
    assert math.isclose(tl.y, sy0, abs_tol=TOL_MM)
    assert math.isclose(br.x, sx1, abs_tol=TOL_MM)
    assert math.isclose(br.y, sy1, abs_tol=TOL_MM)


def test_unlabeled_leaf_has_no_outline_polyline():
    """Leaves with no ``group_labels`` match still produce no silk poly --
    legacy behaviour preserved. The Edge.Cuts stamper then falls back to
    a sharp rectangle (its previous shape), which is fine because there's
    no rounded silk to visually disagree with.
    """
    components = _fake_components()
    cfg_no_labels = dict(CFG)
    cfg_no_labels["group_labels"] = {}

    silk_elements = _build_leaf_silkscreen(
        components, _component_bbox(components), extraction=None, config=cfg_no_labels,
    )
    assert silk_elements == []
    assert _extract_leaf_outline_polyline_mm(silk_elements) is None


def test_extractor_returns_none_for_non_poly_silk():
    """The extractor must ignore silk text and any non-F.SilkS polys so
    Edge.Cuts only ever traces the dedicated leaf-outline poly, not e.g.
    a B.SilkS marking or the leaf label text.
    """
    silkscreen = [
        SilkscreenElement(kind="text", layer="F.SilkS", text="LABEL"),
        SilkscreenElement(
            kind="poly",
            layer="B.SilkS",
            points=[Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)],
        ),
    ]
    assert _extract_leaf_outline_polyline_mm(silkscreen) is None
