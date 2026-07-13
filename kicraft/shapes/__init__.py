"""Named / compound board outline shapes, backed by Shapely.

The intent stage captures a shape NAME (``hexagon``, ``star``, ``heart``,
``snowman``, …); this package turns a name into a concrete polygon for
``Edge.Cuts`` and sizes it to enclose the placed circuit. It is the home of the
boolean-union machinery (a snowman is the union of three circles) and the
containment / scale-to-fit helpers the compose pipeline needs.

Frame: generators build in conventional Y-up math coordinates, then flip to the
KiCad Y-down board frame so an asymmetric shape (heart cusp down, snowman head
up) comes out the right way round. Everything is normalized to a unit shape
(centered at the origin, longest bbox side = 1.0); :func:`circumscribe` then
scales and positions it around the content.

The four parametric convex shapes (``rect`` / ``rounded_rect`` / ``circle`` /
``chamfered_rect``) are handled WITHOUT Shapely by
:class:`kicraft.layout_editor.outline.OutlineSpec`; this package is for the
named and compound shapes that need a general polygon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.affinity import scale as _scale
from shapely.affinity import translate as _translate
from shapely.geometry import Point as _SPoint
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

__all__ = [
    "KNOWN_SHAPES",
    "UnknownShapeError",
    "PolygonOutline",
    "build_unit_polygon",
    "circumscribe",
    "polygon_outline_from_points",
]


class UnknownShapeError(ValueError):
    """Raised for a shape name with no generator (caller should fall back)."""


# --------------------------------------------------------------------------- #
# Primitive generators (Y-up math frame, arbitrary scale; normalized later)
# --------------------------------------------------------------------------- #

def _regular(n: int, *, rot_deg: float = 90.0) -> Polygon:
    """Regular n-gon on the unit circle. Default rotation puts a vertex at top
    (a flat edge at the bottom for even n) -- a natural board orientation."""
    a0 = math.radians(rot_deg)
    return Polygon(
        [(math.cos(a0 + 2 * math.pi * k / n), math.sin(a0 + 2 * math.pi * k / n))
         for k in range(n)]
    )


def _star(points: int = 5, *, inner_ratio: float = 0.45, rot_deg: float = 90.0) -> Polygon:
    a0 = math.radians(rot_deg)
    verts = []
    for k in range(points * 2):
        r = 1.0 if k % 2 == 0 else inner_ratio
        a = a0 + math.pi * k / points
        verts.append((r * math.cos(a), r * math.sin(a)))
    return Polygon(verts)


def _heart(n: int = 240) -> Polygon:
    # Classic parametric heart; lobes at +y, cusp at -y (Y-up).
    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    x = 16.0 * np.sin(t) ** 3
    y = (13.0 * np.cos(t) - 5.0 * np.cos(2 * t)
         - 2.0 * np.cos(3 * t) - np.cos(4 * t))
    return Polygon(np.column_stack([x, y]))


def _snowman(radii: tuple[float, ...] = (1.0, 0.70, 0.50)) -> Polygon:
    """Union of stacked circles, base (largest) at the bottom (-y), head at the
    top (+y). Centers are spaced closer than the sum of radii so the union is a
    single connected silhouette with narrow necks between blobs."""
    circles = []
    cy = 0.0
    prev_r: float | None = None
    for r in radii:
        if prev_r is None:
            cy = 0.0
        else:
            cy += (prev_r + r) * 0.82  # 18% overlap -> connected necks
        circles.append(_SPoint(0.0, cy).buffer(r, quad_segs=72))
        prev_r = r
    return unary_union(circles)


_GENERATORS = {
    "circle": lambda **k: _SPoint(0.0, 0.0).buffer(1.0, quad_segs=96),
    "triangle": lambda **k: _regular(3),
    "pentagon": lambda **k: _regular(5),
    "hexagon": lambda **k: _regular(6),
    "octagon": lambda **k: _regular(8),
    "star": lambda **k: _star(int(k.get("points", 5)), inner_ratio=float(k.get("inner_ratio", 0.45))),
    "heart": lambda **k: _heart(),
    "snowman": lambda **k: _snowman(),
}

KNOWN_SHAPES: frozenset[str] = frozenset(_GENERATORS)


def _normalize(poly: Polygon) -> Polygon:
    """Flip to KiCad Y-down, recenter on the bbox, scale longest side to 1.0."""
    if not poly.is_valid:
        poly = poly.buffer(0)
    poly = _scale(poly, 1.0, -1.0, origin=(0.0, 0.0))  # Y-up math -> Y-down KiCad
    minx, miny, maxx, maxy = poly.bounds
    poly = _translate(poly, -(minx + maxx) / 2.0, -(miny + maxy) / 2.0)
    dim = max(maxx - minx, maxy - miny)
    if dim > 0:
        poly = _scale(poly, 1.0 / dim, 1.0 / dim, origin=(0.0, 0.0))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def build_unit_polygon(name: str, **params) -> Polygon:
    """Validated unit polygon (KiCad frame, centered, longest side 1.0)."""
    key = (name or "").strip().lower()
    gen = _GENERATORS.get(key)
    if gen is None:
        raise UnknownShapeError(f"unknown shape {name!r}; known: {sorted(KNOWN_SHAPES)}")
    return _normalize(gen(**params))


# --------------------------------------------------------------------------- #
# Circumscribe a named polygon around placed content
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class PolygonOutline:
    """A positioned, sized board-outline polygon (KiCad mm frame)."""

    polygon: Polygon

    def points(self) -> list[tuple[float, float]]:
        """Closed exterior ring as (x, y), without the duplicate closing point."""
        return [(float(x), float(y)) for x, y in self.polygon.exterior.coords[:-1]]

    def aabb(self) -> tuple[tuple[float, float], tuple[float, float]]:
        minx, miny, maxx, maxy = self.polygon.bounds
        return ((minx, miny), (maxx, maxy))

    def contains_point(self, x: float, y: float, tol: float = 0.0) -> bool:
        geom = self.polygon.buffer(tol) if tol else self.polygon
        return bool(geom.covers(_SPoint(x, y)))

    def contains_rect(
        self, x0: float, y0: float, x1: float, y1: float, tol: float = 0.0
    ) -> bool:
        geom = self.polygon.buffer(tol) if tol else self.polygon
        return bool(geom.covers(box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))))


def polygon_outline_from_points(points) -> PolygonOutline:
    """Rebuild a :class:`PolygonOutline` from a closed ring of ``[x, y]`` points
    (the on-state ``fitted_polygon``), for containment checks at validation."""
    return PolygonOutline(Polygon([(float(p[0]), float(p[1])) for p in points]))


def circumscribe(
    name: str,
    min_pt,
    max_pt,
    *,
    margin_mm: float = 0.5,
    content_rects=None,
    **params,
) -> PolygonOutline:
    """Smallest named polygon that fully contains the content rectangle
    ``(min_pt, max_pt)``, centered on it (the same circumscribe strategy as the
    parametric path, generalized to any polygon via a scale binary-search).

    ``min_pt`` / ``max_pt`` are anything with ``.x`` / ``.y`` (e.g. autoplacer
    ``Point``). The placed circuit stays put; the shape grows around it so
    nothing lands outside ``Edge.Cuts``.

    ``content_rects`` (``(min_pt, max_pt)`` pairs): the TRUE occupied
    geometry. When given, the polygon must cover every rect (inflated by
    ``margin_mm``) instead of the single AABB, so a hollow composition gets a
    shape sized to its occupied extent, not the AABB's diagonal
    (shaped-compose-leaf-nesting PR-N3). Centering stays on the AABB.
    """
    unit = build_unit_polygon(name, **params)
    cx = (min_pt.x + max_pt.x) / 2.0
    cy = (min_pt.y + max_pt.y) / 2.0
    hw = (max_pt.x - min_pt.x) / 2.0
    hh = (max_pt.y - min_pt.y) / 2.0
    target = box(cx - hw - margin_mm, cy - hh - margin_mm,
                 cx + hw + margin_mm, cy + hh + margin_mm)
    if content_rects is not None and not content_rects:
        content_rects = None  # empty = no signal; fall back to the AABB
    content_boxes = None
    if content_rects is not None:
        content_boxes = [
            box(min(a.x, b.x) - margin_mm, min(a.y, b.y) - margin_mm,
                max(a.x, b.x) + margin_mm, max(a.y, b.y) + margin_mm)
            for a, b in content_rects
        ]

    def _placed(scale: float) -> Polygon:
        return _translate(_scale(unit, scale, scale, origin=(0.0, 0.0)), cx, cy)

    def _ok(scale: float) -> bool:
        placed = _placed(scale)
        if content_boxes is not None:
            return all(placed.covers(b) for b in content_boxes)
        return bool(placed.covers(target))

    hi = max(hw, hh, 1.0) * 2.0 + 2.0 * margin_mm
    guard = 0
    while not _ok(hi) and guard < 64:
        hi *= 1.3
        guard += 1
    lo = 0.0
    for _ in range(52):
        mid = (lo + hi) / 2.0
        if _ok(mid):
            hi = mid
        else:
            lo = mid
    return PolygonOutline(_placed(hi))
