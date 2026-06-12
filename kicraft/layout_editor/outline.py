"""Board outline shapes for the manual layout editor.

``OutlineSpec`` models the user's board shape as first-class data:
an axis-aligned bounding box plus a shape tag (rect / rounded-rect /
circle / chamfered-rect) and its parameter. Everything downstream
consumes one of three views of it:

- ``aabb()``: the bounding rectangle, for board-size bookkeeping and
  every consumer that predates shapes;
- ``polyline()``: a closed, convex, clockwise (KiCad Y-down) point
  loop for Edge.Cuts stamping. Arcs are sampled finely enough that
  the worst-case sagitta stays below ``_MAX_SAGITTA_MM``;
- ``contains_point()`` / ``contains_rect()``: analytic containment
  with tolerance, used by geometry validation. Analytic (not
  polyline-sampled) so the answer is exact; the sampling error of
  the stamped polyline is bounded well under the validation margin.

The canvas JS mirrors the generator and containment math
(``layout_canvas.js``); keep the two in sync. The cross-language
agreement test drives both against shared fixtures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from kicraft.autoplacer.brain.types import Point

SHAPES = ("rect", "rounded_rect", "circle", "chamfered_rect")

# Worst-case arc flattening error for sampled circles. 0.02 mm is far
# under the 0.05 mm geometry-validation margin and any fab tolerance.
_MAX_SAGITTA_MM = 0.02
_CIRCLE_MIN_SEGMENTS = 32
_CIRCLE_MAX_SEGMENTS = 128

# Samples per 90-degree corner arc for rounded rects. Matches the leaf
# outline convention (subcircuit_solver.leaf_outline_polyline) so leaf
# and parent rounded corners render identically.
ROUNDED_RECT_N_ARC = 8

_SQRT2 = math.sqrt(2.0)


def rounded_rect_polyline(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    radius_mm: float,
    n_arc: int = ROUNDED_RECT_N_ARC,
) -> list[Point]:
    """Closed rounded-rectangle polyline (clockwise in KiCad Y-down).

    Single source of truth for rounded-rect outlines: leaf silk +
    Edge.Cuts (via ``subcircuit_solver.leaf_outline_polyline``) and
    shaped parent boards both sample from this. Points form a closed
    loop (last point connects back to first; the first point is not
    duplicated at the end). ``n_arc`` is the number of straight-line
    samples per 90 deg corner.
    """
    r = min(radius_mm, (x1 - x0) / 2, (y1 - y0) / 2)
    points: list[Point] = []
    corners = [
        (x0 + r, y0 + r, math.pi, math.pi / 2),       # top-left
        (x1 - r, y0 + r, math.pi / 2, 0),              # top-right
        (x1 - r, y1 - r, 0, -math.pi / 2),             # bottom-right
        (x0 + r, y1 - r, -math.pi / 2, -math.pi),      # bottom-left
    ]
    for cx, cy, a_start, a_end in corners:
        for i in range(n_arc):
            t = a_start + (a_end - a_start) * i / (n_arc - 1)
            px = cx + r * math.cos(t)
            py = cy - r * math.sin(t)  # KiCad Y-down
            points.append(Point(px, py))
    return points


def circle_segment_count(radius_mm: float) -> int:
    """Segments needed to keep the sagitta under ``_MAX_SAGITTA_MM``."""
    if radius_mm <= _MAX_SAGITTA_MM:
        return _CIRCLE_MIN_SEGMENTS
    n = math.ceil(math.pi / math.acos(1.0 - _MAX_SAGITTA_MM / radius_mm))
    return max(_CIRCLE_MIN_SEGMENTS, min(_CIRCLE_MAX_SEGMENTS, int(n)))


@dataclass(slots=True)
class OutlineSpec:
    """Board outline: AABB + shape tag + shape parameter.

    ``corner_radius_mm`` applies to ``rounded_rect``; ``chamfer_mm``
    to ``chamfered_rect``; both are ignored otherwise. The generators
    clamp parameters to ``min(w, h) / 2`` rather than rejecting them,
    matching the leaf-outline convention.
    """

    shape: str
    min_pt: Point
    max_pt: Point
    corner_radius_mm: float = 0.0
    chamfer_mm: float = 0.0

    # -- Constructors ------------------------------------------------------

    @classmethod
    def rect(cls, min_pt: Point, max_pt: Point) -> "OutlineSpec":
        return cls(shape="rect", min_pt=min_pt, max_pt=max_pt)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutlineSpec":
        if not isinstance(data, dict):
            raise ValueError("outline must be a JSON object")
        shape = str(data.get("shape", "rect"))
        if shape not in SHAPES:
            raise ValueError(f"unknown outline shape {shape!r}; expected one of {SHAPES}")
        try:
            min_pt = Point(float(data["min"]["x"]), float(data["min"]["y"]))
            max_pt = Point(float(data["max"]["x"]), float(data["max"]["y"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid outline min/max: {exc}") from exc
        if max_pt.x <= min_pt.x or max_pt.y <= min_pt.y:
            raise ValueError(
                f"outline must satisfy max>min, got min={min_pt}, max={max_pt}"
            )
        spec = cls(
            shape=shape,
            min_pt=min_pt,
            max_pt=max_pt,
            corner_radius_mm=float(data.get("corner_radius_mm", 0.0) or 0.0),
            chamfer_mm=float(data.get("chamfer_mm", 0.0) or 0.0),
        )
        if shape == "circle" and abs(spec.width_mm - spec.height_mm) > 0.01:
            raise ValueError(
                "circle outline requires a square bounding box, got "
                f"{spec.width_mm:.3f} x {spec.height_mm:.3f} mm"
            )
        if shape == "rounded_rect" and spec.corner_radius_mm <= 0:
            raise ValueError("rounded_rect outline requires corner_radius_mm > 0")
        if shape == "chamfered_rect" and spec.chamfer_mm <= 0:
            raise ValueError("chamfered_rect outline requires chamfer_mm > 0")
        return spec

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "shape": self.shape,
            "min": {"x": self.min_pt.x, "y": self.min_pt.y},
            "max": {"x": self.max_pt.x, "y": self.max_pt.y},
        }
        if self.shape == "rounded_rect":
            out["corner_radius_mm"] = self.corner_radius_mm
        if self.shape == "chamfered_rect":
            out["chamfer_mm"] = self.chamfer_mm
        return out

    # -- Views -------------------------------------------------------------

    @property
    def width_mm(self) -> float:
        return self.max_pt.x - self.min_pt.x

    @property
    def height_mm(self) -> float:
        return self.max_pt.y - self.min_pt.y

    def aabb(self) -> tuple[Point, Point]:
        return (self.min_pt, self.max_pt)

    def is_rect(self) -> bool:
        return self.shape == "rect"

    def _clamped_param(self) -> float:
        """Corner radius / chamfer / circle radius clamped to fit."""
        half = min(self.width_mm, self.height_mm) / 2.0
        if self.shape == "rounded_rect":
            return min(self.corner_radius_mm, half)
        if self.shape == "chamfered_rect":
            return min(self.chamfer_mm, half)
        if self.shape == "circle":
            return half
        return 0.0

    def polyline(self) -> list[Point]:
        """Closed convex clockwise (Y-down) loop; first point not repeated."""
        x0, y0 = self.min_pt.x, self.min_pt.y
        x1, y1 = self.max_pt.x, self.max_pt.y
        if self.shape == "rect":
            return [Point(x0, y0), Point(x1, y0), Point(x1, y1), Point(x0, y1)]
        if self.shape == "rounded_rect":
            return rounded_rect_polyline(x0, y0, x1, y1, self._clamped_param())
        if self.shape == "chamfered_rect":
            c = self._clamped_param()
            return [
                Point(x0, y0 + c),
                Point(x0 + c, y0),
                Point(x1 - c, y0),
                Point(x1, y0 + c),
                Point(x1, y1 - c),
                Point(x1 - c, y1),
                Point(x0 + c, y1),
                Point(x0, y1 - c),
            ]
        if self.shape == "circle":
            r = self._clamped_param()
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            n = circle_segment_count(r)
            pts = []
            for k in range(n):
                t = math.pi - 2.0 * math.pi * k / n
                pts.append(Point(cx + r * math.cos(t), cy - r * math.sin(t)))
            return pts
        raise ValueError(f"unknown outline shape {self.shape!r}")

    # -- Containment (analytic, with tolerance) -----------------------------

    def contains_point(self, x: float, y: float, tol: float = 0.0) -> bool:
        x0, y0 = self.min_pt.x, self.min_pt.y
        x1, y1 = self.max_pt.x, self.max_pt.y
        if x < x0 - tol or x > x1 + tol or y < y0 - tol or y > y1 + tol:
            return False
        if self.shape == "rect":
            return True
        if self.shape == "circle":
            r = self._clamped_param()
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            return math.hypot(x - cx, y - cy) <= r + tol
        if self.shape == "chamfered_rect":
            c = self._clamped_param()
            # Four 45-degree half-planes; distance to a x+y=k line is
            # |dx+dy|/sqrt(2), so scale the tolerance accordingly.
            t = tol * _SQRT2
            return (
                (x - x0) + (y - y0) >= c - t
                and (x1 - x) + (y - y0) >= c - t
                and (x1 - x) + (y1 - y) >= c - t
                and (x - x0) + (y1 - y) >= c - t
            )
        if self.shape == "rounded_rect":
            r = self._clamped_param()
            # Inside the AABB; only the four corner squares constrain
            # further: there the point must lie within r of the arc
            # center.
            nearest_cx = min(max(x, x0 + r), x1 - r)
            nearest_cy = min(max(y, y0 + r), y1 - r)
            return math.hypot(x - nearest_cx, y - nearest_cy) <= r + tol
        raise ValueError(f"unknown outline shape {self.shape!r}")

    def contains_rect(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        tol: float = 0.0,
    ) -> bool:
        """All four corners inside; sufficient for convex shapes."""
        return (
            self.contains_point(min_x, min_y, tol)
            and self.contains_point(max_x, min_y, tol)
            and self.contains_point(max_x, max_y, tol)
            and self.contains_point(min_x, max_y, tol)
        )

    # -- Mounting-hole corner pegs ------------------------------------------

    def mounting_hole_position(self, corner: str, inset_mm: float) -> Point:
        """Corner-pegged hole center, shape-aware.

        Defined as: walk inward along the 45-degree diagonal from the
        AABB corner; start where that diagonal enters the shape, then
        go ``inset_mm * sqrt(2)`` further. For a plain rect the entry
        point IS the corner, which reduces to the historical
        ``corner + (inset, inset)`` behaviour. The entry depth has a
        closed form for every supported shape:

        - rect: 0
        - rounded_rect: r * (sqrt(2) - 1)
        - chamfered_rect: c / sqrt(2)
        - circle (square AABB): R * (sqrt(2) - 1)
        """
        x0, y0 = self.min_pt.x, self.min_pt.y
        x1, y1 = self.max_pt.x, self.max_pt.y
        sx, sy = {
            "top-left": (1.0, 1.0),
            "top-right": (-1.0, 1.0),
            "bottom-left": (1.0, -1.0),
            "bottom-right": (-1.0, -1.0),
        }[corner]
        cx = x0 if sx > 0 else x1
        cy = y0 if sy > 0 else y1

        p = self._clamped_param()
        if self.shape == "rect":
            entry = 0.0
        elif self.shape in ("rounded_rect", "circle"):
            entry = p * (_SQRT2 - 1.0)
        elif self.shape == "chamfered_rect":
            entry = p / _SQRT2
        else:
            raise ValueError(f"unknown outline shape {self.shape!r}")

        per_axis = entry / _SQRT2 + inset_mm
        return Point(cx + sx * per_axis, cy + sy * per_axis)
