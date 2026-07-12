"""Parent-board geometry validation, outline repair, and view rendering.

Split out of ``compose_subcircuits.py`` (Lever 2.5). These operate on a stamped
parent board (pcbnew) + the ``ParentCompositionState``; they call no other
compose internal. Re-exported from ``compose_subcircuits`` so the external API
(``_repair_parent_outline`` / ``_validate_parent_geometry``) keeps resolving.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from kicraft.autoplacer.brain.types import Point
from kicraft.cli._compose_state import ParentCompositionState


def _repair_parent_outline(
    state: ParentCompositionState,
    *,
    margin_mm: float = 2.0,
    pad_edge_clearance_mm: float = 0.2,
    verify_only: bool = False,
) -> dict[str, Any]:
    """Grow the parent board outline so it encloses all placed geometry.

    The constraint-aware outline (``_compute_final_outline``) snaps
    edge-constrained sides to their anchor coordinate and can therefore come
    out *smaller* than the placed-content bbox, leaving footprints, pads or
    stamped leaf copper outside ``Edge.Cuts``. FreeRouting cannot produce an
    SES for a board with copper outside the outline (``rc=-1``), so we repair
    the outline before validation/stamping.

    The required extent mirrors the rule enforced by
    :func:`_validate_parent_geometry`:

    * every non-edge-constrained component **body** (courtyard) must fit,
    * **all** pad copper must fit (edge connectors included),
    * every stamped trace and via must fit,

    with ``margin_mm`` of copper-to-edge breathing room. Edge-constrained
    component *bodies* are exempt (their housing legitimately mounts past the
    PCB edge), so such refs contribute only their pads -- which, for a
    correctly flush-mounted connector, already sit inboard and do not push the
    edge out. The outline is only ever **grown**, never shrunk, so a
    constraint-aware outline that already encloses everything (the normal
    case) is left untouched and flush edge-mounting is preserved.

    Mutates ``state.composition.board_state.board_outline`` in place when a
    grow is needed; :func:`_stamp_parent_board` then derives ``Edge.Cuts``
    from it, keeping the artifact and in-memory state in sync. Returns a small
    dict describing whether the outline changed and its old/new size.

    When ``verify_only=True`` (Phase 3A), the outline is NOT mutated: the
    function computes whether a grow WOULD be needed and emits a diagnostic
    (stderr) when it would, acting as a verify-only assert that the
    containment invariant in :func:`_compute_final_outline` held. The
    returned dict carries ``would_repair``/``would_change_mm`` instead of
    mutating.
    """
    composition = state.composition
    if composition is None:
        return {"repaired": False, "reason": "no composition"}
    if state.manual_outline is not None:
        # Manual mode: the user's outline is authoritative. Growing it
        # silently would deliver a different board than the one drawn
        # in the editor; geometry validation right after this fails
        # loudly instead, and the editor surfaces the violations.
        return {"repaired": False, "reason": "manual outline is authoritative"}
    outline = composition.board_state.board_outline
    if not outline or len(outline) < 2:
        return {"repaired": False, "reason": "no outline"}

    tl, br = outline
    edge_constrained = set(state.edge_constrained_refs or ())
    # Sides of the outline that an edge-mount CONNECTOR's mouth defines. On
    # these the constraint-aware outline (tl/br) already sits at mouth+overhang;
    # the repair must NOT add breathing-room margin there or it buries the port
    # behind a neighbor part sitting just inboard of the mouth.
    conn_sides = set(state.edge_zoned_outline_sides or ())

    # Two requirement boxes over the SAME geometry (non-edge-constrained bodies
    # + all pads + traces + vias): `req` gets margin_mm of copper-to-edge
    # breathing room; `flr` is the zero-margin floor used on connector sides so
    # geometry still stays inside the board without pushing the port-edge out.
    req_min_x = req_min_y = float("inf")
    req_max_x = req_max_y = float("-inf")
    flr_min_x = flr_min_y = float("inf")
    flr_max_x = flr_max_y = float("-inf")

    def _grow(p_tl: Point, p_br: Point) -> None:
        nonlocal req_min_x, req_min_y, req_max_x, req_max_y
        nonlocal flr_min_x, flr_min_y, flr_max_x, flr_max_y
        req_min_x = min(req_min_x, p_tl.x)
        req_min_y = min(req_min_y, p_tl.y)
        req_max_x = max(req_max_x, p_br.x)
        req_max_y = max(req_max_y, p_br.y)
        flr_min_x = min(flr_min_x, p_tl.x)
        flr_min_y = min(flr_min_y, p_tl.y)
        flr_max_x = max(flr_max_x, p_br.x)
        flr_max_y = max(flr_max_y, p_br.y)

    for ref, comp in (composition.board_state.components or {}).items():
        if ref not in edge_constrained:
            b_tl, b_br = comp.bbox()
            _grow(b_tl, b_br)
        for pad in comp.pads:
            p_tl, p_br = pad.bbox()
            _grow(p_tl, p_br)

    for trace in composition.board_state.traces or []:
        _grow(
            Point(min(trace.start.x, trace.end.x), min(trace.start.y, trace.end.y)),
            Point(max(trace.start.x, trace.end.x), max(trace.start.y, trace.end.y)),
        )
    for via in composition.board_state.vias or []:
        _grow(via.pos, via.pos)

    if req_min_x == float("inf"):
        return {"repaired": False, "reason": "no geometry"}

    req_min_x -= margin_mm
    req_min_y -= margin_mm
    req_max_x += margin_mm
    req_max_y += margin_mm

    new_tl = Point(min(tl.x, req_min_x), min(tl.y, req_min_y))
    new_br = Point(max(br.x, req_max_x), max(br.y, req_max_y))

    # On connector-defined sides, keep the edge at the constraint-aware outline
    # (mouth + overhang) and grow ONLY to the floor -- so a neighbor part inboard
    # of the mouth, or the full breathing-room margin, can never push the board
    # out past the port. The floor leaves ``pad_edge_clearance_mm`` of copper-to-
    # edge clearance from the placed copper, so an edge-mount connector whose
    # edge-facing pads sit at its body front (a BNC GND shield, a flush switch)
    # gets the cut line pulled that far outboard of its pads -- the fix for
    # pads-flush-with-edge copper_edge_clearance DRC -- while a connector whose
    # pads already sit well inboard of the mouth is untouched (the mouth+overhang
    # edge stays outboard of pad+clearance). The mouth/body (excluded from the
    # floor) still overhangs, so the port stays accessible.
    clr = max(0.0, float(pad_edge_clearance_mm))
    if "left" in conn_sides:
        new_tl = Point(min(tl.x, flr_min_x - clr), new_tl.y)
    if "top" in conn_sides:
        new_tl = Point(new_tl.x, min(tl.y, flr_min_y - clr))
    if "right" in conn_sides:
        new_br = Point(max(br.x, flr_max_x + clr), new_br.y)
    if "bottom" in conn_sides:
        new_br = Point(new_br.x, max(br.y, flr_max_y + clr))

    changed = (
        abs(new_tl.x - tl.x) > 1e-6
        or abs(new_tl.y - tl.y) > 1e-6
        or abs(new_br.x - br.x) > 1e-6
        or abs(new_br.y - br.y) > 1e-6
    )
    if verify_only:
        # Verify-only assert (Phase 3A): do NOT mutate. The containment
        # invariant in _compute_final_outline should make this unreachable;
        # a would-change hit means the bbox-level clamp missed geometry the
        # repair covers (pads / traces / vias, or a corner-escape case) and
        # is a breadcrumb for tightening the pure function, not a silent fix.
        if changed:
            import sys as _sys
            print(
                "[outline] verify-only _repair_parent_outline WOULD grow: "
                f"{[round(br.x - tl.x, 2), round(br.y - tl.y, 2)]} -> "
                f"{[round(new_br.x - new_tl.x, 2), round(new_br.y - new_tl.y, 2)]} mm "
                f"(delta x=[{round(new_tl.x - tl.x, 3)},{round(new_br.x - br.x, 3)}] "
                f"y=[{round(new_tl.y - tl.y, 3)},{round(new_br.y - br.y, 3)}]); "
                "bbox-level containment clamp did not cover this geometry",
                file=_sys.stderr,
            )
        return {
            "repaired": False,
            "would_repair": changed,
            "old_size_mm": [round(br.x - tl.x, 2), round(br.y - tl.y, 2)],
            "new_size_mm": [round(new_br.x - new_tl.x, 2), round(new_br.y - new_tl.y, 2)],
            "would_change_mm": [
                round(new_br.x - new_tl.x - (br.x - tl.x), 3),
                round(new_br.y - new_tl.y - (br.y - tl.y), 3),
            ],
        }
    if changed:
        composition.board_state.board_outline = (new_tl, new_br)
    return {
        "repaired": changed,
        "old_size_mm": [round(br.x - tl.x, 2), round(br.y - tl.y, 2)],
        "new_size_mm": [round(new_br.x - new_tl.x, 2), round(new_br.y - new_tl.y, 2)],
    }


def _as_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# A circumscribed shape may legitimately be a good deal larger than the
# rectangular content it wraps: a circle/hexagon is ~2x, and an inherently
# low-fill shape (snowman, triangle) reaches ~12x its content area even on a
# reasonable 2:1 content. But a low-circularity shape around an ELONGATED content
# explodes the board (star-ornament shipped fab-ready at 592x563 mm, ~63x the
# content area). Cap the fitted-area / content-area ratio ABOVE the legit worst
# case (snowman ~12x) so real shapes still fit, but reject egregious explosions
# (WS8). Paired with the size_mm check, which needs no such headroom.
_MAX_SHAPE_AREA_RATIO = 15.0
# Slack allowed when honoring a brief-requested size_mm before calling the fit
# oversized.
_SHAPE_SIZE_TOL = 0.05


def _ring_area(points: list[tuple[float, float]]) -> float:
    """Shoelace area of a closed ring given as (x, y) pairs (no repeated last)."""
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _requested_size_pair(size_mm: Any) -> tuple[float, float] | None:
    """Normalize a requested ``size_mm`` (scalar diameter/side, ``[w, h]``, or
    ``{"w":..,"h":..}``) to a ``(w, h)`` target, or None if unspecified/unparsable."""
    if size_mm is None:
        return None
    if isinstance(size_mm, (int, float)):
        v = float(size_mm)
        return (v, v) if v > 0 else None
    if isinstance(size_mm, dict):
        w = _as_float(size_mm.get("w") or size_mm.get("width_mm") or size_mm.get("x"))
        h = _as_float(size_mm.get("h") or size_mm.get("height_mm") or size_mm.get("y"))
        if w and h:
            return (w, h)
        return None
    if isinstance(size_mm, (list, tuple)) and len(size_mm) >= 2:
        w, h = _as_float(size_mm[0]), _as_float(size_mm[1])
        if w and h:
            return (w, h)
    return None


def _shape_fit_guard(
    shape: str,
    req: dict[str, Any],
    content_area: float,
    fitted_w: float,
    fitted_h: float,
    fitted_area: float,
) -> dict[str, Any] | None:
    """Return a loud rejection dict if the circumscribed shape explodes the
    board, else None. Consumes the brief's ``size_mm`` (previously ignored) and
    caps material overshoot so a pathological shape can no longer ship a
    massively oversized 'fab-ready' board (WS8)."""
    reasons: list[str] = []
    target = _requested_size_pair(req.get("size_mm"))
    if target is not None:
        tw, th = target
        slack = 1.0 + _SHAPE_SIZE_TOL
        if fitted_w > tw * slack or fitted_h > th * slack:
            reasons.append(
                f"circumscribed {fitted_w:.1f}x{fitted_h:.1f} mm exceeds requested "
                f"size_mm {tw:.1f}x{th:.1f} mm"
            )
    ratio = fitted_area / max(content_area, 1e-6)
    if ratio > _MAX_SHAPE_AREA_RATIO:
        reasons.append(
            f"fitted area {fitted_area:.0f} mm^2 is {ratio:.1f}x the "
            f"{content_area:.0f} mm^2 content (cap {_MAX_SHAPE_AREA_RATIO:.0f}x)"
        )
    if reasons:
        return {
            "fitted": False,
            "reason": "shape fit rejected: " + "; ".join(reasons),
            "rejected_shape": shape,
            "kept_outline": "rect",
            "fitted_size_mm": [round(fitted_w, 2), round(fitted_h, 2)],
        }
    return None


def _circumscribe_dims(
    shape: str, req: dict[str, Any], w: float, h: float
) -> tuple[float, float] | None:
    """Fitted ``(width, height)`` of the requested shape circumscribed around
    a ``w x h`` content rect, mirroring ``_fit_requested_shape``'s two
    branches exactly. None for a shape neither branch supports."""
    tl, br = Point(0.0, 0.0), Point(max(w, 1e-3), max(h, 1e-3))
    from kicraft.layout_editor.outline import SHAPES, circumscribe

    if shape in SHAPES:
        spec = circumscribe(
            shape,
            tl,
            br,
            corner_radius_mm=_as_float(req.get("corner_radius_mm")),
            chamfer_mm=_as_float(req.get("chamfer_mm")),
        )
        return spec.width_mm, spec.height_mm

    from kicraft.shapes import KNOWN_SHAPES
    from kicraft.shapes import circumscribe as circumscribe_polygon

    if shape in KNOWN_SHAPES:
        poly = circumscribe_polygon(shape, tl, br)
        (minx, miny), (maxx, maxy) = poly.aabb()
        return maxx - minx, maxy - miny
    return None


def inscribed_rect_bound(
    req: dict[str, Any] | None, aspect: float
) -> tuple[float, float] | None:
    """Largest ``(w, h)`` content rectangle at ``aspect`` (= w/h) whose
    circumscribed requested shape lands AT the ``size_mm`` target per axis.

    This is the placement-side half of the shape contract: seed the parent
    solver with (at most) this rectangle and the post-placement circumscribe
    in ``_fit_requested_shape`` clears the size half of ``_shape_fit_guard``
    by construction — the same circumscribe decides both. The bound aims at
    the exact target (NOT ``target * (1 + _SHAPE_SIZE_TOL)``): the guard's
    slack must stay available to absorb packing overshoot past the seed —
    aiming at the ceiling spends it up front (KC-HN59RJ's replay candidates
    missed the cap by 0.5-1.9 mm for exactly that reason). Only the SIZE half
    of the guard is mirrored here: its area-ratio cap is non-monotone in
    content size (a sliver inside a low-fill star trips it), which would
    break the search and matters only when no size target exists. None when
    no non-rect shape with a ``size_mm`` target is requested, or the shape
    has no generator.
    """
    if not req:
        return None
    shape = str(req.get("shape", "rect")).strip().lower()
    if shape in ("", "rect"):
        return None
    target = _requested_size_pair(req.get("size_mm"))
    if target is None:
        return None
    tw, th = target
    slack = 1.0
    a = max(0.1, float(aspect))

    def _fits(s: float) -> bool:
        dims = _circumscribe_dims(shape, req, a * s, s)
        if dims is None:
            return False
        fw, fh = dims
        return fw <= tw * slack and fh <= th * slack

    hi = max(tw, th) * 1.5
    lo = 0.0
    if not _fits(min(tw, th) * 0.05):
        # Even a sliver of content overshoots (unsupported shape name or a
        # degenerate size request) — no usable bound.
        return None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _fits(mid):
            lo = mid
        else:
            hi = mid
    return (a * lo, lo)


def _fit_requested_shape(state: ParentCompositionState) -> dict[str, Any]:
    """Circumscribe the brief-requested outline shape around the (already
    grown) rectangular content AABB, then hand it to the stamp/validate/pour
    path as an authoritative outline.

    Runs after :func:`_repair_parent_outline` on the auto (non-manual) path:
    placement happened in the rectangular AABB, so growing the requested shape
    around it keeps every part inside ``Edge.Cuts`` with no placement changes.
    Two output channels:

    * **Parametric** shapes (circle / rounded_rect / chamfered_rect) -> an
      exact ``OutlineSpec`` on ``state.manual_outline`` (no Shapely).
    * **Named / compound** shapes (hexagon, star, snowman, ...) -> a general
      polygon ring on ``state.fitted_polygon`` (Shapely), kept off the
      JS-mirrored ``OutlineSpec``.

    Either way the stamper writes the shape to ``Edge.Cuts``, the geometry
    validator checks true-shape containment, and the KiCad zone filler clips the
    GND pour to it. No-op when a manual layout is authoritative, no shape was
    requested, the request is rectangular, or the name has no generator.
    """
    if state.manual_outline is not None:
        return {"fitted": False, "reason": "manual outline authoritative"}
    req = state.requested_shape
    if not req:
        return {"fitted": False, "reason": "no requested shape"}

    shape = str(req.get("shape", "rect")).strip().lower()
    if shape in ("", "rect"):
        return {"fitted": False, "reason": "rectangular"}

    composition = state.composition
    outline = composition.board_state.board_outline if composition is not None else None
    if not outline or len(outline) < 2:
        return {"fitted": False, "reason": "no outline"}
    tl, br = outline
    content_area = max((br.x - tl.x) * (br.y - tl.y), 1e-6)

    # Parametric convex shapes: exact OutlineSpec path (circle wins here over the
    # polygon path -- simpler, JS-mirror-compatible).
    from kicraft.layout_editor.outline import SHAPES, OutlineSpec, circumscribe

    if shape in SHAPES:
        spec = circumscribe(
            shape,
            tl,
            br,
            corner_radius_mm=_as_float(req.get("corner_radius_mm")),
            chamfer_mm=_as_float(req.get("chamfer_mm")),
        )
        # Guard BEFORE committing the outline: an oversized fit is rejected so the
        # sane rectangular AABB (already on board_state) ships instead (WS8).
        fitted_area = _ring_area([(p.x, p.y) for p in spec.polyline()])
        guard = _shape_fit_guard(
            shape, req, content_area, spec.width_mm, spec.height_mm, fitted_area
        )
        if guard is not None:
            return guard
        # A brief-requested size_mm is a target, not just a cap: once the
        # content fits, deliver the shape AT that size ("round 60 mm" must not
        # ship a ⌀45 board). Grow only — a fitted axis already inside the
        # guard's 5% slack above target is kept, never shrunk below content.
        target = _requested_size_pair(req.get("size_mm"))
        if target is not None:
            tw, th = target
            new_w = max(spec.width_mm, tw)
            new_h = max(spec.height_mm, th)
            if shape == "circle":
                new_w = new_h = max(new_w, new_h)
            if new_w > spec.width_mm + 1e-9 or new_h > spec.height_mm + 1e-9:
                ccx = (spec.min_pt.x + spec.max_pt.x) / 2.0
                ccy = (spec.min_pt.y + spec.max_pt.y) / 2.0
                spec = OutlineSpec(
                    shape=shape,
                    min_pt=Point(ccx - new_w / 2.0, ccy - new_h / 2.0),
                    max_pt=Point(ccx + new_w / 2.0, ccy + new_h / 2.0),
                    corner_radius_mm=spec.corner_radius_mm,
                    chamfer_mm=spec.chamfer_mm,
                )
        composition.board_state.board_outline = spec.aabb()
        state.manual_outline = spec.to_dict()
        return {
            "fitted": True,
            "shape": shape,
            "kind": "parametric",
            "size_mm": [round(spec.width_mm, 2), round(spec.height_mm, 2)],
        }

    # Named / compound shapes: general polygon via Shapely on fitted_polygon.
    from kicraft.shapes import KNOWN_SHAPES
    from kicraft.shapes import circumscribe as circumscribe_polygon

    if shape in KNOWN_SHAPES:
        poly = circumscribe_polygon(shape, tl, br)
        (minx, miny), (maxx, maxy) = poly.aabb()
        fitted_area = _ring_area(poly.points())
        guard = _shape_fit_guard(
            shape, req, content_area, maxx - minx, maxy - miny, fitted_area
        )
        if guard is not None:
            return guard
        points = [(float(x), float(y)) for x, y in poly.points()]
        # size_mm is a target, not just a cap (same rule as the parametric
        # branch): grow the polygon uniformly about its center up to the
        # requested size. Uniform only — a named shape keeps its proportions.
        target = _requested_size_pair(req.get("size_mm"))
        if target is not None:
            f = min(target[0] / max(maxx - minx, 1e-6), target[1] / max(maxy - miny, 1e-6))
            if f > 1.0 + 1e-9:
                pcx = (minx + maxx) / 2.0
                pcy = (miny + maxy) / 2.0
                points = [
                    (pcx + (x - pcx) * f, pcy + (y - pcy) * f) for x, y in points
                ]
                minx = pcx - (pcx - minx) * f
                maxx = pcx + (maxx - pcx) * f
                miny = pcy - (pcy - miny) * f
                maxy = pcy + (maxy - pcy) * f
        composition.board_state.board_outline = (Point(minx, miny), Point(maxx, maxy))
        state.fitted_polygon = [[x, y] for x, y in points]
        return {
            "fitted": True,
            "shape": shape,
            "kind": "polygon",
            "vertices": len(state.fitted_polygon),
            "size_mm": [round(maxx - minx, 2), round(maxy - miny, 2)],
        }

    return {"fitted": False, "reason": f"shape {shape!r} not supported"}


def _validate_parent_geometry(
    state: ParentCompositionState,
) -> dict[str, Any]:
    """Validate that composed parent geometry fits inside the derived outline.

    Two independent checks per component:

    * **Body** (``comp.bbox()`` = courtyard) must be inside the board outline,
      EXCEPT for edge-constrained refs (USB-C, edge connectors) whose
      housing legitimately extends past the PCB edge to mate with an
      external host.
    * **Pad copper** (``pad.bbox()`` = full pad extent) must be inside the
      board outline. Always. Overhanging pad copper is unfabricable; no
      exemption.

    The previous implementation checked pad **centers** rather than pad
    bboxes, undercounting copper overhang by half the pad width. It also
    accepted a per-ref ``parent_overhang_mm`` exemption that, paired with
    the center-only check, was a band-aid over the same bug.
    """
    composition = state.composition
    if composition is None:
        raise RuntimeError("ParentCompositionState has no composition object")

    outline = composition.board_state.board_outline
    if not outline or len(outline) < 2:
        raise RuntimeError("Parent composition has no valid board outline")

    tl, br = outline
    if br.x <= tl.x or br.y <= tl.y:
        raise RuntimeError(
            "Parent composition produced a degenerate board outline "
            f"({tl.x:.3f}, {tl.y:.3f}) -> ({br.x:.3f}, {br.y:.3f})"
        )

    margin = 0.05
    geometry_union_min_x = float("inf")
    geometry_union_min_y = float("inf")
    geometry_union_max_x = float("-inf")
    geometry_union_max_y = float("-inf")

    min_x = tl.x - margin
    min_y = tl.y - margin
    max_x = br.x + margin
    max_y = br.y + margin

    # Manual non-rect outlines additionally constrain geometry to the
    # true shape (analytic containment), not just the AABB: a leaf
    # tucked into the corner of a circular board is inside the AABB
    # but off the physical board.
    # ``shape_spec`` is any object exposing contains_rect / contains_point with
    # a tolerance -- OutlineSpec for manual + parametric-auto outlines, or a
    # shapely-backed PolygonOutline for named/compound auto outlines. Both share
    # the same surface, so the checks below are identical for either.
    shape_spec = None
    if (
        state.manual_outline is not None
        and state.manual_outline.get("shape", "rect") != "rect"
    ):
        from kicraft.layout_editor.outline import OutlineSpec

        shape_spec = OutlineSpec.from_dict(state.manual_outline)
    elif state.fitted_polygon:
        from kicraft.shapes import polygon_outline_from_points

        shape_spec = polygon_outline_from_points(state.fitted_polygon)

    def _bbox_outside(bx0: float, by0: float, bx1: float, by1: float) -> bool:
        if bx0 < min_x or by0 < min_y or bx1 > max_x or by1 > max_y:
            return True
        return shape_spec is not None and not shape_spec.contains_rect(
            bx0, by0, bx1, by1, tol=margin
        )

    def _point_outside(px: float, py: float) -> bool:
        if px < min_x or px > max_x or py < min_y or py > max_y:
            return True
        return shape_spec is not None and not shape_spec.contains_point(
            px, py, tol=margin
        )

    outside_components: list[dict[str, Any]] = []
    outside_pads = 0
    outside_traces = 0
    outside_vias = 0
    edge_constrained = set(state.edge_constrained_refs or ())

    for ref, comp in (composition.board_state.components or {}).items():
        # geometry_union tracks the full physical extent of every component
        # so the diagnostic shows where the actual copper/courtyard lives
        # relative to the board outline, not just the courtyard centerline.
        phys_tl, phys_br = comp.physical_bbox()
        geometry_union_min_x = min(geometry_union_min_x, phys_tl.x)
        geometry_union_min_y = min(geometry_union_min_y, phys_tl.y)
        geometry_union_max_x = max(geometry_union_max_x, phys_br.x)
        geometry_union_max_y = max(geometry_union_max_y, phys_br.y)

        # Body (courtyard) check: a non-edge-constrained component whose
        # courtyard extends past the board outline is misplaced. Edge-pinned
        # connectors are exempted because the housing legitimately mounts
        # past the PCB edge.
        body_tl, body_br = comp.bbox()
        if ref in edge_constrained:
            component_outside = False
        else:
            component_outside = _bbox_outside(
                body_tl.x, body_tl.y, body_br.x, body_br.y
            )

        # Pad check: pad COPPER (not just the center) must be inside the
        # board outline. No edge-constrained exemption -- pad copper that
        # crosses Edge.Cuts is unfabricable.
        pad_outside_count = 0
        for pad in comp.pads:
            pad_tl, pad_br = pad.bbox()
            if _bbox_outside(pad_tl.x, pad_tl.y, pad_br.x, pad_br.y):
                pad_outside_count += 1
                outside_pads += 1

        if component_outside or pad_outside_count > 0:
            outside_components.append(
                {
                    "ref": ref,
                    "bbox": {
                        "top_left": {"x": body_tl.x, "y": body_tl.y},
                        "bottom_right": {"x": body_br.x, "y": body_br.y},
                    },
                    "physical_bbox": {
                        "top_left": {"x": phys_tl.x, "y": phys_tl.y},
                        "bottom_right": {"x": phys_br.x, "y": phys_br.y},
                    },
                    "outside_body": component_outside,
                    "outside_pad_count": pad_outside_count,
                }
            )

    for trace in composition.board_state.traces or []:
        geometry_union_min_x = min(geometry_union_min_x, trace.start.x, trace.end.x)
        geometry_union_min_y = min(geometry_union_min_y, trace.start.y, trace.end.y)
        geometry_union_max_x = max(geometry_union_max_x, trace.start.x, trace.end.x)
        geometry_union_max_y = max(geometry_union_max_y, trace.start.y, trace.end.y)
        if _point_outside(trace.start.x, trace.start.y) or _point_outside(
            trace.end.x, trace.end.y
        ):
            outside_traces += 1

    for via in composition.board_state.vias or []:
        geometry_union_min_x = min(geometry_union_min_x, via.pos.x)
        geometry_union_min_y = min(geometry_union_min_y, via.pos.y)
        geometry_union_max_x = max(geometry_union_max_x, via.pos.x)
        geometry_union_max_y = max(geometry_union_max_y, via.pos.y)
        if _point_outside(via.pos.x, via.pos.y):
            outside_vias += 1

    validation = {
        "accepted": not outside_components
        and outside_traces == 0
        and outside_vias == 0,
        "geometry_union": {
            "top_left": {
                "x": 0.0 if geometry_union_min_x == float("inf") else geometry_union_min_x,
                "y": 0.0 if geometry_union_min_y == float("inf") else geometry_union_min_y,
            },
            "bottom_right": {
                "x": 0.0 if geometry_union_max_x == float("-inf") else geometry_union_max_x,
                "y": 0.0 if geometry_union_max_y == float("-inf") else geometry_union_max_y,
            },
        },
        "board_outline": {
            "top_left": {"x": tl.x, "y": tl.y},
            "bottom_right": {"x": br.x, "y": br.y},
            "width_mm": max(0.0, br.x - tl.x),
            "height_mm": max(0.0, br.y - tl.y),
        },
        # True outline shape, from whichever channel carried it: manual/parametric
        # spec, or the named/compound polygon path (fitted_polygon). Reading only
        # manual_outline logged a hexagon/star board as "rect" (WS9).
        "outline_shape": (
            str(state.manual_outline.get("shape", "rect"))
            if state.manual_outline is not None
            else (
                str(state.requested_shape.get("shape", "polygon"))
                if state.fitted_polygon and state.requested_shape
                else ("polygon" if state.fitted_polygon else "rect")
            )
        ),
        "outside_component_count": len(outside_components),
        "outside_components": outside_components[:50],
        "outside_pad_count": outside_pads,
        "outside_trace_count": outside_traces,
        "outside_via_count": outside_vias,
    }
    state.geometry_validation = validation
    return validation


def _render_parent_board_views(
    pcb_path: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Render standard parent board preview images via the unified
    ``kicraft.render.render_views`` pipeline. Same code path the GUI
    monitor and the score-time visual check use, so parent previews
    cannot drift from leaf previews."""
    from kicraft.render import render_views

    results = render_views(
        pcb_path,
        output_dir,
        views=["front_all", "back_all", "copper_both"],
    )
    return {name: str(path) for name, path in results.items()}
