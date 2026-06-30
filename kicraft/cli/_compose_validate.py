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

    # Parametric convex shapes: exact OutlineSpec path (circle wins here over the
    # polygon path -- simpler, JS-mirror-compatible).
    from kicraft.layout_editor.outline import SHAPES, circumscribe

    if shape in SHAPES:
        spec = circumscribe(
            shape,
            tl,
            br,
            corner_radius_mm=_as_float(req.get("corner_radius_mm")),
            chamfer_mm=_as_float(req.get("chamfer_mm")),
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
        composition.board_state.board_outline = (Point(minx, miny), Point(maxx, maxy))
        state.fitted_polygon = [[float(x), float(y)] for x, y in poly.points()]
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
        "outline_shape": (
            state.manual_outline.get("shape", "rect")
            if state.manual_outline is not None
            else "rect"
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
