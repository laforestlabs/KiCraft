"""Leaf subcircuit geometry and topology helpers.

Pure-algorithm utilities for computing bounding boxes, building reduced
extractions, generating size-reduction candidates, scoring local component
placement, and repairing placement legality.  These were originally private
helpers inside ``kicraft.cli.solve_subcircuits`` and are extracted here so
they can be reused by other brain-layer modules without depending on the
CLI entry-point.
"""

from __future__ import annotations

import copy
from typing import Any

from kicraft.autoplacer.brain.placement import PlacementScorer, PlacementSolver
from kicraft.autoplacer.brain.subcircuit_extractor import ExtractedSubcircuitBoard
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    PlacementScore,
    Point,
)


# ------------------------------------------------------------------
# Translation helpers (pure geometry, no pcbnew)
# ------------------------------------------------------------------

def copy_components_with_translation(
    components: dict[str, Component],
    delta: Point,
) -> dict[str, Component]:
    """Return a deep-copied dict of *components* shifted by *delta*."""
    translated: dict[str, Component] = {}
    for ref, comp in components.items():
        new_comp = copy.deepcopy(comp)
        new_comp.pos = Point(new_comp.pos.x + delta.x, new_comp.pos.y + delta.y)
        if new_comp.body_center is not None:
            new_comp.body_center = Point(
                new_comp.body_center.x + delta.x,
                new_comp.body_center.y + delta.y,
            )
        for pad in new_comp.pads:
            pad.pos = Point(pad.pos.x + delta.x, pad.pos.y + delta.y)
        translated[ref] = new_comp
    return translated


def copy_traces_with_translation(traces: list[Any], delta: Point) -> list[Any]:
    """Return deep-copied *traces* shifted by *delta*."""
    translated: list[Any] = []
    for trace in traces:
        new_trace = copy.deepcopy(trace)
        new_trace.start = Point(
            new_trace.start.x + delta.x,
            new_trace.start.y + delta.y,
        )
        new_trace.end = Point(
            new_trace.end.x + delta.x,
            new_trace.end.y + delta.y,
        )
        translated.append(new_trace)
    return translated


def copy_vias_with_translation(vias: list[Any], delta: Point) -> list[Any]:
    """Return deep-copied *vias* shifted by *delta*."""
    translated: list[Any] = []
    for via in vias:
        new_via = copy.deepcopy(via)
        new_via.pos = Point(new_via.pos.x + delta.x, new_via.pos.y + delta.y)
        translated.append(new_via)
    return translated


# ------------------------------------------------------------------
# Bounding-box computation
# ------------------------------------------------------------------

def tight_leaf_geometry_bounds(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    routing: dict[str, Any],
    *,
    connector_pad_margin_mm: float = 0.0,
) -> dict[str, float]:
    """Compute a tight axis-aligned bounding box around all leaf geometry.

    Considers component bounding boxes, pad positions, trace segments
    (including half-widths), and vias (including radii).  Falls back to
    the extraction board outline when there is no geometry at all.

    When *connector_pad_margin_mm* > 0, an extra margin is added around
    each pad of connector-type components.  This compensates for the fact
    that the ``Pad`` type stores only the pad center position, not the
    physical copper extent.  Edge-pinned connectors (USB, barrel jacks,
    etc.) often have pads whose copper extends well beyond the center
    point, and without this margin the board outline can end up too close
    to the physical copper, causing ``copper_edge_clearance`` DRC
    violations after size reduction.
    """
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for comp in solved_components.values():
        tl, br = comp.bbox()
        min_x = min(min_x, tl.x)
        min_y = min(min_y, tl.y)
        max_x = max(max_x, br.x)
        max_y = max(max_y, br.y)
        # For connectors, expand pad contributions by connector_pad_margin_mm
        # to account for physical pad copper extent beyond the center point.
        pad_margin = (
            connector_pad_margin_mm
            if connector_pad_margin_mm > 0.0
            and comp.kind == "connector"
            else 0.0
        )
        for pad in comp.pads:
            min_x = min(min_x, pad.pos.x - pad_margin)
            min_y = min(min_y, pad.pos.y - pad_margin)
            max_x = max(max_x, pad.pos.x + pad_margin)
            max_y = max(max_y, pad.pos.y + pad_margin)

    for trace in routing.get("_trace_segments", []):
        half_width = max(0.0, float(getattr(trace, "width_mm", 0.0)) / 2.0)
        min_x = min(min_x, trace.start.x - half_width, trace.end.x - half_width)
        min_y = min(min_y, trace.start.y - half_width, trace.end.y - half_width)
        max_x = max(max_x, trace.start.x + half_width, trace.end.x + half_width)
        max_y = max(max_y, trace.start.y + half_width, trace.end.y + half_width)

    for via in routing.get("_via_objects", []):
        radius = max(0.0, float(getattr(via, "size_mm", 0.0)) / 2.0)
        min_x = min(min_x, via.pos.x - radius)
        min_y = min(min_y, via.pos.y - radius)
        max_x = max(max_x, via.pos.x + radius)
        max_y = max(max_y, via.pos.y + radius)

    if min_x == float("inf"):
        tl, br = extraction.local_state.board_outline
        min_x = tl.x
        min_y = tl.y
        max_x = br.x
        max_y = br.y

    return {
        "min_x": float(min_x),
        "min_y": float(min_y),
        "max_x": float(max_x),
        "max_y": float(max_y),
        "width_mm": float(max_x - min_x),
        "height_mm": float(max_y - min_y),
    }


# ------------------------------------------------------------------
# Reduced extraction builder
# ------------------------------------------------------------------

def build_reduced_leaf_extraction(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    routing: dict[str, Any],
    outline: tuple[Point, Point],
) -> ExtractedSubcircuitBoard:
    """Build a new extraction whose outline is shrunk to *outline*.

    All solved components and routing geometry are translated so that
    the top-left corner of *outline* becomes the origin.
    """
    tl, br = outline
    delta = Point(-tl.x, -tl.y)
    local_state = copy.deepcopy(extraction.local_state)
    local_state.components = copy_components_with_translation(solved_components, delta)
    local_state.traces = copy_traces_with_translation(
        routing.get("_trace_segments", []),
        delta,
    )
    local_state.vias = copy_vias_with_translation(
        routing.get("_via_objects", []),
        delta,
    )
    local_state.board_outline = (
        Point(0.0, 0.0),
        Point(max(1.0, br.x - tl.x), max(1.0, br.y - tl.y)),
    )

    reduced = copy.deepcopy(extraction)
    reduced.local_state = local_state
    reduced.internal_traces = copy.deepcopy(local_state.traces)
    reduced.internal_vias = copy.deepcopy(local_state.vias)
    reduced.translation = Point(
        extraction.translation.x + delta.x,
        extraction.translation.y + delta.y,
    )
    if reduced.envelope is not None:
        reduced.envelope.top_left = Point(0.0, 0.0)
        reduced.envelope.bottom_right = Point(
            local_state.board_width, local_state.board_height
        )
        reduced.envelope.width_mm = local_state.board_width
        reduced.envelope.height_mm = local_state.board_height
    reduced.notes = list(reduced.notes) + [
        f"reduced_outline_width_mm={local_state.board_width:.3f}",
        f"reduced_outline_height_mm={local_state.board_height:.3f}",
    ]
    return reduced


# ------------------------------------------------------------------
# Size-reduction candidate generator
# ------------------------------------------------------------------

def leaf_size_reduction_candidates(
    current_width: float,
    current_height: float,
    min_width: float,
    min_height: float,
) -> list[dict[str, Any]]:
    """Generate candidate smaller outline sizes for a leaf subcircuit.

    Produces a list of dicts, each with keys ``axis``, ``step_mm``,
    ``width_mm``, and ``height_mm``.  Candidates are ordered coarse-first
    (single-axis reductions, then both-axis) followed by fine steps.
    """
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    coarse_steps = (2.0, 1.0, 0.5)
    fine_steps = (0.25,)

    def _add(width: float, height: float, axis: str, step_mm: float) -> None:
        width = round(max(min_width, width), 4)
        height = round(max(min_height, height), 4)
        key = (width, height)
        if key in seen:
            return
        if width >= current_width and height >= current_height:
            return
        seen.add(key)
        candidates.append(
            {
                "axis": axis,
                "step_mm": float(step_mm),
                "width_mm": width,
                "height_mm": height,
            }
        )

    for step in coarse_steps:
        _add(current_width - step, current_height, "width", step)
        _add(current_width, current_height - step, "height", step)
    for step in coarse_steps:
        _add(current_width - step, current_height - step, "both", step)
    for step in fine_steps:
        _add(current_width - step, current_height, "width", step)
        _add(current_width, current_height - step, "height", step)
        _add(current_width - step, current_height - step, "both", step)

    return candidates


# ------------------------------------------------------------------
# Local component scoring
# ------------------------------------------------------------------

def score_local_components(
    local_state: BoardState,
    components: dict[str, Component],
    cfg: dict[str, Any],
) -> PlacementScore:
    """Score a set of *components* against *local_state* using the solver/scorer.

    Returns a :class:`PlacementScore` with legality penalties applied when
    overlaps or pads-outside-board are detected.
    """
    work_state = copy.copy(local_state)
    work_state.components = components
    score = PlacementScorer(work_state, cfg).score()

    legalizer = PlacementSolver(work_state, cfg, seed=0)
    raw_legality = legalizer.legality_diagnostics(components)
    legality = raw_legality if isinstance(raw_legality, dict) else {}
    raw_overlap_count = legality.get("overlap_count", 0)
    raw_pad_outside_count = legality.get("pad_outside_count", 0)
    overlap_count = (
        int(raw_overlap_count)
        if isinstance(raw_overlap_count, (int, float, str))
        else 0
    )
    pad_outside_count = (
        int(raw_pad_outside_count)
        if isinstance(raw_pad_outside_count, (int, float, str))
        else 0
    )

    if overlap_count or pad_outside_count:
        score.courtyard_overlap = max(
            0.0,
            min(score.courtyard_overlap, 100.0 - 25.0 * overlap_count),
        )
        score.board_containment = max(
            0.0,
            min(score.board_containment, 100.0 - 40.0 * pad_outside_count),
        )
        score.compute_total()

    return score


# ------------------------------------------------------------------
# Placement legality repair
# ------------------------------------------------------------------

def repair_leaf_placement_legality(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
) -> tuple[dict[str, Component], dict[str, Any]]:
    """Attempt to repair overlap / out-of-board issues via the legalizer.

    Returns ``(repaired_components, diagnostics_dict)``.  The diagnostics
    dict always contains the key ``"attempted": True`` along with pass
    counts, moved refs, remaining overlaps, and resolution status.
    """
    repaired = copy.deepcopy(solved_components)
    local_state = copy.deepcopy(extraction.local_state)
    local_state.components = repaired

    legalizer = PlacementSolver(local_state, cfg, seed=0)
    legalization = legalizer.legalize_components(
        repaired,
        max_passes=int(cfg.get("leaf_legality_repair_passes", 12)),
    )
    raw_diagnostics = legalization.get("diagnostics", {})
    diagnostics = raw_diagnostics if isinstance(raw_diagnostics, dict) else {}

    raw_moved_refs = legalization.get("moved_refs", [])
    moved_refs = raw_moved_refs if isinstance(raw_moved_refs, list) else []

    raw_overlaps = diagnostics.get("overlaps", [])
    overlaps = raw_overlaps if isinstance(raw_overlaps, list) else []

    raw_pads_outside = diagnostics.get("pads_outside_board", [])
    pads_outside = raw_pads_outside if isinstance(raw_pads_outside, list) else []

    raw_passes = legalization.get("passes", 0)
    passes = int(raw_passes) if isinstance(raw_passes, (int, float, str)) else 0

    return repaired, {
        "attempted": True,
        "passes": passes,
        "moved_components": list(moved_refs),
        "remaining_overlaps": list(overlaps),
        "pads_outside_board": list(pads_outside),
        "resolved": bool(legalization.get("resolved", False)),
        "diagnostics": diagnostics,
    }
