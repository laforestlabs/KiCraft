"""Parent-composition state dataclasses.

Split out of ``compose_subcircuits.py`` (Lever 2.5). ``ParentCompositionState``
is the central machine-readable snapshot threaded through compose -> stamp ->
search -> route -> persist; extracting it lets those layers move to their own
modules without importing the whole monolith. Re-exported from
``compose_subcircuits`` so existing references keep resolving.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kicraft.autoplacer.brain.copper_accounting import CopperManifest
from kicraft.autoplacer.brain.subcircuit_composer import ParentComposition
from kicraft.autoplacer.brain.types import Point


@dataclass(slots=True)
class CompositionEntry:
    """One rigid child instance inside a parent composition."""

    artifact_dir: str
    sheet_name: str
    instance_path: str
    origin: Point
    rotation: float
    transformed_bbox: tuple[float, float]
    component_count: int
    trace_count: int
    via_count: int
    anchor_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_dir": self.artifact_dir,
            "sheet_name": self.sheet_name,
            "instance_path": self.instance_path,
            "origin": {
                "x": self.origin.x,
                "y": self.origin.y,
            },
            "rotation": self.rotation,
            "transformed_bbox": {
                "width_mm": self.transformed_bbox[0],
                "height_mm": self.transformed_bbox[1],
            },
            "component_count": self.component_count,
            "trace_count": self.trace_count,
            "via_count": self.via_count,
            "anchor_count": self.anchor_count,
        }


@dataclass(slots=True)
class ParentCompositionState:
    """Machine-readable parent composition snapshot."""

    project_dir: str
    spacing_mm: float
    entries: list[CompositionEntry] = field(default_factory=list)
    bounding_box: tuple[Point, Point] = field(
        default_factory=lambda: (Point(0.0, 0.0), Point(0.0, 0.0))
    )
    parent_sheet_name: str = "COMPOSED_PARENT"
    parent_instance_path: str = "/COMPOSED_PARENT"
    component_count: int = 0
    trace_count: int = 0
    via_count: int = 0
    interconnect_net_count: int = 0
    inferred_interconnect_net_count: int = 0
    preserved_child_trace_count: int = 0
    preserved_child_via_count: int = 0
    expected_preserved_child_trace_count: int = 0
    expected_preserved_child_via_count: int = 0
    routed_total_trace_count: int = 0
    routed_total_via_count: int = 0
    added_parent_trace_count: int = 0
    added_parent_via_count: int = 0
    packing_metadata: dict[str, Any] = field(default_factory=dict)
    geometry_validation: dict[str, Any] = field(default_factory=dict)
    # Post-route acceptance summary: DRC categories, rejection reasons, etc.
    # Populated after _route_parent_board returns. Persisted unconditionally
    # so callers can diagnose why a routed board was rejected even when the
    # run exits non-zero.
    routed_validation: dict[str, Any] = field(default_factory=dict)
    # Pre-route DRC summary: DRC counts on parent_pre_freerouting.kicad_pcb
    # (i.e., the stamped board BEFORE FreeRouting runs). Distinguishes
    # composer-introduced shorts (shorts>0 here) from router-introduced
    # shorts (shorts==0 here, but >0 after route). Without this split, every
    # route failure looks like a FreeRouting clearance bug even when the
    # composer stamped two leaves' tracks on top of each other.
    stamp_drc: dict[str, Any] = field(default_factory=dict)
    score_total: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    score_notes: list[str] = field(default_factory=list)
    composition_notes: list[str] = field(default_factory=list)
    composition: ParentComposition | None = None
    copper_manifest: CopperManifest | None = None
    # Keep-in rects around parent-local locked components (e.g. mounting
    # holes) that must be stamped onto the parent board as rule-area
    # keep-outs so FreeRouting cannot route tracks or place vias through
    # them. Units: mm, absolute parent-local coords.
    parent_local_keep_in_rects: list[tuple[Point, Point]] = field(default_factory=list)
    # Refs whose components are pinned to a board edge/corner. After the
    # "PCB Edge" marker is honoured as the anchor (D1), these refs' bodies
    # are expected to extend beyond the board outline (e.g. USB-C shell);
    # the geometry validator must only flag them when pads fall outside.
    edge_constrained_refs: frozenset[str] = field(default_factory=frozenset)
    # Outline sides (left/right/top/bottom) whose position is defined by an
    # edge-ZONED part (a connector mouth, OR a switch/header zoned to that
    # edge). _repair_parent_outline keeps these flush with the part instead of
    # adding breathing-room margin, so the zoned part is not buried inboard by
    # the margin or by a neighbor sitting just behind it.
    edge_zoned_outline_sides: frozenset[str] = field(default_factory=frozenset)
    # Serialized OutlineSpec dict when this composition came from a
    # manual layout (kicraft.layout_editor.outline). Non-None marks the
    # outline as USER-AUTHORITATIVE: the outline-repair grow is skipped
    # (violations fail loudly via geometry validation instead), the
    # geometry validator additionally checks the true shape (not just
    # the AABB), and the stamper writes the shape's polyline to
    # Edge.Cuts for non-rect shapes.
    manual_outline: dict[str, Any] | None = None
    # True when the outline is authoritative for a reason OTHER than a manual
    # layout (today: a standard form-factor scaffold, whose rect IS the spec).
    # Same contract as manual_outline: the outline-repair grow is skipped and
    # violations fail loudly via geometry validation instead of an up-size.
    outline_authoritative: bool = False
    # Non-rectangular outline shape requested from the brief (captured at the
    # intent stage as ``intent.form_factor``, emitted into autoplacer.json as the
    # ``board_outline`` block). Distinct from ``manual_outline``: this carries
    # only the shape INTENT (shape tag + params, no min/max), to be sized to the
    # placed content by the compose pipeline (Phase 3 circumscribe/inscribe).
    # None on a rectangular board or when a manual layout supplies the outline.
    requested_shape: dict[str, Any] | None = None
    # Resolved arbitrary (named / compound) outline as a closed ring of [x, y]
    # points in the parent-local frame, set by ``_fit_requested_shape`` for
    # shapes that need a general polygon (hexagon, star, snowman, ...). Kept
    # OFF the JS-mirrored OutlineSpec/manual_outline: the stamper writes these
    # points straight to Edge.Cuts and the geometry validator checks containment
    # via shapely. None for rectangular / parametric / manual boards.
    fitted_polygon: list[list[float]] | None = None
    # Outcome dict of _fit_requested_shape for THIS composition (fitted /
    # rejected + reason + size), set at stamp time. The candidate search reads
    # it to prefer placements whose requested shape actually committed; the
    # round JSON persists it so a rejected fit is diagnosable per candidate.
    shape_fit: dict[str, Any] | None = None
    # Stock mounting-hole footprints to load onto the stamped board for
    # user holes without a backing H-ref (manual mode). Entries:
    # {ref, x, y, lib_dir, fp_name, screw}; coordinates in the same
    # frame as components (A4-shifted alongside them at stamp time).
    synthesized_footprints: list[dict[str, Any]] = field(default_factory=list)
    # Wall-clock per phase of a parent compose+route round. Keys (when
    # populated): place_solve_ms, stamp_ms, stamp_drc_ms, freerouting_ms,
    # candidate_search_ms, plus solve_*_ms sub-phases from the solver.
    # Lets the harness see whether routing or layout dominates a round so
    # the layout-search budget can be tuned without re-instrumenting.
    phase_timings: dict[str, float] = field(default_factory=dict)
    # Per-round candidate search summary written by _search_best_layout.
    # Keys: k, tried, accepted, rejected_drc, best_index, best_seed,
    # total_search_ms, candidates (list of per-trial dicts).
    candidate_search: dict[str, Any] = field(default_factory=dict)

    @property
    def width_mm(self) -> float:
        tl, br = self.bounding_box
        return max(0.0, br.x - tl.x)

    @property
    def height_mm(self) -> float:
        tl, br = self.bounding_box
        return max(0.0, br.y - tl.y)

    def to_dict(self) -> dict[str, Any]:
        tl, br = self.bounding_box
        return {
            "project_dir": self.project_dir,
            "spacing_mm": self.spacing_mm,
            "parent_sheet_name": self.parent_sheet_name,
            "parent_instance_path": self.parent_instance_path,
            "entry_count": len(self.entries),
            "component_count": self.component_count,
            "trace_count": self.trace_count,
            "via_count": self.via_count,
            "interconnect_net_count": self.interconnect_net_count,
            "inferred_interconnect_net_count": self.inferred_interconnect_net_count,
            "preserved_child_trace_count": self.preserved_child_trace_count,
            "preserved_child_via_count": self.preserved_child_via_count,
            "expected_preserved_child_trace_count": self.expected_preserved_child_trace_count,
            "expected_preserved_child_via_count": self.expected_preserved_child_via_count,
            "routed_total_trace_count": self.routed_total_trace_count,
            "routed_total_via_count": self.routed_total_via_count,
            "added_parent_trace_count": self.added_parent_trace_count,
            "added_parent_via_count": self.added_parent_via_count,
            "packing_metadata": dict(self.packing_metadata),
            "geometry_validation": dict(self.geometry_validation),
            "routed_validation": dict(self.routed_validation),
            "stamp_drc": dict(self.stamp_drc),
            "phase_timings": dict(self.phase_timings),
            "candidate_search": dict(self.candidate_search),
            "score_total": self.score_total,
            "score_breakdown": dict(self.score_breakdown),
            "score_notes": list(self.score_notes),
            "composition_notes": list(self.composition_notes),
            "bounding_box": {
                "top_left": {"x": tl.x, "y": tl.y},
                "bottom_right": {"x": br.x, "y": br.y},
                "width_mm": self.width_mm,
                "height_mm": self.height_mm,
            },
            "entries": [entry.to_dict() for entry in self.entries],
            "copper_manifest": self.copper_manifest.to_dict() if self.copper_manifest else None,
            # Outline provenance (WS9): so a shaped board's true outline is
            # recorded, not silently reported as "rect" downstream. requested_shape
            # is the brief intent; manual_outline/fitted_polygon is what actually
            # got stamped (parametric spec / general polygon ring).
            "requested_shape": dict(self.requested_shape) if self.requested_shape else None,
            "manual_outline": dict(self.manual_outline) if self.manual_outline else None,
            "fitted_polygon": (
                [[float(x), float(y)] for x, y in self.fitted_polygon]
                if self.fitted_polygon
                else None
            ),
            "shape_fit": dict(self.shape_fit) if self.shape_fit else None,
        }
