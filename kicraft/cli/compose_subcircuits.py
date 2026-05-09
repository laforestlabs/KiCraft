#!/usr/bin/env python3
"""Compose solved subcircuits into a parent composition state.

This CLI is the first parent-composition entrypoint for the subcircuits
redesign. It loads solved leaf artifacts from `.experiments/subcircuits`,
instantiates them as rigid modules, applies translation/rotation transforms,
and emits a machine-readable composition snapshot.

Current scope:
- load canonical solved subcircuit artifacts
- instantiate rigid child modules
- apply translation + rotation transforms
- build a parent composition state summary
- emit JSON and optional saved composition snapshot
- support simple placement modes for initial composition experiments
- stamp composition onto a real .kicad_pcb file (--stamp)
- route parent interconnects via FreeRouting (--route)
- persist parent-level solved layout artifacts

This command does NOT yet:
- optimize parent placement
- recurse through non-leaf schematic hierarchy automatically

It is intended as a composition-side scaffold so later milestones can build:
- parent-level placement optimization
- interconnect routing
- recursive upward propagation
- final top-level board assembly
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kicraft.autoplacer.brain.hierarchy_parser import parse_hierarchy


from kicraft.autoplacer.brain.copper_accounting import (
    CopperManifest,
    build_copper_manifest,
    verify_copper_preservation,
)
from kicraft.autoplacer.brain.parent_adapter import (
    artifact_to_component,
    attachment_constraints_to_zones,
    infer_interconnect_nets_pre_placement,
    placements_from_solved_state,
    synthetic_block_ref,
)
from kicraft.autoplacer.brain.manual_layout import ManualLayout, load_manual_layout
from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.subcircuit_composer import (
    AttachmentConstraint,
    ChildArtifactPlacement,
    DerivedAttachmentConstraints,
    LeafBlockerSet,
    ParentComposition,
    PlacementModel,
    build_parent_composition,
    dominant_blocker_side,
    edge_anchor_target_coordinate,
    estimate_layer_aware_parent_board_size,
    packed_extents_outline,
    derive_attachment_constraints,
    child_layer_envelopes,
    can_overlap,
    can_overlap_sparse,
    constraint_aware_outline,
    extract_leaf_blocker_set,
)
from kicraft.autoplacer.brain.types import BoardState
from kicraft.autoplacer.brain.subcircuit_extractor import extract_parent_local_components
from kicraft.autoplacer.brain.subcircuit_instances import (
    artifact_debug_dict,
    artifact_summary,
    load_solved_artifacts,
    transform_loaded_artifact,
    transformed_debug_dict,
    transformed_summary,
)
from kicraft.autoplacer.brain.types import Component, Point, SubCircuitDefinition, SubCircuitId


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
        }


def _emit_inspector_bundle(routed_pcb: Path) -> None:
    """Run the parent-PCB inspector and print bundle paths.

    Emits a structured JSON report, a markdown summary, and annotated
    PNGs that downstream callers (especially AI agents) can read to
    understand the layout. Failures are non-fatal -- the inspector is
    a diagnostic, not a gate.
    """
    if not routed_pcb.is_file():
        return
    try:
        from kicraft.cli.inspect_parent import collect, render_annotated_top, \
            render_stacking_heatmap, to_markdown
        out_dir = routed_pcb.parent / "inspect"
        out_dir.mkdir(parents=True, exist_ok=True)
        report = collect(routed_pcb)
        json_path = out_dir / "report.json"
        # Atomic writes: this bundle is auto-emitted after every parent
        # route, and downstream tools (GUI, agents) may read mid-write.
        tmp_json = json_path.with_suffix(json_path.suffix + ".tmp")
        tmp_json.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        tmp_json.replace(json_path)
        pngs: dict[str, Path] = {}
        try:
            pngs["annotated_top"] = render_annotated_top(
                report, out_dir / "annotated_top.png"
            )
            pngs["stacking_heatmap"] = render_stacking_heatmap(
                report, out_dir / "stacking_heatmap.png"
            )
        except Exception as exc:
            print(f"inspect: render failed: {exc}", file=sys.stderr)
        md_path = out_dir / "summary.md"
        tmp_md = md_path.with_suffix(md_path.suffix + ".tmp")
        tmp_md.write_text(to_markdown(report, png_paths=pngs), encoding="utf-8")
        tmp_md.replace(md_path)
        print(f"inspect_summary    : {md_path}")
        print(f"inspect_json       : {json_path}")
        for label, p in pngs.items():
            print(f"inspect_{label:<10s}: {p}")
    except Exception as exc:
        print(f"inspect: failed: {exc}", file=sys.stderr)


def _discover_artifact_dirs(project_dir: Path) -> list[Path]:
    """Find solved subcircuit artifact directories under a project."""
    root = project_dir / ".experiments" / "subcircuits"
    if not root.exists():
        return []

    artifact_dirs: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        metadata = child / "metadata.json"
        debug = child / "debug.json"
        solved_layout = child / "solved_layout.json"
        if metadata.exists() and debug.exists() and solved_layout.exists():
            artifact_dirs.append(child)
    return artifact_dirs


def _resolve_artifact_dirs(
    project_dir: Path | None,
    artifact_args: list[str],
) -> list[str | Path]:
    """Resolve artifact directories from CLI inputs."""
    resolved: list[str | Path] = []

    for artifact in artifact_args:
        path = Path(artifact).resolve()
        if path not in resolved:
            resolved.append(path)

    if project_dir is not None:
        for path in _discover_artifact_dirs(project_dir.resolve()):
            if path not in resolved:
                resolved.append(path)

    return resolved


def _filter_loaded_artifacts(loaded_artifacts, only: list[str]) -> list[Any]:
    """Filter loaded artifacts by sheet name, file name, or instance path."""
    if not only:
        return list(loaded_artifacts)

    only_set = {item.strip().lower() for item in only if item.strip()}
    filtered = []
    for artifact in loaded_artifacts:
        candidates = {
            artifact.layout.subcircuit_id.sheet_name.lower(),
            artifact.layout.subcircuit_id.sheet_file.lower(),
            artifact.layout.subcircuit_id.instance_path.lower(),
        }
        if candidates & only_set:
            filtered.append(artifact)
    return filtered


def _select_parent_definition(
    project_dir: Path | None,
    parent_selector: str | None,
) -> SubCircuitDefinition | None:
    """Resolve a real parent definition from schematic hierarchy."""
    if project_dir is None or not parent_selector:
        return None

    graph = parse_hierarchy(project_dir=project_dir.resolve())
    selector = parent_selector.strip().lower()
    if not selector:
        return None

    root_candidates = {
        graph.root.id.sheet_name.lower(),
        graph.root.id.sheet_file.lower(),
        graph.root.id.instance_path.lower(),
    }
    if selector in root_candidates:
        return graph.root.definition

    for node in graph.non_leaf_nodes():
        if node.id.instance_path == "/":
            continue
        candidates = {
            node.id.sheet_name.lower(),
            node.id.sheet_file.lower(),
            node.id.instance_path.lower(),
        }
        if selector in candidates:
            return node.definition

    raise ValueError(f"Unknown parent subcircuit: {parent_selector}")


def _filter_artifacts_for_parent(
    loaded_artifacts,
    parent_definition: SubCircuitDefinition | None,
) -> list[Any]:
    """Restrict artifacts to direct children of the selected parent."""
    if parent_definition is None:
        return list(loaded_artifacts)

    child_paths = {child_id.instance_path for child_id in parent_definition.child_ids}
    return [
        artifact
        for artifact in loaded_artifacts
        if artifact.layout.subcircuit_id.instance_path in child_paths
    ]


def _bbox_size(bbox: tuple[Point, Point]) -> tuple[float, float]:
    return (bbox[1].x - bbox[0].x, bbox[1].y - bbox[0].y)


def _shift_bbox(bbox: tuple[Point, Point], origin: Point) -> tuple[Point, Point]:
    return (
        Point(bbox[0].x + origin.x, bbox[0].y + origin.y),
        Point(bbox[1].x + origin.x, bbox[1].y + origin.y),
    )


def _shift_envelope(
    envelope: tuple[Point, Point],
    origin: Point,
) -> tuple[Point, Point]:
    return (
        Point(envelope[0].x + origin.x, envelope[0].y + origin.y),
        Point(envelope[1].x + origin.x, envelope[1].y + origin.y),
    )


def _shift_layer_envelopes(
    envelopes: tuple[
        list[tuple[Point, Point]],
        list[tuple[Point, Point]],
        list[tuple[Point, Point]],
    ],
    origin: Point,
) -> tuple[
    list[tuple[Point, Point]],
    list[tuple[Point, Point]],
    list[tuple[Point, Point]],
]:
    return (
        [_shift_envelope(envelope, origin) for envelope in envelopes[0]],
        [_shift_envelope(envelope, origin) for envelope in envelopes[1]],
        [_shift_envelope(envelope, origin) for envelope in envelopes[2]],
    )


def _shift_rect(rect: tuple[Point, Point], origin: Point) -> tuple[Point, Point]:
    return (
        Point(rect[0].x + origin.x, rect[0].y + origin.y),
        Point(rect[1].x + origin.x, rect[1].y + origin.y),
    )


def _shift_rects(
    rects: tuple[tuple[Point, Point], ...],
    origin: Point,
) -> list[tuple[Point, Point]]:
    return [_shift_rect(rect, origin) for rect in rects]


def _rect_area(rect: tuple[Point, Point]) -> float:
    return max(0.0, rect[1].x - rect[0].x) * max(0.0, rect[1].y - rect[0].y)


def _component_geometry_bbox(comp) -> tuple[Point, Point]:
    bbox_min, bbox_max = comp.bbox()
    min_x = bbox_min.x
    min_y = bbox_min.y
    max_x = bbox_max.x
    max_y = bbox_max.y
    for pad in comp.pads:
        min_x = min(min_x, pad.pos.x)
        min_y = min(min_y, pad.pos.y)
        max_x = max(max_x, pad.pos.x)
        max_y = max(max_y, pad.pos.y)
    return Point(min_x, min_y), Point(max_x, max_y)


def _bbox_disjoint(a: tuple[Point, Point], b: tuple[Point, Point]) -> bool:
    return a[1].x <= b[0].x or b[1].x <= a[0].x or a[1].y <= b[0].y or b[1].y <= a[0].y


def _rect_lists_disjoint(
    rects_a: list[tuple[Point, Point]],
    rects_b: list[tuple[Point, Point]],
) -> bool:
    for rect_a in rects_a:
        for rect_b in rects_b:
            if not _bbox_disjoint(rect_a, rect_b):
                return False
    return True
def _resolve_parent_local_allowlist(component_zones: dict[str, Any], loaded_artifacts) -> set[str]:
    child_refs = set()
    for artifact in loaded_artifacts:
        child_refs.update(artifact.layout.components.keys())
    return {
        ref
        for ref in component_zones.keys()
        if ref not in child_refs
    }


def _seed_outline_dimensions(
    loaded_artifacts,
    derived: DerivedAttachmentConstraints,
    spacing_mm: float,
    *,
    area_overhead: float = 2.5,
    aspect_target: float = 1.0,
) -> tuple[float, float]:
    """Estimate (width, height) for the seed board outline the unified
    placer runs in.

    The legacy ``estimate_layer_aware_parent_board_size`` operates on
    layer envelopes (pad rects), which under-counts space for parent
    composition: synthetic blocks need room for their full bbox, not
    just their pad cluster. We bound by the children's whole-bbox area
    with slack, then floor by the per-axis constraint span when any
    axis-pinning constraint exists -- so an aggressive ``USB on left +
    power on right`` config cannot collapse the seed below the children's
    natural width.

    ``area_overhead`` (default 2.5) multiplies child total area to set
    the seed's nominal area: lower = tighter seed = forces compaction;
    higher = looser seed = lets the placer sprawl.

    ``aspect_target`` (default 1.0 = square) is the seed's width/height
    ratio.  Setting <1.0 produces a tall seed (favours vertical
    layouts), >1.0 a wide seed (favours horizontal-strip layouts).  The
    floors below (max widths/heights, sum*0.6, edge spans) still apply,
    so an aggressive aspect cannot collapse the seed below what the
    children actually need.
    """
    if not loaded_artifacts:
        return (max(20.0, spacing_mm * 4),) * 2

    # Use the content bbox of each artifact (not the leaf-PCB outline) so
    # the seed reflects the placer's actual space need.
    n = len(loaded_artifacts)
    widths: list[float] = []
    heights: list[float] = []
    for art in loaded_artifacts:
        transformed = transform_loaded_artifact(art, origin=Point(0.0, 0.0), rotation=0.0)
        tl, br = transformed.bounding_box
        widths.append(max(0.0, br.x - tl.x))
        heights.append(max(0.0, br.y - tl.y))
    total_area = sum(w * h for w, h in zip(widths, heights))
    target_area = max(total_area, 1.0) * max(0.5, area_overhead)
    aspect = max(0.1, aspect_target)
    base_w = math.sqrt(target_area * aspect) + spacing_mm * 2.0
    base_h = math.sqrt(target_area / aspect) + spacing_mm * 2.0
    sum_w = sum(widths) + spacing_mm * (n + 1)
    sum_h = sum(heights) + spacing_mm * (n + 1)
    # Floors: max single child + spacing, sum*0.6 fallback, and the
    # aspect-driven base.  The floors keep the seed solvable even when
    # aspect_target is aggressive (e.g. 0.3 = very wide and short).
    seed_w = max(base_w, sum_w * 0.6, max(widths) + spacing_mm * 4)
    seed_h = max(base_h, sum_h * 0.6, max(heights) + spacing_mm * 4)

    horizontal_widths: list[float] = []
    vertical_heights: list[float] = []
    for spec in derived.child_specs.values():
        if not spec.constraints:
            continue
        primary = next((c for c in spec.constraints if c.strict), spec.constraints[0])
        idx = spec.child_index
        if primary.target == "edge":
            if primary.value in ("left", "right"):
                horizontal_widths.append(widths[idx])
            elif primary.value in ("top", "bottom"):
                vertical_heights.append(heights[idx])
        elif primary.target == "corner":
            horizontal_widths.append(widths[idx])
            vertical_heights.append(heights[idx])

    if horizontal_widths:
        span = sum(horizontal_widths) + spacing_mm * (len(horizontal_widths) + 1)
        seed_w = max(seed_w, span)
    if vertical_heights:
        span = sum(vertical_heights) + spacing_mm * (len(vertical_heights) + 1)
        seed_h = max(seed_h, span)
    return seed_w, seed_h


def _post_solve_geometry(
    placements: dict[str, ChildArtifactPlacement],
    loaded_artifacts,
) -> tuple[
    dict[int, tuple[Point, Point]],
    list[dict[str, Any]],
    dict[str, Point],
    dict[int, Any],
]:
    """Build per-child bbox, envelopes, anchor positions, and transformed
    cache from the solver output. Used by the validation block and the
    final-outline computation downstream."""
    bboxes_by_index: dict[int, tuple[Point, Point]] = {}
    placed_envelopes: list[dict[str, Any]] = []
    anchor_positions: dict[str, Point] = {}
    transformed_by_index: dict[int, Any] = {}

    for child_index, artifact in enumerate(loaded_artifacts):
        placement = placements.get(artifact.instance_path)
        if placement is None:
            continue
        transformed = transform_loaded_artifact(
            artifact,
            origin=placement.origin,
            rotation=placement.rotation,
        )
        bbox = transformed.bounding_box
        bboxes_by_index[child_index] = bbox
        transformed_by_index[child_index] = transformed
        envelopes = child_layer_envelopes(transformed)
        blocker_set = extract_leaf_blocker_set(artifact)
        placed_envelopes.append(
            {
                "bbox": bbox,
                "envelopes": envelopes,
                "blocker_set": blocker_set,
                "origin": placement.origin,
                "rotation": placement.rotation,
                "label": artifact.sheet_name,
            }
        )
        # Anchor position for each constrained ref: the existing per-spec
        # constraint_entries already encode local_anchor_offset at the
        # CHOSEN rotation, so we walk transformed.transformed_components
        # for the constrained refs and recover the world-frame anchor
        # using the same local-anchor math the legacy compose used.
    return bboxes_by_index, placed_envelopes, anchor_positions, transformed_by_index


def _resolve_constraint_anchor_positions(
    derived: DerivedAttachmentConstraints,
    placements: dict[str, ChildArtifactPlacement],
    loaded_artifacts,
    transformed_by_index: dict[int, Any],
    parent_local: dict[str, Component],
) -> dict[str, Point]:
    """Compute the world-frame anchor position for every constrained ref.

    For child-artifact constraints, recompute local_anchor_offset at the
    chosen rotation (the rotation the solver picked, not the spec's first
    candidate) and add it to the placement origin. For parent-local
    constraints, use the component's pad centroid or body center.
    """
    anchors: dict[str, Point] = {}
    for spec in derived.child_specs.values():
        artifact = loaded_artifacts[spec.child_index]
        placement = placements.get(artifact.instance_path)
        if placement is None:
            continue
        transformed = transformed_by_index.get(spec.child_index)
        if transformed is None:
            continue
        blocker_set = extract_leaf_blocker_set(artifact)
        for constraint in spec.constraints:
            try:
                from kicraft.autoplacer.brain.subcircuit_composer import (
                    _compute_local_anchor_offset,
                )
                local_offset = _compute_local_anchor_offset(
                    transformed,
                    constraint,
                    spec.constraints,
                    blocker_set,
                    placement.rotation,
                )
            except Exception:
                continue
            anchors[constraint.ref] = Point(
                placement.origin.x + local_offset.x,
                placement.origin.y + local_offset.y,
            )

    for constraint in derived.parent_local_constraints:
        comp = parent_local.get(constraint.ref)
        if comp is None:
            continue
        if comp.pads:
            cx = sum(p.pos.x for p in comp.pads) / len(comp.pads)
            cy = sum(p.pos.y for p in comp.pads) / len(comp.pads)
        else:
            anchor = comp.body_center if comp.body_center is not None else comp.pos
            cx, cy = anchor.x, anchor.y
        anchors[constraint.ref] = Point(cx, cy)
    return anchors


def _block_artifact_origin(comp: Component) -> Point:
    """Inverse of synthetic-block-pos -> artifact-origin mapping. For a
    synthetic block, ``world_origin = pos - rotated(body_center_offset, rot)``;
    matches ``placements_from_solved_state``. For non-block components
    (parent-local mounting holes) the origin is simply ``comp.pos``.
    """
    if comp.kind != "subcircuit" or comp.block_artifact_origin_offset is None:
        return comp.pos
    body_offset = comp.block_artifact_origin_offset
    rad = math.radians(comp.rotation)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    return Point(
        comp.pos.x - (body_offset.x * cos_r - body_offset.y * sin_r),
        comp.pos.y - (body_offset.x * sin_r + body_offset.y * cos_r),
    )


def _apply_slide(comp: Component, free_axis_y: bool, delta: float) -> None:
    """Translate a Component (its ``pos``, ``body_center``, and pad
    positions) along one axis by ``delta``."""
    if free_axis_y:
        comp.pos = Point(comp.pos.x, comp.pos.y + delta)
        if comp.body_center is not None:
            comp.body_center = Point(comp.body_center.x, comp.body_center.y + delta)
        for pad in comp.pads:
            pad.pos = Point(pad.pos.x, pad.pos.y + delta)
    else:
        comp.pos = Point(comp.pos.x + delta, comp.pos.y)
        if comp.body_center is not None:
            comp.body_center = Point(comp.body_center.x + delta, comp.body_center.y)
        for pad in comp.pads:
            pad.pos = Point(pad.pos.x + delta, pad.pos.y)


def _slide_clearance_ok(solved: dict[str, Component], moving_ref: str) -> bool:
    """True iff the bbox of ``solved[moving_ref]`` does not introduce an
    incompatible overlap with any other placed component. Compatible
    block-on-block overlaps (per ``can_overlap_sparse``) are allowed --
    those represent the dual-layer stacking the parent solver already
    builds. Non-block components and same-side blocks are not allowed
    to overlap.
    """
    moving = solved[moving_ref]
    moving_bbox = moving.bbox()
    moving_blocker = moving.block_blocker_set
    moving_origin = _block_artifact_origin(moving)
    for other_ref, other in solved.items():
        if other_ref == moving_ref:
            continue
        if _bbox_disjoint(moving_bbox, other.bbox()):
            continue
        if moving_blocker is not None and other.block_blocker_set is not None:
            other_origin = _block_artifact_origin(other)
            if can_overlap_sparse(
                moving_blocker,
                moving_origin,
                moving.rotation,
                other.block_blocker_set,
                other_origin,
                other.rotation,
            ):
                continue
        return False
    return True


def _largest_safe_slide(
    solved: dict[str, Component],
    moving_ref: str,
    free_axis_y: bool,
    desired_delta: float,
) -> float:
    """Binary-search the largest |delta| in [0, |desired_delta|] (same sign
    as desired) that keeps ``_slide_clearance_ok`` true. Mutates and
    restores the moving component during probes."""
    if abs(desired_delta) < 1e-3:
        return 0.0

    moving = solved[moving_ref]
    saved_pos = moving.pos
    saved_body = moving.body_center
    saved_pad_positions = [pad.pos for pad in moving.pads]

    def restore() -> None:
        moving.pos = saved_pos
        moving.body_center = saved_body
        for pad, original in zip(moving.pads, saved_pad_positions):
            pad.pos = original

    def safe_at(d: float) -> bool:
        _apply_slide(moving, free_axis_y, d)
        try:
            return _slide_clearance_ok(solved, moving_ref)
        finally:
            restore()

    if safe_at(desired_delta):
        return desired_delta

    sign = 1.0 if desired_delta > 0 else -1.0
    lo, hi = 0.0, abs(desired_delta)
    for _ in range(8):
        mid = (lo + hi) / 2.0
        if safe_at(sign * mid):
            lo = mid
        else:
            hi = mid
    return sign * lo


def _cluster_bbox(
    solved: dict[str, Component], exclude_refs: set[str]
) -> tuple[Point, Point] | None:
    """Bbox union of every component in ``solved`` whose ref is not in
    ``exclude_refs``. Returns None when nothing is left."""
    lo_x = lo_y = math.inf
    hi_x = hi_y = -math.inf
    for ref, comp in solved.items():
        if ref in exclude_refs:
            continue
        b_min, b_max = comp.bbox()
        if b_min.x < lo_x:
            lo_x = b_min.x
        if b_min.y < lo_y:
            lo_y = b_min.y
        if b_max.x > hi_x:
            hi_x = b_max.x
        if b_max.y > hi_y:
            hi_y = b_max.y
    if lo_x == math.inf:
        return None
    return (Point(lo_x, lo_y), Point(hi_x, hi_y))


def _slide_constrained_to_cluster(
    solved: dict[str, Component],
    derived: DerivedAttachmentConstraints,
    synthetic_refs: dict[int, str],
) -> None:
    """In-place: pull each constrained component back to the cluster.

    The parent solver pins constrained components to the seed frame's
    edges or corners. The seed frame is oversized (2.5x area slack) to
    leave routing room, so a constrained component pinned to its corner
    can sit far outside the actual cluster's extent. ``_compute_final_outline``
    then snaps the corresponding board side to include the constrained
    component, inflating PCB area.

    Two operations:

      * **Edge-pinned components** keep their pinned axis fixed but slide
        along the free axis so their bbox falls inside the cluster's
        perpendicular span. Walked back via ``_largest_safe_slide`` if
        the slide would create an incompatible block overlap.
      * **Corner-pinned parent-local components** (mounting holes) jump
        to the corresponding corner of the cluster bbox. They are placed
        BY the constraint -- their absolute position is meant to track
        the board corner -- so dragging them off the seed-frame corner
        onto the cluster corner is the right operation.
    """
    edge_targets: list[tuple[str, str]] = []
    for child_index, spec in derived.child_specs.items():
        block_ref = synthetic_refs.get(child_index)
        if block_ref is None or block_ref not in solved:
            continue
        primary = next((c for c in spec.constraints if c.strict), None)
        if primary is None or primary.target != "edge":
            continue
        edge_targets.append((block_ref, primary.value))
    for c in derived.parent_local_constraints:
        if c.target != "edge" or c.ref not in solved:
            continue
        edge_targets.append((c.ref, c.value))

    corner_targets: list[tuple[str, str]] = [
        (c.ref, c.value)
        for c in derived.parent_local_constraints
        if c.target == "corner" and c.ref in solved
    ]
    if not edge_targets and not corner_targets:
        return

    # Corner-pinned refs sit at board corners by design and would stretch
    # the span to the full seed frame; exclude them from cluster math.
    corner_refs = {c.ref for c in derived.constraints if c.target == "corner"}

    slides_applied = 0
    snapped_corners = 0

    # --- 1. Edge-pinned slides on the free axis ---
    for ref, edge_value in edge_targets:
        comp = solved[ref]
        free_axis_y = edge_value in ("left", "right")
        # Cluster span excludes the moving ref and every corner-pinned ref.
        exclude = {ref} | corner_refs
        cluster = _cluster_bbox(solved, exclude)
        if cluster is None:
            continue
        cluster_lo = cluster[0].y if free_axis_y else cluster[0].x
        cluster_hi = cluster[1].y if free_axis_y else cluster[1].x

        b_min, b_max = comp.bbox()
        cur_lo, cur_hi = (b_min.y, b_max.y) if free_axis_y else (b_min.x, b_max.x)

        if cur_lo >= cluster_lo and cur_hi <= cluster_hi:
            continue
        if (cur_hi - cur_lo) > (cluster_hi - cluster_lo) + 1e-3:
            continue

        if cur_hi > cluster_hi:
            delta = cluster_hi - cur_hi
        elif cur_lo < cluster_lo:
            delta = cluster_lo - cur_lo
        else:
            continue

        safe = _largest_safe_slide(solved, ref, free_axis_y, delta)
        if abs(safe) < 1e-3:
            continue
        _apply_slide(comp, free_axis_y, safe)
        slides_applied += 1

    # --- 2. Corner-pinned parent-local snap to cluster corner ---
    # Edge-slid components are now in the cluster's perpendicular span,
    # so include them in the cluster bbox used for corner placement.
    cluster_for_corners = _cluster_bbox(solved, corner_refs)
    if cluster_for_corners is not None:
        c_min, c_max = cluster_for_corners
        for ref, corner_value in corner_targets:
            comp = solved[ref]
            cur_anchor_x = (
                comp.body_center.x if comp.body_center is not None else comp.pos.x
            )
            cur_anchor_y = (
                comp.body_center.y if comp.body_center is not None else comp.pos.y
            )
            target_x = c_min.x if "left" in corner_value else c_max.x
            target_y = c_min.y if "top" in corner_value else c_max.y
            dx = target_x - cur_anchor_x
            dy = target_y - cur_anchor_y
            if abs(dx) > 1e-3:
                _apply_slide(comp, free_axis_y=False, delta=dx)
            if abs(dy) > 1e-3:
                _apply_slide(comp, free_axis_y=True, delta=dy)
            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                snapped_corners += 1

    if slides_applied or snapped_corners:
        print(
            f"  Cluster-slide: aligned {slides_applied} edge-constrained, "
            f"{snapped_corners} corner-constrained component(s) to the cluster"
        )


def _compute_final_outline(
    placed_bboxes: list[tuple[Point, Point]],
    constraints: list[AttachmentConstraint],
    anchor_positions: dict[str, Point],
    spacing_mm: float,
) -> tuple[Point, Point]:
    """Final outline tracks ``constraint_aware_outline`` (which already
    applies ``margin_mm`` to unconstrained sides and snaps constrained
    sides to anchor targets), expanded if necessary so the outline still
    contains every placed bbox -- never shrinking past geometry."""
    if not placed_bboxes:
        return (Point(0.0, 0.0), Point(0.0, 0.0))

    if not constraints:
        geom_min_x = min(b[0].x for b in placed_bboxes) - spacing_mm
        geom_min_y = min(b[0].y for b in placed_bboxes) - spacing_mm
        geom_max_x = max(b[1].x for b in placed_bboxes) + spacing_mm
        geom_max_y = max(b[1].y for b in placed_bboxes) + spacing_mm
        return (Point(geom_min_x, geom_min_y), Point(geom_max_x, geom_max_y))

    constraint_outline = constraint_aware_outline(
        placed_bboxes=placed_bboxes,
        attachment_constraints=constraints,
        constrained_ref_world_anchors=anchor_positions,
        margin_mm=spacing_mm,
    )
    # On a side constrained by an EDGE constraint (e.g. J1 edge=left)
    # the outline must STOP at the constraint anchor target -- expanding
    # past it would push the connector inboard so its body no longer sits
    # flush with the PCB edge. CORNER-only sides (e.g. only H4 corner=
    # top-left constrains the top edge) are different: the corner mount
    # may have been escaped by _stack_compatible_blocks / corner-escape
    # to dodge an edge connector, and using the escaped mount's centroid
    # as the precise outline edge clips legitimately-placed leaves outside
    # the board. For corner-only sides, expand to fit geometry so escaped
    # mounts never shrink the outline below placed bboxes.
    edge_constrained_sides = {
        "left": False, "right": False, "top": False, "bottom": False,
    }
    corner_constrained_sides = {
        "left": False, "right": False, "top": False, "bottom": False,
    }
    for c in constraints:
        if c.target == "edge":
            edge_constrained_sides[c.value] = True
        elif c.target == "corner":
            for side in c.value.split("-"):
                if side in corner_constrained_sides:
                    corner_constrained_sides[side] = True

    geom_min_x = min(b[0].x for b in placed_bboxes)
    geom_min_y = min(b[0].y for b in placed_bboxes)
    geom_max_x = max(b[1].x for b in placed_bboxes)
    geom_max_y = max(b[1].y for b in placed_bboxes)

    # For corner-constrained sides, ``c_val`` is the corner anchor's
    # coordinate (no margin -- corner anchors are points, and
    # ``constraint_aware_outline`` populates ``{top,bottom,left,right}_edges``
    # from corner constraints without adding ``margin_mm``). When a leaf's
    # geometry extends past the corner anchor on that side, we must inflate
    # ``g_val`` by ``spacing_mm`` so pad copper keeps its full edge clearance
    # -- the unconstrained branch already gets margin via constraint_aware_outline.
    def _resolve_min(side: str, c_val: float, g_val: float) -> float:
        if edge_constrained_sides[side]:
            return c_val
        if corner_constrained_sides[side]:
            return min(c_val, g_val - spacing_mm)
        return min(c_val, g_val)

    def _resolve_max(side: str, c_val: float, g_val: float) -> float:
        if edge_constrained_sides[side]:
            return c_val
        if corner_constrained_sides[side]:
            return max(c_val, g_val + spacing_mm)
        return max(c_val, g_val)

    out_min_x = _resolve_min("left", constraint_outline[0].x, geom_min_x)
    out_min_y = _resolve_min("top", constraint_outline[0].y, geom_min_y)
    out_max_x = _resolve_max("right", constraint_outline[1].x, geom_max_x)
    out_max_y = _resolve_max("bottom", constraint_outline[1].y, geom_max_y)
    return (Point(out_min_x, out_min_y), Point(out_max_x, out_max_y))


def _is_mounting_hole_ref(ref: str, comp: Component | None = None) -> bool:
    """Heuristic: refs like 'H1', 'H86' are mounting holes.

    KiCad convention is 'H' or 'MH' prefix. Components fed to manual
    mode's parent_local list are already filtered to constraint-pinned
    items, so a name match is sufficient here.
    """
    if not ref:
        return False
    upper = ref.upper()
    return upper.startswith("H") and (len(upper) == 1 or upper[1].isdigit() or upper[1] == "_")


def _move_component_to(comp: Component, new_pos: Point) -> None:
    """Translate a Component (and its pads / body_center) so its anchor
    lands at ``new_pos``. Preserves rotation; mirrors the same delta
    pattern used by _snap_parent_local.
    """
    dx = new_pos.x - comp.pos.x
    dy = new_pos.y - comp.pos.y
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return
    comp.pos = Point(comp.pos.x + dx, comp.pos.y + dy)
    if comp.body_center is not None:
        comp.body_center = Point(comp.body_center.x + dx, comp.body_center.y + dy)
    for pad in comp.pads:
        pad.pos = Point(pad.pos.x + dx, pad.pos.y + dy)


def _snap_parent_local(
    comps: dict[str, Component],
    constraints: list[AttachmentConstraint],
    outline: tuple[Point, Point],
) -> None:
    """Translate each parent-local component with an edge/corner/zone
    constraint so its anchor lands at the exact constraint coordinate
    via edge_anchor_target_coordinate. Restores sub-mm precision the
    geometry validator enforces (the solver's edge-pinning math may
    leave it within the connector_inset_mm jitter window)."""
    min_pt, max_pt = outline
    for c in constraints:
        comp = comps.get(c.ref)
        if comp is None:
            continue
        target_x = comp.pos.x
        target_y = comp.pos.y
        if c.target == "edge":
            if c.value in ("left", "right"):
                target_x = edge_anchor_target_coordinate(c.value, c, min_pt, max_pt)
            elif c.value in ("top", "bottom"):
                target_y = edge_anchor_target_coordinate(c.value, c, min_pt, max_pt)
        elif c.target == "corner":
            corner_sides = {
                "top-left": ("left", "top"),
                "top-right": ("right", "top"),
                "bottom-left": ("left", "bottom"),
                "bottom-right": ("right", "bottom"),
            }.get(c.value)
            if corner_sides is None:
                continue
            target_x = edge_anchor_target_coordinate(corner_sides[0], c, min_pt, max_pt)
            target_y = edge_anchor_target_coordinate(corner_sides[1], c, min_pt, max_pt)
        elif c.target == "zone" and c.value == "bottom":
            target_y = max_pt.y - c.inward_keep_in_mm
        else:
            continue

        # Compute centroid of pads (or body center fallback) and shift to
        # the target. Anchor is what the validator checks, so we move the
        # whole comp by the delta needed to bring the anchor on target.
        if comp.pads:
            anchor_x = sum(p.pos.x for p in comp.pads) / len(comp.pads)
            anchor_y = sum(p.pos.y for p in comp.pads) / len(comp.pads)
        else:
            ref = comp.body_center if comp.body_center is not None else comp.pos
            anchor_x, anchor_y = ref.x, ref.y

        dx = target_x - anchor_x
        dy = target_y - anchor_y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        comp.pos = Point(comp.pos.x + dx, comp.pos.y + dy)
        if comp.body_center is not None:
            comp.body_center = Point(comp.body_center.x + dx, comp.body_center.y + dy)
        for pad in comp.pads:
            pad.pos = Point(pad.pos.x + dx, pad.pos.y + dy)


def _synthetic_parent_definition(loaded_artifacts) -> SubCircuitDefinition:
    return SubCircuitDefinition(
        id=SubCircuitId(
            sheet_name="COMPOSED_PARENT",
            sheet_file="COMPOSED_PARENT.kicad_sch",
            instance_path="/COMPOSED_PARENT",
            parent_instance_path=None,
        ),
        schematic_path="",
        component_refs=[],
        ports=[],
        child_ids=[artifact.layout.subcircuit_id for artifact in loaded_artifacts],
        parent_id=None,
        is_leaf=False,
        sheet_uuid="",
        notes=[
            "synthetic_parent=true",
            f"artifact_count={len(loaded_artifacts)}",
        ],
    )


def _compose_artifacts(
    loaded_artifacts,
    *,
    spacing_mm: float,
    rotation_step_deg: float,
    parent_definition: SubCircuitDefinition | None = None,
    pcb_path: Path | None = None,
    cfg: dict[str, Any] | None = None,
    seed: int = 0,
    seed_area_overhead: float = 2.5,
    seed_aspect_target: float = 1.0,
    manual_layout: ManualLayout | None = None,
) -> tuple[ParentCompositionState, list[dict[str, Any]]]:
    """Compose loaded artifacts into a parent composition snapshot.

    Replaces the legacy 3-iteration constraint-placement loop with a
    single pass through the unified ``PlacementSolver``: each artifact
    becomes a synthetic block component carrying its sparse blocker
    set, child attachment constraints become block-level zone entries
    with anchor offsets, and the solver picks positions and rotations
    in one shot.
    """
    from kicraft.autoplacer.config import discover_project_config, load_project_config
    import logging

    logger = logging.getLogger(__name__)

    # Merge caller-supplied cfg over project config so explicit overrides
    # (e.g., opposite_side_attraction_k from the parent route command,
    # connector_edge_inset_mm tuning) survive. Without this merge,
    # load_project_config used to clobber whatever the caller passed in.
    user_cfg = dict(cfg or {})
    cfg = dict(user_cfg)
    component_zones: dict[str, Any] = cfg.get("component_zones", {})
    parent_local: dict[str, Component] = {}

    if pcb_path:
        # No try/except here: a project-config parse error or a
        # parent_local extraction failure leaves the solver without
        # mounting-hole keepouts, edge-connector pins, etc. -- which
        # silently produces broken placements (items_not_allowed
        # violations, leaves over mounting holes). Better to fail loudly
        # at the cfg loading boundary than to ship a bad layout.
        project_dir = Path(pcb_path).resolve().parent
        cfg_file = discover_project_config(project_dir)
        if cfg_file is not None:
            project_cfg = load_project_config(str(cfg_file))
            cfg = {**project_cfg, **user_cfg}
            component_zones = cfg.get("component_zones", {})
        parent_local = extract_parent_local_components(
            str(pcb_path),
            loaded_artifacts,
            allowlist=_resolve_parent_local_allowlist(
                component_zones, loaded_artifacts
            ),
        )

    derived = derive_attachment_constraints(
        loaded_artifacts,
        parent_local,
        component_zones,
        cfg,
        rotation_step_deg=rotation_step_deg,
    )
    all_constraints = derived.constraints
    logger.info("composition: %d attachment constraints derived", len(all_constraints))

    # --- Build synthetic block components for each child artifact ---
    synthetic_refs = {
        i: synthetic_block_ref(i, art.sheet_name)
        for i, art in enumerate(loaded_artifacts)
    }
    block_zones, allowed_rotations = attachment_constraints_to_zones(
        derived, synthetic_refs, list(loaded_artifacts)
    )
    # Project-level override for the layer-intent heuristic. Sheet
    # names listed here force can_overlap_sparse to treat the leaf as
    # having no front-side copper intent, so SMT-on-front candidates
    # may stack on it. Generalises the THT-back-anchor case (battery
    # holders, terminal blocks) without per-project hardcoding -- any
    # project lists its own offending sheet names. Default empty: the
    # heuristic alone handles the common shadow-PTH case.
    _back_through_hole_leaves = set(
        cfg.get("parent_placement", {}).get("backside_through_hole_leaves", []) or []
    )
    synthetic_comps: dict[str, Component] = {}
    for i, art in enumerate(loaded_artifacts):
        ref = synthetic_refs[i]
        rot = float(block_zones.get(ref, {}).get("rotation", 0.0))
        spec = derived.child_specs.get(i)
        rotation_models = spec.models if spec is not None else None
        comp = artifact_to_component(
            art, ref=ref, rotation=rot, rotation_models=rotation_models
        )
        if ref in allowed_rotations:
            comp.allowed_rotations = list(allowed_rotations[ref])
        if art.sheet_name in _back_through_hole_leaves:
            comp.block_force_back_only = True
        synthetic_comps[ref] = comp

    # Parent-local components (mounting holes etc.) join the same solver
    # state. They keep their loaded positions; _snap_parent_local applies
    # the exact constraint-target snap after solve.
    for ref, comp in parent_local.items():
        synthetic_comps[ref] = comp

    seed_w, seed_h = _seed_outline_dimensions(
        loaded_artifacts,
        derived,
        spacing_mm,
        area_overhead=seed_area_overhead,
        aspect_target=seed_aspect_target,
    )

    parent_subcircuit = parent_definition or _synthetic_parent_definition(loaded_artifacts)
    interconnect_nets = infer_interconnect_nets_pre_placement(
        parent_subcircuit, loaded_artifacts, synthetic_refs
    )

    state_in = BoardState(
        components=synthetic_comps,
        nets=interconnect_nets,
        traces=[],
        vias=[],
        silkscreen=[],
        board_outline=(Point(0.0, 0.0), Point(seed_w, seed_h)),
    )
    # Forward project-level cfg so the solver sees connector_edge_inset_mm,
    # edge_margin_mm, force_attract_k, etc. parent_placement is an optional
    # override layer for parent-specific tuning. component_zones is forced
    # to block-level only so the leaf-pad warning doesn't fire and so the
    # solver doesn't try to pin J1/J2/J3 directly (those flow through
    # attachment_constraints_to_zones into the synthetic block zones).
    #
    # parent_keep_in_rects: hard-keepout zones derived from parent-local
    # attachment constraints (e.g. mounting-hole inward_keep_in_mm). The
    # solver's _resolve_keep_in_rects pass pushes any unlocked component
    # whose bbox enters one of these zones back toward the board center.
    # Without this, fast-cfg / SA-disabled runs were producing layouts
    # with leaves stamped on top of mounting-hole keep-ins -> stamped
    # DRC items_not_allowed violations (14 on a recent failed candidate).
    parent_keep_in_specs = [
        {"ref": c.ref, "margin_mm": float(c.inward_keep_in_mm)}
        for c in derived.parent_local_constraints
        if c.ref in parent_local
    ]
    if manual_layout is not None:
        # Manual mode: user-supplied placements + outline. Skip the solver
        # and the auto outline-fit pass entirely. Validation, stamping and
        # routing run unchanged on these placements.
        manual_by_path = manual_layout.placement_by_path()
        missing = [
            art.instance_path
            for art in loaded_artifacts
            if art.instance_path not in manual_by_path
        ]
        if missing:
            raise ValueError(
                "manual layout missing placements for instance paths: "
                + ", ".join(missing)
            )
        placements_dict = {
            art.instance_path: ChildArtifactPlacement(
                artifact=art,
                origin=manual_by_path[art.instance_path].origin,
                rotation=manual_by_path[art.instance_path].rotation,
            )
            for art in loaded_artifacts
        }
        # Honour user-supplied parent-local positions when present; else
        # keep the extracted positions and let _snap_parent_local apply
        # constraint targets within the manual outline.
        parent_local_solved = {
            ref: copy.deepcopy(comp) for ref, comp in parent_local.items()
        }
        manual_pl_by_ref = manual_layout.parent_local_by_ref()
        for ref, mpl in manual_pl_by_ref.items():
            comp = parent_local_solved.get(ref)
            if comp is not None:
                comp.pos = mpl.pos

        # GUI mounting-hole panel maps to the parent's existing
        # parent-local mounting-hole footprints in alphabetical ref
        # order (so H4 < H86 etc.). Override each paired component's
        # position with the user's chosen corner+inset; mark it
        # user-positioned so the constraint snap below leaves it
        # alone.
        user_positioned_refs: set[str] = set()
        gui_holes = sorted(
            getattr(manual_layout, "mounting_holes", []) or [],
            key=lambda h: h.index,
        )
        if gui_holes:
            mh_refs = sorted(
                ref for ref, comp in parent_local_solved.items()
                if _is_mounting_hole_ref(ref, comp)
            )
            for hole, ref in zip(gui_holes, mh_refs):
                comp = parent_local_solved.get(ref)
                if comp is None:
                    continue
                _move_component_to(comp, hole.pos)
                user_positioned_refs.add(ref)
            if len(gui_holes) > len(mh_refs):
                print(
                    f"[manual-layout] note: GUI has {len(gui_holes)} mounting "
                    f"holes but parent has {len(mh_refs)} matching footprints "
                    f"({', '.join(mh_refs) or 'none'}); the extra GUI holes "
                    f"are decorative until composer-side footprint synthesis "
                    f"lands.",
                    file=sys.stderr,
                )

        solver_phase_timings = {}

        placed_child_bboxes, placed_envelopes, _ignored_anchors, transformed_by_index = (
            _post_solve_geometry(placements_dict, loaded_artifacts)
        )
        child_anchor_positions = _resolve_constraint_anchor_positions(
            derived, placements_dict, loaded_artifacts, transformed_by_index, parent_local_solved
        )
        exact_outline = manual_layout.board_outline
        _snap_parent_local(
            parent_local_solved,
            [
                c for c in derived.parent_local_constraints
                if c.ref not in user_positioned_refs
            ],
            exact_outline,
        )
    else:
        solver_cfg = {
            **cfg,
            **cfg.get("parent_placement", {}),
            "component_zones": dict(block_zones),
            "placement_clearance_mm": spacing_mm,
            "clearance_mm": spacing_mm,
            "parent_keep_in_rects": parent_keep_in_specs,
        }
        solver = PlacementSolver(state_in, config=solver_cfg, seed=seed)
        solved = solver.solve()
        solver_phase_timings = dict(getattr(solver, "last_solve_phase_timings", {}))

        # Slide any edge-constrained block whose free axis drifted outside the
        # rest of the cluster's perpendicular span. The solver pins X (or Y)
        # to the board edge but lets the free axis float; a leaf parked in a
        # corner inflates the final outline because the orthogonal sides snap
        # to include it. Bringing it back inside the cluster span lets
        # _compute_final_outline shrink the board.
        _slide_constrained_to_cluster(solved, derived, synthetic_refs)

        # --- Recover artifact placements from solver output ---
        placements_dict = placements_from_solved_state(solved, list(loaded_artifacts), synthetic_refs)
        parent_local_solved: dict[str, Component] = {
            ref: solved[ref] for ref in parent_local if ref in solved
        }

        # Build per-child geometry for outline + validation.
        placed_child_bboxes, placed_envelopes, _ignored_anchors, transformed_by_index = (
            _post_solve_geometry(placements_dict, loaded_artifacts)
        )
        child_anchor_positions = _resolve_constraint_anchor_positions(
            derived, placements_dict, loaded_artifacts, transformed_by_index, parent_local_solved
        )

        placed_bbox_list = [
            placed_child_bboxes[index] for index in sorted(placed_child_bboxes)
        ]
        exact_outline = _compute_final_outline(
            placed_bbox_list, all_constraints, child_anchor_positions, spacing_mm
        )

        # Snap parent-local components to exact constraint coordinates.
        _snap_parent_local(
            parent_local_solved,
            derived.parent_local_constraints,
            exact_outline,
        )

    # --- Build CompositionEntry list + transformed_payloads in artifact order ---
    entries: list[CompositionEntry] = []
    transformed_payloads: list[dict[str, Any]] = []
    child_artifact_placements: list[ChildArtifactPlacement] = []
    for child_index, artifact in enumerate(loaded_artifacts):
        placement = placements_dict.get(artifact.instance_path)
        if placement is None:
            raise ValueError(
                f"Solver did not produce a placement for {artifact.instance_path}"
            )
        transformed = transformed_by_index.get(child_index)
        if transformed is None:
            transformed = transform_loaded_artifact(
                artifact, origin=placement.origin, rotation=placement.rotation
            )
        entry = CompositionEntry(
            artifact_dir=artifact.artifact_dir,
            sheet_name=artifact.sheet_name,
            instance_path=artifact.instance_path,
            origin=placement.origin,
            rotation=placement.rotation,
            transformed_bbox=transformed.instance.transformed_bbox,
            component_count=len(transformed.transformed_components),
            trace_count=len(transformed.transformed_traces),
            via_count=len(transformed.transformed_vias),
            anchor_count=len(transformed.transformed_anchors),
        )
        entries.append(entry)
        child_artifact_placements.append(placement)
        transformed_payloads.append(
            {
                "artifact": artifact_debug_dict(artifact),
                "transformed": transformed_debug_dict(transformed),
                "summary": transformed_summary(transformed),
            }
        )

    composition = build_parent_composition(
        parent_subcircuit,
        child_artifact_placements=child_artifact_placements,
        board_outline=exact_outline,
        local_components=parent_local_solved,
    )

    copper_manifest = build_copper_manifest(
        composed_children=composition.composed_children,
        final_child_bboxes={
            loaded_artifacts[index].instance_path: (
                (bbox[0].x, bbox[0].y),
                (bbox[1].x, bbox[1].y),
            )
            for index, bbox in placed_child_bboxes.items()
        },
    )

    for ref, comp in parent_local_solved.items():
        if ref in composition.board_state.components:
            composition.board_state.components[ref] = copy.deepcopy(comp)
        if ref in composition.hierarchy_state.local_components:
            composition.hierarchy_state.local_components[ref] = copy.deepcopy(comp)

    # --- Validation block (preserved from legacy flow) ---
    import itertools

    edge_attachment_satisfied: dict[str, bool] = {}
    mounting_hole_keep_in_satisfied: dict[str, bool] = {}

    for c in all_constraints:
        expected_x = None
        expected_y = None
        actual_x = None
        actual_y = None

        min_pt, max_pt = exact_outline

        if c.source == "child_artifact":
            anchor = child_anchor_positions.get(c.ref)
            if anchor is None:
                continue
            actual_x = anchor.x
            actual_y = anchor.y
        else:
            comp = parent_local_solved.get(c.ref)
            if not comp:
                continue
            if comp.pads:
                actual_x = sum(pad.pos.x for pad in comp.pads) / len(comp.pads)
                actual_y = sum(pad.pos.y for pad in comp.pads) / len(comp.pads)
            else:
                anchor = comp.body_center if comp.body_center is not None else comp.pos
                actual_x = anchor.x
                actual_y = anchor.y

        if c.target == "edge":
            if c.value in ("left", "right"):
                expected_x = edge_anchor_target_coordinate(c.value, c, min_pt, max_pt)
            elif c.value in ("top", "bottom"):
                expected_y = edge_anchor_target_coordinate(c.value, c, min_pt, max_pt)
        elif c.target == "corner":
            corner_sides = {
                "top-left": ("left", "top"),
                "top-right": ("right", "top"),
                "bottom-left": ("left", "bottom"),
                "bottom-right": ("right", "bottom"),
            }.get(c.value)
            if corner_sides is not None:
                x_side, y_side = corner_sides
                expected_x = edge_anchor_target_coordinate(x_side, c, min_pt, max_pt)
                expected_y = edge_anchor_target_coordinate(y_side, c, min_pt, max_pt)
        elif c.target == "zone" and c.value == "bottom":
            expected_y = max_pt.y - c.inward_keep_in_mm

        ok_x = expected_x is None or abs(actual_x - expected_x) <= 1e-3
        ok_y = expected_y is None or abs(actual_y - expected_y) <= 1e-3
        ok = ok_x and ok_y

        edge_attachment_satisfied[c.ref] = ok

        is_hole = c.ref.startswith("H") or (
            c.inward_keep_in_mm > 0 and "hole" in c.ref.lower()
        )
        if is_hole:
            mounting_hole_keep_in_satisfied[c.ref] = ok

    same_side_overlap_conflicts: list[tuple[str, str]] = []
    tht_keepout_violations: list[tuple[str, str]] = []

    ordered_entry_indices = [
        next(
            index
            for index, artifact in enumerate(loaded_artifacts)
            if artifact.instance_path == entry.instance_path
        )
        for entry in entries
    ]
    for i, j in itertools.combinations(range(len(placed_envelopes)), 2):
        item_a = placed_envelopes[i]
        item_b = placed_envelopes[j]
        env_a = item_a["envelopes"]
        env_b = item_b["envelopes"]
        art_a = loaded_artifacts[ordered_entry_indices[i]]
        art_b = loaded_artifacts[ordered_entry_indices[j]]
        rect_a = placed_child_bboxes[ordered_entry_indices[i]]
        rect_b = placed_child_bboxes[ordered_entry_indices[j]]

        if not _bbox_disjoint(rect_a, rect_b):
            blocker_a = item_a.get("blocker_set")
            blocker_b = item_b.get("blocker_set")
            if blocker_a is not None and blocker_b is not None:
                overlap_ok = can_overlap_sparse(
                    blocker_a,
                    item_a["origin"],
                    item_a["rotation"],
                    blocker_b,
                    item_b["origin"],
                    item_b["rotation"],
                )
            else:
                overlap_ok = can_overlap(env_a, env_b)
            if not overlap_ok:
                a_label = getattr(
                    art_a,
                    "label",
                    getattr(art_a, "slug", getattr(art_a, "sheet_name", f"child[{i}]")),
                )
                b_label = getattr(
                    art_b,
                    "label",
                    getattr(art_b, "slug", getattr(art_b, "sheet_name", f"child[{j}]")),
                )

                a_front, a_back, a_tht = env_a
                b_front, b_back, b_tht = env_b

                if (
                    not _rect_lists_disjoint(a_tht, b_front)
                    or not _rect_lists_disjoint(a_tht, b_back)
                    or not _rect_lists_disjoint(b_tht, a_front)
                    or not _rect_lists_disjoint(b_tht, a_back)
                    or not _rect_lists_disjoint(a_tht, b_tht)
                ):
                    tht_keepout_violations.append((a_label, b_label))
                elif (
                    not _rect_lists_disjoint(a_front, b_front)
                    or not _rect_lists_disjoint(a_back, b_back)
                    or (
                        blocker_a is not None
                        and blocker_b is not None
                        and dominant_blocker_side(blocker_a) in {"front", "back"}
                        and dominant_blocker_side(blocker_a)
                        == dominant_blocker_side(blocker_b)
                    )
                ):
                    same_side_overlap_conflicts.append((a_label, b_label))

    validation_data = {
        "edge_attachment_satisfied": edge_attachment_satisfied,
        "mounting_hole_keep_in_satisfied": mounting_hole_keep_in_satisfied,
        "same_side_overlap_conflicts": same_side_overlap_conflicts,
        "tht_keepout_violations": tht_keepout_violations,
        "constraint_count": len(all_constraints),
        "parent_local_count": len(parent_local),
    }
    unsatisfied_edges = sum(
        1 for ok in edge_attachment_satisfied.values() if not ok
    )
    logger.info(
        "composition: %d constraints, %d unsatisfied edges, %d overlap conflicts, "
        "%d THT violations",
        len(all_constraints),
        unsatisfied_edges,
        len(same_side_overlap_conflicts),
        len(tht_keepout_violations),
    )

    parent_local_keep_in_rects: list[tuple[Point, Point]] = []
    for constraint in derived.parent_local_constraints:
        comp = parent_local_solved.get(constraint.ref)
        if comp is None:
            continue
        bbox_min, bbox_max = _component_geometry_bbox(comp)
        keep_in = (
            Point(
                bbox_min.x - constraint.inward_keep_in_mm,
                bbox_min.y - constraint.inward_keep_in_mm,
            ),
            Point(
                bbox_max.x + constraint.inward_keep_in_mm,
                bbox_max.y + constraint.inward_keep_in_mm,
            ),
        )
        parent_local_keep_in_rects.append(keep_in)

    outline_w = exact_outline[1].x - exact_outline[0].x
    outline_h = exact_outline[1].y - exact_outline[0].y
    packing_metadata: dict[str, Any] = {
        "strategy": "unified_solver",
        "board_width_mm": round(outline_w, 2),
        "board_height_mm": round(outline_h, 2),
    }

    project_dir = ""
    if loaded_artifacts:
        try:
            project_dir = str(
                Path(loaded_artifacts[0].artifact_dir).resolve().parents[2]
            )
        except IndexError:
            project_dir = ""

    state = ParentCompositionState(
        project_dir=project_dir,
        spacing_mm=spacing_mm,
        entries=entries,
        bounding_box=composition.board_state.board_outline,
        parent_sheet_name=composition.hierarchy_state.subcircuit.id.sheet_name,
        parent_instance_path=composition.hierarchy_state.subcircuit.id.instance_path,
        component_count=composition.component_count,
        trace_count=composition.trace_count,
        via_count=composition.via_count,
        interconnect_net_count=len(composition.hierarchy_state.interconnect_nets),
        inferred_interconnect_net_count=len(composition.inferred_interconnect_nets),
        preserved_child_trace_count=composition.trace_count,
        preserved_child_via_count=composition.via_count,
        expected_preserved_child_trace_count=composition.trace_count,
        expected_preserved_child_via_count=composition.via_count,
        routed_total_trace_count=composition.trace_count,
        routed_total_via_count=composition.via_count,
        added_parent_trace_count=0,
        added_parent_via_count=0,
        packing_metadata=packing_metadata,
        geometry_validation=validation_data,
        score_total=composition.score.total if composition.score else 0.0,
        score_breakdown=dict(composition.score.breakdown) if composition.score else {},
        score_notes=list(composition.score.notes) if composition.score else [],
        composition_notes=list(composition.notes),
        copper_manifest=copper_manifest,
        composition=composition,
        parent_local_keep_in_rects=list(parent_local_keep_in_rects),
        edge_constrained_refs=frozenset(
            c.ref for c in all_constraints if c.target in ("edge", "corner")
        ),
    )
    state.phase_timings.update(solver_phase_timings)
    return state, transformed_payloads


def _save_composition_snapshot(
    output_path: Path,
    state: ParentCompositionState,
    transformed_payloads: list[dict[str, Any]],
) -> str:
    """Write a composition snapshot JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composition_payload = {
        "summary": (
            f"{state.parent_sheet_name} "
            f"[{state.parent_instance_path}] "
            f"children={len(state.entries)} "
            f"components={state.component_count} "
            f"traces={state.trace_count} "
            f"vias={state.via_count} "
            f"interconnects={state.interconnect_net_count} "
            f"score={state.score_total:.1f} "
            f"size={state.width_mm:.1f}x{state.height_mm:.1f}mm"
        ),
        "debug": {
            "parent": {
                "sheet_name": state.parent_sheet_name,
                "instance_path": state.parent_instance_path,
            },
            "child_count": len(state.entries),
            "component_count": state.component_count,
            "trace_count": state.trace_count,
            "via_count": state.via_count,
            "interconnect_net_count": state.interconnect_net_count,
            "inferred_interconnect_net_count": state.inferred_interconnect_net_count,
            "score": {
                "total": state.score_total,
                "breakdown": dict(state.score_breakdown),
                "notes": list(state.score_notes),
            },
            "notes": list(state.composition_notes),
            "board_outline": state.to_dict()["bounding_box"],
            "packing_metadata": dict(state.packing_metadata),
            "geometry_validation": dict(state.geometry_validation),
        },
    }
    payload = {
        "composition": composition_payload,
        "state": state.to_dict(),
        "artifacts": transformed_payloads,
    }
    # Atomic write: autoexperiment reads this output via
    # _read_composer_quality_score to extract the round score; a torn
    # mid-write read would crash the score-extraction path and reject
    # an otherwise-valid round.
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_output.replace(output_path)
    return str(output_path)


def _print_human_summary(
    loaded_artifacts,
    state: ParentCompositionState,
    transformed_payloads: list[dict[str, Any]],
    output_path: str | None,
) -> None:
    print("=== Subcircuit Composition ===")
    print(f"artifacts              : {len(loaded_artifacts)}")
    print(f"spacing_mm             : {state.spacing_mm:.2f}")
    print(f"parent                 : {state.parent_sheet_name}")
    print(f"parent_instance_path   : {state.parent_instance_path}")
    print(f"composition_mm         : {state.width_mm:.2f} x {state.height_mm:.2f}")
    print(f"components             : {state.component_count}")
    print(f"traces                 : {state.trace_count}")
    print(f"vias                   : {state.via_count}")
    print(f"interconnect_nets      : {state.interconnect_net_count}")
    print(f"inferred_interconnects : {state.inferred_interconnect_net_count}")
    print(f"score_total            : {state.score_total:.2f}")
    if output_path:
        print(f"output_json            : {output_path}")
    print()

    for artifact, transformed in zip(loaded_artifacts, transformed_payloads):
        print(f"- {artifact_summary(artifact)}")
        print(f"  artifact_dir : {artifact.artifact_dir}")
        print(f"  transformed  : {transformed['summary']}")
        print()

    if state.score_breakdown:
        print("score_breakdown:")
        for key, value in sorted(state.score_breakdown.items()):
            print(f"  - {key}: {value:.2f}")

    # Copper accounting summary
    if state.preserved_child_trace_count or state.added_parent_trace_count:
        print()
        print("copper_accounting:")
        print(
            f"  child_traces         : {state.preserved_child_trace_count}"
            f" / {state.expected_preserved_child_trace_count} preserved"
        )
        print(
            f"  child_vias           : {state.preserved_child_via_count}"
            f" / {state.expected_preserved_child_via_count} preserved"
        )
        print(f"  parent_traces        : +{state.added_parent_trace_count} new")
        print(f"  parent_vias          : +{state.added_parent_via_count} new")
        print(
            f"  total_routed         : {state.routed_total_trace_count} traces,"
            f" {state.routed_total_via_count} vias"
        )
        print()

    if state.score_notes:
        print("score_notes:")
        for note in state.score_notes:
            print(f"  - {note}")
        print()

    if state.composition_notes:
        print("composition_notes:")
        for note in state.composition_notes:
            print(f"  - {note}")
        print()


def _json_payload(
    loaded_artifacts,
    state: ParentCompositionState,
    transformed_payloads: list[dict[str, Any]],
    output_path: str | None,
) -> dict[str, Any]:
    composition_payload = {
        "summary": (
            f"{state.parent_sheet_name} "
            f"[{state.parent_instance_path}] "
            f"children={len(state.entries)} "
            f"components={state.component_count} "
            f"traces={state.trace_count} "
            f"vias={state.via_count} "
            f"interconnects={state.interconnect_net_count} "
            f"score={state.score_total:.1f} "
            f"size={state.width_mm:.1f}x{state.height_mm:.1f}mm"
        ),
        "debug": {
            "parent": {
                "sheet_name": state.parent_sheet_name,
                "instance_path": state.parent_instance_path,
            },
            "child_count": len(state.entries),
            "component_count": state.component_count,
            "trace_count": state.trace_count,
            "via_count": state.via_count,
            "interconnect_net_count": state.interconnect_net_count,
            "inferred_interconnect_net_count": state.inferred_interconnect_net_count,
            "score": {
                "total": state.score_total,
                "breakdown": dict(state.score_breakdown),
                "notes": list(state.score_notes),
            },
            "notes": list(state.composition_notes),
            "board_outline": state.to_dict()["bounding_box"],
        },
    }
    return {
        "artifact_count": len(loaded_artifacts),
        "composition": composition_payload,
        "state": state.to_dict(),
        "output_json": output_path,
        "artifacts": transformed_payloads,
    }


# ---------------------------------------------------------------------------
# Parent board stamping, routing, validation, rendering, and artifact persistence
# ---------------------------------------------------------------------------


def _compact_routed_validation(validation: dict[str, Any]) -> dict[str, Any]:
    """Strip the ~100 KB DRC report_text, keep the diagnostic signal.

    Preserved fields: accepted, rejection_reasons, obviously_illegal_routed_geometry,
    DRC violation counts, first N violation excerpts (~1 KB), track/anchor
    summaries. Callers that need the full report can still run
    `validate_routed_board` themselves.
    """
    if not isinstance(validation, dict):
        return {}
    drc = validation.get("drc", {}) or {}
    report_text = drc.get("report_text", "") or ""
    drc_slim: dict[str, Any] = {
        k: v for k, v in drc.items() if k != "report_text"
    }
    if report_text:
        # Retain only the first few violation blocks for post-hoc analysis.
        # A KiCad DRC report is line-oriented; "[category]" lines introduce
        # each violation. Take the first 800 chars, which is usually enough
        # to identify the dominant failure mode without bloating the JSON.
        drc_slim["report_excerpt"] = report_text[:800]
    out = {
        "accepted": bool(validation.get("accepted", False)),
        "rejection_reasons": list(validation.get("rejection_reasons", []) or []),
        "obviously_illegal_routed_geometry": bool(
            validation.get("obviously_illegal_routed_geometry", False)
        ),
        "malformed_board_geometry": bool(
            validation.get("malformed_board_geometry", False)
        ),
        "track_summary": dict(validation.get("track_summary", {}) or {}),
        "anchor_summary": {
            k: v
            for k, v in (validation.get("anchor_summary", {}) or {}).items()
            if k
            in {
                "expected_count",
                "actual_count",
                "required_count",
                "all_required_present",
            }
        },
        "drc": drc_slim,
    }
    for k in (
        "footprint_internal_clearance_count",
        "footprint_internal_copper_edge_count",
    ):
        if k in validation:
            out[k] = validation[k]
    return out


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
            component_outside = (
                body_tl.x < min_x
                or body_tl.y < min_y
                or body_br.x > max_x
                or body_br.y > max_y
            )

        # Pad check: pad COPPER (not just the center) must be inside the
        # board outline. No edge-constrained exemption -- pad copper that
        # crosses Edge.Cuts is unfabricable.
        pad_outside_count = 0
        for pad in comp.pads:
            pad_tl, pad_br = pad.bbox()
            if (
                pad_tl.x < min_x
                or pad_br.x > max_x
                or pad_tl.y < min_y
                or pad_br.y > max_y
            ):
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
        if (
            trace.start.x < min_x
            or trace.start.x > max_x
            or trace.start.y < min_y
            or trace.start.y > max_y
            or trace.end.x < min_x
            or trace.end.x > max_x
            or trace.end.y < min_y
            or trace.end.y > max_y
        ):
            outside_traces += 1

    for via in composition.board_state.vias or []:
        geometry_union_min_x = min(geometry_union_min_x, via.pos.x)
        geometry_union_min_y = min(geometry_union_min_y, via.pos.y)
        geometry_union_max_x = max(geometry_union_max_x, via.pos.x)
        geometry_union_max_y = max(geometry_union_max_y, via.pos.y)
        if (
            via.pos.x < min_x
            or via.pos.x > max_x
            or via.pos.y < min_y
            or via.pos.y > max_y
        ):
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
    """Render standard parent board preview images."""
    script_dir = Path(__file__).resolve().parent
    render_script = script_dir / "render_pcb.py"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(render_script),
        str(pcb_path),
        "--output-dir",
        str(output_dir),
        "--views",
        "front_all",
        "back_all",
        "copper_both",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Parent render failed for "
            f"{pcb_path}: {result.stderr.strip() or result.stdout.strip()}"
        )

    rendered = {
        "front_all": str(output_dir / "front_all.png"),
        "back_all": str(output_dir / "back_all.png"),
        "copper_both": str(output_dir / "copper_both.png"),
    }
    return {name: path for name, path in rendered.items() if Path(path).exists()}


def _stamp_parent_board(
    state: ParentCompositionState,
    pcb_path: Path,
    project_dir: Path,
    cfg: dict[str, Any],
    output_pcb_path: Path | None = None,
) -> Path:
    """Stamp the parent composition onto a real .kicad_pcb file.

    Uses a subprocess to run pcbnew operations so the main process does not
    need pcbnew installed.  The subprocess:
    1. Loads the copied board
    2. Moves footprints to their composed positions
    3. Clears existing tracks/zones
    4. Recreates traces/vias from the merged child copper
    5. Rebuilds connectivity and saves

    If ``output_pcb_path`` is provided, the stamped board is written there
    instead of the canonical ``<artifact_dir>/parent_pre_freerouting.kicad_pcb``.
    The candidate-search loop uses this to stamp each trial to a distinct
    file under ``<artifact_dir>/_search/``.

    Returns the stamped board path.
    """
    import json as _json
    import os
    import tempfile

    from kicraft.autoplacer.brain.subcircuit_artifacts import slugify_subcircuit_id
    from kicraft.autoplacer.brain.types import Layer
    from kicraft.autoplacer.freerouting_runner import _run_pcbnew_script_file

    composition = state.composition
    if composition is None:
        raise RuntimeError("ParentCompositionState has no composition object")

    parent_id = composition.hierarchy_state.subcircuit.id
    slug = slugify_subcircuit_id(parent_id)
    artifact_dir = project_dir / ".experiments" / "subcircuits" / slug
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if output_pcb_path is not None:
        output_pcb = Path(output_pcb_path)
        output_pcb.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_pcb = artifact_dir / "parent_pre_freerouting.kicad_pcb"
    shutil.copy2(str(pcb_path), str(output_pcb))

    # Serialize board state for the subprocess
    board_state = composition.board_state
    components_json = []
    for ref, comp in (board_state.components or {}).items():
        components_json.append(
            {
                "ref": ref,
                "x": comp.pos.x,
                "y": comp.pos.y,
                "rotation": comp.rotation,
                "layer": 0 if comp.layer == Layer.FRONT else 1,
            }
        )

    traces_json = []
    for trace in board_state.traces or []:
        traces_json.append(
            {
                "start_x": trace.start.x,
                "start_y": trace.start.y,
                "end_x": trace.end.x,
                "end_y": trace.end.y,
                "width": trace.width_mm,
                "layer": "F.Cu" if trace.layer == Layer.FRONT else "B.Cu",
                "net_name": trace.net or "",
            }
        )

    vias_json = []
    for via in board_state.vias or []:
        vias_json.append(
            {
                "x": via.pos.x,
                "y": via.pos.y,
                "size": via.size_mm,
                "drill": via.drill_mm,
                "net_name": via.net or "",
            }
        )

    silkscreen_json = []
    for elem in board_state.silkscreen or []:
        if elem.kind == "poly":
            silkscreen_json.append({
                "kind": "poly",
                "layer": elem.layer,
                "points": [{"x": p.x, "y": p.y} for p in elem.points],
                "stroke_width": elem.stroke_width,
            })
        elif elem.kind == "text":
            silkscreen_json.append({
                "kind": "text",
                "layer": elem.layer,
                "text": elem.text,
                "pos": {"x": elem.pos.x, "y": elem.pos.y},
                "font_height": elem.font_height,
                "font_width": elem.font_width,
                "font_thickness": elem.font_thickness,
            })

    # Compute the board outline from the composition
    outline = board_state.board_outline
    outline_data = None
    if outline and len(outline) >= 2:
        outline_data = {
            "tl_x": outline[0].x,
            "tl_y": outline[0].y,
            "br_x": outline[1].x,
            "br_y": outline[1].y,
        }

    geometry_validation = _validate_parent_geometry(state)
    if not geometry_validation.get("accepted", False):
        # Don't raise -- the caller wants to stamp + render even on
        # geometry rejection so the user has a diagnostic image showing
        # where components ended up. The outer code's geometry_accepted
        # flag still gates routing, so we won't attempt to route an
        # off-board layout. Just surface a warning so the failure is
        # visible in the log.
        print(
            "warning: parent composition geometry is invalid before "
            "stamping (continuing to stamp for diagnostic render): "
            f"outside_components={geometry_validation.get('outside_component_count', 0)} "
            f"outside_pads={geometry_validation.get('outside_pad_count', 0)} "
            f"outside_traces={geometry_validation.get('outside_trace_count', 0)} "
            f"outside_vias={geometry_validation.get('outside_via_count', 0)}",
            file=sys.stderr,
        )

    keepout_json = [
        {
            "tl_x": rect[0].x,
            "tl_y": rect[0].y,
            "br_x": rect[1].x,
            "br_y": rect[1].y,
        }
        for rect in (state.parent_local_keep_in_rects or [])
    ]

    payload = {
        "pcb_path": str(output_pcb),
        "output_path": str(output_pcb),
        "components": components_json,
        "traces": traces_json,
        "vias": vias_json,
        "silkscreen": silkscreen_json,
        "outline": outline_data,
        "keepouts": keepout_json,
    }

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="stamp_parent_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            _json.dump(payload, f)

        _run_pcbnew_script_file(_PARENT_STAMP_SCRIPT_PATH, tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    print(f"Parent board stamped to {output_pcb} (subprocess)")
    return output_pcb



# ---------------------------------------------------------------------------
# Self-contained pcbnew script executed in a subprocess by
# _stamp_parent_board(). Lifted to its own file (_parent_stamp_subprocess.py)
# so import-time errors fire when the file is parsed and so linters / IDEs
# see the pcbnew API calls.
# ---------------------------------------------------------------------------
_PARENT_STAMP_SCRIPT_PATH = str(
    Path(__file__).parent / "_parent_stamp_subprocess.py"
)


@dataclass(slots=True)
class CandidateRecord:
    """One trial in the layout search loop."""

    seed: int
    shorts: int
    score: float
    place_solve_ms: float
    stamp_ms: float
    stamp_drc_ms: float
    accepted: bool
    pcb_path: str
    bbox_h_mm: float = 0.0
    bbox_w_mm: float = 0.0
    outline_h_mm: float = 0.0
    outline_w_mm: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    geometry_accepted: bool = False
    outside_component_count: int = 0
    outside_pad_count: int = 0
    phase_timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "shorts": self.shorts,
            "score": self.score,
            "place_solve_ms": self.place_solve_ms,
            "stamp_ms": self.stamp_ms,
            "stamp_drc_ms": self.stamp_drc_ms,
            "accepted": self.accepted,
            "pcb_path": self.pcb_path,
            "bbox_h_mm": self.bbox_h_mm,
            "bbox_w_mm": self.bbox_w_mm,
            "outline_h_mm": self.outline_h_mm,
            "outline_w_mm": self.outline_w_mm,
            "breakdown": dict(self.breakdown),
            "geometry_accepted": self.geometry_accepted,
            "outside_component_count": self.outside_component_count,
            "outside_pad_count": self.outside_pad_count,
            "phase_timings": dict(self.phase_timings),
        }


@dataclass(slots=True)
class SearchResult:
    """Outcome of _search_best_layout: winner state + per-candidate records."""

    winner_idx: int
    winner_seed: int
    winner_state: ParentCompositionState
    winner_payloads: list[dict[str, Any]]
    winner_pcb_path: Path
    candidates: list[CandidateRecord]
    total_search_ms: float


def _search_best_layout(
    loaded_artifacts,
    *,
    spacing_mm: float,
    rotation_step_deg: float,
    parent_definition,
    pcb_path: Path | None,
    project_dir: Path,
    cfg: dict[str, Any] | None,
    base_seed: int,
    k: int = 8,
    time_budget_s: float = 60.0,
    manual_layout: ManualLayout | None = None,
) -> SearchResult:
    """Generate K placement candidates, hard-prefer shorts==0, return the winner.

    Each iteration calls ``_compose_artifacts`` with the project cfg
    UNCHANGED -- one solver pipeline, full fidelity per candidate. K is a
    diversity knob across seeds, not a quality mode.

    Winner selection is lexicographic: shorts ascending first, composite
    score descending second. If no candidate has shorts==0, the function
    raises ``RuntimeError`` -- the round fails loudly. There is no
    "best of bad options" fallback: a stamped board with shorts is not
    a useful artifact, and silently emitting one masks solver bugs that
    deserve attention.
    """
    from kicraft.autoplacer.brain.graph import total_ratsnest_length
    from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
    from kicraft.autoplacer.freerouting_runner import _run_kicad_cli_drc

    base_cfg = dict(cfg or {})
    base_parent_placement = dict(base_cfg.get("parent_placement", {}))
    _search_cfg = dict(base_parent_placement.get("candidate_search", {}))

    # Manual mode: collapse the candidate search to a single trial. The
    # user has chosen the layout, so seed-aspect sweeping is meaningless
    # and stamp_drc is informational rather than gating.
    if manual_layout is not None:
        k = 1

    # Resolve artifact_dir for candidate stamping. Mirrors the path
    # _stamp_parent_board() builds so downstream code finds the winner
    # under the canonical slug.
    from kicraft.autoplacer.brain.subcircuit_artifacts import slugify_subcircuit_id

    if parent_definition is not None and getattr(parent_definition, "id", None):
        slug = slugify_subcircuit_id(parent_definition.id)
    else:
        slug = "parent"
    artifact_dir = project_dir / ".experiments" / "subcircuits" / slug
    search_dir = artifact_dir / "_search"
    search_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[CandidateRecord] = []
    cand_states: list[ParentCompositionState] = []
    cand_payloads: list[list[dict[str, Any]]] = []
    cand_pcb_paths: list[Path] = []

    t_search_start = time.perf_counter()
    for i in range(k):
        elapsed_ms = (time.perf_counter() - t_search_start) * 1000.0
        if elapsed_ms > time_budget_s * 1000.0:
            print(
                f"[candidate-search] time budget {time_budget_s:.0f}s exhausted "
                f"after {i} candidate(s)",
                file=sys.stderr,
            )
            break
        seed_i = base_seed + i
        # Sweep the seed aspect linearly across the K candidates so each
        # parents-only invocation explores a spread of horizontal-strip
        # to vertical-stack shapes instead of K placements all starting
        # from the same square seed.  i=0 is the most horizontal
        # (aspect=0.6, wide+short), i=k-1 is the most vertical
        # (aspect=1.7, tall+narrow).  K=1 stays at aspect=1.0 (square)
        # for backward compatibility with single-candidate searches.
        # area_overhead stays fixed -- it is config-tunable per round
        # via parent_seed_area_overhead so users can dial overall seed
        # tightness without forking the per-candidate aspect sweep.
        if k > 1:
            seed_aspect_i = 0.6 + (i / (k - 1)) * 1.1
        else:
            seed_aspect_i = 1.0
        seed_overhead_i = float(base_cfg.get("parent_seed_area_overhead", 2.5))
        t_solve = time.perf_counter()
        state, payloads = _compose_artifacts(
            loaded_artifacts,
            spacing_mm=spacing_mm,
            rotation_step_deg=rotation_step_deg,
            parent_definition=parent_definition,
            pcb_path=pcb_path,
            cfg=base_cfg,
            seed=seed_i,
            seed_area_overhead=seed_overhead_i,
            seed_aspect_target=seed_aspect_i,
            manual_layout=manual_layout,
        )
        place_solve_ms = (time.perf_counter() - t_solve) * 1000.0
        state.phase_timings["place_solve_ms"] = place_solve_ms

        if pcb_path is None:
            # No source PCB → no stamping, no DRC. Search degrades to
            # placement-quality ranking (composite score only); shorts
            # is unknowable so all candidates are treated as
            # accepted=True. This keeps a single search code path
            # for the rare interactive `compose-subcircuits --output`
            # case without --pcb.
            stamp_ms = 0.0
            stamp_drc_ms = 0.0
            shorts = 0
            stamped = Path("")
        else:
            cand_pcb = search_dir / f"cand_{i:02d}.kicad_pcb"
            t_stamp = time.perf_counter()
            stamped = _stamp_parent_board(
                state, pcb_path, project_dir, base_cfg,
                output_pcb_path=cand_pcb,
            )
            stamp_ms = (time.perf_counter() - t_stamp) * 1000.0
            state.phase_timings["stamp_ms"] = stamp_ms

            t_drc = time.perf_counter()
            drc = _run_kicad_cli_drc(str(stamped), timeout_s=30) or {}
            stamp_drc_ms = (time.perf_counter() - t_drc) * 1000.0
            state.phase_timings["stamp_drc_ms"] = stamp_drc_ms
            state.stamp_drc = dict(drc)
            shorts = int(drc.get("shorts", 0) or 0)

        board_state = state.composition.board_state if state.composition else None
        if board_state is None:
            raise RuntimeError(
                f"candidate-search cand={i} seed={seed_i}: composition has no "
                "board_state; cannot compute composite score"
            )
        scorer = PlacementScorer(board_state, base_parent_placement)
        opp_side = float(scorer._score_block_opposite_side())
        overlap = float(scorer._score_courtyard_overlap())
        ratsnest_mm = float(total_ratsnest_length(board_state))
        net_dist = max(0.0, 100.0 - ratsnest_mm * 0.1)
        # Compactness penalty: candidates with sprawling placements grow
        # the parent outline (sometimes 200+ mm tall) and break geometry
        # validation. The previous composite saturated to 4.37 across
        # most candidates (opp_side=0, overlap=12.5, net_dist=0),
        # leaving K=4 selection essentially random. Add a direct bbox
        # term so smaller boards beat bigger ones when the rest is
        # tied. Uses _score_bbox_packing (0..100, higher = denser).
        bbox_packing = float(scorer._score_bbox_packing())
        composite = (
            0.30 * opp_side
            + 0.25 * overlap
            + 0.15 * net_dist
            + 0.30 * bbox_packing
        )

        # Diagnostic capture: unlocked-component cluster AABB (the
        # spread the solver actually controls) and the auto-grown
        # outline are NOT the same. Locked components (corner-pinned
        # mounting holes, edge-pinned connectors) sit at fixed
        # template positions and would constant-pad the cluster
        # measurement -- including them here would compare the
        # outline cap against a frame the solver can't shrink, so a
        # template board larger than the cap would always fail
        # regardless of how compact the unlocked placement was.
        # Mirrors _record_placed_extent so this measurement is
        # apples-to-apples with the per-phase solve_*_placed_*_mm
        # extents persisted alongside.
        comps = [
            c for c in board_state.components.values() if not c.locked
        ]
        if comps:
            phys = [c.physical_bbox() for c in comps]
            placed_w_mm = max(b[1].x for b in phys) - min(b[0].x for b in phys)
            placed_h_mm = max(b[1].y for b in phys) - min(b[0].y for b in phys)
        else:
            placed_w_mm = 0.0
            placed_h_mm = 0.0
        outline_tl, outline_br = board_state.board_outline
        outline_w_mm = max(0.0, outline_br.x - outline_tl.x)
        outline_h_mm = max(0.0, outline_br.y - outline_tl.y)
        # state.geometry_validation is populated inside _stamp_parent_board
        # via _validate_parent_geometry. When pcb_path is None the stamp
        # path is skipped, leaving geometry_validation = {} -- treat that
        # as accepted=True (no stamping happened, so nothing to reject).
        gv = state.geometry_validation or {}
        geometry_accepted = bool(gv.get("accepted", True))
        outside_component_count = int(gv.get("outside_component_count", 0) or 0)
        outside_pad_count = int(gv.get("outside_pad_count", 0) or 0)

        # Hard gate: shorts==0 only. Stamped electrical shorts are an
        # objective truth (DRC counts them), so a candidate with shorts
        # cannot win regardless of its routing prospects. Geometry
        # violations (components/pads outside the auto-grown outline)
        # are RECORDED on the CandidateRecord (geometry_accepted,
        # outside_component_count, outside_pad_count) but no longer
        # short-circuit the picker -- they're a guess at unfabricability
        # that prevents FreeRouting from running and starves the search
        # of real signal. Let routing run; a layout that violates
        # geometry will produce a routed PNG showing exactly where the
        # problem is, which is more actionable than "round aborted".
        #
        # Manual mode bypass: the user explicitly chose this placement.
        # stamp_drc is informational (surfaced in the GUI), not a gate;
        # auto-rejecting on shorts would force the user to redo the
        # whole drag-route cycle when often a single track tweak in the
        # routed result is enough.
        if manual_layout is not None:
            accepted = True
        else:
            accepted = shorts == 0

        rec = CandidateRecord(
            seed=seed_i,
            shorts=shorts,
            score=composite,
            place_solve_ms=place_solve_ms,
            stamp_ms=stamp_ms,
            stamp_drc_ms=stamp_drc_ms,
            accepted=accepted,
            pcb_path=str(stamped),
            bbox_h_mm=placed_h_mm,
            bbox_w_mm=placed_w_mm,
            outline_h_mm=outline_h_mm,
            outline_w_mm=outline_w_mm,
            breakdown={
                "opp_side": opp_side,
                "overlap": overlap,
                "net_dist": net_dist,
                "bbox_packing": bbox_packing,
            },
            geometry_accepted=geometry_accepted,
            outside_component_count=outside_component_count,
            outside_pad_count=outside_pad_count,
            phase_timings=dict(state.phase_timings),
        )
        candidates.append(rec)
        cand_states.append(state)
        cand_payloads.append(payloads)
        cand_pcb_paths.append(stamped)
        print(
            f"[candidate-search] cand={i} seed={seed_i} "
            f"aspect={seed_aspect_i:.2f} "
            f"shorts={shorts} score={composite:.1f} "
            f"bh={placed_h_mm:.1f}mm bw={placed_w_mm:.1f}mm "
            f"oh={outline_h_mm:.1f}mm geom_ok={geometry_accepted} "
            f"place={place_solve_ms / 1000:.1f}s drc={stamp_drc_ms / 1000:.1f}s"
        )

    total_search_ms = (time.perf_counter() - t_search_start) * 1000.0

    if not candidates:
        # K iterations exited the budget loop with zero successful
        # appends. Either k <= 0 was passed or the time_budget_s was so
        # tight that the very first compose blew it. Either way it's a
        # configuration problem, not a placement bug -- surface it.
        raise RuntimeError(
            f"candidate-search ran zero candidates in K={k}, "
            f"time_budget_s={time_budget_s:.0f}; check candidate_search cfg"
        )

    accepted_recs = [c for c in candidates if c.accepted]
    if not accepted_recs:
        # All K candidates failed at least one hard gate (shorts > 0
        # or geometry_accepted False). Fail loudly -- per "no
        # fallbacks", a masked failure here would let real solver
        # bugs ship under a green checkmark.
        #
        # Before raising, persist per-candidate diagnostics. Each record
        # already carries the per-phase solve_*_placed_{w,h}_mm extents
        # captured by _record_placed_extent(); without writing them to
        # disk now, the round abort throws this data away and the next
        # operator has nothing to triage from. The success path scrubs
        # search_dir at line ~2703; the abort path leaves cand_*.kicad_pcb
        # AND now this JSON for inspection.
        rejected_payload = {
            "k": k,
            "tried": len(candidates),
            "total_search_ms": total_search_ms,
            "candidates": [c.to_dict() for c in candidates],
        }
        try:
            (search_dir / "_rejected_candidates.json").write_text(
                json.dumps(rejected_payload, indent=2, default=float)
            )
        except OSError as exc:
            print(
                f"[candidate-search] failed to write rejected-candidate "
                f"diagnostics to {search_dir}: {exc}",
                file=sys.stderr,
            )
        rejection_summary = ", ".join(
            (
                f"seed={c.seed}:"
                f"shorts={c.shorts},"
                f"geom_ok={c.geometry_accepted},"
                f"bh={c.bbox_h_mm:.1f}mm,"
                f"bw={c.bbox_w_mm:.1f}mm"
            )
            for c in candidates
        )
        raise RuntimeError(
            f"candidate-search produced no acceptable placement in K={len(candidates)} "
            f"(per-candidate: {rejection_summary}). "
            f"Round aborted; investigate solver before re-running. "
            f"Per-candidate phase timings written to "
            f"{search_dir / '_rejected_candidates.json'}."
        )
    winner_rec = max(accepted_recs, key=lambda c: c.score)

    winner_idx = candidates.index(winner_rec)
    winner_state = cand_states[winner_idx]
    winner_payloads = cand_payloads[winner_idx]
    winner_pcb = cand_pcb_paths[winner_idx]

    winner_state.candidate_search = {
        "k": k,
        "tried": len(candidates),
        "accepted": len(accepted_recs),
        "rejected_drc": len(candidates) - len(accepted_recs),
        "best_index": winner_idx,
        "best_seed": winner_rec.seed,
        "total_search_ms": total_search_ms,
        "candidates": [c.to_dict() for c in candidates],
    }
    winner_state.phase_timings["candidate_search_ms"] = total_search_ms

    # Drop the per-trial stamped boards now that the winner is selected.
    # The search dir leaks ~16 MB/round of intermediates and the winner
    # is captured in winner_state + winner_pcb_path.
    try:
        shutil.rmtree(search_dir)
    except OSError:
        pass

    return SearchResult(
        winner_idx=winner_idx,
        winner_seed=winner_rec.seed,
        winner_state=winner_state,
        winner_payloads=winner_payloads,
        winner_pcb_path=winner_pcb,
        candidates=candidates,
        total_search_ms=total_search_ms,
    )


def _route_parent_board(
    stamped_pcb: Path,
    state: ParentCompositionState,
    project_dir: Path,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Route parent interconnects via FreeRouting, then import and validate.

    1. Resolve output path for the routed board
    2. Run FreeRouting (preserving stamped child copper in the DSN export)
    3. Import routed copper from the result
    4. Validate the routed board
    5. Return a result dict
    """
    from kicraft.autoplacer.freerouting_runner import (
        import_routed_copper,
        route_with_freerouting,
        validate_routed_board,
    )

    composition = state.composition
    if composition is None:
        raise RuntimeError("ParentCompositionState has no composition object")

    routed_pcb = stamped_pcb.parent / "parent_routed.kicad_pcb"

    jar_path = cfg.get("freerouting_jar", "")
    if not jar_path:
        raise RuntimeError(
            "No FreeRouting JAR path configured; pass --jar or set "
            "freerouting_jar in project config"
        )

    # Build a routing config that preserves child copper already stamped
    # onto the board.  FreeRouting's DSN export will see those traces as
    # wires so it only routes the remaining unconnected (interconnect) nets.
    route_cfg = dict(cfg)
    route_cfg["freerouting_preserve_existing_copper"] = True
    route_cfg["freerouting_clear_existing_copper"] = False
    route_cfg["freerouting_clear_zones"] = False

    try:
        freerouting_stats = route_with_freerouting(
            kicad_pcb_path=str(stamped_pcb),
            output_path=str(routed_pcb),
            jar_path=jar_path,
            config=route_cfg,
        )
    except Exception as exc:
        return {
            "failed": True,
            "error": str(exc),
            "routed_board_path": str(routed_pcb),
            "_trace_segments": [],
            "_via_objects": [],
            "validation": {},
            "freerouting_stats": {},
        }

    # Import all copper from the routed board (child + new parent traces)
    copper = import_routed_copper(str(routed_pcb))

    # Root parent has no interface anchors -- skip anchor validation.
    # Anchor completeness is a leaf-level gate, not a parent-level gate.
    validation = validate_routed_board(
        str(routed_pcb),
        cfg=cfg,
        expected_anchor_names=[],
        actual_anchor_names=[],
        required_anchor_names=[],
    )

    return {
        "failed": False,
        "routed_board_path": str(routed_pcb),
        "_trace_segments": copper.get("traces", []),
        "_via_objects": copper.get("vias", []),
        "validation": validation,
        "freerouting_stats": freerouting_stats,
    }


def _persist_parent_artifact(
    state: ParentCompositionState,
    routing_result: dict[str, Any],
    project_dir: Path,
    cfg: dict[str, Any],
) -> str:
    """Persist a parent-level solved layout artifact.

    1. Build a SubCircuitLayout from the composition's board_state
       with routed copper from the routing result
    2. Build and save the solved layout artifact payload
    3. Save metadata and debug payloads
    4. Return the artifact directory path
    """
    from kicraft.autoplacer.brain.subcircuit_artifacts import (
        build_solved_layout_artifact,
        resolve_artifact_paths,
        save_solved_layout_artifact,
    )
    from kicraft.autoplacer.brain.types import SubCircuitLayout

    composition = state.composition
    if composition is None:
        raise RuntimeError("ParentCompositionState has no composition object")

    parent_id = composition.hierarchy_state.subcircuit.id
    parent_def = composition.hierarchy_state.subcircuit

    # Use routed copper (all traces: child + new parent interconnect)
    all_traces = routing_result.get("_trace_segments", [])
    all_vias = routing_result.get("_via_objects", [])

    # Fall back to composition board_state copper if routing returned nothing
    if not all_traces and not all_vias:
        all_traces = list(composition.board_state.traces)
        all_vias = list(composition.board_state.vias)

    # Compute bounding box from the composition's board outline
    tl, br = composition.board_state.board_outline
    width = max(0.0, br.x - tl.x)
    height = max(0.0, br.y - tl.y)

    layout = SubCircuitLayout(
        subcircuit_id=parent_id,
        components=dict(composition.board_state.components),
        traces=list(all_traces),
        vias=list(all_vias),
        bounding_box=(width, height),
        ports=list(parent_def.ports),
        interface_anchors=[],
        score=state.score_total,
        frozen=True,
    )

    # Build notes for the artifact
    # Use copper manifest for accurate child-only counts (state.trace_count
    # includes parent interconnect traces, which inflates the expectation).
    if state.copper_manifest is not None:
        expected_child_trace_count = state.copper_manifest.total_child_traces
        expected_child_via_count = state.copper_manifest.total_child_vias
    else:
        expected_child_trace_count = int(state.trace_count)
        expected_child_via_count = int(state.via_count)
    routed_total_trace_count = len(all_traces)
    routed_total_via_count = len(all_vias)

    # -- Real copper accounting via fingerprint matching --
    if state.copper_manifest is not None:
        copper_verification = verify_copper_preservation(
            manifest=state.copper_manifest,
            post_route_traces=all_traces,
            post_route_vias=all_vias,
        )
        preserved_child_trace_count = copper_verification["matched_child_traces"]
        preserved_child_via_count = copper_verification["matched_child_vias"]
        added_parent_trace_count = copper_verification["new_route_traces"]
        added_parent_via_count = copper_verification["new_route_vias"]
    else:
        # Fallback to count-based estimation when manifest unavailable
        copper_verification = None
        preserved_child_trace_count = min(
            expected_child_trace_count, routed_total_trace_count
        )
        preserved_child_via_count = min(
            expected_child_via_count, routed_total_via_count
        )
        added_parent_trace_count = max(
            0, routed_total_trace_count - preserved_child_trace_count
        )
        added_parent_via_count = max(
            0, routed_total_via_count - preserved_child_via_count
        )

    state.expected_preserved_child_trace_count = expected_child_trace_count
    state.expected_preserved_child_via_count = expected_child_via_count
    state.preserved_child_trace_count = preserved_child_trace_count
    state.preserved_child_via_count = preserved_child_via_count
    state.routed_total_trace_count = routed_total_trace_count
    state.routed_total_via_count = routed_total_via_count
    state.added_parent_trace_count = added_parent_trace_count
    state.added_parent_via_count = added_parent_via_count

    notes = [
        "parent_composition=true",
        f"child_count={len(state.entries)}",
        f"interconnect_nets={state.interconnect_net_count}",
        f"inferred_interconnects={state.inferred_interconnect_net_count}",
        f"expected_child_traces={expected_child_trace_count}",
        f"expected_child_vias={expected_child_via_count}",
        f"preserved_child_traces={preserved_child_trace_count}",
        f"preserved_child_vias={preserved_child_via_count}",
        f"routed_total_traces={routed_total_trace_count}",
        f"routed_total_vias={routed_total_via_count}",
        f"added_parent_traces={added_parent_trace_count}",
        f"added_parent_vias={added_parent_via_count}",
    ]
    if copper_verification:
        notes.append(f"copper_status={copper_verification['status']}")
        notes.append(
            f"copper_trace_preservation="
            f"{copper_verification['matched_child_traces']}/"
            f"{copper_verification['expected_child_traces']}"
        )
        notes.append(
            f"copper_via_preservation="
            f"{copper_verification['matched_child_vias']}/"
            f"{copper_verification['expected_child_vias']}"
        )
    validation = routing_result.get("validation", {})
    if validation:
        notes.append(f"validation_accepted={validation.get('accepted', False)}")

    payload = build_solved_layout_artifact(
        layout,
        project_dir=str(project_dir),
        solver_version="parent-compose-v1",
        notes=notes,
    )

    save_solved_layout_artifact(payload)

    # Save additional metadata alongside the solved layout
    artifact_paths = resolve_artifact_paths(str(project_dir), parent_id)
    artifact_dir = Path(artifact_paths.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Write a metadata.json for the parent artifact
    renders_dir = artifact_dir / "renders"
    parent_pre_route_board_path = artifact_dir / "parent_pre_freerouting.kicad_pcb"
    parent_routed_board_path = artifact_dir / "parent_routed.kicad_pcb"
    parent_stamped_preview_path = renders_dir / "parent_stamped.png"
    parent_routed_preview_path = renders_dir / "parent_routed.png"

    metadata_payload = {
        "schema_version": "parent-compose-v1",
        "subcircuit_id": {
            "sheet_name": parent_id.sheet_name,
            "sheet_file": parent_id.sheet_file,
            "instance_path": parent_id.instance_path,
            "parent_instance_path": parent_id.parent_instance_path,
        },
        "parent_composition": True,
        "child_count": len(state.entries),
        "component_count": state.component_count,
        "trace_count": len(all_traces),
        "via_count": len(all_vias),
        "interconnect_net_count": state.interconnect_net_count,
        "inferred_interconnect_net_count": state.inferred_interconnect_net_count,
        "preserved_child_trace_count": state.preserved_child_trace_count,
        "preserved_child_via_count": state.preserved_child_via_count,
        "expected_preserved_child_trace_count": state.expected_preserved_child_trace_count,
        "expected_preserved_child_via_count": state.expected_preserved_child_via_count,
        "routed_total_trace_count": state.routed_total_trace_count,
        "routed_total_via_count": state.routed_total_via_count,
        "added_parent_trace_count": state.added_parent_trace_count,
        "added_parent_via_count": state.added_parent_via_count,
        "score_total": state.score_total,
        "copper_verification": copper_verification if copper_verification else {},
        "validation": validation,
        "artifact_paths": {
            "artifact_dir": str(artifact_dir),
            "renders_dir": str(renders_dir),
            "parent_pre_freerouting_board": str(parent_pre_route_board_path)
            if parent_pre_route_board_path.exists()
            else "",
            "parent_routed_board": str(parent_routed_board_path)
            if parent_routed_board_path.exists()
            else "",
            "parent_stamped_preview": str(parent_stamped_preview_path)
            if parent_stamped_preview_path.exists()
            else "",
            "parent_routed_preview": str(parent_routed_preview_path)
            if parent_routed_preview_path.exists()
            else "",
        },
        "notes": notes,
    }
    metadata_path = artifact_dir / "metadata.json"
    # Atomic write: GUI and other downstream consumers may read this
    # while compose is still finishing other artifacts in the same dir.
    tmp_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    tmp_metadata.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    tmp_metadata.replace(metadata_path)

    # Write a debug.json with routing details
    debug_payload = {
        "schema_version": "parent-compose-v1",
        "parent_composition": True,
        "geometry_validation": dict(state.geometry_validation),
        "artifact_paths": {
            "artifact_dir": str(artifact_dir),
            "renders_dir": str(renders_dir),
            "parent_pre_freerouting_board": str(parent_pre_route_board_path)
            if parent_pre_route_board_path.exists()
            else "",
            "parent_routed_board": str(parent_routed_board_path)
            if parent_routed_board_path.exists()
            else "",
            "parent_stamped_preview": str(parent_stamped_preview_path)
            if parent_stamped_preview_path.exists()
            else "",
            "parent_routed_preview": str(parent_routed_preview_path)
            if parent_routed_preview_path.exists()
            else "",
        },
        "routing_result": {
            "routed_board_path": routing_result.get("routed_board_path", ""),
            "trace_count": len(all_traces),
            "via_count": len(all_vias),
            "freerouting_stats": routing_result.get("freerouting_stats", {}),
            "preview_paths": {
                "parent_stamped_preview": str(parent_stamped_preview_path)
                if parent_stamped_preview_path.exists()
                else "",
                "parent_routed_preview": str(parent_routed_preview_path)
                if parent_routed_preview_path.exists()
                else "",
            },
            "board_paths": {
                "parent_pre_freerouting_board": str(parent_pre_route_board_path)
                if parent_pre_route_board_path.exists()
                else "",
                "parent_routed_board": str(parent_routed_board_path)
                if parent_routed_board_path.exists()
                else "",
            },
            "copper_accounting": {
                "expected_preserved_child_trace_count": state.expected_preserved_child_trace_count,
                "expected_preserved_child_via_count": state.expected_preserved_child_via_count,
                "preserved_child_trace_count": state.preserved_child_trace_count,
                "preserved_child_via_count": state.preserved_child_via_count,
                "routed_total_trace_count": state.routed_total_trace_count,
                "routed_total_via_count": state.routed_total_via_count,
                "added_parent_trace_count": state.added_parent_trace_count,
                "added_parent_via_count": state.added_parent_via_count,
            },
            "copper_verification": copper_verification,
            "copper_manifest": state.copper_manifest.to_dict() if state.copper_manifest else None,
        },
        "validation": validation,
        "hierarchical_status": {
            "current_parent": state.parent_sheet_name,
            "current_node": state.parent_instance_path,
            "top_level_status": "accepted"
            if validation.get("accepted")
            else "rejected",
            "composition_status": "routed"
            if not routing_result.get("failed")
            else "failed",
            "preview_paths": {
                "parent_stamped_preview": str(parent_stamped_preview_path)
                if parent_stamped_preview_path.exists()
                else "",
                "parent_routed_preview": str(parent_routed_preview_path)
                if parent_routed_preview_path.exists()
                else "",
                "parent_stamped_board": str(parent_pre_route_board_path)
                if parent_pre_route_board_path.exists()
                else "",
                "parent_routed_board": str(parent_routed_board_path)
                if parent_routed_board_path.exists()
                else "",
            },
            "leaf_workers": {
                "total": 0,
                "active": 0,
                "idle": 0,
                "queued": 0,
                "completed": len(state.entries),
            },
            "copper_accounting": {
                "expected_preserved_child_trace_count": state.expected_preserved_child_trace_count,
                "expected_preserved_child_via_count": state.expected_preserved_child_via_count,
                "preserved_child_trace_count": state.preserved_child_trace_count,
                "preserved_child_via_count": state.preserved_child_via_count,
                "routed_total_trace_count": state.routed_total_trace_count,
                "routed_total_via_count": state.routed_total_via_count,
                "added_parent_trace_count": state.added_parent_trace_count,
                "added_parent_via_count": state.added_parent_via_count,
            },
        },
        "composition_state": state.to_dict(),
    }
    debug_path = artifact_dir / "debug.json"
    # Atomic write: same rationale as metadata.json above.
    tmp_debug = debug_path.with_suffix(debug_path.suffix + ".tmp")
    tmp_debug.write_text(
        json.dumps(debug_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    tmp_debug.replace(debug_path)

    return str(artifact_dir)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose solved subcircuits into a parent composition state"
    )
    parser.add_argument(
        "--project",
        help="Project directory containing .experiments/subcircuits",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Specific solved artifact directory to include (repeatable)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Restrict composition to sheet name, sheet file, or instance path",
    )
    parser.add_argument(
        "--parent",
        help="Compose a real parent by sheet name, sheet file, or instance path (including root)",
    )
    parser.add_argument(
        "--spacing-mm",
        type=float,
        default=2.0,
        help="Spacing between rigid child modules in mm (default: 2)",
    )
    parser.add_argument(
        "--rotation-step-deg",
        type=float,
        default=0.0,
        help="Per-artifact rotation increment in degrees (default: 0)",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON file path to save the composition snapshot",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text",
    )
    # --- New flags for parent board stamping and routing ---
    parser.add_argument(
        "--pcb",
        help="Source .kicad_pcb file (template for stamping; needed for --stamp/--route)",
    )
    parser.add_argument(
        "--stamp",
        action="store_true",
        help="Stamp composition into a real .kicad_pcb file",
    )
    parser.add_argument(
        "--route",
        action="store_true",
        help="Route parent interconnects via FreeRouting (implies --stamp)",
    )
    parser.add_argument(
        "--jar",
        help="Path to FreeRouting JAR (overrides config)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of placement rounds for parent composition (default: 1)",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file to merge on top of default/project config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "RNG seed forwarded to the parent PlacementSolver. Different seeds "
            "produce different parent placements at fixed config; required for "
            "random-search to actually explore parent layouts (default: 0)."
        ),
    )
    parser.add_argument(
        "--manual-layout",
        help=(
            "Path to a manual_layout.v1 JSON file. When set, the candidate "
            "search is skipped: leaf placements and the board outline are "
            "taken verbatim from the file, and the rest of the pipeline "
            "(stamp, stamp_drc, route) runs unchanged."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    project_dir = Path(args.project).resolve() if args.project else None
    artifact_dirs = _resolve_artifact_dirs(project_dir, args.artifact)

    if not artifact_dirs:
        print(
            "error: no solved subcircuit artifacts found; provide --artifact or --project",
            file=sys.stderr,
        )
        return 2

    # Apply any active leaf pins so the canonical artifact files reflect
    # the pinned round, then load. ensure_applied is idempotent and a no-op
    # when no pins.json exists, so this is safe on every compose run. If
    # ensure_applied fails (corrupt pins.json, missing pinned round file),
    # we let it raise -- silently continuing with stale leaf state would
    # produce wrong placements without surfacing why.
    if project_dir is not None:
        from kicraft.autoplacer.brain import pins
        pin_status = pins.ensure_applied(project_dir / ".experiments")
        for leaf_key, status in pin_status.items():
            print(f"[pins] {leaf_key}: {status}")

    try:
        loaded_artifacts = load_solved_artifacts(list(artifact_dirs))
        loaded_artifacts = _filter_loaded_artifacts(loaded_artifacts, args.only)
        parent_definition = _select_parent_definition(project_dir, args.parent)
        loaded_artifacts = _filter_artifacts_for_parent(
            loaded_artifacts,
            parent_definition,
        )
        if not loaded_artifacts:
            if args.parent:
                print(
                    "error: no solved child artifacts found for selected parent",
                    file=sys.stderr,
                )
            else:
                print(
                    "error: no matching solved artifacts after filtering",
                    file=sys.stderr,
                )
            return 1

        # Resolve project clearance early so the composer's pad-margin can adapt
        # to the project's design rules even on plain compose runs without
        # --stamp/--route.
        from kicraft.autoplacer.config import discover_project_config, load_project_config
        compose_cfg: dict[str, Any] = {}
        if project_dir:
            proj_cfg_path = discover_project_config(str(project_dir))
            if proj_cfg_path:
                compose_cfg.update(load_project_config(str(proj_cfg_path)))
        if args.config:
            compose_cfg.update(load_project_config(args.config))

        # Candidate-search loop: K fast-cfg solves + DRC, pick shorts==0
        # winner (lex order: shorts asc, composite score desc). K=1 collapses
        # to a single fast solve; same code path. K and time_budget come
        # from cfg["parent_placement"]["candidate_search"], with project
        # config overrides applied via load_project_config above.
        search_cfg_raw = (
            compose_cfg.get("parent_placement", {}).get("candidate_search", {})
        )
        try:
            k = max(1, int(search_cfg_raw.get("k", 4)))
        except (TypeError, ValueError):
            k = 4
        try:
            time_budget_s = max(1.0, float(search_cfg_raw.get("time_budget_s", 240.0)))
        except (TypeError, ValueError):
            time_budget_s = 240.0

        manual_layout: ManualLayout | None = None
        if args.manual_layout:
            manual_layout = load_manual_layout(args.manual_layout)
            print(
                f"[manual-layout] loaded {len(manual_layout.placements)} "
                f"leaf placements + {len(manual_layout.parent_local)} parent-local "
                f"overrides; outline={manual_layout.board_outline}"
            )

        search_result = _search_best_layout(
            loaded_artifacts,
            spacing_mm=max(0.0, args.spacing_mm),
            rotation_step_deg=args.rotation_step_deg,
            parent_definition=parent_definition,
            pcb_path=Path(args.pcb) if args.pcb else None,
            project_dir=project_dir if project_dir else Path("."),
            cfg=compose_cfg,
            base_seed=int(args.seed),
            k=k,
            time_budget_s=time_budget_s,
            manual_layout=manual_layout,
        )
        state = search_result.winner_state
        transformed_payloads = search_result.winner_payloads

        output_path = None
        if args.output:
            output_path = _save_composition_snapshot(
                Path(args.output).resolve(),
                state,
                transformed_payloads,
            )

    except Exception as exc:
        print(f"error: failed to compose subcircuits: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                _json_payload(
                    loaded_artifacts, state, transformed_payloads, output_path
                ),
                indent=2,
            )
        )
        return 0

    _print_human_summary(loaded_artifacts, state, transformed_payloads, output_path)

    # --- Parent board stamping and routing ---
    if args.route or args.stamp:
        if not args.pcb:
            print(
                "error: --pcb is required for --stamp and --route",
                file=sys.stderr,
            )
            return 2

        pcb_path = Path(args.pcb)
        if not pcb_path.exists():
            print(f"error: PCB file not found: {pcb_path}", file=sys.stderr)
            return 2

        if project_dir is None:
            print(
                "error: --project is required for --stamp and --route",
                file=sys.stderr,
            )
            return 2

        # Build config: project config -> --config overlay -> --jar override
        from kicraft.autoplacer.config import discover_project_config, load_project_config

        cfg: dict[str, Any] = {"pcb_path": str(pcb_path)}
        proj_cfg_path = discover_project_config(str(project_dir))
        if proj_cfg_path:
            cfg.update(load_project_config(str(proj_cfg_path)))
        if args.config:
            cfg.update(load_project_config(args.config))
        if args.jar:
            cfg["freerouting_jar"] = args.jar

        try:
            geometry_validation = _validate_parent_geometry(state)
            geometry_accepted = bool(geometry_validation.get("accepted", False))

            # Always stamp + render, even when geometry validation fails.
            # The PNG is the user's diagnostic for why composition was
            # rejected, and the Monitor tab falls back to it when
            # parent_routed.png is absent (failed routing).
            try:
                _t_stamp = time.perf_counter()
                stamped_pcb = _stamp_parent_board(state, pcb_path, project_dir, cfg)
                state.phase_timings["stamp_ms"] = (
                    time.perf_counter() - _t_stamp
                ) * 1000.0
                print(f"parent_stamped_pcb : {stamped_pcb}")

                stamped_render_dir = stamped_pcb.parent / "renders"
                stamped_renders = _render_parent_board_views(
                    stamped_pcb,
                    stamped_render_dir / ".tmp_parent_stamped_views",
                )
                if stamped_renders.get("front_all"):
                    shutil.copy2(
                        stamped_renders["front_all"],
                        stamped_render_dir / "parent_stamped.png",
                    )
                    print(
                        f"parent_stamped_png : {stamped_render_dir / 'parent_stamped.png'}"
                    )
            except Exception as stamp_exc:
                if geometry_accepted:
                    raise
                # If geometry was already going to be rejected, a stamping
                # failure here is secondary -- keep the original rejection
                # as the reason but note the stamp problem.
                print(
                    f"warning: stamp/render failed after geometry rejection: {stamp_exc}",
                    file=sys.stderr,
                )

            if not geometry_accepted:
                # Diagnostic dump but DON'T exit. The previous behaviour
                # was to return 1 here, which is the THIRD pre-route
                # rejection gate that starved the search of signal:
                # the stamped board exists, the user wants to see what
                # came out, and FreeRouting can still produce a useful
                # routed render even on a layout whose components extend
                # past the auto-grown outline. Surface as a warning and
                # let routing run.
                print(
                    json.dumps(
                        {
                            "parent_geometry_validation_failed": True,
                            "geometry_validation": geometry_validation,
                        },
                        indent=2,
                    ),
                    file=sys.stderr,
                )
                print(
                    "warning: parent composition geometry validation failed; "
                    "continuing to route so the routed render is available "
                    "as a diagnostic",
                    file=sys.stderr,
                )

            # Stamp-time DRC guard: kicad-cli DRC on the pre-route board so
            # composer-introduced shorts (two leaves' locked tracks stamped
            # on top of each other) are caught and labeled as such, instead
            # of being misattributed to FreeRouting later. When shorts are
            # detected, routing is skipped: FreeRouting cannot fix
            # overlapping copper, and a 200 s+ routing pass on a known-bad
            # layout is wasted CPU. The composer-vs-router attribution is
            # recorded in state.stamp_drc and surfaces in the round JSON.
            stamp_shorts = 0
            stamp_clearance = 0
            try:
                from kicraft.autoplacer.freerouting_runner import _run_kicad_cli_drc
                _t_stamp_drc = time.perf_counter()
                _stamp_drc = _run_kicad_cli_drc(str(stamped_pcb), timeout_s=30)
                state.phase_timings["stamp_drc_ms"] = (
                    time.perf_counter() - _t_stamp_drc
                ) * 1000.0
                stamp_shorts = int(_stamp_drc.get("shorts", 0))
                stamp_clearance = int(_stamp_drc.get("clearance", 0))
                state.stamp_drc = {
                    "ran": bool(_stamp_drc.get("ran", False)),
                    "shorts": stamp_shorts,
                    "clearance": stamp_clearance,
                    "copper_edge_clearance": int(_stamp_drc.get("copper_edge_clearance", 0)),
                    "courtyard": int(_stamp_drc.get("courtyard", 0)),
                    "report_excerpt": (str(_stamp_drc.get("report_text", ""))[:2000]),
                }
                if stamp_shorts > 0:
                    print(
                        f"warning: stamp-time DRC found {stamp_shorts} shorts on "
                        f"parent_pre_freerouting -- composer stamped overlapping "
                        f"leaf tracks; skipping FreeRouting",
                        file=sys.stderr,
                    )
            except Exception as drc_exc:
                # kicad-cli failure or subprocess crash. Without a stamp
                # DRC result we can't verify the placement is routable;
                # proceeding to FreeRouting on an unverified board would
                # let composer-introduced shorts ship as routing failures
                # without attribution. Persist what we know and re-raise
                # so the round fails loudly.
                state.stamp_drc = {"ran": False, "error": str(drc_exc)}
                if args.output:
                    _save_composition_snapshot(
                        Path(args.output).resolve(),
                        state,
                        transformed_payloads,
                    )
                raise

            # Re-save the snapshot with stamp + stamp_drc timings populated
            # so --stamp-only runs (and the route-skipped path below) carry
            # the full phase_timings breakdown into the output JSON.
            if args.output:
                _save_composition_snapshot(
                    Path(args.output).resolve(),
                    state,
                    transformed_payloads,
                )

            # Early bail: stamp DRC shows the placement is unroutable as
            # stamped. Surface as a routing rejection so callers (the
            # autoexperiment harness, etc.) treat the round as failed.
            if stamp_shorts > 0 and args.route:
                state.routed_validation = {
                    "accepted": False,
                    "rejection_reasons": [f"stamp_shorts={stamp_shorts}"],
                    "drc": {
                        "shorts": stamp_shorts,
                        "clearance": stamp_clearance,
                        "skipped_routing": True,
                    },
                }
                if args.output:
                    _save_composition_snapshot(
                        Path(args.output).resolve(),
                        state,
                        transformed_payloads,
                    )
                print(
                    f"parent_status      : rejected (stamp_shorts={stamp_shorts})"
                )
                print(
                    f"error: parent stamped with {stamp_shorts} shorts; "
                    f"skipped routing",
                    file=sys.stderr,
                )
                return 1

            if args.route:
                _t_route = time.perf_counter()
                routing_result = _route_parent_board(
                    stamped_pcb, state, project_dir, cfg
                )
                state.phase_timings["freerouting_ms"] = (
                    time.perf_counter() - _t_route
                ) * 1000.0
                if not routing_result.get("failed"):
                    routed_board_path = Path(routing_result["routed_board_path"])
                    routed_renders = _render_parent_board_views(
                        routed_board_path,
                        stamped_render_dir / ".tmp_parent_routed_views",
                    )
                    if routed_renders.get("front_all"):
                        shutil.copy2(
                            routed_renders["front_all"],
                            stamped_render_dir / "parent_routed.png",
                        )
                        print(
                            f"parent_routed_png  : {stamped_render_dir / 'parent_routed.png'}"
                        )

                    validation = routing_result.get("validation", {})
                    state.routed_validation = _compact_routed_validation(validation)
                    if args.output:
                        # Re-save the snapshot so the --output file reflects
                        # the post-route DRC summary (rejection reasons, DRC
                        # category counts). Necessary because the first save
                        # ran before routing; otherwise the data is lost when
                        # the run exits non-zero on rejection.
                        _save_composition_snapshot(
                            Path(args.output).resolve(),
                            state,
                            transformed_payloads,
                        )

                    # Apply post-routing DRC penalty to parent score.
                    # Shorts tank the score to near-zero; clearance
                    # violations apply proportional reduction.
                    drc = validation.get("drc", {})
                    shorts = drc.get("shorts", 0)
                    clearance = drc.get("clearance", 0)
                    if shorts > 0:
                        state.score_total *= 0.01
                        state.score_breakdown["drc_penalty"] = 0.01
                        state.score_notes.append(
                            f"DRC penalty: {shorts} short(s) -- score *= 0.01"
                        )
                    elif clearance > 0:
                        penalty = min(0.9, clearance * 0.1)
                        state.score_total *= (1.0 - penalty)
                        state.score_breakdown["drc_penalty"] = 1.0 - penalty
                        state.score_notes.append(
                            f"DRC penalty: {clearance} clearance violation(s) -- score *= {1.0 - penalty:.3f}"
                        )

                    if validation.get("accepted"):
                        artifact_dir = _persist_parent_artifact(
                            state, routing_result, project_dir, cfg
                        )
                        print(f"parent_artifact    : {artifact_dir}")
                        print("parent_status      : accepted")
                        _emit_inspector_bundle(
                            Path(artifact_dir) / "parent_routed.kicad_pcb"
                        )
                    else:
                        reasons = validation.get("rejection_reasons", [])
                        reason_str = ', '.join(reasons) if reasons else 'unknown'
                        print(
                            f"parent_status      : rejected ({reason_str})"
                        )
                        # Still run the inspector on the rejected board so
                        # an AI agent can see exactly what failed: which
                        # DRC violations triggered the rejection, where
                        # the marker is vs. the board edge, etc.
                        rejected_pcb = (
                            Path(routing_result.get("routed_pcb", ""))
                            if routing_result.get("routed_pcb")
                            else None
                        )
                        if rejected_pcb and rejected_pcb.is_file():
                            _emit_inspector_bundle(rejected_pcb)
                        print(
                            f"error: parent board rejected by acceptance gate: {reason_str}",
                            file=sys.stderr,
                        )
                        return 1
                else:
                    error_msg = routing_result.get("error", "unknown error")
                    print(
                        f"warning: parent routing failed: {error_msg}",
                        file=sys.stderr,
                    )
        except Exception as exc:
            print(f"error: parent stamping/routing failed: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
