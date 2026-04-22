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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kicraft.autoplacer.brain.hierarchy_parser import parse_hierarchy


from kicraft.autoplacer.brain.copper_accounting import (
    CopperManifest,
    build_copper_manifest,
    verify_copper_preservation,
)
from kicraft.autoplacer.brain.subcircuit_composer import (
    ChildArtifactPlacement,
    DerivedAttachmentConstraints,
    LeafBlockerSet,
    ParentComposition,
    PlacementModel,
    build_parent_composition,
    dominant_blocker_side,
    estimate_layer_aware_parent_board_size,
    packed_extents_outline,
    derive_attachment_constraints,
    expand_rotation_candidates,
    child_layer_envelopes,
    can_overlap,
    can_overlap_sparse,
    constraint_aware_outline,
    extract_leaf_blocker_set,
    place_constrained_child,
    validate_child_constraints,
)
from kicraft.autoplacer.brain.subcircuit_extractor import extract_parent_local_components
from kicraft.autoplacer.brain.subcircuit_instances import (
    artifact_debug_dict,
    artifact_summary,
    load_solved_artifacts,
    transform_loaded_artifact,
    transformed_debug_dict,
    transformed_summary,
)
from kicraft.autoplacer.brain.types import Point, SubCircuitDefinition, SubCircuitId


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
    mode: str
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
            "mode": self.mode,
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


def _build_parent_local_blocker_set(comp) -> LeafBlockerSet:
    bbox = _component_geometry_bbox(comp)
    pad_rects = tuple(_shift_rect((pad.pos, pad.pos), Point(0.2, 0.2)) for pad in [])
    front_rects: list[tuple[Point, Point]] = []
    back_rects: list[tuple[Point, Point]] = []
    tht_rects: list[tuple[Point, Point]] = []
    if comp.pads:
        for pad in comp.pads:
            rect = (
                Point(pad.pos.x - 0.2, pad.pos.y - 0.2),
                Point(pad.pos.x + 0.2, pad.pos.y + 0.2),
            )
            if comp.is_through_hole:
                front_rects.append(rect)
                back_rects.append(rect)
                tht_rects.append(rect)
            elif comp.layer == 0:
                front_rects.append(rect)
            else:
                back_rects.append(rect)
    elif comp.is_through_hole:
        tht_rects.append(bbox)
    else:
        if comp.layer == 0:
            front_rects.append(bbox)
        else:
            back_rects.append(bbox)
    return LeafBlockerSet(
        front_pads=tuple(front_rects),
        back_pads=tuple(back_rects),
        tht_drills=tuple(tht_rects),
        leaf_outline=bbox,
    )


def _preview_parent_local_keep_in_rects(
    parent_local: dict[str, Any],
    local_constraints,
    outline: tuple[Point, Point],
    occupied_rects: list[tuple[Point, Point]] | None = None,
) -> list[tuple[Point, Point]]:
    if not parent_local or not local_constraints:
        return []
    preview_parent_local = copy.deepcopy(parent_local)
    _place_parent_local_components(preview_parent_local, local_constraints, outline)
    _reposition_parent_local_components(
        preview_parent_local,
        local_constraints,
        outline,
        occupied_rects or [],
    )
    keep_in_rects: list[tuple[Point, Point]] = []
    for constraint in local_constraints:
        comp = preview_parent_local.get(constraint.ref)
        if comp is None:
            continue
        bbox_min, bbox_max = _component_geometry_bbox(comp)
        keep_in_rects.append(
            (
                Point(
                    bbox_min.x - constraint.inward_keep_in_mm,
                    bbox_min.y - constraint.inward_keep_in_mm,
                ),
                Point(
                    bbox_max.x + constraint.inward_keep_in_mm,
                    bbox_max.y + constraint.inward_keep_in_mm,
                ),
            )
        )
    return keep_in_rects


def _make_placed_item(
    *,
    bbox: tuple[Point, Point],
    envelopes,
    blocker_set: LeafBlockerSet | None,
    origin: Point,
    rotation: float,
    label: str,
) -> dict[str, Any]:
    return {
        "bbox": bbox,
        "envelopes": envelopes,
        "blocker_set": blocker_set,
        "origin": origin,
        "rotation": rotation,
        "label": label,
    }


def _placed_items_conflict(
    existing_item: dict[str, Any],
    candidate_bbox: tuple[Point, Point],
    candidate_envelopes,
    candidate_blocker_set: LeafBlockerSet | None,
    candidate_origin: Point,
    candidate_rotation: float,
) -> bool:
    existing_bbox = existing_item["bbox"]
    if _bbox_disjoint(existing_bbox, candidate_bbox):
        return False
    existing_blocker_set = existing_item.get("blocker_set")
    if existing_blocker_set is not None and candidate_blocker_set is not None:
        return not can_overlap_sparse(
            existing_blocker_set,
            existing_item["origin"],
            existing_item["rotation"],
            candidate_blocker_set,
            candidate_origin,
            candidate_rotation,
        )
    return not can_overlap(existing_item["envelopes"], candidate_envelopes)


def _keep_in_conflict(
    blocker_set: LeafBlockerSet | None,
    origin: Point,
    rotation: float,
    keep_in_rects: list[tuple[Point, Point]],
) -> bool:
    if blocker_set is None:
        return False
    from kicraft.autoplacer.brain.subcircuit_composer import _transform_rect

    rects = [blocker_set.leaf_outline]
    for rect in rects:
        transformed_rect = _shift_bbox(rect, origin)
        if rotation % 360.0:
            transformed_rect = _transform_rect(rect, origin, rotation)
        if any(not _bbox_disjoint(transformed_rect, keep_in_rect) for keep_in_rect in keep_in_rects):
            return True
    return False


def _expand_rect(
    rect: tuple[Point, Point],
    margin_mm: float,
) -> tuple[Point, Point]:
    return (
        Point(rect[0].x - margin_mm, rect[0].y - margin_mm),
        Point(rect[1].x + margin_mm, rect[1].y + margin_mm),
    )


def _placed_item_blocker_rects(item: dict[str, Any]) -> list[tuple[Point, Point]]:
    blocker_set = item.get("blocker_set")
    if blocker_set is None:
        return []
    from kicraft.autoplacer.brain.subcircuit_composer import _transform_rect

    world_rects: list[tuple[Point, Point]] = []
    # Pads + drills are the copper/drill keep-out; component_rects are the
    # courtyard/body bboxes. Both matter for parent-local keep-ins: a
    # mounting hole needs clearance not just from copper but from any
    # component body (e.g. USB-C receptacle housing) that would block
    # the screw head. Without the component rects the mounting hole can
    # land visually inside a leaf silkscreen box even when no pad
    # happens to sit at that exact position.
    rects = (
        list(blocker_set.front_pads)
        + list(blocker_set.back_pads)
        + list(blocker_set.tht_drills)
        + list(blocker_set.component_rects.values())
    )
    for rect in rects:
        transformed_rect = _shift_bbox(rect, item["origin"])
        if item["rotation"] % 360.0:
            transformed_rect = _transform_rect(rect, item["origin"], item["rotation"])
        world_rects.append(transformed_rect)
    return world_rects


def _constraint_target_sides(constraint) -> set[str]:
    sides: set[str] = set()
    if constraint.target not in {"edge", "corner", "zone"}:
        return sides
    value = constraint.value
    for side in ("left", "right", "top", "bottom"):
        if side in value:
            sides.add(side)
    return sides


def _reposition_parent_local_components(
    parent_local: dict[str, Any],
    local_constraints,
    outline: tuple[Point, Point],
    occupied_rects: list[tuple[Point, Point]],
    *,
    step_mm: float = 0.5,
) -> None:
    if not occupied_rects:
        return
    outline_min, outline_max = outline
    for constraint in local_constraints:
        comp = parent_local.get(constraint.ref)
        if comp is None:
            continue

        bbox_min, bbox_max = _component_geometry_bbox(comp)
        width = bbox_max.x - bbox_min.x
        height = bbox_max.y - bbox_min.y
        sides = _constraint_target_sides(constraint)

        min_x = outline_min.x + constraint.inward_keep_in_mm
        max_x = outline_max.x - constraint.inward_keep_in_mm - width
        min_y = outline_min.y + constraint.inward_keep_in_mm
        max_y = outline_max.y - constraint.inward_keep_in_mm - height

        if "left" in sides:
            min_x = max(min_x, bbox_min.x)
        if "right" in sides:
            max_x = min(max_x, bbox_max.x - width)
        if "top" in sides:
            min_y = max(min_y, bbox_min.y)
        if "bottom" in sides:
            max_y = min(max_y, bbox_max.y - height)

        x_candidates: list[float] = []
        x = min_x
        while x <= max_x + 1e-9:
            x_candidates.append(round(x, 6))
            x += step_mm
        y_candidates: list[float] = []
        y = min_y
        while y <= max_y + 1e-9:
            y_candidates.append(round(y, 6))
            y += step_mm
        current_x = bbox_min.x
        current_y = bbox_min.y
        candidates = sorted(
            ((x, y) for x in x_candidates for y in y_candidates),
            key=lambda candidate: (
                abs(candidate[0] - current_x) + abs(candidate[1] - current_y),
                abs(candidate[0] - current_x),
                abs(candidate[1] - current_y),
            ),
        )
        for candidate_x, candidate_y in candidates:
            candidate_bbox = (
                Point(candidate_x, candidate_y),
                Point(candidate_x + width, candidate_y + height),
            )
            candidate_keep_in = _expand_rect(candidate_bbox, constraint.inward_keep_in_mm)
            if any(
                not _bbox_disjoint(candidate_keep_in, occupied_rect)
                for occupied_rect in occupied_rects
            ):
                continue
            delta = Point(candidate_x - bbox_min.x, candidate_y - bbox_min.y)
            if abs(delta.x) > 1e-9 or abs(delta.y) > 1e-9:
                _translate_component_geometry(comp, delta)
            break


def _occupied_pad_rects(placed_envelopes: list[dict[str, Any]]) -> list[tuple[Point, Point]]:
    rects: list[tuple[Point, Point]] = []
    for item in placed_envelopes:
        rects.extend(_placed_item_blocker_rects(item))
    return rects


def _constraint_world_anchors(
    child_anchor_positions: dict[str, Point],
    parent_local: dict[str, Any],
    constraints,
) -> dict[str, Point]:
    anchors = dict(child_anchor_positions)
    for constraint in constraints:
        if constraint.source != "parent_local":
            continue
        comp = parent_local.get(constraint.ref)
        if comp is None:
            continue
        if comp.pads:
            anchors[constraint.ref] = Point(
                sum(pad.pos.x for pad in comp.pads) / len(comp.pads),
                sum(pad.pos.y for pad in comp.pads) / len(comp.pads),
            )
        else:
            anchor = comp.body_center if comp.body_center is not None else comp.pos
            anchors[constraint.ref] = Point(anchor.x, anchor.y)
    return anchors


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
def _bbox_inside_frame(
    bbox: tuple[Point, Point],
    frame_min: Point,
    frame_max: Point,
    tolerance_mm: float = 1e-3,
) -> bool:
    return (
        bbox[0].x >= frame_min.x - tolerance_mm
        and bbox[0].y >= frame_min.y - tolerance_mm
        and bbox[1].x <= frame_max.x + tolerance_mm
        and bbox[1].y <= frame_max.y + tolerance_mm
    )


def _resolve_parent_local_allowlist(component_zones: dict[str, Any], loaded_artifacts) -> set[str]:
    child_refs = set()
    for artifact in loaded_artifacts:
        child_refs.update(artifact.layout.components.keys())
    return {
        ref
        for ref in component_zones.keys()
        if ref not in child_refs
    }


def _make_unconstrained_model(index: int, artifact, rotation_step_deg: float) -> PlacementModel:
    rotation = (index * rotation_step_deg) % 360.0
    transformed = transform_loaded_artifact(artifact, origin=Point(0.0, 0.0), rotation=rotation)
    return PlacementModel(
        rotation=rotation,
        transformed=transformed,
        blocker_set=extract_leaf_blocker_set(artifact),
        layer_envelopes=child_layer_envelopes(transformed),
        constraint_entries=[],
    )


def _candidate_positions(
    start: Point,
    frame_min: Point,
    frame_max: Point,
    bbox: tuple[Point, Point],
    spacing_mm: float,
    *,
    scan_from_frame_origin: bool = False,
):
    step = max(0.5, spacing_mm)
    if scan_from_frame_origin:
        x_start = frame_min.x - bbox[0].x
        y_start = frame_min.y - bbox[0].y
    else:
        x_start = min(max(start.x, frame_min.x - bbox[0].x), frame_max.x - bbox[0].x + step * 4)
        y_start = min(max(start.y, frame_min.y - bbox[0].y), frame_max.y - bbox[0].y + step * 4)
    y = y_start
    while y <= frame_max.y - bbox[0].y + step * 4:
        x = x_start if abs(y - y_start) < 1e-9 else frame_min.x - bbox[0].x
        while x <= frame_max.x - bbox[0].x + step * 4:
            yield Point(x, y)
            x += step
        y += step


def _constraint_axes_used(model: PlacementModel) -> tuple[bool, bool]:
    """Return (x_constrained, y_constrained) for a placed leaf."""
    x_constrained = False
    y_constrained = False
    for entry in model.constraint_entries:
        v = entry.constraint.value or ""
        if "left" in v or "right" in v:
            x_constrained = True
        if "top" in v or "bottom" in v:
            y_constrained = True
    return x_constrained, y_constrained


def _compact_free_axes(
    *,
    final_models: dict[int, PlacementModel],
    entries: list[CompositionEntry],
    loaded_artifacts,
    placed_envelopes: list[dict[str, Any]],
    placed_child_bboxes: dict[int, tuple[Point, Point]],
    child_artifact_placements: list[Any],
    transformed_payloads: list[dict[str, Any]],
    spacing_mm: float,
    all_constraints,
    child_anchor_positions: dict[str, Point],
) -> None:
    """Shift each leaf along its non-constrained axes toward the cluster
    centroid, stopping when another leaf or a child anchor would move.

    Mutates ``entries``, ``placed_envelopes``, ``placed_child_bboxes``,
    and ``child_anchor_positions`` in place.
    """
    if not final_models:
        return
    # Compute centroid of placed bboxes (targets to shift toward)
    bboxes = list(placed_child_bboxes.values())
    mid_x = sum((bb[0].x + bb[1].x) for bb in bboxes) / (2 * len(bboxes))
    mid_y = sum((bb[0].y + bb[1].y) for bb in bboxes) / (2 * len(bboxes))

    # Step sizes for binary search
    max_step = 64.0
    min_step = 0.5

    entry_by_path = {e.instance_path: e for e in entries}

    def _child_index_of(item):
        path = item.get("label") or ""
        # match via instance_path through loaded_artifacts
        for idx, art in enumerate(loaded_artifacts):
            if art.sheet_name == path:
                return idx
        return None

    # Precompute which envelope corresponds to which child_index
    env_child_index: dict[int, int] = {}
    for env_i, item in enumerate(placed_envelopes):
        label = item.get("label")
        for idx, art in enumerate(loaded_artifacts):
            if art.sheet_name == label:
                env_child_index[env_i] = idx
                break

    # Iterate: each round shifts each leaf by up to max_step toward centroid
    for _round in range(4):
        any_moved = False
        for child_index, model in final_models.items():
            x_con, y_con = _constraint_axes_used(model)
            if x_con and y_con:
                continue
            art = loaded_artifacts[child_index]
            entry = entry_by_path.get(art.instance_path)
            if entry is None:
                continue
            cur = entry.origin
            # Locate this leaf's envelope index
            env_idx = None
            for env_i, idx in env_child_index.items():
                if idx == child_index:
                    env_idx = env_i
                    break
            if env_idx is None:
                continue
            item = placed_envelopes[env_idx]
            bbox_local = (
                Point(placed_child_bboxes[child_index][0].x - cur.x, placed_child_bboxes[child_index][0].y - cur.y),
                Point(placed_child_bboxes[child_index][1].x - cur.x, placed_child_bboxes[child_index][1].y - cur.y),
            )

            def _try_move(new_origin: Point) -> bool:
                new_bbox = _shift_bbox(bbox_local, new_origin)
                for other_i, other_item in enumerate(placed_envelopes):
                    if other_i == env_idx:
                        continue
                    if _bbox_disjoint(other_item["bbox"], new_bbox):
                        continue
                    # Bboxes overlap -- check sparse
                    if _placed_items_conflict(
                        other_item,
                        new_bbox,
                        _shift_layer_envelopes(model.layer_envelopes, new_origin),
                        model.blocker_set,
                        new_origin,
                        model.rotation,
                    ):
                        return False
                return True

            dx_target = 0.0 if x_con else (mid_x - (cur.x + (bbox_local[0].x + bbox_local[1].x) / 2))
            dy_target = 0.0 if y_con else (mid_y - (cur.y + (bbox_local[0].y + bbox_local[1].y) / 2))
            if abs(dx_target) < min_step and abs(dy_target) < min_step:
                continue

            # Binary search for the max safe shift toward target
            best_dx = 0.0
            best_dy = 0.0
            # Try progressively smaller shifts
            step = min(max_step, max(abs(dx_target), abs(dy_target)))
            while step >= min_step:
                cand_dx = (dx_target / max(abs(dx_target), 1e-9)) * min(step, abs(dx_target)) if abs(dx_target) > min_step else 0.0
                cand_dy = (dy_target / max(abs(dy_target), 1e-9)) * min(step, abs(dy_target)) if abs(dy_target) > min_step else 0.0
                new_origin = Point(cur.x + best_dx + cand_dx, cur.y + best_dy + cand_dy)
                if _try_move(new_origin):
                    best_dx += cand_dx
                    best_dy += cand_dy
                step /= 2.0

            if abs(best_dx) < min_step and abs(best_dy) < min_step:
                continue

            # Commit the shift
            any_moved = True
            new_origin = Point(cur.x + best_dx, cur.y + best_dy)
            entry.origin = new_origin
            new_bbox = _shift_bbox(bbox_local, new_origin)
            placed_child_bboxes[child_index] = new_bbox
            placed_envelopes[env_idx]["bbox"] = new_bbox
            placed_envelopes[env_idx]["origin"] = new_origin
            placed_envelopes[env_idx]["envelopes"] = _shift_layer_envelopes(
                model.layer_envelopes, new_origin
            )
            # Update child anchor positions (world coords)
            for centry in model.constraint_entries:
                child_anchor_positions[centry.constraint.ref] = Point(
                    new_origin.x + centry.local_anchor_offset.x,
                    new_origin.y + centry.local_anchor_offset.y,
                )
            # Update the ChildArtifactPlacement (used for actual stamping)
            # and the transformed debug payload. Both are keyed by
            # instance_path.
            for cap in child_artifact_placements:
                if cap.artifact.instance_path == art.instance_path:
                    cap.origin = new_origin
                    break
            for tp in transformed_payloads:
                if tp.get("artifact", {}).get("instance_path") == art.instance_path:
                    from kicraft.autoplacer.brain.subcircuit_instances import (
                        transform_loaded_artifact,
                        transformed_debug_dict,
                        transformed_summary,
                    )
                    new_tr = transform_loaded_artifact(
                        art, origin=new_origin, rotation=model.rotation
                    )
                    tp["transformed"] = transformed_debug_dict(new_tr)
                    tp["summary"] = transformed_summary(new_tr)
                    break
        if not any_moved:
            break


def _bbox_overlap_area(a: tuple[Point, Point], b: tuple[Point, Point]) -> float:
    dx = max(0.0, min(a[1].x, b[1].x) - max(a[0].x, b[0].x))
    dy = max(0.0, min(a[1].y, b[1].y) - max(a[0].y, b[0].y))
    return dx * dy


def _opposite_side_overlap_weight(
    candidate_side: str, placed_side: str
) -> float:
    """Weight bbox overlap by whether it represents efficient dual-side packing.

    Opposite-side overlap (e.g. a front-dominant SMT leaf stacking over a
    back-dominant THT leaf) is the goal -- both leaves get their own copper
    area, halving XY usage. Same-side overlap is still legal when pad rects
    miss, but it doesn't buy density, so score it much lower.
    """
    if candidate_side == "none" or placed_side == "none":
        return 0.2
    if candidate_side == "dual" or placed_side == "dual":
        # Dual-side leaf is effectively both -- any overlap is partial win
        return 0.5
    if candidate_side != placed_side:
        return 1.0  # true opposite-side stacking -- the target behavior
    return 0.2  # same-side overlap -- legal by sparse check but not dense


def _find_non_overlapping_origin(
    proposed: Point,
    frame_min: Point,
    frame_max: Point,
    model: PlacementModel,
    placed_bboxes: list[tuple[Point, Point]],
    placed_envelopes,
    spacing_mm: float,
    parent_local_keep_in_rects: list[tuple[Point, Point]] | None = None,
) -> Point:
    parent_local_keep_in_rects = parent_local_keep_in_rects or []
    # Score each legal candidate by opposite-side-weighted overlap area so
    # front-dominant SMT leaves stack onto back-dominant leaves (battery
    # holders) instead of onto each other. Same-side overlap is legal when
    # individual pad rects miss, but it wastes density -- weight it lower so
    # opposite-side candidates win when both exist.
    candidate_side = dominant_blocker_side(model.blocker_set)
    best_overlap_candidate: Point | None = None
    best_overlap_score = 0.0
    best_overlap_dist = float("inf")
    best_empty_candidate: Point | None = None
    best_empty_dist = float("inf")
    placed_sides = [
        dominant_blocker_side(item["blocker_set"])
        if item.get("blocker_set") is not None
        else "none"
        for item in placed_envelopes
    ]
    for candidate in _candidate_positions(
        proposed,
        frame_min,
        frame_max,
        model.transformed.bounding_box,
        spacing_mm,
        scan_from_frame_origin=bool(placed_envelopes),
    ):
        candidate_bbox = _shift_bbox(model.transformed.bounding_box, candidate)
        if not _bbox_inside_frame(candidate_bbox, frame_min, frame_max):
            continue
        if _keep_in_conflict(
            model.blocker_set,
            candidate,
            model.rotation,
            parent_local_keep_in_rects,
        ):
            continue
        candidate_envelopes = _shift_layer_envelopes(model.layer_envelopes, candidate)
        collision = False
        overlap_score = 0.0
        for placed_side, existing_item in zip(placed_sides, placed_envelopes):
            area = _bbox_overlap_area(existing_item["bbox"], candidate_bbox)
            if area > 0.0:
                weight = _opposite_side_overlap_weight(candidate_side, placed_side)
                overlap_score += weight * area
            if _placed_items_conflict(
                existing_item,
                candidate_bbox,
                candidate_envelopes,
                model.blocker_set,
                candidate,
                model.rotation,
            ):
                collision = True
                break
        if collision:
            continue
        dist = (candidate.x - proposed.x) ** 2 + (candidate.y - proposed.y) ** 2
        if overlap_score > 0.0:
            # Prefer more opposite-side overlap; tiebreak by proximity
            if (
                overlap_score > best_overlap_score
                or (overlap_score == best_overlap_score and dist < best_overlap_dist)
            ):
                best_overlap_score = overlap_score
                best_overlap_dist = dist
                best_overlap_candidate = candidate
        elif best_overlap_candidate is None and dist < best_empty_dist:
            best_empty_dist = dist
            best_empty_candidate = candidate
    if best_overlap_candidate is not None:
        return best_overlap_candidate
    if best_empty_candidate is not None:
        return best_empty_candidate
    raise ValueError(
        "Unable to place unconstrained child inside composition frame "
        f"({frame_min.x:.3f},{frame_min.y:.3f})-({frame_max.x:.3f},{frame_max.y:.3f})"
    )


def _ordered_unconstrained_indices(mode: str, unconstrained_artifacts, spacing_mm: float):
    if mode != "packed":
        return list(unconstrained_artifacts)

    def _bbox_area(bbox: tuple[Point, Point] | None) -> float:
        if bbox is None:
            return 0.0
        return max(0.0, bbox[1].x - bbox[0].x) * max(0.0, bbox[1].y - bbox[0].y)

    def _sort_metrics(item):
        index, artifact = item
        blocker_set = extract_leaf_blocker_set(artifact)
        front_blocker_area = sum(_rect_area(rect) for rect in blocker_set.front_pads) + sum(_rect_area(rect) for rect in blocker_set.tht_drills)
        back_blocker_area = sum(_rect_area(rect) for rect in blocker_set.back_pads) + sum(_rect_area(rect) for rect in blocker_set.tht_drills)
        has_tht_or_dual = int(
            bool(blocker_set.tht_drills)
            or dominant_blocker_side(blocker_set) == "dual"
        )
        total_bbox_area = max(0.0, artifact.layout.width) * max(0.0, artifact.layout.height)
        return (
            -max(front_blocker_area, back_blocker_area),
            -has_tht_or_dual,
            -total_bbox_area,
            index,
        )

    indexed_artifacts = list(unconstrained_artifacts)
    indexed_artifacts.sort(key=_sort_metrics)
    return indexed_artifacts


def _translate_component_geometry(comp, delta: Point) -> None:
    comp.pos = Point(comp.pos.x + delta.x, comp.pos.y + delta.y)
    if comp.body_center is not None:
        comp.body_center = Point(comp.body_center.x + delta.x, comp.body_center.y + delta.y)
    comp.pads = [
        pad.__class__(
            ref=pad.ref,
            pad_id=pad.pad_id,
            pos=Point(pad.pos.x + delta.x, pad.pos.y + delta.y),
            net=pad.net,
            layer=pad.layer,
        )
        for pad in comp.pads
    ]


def _place_parent_local_components(
    parent_local: dict[str, Any],
    local_constraints,
    outline: tuple[Point, Point],
) -> None:
    min_pt, max_pt = outline
    for constraint in local_constraints:
        comp = parent_local.get(constraint.ref)
        if comp is None:
            continue

        bbox_min, bbox_max = _component_geometry_bbox(comp)
        delta_x = 0.0
        delta_y = 0.0
        left_target = min_pt.x + constraint.inward_keep_in_mm - constraint.outward_overhang_mm
        right_target = max_pt.x - constraint.inward_keep_in_mm + constraint.outward_overhang_mm
        top_target = min_pt.y + constraint.inward_keep_in_mm - constraint.outward_overhang_mm
        bottom_target = max_pt.y - constraint.inward_keep_in_mm + constraint.outward_overhang_mm

        if constraint.target == "corner":
            if constraint.value == "top-left":
                delta_x = left_target - bbox_min.x
                delta_y = top_target - bbox_min.y
            elif constraint.value == "top-right":
                delta_x = right_target - bbox_max.x
                delta_y = top_target - bbox_min.y
            elif constraint.value == "bottom-left":
                delta_x = left_target - bbox_min.x
                delta_y = bottom_target - bbox_max.y
            elif constraint.value == "bottom-right":
                delta_x = right_target - bbox_max.x
                delta_y = bottom_target - bbox_max.y
        elif constraint.target == "edge":
            if constraint.value == "left":
                delta_x = left_target - bbox_min.x
            elif constraint.value == "right":
                delta_x = right_target - bbox_max.x
            elif constraint.value == "top":
                delta_y = top_target - bbox_min.y
            elif constraint.value == "bottom":
                delta_y = bottom_target - bbox_max.y

        delta = Point(delta_x, delta_y)
        _translate_component_geometry(comp, delta)


def _compose_artifacts(
    loaded_artifacts,
    *,
    mode: str,
    spacing_mm: float,
    rotation_step_deg: float,
    parent_definition: SubCircuitDefinition | None = None,
    pcb_path: Path | None = None,
) -> tuple[ParentCompositionState, list[dict[str, Any]]]:
    """Compose loaded artifacts into a parent composition snapshot."""
    entries: list[CompositionEntry] = []
    transformed_payloads: list[dict[str, Any]] = []
    child_artifact_placements: list[ChildArtifactPlacement] = []
    packing_metadata: dict[str, Any] = {}
    final_models: dict[int, PlacementModel] = {}
    placed_child_bboxes: dict[int, tuple[Point, Point]] = {}
    child_anchor_positions: dict[str, Point] = {}
    parent_local_keep_in_rects: list[tuple[Point, Point]] = []

    def _append_entry(index: int, artifact, origin: Point, model: PlacementModel):
        transformed = transform_loaded_artifact(
            artifact,
            origin=origin,
            rotation=model.rotation,
        )

        entry = CompositionEntry(
            artifact_dir=artifact.artifact_dir,
            sheet_name=artifact.sheet_name,
            instance_path=artifact.instance_path,
            origin=origin,
            rotation=model.rotation,
            transformed_bbox=transformed.instance.transformed_bbox,
            component_count=len(transformed.transformed_components),
            trace_count=len(transformed.transformed_traces),
            via_count=len(transformed.transformed_vias),
            anchor_count=len(transformed.transformed_anchors),
        )
        entries.append(entry)
        child_artifact_placements.append(
            ChildArtifactPlacement(
                artifact=artifact,
                origin=origin,
                rotation=model.rotation,
            )
        )
        transformed_payloads.append(
            {
                "artifact": artifact_debug_dict(artifact),
                "transformed": transformed_debug_dict(transformed),
                "summary": transformed_summary(transformed),
            }
        )
        return transformed

    from kicraft.autoplacer.config import load_project_config, discover_project_config
    import logging
    logger = logging.getLogger(__name__)

    cfg = {}
    component_zones = {}
    parent_local = {}

    if pcb_path:
        try:
            project_dir = Path(pcb_path).resolve().parent
            cfg_file = discover_project_config(project_dir)
            if cfg_file is not None:
                cfg = load_project_config(str(cfg_file))
                component_zones = cfg.get("component_zones", {})
            parent_local = extract_parent_local_components(
                str(pcb_path),
                loaded_artifacts,
                allowlist=_resolve_parent_local_allowlist(component_zones, loaded_artifacts),
            )
        except Exception as e:
            logger.warning(f"Could not load config/local components: {e}")

    derived_constraints = derive_attachment_constraints(
        loaded_artifacts,
        parent_local,
        component_zones,
        cfg,
        rotation_step_deg=rotation_step_deg,
    )
    all_constraints = derived_constraints.constraints
    logger.info("composition: %d attachment constraints derived", len(all_constraints))

    constrained_indices = set(derived_constraints.child_specs.keys())
    unconstrained_artifacts = [
        (i, art) for i, art in enumerate(loaded_artifacts) if i not in constrained_indices
    ]

    child_bbox_sizes = [
        _bbox_size(spec.models[spec.rotation_candidates[0]].transformed.bounding_box)
        for spec in derived_constraints.child_specs.values()
    ] + [
        _bbox_size(_make_unconstrained_model(index, artifact, rotation_step_deg).transformed.bounding_box)
        for index, artifact in unconstrained_artifacts
    ]
    seed_child_envelopes = [
        child_layer_envelopes(
            transform_loaded_artifact(artifact, origin=Point(0.0, 0.0), rotation=0.0)
        )
        for artifact in loaded_artifacts
    ]
    seed_width, seed_height = estimate_layer_aware_parent_board_size(
        seed_child_envelopes,
        interconnect_net_count=0,
        margin_mm=spacing_mm,
    )
    seed_frame_min = Point(0.0, 0.0)
    seed_frame_max = Point(seed_width, seed_height)

    final_placed_envelopes = []
    last_outline = (seed_frame_min, seed_frame_max)
    last_packed_extents = (seed_frame_min, seed_frame_max)

    for iteration in range(3):
        entries = []
        transformed_payloads = []
        child_artifact_placements = []
        final_models = {}
        placed_child_bboxes = {}
        child_anchor_positions = {}
        placed_envelopes = []
        parent_local_keep_in_rects = []

        for child_index in sorted(derived_constraints.child_specs):
            spec = derived_constraints.child_specs[child_index]
            artifact = loaded_artifacts[child_index]
            selected_model = None
            selected_origin = None
            selected_bbox = None

            for rotation in spec.rotation_candidates:
                model = spec.models[rotation]
                try:
                    origin, placed_bbox = place_constrained_child(
                        model,
                        parent_outline_min=seed_frame_min,
                        parent_outline_max=seed_frame_max,
                    )
                except ValueError:
                    continue

                shifted_envelopes = _shift_layer_envelopes(model.layer_envelopes, origin)
                overlap_conflict = False
                for existing_item in placed_envelopes:
                    if _placed_items_conflict(
                        existing_item,
                        placed_bbox,
                        shifted_envelopes,
                        model.blocker_set,
                        origin,
                        model.rotation,
                    ):
                        overlap_conflict = True
                        break
                if overlap_conflict:
                    continue

                selected_model = model
                selected_origin = origin
                selected_bbox = placed_bbox
                break

            if selected_model is None:
                if len(spec.rotation_candidates) == 1:
                    expand_rotation_candidates(spec)
                    for rotation in spec.rotation_candidates[1:]:
                        model = spec.models[rotation]
                        try:
                            origin, placed_bbox = place_constrained_child(
                                model,
                                parent_outline_min=seed_frame_min,
                                parent_outline_max=seed_frame_max,
                            )
                        except ValueError:
                            continue
                        shifted_envelopes = _shift_layer_envelopes(model.layer_envelopes, origin)
                        overlap_conflict = False
                        for existing_item in placed_envelopes:
                            if _placed_items_conflict(
                                existing_item,
                                placed_bbox,
                                shifted_envelopes,
                                model.blocker_set,
                                origin,
                                model.rotation,
                            ):
                                overlap_conflict = True
                                break
                        if overlap_conflict:
                            continue
                        selected_model = model
                        selected_origin = origin
                        selected_bbox = placed_bbox
                        break

            if selected_model is None or selected_origin is None or selected_bbox is None:
                refs = ", ".join(constraint.ref for constraint in spec.constraints)
                raise ValueError(
                    f"Unable to place constrained child {artifact.instance_path} for refs {refs}"
                )

            transformed = _append_entry(child_index, artifact, selected_origin, selected_model)
            final_models[child_index] = selected_model
            placed_child_bboxes[child_index] = selected_bbox
            placed_envelopes.append(
                _make_placed_item(
                    bbox=selected_bbox,
                    envelopes=_shift_layer_envelopes(selected_model.layer_envelopes, selected_origin),
                    blocker_set=selected_model.blocker_set,
                    origin=selected_origin,
                    rotation=selected_model.rotation,
                    label=artifact.sheet_name,
                )
            )
            for entry in selected_model.constraint_entries:
                child_anchor_positions[entry.constraint.ref] = Point(
                    selected_origin.x + entry.local_anchor_offset.x,
                    selected_origin.y + entry.local_anchor_offset.y,
                )

        parent_local_keep_in_rects = _preview_parent_local_keep_in_rects(
            parent_local,
            derived_constraints.parent_local_constraints,
            (seed_frame_min, seed_frame_max),
            occupied_rects=_occupied_pad_rects(placed_envelopes),
        )

        ordered_unconstrained = _ordered_unconstrained_indices(mode, unconstrained_artifacts, spacing_mm)
        row_count = 0
        row_widths: list[float] = []
        row_heights: list[float] = []
        row_item_counts: list[int] = []
        row_x = seed_frame_min.x + spacing_mm
        row_y = seed_frame_min.y + spacing_mm
        row_height = 0.0
        current_row_items = 0
        target_row_width = seed_width
        if mode == "packed":
            total_area = sum(
                max(0.0, artifact.layout.width) * max(0.0, artifact.layout.height)
                for _, artifact in ordered_unconstrained
            )
            max_child_width = max(
                (max(0.0, artifact.layout.width) for _, artifact in ordered_unconstrained),
                default=0.0,
            )
            est_width, _ = estimate_layer_aware_parent_board_size(
                [
                    child_layer_envelopes(
                        transform_loaded_artifact(
                            artifact,
                            origin=Point(0.0, 0.0),
                            rotation=0.0,
                        )
                    )
                    for index, artifact in ordered_unconstrained
                ],
                margin_mm=spacing_mm,
            )
            target_row_width = max(max_child_width + spacing_mm, est_width)
            packing_metadata = {
                "strategy": "packed_rows",
                "sort_key": "area_desc_width_desc_height_desc",
                "target_row_width_mm": target_row_width,
                "estimated_total_child_area_mm2": total_area,
                "max_child_width_mm": max_child_width,
            }
        elif mode == "row":
            packing_metadata = {"strategy": "row", "sort_key": "input_order"}
        elif mode == "column":
            packing_metadata = {"strategy": "column", "sort_key": "input_order"}
        elif mode == "grid":
            cols = max(1, math.ceil(math.sqrt(len(ordered_unconstrained))))
            packing_metadata = {"strategy": "grid", "grid_columns": cols, "sort_key": "input_order"}
        else:
            raise ValueError(f"Unsupported composition mode: {mode}")

        for ordinal, (index, artifact) in enumerate(ordered_unconstrained):
            model = _make_unconstrained_model(index, artifact, rotation_step_deg)
            final_models[index] = model
            if mode == "column":
                proposed = Point(seed_frame_min.x + spacing_mm, row_y)
            elif mode == "grid":
                cols = packing_metadata.get("grid_columns", 1)
                max_width = max((item[1].layout.width for item in ordered_unconstrained), default=0.0)
                max_height = max((item[1].layout.height for item in ordered_unconstrained), default=0.0)
                cell_w = max_width + spacing_mm
                cell_h = max_height + spacing_mm
                row = ordinal // cols
                col = ordinal % cols
                proposed = Point(seed_frame_min.x + spacing_mm + col * cell_w, seed_frame_min.y + spacing_mm + row * cell_h)
                packing_metadata["grid_rows"] = math.ceil(len(ordered_unconstrained) / cols) if ordered_unconstrained else 0
                packing_metadata["cell_width_mm"] = cell_w
                packing_metadata["cell_height_mm"] = cell_h
            else:
                should_wrap = (
                    mode == "packed"
                    and current_row_items > 0
                    and (row_x + model.transformed.width_mm) > (seed_frame_min.x + target_row_width)
                )
                if should_wrap:
                    row_widths.append(max(0.0, row_x - spacing_mm - seed_frame_min.x))
                    row_heights.append(row_height)
                    row_item_counts.append(current_row_items)
                    row_count += 1
                    row_y += row_height + spacing_mm
                    row_x = seed_frame_min.x + spacing_mm
                    row_height = 0.0
                    current_row_items = 0
                proposed = Point(row_x, row_y)

            origin = _find_non_overlapping_origin(
                proposed,
                seed_frame_min,
                seed_frame_max,
                model,
                list(placed_child_bboxes.values()),
                placed_envelopes,
                spacing_mm,
                parent_local_keep_in_rects=parent_local_keep_in_rects,
            )
            placed_bbox = _shift_bbox(model.transformed.bounding_box, origin)
            transformed = _append_entry(index, artifact, origin, model)
            placed_child_bboxes[index] = placed_bbox
            placed_envelopes.append(
                _make_placed_item(
                    bbox=placed_bbox,
                    envelopes=_shift_layer_envelopes(model.layer_envelopes, origin),
                    blocker_set=model.blocker_set,
                    origin=origin,
                    rotation=model.rotation,
                    label=artifact.sheet_name,
                )
            )

            if mode == "column":
                row_y = placed_bbox[1].y + spacing_mm
            elif mode in {"row", "packed"}:
                row_x = placed_bbox[1].x + spacing_mm
                row_height = max(row_height, placed_bbox[1].y - placed_bbox[0].y)
                current_row_items += 1

        if mode == "packed" and current_row_items > 0:
            row_widths.append(max(0.0, row_x - spacing_mm - seed_frame_min.x))
            row_heights.append(row_height)
            row_item_counts.append(current_row_items)
            row_count += 1
            packing_metadata["row_count"] = row_count
            packing_metadata["row_widths_mm"] = row_widths
            packing_metadata["row_heights_mm"] = row_heights
            packing_metadata["row_item_counts"] = row_item_counts

        placed_bbox_list = [placed_child_bboxes[index] for index in sorted(placed_child_bboxes)]
        if all_constraints:
            last_outline = constraint_aware_outline(
                placed_bboxes=placed_bbox_list,
                attachment_constraints=all_constraints,
                constrained_ref_world_anchors=child_anchor_positions,
                margin_mm=spacing_mm,
            )
        else:
            last_outline = packed_extents_outline(placed_bbox_list, margin_mm=spacing_mm)
        last_packed_extents = packed_extents_outline(placed_bbox_list, margin_mm=0.0)

        if (
            last_packed_extents[1].x <= seed_frame_max.x + 1e-3
            and last_packed_extents[1].y <= seed_frame_max.y + 1e-3
            and last_packed_extents[0].x >= seed_frame_min.x - 1e-3
            and last_packed_extents[0].y >= seed_frame_min.y - 1e-3
        ):
            final_placed_envelopes = placed_envelopes
            seed_frame_min = Point(last_outline[0].x, last_outline[0].y)
            seed_frame_max = Point(last_outline[1].x, last_outline[1].y)
            break

        seed_frame_min = Point(min(seed_frame_min.x, last_packed_extents[0].x - spacing_mm), min(seed_frame_min.y, last_packed_extents[0].y - spacing_mm))
        seed_frame_max = Point(max(seed_frame_max.x, last_packed_extents[1].x + spacing_mm), max(seed_frame_max.y, last_packed_extents[1].y + spacing_mm))
        final_placed_envelopes = placed_envelopes

    # Post-iteration compaction: shift leaves along axes that have no
    # explicit constraint toward the cluster centroid. Constrained axes
    # (e.g. USB INPUT's x is pinned to the left edge, LDO 3.3V's x is
    # pinned to the right edge) are left alone. This closes the empty
    # top/bottom strips that appear because leaves default to the frame
    # origin on their free axes -- without this pass USB INPUT and LDO
    # 3.3V hug the top of the frame and force the board height to be
    # much larger than the leaf cluster actually needs.
    _compact_free_axes(
        final_models=final_models,
        entries=entries,
        loaded_artifacts=loaded_artifacts,
        placed_envelopes=final_placed_envelopes,
        placed_child_bboxes=placed_child_bboxes,
        child_artifact_placements=child_artifact_placements,
        transformed_payloads=transformed_payloads,
        spacing_mm=spacing_mm,
        all_constraints=all_constraints,
        child_anchor_positions=child_anchor_positions,
    )

    # Recompute outline now that leaves may have shifted inward
    placed_bbox_list = [placed_child_bboxes[index] for index in sorted(placed_child_bboxes)]
    if all_constraints:
        last_outline = constraint_aware_outline(
            placed_bboxes=placed_bbox_list,
            attachment_constraints=all_constraints,
            constrained_ref_world_anchors=child_anchor_positions,
            margin_mm=spacing_mm,
        )
    else:
        last_outline = packed_extents_outline(placed_bbox_list, margin_mm=spacing_mm)

    exact_outline = last_outline
    for child_index, model in final_models.items():
        entry = next(item for item in entries if item.instance_path == loaded_artifacts[child_index].instance_path)
        validate_child_constraints(
            model,
            origin=entry.origin,
            parent_outline_min=exact_outline[0],
            parent_outline_max=exact_outline[1],
        )

    _place_parent_local_components(parent_local, derived_constraints.parent_local_constraints, exact_outline)
    _reposition_parent_local_components(
        parent_local,
        derived_constraints.parent_local_constraints,
        exact_outline,
        _occupied_pad_rects(final_placed_envelopes),
    )
    parent_local_keep_in_rects = []
    for constraint in derived_constraints.parent_local_constraints:
        comp = parent_local.get(constraint.ref)
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
    packing_metadata["board_width_mm"] = round(outline_w, 2)
    packing_metadata["board_height_mm"] = round(outline_h, 2)

    project_dir = (
        str(Path(loaded_artifacts[0].artifact_dir).resolve().parents[2])
        if loaded_artifacts
        else ""
    )
    parent_subcircuit = parent_definition or SubCircuitDefinition(
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
            f"mode={mode}",
            f"artifact_count={len(loaded_artifacts)}",
        ],
    )
    composition = build_parent_composition(
        parent_subcircuit,
        child_artifact_placements=child_artifact_placements,
        board_outline=exact_outline,
        local_components=parent_local,
    )

    # Build copper manifest before the flat merge loses provenance
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

    for ref, comp in parent_local.items():
        if ref in composition.board_state.components:
            composition.board_state.components[ref] = copy.deepcopy(comp)
        if ref in composition.hierarchy_state.local_components:
            composition.hierarchy_state.local_components[ref] = copy.deepcopy(comp)

    import itertools

    edge_attachment_satisfied = {}
    mounting_hole_keep_in_satisfied = {}

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
            comp = parent_local.get(c.ref)
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
            if c.value == "left":
                expected_x = min_pt.x + c.inward_keep_in_mm - c.outward_overhang_mm
            elif c.value == "right":
                expected_x = max_pt.x - c.inward_keep_in_mm + c.outward_overhang_mm
            elif c.value == "top":
                expected_y = min_pt.y + c.inward_keep_in_mm - c.outward_overhang_mm
            elif c.value == "bottom":
                expected_y = max_pt.y - c.inward_keep_in_mm + c.outward_overhang_mm
        elif c.target == "corner":
            if c.value == "top-left":
                expected_x = min_pt.x + c.inward_keep_in_mm - c.outward_overhang_mm
                expected_y = min_pt.y + c.inward_keep_in_mm - c.outward_overhang_mm
            elif c.value == "top-right":
                expected_x = max_pt.x - c.inward_keep_in_mm + c.outward_overhang_mm
                expected_y = min_pt.y + c.inward_keep_in_mm - c.outward_overhang_mm
            elif c.value == "bottom-left":
                expected_x = min_pt.x + c.inward_keep_in_mm - c.outward_overhang_mm
                expected_y = max_pt.y - c.inward_keep_in_mm + c.outward_overhang_mm
            elif c.value == "bottom-right":
                expected_x = max_pt.x - c.inward_keep_in_mm + c.outward_overhang_mm
                expected_y = max_pt.y - c.inward_keep_in_mm + c.outward_overhang_mm
        elif c.target == "zone" and c.value == "bottom":
            expected_y = max_pt.y - c.inward_keep_in_mm

        ok_x = expected_x is None or abs(actual_x - expected_x) <= 1e-3
        ok_y = expected_y is None or abs(actual_y - expected_y) <= 1e-3
        ok = ok_x and ok_y

        edge_attachment_satisfied[c.ref] = ok

        is_hole = c.ref.startswith("H") or (c.inward_keep_in_mm > 0 and "hole" in c.ref.lower())
        if is_hole:
            mounting_hole_keep_in_satisfied[c.ref] = ok


    same_side_overlap_conflicts = []
    tht_keepout_violations = []

    ordered_entry_indices = [
        next(index for index, artifact in enumerate(loaded_artifacts) if artifact.instance_path == entry.instance_path)
        for entry in entries
    ]
    for i, j in itertools.combinations(range(len(final_placed_envelopes)), 2):
        item_a = final_placed_envelopes[i]
        item_b = final_placed_envelopes[j]
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
                a_label = getattr(art_a, "label", getattr(art_a, "slug", getattr(art_a, "sheet_name", f"child[{i}]")))
                b_label = getattr(art_b, "label", getattr(art_b, "slug", getattr(art_b, "sheet_name", f"child[{j}]")))

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
                        and dominant_blocker_side(blocker_a) == dominant_blocker_side(blocker_b)
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

    unsatisfied_edges = sum(1 for ok in edge_attachment_satisfied.values() if not ok)
    logger.info("composition: %d constraints, %d unsatisfied edges, %d overlap conflicts, %d THT violations", 
        len(all_constraints), unsatisfied_edges, len(same_side_overlap_conflicts), len(tht_keepout_violations))


    state = ParentCompositionState(
        project_dir=project_dir,
        mode=mode,
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
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(output_path)


def _print_human_summary(
    loaded_artifacts,
    state: ParentCompositionState,
    transformed_payloads: list[dict[str, Any]],
    output_path: str | None,
) -> None:
    print("=== Subcircuit Composition ===")
    print(f"artifacts              : {len(loaded_artifacts)}")
    print(f"mode                   : {state.mode}")
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


def _validate_parent_geometry(
    state: ParentCompositionState,
    overhangs: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Validate that composed parent geometry fits inside the derived outline."""
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
    overhang_map = {str(ref): float(value) for ref, value in (overhangs or {}).items()}

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
        bbox_tl, bbox_br = comp.bbox()
        geometry_union_min_x = min(geometry_union_min_x, bbox_tl.x)
        geometry_union_min_y = min(geometry_union_min_y, bbox_tl.y)
        geometry_union_max_x = max(geometry_union_max_x, bbox_br.x)
        geometry_union_max_y = max(geometry_union_max_y, bbox_br.y)
        allowed_overhang = max(0.0, overhang_map.get(ref, 0.0))
        ref_min_x = min_x - allowed_overhang
        ref_max_x = max_x + allowed_overhang
        if ref in edge_constrained:
            # Edge-pinned connectors (e.g. USB-C) are allowed to have their
            # body extend beyond the outline on the constrained side -- the
            # PCB Edge marker (D1) anchors pads to the fab edge, so the
            # shell naturally hangs off. We still verify pads are inside.
            component_outside = False
        else:
            component_outside = (
                bbox_tl.x < ref_min_x
                or bbox_tl.y < min_y
                or bbox_br.x > ref_max_x
                or bbox_br.y > max_y
            )
        pad_outside_count = 0
        for pad in comp.pads:
            if (
                pad.pos.x < ref_min_x
                or pad.pos.x > ref_max_x
                or pad.pos.y < min_y
                or pad.pos.y > max_y
            ):
                pad_outside_count += 1
                outside_pads += 1
        if component_outside or pad_outside_count > 0:
            outside_components.append(
                {
                    "ref": ref,
                    "bbox": {
                        "top_left": {"x": bbox_tl.x, "y": bbox_tl.y},
                        "bottom_right": {"x": bbox_br.x, "y": bbox_br.y},
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
) -> Path:
    """Stamp the parent composition onto a real .kicad_pcb file.

    Uses a subprocess to run pcbnew operations so the main process does not
    need pcbnew installed.  The subprocess:
    1. Loads the copied board
    2. Moves footprints to their composed positions
    3. Clears existing tracks/zones
    4. Recreates traces/vias from the merged child copper
    5. Rebuilds connectivity and saves

    Returns the stamped board path.
    """
    import json as _json
    import os
    import tempfile

    from kicraft.autoplacer.brain.subcircuit_artifacts import slugify_subcircuit_id
    from kicraft.autoplacer.brain.types import Layer
    from kicraft.autoplacer.freerouting_runner import _run_pcbnew_script

    composition = state.composition
    if composition is None:
        raise RuntimeError("ParentCompositionState has no composition object")

    parent_id = composition.hierarchy_state.subcircuit.id
    slug = slugify_subcircuit_id(parent_id)
    artifact_dir = project_dir / ".experiments" / "subcircuits" / slug
    artifact_dir.mkdir(parents=True, exist_ok=True)

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

    geometry_validation = _validate_parent_geometry(
        state,
        overhangs=cfg.get("parent_overhang_mm", {}),
    )
    if not geometry_validation.get("accepted", False):
        raise RuntimeError(
            "Parent composition geometry is invalid before stamping: "
            f"outside_components={geometry_validation.get('outside_component_count', 0)} "
            f"outside_pads={geometry_validation.get('outside_pad_count', 0)} "
            f"outside_traces={geometry_validation.get('outside_trace_count', 0)} "
            f"outside_vias={geometry_validation.get('outside_via_count', 0)}"
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

        script = _PARENT_STAMP_SCRIPT.replace("__JSON_PATH__", tmp_path)
        _run_pcbnew_script(script)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    print(f"Parent board stamped to {output_pcb} (subprocess)")
    return output_pcb


_PARENT_STAMP_SCRIPT = r"""
import json, pcbnew

with open("__JSON_PATH__") as _f:
    _data = json.load(_f)

_pcb_path = _data["pcb_path"]
_out_path = _data["output_path"]
_components = _data["components"]
_traces = _data["traces"]
_vias = _data["vias"]
_silkscreen = _data.get("silkscreen", [])
_keepouts = _data.get("keepouts", [])

_LAYER_MAP = {0: pcbnew.F_Cu, 1: pcbnew.B_Cu}
_LAYER_NAME_MAP = {"F.Cu": pcbnew.F_Cu, "B.Cu": pcbnew.B_Cu}

board = pcbnew.LoadBoard(_pcb_path)

# --- rewrite board outline if provided ---
_outline = _data.get("outline")
if _outline:
    _width_mm = max(1.0, _outline["br_x"] - _outline["tl_x"])
    _height_mm = max(1.0, _outline["br_y"] - _outline["tl_y"])
    _left = pcbnew.FromMM(_outline["tl_x"])
    _top = pcbnew.FromMM(_outline["tl_y"])
    _right = pcbnew.FromMM(_outline["tl_x"] + _width_mm)
    _bottom = pcbnew.FromMM(_outline["tl_y"] + _height_mm)

    _edge_remove = [d for d in board.GetDrawings() if d.GetLayer() == pcbnew.Edge_Cuts]
    for d in _edge_remove:
        board.Remove(d)

    _corners = [(_left, _top), (_right, _top), (_right, _bottom), (_left, _bottom)]
    for _i in range(4):
        _seg = pcbnew.PCB_SHAPE(board)
        _seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        _seg.SetLayer(pcbnew.Edge_Cuts)
        _seg.SetWidth(pcbnew.FromMM(0.05))
        _x1, _y1 = _corners[_i]
        _x2, _y2 = _corners[(_i + 1) % 4]
        _seg.SetStart(pcbnew.VECTOR2I(_x1, _y1))
        _seg.SetEnd(pcbnew.VECTOR2I(_x2, _y2))
        board.Add(_seg)

# --- build component lookup ---
_comp_map = {c["ref"]: c for c in _components}

# --- move footprints to composed positions (keep all footprints) ---
for _fp in board.Footprints():
    _ref = _fp.GetReferenceAsString()
    _comp = _comp_map.get(_ref)
    if _comp is None:
        continue
    _cur_layer = 1 if _fp.GetLayer() == pcbnew.B_Cu else 0
    if _comp["layer"] != _cur_layer:
        _fp.Flip(_fp.GetPosition(), False)
    _fp.SetPosition(
        pcbnew.VECTOR2I(pcbnew.FromMM(_comp["x"]), pcbnew.FromMM(_comp["y"]))
    )
    _fp.SetOrientationDegrees(_comp["rotation"])

# --- clear existing tracks ---
_tr = list(board.GetTracks())
for _t in _tr:
    board.Remove(_t)

# --- clear existing zones ---
_zr = [z for z in board.Zones() if not z.GetIsRuleArea()]
for _z in _zr:
    board.Remove(_z)

# --- strip all non-outline drawings from the source/template board ---
# The parent stamp rebuilds the board from scratch: tracks, zones, and
# drawings are all cleared, then recreated from composed child geometry.
# Only Edge_Cuts outlines survive (matching the leaf stamp in adapter.py).
# Source-board silkscreen (group labels, boundary shapes) would otherwise
# duplicate the composed child silkscreen stamped below.
_draw_remove = []
for _d in board.GetDrawings():
    try:
        if _d.GetLayer() == pcbnew.Edge_Cuts:
            continue
    except Exception:
        pass
    _draw_remove.append(_d)
for _d in _draw_remove:
    board.Remove(_d)

# --- resolve net code ---
_netinfo = board.GetNetInfo()

def _resolve_net(name):
    if not name:
        return 0
    ni = _netinfo.GetNetItem(name)
    if ni is None:
        return 0
    try:
        return int(ni.GetNetCode())
    except Exception:
        return 0

# --- recreate child traces ---
for _t in _traces:
    _s = pcbnew.PCB_TRACK(board)
    _s.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(_t["start_x"]), pcbnew.FromMM(_t["start_y"])))
    _s.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(_t["end_x"]), pcbnew.FromMM(_t["end_y"])))
    _s.SetLayer(_LAYER_NAME_MAP.get(_t["layer"], pcbnew.F_Cu))
    _s.SetWidth(pcbnew.FromMM(_t["width"]))
    _nc = _resolve_net(_t["net_name"])
    if _nc > 0:
        _s.SetNetCode(_nc)
    board.Add(_s)

# --- recreate vias ---
for _v in _vias:
    _tv = pcbnew.PCB_VIA(board)
    _tv.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(_v["x"]), pcbnew.FromMM(_v["y"])))
    _tv.SetDrill(pcbnew.FromMM(_v["drill"]))
    try:
        _tv.SetWidth(pcbnew.FromMM(_v["size"]))
    except TypeError:
        _tv.SetWidth(pcbnew.F_Cu, pcbnew.FromMM(_v["size"]))
    _nc = _resolve_net(_v["net_name"])
    if _nc > 0:
        _tv.SetNetCode(_nc)
    board.Add(_tv)

# --- stamp silkscreen graphics ---
_SILK_LAYER_MAP = {"F.SilkS": pcbnew.F_SilkS, "B.SilkS": pcbnew.B_SilkS}

for _silk in _silkscreen:
    _slayer = _SILK_LAYER_MAP.get(_silk.get("layer", "F.SilkS"), pcbnew.F_SilkS)
    if _silk["kind"] == "poly":
        _shape = pcbnew.PCB_SHAPE(board)
        _shape.SetShape(pcbnew.SHAPE_T_POLY)
        _shape.SetLayer(_slayer)
        _shape.SetFilled(False)
        _shape.SetWidth(pcbnew.FromMM(_silk.get("stroke_width", 0.15)))
        _poly = pcbnew.VECTOR_VECTOR2I()
        for _pt in _silk.get("points", []):
            _poly.append(pcbnew.VECTOR2I(pcbnew.FromMM(_pt["x"]), pcbnew.FromMM(_pt["y"])))
        _shape.SetPolyPoints(_poly)
        board.Add(_shape)
    elif _silk["kind"] == "text":
        _txt = pcbnew.PCB_TEXT(board)
        _txt.SetText(_silk.get("text", ""))
        _txt.SetLayer(_slayer)
        _pos = _silk.get("pos", {"x": 0, "y": 0})
        _txt.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(_pos["x"]), pcbnew.FromMM(_pos["y"])))
        _txt.SetTextSize(pcbnew.VECTOR2I(
            pcbnew.FromMM(_silk.get("font_width", 1.0)),
            pcbnew.FromMM(_silk.get("font_height", 1.0)),
        ))
        _txt.SetTextThickness(pcbnew.FromMM(_silk.get("font_thickness", 0.15)))
        _txt.SetHorizJustify(pcbnew.GR_TEXT_H_ALIGN_LEFT)
        board.Add(_txt)

# --- stamp parent-local rule-area keepouts (mounting holes etc.) ---
# These zones survive freerouting_runner.strip_zones() because that helper
# preserves GetIsRuleArea()==True zones. The DSN export for FreeRouting
# reads rule-area keepouts so no track or via can be placed inside them.
_KEEPOUT_LAYERS = [pcbnew.F_Cu, pcbnew.B_Cu]
for _ko in _keepouts:
    for _layer in _KEEPOUT_LAYERS:
        _zone = pcbnew.ZONE(board)
        _zone.SetLayer(_layer)
        _zone.SetIsRuleArea(True)
        _zone.SetDoNotAllowTracks(True)
        _zone.SetDoNotAllowVias(True)
        _zone.SetDoNotAllowPads(True)
        _zone.SetDoNotAllowCopperPour(True)
        _outline = _zone.Outline()
        _outline.NewOutline()
        _x1 = pcbnew.FromMM(_ko["tl_x"])
        _y1 = pcbnew.FromMM(_ko["tl_y"])
        _x2 = pcbnew.FromMM(_ko["br_x"])
        _y2 = pcbnew.FromMM(_ko["br_y"])
        _outline.Append(_x1, _y1)
        _outline.Append(_x2, _y1)
        _outline.Append(_x2, _y2)
        _outline.Append(_x1, _y2)
        board.Add(_zone)

board.BuildConnectivity()
board.Save(_out_path)
print("OK")
"""


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
        f"mode={state.mode}",
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
        "mode": state.mode,
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
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

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
    debug_path.write_text(
        json.dumps(debug_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

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
        "--mode",
        choices=("row", "column", "grid", "packed"),
        default="packed",
        help="Initial rigid composition mode (default: packed)",
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

        state, transformed_payloads = _compose_artifacts(
            loaded_artifacts,
            mode=args.mode,
            spacing_mm=max(0.0, args.spacing_mm),
            rotation_step_deg=args.rotation_step_deg,
            parent_definition=parent_definition,
            pcb_path=Path(args.pcb) if args.pcb else None,
        )

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
            geometry_validation = _validate_parent_geometry(
                state,
                overhangs=cfg.get("parent_overhang_mm", {}),
            )
            if not geometry_validation.get("accepted", False):
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
                    "error: parent composition geometry validation failed before stamping",
                    file=sys.stderr,
                )
                return 1

            # Stamp
            stamped_pcb = _stamp_parent_board(state, pcb_path, project_dir, cfg)
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

            if args.route:
                routing_result = _route_parent_board(
                    stamped_pcb, state, project_dir, cfg
                )
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
                    else:
                        reasons = validation.get("rejection_reasons", [])
                        reason_str = ', '.join(reasons) if reasons else 'unknown'
                        print(
                            f"parent_status      : rejected ({reason_str})"
                        )
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
