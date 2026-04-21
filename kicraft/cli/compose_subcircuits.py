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
    ParentComposition,
    build_parent_composition,
    estimate_parent_board_size,
    packed_extents_outline,
    derive_attachment_constraints,
    constrained_child_offset,
    child_layer_envelopes,
    can_overlap,
    constraint_aware_outline,
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

    def _append_entry(index: int, artifact, origin: Point):
        rotation = (index * rotation_step_deg) % 360.0
        transformed = transform_loaded_artifact(
            artifact,
            origin=origin,
            rotation=rotation,
        )

        entry = CompositionEntry(
            artifact_dir=artifact.artifact_dir,
            sheet_name=artifact.sheet_name,
            instance_path=artifact.instance_path,
            origin=origin,
            rotation=rotation,
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
                rotation=rotation,
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
            parent_local = extract_parent_local_components(str(pcb_path), loaded_artifacts)
        except Exception as e:
            logger.warning(f"Could not load config/local components: {e}")

    constraints = derive_attachment_constraints(loaded_artifacts, parent_local, component_zones, cfg)
    logger.info("composition: %d attachment constraints derived", len(constraints))

    child_constraints = [c for c in constraints if c.source == "child_artifact" and c.child_index is not None]
    local_constraints = [c for c in constraints if c.source == "parent_local"]

    constrained_indices = {c.child_index for c in child_constraints}
    unconstrained_artifacts = [(i, art) for i, art in enumerate(loaded_artifacts) if i not in constrained_indices]

    placed_envelopes = []

    parent_min_x, parent_min_y = 0.0, 0.0
    parent_max_x, parent_max_y = 0.0, 0.0

    for c in child_constraints:
        art_idx = int(c.child_index) if c.child_index is not None else 0
        art = loaded_artifacts[art_idx]
        rot = (art_idx * rotation_step_deg) % 360.0

        origin = constrained_child_offset(
            artifact=art,
            constraint=c,
            rotation_deg=rot,
            parent_outline_min=Point(parent_min_x, parent_min_y),
            parent_outline_max=Point(parent_max_x, parent_max_y),
        )

        transformed = _append_entry(art_idx, art, origin)
        placed_envelopes.append(child_layer_envelopes(transformed))

        w, h = transformed.instance.transformed_bbox
        if parent_min_x == 0.0 and parent_max_x == 0.0 and parent_min_y == 0.0 and parent_max_y == 0.0:
            parent_min_x = origin.x
            parent_min_y = origin.y
            parent_max_x = origin.x + w
            parent_max_y = origin.y + h
        else:
            parent_min_x = min(parent_min_x, origin.x)
            parent_min_y = min(parent_min_y, origin.y)
            parent_max_x = max(parent_max_x, origin.x + w)
            parent_max_y = max(parent_max_y, origin.y + h)

    def find_non_overlapping_origin(start_x, start_y, transformed_art) -> Point:
        cursor_x, cursor_y = start_x, start_y
        w, h = transformed_art.instance.transformed_bbox
        
        while True:
            env = child_layer_envelopes(transformed_art)
            def shift_env(e):
                if e is None:
                    return None
                return (Point(e[0].x + cursor_x, e[0].y + cursor_y), Point(e[1].x + cursor_x, e[1].y + cursor_y))
            
            shifted_env = (shift_env(env[0]), shift_env(env[1]), shift_env(env[2]))
            
            conflict = False
            for pe in placed_envelopes:
                if not can_overlap(pe, shifted_env):
                    conflict = True
                    break
                    
            if not conflict:
                return Point(cursor_x, cursor_y)
                
            cursor_x += spacing_mm
            if cursor_x > parent_max_x + spacing_mm:
                cursor_x = 0.0
                cursor_y += spacing_mm
            if cursor_y > 1000.0:
                return Point(start_x, start_y)

    if mode == "row":
        cursor_x = 0.0
        cursor_y = 0.0

        for index, artifact in unconstrained_artifacts:
            rot = (index * rotation_step_deg) % 360.0
            t_art = transform_loaded_artifact(artifact, origin=Point(0,0), rotation=rot)
            
            origin = find_non_overlapping_origin(cursor_x, cursor_y, t_art)
            transformed = _append_entry(index, artifact, origin)
            placed_envelopes.append(child_layer_envelopes(transformed))
            
            cursor_x = origin.x + entries[-1].transformed_bbox[0] + spacing_mm

        packing_metadata = {
            "strategy": "row",
            "row_count": 1 if entries else 0,
            "sort_key": "input_order",
        }

    elif mode == "column":
        cursor_x = 0.0
        cursor_y = 0.0

        for index, artifact in unconstrained_artifacts:
            rot = (index * rotation_step_deg) % 360.0
            t_art = transform_loaded_artifact(artifact, origin=Point(0,0), rotation=rot)
            
            origin = find_non_overlapping_origin(cursor_x, cursor_y, t_art)
            transformed = _append_entry(index, artifact, origin)
            placed_envelopes.append(child_layer_envelopes(transformed))
            
            cursor_y = origin.y + entries[-1].transformed_bbox[1] + spacing_mm

        packing_metadata = {
            "strategy": "column",
            "column_count": 1 if entries else 0,
            "sort_key": "input_order",
        }

    elif mode == "grid":
        count = len(unconstrained_artifacts)
        cols = max(1, math.ceil(math.sqrt(count)))

        max_width = 0.0
        max_height = 0.0
        for _, artifact in unconstrained_artifacts:
            max_width = max(max_width, artifact.layout.width)
            max_height = max(max_height, artifact.layout.height)

        cell_w = max_width + spacing_mm
        cell_h = max_height + spacing_mm

        for idx, (original_index, artifact) in enumerate(unconstrained_artifacts):
            row = idx // cols
            col = idx % cols
            
            rot = (original_index * rotation_step_deg) % 360.0
            t_art = transform_loaded_artifact(artifact, origin=Point(0,0), rotation=rot)
            
            proposed = Point(col * cell_w, row * cell_h)
            origin = find_non_overlapping_origin(proposed.x, proposed.y, t_art)
            
            transformed = _append_entry(original_index, artifact, origin)
            placed_envelopes.append(child_layer_envelopes(transformed))

        packing_metadata = {
            "strategy": "grid",
            "grid_columns": cols,
            "grid_rows": math.ceil(count / cols) if count else 0,
            "cell_width_mm": cell_w,
            "cell_height_mm": cell_h,
            "max_child_width_mm": max_width,
            "max_child_height_mm": max_height,
            "sort_key": "input_order",
        }

    elif mode == "packed":
        indexed_artifacts = list(unconstrained_artifacts)
        indexed_artifacts.sort(
            key=lambda item: (
                -(item[1].layout.width * item[1].layout.height),
                -item[1].layout.width,
                -item[1].layout.height,
                item[0],
            )
        )

        total_area = sum(
            max(0.0, artifact.layout.width) * max(0.0, artifact.layout.height)
            for _, artifact in indexed_artifacts
        )
        max_child_width = max(
            (max(0.0, artifact.layout.width) for _, artifact in indexed_artifacts),
            default=0.0,
        )
        child_bboxes = [
            (max(0.0, artifact.layout.width), max(0.0, artifact.layout.height))
            for _, artifact in indexed_artifacts
        ]
        estimated_w, _ = estimate_parent_board_size(
            child_bboxes, margin_mm=spacing_mm
        )
        target_row_width = max(
            max_child_width + spacing_mm,
            estimated_w if estimated_w > 0.0 else max_child_width,
        )

        row_count = 0
        row_y = 0.0
        row_x = 0.0
        row_height = 0.0
        row_widths: list[float] = []
        row_heights: list[float] = []
        row_item_counts: list[int] = []
        current_row_items = 0

        for sorted_pos, (original_index, artifact) in enumerate(indexed_artifacts):
            width = max(0.0, artifact.layout.width)

            should_wrap = (
                current_row_items > 0
                and row_x > 0.0
                and (row_x + width) > target_row_width
            )
            if should_wrap:
                row_widths.append(max(0.0, row_x - spacing_mm))
                row_heights.append(row_height)
                row_item_counts.append(current_row_items)
                row_count += 1
                row_y += row_height + spacing_mm
                row_x = 0.0
                row_height = 0.0
                current_row_items = 0

            rot = (original_index * rotation_step_deg) % 360.0
            t_art = transform_loaded_artifact(artifact, origin=Point(0,0), rotation=rot)
            
            proposed = Point(row_x, row_y)
            origin = find_non_overlapping_origin(proposed.x, proposed.y, t_art)
            
            transformed = _append_entry(original_index, artifact, origin)
            placed_envelopes.append(child_layer_envelopes(transformed))

            row_x = origin.x + entries[-1].transformed_bbox[0] + spacing_mm
            row_height = max(row_height, entries[-1].transformed_bbox[1])
            current_row_items += 1

        if current_row_items > 0:
            row_widths.append(max(0.0, row_x - spacing_mm))
            row_heights.append(row_height)
            row_item_counts.append(current_row_items)
            row_count += 1

        packing_metadata = {
            "strategy": "packed_rows",
            "sort_key": "area_desc_width_desc_height_desc",
            "target_row_width_mm": target_row_width,
            "estimated_total_child_area_mm2": total_area,
            "max_child_width_mm": max_child_width,
            "row_count": row_count,
            "row_widths_mm": row_widths,
            "row_heights_mm": row_heights,
            "row_item_counts": row_item_counts,
        }

    else:
        raise ValueError(f"Unsupported composition mode: {mode}")

    child_bboxes = [e.transformed_bbox for e in entries]
    child_origins_pt = [e.origin for e in entries]
    child_origins = [(e.origin.x, e.origin.y) for e in entries]
    
    if constraints:
        centers = {}
        for c in child_constraints:
            art_idx = int(c.child_index) if c.child_index is not None else 0
            art = loaded_artifacts[art_idx]
            comp = art.layout.components.get(c.ref)
            if comp:
                center = comp.body_center if comp.body_center else comp.pos
                rad = math.radians((art_idx * rotation_step_deg) % 360.0)
                rx = center.x * math.cos(rad) - center.y * math.sin(rad)
                ry = center.x * math.sin(rad) + center.y * math.cos(rad)
                origin = next(e.origin for e in entries if e.instance_path == art.instance_path)
                centers[c.ref] = Point(origin.x + rx, origin.y + ry)
                
        exact_outline = constraint_aware_outline(
            child_origins=child_origins_pt,
            child_bboxes=child_bboxes,
            attachment_constraints=constraints,
            constrained_ref_world_centers=centers,
            margin_mm=spacing_mm
        )
        
        for c in local_constraints:
            comp = parent_local.get(c.ref)
            if not comp:
                continue
            
            target_x = comp.pos.x
            target_y = comp.pos.y
            min_pt, max_pt = exact_outline
            
            if c.target == "corner":
                if c.value == "top-left":
                    target_x = min_pt.x + c.inward_keep_in_mm
                    target_y = min_pt.y + c.inward_keep_in_mm
                elif c.value == "top-right":
                    target_x = max_pt.x - c.inward_keep_in_mm
                    target_y = min_pt.y + c.inward_keep_in_mm
                elif c.value == "bottom-left":
                    target_x = min_pt.x + c.inward_keep_in_mm
                    target_y = max_pt.y - c.inward_keep_in_mm
                elif c.value == "bottom-right":
                    target_x = max_pt.x - c.inward_keep_in_mm
                    target_y = max_pt.y - c.inward_keep_in_mm
            elif c.target == "edge":
                if c.value == "left":
                    target_x = min_pt.x + c.inward_keep_in_mm
                elif c.value == "right":
                    target_x = max_pt.x - c.inward_keep_in_mm
                elif c.value == "top":
                    target_y = min_pt.y + c.inward_keep_in_mm
                elif c.value == "bottom":
                    target_y = max_pt.y - c.inward_keep_in_mm

            comp.body_center = Point(target_x, target_y)
            comp.pos = Point(target_x, target_y)
    else:
        exact_outline = packed_extents_outline(child_origins, child_bboxes)
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
    )

    import itertools

    edge_attachment_satisfied = {}
    mounting_hole_keep_in_satisfied = {}

    for c in constraints:
        expected_x = None
        expected_y = None
        actual_x = None
        actual_y = None
        
        min_pt, max_pt = exact_outline

        if c.source == "child_artifact":
            art_idx = c.child_index if c.child_index is not None else 0
            art = loaded_artifacts[art_idx]
            comp = art.layout.components.get(c.ref)
            if not comp:
                continue

            center = comp.body_center if comp.body_center else comp.pos
            rad = math.radians((art_idx * rotation_step_deg) % 360.0)
            rx = center.x * math.cos(rad) - center.y * math.sin(rad)
            ry = center.x * math.sin(rad) + center.y * math.cos(rad)

            try:
                origin = next(e.origin for e in entries if e.instance_path == art.instance_path)
            except StopIteration:
                continue

            actual_x = origin.x + rx
            actual_y = origin.y + ry
        else:
            comp = parent_local.get(c.ref)
            if not comp:
                continue
            actual_x = comp.pos.x
            actual_y = comp.pos.y

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


    def bbox_disjoint(a, b):
        if a is None or b is None: return True
        return a[1].x <= b[0].x or b[1].x <= a[0].x or a[1].y <= b[0].y or b[1].y <= a[0].y

    same_side_overlap_conflicts = []
    tht_keepout_violations = []

    for i, j in itertools.combinations(range(len(placed_envelopes)), 2):
        env_a = placed_envelopes[i]
        env_b = placed_envelopes[j]
        art_a = loaded_artifacts[i]
        art_b = loaded_artifacts[j]

        bbox_a = entries[i].transformed_bbox
        origin_a = entries[i].origin
        rect_a = (Point(origin_a.x, origin_a.y), Point(origin_a.x + bbox_a[0], origin_a.y + bbox_a[1]))

        bbox_b = entries[j].transformed_bbox
        origin_b = entries[j].origin
        rect_b = (Point(origin_b.x, origin_b.y), Point(origin_b.x + bbox_b[0], origin_b.y + bbox_b[1]))

        if not bbox_disjoint(rect_a, rect_b):
            if not can_overlap(env_a, env_b):
                a_label = getattr(art_a, "label", getattr(art_a, "slug", getattr(art_a, "sheet_name", f"child[{i}]")))
                b_label = getattr(art_b, "label", getattr(art_b, "slug", getattr(art_b, "sheet_name", f"child[{j}]")))

                a_front, a_back, a_tht = env_a
                b_front, b_back, b_tht = env_b

                if not bbox_disjoint(a_tht, b_front) or not bbox_disjoint(a_tht, b_back) or                    not bbox_disjoint(b_tht, a_front) or not bbox_disjoint(b_tht, a_back) or                    not bbox_disjoint(a_tht, b_tht):
                    tht_keepout_violations.append((a_label, b_label))
                elif not bbox_disjoint(a_front, b_front) or not bbox_disjoint(a_back, b_back):
                    same_side_overlap_conflicts.append((a_label, b_label))

    validation_data = {
        "edge_attachment_satisfied": edge_attachment_satisfied,
        "mounting_hole_keep_in_satisfied": mounting_hole_keep_in_satisfied,
        "same_side_overlap_conflicts": same_side_overlap_conflicts,
        "tht_keepout_violations": tht_keepout_violations,
        "constraint_count": len(constraints),
        "parent_local_count": len(parent_local),
    }

    unsatisfied_edges = sum(1 for ok in edge_attachment_satisfied.values() if not ok)
    logger.info("composition: %d constraints, %d unsatisfied edges, %d overlap conflicts, %d THT violations", 
        len(constraints), unsatisfied_edges, len(same_side_overlap_conflicts), len(tht_keepout_violations))


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
    min_x = tl.x - margin
    min_y = tl.y - margin
    max_x = br.x + margin
    max_y = br.y + margin

    outside_components: list[dict[str, Any]] = []
    outside_pads = 0
    outside_traces = 0
    outside_vias = 0

    for ref, comp in (composition.board_state.components or {}).items():
        bbox_tl, bbox_br = comp.bbox()
        component_outside = (
            bbox_tl.x < min_x
            or bbox_tl.y < min_y
            or bbox_br.x > max_x
            or bbox_br.y > max_y
        )
        pad_outside_count = 0
        for pad in comp.pads:
            if (
                pad.pos.x < min_x
                or pad.pos.x > max_x
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

    geometry_validation = _validate_parent_geometry(state)
    if not geometry_validation.get("accepted", False):
        raise RuntimeError(
            "Parent composition geometry is invalid before stamping: "
            f"outside_components={geometry_validation.get('outside_component_count', 0)} "
            f"outside_pads={geometry_validation.get('outside_pad_count', 0)} "
            f"outside_traces={geometry_validation.get('outside_trace_count', 0)} "
            f"outside_vias={geometry_validation.get('outside_via_count', 0)}"
        )

    payload = {
        "pcb_path": str(output_pcb),
        "output_path": str(output_pcb),
        "components": components_json,
        "traces": traces_json,
        "vias": vias_json,
        "silkscreen": silkscreen_json,
        "outline": outline_data,
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
    if _fp.IsLocked():
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
            geometry_validation = _validate_parent_geometry(state)
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
