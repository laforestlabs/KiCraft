"""Parent composition state builder for rigid solved subcircuits.

This module is the first composition-side builder for the subcircuits redesign.
It takes already-solved rigid child subcircuits, applies rigid transforms, and
builds a parent-level composition state that can later be used for:

- parent-level placement optimization
- inter-subcircuit routing
- top-level hierarchical assembly
- rigid child stamping into a final board state

Current scope:
- accept solved child layouts or loaded solved artifacts
- instantiate rigid child modules with translation + rotation
- transform child geometry into parent coordinates
- merge transformed child geometry into a parent `BoardState`
- preserve child internals exactly (components, pads, traces, vias)
- expose transformed interface anchors for parent-level routing
- include optional parent-local components and nets
- build a `HierarchyLevelState` plus a merged `BoardState`

This module intentionally does not yet:
- optimize parent placement
- merge copper zones
- support whole-subcircuit flipping to the opposite board side
- recursively solve parent sheets end-to-end

Those capabilities belong to later milestones.
"""

from __future__ import annotations

import copy
import importlib
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .subcircuit_instances import (
    LoadedSubcircuitArtifact,
    TransformedSubcircuit,
    instantiate_subcircuit,
    transform_loaded_artifact,
    transform_subcircuit_instance,
)
from .types import (
    BoardState,
    Component,
    HierarchyLevelState,
    InterfaceAnchor,
    Net,
    Point,
    SilkscreenElement,
    SubCircuitDefinition,
    SubCircuitInstance,
    SubCircuitLayout,
    TraceSegment,
    Via,
)


@dataclass(slots=True)
class ChildPlacement:
    """Rigid placement request for one solved child subcircuit."""

    layout: SubCircuitLayout
    origin: Point
    rotation: float = 0.0

    @property
    def instance_path(self) -> str:
        return self.layout.subcircuit_id.instance_path


@dataclass(slots=True)
class ChildArtifactPlacement:
    """Rigid placement request for one loaded solved artifact."""

    artifact: LoadedSubcircuitArtifact
    origin: Point
    rotation: float = 0.0

    @property
    def instance_path(self) -> str:
        return self.artifact.layout.subcircuit_id.instance_path


@dataclass(slots=True)
class ComposedChild:
    """One transformed rigid child inside a parent composition."""

    instance: SubCircuitInstance
    transformed: TransformedSubcircuit
    source: str = "layout"

    @property
    def instance_path(self) -> str:
        return self.instance.layout_id.instance_path

    @property
    def sheet_name(self) -> str:
        return self.instance.layout_id.sheet_name


@dataclass(slots=True)
class ParentCompositionScore:
    """Lightweight parent-level composition score and breakdown."""

    total: float
    breakdown: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParentComposition:
    """Complete parent composition result.

    Contains:
    - parent-level `HierarchyLevelState`
    - merged rigid child geometry as a `BoardState`
    - transformed child anchor maps for later routing
    """

    hierarchy_state: HierarchyLevelState
    board_state: BoardState
    composed_children: list[ComposedChild] = field(default_factory=list)
    child_anchor_maps: dict[str, dict[str, InterfaceAnchor]] = field(
        default_factory=dict
    )
    inferred_interconnect_nets: dict[str, Net] = field(default_factory=dict)
    score: ParentCompositionScore | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def child_count(self) -> int:
        return len(self.composed_children)

    @property
    def component_count(self) -> int:
        return len(self.board_state.components)

    @property
    def trace_count(self) -> int:
        return len(self.board_state.traces)

    @property
    def via_count(self) -> int:
        return len(self.board_state.vias)


@dataclass(frozen=True)
class LeafBlockerSet:
    """Sparse per-feature keepouts for one leaf in local coordinates."""

    front_pads: tuple[tuple[Point, Point], ...]
    back_pads: tuple[tuple[Point, Point], ...]
    tht_drills: tuple[tuple[Point, Point], ...]
    leaf_outline: tuple[Point, Point]


@dataclass(frozen=True)
class AttachmentConstraint:
    ref: str
    target: Literal["edge", "corner", "zone"]
    value: str
    inward_keep_in_mm: float
    outward_overhang_mm: float
    source: Literal["child_artifact", "parent_local"]
    child_index: int | None
    strict: bool = True


@dataclass(slots=True)
class PlacementConstraintEntry:
    constraint: AttachmentConstraint
    local_anchor_offset: Point


@dataclass(slots=True)
class PlacementModel:
    rotation: float
    transformed: TransformedSubcircuit
    layer_envelopes: tuple[
        list[tuple[Point, Point]],
        list[tuple[Point, Point]],
        list[tuple[Point, Point]],
    ]
    blocker_set: LeafBlockerSet | None = None
    constraint_entries: list[PlacementConstraintEntry] = field(default_factory=list)


@dataclass(slots=True)
class PlacementSpec:
    child_index: int
    instance_path: str
    rotation_candidates: list[float]
    all_rotation_candidates: list[float]
    constraints: list[AttachmentConstraint] = field(default_factory=list)
    models: dict[float, PlacementModel] = field(default_factory=dict)


@dataclass(slots=True)
class DerivedAttachmentConstraints:
    constraints: list[AttachmentConstraint] = field(default_factory=list)
    child_specs: dict[int, PlacementSpec] = field(default_factory=dict)
    parent_local_constraints: list[AttachmentConstraint] = field(default_factory=list)


def derive_attachment_constraints(
    loaded_artifacts: list[Any],
    parent_local_components: dict[str, Component],
    component_zones: dict[str, str],
    cfg: dict[str, Any],
    *,
    rotation_step_deg: float = 0.0,
) -> DerivedAttachmentConstraints:
    constraints: list[AttachmentConstraint] = []
    child_constraints: dict[int, list[AttachmentConstraint]] = {}
    logger = logging.getLogger(__name__)

    for ref, zone_spec in component_zones.items():
        if isinstance(zone_spec, dict):
            if not zone_spec:
                continue
            target = next(iter(zone_spec.keys()))
            value = zone_spec[target]
        elif isinstance(zone_spec, str):
            if ":" in zone_spec:
                target, value = zone_spec.split(":", 1)
            else:
                target = "edge"
                value = zone_spec
        else:
            continue

        if target not in ("edge", "corner", "zone"):
            target = "zone"

        source: Literal["child_artifact", "parent_local"] | None = None
        child_idx: int | None = None
        comp: Component | None = None

        for i, artifact in enumerate(loaded_artifacts):
            if ref in artifact.layout.components:
                source = "child_artifact"
                child_idx = i
                comp = artifact.layout.components[ref]
                break

        if source is None:
            if ref in parent_local_components:
                source = "parent_local"
                comp = parent_local_components[ref]

        if source is None or comp is None:
            logger.warning(
                "Component %s constrained to %s:%s but not found in any artifact or local components",
                ref,
                target,
                value,
            )
            continue

        is_hole = ref.startswith("H") or ("hole" in getattr(comp, "kind", "").lower())
        is_conn = ref.startswith("J") or ("connector" in getattr(comp, "kind", "").lower())

        if is_hole:
            inward = cfg.get("mounting_hole_keep_in_mm", 2.5)
            outward = 0.0
        elif is_conn:
            inset = cfg.get("connector_edge_inset_mm", 1.0)
            per_ref_overhang = cfg.get("parent_overhang_mm", {}).get(ref, 0.0)
            inward = max(0.0, inset)
            outward = max(0.0, -inset) + per_ref_overhang
        else:
            inward = 0.0
            outward = 0.0

        constraints.append(
            AttachmentConstraint(
                ref=ref,
                target=target,
                value=value,
                inward_keep_in_mm=inward,
                outward_overhang_mm=outward,
                source=source,
                child_index=child_idx,
                strict=target in ("edge", "corner"),
            )
        )
        constraint = constraints[-1]
        if constraint.source == "child_artifact" and constraint.child_index is not None:
            child_constraints.setdefault(int(constraint.child_index), []).append(constraint)

    child_specs: dict[int, PlacementSpec] = {}
    for child_index, grouped_constraints in child_constraints.items():
        artifact = loaded_artifacts[child_index]
        base_rotation = (child_index * rotation_step_deg) % 360.0
        all_rotation_candidates = [
            (base_rotation + delta) % 360.0 for delta in (0.0, 90.0, 180.0, 270.0)
        ]
        child_specs[child_index] = PlacementSpec(
            child_index=child_index,
            instance_path=artifact.instance_path,
            rotation_candidates=[all_rotation_candidates[0]],
            all_rotation_candidates=all_rotation_candidates,
            constraints=list(grouped_constraints),
            models=_build_models_for_artifact(artifact, grouped_constraints, all_rotation_candidates),
        )

    return DerivedAttachmentConstraints(
        constraints=constraints,
        child_specs=child_specs,
        parent_local_constraints=[
            constraint for constraint in constraints if constraint.source == "parent_local"
        ],
    )


def _build_models_for_artifact(
    artifact: Any,
    constraints: list[AttachmentConstraint],
    rotation_candidates: list[float],
) -> dict[float, PlacementModel]:
    models: dict[float, PlacementModel] = {}
    for rotation in rotation_candidates:
        transformed = transform_loaded_artifact(
            artifact,
            origin=Point(0.0, 0.0),
            rotation=rotation,
        )
        models[rotation] = PlacementModel(
            rotation=rotation,
            transformed=transformed,
            blocker_set=extract_leaf_blocker_set(artifact),
            layer_envelopes=child_layer_envelopes(transformed),
            constraint_entries=[
                PlacementConstraintEntry(
                    constraint=constraint,
                    local_anchor_offset=_compute_local_anchor_offset(
                        transformed,
                        constraint,
                        rotation,
                    ),
                )
                for constraint in constraints
            ],
        )
    return models


def expand_rotation_candidates(spec: PlacementSpec) -> None:
    spec.rotation_candidates = list(spec.all_rotation_candidates)


def _constraint_sides(constraint: AttachmentConstraint) -> list[str]:
    if constraint.target == "edge":
        return [constraint.value]
    if constraint.target == "corner":
        return [side for side in constraint.value.split("-") if side]
    if constraint.target == "zone" and constraint.value in {"left", "right", "top", "bottom"}:
        return [constraint.value]
    return []


def _compute_mounting_hole_anchor(component: Component) -> Point:
    if component.pads:
        pad_x = sum(pad.pos.x for pad in component.pads) / len(component.pads)
        pad_y = sum(pad.pos.y for pad in component.pads) / len(component.pads)
        return Point(pad_x, pad_y)
    return component.body_center if component.body_center is not None else component.pos


def _compute_local_anchor_offset(
    transformed: TransformedSubcircuit,
    constraint: AttachmentConstraint,
    rotation_deg: float,
) -> Point:
    component = transformed.transformed_components[constraint.ref]
    bbox_min, bbox_max = transformed.bounding_box

    if constraint.ref.startswith("H") or "hole" in getattr(component, "kind", "").lower():
        return _compute_mounting_hole_anchor(component)

    anchor = Point(
        (bbox_min.x + bbox_max.x) / 2.0,
        (bbox_min.y + bbox_max.y) / 2.0,
    )
    sides = _constraint_sides(constraint)

    for side in sides:
        if side == "left":
            anchor.x = bbox_min.x
        elif side == "right":
            anchor.x = bbox_max.x
        elif side == "top":
            anchor.y = bbox_min.y
        elif side == "bottom":
            anchor.y = bbox_max.y

    return anchor


def constrained_child_offset(
    placement_model: PlacementModel,
    parent_outline_min: Point,
    parent_outline_max: Point,
) -> Point:
    origin, _ = place_constrained_child(
        placement_model,
        parent_outline_min=parent_outline_min,
        parent_outline_max=parent_outline_max,
    )
    return origin


def _exact_target_coordinate(
    side: str,
    constraint: AttachmentConstraint,
    parent_outline_min: Point,
    parent_outline_max: Point,
) -> float:
    if side == "left":
        return parent_outline_min.x + constraint.inward_keep_in_mm - constraint.outward_overhang_mm
    if side == "right":
        return parent_outline_max.x - constraint.inward_keep_in_mm + constraint.outward_overhang_mm
    if side == "top":
        return parent_outline_min.y + constraint.inward_keep_in_mm - constraint.outward_overhang_mm
    if side == "bottom":
        return parent_outline_max.y - constraint.inward_keep_in_mm + constraint.outward_overhang_mm
    raise ValueError(f"Unsupported constrained side: {side}")


def _zone_band_interval(
    side: str,
    entry: PlacementConstraintEntry,
    parent_outline_min: Point,
    parent_outline_max: Point,
) -> tuple[float, float]:
    frame_width = max(0.0, parent_outline_max.x - parent_outline_min.x)
    frame_height = max(0.0, parent_outline_max.y - parent_outline_min.y)
    if side == "bottom":
        band_min = parent_outline_min.y + frame_height * 0.5
        band_max = parent_outline_max.y - entry.constraint.inward_keep_in_mm
        return (
            band_min - entry.local_anchor_offset.y,
            band_max - entry.local_anchor_offset.y,
        )
    if side == "top":
        band_min = parent_outline_min.y + entry.constraint.inward_keep_in_mm
        band_max = parent_outline_min.y + frame_height * 0.5
        return (
            band_min - entry.local_anchor_offset.y,
            band_max - entry.local_anchor_offset.y,
        )
    if side == "left":
        band_min = parent_outline_min.x + entry.constraint.inward_keep_in_mm
        band_max = parent_outline_min.x + frame_width * 0.5
        return (
            band_min - entry.local_anchor_offset.x,
            band_max - entry.local_anchor_offset.x,
        )
    if side == "right":
        band_min = parent_outline_min.x + frame_width * 0.5
        band_max = parent_outline_max.x - entry.constraint.inward_keep_in_mm
        return (
            band_min - entry.local_anchor_offset.x,
            band_max - entry.local_anchor_offset.x,
        )
    raise ValueError(f"Unsupported zone side: {side}")


def _resolve_exact_axis_origin(
    axis: Literal["x", "y"],
    entries: list[PlacementConstraintEntry],
    parent_outline_min: Point,
    parent_outline_max: Point,
    tolerance_mm: float = 1e-3,
) -> float | None:
    exact_targets: list[tuple[str, float]] = []
    zone_ranges: list[tuple[str, tuple[float, float]]] = []
    for entry in entries:
        for side in _constraint_sides(entry.constraint):
            if axis == "x" and side not in {"left", "right"}:
                continue
            if axis == "y" and side not in {"top", "bottom"}:
                continue
            if entry.constraint.strict:
                exact_targets.append(
                    (
                        entry.constraint.ref,
                        _exact_target_coordinate(
                            side,
                            entry.constraint,
                            parent_outline_min,
                            parent_outline_max,
                        )
                        - (entry.local_anchor_offset.x if axis == "x" else entry.local_anchor_offset.y),
                    )
                )
            else:
                zone_ranges.append(
                    (
                        entry.constraint.ref,
                        _zone_band_interval(
                            side,
                            entry,
                            parent_outline_min,
                            parent_outline_max,
                        ),
                    )
                )

    if exact_targets:
        values = [value for _, value in exact_targets]
        reference = values[0]
        if any(abs(value - reference) > tolerance_mm for value in values[1:]):
            detail = ", ".join(
                f"{ref}={value:.3f}" for ref, value in exact_targets
            )
            axis_name = "x" if axis == "x" else "y"
            raise ValueError(f"Conflicting attachment targets on {axis_name}-axis: {detail}")
        return reference

    if zone_ranges:
        lower = max(interval[0] for _, interval in zone_ranges)
        upper = min(interval[1] for _, interval in zone_ranges)
        if lower > upper + tolerance_mm:
            detail = ", ".join(
                f"{ref}=[{interval[0]:.3f},{interval[1]:.3f}]"
                for ref, interval in zone_ranges
            )
            axis_name = "x" if axis == "x" else "y"
            raise ValueError(f"Infeasible zone attachment band on {axis_name}-axis: {detail}")
        return upper

    return None


def place_constrained_child(
    placement_model: PlacementModel,
    *,
    parent_outline_min: Point,
    parent_outline_max: Point,
) -> tuple[Point, tuple[Point, Point]]:
    origin_x = _resolve_exact_axis_origin(
        "x",
        placement_model.constraint_entries,
        parent_outline_min,
        parent_outline_max,
    )
    origin_y = _resolve_exact_axis_origin(
        "y",
        placement_model.constraint_entries,
        parent_outline_min,
        parent_outline_max,
    )

    bbox_min, bbox_max = placement_model.transformed.bounding_box
    if origin_x is None:
        origin_x = parent_outline_min.x - bbox_min.x
    if origin_y is None:
        origin_y = parent_outline_min.y - bbox_min.y

    placed_bbox = (
        Point(origin_x + bbox_min.x, origin_y + bbox_min.y),
        Point(origin_x + bbox_max.x, origin_y + bbox_max.y),
    )
    return Point(origin_x, origin_y), placed_bbox


def validate_child_constraints(
    placement_model: PlacementModel,
    origin: Point,
    parent_outline_min: Point,
    parent_outline_max: Point,
    *,
    tolerance_mm: float = 1e-3,
) -> None:
    for entry in placement_model.constraint_entries:
        world_anchor = Point(
            origin.x + entry.local_anchor_offset.x,
            origin.y + entry.local_anchor_offset.y,
        )
        for side in _constraint_sides(entry.constraint):
            if entry.constraint.strict:
                expected = _exact_target_coordinate(
                    side,
                    entry.constraint,
                    parent_outline_min,
                    parent_outline_max,
                )
                actual = world_anchor.x if side in {"left", "right"} else world_anchor.y
                if abs(actual - expected) > tolerance_mm:
                    raise ValueError(
                        f"Constraint not satisfied for {entry.constraint.ref} on {side}: "
                        f"expected {expected:.3f}, got {actual:.3f}"
                    )
                continue

            lower, upper = _zone_band_interval(
                side,
                entry,
                parent_outline_min,
                parent_outline_max,
            )
            actual_origin = origin.x if side in {"left", "right"} else origin.y
            if actual_origin < lower - tolerance_mm or actual_origin > upper + tolerance_mm:
                raise ValueError(
                    f"Zone constraint not satisfied for {entry.constraint.ref} on {side}: "
                    f"expected origin in [{lower:.3f}, {upper:.3f}], got {actual_origin:.3f}"
                )


def build_parent_composition(
    parent_subcircuit: SubCircuitDefinition,
    *,
    child_placements: list[ChildPlacement] | None = None,
    child_artifact_placements: list[ChildArtifactPlacement] | None = None,
    local_components: dict[str, Component] | None = None,
    interconnect_nets: dict[str, Net] | None = None,
    board_outline: tuple[Point, Point] | None = None,
    constraints: dict[str, object] | None = None,
) -> ParentComposition:
    """Build a parent composition state from rigid solved children.

    Args:
        parent_subcircuit: Parent-level logical subcircuit definition.
        child_placements: Solved child layouts with rigid transforms.
        child_artifact_placements: Loaded solved artifacts with rigid transforms.
        local_components: Optional parent-local components.
        interconnect_nets: Optional parent-level interconnect nets.
        board_outline: Optional parent board outline. If omitted, derived from
            merged geometry.
        constraints: Optional parent-level composition constraints.

    Returns:
        `ParentComposition` containing:
        - merged rigid child geometry
        - parent-local components
        - parent-level hierarchy state
        - transformed child anchor maps

    Notes:
        - Child internals are preserved exactly.
        - Child refs must be globally unique across the composition.
        - Parent-local components are copied into the merged board state.
        - Interconnect nets are not routed here; they are only carried forward.
    """
    composed_children: list[ComposedChild] = []

    for placement in child_placements or []:
        instance = instantiate_subcircuit(
            placement.layout,
            origin=placement.origin,
            rotation=placement.rotation,
        )
        transformed = transform_subcircuit_instance(placement.layout, instance)
        composed_children.append(
            ComposedChild(
                instance=instance,
                transformed=transformed,
                source="layout",
            )
        )

    for placement in child_artifact_placements or []:
        transformed = transform_loaded_artifact(
            placement.artifact,
            origin=placement.origin,
            rotation=placement.rotation,
        )
        composed_children.append(
            ComposedChild(
                instance=transformed.instance,
                transformed=transformed,
                source="artifact",
            )
        )

    merged_components: dict[str, Component] = {}
    merged_traces: list[TraceSegment] = []
    merged_vias: list[Via] = []
    merged_silkscreen: list[SilkscreenElement] = []
    child_anchor_maps: dict[str, dict[str, InterfaceAnchor]] = {}

    for child in composed_children:
        _merge_child_geometry(
            child,
            merged_components,
            merged_traces,
            merged_vias,
            merged_silkscreen,
            child_anchor_maps,
        )

    for ref, comp in (local_components or {}).items():
        if ref in merged_components:
            raise ValueError(
                f"Parent-local component ref '{ref}' collides with a child component"
            )
        merged_components[ref] = copy.deepcopy(comp)

    inferred_interconnect_nets = _infer_parent_interconnect_nets(
        parent_subcircuit,
        composed_children,
        child_anchor_maps,
        local_components or {},
    )
    explicit_interconnect_nets = interconnect_nets or {}
    combined_interconnect_nets = _merge_interconnect_net_maps(
        inferred_interconnect_nets,
        explicit_interconnect_nets,
    )

    merged_nets = _build_merged_nets(
        merged_components,
        combined_interconnect_nets,
    )

    outline = board_outline or _derive_board_outline(
        merged_components,
        merged_traces,
        merged_vias,
        child_anchor_maps,
    )

    hierarchy_state = HierarchyLevelState(
        subcircuit=parent_subcircuit,
        child_instances=[child.instance for child in composed_children],
        local_components={
            ref: copy.deepcopy(comp) for ref, comp in (local_components or {}).items()
        },
        interconnect_nets={
            name: copy.deepcopy(net) for name, net in combined_interconnect_nets.items()
        },
        board_outline=outline,
        constraints=dict(constraints or {}),
    )

    board_state = BoardState(
        components=merged_components,
        nets=merged_nets,
        traces=merged_traces,
        vias=merged_vias,
        silkscreen=merged_silkscreen,
        board_outline=outline,
    )

    score = _score_parent_composition(
        parent_subcircuit,
        composed_children,
        child_anchor_maps,
        combined_interconnect_nets,
        board_state,
    )

    notes = [
        f"parent={parent_subcircuit.id.sheet_name}",
        f"child_count={len(composed_children)}",
        f"component_count={len(merged_components)}",
        f"trace_count={len(merged_traces)}",
        f"via_count={len(merged_vias)}",
        f"inferred_interconnect_nets={len(inferred_interconnect_nets)}",
        f"interconnect_nets={len(combined_interconnect_nets)}",
        f"score_total={score.total:.3f}",
    ]

    return ParentComposition(
        hierarchy_state=hierarchy_state,
        board_state=board_state,
        composed_children=composed_children,
        child_anchor_maps=child_anchor_maps,
        inferred_interconnect_nets=inferred_interconnect_nets,
        score=score,
        notes=notes,
    )


def composition_debug_dict(composition: ParentComposition) -> dict[str, Any]:
    """Return a JSON-serializable debug view of a parent composition."""
    tl, br = composition.board_state.board_outline
    return {
        "parent": {
            "sheet_name": composition.hierarchy_state.subcircuit.id.sheet_name,
            "sheet_file": composition.hierarchy_state.subcircuit.id.sheet_file,
            "instance_path": composition.hierarchy_state.subcircuit.id.instance_path,
        },
        "child_count": composition.child_count,
        "component_count": composition.component_count,
        "trace_count": composition.trace_count,
        "via_count": composition.via_count,
        "board_outline": {
            "top_left": {"x": tl.x, "y": tl.y},
            "bottom_right": {"x": br.x, "y": br.y},
            "width_mm": br.x - tl.x,
            "height_mm": br.y - tl.y,
        },
        "children": [
            {
                "sheet_name": child.sheet_name,
                "instance_path": child.instance_path,
                "origin": {
                    "x": child.instance.origin.x,
                    "y": child.instance.origin.y,
                },
                "rotation": child.instance.rotation,
                "source": child.source,
                "component_count": len(child.transformed.transformed_components),
                "trace_count": len(child.transformed.transformed_traces),
                "via_count": len(child.transformed.transformed_vias),
                "anchor_count": len(child.transformed.transformed_anchors),
            }
            for child in composition.composed_children
        ],
        "anchor_maps": {
            instance_path: {
                port_name: {
                    "x": anchor.pos.x,
                    "y": anchor.pos.y,
                    "layer": "B.Cu"
                    if getattr(anchor.layer, "name", "") == "BACK"
                    else "F.Cu",
                    "pad_ref": list(anchor.pad_ref) if anchor.pad_ref else None,
                }
                for port_name, anchor in anchors.items()
            }
            for instance_path, anchors in composition.child_anchor_maps.items()
        },
        "inferred_interconnect_nets": {
            name: {
                "pad_refs": [list(pad_ref) for pad_ref in net.pad_refs],
                "priority": net.priority,
                "width_mm": net.width_mm,
                "is_power": net.is_power,
            }
            for name, net in composition.inferred_interconnect_nets.items()
        },
        "score": {
            "total": composition.score.total if composition.score else 0.0,
            "breakdown": dict(composition.score.breakdown) if composition.score else {},
            "notes": list(composition.score.notes) if composition.score else [],
        },
        "notes": list(composition.notes),
    }


def composition_summary(composition: ParentComposition) -> str:
    """Return a compact one-line summary for logs/debug output."""
    tl, br = composition.board_state.board_outline
    width = br.x - tl.x
    height = br.y - tl.y
    score_total = composition.score.total if composition.score else 0.0
    interconnect_count = len(composition.hierarchy_state.interconnect_nets)
    return (
        f"{composition.hierarchy_state.subcircuit.id.sheet_name} "
        f"[{composition.hierarchy_state.subcircuit.id.instance_path}] "
        f"children={composition.child_count} "
        f"components={composition.component_count} "
        f"traces={composition.trace_count} "
        f"vias={composition.via_count} "
        f"interconnects={interconnect_count} "
        f"score={score_total:.1f} "
        f"size={width:.1f}x{height:.1f}mm"
    )


def child_anchor_map(
    composition: ParentComposition, instance_path: str
) -> dict[str, InterfaceAnchor]:
    """Return the transformed anchor map for one child instance path."""
    return dict(composition.child_anchor_maps.get(instance_path, {}))


def child_component_refs(
    composition: ParentComposition, instance_path: str
) -> list[str]:
    """Return component refs belonging to one composed child."""
    for child in composition.composed_children:
        if child.instance_path == instance_path:
            return sorted(child.transformed.transformed_components.keys())
    return []


def _merge_child_geometry(
    child: ComposedChild,
    merged_components: dict[str, Component],
    merged_traces: list[TraceSegment],
    merged_vias: list[Via],
    merged_silkscreen: list[SilkscreenElement],
    child_anchor_maps: dict[str, dict[str, InterfaceAnchor]],
) -> None:
    """Merge one transformed rigid child into the parent composition."""
    for ref, comp in child.transformed.transformed_components.items():
        if ref in merged_components:
            raise ValueError(
                f"Component ref collision while composing child '{child.sheet_name}': {ref}"
            )
        merged_components[ref] = copy.deepcopy(comp)

    merged_traces.extend(
        copy.deepcopy(trace) for trace in child.transformed.transformed_traces
    )
    merged_vias.extend(copy.deepcopy(via) for via in child.transformed.transformed_vias)
    merged_silkscreen.extend(
        copy.deepcopy(elem) for elem in child.transformed.transformed_silkscreen
    )

    child_anchor_maps[child.instance_path] = {
        anchor.port_name: copy.deepcopy(anchor)
        for anchor in child.transformed.transformed_anchors
    }


def _build_merged_nets(
    components: dict[str, Component],
    interconnect_nets: dict[str, Net],
) -> dict[str, Net]:
    """Build merged net map from component pads plus optional parent nets."""
    merged: dict[str, Net] = {}

    for comp in components.values():
        for pad in comp.pads:
            if not pad.net:
                continue
            net = merged.get(pad.net)
            if net is None:
                net = Net(name=pad.net)
                merged[pad.net] = net
            pad_ref = (pad.ref, pad.pad_id)
            if pad_ref not in net.pad_refs:
                net.pad_refs.append(pad_ref)

    for name, net in interconnect_nets.items():
        existing = merged.get(name)
        if existing is None:
            merged[name] = copy.deepcopy(net)
            continue

        existing.priority = max(existing.priority, net.priority)
        existing.width_mm = max(existing.width_mm, net.width_mm)
        existing.is_power = existing.is_power or net.is_power

        seen = set(existing.pad_refs)
        for pad_ref in net.pad_refs:
            if pad_ref not in seen:
                existing.pad_refs.append(pad_ref)
                seen.add(pad_ref)

    return merged


def _infer_parent_interconnect_nets(
    parent_subcircuit: SubCircuitDefinition,
    composed_children: list[ComposedChild],
    child_anchor_maps: dict[str, dict[str, InterfaceAnchor]],
    local_components: dict[str, Component],
) -> dict[str, Net]:
    """Infer parent interconnect nets from layout ports and local pads."""
    inferred: dict[str, Net] = {}
    child_by_path = {child.instance_path: child for child in composed_children}

    for child_id in parent_subcircuit.child_ids:
        child = child_by_path.get(child_id.instance_path)
        if child is None:
            continue

        for port in _child_interface_ports(child):
            if not port.net_name:
                continue
            pad_ref = _resolve_child_port_pad_ref(
                child,
                child_anchor_maps.get(child.instance_path, {}),
                port.name,
                port.net_name,
            )
            if pad_ref is None:
                continue
            _append_pad_ref(
                inferred,
                port.net_name,
                pad_ref,
            )

    for comp in local_components.values():
        for pad in comp.pads:
            if not pad.net:
                continue
            _append_pad_ref(
                inferred,
                pad.net,
                (pad.ref, pad.pad_id),
            )

    deduped: dict[str, Net] = {}
    for net in inferred.values():
        normalized_name = _normalize_net_name(net.name)
        existing = deduped.get(normalized_name)
        if existing is None:
            deduped[normalized_name] = Net(
                name=net.name,
                pad_refs=list(net.pad_refs),
                priority=net.priority,
                width_mm=net.width_mm,
                is_power=net.is_power,
            )
            continue

        existing.priority = max(existing.priority, net.priority)
        existing.width_mm = max(existing.width_mm, net.width_mm)
        existing.is_power = existing.is_power or net.is_power
        seen = set(existing.pad_refs)
        for pad_ref in net.pad_refs:
            if pad_ref not in seen:
                existing.pad_refs.append(pad_ref)
                seen.add(pad_ref)

    return {
        net.name: net
        for net in deduped.values()
        if len({ref for ref, _ in net.pad_refs}) >= 2
    }


def _append_pad_ref(
    inferred: dict[str, Net],
    net_name: str,
    pad_ref: tuple[str, str],
) -> None:
    """Append one pad ref into an inferred net, creating it if needed."""
    net = inferred.get(net_name)
    if net is None:
        net = Net(
            name=net_name,
            priority=1,
            width_mm=0.127,
            is_power=_looks_like_power_net(net_name),
        )
        inferred[net_name] = net
    if pad_ref not in net.pad_refs:
        net.pad_refs.append(pad_ref)


def _child_interface_ports(child: ComposedChild) -> list[Any]:
    """Return logical interface ports for one composed child from its layout."""
    return list(child.transformed.layout.ports)


def _resolve_child_port_pad_ref(
    child: ComposedChild,
    anchors: dict[str, InterfaceAnchor],
    port_name: str,
    net_name: str,
) -> tuple[str, str] | None:
    """Resolve a representative pad ref for one child port/net."""
    anchor = anchors.get(port_name)
    if anchor is not None and anchor.pad_ref:
        return anchor.pad_ref

    best_pad_ref: tuple[str, str] | None = None
    best_distance = float("inf")
    center = _child_center(child)

    for comp in child.transformed.transformed_components.values():
        for pad in comp.pads:
            if not _nets_match(pad.net, net_name):
                continue
            distance = pad.pos.dist(center)
            if distance < best_distance:
                best_distance = distance
                best_pad_ref = (pad.ref, pad.pad_id)

    return best_pad_ref


def _child_center(child: ComposedChild) -> Point:
    """Return the geometric center of one transformed child."""
    tl, br = child.transformed.bounding_box
    return Point((tl.x + br.x) / 2.0, (tl.y + br.y) / 2.0)


def _merge_interconnect_net_maps(
    inferred_nets: dict[str, Net],
    explicit_nets: dict[str, Net],
) -> dict[str, Net]:
    """Merge inferred and explicit parent interconnect nets."""
    merged = {name: copy.deepcopy(net) for name, net in inferred_nets.items()}

    for name, net in explicit_nets.items():
        existing = merged.get(name)
        if existing is None:
            merged[name] = copy.deepcopy(net)
            continue

        existing.priority = max(existing.priority, net.priority)
        existing.width_mm = max(existing.width_mm, net.width_mm)
        existing.is_power = existing.is_power or net.is_power

        seen = set(existing.pad_refs)
        for pad_ref in net.pad_refs:
            if pad_ref not in seen:
                existing.pad_refs.append(pad_ref)
                seen.add(pad_ref)

    return merged
def _score_parent_composition(
    parent_subcircuit: SubCircuitDefinition,
    composed_children: list[ComposedChild],
    child_anchor_maps: dict[str, dict[str, InterfaceAnchor]],
    interconnect_nets: dict[str, Net],
    board_state: BoardState,
) -> ParentCompositionScore:
    """Compute a lightweight parent-level composition score."""
    child_scores = [
        child.transformed.layout.score
        for child in composed_children
        if child.transformed.layout.score > 0.0
    ]
    avg_child_score = sum(child_scores) / len(child_scores) if child_scores else 0.0

    total_anchor_count = sum(len(anchors) for anchors in child_anchor_maps.values())
    connected_anchor_count = 0
    anchor_distance_total = 0.0
    anchor_distance_pairs = 0

    for net in interconnect_nets.values():
        anchor_points: list[Point] = []
        for pad_ref in net.pad_refs:
            anchor = _find_anchor_for_pad_ref(child_anchor_maps, pad_ref)
            if anchor is not None:
                anchor_points.append(anchor.pos)

        connected_anchor_count += len(anchor_points)
        if len(anchor_points) >= 2:
            for index in range(len(anchor_points)):
                for other_index in range(index + 1, len(anchor_points)):
                    anchor_distance_total += anchor_points[index].dist(
                        anchor_points[other_index]
                    )
                    anchor_distance_pairs += 1

    anchor_coverage = (
        connected_anchor_count / total_anchor_count if total_anchor_count else 1.0
    )
    avg_anchor_distance = (
        anchor_distance_total / anchor_distance_pairs if anchor_distance_pairs else 0.0
    )

    tl, br = board_state.board_outline
    board_area = max(1.0, (br.x - tl.x) * (br.y - tl.y))
    component_area = sum(comp.area for comp in board_state.components.values())
    area_utilization = min(1.0, component_area / board_area)

    child_score_component = max(0.0, min(100.0, avg_child_score))
    anchor_coverage_component = max(0.0, min(100.0, anchor_coverage * 100.0))
    interconnect_component = max(
        0.0,
        min(100.0, 100.0 - min(avg_anchor_distance, 100.0)),
    )
    utilization_component = max(0.0, min(100.0, area_utilization * 100.0))

    total = (
        child_score_component * 0.35
        + anchor_coverage_component * 0.20
        + interconnect_component * 0.20
        + utilization_component * 0.25
    )

    notes = [
        f"parent={parent_subcircuit.id.sheet_name}",
        f"child_score_avg={avg_child_score:.3f}",
        f"anchor_coverage={anchor_coverage:.3f}",
        f"avg_anchor_distance_mm={avg_anchor_distance:.3f}",
        f"area_utilization={area_utilization:.3f}",
        f"interconnect_nets={len(interconnect_nets)}",
    ]

    return ParentCompositionScore(
        total=total,
        breakdown={
            "child_layout_quality": child_score_component,
            "anchor_coverage": anchor_coverage_component,
            "interconnect_compactness": interconnect_component,
            "area_utilization": utilization_component,
        },
        notes=notes,
    )


def _find_anchor_for_pad_ref(
    child_anchor_maps: dict[str, dict[str, InterfaceAnchor]],
    pad_ref: tuple[str, str],
) -> InterfaceAnchor | None:
    """Find a transformed child anchor by backing pad reference."""
    for anchors in child_anchor_maps.values():
        for anchor in anchors.values():
            if anchor.pad_ref == pad_ref:
                return anchor
    return None


def _normalize_net_name(net_name: str) -> str:
    """Normalize schematic/PCB net names for logical interconnect matching."""
    return str(net_name or "").strip().lstrip("/").upper()


def _nets_match(left: str, right: str) -> bool:
    """Return True when two net names refer to the same logical net."""
    return _normalize_net_name(left) == _normalize_net_name(right)


def _looks_like_power_net(net_name: str) -> bool:
    """Heuristic power-net classifier for inferred parent interconnects."""
    upper = _normalize_net_name(net_name)
    return (
        "GND" in upper
        or "VCC" in upper
        or "VIN" in upper
        or "VBUS" in upper
        or "VBAT" in upper
        or upper.startswith("+")
        or upper.startswith("-")
        or "3V3" in upper
        or "5V" in upper
        or "12V" in upper
    )


def _rect_from_center(center: Point, half_w: float, half_h: float) -> tuple[Point, Point]:
    return (
        Point(center.x - half_w, center.y - half_h),
        Point(center.x + half_w, center.y + half_h),
    )


def _rect_area(rect: tuple[Point, Point]) -> float:
    return max(0.0, rect[1].x - rect[0].x) * max(0.0, rect[1].y - rect[0].y)


def _transform_local_point(point: Point, origin: Point, rotation_deg: float) -> Point:
    rotation = rotation_deg % 360.0
    if abs(rotation - 0.0) < 1e-9:
        return Point(point.x + origin.x, point.y + origin.y)
    if abs(rotation - 90.0) < 1e-9:
        return Point(-point.y + origin.x, point.x + origin.y)
    if abs(rotation - 180.0) < 1e-9:
        return Point(-point.x + origin.x, -point.y + origin.y)
    if abs(rotation - 270.0) < 1e-9:
        return Point(point.y + origin.x, -point.x + origin.y)
    theta = math.radians(rotation)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return Point(
        point.x * cos_theta - point.y * sin_theta + origin.x,
        point.x * sin_theta + point.y * cos_theta + origin.y,
    )


def _points_bbox(points: list[Point]) -> tuple[Point, Point]:
    min_x = min(point.x for point in points)
    min_y = min(point.y for point in points)
    max_x = max(point.x for point in points)
    max_y = max(point.y for point in points)
    return (Point(min_x, min_y), Point(max_x, max_y))


def _transform_rect(
    rect: tuple[Point, Point],
    origin: Point,
    rotation_deg: float,
) -> tuple[Point, Point]:
    min_point, max_point = rect
    corners = [
        Point(min_point.x, min_point.y),
        Point(max_point.x, min_point.y),
        Point(max_point.x, max_point.y),
        Point(min_point.x, max_point.y),
    ]
    return _points_bbox(
        [_transform_local_point(corner, origin, rotation_deg) for corner in corners]
    )


def _rects_intersect(a: tuple[Point, Point], b: tuple[Point, Point]) -> bool:
    return not _bbox_disjoint(a, b)


def _any_rect_overlap(
    rects_a: tuple[tuple[Point, Point], ...],
    origin_a: Point,
    rotation_a: float,
    rects_b: tuple[tuple[Point, Point], ...],
    origin_b: Point,
    rotation_b: float,
) -> bool:
    for rect_a in rects_a:
        world_a = _transform_rect(rect_a, origin_a, rotation_a)
        for rect_b in rects_b:
            world_b = _transform_rect(rect_b, origin_b, rotation_b)
            if _rects_intersect(world_a, world_b):
                return True
    return False


def _component_local_bbox(comp: Component) -> tuple[Point, Point]:
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
    return (Point(min_x, min_y), Point(max_x, max_y))


def _artifact_outline(artifact: LoadedSubcircuitArtifact) -> tuple[Point, Point]:
    transformed = transform_loaded_artifact(
        artifact,
        origin=Point(0.0, 0.0),
        rotation=0.0,
    )
    return transformed.bounding_box


def _coalesce_rects(
    rects: list[tuple[Point, Point]],
    *,
    max_rects: int = 200,
) -> tuple[tuple[Point, Point], ...]:
    if len(rects) <= max_rects:
        return tuple(rects)
    remaining = list(rects)
    merged: list[tuple[Point, Point]] = []
    while remaining:
        current_min, current_max = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            next_remaining: list[tuple[Point, Point]] = []
            for other_min, other_max in remaining:
                touches_x = other_min.x <= current_max.x and other_max.x >= current_min.x
                touches_y = other_min.y <= current_max.y and other_max.y >= current_min.y
                if touches_x and touches_y:
                    current_min = Point(
                        min(current_min.x, other_min.x),
                        min(current_min.y, other_min.y),
                    )
                    current_max = Point(
                        max(current_max.x, other_max.x),
                        max(current_max.y, other_max.y),
                    )
                    changed = True
                    continue
                next_remaining.append((other_min, other_max))
            remaining = next_remaining
        merged.append((current_min, current_max))
    return tuple(merged)


def _extract_blockers_from_layout(
    artifact: LoadedSubcircuitArtifact,
    *,
    pad_margin_mm: float,
    drill_margin_mm: float,
) -> LeafBlockerSet:
    front_pads: list[tuple[Point, Point]] = []
    back_pads: list[tuple[Point, Point]] = []
    tht_drills: list[tuple[Point, Point]] = []

    for component in artifact.layout.components.values():
        component_bbox = _component_local_bbox(component)
        body_width = max(0.6, component_bbox[1].x - component_bbox[0].x)
        body_height = max(0.6, component_bbox[1].y - component_bbox[0].y)
        pad_half_w = max(0.3, body_width * 0.1) + pad_margin_mm
        pad_half_h = max(0.3, body_height * 0.1) + pad_margin_mm
        drill_half_w = max(0.3, body_width * 0.08) + drill_margin_mm
        drill_half_h = max(0.3, body_height * 0.08) + drill_margin_mm

        for pad in component.pads:
            pad_rect = _rect_from_center(pad.pos, pad_half_w, pad_half_h)
            if component.is_through_hole:
                front_pads.append(pad_rect)
                back_pads.append(pad_rect)
                tht_drills.append(
                    _rect_from_center(pad.pos, drill_half_w, drill_half_h)
                )
            elif pad.layer == 0:
                front_pads.append(pad_rect)
            else:
                back_pads.append(pad_rect)

        if component.is_through_hole and not component.pads:
            tht_drills.append(component_bbox)

    return LeafBlockerSet(
        front_pads=_coalesce_rects(front_pads),
        back_pads=_coalesce_rects(back_pads),
        tht_drills=_coalesce_rects(tht_drills),
        leaf_outline=_artifact_outline(artifact),
    )


def _extract_blockers_from_pcb(
    artifact: LoadedSubcircuitArtifact,
    *,
    pad_margin_mm: float,
    drill_margin_mm: float,
) -> LeafBlockerSet | None:
    mini_pcb = artifact.source_files.get("mini_pcb") or artifact.metadata.get(
        "artifact_paths", {}
    ).get("mini_pcb", "")
    if not mini_pcb:
        return None
    mini_pcb_path = Path(str(mini_pcb))
    if not mini_pcb_path.exists():
        return None

    try:
        pcbnew = importlib.import_module("pcbnew")
    except Exception:
        return None

    board = pcbnew.LoadBoard(str(mini_pcb_path))
    front_pads: list[tuple[Point, Point]] = []
    back_pads: list[tuple[Point, Point]] = []
    tht_drills: list[tuple[Point, Point]] = []
    copper_layers = {
        pcbnew.F_Cu: front_pads,
        pcbnew.B_Cu: back_pads,
    }

    for footprint in board.GetFootprints():
        for pad in footprint.Pads():
            drill = pad.GetDrillSize()
            drill_x_mm = drill.x / 1e6
            drill_y_mm = drill.y / 1e6
            is_through_hole = (
                pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH
                or drill_x_mm > 0.0
                or drill_y_mm > 0.0
            )
            for layer_id, target in copper_layers.items():
                if not pad.CanFlashLayer(layer_id):
                    continue
                bbox = pad.GetBoundingBox(layer_id)
                target.append(
                    (
                        Point(
                            bbox.GetLeft() / 1e6 - pad_margin_mm,
                            bbox.GetTop() / 1e6 - pad_margin_mm,
                        ),
                        Point(
                            bbox.GetRight() / 1e6 + pad_margin_mm,
                            bbox.GetBottom() / 1e6 + pad_margin_mm,
                        ),
                    )
                )
            if is_through_hole:
                position = pad.GetPosition()
                tht_drills.append(
                    _rect_from_center(
                        Point(position.x / 1e6, position.y / 1e6),
                        max(0.1, drill_x_mm / 2.0) + drill_margin_mm,
                        max(0.1, drill_y_mm / 2.0) + drill_margin_mm,
                    )
                )

    return LeafBlockerSet(
        front_pads=_coalesce_rects(front_pads),
        back_pads=_coalesce_rects(back_pads),
        tht_drills=_coalesce_rects(tht_drills),
        leaf_outline=_artifact_outline(artifact),
    )


def extract_leaf_blocker_set(
    artifact: LoadedSubcircuitArtifact,
    *,
    pad_margin_mm: float = 0.2,
    drill_margin_mm: float = 0.2,
) -> LeafBlockerSet:
    blocker_set = _extract_blockers_from_pcb(
        artifact,
        pad_margin_mm=pad_margin_mm,
        drill_margin_mm=drill_margin_mm,
    )
    if blocker_set is not None:
        return blocker_set
    return _extract_blockers_from_layout(
        artifact,
        pad_margin_mm=pad_margin_mm,
        drill_margin_mm=drill_margin_mm,
    )


def dominant_blocker_side(blocker_set: LeafBlockerSet) -> Literal["front", "back", "dual", "none"]:
    front_area = sum(_rect_area(rect) for rect in blocker_set.front_pads)
    back_area = sum(_rect_area(rect) for rect in blocker_set.back_pads)
    if front_area <= 0.0 and back_area <= 0.0:
        return "none"
    if front_area > 0.0 and back_area <= 0.0:
        return "front"
    if back_area > 0.0 and front_area <= 0.0:
        return "back"
    if front_area >= back_area * 1.5:
        return "front"
    if back_area >= front_area * 1.5:
        return "back"
    return "dual"


def can_overlap_sparse(
    blocker_a: LeafBlockerSet,
    origin_a: Point,
    rotation_a: float,
    blocker_b: LeafBlockerSet,
    origin_b: Point,
    rotation_b: float,
) -> bool:
    if _any_rect_overlap(
        blocker_a.front_pads,
        origin_a,
        rotation_a,
        blocker_b.front_pads,
        origin_b,
        rotation_b,
    ):
        return False
    if _any_rect_overlap(
        blocker_a.back_pads,
        origin_a,
        rotation_a,
        blocker_b.back_pads,
        origin_b,
        rotation_b,
    ):
        return False
    if _any_rect_overlap(
        blocker_a.front_pads,
        origin_a,
        rotation_a,
        blocker_b.tht_drills,
        origin_b,
        rotation_b,
    ):
        return False
    if _any_rect_overlap(
        blocker_a.tht_drills,
        origin_a,
        rotation_a,
        blocker_b.front_pads,
        origin_b,
        rotation_b,
    ):
        return False
    if _any_rect_overlap(
        blocker_a.back_pads,
        origin_a,
        rotation_a,
        blocker_b.tht_drills,
        origin_b,
        rotation_b,
    ):
        return False
    if _any_rect_overlap(
        blocker_a.tht_drills,
        origin_a,
        rotation_a,
        blocker_b.back_pads,
        origin_b,
        rotation_b,
    ):
        return False
    if _any_rect_overlap(
        blocker_a.tht_drills,
        origin_a,
        rotation_a,
        blocker_b.tht_drills,
        origin_b,
        rotation_b,
    ):
        return False

    outline_a = _transform_rect(blocker_a.leaf_outline, origin_a, rotation_a)
    outline_b = _transform_rect(blocker_b.leaf_outline, origin_b, rotation_b)
    side_a = dominant_blocker_side(blocker_a)
    side_b = dominant_blocker_side(blocker_b)
    if side_a in {"front", "back"} and side_a == side_b and _rects_intersect(outline_a, outline_b):
        return False
    return True


def child_layer_envelopes(
    transformed: TransformedSubcircuit,
) -> tuple[list[tuple[Point, Point]], list[tuple[Point, Point]], list[tuple[Point, Point]]]:
    """Compute layer-aware geometry envelopes for one transformed child artifact.

    Returns:
        (front_surface_rects, back_surface_rects, tht_keepout_rects)
    """
    front_rects: list[tuple[Point, Point]] = []
    back_rects: list[tuple[Point, Point]] = []
    tht_rects: list[tuple[Point, Point]] = []

    for comp in transformed.transformed_components.values():
        c_min, c_max = comp.bbox()
        if comp.is_through_hole:
            if comp.pads:
                for pad in comp.pads:
                    pad_half = 0.6
                    tht_rects.append(
                        (
                            Point(pad.pos.x - pad_half, pad.pos.y - pad_half),
                            Point(pad.pos.x + pad_half, pad.pos.y + pad_half),
                        )
                    )
            else:
                tht_rects.append((c_min, c_max))
            # THT also contributes to its placement layer's surface envelope
            if comp.layer == 0:  # Layer.FRONT
                front_rects.append((c_min, c_max))
            else:
                back_rects.append((c_min, c_max))
        else:
            if comp.layer == 0:  # Layer.FRONT
                front_rects.append((c_min, c_max))
            else:
                back_rects.append((c_min, c_max))

    return front_rects, back_rects, tht_rects


def _bbox_disjoint(a: tuple[Point, Point] | None, b: tuple[Point, Point] | None) -> bool:
    """Return True if two bounding boxes are disjoint (do not overlap).

    None is considered disjoint from everything.
    """
    if a is None or b is None:
        return True
    a_min, a_max = a
    b_min, b_max = b

    # One is entirely to the left/right
    if a_max.x <= b_min.x or b_max.x <= a_min.x:
        return True
    # One is entirely above/below
    if a_max.y <= b_min.y or b_max.y <= a_min.y:
        return True
    return False


def can_overlap(
    a_envelopes: tuple[list[tuple[Point, Point]], list[tuple[Point, Point]], list[tuple[Point, Point]]],
    b_envelopes: tuple[list[tuple[Point, Point]], list[tuple[Point, Point]], list[tuple[Point, Point]]],
) -> bool:
    """Determine if two children can safely XY-overlap without geometric conflict.

    Two children can overlap iff:
    - Their front surfaces are disjoint
    - Their back surfaces are disjoint
    - Child A's THT keepout is disjoint from Child B's front and back surfaces
    - Child B's THT keepout is disjoint from Child A's front and back surfaces
    - Their THT keepouts are disjoint from each other
    """
    a_front, a_back, a_tht = a_envelopes
    b_front, b_back, b_tht = b_envelopes

    for rect_a in a_front:
        for rect_b in b_front:
            if not _bbox_disjoint(rect_a, rect_b):
                return False
    for rect_a in a_back:
        for rect_b in b_back:
            if not _bbox_disjoint(rect_a, rect_b):
                return False
    for rect_a in a_tht:
        for rect_b in b_tht:
            if not _bbox_disjoint(rect_a, rect_b):
                return False
        for rect_b in b_front:
            if not _bbox_disjoint(rect_a, rect_b):
                return False
        for rect_b in b_back:
            if not _bbox_disjoint(rect_a, rect_b):
                return False
    for rect_a in b_tht:
        for rect_b in a_front:
            if not _bbox_disjoint(rect_a, rect_b):
                return False
        for rect_b in a_back:
            if not _bbox_disjoint(rect_a, rect_b):
                return False

    return True


def constraint_aware_outline(
    placed_bboxes: list[tuple[Point, Point]],
    attachment_constraints: list[AttachmentConstraint],
    constrained_ref_world_anchors: dict[str, Point],
    margin_mm: float = 1.5,
) -> tuple[Point, Point]:
    """Compute parent board outline respecting both geometry and constraint targets.

    For unconstrained sides, uses the maximum geometry extent plus margin.
    For constrained sides, uses the exact target coordinate derived from the
    constrained component's world position and keep-in/overhang values.
    """
    if not placed_bboxes:
        return (Point(0.0, 0.0), Point(10.0, 10.0))

    min_geom_x = min(bbox[0].x for bbox in placed_bboxes)
    min_geom_y = min(bbox[0].y for bbox in placed_bboxes)
    max_geom_x = max(bbox[1].x for bbox in placed_bboxes)
    max_geom_y = max(bbox[1].y for bbox in placed_bboxes)

    out_min_x = min_geom_x
    out_min_y = min_geom_y
    out_max_x = max_geom_x
    out_max_y = max_geom_y

    # Override with constraint targets
    left_edges = []
    right_edges = []
    top_edges = []
    bottom_edges = []

    for c in attachment_constraints:
        if c.ref not in constrained_ref_world_anchors:
            continue
        anchor = constrained_ref_world_anchors[c.ref]

        if c.target == "edge" or c.target == "corner":
            v = c.value
            if "left" in v:
                left_edges.append(anchor.x - c.inward_keep_in_mm + c.outward_overhang_mm)
            if "right" in v:
                right_edges.append(anchor.x + c.inward_keep_in_mm - c.outward_overhang_mm)
            if "top" in v:
                top_edges.append(anchor.y - c.inward_keep_in_mm + c.outward_overhang_mm)
            if "bottom" in v:
                bottom_edges.append(anchor.y + c.inward_keep_in_mm - c.outward_overhang_mm)

    if left_edges:
        out_min_x = min(min_geom_x, min(left_edges))
    else:
        out_min_x = min_geom_x - margin_mm
    if right_edges:
        out_max_x = max(max_geom_x, max(right_edges))
    else:
        out_max_x = max_geom_x + margin_mm
    if top_edges:
        out_min_y = min(min_geom_y, min(top_edges))
    else:
        out_min_y = min_geom_y - margin_mm
    if bottom_edges:
        out_max_y = max(max_geom_y, max(bottom_edges))
    else:
        out_max_y = max_geom_y + margin_mm

    return Point(out_min_x, out_min_y), Point(out_max_x, out_max_y)


def estimate_parent_board_size(
    child_bboxes: list[tuple[float, float]],
    interconnect_net_count: int = 0,
    routing_overhead_factor: float = 1.3,
    margin_mm: float = 1.5,
) -> tuple[float, float]:
    """Estimate a reasonable parent board size from child bounding boxes.

    Use this BEFORE packing to get a starting size for config/search bounds.
    After packing, prefer ``packed_extents_outline()`` for exact dimensions.

    Args:
        child_bboxes: List of (width_mm, height_mm) for each child.
        interconnect_net_count: Number of interconnect nets between children.
        routing_overhead_factor: Multiplier on total child area for routing.
        margin_mm: Edge margin around the packed children.

    Returns:
        (estimated_width_mm, estimated_height_mm).
    """
    import math

    if not child_bboxes:
        return (10.0, 10.0)

    total_child_area = sum(
        max(0.0, w) * max(0.0, h) for w, h in child_bboxes
    )
    max_child_width = max((max(0.0, w) for w, h in child_bboxes), default=0.0)
    max_child_height = max((max(0.0, h) for w, h in child_bboxes), default=0.0)

    net_overhead = min(0.3, interconnect_net_count * 0.01)
    effective_overhead = routing_overhead_factor + net_overhead

    estimated_area = total_child_area * effective_overhead

    side = math.sqrt(estimated_area)
    estimated_width = max(side, max_child_width) + 2.0 * margin_mm
    estimated_height = (
        max(
            estimated_area / max(1.0, estimated_width - 2.0 * margin_mm),
            max_child_height,
        )
        + 2.0 * margin_mm
    )

    return (round(estimated_width, 2), round(estimated_height, 2))


def estimate_layer_aware_parent_board_size(
    child_envelopes: list[
        tuple[
            list[tuple[Point, Point]],
            list[tuple[Point, Point]],
            list[tuple[Point, Point]],
        ]
    ],
    interconnect_net_count: int = 0,
    routing_overhead_factor: float = 1.3,
    margin_mm: float = 1.5,
) -> tuple[float, float]:
    """Estimate a parent board size using layer-aware shareable occupancy.

    Front-only and back-only leaves can share XY area, but through-hole geometry
    blocks both layers. This estimator approximates that by computing a blocker
    area per layer and using the larger of the summed front/back blocker areas.
    """
    import math

    def union_rect_area(rects: list[tuple[Point, Point]]) -> float:
        if not rects:
            return 0.0
        min_x = min(rect[0].x for rect in rects)
        min_y = min(rect[0].y for rect in rects)
        max_x = max(rect[1].x for rect in rects)
        max_y = max(rect[1].y for rect in rects)
        return max(0.0, max_x - min_x) * max(0.0, max_y - min_y)

    def envelope_span(
        envelope: tuple[
            list[tuple[Point, Point]],
            list[tuple[Point, Point]],
            list[tuple[Point, Point]],
        ]
    ) -> tuple[float, float]:
        points: list[Point] = []
        for rects in envelope:
            for rect in rects:
                points.extend([rect[0], rect[1]])
        if not points:
            return (0.0, 0.0)
        min_x = min(point.x for point in points)
        min_y = min(point.y for point in points)
        max_x = max(point.x for point in points)
        max_y = max(point.y for point in points)
        return (max(0.0, max_x - min_x), max(0.0, max_y - min_y))

    if not child_envelopes:
        return (10.0, 10.0)

    front_sum = 0.0
    back_sum = 0.0
    max_child_width = 0.0
    max_child_height = 0.0

    for front_surface, back_surface, tht_keepout in child_envelopes:
        front_sum += union_rect_area(front_surface + tht_keepout)
        back_sum += union_rect_area(back_surface + tht_keepout)
        child_width, child_height = envelope_span(
            (front_surface, back_surface, tht_keepout)
        )
        max_child_width = max(max_child_width, child_width)
        max_child_height = max(max_child_height, child_height)

    effective_area = max(front_sum, back_sum)
    net_overhead = min(0.3, interconnect_net_count * 0.01)
    effective_overhead = routing_overhead_factor + net_overhead
    estimated_area = effective_area * effective_overhead

    side = math.sqrt(estimated_area)
    estimated_width = max(side, max_child_width) + 2.0 * margin_mm
    estimated_height = (
        max(
            estimated_area / max(1.0, estimated_width - 2.0 * margin_mm),
            max_child_height,
        )
        + 2.0 * margin_mm
    )
    return (round(estimated_width, 2), round(estimated_height, 2))


def packed_extents_outline(
    placed_bboxes: list[tuple[Point, Point]],
    margin_mm: float = 1.5,
) -> tuple[Point, Point]:
    """Compute an exact board outline from already-packed child placements.

    Args:
        origins: List of (x, y) placement origins for each child.
        bboxes: List of (width, height) transformed bounding boxes.
        margin_mm: Edge clearance around outermost geometry.

    Returns:
        (top_left, bottom_right) ``Point`` pair for the board outline.
    """
    if not placed_bboxes:
        return (Point(0.0, 0.0), Point(10.0, 10.0))

    min_x = min(bbox[0].x for bbox in placed_bboxes)
    min_y = min(bbox[0].y for bbox in placed_bboxes)
    max_x = max(bbox[1].x for bbox in placed_bboxes)
    max_y = max(bbox[1].y for bbox in placed_bboxes)

    return (Point(min_x - margin_mm, min_y - margin_mm), Point(max_x + margin_mm, max_y + margin_mm))


def _derive_board_outline(
    components: dict[str, Component],
    traces: list[TraceSegment],
    vias: list[Via],
    child_anchor_maps: dict[str, dict[str, InterfaceAnchor]],
    margin_mm: float = 2.0,
) -> tuple[Point, Point]:
    """Derive a parent board outline from merged geometry."""
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for comp in components.values():
        tl, br = comp.bbox()
        min_x = min(min_x, tl.x)
        min_y = min(min_y, tl.y)
        max_x = max(max_x, br.x)
        max_y = max(max_y, br.y)
        for pad in comp.pads:
            min_x = min(min_x, pad.pos.x)
            min_y = min(min_y, pad.pos.y)
            max_x = max(max_x, pad.pos.x)
            max_y = max(max_y, pad.pos.y)

    for trace in traces:
        min_x = min(min_x, trace.start.x, trace.end.x)
        min_y = min(min_y, trace.start.y, trace.end.y)
        max_x = max(max_x, trace.start.x, trace.end.x)
        max_y = max(max_y, trace.start.y, trace.end.y)

    for via in vias:
        min_x = min(min_x, via.pos.x)
        min_y = min(min_y, via.pos.y)
        max_x = max(max_x, via.pos.x)
        max_y = max(max_y, via.pos.y)

    for anchors in child_anchor_maps.values():
        for anchor in anchors.values():
            min_x = min(min_x, anchor.pos.x)
            min_y = min(min_y, anchor.pos.y)
            max_x = max(max_x, anchor.pos.x)
            max_y = max(max_y, anchor.pos.y)

    if min_x == float("inf"):
        return (Point(0.0, 0.0), Point(0.0, 0.0))

    return (
        Point(min_x - margin_mm, min_y - margin_mm),
        Point(max_x + margin_mm, max_y + margin_mm),
    )


__all__ = [
    "AttachmentConstraint",
    "ChildArtifactPlacement",
    "ChildPlacement",
    "ComposedChild",
    "LeafBlockerSet",
    "ParentComposition",
    "ParentCompositionScore",
    "build_parent_composition",
    "can_overlap_sparse",
    "child_anchor_map",
    "child_component_refs",
    "composition_debug_dict",
    "composition_summary",
    "constrained_child_offset",
    "derive_attachment_constraints",
    "dominant_blocker_side",
    "estimate_parent_board_size",
    "estimate_layer_aware_parent_board_size",
    "extract_leaf_blocker_set",
    "packed_extents_outline",
    "child_layer_envelopes",
    "can_overlap",
    "constraint_aware_outline",
]
