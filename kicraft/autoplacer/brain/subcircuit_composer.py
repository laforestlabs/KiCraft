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
import logging
from dataclasses import dataclass, field
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
class AttachmentConstraint:
    ref: str
    target: Literal["edge", "corner", "zone"]
    value: str
    inward_keep_in_mm: float
    outward_overhang_mm: float
    source: Literal["child_artifact", "parent_local"]
    child_index: int | None


def derive_attachment_constraints(
    loaded_artifacts: list[Any],
    parent_local_components: dict[str, Component],
    component_zones: dict[str, str],
    cfg: dict[str, Any],
) -> list[AttachmentConstraint]:
    constraints = []
    for ref, zone_str in component_zones.items():
        if ":" in zone_str:
            target, value = zone_str.split(":", 1)
        else:
            target = "edge"
            value = zone_str

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
            logging.getLogger(__name__).warning(
                f"Component {ref} constrained to {zone_str} but not found in any artifact or local components"
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
            )
        )
    return constraints


def constrained_child_offset(
    artifact: Any,
    constraint: AttachmentConstraint,
    rotation_deg: float,
    parent_outline_min: Point,
    parent_outline_max: Point,
) -> Point:
    import math

    comp = artifact.layout.components[constraint.ref]
    center = comp.body_center if comp.body_center else comp.pos

    rad = math.radians(rotation_deg)
    rx = center.x * math.cos(rad) - center.y * math.sin(rad)
    ry = center.x * math.sin(rad) + center.y * math.cos(rad)

    target_x = rx
    target_y = ry

    if constraint.target == "edge":
        if constraint.value == "left":
            target_x = parent_outline_min.x + constraint.inward_keep_in_mm - constraint.outward_overhang_mm
        elif constraint.value == "right":
            target_x = parent_outline_max.x - constraint.inward_keep_in_mm + constraint.outward_overhang_mm
        elif constraint.value == "top":
            target_y = parent_outline_min.y + constraint.inward_keep_in_mm - constraint.outward_overhang_mm
        elif constraint.value == "bottom":
            target_y = parent_outline_max.y - constraint.inward_keep_in_mm + constraint.outward_overhang_mm
    elif constraint.target == "corner":
        if constraint.value == "top-left":
            target_x = parent_outline_min.x + constraint.inward_keep_in_mm - constraint.outward_overhang_mm
            target_y = parent_outline_min.y + constraint.inward_keep_in_mm - constraint.outward_overhang_mm
        elif constraint.value == "top-right":
            target_x = parent_outline_max.x - constraint.inward_keep_in_mm + constraint.outward_overhang_mm
            target_y = parent_outline_min.y + constraint.inward_keep_in_mm - constraint.outward_overhang_mm
        elif constraint.value == "bottom-left":
            target_x = parent_outline_min.x + constraint.inward_keep_in_mm - constraint.outward_overhang_mm
            target_y = parent_outline_max.y - constraint.inward_keep_in_mm + constraint.outward_overhang_mm
        elif constraint.value == "bottom-right":
            target_x = parent_outline_max.x - constraint.inward_keep_in_mm + constraint.outward_overhang_mm
            target_y = parent_outline_max.y - constraint.inward_keep_in_mm + constraint.outward_overhang_mm
    elif constraint.target == "zone":
        if constraint.value == "bottom":
            target_y = parent_outline_max.y - constraint.inward_keep_in_mm

    return Point(target_x - rx, target_y - ry)


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


def child_layer_envelopes(
    transformed: TransformedSubcircuit,
) -> tuple[tuple[Point, Point] | None, tuple[Point, Point] | None, tuple[Point, Point] | None]:
    """Compute layer-aware geometry envelopes for one transformed child artifact.

    Returns:
        (front_surface_bbox, back_surface_bbox, tht_keepout_bbox)
        Each is either (min_pt, max_pt) or None if no geometry exists for that envelope.
    """
    front_pts = []
    back_pts = []
    tht_pts = []

    for comp in transformed.board_state.components.values():
        c_min, c_max = comp.bbox()
        if comp.is_through_hole:
            tht_pts.extend([c_min, c_max])
            # THT also contributes to its placement layer's surface envelope
            if comp.layer == 0:  # Layer.FRONT
                front_pts.extend([c_min, c_max])
            else:
                back_pts.extend([c_min, c_max])
        else:
            if comp.layer == 0:  # Layer.FRONT
                front_pts.extend([c_min, c_max])
            else:
                back_pts.extend([c_min, c_max])

    def make_bbox(pts: list[Point]) -> tuple[Point, Point] | None:
        if not pts:
            return None
        min_x = min(p.x for p in pts)
        min_y = min(p.y for p in pts)
        max_x = max(p.x for p in pts)
        max_y = max(p.y for p in pts)
        return (Point(min_x, min_y), Point(max_x, max_y))

    return make_bbox(front_pts), make_bbox(back_pts), make_bbox(tht_pts)


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
    a_envelopes: tuple[tuple[Point, Point] | None, tuple[Point, Point] | None, tuple[Point, Point] | None],
    b_envelopes: tuple[tuple[Point, Point] | None, tuple[Point, Point] | None, tuple[Point, Point] | None],
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

    if not _bbox_disjoint(a_front, b_front):
        return False
    if not _bbox_disjoint(a_back, b_back):
        return False

    if not _bbox_disjoint(a_tht, b_tht):
        return False

    if not _bbox_disjoint(a_tht, b_front):
        return False
    if not _bbox_disjoint(a_tht, b_back):
        return False

    if not _bbox_disjoint(b_tht, a_front):
        return False
    if not _bbox_disjoint(b_tht, a_back):
        return False

    return True


def constraint_aware_outline(
    child_origins: list[Point],
    child_bboxes: list[tuple[float, float]],
    attachment_constraints: list[AttachmentConstraint],
    constrained_ref_world_centers: dict[str, Point],
    margin_mm: float = 1.5,
) -> tuple[Point, Point]:
    """Compute parent board outline respecting both geometry and constraint targets.

    For unconstrained sides, uses the maximum geometry extent plus margin.
    For constrained sides, uses the exact target coordinate derived from the
    constrained component's world position and keep-in/overhang values.
    """
    if not child_origins:
        return (Point(0.0, 0.0), Point(10.0, 10.0))

    # Base geometry extents
    min_geom_x = min(o.x for o in child_origins)
    min_geom_y = min(o.y for o in child_origins)
    max_geom_x = max(o.x + w for o, (w, h) in zip(child_origins, child_bboxes))
    max_geom_y = max(o.y + h for o, (w, h) in zip(child_origins, child_bboxes))

    out_min_x = min_geom_x - margin_mm
    out_min_y = min_geom_y - margin_mm
    out_max_x = max_geom_x + margin_mm
    out_max_y = max_geom_y + margin_mm

    # Override with constraint targets
    left_edges = []
    right_edges = []
    top_edges = []
    bottom_edges = []

    for c in attachment_constraints:
        if c.ref not in constrained_ref_world_centers:
            continue
        center = constrained_ref_world_centers[c.ref]

        if c.target == "edge" or c.target == "corner":
            v = c.value
            if "left" in v:
                left_edges.append(center.x - c.inward_keep_in_mm + c.outward_overhang_mm)
            if "right" in v:
                right_edges.append(center.x + c.inward_keep_in_mm - c.outward_overhang_mm)
            if "top" in v:
                top_edges.append(center.y - c.inward_keep_in_mm + c.outward_overhang_mm)
            if "bottom" in v:
                bottom_edges.append(center.y + c.inward_keep_in_mm - c.outward_overhang_mm)

    if left_edges:
        out_min_x = min(left_edges)
    if right_edges:
        out_max_x = max(right_edges)
    if top_edges:
        out_min_y = min(top_edges)
    if bottom_edges:
        out_max_y = max(bottom_edges)

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


def packed_extents_outline(
    origins: list[tuple[float, float]],
    bboxes: list[tuple[float, float]],
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
    if not origins or not bboxes:
        return (Point(0.0, 0.0), Point(10.0, 10.0))

    max_x = 0.0
    max_y = 0.0
    for (ox, oy), (w, h) in zip(origins, bboxes):
        max_x = max(max_x, ox + w)
        max_y = max(max_y, oy + h)

    return (
        Point(-margin_mm, -margin_mm),
        Point(max_x + margin_mm, max_y + margin_mm),
    )


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
    "ParentComposition",
    "ParentCompositionScore",
    "build_parent_composition",
    "child_anchor_map",
    "child_component_refs",
    "composition_debug_dict",
    "composition_summary",
    "constrained_child_offset",
    "derive_attachment_constraints",
    "estimate_parent_board_size",
    "packed_extents_outline",
    "child_layer_envelopes",
    "can_overlap",
    "constraint_aware_outline",
]
