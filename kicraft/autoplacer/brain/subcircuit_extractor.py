"""Leaf board-state extraction for hierarchical subcircuits.

This module bridges the new schematic-driven subcircuit hierarchy with the
existing full-board PCB model. It extracts a leaf subcircuit's local
`BoardState` from the full project `BoardState` while preserving the original
component/pad geometry.

Current scope:
- extract leaf-local components from a full board state
- partition nets into internal vs external/interface nets
- collect traces/vias that are fully internal to the leaf
- derive a local solving envelope and translated local board state
- build extraction records suitable for artifact generation and later solving

This module is intentionally pure Python and does not depend on pcbnew.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .types import (
    BoardState,
    Component,
    InterfaceDirection,
    InterfacePort,
    InterfaceRole,
    InterfaceSide,
    Net,
    Pad,
    Point,
    SubCircuitDefinition,
    SubcircuitAccessPolicy,
    TraceSegment,
    Via,
)


@dataclass(slots=True)
class NetPartition:
    """Partition of nets for one extracted leaf subcircuit."""

    internal: dict[str, Net] = field(default_factory=dict)
    external: dict[str, Net] = field(default_factory=dict)
    ignored: dict[str, Net] = field(default_factory=dict)

    @property
    def internal_names(self) -> list[str]:
        return sorted(self.internal.keys())

    @property
    def external_names(self) -> list[str]:
        return sorted(self.external.keys())

    @property
    def ignored_names(self) -> list[str]:
        return sorted(self.ignored.keys())


@dataclass(slots=True)
class LocalEnvelope:
    """Synthetic local board envelope for a leaf subcircuit."""

    top_left: Point
    bottom_right: Point
    width_mm: float
    height_mm: float
    margin_mm: float
    source_board_outline: tuple[Point, Point] | None = None

    @property
    def board_outline(self) -> tuple[Point, Point]:
        return (self.top_left, self.bottom_right)

    @property
    def origin(self) -> Point:
        return self.top_left


@dataclass(slots=True)
class ExtractedSubcircuitBoard:
    """Result of extracting a leaf subcircuit from a full board state."""

    subcircuit: SubCircuitDefinition
    full_state: BoardState
    local_state: BoardState
    component_refs: list[str]
    interface_ports: list[InterfacePort]
    net_partition: NetPartition
    internal_traces: list[TraceSegment] = field(default_factory=list)
    internal_vias: list[Via] = field(default_factory=list)
    envelope: LocalEnvelope | None = None
    translation: Point = field(default_factory=lambda: Point(0.0, 0.0))
    notes: list[str] = field(default_factory=list)

    @property
    def internal_net_names(self) -> list[str]:
        return self.net_partition.internal_names

    @property
    def external_net_names(self) -> list[str]:
        return self.net_partition.external_names

    @property
    def ignored_net_names(self) -> list[str]:
        return self.net_partition.ignored_names



def _infer_implicit_interface_ports(
    interface_ports: list[InterfacePort],
    net_partition: NetPartition,
) -> list[InterfacePort]:
    """Add implicit interface ports for external nets without declared ports.

    When a net crosses the subcircuit boundary (has both internal and
    external component connections) but no schematic sheet pin declares it
    as an interface port, it is "implicitly shared."  Power nets like GND
    are the most common case -- they connect to every subcircuit but are
    rarely declared as explicit sheet pins.

    This function creates synthetic InterfacePort entries for such nets so
    that the parent composition step can discover and route them.

    Implicit ports are marked ``required=False`` (optional for anchor
    completeness) and ``source_kind="implicit_external_net"`` to
    distinguish them from schematic-declared ports.
    """
    declared_nets = {p.net_name for p in interface_ports}
    augmented = list(interface_ports)

    for net_name in sorted(net_partition.external.keys()):
        if net_name in declared_nets:
            continue
        net = net_partition.external[net_name]
        # Determine role: power nets get POWER_IN role, others get BIDIR
        if net.is_power:
            role = InterfaceRole.POWER_IN
        else:
            role = InterfaceRole.BIDIR

        # Use the net name (without leading /) as the port name
        port_name = net_name.lstrip("/")
        augmented.append(
            InterfacePort(
                name=port_name,
                net_name=net_name,
                role=role,
                direction=InterfaceDirection.UNKNOWN,
                preferred_side=InterfaceSide.ANY,
                access_policy=SubcircuitAccessPolicy.INTERFACE_ONLY,
                cardinality=1,
                bus_index=None,
                required=False,
                description="implicit interface port for external net without declared sheet pin",
                raw_direction="",
                source_uuid=None,
                source_kind="implicit_external_net",
            )
        )

    return augmented


def extract_leaf_board_state(
    subcircuit: SubCircuitDefinition,
    full_state: BoardState,
    *,
    margin_mm: float = 5.0,
    include_power_externals: bool = True,
    ignored_nets: set[str] | None = None,
) -> ExtractedSubcircuitBoard:
    """Extract a leaf-local board state from the full board state.

    Args:
        subcircuit: Parsed leaf subcircuit definition from schematic hierarchy.
        full_state: Full project board state loaded from the PCB.
        margin_mm: Extra margin around the extracted component bounding box.
        include_power_externals: Keep external power nets in the external
            partition. When False, they are moved to ignored.
        ignored_nets: Optional set of net names to exclude from both internal
            and external partitions.

    Returns:
        `ExtractedSubcircuitBoard` containing:
        - translated local `BoardState`
        - internal/external net partition
        - internal traces/vias
        - synthetic local envelope

    Raises:
        ValueError: if the subcircuit is not a leaf or has no matching
            components on the board.
    """
    if not subcircuit.is_leaf:
        raise ValueError(
            f"Subcircuit '{subcircuit.id.sheet_name}' is not a leaf and cannot be extracted as a leaf board state"
        )

    ignored = {n.upper() for n in (ignored_nets or set())}
    component_refs = [
        ref for ref in subcircuit.component_refs if ref in full_state.components
    ]
    if not component_refs:
        raise ValueError(
            f"Leaf subcircuit '{subcircuit.id.sheet_name}' has no matching components in the full board state"
        )

    component_set = set(component_refs)
    interface_ports = list(subcircuit.ports)

    local_components = _copy_components(full_state.components, component_refs)
    net_partition = _partition_nets(
        full_state.nets,
        component_set,
        interface_ports,
        include_power_externals=include_power_externals,
        ignored_nets=ignored,
    )

    # Augment interface ports with implicit ports for external nets that
    # have no declared schematic sheet pin.  This ensures power nets like
    # GND appear as interface ports for parent composition.
    interface_ports = _infer_implicit_interface_ports(interface_ports, net_partition)

    internal_traces = _extract_internal_traces(
        full_state.traces, net_partition.internal
    )
    internal_vias = _extract_internal_vias(full_state.vias, net_partition.internal)

    envelope = _derive_local_envelope(
        local_components,
        margin_mm=margin_mm,
        board_outline=full_state.board_outline,
    )
    translation = Point(-envelope.top_left.x, -envelope.top_left.y)

    translated_components = _translate_components(local_components, translation)
    translated_traces = _translate_traces(internal_traces, translation)
    translated_vias = _translate_vias(internal_vias, translation)

    local_state = BoardState(
        components=translated_components,
        nets={**net_partition.internal, **net_partition.external},
        traces=translated_traces,
        vias=translated_vias,
        board_outline=(
            Point(0.0, 0.0),
            Point(envelope.width_mm, envelope.height_mm),
        ),
    )

    notes = [
        f"leaf={subcircuit.id.sheet_name}",
        f"instance_path={subcircuit.id.instance_path}",
        f"component_count={len(component_refs)}",
        f"internal_nets={len(net_partition.internal)}",
        f"external_nets={len(net_partition.external)}",
        f"ignored_nets={len(net_partition.ignored)}",
        f"margin_mm={margin_mm:.3f}",
    ]

    return ExtractedSubcircuitBoard(
        subcircuit=subcircuit,
        full_state=full_state,
        local_state=local_state,
        component_refs=component_refs,
        interface_ports=interface_ports,
        net_partition=net_partition,
        internal_traces=translated_traces,
        internal_vias=translated_vias,
        envelope=LocalEnvelope(
            top_left=Point(0.0, 0.0),
            bottom_right=Point(envelope.width_mm, envelope.height_mm),
            width_mm=envelope.width_mm,
            height_mm=envelope.height_mm,
            margin_mm=envelope.margin_mm,
            source_board_outline=envelope.source_board_outline,
        ),
        translation=translation,
        notes=notes,
    )


def summarize_extraction(extraction: ExtractedSubcircuitBoard) -> str:
    """Return a compact human-readable summary."""
    return (
        f"{extraction.subcircuit.id.sheet_name} "
        f"[{extraction.subcircuit.id.instance_path}] "
        f"refs={len(extraction.component_refs)} "
        f"internal_nets={len(extraction.net_partition.internal)} "
        f"external_nets={len(extraction.net_partition.external)} "
        f"traces={len(extraction.internal_traces)} "
        f"vias={len(extraction.internal_vias)} "
        f"size={extraction.local_state.board_width:.1f}x{extraction.local_state.board_height:.1f}mm"
    )


def extract_parent_local_components(
    pcb_path: str,
    child_artifacts: list[Any],
    *,
    allowlist: set[str] | None = None,
) -> dict[str, Component]:
    from kicraft.autoplacer.hardware.adapter import KiCadAdapter

    adapter = KiCadAdapter(pcb_path, config={})
    full_state = adapter.load()
    
    child_refs = set().union(
        *(set(art.layout.components.keys()) for art in child_artifacts)
    )
    
    allowed_refs = {ref.upper() for ref in (allowlist or set())}
    local_components = {}
    for ref, comp in full_state.components.items():
        if ref in child_refs:
            continue
        if allowed_refs and ref.upper() not in allowed_refs:
            continue
        if not allowed_refs or ref.upper() in allowed_refs:
            local_components[ref] = copy.deepcopy(comp)
            
    return local_components


def extraction_debug_dict(extraction: ExtractedSubcircuitBoard) -> dict[str, Any]:
    """Return a JSON-serializable debug view of an extraction."""
    return {
        "subcircuit": {
            "sheet_name": extraction.subcircuit.id.sheet_name,
            "sheet_file": extraction.subcircuit.id.sheet_file,
            "instance_path": extraction.subcircuit.id.instance_path,
            "parent_instance_path": extraction.subcircuit.id.parent_instance_path,
        },
        "component_refs": list(extraction.component_refs),
        "interface_ports": [
            {
                "name": p.name,
                "net_name": p.net_name,
                "role": getattr(p.role, "value", str(p.role)),
                "direction": getattr(p.direction, "value", str(p.direction)),
                "preferred_side": getattr(
                    p.preferred_side, "value", str(p.preferred_side)
                ),
                "access_policy": getattr(
                    p.access_policy, "value", str(p.access_policy)
                ),
                "cardinality": p.cardinality,
                "bus_index": p.bus_index,
                "required": p.required,
                "description": p.description,
            }
            for p in extraction.interface_ports
        ],
        "net_partition": {
            "internal": sorted(extraction.net_partition.internal.keys()),
            "external": sorted(extraction.net_partition.external.keys()),
            "ignored": sorted(extraction.net_partition.ignored.keys()),
        },
        "local_board_outline": {
            "top_left": {
                "x": extraction.local_state.board_outline[0].x,
                "y": extraction.local_state.board_outline[0].y,
            },
            "bottom_right": {
                "x": extraction.local_state.board_outline[1].x,
                "y": extraction.local_state.board_outline[1].y,
            },
            "width_mm": extraction.local_state.board_width,
            "height_mm": extraction.local_state.board_height,
        },
        "translation": {
            "x": extraction.translation.x,
            "y": extraction.translation.y,
        },
        "trace_count": len(extraction.internal_traces),
        "via_count": len(extraction.internal_vias),
        "notes": list(extraction.notes),
    }


def _copy_components(
    components: dict[str, Component],
    refs: list[str],
) -> dict[str, Component]:
    copied: dict[str, Component] = {}
    for ref in refs:
        if ref not in components:
            continue
        copied[ref] = copy.deepcopy(components[ref])
    return copied


def _partition_nets(
    nets: dict[str, Net],
    component_refs: set[str],
    interface_ports: list[InterfacePort],
    *,
    include_power_externals: bool,
    ignored_nets: set[str],
) -> NetPartition:
    interface_net_names = {port.net_name for port in interface_ports}
    partition = NetPartition()

    for net_name, net in nets.items():
        if net_name.upper() in ignored_nets:
            partition.ignored[net_name] = copy.deepcopy(net)
            continue

        refs_on_net = net.component_refs
        internal_refs = refs_on_net & component_refs
        external_refs = refs_on_net - component_refs

        if not internal_refs:
            continue

        if not external_refs:
            partition.internal[net_name] = _filter_net_to_components(
                net, component_refs
            )
            continue

        if net_name in interface_net_names:
            if not include_power_externals and net.is_power:
                partition.ignored[net_name] = _filter_net_to_components(
                    net, component_refs
                )
            else:
                partition.external[net_name] = _filter_net_to_components(
                    net, component_refs
                )
            continue

        if net.is_power and not include_power_externals:
            partition.ignored[net_name] = _filter_net_to_components(net, component_refs)
            continue

        partition.external[net_name] = _filter_net_to_components(net, component_refs)

    return partition


def _filter_net_to_components(net: Net, component_refs: set[str]) -> Net:
    return Net(
        name=net.name,
        pad_refs=[pad_ref for pad_ref in net.pad_refs if pad_ref[0] in component_refs],
        priority=net.priority,
        width_mm=net.width_mm,
        is_power=net.is_power,
    )


def _extract_internal_traces(
    traces: list[TraceSegment],
    internal_nets: dict[str, Net],
) -> list[TraceSegment]:
    internal_names = set(internal_nets.keys())
    return [copy.deepcopy(trace) for trace in traces if trace.net in internal_names]


def _extract_internal_vias(
    vias: list[Via],
    internal_nets: dict[str, Net],
) -> list[Via]:
    internal_names = set(internal_nets.keys())
    return [copy.deepcopy(via) for via in vias if via.net in internal_names]


def _derive_local_envelope(
    components: dict[str, Component],
    *,
    margin_mm: float,
    board_outline: tuple[Point, Point] | None = None,
) -> LocalEnvelope:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for comp in components.values():
        tl, br = comp.physical_bbox()
        min_x = min(min_x, tl.x)
        min_y = min(min_y, tl.y)
        max_x = max(max_x, br.x)
        max_y = max(max_y, br.y)

    if min_x == float("inf"):
        min_x = min_y = 0.0
        max_x = max_y = 0.0

    top_left = Point(min_x - margin_mm, min_y - margin_mm)
    bottom_right = Point(max_x + margin_mm, max_y + margin_mm)

    source_board_outline: tuple[Point, Point] | None = None
    if board_outline is not None:
        board_tl, board_br = board_outline
        source_board_outline = (
            Point(board_tl.x, board_tl.y),
            Point(board_br.x, board_br.y),
        )

        left_offset = max(0.0, min_x - board_tl.x)
        right_offset = max(0.0, board_br.x - max_x)
        top_offset = max(0.0, min_y - board_tl.y)
        bottom_offset = max(0.0, board_br.y - max_y)

        top_left = Point(
            min_x - min(left_offset, margin_mm),
            min_y - min(top_offset, margin_mm),
        )
        bottom_right = Point(
            max_x + min(right_offset, margin_mm),
            max_y + min(bottom_offset, margin_mm),
        )

    width = max(1.0, bottom_right.x - top_left.x)
    height = max(1.0, bottom_right.y - top_left.y)

    return LocalEnvelope(
        top_left=top_left,
        bottom_right=bottom_right,
        width_mm=width,
        height_mm=height,
        margin_mm=margin_mm,
        source_board_outline=source_board_outline,
    )


def _translate_components(
    components: dict[str, Component],
    delta: Point,
) -> dict[str, Component]:
    translated: dict[str, Component] = {}
    for ref, comp in components.items():
        new_comp = copy.deepcopy(comp)
        new_comp.pos = _translate_point(new_comp.pos, delta)
        if new_comp.body_center is not None:
            new_comp.body_center = _translate_point(new_comp.body_center, delta)
        new_comp.pads = [_translate_pad(pad, delta) for pad in new_comp.pads]
        translated[ref] = new_comp
    return translated


def _translate_pad(pad: Pad, delta: Point) -> Pad:
    new_pad = copy.deepcopy(pad)
    new_pad.pos = _translate_point(new_pad.pos, delta)
    return new_pad


def _translate_traces(traces: list[TraceSegment], delta: Point) -> list[TraceSegment]:
    translated: list[TraceSegment] = []
    for trace in traces:
        new_trace = copy.deepcopy(trace)
        new_trace.start = _translate_point(new_trace.start, delta)
        new_trace.end = _translate_point(new_trace.end, delta)
        translated.append(new_trace)
    return translated


def _translate_vias(vias: list[Via], delta: Point) -> list[Via]:
    translated: list[Via] = []
    for via in vias:
        new_via = copy.deepcopy(via)
        new_via.pos = _translate_point(new_via.pos, delta)
        translated.append(new_via)
    return translated


def _translate_point(point: Point, delta: Point) -> Point:
    return Point(point.x + delta.x, point.y + delta.y)


__all__ = [
    "ExtractedSubcircuitBoard",
    "LocalEnvelope",
    "NetPartition",
    "extract_leaf_board_state",
    "extract_parent_local_components",
    "extraction_debug_dict",
    "summarize_extraction",
]
