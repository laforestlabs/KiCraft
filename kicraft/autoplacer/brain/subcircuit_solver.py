"""Leaf placement and routing solver for extracted subcircuits.

This module is the first solver-stage bridge between the new subcircuits
architecture and the existing placement/routing engine.

Current scope:
- accept an extracted leaf-local `BoardState`
- run local placement optimization using the existing `PlacementSolver`
- optionally run local autorouting for internal nets
- score the result with `PlacementScorer`
- compute a local bounding box for the solved geometry
- package the result into a `SubCircuitLayout`
- provide JSON-serializable debug summaries for artifact persistence

This module intentionally does not yet:
- stamp solved layouts back into a parent board
- optimize interface anchor placement beyond simple net/pad selection
- perform parent-level rigid composition

Design notes:
- The extracted local board state already lives in a translated local
  coordinate system with a synthetic local board outline.
- The existing `PlacementSolver` can therefore be reused directly.
- Local routing is currently a lightweight Manhattan router for internal nets.
- The resulting `SubCircuitLayout` is treated as a frozen rigid artifact
  candidate for later parent-level composition.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .placement import PlacementScorer, PlacementSolver
from .subcircuit_extractor import ExtractedSubcircuitBoard
from .types import (
    BoardState,
    Component,
    InterfaceAnchor,
    InterfacePort,
    Layer,
    Net,
    Pad,
    Point,
    SubCircuitLayout,
    TraceSegment,
    Via,
)


@dataclass(slots=True)
class LocalRoutingResult:
    """Typed result for lightweight local subcircuit routing."""

    traces: list[TraceSegment] = field(default_factory=list)
    vias: list[Via] = field(default_factory=list)
    routed_internal_nets: list[str] = field(default_factory=list)
    failed_internal_nets: list[str] = field(default_factory=list)

    @property
    def trace_count(self) -> int:
        return len(self.traces)

    @property
    def via_count(self) -> int:
        return len(self.vias)

    @property
    def total_length_mm(self) -> float:
        return sum(trace.length for trace in self.traces)


@dataclass(slots=True)
class LeafPlacementResult:
    """Result of solving local placement and optional routing for one leaf."""

    layout: SubCircuitLayout
    score_total: float
    score_breakdown: dict[str, float]
    solved_state: BoardState
    interface_anchors: list[InterfaceAnchor] = field(default_factory=list)
    routed_internal_nets: list[str] = field(default_factory=list)
    failed_internal_nets: list[str] = field(default_factory=list)
    route_trace_count: int = 0
    route_via_count: int = 0
    route_length_mm: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def width_mm(self) -> float:
        return self.layout.width

    @property
    def height_mm(self) -> float:
        return self.layout.height


def solve_leaf_placement(
    extraction: ExtractedSubcircuitBoard,
    config: dict | None = None,
    seed: int = 0,
) -> LeafPlacementResult:
    """Solve local placement and optional routing for one extracted leaf.

    Args:
        extraction: Leaf-local extracted board state and metadata.
        config: Optional solver config merged/consumed by the local solver.
        seed: Random seed for deterministic local placement variation.

    Returns:
        `LeafPlacementResult` containing:
        - frozen `SubCircuitLayout`
        - placement score breakdown
        - solved local `BoardState`
        - inferred interface anchors
        - local routing summary for internal nets

    Raises:
        ValueError: if the extraction is missing a local board state or has
            no components to solve.
    """
    if extraction.local_state is None:
        raise ValueError("Leaf extraction is missing local_state")
    if not extraction.local_state.components:
        raise ValueError(
            f"Leaf extraction '{extraction.subcircuit.id.sheet_name}' has no local components"
        )

    cfg = dict(config or {})
    work_state = _deepcopy_board_state(extraction.local_state)

    solver = PlacementSolver(work_state, cfg, seed=seed)
    solved_components = solver.solve()
    work_state.components = solved_components

    routed_internal_nets: list[str] = []
    failed_internal_nets: list[str] = []
    route_trace_count = 0
    route_via_count = 0
    route_length_mm = 0.0

    if cfg.get("subcircuit_route_internal_nets", True):
        routing = route_leaf_internal_nets(extraction, solved_components, cfg)

        work_state.traces = routing.traces
        work_state.vias = routing.vias
        routed_internal_nets = routing.routed_internal_nets
        failed_internal_nets = routing.failed_internal_nets
        route_trace_count = routing.trace_count
        route_via_count = routing.via_count
        route_length_mm = routing.total_length_mm
    else:
        work_state.traces = [
            copy.deepcopy(trace) for trace in extraction.internal_traces
        ]
        work_state.vias = [copy.deepcopy(via) for via in extraction.internal_vias]

    score = PlacementScorer(work_state, cfg).score()
    bbox = _compute_component_bbox(solved_components)
    anchors = infer_interface_anchors(
        extraction.interface_ports,
        solved_components,
    )

    layout = SubCircuitLayout(
        subcircuit_id=extraction.subcircuit.id,
        components={
            ref: copy.deepcopy(comp) for ref, comp in solved_components.items()
        },
        traces=[copy.deepcopy(trace) for trace in work_state.traces],
        vias=[copy.deepcopy(via) for via in work_state.vias],
        bounding_box=(bbox["width_mm"], bbox["height_mm"]),
        interface_anchors=anchors,
        score=score.total,
        artifact_paths={},
        frozen=True,
    )

    notes = [
        f"leaf={extraction.subcircuit.id.sheet_name}",
        f"instance_path={extraction.subcircuit.id.instance_path}",
        f"seed={seed}",
        f"component_count={len(solved_components)}",
        f"interface_anchor_count={len(anchors)}",
        f"score_total={score.total:.3f}",
        f"bbox_width_mm={bbox['width_mm']:.3f}",
        f"bbox_height_mm={bbox['height_mm']:.3f}",
        f"routed_internal_nets={len(routed_internal_nets)}",
        f"failed_internal_nets={len(failed_internal_nets)}",
        f"route_trace_count={route_trace_count}",
        f"route_via_count={route_via_count}",
        f"route_length_mm={route_length_mm:.3f}",
    ]

    return LeafPlacementResult(
        layout=layout,
        score_total=score.total,
        score_breakdown=_score_to_dict(score),
        solved_state=work_state,
        interface_anchors=anchors,
        routed_internal_nets=routed_internal_nets,
        failed_internal_nets=failed_internal_nets,
        route_trace_count=route_trace_count,
        route_via_count=route_via_count,
        route_length_mm=route_length_mm,
        notes=notes,
    )


def route_leaf_internal_nets(
    extraction: ExtractedSubcircuitBoard,
    components: dict[str, Component],
    config: dict | None = None,
) -> LocalRoutingResult:
    """Route internal leaf nets with a lightweight local Manhattan router."""
    return route_interconnect_nets(
        extraction.net_partition.internal,
        components,
        config=config,
    )


def route_interconnect_nets(
    nets: dict[str, Net],
    components: dict[str, Component],
    config: dict | None = None,
) -> LocalRoutingResult:
    """Route arbitrary nets across a component map with a simple Manhattan router.

    Current strategy:
    - route only nets with at least two pad refs
    - resolve pad refs against the provided component map
    - for 2-pin nets, draw a direct Manhattan path
    - for multi-pin nets, connect all pads to the first pad as a star
    - insert a via when the endpoints are on different layers
    - keep routing simple and deterministic for early hierarchical composition
    """
    cfg = dict(config or {})
    width_default = float(cfg.get("signal_width_mm", 0.127))
    via_drill = float(cfg.get("via_drill_mm", 0.3))
    via_size = float(cfg.get("via_size_mm", 0.6))
    traces: list[TraceSegment] = []
    vias: list[Via] = []
    routed_net_names: list[str] = []
    failed_net_names: list[str] = []

    for net_name, net in nets.items():
        if net is None or len(net.pad_refs) < 2:
            failed_net_names.append(net_name)
            continue

        pads = _resolve_net_pads(net, components)
        if len(pads) < 2:
            failed_net_names.append(net_name)
            continue

        width_mm = float(net.width_mm or width_default)
        try:
            net_traces, net_vias = _route_net_manhattan(
                net_name, pads, width_mm, via_drill, via_size
            )
        except Exception:
            failed_net_names.append(net_name)
            continue

        traces.extend(net_traces)
        vias.extend(net_vias)
        routed_net_names.append(net_name)

    return LocalRoutingResult(
        traces=traces,
        vias=vias,
        routed_internal_nets=routed_net_names,
        failed_internal_nets=failed_net_names,
    )


def infer_interface_anchors(
    ports: list[InterfacePort],
    components: dict[str, Component],
) -> list[InterfaceAnchor]:
    """Infer physical interface anchors from solved component pads.

    Current heuristic:
    - for each interface port, find all pads on the matching net
    - choose the pad closest to the outer edge of the solved local bbox
    - use that pad position as the interface anchor

    Net matching is normalized so schematic-facing names like `VBUS` can match
    PCB pad nets like `/VBUS`.

    This is intentionally simple for the first milestone. Later versions can
    incorporate:
    - preferred-side-aware anchor selection
    - connector pin ordering
    - explicit interface annotations
    - synthetic anchor points distinct from actual pads
    """
    if not ports or not components:
        return []

    bbox = _compute_component_bbox(components)
    anchors: list[InterfaceAnchor] = []

    for port in ports:
        candidate_pads = []
        for comp in components.values():
            for pad in comp.pads:
                if _nets_match(pad.net, port.net_name):
                    candidate_pads.append(pad)

        if not candidate_pads:
            continue

        best_pad = min(
            candidate_pads,
            key=lambda pad: _edge_distance_score(
                pad.pos,
                bbox["min_x"],
                bbox["min_y"],
                bbox["max_x"],
                bbox["max_y"],
            ),
        )

        anchors.append(
            InterfaceAnchor(
                port_name=port.name,
                pos=Point(best_pad.pos.x, best_pad.pos.y),
                layer=best_pad.layer,
                pad_ref=(best_pad.ref, best_pad.pad_id),
            )
        )

    return anchors


def placement_result_debug_dict(result: LeafPlacementResult) -> dict:
    """Return a JSON-serializable debug view of a leaf placement result."""
    return {
        "subcircuit_id": {
            "sheet_name": result.layout.subcircuit_id.sheet_name,
            "sheet_file": result.layout.subcircuit_id.sheet_file,
            "instance_path": result.layout.subcircuit_id.instance_path,
            "parent_instance_path": result.layout.subcircuit_id.parent_instance_path,
        },
        "score_total": result.score_total,
        "score_breakdown": dict(result.score_breakdown),
        "bounding_box": {
            "width_mm": result.layout.width,
            "height_mm": result.layout.height,
        },
        "component_count": len(result.layout.components),
        "trace_count": len(result.layout.traces),
        "via_count": len(result.layout.vias),
        "routed_internal_nets": list(result.routed_internal_nets),
        "failed_internal_nets": list(result.failed_internal_nets),
        "route_trace_count": result.route_trace_count,
        "route_via_count": result.route_via_count,
        "route_length_mm": result.route_length_mm,
        "interface_anchors": [
            {
                "port_name": anchor.port_name,
                "x": anchor.pos.x,
                "y": anchor.pos.y,
                "layer": _layer_name(anchor.layer),
                "pad_ref": list(anchor.pad_ref) if anchor.pad_ref else None,
            }
            for anchor in result.interface_anchors
        ],
        "notes": list(result.notes),
    }


def summarize_placement_result(result: LeafPlacementResult) -> str:
    """Return a compact one-line summary for logs/debug output."""
    return (
        f"{result.layout.subcircuit_id.sheet_name} "
        f"[{result.layout.subcircuit_id.instance_path}] "
        f"score={result.score_total:.1f} "
        f"size={result.layout.width:.1f}x{result.layout.height:.1f}mm "
        f"anchors={len(result.interface_anchors)} "
        f"traces={len(result.layout.traces)} "
        f"vias={len(result.layout.vias)} "
        f"routed={len(result.routed_internal_nets)} "
        f"failed={len(result.failed_internal_nets)}"
    )


def _deepcopy_board_state(state: BoardState) -> BoardState:
    """Deep-copy a board state for isolated local solving."""
    return BoardState(
        components={ref: copy.deepcopy(comp) for ref, comp in state.components.items()},
        nets={name: copy.deepcopy(net) for name, net in state.nets.items()},
        traces=[copy.deepcopy(trace) for trace in state.traces],
        vias=[copy.deepcopy(via) for via in state.vias],
        board_outline=(
            Point(state.board_outline[0].x, state.board_outline[0].y),
            Point(state.board_outline[1].x, state.board_outline[1].y),
        ),
    )


def _resolve_net_pads(net, components: dict[str, Component]) -> list[Pad]:
    """Resolve a net's pad refs into actual pad objects from solved components."""
    resolved: list[Pad] = []
    for ref, pad_id in net.pad_refs:
        comp = components.get(ref)
        if comp is None:
            continue
        for pad in comp.pads:
            if pad.pad_id == pad_id and pad.net == net.name:
                resolved.append(pad)
                break
    return resolved


def _route_net_manhattan(
    net_name: str,
    pads: list[Pad],
    width_mm: float,
    via_drill: float = 0.3,
    via_size: float = 0.6,
) -> tuple[list[TraceSegment], list[Via]]:
    """Route one net with simple Manhattan segments and optional vias."""
    traces: list[TraceSegment] = []
    vias: list[Via] = []

    root = pads[0]
    for target in pads[1:]:
        net_traces, net_vias = _route_pad_pair_manhattan(
            net_name,
            root,
            target,
            width_mm,
            via_drill,
            via_size,
        )
        traces.extend(net_traces)
        vias.extend(net_vias)

    return traces, vias


def _route_pad_pair_manhattan(
    net_name: str,
    start_pad: Pad,
    end_pad: Pad,
    width_mm: float,
    via_drill: float = 0.3,
    via_size: float = 0.6,
) -> tuple[list[TraceSegment], list[Via]]:
    """Route a pair of pads with a simple orthogonal path."""
    traces: list[TraceSegment] = []
    vias: list[Via] = []

    start = Point(start_pad.pos.x, start_pad.pos.y)
    end = Point(end_pad.pos.x, end_pad.pos.y)
    mid = Point(end.x, start.y)

    if start_pad.layer == end_pad.layer:
        if start.dist(mid) > 0:
            traces.append(
                TraceSegment(
                    start=start,
                    end=mid,
                    layer=start_pad.layer,
                    net=net_name,
                    width_mm=width_mm,
                )
            )
        if mid.dist(end) > 0:
            traces.append(
                TraceSegment(
                    start=mid,
                    end=end,
                    layer=start_pad.layer,
                    net=net_name,
                    width_mm=width_mm,
                )
            )
        return traces, vias

    via_pos = Point(mid.x, mid.y)
    if start.dist(via_pos) > 0:
        traces.append(
            TraceSegment(
                start=start,
                end=via_pos,
                layer=start_pad.layer,
                net=net_name,
                width_mm=width_mm,
            )
        )
    vias.append(
        Via(
            pos=via_pos,
            net=net_name,
            drill_mm=via_drill,
            size_mm=via_size,
        )
    )
    if via_pos.dist(end) > 0:
        traces.append(
            TraceSegment(
                start=via_pos,
                end=end,
                layer=end_pad.layer,
                net=net_name,
                width_mm=width_mm,
            )
        )
    return traces, vias


def _compute_component_bbox(components: dict[str, Component]) -> dict[str, float]:
    """Compute a tight bbox around solved components and pads."""
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

    if min_x == float("inf"):
        min_x = min_y = max_x = max_y = 0.0

    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "width_mm": max(0.0, max_x - min_x),
        "height_mm": max(0.0, max_y - min_y),
    }


def _edge_distance_score(
    pos: Point,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> float:
    """Lower score means closer to an outer edge of the local bbox."""
    return min(
        abs(pos.x - min_x),
        abs(pos.x - max_x),
        abs(pos.y - min_y),
        abs(pos.y - max_y),
    )


def _score_to_dict(score) -> dict[str, float]:
    """Convert `PlacementScore` into a plain dict."""
    return {
        "total": float(score.total),
        "net_distance": float(score.net_distance),
        "crossover_score": float(score.crossover_score),
        "crossover_count": float(score.crossover_count),
        "compactness": float(score.compactness),
        "edge_compliance": float(score.edge_compliance),
        "rotation_score": float(score.rotation_score),
        "board_containment": float(score.board_containment),
        "courtyard_overlap": float(score.courtyard_overlap),
        "smt_opposite_tht": float(score.smt_opposite_tht),
        "group_coherence": float(score.group_coherence),
        "aspect_ratio": float(score.aspect_ratio),
    }


def _normalize_net_name(net_name: str) -> str:
    """Normalize schematic/PCB net names for interface matching."""
    return str(net_name or "").strip().lstrip("/").upper()


def _nets_match(left: str, right: str) -> bool:
    """Return True when two net names refer to the same logical net."""
    return _normalize_net_name(left) == _normalize_net_name(right)


def _layer_name(layer: Layer) -> str:
    return "B.Cu" if layer == Layer.BACK else "F.Cu"


__all__ = [
    "LeafPlacementResult",
    "LocalRoutingResult",
    "infer_interface_anchors",
    "placement_result_debug_dict",
    "route_interconnect_nets",
    "route_leaf_internal_nets",
    "solve_leaf_placement",
    "summarize_placement_result",
]
