#!/usr/bin/env python3
"""Solve leaf subcircuits with local placement search.

This CLI is an early execution entrypoint for the subcircuits redesign.

It performs the following steps:

1. Parse the true-sheet schematic hierarchy from a top-level `.kicad_sch`
2. Load the full project `.kicad_pcb`
3. Extract each leaf sheet into a local synthetic board state
4. Run local placement search for each leaf subcircuit
5. Save JSON metadata/debug artifacts for each solved leaf
6. Print a human-readable or JSON summary

Current scope:
- leaf-only solving
- placement search with optional local routing
- local routing currently uses the lightweight internal-net router from `subcircuit_solver.py`
- no parent/composite composition yet

The goal is to establish a stable bottom-up local solve loop that can later
be extended with:
- local routing
- frozen subcircuit layout artifacts
- parent-level rigid composition
- final top-level assembly

Usage:
    python3 solve_subcircuits.py LLUPS.kicad_sch
    python3 solve_subcircuits.py LLUPS.kicad_sch --pcb LLUPS.kicad_pcb
    python3 solve_subcircuits.py LLUPS.kicad_sch --rounds 8
    python3 solve_subcircuits.py LLUPS.kicad_sch --json
    python3 solve_subcircuits.py LLUPS.kicad_sch --only CHARGER
    python3 solve_subcircuits.py LLUPS.kicad_sch --route
    python3 solve_subcircuits.py LLUPS.kicad_sch --route --fast-smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import re
import shutil
import site
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any



def _ensure_kicad_python_path() -> None:
    """Ensure KiCad Python bindings are importable."""
    try:
        import pcbnew  # noqa: F401

        return
    except Exception:
        pass

    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        f"/usr/lib/python{ver}/site-packages",
        f"/usr/lib64/python{ver}/site-packages",
        "/usr/lib/python3/dist-packages",
        "/usr/lib64/python3/dist-packages",
    ]

    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        pcbnew_py = Path(path) / "pcbnew.py"
        pcbnew_pkg = Path(path) / "pcbnew"
        if pcbnew_py.exists() or pcbnew_pkg.exists():
            if path not in sys.path:
                sys.path.append(path)

    try:
        import pcbnew  # noqa: F401
    except Exception as exc:
        raise ModuleNotFoundError(
            "KiCad Python module 'pcbnew' not found. "
            "Install KiCad bindings or set PYTHONPATH to KiCad site-packages."
        ) from exc


_ensure_kicad_python_path()

from kicad_helper.autoplacer.brain.hierarchy_parser import (
    HierarchyGraph,
    HierarchyNode,
    parse_hierarchy,
)
from kicad_helper.autoplacer.brain.placement import (
    PlacementScorer,
    PlacementSolver,
    _update_pad_positions,
)
from kicad_helper.autoplacer.brain.subcircuit_artifacts import (
    build_anchor_validation,
    build_artifact_metadata,
    build_leaf_extraction,
    build_solved_layout_artifact,
    resolve_artifact_paths,
    save_artifact_metadata,
    save_debug_payload,
    save_solved_layout_artifact,
    serialize_components,
)
from kicad_helper.autoplacer.brain.subcircuit_board_export import (
    ExportOptions,
    export_subcircuit_board,
)
from kicad_helper.autoplacer.brain.subcircuit_extractor import (
    ExtractedSubcircuitBoard,
    extract_leaf_board_state,
    extraction_debug_dict,
    summarize_extraction,
)
from kicad_helper.autoplacer.brain.subcircuit_render_diagnostics import (
    generate_leaf_diagnostic_artifacts,
    generate_stage_diagnostic_artifacts,
)
from kicad_helper.autoplacer.brain.types import (
    BoardState,
    Component,
    PlacementScore,
    Point,
    SubCircuitLayout,
)
from kicad_helper.autoplacer.config import DEFAULT_CONFIG, load_project_config
from kicad_helper.autoplacer.freerouting_runner import (
    import_routed_copper,
    route_with_freerouting,
    validate_routed_board,
)
from kicad_helper.autoplacer.hardware.adapter import KiCadAdapter


@dataclass(slots=True)
class SolveRoundResult:
    """One local placement-search round for a leaf subcircuit."""

    round_index: int
    seed: int
    score: float
    placement: PlacementScore
    components: dict[str, Component] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)
    routed: bool = False
    timing_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        routing = {
            key: value for key, value in self.routing.items() if not key.startswith("_")
        }
        return {
            "round_index": self.round_index,
            "seed": self.seed,
            "score": self.score,
            "routed": self.routed,
            "placement": {
                "total": self.placement.total,
                "net_distance": self.placement.net_distance,
                "crossover_count": self.placement.crossover_count,
                "crossover_score": self.placement.crossover_score,
                "compactness": self.placement.compactness,
                "edge_compliance": self.placement.edge_compliance,
                "rotation_score": self.placement.rotation_score,
                "board_containment": self.placement.board_containment,
                "courtyard_overlap": self.placement.courtyard_overlap,
                "smt_opposite_tht": self.placement.smt_opposite_tht,
                "group_coherence": self.placement.group_coherence,
                "aspect_ratio": self.placement.aspect_ratio,
            },
            "routing": routing,
            "timing_breakdown": dict(self.timing_breakdown),
            "preview_paths": {
                "pre_route_front": routing.get("round_preview_pre_route_front", ""),
                "pre_route_back": routing.get("round_preview_pre_route_back", ""),
                "pre_route_copper": routing.get("round_preview_pre_route_copper", ""),
                "routed_front": routing.get("round_preview_routed_front", ""),
                "routed_back": routing.get("round_preview_routed_back", ""),
                "routed_copper": routing.get("round_preview_routed_copper", ""),
            },
            "board_paths": {
                "illegal_pre_stamp": routing.get("round_board_illegal_pre_stamp", ""),
                "pre_route": routing.get("round_board_pre_route", ""),
                "routed": routing.get("round_board_routed", ""),
            },
            "log_summary": {
                "router": routing.get("router", ""),
                "reason": routing.get("reason", ""),
                "failed": bool(routing.get("failed", False)),
                "skipped": bool(routing.get("skipped", False)),
                "traces": int(routing.get("traces", 0) or 0),
                "vias": int(routing.get("vias", 0) or 0),
                "total_length_mm": float(routing.get("total_length_mm", 0.0) or 0.0),
                "failed_internal_nets": list(
                    routing.get("failed_internal_nets", []) or []
                ),
                "routed_internal_nets": list(
                    routing.get("routed_internal_nets", []) or []
                ),
            },
        }


@dataclass(slots=True)
class SolvedLeafSubcircuit:
    """Solved local placement result for one leaf subcircuit."""

    node: HierarchyNode
    extraction: ExtractedSubcircuitBoard
    best_round: SolveRoundResult
    all_rounds: list[SolveRoundResult] = field(default_factory=list)
    size_reduction: dict[str, Any] = field(default_factory=dict)
    scheduling_metadata: dict[str, Any] = field(default_factory=dict)
    failure_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def sheet_name(self) -> str:
        return self.node.id.sheet_name

    @property
    def instance_path(self) -> str:
        return self.node.id.instance_path

    def best_round_to_layout(self):
        from kicad_helper.autoplacer.brain.subcircuit_solver import infer_interface_anchors
        from kicad_helper.autoplacer.brain.types import SubCircuitLayout

        anchors = infer_interface_anchors(
            self.extraction.interface_ports,
            self.best_round.components,
        )
        routed_traces = [
            copy.deepcopy(trace)
            for trace in self.best_round.routing.get("_trace_segments", [])
        ]
        routed_vias = [
            copy.deepcopy(via)
            for via in self.best_round.routing.get("_via_objects", [])
        ]

        return SubCircuitLayout(
            subcircuit_id=self.node.definition.id,
            components=copy.deepcopy(self.best_round.components),
            traces=routed_traces,
            vias=routed_vias,
            bounding_box=(
                self.extraction.local_state.board_width,
                self.extraction.local_state.board_height,
            ),
            ports=[copy.deepcopy(port) for port in self.extraction.interface_ports],
            interface_anchors=anchors,
            score=self.best_round.score,
            artifact_paths={},
            frozen=True,
        )

    def canonical_layout_artifact(self, cfg: dict[str, Any]) -> dict[str, Any]:
        layout = self.best_round_to_layout()
        project_dir = Path(self.extraction.subcircuit.schematic_path).parent
        return build_solved_layout_artifact(
            layout,
            project_dir=project_dir,
            source_hash=self.extraction.subcircuit.id.instance_path,
            config_hash=json.dumps(cfg, sort_keys=True, default=str),
            solver_version="subcircuits-m3-placement",
            notes=[
                f"round_index={self.best_round.round_index}",
                f"seed={self.best_round.seed}",
                f"routing={json.dumps({key: value for key, value in self.best_round.routing.items() if not key.startswith('_')}, sort_keys=True)}",
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sheet_name": self.sheet_name,
            "instance_path": self.instance_path,
            "summary": summarize_extraction(self.extraction),
            "best_round": self.best_round.to_dict(),
            "rounds": [round_result.to_dict() for round_result in self.all_rounds],
            "extraction": extraction_debug_dict(self.extraction),
            "scheduling_metadata": dict(self.scheduling_metadata),
            "failure_summary": dict(self.failure_summary),
        }


def _iter_children(node: HierarchyNode):
    for child in node.children:
        yield child
        yield from _iter_children(child)


def _iter_non_root_nodes(graph: HierarchyGraph):
    for child in graph.root.children:
        yield child
        yield from _iter_children(child)


def _default_pcb_path(top_schematic: Path) -> Path:
    return top_schematic.with_suffix(".kicad_pcb")


def _load_config(
    config_path: str | None, project_dir: Path | None = None
) -> dict[str, Any]:
    cfg: dict[str, Any] = {**DEFAULT_CONFIG}

    # Auto-discover project-specific config if no explicit path given
    if not config_path and project_dir:
        from kicad_helper.autoplacer.config import discover_project_config

        discovered = discover_project_config(project_dir)
        if discovered:
            config_path = str(discovered)

    if config_path:
        cfg.update(load_project_config(config_path))
    return cfg


def _load_board_state(pcb_path: Path, config: dict[str, Any]) -> BoardState:
    adapter = KiCadAdapter(str(pcb_path), config=config)
    return adapter.load()


def _leaf_nodes(
    graph: HierarchyGraph,
    only: list[str] | None = None,
    preferred_order: list[str] | None = None,
) -> list[HierarchyNode]:
    selected = []
    only_set = {item.strip().lower() for item in (only or []) if item.strip()}
    for node in _iter_non_root_nodes(graph):
        if not node.is_leaf:
            continue
        if only_set:
            name_match = node.id.sheet_name.lower() in only_set
            path_match = node.id.instance_path.lower() in only_set
            file_match = node.id.sheet_file.lower() in only_set
            if not (name_match or path_match or file_match):
                continue
        selected.append(node)

    preferred_rank: dict[str, int] = {}
    for index, item in enumerate(preferred_order or []):
        key = item.strip().lower()
        if key and key not in preferred_rank:
            preferred_rank[key] = index

    if preferred_rank:
        selected.sort(
            key=lambda node: (
                preferred_rank.get(node.id.sheet_name.lower(), len(preferred_rank)),
                preferred_rank.get(node.id.instance_path.lower(), len(preferred_rank)),
                preferred_rank.get(node.id.sheet_file.lower(), len(preferred_rank)),
                node.id.sheet_name.lower(),
                node.id.instance_path.lower(),
            )
        )

    return selected


def _local_solver_config(
    base_cfg: dict[str, Any], extraction: ExtractedSubcircuitBoard
) -> dict[str, Any]:
    cfg = dict(base_cfg)

    cfg["enable_board_size_search"] = False
    cfg["hierarchical_placement"] = False
    cfg["subcircuit_route_internal_nets"] = bool(
        base_cfg.get("subcircuit_route_internal_nets", False)
    )

    local_component_zones: dict[str, Any] = {}
    source_outline = (
        extraction.envelope.source_board_outline
        if extraction.envelope is not None
        else None
    )
    source_board_tl = source_outline[0] if source_outline is not None else None
    source_board_br = source_outline[1] if source_outline is not None else None
    translation = extraction.translation

    for ref, comp in extraction.local_state.components.items():
        if comp.kind == "connector":
            if source_board_tl is not None and source_board_br is not None:
                if comp.body_center is not None:
                    source_center_x = comp.body_center.x - translation.x
                    source_center_y = comp.body_center.y - translation.y
                else:
                    source_center_x = comp.pos.x - translation.x
                    source_center_y = comp.pos.y - translation.y

                distances = {
                    "left": max(0.0, source_center_x - source_board_tl.x),
                    "right": max(0.0, source_board_br.x - source_center_x),
                    "top": max(0.0, source_center_y - source_board_tl.y),
                    "bottom": max(0.0, source_board_br.y - source_center_y),
                }
            else:
                if comp.body_center is not None:
                    local_center_x = comp.body_center.x
                    local_center_y = comp.body_center.y
                else:
                    local_center_x = comp.pos.x
                    local_center_y = comp.pos.y

                distances = {
                    "left": local_center_x,
                    "right": extraction.local_state.board_width - local_center_x,
                    "top": local_center_y,
                    "bottom": extraction.local_state.board_height - local_center_y,
                }

            nearest_edge = min(distances, key=lambda edge: distances[edge])
            local_component_zones[ref] = {"edge": nearest_edge}

    cfg["component_zones"] = local_component_zones
    cfg["unlock_all_footprints"] = False

    cfg["board_width_mm"] = extraction.local_state.board_width
    cfg["board_height_mm"] = extraction.local_state.board_height

    board_area = (
        extraction.local_state.board_width * extraction.local_state.board_height
    )
    total_component_area = sum(
        c.width_mm * c.height_mm for c in extraction.local_state.components.values()
    )
    density = total_component_area / max(board_area, 1.0)

    # Dense leaves benefit from tighter packing and stronger ordering.
    if density > 0.3:
        adaptive_clearance = max(0.5, 3.0 * (1.0 - density))
    else:
        adaptive_clearance = float(base_cfg.get("placement_clearance_mm", 3.0))

    passive_count = sum(
        1 for c in extraction.local_state.components.values() if c.kind == "passive"
    )
    connector_count = sum(
        1 for c in extraction.local_state.components.values() if c.kind == "connector"
    )
    ic_like_count = sum(
        1
        for c in extraction.local_state.components.values()
        if c.kind in ("ic", "regulator", "connector")
    )
    component_count = max(1, len(extraction.local_state.components))
    passive_ratio = passive_count / component_count

    cfg["placement_clearance_mm"] = max(0.5, adaptive_clearance)
    cfg["edge_margin_mm"] = max(
        0.5,
        min(2.0, float(base_cfg.get("edge_margin_mm", 2.0))),
    )
    cfg["placement_grid_mm"] = float(base_cfg.get("placement_grid_mm", 0.5))
    cfg["max_placement_iterations"] = max(
        300,
        int(base_cfg.get("max_placement_iterations", 300)),
    )
    cfg["placement_convergence_threshold"] = min(
        0.2,
        float(base_cfg.get("placement_convergence_threshold", 0.2)),
    )

    # Leaf layouts should still respect grouping/ordering, but routed search needs
    # enough exploration to produce meaningfully different candidate placements.
    cfg["group_source"] = str(base_cfg.get("leaf_group_source", "netlist"))
    cfg["signal_flow_order"] = list(base_cfg.get("leaf_signal_flow_order", []))
    cfg["ic_groups"] = dict(
        base_cfg.get("leaf_ic_groups", base_cfg.get("ic_groups", {}))
    )
    cfg["group_labels"] = dict(
        base_cfg.get("leaf_group_labels", base_cfg.get("group_labels", {}))
    )
    cfg["orderedness"] = float(
        base_cfg.get(
            "leaf_orderedness",
            0.35 if passive_ratio >= 0.35 or passive_count >= 4 else 0.20,
        )
    )
    cfg["randomize_group_layout"] = bool(
        base_cfg.get("leaf_randomize_group_layout", True)
    )
    cfg["scatter_mode"] = str(base_cfg.get("leaf_scatter_mode", "groups"))
    cfg["placement_score_every_n"] = 1
    cfg["unlock_all_footprints"] = False
    cfg["align_large_pairs"] = bool(base_cfg.get("leaf_align_large_pairs", True))
    cfg["prefer_legal_states"] = True
    cfg["legalize_during_force"] = True
    cfg["legalize_every_n"] = 1
    cfg["legalize_during_force_passes"] = max(
        2,
        int(base_cfg.get("legalize_during_force_passes", 2)),
    )
    cfg["enable_swap_optimization"] = bool(
        base_cfg.get("leaf_enable_swap_optimization", True)
    )
    cfg["leaf_legality_repair_passes"] = max(
        24,
        int(base_cfg.get("leaf_legality_repair_passes", 24)),
    )
    cfg["leaf_min_route_rounds"] = max(
        16,
        int(base_cfg.get("leaf_min_route_rounds", 16)),
    )

    # Encourage more structured passive rows around IC-heavy leaves.
    cfg["leaf_passive_ordering_enabled"] = bool(
        base_cfg.get("leaf_passive_ordering_enabled", passive_count >= 4)
    )
    cfg["leaf_passive_ordering_axis_bias"] = str(
        base_cfg.get(
            "leaf_passive_ordering_axis_bias",
            "horizontal" if connector_count <= 1 and ic_like_count >= 1 else "auto",
        )
    )
    cfg["leaf_passive_ordering_net_bias"] = bool(
        base_cfg.get("leaf_passive_ordering_net_bias", True)
    )
    cfg["leaf_passive_ordering_strength"] = float(
        base_cfg.get("leaf_passive_ordering_strength", 0.35)
    )
    cfg["leaf_passive_ordering_max_displacement_mm"] = float(
        base_cfg.get("leaf_passive_ordering_max_displacement_mm", 2.5)
    )
    cfg["leaf_passive_ordering_min_anchor_clearance_mm"] = float(
        base_cfg.get("leaf_passive_ordering_min_anchor_clearance_mm", 1.0)
    )

    return cfg


def _copy_components_with_translation(
    components: dict[str, Component],
    delta: Point,
) -> dict[str, Component]:
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


def _copy_traces_with_translation(traces: list[Any], delta: Point) -> list[Any]:
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


def _copy_vias_with_translation(vias: list[Any], delta: Point) -> list[Any]:
    translated: list[Any] = []
    for via in vias:
        new_via = copy.deepcopy(via)
        new_via.pos = Point(new_via.pos.x + delta.x, new_via.pos.y + delta.y)
        translated.append(new_via)
    return translated


def _component_net_degree_map(extraction: ExtractedSubcircuitBoard) -> dict[str, int]:
    degree_by_ref: dict[str, int] = {}
    for net in extraction.local_state.nets.values():
        refs = {ref for ref, _ in net.pad_refs}
        if len(refs) < 2:
            continue
        weight = max(1, len(refs) - 1)
        for ref in refs:
            degree_by_ref[ref] = degree_by_ref.get(ref, 0) + weight
    return degree_by_ref


def _component_primary_net_map(
    extraction: ExtractedSubcircuitBoard,
) -> dict[str, tuple[str, int]]:
    primary: dict[str, tuple[str, int]] = {}
    for net in extraction.local_state.nets.values():
        refs = [ref for ref, _ in net.pad_refs]
        if len(refs) < 2:
            continue
        weight = len(refs)
        for ref in refs:
            current = primary.get(ref)
            candidate = (net.name, weight)
            if (
                current is None
                or candidate[1] > current[1]
                or (candidate[1] == current[1] and candidate[0] < current[0])
            ):
                primary[ref] = candidate
    return primary


def _component_net_map(
    extraction: ExtractedSubcircuitBoard,
) -> dict[str, set[str]]:
    nets_by_ref: dict[str, set[str]] = {}
    for net in extraction.local_state.nets.values():
        refs = {ref for ref, _ in net.pad_refs}
        if len(refs) < 2:
            continue
        for ref in refs:
            nets_by_ref.setdefault(ref, set()).add(net.name)
    return nets_by_ref


def _component_adjacency_map(
    extraction: ExtractedSubcircuitBoard,
) -> dict[str, dict[str, int]]:
    adjacency: dict[str, dict[str, int]] = {}
    for net in extraction.local_state.nets.values():
        refs = sorted({ref for ref, _ in net.pad_refs})
        if len(refs) < 2:
            continue
        weight = max(1, len(refs) - 1)
        for ref in refs:
            adjacency.setdefault(ref, {})
        for i, ref_a in enumerate(refs):
            for ref_b in refs[i + 1 :]:
                adjacency[ref_a][ref_b] = adjacency[ref_a].get(ref_b, 0) + weight
                adjacency[ref_b][ref_a] = adjacency[ref_b].get(ref_a, 0) + weight
    return adjacency


def _build_leaf_passive_topology_groups(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
) -> list[dict[str, Any]]:
    components = solved_components
    passives = {
        ref
        for ref, comp in components.items()
        if not comp.locked and comp.kind == "passive"
    }
    if len(passives) < 4:
        return []

    degree_by_ref = _component_net_degree_map(extraction)
    primary_net_by_ref = _component_primary_net_map(extraction)
    nets_by_ref = _component_net_map(extraction)
    adjacency = _component_adjacency_map(extraction)

    anchor_refs = [
        ref
        for ref, comp in components.items()
        if comp.kind in ("ic", "regulator", "connector") and ref not in passives
    ]
    if not anchor_refs:
        return []

    anchor_to_passives: dict[str, list[str]] = {}
    for passive_ref in sorted(passives):
        best_anchor = None
        best_key = None
        passive_nets = nets_by_ref.get(passive_ref, set())
        for anchor_ref in anchor_refs:
            shared_nets = len(passive_nets & nets_by_ref.get(anchor_ref, set()))
            edge_weight = adjacency.get(passive_ref, {}).get(anchor_ref, 0)
            anchor_degree = degree_by_ref.get(anchor_ref, 0)
            key = (shared_nets, edge_weight, anchor_degree, components[anchor_ref].area)
            if best_key is None or key > best_key:
                best_key = key
                best_anchor = anchor_ref
        if (
            best_anchor is not None
            and best_key is not None
            and (best_key[0] > 0 or best_key[1] > 0)
        ):
            anchor_to_passives.setdefault(best_anchor, []).append(passive_ref)

    topology_groups: list[dict[str, Any]] = []
    for anchor_ref, passive_refs in anchor_to_passives.items():
        if len(passive_refs) < 2:
            continue

        remaining = set(passive_refs)
        chains: list[list[str]] = []

        while remaining:
            seed = max(
                remaining,
                key=lambda ref: (
                    degree_by_ref.get(ref, 0),
                    primary_net_by_ref.get(ref, ("", 0))[1],
                    ref,
                ),
            )
            chain = [seed]
            remaining.remove(seed)

            while True:
                tail = chain[-1]
                candidates = [
                    ref for ref in remaining if adjacency.get(tail, {}).get(ref, 0) > 0
                ]
                if not candidates:
                    break
                next_ref = max(
                    candidates,
                    key=lambda ref: (
                        adjacency.get(tail, {}).get(ref, 0),
                        len(nets_by_ref.get(tail, set()) & nets_by_ref.get(ref, set())),
                        primary_net_by_ref.get(ref, ("", 0))[1],
                        -components[ref].area,
                        ref,
                    ),
                )
                chain.append(next_ref)
                remaining.remove(next_ref)

            extended = True
            while extended:
                extended = False
                head = chain[0]
                candidates = [
                    ref for ref in remaining if adjacency.get(head, {}).get(ref, 0) > 0
                ]
                if candidates:
                    prev_ref = max(
                        candidates,
                        key=lambda ref: (
                            adjacency.get(head, {}).get(ref, 0),
                            len(
                                nets_by_ref.get(head, set())
                                & nets_by_ref.get(ref, set())
                            ),
                            primary_net_by_ref.get(ref, ("", 0))[1],
                            -components[ref].area,
                            ref,
                        ),
                    )
                    chain.insert(0, prev_ref)
                    remaining.remove(prev_ref)
                    extended = True

            chains.append(chain)

        topology_groups.append(
            {
                "anchor_ref": anchor_ref,
                "chains": chains,
            }
        )

    return topology_groups


def _apply_leaf_passive_ordering(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
) -> dict[str, Component]:
    if not bool(cfg.get("leaf_passive_ordering_enabled", False)):
        return copy.deepcopy(solved_components)

    ordered = copy.deepcopy(solved_components)
    topology_groups = _build_leaf_passive_topology_groups(extraction, ordered)
    if not topology_groups:
        return ordered

    degree_by_ref = _component_net_degree_map(extraction)
    axis_bias = str(cfg.get("leaf_passive_ordering_axis_bias", "auto")).lower()
    grid = max(0.25, float(cfg.get("placement_grid_mm", 0.5)))
    gap = max(
        0.6,
        float(cfg.get("placement_clearance_mm", 1.0)) * 0.6,
    )
    blend_strength = max(
        0.0,
        min(1.0, float(cfg.get("leaf_passive_ordering_strength", 0.35))),
    )
    max_displacement = max(
        0.5,
        float(cfg.get("leaf_passive_ordering_max_displacement_mm", 2.5)),
    )
    min_anchor_clearance = max(
        0.0,
        float(cfg.get("leaf_passive_ordering_min_anchor_clearance_mm", 1.0)),
    )

    def _bbox_for(comp: Component, pos: Point) -> tuple[float, float, float, float]:
        cx = comp.body_center.x if comp.body_center is not None else comp.pos.x
        cy = comp.body_center.y if comp.body_center is not None else comp.pos.y
        dx = pos.x - comp.pos.x
        dy = pos.y - comp.pos.y
        cx += dx
        cy += dy
        return (
            cx - comp.width_mm / 2,
            cy - comp.height_mm / 2,
            cx + comp.width_mm / 2,
            cy + comp.height_mm / 2,
        )

    def _overlaps_anchor(
        anchor_comp: Component | None,
        comp: Component,
        pos: Point,
    ) -> bool:
        if anchor_comp is None or comp.ref == anchor_comp.ref:
            return False
        a_l, a_t, a_r, a_b = _bbox_for(anchor_comp, anchor_comp.pos)
        c_l, c_t, c_r, c_b = _bbox_for(comp, pos)
        return not (
            c_r <= a_l - min_anchor_clearance
            or c_l >= a_r + min_anchor_clearance
            or c_b <= a_t - min_anchor_clearance
            or c_t >= a_b + min_anchor_clearance
        )

    total_aligned = 0

    for topology_group in sorted(
        topology_groups,
        key=lambda item: (
            -degree_by_ref.get(item["anchor_ref"], 0),
            item["anchor_ref"],
        ),
    ):
        anchor_ref = topology_group["anchor_ref"]
        anchor_comp = ordered.get(anchor_ref)
        if anchor_comp is None:
            continue

        anchor_pos = Point(anchor_comp.pos.x, anchor_comp.pos.y)
        chains = [chain for chain in topology_group["chains"] if len(chain) >= 2]
        if not chains:
            continue

        horizontal = axis_bias == "horizontal"
        if axis_bias == "auto":
            chain_points = [
                ordered[ref].pos for chain in chains for ref in chain if ref in ordered
            ]
            xs = [pt.x for pt in chain_points]
            ys = [pt.y for pt in chain_points]
            horizontal = (max(xs) - min(xs)) >= (max(ys) - min(ys))

        row_offset = 0.0
        for chain in chains:
            chain_refs = [ref for ref in chain if ref in ordered]
            if len(chain_refs) < 2:
                continue

            max_w = max(ordered[ref].width_mm for ref in chain_refs)
            max_h = max(ordered[ref].height_mm for ref in chain_refs)

            if horizontal:
                pitch = max_w + gap
                start_x = anchor_pos.x - ((len(chain_refs) - 1) * pitch) / 2.0
                target_y = anchor_pos.y + row_offset
                for idx, ref in enumerate(chain_refs):
                    comp = ordered[ref]
                    raw_tx = round((start_x + idx * pitch) / grid) * grid
                    raw_ty = round(target_y / grid) * grid
                    dx = max(
                        -max_displacement,
                        min(max_displacement, raw_tx - comp.pos.x),
                    )
                    dy = max(
                        -max_displacement,
                        min(max_displacement, raw_ty - comp.pos.y),
                    )
                    tx = comp.pos.x + dx * blend_strength
                    ty = comp.pos.y + dy * blend_strength
                    candidate = Point(tx, ty)
                    if _overlaps_anchor(anchor_comp, comp, candidate):
                        continue
                    old_pos = Point(comp.pos.x, comp.pos.y)
                    comp.pos = candidate
                    _update_pad_positions(comp, old_pos, comp.rotation)
                    total_aligned += 1
                row_offset += max_h + gap
            else:
                pitch = max_h + gap
                start_y = anchor_pos.y - ((len(chain_refs) - 1) * pitch) / 2.0
                target_x = anchor_pos.x + row_offset
                for idx, ref in enumerate(chain_refs):
                    comp = ordered[ref]
                    raw_tx = round(target_x / grid) * grid
                    raw_ty = round((start_y + idx * pitch) / grid) * grid
                    dx = max(
                        -max_displacement,
                        min(max_displacement, raw_tx - comp.pos.x),
                    )
                    dy = max(
                        -max_displacement,
                        min(max_displacement, raw_ty - comp.pos.y),
                    )
                    tx = comp.pos.x + dx * blend_strength
                    ty = comp.pos.y + dy * blend_strength
                    candidate = Point(tx, ty)
                    if _overlaps_anchor(anchor_comp, comp, candidate):
                        continue
                    old_pos = Point(comp.pos.x, comp.pos.y)
                    comp.pos = candidate
                    _update_pad_positions(comp, old_pos, comp.rotation)
                    total_aligned += 1
                row_offset += max_w + gap

    return ordered


def _tight_leaf_geometry_bounds(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    routing: dict[str, Any],
) -> dict[str, float]:
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
        for pad in comp.pads:
            min_x = min(min_x, pad.pos.x)
            min_y = min(min_y, pad.pos.y)
            max_x = max(max_x, pad.pos.x)
            max_y = max(max_y, pad.pos.y)

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


def _build_reduced_leaf_extraction(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    routing: dict[str, Any],
    outline: tuple[Point, Point],
) -> ExtractedSubcircuitBoard:
    tl, br = outline
    delta = Point(-tl.x, -tl.y)
    local_state = copy.deepcopy(extraction.local_state)
    local_state.components = _copy_components_with_translation(solved_components, delta)
    local_state.traces = _copy_traces_with_translation(
        routing.get("_trace_segments", []),
        delta,
    )
    local_state.vias = _copy_vias_with_translation(
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


def _leaf_size_reduction_candidates(
    current_width: float,
    current_height: float,
    min_width: float,
    min_height: float,
) -> list[dict[str, Any]]:
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


def _attempt_leaf_size_reduction(
    extraction: ExtractedSubcircuitBoard,
    best_round: SolveRoundResult,
    cfg: dict[str, Any],
) -> tuple[ExtractedSubcircuitBoard, SolveRoundResult, dict[str, Any]]:
    original_tl, original_br = extraction.local_state.board_outline
    original_width = extraction.local_state.board_width
    original_height = extraction.local_state.board_height
    pad_inset_margin = max(
        0.0,
        float(cfg.get("pad_inset_margin_mm", 0.3)),
    )
    outline_margin = max(
        pad_inset_margin,
        float(cfg.get("leaf_size_reduction_margin_mm", 0.5)),
    )

    geometry_bounds = _tight_leaf_geometry_bounds(
        extraction,
        best_round.components,
        best_round.routing,
    )
    min_width = max(1.0, geometry_bounds["width_mm"] + 2.0 * outline_margin)
    min_height = max(1.0, geometry_bounds["height_mm"] + 2.0 * outline_margin)

    summary: dict[str, Any] = {
        "attempted": True,
        "enabled": True,
        "accepted": False,
        "passes": 0,
        "original_outline": {
            "top_left_x": original_tl.x,
            "top_left_y": original_tl.y,
            "bottom_right_x": original_br.x,
            "bottom_right_y": original_br.y,
            "width_mm": original_width,
            "height_mm": original_height,
        },
        "reduced_outline": {
            "top_left_x": original_tl.x,
            "top_left_y": original_tl.y,
            "bottom_right_x": original_br.x,
            "bottom_right_y": original_br.y,
            "width_mm": original_width,
            "height_mm": original_height,
        },
        "tight_geometry_bounds": geometry_bounds,
        "outline_margin_mm": outline_margin,
        "pad_inset_margin_mm": pad_inset_margin,
        "attempts": [],
        "outline_reduction_mm": {
            "width_mm": 0.0,
            "height_mm": 0.0,
            "area_mm2": 0.0,
        },
        "outline_reduction_percent": {
            "width_percent": 0.0,
            "height_percent": 0.0,
            "area_percent": 0.0,
        },
        "validation": {
            "accepted": True,
            "reason": "original_outline_retained",
        },
    }

    if best_round.routing.get("failed", False):
        summary["validation"] = {
            "accepted": False,
            "reason": "best_round_not_accepted",
        }
        return extraction, best_round, summary

    if best_round.routing.get("reason") == "no_internal_nets":
        summary["validation"] = {
            "accepted": False,
            "reason": "no_internal_nets",
        }
        return extraction, best_round, summary

    if min_width >= original_width and min_height >= original_height:
        summary["validation"] = {
            "accepted": False,
            "reason": "no_shrink_headroom",
        }
        return extraction, best_round, summary

    current_width = original_width
    current_height = original_height
    current_extraction = extraction
    current_round = best_round

    max_attempts = max(1, int(cfg.get("leaf_size_reduction_max_attempts", 3)))
    max_passes = max(1, int(cfg.get("leaf_size_reduction_max_passes", 1)))
    total_attempts = 0

    while True:
        if total_attempts >= max_attempts:
            summary["validation"] = {
                "accepted": True,
                "reason": "attempt_limit_reached",
                "passes": summary["passes"],
                "attempts": total_attempts,
            }
            break

        candidates = _leaf_size_reduction_candidates(
            current_width,
            current_height,
            min_width,
            min_height,
        )
        if not candidates:
            break

        accepted_candidate = False
        for candidate in candidates:
            if total_attempts >= max_attempts:
                break
            if int(summary["passes"]) >= max_passes:
                summary["validation"] = {
                    "accepted": True,
                    "reason": "pass_limit_reached",
                    "passes": summary["passes"],
                    "attempts": total_attempts,
                }
                break
            candidate_width = float(candidate["width_mm"])
            candidate_height = float(candidate["height_mm"])
            candidate_outline = (
                Point(0.0, 0.0),
                Point(candidate_width, candidate_height),
            )
            candidate_extraction = _build_reduced_leaf_extraction(
                current_extraction,
                current_round.components,
                current_round.routing,
                candidate_outline,
            )
            candidate_cfg = _local_solver_config(cfg, candidate_extraction)
            candidate_cfg["board_width_mm"] = candidate_width
            candidate_cfg["board_height_mm"] = candidate_height

            legality_solver = PlacementSolver(
                candidate_extraction.local_state, candidate_cfg, seed=0
            )
            legality = legality_solver.legality_diagnostics(
                candidate_extraction.local_state.components
            )
            total_attempts += 1
            attempt_record = {
                "axis": candidate["axis"],
                "step_mm": candidate["step_mm"],
                "attempt_index": total_attempts,
                "candidate_outline": {
                    "top_left_x": 0.0,
                    "top_left_y": 0.0,
                    "bottom_right_x": candidate_width,
                    "bottom_right_y": candidate_height,
                    "width_mm": candidate_width,
                    "height_mm": candidate_height,
                },
                "legality": copy.deepcopy(legality),
                "accepted": False,
            }

            if legality.get("pad_outside_count", 0) or legality.get("overlap_count", 0):
                attempt_record["rejection_reason"] = "placement_legality_failed"
                summary["attempts"].append(attempt_record)
                continue

            width_delta = current_width - candidate_width
            height_delta = current_height - candidate_height
            reroute_threshold = float(
                cfg.get("leaf_size_reduction_reroute_threshold_mm", 1.5)
            )
            should_reroute = (
                candidate["axis"] == "both"
                or width_delta > reroute_threshold
                or height_delta > reroute_threshold
            )

            rerouted: dict[str, Any] = {}
            reroute_timing: dict[str, float] = {}

            if not should_reroute:
                if current_round.routing.get("validation", {}).get("accepted", False):
                    rerouted = copy.deepcopy(current_round.routing)
                    rerouted["validation"] = copy.deepcopy(
                        current_round.routing.get("validation", {})
                    )
                    rerouted["render_diagnostics"] = {
                        "skipped": True,
                        "reason": "size_reduction_reused_previous_route",
                    }
                    rerouted["size_reduction_reused_route"] = True
                else:
                    try:
                        rerouted, reroute_timing = _route_local_subcircuit(
                            candidate_extraction,
                            candidate_extraction.local_state.components,
                            candidate_cfg,
                            generate_diagnostics=False,
                            round_index=current_round.round_index,
                        )
                    except Exception as exc:
                        attempt_record["rejection_reason"] = f"reroute_exception:{exc}"
                        summary["attempts"].append(attempt_record)
                        continue

            attempt_record["routing"] = {
                key: value for key, value in rerouted.items() if not key.startswith("_")
            }
            attempt_record["timing_breakdown"] = dict(reroute_timing)
            validation = rerouted.get("validation", {}) or {}
            attempt_record["size_reduction_validation"] = copy.deepcopy(validation)

            if rerouted.get("failed", False) or not validation.get("accepted", False):
                attempt_record["rejection_reason"] = (
                    validation.get("rejection_stage")
                    or validation.get("rejection_message")
                    or rerouted.get("reason")
                    or "reroute_validation_failed"
                )
                summary["attempts"].append(attempt_record)
                continue

            accepted_round = SolveRoundResult(
                round_index=current_round.round_index,
                seed=current_round.seed,
                score=current_round.score,
                placement=current_round.placement,
                components=copy.deepcopy(candidate_extraction.local_state.components),
                routing=rerouted,
                routed=True,
                timing_breakdown=dict(reroute_timing),
            )
            current_width = candidate_width
            current_height = candidate_height
            current_extraction = candidate_extraction
            current_round = accepted_round
            attempt_record["accepted"] = True
            summary["attempts"].append(attempt_record)
            summary["passes"] = int(summary["passes"]) + 1
            accepted_candidate = True
            break

        if int(summary["passes"]) >= max_passes:
            summary["validation"] = {
                "accepted": True,
                "reason": "pass_limit_reached",
                "passes": summary["passes"],
                "attempts": total_attempts,
            }
            break

        if not accepted_candidate:
            break

    reduced_tl, reduced_br = current_extraction.local_state.board_outline
    reduced_width = current_extraction.local_state.board_width
    reduced_height = current_extraction.local_state.board_height
    original_area = original_width * original_height
    reduced_area = reduced_width * reduced_height
    width_reduction = max(0.0, original_width - reduced_width)
    height_reduction = max(0.0, original_height - reduced_height)
    area_reduction = max(0.0, original_area - reduced_area)

    summary["accepted"] = summary["passes"] > 0
    summary["reduced_outline"] = {
        "top_left_x": reduced_tl.x,
        "top_left_y": reduced_tl.y,
        "bottom_right_x": reduced_br.x,
        "bottom_right_y": reduced_br.y,
        "width_mm": reduced_width,
        "height_mm": reduced_height,
    }
    summary["outline_reduction_mm"] = {
        "width_mm": width_reduction,
        "height_mm": height_reduction,
        "area_mm2": area_reduction,
    }
    summary["outline_reduction_percent"] = {
        "width_percent": 0.0
        if original_width <= 0.0
        else (width_reduction / original_width) * 100.0,
        "height_percent": 0.0
        if original_height <= 0.0
        else (height_reduction / original_height) * 100.0,
        "area_percent": 0.0
        if original_area <= 0.0
        else (area_reduction / original_area) * 100.0,
    }
    if summary.get("validation", {}).get("reason") not in {
        "attempt_limit_reached",
        "pass_limit_reached",
    }:
        summary["validation"] = {
            "accepted": True,
            "reason": "reduced_outline_kept"
            if summary["accepted"]
            else "original_outline_retained",
            "passes": summary["passes"],
            "attempts": total_attempts,
        }
    else:
        summary["validation"]["attempts"] = total_attempts

    return current_extraction, current_round, summary


def _score_local_components(
    local_state: BoardState,
    components: dict[str, Component],
    cfg: dict[str, Any],
) -> PlacementScore:
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


def _repair_leaf_placement_legality(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
) -> tuple[dict[str, Component], dict[str, Any]]:
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


def _route_local_subcircuit(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
    *,
    generate_diagnostics: bool = True,
    round_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    fast_smoke_mode = bool(cfg.get("subcircuit_fast_smoke_mode", False))
    render_pre_route_board_views = bool(
        cfg.get("subcircuit_render_pre_route_board_views", not fast_smoke_mode)
    )
    render_routed_board_views = bool(
        cfg.get("subcircuit_render_routed_board_views", True)
    )
    render_pre_route_drc_overlay = bool(
        cfg.get("subcircuit_render_pre_route_drc_overlay", not fast_smoke_mode)
    )
    render_routed_drc_overlay = bool(
        cfg.get("subcircuit_render_routed_drc_overlay", not fast_smoke_mode)
    )
    write_pre_route_drc_json = bool(
        cfg.get("subcircuit_write_pre_route_drc_json", not fast_smoke_mode)
    )
    write_routed_drc_json = bool(cfg.get("subcircuit_write_routed_drc_json", True))
    write_pre_route_drc_report = bool(
        cfg.get("subcircuit_write_pre_route_drc_report", not fast_smoke_mode)
    )
    write_routed_drc_report = bool(cfg.get("subcircuit_write_routed_drc_report", True))
    build_comparison_contact_sheet = bool(
        cfg.get("subcircuit_build_comparison_contact_sheet", not fast_smoke_mode)
    )
    if not extraction.internal_net_names:
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": "no_internal_nets",
                "router": "none",
                "traces": 0,
                "vias": 0,
                "total_length_mm": 0.0,
                "routed_internal_nets": [],
                "failed_internal_nets": [],
                "_trace_segments": [],
                "_via_objects": [],
                "validation": {
                    "accepted": True,
                    "reason": "no_internal_nets",
                },
                "failed": False,
            },
            {},
        )

    artifact_paths = resolve_artifact_paths(
        Path(extraction.subcircuit.schematic_path).parent,
        extraction.subcircuit.id,
    )
    pre_route_board = (
        Path(artifact_paths.artifact_dir) / "leaf_pre_freerouting.kicad_pcb"
    )
    routed_board = Path(artifact_paths.artifact_dir) / "leaf_routed.kicad_pcb"
    illegal_board = (
        Path(artifact_paths.artifact_dir) / "leaf_illegal_pre_stamp.kicad_pcb"
    )

    route_timing: dict[str, float] = {}
    route_total_start = time.monotonic()

    legality_start = time.monotonic()
    repaired_components, legality_repair = _repair_leaf_placement_legality(
        extraction,
        solved_components,
        cfg,
    )
    route_timing["legality_repair_s"] = round(
        max(0.0, time.monotonic() - legality_start), 3
    )

    source_pcb = Path(cfg.get("subcircuit_route_source_pcb", cfg.get("pcb_path", "")))
    if not source_pcb.exists():
        source_pcb = _default_pcb_path(Path(extraction.subcircuit.schematic_path))

    if not source_pcb.exists():
        raise RuntimeError(
            "Leaf FreeRouting requires a real source PCB to stamp from; "
            f"could not resolve base board for {extraction.subcircuit.id.instance_path}"
        )

    if not legality_repair.get("resolved", False):
        diagnostics = legality_repair.get("diagnostics", {}) or {}
        overlap_count = int(diagnostics.get("overlap_count", 0) or 0)
        pad_outside_count = int(diagnostics.get("pad_outside_count", 0) or 0)
        overlap_pairs = [
            f"{item.get('a', '?')}:{item.get('b', '?')}"
            for item in diagnostics.get("overlaps", [])
        ]
        pad_violations = [
            f"{item.get('ref', '?')}:{item.get('pad_id', '?')}:{','.join(item.get('sides', []))}"
            for item in diagnostics.get("pads_outside_board", [])
        ]

        overlap_details = []
        for item in diagnostics.get("overlaps", []):
            overlap_details.append(
                {
                    "a": item.get("a"),
                    "b": item.get("b"),
                    "overlap_x_mm": item.get("overlap_x_mm"),
                    "overlap_y_mm": item.get("overlap_y_mm"),
                    "overlap_area_mm2": item.get("overlap_area_mm2"),
                }
            )

        component_debug = []
        repaired_by_ref = repaired_components or {}
        for ref in sorted(repaired_by_ref.keys()):
            comp = repaired_by_ref[ref]
            component_debug.append(
                {
                    "ref": ref,
                    "kind": comp.kind,
                    "layer": str(comp.layer),
                    "locked": bool(comp.locked),
                    "x_mm": round(comp.pos.x, 4),
                    "y_mm": round(comp.pos.y, 4),
                    "rotation_deg": round(comp.rotation, 4),
                    "width_mm": round(comp.width_mm, 4),
                    "height_mm": round(comp.height_mm, 4),
                    "pad_count": len(comp.pads),
                }
            )

        print(
            "  Leaf legality repair rejected placement: "
            f"overlaps={overlap_count} "
            f"pads_outside={pad_outside_count} "
            f"overlap_pairs={overlap_pairs} "
            f"pad_violations={pad_violations}"
        )
        if overlap_details:
            print(f"  Leaf legality overlap details: {overlap_details}")
        if component_debug:
            print(f"  Leaf legality component states: {component_debug}")

        illegal_input_board = copy.deepcopy(extraction.local_state)
        illegal_input_board.components = copy.deepcopy(repaired_components)
        illegal_input_board.traces = []
        illegal_input_board.vias = []

        illegal_render_diagnostics: dict[str, Any] = {
            "artifact_dir": artifact_paths.artifact_dir,
            "renders_dir": str(Path(artifact_paths.artifact_dir) / "renders"),
            "illegal_pre_stamp": None,
            "errors": [],
        }

        try:
            route_adapter = KiCadAdapter(str(source_pcb), config=cfg)
            route_adapter.stamp_subcircuit_board(
                illegal_input_board,
                output_path=str(illegal_board),
                clear_existing_tracks=True,
                clear_existing_zones=True,
                remove_unmapped_footprints=True,
            )
            illegal_validation = {
                "accepted": False,
                "rejected": True,
                "rejection_stage": "leaf_pre_stamp_legality_repair",
                "rejection_reasons": ["illegal_unrepaired_leaf_placement"],
                "leaf_legality_repair": copy.deepcopy(legality_repair),
                "drc": {
                    "violations": [],
                    "report_text": (
                        "Leaf placement rejected before routing due to placement legality.\n"
                        f"overlap_count={overlap_count}\n"
                        f"pad_outside_count={pad_outside_count}\n"
                        f"overlap_pairs={overlap_pairs}\n"
                        f"pad_violations={pad_violations}\n"
                    ),
                },
            }
            illegal_render_diagnostics["illegal_pre_stamp"] = (
                generate_stage_diagnostic_artifacts(
                    pcb_path=str(illegal_board),
                    validation=illegal_validation,
                    artifact_dir=artifact_paths.artifact_dir,
                    stage="illegal_pre_stamp",
                    render_board_views=not fast_smoke_mode,
                    write_drc_json=not fast_smoke_mode,
                    write_drc_report=not fast_smoke_mode,
                    render_drc_overlay=not fast_smoke_mode,
                )
            )
        except Exception as exc:
            illegal_render_diagnostics["errors"].append(
                f"illegal_pre_stamp_render_failed:{exc}"
            )

        route_timing["route_local_subcircuit_total_s"] = round(
            max(0.0, time.monotonic() - route_total_start), 3
        )
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": "illegal_unrepaired_leaf_placement",
                "router": "freerouting",
                "traces": 0,
                "vias": 0,
                "total_length_mm": 0.0,
                "routed_internal_nets": [],
                "failed_internal_nets": list(sorted(extraction.internal_net_names)),
                "_trace_segments": [],
                "_via_objects": [],
                "validation": {
                    "accepted": False,
                    "rejected": True,
                    "rejection_stage": "leaf_pre_stamp_legality_repair",
                    "rejection_reasons": ["illegal_unrepaired_leaf_placement"],
                    "leaf_legality_repair": copy.deepcopy(legality_repair),
                    "render_diagnostics": copy.deepcopy(illegal_render_diagnostics)
                    if generate_diagnostics
                    else {"skipped": True, "reason": "size_reduction_fast_path"},
                    "illegal_pre_stamp_board_path": str(illegal_board),
                },
                "leaf_legality_repair": copy.deepcopy(legality_repair),
                "render_diagnostics": copy.deepcopy(illegal_render_diagnostics)
                if generate_diagnostics
                else {"skipped": True, "reason": "size_reduction_fast_path"},
                "illegal_pre_stamp_board_path": str(illegal_board),
                "failed": True,
            },
            route_timing,
        )

    route_input_board = copy.deepcopy(extraction.local_state)
    route_input_board.components = copy.deepcopy(repaired_components)
    route_input_board.traces = []
    route_input_board.vias = []

    stamp_start = time.monotonic()
    route_adapter = KiCadAdapter(str(source_pcb), config=cfg)
    route_adapter.stamp_subcircuit_board(
        route_input_board,
        output_path=str(pre_route_board),
        clear_existing_tracks=True,
        clear_existing_zones=True,
        remove_unmapped_footprints=True,
    )
    route_timing["stamp_pre_route_board_s"] = round(
        max(0.0, time.monotonic() - stamp_start), 3
    )

    jar_path = cfg.get("freerouting_jar")
    if not jar_path:
        raise RuntimeError(
            "Leaf FreeRouting requires 'freerouting_jar' to be configured"
        )

    freerouting_start = time.monotonic()
    freerouting_stats = route_with_freerouting(
        str(pre_route_board),
        str(routed_board),
        str(jar_path),
        {
            **cfg,
            "pcb_path": str(source_pcb),
            "freerouting_preserve_existing_copper": False,
        },
    )
    route_timing["freerouting_s"] = round(
        max(0.0, time.monotonic() - freerouting_start), 3
    )

    pre_route_validation_start = time.monotonic()
    pre_route_validation = validate_routed_board(
        str(pre_route_board),
        expected_anchor_names=[port.name for port in extraction.interface_ports],
        actual_anchor_names=[port.name for port in extraction.interface_ports],
        required_anchor_names=[
            port.name for port in extraction.interface_ports if port.required
        ],
        timeout_s=int(cfg.get("subcircuit_validation_timeout_s", 30)),
    )
    route_timing["pre_route_validation_s"] = round(
        max(0.0, time.monotonic() - pre_route_validation_start), 3
    )
    # Pre-route DRC is informational only — we let FreeRouting attempt routing
    # regardless of pre-route violations. The post-route DRC gate handles acceptance.
    pre_route_drc = pre_route_validation.get("drc", {})
    if pre_route_drc.get("violations"):
        pre_route_violation_types = {v.get("type") for v in pre_route_drc["violations"]}
        print(
            f"  Pre-route DRC info: {len(pre_route_drc['violations'])} violations ({', '.join(sorted(pre_route_violation_types))})"
        )
    if generate_diagnostics:
        pre_route_render_start = time.monotonic()
        leaf_diagnostics = generate_leaf_diagnostic_artifacts(
            artifact_dir=artifact_paths.artifact_dir,
            pre_route_board=str(pre_route_board),
            routed_board=str(routed_board) if routed_board.exists() else None,
            pre_route_validation=pre_route_validation,
            routed_validation=None,
            render_pre_route_board_views=render_pre_route_board_views,
            render_routed_board_views=False,
            write_pre_route_drc_json=write_pre_route_drc_json,
            write_routed_drc_json=False,
            write_pre_route_drc_report=write_pre_route_drc_report,
            write_routed_drc_report=False,
            render_pre_route_drc_overlay=render_pre_route_drc_overlay,
            render_routed_drc_overlay=False,
            build_comparison_contact_sheet_enabled=False,
            quiet_board_render=fast_smoke_mode,
        )
        route_timing["pre_route_render_diagnostics_s"] = round(
            max(0.0, time.monotonic() - pre_route_render_start), 3
        )
    else:
        leaf_diagnostics = {
            "skipped": True,
            "reason": "size_reduction_fast_path",
        }
        route_timing["pre_route_render_diagnostics_s"] = 0.0

    round_board_illegal_pre_stamp = ""
    round_board_pre_route = ""
    round_board_routed = ""

    if round_index is not None:
        round_prefix = f"round_{int(round_index):04d}"

        def _copy_round_board(
            source_path: Path,
            suffix: str,
        ) -> str:
            if not source_path.exists():
                return ""
            destination = (
                source_path.parent / f"{round_prefix}_{suffix}{source_path.suffix}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            return str(destination)

        round_board_pre_route = _copy_round_board(
            pre_route_board,
            "leaf_pre_freerouting",
        )

    if round_index is not None and not leaf_diagnostics.get("skipped", False):
        renders_dir = Path(artifact_paths.artifact_dir) / "renders"
        round_prefix = f"round_{int(round_index):04d}"

        def _copy_round_preview(
            source_path: str | None,
            suffix: str,
        ) -> str:
            if not source_path:
                return ""
            source = Path(source_path)
            if not source.exists():
                return ""
            destination = renders_dir / f"{round_prefix}_{suffix}{source.suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return str(destination)

        pre_route_views = leaf_diagnostics.get("pre_route", {}).get("board_views", {})
        if isinstance(pre_route_views, dict):
            pre_route_paths = pre_route_views.get("paths", {})
            if isinstance(pre_route_paths, dict):
                round_pre_front = _copy_round_preview(
                    pre_route_paths.get("front_all"),
                    "pre_route_front_all",
                )
                round_pre_back = _copy_round_preview(
                    pre_route_paths.get("back_all"),
                    "pre_route_back_all",
                )
                round_pre_copper = _copy_round_preview(
                    pre_route_paths.get("copper_both"),
                    "pre_route_copper_both",
                )
                if round_pre_front:
                    pre_route_paths["round_front_all"] = round_pre_front
                if round_pre_back:
                    pre_route_paths["round_back_all"] = round_pre_back
                if round_pre_copper:
                    pre_route_paths["round_copper_both"] = round_pre_copper

    pre_route_validation["render_diagnostics"] = copy.deepcopy(leaf_diagnostics)
    pre_route_validation["leaf_legality_repair"] = copy.deepcopy(legality_repair)
    if round_board_pre_route:
        pre_route_validation["round_board_pre_route"] = round_board_pre_route

    import_copper_start = time.monotonic()
    imported_copper = import_routed_copper(str(routed_board))
    route_timing["import_routed_copper_s"] = round(
        max(0.0, time.monotonic() - import_copper_start), 3
    )

    routed_validation_start = time.monotonic()
    validation = validate_routed_board(
        str(routed_board),
        expected_anchor_names=[port.name for port in extraction.interface_ports],
        actual_anchor_names=[port.name for port in extraction.interface_ports],
        required_anchor_names=[
            port.name for port in extraction.interface_ports if port.required
        ],
        timeout_s=int(cfg.get("subcircuit_validation_timeout_s", 30)),
    )
    route_timing["routed_validation_s"] = round(
        max(0.0, time.monotonic() - routed_validation_start), 3
    )
    if generate_diagnostics:
        routed_render_start = time.monotonic()
        leaf_diagnostics = generate_leaf_diagnostic_artifacts(
            artifact_dir=artifact_paths.artifact_dir,
            pre_route_board=str(pre_route_board),
            routed_board=str(routed_board),
            pre_route_validation=pre_route_validation,
            routed_validation=validation,
            render_pre_route_board_views=render_pre_route_board_views,
            render_routed_board_views=render_routed_board_views,
            write_pre_route_drc_json=write_pre_route_drc_json,
            write_routed_drc_json=write_routed_drc_json,
            write_pre_route_drc_report=write_pre_route_drc_report,
            write_routed_drc_report=write_routed_drc_report,
            render_pre_route_drc_overlay=render_pre_route_drc_overlay,
            render_routed_drc_overlay=render_routed_drc_overlay,
            build_comparison_contact_sheet_enabled=build_comparison_contact_sheet,
            quiet_board_render=fast_smoke_mode,
        )
        route_timing["routed_render_diagnostics_s"] = round(
            max(0.0, time.monotonic() - routed_render_start), 3
        )
    else:
        leaf_diagnostics = {
            "skipped": True,
            "reason": "size_reduction_fast_path",
        }
        route_timing["routed_render_diagnostics_s"] = 0.0

    round_preview_pre_route_front = ""
    round_preview_pre_route_back = ""
    round_preview_pre_route_copper = ""
    round_preview_routed_front = ""
    round_preview_routed_back = ""
    round_preview_routed_copper = ""

    if round_index is not None:
        round_prefix = f"round_{int(round_index):04d}"

        def _copy_round_board(
            source_path: Path,
            suffix: str,
        ) -> str:
            if not source_path.exists():
                return ""
            destination = (
                source_path.parent / f"{round_prefix}_{suffix}{source_path.suffix}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            return str(destination)

        round_board_routed = _copy_round_board(
            routed_board,
            "leaf_routed",
        )

    if round_index is not None and not leaf_diagnostics.get("skipped", False):
        renders_dir = Path(artifact_paths.artifact_dir) / "renders"
        round_prefix = f"round_{int(round_index):04d}"

        def _copy_round_preview(
            source_path: str | None,
            suffix: str,
        ) -> str:
            if not source_path:
                return ""
            source = Path(source_path)
            if not source.exists():
                return ""
            destination = renders_dir / f"{round_prefix}_{suffix}{source.suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return str(destination)

        pre_route_views = leaf_diagnostics.get("pre_route", {}).get("board_views", {})
        if isinstance(pre_route_views, dict):
            pre_route_paths = pre_route_views.get("paths", {})
            if isinstance(pre_route_paths, dict):
                round_preview_pre_route_front = _copy_round_preview(
                    pre_route_paths.get("front_all"),
                    "pre_route_front_all",
                )
                round_preview_pre_route_back = _copy_round_preview(
                    pre_route_paths.get("back_all"),
                    "pre_route_back_all",
                )
                round_preview_pre_route_copper = _copy_round_preview(
                    pre_route_paths.get("copper_both"),
                    "pre_route_copper_both",
                )
                if round_preview_pre_route_front:
                    pre_route_paths["round_front_all"] = round_preview_pre_route_front
                if round_preview_pre_route_back:
                    pre_route_paths["round_back_all"] = round_preview_pre_route_back
                if round_preview_pre_route_copper:
                    pre_route_paths["round_copper_both"] = (
                        round_preview_pre_route_copper
                    )

        routed_views = leaf_diagnostics.get("routed", {}).get("board_views", {})
        if isinstance(routed_views, dict):
            routed_paths = routed_views.get("paths", {})
            if isinstance(routed_paths, dict):
                round_preview_routed_front = _copy_round_preview(
                    routed_paths.get("front_all"),
                    "routed_front_all",
                )
                round_preview_routed_back = _copy_round_preview(
                    routed_paths.get("back_all"),
                    "routed_back_all",
                )
                round_preview_routed_copper = _copy_round_preview(
                    routed_paths.get("copper_both"),
                    "routed_copper_both",
                )
                if round_preview_routed_front:
                    routed_paths["round_front_all"] = round_preview_routed_front
                if round_preview_routed_back:
                    routed_paths["round_back_all"] = round_preview_routed_back
                if round_preview_routed_copper:
                    routed_paths["round_copper_both"] = round_preview_routed_copper

    validation["pre_route_validation"] = copy.deepcopy(pre_route_validation)
    validation["render_diagnostics"] = copy.deepcopy(leaf_diagnostics)
    if round_board_pre_route:
        validation["round_board_pre_route"] = round_board_pre_route
    if round_board_routed:
        validation["round_board_routed"] = round_board_routed

    drc = validation.get("drc", {})
    drc_stdout = str(drc.get("stdout", ""))
    drc_stderr = str(drc.get("stderr", ""))
    drc_report_text = "\n".join(
        part for part in (drc_stdout, drc_stderr) if part.strip()
    )

    # Post-route ignorable violation types: cosmetic issues and violations
    # that are inherent to the footprint or subcircuit outline, not caused
    # by the routing itself.
    ignorable_warning_types = {
        "silk_overlap",
        "lib_footprint_mismatch",
        "copper_edge_clearance",  # tight subcircuit outlines
        "silk_edge_clearance",  # cosmetic
        "silk_over_copper",  # cosmetic
        "solder_mask_bridge",  # footprint-internal
        "unconnected_items",  # FreeRouting may not route all nets
    }
    significant_violations = [
        violation
        for violation in drc.get("violations", [])
        if violation.get("type") not in ignorable_warning_types
    ]

    # --- Generalized DRC exception: config-driven patterns ---
    # If the config provides ignorable_drc_patterns (list of regex strings),
    # check whether ALL significant violations match at least one pattern.
    ignorable_drc_patterns = cfg.get("ignorable_drc_patterns", [])
    _compiled_drc_patterns = [re.compile(p) for p in ignorable_drc_patterns]
    _all_match_config_patterns = (
        significant_violations
        and _compiled_drc_patterns
        and all(
            any(pat.search(v.get("description", "")) for pat in _compiled_drc_patterns)
            for v in significant_violations
        )
        and not drc.get("shorts", 0)
    )

    # --- Generalized DRC exception: footprint-baseline clearance heuristic ---
    # If ALL significant violations are clearance-type violations whose
    # descriptions reference pads from the SAME single footprint, treat them
    # as footprint-internal baseline clearance issues (e.g. dense USB-C,
    # fine-pitch IC pads closer together than the board clearance rule).
    _footprint_ref_re = re.compile(r"\bof\s+(\S+)")
    _clearance_types = {"clearance", "hole_clearance", "solder_mask_bridge"}
    _all_clearance = (
        significant_violations
        and all(v.get("type") in _clearance_types for v in significant_violations)
        and not drc.get("shorts", 0)
    )
    _single_footprint_baseline = False
    _baseline_footprint_ref = None
    if _all_clearance:
        # Collect all footprint references mentioned across violations
        _violation_footprint_refs: set[str] = set()
        for v in significant_violations:
            desc = v.get("description", "")
            for m in _footprint_ref_re.finditer(desc):
                _violation_footprint_refs.add(m.group(1))
        # If every violation references pads from exactly one footprint,
        # this is a footprint-internal clearance issue.
        if len(_violation_footprint_refs) == 1:
            _single_footprint_baseline = True
            _baseline_footprint_ref = next(iter(_violation_footprint_refs))

    if _all_match_config_patterns or _single_footprint_baseline:
        _ignore_reason = (
            "config_ignorable_drc_patterns"
            if _all_match_config_patterns
            else f"footprint_baseline_clearance:{_baseline_footprint_ref}"
        )
        _ignored_types = {v.get("type") for v in significant_violations}
        validation["obviously_illegal_routed_geometry"] = False
        validation["rejection_reasons"] = [
            reason
            for reason in validation.get("rejection_reasons", [])
            if reason != "illegal_routed_geometry"
        ]
        validation["accepted"] = not validation["rejection_reasons"]
        validation["drc"]["ignored_violation_types"] = sorted(
            ignorable_warning_types | _ignored_types
        )
        validation["drc"]["ignored_violation_count"] = len(drc.get("violations", []))
        validation["drc"]["significant_violation_count"] = 0
        validation["drc"]["ignored_clearance_reason"] = _ignore_reason

    accepted = bool(validation.get("accepted", False))
    if not accepted:
        validation["accepted"] = False
        validation["rejected"] = True
        validation["rejection_stage"] = "leaf_routed_artifact_validation"
        validation["routed_board_path"] = str(routed_board)
        validation["pre_route_board_path"] = str(pre_route_board)
        validation["router"] = "freerouting"
        validation["internal_net_names"] = list(sorted(extraction.internal_net_names))
        validation["interface_port_names"] = [
            port.name for port in extraction.interface_ports
        ]
        validation["imported_copper_summary"] = {
            "trace_count": int(imported_copper.get("trace_count", 0)),
            "via_count": int(imported_copper.get("via_count", 0)),
            "total_length_mm": float(imported_copper.get("total_length_mm", 0.0)),
        }
        validation["freerouting_stats"] = copy.deepcopy(freerouting_stats)
        validation["rejection_message"] = "Leaf routed artifact rejected: " + ",".join(
            validation.get("rejection_reasons", [])
        )
        print("  Routed DRC rejected placement: " + validation["rejection_message"])
        route_timing["route_local_subcircuit_total_s"] = round(
            max(0.0, time.monotonic() - route_total_start), 3
        )
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": "routed_drc_rejection",
                "router": "freerouting",
                "traces": int(imported_copper.get("trace_count", 0)),
                "vias": int(imported_copper.get("via_count", 0)),
                "total_length_mm": float(imported_copper.get("total_length_mm", 0.0)),
                "round_board_illegal_pre_stamp": round_board_illegal_pre_stamp,
                "round_board_pre_route": round_board_pre_route,
                "round_board_routed": round_board_routed,
                "routed_internal_nets": [],
                "failed_internal_nets": list(sorted(extraction.internal_net_names)),
                "_trace_segments": [],
                "_via_objects": [],
                "validation": copy.deepcopy(validation),
                "freerouting_stats": copy.deepcopy(freerouting_stats),
                "render_diagnostics": copy.deepcopy(leaf_diagnostics),
                "routed_board_path": str(routed_board),
                "pre_route_board_path": str(pre_route_board),
                "round_preview_pre_route_front": round_preview_pre_route_front,
                "round_preview_pre_route_back": round_preview_pre_route_back,
                "round_preview_pre_route_copper": round_preview_pre_route_copper,
                "round_preview_routed_front": round_preview_routed_front,
                "round_preview_routed_back": round_preview_routed_back,
                "round_preview_routed_copper": round_preview_routed_copper,
                "failed": True,
            },
            route_timing,
        )

    route_timing["route_local_subcircuit_total_s"] = round(
        max(0.0, time.monotonic() - route_total_start), 3
    )
    return (
        {
            "enabled": True,
            "skipped": False,
            "reason": "",
            "router": "freerouting",
            "traces": int(imported_copper.get("trace_count", 0)),
            "vias": int(imported_copper.get("via_count", 0)),
            "total_length_mm": float(imported_copper.get("total_length_mm", 0.0)),
            "round_board_illegal_pre_stamp": round_board_illegal_pre_stamp,
            "round_board_pre_route": round_board_pre_route,
            "round_board_routed": round_board_routed,
            "routed_internal_nets": list(sorted(extraction.internal_net_names)),
            "failed_internal_nets": [],
            "_trace_segments": [
                copy.deepcopy(trace) for trace in imported_copper.get("traces", [])
            ],
            "_via_objects": [
                copy.deepcopy(via) for via in imported_copper.get("vias", [])
            ],
            "freerouting_stats": freerouting_stats,
            "validation": validation,
            "render_diagnostics": copy.deepcopy(leaf_diagnostics),
            "leaf_legality_repair": copy.deepcopy(legality_repair),
            "routed_board_path": str(routed_board),
            "pre_route_board_path": str(pre_route_board),
            "round_preview_pre_route_front": round_preview_pre_route_front,
            "round_preview_pre_route_back": round_preview_pre_route_back,
            "round_preview_pre_route_copper": round_preview_pre_route_copper,
            "round_preview_routed_front": round_preview_routed_front,
            "round_preview_routed_back": round_preview_routed_back,
            "round_preview_routed_copper": round_preview_routed_copper,
            "failed": False,
        },
        route_timing,
    )


def _solve_one_round(
    extraction: ExtractedSubcircuitBoard,
    cfg: dict[str, Any],
    seed: int,
    round_index: int,
    route: bool,
) -> SolveRoundResult:
    round_timing: dict[str, float] = {}
    round_total_start = time.monotonic()

    local_state = copy.deepcopy(extraction.local_state)

    placement_start = time.monotonic()
    solver = PlacementSolver(local_state, cfg, seed=seed)
    solved_components = solver.solve()
    round_timing["placement_solve_s"] = round(
        max(0.0, time.monotonic() - placement_start), 3
    )

    passive_ordering_start = time.monotonic()
    solved_components = _apply_leaf_passive_ordering(extraction, solved_components, cfg)
    round_timing["passive_ordering_s"] = round(
        max(0.0, time.monotonic() - passive_ordering_start), 3
    )

    ordering_legality_start = time.monotonic()
    repaired_components, ordering_legality = _repair_leaf_placement_legality(
        extraction,
        solved_components,
        cfg,
    )
    round_timing["post_ordering_legality_repair_s"] = round(
        max(0.0, time.monotonic() - ordering_legality_start), 3
    )

    if ordering_legality.get("resolved", False):
        solved_components = repaired_components

    placement_score_start = time.monotonic()
    placement = _score_local_components(local_state, solved_components, cfg)
    round_timing["placement_scoring_s"] = round(
        max(0.0, time.monotonic() - placement_score_start), 3
    )

    if route:
        try:
            routing, route_timing = _route_local_subcircuit(
                extraction,
                solved_components,
                cfg,
                round_index=round_index,
            )
            round_timing.update(route_timing)
        except Exception as exc:
            print(f"  WARNING: unexpected routing error in round {round_index}: {exc}")
            routing = {
                "enabled": True,
                "skipped": True,
                "reason": "routing_exception",
                "router": "freerouting",
                "traces": 0,
                "vias": 0,
                "total_length_mm": 0.0,
                "round_board_illegal_pre_stamp": "",
                "round_board_pre_route": "",
                "round_board_routed": "",
                "routed_internal_nets": [],
                "failed_internal_nets": list(sorted(extraction.internal_net_names)),
                "_trace_segments": [],
                "_via_objects": [],
                "validation": {
                    "accepted": False,
                    "rejected": True,
                    "rejection_stage": "routing_exception",
                    "rejection_reasons": [str(exc)],
                },
                "failed": True,
            }
            round_timing["route_local_subcircuit_total_s"] = 0.0
    else:
        routing = {
            "enabled": False,
            "skipped": True,
            "reason": "routing_disabled",
            "router": "disabled",
            "traces": 0,
            "vias": 0,
            "total_length_mm": 0.0,
            "round_board_illegal_pre_stamp": "",
            "round_board_pre_route": "",
            "round_board_routed": "",
            "routed_internal_nets": [],
            "failed_internal_nets": [],
            "_trace_segments": [],
            "_via_objects": [],
            "failed": False,
        }
        round_timing["route_local_subcircuit_total_s"] = 0.0

    round_timing["solve_one_round_total_s"] = round(
        max(0.0, time.monotonic() - round_total_start), 3
    )

    if route and routing.get("failed", False):
        return SolveRoundResult(
            round_index=round_index,
            seed=seed,
            score=float("-inf"),
            placement=placement,
            components=solved_components,
            routing=routing,
            routed=False,
            timing_breakdown=round_timing,
        )
    return SolveRoundResult(
        round_index=round_index,
        seed=seed,
        score=placement.total,
        placement=placement,
        components=solved_components,
        routing=routing,
        routed=bool(route and not routing.get("failed", False)),
        timing_breakdown=round_timing,
    )


def _solve_leaf_subcircuit(
    node: HierarchyNode,
    full_state: BoardState,
    cfg: dict[str, Any],
    rounds: int,
    base_seed: int,
    route: bool,
) -> SolvedLeafSubcircuit:
    leaf_total_start = time.monotonic()

    extraction_start = time.monotonic()
    extraction = extract_leaf_board_state(
        subcircuit=node.definition,
        full_state=full_state,
        margin_mm=float(cfg.get("subcircuit_margin_mm", 0.0)),
        include_power_externals=bool(
            cfg.get("subcircuit_include_power_externals", True)
        ),
        ignored_nets=set(cfg.get("subcircuit_ignored_nets", [])),
    )
    extraction_elapsed_s = round(max(0.0, time.monotonic() - extraction_start), 3)

    local_cfg_start = time.monotonic()
    local_cfg = _local_solver_config(cfg, extraction)
    local_cfg_elapsed_s = round(max(0.0, time.monotonic() - local_cfg_start), 3)
    rng = random.Random(base_seed)

    round_results: list[SolveRoundResult] = []
    best: SolveRoundResult | None = None

    effective_rounds = rounds
    if route:
        if bool(local_cfg.get("subcircuit_fast_smoke_mode", False)):
            effective_rounds = max(
                1,
                int(local_cfg.get("leaf_fast_smoke_route_rounds", rounds)),
            )
        else:
            effective_rounds = max(
                rounds,
                int(local_cfg.get("leaf_min_route_rounds", 8)),
            )

    fast_smoke_mode = bool(local_cfg.get("subcircuit_fast_smoke_mode", False))

    failure_reasons: list[str] = []
    failure_rows: list[dict[str, Any]] = []
    accepted_round_count = 0
    failed_round_count = 0

    for round_index in range(effective_rounds):
        seed = rng.randint(0, 2**31 - 1)
        round_cfg = dict(local_cfg)
        if route and not fast_smoke_mode:
            if round_index % 3 == 1:
                round_cfg["randomize_group_layout"] = True
                round_cfg["orderedness"] = max(
                    0.15,
                    float(round_cfg.get("orderedness", 0.25)) - 0.10,
                )
            elif round_index % 3 == 2:
                round_cfg["randomize_group_layout"] = True
                round_cfg["scatter_mode"] = "random"
                round_cfg["orderedness"] = max(
                    0.10,
                    float(round_cfg.get("orderedness", 0.25)) - 0.15,
                )
        result = _solve_one_round(extraction, round_cfg, seed, round_index, route)
        result.timing_breakdown["leaf_extraction_s"] = extraction_elapsed_s
        result.timing_breakdown["local_solver_config_s"] = local_cfg_elapsed_s
        round_results.append(result)

        routing = result.routing or {}
        validation = routing.get("validation", {}) or {}
        accepted = not (
            route
            and (
                routing.get("failed", False)
                or (
                    not routing.get("routed_board_path")
                    and routing.get("reason") != "no_internal_nets"
                )
            )
        )
        if accepted:
            accepted_round_count += 1
        else:
            failed_round_count += 1
            reason = (
                validation.get("rejection_stage")
                or validation.get("rejection_message")
                or routing.get("reason")
                or "unknown_leaf_failure"
            )
            failure_reasons.append(str(reason))
            failure_rows.append(
                {
                    "round_index": result.round_index,
                    "seed": result.seed,
                    "reason": str(reason),
                    "router": str(routing.get("router", "") or ""),
                    "failed": bool(routing.get("failed", False)),
                    "failed_internal_nets": list(
                        routing.get("failed_internal_nets", []) or []
                    ),
                    "timing_breakdown": dict(result.timing_breakdown),
                }
            )
            continue

        if best is None or result.score > best.score:
            best = result

    if best is None:
        unique_reasons = sorted(set(failure_reasons))
        raise RuntimeError(
            "No accepted routed leaf artifact produced for "
            f"{node.definition.id.instance_path} after {effective_rounds} round(s): "
            + ",".join(unique_reasons or ["unknown_leaf_failure"])
        )

    size_reduction_start = time.monotonic()
    reduced_extraction, reduced_best, size_reduction = _attempt_leaf_size_reduction(
        extraction,
        best,
        cfg,
    )
    size_reduction_elapsed_s = round(
        max(0.0, time.monotonic() - size_reduction_start), 3
    )
    leaf_total_elapsed_s = round(max(0.0, time.monotonic() - leaf_total_start), 3)

    for round_result in round_results:
        round_result.timing_breakdown["leaf_size_reduction_s"] = (
            size_reduction_elapsed_s
        )
        round_result.timing_breakdown["leaf_total_s"] = leaf_total_elapsed_s

    scheduling_metadata = {
        "sheet_name": node.definition.id.sheet_name,
        "instance_path": node.definition.id.instance_path,
        "internal_net_count": len(extraction.internal_net_names),
        "external_net_count": len(extraction.external_net_names),
        "historically_trivial_candidate": len(extraction.internal_net_names) == 0,
        "trace_count": len(extraction.internal_traces),
        "via_count": len(extraction.internal_vias),
        "effective_rounds": effective_rounds,
        "fast_smoke_mode": fast_smoke_mode,
        "best_round_index": reduced_best.round_index,
        "best_score": reduced_best.score,
        "leaf_total_s": leaf_total_elapsed_s,
        "route_total_s": float(
            reduced_best.timing_breakdown.get("route_local_subcircuit_total_s", 0.0)
            or 0.0
        ),
        "freerouting_s": float(
            reduced_best.timing_breakdown.get("freerouting_s", 0.0) or 0.0
        ),
        "accepted_round_count": accepted_round_count,
        "failed_round_count": failed_round_count,
    }

    failure_summary = {
        "had_failures": bool(failure_rows),
        "failure_count": len(failure_rows),
        "accepted_round_count": accepted_round_count,
        "failed_round_count": failed_round_count,
        "unique_reasons": sorted(set(failure_reasons)),
        "failures": failure_rows,
    }

    return SolvedLeafSubcircuit(
        node=node,
        extraction=reduced_extraction,
        best_round=reduced_best,
        all_rounds=round_results,
        size_reduction=size_reduction,
        scheduling_metadata=scheduling_metadata,
        failure_summary=failure_summary,
    )


def _solved_local_outline(extraction: ExtractedSubcircuitBoard) -> dict[str, float]:
    tl, br = extraction.local_state.board_outline
    return {
        "top_left_x": tl.x,
        "top_left_y": tl.y,
        "bottom_right_x": br.x,
        "bottom_right_y": br.y,
        "width_mm": extraction.local_state.board_width,
        "height_mm": extraction.local_state.board_height,
    }


def _solved_local_translation(extraction: ExtractedSubcircuitBoard) -> dict[str, float]:
    return {
        "x": extraction.translation.x,
        "y": extraction.translation.y,
    }


def _persist_solution(
    solved: SolvedLeafSubcircuit,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    persist_start = time.monotonic()
    solved_layout = solved.best_round_to_layout()
    solved_geometry = serialize_components(solved.best_round.components)
    anchor_validation = build_anchor_validation(
        solved_layout.ports,
        solved_layout.interface_anchors,
    )
    routing_validation = dict(solved.best_round.routing.get("validation", {}))
    canonical_layout = solved.canonical_layout_artifact(cfg)
    canonical_layout["validation"] = routing_validation
    canonical_layout["scheduling_metadata"] = dict(solved.scheduling_metadata or {})
    canonical_layout["failure_summary"] = dict(solved.failure_summary or {})
    size_reduction = dict(solved.size_reduction or {})
    reduced_outline = dict(
        size_reduction.get("reduced_outline", _solved_local_outline(solved.extraction))
    )
    original_outline = dict(size_reduction.get("original_outline", reduced_outline))
    extraction = build_leaf_extraction(
        subcircuit=solved.node.definition,
        project_dir=solved.extraction.subcircuit.schematic_path
        and Path(solved.extraction.subcircuit.schematic_path).parent
        or ".",
        internal_nets=solved.extraction.internal_net_names,
        external_nets=solved.extraction.external_net_names,
        local_board_outline=reduced_outline,
        local_translation=_solved_local_translation(solved.extraction),
        internal_trace_count=len(solved.extraction.internal_traces),
        internal_via_count=len(solved.extraction.internal_vias),
        notes=[
            "solved by solve_subcircuits.py",
            f"best_round={solved.best_round.round_index}",
            f"best_seed={solved.best_round.seed}",
            f"best_score={solved.best_round.score:.3f}",
            f"rounds={len(solved.all_rounds)}",
            f"solved_component_count={len(solved.best_round.components)}",
            f"canonical_layout_schema={canonical_layout['schema_version']}",
            f"router={solved.best_round.routing.get('router', 'unknown')}",
            f"accepted={routing_validation.get('accepted', False)}",
            f"failed_round_count={int(solved.failure_summary.get('failed_round_count', 0) or 0)}",
            f"failure_reasons={','.join(solved.failure_summary.get('unique_reasons', []) or ['none'])}",
            f"historically_trivial_candidate={bool(solved.scheduling_metadata.get('historically_trivial_candidate', False))}",
            f"leaf_total_s={float(solved.scheduling_metadata.get('leaf_total_s', 0.0) or 0.0):.3f}",
            f"route_total_s={float(solved.scheduling_metadata.get('route_total_s', 0.0) or 0.0):.3f}",
            f"freerouting_s={float(solved.scheduling_metadata.get('freerouting_s', 0.0) or 0.0):.3f}",
            f"size_reduction_attempted={size_reduction.get('attempted', False)}",
            f"size_reduction_passes={size_reduction.get('passes', 0)}",
            f"original_outline_width_mm={original_outline.get('width_mm', solved.extraction.local_state.board_width):.3f}",
            f"original_outline_height_mm={original_outline.get('height_mm', solved.extraction.local_state.board_height):.3f}",
            f"reduced_outline_width_mm={reduced_outline.get('width_mm', solved.extraction.local_state.board_width):.3f}",
            f"reduced_outline_height_mm={reduced_outline.get('height_mm', solved.extraction.local_state.board_height):.3f}",
        ]
        + list(solved.extraction.notes),
    )

    metadata = build_artifact_metadata(
        extraction=extraction,
        config=cfg,
        solver_version="subcircuits-m4-freerouting",
    )

    routed_board_path = solved.best_round.routing.get("routed_board_path")
    if routed_board_path:
        metadata.artifact_paths["mini_pcb"] = routed_board_path
    elif solved.best_round.routing.get("reason") == "no_internal_nets":
        # Leaves with no internal nets have no routed board — use the layout.kicad_pcb instead
        layout_pcb = Path(metadata.artifact_paths.get("layout_pcb", ""))
        if layout_pcb.exists():
            metadata.artifact_paths["mini_pcb"] = str(layout_pcb)
    else:
        raise RuntimeError(
            f"Accepted leaf artifact for {solved.instance_path} is missing routed_board_path"
        )

    canonical_layout["original_outline"] = original_outline
    canonical_layout["reduced_outline"] = reduced_outline
    canonical_layout["size_reduction"] = size_reduction

    solved_layout_json = save_solved_layout_artifact(canonical_layout)
    metadata.artifact_paths["solved_layout_json"] = solved_layout_json

    save_artifact_metadata(metadata)
    metadata.notes = list(metadata.notes) + [
        f"size_reduction_attempted={size_reduction.get('attempted', False)}",
        f"size_reduction_passes={size_reduction.get('passes', 0)}",
        f"outline_reduction_width_mm={size_reduction.get('outline_reduction_mm', {}).get('width_mm', 0.0):.3f}",
        f"outline_reduction_height_mm={size_reduction.get('outline_reduction_mm', {}).get('height_mm', 0.0):.3f}",
        f"outline_reduction_area_mm2={size_reduction.get('outline_reduction_mm', {}).get('area_mm2', 0.0):.3f}",
    ]
    save_artifact_metadata(metadata)

    persist_elapsed_s = round(max(0.0, time.monotonic() - persist_start), 3)
    solved.best_round.timing_breakdown["persist_solution_s"] = persist_elapsed_s

    save_debug_payload(
        extraction=extraction,
        metadata=metadata,
        extra={
            "solve_summary": solved.to_dict(),
            "best_round": solved.best_round.to_dict(),
            "all_rounds": [
                round_result.to_dict() for round_result in solved.all_rounds
            ],
            "leaf_board_state": extraction_debug_dict(solved.extraction),
            "solved_local_placement": {
                "component_count": len(solved.best_round.components),
                "components": solved_geometry,
            },
            "best_round_routing": {
                key: value
                for key, value in solved.best_round.routing.items()
                if not key.startswith("_")
            },
            "leaf_acceptance": routing_validation,
            "leaf_render_diagnostics": solved.best_round.routing.get(
                "render_diagnostics", {}
            ),
            "interface_anchor_validation": anchor_validation,
            "size_reduction": size_reduction,
            "original_outline": original_outline,
            "reduced_outline": reduced_outline,
            "size_reduction_validation": size_reduction.get("validation", {}),
            "canonical_solved_layout": canonical_layout,
            "canonical_solved_layout_path": str(solved_layout_json),
            "timing_breakdown": dict(solved.best_round.timing_breakdown),
            "scheduling_metadata": dict(solved.scheduling_metadata or {}),
            "failure_summary": dict(solved.failure_summary or {}),
        },
    )
    return metadata.to_dict()


def _print_human_summary(
    results: list[SolvedLeafSubcircuit], persisted: list[dict[str, Any]]
) -> None:
    print("=== Leaf Subcircuit Solve ===")
    print(f"leaf_subcircuits : {len(results)}")
    print()

    for solved, metadata in zip(results, persisted):
        best = solved.best_round
        print(f"- {solved.sheet_name} [{solved.instance_path}]")
        print(f"  best_score    : {best.score:.2f}")
        print(f"  best_round    : {best.round_index}")
        print(f"  best_seed     : {best.seed}")
        print(
            f"  local_size_mm : "
            f"{solved.extraction.local_state.board_width:.1f} x "
            f"{solved.extraction.local_state.board_height:.1f}"
        )
        print(f"  internal_nets : {len(solved.extraction.internal_net_names)}")
        print(f"  external_nets : {len(solved.extraction.external_net_names)}")
        print(f"  traces        : {len(solved.extraction.internal_traces)}")
        print(f"  vias          : {len(solved.extraction.internal_vias)}")
        print(f"  routed        : {best.routed}")
        print(f"  route_traces  : {best.routing.get('traces', 0)}")
        print(f"  route_vias    : {best.routing.get('vias', 0)}")
        print(f"  metadata_json : {metadata['artifact_paths']['metadata_json']}")
        print(f"  debug_json    : {metadata['artifact_paths']['debug_json']}")
        print()


def _json_summary(
    results: list[SolvedLeafSubcircuit], persisted: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "leaf_subcircuits": len(results),
        "results": [
            {
                "solved": solved.to_dict(),
                "artifact_metadata": metadata,
            }
            for solved, metadata in zip(results, persisted)
        ],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve leaf subcircuits with local placement search"
    )
    parser.add_argument(
        "schematic",
        help="Top-level .kicad_sch file",
    )
    parser.add_argument(
        "--pcb",
        help="Optional .kicad_pcb file (defaults to schematic stem with .kicad_pcb)",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file to merge on top of default/project config",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=6,
        help="Placement-search rounds per leaf subcircuit (default: 6)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed for leaf solve search (default: 0)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Restrict solving to a specific leaf by sheet name, sheet file, or instance path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary instead of human-readable text",
    )
    parser.add_argument(
        "--route",
        action="store_true",
        help="Run optional local routing for internal leaf nets after placement",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers for solving independent leaf subcircuits at the same hierarchy level (0 = auto-select)",
    )
    parser.add_argument(
        "--leaf-order",
        action="append",
        default=[],
        help="Preferred leaf solve order by sheet name, sheet file, or instance path; may be repeated",
    )
    parser.add_argument(
        "--fast-smoke",
        action="store_true",
        help="Reduce nonessential render diagnostics for faster smoke-test verification while preserving canonical board artifacts",
    )
    return parser.parse_args(argv)


def _solve_leaf_worker(
    args: tuple[HierarchyNode, BoardState, dict[str, Any], int, int, bool],
) -> tuple[str, SolvedLeafSubcircuit | None, dict[str, Any] | None]:
    node, full_state, cfg, rounds, base_seed, route = args
    try:
        solved = _solve_leaf_subcircuit(
            node=node,
            full_state=full_state,
            cfg=cfg,
            rounds=rounds,
            base_seed=base_seed,
            route=route,
        )
        return (node.id.instance_path, solved, None)
    except Exception as exc:
        return (
            node.id.instance_path,
            None,
            {
                "sheet_name": node.id.sheet_name,
                "instance_path": node.id.instance_path,
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    top_schematic = Path(args.schematic).resolve()

    if not top_schematic.exists():
        print(f"error: schematic not found: {top_schematic}", file=sys.stderr)
        return 2
    if top_schematic.suffix != ".kicad_sch":
        print(
            f"error: expected a .kicad_sch file, got: {top_schematic}",
            file=sys.stderr,
        )
        return 2

    pcb_path = (
        Path(args.pcb).resolve() if args.pcb else _default_pcb_path(top_schematic)
    )
    if not pcb_path.exists():
        print(f"error: pcb not found: {pcb_path}", file=sys.stderr)
        return 2

    try:
        cfg = _load_config(args.config, project_dir=top_schematic.parent)
        cfg["pcb_path"] = str(pcb_path)
        if args.fast_smoke:
            cfg["subcircuit_fast_smoke_mode"] = True
            cfg["subcircuit_render_pre_route_board_views"] = False
            cfg["subcircuit_render_routed_board_views"] = False
            cfg["subcircuit_render_pre_route_drc_overlay"] = False
            cfg["subcircuit_render_routed_drc_overlay"] = False
            cfg["subcircuit_write_pre_route_drc_json"] = False
            cfg["subcircuit_write_routed_drc_json"] = False
            cfg["subcircuit_write_pre_route_drc_report"] = False
            cfg["subcircuit_write_routed_drc_report"] = True
            cfg["subcircuit_build_comparison_contact_sheet"] = False
            cfg["leaf_fast_smoke_route_rounds"] = 1
        graph = parse_hierarchy(
            project_dir=top_schematic.parent,
            top_schematic=top_schematic,
        )
        board_state = _load_board_state(pcb_path, cfg)
        leaves = _leaf_nodes(graph, args.only, args.leaf_order)
        if not leaves:
            print("error: no matching leaf subcircuits found", file=sys.stderr)
            return 1

        solved_results: list[SolvedLeafSubcircuit] = []
        persisted: list[dict[str, Any]] = []

        requested_workers = int(args.workers or 0)
        available_cpus = max(1, int(os.cpu_count() or 1))
        if requested_workers > 0:
            worker_count = max(1, requested_workers)
        else:
            worker_count = min(len(leaves), max(1, available_cpus - 1))
        rounds = max(1, args.rounds)

        if worker_count == 1 or len(leaves) <= 1:
            for index, node in enumerate(leaves):
                solved = _solve_leaf_subcircuit(
                    node=node,
                    full_state=board_state,
                    cfg=cfg,
                    rounds=rounds,
                    base_seed=args.seed + index * 1009,
                    route=args.route,
                )
                solved_results.append(solved)
        else:
            ctx = mp.get_context("spawn")
            worker_args = [
                (
                    node,
                    board_state,
                    cfg,
                    rounds,
                    args.seed + index * 1009,
                    args.route,
                )
                for index, node in enumerate(leaves)
            ]
            solved_by_path: dict[str, SolvedLeafSubcircuit] = {}
            failed_by_path: dict[str, dict[str, Any]] = {}
            infrastructure_failure: Exception | None = None
            try:
                with ProcessPoolExecutor(
                    max_workers=min(worker_count, len(worker_args)),
                    mp_context=ctx,
                ) as pool:
                    future_map = {
                        pool.submit(_solve_leaf_worker, item): item[0].id.instance_path
                        for item in worker_args
                    }
                    for future in as_completed(future_map):
                        instance_path = future_map[future]
                        try:
                            solved_path, solved, failure = future.result()
                        except Exception as exc:
                            infrastructure_failure = exc
                            print(
                                "warning: parallel leaf worker infrastructure failure: "
                                f"{instance_path}: {exc}",
                                file=sys.stderr,
                            )
                            continue
                        if failure is not None:
                            failed_by_path[solved_path] = dict(failure)
                            print(
                                "warning: parallel leaf solve failed for "
                                f"{solved_path}: {failure.get('error', 'unknown_error')}",
                                file=sys.stderr,
                            )
                            continue
                        if solved is not None:
                            solved_by_path[solved.instance_path] = solved
            except Exception as exc:
                infrastructure_failure = exc
                print(
                    "warning: parallel leaf solve infrastructure failed; preserving completed results where possible: "
                    f"{exc}",
                    file=sys.stderr,
                )

            if infrastructure_failure is not None:
                for index, node in enumerate(leaves):
                    if node.id.instance_path in solved_by_path:
                        continue
                    try:
                        solved = _solve_leaf_subcircuit(
                            node=node,
                            full_state=board_state,
                            cfg=cfg,
                            rounds=rounds,
                            base_seed=args.seed + index * 1009,
                            route=args.route,
                        )
                        solved_by_path[solved.instance_path] = solved
                    except Exception as exc:
                        failed_by_path[node.id.instance_path] = {
                            "sheet_name": node.id.sheet_name,
                            "instance_path": node.id.instance_path,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "recovery_mode": "serial_after_parallel_infrastructure_failure",
                        }

            if failed_by_path:
                failure_lines = [
                    f"{item.get('instance_path', path)}:{item.get('error', 'unknown_error')}"
                    for path, item in sorted(failed_by_path.items())
                ]
                raise RuntimeError(
                    "Leaf solve failures encountered after preserving successful parallel results: "
                    + "; ".join(failure_lines)
                )

            solved_results = [
                solved_by_path[node.id.instance_path]
                for node in leaves
                if node.id.instance_path in solved_by_path
            ]

        for solved in solved_results:
            persisted.append(_persist_solution(solved, cfg))

    except Exception as exc:
        print(f"error: failed to solve subcircuits: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = json.dumps(
            _json_summary(solved_results, persisted), indent=2, default=str
        )
        print("===SOLVE_SUBCIRCUITS_JSON_START===")
        print(payload)
        print("===SOLVE_SUBCIRCUITS_JSON_END===")
        return 0

    _print_human_summary(solved_results, persisted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
