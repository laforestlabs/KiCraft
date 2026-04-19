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

from kicraft.autoplacer.brain.hierarchy_parser import (
    HierarchyGraph,
    HierarchyNode,
    parse_hierarchy,
)
from kicraft.autoplacer.brain.leaf_acceptance import (
    acceptance_config_from_dict,
    evaluate_leaf_acceptance,
)
from kicraft.autoplacer.brain.placement import (
    PlacementSolver,
)
from kicraft.autoplacer.brain.leaf_routing import route_local_subcircuit
from kicraft.autoplacer.brain.leaf_size_reduction import (
    attempt_leaf_size_reduction,
    local_solver_config,
)
from kicraft.autoplacer.brain.leaf_geometry import (
    build_reduced_leaf_extraction,
    copy_components_with_translation,
    copy_traces_with_translation,
    copy_vias_with_translation,
    leaf_size_reduction_candidates,
    repair_leaf_placement_legality,
    score_local_components,
    tight_leaf_geometry_bounds,
)
from kicraft.autoplacer.brain.leaf_passive_ordering import (
    apply_leaf_passive_ordering,
    build_leaf_passive_topology_groups,
    component_adjacency_map,
    component_net_degree_map,
    component_net_map,
    component_primary_net_map,
)
from kicraft.autoplacer.brain.subcircuit_extractor import (
    ExtractedSubcircuitBoard,
    extract_leaf_board_state,
    extraction_debug_dict,
    summarize_extraction,
)
from kicraft.autoplacer.brain.subcircuit_artifacts import (
    build_anchor_validation,
    build_artifact_metadata,
    build_leaf_extraction,
    build_solved_layout_artifact,
    save_artifact_metadata,
    save_debug_payload,
    save_solved_layout_artifact,
    serialize_components,
)
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    PlacementScore,
    Point,
    SolveRoundResult,
    SubCircuitLayout,
)
from kicraft.autoplacer.config import DEFAULT_CONFIG, load_project_config
from kicraft.autoplacer.hardware.adapter import KiCadAdapter


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
        from kicraft.autoplacer.brain.subcircuit_solver import infer_interface_anchors

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
        from kicraft.autoplacer.config import discover_project_config

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
    return local_solver_config(base_cfg, extraction)


def _copy_components_with_translation(
    components: dict[str, Component],
    delta: Point,
) -> dict[str, Component]:
    return copy_components_with_translation(components, delta)


def _copy_traces_with_translation(traces: list[Any], delta: Point) -> list[Any]:
    return copy_traces_with_translation(traces, delta)


def _copy_vias_with_translation(vias: list[Any], delta: Point) -> list[Any]:
    return copy_vias_with_translation(vias, delta)


def _component_net_degree_map(extraction: ExtractedSubcircuitBoard) -> dict[str, int]:
    return component_net_degree_map(extraction)


def _component_primary_net_map(
    extraction: ExtractedSubcircuitBoard,
) -> dict[str, tuple[str, int]]:
    return component_primary_net_map(extraction)


def _component_net_map(
    extraction: ExtractedSubcircuitBoard,
) -> dict[str, set[str]]:
    return component_net_map(extraction)


def _component_adjacency_map(
    extraction: ExtractedSubcircuitBoard,
) -> dict[str, dict[str, int]]:
    return component_adjacency_map(extraction)


def _build_leaf_passive_topology_groups(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
) -> list[dict[str, Any]]:
    return build_leaf_passive_topology_groups(extraction, solved_components)


def _apply_leaf_passive_ordering(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
) -> dict[str, Component]:
    return apply_leaf_passive_ordering(extraction, solved_components, cfg)


def _tight_leaf_geometry_bounds(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    routing: dict[str, Any],
    *,
    connector_pad_margin_mm: float = 0.0,
) -> dict[str, float]:
    return tight_leaf_geometry_bounds(
        extraction, solved_components, routing,
        connector_pad_margin_mm=connector_pad_margin_mm,
    )


def _build_reduced_leaf_extraction(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    routing: dict[str, Any],
    outline: tuple[Point, Point],
) -> ExtractedSubcircuitBoard:
    return build_reduced_leaf_extraction(
        extraction, solved_components, routing, outline,
    )


def _leaf_size_reduction_candidates(
    current_width: float,
    current_height: float,
    min_width: float,
    min_height: float,
) -> list[dict[str, Any]]:
    return leaf_size_reduction_candidates(
        current_width, current_height, min_width, min_height,
    )


def _attempt_leaf_size_reduction(
    extraction: ExtractedSubcircuitBoard,
    best_round: SolveRoundResult,
    cfg: dict[str, Any],
) -> tuple[ExtractedSubcircuitBoard, SolveRoundResult, dict[str, Any]]:
    return attempt_leaf_size_reduction(extraction, best_round, cfg)


def _score_local_components(
    local_state: BoardState,
    components: dict[str, Component],
    cfg: dict[str, Any],
) -> PlacementScore:
    return score_local_components(local_state, components, cfg)


def _repair_leaf_placement_legality(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
) -> tuple[dict[str, Component], dict[str, Any]]:
    return repair_leaf_placement_legality(extraction, solved_components, cfg)


def _route_local_subcircuit(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
    *,
    generate_diagnostics: bool = True,
    round_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    return route_local_subcircuit(
        extraction,
        solved_components,
        cfg,
        generate_diagnostics=generate_diagnostics,
        round_index=round_index,
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
    acceptance_cfg = acceptance_config_from_dict(cfg)

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

        # -- Structured round acceptance via leaf_acceptance gates --
        if not route:
            # No routing requested -- placement-only round is always accepted
            accepted = True
            round_acceptance = None
        elif routing.get("reason") == "no_internal_nets":
            # Trivial pass -- nothing to route for this leaf
            accepted = True
            round_acceptance = None
        elif routing.get("failed", False):
            # Clear routing infrastructure failure -- reject without gate eval
            accepted = False
            round_acceptance = None
        else:
            # Normal routed result -- evaluate through structured acceptance
            # gates.  Anchor validation is deferred to persist time; pass
            # empty dict here so anchor gates are skipped for per-round eval.
            round_acceptance = evaluate_leaf_acceptance(
                validation=validation,
                anchor_validation={},
                config=acceptance_cfg,
            )
            accepted = round_acceptance.accepted

        # Stash the structured acceptance result on the routing dict for
        # downstream persistence and debugging.
        result.routing["_round_acceptance"] = (
            {
                "accepted": round_acceptance.accepted,
                "rejection_reasons": list(round_acceptance.rejection_reasons),
                "gate_results": dict(round_acceptance.gate_results),
                "drc_summary": dict(round_acceptance.drc_summary),
                "notes": list(round_acceptance.notes),
            }
            if round_acceptance is not None
            else {
                "accepted": accepted,
                "rejection_reasons": [],
                "gate_results": {},
                "drc_summary": {},
                "notes": [],
            }
        )

        if accepted:
            accepted_round_count += 1
        else:
            failed_round_count += 1
            if round_acceptance is not None:
                reason = (
                    ",".join(round_acceptance.rejection_reasons)
                    or "unknown_gate_failure"
                )
            else:
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
                    "acceptance_gate_results": (
                        dict(round_acceptance.gate_results)
                        if round_acceptance
                        else {}
                    ),
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
    # Warn about incomplete anchors
    if not anchor_validation.get("all_required_anchored", True):
        missing = anchor_validation.get("missing_required", [])
        print(
            f"  WARNING: leaf {solved.node.definition.id.sheet_name} has "
            f"{len(missing)} unanchored required port(s): {missing}"
        )
    routing_validation = dict(solved.best_round.routing.get("validation", {}))

    # -- Full acceptance evaluation with anchor validation --
    persist_acceptance_cfg = acceptance_config_from_dict(cfg)
    full_acceptance = evaluate_leaf_acceptance(
        validation=routing_validation,
        anchor_validation=anchor_validation,
        config=persist_acceptance_cfg,
    )
    routing_validation["leaf_acceptance_result"] = {
        "accepted": full_acceptance.accepted,
        "rejection_reasons": list(full_acceptance.rejection_reasons),
        "gate_results": dict(full_acceptance.gate_results),
        "drc_summary": dict(full_acceptance.drc_summary),
        "anchor_summary": dict(full_acceptance.anchor_summary),
        "notes": list(full_acceptance.notes),
    }

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
            "leaf_acceptance_structured": routing_validation.get("leaf_acceptance_result", {}),
            "round_acceptance": solved.best_round.routing.get("_round_acceptance", {}),
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
