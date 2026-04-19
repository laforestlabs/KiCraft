from __future__ import annotations

import copy
from typing import Any

from kicraft.autoplacer.brain.leaf_geometry import (
    build_reduced_leaf_extraction,
    leaf_size_reduction_candidates,
    tight_leaf_geometry_bounds,
)
from kicraft.autoplacer.brain.leaf_routing import route_local_subcircuit
from kicraft.autoplacer.brain.placement import PlacementSolver
from kicraft.autoplacer.brain.subcircuit_extractor import ExtractedSubcircuitBoard
from kicraft.autoplacer.brain.types import Point, SolveRoundResult


def local_solver_config(
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

    board_area = extraction.local_state.board_width * extraction.local_state.board_height
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


def attempt_leaf_size_reduction(
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

    connector_pad_margin = max(
        0.0,
        float(cfg.get("connector_pad_margin_mm", 1.0)),
    )
    geometry_bounds = tight_leaf_geometry_bounds(
        extraction,
        best_round.components,
        best_round.routing,
        connector_pad_margin_mm=connector_pad_margin,
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

        candidates = leaf_size_reduction_candidates(
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
            candidate_extraction = build_reduced_leaf_extraction(
                current_extraction,
                current_round.components,
                current_round.routing,
                candidate_outline,
            )
            candidate_cfg = local_solver_config(cfg, candidate_extraction)
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
                        rerouted, reroute_timing = route_local_subcircuit(
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
