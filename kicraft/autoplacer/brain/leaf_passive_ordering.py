"""Leaf subcircuit passive topology and ordering helpers.

Pure-algorithm utilities for analyzing component-net connectivity,
building passive topology groups around anchor ICs/connectors, and
applying ordered passive placement.  Extracted from the CLI
``solve_subcircuits`` module so they can be unit-tested and reused
independently.
"""

from __future__ import annotations

import copy
from typing import Any

from kicraft.autoplacer.brain.placement import _update_pad_positions
from kicraft.autoplacer.brain.subcircuit_extractor import ExtractedSubcircuitBoard
from kicraft.autoplacer.brain.types import Component, Point


def component_net_degree_map(extraction: ExtractedSubcircuitBoard) -> dict[str, int]:
    degree_by_ref: dict[str, int] = {}
    for net in extraction.local_state.nets.values():
        refs = {ref for ref, _ in net.pad_refs}
        if len(refs) < 2:
            continue
        weight = max(1, len(refs) - 1)
        for ref in refs:
            degree_by_ref[ref] = degree_by_ref.get(ref, 0) + weight
    return degree_by_ref


def component_primary_net_map(
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


def component_net_map(
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


def component_adjacency_map(
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


def build_leaf_passive_topology_groups(
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

    degree_by_ref = component_net_degree_map(extraction)
    primary_net_by_ref = component_primary_net_map(extraction)
    nets_by_ref = component_net_map(extraction)
    adjacency = component_adjacency_map(extraction)

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


def apply_leaf_passive_ordering(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
) -> dict[str, Component]:
    if not bool(cfg.get("leaf_passive_ordering_enabled", False)):
        return copy.deepcopy(solved_components)

    ordered = copy.deepcopy(solved_components)
    topology_groups = build_leaf_passive_topology_groups(extraction, ordered)
    if not topology_groups:
        return ordered

    degree_by_ref = component_net_degree_map(extraction)
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

