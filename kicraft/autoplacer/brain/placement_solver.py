"""PlacementSolver -- force-directed placement with edge-first constraints.

Extracted from placement.py for modularity.  Import from
``placement`` (the re-export hub) for backward compatibility,
or directly from this module in new code.
"""

from __future__ import annotations

import copy
import math
import random

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from .graph import (
    AdjacencyGraph,
    build_connectivity_graph,
    count_crossings,
    find_communities,
)
from .placement_scorer import PlacementScorer
from .placement_utils import (
    _bbox_overlap_amount,
    _bbox_overlap_xy,
    _effective_bbox,
    _pad_half_extents,
    _swap_pad_positions,
    _update_pad_positions,
)
from .types import (
    BoardState,
    Component,
    FunctionalGroup,
    Layer,
    Net,
    PlacedGroup,
    Point,
)

class PlacementSolver:
    """Force-directed placement with edge-first constraints and scoring feedback.

    The solver iterates locally — all geometric computation in Python.
    Placement quality is scored each iteration; the solver converges
    when score improvement plateaus.
    """

    def __init__(self, state: BoardState, config: dict = None, seed: int = 0):
        self.state = state
        self.cfg = config or {}
        self.seed = seed
        self.rng = random.Random(seed)
        self.k_attract = max(0.001, min(1.0, self.cfg.get("force_attract_k", 0.08)))
        self.k_repel = max(1.0, min(5000.0, self.cfg.get("force_repel_k", 40.0)))
        self.cooling = max(0.5, min(0.999, self.cfg.get("cooling_factor", 0.97)))
        self.edge_margin = max(0.5, min(30.0, self.cfg.get("edge_margin_mm", 2.0)))
        self.grid_snap = self.cfg.get("placement_grid_mm", 0.5)
        self.max_iterations = max(
            10, min(2000, int(self.cfg.get("max_placement_iterations", 300)))
        )
        self.convergence_threshold = self.cfg.get(
            "placement_convergence_threshold", 0.5
        )
        self.score_every_n = self.cfg.get("placement_score_every_n", 1)
        self.intra_cluster_iters = self.cfg.get("intra_cluster_iters", 80)
        # placement_clearance_mm is the min gap between component bboxes.
        # Falls back to clearance_mm for backwards compatibility, then 2.5mm.
        self.clearance = self.cfg.get(
            "placement_clearance_mm", self.cfg.get("clearance_mm", 2.5)
        )
        self._seen_force_states: set[int] = set()
        # Aligned pairs: list of (ref_a, ref_b, axis) tuples.
        # Populated by _align_large_pairs(); used by _force_step().
        self._aligned_pairs: list[tuple[str, str, str]] = []

    # ------------------------------------------------------------------
    # Intra-group placement (Phase 2 of hierarchical pipeline)
    # ------------------------------------------------------------------

    def solve_group(
        self, group: FunctionalGroup, comps: dict[str, Component], nets: dict[str, Net]
    ) -> PlacedGroup:
        """Place components within a single functional group.

        Creates a tight virtual bounding rectangle, runs a mini force-directed
        simulation (reusing existing force logic), then returns a PlacedGroup
        with positions relative to (0, 0).

        Args:
            group: The functional group to place.
            comps: Full component dict (only group members are moved).
            nets: Full net dict (used for connectivity).

        Returns:
            PlacedGroup with component positions/rotations relative to group
            origin (0, 0) and bounding box dimensions.
        """
        member_comps = {ref: comps[ref] for ref in group.member_refs if ref in comps}
        if not member_comps:
            return PlacedGroup(
                group=group,
                bounding_box=(0, 0),
                component_positions={},
                component_layers={},
            )

        # Skip rigid group placement for groups of only large THT components.
        # These will be aligned by _align_large_pairs() during post-processing,
        # which is more flexible than a rigid group block.
        tht_area_threshold = self.cfg.get("tht_backside_min_area_mm2", 50.0)
        if all(
            c.area > tht_area_threshold and c.is_through_hole
            for c in member_comps.values()
        ):
            # Try both horizontal and vertical arrangements, pick the more compact one
            def _tht_strip(horiz: bool):
                positions = {}
                layers = {}
                tw = 0.0
                th = 0.0
                cursor = 0.0
                for ref, c in member_comps.items():
                    if horiz:
                        positions[ref] = (
                            cursor + c.width_mm / 2,
                            c.height_mm / 2,
                            c.rotation,
                        )
                        layers[ref] = c.layer
                        cursor += c.width_mm + 2.0
                        th = max(th, c.height_mm)
                    else:
                        positions[ref] = (
                            c.width_mm / 2,
                            cursor + c.height_mm / 2,
                            c.rotation,
                        )
                        layers[ref] = c.layer
                        cursor += c.height_mm + 2.0
                        tw = max(tw, c.width_mm)
                if horiz:
                    tw = cursor - 2.0 if cursor > 2.0 else 0.0
                else:
                    th = cursor - 2.0 if cursor > 2.0 else 0.0
                return positions, layers, tw, th

            h_pos, h_lay, h_w, h_h = _tht_strip(True)
            v_pos, v_lay, v_w, v_h = _tht_strip(False)
            # Pick arrangement closest to 1:1 aspect ratio
            h_ratio = max(h_w, h_h) / max(0.1, min(h_w, h_h))
            v_ratio = max(v_w, v_h) / max(0.1, min(v_w, v_h))
            if v_ratio < h_ratio:
                component_positions, component_layers, total_w, max_h = (
                    v_pos,
                    v_lay,
                    v_w,
                    v_h,
                )
            else:
                component_positions, component_layers, total_w, max_h = (
                    h_pos,
                    h_lay,
                    h_w,
                    h_h,
                )
            # Return with tight bbox (no clearance padding)
            return PlacedGroup(
                group=group,
                bounding_box=(total_w, max_h),
                component_positions=component_positions,
                component_layers=component_layers,
            )

        # --- Compute virtual board size from total component area ---
        total_area = sum(c.area for c in member_comps.values())
        n = len(member_comps)
        # Overhead: enough room for clearance between components.
        # Small groups need proportionally more margin.
        overhead = max(2.0, 3.5 - 0.15 * n)
        virt_area = max(total_area * overhead, 25.0)  # at least 5x5mm
        # Aspect ratio: slightly landscape (1.2:1) for typical IC+passives layout
        aspect = 1.2
        virt_w = math.sqrt(virt_area * aspect)
        virt_h = virt_w / aspect
        # Round up to nearest mm
        virt_w = math.ceil(virt_w)
        virt_h = math.ceil(virt_h)

        # --- Build group-scoped connectivity graph ---
        member_set = set(group.member_refs)
        group_conn = AdjacencyGraph()
        for net in nets.values():
            if net.name in ("GND", "/GND"):
                continue
            refs_in_net = [r for r, _ in net.pad_refs if r in member_set]
            if len(refs_in_net) < 2:
                continue
            weight = 3.0 if net.is_power else 1.0
            for i in range(len(refs_in_net)):
                for j in range(i + 1, len(refs_in_net)):
                    group_conn.add_edge(refs_in_net[i], refs_in_net[j], weight)

        # --- Deep copy group members for manipulation ---
        work_comps = {ref: copy.deepcopy(c) for ref, c in member_comps.items()}

        # --- Initial placement: leader at center, others around it ---
        center = Point(virt_w / 2, virt_h / 2)
        leader = work_comps.get(group.leader_ref)

        # Sort members: leader first, then ICs, then by connectivity, then passives
        sorted_refs = sorted(
            work_comps.keys(),
            key=lambda r: (
                0
                if r == group.leader_ref
                else 1
                if work_comps[r].kind == "ic"
                else 2
                if work_comps[r].kind not in ("passive", "misc", "")
                else 3,
                -group_conn.degree(r),
            ),
        )

        # Place in radial pattern around center
        if leader:
            old_pos = Point(leader.pos.x, leader.pos.y)
            leader.pos = Point(center.x, center.y)
            _update_pad_positions(leader, old_pos, leader.rotation)

        radius_base = max(3.0, math.sqrt(n) * 2.0)
        others = [r for r in sorted_refs if r != group.leader_ref]
        for i, ref in enumerate(others):
            c = work_comps[ref]
            angle = 2 * math.pi * i / max(len(others), 1)
            # Tighter radius for decoupling caps
            is_cap = ref.startswith("C")
            r = radius_base * (0.4 if is_cap else 0.8)
            old_pos = Point(c.pos.x, c.pos.y)
            c.pos = Point(
                center.x + r * math.cos(angle),
                center.y + r * math.sin(angle),
            )
            _update_pad_positions(c, old_pos, c.rotation)

        # --- Mini force-directed simulation ---
        iters = min(120, max(40, n * 15))
        damping = 1.0
        virt_tl = Point(0, 0)
        virt_br = Point(virt_w, virt_h)

        for _ in range(iters):
            forces = {r: Point(0, 0) for r in work_comps}

            # Attraction: connected pairs
            refs_list = list(work_comps.keys())
            for i_r, ra in enumerate(refs_list):
                for rb in refs_list[i_r + 1 :]:
                    w = group_conn.weight(ra, rb)
                    if w <= 0:
                        continue
                    a, b = work_comps[ra], work_comps[rb]
                    d = max(a.pos.dist(b.pos), 0.1)
                    target = (a.width_mm + b.width_mm) / 2 + self.clearance
                    f = (
                        self.k_attract * w * (d - target) * 2.0
                    )  # 2x stronger within group
                    dx = (b.pos.x - a.pos.x) / d * f
                    dy = (b.pos.y - a.pos.y) / d * f
                    forces[ra].x += dx
                    forces[ra].y += dy
                    forces[rb].x -= dx
                    forces[rb].y -= dy

            # Repulsion: overlap-based
            for i_r, ra in enumerate(refs_list):
                for rb in refs_list[i_r + 1 :]:
                    a, b = work_comps[ra], work_comps[rb]
                    overlap = _bbox_overlap_amount(a, b)
                    d = max(a.pos.dist(b.pos), 0.1)
                    # Lighter repulsion within groups (we want tight packing)
                    if overlap > 0:
                        f = 4.0 * math.sqrt(overlap)
                    elif (
                        d
                        < (max(a.width_mm, a.height_mm) + max(b.width_mm, b.height_mm))
                        / 2
                        + self.clearance
                    ):
                        f = self.k_repel * 0.01 * (a.area * b.area) / (d * d)
                    else:
                        continue
                    dx = (a.pos.x - b.pos.x) / d * f
                    dy = (a.pos.y - b.pos.y) / d * f
                    forces[ra].x += dx
                    forces[ra].y += dy
                    forces[rb].x -= dx
                    forces[rb].y -= dy

            # Centering force: gently pull toward virtual board center
            for ref in work_comps:
                c = work_comps[ref]
                cx_off = center.x - c.pos.x
                cy_off = center.y - c.pos.y
                forces[ref].x += cx_off * 0.02
                forces[ref].y += cy_off * 0.02

            # Apply forces
            for ref in work_comps:
                c = work_comps[ref]
                dx = forces[ref].x * damping
                dy = forces[ref].y * damping
                mag = math.hypot(dx, dy)
                max_step = 2.0 * damping
                if mag > max_step:
                    dx *= max_step / mag
                    dy *= max_step / mag

                old_pos = Point(c.pos.x, c.pos.y)
                c.pos.x += dx
                c.pos.y += dy
                # Clamp to virtual board
                hw, hh = c.width_mm / 2, c.height_mm / 2
                c.pos.x = max(virt_tl.x + hw + 0.5, min(virt_br.x - hw - 0.5, c.pos.x))
                c.pos.y = max(virt_tl.y + hh + 0.5, min(virt_br.y - hh - 0.5, c.pos.y))
                _update_pad_positions(c, old_pos, c.rotation)

            damping *= 0.96

        # --- Resolve overlaps within the virtual board ---
        self._resolve_overlaps_bounded(work_comps, virt_tl, virt_br)

        # --- Rotation optimization within group ---
        # Build a mini BoardState for rotation scoring
        mini_state = BoardState(
            components=work_comps,
            nets={
                n.name: n
                for n in nets.values()
                if any(r in member_set for r, _ in n.pad_refs)
            },
            board_outline=(virt_tl, virt_br),
        )
        for ref, comp in work_comps.items():
            if comp.kind not in ("ic", "connector") or len(comp.pads) < 2:
                continue
            pad_offsets = [
                (p.pos.x - comp.pos.x, p.pos.y - comp.pos.y) for p in comp.pads
            ]
            orig_rot = comp.rotation
            best_rot = orig_rot
            best_score = self._score_rotation_for_routing(mini_state, comp)
            for rot in [0, 90, 180, 270]:
                if rot == orig_rot:
                    continue
                delta = math.radians(rot - orig_rot)
                cos_d, sin_d = math.cos(delta), math.sin(delta)
                for k, p in enumerate(comp.pads):
                    ox, oy = pad_offsets[k]
                    p.pos = Point(
                        comp.pos.x + ox * cos_d + oy * sin_d,
                        comp.pos.y - ox * sin_d + oy * cos_d,
                    )
                comp.rotation = rot
                rot_score = self._score_rotation_for_routing(mini_state, comp)
                if rot_score > best_score:
                    best_score = rot_score
                    best_rot = rot
            # Apply best
            delta = math.radians(best_rot - orig_rot)
            cos_d, sin_d = math.cos(delta), math.sin(delta)
            for k, p in enumerate(comp.pads):
                ox, oy = pad_offsets[k]
                p.pos = Point(
                    comp.pos.x + ox * cos_d + oy * sin_d,
                    comp.pos.y - ox * sin_d + oy * cos_d,
                )
            comp.rotation = best_rot

        # --- Compute tight bounding box and normalize to origin ---
        # For small groups of large components (e.g. battery holder pairs),
        # reduce the inter-component clearance padding so the group block
        # doesn't dominate the board.
        large_area_threshold = 100.0  # mm²
        if n <= 2 and all(c.area > large_area_threshold for c in work_comps.values()):
            bbox_clearance = self.clearance * 0.15
        else:
            bbox_clearance = self.clearance / 2

        all_tls = []
        all_brs = []
        for c in work_comps.values():
            c_tl, c_br = c.bbox(bbox_clearance)
            all_tls.append(c_tl)
            all_brs.append(c_br)
        if not all_tls:
            return PlacedGroup(
                group=group,
                bounding_box=(0, 0),
                component_positions={},
                component_layers={},
            )

        min_x = min(p.x for p in all_tls)
        min_y = min(p.y for p in all_tls)
        max_x = max(p.x for p in all_brs)
        max_y = max(p.y for p in all_brs)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y

        # Normalize positions: make min corner = (0, 0)
        component_positions = {}
        component_layers = {}
        for ref, c in work_comps.items():
            rel_x = c.pos.x - min_x
            rel_y = c.pos.y - min_y
            component_positions[ref] = (rel_x, rel_y, c.rotation)
            component_layers[ref] = c.layer

        # --- Constrain group aspect ratio to prevent overly elongated blocks ---
        max_aspect = 3.0
        if bbox_w > 0 and bbox_h > 0:
            ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h)
            if ratio > max_aspect:
                if n <= 3:
                    # For small groups, try the perpendicular arrangement.
                    # Compute both horizontal and vertical layouts and pick
                    # the one with the better (closer to 1:1) aspect ratio.
                    refs = list(work_comps.keys())
                    if n == 2:
                        c1 = work_comps[refs[0]]
                        c2 = work_comps[refs[1]]
                        gap = bbox_clearance * 2

                        # Horizontal arrangement
                        h_w = c1.width_mm + c2.width_mm + gap + bbox_clearance * 2
                        h_h = max(c1.height_mm, c2.height_mm) + bbox_clearance * 2
                        h_ratio = max(h_w, h_h) / max(min(h_w, h_h), 0.1)

                        # Vertical arrangement
                        v_w = max(c1.width_mm, c2.width_mm) + bbox_clearance * 2
                        v_h = c1.height_mm + c2.height_mm + gap + bbox_clearance * 2
                        v_ratio = max(v_w, v_h) / max(min(v_w, v_h), 0.1)

                        # Pick whichever is closer to 1:1
                        if v_ratio < h_ratio and v_ratio < ratio:
                            # Reflow to vertical: stack components
                            x = bbox_clearance + max(c1.width_mm, c2.width_mm) / 2
                            y1 = bbox_clearance + c1.height_mm / 2
                            y2 = bbox_clearance + c1.height_mm + gap + c2.height_mm / 2
                            component_positions[refs[0]] = (x, y1, c1.rotation)
                            component_positions[refs[1]] = (x, y2, c2.rotation)
                            bbox_w, bbox_h = v_w, v_h
                        elif h_ratio < v_ratio and h_ratio < ratio:
                            # Reflow to horizontal: place side by side
                            y = bbox_clearance + max(c1.height_mm, c2.height_mm) / 2
                            x1 = bbox_clearance + c1.width_mm / 2
                            x2 = bbox_clearance + c1.width_mm + gap + c2.width_mm / 2
                            component_positions[refs[0]] = (x1, y, c1.rotation)
                            component_positions[refs[1]] = (x2, y, c2.rotation)
                            bbox_w, bbox_h = h_w, h_h
                    # else n==3: keep current layout (reflow is complex)
                else:
                    # For larger groups, compress the long axis toward
                    # the target ratio to reduce elongation.
                    target_ratio = max_aspect
                    if bbox_w > bbox_h:
                        target_w = bbox_h * target_ratio
                        if target_w < bbox_w:
                            scale = target_w / bbox_w
                            cx = bbox_w / 2
                            new_pos = {}
                            for ref, (rx, ry, rot) in component_positions.items():
                                new_pos[ref] = (cx + (rx - cx) * scale, ry, rot)
                            component_positions = new_pos
                            bbox_w = target_w
                    else:
                        target_h = bbox_w * target_ratio
                        if target_h < bbox_h:
                            scale = target_h / bbox_h
                            cy = bbox_h / 2
                            new_pos = {}
                            for ref, (rx, ry, rot) in component_positions.items():
                                new_pos[ref] = (rx, cy + (ry - cy) * scale, rot)
                            component_positions = new_pos
                            bbox_h = target_h

        return PlacedGroup(
            group=group,
            bounding_box=(bbox_w, bbox_h),
            component_positions=component_positions,
            component_layers=component_layers,
        )

    def _resolve_overlaps_bounded(
        self, comps: dict[str, Component], tl: Point, br: Point
    ):
        """Resolve overlaps within bounded area (virtual board for groups).

        Simpler version of _resolve_overlaps that uses the provided bounds
        instead of self.state.board_outline.
        """
        refs = list(comps.keys())
        half_gap = self.clearance / 2.0

        for iteration in range(200):
            moved = False
            for i in range(len(refs)):
                a = comps[refs[i]]
                a_tl, a_br = a.bbox(half_gap)
                for j in range(i + 1, len(refs)):
                    b = comps[refs[j]]
                    b_tl, b_br = b.bbox(half_gap)
                    ox = min(a_br.x, b_br.x) - max(a_tl.x, b_tl.x)
                    oy = min(a_br.y, b_br.y) - max(a_tl.y, b_tl.y)
                    if ox <= 0 or oy <= 0:
                        continue
                    # Both free: split the push evenly
                    hw_a, hh_a = a.width_mm / 2, a.height_mm / 2
                    hw_b, hh_b = b.width_mm / 2, b.height_mm / 2
                    if ox < oy:
                        push = (ox + 0.1) / 2
                        sign = 1.0 if a.pos.x >= b.pos.x else -1.0
                        old_a = Point(a.pos.x, a.pos.y)
                        old_b = Point(b.pos.x, b.pos.y)
                        a.pos.x = max(
                            tl.x + hw_a + 0.5,
                            min(br.x - hw_a - 0.5, a.pos.x + sign * push),
                        )
                        b.pos.x = max(
                            tl.x + hw_b + 0.5,
                            min(br.x - hw_b - 0.5, b.pos.x - sign * push),
                        )
                    else:
                        push = (oy + 0.1) / 2
                        sign = 1.0 if a.pos.y >= b.pos.y else -1.0
                        old_a = Point(a.pos.x, a.pos.y)
                        old_b = Point(b.pos.x, b.pos.y)
                        a.pos.y = max(
                            tl.y + hh_a + 0.5,
                            min(br.y - hh_a - 0.5, a.pos.y + sign * push),
                        )
                        b.pos.y = max(
                            tl.y + hh_b + 0.5,
                            min(br.y - hh_b - 0.5, b.pos.y - sign * push),
                        )
                    _update_pad_positions(a, old_a, a.rotation)
                    _update_pad_positions(b, old_b, b.rotation)
                    a_tl, a_br = a.bbox(half_gap)
                    moved = True
            if not moved:
                break

    def solve(
        self, max_iterations: int = None, convergence_threshold: float = None
    ) -> dict[str, Component]:
        """Run full placement pipeline. Returns updated components dict."""
        # Deep copy so we don't mutate the original
        comps = {ref: copy.deepcopy(c) for ref, c in self.state.components.items()}
        # Build a working state for scoring
        work_state = copy.copy(self.state)
        work_state.components = comps

        # Detect alignment groups from the INITIAL component positions.
        # SA refinement happily scrambles paired components (parallel
        # batteries, header arrays, LED rows) far enough apart that
        # post-SA position-based axis inference can't tell which axis
        # they were meant to share. Detecting up-front captures the
        # user's schematic-time intent; we apply the snap at the end of
        # solve() once the SA-chosen group center is known.
        from kicraft.autoplacer.brain.placement_alignment import (
            apply_alignment_repair,
            detect_alignment_groups,
        )

        alignment_groups = detect_alignment_groups(self.cfg, comps)

        # Build connectivity graph
        conn_graph = build_connectivity_graph(self.state.nets)

        # Step 0.5: Assign layers BEFORE edge pinning so pad positions
        # reflect the flip when computing connector placement
        self._assign_layers(comps)

        # Step 1: Pin edge components (connectors, mounting holes)
        self._pin_edge_components(comps)

        # Step 1.3: Align large paired components side-by-side
        self._align_large_pairs(comps)

        # Step 1.5: Use explicit IC groups to boost connectivity weights
        ic_groups = self.cfg.get("ic_groups", {})
        if ic_groups:
            # Add extra weight to connections within IC groups
            for ic_ref, supporting in ic_groups.items():
                for sup_ref in supporting:
                    if sup_ref in comps and ic_ref in comps:
                        conn_graph.add_edge(sup_ref, ic_ref, 2.0)  # Strong bond
            clusters = find_communities(conn_graph, seed=self.seed)
            print(
                f"  Found {len(clusters)} component clusters (with {len(ic_groups)} IC groups)"
            )
        else:
            # Step 2: Cluster by connectivity (seeded for reproducible variation)
            clusters = find_communities(conn_graph, seed=self.seed)
            print(f"  Found {len(clusters)} component clusters")

        # Step 1.6: Sibling grouping — components with the same kind and
        # similar dimensions should be placed adjacent to conserve space.
        # Detects siblings by kind+value or kind+similar area.
        sibling_pairs = []
        comp_list = list(comps.values())
        for i, a in enumerate(comp_list):
            for b in comp_list[i + 1 :]:
                if a.locked or b.locked:
                    continue
                same_kind = a.kind == b.kind and a.kind not in ("", "misc", "passive")
                similar_size = (
                    a.area > 0
                    and b.area > 0
                    and min(a.area, b.area) / max(a.area, b.area) > 0.7
                )
                if same_kind and similar_size:
                    # Weight proportional to component area — larger siblings
                    # benefit more from adjacency (saves more board space)
                    weight = min(3.0, 1.0 + (a.area + b.area) / 200.0)
                    conn_graph.add_edge(a.ref, b.ref, weight)
                    sibling_pairs.append((a.ref, b.ref))
        if sibling_pairs:
            print(
                f"  Sibling grouping: {len(sibling_pairs)} pair(s) "
                f"({', '.join(f'{a}+{b}' for a, b in sibling_pairs)})"
            )

        # Step 3: Initial cluster placement (with seeded jitter)
        self._place_clusters(comps, clusters, conn_graph)

        # Step 4: Optimize layout within each cluster before global layout
        self._optimize_intra_cluster(comps, clusters, conn_graph)

        # Step 5: Try 4 rotations per IC/connector, keep best
        self._optimize_rotations(comps, work_state)

        # Step 6: Force-directed refinement with scoring feedback
        scorer = PlacementScorer(work_state, self.cfg)
        best_score = scorer.score()
        best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}

        # Legalize-during-force: periodically call legalize_components()
        # during force iterations to prevent deeply illegal states
        legalize_during = self.cfg.get("legalize_during_force", False)
        legalize_every = max(1, int(self.cfg.get("legalize_every_n", 5)))
        legalize_passes = max(1, int(self.cfg.get("legalize_during_force_passes", 2)))
        prefer_legal = self.cfg.get("prefer_legal_states", False)
        enable_swap = self.cfg.get("enable_swap_optimization", True)

        best_violations = float("inf")  # track legality for prefer_legal_states
        damping = 1.0
        stagnant = 0
        reheat_strength = self.cfg.get("reheat_strength", 0.0)
        reheat_done = False

        print(
            f"  Initial placement score: {best_score.total:.1f} "
            f"(nets={best_score.net_distance:.0f} "
            f"cross={best_score.crossover_score:.0f} "
            f"xovers={best_score.crossover_count})"
        )

        for iteration in range(self.max_iterations):
            # Temperature reheat: at 50% of iterations, apply perturbation kick
            if (
                not reheat_done
                and reheat_strength > 0
                and iteration == self.max_iterations // 2
            ):
                reheat_done = True
                tl_r, br_r = self.state.board_outline
                diag = math.hypot(br_r.x - tl_r.x, br_r.y - tl_r.y)
                kick_mag = diag * reheat_strength
                unlocked_refs = [r for r in comps if not comps[r].locked]
                for ref in unlocked_refs:
                    old_pos = Point(comps[ref].pos.x, comps[ref].pos.y)
                    comps[ref].pos.x += self.rng.gauss(0, kick_mag)
                    comps[ref].pos.y += self.rng.gauss(0, kick_mag)
                    # Clamp to board (pad-aware)
                    hw, hh = _pad_half_extents(comps[ref])
                    comps[ref].pos.x = max(
                        tl_r.x + hw + 1, min(br_r.x - hw - 1, comps[ref].pos.x)
                    )
                    comps[ref].pos.y = max(
                        tl_r.y + hh + 1, min(br_r.y - hh - 1, comps[ref].pos.y)
                    )
                    _update_pad_positions(comps[ref], old_pos, comps[ref].rotation)
                damping = 0.7  # partial reheat of damping
                stagnant = 0
                self._seen_force_states.clear()

            max_disp = self._force_step(comps, conn_graph, damping)
            self._resolve_overlaps(comps)
            self._clamp_pads_to_board(comps)
            # Periodic legalization during force simulation
            if legalize_during and iteration > 0 and iteration % legalize_every == 0:
                self.legalize_components(comps, max_passes=legalize_passes)
            damping *= self.cooling

            # Score more frequently for faster convergence detection
            if iteration % self.score_every_n == 0:
                work_state.components = comps
                s = scorer.score()
                # When prefer_legal_states is on, factor legality into
                # best-state selection: fewer violations wins even if
                # placement score is slightly lower.
                if prefer_legal:
                    diag = self.legality_diagnostics(comps)
                    violations = diag["overlap_count"] + diag["pad_outside_count"]
                    # Accept if: fewer violations, OR same violations + better score
                    if violations < best_violations or (
                        violations == best_violations and s.total > best_score.total
                    ):
                        best_score = s
                        best_violations = violations
                        best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}
                        stagnant = 0
                    else:
                        stagnant += 1
                        if stagnant >= 3 and stagnant % 3 == 0:
                            comps = {r: copy.deepcopy(c) for r, c in best_comps.items()}
                elif s.total > best_score.total:
                    best_score = s
                    best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}
                    stagnant = 0
                else:
                    stagnant += 1
                    if stagnant >= 3 and stagnant % 3 == 0:
                        comps = {r: copy.deepcopy(c) for r, c in best_comps.items()}

                if stagnant >= 20:
                    print(f"  Converged at iteration {iteration + 1}")
                    break

            if max_disp < self.convergence_threshold and iteration > 30:
                print(f"  Displacement converged at iteration {iteration + 1}")
                break

            # Adaptive convergence: early exit when placement is good and stable
            if (
                iteration > 15
                and best_score.total > 85.0
                and max_disp < 3.0
                and stagnant >= 3
            ):
                print(
                    f"  Adaptive early exit at iteration {iteration + 1} "
                    f"(score={best_score.total:.1f}, disp={max_disp:.2f})"
                )
                break

        # SA refinement: escape local minima after FD convergence
        if self.cfg.get("sa_refine_enabled", True):
            self._seen_force_states.clear()
            work_state.components = best_comps
            best_comps = self._sa_refine(
                {r: copy.deepcopy(c) for r, c in best_comps.items()},
                work_state,
                scorer,
                max_iters=int(self.cfg.get("sa_refine_iterations", 1000)),
                init_temp=float(self.cfg.get("sa_refine_initial_temp", 5.0)),
                cooling_rate=float(self.cfg.get("sa_refine_cooling_rate", 0.995)),
                move_radius=float(self.cfg.get("sa_refine_move_radius_mm", 2.0)),
                swap_prob=float(self.cfg.get("sa_refine_swap_probability", 0.3)),
                rotation_prob=float(self.cfg.get("sa_refine_rotation_probability", 0.2)),
            )

        # Alignment repair: apply the alignment_groups detected from the
        # INITIAL positions (before SA could scramble them). Runs after
        # SA so the group's parallel-axis center reflects the solver's
        # chosen position; the repair snaps perpendicular-axis to the
        # current group mean and redistributes at fixed pitch.
        if alignment_groups:
            apply_alignment_repair(best_comps, alignment_groups)

        # Step 7: Swap optimization — directly minimize crossovers
        comps = best_comps
        if enable_swap:
            self._seen_force_states.clear()
            work_state.components = comps
            best_cross = count_crossings(work_state)
            print(f"  Starting swap optimization ({best_cross} crossings)")

            # Build set of refs in aligned pairs — exclude from swaps to
            # preserve side-by-side alignment
            aligned_refs = set()
            for ref_a, ref_b, _axis in self._aligned_pairs:
                aligned_refs.add(ref_a)
                aligned_refs.add(ref_b)

            improved = True
            swap_round = 0
            while improved and swap_round < 5:
                improved = False
                swap_round += 1
                unlocked = [
                    r for r in comps if not comps[r].locked and r not in aligned_refs
                ]
                for i in range(len(unlocked)):
                    for j in range(i + 1, len(unlocked)):
                        a, b = comps[unlocked[i]], comps[unlocked[j]]
                        # Only swap components of similar size
                        size_ratio = max(a.area, b.area) / max(
                            min(a.area, b.area), 0.01
                        )
                        if size_ratio > 4:
                            continue
                        # Swap positions and update pads
                        a.pos, b.pos = Point(b.pos.x, b.pos.y), Point(a.pos.x, a.pos.y)
                        _swap_pad_positions(a, b)
                        cross = count_crossings(work_state)
                        if cross < best_cross:
                            best_cross = cross
                            improved = True
                        else:
                            # Revert
                            a.pos, b.pos = (
                                Point(b.pos.x, b.pos.y),
                                Point(a.pos.x, a.pos.y),
                            )
                            _swap_pad_positions(a, b)
                if improved:
                    print(f"    Swap round {swap_round}: {best_cross} crossings")

            best_comps = comps
        else:
            self._seen_force_states.clear()

        # Re-snap aligned pairs after swap optimization
        self._re_snap_aligned_pairs(best_comps)

        # Step 8: Snap to grid
        self._snap_to_grid(best_comps)

        # Re-snap aligned pairs after grid snap
        self._re_snap_aligned_pairs(best_comps)

        # Step 8.5: Orderedness — align passives into neat rows/columns
        orderedness = self.cfg.get("orderedness", 0.0)
        if orderedness > 0.01:
            self._apply_orderedness(best_comps, orderedness)
            # Re-snap aligned pairs after orderedness
            self._re_snap_aligned_pairs(best_comps)

        # Step 9: Final exhaustive overlap resolution — guarantee no courtyard
        # overlaps before routing. Must run after snap since snapping can
        # re-introduce small overlaps.
        self._resolve_overlaps(best_comps)

        # Re-snap aligned pairs after overlap resolution
        self._re_snap_aligned_pairs(best_comps)

        # Step 9.5: Comprehensive legalization repair for subcircuit mode
        if prefer_legal:
            repair_passes = int(self.cfg.get("leaf_legality_repair_passes", 12))
            self.legalize_components(best_comps, max_passes=repair_passes)

        # Step 10: Hard clamp — nothing outside the board
        self._clamp_to_board(best_comps)

        # Step 11: Ensure all pads are inside the board boundary
        self._clamp_pads_to_board(best_comps)

        # Step 12: Validate pad containment — re-clamp if any pads still outside
        for clamp_pass in range(3):
            tl_v, br_v = self.state.board_outline
            inset_v = self.cfg.get("pad_inset_margin_mm", 0.3)
            any_outside = False
            for comp in best_comps.values():
                for pad in comp.pads:
                    if (
                        pad.pos.x < tl_v.x + inset_v
                        or pad.pos.x > br_v.x - inset_v
                        or pad.pos.y < tl_v.y + inset_v
                        or pad.pos.y > br_v.y - inset_v
                    ):
                        any_outside = True
                        break
                if any_outside:
                    break
            if not any_outside:
                break
            self._clamp_to_board(best_comps)
            self._clamp_pads_to_board(best_comps)
            if clamp_pass == 2:
                print("  WARNING: some pads still outside board after 3 clamp passes")

        # Step 13: Re-pin edge/corner components that may have drifted
        # during overlap resolution (both-locked case can push pinned parts)
        self._restore_pinned_positions(best_comps)

        # Step 13.5: Overlap resolution after restoration — restoring pinned
        # components can introduce new overlaps (e.g. a mounting hole restored
        # to its corner now overlaps a component that was pushed there during
        # the force simulation).  Re-resolve, then re-restore to ensure both
        # overlap-free placement AND correct pinned positions.
        self._resolve_overlaps(best_comps)
        self._restore_pinned_positions(best_comps)

        # Step 14: Re-validate pad containment after restoring pinned positions
        self._clamp_pads_to_board(best_comps)

        # Final score
        work_state.components = best_comps
        final = PlacementScorer(work_state, self.cfg).score()
        print(
            f"  Final placement score: {final.total:.1f} "
            f"(nets={final.net_distance:.0f} "
            f"cross={final.crossover_score:.0f} "
            f"xovers={final.crossover_count})"
        )

        return best_comps

    def _score_rotation_for_routing(
        self, work_state: BoardState, comp: Component
    ) -> float:
        """Score component rotation for routability.

        Considers: crossovers, pad accessibility (pads not blocked by component body),
        and net distance.
        """
        cross = count_crossings(work_state)
        cross_score = 100 / (1 + cross) if cross > 0 else 100

        # Prefer rotations where pads face outward (toward board edge or open space)
        # Check if pads have clear path to edges
        tl, br = work_state.board_outline
        accessible = 0
        for pad in comp.pads:
            px, py = pad.pos.x, pad.pos.y
            # Check each quadrant for openness
            dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dx, dy in dirs:
                dist = 0
                ox, oy = px, py
                while dist < 30:
                    ox += dx * 2
                    oy += dy * 2
                    if tl.x < ox < br.x and tl.y < oy < br.y:
                        dist += 2
                    else:
                        break
            accessible += dist

        # Higher = more accessible area around pads
        access_score = min(100, accessible / 10)

        # Net distance matters for routing
        from .graph import total_ratsnest_length

        net_dist = total_ratsnest_length(work_state)
        dist_score = max(0, 100 - net_dist / 5)

        return cross_score * 0.5 + access_score * 0.3 + dist_score * 0.2

    @staticmethod
    def _best_rotation_for_edge(comp: Component, edge: str) -> float:
        """Find the rotation (0/90/180/270) that orients a connector flush
        against the named edge with its opening facing outward.

        Strategy:
        1. If the component has a known opening_direction (detected from
           body-extension-beyond-pads in local coords), compute the exact
           rotation that points the opening outward from the given edge.
        2. Otherwise fall back to aspect-ratio heuristics (long axis
           parallel to the edge).
        """
        # Expected outward direction per edge (board-space angle).
        # On B.Cu, Flip() mirrors the local X-axis, so left/right swap.
        if comp.layer == Layer.BACK:
            outward = {"left": 0, "right": 180, "top": 270, "bottom": 90}
        else:
            outward = {"left": 180, "right": 0, "top": 270, "bottom": 90}

        if comp.opening_direction is not None:
            # Direct computation: we need the opening (local-frame angle)
            # to end up pointing at outward[edge] in board-space.
            # KiCad forward: board_angle = local_angle - rotation.
            # So: rotation = opening_direction - outward[edge]
            rot = (comp.opening_direction - outward[edge]) % 360
            return rot

        # -- Fallback: no detectable opening direction --
        # Orient the long axis parallel to the edge.
        if not comp.pads:
            return comp.rotation

        w, h = comp.width_mm, comp.height_mm
        if edge in ("left", "right"):
            # Want height >= width (long axis vertical, parallel to edge).
            if w > h * 1.1:
                return (comp.rotation + 90) % 360
            return comp.rotation
        else:
            # top/bottom: want width >= height (long axis horizontal).
            if h > w * 1.1:
                return (comp.rotation + 90) % 360
            return comp.rotation

    def _pin_edge_components(self, comps: dict[str, Component]):
        """Pin components based on component_zones config, with fallback heuristics.

        Supports three constraint types:
          - edge: snap to named edge (left/right/top/bottom), lock in place
          - corner: pin to named corner (top-left/top-right/bottom-left/bottom-right)
          - zone: confine to a board region (used during _place_clusters, not locked)

        Connectors on the same edge are grouped together in a row/column
        with spacing, preventing them from scattering or falling off the edge.

        Connector orientation is auto-corrected so pads face the board
        center (e.g., USB connector opening faces outward, pads inward).

        Connectors without explicit zone config fall back to nearest-edge heuristic.
        Mounting holes without config fall back to nearest-corner.

        Positions are randomized along the assigned edge/zone each round
        (controlled by self.rng and edge_jitter_mm config) so that placements
        vary across experiment rounds.

        When unlock_all_footprints is True, initial positions are still set for
        edge/corner constraints but components are NOT locked — the force
        simulation can move them, and edge_compliance scoring incentivizes
        keeping them near edges.

        Saves target positions in self._pinned_targets for later restoration
        by _restore_pinned_positions().
        """
        self._pinned_targets: dict[str, Point] = {}
        tl, br = self.state.board_outline
        margin = self.edge_margin
        zones = self.cfg.get("component_zones", {})
        unlock_all = self.cfg.get("unlock_all_footprints", False)
        jitter = self.cfg.get("edge_jitter_mm", 5.0)
        pad_inset = self.cfg.get("pad_inset_margin_mm", 0.3)
        connector_gap = self.cfg.get("connector_gap_mm", 2.0)
        connector_inset = self.cfg.get("connector_edge_inset_mm", 1.0)

        # Validate configured refs against actual components
        missing_refs = [ref for ref in zones if ref not in comps]
        if missing_refs:
            print(
                f"  WARNING: component_zones references not found on board: "
                f"{', '.join(missing_refs)}"
            )

        def _random_in_corner(corner: str, comp: Component) -> Point:
            """Return a position near the named corner with small jitter."""
            cx = tl.x + margin if "left" in corner else br.x - margin
            cy = tl.y + margin if "top" in corner else br.y - margin
            cx += self.rng.uniform(-jitter, jitter)
            cy += self.rng.uniform(-jitter, jitter)
            # Clamp to board
            hw, hh = comp.width_mm / 2, comp.height_mm / 2
            cx = max(tl.x + hw + 1, min(br.x - hw - 1, cx))
            cy = max(tl.y + hh + 1, min(br.y - hh - 1, cy))
            return Point(cx, cy)

        def _shift_pads_inside(comp: Component, assigned_edge: str = None):
            """Shift component so ALL pads are inside the board boundary.

            If assigned_edge is set, skip shifting on the axis perpendicular
            to the edge — don't pull an edge-pinned connector away from its
            assigned edge.  Only enforce containment on the other 3 sides.
            """
            if not comp.pads:
                return
            pad_xs = [p.pos.x for p in comp.pads]
            pad_ys = [p.pos.y for p in comp.pads]
            shift_x = shift_y = 0.0

            # X axis shifts (skip the assigned-edge side)
            if min(pad_xs) < tl.x + pad_inset and assigned_edge != "left":
                shift_x = tl.x + pad_inset - min(pad_xs)
            elif max(pad_xs) > br.x - pad_inset and assigned_edge != "right":
                shift_x = br.x - pad_inset - max(pad_xs)

            # Y axis shifts (skip the assigned-edge side)
            if min(pad_ys) < tl.y + pad_inset and assigned_edge != "top":
                shift_y = tl.y + pad_inset - min(pad_ys)
            elif max(pad_ys) > br.y - pad_inset and assigned_edge != "bottom":
                shift_y = br.y - pad_inset - max(pad_ys)

            if abs(shift_x) > 0.01 or abs(shift_y) > 0.01:
                comp.pos.x += shift_x
                comp.pos.y += shift_y
                for pad in comp.pads:
                    pad.pos.x += shift_x
                    pad.pos.y += shift_y
                if comp.body_center is not None:
                    comp.body_center = Point(
                        comp.body_center.x + shift_x,
                        comp.body_center.y + shift_y,
                    )

        def _connector_edge_x(comp: Component, edge: str) -> float:
            """Compute X position so connector body edge is flush with the
            board edge (plus connector_inset_mm offset).

            For left edge: body left edge at tl.x + connector_inset
            For right edge: body right edge at br.x - connector_inset
            """
            hw = comp.width_mm / 2
            if edge == "left":
                return tl.x + connector_inset + hw
            else:  # right
                return br.x - connector_inset - hw

        def _connector_edge_y(comp: Component, edge: str) -> float:
            """Compute Y position so connector body edge is flush with the
            board edge (plus connector_inset_mm offset).

            For top edge: body top edge at tl.y + connector_inset
            For bottom edge: body bottom edge at br.y - connector_inset
            """
            hh = comp.height_mm / 2
            if edge == "top":
                return tl.y + connector_inset + hh
            else:  # bottom
                return br.y - connector_inset - hh

        def _orient_and_place(comp: Component, edge: str, pos: Point):
            """Orient connector to face inward and move to position."""
            old_pos = Point(comp.pos.x, comp.pos.y)
            old_rot = comp.rotation
            # Auto-orient unless config specifies explicit rotation
            zone_cfg = zones.get(comp.ref, {})
            if "rotation" in zone_cfg:
                comp.rotation = zone_cfg["rotation"]
            else:
                comp.rotation = self._best_rotation_for_edge(comp, edge)
            comp.pos = pos
            _update_pad_positions(comp, old_pos, old_rot)
            _shift_pads_inside(comp, assigned_edge=edge)

        # --- Collect edge-pinned connectors by edge for grouped placement ---
        edge_groups: dict[str, list[str]] = {}  # edge -> [ref, ...]
        for ref, comp in comps.items():
            zone_cfg = zones.get(ref, {})
            if "edge" in zone_cfg:
                edge = zone_cfg["edge"]
                edge_groups.setdefault(edge, []).append(ref)
            elif (
                comp.kind == "connector"
                and "corner" not in zone_cfg
                and "zone" not in zone_cfg
            ):
                # Fallback: assign to nearest edge
                x, y = comp.pos.x, comp.pos.y
                distances = {
                    "left": x - tl.x,
                    "right": br.x - x,
                    "top": y - tl.y,
                    "bottom": br.y - y,
                }
                nearest = min(distances, key=distances.get)
                edge_groups.setdefault(nearest, []).append(ref)

        # --- Place each edge group as a compact row/column ---
        for edge, refs in edge_groups.items():
            group_comps = [comps[r] for r in refs]
            # Sort by component area descending (largest first = anchor)
            order = sorted(
                range(len(refs)), key=lambda i: group_comps[i].area, reverse=True
            )

            if edge in ("left", "right"):
                # Column along Y axis — body edge flush with board edge
                # Total height needed for the group
                sizes = [group_comps[i].height_mm for i in order]
                total_h = sum(sizes) + connector_gap * (len(sizes) - 1)
                # Randomize the group's starting Y within usable range
                usable_top = tl.y + margin + sizes[0] / 2
                usable_bot = br.y - margin - sizes[-1] / 2
                group_span = total_h
                if group_span < (usable_bot - usable_top):
                    start_y = self.rng.uniform(
                        usable_top, usable_bot - group_span + sizes[0] / 2
                    )
                else:
                    start_y = usable_top  # not enough room, pack from top

                cursor_y = start_y
                for idx in order:
                    comp = group_comps[idx]
                    # Place connector body flush to board edge
                    fixed_x = _connector_edge_x(comp, edge)
                    pos = Point(fixed_x, cursor_y)
                    _orient_and_place(comp, edge, pos)
                    self._pinned_targets[refs[idx]] = Point(comp.pos.x, comp.pos.y)
                    comp.locked = not unlock_all
                    cursor_y += comp.height_mm + connector_gap
            else:
                # Row along X axis — body edge flush with board edge
                sizes = [group_comps[i].width_mm for i in order]
                total_w = sum(sizes) + connector_gap * (len(sizes) - 1)
                usable_left = tl.x + margin + sizes[0] / 2
                usable_right = br.x - margin - sizes[-1] / 2
                group_span = total_w
                if group_span < (usable_right - usable_left):
                    start_x = self.rng.uniform(
                        usable_left, usable_right - group_span + sizes[0] / 2
                    )
                else:
                    start_x = usable_left
                cursor_x = start_x
                for idx in order:
                    comp = group_comps[idx]
                    # Place connector body flush to board edge
                    fixed_y = _connector_edge_y(comp, edge)
                    pos = Point(cursor_x, fixed_y)
                    _orient_and_place(comp, edge, pos)
                    self._pinned_targets[refs[idx]] = Point(comp.pos.x, comp.pos.y)
                    comp.locked = not unlock_all
                    cursor_x += comp.width_mm + connector_gap

        # --- Non-edge constraints (corners, zones, mounting holes) ---
        for ref, comp in comps.items():
            zone_cfg = zones.get(ref, {})
            # Skip if already handled as edge group
            if ref in self._pinned_targets:
                continue

            if "corner" in zone_cfg:
                corner = zone_cfg["corner"]
                old_pos = Point(comp.pos.x, comp.pos.y)
                comp.pos = _random_in_corner(corner, comp)
                _update_pad_positions(comp, old_pos, comp.rotation)
                self._pinned_targets[ref] = Point(comp.pos.x, comp.pos.y)
                comp.locked = not unlock_all

            elif "zone" in zone_cfg:
                zx0, zy0, zx1, zy1 = self._get_zone_bounds(zone_cfg["zone"])
                hw, hh = comp.width_mm / 2, comp.height_mm / 2
                old_pos = Point(comp.pos.x, comp.pos.y)
                comp.pos = Point(
                    self.rng.uniform(zx0 + hw, max(zx0 + hw + 1, zx1 - hw)),
                    self.rng.uniform(zy0 + hh, max(zy0 + hh + 1, zy1 - hh)),
                )
                _update_pad_positions(comp, old_pos, comp.rotation)

            elif comp.kind == "mounting_hole":
                pass  # handled in batch below

        # --- Batch mounting hole placement: force diagonal for 2, corners for 4 ---
        mh_refs = [
            ref
            for ref, comp in comps.items()
            if comp.kind == "mounting_hole"
            and ref not in self._pinned_targets
            and zones.get(ref, {}).get("corner") is None
            and zones.get(ref, {}).get("edge") is None
        ]
        if len(mh_refs) == 2:
            # Force to diagonally opposite corners
            diag = self.rng.choice(
                [
                    ("top-left", "bottom-right"),
                    ("top-right", "bottom-left"),
                ]
            )
            for ref, corner in zip(mh_refs, diag):
                comp = comps[ref]
                old_pos = Point(comp.pos.x, comp.pos.y)
                comp.pos = _random_in_corner(corner, comp)
                _update_pad_positions(comp, old_pos, comp.rotation)
                self._pinned_targets[ref] = Point(comp.pos.x, comp.pos.y)
                comp.locked = not unlock_all
        elif len(mh_refs) == 4:
            # One per corner
            corners = ["top-left", "top-right", "bottom-left", "bottom-right"]
            self.rng.shuffle(corners)
            for ref, corner in zip(mh_refs, corners):
                comp = comps[ref]
                old_pos = Point(comp.pos.x, comp.pos.y)
                comp.pos = _random_in_corner(corner, comp)
                _update_pad_positions(comp, old_pos, comp.rotation)
                self._pinned_targets[ref] = Point(comp.pos.x, comp.pos.y)
                comp.locked = not unlock_all
        else:
            # 1 or 3+ mounting holes: nearest-corner heuristic
            for ref in mh_refs:
                comp = comps[ref]
                corner = ""
                corner += "top" if comp.pos.y < (tl.y + br.y) / 2 else "bottom"
                corner += "-"
                corner += "left" if comp.pos.x < (tl.x + br.x) / 2 else "right"
                old_pos = Point(comp.pos.x, comp.pos.y)
                comp.pos = _random_in_corner(corner, comp)
                _update_pad_positions(comp, old_pos, comp.rotation)
                self._pinned_targets[ref] = Point(comp.pos.x, comp.pos.y)
                comp.locked = not unlock_all

    def _restore_pinned_positions(self, comps: dict[str, Component]):
        """Restore edge/corner-pinned components to their target positions.

        Called after overlap resolution as a safety net: the both-locked
        branch can still push pinned components if both are edge/corner
        pinned.  This snaps them back to the positions recorded during
        _pin_edge_components.
        """
        for ref, target in self._pinned_targets.items():
            comp = comps.get(ref)
            if comp is None:
                continue
            dx = target.x - comp.pos.x
            dy = target.y - comp.pos.y
            if abs(dx) < 0.01 and abs(dy) < 0.01:
                continue
            old_pos = Point(comp.pos.x, comp.pos.y)
            comp.pos.x = target.x
            comp.pos.y = target.y
            _update_pad_positions(comp, old_pos, comp.rotation)

    def _get_zone_bounds(self, zone_name: str) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) for a named board zone."""
        tl, br = self.state.board_outline
        margin = self.edge_margin
        mid_x = (tl.x + br.x) / 2
        mid_y = (tl.y + br.y) / 2

        zone_map = {
            "center": (tl.x + margin, tl.y + margin, br.x - margin, br.y - margin),
            "top": (tl.x + margin, tl.y + margin, br.x - margin, mid_y),
            "bottom": (tl.x + margin, mid_y, br.x - margin, br.y - margin),
            "left": (tl.x + margin, tl.y + margin, mid_x, br.y - margin),
            "right": (mid_x, tl.y + margin, br.x - margin, br.y - margin),
            "center-top": (tl.x + margin, tl.y + margin, br.x - margin, mid_y),
            "center-bottom": (tl.x + margin, mid_y, br.x - margin, br.y - margin),
            "center-left": (tl.x + margin, tl.y + margin, mid_x, br.y - margin),
            "center-right": (mid_x, tl.y + margin, br.x - margin, br.y - margin),
            "top-left": (tl.x + margin, tl.y + margin, mid_x, mid_y),
            "top-right": (mid_x, tl.y + margin, br.x - margin, mid_y),
            "bottom-left": (tl.x + margin, mid_y, mid_x, br.y - margin),
            "bottom-right": (mid_x, mid_y, br.x - margin, br.y - margin),
        }
        return zone_map.get(zone_name, zone_map["center"])

    def _place_clusters(
        self,
        comps: dict[str, Component],
        clusters: list[set[str]],
        conn_graph: AdjacencyGraph,
    ):
        """Place each cluster's components near their connectivity centroid.

        Supports three placement strategies controlled by config:
          - scatter_mode="cluster": centroid-based with jitter (default, exploit)
          - scatter_mode="random": uniform random within board bounds (explore)
          - signal_flow_order: biases cluster centroids left-to-right
          - component_zones with "zone": confines components to named regions
          - Decoupling caps (C* in ic_groups) placed at tighter radius to IC leader
        """
        tl, br = self.state.board_outline
        margin = self.edge_margin + 5.0  # keep away from edges
        scatter_mode = self.cfg.get("scatter_mode", "cluster")
        signal_flow = self.cfg.get("signal_flow_order", [])
        ic_groups = self.cfg.get("ic_groups", {})
        zones_cfg = self.cfg.get("component_zones", {})
        randomize_group = self.cfg.get("randomize_group_layout", False)

        # Build reverse map: component ref -> group leader
        ref_to_leader = {}
        for leader, members in ic_groups.items():
            ref_to_leader[leader] = leader
            for m in members:
                ref_to_leader[m] = leader

        # Build signal-flow X targets (evenly spaced across board width)
        flow_x_targets = {}
        if signal_flow:
            usable_left = tl.x + margin
            usable_right = br.x - margin
            for i, leader in enumerate(signal_flow):
                frac = (i + 0.5) / len(signal_flow)
                flow_x_targets[leader] = usable_left + frac * (
                    usable_right - usable_left
                )

        # Find locked component positions for attraction
        locked_positions = {ref: comp.pos for ref, comp in comps.items() if comp.locked}

        # Sort clusters by total connectivity (highest first) so the most
        # connected cluster gets placed first, improving net-topology bias.
        clusters = sorted(
            clusters,
            key=lambda c: sum(conn_graph.degree(r) for r in c),
            reverse=True,
        )

        for cluster in clusters:
            unlocked = [r for r in cluster if not comps[r].locked]
            if not unlocked:
                continue

            if scatter_mode == "random":
                # --- Random scatter: uniform random positions within bounds ---
                # Sort by area descending: large components placed first
                unlocked.sort(key=lambda r: comps[r].area, reverse=True)
                for ref in unlocked:
                    zone_cfg = zones_cfg.get(ref, {})
                    if "zone" in zone_cfg:
                        zx0, zy0, zx1, zy1 = self._get_zone_bounds(zone_cfg["zone"])
                    else:
                        zx0, zy0 = tl.x + margin, tl.y + margin
                        zx1, zy1 = br.x - margin, br.y - margin

                    hw, hh = comps[ref].width_mm / 2, comps[ref].height_mm / 2
                    old_pos = Point(comps[ref].pos.x, comps[ref].pos.y)
                    old_rot = comps[ref].rotation
                    comps[ref].pos = Point(
                        self.rng.uniform(zx0 + hw, max(zx0 + hw + 1, zx1 - hw)),
                        self.rng.uniform(zy0 + hh, max(zy0 + hh + 1, zy1 - hh)),
                    )
                    # Random allowed rotation
                    if comps[ref].kind == "ic":
                        comps[ref].rotation = self.rng.choice([0, 90, 180, 270])
                    elif comps[ref].kind == "passive":
                        comps[ref].rotation = self.rng.choice([0, 90])
                    _update_pad_positions(comps[ref], old_pos, old_rot)
                continue

            # --- Cluster mode: centroid-based with signal-flow bias ---
            # Compute centroid from locked neighbors' positions
            cx, cy, weight_sum = 0.0, 0.0, 0.0
            for ref in unlocked:
                for locked_ref, lpos in locked_positions.items():
                    w = conn_graph.weight(ref, locked_ref)
                    if w > 0:
                        cx += lpos.x * w
                        cy += lpos.y * w
                        weight_sum += w

            if weight_sum > 0:
                cx /= weight_sum
                cy /= weight_sum
            else:
                # Default to board center
                cx = (tl.x + br.x) / 2
                cy = (tl.y + br.y) / 2

            # Apply signal-flow X bias: blend centroid toward target X
            # Find the cluster's group leader (if any)
            cluster_leader = None
            for ref in cluster:
                leader = ref_to_leader.get(ref)
                if leader and leader in flow_x_targets:
                    cluster_leader = leader
                    break
            if cluster_leader and cluster_leader in flow_x_targets:
                target_x = flow_x_targets[cluster_leader]
                # 60% bias toward signal-flow target, 40% toward connectivity
                cx = 0.4 * cx + 0.6 * target_x

            # Clamp to board interior
            cx = max(tl.x + margin, min(br.x - margin, cx))
            cy = max(tl.y + margin, min(br.y - margin, cy))

            # Apply zone constraints: override centroid if component has a zone
            # (uses first zone-constrained component in cluster to bias centroid)
            for ref in unlocked:
                zone_cfg = zones_cfg.get(ref, {})
                if "zone" in zone_cfg:
                    zx0, zy0, zx1, zy1 = self._get_zone_bounds(zone_cfg["zone"])
                    cx = max(zx0, min(zx1, cx))
                    cy = max(zy0, min(zy1, cy))
                    break

            # Spread components around centroid (with seeded jitter)
            n = len(unlocked)
            # Sort by area descending: ICs and large components placed first,
            # then passives fill in around them.
            unlocked.sort(key=lambda r: comps[r].area, reverse=True)
            radius = math.sqrt(n) * 3.0  # spread based on count

            # Radius variation: wider for randomize_group_layout mode
            r_lo, r_hi = (0.3, 1.8) if randomize_group else (0.8, 1.2)

            # Track placed components for net-topology bias
            placed_this_cluster: set[str] = set()

            for i, ref in enumerate(unlocked):
                # Net-topology bias: if this component has already-placed
                # connected neighbors, bias position toward their centroid.
                nbr_cx, nbr_cy, nbr_w = 0.0, 0.0, 0.0
                for nbr, w in conn_graph.neighbors(ref).items():
                    if nbr in comps and (
                        comps[nbr].locked or nbr in placed_this_cluster
                    ):
                        nbr_cx += comps[nbr].pos.x * w
                        nbr_cy += comps[nbr].pos.y * w
                        nbr_w += w
                if nbr_w > 0:
                    # Blend 50% toward connected neighbors, 50% toward cluster centroid
                    local_cx = 0.5 * cx + 0.5 * (nbr_cx / nbr_w)
                    local_cy = 0.5 * cy + 0.5 * (nbr_cy / nbr_w)
                else:
                    local_cx, local_cy = cx, cy

                # Decoupling cap proximity: caps in IC groups get tighter radius
                is_decoupling_cap = (
                    ref.startswith("C")
                    and ref in ref_to_leader
                    and ref_to_leader[ref] != ref  # not the leader itself
                )
                if is_decoupling_cap:
                    # Place within 1.5× clearance of centroid (very tight)
                    r = self.clearance * 1.5 * self.rng.uniform(0.6, 1.0)
                else:
                    r = radius * (0.5 + 0.5 * (i % 2)) * self.rng.uniform(r_lo, r_hi)

                angle = 2 * math.pi * i / max(n, 1) + self.rng.gauss(0, 0.3)

                old_pos = Point(comps[ref].pos.x, comps[ref].pos.y)
                old_rot = comps[ref].rotation
                new_x = local_cx + r * math.cos(angle)
                new_y = local_cy + r * math.sin(angle)

                # Enforce zone bounds if component has a zone constraint
                zone_cfg = zones_cfg.get(ref, {})
                if "zone" in zone_cfg:
                    zx0, zy0, zx1, zy1 = self._get_zone_bounds(zone_cfg["zone"])
                    hw, hh = comps[ref].width_mm / 2, comps[ref].height_mm / 2
                    new_x = max(zx0 + hw, min(zx1 - hw, new_x))
                    new_y = max(zy0 + hh, min(zy1 - hh, new_y))

                comps[ref].pos = Point(new_x, new_y)
                _update_pad_positions(comps[ref], old_pos, old_rot)

                # Early rotation: try all 4 orientations for ICs at placement
                # time — prevents suboptimal rotations from locking in.
                if comps[ref].kind == "ic" and len(comps[ref].pads) >= 2:
                    pad_offsets = [
                        (p.pos.x - comps[ref].pos.x, p.pos.y - comps[ref].pos.y)
                        for p in comps[ref].pads
                    ]
                    orig_rot = comps[ref].rotation
                    best_rot = orig_rot
                    best_rscore = -1.0
                    temp_state = copy.copy(self.state)
                    temp_state.components = comps
                    for rot in [0, 90, 180, 270]:
                        delta = math.radians(rot - orig_rot)
                        cos_d, sin_d = math.cos(delta), math.sin(delta)
                        for k, p in enumerate(comps[ref].pads):
                            ox, oy = pad_offsets[k]
                            p.pos = Point(
                                comps[ref].pos.x + ox * cos_d + oy * sin_d,
                                comps[ref].pos.y - ox * sin_d + oy * cos_d,
                            )
                        comps[ref].rotation = rot
                        rscore = self._score_rotation_for_routing(
                            temp_state, comps[ref]
                        )
                        if rscore > best_rscore:
                            best_rscore = rscore
                            best_rot = rot
                    # Apply best rotation
                    delta = math.radians(best_rot - orig_rot)
                    cos_d, sin_d = math.cos(delta), math.sin(delta)
                    for k, p in enumerate(comps[ref].pads):
                        ox, oy = pad_offsets[k]
                        p.pos = Point(
                            comps[ref].pos.x + ox * cos_d + oy * sin_d,
                            comps[ref].pos.y - ox * sin_d + oy * cos_d,
                        )
                    comps[ref].rotation = best_rot

                placed_this_cluster.add(ref)

    def _optimize_intra_cluster(
        self,
        comps: dict[str, Component],
        clusters: list[set[str]],
        conn_graph: AdjacencyGraph,
    ):
        """Run a short force-directed pass within each cluster independently.

        This arranges components within functional groups (e.g. charger IC
        with its caps and resistors) before the global layout decides
        where groups go relative to each other.
        """
        tl, br = self.state.board_outline
        for cluster in clusters:
            unlocked = [r for r in cluster if not comps[r].locked]
            if len(unlocked) < 2:
                continue

            # Compute cluster centroid
            sum(comps[r].pos.x for r in unlocked) / len(unlocked)
            sum(comps[r].pos.y for r in unlocked) / len(unlocked)

            # Mini force-directed loop: attract connected, repel overlapping
            damping = 1.0
            for _ in range(self.intra_cluster_iters):
                forces = {r: Point(0, 0) for r in unlocked}

                # Attract connected pairs within cluster
                for i, ra in enumerate(unlocked):
                    for rb in unlocked[i + 1 :]:
                        w = conn_graph.weight(ra, rb)
                        if w <= 0:
                            continue
                        a, b = comps[ra], comps[rb]
                        d = max(a.pos.dist(b.pos), 0.1)
                        # Pull together proportional to distance and weight
                        f = self.k_attract * w * d
                        dx = (b.pos.x - a.pos.x) / d * f
                        dy = (b.pos.y - a.pos.y) / d * f
                        forces[ra].x += dx
                        forces[ra].y += dy
                        forces[rb].x -= dx
                        forces[rb].y -= dy

                # Repel overlapping bboxes
                for i, ra in enumerate(unlocked):
                    for rb in unlocked[i + 1 :]:
                        a, b = comps[ra], comps[rb]
                        overlap = _bbox_overlap_amount(a, b)
                        if overlap <= 0:
                            continue
                        d = max(a.pos.dist(b.pos), 0.1)
                        f = 3.0 * math.sqrt(overlap)
                        dx = (a.pos.x - b.pos.x) / d * f
                        dy = (a.pos.y - b.pos.y) / d * f
                        forces[ra].x += dx
                        forces[ra].y += dy
                        forces[rb].x -= dx
                        forces[rb].y -= dy

                # Apply forces
                for r in unlocked:
                    dx = forces[r].x * damping
                    dy = forces[r].y * damping
                    mag = math.hypot(dx, dy)
                    max_step = 1.5 * damping
                    if mag > max_step:
                        dx *= max_step / mag
                        dy *= max_step / mag

                    old_pos = Point(comps[r].pos.x, comps[r].pos.y)
                    comps[r].pos.x += dx
                    comps[r].pos.y += dy
                    # Clamp to board
                    hw, hh = comps[r].width_mm / 2, comps[r].height_mm / 2
                    comps[r].pos.x = max(
                        tl.x + hw + 1.0, min(br.x - hw - 1.0, comps[r].pos.x)
                    )
                    comps[r].pos.y = max(
                        tl.y + hh + 1.0, min(br.y - hh - 1.0, comps[r].pos.y)
                    )
                    _update_pad_positions(comps[r], old_pos, comps[r].rotation)

                damping *= 0.95

        print(f"  Intra-cluster optimization done ({len(clusters)} clusters)")

    def _optimize_rotations(self, comps: dict[str, Component], work_state: BoardState):
        """Try 0/90/180/270 rotations - optimize for routing (low crossovers + accessible pads)."""
        work_state.components = comps

        for ref, comp in comps.items():
            if comp.locked or comp.kind == "mounting_hole":
                continue
            # Skip edge-pinned connectors — rotation set by _best_rotation_for_edge
            if ref in self._pinned_targets:
                continue
            if len(comp.pads) < 2:
                continue

            # Store pad offsets relative to component center
            pad_offsets = []
            for p in comp.pads:
                pad_offsets.append((p.pos.x - comp.pos.x, p.pos.y - comp.pos.y))

            orig_rot = comp.rotation
            best_rot = orig_rot
            best_score = self._score_rotation_for_routing(work_state, comp)

            for rot in [0, 90, 180, 270]:
                if rot == orig_rot:
                    continue
                # Apply rotation: rotate pad offsets by (rot - orig_rot)
                # using KiCad convention (cos+sin, -sin+cos)
                delta = math.radians(rot - orig_rot)
                cos_d, sin_d = math.cos(delta), math.sin(delta)
                for i, p in enumerate(comp.pads):
                    ox, oy = pad_offsets[i]
                    p.pos = Point(
                        comp.pos.x + ox * cos_d + oy * sin_d,
                        comp.pos.y - ox * sin_d + oy * cos_d,
                    )
                comp.rotation = rot

                rot_score = self._score_rotation_for_routing(work_state, comp)
                if rot_score > best_score:
                    best_score = rot_score
                    best_rot = rot

            # Apply best rotation
            if best_rot != orig_rot:
                delta = math.radians(best_rot - orig_rot)
            else:
                delta = 0.0
            cos_d, sin_d = math.cos(delta), math.sin(delta)
            for i, p in enumerate(comp.pads):
                ox, oy = pad_offsets[i]
                p.pos = Point(
                    comp.pos.x + ox * cos_d + oy * sin_d,
                    comp.pos.y - ox * sin_d + oy * cos_d,
                )
            comp.rotation = best_rot

    def _force_step(
        self, comps: dict[str, Component], conn_graph: AdjacencyGraph, damping: float
    ) -> float:
        """One iteration of force-directed simulation. Returns max displacement.

        Uses numpy-accelerated repulsion when available, otherwise falls back
        to pure Python pairwise computation.
        """
        # State dedup: skip if we've seen this exact layout before
        state_h = hash(
            tuple(
                (r, round(comps[r].pos.x, 2), round(comps[r].pos.y, 2))
                for r in sorted(comps.keys())
            )
        )
        if state_h in self._seen_force_states:
            return 0.01  # signal convergence
        self._seen_force_states.add(state_h)

        tl, br = self.state.board_outline
        forces: dict[str, Point] = {ref: Point(0, 0) for ref in comps}
        refs = [r for r in comps if not comps[r].locked]

        # Accumulate all force contributions
        self._accumulate_attraction(comps, refs, forces, conn_graph)
        if _HAS_NUMPY:
            self._accumulate_repulsion_numpy(comps, forces)
        else:
            self._accumulate_repulsion_python(comps, forces)
        self._accumulate_smt_opposite_tht_force(comps, refs, forces)
        self._accumulate_boundary_force(comps, refs, forces, tl, br)
        self._accumulate_center_attraction(comps, refs, forces, tl, br)
        self._accumulate_alignment_force(comps, forces)

        # Integrate and clamp
        max_disp = self._apply_forces(comps, refs, forces, damping, tl, br)
        self._post_step_clamp(comps, refs)

        return max_disp

    def _sa_refine(
        self,
        comps: dict,
        work_state,
        scorer,
        *,
        max_iters: int = 1000,
        init_temp: float = 5.0,
        cooling_rate: float = 0.995,
        move_radius: float = 2.0,
        swap_prob: float = 0.3,
        rotation_prob: float = 0.2,
    ) -> dict:
        """Simulated annealing refinement after force-directed placement.

        Performs single-component moves, pairwise swaps, and rotation
        perturbations with Metropolis acceptance criterion to escape
        local minima found by the force-directed solver.
        """
        import copy
        import math
        import random

        rng = random.Random(self.seed + 9999)

        # Score current state
        work_state.components = comps
        current_score = scorer.score().total
        best_score = current_score
        best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}

        # Get unlocked component refs
        unlocked = [r for r, c in comps.items() if not c.locked]
        if not unlocked:
            return best_comps

        # Board bounds for clamping
        tl = work_state.board_outline[0]
        br = work_state.board_outline[1]

        temp = init_temp
        accepted = 0
        improved = 0

        for iteration in range(max_iters):
            # Choose move type
            roll = rng.random()
            if roll < swap_prob and len(unlocked) >= 2:
                # Pairwise swap
                ref_a, ref_b = rng.sample(unlocked, 2)
                comp_a = comps[ref_a]
                comp_b = comps[ref_b]

                # Save old positions
                old_a = Point(comp_a.pos.x, comp_a.pos.y)
                old_b = Point(comp_b.pos.x, comp_b.pos.y)

                # Swap positions
                comp_a.pos = Point(old_b.x, old_b.y)
                comp_b.pos = Point(old_a.x, old_a.y)
                _update_pad_positions(comp_a, old_a, comp_a.rotation)
                _update_pad_positions(comp_b, old_b, comp_b.rotation)

                # Evaluate
                work_state.components = comps
                new_score = scorer.score().total
                delta = new_score - current_score

                if delta > 0 or rng.random() < math.exp(delta / max(temp, 0.001)):
                    current_score = new_score
                    accepted += 1
                    if new_score > best_score:
                        best_score = new_score
                        best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}
                        improved += 1
                else:
                    # Revert swap
                    comp_a.pos = Point(old_a.x, old_a.y)
                    comp_b.pos = Point(old_b.x, old_b.y)
                    _update_pad_positions(comp_a, old_b, comp_a.rotation)
                    _update_pad_positions(comp_b, old_a, comp_b.rotation)

            elif roll < swap_prob + rotation_prob:
                # Rotation perturbation
                ref = rng.choice(unlocked)
                comp = comps[ref]
                old_rot = comp.rotation
                # Try 90-degree rotation increments
                new_rot = (old_rot + rng.choice([90.0, 180.0, 270.0])) % 360.0
                old_pos = Point(comp.pos.x, comp.pos.y)
                comp.rotation = new_rot
                _update_pad_positions(comp, old_pos, old_rot)

                work_state.components = comps
                new_score = scorer.score().total
                delta = new_score - current_score

                if delta > 0 or rng.random() < math.exp(delta / max(temp, 0.001)):
                    current_score = new_score
                    accepted += 1
                    if new_score > best_score:
                        best_score = new_score
                        best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}
                        improved += 1
                else:
                    # Revert rotation
                    comp.rotation = old_rot
                    _update_pad_positions(comp, old_pos, new_rot)

            else:
                # Single component displacement
                ref = rng.choice(unlocked)
                comp = comps[ref]
                old_pos = Point(comp.pos.x, comp.pos.y)

                # Random displacement within move_radius
                dx = rng.gauss(0, move_radius * 0.5)
                dy = rng.gauss(0, move_radius * 0.5)
                new_x = max(tl.x, min(br.x, comp.pos.x + dx))
                new_y = max(tl.y, min(br.y, comp.pos.y + dy))
                comp.pos = Point(new_x, new_y)
                _update_pad_positions(comp, old_pos, comp.rotation)

                work_state.components = comps
                new_score = scorer.score().total
                delta = new_score - current_score

                if delta > 0 or rng.random() < math.exp(delta / max(temp, 0.001)):
                    current_score = new_score
                    accepted += 1
                    if new_score > best_score:
                        best_score = new_score
                        best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}
                        improved += 1
                else:
                    # Revert displacement
                    comp.pos = Point(old_pos.x, old_pos.y)
                    _update_pad_positions(comp, Point(new_x, new_y), comp.rotation)

            # Cool down
            temp *= cooling_rate

        if improved > 0:
            print(f"  SA refine: {improved} improvements, {accepted} accepted of {max_iters} "
                  f"(best {best_score:.1f} vs initial {current_score:.1f})")
        else:
            print(f"  SA refine: no improvement after {max_iters} iterations")

        return best_comps

    def _accumulate_attraction(
        self,
        comps: dict[str, Component],
        refs: list[str],
        forces: dict[str, Point],
        conn_graph: AdjacencyGraph,
    ) -> None:
        """Attraction: pull connected components together."""
        for ref in refs:
            for nbr, weight in conn_graph.neighbors(ref).items():
                if nbr not in comps:
                    continue
                a = comps[ref]
                b = comps[nbr]
                d = a.pos.dist(b.pos)
                if d < 0.1:
                    continue
                # Target distance based on component sizes
                target = (a.width_mm + b.width_mm) / 2 + self.clearance
                f_mag = self.k_attract * weight * (d - target)
                angle = math.atan2(b.pos.y - a.pos.y, b.pos.x - a.pos.x)
                forces[ref].x += f_mag * math.cos(angle)
                forces[ref].y += f_mag * math.sin(angle)

    def _accumulate_repulsion_python(
        self,
        comps: dict[str, Component],
        forces: dict[str, Point],
    ) -> None:
        """Repulsion (pure Python): push overlapping/close components apart.

        Locked components (connectors, holes) act as repellers even though
        they don't move — this keeps unlocked parts from clustering against them.
        """
        ref_list = list(comps.keys())
        for i in range(len(ref_list)):
            a = comps[ref_list[i]]
            for j in range(i + 1, len(ref_list)):
                b = comps[ref_list[j]]
                if a.locked and b.locked:
                    continue  # both fixed, nothing to do
                d = a.pos.dist(b.pos)
                min_dist = (
                    max(a.width_mm, a.height_mm) + max(b.width_mm, b.height_mm)
                ) / 2 + self.clearance
                if d > min_dist * 2:
                    continue  # too far to matter
                if d < 0.1:
                    d = 0.1
                f_mag = self.k_repel * (a.area * b.area) / (d * d)
                angle = math.atan2(a.pos.y - b.pos.y, a.pos.x - b.pos.x)
                fx = f_mag * math.cos(angle)
                fy = f_mag * math.sin(angle)
                if not a.locked:
                    forces[ref_list[i]].x += fx
                    forces[ref_list[i]].y += fy
                if not b.locked:
                    forces[ref_list[j]].x -= fx
                    forces[ref_list[j]].y -= fy

    def _accumulate_repulsion_numpy(
        self,
        comps: dict[str, Component],
        forces: dict[str, Point],
    ) -> None:
        """Repulsion (numpy-accelerated): push overlapping/close components apart.

        Locked components (connectors, holes) act as repellers even though
        they don't move — this keeps unlocked parts from clustering against them.
        """
        ref_list = list(comps.keys())

        pos_x = np.array([comps[r].pos.x for r in ref_list], dtype=np.float64)
        pos_y = np.array([comps[r].pos.y for r in ref_list], dtype=np.float64)
        areas = np.array([comps[r].area for r in ref_list], dtype=np.float64)
        widths = np.array([comps[r].width_mm for r in ref_list], dtype=np.float64)
        heights = np.array([comps[r].height_mm for r in ref_list], dtype=np.float64)
        locked = np.array([comps[r].locked for r in ref_list], dtype=bool)

        max_dims = np.maximum(widths, heights)
        min_dists = (
            max_dims[:, np.newaxis] + max_dims[np.newaxis, :]
        ) / 2 + self.clearance

        dx = pos_x[:, np.newaxis] - pos_x[np.newaxis, :]
        dy = pos_y[:, np.newaxis] - pos_y[np.newaxis, :]
        dists = np.sqrt(dx * dx + dy * dy)

        skip_mask = (dists > min_dists * 2) | (dists < 0.001)

        force_mags = (
            self.k_repel
            * (areas[:, np.newaxis] * areas[np.newaxis, :])
            / (dists * dists + 0.01)
        )
        np.fill_diagonal(force_mags, 0)
        force_mags = np.where(skip_mask, 0, force_mags)

        safe_dists = np.where(dists > 0.1, dists, 0.1)
        norm_dx = dx / safe_dists
        norm_dy = dy / safe_dists

        fx_matrix = force_mags * norm_dx
        fy_matrix = force_mags * norm_dy

        both_locked = locked[:, np.newaxis] & locked[np.newaxis, :]
        np.fill_diagonal(both_locked, False)

        fx_matrix = np.where(both_locked, 0, fx_matrix)
        fy_matrix = np.where(both_locked, 0, fy_matrix)

        fx_totals = fx_matrix.sum(axis=1)
        fy_totals = fy_matrix.sum(axis=1)

        for i, ref in enumerate(ref_list):
            if not comps[ref].locked:
                forces[ref].x += float(fx_totals[i])
                forces[ref].y += float(fy_totals[i])

    def _accumulate_smt_opposite_tht_force(
        self,
        comps: dict[str, Component],
        refs: list[str],
        forces: dict[str, Point],
    ) -> None:
        """SMT-opposite-THT attraction: pull unlocked SMT components toward
        the nearest point on the nearest back-layer THT bounding box.

        This distributes SMT across the available THT courtyard space
        rather than clustering them all at the centroid.
        """
        if not self.cfg.get("smt_opposite_tht", True):
            return
        back_tht = [
            c for c in comps.values() if c.is_through_hole and c.layer == Layer.BACK
        ]
        if not back_tht:
            return
        smt_k = self.k_attract * 0.6
        # Pre-compute back-THT bboxes
        btht_bboxes = [
            (
                t.pos.x - t.width_mm / 2,
                t.pos.y - t.height_mm / 2,
                t.pos.x + t.width_mm / 2,
                t.pos.y + t.height_mm / 2,
            )
            for t in back_tht
        ]
        for ref in refs:
            c = comps[ref]
            if c.is_through_hole or c.layer == Layer.BACK:
                continue
            # Find nearest point on nearest back-THT bbox
            best_dist = float("inf")
            best_tx, best_ty = c.pos.x, c.pos.y
            for bx0, by0, bx1, by1 in btht_bboxes:
                # Clamp SMT center to THT bbox = nearest point on bbox
                nx = max(bx0, min(bx1, c.pos.x))
                ny = max(by0, min(by1, c.pos.y))
                nd = math.hypot(c.pos.x - nx, c.pos.y - ny)
                if nd < best_dist:
                    best_dist = nd
                    best_tx, best_ty = nx, ny
            if best_dist < 0.1:
                continue
            f_mag = smt_k * best_dist
            angle = math.atan2(best_ty - c.pos.y, best_tx - c.pos.x)
            forces[ref].x += f_mag * math.cos(angle)
            forces[ref].y += f_mag * math.sin(angle)

    def _accumulate_boundary_force(
        self,
        comps: dict[str, Component],
        refs: list[str],
        forces: dict[str, Point],
        tl: Point,
        br: Point,
    ) -> None:
        """Boundary: strong spring force at edges (pad-aware extents)."""
        margin = self.edge_margin + 2.0
        k_boundary = 10.0
        for ref in refs:
            c = comps[ref]
            hw, hh = _pad_half_extents(c)
            if c.pos.x - hw < tl.x + margin:
                forces[ref].x += k_boundary * (tl.x + margin - (c.pos.x - hw))
            if c.pos.x + hw > br.x - margin:
                forces[ref].x -= k_boundary * ((c.pos.x + hw) - (br.x - margin))
            if c.pos.y - hh < tl.y + margin:
                forces[ref].y += k_boundary * (tl.y + margin - (c.pos.y - hh))
            if c.pos.y + hh > br.y - margin:
                forces[ref].y -= k_boundary * ((c.pos.y + hh) - (br.y - margin))

    def _accumulate_center_attraction(
        self,
        comps: dict[str, Component],
        refs: list[str],
        forces: dict[str, Point],
        tl: Point,
        br: Point,
    ) -> None:
        """Center attraction: weak force pulling components toward board center
        to prevent edge-clumping bias."""
        cx = (tl.x + br.x) / 2.0
        cy = (tl.y + br.y) / 2.0
        k_center = 0.02  # weak — just enough to break edge-clumping symmetry
        for ref in refs:
            c = comps[ref]
            dx = cx - c.pos.x
            dy = cy - c.pos.y
            dist = max(0.1, (dx * dx + dy * dy) ** 0.5)
            # Scale by distance from center — stronger pull for far-flung components
            strength = k_center * dist / max(1.0, (br.x - tl.x))
            forces[ref].x += strength * dx
            forces[ref].y += strength * dy

    def _accumulate_alignment_force(
        self,
        comps: dict[str, Component],
        forces: dict[str, Point],
    ) -> None:
        """Large-pair alignment: keep paired components sharing an axis."""
        if not self._aligned_pairs:
            return
        for ref_a, ref_b, axis in self._aligned_pairs:
            if ref_a not in comps or ref_b not in comps:
                continue
            a, b = comps[ref_a], comps[ref_b]
            if axis == "y":  # horizontal side-by-side: share Y
                mid_y = (a.pos.y + b.pos.y) / 2
                if ref_a in forces:
                    forces[ref_a].y += 1.5 * (mid_y - a.pos.y)
                if ref_b in forces:
                    forces[ref_b].y += 1.5 * (mid_y - b.pos.y)
            else:  # vertical: share X
                mid_x = (a.pos.x + b.pos.x) / 2
                if ref_a in forces:
                    forces[ref_a].x += 1.5 * (mid_x - a.pos.x)
                if ref_b in forces:
                    forces[ref_b].x += 1.5 * (mid_x - b.pos.x)

    def _apply_forces(
        self,
        comps: dict[str, Component],
        refs: list[str],
        forces: dict[str, Point],
        damping: float,
        tl: Point,
        br: Point,
    ) -> float:
        """Apply accumulated forces with damping and displacement clamping.
        Returns max displacement."""
        max_disp = 0.0
        for ref in refs:
            dx = forces[ref].x * damping
            dy = forces[ref].y * damping
            # Clamp max displacement per step
            mag = math.hypot(dx, dy)
            max_step = 2.0 * damping
            if mag > max_step:
                dx *= max_step / mag
                dy *= max_step / mag
                mag = max_step

            old_pos = Point(comps[ref].pos.x, comps[ref].pos.y)
            old_rot = comps[ref].rotation
            comps[ref].pos.x += dx
            comps[ref].pos.y += dy

            # Hard clamp: pad-aware extents must stay inside board
            c = comps[ref]
            hw, hh = _pad_half_extents(c)
            c.pos.x = max(tl.x + hw + 1.0, min(br.x - hw - 1.0, c.pos.x))
            c.pos.y = max(tl.y + hh + 1.0, min(br.y - hh - 1.0, c.pos.y))

            _update_pad_positions(comps[ref], old_pos, old_rot)

            max_disp = max(max_disp, mag)

        return max_disp

    def _post_step_clamp(
        self,
        comps: dict[str, Component],
        refs: list[str],
    ) -> None:
        """Post-step: zone re-clamping and aligned-pair re-snapping.

        Keep zone-constrained components within their designated zone bounds
        (prevents drift during force simulation).
        """
        zones_cfg = self.cfg.get("component_zones", {})
        for ref in refs:
            zone_cfg = zones_cfg.get(ref, {})
            if "zone" not in zone_cfg:
                continue
            c = comps[ref]
            zx0, zy0, zx1, zy1 = self._get_zone_bounds(zone_cfg["zone"])
            hw, hh = _pad_half_extents(c)
            clamped_x = max(zx0 + hw, min(zx1 - hw, c.pos.x))
            clamped_y = max(zy0 + hh, min(zy1 - hh, c.pos.y))
            if abs(clamped_x - c.pos.x) > 0.01 or abs(clamped_y - c.pos.y) > 0.01:
                old_pos = Point(c.pos.x, c.pos.y)
                c.pos.x = clamped_x
                c.pos.y = clamped_y
                _update_pad_positions(c, old_pos, c.rotation)

        # Post-step: re-snap aligned pairs to shared coordinate
        self._re_snap_aligned_pairs(comps)

    def _resolve_overlaps(self, comps: dict[str, Component]):
        """Push apart components until no bboxes overlap (including clearance gap).

        For each overlapping pair, picks the escape direction that requires the
        least travel distance AND keeps the free component within board bounds.
        This handles edge cases where the shortest-axis push would send a component
        into a board edge (e.g. a small part trapped between a large locked battery
        holder and the board boundary).
        """
        refs = list(comps.keys())
        half_gap = self.clearance / 2.0
        tl, br = self.state.board_outline

        def _clamp_comp_to_board(
            comp: Component, nx: float, ny: float
        ) -> tuple[float, float]:
            hw, hh = _pad_half_extents(comp)
            return (
                max(tl.x + hw + 1.0, min(br.x - hw - 1.0, nx)),
                max(tl.y + hh + 1.0, min(br.y - hh - 1.0, ny)),
            )

        def _total_overlap_area_for(
            comp: Component, others: dict[str, Component]
        ) -> float:
            comp_tl, comp_br = _effective_bbox(comp, half_gap)
            total = 0.0
            for other in others.values():
                if other is comp:
                    continue
                other_tl, other_br = _effective_bbox(other, half_gap)
                _ox = min(comp_br.x, other_br.x) - max(comp_tl.x, other_tl.x)
                _oy = min(comp_br.y, other_br.y) - max(comp_tl.y, other_tl.y)
                if _ox > 0 and _oy > 0:
                    total += _ox * _oy
            return total

        def _escape(free_c: Component, lock_tl: Point, lock_br: Point) -> bool:
            """Push free_c fully out of lock bbox. Returns True if moved."""
            fc_tl, fc_br = _effective_bbox(free_c, half_gap)
            ox, oy = _bbox_overlap_xy(lock_tl, lock_br, fc_tl, fc_br)
            if ox <= 0 or oy <= 0:
                return False

            # Full-clearance distances: move so trailing edge of free_c
            # clears the leading edge of the lock bbox entirely.
            clear_right = lock_br.x - fc_tl.x + 0.1
            clear_left = fc_br.x - lock_tl.x + 0.1
            clear_down = lock_br.y - fc_tl.y + 0.1
            clear_up = fc_br.y - lock_tl.y + 0.1

            moves = [
                (clear_right, free_c.pos.x + clear_right, free_c.pos.y),
                (clear_left, free_c.pos.x - clear_left, free_c.pos.y),
                (clear_down, free_c.pos.x, free_c.pos.y + clear_down),
                (clear_up, free_c.pos.x, free_c.pos.y - clear_up),
            ]

            old = Point(free_c.pos.x, free_c.pos.y)
            old_overlap = _total_overlap_area_for(free_c, comps)
            best_key: tuple[float, int, float] | None = None
            best_move = (free_c.pos.x, free_c.pos.y)

            for travel, nx, ny in moves:
                nx_c, ny_c = _clamp_comp_to_board(free_c, nx, ny)
                clamped = abs(nx_c - nx) > 0.01 or abs(ny_c - ny) > 0.01

                free_c.pos.x, free_c.pos.y = nx_c, ny_c
                _update_pad_positions(free_c, old, free_c.rotation)
                new_overlap = _total_overlap_area_for(free_c, comps)
                improvement = old_overlap - new_overlap

                key = (-improvement, 1 if clamped else 0, travel)
                if best_key is None or key < best_key:
                    best_key = key
                    best_move = (nx_c, ny_c)

                free_c.pos.x, free_c.pos.y = old.x, old.y
                _update_pad_positions(free_c, Point(nx_c, ny_c), free_c.rotation)

            nx, ny = best_move
            free_c.pos.x, free_c.pos.y = nx, ny
            _update_pad_positions(free_c, old, free_c.rotation)
            return abs(nx - old.x) > 0.01 or abs(ny - old.y) > 0.01

        for iteration in range(300):
            moved = False

            # --- Pass 1: resolve free-free overlaps first ---
            for i in range(len(refs)):
                a = comps[refs[i]]
                if a.locked:
                    continue
                a_tl, a_br = _effective_bbox(a, half_gap)
                for j in range(i + 1, len(refs)):
                    b = comps[refs[j]]
                    if b.locked:
                        continue

                    b_tl, b_br = _effective_bbox(b, half_gap)
                    ox, oy = _bbox_overlap_xy(a_tl, a_br, b_tl, b_br)
                    if ox <= 0 or oy <= 0:
                        continue

                    hw_a, hh_a = _pad_half_extents(a)
                    hw_b, hh_b = _pad_half_extents(b)
                    if ox < oy:
                        push = (ox + 0.1) / 2
                        sign = 1.0 if a.pos.x >= b.pos.x else -1.0
                        old_a = Point(a.pos.x, a.pos.y)
                        old_b = Point(b.pos.x, b.pos.y)
                        a.pos.x = max(
                            tl.x + hw_a + 1.0,
                            min(br.x - hw_a - 1.0, a.pos.x + sign * push),
                        )
                        b.pos.x = max(
                            tl.x + hw_b + 1.0,
                            min(br.x - hw_b - 1.0, b.pos.x - sign * push),
                        )
                    else:
                        push = (oy + 0.1) / 2
                        sign = 1.0 if a.pos.y >= b.pos.y else -1.0
                        old_a = Point(a.pos.x, a.pos.y)
                        old_b = Point(b.pos.x, b.pos.y)
                        a.pos.y = max(
                            tl.y + hh_a + 1.0,
                            min(br.y - hh_a - 1.0, a.pos.y + sign * push),
                        )
                        b.pos.y = max(
                            tl.y + hh_b + 1.0,
                            min(br.y - hh_b - 1.0, b.pos.y - sign * push),
                        )
                    _update_pad_positions(a, old_a, a.rotation)
                    _update_pad_positions(b, old_b, b.rotation)
                    a_tl, a_br = _effective_bbox(a, half_gap)
                    moved = True

            # --- Pass 2: resolve locked-involving overlaps (escape) ---
            for i in range(len(refs)):
                a = comps[refs[i]]
                a_tl, a_br = _effective_bbox(a, half_gap)
                for j in range(i + 1, len(refs)):
                    b = comps[refs[j]]
                    if not a.locked and not b.locked:
                        continue  # already handled in pass 1

                    b_tl, b_br = _effective_bbox(b, half_gap)
                    ox, oy = _bbox_overlap_xy(a_tl, a_br, b_tl, b_br)
                    if ox <= 0 or oy <= 0:
                        continue

                    if a.locked and b.locked:
                        zones = self.cfg.get("component_zones", {})
                        a_pinned = refs[i] in zones and (
                            "edge" in zones[refs[i]] or "corner" in zones[refs[i]]
                        )
                        b_pinned = refs[j] in zones and (
                            "edge" in zones[refs[j]] or "corner" in zones[refs[j]]
                        )
                        if a_pinned and not b_pinned:
                            if _escape(b, a_tl, a_br):
                                b_tl, b_br = _effective_bbox(b, half_gap)
                                moved = True
                        elif b_pinned and not a_pinned:
                            if _escape(a, b_tl, b_br):
                                a_tl, a_br = _effective_bbox(a, half_gap)
                                moved = True
                        else:
                            a_area = a.width_mm * a.height_mm
                            b_area = b.width_mm * b.height_mm
                            if a_area <= b_area:
                                if _escape(a, b_tl, b_br):
                                    a_tl, a_br = _effective_bbox(a, half_gap)
                                    moved = True
                            else:
                                if _escape(b, a_tl, a_br):
                                    b_tl, b_br = _effective_bbox(b, half_gap)
                                    moved = True
                    elif a.locked:
                        if _escape(b, a_tl, a_br):
                            b_tl, b_br = _effective_bbox(b, half_gap)
                            moved = True
                    elif b.locked:
                        if _escape(a, b_tl, b_br):
                            a_tl, a_br = _effective_bbox(a, half_gap)
                            moved = True

            if not moved:
                break  # fully separated

    def legality_diagnostics(self, comps: dict[str, Component]) -> dict[str, object]:
        tl, br = self.state.board_outline
        inset = self.cfg.get("pad_inset_margin_mm", 0.3)
        half_gap = self.clearance / 2.0
        pads_outside: list[dict[str, object]] = []
        overlaps: list[dict[str, object]] = []
        refs = list(comps.keys())
        for ref, comp in comps.items():
            for pad in comp.pads:
                violations: list[str] = []
                if pad.pos.x < tl.x + inset:
                    violations.append("left")
                if pad.pos.x > br.x - inset:
                    violations.append("right")
                if pad.pos.y < tl.y + inset:
                    violations.append("top")
                if pad.pos.y > br.y - inset:
                    violations.append("bottom")
                if violations:
                    pads_outside.append(
                        {
                            "ref": ref,
                            "pad_id": pad.pad_id,
                            "sides": violations,
                            "x_mm": round(pad.pos.x, 4),
                            "y_mm": round(pad.pos.y, 4),
                        }
                    )
        locked_overlap_count = 0
        for i in range(len(refs)):
            a = comps[refs[i]]
            a_tl, a_br = _effective_bbox(a, half_gap)
            for j in range(i + 1, len(refs)):
                b = comps[refs[j]]
                b_tl, b_br = _effective_bbox(b, half_gap)
                ox, oy = _bbox_overlap_xy(a_tl, a_br, b_tl, b_br)
                if ox > 0.0 and oy > 0.0:
                    involves_locked = a.locked or b.locked
                    if involves_locked:
                        locked_overlap_count += 1
                    overlaps.append(
                        {
                            "a": refs[i],
                            "b": refs[j],
                            "overlap_x_mm": round(ox, 4),
                            "overlap_y_mm": round(oy, 4),
                            "overlap_area_mm2": round(ox * oy, 4),
                            "involves_locked": involves_locked,
                        }
                    )
        return {
            "pads_outside_board": pads_outside,
            "overlaps": overlaps,
            "pad_outside_count": len(pads_outside),
            "overlap_count": len(overlaps),
            "locked_overlap_count": locked_overlap_count,
            "legal": not pads_outside and not overlaps,
        }

    def legalize_components(
        self, comps: dict[str, Component], *, max_passes: int = 12
    ) -> dict[str, object]:
        moved_refs: set[str] = set()
        if not hasattr(self, "_pinned_targets"):
            self._pinned_targets = {}
        best_snapshot = {ref: copy.deepcopy(comp) for ref, comp in comps.items()}
        best_diagnostics = self.legality_diagnostics(best_snapshot)

        def _diag_key(diag):
            locked = int(diag.get("locked_overlap_count", 0))
            free = int(diag["overlap_count"]) - locked
            pads = int(diag["pad_outside_count"])
            # Locked overlaps weigh 3x: they require escape pushes that
            # cascade into free-free overlaps, so regressing on them
            # is costlier than having temporary free-free overlaps.
            weighted = locked * 3 + free + pads
            return (weighted, locked, pads)

        def _move_component(comp, nx, ny):
            old_pos = Point(comp.pos.x, comp.pos.y)
            if abs(nx - old_pos.x) <= 0.01 and abs(ny - old_pos.y) <= 0.01:
                return False
            comp.pos.x = nx
            comp.pos.y = ny
            _update_pad_positions(comp, old_pos, comp.rotation)
            return True

        def _clamp_component_to_board(comp, nx, ny):
            tl, br = self.state.board_outline
            hw, hh = _pad_half_extents(comp)
            return (
                max(tl.x + hw + 1.0, min(br.x - hw - 1.0, nx)),
                max(tl.y + hh + 1.0, min(br.y - hh - 1.0, ny)),
            )

        def _keep_out_of_pinned_edge_connectors():
            zones = self.cfg.get("component_zones", {})
            half_gap = self.clearance / 2.0
            pinned_connectors = []
            for ref, comp in comps.items():
                zone_cfg = zones.get(ref, {})
                edge = zone_cfg.get("edge")
                if (
                    edge in {"left", "right", "top", "bottom"}
                    and comp.locked
                    and comp.kind == "connector"
                ):
                    keepout_tl, keepout_br = _effective_bbox(comp, half_gap)
                    pinned_connectors.append((ref, comp, edge, keepout_tl, keepout_br))
            if not pinned_connectors:
                return
            for ref, comp in comps.items():
                if comp.locked:
                    continue
                for _conn_ref, _conn, edge, keepout_tl, keepout_br in pinned_connectors:
                    comp_tl, comp_br = _effective_bbox(comp, half_gap)
                    ox, oy = _bbox_overlap_xy(keepout_tl, keepout_br, comp_tl, comp_br)
                    if ox <= 0.0 or oy <= 0.0:
                        continue
                    old_pos = Point(comp.pos.x, comp.pos.y)
                    candidates = []
                    if edge == "left":
                        candidates.append(
                            (keepout_br.x + (comp_br.x - comp.pos.x) + 0.1, comp.pos.y)
                        )
                    elif edge == "right":
                        candidates.append(
                            (keepout_tl.x - (comp.pos.x - comp_tl.x) - 0.1, comp.pos.y)
                        )
                    elif edge == "top":
                        candidates.append(
                            (comp.pos.x, keepout_br.y + (comp_br.y - comp.pos.y) + 0.1)
                        )
                    else:
                        candidates.append(
                            (comp.pos.x, keepout_tl.y - (comp.pos.y - comp_tl.y) - 0.1)
                        )
                    candidates.extend(
                        [
                            (comp.pos.x + ox + 0.1, comp.pos.y),
                            (comp.pos.x - ox - 0.1, comp.pos.y),
                            (comp.pos.x, comp.pos.y + oy + 0.1),
                            (comp.pos.x, comp.pos.y - oy - 0.1),
                        ]
                    )
                    best_key = None
                    best_move = (comp.pos.x, comp.pos.y)
                    for nx, ny in candidates:
                        nx, ny = _clamp_component_to_board(comp, nx, ny)
                        moved = _move_component(comp, nx, ny)
                        trial_tl, trial_br = _effective_bbox(comp, half_gap)
                        trial_ox, trial_oy = _bbox_overlap_xy(
                            keepout_tl, keepout_br, trial_tl, trial_br
                        )
                        still_overlapping = (
                            1 if trial_ox > 0.0 and trial_oy > 0.0 else 0
                        )
                        travel = old_pos.dist(Point(nx, ny))
                        key = (still_overlapping, travel)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_move = (nx, ny)
                        if moved:
                            _move_component(comp, old_pos.x, old_pos.y)
                    _move_component(comp, best_move[0], best_move[1])

        for _ in range(max_passes):
            before = {ref: (comp.pos.x, comp.pos.y) for ref, comp in comps.items()}
            self._clamp_pads_to_board(comps)
            self._clamp_to_board(comps)
            self._resolve_overlaps(comps)
            self._clamp_to_board(comps)
            self._clamp_pads_to_board(comps)
            self._restore_pinned_positions(comps)
            _keep_out_of_pinned_edge_connectors()
            # Resolve cascading overlaps from connector keepout pushes
            self._resolve_overlaps(comps)
            self._clamp_to_board(comps)
            self._clamp_pads_to_board(comps)
            diagnostics = self.legality_diagnostics(comps)
            if _diag_key(diagnostics) < _diag_key(best_diagnostics):
                best_snapshot = {
                    ref: copy.deepcopy(comp) for ref, comp in comps.items()
                }
                best_diagnostics = diagnostics
            for ref, comp in comps.items():
                old_x, old_y = before[ref]
                if abs(comp.pos.x - old_x) > 0.01 or abs(comp.pos.y - old_y) > 0.01:
                    moved_refs.add(ref)
            if diagnostics["legal"]:
                return {
                    "resolved": True,
                    "passes": _ + 1,
                    "moved_refs": sorted(moved_refs),
                    "diagnostics": diagnostics,
                }
        for ref in list(comps.keys()):
            comps[ref] = copy.deepcopy(best_snapshot[ref])
        return {
            "resolved": best_diagnostics.get("legal", False),
            "passes": max_passes,
            "moved_refs": sorted(moved_refs),
            "diagnostics": best_diagnostics,
        }

    def _re_snap_aligned_pairs(self, comps: dict[str, Component]):
        """Re-snap aligned pairs to shared coordinate after pipeline steps.

        Steps like swap optimization, grid snap, orderedness, and overlap
        resolution can break the alignment set up by _align_large_pairs().
        Call this after any such step to restore side-by-side alignment.
        """
        if not self._aligned_pairs:
            return
        for ref_a, ref_b, axis in self._aligned_pairs:
            if ref_a not in comps or ref_b not in comps:
                continue
            a, b = comps[ref_a], comps[ref_b]
            if axis == "y":
                mid_y = (a.pos.y + b.pos.y) / 2
                old_a = Point(a.pos.x, a.pos.y)
                old_b = Point(b.pos.x, b.pos.y)
                a.pos.y = mid_y
                b.pos.y = mid_y
                _update_pad_positions(a, old_a, a.rotation)
                _update_pad_positions(b, old_b, b.rotation)
            else:
                mid_x = (a.pos.x + b.pos.x) / 2
                old_a = Point(a.pos.x, a.pos.y)
                old_b = Point(b.pos.x, b.pos.y)
                a.pos.x = mid_x
                b.pos.x = mid_x
                _update_pad_positions(a, old_a, a.rotation)
                _update_pad_positions(b, old_b, b.rotation)

    def _clamp_to_board(self, comps: dict[str, Component]):
        """Hard clamp: force every component's bounding box inside the board.

        Uses pad-aware half-extents so that components with pads extending
        beyond the body (e.g. battery holders) are clamped correctly.
        """
        tl, br = self.state.board_outline
        for comp in comps.values():
            if comp.locked:
                continue
            hw, hh = _pad_half_extents(comp)
            old_pos = Point(comp.pos.x, comp.pos.y)
            comp.pos.x = max(tl.x + hw + 1.0, min(br.x - hw - 1.0, comp.pos.x))
            comp.pos.y = max(tl.y + hh + 1.0, min(br.y - hh - 1.0, comp.pos.y))
            if comp.pos.x != old_pos.x or comp.pos.y != old_pos.y:
                _update_pad_positions(comp, old_pos, comp.rotation)

    def _assign_layers(self, comps: dict[str, Component]):
        """Assign large through-hole components to B.Cu (back layer).

        SMT components always stay on F.Cu.  Small THT passives (e.g. axial
        resistors) also stay on F.Cu.  Large THT parts (batteries,
        large connectors) go to back so they don't block SMT placement
        and routing on the front side.

        SMT passives stay on F.Cu even when their IC group contains a
        back-layer THT component — IC group connectivity forces keep them
        nearby in the same XY region, achieving dual-sided board usage.
        """
        min_area = self.cfg.get("tht_backside_min_area_mm2", 50.0)
        moved = []
        for ref, comp in comps.items():
            if not comp.is_through_hole:
                continue
            if comp.area < min_area:
                continue
            if comp.layer != Layer.BACK:
                # Mirror pad X offsets to match KiCad Flip() behavior:
                # Flip negates absolute X offset from component center
                for pad in comp.pads:
                    pad.pos.x = 2 * comp.pos.x - pad.pos.x
                comp.layer = Layer.BACK
                moved.append(ref)
        if moved:
            print(
                f"  Assigned {len(moved)} large THT component(s) to back layer: "
                f"{', '.join(moved)}"
            )

    def _align_large_pairs(self, comps: dict[str, Component]):
        """Detect and align pairs of large, similarly-sized components side-by-side.

        Finds components with same kind (not passive/misc), similar area
        (ratio > 0.85), and area above tht_backside_min_area_mm2.  Places
        them adjacent on a randomly chosen axis (horizontal or vertical),
        sharing one coordinate.  Respects zone constraints.

        Populates self._aligned_pairs for use by _force_step() to maintain
        alignment during force simulation.
        """
        if not self.cfg.get("align_large_pairs", True):
            return

        min_area = self.cfg.get("tht_backside_min_area_mm2", 50.0)
        zones = self.cfg.get("component_zones", {})
        tl, br = self.state.board_outline

        # Find candidates: large, non-passive, non-misc
        candidates = [
            (ref, comp)
            for ref, comp in comps.items()
            if comp.area >= min_area
            and comp.kind not in ("", "misc", "passive", "connector", "mounting_hole")
        ]

        # Detect pairs: same kind, similar area
        paired = set()
        pairs = []
        for i, (ref_a, a) in enumerate(candidates):
            if ref_a in paired:
                continue
            for ref_b, b in candidates[i + 1 :]:
                if ref_b in paired:
                    continue
                if a.kind != b.kind:
                    continue
                ratio = min(a.area, b.area) / max(a.area, b.area)
                if ratio < 0.85:
                    continue
                pairs.append((ref_a, ref_b))
                paired.add(ref_a)
                paired.add(ref_b)
                break  # one partner per component

        if not pairs:
            return

        gap = 1.5  # mm gap between paired components

        for ref_a, ref_b in pairs:
            a, b = comps[ref_a], comps[ref_b]

            # Choose axis based on component shape: place along the longer
            # dimension to minimize total footprint width
            if max(a.width_mm, b.width_mm) >= max(a.height_mm, b.height_mm):
                axis = "y"  # side-by-side horizontally (share Y)
            else:
                axis = "x"  # stacked vertically (share X)

            # Compute zone bounds for clamping (use first component's zone)
            zone_a = zones.get(ref_a, {})
            zone_b = zones.get(ref_b, {})
            zone_name = zone_a.get("zone") or zone_b.get("zone")
            if zone_name:
                zx0, zy0, zx1, zy1 = self._get_zone_bounds(zone_name)
            else:
                margin = self.edge_margin
                zx0, zy0 = tl.x + margin, tl.y + margin
                zx1, zy1 = br.x - margin, br.y - margin

            old_a = Point(a.pos.x, a.pos.y)
            old_b = Point(b.pos.x, b.pos.y)

            if axis == "y":
                # Horizontal side-by-side: same Y, adjacent X
                mid_y = (a.pos.y + b.pos.y) / 2
                total_w = a.width_mm + b.width_mm + gap
                # Center the pair in their zone on X
                pair_cx = self.rng.uniform(
                    zx0 + total_w / 2,
                    max(zx0 + total_w / 2 + 1, zx1 - total_w / 2),
                )
                a.pos.x = pair_cx - (b.width_mm + gap) / 2
                b.pos.x = pair_cx + (a.width_mm + gap) / 2
                # Clamp Y to zone
                mid_y = max(
                    zy0 + max(a.height_mm, b.height_mm) / 2,
                    min(zy1 - max(a.height_mm, b.height_mm) / 2, mid_y),
                )
                a.pos.y = mid_y
                b.pos.y = mid_y
            else:
                # Vertical stack: same X, adjacent Y
                mid_x = (a.pos.x + b.pos.x) / 2
                total_h = a.height_mm + b.height_mm + gap
                pair_cy = self.rng.uniform(
                    zy0 + total_h / 2,
                    max(zy0 + total_h / 2 + 1, zy1 - total_h / 2),
                )
                a.pos.y = pair_cy - (b.height_mm + gap) / 2
                b.pos.y = pair_cy + (a.height_mm + gap) / 2
                mid_x = max(
                    zx0 + max(a.width_mm, b.width_mm) / 2,
                    min(zx1 - max(a.width_mm, b.width_mm) / 2, mid_x),
                )
                a.pos.x = mid_x
                b.pos.x = mid_x

            _update_pad_positions(a, old_a, a.rotation)
            _update_pad_positions(b, old_b, b.rotation)

            self._aligned_pairs.append((ref_a, ref_b, axis))

        if pairs:
            print(
                f"  Aligned {len(pairs)} large pair(s) side-by-side: "
                f"{', '.join(f'{a}+{b}' for a, b in pairs)}"
            )

    def _clamp_pads_to_board(self, comps: dict[str, Component]):
        """Hard clamp: shift components inward so all pads are inside the board."""
        tl, br = self.state.board_outline
        inset = self.cfg.get("pad_inset_margin_mm", 0.3)
        min_x = tl.x + inset
        min_y = tl.y + inset
        max_x = br.x - inset
        max_y = br.y - inset

        for comp in comps.values():
            if not comp.pads:
                continue
            if comp.locked:
                continue

            # Track left/right and top/bottom violations separately
            shift_left = 0.0  # positive = need to move right
            shift_right = 0.0  # negative = need to move left
            shift_up = 0.0  # positive = need to move down
            shift_down = 0.0  # negative = need to move up
            for pad in comp.pads:
                if pad.pos.x < min_x:
                    shift_left = max(shift_left, min_x - pad.pos.x)
                if pad.pos.x > max_x:
                    shift_right = min(shift_right, max_x - pad.pos.x)
                if pad.pos.y < min_y:
                    shift_up = max(shift_up, min_y - pad.pos.y)
                if pad.pos.y > max_y:
                    shift_down = min(shift_down, max_y - pad.pos.y)

            # Use the larger magnitude violation for each axis
            shift_x = shift_left if abs(shift_left) >= abs(shift_right) else shift_right
            shift_y = shift_up if abs(shift_up) >= abs(shift_down) else shift_down

            if abs(shift_x) > 0.001 or abs(shift_y) > 0.001:
                old_pos = Point(comp.pos.x, comp.pos.y)
                comp.pos.x += shift_x
                comp.pos.y += shift_y
                _update_pad_positions(comp, old_pos, comp.rotation)

    def _snap_to_grid(self, comps: dict[str, Component]):
        """Snap all unlocked components to placement grid."""
        g = self.grid_snap
        for comp in comps.values():
            if comp.locked:
                continue
            old_pos = Point(comp.pos.x, comp.pos.y)
            comp.pos.x = round(comp.pos.x / g) * g
            comp.pos.y = round(comp.pos.y / g) * g
            _update_pad_positions(comp, old_pos, comp.rotation)

    def _apply_orderedness(self, comps: dict[str, Component], strength: float):
        """Align passives into neat rows/columns near their IC group leader.

        strength: 0.0 = no effect (organic), 1.0 = full grid alignment.
        Intermediate values blend between organic position and grid position.

        Groups passives by IC group, sorts them by size class, and arranges
        each size class into rows. Components not in any IC group are grouped
        by spatial proximity.
        """
        ic_groups = self.cfg.get("ic_groups", {})
        grid = self.grid_snap

        # Build map: ref -> group leader
        ref_to_leader: dict[str, str] = {}
        for leader, members in ic_groups.items():
            ref_to_leader[leader] = leader
            for m in members:
                ref_to_leader[m] = leader

        # Collect passives by group leader
        grouped: dict[str, list[str]] = {}
        ungrouped: list[str] = []
        for ref, comp in comps.items():
            if comp.locked or comp.kind not in ("passive",):
                continue
            leader = ref_to_leader.get(ref)
            if leader and leader in comps:
                grouped.setdefault(leader, []).append(ref)
            else:
                ungrouped.append(ref)

        # Cluster ungrouped passives by proximity (simple greedy clustering)
        if ungrouped:
            remaining = set(ungrouped)
            cluster_radius = 20.0  # mm
            while remaining:
                seed = remaining.pop()
                cluster = [seed]
                for ref in list(remaining):
                    if comps[ref].pos.dist(comps[seed].pos) < cluster_radius:
                        cluster.append(ref)
                        remaining.discard(ref)
                if len(cluster) >= 2:
                    # Use first component as virtual "leader"
                    grouped[cluster[0]] = cluster

        total_aligned = 0
        for leader, members in grouped.items():
            if len(members) < 2:
                continue

            # Find anchor position: IC leader center or centroid of group
            if leader in comps and leader not in members:
                anchor = comps[leader].pos
            else:
                anchor = Point(
                    sum(comps[r].pos.x for r in members) / len(members),
                    sum(comps[r].pos.y for r in members) / len(members),
                )

            # Bin passives by size class (similar dimensions → same row)
            size_bins: dict[tuple[float, float], list[str]] = {}
            for ref in members:
                c = comps[ref]
                # Round dimensions to nearest 0.5mm for binning
                w_key = round(min(c.width_mm, c.height_mm) * 2) / 2
                h_key = round(max(c.width_mm, c.height_mm) * 2) / 2
                size_bins.setdefault((w_key, h_key), []).append(ref)

            # Arrange each size bin as a row
            row_y_offset = 0.0
            for (w_key, h_key), bin_refs in size_bins.items():
                if not bin_refs:
                    continue
                bin_refs.sort(key=lambda r: comps[r].pos.x)  # left-to-right

                # Determine row direction: horizontal if wider spread, else vertical
                xs = [comps[r].pos.x for r in bin_refs]
                ys = [comps[r].pos.y for r in bin_refs]
                x_spread = max(xs) - min(xs)
                y_spread = max(ys) - min(ys)
                horizontal = x_spread >= y_spread

                # Compute grid-aligned target positions
                sample = comps[bin_refs[0]]
                gap = max(sample.width_mm, sample.height_mm) + self.clearance

                if horizontal:
                    # Row: same Y, evenly spaced X
                    row_cx = sum(xs) / len(xs)
                    row_cy = anchor.y + row_y_offset
                    targets = []
                    start_x = row_cx - (len(bin_refs) - 1) * gap / 2
                    for k, ref in enumerate(bin_refs):
                        tx = round((start_x + k * gap) / grid) * grid
                        ty = round(row_cy / grid) * grid
                        targets.append((ref, tx, ty))
                    row_y_offset += h_key + self.clearance
                else:
                    # Column: same X, evenly spaced Y
                    bin_refs.sort(key=lambda r: comps[r].pos.y)
                    row_cx = anchor.x + row_y_offset
                    row_cy = sum(ys) / len(ys)
                    targets = []
                    start_y = row_cy - (len(bin_refs) - 1) * gap / 2
                    for k, ref in enumerate(bin_refs):
                        tx = round(row_cx / grid) * grid
                        ty = round((start_y + k * gap) / grid) * grid
                        targets.append((ref, tx, ty))
                    row_y_offset += w_key + self.clearance

                # Blend between organic position and grid target
                for ref, tx, ty in targets:
                    comp = comps[ref]
                    old_pos = Point(comp.pos.x, comp.pos.y)
                    comp.pos.x = comp.pos.x + (tx - comp.pos.x) * strength
                    comp.pos.y = comp.pos.y + (ty - comp.pos.y) * strength
                    _update_pad_positions(comp, old_pos, comp.rotation)
                    total_aligned += 1

        if total_aligned > 0:
            print(f"  Orderedness ({strength:.0%}): aligned {total_aligned} passives")
