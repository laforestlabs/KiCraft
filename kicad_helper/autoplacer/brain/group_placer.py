"""GroupPlacer — arrange pre-placed functional groups on the board as rigid blocks.

After intra-group placement produces PlacedGroup objects (each a tight
rectangular block with internal component positions), this module places
those blocks on the board respecting:

  - Signal flow order (left-to-right bias)
  - Edge assignments (groups containing edge-pinned connectors)
  - Component zone constraints
  - Inter-group connectivity (attract groups sharing nets)
  - Non-overlap of group bounding boxes

The result is a dict mapping component refs to their final board-global
positions and rotations.

Pure Python, no pcbnew dependency.
"""
from __future__ import annotations

import copy
import math
import random
from collections import defaultdict

from .types import (
    Point, Component, Net, BoardState, Layer,
    FunctionalGroup, GroupSet, PlacedGroup,
)
from .graph import build_connectivity_graph, AdjacencyGraph


class GroupPlacer:
    """Arrange PlacedGroup blocks on the board.

    Treats each group as a rigid rectangle. Uses a force-directed approach
    at the group level (typically 5-10 entities), then translates group-
    local positions to board-global coordinates.
    """

    def __init__(self, state: BoardState, config: dict = None, seed: int = 0):
        self.state = state
        self.cfg = config or {}
        self.rng = random.Random(seed)
        self.clearance = self.cfg.get("placement_clearance_mm", 2.5)
        self.edge_margin = self.cfg.get("edge_margin_mm", 6.0)

    def place_groups(
        self,
        placed_groups: list[PlacedGroup],
        ungrouped_comps: dict[str, Component],
        nets: dict[str, Net],
        signal_flow_order: list[str] | None = None,
    ) -> dict[str, tuple[float, float, float, Layer]]:
        """Place group blocks and ungrouped components on the board.

        Args:
            placed_groups: List of pre-placed groups (from solve_group).
            ungrouped_comps: Components not in any group.
            nets: Full net dict for inter-group connectivity.
            signal_flow_order: Optional list of group leader refs in
                left-to-right signal flow order.

        Returns:
            Dict mapping component ref -> (global_x, global_y, rotation, layer)
            for all components (grouped and ungrouped).
        """
        tl, br = self.state.board_outline
        board_w = br.x - tl.x
        board_h = br.y - tl.y
        margin = self.edge_margin
        zones = self.cfg.get("component_zones", {})

        # --- Build inter-group connectivity graph ---
        group_conn = self._build_inter_group_graph(placed_groups, nets)

        # --- Classify groups by edge affinity ---
        edge_groups, interior_groups = self._classify_groups(
            placed_groups, zones)

        # --- Initial placement: assign positions to group blocks ---
        # Each block is represented by its center point on the board.
        group_positions: dict[str, Point] = {}  # leader_ref -> center position

        # Place edge-assigned groups first
        for leader_ref, edge, pg in edge_groups:
            pos = self._place_on_edge(pg, edge, tl, br, margin)
            group_positions[leader_ref] = pos

        # Place interior groups using signal flow order
        flow = signal_flow_order or [pg.group.leader_ref for pg in interior_groups]
        interior_by_leader = {pg.group.leader_ref: pg for pg in interior_groups}

        # Compute usable interior region
        int_left = tl.x + margin + 2.0
        int_right = br.x - margin - 2.0
        int_top = tl.y + margin + 2.0
        int_bottom = br.y - margin - 2.0

        # Distribute interior groups along signal flow X axis
        flow_refs = [r for r in flow if r in interior_by_leader]
        if flow_refs:
            n_flow = len(flow_refs)
            for i, leader_ref in enumerate(flow_refs):
                pg = interior_by_leader[leader_ref]
                frac = (i + 0.5) / n_flow
                cx = int_left + frac * (int_right - int_left)
                cy = (int_top + int_bottom) / 2
                # Add small random jitter for diversity
                cx += self.rng.gauss(0, board_w * 0.03)
                cy += self.rng.gauss(0, board_h * 0.05)
                # Clamp
                cx = max(int_left + pg.width / 2,
                         min(int_right - pg.width / 2, cx))
                cy = max(int_top + pg.height / 2,
                         min(int_bottom - pg.height / 2, cy))
                group_positions[leader_ref] = Point(cx, cy)

        # Place any remaining interior groups not in flow order
        for pg in interior_groups:
            lr = pg.group.leader_ref
            if lr not in group_positions:
                cx = self.rng.uniform(int_left + pg.width / 2,
                                      int_right - pg.width / 2)
                cy = self.rng.uniform(int_top + pg.height / 2,
                                      int_bottom - pg.height / 2)
                group_positions[lr] = Point(cx, cy)

        # --- Force-directed refinement at group level ---
        all_pgs = {pg.group.leader_ref: pg for pg in placed_groups}
        group_positions = self._refine_group_positions(
            all_pgs, group_positions, group_conn, tl, br, edge_groups)

        # --- Resolve group-level overlaps ---
        self._resolve_group_overlaps(all_pgs, group_positions, tl, br)

        # --- Translate group-local positions to board-global ---
        result: dict[str, tuple[float, float, float, Layer]] = {}
        for pg in placed_groups:
            lr = pg.group.leader_ref
            center = group_positions[lr]
            # Group origin is at top-left of bounding box
            origin_x = center.x - pg.width / 2
            origin_y = center.y - pg.height / 2
            for ref, (rel_x, rel_y, rot) in pg.component_positions.items():
                global_x = origin_x + rel_x
                global_y = origin_y + rel_y
                layer = pg.component_layers.get(ref, Layer.FRONT)
                result[ref] = (global_x, global_y, rot, layer)

        # --- Place ungrouped components ---
        # Simple: scatter them in available space, biased toward connectivity
        ungrouped_result = self._place_ungrouped(
            ungrouped_comps, result, nets, tl, br, margin)
        result.update(ungrouped_result)

        return result

    def _build_inter_group_graph(
        self,
        placed_groups: list[PlacedGroup],
        nets: dict[str, Net],
    ) -> AdjacencyGraph:
        """Build connectivity graph between groups based on shared nets."""
        # Map ref -> group leader
        ref_to_leader: dict[str, str] = {}
        for pg in placed_groups:
            for ref in pg.component_positions:
                ref_to_leader[ref] = pg.group.leader_ref

        graph = AdjacencyGraph()
        for pg in placed_groups:
            graph.add_node(pg.group.leader_ref)

        for net in nets.values():
            if net.name in ("GND", "/GND"):
                continue
            connected_leaders = set()
            for ref, _ in net.pad_refs:
                leader = ref_to_leader.get(ref)
                if leader:
                    connected_leaders.add(leader)
            if len(connected_leaders) >= 2:
                leaders = sorted(connected_leaders)
                weight = 3.0 if net.is_power else 1.0
                for i in range(len(leaders)):
                    for j in range(i + 1, len(leaders)):
                        graph.add_edge(leaders[i], leaders[j], weight)

        return graph

    def _classify_groups(
        self,
        placed_groups: list[PlacedGroup],
        zones: dict,
    ) -> tuple[list[tuple[str, str, PlacedGroup]], list[PlacedGroup]]:
        """Classify groups into edge-assigned and interior.

        A group is edge-assigned if any of its members has an "edge" constraint
        in component_zones.

        Returns:
            (edge_groups, interior_groups) where edge_groups is a list of
            (leader_ref, edge_name, PlacedGroup) tuples.
        """
        edge_groups = []
        interior_groups = []

        for pg in placed_groups:
            edge_assignment = None
            for ref in pg.component_positions:
                zone_cfg = zones.get(ref, {})
                if "edge" in zone_cfg:
                    edge_assignment = zone_cfg["edge"]
                    break
            if edge_assignment:
                edge_groups.append((pg.group.leader_ref, edge_assignment, pg))
            else:
                interior_groups.append(pg)

        return edge_groups, interior_groups

    def _place_on_edge(
        self,
        pg: PlacedGroup,
        edge: str,
        tl: Point,
        br: Point,
        margin: float,
    ) -> Point:
        """Compute center position for a group placed on a board edge."""
        inset = self.cfg.get("connector_edge_inset_mm", 1.0)

        if edge == "left":
            cx = tl.x + inset + pg.width / 2
            cy = self.rng.uniform(
                tl.y + margin + pg.height / 2,
                max(tl.y + margin + pg.height / 2 + 1,
                    br.y - margin - pg.height / 2))
        elif edge == "right":
            cx = br.x - inset - pg.width / 2
            cy = self.rng.uniform(
                tl.y + margin + pg.height / 2,
                max(tl.y + margin + pg.height / 2 + 1,
                    br.y - margin - pg.height / 2))
        elif edge == "top":
            cx = self.rng.uniform(
                tl.x + margin + pg.width / 2,
                max(tl.x + margin + pg.width / 2 + 1,
                    br.x - margin - pg.width / 2))
            cy = tl.y + inset + pg.height / 2
        elif edge == "bottom":
            cx = self.rng.uniform(
                tl.x + margin + pg.width / 2,
                max(tl.x + margin + pg.width / 2 + 1,
                    br.x - margin - pg.width / 2))
            cy = br.y - inset - pg.height / 2
        else:
            cx = (tl.x + br.x) / 2
            cy = (tl.y + br.y) / 2

        return Point(cx, cy)

    def _refine_group_positions(
        self,
        all_pgs: dict[str, PlacedGroup],
        positions: dict[str, Point],
        group_conn: AdjacencyGraph,
        tl: Point,
        br: Point,
        edge_groups: list[tuple[str, str, PlacedGroup]],
    ) -> dict[str, Point]:
        """Force-directed refinement at the group level.

        Groups are few (~5-10), so this is fast. Edge-assigned groups
        are pinned on their edge axis but can slide along it.
        """
        edge_locked = {lr: edge for lr, edge, _ in edge_groups}
        margin = self.edge_margin

        damping = 1.0
        for _ in range(80):
            forces: dict[str, Point] = {lr: Point(0, 0) for lr in positions}

            # Attraction: pull connected groups closer
            leaders = list(positions.keys())
            for i, la in enumerate(leaders):
                for lb in leaders[i + 1:]:
                    w = group_conn.weight(la, lb)
                    if w <= 0:
                        continue
                    pa, pb = positions[la], positions[lb]
                    d = max(pa.dist(pb), 0.1)
                    # Target distance: sum of half-widths + clearance
                    pg_a = all_pgs.get(la)
                    pg_b = all_pgs.get(lb)
                    if pg_a and pg_b:
                        target = (pg_a.width + pg_b.width) / 2 + self.clearance
                    else:
                        target = self.clearance * 2
                    f = 0.1 * w * (d - target)
                    dx = (pb.x - pa.x) / d * f
                    dy = (pb.y - pa.y) / d * f
                    forces[la].x += dx
                    forces[la].y += dy
                    forces[lb].x -= dx
                    forces[lb].y -= dy

            # Repulsion: prevent group overlap
            for i, la in enumerate(leaders):
                pg_a = all_pgs.get(la)
                if not pg_a:
                    continue
                for lb in leaders[i + 1:]:
                    pg_b = all_pgs.get(lb)
                    if not pg_b:
                        continue
                    pa, pb = positions[la], positions[lb]
                    # Check bbox overlap of group blocks
                    a_hw, a_hh = pg_a.width / 2 + self.clearance, pg_a.height / 2 + self.clearance
                    b_hw, b_hh = pg_b.width / 2 + self.clearance, pg_b.height / 2 + self.clearance
                    ox = (a_hw + b_hw) - abs(pa.x - pb.x)
                    oy = (a_hh + b_hh) - abs(pa.y - pb.y)
                    if ox > 0 and oy > 0:
                        # Overlapping — push apart
                        d = max(pa.dist(pb), 0.1)
                        f = 5.0 * min(ox, oy)
                        dx = (pa.x - pb.x) / d * f
                        dy = (pa.y - pb.y) / d * f
                        forces[la].x += dx
                        forces[la].y += dy
                        forces[lb].x -= dx
                        forces[lb].y -= dy

            # Apply forces
            for lr in leaders:
                pg = all_pgs.get(lr)
                if not pg:
                    continue
                dx = forces[lr].x * damping
                dy = forces[lr].y * damping
                mag = math.hypot(dx, dy)
                max_step = 3.0 * damping
                if mag > max_step:
                    dx *= max_step / mag
                    dy *= max_step / mag

                pos = positions[lr]
                # Edge-locked groups: only move along the edge axis
                edge = edge_locked.get(lr)
                if edge in ("left", "right"):
                    pos.y += dy  # slide vertically only
                elif edge in ("top", "bottom"):
                    pos.x += dx  # slide horizontally only
                else:
                    pos.x += dx
                    pos.y += dy

                # Clamp to board
                hw, hh = pg.width / 2, pg.height / 2
                pos.x = max(tl.x + hw + 1, min(br.x - hw - 1, pos.x))
                pos.y = max(tl.y + hh + 1, min(br.y - hh - 1, pos.y))

            damping *= 0.96

        return positions

    def _resolve_group_overlaps(
        self,
        all_pgs: dict[str, PlacedGroup],
        positions: dict[str, Point],
        tl: Point,
        br: Point,
    ):
        """Push group blocks apart until no bounding boxes overlap."""
        leaders = list(positions.keys())
        gap = self.clearance

        for _ in range(100):
            moved = False
            for i, la in enumerate(leaders):
                pg_a = all_pgs.get(la)
                if not pg_a:
                    continue
                pa = positions[la]
                for lb in leaders[i + 1:]:
                    pg_b = all_pgs.get(lb)
                    if not pg_b:
                        continue
                    pb = positions[lb]

                    a_hw = pg_a.width / 2 + gap / 2
                    a_hh = pg_a.height / 2 + gap / 2
                    b_hw = pg_b.width / 2 + gap / 2
                    b_hh = pg_b.height / 2 + gap / 2

                    ox = (a_hw + b_hw) - abs(pa.x - pb.x)
                    oy = (a_hh + b_hh) - abs(pa.y - pb.y)
                    if ox <= 0 or oy <= 0:
                        continue

                    # Push apart on shortest axis
                    if ox < oy:
                        push = (ox + 0.1) / 2
                        sign = 1.0 if pa.x >= pb.x else -1.0
                        pa.x = max(tl.x + pg_a.width / 2 + 1,
                                   min(br.x - pg_a.width / 2 - 1,
                                       pa.x + sign * push))
                        pb.x = max(tl.x + pg_b.width / 2 + 1,
                                   min(br.x - pg_b.width / 2 - 1,
                                       pb.x - sign * push))
                    else:
                        push = (oy + 0.1) / 2
                        sign = 1.0 if pa.y >= pb.y else -1.0
                        pa.y = max(tl.y + pg_a.height / 2 + 1,
                                   min(br.y - pg_a.height / 2 - 1,
                                       pa.y + sign * push))
                        pb.y = max(tl.y + pg_b.height / 2 + 1,
                                   min(br.y - pg_b.height / 2 - 1,
                                       pb.y - sign * push))
                    moved = True
            if not moved:
                break

    def _place_ungrouped(
        self,
        ungrouped_comps: dict[str, Component],
        grouped_result: dict[str, tuple[float, float, float, Layer]],
        nets: dict[str, Net],
        tl: Point,
        br: Point,
        margin: float,
    ) -> dict[str, tuple[float, float, float, Layer]]:
        """Place ungrouped components (mounting holes, misc parts).

        Strategy: bias toward connected grouped components, respect zone
        constraints, and avoid overlapping with already-placed groups.
        """
        result: dict[str, tuple[float, float, float, Layer]] = {}
        zones = self.cfg.get("component_zones", {})

        # Build map: ref -> set of connected refs (for attraction)
        ref_connections: dict[str, set[str]] = defaultdict(set)
        for net in nets.values():
            refs = [r for r, _ in net.pad_refs]
            for r in refs:
                ref_connections[r].update(refs)

        for ref, comp in ungrouped_comps.items():
            zone_cfg = zones.get(ref, {})

            if "edge" in zone_cfg:
                # Place on specified edge
                edge = zone_cfg["edge"]
                hw, hh = comp.width_mm / 2, comp.height_mm / 2
                if edge == "left":
                    x = tl.x + margin
                    y = self.rng.uniform(tl.y + margin + hh, br.y - margin - hh)
                elif edge == "right":
                    x = br.x - margin
                    y = self.rng.uniform(tl.y + margin + hh, br.y - margin - hh)
                elif edge == "top":
                    x = self.rng.uniform(tl.x + margin + hw, br.x - margin - hw)
                    y = tl.y + margin
                elif edge == "bottom":
                    x = self.rng.uniform(tl.x + margin + hw, br.x - margin - hw)
                    y = br.y - margin
                else:
                    x, y = (tl.x + br.x) / 2, (tl.y + br.y) / 2
                result[ref] = (x, y, comp.rotation, comp.layer)

            elif "corner" in zone_cfg:
                corner = zone_cfg["corner"]
                cx = tl.x + margin if "left" in corner else br.x - margin
                cy = tl.y + margin if "top" in corner else br.y - margin
                cx += self.rng.uniform(-2, 2)
                cy += self.rng.uniform(-2, 2)
                result[ref] = (cx, cy, comp.rotation, comp.layer)

            elif "zone" in zone_cfg:
                zone_name = zone_cfg["zone"]
                zx0, zy0, zx1, zy1 = self._get_zone_bounds(zone_name)
                hw, hh = comp.width_mm / 2, comp.height_mm / 2
                x = self.rng.uniform(zx0 + hw, max(zx0 + hw + 1, zx1 - hw))
                y = self.rng.uniform(zy0 + hh, max(zy0 + hh + 1, zy1 - hh))
                result[ref] = (x, y, comp.rotation, comp.layer)

            else:
                # Default: place near connected components
                connected = ref_connections.get(ref, set())
                cx, cy, n_conn = 0, 0, 0
                for cr in connected:
                    if cr in grouped_result:
                        gx, gy, _, _ = grouped_result[cr]
                        cx += gx
                        cy += gy
                        n_conn += 1
                    elif cr in result:
                        rx, ry, _, _ = result[cr]
                        cx += rx
                        cy += ry
                        n_conn += 1
                if n_conn > 0:
                    cx /= n_conn
                    cy /= n_conn
                else:
                    cx = (tl.x + br.x) / 2
                    cy = (tl.y + br.y) / 2
                # Add jitter
                cx += self.rng.gauss(0, 5.0)
                cy += self.rng.gauss(0, 5.0)
                hw, hh = comp.width_mm / 2, comp.height_mm / 2
                cx = max(tl.x + hw + 1, min(br.x - hw - 1, cx))
                cy = max(tl.y + hh + 1, min(br.y - hh - 1, cy))
                result[ref] = (cx, cy, comp.rotation, comp.layer)

        return result

    def _get_zone_bounds(self, zone_name: str) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) for a named board zone."""
        tl, br = self.state.board_outline
        margin = self.edge_margin
        mid_x = (tl.x + br.x) / 2
        mid_y = (tl.y + br.y) / 2

        zone_map = {
            "center":        (tl.x + margin, tl.y + margin, br.x - margin, br.y - margin),
            "top":           (tl.x + margin, tl.y + margin, br.x - margin, mid_y),
            "bottom":        (tl.x + margin, mid_y, br.x - margin, br.y - margin),
            "left":          (tl.x + margin, tl.y + margin, mid_x, br.y - margin),
            "right":         (mid_x, tl.y + margin, br.x - margin, br.y - margin),
            "center-top":    (tl.x + margin, tl.y + margin, br.x - margin, mid_y),
            "center-bottom": (tl.x + margin, mid_y, br.x - margin, br.y - margin),
            "center-left":   (tl.x + margin, tl.y + margin, mid_x, br.y - margin),
            "center-right":  (mid_x, tl.y + margin, br.x - margin, br.y - margin),
            "top-left":      (tl.x + margin, tl.y + margin, mid_x, mid_y),
            "top-right":     (mid_x, tl.y + margin, br.x - margin, mid_y),
            "bottom-left":   (tl.x + margin, mid_y, mid_x, br.y - margin),
            "bottom-right":  (mid_x, mid_y, br.x - margin, br.y - margin),
        }
        return zone_map.get(zone_name, zone_map["center"])
