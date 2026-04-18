"""Graph utilities — connectivity graph, community detection, MST, crossing estimator.

Pure Python, no external dependencies.
"""
from __future__ import annotations
import random
from collections import defaultdict

from .types import Net, Point, BoardState


class AdjacencyGraph:
    """Simple weighted undirected graph."""

    def __init__(self):
        self.nodes: set[str] = set()
        self._adj: dict[str, dict[str, float]] = defaultdict(dict)

    def add_node(self, n: str):
        self.nodes.add(n)

    def add_edge(self, a: str, b: str, weight: float = 1.0):
        self.nodes.add(a)
        self.nodes.add(b)
        self._adj[a][b] = self._adj[a].get(b, 0) + weight
        self._adj[b][a] = self._adj[b].get(a, 0) + weight

    def neighbors(self, node: str) -> dict[str, float]:
        return self._adj.get(node, {})

    def weight(self, a: str, b: str) -> float:
        return self._adj.get(a, {}).get(b, 0.0)

    def degree(self, node: str) -> float:
        return sum(self._adj.get(node, {}).values())


DEFAULT_POWER_NETS: set[str] = {
    "GND", "VCC", "VDD", "5V", "3V3", "3.3V", "+5V", "+3V3", "+3.3V",
}

def build_connectivity_graph(
    nets: dict[str, Net],
    power_nets: set[str] | None = None,
) -> AdjacencyGraph:
    """Build graph: nodes=component refs, edge weight=shared net count.

    Power nets create stronger connections (critical for grouping).
    GND is skipped (connects everything, dominates clustering).

    Args:
        nets: Mapping of net name to Net objects.
        power_nets: Optional set of net names to treat as power nets
            (weighted more heavily). Falls back to DEFAULT_POWER_NETS.
    """
    active_power_nets = power_nets if power_nets is not None else DEFAULT_POWER_NETS
    g = AdjacencyGraph()
    for net in nets.values():
        if net.name in ("GND", "/GND"):
            continue
        refs = list(net.component_refs)
        if len(refs) < 2:
            continue
        # Power nets = strong connection, signal nets = weak
        weight = 3.0 if net.name in active_power_nets else 1.0
        for i in range(len(refs)):
            g.add_node(refs[i])
            for j in range(i + 1, len(refs)):
                g.add_edge(refs[i], refs[j], weight)
    return g


def find_communities(graph: AdjacencyGraph, max_iter: int = 20,
                     seed: int = 42) -> list[set[str]]:
    """Weighted label propagation community detection.

    O(V+E) per iteration, converges in ~5-10 iterations for small graphs.
    Returns list of component-ref sets (communities).
    """
    rng = random.Random(seed)
    labels: dict[str, str] = {n: n for n in graph.nodes}

    for _ in range(max_iter):
        changed = False
        nodes = list(graph.nodes)
        rng.shuffle(nodes)

        for node in nodes:
            nbrs = graph.neighbors(node)
            if not nbrs:
                continue
            # Tally weighted votes per label
            votes: dict[str, float] = defaultdict(float)
            for nbr, w in nbrs.items():
                votes[labels[nbr]] += w

            best_label = max(votes, key=votes.get)
            if labels[node] != best_label:
                labels[node] = best_label
                changed = True

        if not changed:
            break

    # Group by label
    communities: dict[str, set[str]] = defaultdict(set)
    for node, label in labels.items():
        communities[label].add(node)

    return [c for c in communities.values() if len(c) >= 1]


def minimum_spanning_tree(nodes: list[str],
                          dist_fn) -> list[tuple[str, str, float]]:
    """Prim's MST. dist_fn(a, b) -> float distance.

    Returns [(node_a, node_b, distance), ...].
    """
    if len(nodes) < 2:
        return []
    visited = {nodes[0]}
    edges = []
    remaining = set(nodes[1:])

    while remaining:
        best_d = float("inf")
        best_pair = None
        for v in visited:
            for r in remaining:
                d = dist_fn(v, r)
                if d < best_d:
                    best_d = d
                    best_pair = (v, r)
        if best_pair is None:
            break
        edges.append((best_pair[0], best_pair[1], best_d))
        visited.add(best_pair[1])
        remaining.remove(best_pair[1])

    return edges


def count_crossings(state: BoardState) -> int:
    """Estimate ratsnest crossings by counting intersecting MST edges across all nets.

    For each net, build MST of its pad positions. Then count how many
    MST edges from different nets cross each other. This is a fast
    O(E^2) proxy for routing difficulty.
    """
    # Build MST edges per net (as line segments)
    all_edges: list[tuple[Point, Point, str]] = []  # (start, end, net_name)

    for net in state.nets.values():
        if net.name in ("GND", "/GND") or len(net.pad_refs) < 2:
            continue
        # Gather pad positions
        pad_positions: list[Point] = []
        for ref, pad_id in net.pad_refs:
            comp = state.components.get(ref)
            if comp:
                for p in comp.pads:
                    if p.pad_id == pad_id and p.net == net.name:
                        pad_positions.append(p.pos)
                        break

        if len(pad_positions) < 2:
            continue

        # MST via Prim's
        pnames = [f"{i}" for i in range(len(pad_positions))]
        pos_map = {pnames[i]: pad_positions[i] for i in range(len(pad_positions))}
        mst = minimum_spanning_tree(
            pnames, lambda a, b: pos_map[a].dist(pos_map[b])
        )
        for a, b, _ in mst:
            all_edges.append((pos_map[a], pos_map[b], net.name))

    # Count crossings between edges of different nets
    crossings = 0
    for i in range(len(all_edges)):
        p1, p2, net_i = all_edges[i]
        for j in range(i + 1, len(all_edges)):
            p3, p4, net_j = all_edges[j]
            if net_i == net_j:
                continue
            if _segments_intersect(p1, p2, p3, p4):
                crossings += 1

    return crossings


def _ccw(a: Point, b: Point, c: Point) -> float:
    """Cross product sign: positive if CCW, negative if CW, 0 if collinear."""
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Test if line segments (p1-p2) and (p3-p4) intersect (proper crossing)."""
    d1 = _ccw(p3, p4, p1)
    d2 = _ccw(p3, p4, p2)
    d3 = _ccw(p1, p2, p3)
    d4 = _ccw(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    return False


def total_ratsnest_length(state: BoardState) -> float:
    """Sum of MST edge lengths across all nets. Lower = better placement."""
    total = 0.0
    for net in state.nets.values():
        if net.name in ("GND", "/GND") or len(net.pad_refs) < 2:
            continue
        pad_positions: list[Point] = []
        for ref, pad_id in net.pad_refs:
            comp = state.components.get(ref)
            if comp:
                for p in comp.pads:
                    if p.pad_id == pad_id and p.net == net.name:
                        pad_positions.append(p.pos)
                        break

        if len(pad_positions) < 2:
            continue

        pnames = [f"{i}" for i in range(len(pad_positions))]
        pos_map = {pnames[i]: pad_positions[i] for i in range(len(pad_positions))}
        mst = minimum_spanning_tree(
            pnames, lambda a, b: pos_map[a].dist(pos_map[b])
        )
        total += sum(d for _, _, d in mst)

    return total
