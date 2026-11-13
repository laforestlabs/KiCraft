"""A placement seed must not depend on Python's per-process set order."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_PROGRAM = """
import copy
import json
import sys
from kicraft.autoplacer.brain.graph import AdjacencyGraph, find_communities
from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.types import BoardState, Component, Layer, Point
from kicraft.autoplacer.config import DEFAULT_CONFIG

graph = AdjacencyGraph()
for group in (("B0", "B1", "B2", "B3"), ("B4", "B5", "B6", "B7")):
    for index, first in enumerate(group):
        for second in group[index + 1:]:
            graph.add_edge(first, second, 3.0)
graph.add_edge("B3", "B4", 1.0)
results = {}
for seed in (4908357, 4908358):
    components = {
        ref: Component(
            ref=ref, value="", pos=Point(40, 30), rotation=0,
            width_mm=4, height_mm=4, pads=[], layer=Layer.FRONT, kind="ic",
        )
        for ref in sorted(graph.nodes)
    }
    state = BoardState(
        components=components, nets={},
        board_outline=(Point(0, 0), Point(80, 60)),
    )
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(
        component_zones={}, ic_groups={}, scatter_mode=sys.argv[1],
        intra_cluster_iters=8,
    )
    solver = PlacementSolver(state, cfg, seed=seed)
    clusters = find_communities(graph, seed=seed)
    solver._place_clusters(components, clusters, graph)
    solver._optimize_intra_cluster(components, clusters, graph)
    results[seed] = {
        ref: [component.pos.x, component.pos.y, component.rotation]
        for ref, component in sorted(components.items())
    }
print(json.dumps(results, sort_keys=True))
"""


@pytest.mark.parametrize("mode", ["cluster", "random"])
def test_seeded_positions_ignore_process_hash_order(mode):
    outcomes = []
    for hash_seed in ("1", "3"):
        output = subprocess.check_output(
            [sys.executable, "-c", _PROGRAM, mode],
            cwd=_ROOT,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            text=True,
        )
        outcomes.append(json.loads(output.splitlines()[-1]))

    assert outcomes[0] == outcomes[1]
    assert outcomes[0]["4908357"] != outcomes[0]["4908358"]
