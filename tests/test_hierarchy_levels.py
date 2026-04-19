"""Tests for _compute_levels from kicraft.cli.solve_hierarchy.

All tests use synthetic/mock data only; no pcbnew dependency.
"""

from __future__ import annotations

import pytest

from kicraft.cli.solve_hierarchy import _compute_levels
from kicraft.autoplacer.brain.types import (
    SubCircuitDefinition,
    SubCircuitId,
    InterfacePort,
)
from kicraft.autoplacer.brain.hierarchy_parser import HierarchyNode, HierarchyGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    name: str,
    children: list[HierarchyNode] | None = None,
    is_leaf: bool | None = None,
) -> HierarchyNode:
    """Build a HierarchyNode with a SubCircuitDefinition.

    If is_leaf is None, it is auto-detected: True when children is empty.
    """
    kids = list(children or [])
    auto_leaf = (len(kids) == 0) if is_leaf is None else is_leaf
    child_ids = [child.definition.id for child in kids]
    defn = SubCircuitDefinition(
        id=SubCircuitId(
            sheet_name=name,
            sheet_file=f"{name.lower()}.kicad_sch",
            instance_path=f"/{name.lower()}",
        ),
        schematic_path=f"/fake/{name.lower()}.kicad_sch",
        component_refs=[],
        ports=[],
        child_ids=child_ids,
        parent_id=None,
        is_leaf=auto_leaf,
    )
    return HierarchyNode(definition=defn, children=kids)


def _make_graph(root: HierarchyNode) -> HierarchyGraph:
    """Build a HierarchyGraph from a root node."""
    nodes_by_path: dict[str, HierarchyNode] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        nodes_by_path[node.id.instance_path] = node
        stack.extend(reversed(node.children))
    return HierarchyGraph(
        project_dir="/fake",
        root_schematic_path="/fake/root.kicad_sch",
        root=root,
        nodes_by_path=nodes_by_path,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleLeaf:
    def test_single_leaf(self):
        """1 leaf node => levels = [[leaf]]."""
        leaf = _make_node("LEAF")
        graph = _make_graph(leaf)
        levels = _compute_levels(graph)

        assert len(levels) == 1
        assert len(levels[0]) == 1
        assert levels[0][0].id.sheet_name == "LEAF"
        assert levels[0][0].is_leaf is True


class TestTwoLeavesOneParent:
    def test_two_leaves_one_parent(self):
        """2 leaves + 1 parent => levels = [[leaf1, leaf2], [parent]]."""
        leaf1 = _make_node("LEAF1")
        leaf2 = _make_node("LEAF2")
        parent = _make_node("PARENT", children=[leaf1, leaf2])
        graph = _make_graph(parent)
        levels = _compute_levels(graph)

        assert len(levels) == 2
        # Level 0: both leaves
        level0_names = sorted(n.id.sheet_name for n in levels[0])
        assert level0_names == ["LEAF1", "LEAF2"]
        # Level 1: parent
        assert len(levels[1]) == 1
        assert levels[1][0].id.sheet_name == "PARENT"


class TestThreeLevelHierarchy:
    def test_three_level_hierarchy(self):
        """leaves -> mid-parent -> root => 3 levels."""
        leaf_a = _make_node("LEAF_A")
        leaf_b = _make_node("LEAF_B")
        mid = _make_node("MID", children=[leaf_a, leaf_b])
        root = _make_node("ROOT", children=[mid])
        graph = _make_graph(root)
        levels = _compute_levels(graph)

        assert len(levels) == 3
        # Level 0: leaves
        level0_names = sorted(n.id.sheet_name for n in levels[0])
        assert level0_names == ["LEAF_A", "LEAF_B"]
        # Level 1: mid
        assert len(levels[1]) == 1
        assert levels[1][0].id.sheet_name == "MID"
        # Level 2: root
        assert len(levels[2]) == 1
        assert levels[2][0].id.sheet_name == "ROOT"


class TestEmptyHierarchy:
    def test_empty_hierarchy(self):
        """A root with no children that is itself a leaf => one level.

        Note: _compute_levels needs a valid graph with at least a root.
        The root as a leaf produces one level.
        """
        root_leaf = _make_node("SOLO")
        graph = _make_graph(root_leaf)
        levels = _compute_levels(graph)
        # A single leaf root gives exactly 1 level
        assert len(levels) == 1
        assert levels[0][0].id.sheet_name == "SOLO"


class TestLeafNodesAllLevelZero:
    def test_leaf_nodes_all_level_zero(self):
        """All leaves are at level 0."""
        leaf1 = _make_node("L1")
        leaf2 = _make_node("L2")
        leaf3 = _make_node("L3")
        parent = _make_node("P", children=[leaf1, leaf2, leaf3])
        graph = _make_graph(parent)
        levels = _compute_levels(graph)

        # Every node in level 0 must be a leaf
        for node in levels[0]:
            assert node.is_leaf is True
        # And all 3 leaves should be there
        assert len(levels[0]) == 3


class TestDeepChain:
    def test_deep_chain(self):
        """Linear chain of 4 nodes (leaf -> p1 -> p2 -> root) => 4 levels."""
        leaf = _make_node("LEAF")
        p1 = _make_node("P1", children=[leaf])
        p2 = _make_node("P2", children=[p1])
        root = _make_node("ROOT", children=[p2])
        graph = _make_graph(root)
        levels = _compute_levels(graph)

        assert len(levels) == 4
        assert levels[0][0].id.sheet_name == "LEAF"
        assert levels[1][0].id.sheet_name == "P1"
        assert levels[2][0].id.sheet_name == "P2"
        assert levels[3][0].id.sheet_name == "ROOT"
