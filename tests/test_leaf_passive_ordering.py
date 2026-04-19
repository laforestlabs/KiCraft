"""Tests for kicraft.autoplacer.brain.leaf_passive_ordering.

All tests use synthetic/mock data only; no pcbnew dependency.
"""

from __future__ import annotations


from kicraft.autoplacer.brain.leaf_passive_ordering import (
    component_adjacency_map,
    component_net_degree_map,
    component_net_map,
    component_primary_net_map,
    build_leaf_passive_topology_groups,
    apply_leaf_passive_ordering,
)
from kicraft.autoplacer.brain.subcircuit_extractor import (
    ExtractedSubcircuitBoard,
    NetPartition,
)
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Net,
    Pad,
    Point,
    SubCircuitDefinition,
    SubCircuitId,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad(ref: str, pad_id: str, x: float, y: float, net: str) -> Pad:
    return Pad(ref=ref, pad_id=pad_id, pos=Point(x, y), net=net, layer=Layer.FRONT)


def _comp(
    ref: str,
    x: float,
    y: float,
    kind: str = "passive",
    w: float = 2.0,
    h: float = 1.0,
    pads: list[Pad] | None = None,
    locked: bool = False,
) -> Component:
    return Component(
        ref=ref,
        value="val",
        pos=Point(x, y),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=w,
        height_mm=h,
        pads=list(pads or []),
        locked=locked,
        kind=kind,
    )


def _leaf_def(refs: list[str], name: str = "LEAF") -> SubCircuitDefinition:
    return SubCircuitDefinition(
        id=SubCircuitId(
            sheet_name=name,
            sheet_file="test.kicad_sch",
            instance_path=f"/{name.lower()}",
        ),
        component_refs=list(refs),
        is_leaf=True,
    )


def _extraction(
    components: dict[str, Component],
    nets: dict[str, Net],
) -> ExtractedSubcircuitBoard:
    """Build a minimal ExtractedSubcircuitBoard for testing the ordering helpers."""
    refs = list(components.keys())
    local_state = BoardState(
        components=dict(components),
        nets=dict(nets),
    )
    return ExtractedSubcircuitBoard(
        subcircuit=_leaf_def(refs),
        full_state=local_state,
        local_state=local_state,
        component_refs=refs,
        interface_ports=[],
        net_partition=NetPartition(
            internal=dict(nets),
            external={},
            ignored={},
        ),
    )


# ---------------------------------------------------------------------------
# component_net_degree_map
# ---------------------------------------------------------------------------


class TestComponentNetDegreeMap:
    """Tests for component_net_degree_map."""

    def test_empty_nets(self):
        ext = _extraction({"R1": _comp("R1", 0, 0)}, {})
        result = component_net_degree_map(ext)
        assert result == {}

    def test_single_net_two_refs(self):
        nets = {
            "N1": Net(name="N1", pad_refs=[("R1", "1"), ("R2", "1")]),
        }
        ext = _extraction(
            {"R1": _comp("R1", 0, 0), "R2": _comp("R2", 5, 0)},
            nets,
        )
        result = component_net_degree_map(ext)
        # 2 unique refs -> weight = max(1, 2-1) = 1
        assert result == {"R1": 1, "R2": 1}

    def test_multiple_nets_shared_refs(self):
        nets = {
            "N1": Net(name="N1", pad_refs=[("R1", "1"), ("R2", "1")]),
            "N2": Net(name="N2", pad_refs=[("R1", "2"), ("R2", "2"), ("R3", "1")]),
        }
        ext = _extraction(
            {
                "R1": _comp("R1", 0, 0),
                "R2": _comp("R2", 5, 0),
                "R3": _comp("R3", 10, 0),
            },
            nets,
        )
        result = component_net_degree_map(ext)
        # N1: weight = max(1, 2-1) = 1, adds 1 to R1 and R2
        # N2: weight = max(1, 3-1) = 2, adds 2 to R1, R2, R3
        assert result["R1"] == 3
        assert result["R2"] == 3
        assert result["R3"] == 2

    def test_single_ref_nets_ignored(self):
        nets = {
            "N1": Net(name="N1", pad_refs=[("R1", "1")]),
        }
        ext = _extraction({"R1": _comp("R1", 0, 0)}, nets)
        result = component_net_degree_map(ext)
        assert result == {}

    def test_duplicate_ref_pads_collapsed(self):
        """Same ref on multiple pads of the same net counts as one unique ref."""
        nets = {
            "N1": Net(name="N1", pad_refs=[("R1", "1"), ("R1", "2"), ("R2", "1")]),
        }
        ext = _extraction(
            {"R1": _comp("R1", 0, 0), "R2": _comp("R2", 5, 0)},
            nets,
        )
        result = component_net_degree_map(ext)
        # Unique refs: {R1, R2} -> len=2, weight=max(1, 2-1)=1
        assert result == {"R1": 1, "R2": 1}


# ---------------------------------------------------------------------------
# component_primary_net_map
# ---------------------------------------------------------------------------


class TestComponentPrimaryNetMap:
    """Tests for component_primary_net_map."""

    def test_single_net(self):
        nets = {
            "N1": Net(name="N1", pad_refs=[("R1", "1"), ("R2", "1")]),
        }
        ext = _extraction(
            {"R1": _comp("R1", 0, 0), "R2": _comp("R2", 5, 0)},
            nets,
        )
        result = component_primary_net_map(ext)
        assert result["R1"] == ("N1", 2)
        assert result["R2"] == ("N1", 2)

    def test_picks_higher_weight(self):
        nets = {
            "SMALL": Net(name="SMALL", pad_refs=[("R1", "1"), ("R2", "1")]),
            "BIG": Net(
                name="BIG",
                pad_refs=[("R1", "2"), ("R2", "2"), ("R3", "1")],
            ),
        }
        ext = _extraction(
            {
                "R1": _comp("R1", 0, 0),
                "R2": _comp("R2", 5, 0),
                "R3": _comp("R3", 10, 0),
            },
            nets,
        )
        result = component_primary_net_map(ext)
        # BIG has weight=3 (3 pad_refs), SMALL has weight=2
        assert result["R1"] == ("BIG", 3)
        assert result["R2"] == ("BIG", 3)
        assert result["R3"] == ("BIG", 3)

    def test_tiebreaker_alphabetical(self):
        nets = {
            "BETA": Net(name="BETA", pad_refs=[("R1", "1"), ("R2", "1")]),
            "ALPHA": Net(name="ALPHA", pad_refs=[("R1", "2"), ("R2", "2")]),
        }
        ext = _extraction(
            {"R1": _comp("R1", 0, 0), "R2": _comp("R2", 5, 0)},
            nets,
        )
        result = component_primary_net_map(ext)
        # Both nets have weight=2, so alphabetically earlier wins
        assert result["R1"] == ("ALPHA", 2)
        assert result["R2"] == ("ALPHA", 2)

    def test_single_ref_nets_excluded(self):
        nets = {
            "SOLO": Net(name="SOLO", pad_refs=[("R1", "1")]),
        }
        ext = _extraction({"R1": _comp("R1", 0, 0)}, nets)
        result = component_primary_net_map(ext)
        assert result == {}


# ---------------------------------------------------------------------------
# component_net_map
# ---------------------------------------------------------------------------


class TestComponentNetMap:
    """Tests for component_net_map."""

    def test_component_on_two_nets(self):
        nets = {
            "N1": Net(name="N1", pad_refs=[("R1", "1"), ("R2", "1")]),
            "N2": Net(name="N2", pad_refs=[("R1", "2"), ("R3", "1")]),
        }
        ext = _extraction(
            {
                "R1": _comp("R1", 0, 0),
                "R2": _comp("R2", 5, 0),
                "R3": _comp("R3", 10, 0),
            },
            nets,
        )
        result = component_net_map(ext)
        assert result["R1"] == {"N1", "N2"}
        assert result["R2"] == {"N1"}
        assert result["R3"] == {"N2"}

    def test_single_ref_nets_excluded(self):
        nets = {
            "MULTI": Net(name="MULTI", pad_refs=[("R1", "1"), ("R2", "1")]),
            "SOLO": Net(name="SOLO", pad_refs=[("R1", "2")]),
        }
        ext = _extraction(
            {"R1": _comp("R1", 0, 0), "R2": _comp("R2", 5, 0)},
            nets,
        )
        result = component_net_map(ext)
        # SOLO has only 1 unique ref, so it is excluded
        assert result["R1"] == {"MULTI"}
        assert "SOLO" not in result.get("R1", set())

    def test_empty_nets(self):
        ext = _extraction({"R1": _comp("R1", 0, 0)}, {})
        result = component_net_map(ext)
        assert result == {}


# ---------------------------------------------------------------------------
# component_adjacency_map
# ---------------------------------------------------------------------------


class TestComponentAdjacencyMap:
    """Tests for component_adjacency_map."""

    def test_two_components_same_net(self):
        nets = {
            "N1": Net(name="N1", pad_refs=[("R1", "1"), ("R2", "1")]),
        }
        ext = _extraction(
            {"R1": _comp("R1", 0, 0), "R2": _comp("R2", 5, 0)},
            nets,
        )
        adj = component_adjacency_map(ext)
        # weight = max(1, 2-1) = 1
        assert adj["R1"]["R2"] == 1
        assert adj["R2"]["R1"] == 1

    def test_three_components_same_net(self):
        nets = {
            "N1": Net(
                name="N1",
                pad_refs=[("R1", "1"), ("R2", "1"), ("R3", "1")],
            ),
        }
        ext = _extraction(
            {
                "R1": _comp("R1", 0, 0),
                "R2": _comp("R2", 5, 0),
                "R3": _comp("R3", 10, 0),
            },
            nets,
        )
        adj = component_adjacency_map(ext)
        # weight = max(1, 3-1) = 2; all pairs connected
        assert adj["R1"]["R2"] == 2
        assert adj["R1"]["R3"] == 2
        assert adj["R2"]["R1"] == 2
        assert adj["R2"]["R3"] == 2
        assert adj["R3"]["R1"] == 2
        assert adj["R3"]["R2"] == 2

    def test_adjacency_accumulates_across_nets(self):
        nets = {
            "N1": Net(name="N1", pad_refs=[("R1", "1"), ("R2", "1")]),
            "N2": Net(name="N2", pad_refs=[("R1", "2"), ("R2", "2")]),
        }
        ext = _extraction(
            {"R1": _comp("R1", 0, 0), "R2": _comp("R2", 5, 0)},
            nets,
        )
        adj = component_adjacency_map(ext)
        # Each net contributes weight=1, so total = 2
        assert adj["R1"]["R2"] == 2
        assert adj["R2"]["R1"] == 2

    def test_single_ref_net_no_adjacency(self):
        nets = {
            "SOLO": Net(name="SOLO", pad_refs=[("R1", "1")]),
        }
        ext = _extraction({"R1": _comp("R1", 0, 0)}, nets)
        adj = component_adjacency_map(ext)
        assert adj == {}


# ---------------------------------------------------------------------------
# build_leaf_passive_topology_groups
# ---------------------------------------------------------------------------


def _ic_and_passives_extraction(
    n_passives: int = 5,
) -> tuple[ExtractedSubcircuitBoard, dict[str, Component]]:
    """Build a small topology: 1 IC anchor + n passives sharing nets.

    Creates nets so that each passive shares a net with the IC and at least
    one other passive, giving enough connectivity to form chains.
    """
    comps: dict[str, Component] = {}
    # Anchor IC
    comps["U1"] = _comp("U1", 20.0, 20.0, kind="ic", w=5.0, h=5.0)

    pad_refs_main: list[tuple[str, str]] = [("U1", "1")]
    for i in range(1, n_passives + 1):
        ref = f"R{i}"
        comps[ref] = _comp(ref, 5.0 + i * 3.0, 20.0, kind="passive")
        pad_refs_main.append((ref, "1"))

    # Main shared net (IC + all passives)
    nets: dict[str, Net] = {
        "MAIN": Net(name="MAIN", pad_refs=list(pad_refs_main)),
    }

    # Additional pairwise nets between consecutive passives for chain connectivity
    for i in range(1, n_passives):
        ref_a = f"R{i}"
        ref_b = f"R{i + 1}"
        net_name = f"PAIR_{i}"
        nets[net_name] = Net(
            name=net_name,
            pad_refs=[(ref_a, "2"), (ref_b, "2")],
        )

    ext = _extraction(comps, nets)
    return ext, dict(comps)


class TestBuildLeafPassiveTopologyGroups:
    """Tests for build_leaf_passive_topology_groups."""

    def test_fewer_than_4_passives_returns_empty(self):
        ext, comps = _ic_and_passives_extraction(n_passives=3)
        result = build_leaf_passive_topology_groups(ext, comps)
        assert result == []

    def test_no_anchor_refs_returns_empty(self):
        """All components are passives -- no IC/connector anchor."""
        comps = {
            f"R{i}": _comp(f"R{i}", i * 3.0, 0.0, kind="passive")
            for i in range(1, 6)
        }
        pad_refs = [(f"R{i}", "1") for i in range(1, 6)]
        nets = {"N1": Net(name="N1", pad_refs=pad_refs)}
        ext = _extraction(comps, nets)
        result = build_leaf_passive_topology_groups(ext, comps)
        assert result == []

    def test_valid_topology_returns_groups(self):
        ext, comps = _ic_and_passives_extraction(n_passives=5)
        result = build_leaf_passive_topology_groups(ext, comps)
        assert len(result) >= 1
        group = result[0]
        assert group["anchor_ref"] == "U1"
        assert "chains" in group
        assert len(group["chains"]) >= 1

    def test_chains_contain_passive_refs(self):
        ext, comps = _ic_and_passives_extraction(n_passives=5)
        result = build_leaf_passive_topology_groups(ext, comps)
        all_chain_refs = {
            ref for g in result for chain in g["chains"] for ref in chain
        }
        # All chain refs should be passives (not the anchor)
        for ref in all_chain_refs:
            assert comps[ref].kind == "passive"

    def test_locked_passives_excluded(self):
        """Locked passives should not count toward the 4-passive threshold."""
        comps: dict[str, Component] = {}
        comps["U1"] = _comp("U1", 20.0, 20.0, kind="ic", w=5.0, h=5.0)
        pad_refs: list[tuple[str, str]] = [("U1", "1")]
        for i in range(1, 6):
            ref = f"R{i}"
            # Lock 2 of them, leaving only 3 unlocked (< 4 threshold)
            locked = i <= 2
            comps[ref] = _comp(
                ref, 5.0 + i * 3.0, 20.0, kind="passive", locked=locked,
            )
            pad_refs.append((ref, "1"))
        nets = {"N1": Net(name="N1", pad_refs=pad_refs)}
        ext = _extraction(comps, nets)
        result = build_leaf_passive_topology_groups(ext, comps)
        assert result == []

    def test_multiple_anchors(self):
        """Two ICs with their own passives can produce separate groups."""
        comps: dict[str, Component] = {}
        comps["U1"] = _comp("U1", 10.0, 20.0, kind="ic", w=5.0, h=5.0)
        comps["U2"] = _comp("U2", 40.0, 20.0, kind="ic", w=5.0, h=5.0)
        for i in range(1, 5):
            comps[f"R{i}"] = _comp(
                f"R{i}", 5.0 + i * 3.0, 20.0, kind="passive",
            )
        for i in range(5, 9):
            comps[f"R{i}"] = _comp(
                f"R{i}", 35.0 + (i - 4) * 3.0, 20.0, kind="passive",
            )

        nets: dict[str, Net] = {
            "NET_U1": Net(
                name="NET_U1",
                pad_refs=[("U1", "1")] + [(f"R{i}", "1") for i in range(1, 5)],
            ),
            "NET_U2": Net(
                name="NET_U2",
                pad_refs=[("U2", "1")] + [(f"R{i}", "1") for i in range(5, 9)],
            ),
        }
        # Add pairwise nets for chain connectivity within each group
        for i in range(1, 4):
            nets[f"P1_{i}"] = Net(
                name=f"P1_{i}",
                pad_refs=[(f"R{i}", "2"), (f"R{i+1}", "2")],
            )
        for i in range(5, 8):
            nets[f"P2_{i}"] = Net(
                name=f"P2_{i}",
                pad_refs=[(f"R{i}", "2"), (f"R{i+1}", "2")],
            )
        ext = _extraction(comps, nets)
        result = build_leaf_passive_topology_groups(ext, comps)
        # Should have groups for both anchors
        anchor_refs = {g["anchor_ref"] for g in result}
        assert "U1" in anchor_refs or "U2" in anchor_refs
        # At least one group should exist
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# apply_leaf_passive_ordering
# ---------------------------------------------------------------------------


class TestApplyLeafPassiveOrdering:
    """Tests for apply_leaf_passive_ordering."""

    def test_disabled_returns_deepcopy_unchanged(self):
        ext, comps = _ic_and_passives_extraction(n_passives=5)
        cfg: dict = {"leaf_passive_ordering_enabled": False}
        result = apply_leaf_passive_ordering(ext, comps, cfg)
        # Should be a deepcopy -- same positions, different objects
        for ref in comps:
            assert result[ref].pos.x == comps[ref].pos.x
            assert result[ref].pos.y == comps[ref].pos.y
            assert result[ref] is not comps[ref]

    def test_fewer_than_4_passives_unchanged(self):
        ext, comps = _ic_and_passives_extraction(n_passives=3)
        cfg: dict = {"leaf_passive_ordering_enabled": True}
        result = apply_leaf_passive_ordering(ext, comps, cfg)
        for ref in comps:
            assert result[ref].pos.x == comps[ref].pos.x
            assert result[ref].pos.y == comps[ref].pos.y

    def test_valid_topology_some_positions_change(self):
        ext, comps = _ic_and_passives_extraction(n_passives=6)
        cfg: dict = {
            "leaf_passive_ordering_enabled": True,
            "leaf_passive_ordering_strength": 1.0,
            "leaf_passive_ordering_max_displacement_mm": 20.0,
            "placement_grid_mm": 0.5,
            "placement_clearance_mm": 1.0,
        }
        result = apply_leaf_passive_ordering(ext, comps, cfg)
        # At least some passive positions should have changed
        moved = 0
        for ref in comps:
            if comps[ref].kind == "passive":
                dx = abs(result[ref].pos.x - comps[ref].pos.x)
                dy = abs(result[ref].pos.y - comps[ref].pos.y)
                if dx > 0.01 or dy > 0.01:
                    moved += 1
        assert moved > 0, "Expected at least one passive to be repositioned"

    def test_result_is_deepcopy(self):
        """Mutations on the result should not affect the original."""
        ext, comps = _ic_and_passives_extraction(n_passives=5)
        cfg: dict = {"leaf_passive_ordering_enabled": True}
        result = apply_leaf_passive_ordering(ext, comps, cfg)
        # Mutate result and verify original is untouched
        for ref in result:
            result[ref].pos = Point(999.0, 999.0)
        for ref in comps:
            assert comps[ref].pos.x != 999.0

    def test_anchor_position_unchanged(self):
        """The IC anchor should not be repositioned by ordering."""
        ext, comps = _ic_and_passives_extraction(n_passives=6)
        cfg: dict = {
            "leaf_passive_ordering_enabled": True,
            "leaf_passive_ordering_strength": 1.0,
        }
        result = apply_leaf_passive_ordering(ext, comps, cfg)
        assert result["U1"].pos.x == comps["U1"].pos.x
        assert result["U1"].pos.y == comps["U1"].pos.y

    def test_connector_anchor(self):
        """Connectors also serve as anchors."""
        comps: dict[str, Component] = {}
        comps["J1"] = _comp("J1", 20.0, 20.0, kind="connector", w=6.0, h=3.0)
        pad_refs: list[tuple[str, str]] = [("J1", "1")]
        for i in range(1, 6):
            ref = f"C{i}"
            comps[ref] = _comp(ref, 5.0 + i * 3.0, 20.0, kind="passive")
            pad_refs.append((ref, "1"))
        nets: dict[str, Net] = {
            "SHARED": Net(name="SHARED", pad_refs=pad_refs),
        }
        for i in range(1, 5):
            nets[f"PAIR_{i}"] = Net(
                name=f"PAIR_{i}",
                pad_refs=[(f"C{i}", "2"), (f"C{i+1}", "2")],
            )
        ext = _extraction(comps, nets)
        cfg: dict = {
            "leaf_passive_ordering_enabled": True,
            "leaf_passive_ordering_strength": 1.0,
            "leaf_passive_ordering_max_displacement_mm": 20.0,
        }
        result = apply_leaf_passive_ordering(ext, comps, cfg)
        # Connector should stay in place
        assert result["J1"].pos.x == comps["J1"].pos.x
        assert result["J1"].pos.y == comps["J1"].pos.y

    def test_axis_bias_horizontal(self):
        """Explicit horizontal axis bias should be accepted."""
        ext, comps = _ic_and_passives_extraction(n_passives=5)
        cfg: dict = {
            "leaf_passive_ordering_enabled": True,
            "leaf_passive_ordering_axis_bias": "horizontal",
            "leaf_passive_ordering_strength": 1.0,
            "leaf_passive_ordering_max_displacement_mm": 20.0,
        }
        # Should not raise
        result = apply_leaf_passive_ordering(ext, comps, cfg)
        assert "U1" in result

    def test_axis_bias_vertical(self):
        """Explicit vertical axis bias should be accepted."""
        ext, comps = _ic_and_passives_extraction(n_passives=5)
        cfg: dict = {
            "leaf_passive_ordering_enabled": True,
            "leaf_passive_ordering_axis_bias": "vertical",
            "leaf_passive_ordering_strength": 1.0,
            "leaf_passive_ordering_max_displacement_mm": 20.0,
        }
        result = apply_leaf_passive_ordering(ext, comps, cfg)
        assert "U1" in result
