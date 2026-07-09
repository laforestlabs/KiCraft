"""Tests for kicraft.autoplacer.brain.leaf_tidiness.

Pure/synthetic data — no pcbnew, no extraction. Builds PlacedPart views
directly so the metric logic is exercised in isolation.
"""

from __future__ import annotations

from kicraft.autoplacer.brain.leaf_tidiness import (
    PlacedPart,
    aggregate,
    aggregate_pin_locality,
    build_anchor_pad_index,
    functional_passive_groups,
    leaf_pin_locality,
    leaf_tidiness,
    orientation_axis,
    pin_locality_for_passive,
)


def _passive(ref, x, y, rot=0.0, nets=()):
    return PlacedPart(
        ref=ref, kind="passive", locked=False, rotation=rot,
        cx=x, cy=y, w=2.0, h=1.0, nets=tuple(nets),
    )


def _ic(ref, x, y, nets=()):
    return PlacedPart(
        ref=ref, kind="ic", locked=False, rotation=0.0,
        cx=x, cy=y, w=6.0, h=6.0, nets=tuple(nets),
    )


class TestOrientationAxis:
    def test_cardinal_folding(self):
        assert orientation_axis(0) == "H"
        assert orientation_axis(180) == "H"
        assert orientation_axis(360) == "H"
        assert orientation_axis(90) == "V"
        assert orientation_axis(270) == "V"

    def test_near_cardinal(self):
        assert orientation_axis(44) == "H"
        assert orientation_axis(46) == "V"
        assert orientation_axis(134) == "V"
        assert orientation_axis(136) == "H"


class TestGrouping:
    def test_passives_group_under_shared_anchor(self):
        # U1 + three caps each sharing a net with U1 -> one group of 3.
        parts = [
            _ic("U1", 0, 0, nets=("VCC", "GND", "SIG")),
            _passive("C1", 5, 0, nets=("VCC", "GND")),
            _passive("C2", 7, 0, nets=("VCC", "GND")),
            _passive("C3", 9, 0, nets=("SIG", "GND")),
        ]
        groups = functional_passive_groups(parts)
        assert len(groups) == 1
        assert set(groups[0]) == {"C1", "C2", "C3"}

    def test_anchorless_array_groups_by_signal_net(self):
        # No IC/connector anchor: two passives sharing a low-fanout signal net
        # still form an "array" group (an R-2R ladder whose IC is on another
        # sheet) -- the case the old anchor-only grouping dropped.
        parts = [_passive("R1", 0, 0, nets=("A",)), _passive("R2", 2, 0, nets=("A",))]
        groups = functional_passive_groups(parts)
        assert len(groups) == 1
        assert set(groups[0]) == {"R1", "R2"}

    def test_anchorless_ladder_chains_into_one_group(self):
        # Ladder topology: each rung shares a node net with the next. The whole
        # chain must land in a single connected-component group.
        parts = [
            _passive("R1", 0, 0, nets=("IN", "N1")),
            _passive("R2", 2, 0, nets=("N1", "N2")),
            _passive("R3", 4, 0, nets=("N2", "N3")),
            _passive("R4", 6, 0, nets=("N3", "OUT")),
        ]
        groups = functional_passive_groups(parts)
        assert len(groups) == 1
        assert set(groups[0]) == {"R1", "R2", "R3", "R4"}

    def test_high_fanout_bus_does_not_merge_arrays(self):
        # Six passives sharing only a high-fanout rail (GND) must NOT collapse
        # into one group -- a rail isn't a "belongs-together" signal.
        parts = [_passive(f"R{i}", i * 2, 0, nets=("GND",)) for i in range(6)]
        assert functional_passive_groups(parts) == []

    def test_long_ladder_splits_into_rows(self):
        # A 9-rung chain (each rung shares a node net with the next) exceeds the
        # per-row cap and must split into contiguous sub-rows (6 + 3), each a
        # crisp row rather than one impossible 9-wide group.
        nets = [(f"N{i}", f"N{i+1}") for i in range(9)]
        parts = [_passive(f"R{i+1}", i * 2, 0, nets=nets[i]) for i in range(9)]
        groups = functional_passive_groups(parts)
        assert len(groups) == 2
        assert sorted(len(g) for g in groups) == [3, 6]
        # every resistor lands in exactly one row; rows are contiguous chain slices
        assert {r for g in groups for r in g} == {f"R{i+1}" for i in range(9)}

    def test_unconnected_passive_left_out(self):
        parts = [
            _ic("U1", 0, 0, nets=("VCC", "GND")),
            _passive("C1", 5, 0, nets=("VCC", "GND")),
            _passive("C2", 7, 0, nets=("VCC", "GND")),
            _passive("C9", 40, 40, nets=("ISOLATED",)),  # shares nothing
        ]
        groups = functional_passive_groups(parts)
        assert len(groups) == 1
        assert "C9" not in groups[0]


class TestTidinessMetrics:
    def test_perfect_row_scores_ideal(self):
        # Three caps in a straight horizontal row, all same orientation.
        parts = [
            _ic("U1", 0, 0, nets=("VCC", "GND")),
            _passive("C1", 5, 10, rot=0, nets=("VCC", "GND")),
            _passive("C2", 8, 10, rot=0, nets=("VCC", "GND")),
            _passive("C3", 11, 10, rot=180, nets=("VCC", "GND")),  # 180==0 axis
        ]
        m = leaf_tidiness(parts)
        assert m.orientation_consensus_grouped_pct == 100.0
        assert m.alignment_residual_mm == 0.0  # all share y=10

    def test_mixed_orientation_scores_low(self):
        parts = [
            _ic("U1", 0, 0, nets=("VCC", "GND")),
            _passive("C1", 5, 10, rot=0, nets=("VCC", "GND")),
            _passive("C2", 8, 10, rot=90, nets=("VCC", "GND")),  # disagrees
            _passive("C3", 11, 10, rot=90, nets=("VCC", "GND")),  # disagrees
            _passive("C4", 14, 10, rot=0, nets=("VCC", "GND")),
        ]
        m = leaf_tidiness(parts)
        # dominant axis is a 2-2 tie -> 'H' wins; 2 of 4 match.
        assert m.orientation_consensus_grouped_pct == 50.0

    def test_scattered_row_has_residual(self):
        parts = [
            _ic("U1", 0, 0, nets=("VCC", "GND")),
            _passive("C1", 5, 10, nets=("VCC", "GND")),
            _passive("C2", 8, 13, nets=("VCC", "GND")),  # off the row
            _passive("C3", 11, 7, nets=("VCC", "GND")),
        ]
        m = leaf_tidiness(parts)
        assert m.alignment_residual_mm is not None
        assert m.alignment_residual_mm > 1.0

    def test_packing_fill_tight_vs_sparse(self):
        tight = [_passive("R1", 0, 0), _passive("R2", 2, 0)]  # touching
        sparse = [_passive("R1", 0, 0), _passive("R2", 100, 100)]
        assert leaf_tidiness(tight).packing_fill_pct > 40.0
        assert leaf_tidiness(sparse).packing_fill_pct < 1.0

    def test_degenerate_guards(self):
        m = leaf_tidiness([])
        assert m.n_components == 0
        assert m.orientation_consensus_grouped_pct is None
        assert m.orientation_consensus_leaf_pct is None
        assert m.alignment_residual_mm is None
        assert m.packing_fill_pct is None

    def test_single_component_no_fill(self):
        # A lone part fills its own bbox -> trivially 100; guarded to None.
        assert leaf_tidiness([_passive("R1", 0, 0)]).packing_fill_pct is None


def _cap(ref, cx, cy, pads, rot=0.0):
    """A 2-pad passive with explicit per-pad geometry (pads = [(x,y,net),...])."""
    return PlacedPart(
        ref=ref, kind="passive", locked=False, rotation=rot,
        cx=cx, cy=cy, w=2.0, h=1.0,
        nets=tuple(sorted({n for (_x, _y, n) in pads})),
        pads=tuple(pads),
    )


def _chip(ref, cx, cy, pads):
    return PlacedPart(
        ref=ref, kind="ic", locked=False, rotation=0.0,
        cx=cx, cy=cy, w=6.0, h=6.0,
        nets=tuple(sorted({n for (_x, _y, n) in pads})),
        pads=tuple(pads),
    )


class TestPinLocality:
    def test_cap_hugging_pins_beats_cap_far_away(self):
        # U1: +3V3 pin at (0,0), GND pin at (0,2). A decap belongs ~1 mm from
        # those pins; one 10 mm away is the shipped bug.
        u1 = _chip("U1", -3, 1, [(0.0, 0.0, "+3V3"), (0.0, 2.0, "GND")])
        near = _cap("C1", 0.5, 1, [(0.5, 0.0, "+3V3"), (0.5, 2.0, "GND")])
        far = _cap("C2", 10.0, 1, [(10.0, 0.0, "+3V3"), (10.0, 2.0, "GND")])

        m_near = leaf_pin_locality([u1, near])
        m_far = leaf_pin_locality([u1, far])

        assert m_near.mean_worst_pad_dist_mm == 0.5  # the +3V3 pad, 0.5 mm off
        assert 10.0 - 1e-6 <= m_far.mean_worst_pad_dist_mm <= 10.0 + 1e-6
        assert m_near.pin_locality_pct > 75.0
        assert m_far.pin_locality_pct < m_near.pin_locality_pct

    def test_gnd_pad_is_via_reachable_not_dragged(self):
        # A poured-GND pad is via-reachable: its distance to the nearest GND pad
        # must NOT govern the passive's score -- only the non-plane pad does.
        anchors = {"+3V3": [(0.0, 0.0)], "GND": [(0.0, 2.0)]}
        bodies = [(-3.0, 1.0, frozenset({"+3V3", "GND"}))]
        # Same +3V3 pad (1 mm off); GND pad near vs. absurdly far.
        near_gnd = pin_locality_for_passive(
            [(1.0, 0.0, "+3V3"), (1.0, 2.0, "GND")], (1.0, 1.0), anchors, bodies)
        far_gnd = pin_locality_for_passive(
            [(1.0, 0.0, "+3V3"), (1.0, 50.0, "GND")], (1.0, 25.0), anchors, bodies)
        # d_worst identical (governed by the +3V3 pad at 1 mm), regardless of GND.
        assert near_gnd[1] == far_gnd[1] == 1.0

    def test_orientation_to_span_rewarded(self):
        # U1 pins are a horizontal pair: +3V3 (0,0), GND (2,0). Two caps at the
        # SAME pad distance -- one straddling the pins (pad-axis horizontal),
        # one perpendicular -- the aligned one scores higher.
        anchors = {"+3V3": [(0.0, 0.0)], "GND": [(2.0, 0.0)]}
        bodies = [(1.0, -3.0, frozenset({"+3V3", "GND"}))]
        aligned = pin_locality_for_passive(
            [(0.0, 3.0, "+3V3"), (2.0, 3.0, "GND")], (1.0, 3.0), anchors, bodies)
        perp = pin_locality_for_passive(
            [(0.0, 3.0, "+3V3"), (0.0, 5.0, "GND")], (0.0, 4.0), anchors, bodies)
        assert aligned[1] == perp[1] == 3.0  # same worst-pad distance
        assert aligned[2] > perp[2]  # orient_score
        assert aligned[0] > perp[0]  # total

    def test_unscorable_passive_returns_none(self):
        # A passive whose nets reach no anchor at all is unscored.
        res = pin_locality_for_passive(
            [(0.0, 0.0, "ISO_A"), (2.0, 0.0, "ISO_B")], (1.0, 0.0), {}, [])
        assert res is None

    def test_leaf_degenerate_guards(self):
        # No anchors -> nothing scorable.
        no_anchor = leaf_pin_locality([_cap("C1", 0, 0, [(0, 0, "A"), (2, 0, "B")])])
        assert no_anchor.n_scored == 0
        assert no_anchor.pin_locality_pct is None
        # No passives.
        no_pass = leaf_pin_locality([_chip("U1", 0, 0, [(0, 0, "A")])])
        assert no_pass.n_passives == 0
        assert no_pass.pin_locality_pct is None

    def test_anchor_index_covers_all_anchor_kinds(self):
        parts = [
            _chip("U1", 0, 0, [(0, 0, "A")]),
            PlacedPart(ref="J1", kind="connector", locked=False, rotation=0.0,
                       cx=5, cy=0, w=4, h=4, nets=("B",), pads=((5.0, 0.0, "B"),)),
        ]
        idx, bodies = build_anchor_pad_index(parts)
        assert set(idx) == {"A", "B"}
        assert len(bodies) == 2

    def test_aggregate_skips_none(self):
        u1 = _chip("U1", -3, 1, [(0.0, 0.0, "+3V3"), (0.0, 2.0, "GND")])
        good = leaf_pin_locality([u1, _cap("C1", 0.5, 1, [(0.5, 0, "+3V3"), (0.5, 2, "GND")])])
        empty = leaf_pin_locality([])
        agg = aggregate_pin_locality([good, empty])
        assert agg["n_leaves"] == 2
        assert agg["n_leaves_scored"] == 1
        assert agg["pin_locality_pct"] is not None


class TestAggregate:
    def test_skips_none(self):
        good = leaf_tidiness([
            _ic("U1", 0, 0, nets=("V", "G")),
            _passive("C1", 5, 10, nets=("V", "G")),
            _passive("C2", 8, 10, nets=("V", "G")),
        ])
        empty = leaf_tidiness([])
        agg = aggregate([good, empty])
        assert agg["n_leaves"] == 2
        assert agg["orientation_consensus_grouped_pct"] is not None
