"""Tests for the discrete anchor-relative grid + SA-as-assignment.

Synthetic Components (no pcbnew). Cover: slot generation legality (no slot
overlaps an anchor courtyard or the board edge), courtyard-legal pitch,
over-provision bound, anchor-less lane build, deterministic greedy init that
lands a decap in a pin-adjacent slot, and a deterministic assignment-SA smoke.
"""

from __future__ import annotations

import random

from kicraft.autoplacer.brain.leaf_grid_assignment import (
    _overlaps_rect,
    assign_initial,
    build_anchor_grid,
    grid_assignment_sa,
    resnap_to_grid,
)
from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Pad,
    Point,
)

BOARD = (Point(0.0, 0.0), Point(60.0, 60.0))


def _pad(owner, pad_id, x, y, net):
    return Pad(ref=owner, pad_id=pad_id, pos=Point(x, y), net=net, layer=Layer.FRONT)


def _ic(ref, x, y, pads):
    return Component(
        ref=ref, value="", pos=Point(x, y), rotation=0.0, layer=Layer.FRONT,
        width_mm=6.0, height_mm=6.0, kind="ic", pads=pads, body_center=Point(x, y),
    )


def _cap(ref, x, y, pads, rot=0.0):
    return Component(
        ref=ref, value="", pos=Point(x, y), rotation=rot, layer=Layer.FRONT,
        width_mm=2.0, height_mm=1.0, kind="passive", pads=pads, body_center=Point(x, y),
    )


def _ic_4pads(ref="U1", cx=30.0, cy=30.0):
    # pads on all four courtyard edges (IC is 6x6 centered at cx,cy).
    return _ic(ref, cx, cy, [
        _pad(ref, "1", cx, cy - 3.0, "+3V3"),  # N edge
        _pad(ref, "2", cx + 3.0, cy, "GND"),   # E edge
        _pad(ref, "3", cx, cy + 3.0, "SIG"),   # S edge
        _pad(ref, "4", cx - 3.0, cy, "GND"),   # W edge
    ])


def _decap(ref, x, y, na="+3V3", nb="GND", rot=0.0):
    return _cap(ref, x, y, [_pad(ref, "1", x, y - 0.9, na), _pad(ref, "2", x, y + 0.9, nb)], rot)


def test_slots_generated_and_legal():
    u1 = _ic_4pads()
    comps = {u1.ref: u1, "C1": _decap("C1", 40.0, 30.0)}
    grid = build_anchor_grid(comps, board_outline=BOARD, pitch_gap_mm=1.0, grid_snap=0.5)
    assert grid.slots, "expected pin-adjacent slots around the anchor"
    half = 2.0 / 2.0  # max passive long extent / 2
    a_tl = Point(30.0 - 3.0, 30.0 - 3.0)
    a_br = Point(30.0 + 3.0, 30.0 + 3.0)
    for slot in grid.slots:
        # inside the board (with the half-extent margin)
        assert half <= slot.pos.x <= 60.0 - half
        assert half <= slot.pos.y <= 60.0 - half
        # never overlapping the anchor courtyard
        assert not _overlaps_rect(slot.pos, half, a_tl, a_br)


def test_no_two_slots_overlap():
    # Any two slots are >= the passive courtyard extent apart, so simultaneous
    # occupancy is overlap-free (a courtyard-DRC guarantee, by construction).
    u1 = _ic_4pads()
    grid = build_anchor_grid({u1.ref: u1, "C1": _decap("C1", 40, 30)},
                             board_outline=BOARD, pitch_gap_mm=1.0, rings=3, lateral=2)
    extent = 2.0  # max passive long side
    for i, a in enumerate(grid.slots):
        for b in grid.slots[i + 1:]:
            assert (abs(a.pos.x - b.pos.x) >= extent - 1e-6
                    or abs(a.pos.y - b.pos.y) >= extent - 1e-6)


def test_slots_carry_adjacent_pin_nets():
    u1 = _ic_4pads()
    grid = build_anchor_grid({u1.ref: u1, "C1": _decap("C1", 40, 30)},
                             board_outline=BOARD, pitch_gap_mm=1.0)
    all_nets = set().union(*(s.nets for s in grid.slots))
    assert {"+3V3", "GND", "SIG"} & all_nets  # slots know the pins they sit by


def test_overprovision_bounded():
    u1 = _ic_4pads()
    grid = build_anchor_grid({u1.ref: u1, "C1": _decap("C1", 40, 30)},
                             board_outline=BOARD, pitch_gap_mm=1.0,
                             rings=4, lateral=3, max_slots=20)
    assert len(grid.slots) <= 20


def test_anchorless_array_builds_a_lane():
    # An R-2R-style ladder: no IC anchor, rungs chained by low-fanout nets.
    comps = {
        "R1": _cap("R1", 20, 30, [_pad("R1", "1", 19, 30, "IN"), _pad("R1", "2", 21, 30, "N1")]),
        "R2": _cap("R2", 24, 30, [_pad("R2", "1", 23, 30, "N1"), _pad("R2", "2", 25, 30, "N2")]),
        "R3": _cap("R3", 28, 30, [_pad("R3", "1", 27, 30, "N2"), _pad("R3", "2", 29, 30, "OUT")]),
    }
    grid = build_anchor_grid(comps, board_outline=BOARD, pitch_gap_mm=1.0)
    assert grid.slots
    assert all(s.side == "lane" for s in grid.slots)
    assert len(grid.slots) >= 3  # over-provisioned lane


def test_greedy_init_lands_decap_next_to_its_pin():
    u1 = _ic_4pads()
    # decap starts far away; greedy init should snap it to a +3V3/GND slot.
    comps = {u1.ref: u1, "C1": _decap("C1", 55.0, 55.0)}
    grid = build_anchor_grid(comps, board_outline=BOARD, pitch_gap_mm=1.0)
    assign_initial(comps, grid)
    assert "C1" in grid.occupied_by_ref
    sid = grid.occupied_by_ref["C1"]
    slot = grid.slots[sid]
    assert {"+3V3", "GND"} & slot.nets  # matched by a power/ground pin
    # and it physically moved next to the chip (well inside the board center)
    assert abs(comps["C1"].body_center.x - 30.0) < 12.0
    assert abs(comps["C1"].body_center.y - 30.0) < 12.0


def _state(comps):
    return BoardState(components=dict(comps), board_outline=BOARD)


def test_assignment_sa_is_deterministic_and_grid_aligned():
    cfg = {"psw_pin_locality": 1.0, "psw_tidiness": 0.0}

    def _run():
        u1 = _ic_4pads()
        comps = {u1.ref: u1, "C1": _decap("C1", 52, 52), "C2": _decap("C2", 8, 8, "GND", "SIG")}
        state = _state(comps)
        grid = build_anchor_grid(comps, board_outline=BOARD, pitch_gap_mm=1.0)
        best = grid_assignment_sa(
            comps, grid, state, PlacementScorer(state, cfg),
            rng=random.Random(0), max_iters=120,
        )
        return best, grid

    (b1, g1), (b2, g2) = _run(), _run()
    # deterministic: same seed -> identical placement
    assert b1["C1"].body_center.x == b2["C1"].body_center.x
    assert b1["C1"].body_center.y == b2["C1"].body_center.y
    # each passive ended on one of its grid slots (tidy by construction)
    assert "C1" in g1.occupied_by_ref and "C2" in g1.occupied_by_ref
    sid = g1.occupied_by_ref["C1"]
    assert (b1["C1"].body_center.x, b1["C1"].body_center.y) == (
        g1.slots[sid].pos.x, g1.slots[sid].pos.y)


def test_slots_avoid_non_anchor_fixed_parts():
    # A non-anchor fixed part (e.g. an inductor) must be an obstacle so a passive
    # slot never lands on top of it. Culling only _ANCHOR_KINDS let a slot overlap
    # an inductor/LED/diode, producing the unrepairable 'R1:L1' / 'LED1:C2'
    # courtyard overlaps (WS1).
    u1 = _ic_4pads()
    l1 = Component(
        ref="L1", value="", pos=Point(38.0, 30.0), rotation=0.0, layer=Layer.FRONT,
        width_mm=6.0, height_mm=6.0, kind="inductor", pads=[], body_center=Point(38.0, 30.0),
    )
    comps = {u1.ref: u1, "L1": l1, "C1": _decap("C1", 45.0, 45.0)}
    grid = build_anchor_grid(comps, board_outline=BOARD, pitch_gap_mm=1.0, rings=3, lateral=2)
    half = 2.0 / 2.0
    l_tl = Point(38.0 - 3.0, 30.0 - 3.0)
    l_br = Point(38.0 + 3.0, 30.0 + 3.0)
    for slot in grid.slots:
        assert not _overlaps_rect(slot.pos, half, l_tl, l_br), (
            f"slot {slot.pos} overlaps the fixed inductor L1 courtyard"
        )


class _CollapseScorer:
    """Models the buck-3a regression: the input placement scores well, every
    gridded arrangement scores far worse (crossings/packing collapse)."""

    def __init__(self, state, home=(52.0, 52.0)):
        self.state = state
        self.home = home

    def score(self):
        from types import SimpleNamespace

        c = self.state.components["C1"]
        at_home = (abs(c.body_center.x - self.home[0]) < 0.01
                   and abs(c.body_center.y - self.home[1]) < 0.01)
        return SimpleNamespace(total=80.0 if at_home else 40.0)


def test_assignment_sa_keeps_input_when_score_collapses():
    # Accept-if-better still holds for a score COLLAPSE: the buck-3a regression
    # (65.8 -> 43.4 with +18 crossovers) must not ship even though gridding
    # improves pin-adjacency. Input returned verbatim, grid neutralized.
    u1 = _ic_4pads()
    comps = {u1.ref: u1, "C1": _decap("C1", 52.0, 52.0)}
    state = _state(comps)
    grid = build_anchor_grid(comps, board_outline=BOARD, pitch_gap_mm=1.0)
    assert grid.slots  # slots exist, so the guard (not the empty-grid path) fires

    best = grid_assignment_sa(
        comps, grid, state, _CollapseScorer(state), rng=random.Random(0),
        max_iters=60,
    )
    # Input kept verbatim (C1 not moved onto a slot) ...
    assert best["C1"].body_center.x == 52.0
    assert best["C1"].body_center.y == 52.0
    # ... and the grid is neutralized so resnap_to_grid is a no-op.
    assert grid.occupied_by_ref == {}
    assert resnap_to_grid(best, grid) == 0
    assert grid.stats["guard"] == "discard_score"
    assert grid.stats["grid_discarded"] is True


def test_pin_locality_floor_wins_a_score_tie():
    # The pin-locality floor: when the total score cannot tell the two apart
    # (it is a weighted average in which pin-locality holds ~18% of the vote),
    # the arrangement that puts the decap on its pins wins. Silently reverting
    # here is what left the dense-SoC leaf with 30 mm decap hauls.
    from types import SimpleNamespace

    class _FlatScorer:
        def score(self):
            return SimpleNamespace(total=0.0)

    u1 = _ic_4pads()
    comps = {u1.ref: u1, "C1": _decap("C1", 52.0, 52.0)}
    state = _state(comps)
    grid = build_anchor_grid(comps, board_outline=BOARD, pitch_gap_mm=1.0)

    best = grid_assignment_sa(
        comps, grid, state, _FlatScorer(), rng=random.Random(0), max_iters=60,
    )
    assert grid.stats["guard"] == "accept_pin_locality"
    assert grid.stats["grid_pin_median_mm"] < grid.stats["input_pin_median_mm"]
    assert "C1" in grid.occupied_by_ref
    # and it physically moved onto a pin-adjacent slot
    assert abs(best["C1"].body_center.x - 30.0) < 12.0
    assert abs(best["C1"].body_center.y - 30.0) < 12.0


def test_grid_build_stats_report_provisioning():
    # P0.2: slot starvation (23 passives sharing 25 slots) was invisible.
    u1 = _ic_4pads()
    comps = {u1.ref: u1}
    for i in range(4):
        comps[f"C{i}"] = _decap(f"C{i}", 50.0, 40.0 + i * 3)
    grid = build_anchor_grid(comps, board_outline=BOARD, pitch_gap_mm=0.5)
    st = grid.stats
    assert st["gridable_passives"] == 4
    assert st["slots_total"] == len(grid.slots)
    assert st["provisioning_ratio"] >= 3.0  # honored over-provisioning target
    assert st["slots_per_anchor"]["U1"] > 0


def test_crystal_and_button_anchor_their_companions():
    # P0.3: X1/SW1/BT1 classify as misc/battery, so with only ic/regulator/
    # connector anchored their companions got no slots anywhere near them.
    u1 = _ic_4pads()
    x1 = Component(
        ref="X1", value="", pos=Point(45.0, 30.0), rotation=0.0, layer=Layer.FRONT,
        width_mm=3.2, height_mm=2.5, kind="misc", body_center=Point(45.0, 30.0),
        pads=[_pad("X1", "1", 44.0, 30.0, "OSC1"), _pad("X1", "2", 46.0, 30.0, "OSC2")],
    )
    comps = {u1.ref: u1, "X1": x1, "C1": _decap("C1", 10.0, 10.0, "OSC1", "GND")}
    grid = build_anchor_grid(comps, board_outline=BOARD, pitch_gap_mm=0.5)
    assert "X1" in grid.stats["anchors"]
    assert any("OSC1" in s.nets for s in grid.slots)


def test_resnap_is_idempotent():
    u1 = _ic_4pads()
    comps = {u1.ref: u1, "C1": _decap("C1", 52, 52)}
    state = _state(comps)
    grid = build_anchor_grid(comps, board_outline=BOARD, pitch_gap_mm=1.0)
    grid_assignment_sa(comps, grid, state, PlacementScorer(state, {"psw_pin_locality": 1.0}),
                       rng=random.Random(0), max_iters=60)
    # already on-grid -> nothing to snap
    assert resnap_to_grid(comps, grid) == 0


def _drift(comp, dx: float) -> None:
    """Displace a component the way the legality tail does: pos AND
    body_center move together (resnap compares the body centre)."""
    comp.pos.x += dx
    if comp.body_center is not None:
        comp.body_center.x += dx


def test_resnap_snaps_back_a_drifted_occupant():
    u1 = _ic_4pads()
    comps = {u1.ref: u1, "C1": _decap("C1", 52, 52)}
    state = _state(comps)
    grid = build_anchor_grid(comps, board_outline=BOARD, pitch_gap_mm=1.0)
    grid_assignment_sa(comps, grid, state, PlacementScorer(state, {"psw_pin_locality": 1.0}),
                       rng=random.Random(0), max_iters=60)
    if "C1" not in grid.occupied_by_ref:
        return  # assignment kept the input; nothing gridded to exercise
    _drift(comps["C1"], 3.0)
    assert resnap_to_grid(comps, grid) == 1


def test_resnap_exclude_preserves_step16_moves():
    # 2026-07-19 review §3.1: an occupant the courtyard-legalization pass
    # (Step 16) moved must NOT be snapped back -- that reinstated the exact
    # overlap Step 16 had just cleared.
    u1 = _ic_4pads()
    comps = {u1.ref: u1, "C1": _decap("C1", 52, 52)}
    state = _state(comps)
    grid = build_anchor_grid(comps, board_outline=BOARD, pitch_gap_mm=1.0)
    grid_assignment_sa(comps, grid, state, PlacementScorer(state, {"psw_pin_locality": 1.0}),
                       rng=random.Random(0), max_iters=60)
    if "C1" not in grid.occupied_by_ref:
        return
    _drift(comps["C1"], 3.0)
    moved_x = comps["C1"].pos.x
    assert resnap_to_grid(comps, grid, exclude={"C1"}) == 0
    assert comps["C1"].pos.x == moved_x
