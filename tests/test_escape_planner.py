"""Golden tests for the fine-pitch escape planner (pure geometry, no pcbnew).

The fixture is the real nRF52840 aQFN-73 pad table from KC-RYVSQV -- the board
whose MCU leaf failed ``no_unconnected`` on DEC3/DECUSB/XL1/nRESET in all nine
rounds. These tests pin the *margins* that decide those verdicts (the 2 um that
closes a same-row diagonal, the 15 um that decides whether two escapes share a
0.75 mm depopulated lane, the via diameter at which an inner ring can be
dog-boned at all), so a future "harmless" constant bump fails here in
milliseconds instead of in a self-eval batch two days later.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.autoplacer.brain.escape_planner import (
    Pad,
    Rules,
    find_lanes,
    pads_from_dicts,
    plan_escapes,
    via_fanout_center,
)
from kicraft.autoplacer.brain.breakout_stubs import STAMP_CLEARANCE_GUARD_MM
from kicraft.autoplacer.fab_profile import (
    FAB_CAPABILITY,
    NETCLASS_CLEARANCE_MM,
    fab_floors,
    fanout_via,
)

FIXTURE = Path(__file__).parent / "data" / "aqfn73_pads.json"

_VIA_DIA, _VIA_DRILL = fanout_via()
_NETCLASS_CLEARANCE = NETCLASS_CLEARANCE_MM  # unchanged by this work
# What the stamper enforces, and therefore what the planner must plan at.
_EFFECTIVE_CLEARANCE = _NETCLASS_CLEARANCE + STAMP_CLEARANCE_GUARD_MM

# The rule sets that matter, named for what they mean.
HEAD_RULES = Rules(
    track_mm=0.153,
    clearance_mm=_EFFECTIVE_CLEARANCE,
    via_diameter_mm=0.6,
    via_drill_mm=0.3,
)
"""What shipped: the legacy 6 mil floor and the 0.6/0.3 netclass via."""

PLAN_RULES = Rules(
    track_mm=fab_floors()["track_mm"],
    clearance_mm=_EFFECTIVE_CLEARANCE,
    via_diameter_mm=_VIA_DIA,
    via_drill_mm=_VIA_DRILL,
)
"""What the pipeline now plans at: capability track, netclass clearance (plus the
stamper's geometry guard), fanout via class."""

CAP_RULES = Rules(
    track_mm=fab_floors()["track_mm"],
    clearance_mm=fab_floors()["clearance_mm"] + STAMP_CLEARANCE_GUARD_MM,
    via_diameter_mm=_VIA_DIA,
    via_drill_mm=_VIA_DRILL,
)
"""If the netclasses ever followed the floors too -- the "feasible only at
capability" probe."""

# The six pads the investigation named, with the net each carries.
WITNESS = {
    "AC13": "nRESET",
    "AC5": "DECUSB",
    "D2": "XL1",
    "D23": "DEC3",
    "B7": "GND",
    "F23": "GND",
}


@pytest.fixture(scope="module")
def aqfn_pads() -> list[Pad]:
    return pads_from_dicts(json.loads(FIXTURE.read_text())["pads"])


# --------------------------------------------------------------------------- #
# The fixture itself -- if this drifts, every margin below is meaningless.
# --------------------------------------------------------------------------- #


def test_fixture_matches_the_measured_package(aqfn_pads):
    by_num = {p.number: p for p in aqfn_pads}
    assert len(aqfn_pads) == 74  # 73 signal pads + the exposed pad
    ep = by_num["0"]
    assert (ep.w, ep.h) == (4.85, 4.85)
    for num, net in WITNESS.items():
        assert by_num[num].net == net
    # Ring geometry the whole analysis rests on.
    assert by_num["AC13"].x == pytest.approx(2.75) and by_num["AC13"].y == pytest.approx(0.0)
    inner = [p for p in aqfn_pads if p.number in ("D2", "F2", "D23", "F23")]
    assert all(max(abs(p.x), abs(p.y)) == pytest.approx(2.75) for p in inner)
    signal = [p for p in aqfn_pads if p.number != "0"]
    assert all((p.w, p.h) == (0.25, 0.25) for p in signal)
    # inner-pad -> EP gap 0.20; inner -> outer ring channel 0.25.
    assert 2.75 - 0.125 - ep.w / 2 == pytest.approx(0.20)
    assert 3.25 - 0.125 - (2.75 + 0.125) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Lane capacity -- the 2 um and 15 um margins, stated as arithmetic.
# --------------------------------------------------------------------------- #


_LEGACY = Rules(track_mm=0.153, clearance_mm=0.153)
_CAPABILITY = Rules(track_mm=0.127, clearance_mm=0.127)


@pytest.mark.parametrize(
    "gap, rules, expected",
    [
        # Adjacent pads / the inner-to-outer ring channel: never passable.
        (0.25, _LEGACY, 0),
        (0.25, _CAPABILITY, 0),
        # inner-pad -> EP gap: never passable.
        (0.20, _LEGACY, 0),
        (0.20, _CAPABILITY, 0),
        # Same-row diagonal opening: misses by 2 um at 0.153 (needs 0.459),
        # carries one track at 0.127 (needs 0.381).
        (0.457, _LEGACY, 0),
        (0.457, _CAPABILITY, 1),
        # A depopulated ring position: one track at 0.153 (two need 0.765),
        # two at 0.127 (two need 0.635) -- the 15 um that strands XL1.
        (0.75, _LEGACY, 1),
        (0.75, _CAPABILITY, 2),
    ],
)
def test_lane_capacity_pins_the_corridor_margins(gap, rules, expected):
    assert rules.lane_capacity(gap) == expected


def test_capacity_one_is_exactly_track_plus_two_clearances():
    need = _LEGACY.track_mm + 2 * _LEGACY.clearance_mm
    assert _LEGACY.lane_capacity(need - 0.000001) == 0
    assert _LEGACY.lane_capacity(need) == 1


def test_lanes_found_on_the_aqfn_outer_row(aqfn_pads):
    lanes = {(ln.side, round(ln.gap_mm, 3)) for ln in find_lanes(aqfn_pads, PLAN_RULES)}
    # The package's designed openings: depopulated ring positions (0.75 mm),
    # corner exits (0.5 mm) and the wide depopulated spans.
    assert ("y+", 0.75) in lanes  # C1 -> G1, the lane XL2's radial ray threads
    assert ("y-", 0.75) in lanes  # B24 -> E24, DEC3's private lane
    assert ("x-", 2.0) in lanes
    # The fully-populated AD column leaves NO lane on the x+ side near AC13.
    x_plus = [ln for ln in find_lanes(aqfn_pads, PLAN_RULES) if ln.side == "x+"]
    assert all(not (ln.lo < 0.0 < ln.hi) for ln in x_plus)


# --------------------------------------------------------------------------- #
# Via fanout -- the load-bearing claim: a 0.4 mm via fits beside the ring.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("num", sorted(WITNESS))
def test_every_witness_pad_has_a_legal_fanout_via(aqfn_pads, num):
    """The shipped fanout class clears every trapped inner pad at the clearance
    the stamper actually enforces -- the unlock this whole change rests on."""
    src = next(p for p in aqfn_pads if p.number == num)
    got = via_fanout_center(src, aqfn_pads, PLAN_RULES)
    assert got is not None, f"{num} has no legal 0.4 mm via centre"
    (vx, vy), margin = got
    assert margin >= 0.0
    assert abs(vx - src.x) <= 0.75 and abs(vy - src.y) <= 0.75


def test_ac13_via_window_is_real_but_thin(aqfn_pads):
    """AC13 faces the fully-populated AD column: via-fanout or nothing.

    Bounded below by the exposed pad and above by the diagonal to AD12/AD14.
    Pinned because it is the single number that decides whether an nRF52840
    reset pin can be escaped at all, and because it is only ~23 um wide -- any
    "harmless" bump to the via class or the clearance closes it.
    """
    from kicraft.autoplacer.brain.escape_planner import _dist_point_rect, _rect

    src = next(p for p in aqfn_pads if p.number == "AC13")
    foreign = [_rect(p) for p in aqfn_pads if p.net != src.net]
    keep = PLAN_RULES.via_keepout_mm
    legal = [
        round(i * 0.001, 3)
        for i in range(0, 200)
        if all(
            _dist_point_rect(src.x + i * 0.001, src.y, r) >= keep for r in foreign
        )
    ]
    assert legal, "no legal outward offset for AC13's fanout via"
    assert 0.0 < max(legal) - min(legal) < 0.04  # there is no slack here
    # The planner actually finds a centre in that channel: outward, close, and
    # clearing every foreign pad by the full via keep-out.
    got = via_fanout_center(src, aqfn_pads, PLAN_RULES)
    assert got is not None
    (vx, vy), _margin = got
    assert 0.0 < vx - src.x <= max(legal)
    assert abs(vy - src.y) <= 0.05
    assert all(_dist_point_rect(vx, vy, r) >= keep for r in foreign)


@pytest.mark.parametrize("dia, drill", [(0.6, 0.3), (0.5, 0.3), (0.45, 0.2), (0.4, 0.2)])
def test_ac13_has_no_via_window_at_larger_diameters(aqfn_pads, dia, drill):
    """Everything above the shipped class fails AC13 -- including 0.4/0.2.

    The plan this implements proposed 0.4/0.2 and expected a 0.5/0.3 window to
    open at c = 0.127. Honest geometry says otherwise on both counts. 0.4/0.2
    has a real 17 um window at the BARE 0.153 rule, but the stamper holds a
    +10 um geometry guard above the rule (measured on this very footprint), and
    that closes it. Hence 0.35/0.2 -- see autoplacer/fab_profile.py.
    """
    src = next(p for p in aqfn_pads if p.number == "AC13")
    rules = Rules(
        track_mm=fab_floors()["track_mm"],
        clearance_mm=_EFFECTIVE_CLEARANCE,
        via_diameter_mm=dia,
        via_drill_mm=drill,
    )
    assert via_fanout_center(src, aqfn_pads, rules) is None


def test_the_rejected_04_via_really_did_fit_at_the_bare_rule(aqfn_pads):
    """Records WHY 0.4/0.2 was rejected: the rule allows it, the guard does not."""
    src = next(p for p in aqfn_pads if p.number == "AC13")
    bare = Rules(track_mm=0.127, clearance_mm=0.153, via_diameter_mm=0.4, via_drill_mm=0.2)
    guarded = Rules(
        track_mm=0.127,
        clearance_mm=_EFFECTIVE_CLEARANCE,
        via_diameter_mm=0.4,
        via_drill_mm=0.2,
    )
    assert via_fanout_center(src, aqfn_pads, bare) is not None
    assert via_fanout_center(src, aqfn_pads, guarded) is None


def test_two_adjacent_ring_pads_get_separated_vias(aqfn_pads):
    """XL1/XL2 sit one 0.5 mm ring pitch apart; their vias must not collide."""
    d2 = next(p for p in aqfn_pads if p.number == "D2")
    f2 = next(p for p in aqfn_pads if p.number == "F2")
    first = via_fanout_center(d2, aqfn_pads, PLAN_RULES)
    assert first is not None
    second = via_fanout_center(
        f2, aqfn_pads, PLAN_RULES, taken_vias=[(d2.net, first[0][0], first[0][1])]
    )
    assert second is not None
    gap = (
        (first[0][0] - second[0][0]) ** 2 + (first[0][1] - second[0][1]) ** 2
    ) ** 0.5
    assert gap >= PLAN_RULES.via_diameter_mm + PLAN_RULES.clearance_mm


# --------------------------------------------------------------------------- #
# Whole-package verdicts at each rule set.
# --------------------------------------------------------------------------- #


def test_head_rules_strand_the_wall_locked_pads(aqfn_pads):
    """What shipped: a 0.6 mm via cannot dog-bone a 0.5 mm-pitch inner ring, and
    a 0.153 mm escape track cannot share the package's designed lanes. Three
    pads have no exit at all -- nRESET, DECUSB and one of the crystal pins,
    which is the KC-RYVSQV failure list."""
    plan = plan_escapes(aqfn_pads, HEAD_RULES)
    assert set(plan.infeasible) == {"AC13", "AC5", "F2"}


def test_fanout_via_class_resolves_every_inner_pad(aqfn_pads):
    """The primary result: at TODAY's netclass clearance nothing is stranded."""
    plan = plan_escapes(aqfn_pads, PLAN_RULES)
    assert plan.infeasible == []
    for num in WITNESS:
        assert plan.escapes[num].feasible
    # Only the wall-locked pad needs a drill; everything else leaves on-layer.
    assert {e.pad for e in plan.escapes.values() if e.kind == "via"} == {"AC13"}
    # ... and every inner GND pad reaches the exposed pad without one.
    assert {e.pad for e in plan.escapes.values() if e.kind == "tie"} >= {"B7", "F23"}


def test_capability_track_is_what_reopens_the_designed_lanes(aqfn_pads):
    """The 0.75 mm depopulated lanes carry TWO escapes once the escape track
    follows the fab capability, ONE at the legacy 6 mil floor. That 26 um is
    what put XL1 and XL2 in contention for a single opening."""
    def caps(rules):
        return {round(ln.gap_mm, 3): ln.capacity for ln in find_lanes(aqfn_pads, rules)}

    assert caps(PLAN_RULES)[0.75] == 2          # 0.127 escape track
    assert caps(HEAD_RULES)[0.75] == 1          # 0.153 escape track
    # And at the capability clearance too, nothing is left stranded.
    plan = plan_escapes(aqfn_pads, CAP_RULES)
    assert plan.infeasible == []


def test_without_a_via_class_the_wall_locked_pads_are_honestly_infeasible(aqfn_pads):
    """No nub, no false progress -- an unreachable pad says so."""
    plan = plan_escapes(aqfn_pads, PLAN_RULES, allow_via=False)
    assert set(plan.infeasible) == {"AC13"}
    for num in plan.infeasible:
        assert plan.escapes[num].polyline == ()
        assert plan.escapes[num].via_center is None


def test_outer_row_pads_are_left_to_the_router(aqfn_pads):
    plan = plan_escapes(aqfn_pads, PLAN_RULES)
    # Pads on the ring's outer row have open copper in front of them.
    assert plan.escapes["A18"].kind == "open"   # x- outer column
    assert plan.escapes["AD8"].kind == "open"   # x+ outer column
    assert plan.escapes["B1"].kind == "open"    # y+ outer row
    assert plan.escapes["E24"].kind == "open"   # y- outer row
    # The exposed pad hosts its own thermal vias; it is not an escape problem.
    assert "0" not in plan.escapes


def test_plan_is_deterministic(aqfn_pads):
    a = plan_escapes(aqfn_pads, PLAN_RULES).to_dict()
    b = plan_escapes(aqfn_pads, PLAN_RULES).to_dict()
    assert a == b


def test_planned_escapes_hold_clearance_from_each_other(aqfn_pads):
    """Capacity rations how many share a lane; positions must differ too."""
    from kicraft.autoplacer.brain.escape_planner import _seg_seg_dist

    plan = plan_escapes(aqfn_pads, PLAN_RULES)
    segs = [
        (e.net, a, b)
        for e in plan.escapes.values()
        for a, b in zip(e.polyline, e.polyline[1:])
    ]
    need = PLAN_RULES.track_mm + PLAN_RULES.clearance_mm
    for i, (net_a, a1, a2) in enumerate(segs):
        for net_b, b1, b2 in segs[i + 1 :]:
            if net_a == net_b:
                continue
            assert _seg_seg_dist(a1, a2, b1, b2) >= need - 1e-9


def test_escapes_stay_short(aqfn_pads):
    """An escape leaves the pad field; routing the net is FreeRouting's job."""
    plan = plan_escapes(aqfn_pads, PLAN_RULES, max_escape_len_mm=2.5)
    for e in plan.escapes.values():
        length = sum(
            ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
            for a, b in zip(e.polyline, e.polyline[1:])
        )
        assert length <= 2.5 + 1e-9, f"{e.pad} escape is {length:.2f} mm"


def test_a_simple_two_pad_footprint_needs_no_plan():
    """Nothing changes for parts that were never trapped."""
    pads = [
        Pad(number="1", net="VCC", x=-0.8, y=0.0, w=0.9, h=0.9),
        Pad(number="2", net="GND", x=0.8, y=0.0, w=0.9, h=0.9),
    ]
    plan = plan_escapes(pads, PLAN_RULES)
    assert all(e.kind == "open" for e in plan.escapes.values())
    assert plan.stampable == []


# --------------------------------------------------------------------------- #
# The fab capability profile these rule sets come from.
# --------------------------------------------------------------------------- #


def test_fanout_via_ring_covers_the_hole_clearance_rule():
    """The invariant that keeps FreeRouting from generating hole_clearance errors.

    FreeRouting only knows the COPPER clearance. A track it places at exactly the
    copper rule from a fanout via's annulus sits ``clearance + ring`` from that
    via's HOLE, so unless

        netclass_clearance + annular_ring >= hole_to_copper_clearance

    the router can legally produce a board KiCad fails. A 0.35/0.2 via (ring
    0.075) misses this by 22 um, and on the witness board it did: a B.Cu track
    landed 0.2417 mm from the nRESET fanout hole against the 0.25 mm rule, and an
    otherwise clean zero-unconnected leaf round was discarded for it.
    """
    dia, drill = fanout_via()
    ring = (dia - drill) / 2.0
    hole_to_copper = Rules().hole_clearance_mm
    assert NETCLASS_CLEARANCE_MM + ring >= hole_to_copper, (
        f"fanout via {dia}/{drill} has a {ring:.4f} mm ring; it needs at least "
        f"{hole_to_copper - NETCLASS_CLEARANCE_MM:.4f} mm or FreeRouting will "
        "route hole-clearance violations it cannot see"
    )


def test_capability_profile_is_within_the_verified_fab_limits():
    """JLCPCB 2-layer 1 oz, checked 2026-07-23: 0.10 mm track/space minimum,
    0.15 mm via hole, 0.25 mm via diameter. Ours must sit at or above those."""
    floors = fab_floors()
    assert floors["track_mm"] >= 0.10
    assert floors["clearance_mm"] >= 0.10
    dia, drill = fanout_via()
    assert dia >= 0.25
    assert drill >= 0.15
    assert drill < dia
    # 0.127 mm is exactly 127 um: no whole-micron DSN rounding trap.
    assert round(floors["track_mm"] * 1000, 6) == 127.0
    assert round(floors["clearance_mm"] * 1000, 6) == 127.0


def test_synthesis_floors_mirror_the_capability_profile():
    from kicraft.design.synthesis.kicad_pro import DEFAULT_RULES

    floors = fab_floors()
    dia, drill = fanout_via()
    assert DEFAULT_RULES["min_track_width"] == floors["track_mm"]
    assert DEFAULT_RULES["min_clearance"] == floors["clearance_mm"]
    # The fanout via must be DRC-legal on a generated board, ring included.
    assert DEFAULT_RULES["min_via_diameter"] <= dia
    assert DEFAULT_RULES["min_via_annular_width"] <= (dia - drill) / 2.0 + 1e-9
    assert DEFAULT_RULES["min_through_hole_diameter"] <= drill


def test_autoplacer_defaults_track_the_capability_profile():
    from kicraft.autoplacer.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["fab_capability"] == FAB_CAPABILITY
    floors = fab_floors()
    assert DEFAULT_CONFIG["freerouting_min_clearance_mm"] == floors["clearance_mm"]
    assert DEFAULT_CONFIG["freerouting_fine_pitch_track_mm"] == floors["track_mm"]
    # Scope guard: the netclass via is NOT the fanout via.
    assert DEFAULT_CONFIG["via_size_mm"] == 0.6
    assert DEFAULT_CONFIG["via_drill_mm"] == 0.3


def test_stamp_fab_floors_only_lowers(tmp_path):
    from kicraft.autoplacer.fab_profile import stamp_fab_floors_into_pro

    pro = tmp_path / "p.kicad_pro"
    pro.write_text(
        json.dumps(
            {
                "board": {
                    "design_settings": {
                        "rules": {
                            "min_clearance": 0.153,
                            "min_track_width": 0.153,
                            "min_via_diameter": 0.508,
                            "min_via_annular_width": 0.127,
                            "min_copper_edge_clearance": 0.2,
                        }
                    }
                }
            }
        )
    )
    assert stamp_fab_floors_into_pro(pro) is True
    rules = json.loads(pro.read_text())["board"]["design_settings"]["rules"]
    dia, drill = fanout_via()
    assert rules["min_clearance"] == fab_floors()["clearance_mm"]
    assert rules["min_via_diameter"] == dia
    assert rules["min_via_annular_width"] == pytest.approx((dia - drill) / 2.0)
    # Untouched rules survive, and a second pass is a no-op.
    assert rules["min_copper_edge_clearance"] == 0.2
    assert stamp_fab_floors_into_pro(pro) is False


def test_stamp_fab_floors_never_widens_a_stricter_project(tmp_path):
    from kicraft.autoplacer.fab_profile import stamp_fab_floors_into_pro

    pro = tmp_path / "p.kicad_pro"
    pro.write_text(
        json.dumps(
            {"board": {"design_settings": {"rules": {"min_clearance": 0.05}}}}
        )
    )
    stamp_fab_floors_into_pro(pro)
    rules = json.loads(pro.read_text())["board"]["design_settings"]["rules"]
    assert rules["min_clearance"] == 0.05
