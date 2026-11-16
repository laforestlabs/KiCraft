"""Regression coverage for packing independent owned supply banks."""

from itertools import combinations

import pytest

from kicraft.autoplacer.brain.placement_solver import (
    LocalBypassPlacementError,
    PlacementSolver,
    local_contract_violations,
)
from kicraft.autoplacer.brain.types import BoardState, Component, Layer, Pad, Point


def test_independent_joint_supply_banks_stay_compact_on_wide_canvas():
    margin = 2.0
    clearance = 2.84
    family_gap = 0.2
    reach = 3.0
    owner_width = 2.95
    cap_width = 3.01
    components = []
    specs = []
    families = {}

    def pad(ref, pin, x, y, net):
        return Pad(ref=ref, pad_id=pin, pos=Point(x, y), net=net, layer=Layer.FRONT)

    for index in (1, 2):
        owner_ref = f"U{index}"
        input_net, output_net = f"VIN{index}", f"VOUT{index}"
        components.append(Component(
            ref=owner_ref, value="ME6211", pos=Point(100, 40), rotation=0,
            layer=Layer.FRONT, width_mm=owner_width, height_mm=1.61, kind="ic",
            body_center=Point(100, 40),
            pads=[pad(owner_ref, "1", 99.05, 41.15, input_net),
                  pad(owner_ref, "2", 100, 41.15, "GND"),
                  pad(owner_ref, "5", 99.05, 38.85, output_net)],
        ))
        families[owner_ref] = index
        for cap_index, power_pin, net in ((1, "1", input_net), (2, "5", output_net)):
            ref = f"C{2 * (index - 1) + cap_index}"
            components.append(Component(
                ref=ref, value="1uF", pos=Point(150, 60), rotation=0,
                layer=Layer.FRONT, width_mm=cap_width, height_mm=1.51,
                kind="passive", body_center=Point(150, 60),
                pads=[pad(ref, "1", 149.225, 60, net),
                      pad(ref, "2", 150.775, 60, "GND")],
            ))
            families[ref] = index
            specs.append({
                "owner_ref": owner_ref, "bypass_ref": ref,
                "power_pins": [power_pin, "1"], "ground_pins": [["2"], "2"],
                "max_distance_mm": reach,
            })

    state = BoardState(
        components={component.ref: component for component in components},
        board_outline=(Point(0, 0), Point(200, 80)),
    )
    solver = PlacementSolver(state, {
        "local_bypasses": specs, "edge_margin_mm": margin,
        "placement_clearance_mm": clearance, "legality_clearance_mm": family_gap,
        "courtyard_overlap_min_gap_mm": family_gap,
    })
    solver._reserve_local_bypasses(state.components)

    assert local_contract_violations(state, specs) == []
    placed = list(state.components.values())
    for component in placed:
        low, high = component.physical_bbox()
        assert margin - 1e-9 <= low.x <= high.x <= 200 - margin + 1e-9
        assert margin - 1e-9 <= low.y <= high.y <= 80 - margin + 1e-9

    for a, b in combinations(placed, 2):
        gap = family_gap if families[a.ref] == families[b.ref] else clearance
        a_low, a_high = a.bbox(gap / 2)
        b_low, b_high = b.bbox(gap / 2)
        assert (a_high.x <= b_low.x + 1e-9 or b_high.x <= a_low.x + 1e-9
                or a_high.y <= b_low.y + 1e-9 or b_high.y <= a_low.y + 1e-9)

    bounds = [component.physical_bbox() for component in placed]
    span = max(high.x for _, high in bounds) - min(low.x for low, _ in bounds)
    # Two local neighborhoods, not opposite ends of the 200 mm canvas.
    assert span <= 2 * (2 * reach + owner_width + cap_width + clearance)


def test_nested_bank_reconsiders_parent_member_for_fixed_dependent():
    def component(ref, x, pads, *, locked=False):
        return Component(
            ref=ref, value=ref, pos=Point(x, 3), rotation=0,
            layer=Layer.FRONT, width_mm=0.4, height_mm=0.4, kind="passive",
            body_center=Point(x, 3), locked=locked,
            pads=[Pad(ref=ref, pad_id=pin, pos=Point(x, 3), net=net,
                      layer=Layer.FRONT) for pin, net in pads],
        )

    parts = [
        component("U1", 3, [("1", "A")], locked=True),
        component("L1", 4.2, [("1", "A"), ("2", "B")]),
        component("C1", 8, [("1", "B")], locked=True),
    ]
    specs = [
        {"owner_ref": "U1", "bypass_ref": "L1",
         "pin_pairs": [["1", "1"]], "max_distance_mm": 4.5},
        {"owner_ref": "L1", "bypass_ref": "C1",
         "pin_pairs": [["2", "1"]], "max_distance_mm": 1.0},
    ]
    state = BoardState(components={part.ref: part for part in parts},
                       board_outline=(Point(0, 0), Point(12, 6)))
    solver = PlacementSolver(state, {
        "local_bypasses": specs, "edge_margin_mm": 0.5,
        "placement_clearance_mm": 2.84, "legality_clearance_mm": 0.2,
        "courtyard_overlap_min_gap_mm": 0.2,
    })

    solver._reserve_local_bypasses(state.components)

    assert local_contract_violations(state, specs) == []
    assert state.components["U1"].pos == Point(3, 3)
    assert state.components["C1"].pos == Point(8, 3)
    for first, second in combinations(state.components.values(), 2):
        first_low, first_high = first.bbox(0.1)
        second_low, second_high = second.bbox(0.1)
        assert (first_high.x <= second_low.x + 1e-9
                or second_high.x <= first_low.x + 1e-9
                or first_high.y <= second_low.y + 1e-9
                or second_high.y <= first_low.y + 1e-9)


def test_nested_family_reserves_siblings_without_moving_fixed_dependent():
    def part(ref, x, net, *, locked=False):
        return Component(
            ref=ref, value=ref, pos=Point(x, 3), rotation=0,
            layer=Layer.FRONT, width_mm=0.4, height_mm=0.4, kind="passive",
            body_center=Point(x, 3), locked=locked,
            pads=[Pad(ref=ref, pad_id="1", pos=Point(x, 3), net=net,
                      layer=Layer.FRONT)],
        )

    owner = part("U1", 3, "A", locked=True)
    inductor = part("L1", 4.2, "A")
    inductor.pads.append(Pad(ref="L1", pad_id="2", pos=Point(4.2, 3),
                             net="B", layer=Layer.FRONT))
    siblings = [part(f"C{index}", 1.5, "A") for index in range(2, 5)]
    dependent = part("C1", 8, "B", locked=True)
    specs = [
        {"owner_ref": "U1", "bypass_ref": "L1",
         "pin_pairs": [["1", "1"]], "max_distance_mm": 4.5},
        *({"owner_ref": "U1", "bypass_ref": cap.ref,
           "pin_pairs": [["1", "1"]], "max_distance_mm": 1.5}
          for cap in siblings),
        {"owner_ref": "L1", "bypass_ref": "C1",
         "pin_pairs": [["2", "1"]], "max_distance_mm": 1.0},
    ]
    state = BoardState(
        components={part.ref: part for part in [owner, inductor, *siblings, dependent]},
        board_outline=(Point(0, 0), Point(12, 6)),
    )
    solver = PlacementSolver(state, {
        "local_bypasses": specs, "edge_margin_mm": 0.5,
        "placement_clearance_mm": 2.84, "legality_clearance_mm": 0.2,
        "courtyard_overlap_min_gap_mm": 0.2,
    })
    solver._reserve_local_bypasses(state.components)

    assert local_contract_violations(state, specs) == []
    assert state.components["C1"].pos == Point(8, 3)
    assert all(state.components[ref].locked for ref in state.components)


def test_failed_nested_reservation_restores_poses_and_locks():
    def part(ref, x, net, locked=False):
        return Component(
            ref=ref, value=ref, pos=Point(x, 2), rotation=0,
            layer=Layer.FRONT, width_mm=0.4, height_mm=0.4,
            kind="passive", body_center=Point(x, 2), locked=locked,
            pads=[Pad(ref=ref, pad_id="1", pos=Point(x, 2), net=net,
                      layer=Layer.FRONT)],
        )

    owner = part("U1", 2, "A", True)
    intermediate = part("L1", 3, "A")
    intermediate.pads.append(Pad(ref="L1", pad_id="2", pos=Point(3, 2),
                                 net="B", layer=Layer.FRONT))
    dependent = part("C1", 7, "B", True)
    specs = [
        {"owner_ref": "U1", "bypass_ref": "L1",
         "pin_pairs": [["1", "1"]], "max_distance_mm": 0.7},
        {"owner_ref": "L1", "bypass_ref": "C1",
         "pin_pairs": [["2", "1"]], "max_distance_mm": 0.7},
    ]
    parts = [owner, intermediate, dependent]
    state = BoardState(components={part.ref: part for part in parts},
                       board_outline=(Point(0, 0), Point(9, 4)))
    solver = PlacementSolver(state, {
        "local_bypasses": specs, "edge_margin_mm": 0.2,
        "legality_clearance_mm": 0.2,
    })

    with pytest.raises(LocalBypassPlacementError):
        solver._place_local_bypasses(state.components, reserve=True)

    assert intermediate.pos == Point(3, 2)
    assert intermediate.pads[1].pos == Point(3, 2)
    assert [part.locked for part in parts] == [True, False, True]


def _channel_part(ref, x, pads, *, locked=False):
    return Component(
        ref=ref, value=ref, pos=Point(x, 1), rotation=0,
        layer=Layer.FRONT, width_mm=0.4, height_mm=0.4,
        kind="passive", body_center=Point(x, 1), locked=locked,
        allowed_rotations=[0],
        pads=[Pad(ref=ref, pad_id=pin, pos=Point(x + offset, 1),
                  net=net, layer=Layer.FRONT)
              for pin, offset, net in pads],
    )


def _channel_solver(parts, specs, width):
    state = BoardState(
        components={part.ref: part for part in parts},
        board_outline=(Point(0, 0), Point(width, 2)),
    )
    solver = PlacementSolver(state, {
        "local_bypasses": specs, "edge_margin_mm": 0.8,
        "placement_clearance_mm": 0.2, "legality_clearance_mm": 0.2,
        "courtyard_overlap_min_gap_mm": 0.2,
    })
    return state, solver


def _assert_channel_geometry(state, specs, width):
    assert local_contract_violations(state, specs) == []
    for spec in specs:
        owner = state.components[spec["owner_ref"]]
        member = state.components[spec["bypass_ref"]]
        for owner_pin, member_pin in spec["pin_pairs"]:
            owner_pad = next(p for p in owner.pads if p.pad_id == owner_pin)
            member_pad = next(p for p in member.pads if p.pad_id == member_pin)
            assert owner_pad.net == member_pad.net
            assert owner_pad.pos.dist(member_pad.pos) <= spec["max_distance_mm"] + 1e-9
    for part in state.components.values():
        low, high = part.physical_bbox()
        assert 0.8 - 1e-9 <= low.x <= high.x <= width - 0.8 + 1e-9
        assert 0.8 - 1e-9 <= low.y <= high.y <= 1.2 + 1e-9
        assert part.locked
    for first, second in combinations(state.components.values(), 2):
        first_low, first_high = first.bbox(0.1)
        second_low, second_high = second.bbox(0.1)
        assert (first_high.x <= second_low.x + 1e-9
                or second_high.x <= first_low.x + 1e-9
                or first_high.y <= second_low.y + 1e-9
                or second_high.y <= first_low.y + 1e-9)


def test_nested_movable_banks_share_remaining_channel_space():
    owner = _channel_part("U1", 1, [("1", 0.2, "A")], locked=True)
    dependent = _channel_part("L1", 2.3, [("1", 0.2, "A"), ("2", -0.2, "B")],
                              locked=True)
    parent_cap = _channel_part("C1", 5, [("1", 0, "A")])
    child_cap = _channel_part("C2", 5, [("1", 0, "B")])
    specs = [
        {"owner_ref": "U1", "bypass_ref": "L1",
         "pin_pairs": [["1", "1"]], "max_distance_mm": 1.4},
        {"owner_ref": "U1", "bypass_ref": "C1",
         "pin_pairs": [["1", "1"]], "max_distance_mm": 0.5},
        {"owner_ref": "L1", "bypass_ref": "C2",
         "pin_pairs": [["2", "1"]], "max_distance_mm": 1.5},
    ]
    state, solver = _channel_solver([owner, dependent, parent_cap, child_cap],
                                    specs, 6)

    solver._reserve_local_bypasses(state.components)

    _assert_channel_geometry(state, specs, 6)
    assert state.components["U1"].pos == Point(1, 1)
    assert state.components["L1"].pos == Point(2.3, 1)
    assert state.components["C2"].pos.x > state.components["L1"].pos.x


def test_nested_moving_anchor_reconsidered_with_both_banks_movable():
    owner = _channel_part("U1", 2, [("1", 0, "A")], locked=True)
    dependent = _channel_part("L1", 2.6, [("1", 0, "A"), ("2", 0, "B")])
    parent_cap = _channel_part("C1", 6, [("1", 0, "A")])
    child_cap = _channel_part("C2", 1, [("1", 0, "B")])
    fixed_cap = _channel_part("C3", 5.4, [("1", 0, "B")], locked=True)
    specs = [
        {"owner_ref": "U1", "bypass_ref": "L1",
         "pin_pairs": [["1", "1"]], "max_distance_mm": 3.5},
        {"owner_ref": "U1", "bypass_ref": "C1",
         "pin_pairs": [["1", "1"]], "max_distance_mm": 1.2},
        {"owner_ref": "L1", "bypass_ref": "C2",
         "pin_pairs": [["2", "1"]], "max_distance_mm": 1.2},
        {"owner_ref": "L1", "bypass_ref": "C3",
         "pin_pairs": [["2", "1"]], "max_distance_mm": 0.8},
    ]
    state, solver = _channel_solver(
        [owner, dependent, parent_cap, child_cap, fixed_cap], specs, 7.5)

    solver._reserve_local_bypasses(state.components)

    _assert_channel_geometry(state, specs, 7.5)
    assert state.components["U1"].pos == Point(2, 1)
    assert state.components["C3"].pos == Point(5.4, 1)
    assert state.components["L1"].pos.x >= 4.6 - 1e-9
    assert state.components["L1"].pos != Point(2.6, 1)


def test_shared_member_satisfies_both_owners():
    first = _channel_part("U1", 2, [("1", 0, "A")], locked=True)
    second = _channel_part("U2", 4, [("1", 0, "A")], locked=True)
    shared = _channel_part("C1", 5, [("1", 0, "A")])
    specs = [
        {"owner_ref": "U1", "bypass_ref": "C1",
         "pin_pairs": [["1", "1"]], "max_distance_mm": 1.2},
        {"owner_ref": "U2", "bypass_ref": "C1",
         "pin_pairs": [["1", "1"]], "max_distance_mm": 1.2},
    ]
    state, solver = _channel_solver([first, second, shared], specs, 6)

    solver._reserve_local_bypasses(state.components)

    _assert_channel_geometry(state, specs, 6)
    assert state.components["U1"].pos == Point(2, 1)
    assert state.components["U2"].pos == Point(4, 1)


def test_failed_moving_family_restores_every_pose_and_lock():
    owner = _channel_part("U1", 2, [("1", 0, "A")], locked=True)
    dependent = _channel_part("L1", 2.6, [("1", 0, "A"), ("2", 0, "B")])
    parent_cap = _channel_part("C1", 6, [("1", 0, "A")])
    child_cap = _channel_part("C2", 1, [("1", 0, "B")])
    fixed_cap = _channel_part("C3", 5.4, [("1", 0, "B")], locked=True)
    specs = [
        {"owner_ref": "U1", "bypass_ref": "L1",
         "pin_pairs": [["1", "1"]], "max_distance_mm": 3.5},
        {"owner_ref": "U1", "bypass_ref": "C1",
         "pin_pairs": [["1", "1"]], "max_distance_mm": 1.2},
        {"owner_ref": "L1", "bypass_ref": "C2",
         "pin_pairs": [["2", "1"]], "max_distance_mm": 0.4},
        {"owner_ref": "L1", "bypass_ref": "C3",
         "pin_pairs": [["2", "1"]], "max_distance_mm": 0.8},
    ]
    state, solver = _channel_solver(
        [owner, dependent, parent_cap, child_cap, fixed_cap], specs, 7.5)

    def snapshot():
        return {
            ref: ((part.pos.x, part.pos.y),
                  (part.body_center.x, part.body_center.y),
                  part.rotation, part.locked,
                  tuple((pad.pos.x, pad.pos.y) for pad in part.pads))
            for ref, part in state.components.items()
        }

    before = snapshot()
    with pytest.raises(LocalBypassPlacementError):
        solver._reserve_local_bypasses(state.components)
    assert snapshot() == before
