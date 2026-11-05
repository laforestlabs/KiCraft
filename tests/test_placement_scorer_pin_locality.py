"""Tests for PlacementScorer._score_pin_locality -- the connectivity-first
objective that pulls each 2-pad passive against the anchor pin pair it bridges.

Drives the sub-metric directly (and via ``score()`` for the byte-identical
early-out) so a regression here is not masked by another sub-metric. Shares the
single-source ``pin_locality_for_passive`` kernel with the metric module, so
these also guard that the scorer's view agrees with ``leaf_pin_locality``.
"""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Pad,
    Point,
)


def _pad(owner: str, pad_id: str, x: float, y: float, net: str) -> Pad:
    return Pad(ref=owner, pad_id=pad_id, pos=Point(x, y), net=net, layer=Layer.FRONT)


def _ic(ref: str, x: float, y: float, pads: list[Pad]) -> Component:
    return Component(
        ref=ref, value="", pos=Point(x, y), rotation=0.0, layer=Layer.FRONT,
        width_mm=6.0, height_mm=6.0, kind="ic", pads=pads, body_center=Point(x, y),
    )


def _cap(ref: str, x: float, y: float, pads: list[Pad], rot: float = 0.0) -> Component:
    return Component(
        ref=ref, value="", pos=Point(x, y), rotation=rot, layer=Layer.FRONT,
        width_mm=2.0, height_mm=1.0, kind="passive", pads=pads, body_center=Point(x, y),
    )


def _state(comps: list[Component]) -> BoardState:
    return BoardState(
        components={c.ref: c for c in comps},
        board_outline=(Point(0.0, 0.0), Point(100.0, 100.0)),
    )


# U1 power pins: +3V3 at (0,0), GND at (0,2).
def _u1() -> Component:
    return _ic("U1", -3, 1, [_pad("U1", "1", 0.0, 0.0, "+3V3"), _pad("U1", "2", 0.0, 2.0, "GND")])


def _decap(ref: str, x: float) -> Component:
    """A decap straddling +3V3/GND, offset ``x`` mm to the right of U1's pins."""
    return _cap(ref, x, 1, [_pad(ref, "1", x, 0.0, "+3V3"), _pad(ref, "2", x, 2.0, "GND")])


def test_early_out_neutral_when_unweighted():
    # No psw_pin_locality -> the term is a no-op (100), scoring byte-identical.
    scorer = PlacementScorer(_state([_u1(), _decap("C1", 8.0)]))
    assert scorer._score_pin_locality() == pytest.approx(100.0)


def test_closer_cap_scores_higher():
    cfg = {"psw_pin_locality": 0.2}
    near = PlacementScorer(_state([_u1(), _decap("C1", 0.5)]), cfg)._score_pin_locality()
    far = PlacementScorer(_state([_u1(), _decap("C2", 10.0)]), cfg)._score_pin_locality()
    assert near > far
    assert far < 100.0  # a 10 mm decap is not "local"


def test_neutral_when_no_anchors():
    # Two passives, no IC/connector anchor -> nothing to be local to.
    cfg = {"psw_pin_locality": 0.2}
    comps = [_decap("C1", 1.0), _decap("C2", 3.0)]
    assert PlacementScorer(_state(comps), cfg)._score_pin_locality() == pytest.approx(100.0)


def test_total_penalizes_a_far_decap_when_weighted():
    # With the term weighted, the full weighted total must drop for a far decap.
    cfg = {"psw_pin_locality": 0.3}
    near = PlacementScorer(_state([_u1(), _decap("C1", 0.5)]), cfg).score().total
    far = PlacementScorer(_state([_u1(), _decap("C2", 12.0)]), cfg).score().total
    assert near > far


def test_weight_toggles_total_on_same_state():
    # On a FIXED placement, turning the term ON must move the total (a far decap
    # is penalized); OFF it is neutral. Isolates the term from geometry changes
    # (comparing two different placements would also move net_distance etc.).
    state = _state([_u1(), _decap("C1", 12.0)])
    off = PlacementScorer(state).score().total
    on = PlacementScorer(state, {"psw_pin_locality": 0.3}).score().total
    assert on < off


def test_owned_bypasses_follow_their_own_ic_on_a_shared_rail():
    from kicraft.autoplacer.brain.placement_solver import PlacementSolver

    def sensor(ref, x):
        return Component(
            ref=ref, value="TMP102", pos=Point(x, 20), rotation=0, layer=Layer.FRONT,
            width_mm=2.79, height_mm=2.29, body_center=Point(x, 20),
            pads=[_pad(ref, "5", x - 0.74, 20, "3V3"),
                  _pad(ref, "2", x + 0.74, 20, "GND")],
        )

    def bypass(ref, x):
        return Component(
            ref=ref, value="100nF", pos=Point(x, 30), rotation=0, layer=Layer.FRONT,
            width_mm=1.55, height_mm=3.05, body_center=Point(x, 30),
            pads=[_pad(ref, "1", x, 30.775, "3V3"),
                  _pad(ref, "2", x, 29.225, "GND")],
        )

    state = _state([sensor("U1", 20), sensor("U2", 40),
                    bypass("C1", 40), bypass("C2", 20)])
    specs = [
        {"owner_ref": f"U{i}", "bypass_ref": f"C{i}",
         "power_pins": ["5", "1"], "ground_pins": ["2", "2"],
         "max_distance_mm": 2.5}
        for i in (1, 2)
    ]
    solver = PlacementSolver(state, {
        "local_bypasses": specs, "legality_clearance_mm": 0,
        "courtyard_overlap_min_gap_mm": 0.15,
    })
    solver._place_local_bypasses(state.components)
    for i in (1, 2):
        owner = {p.pad_id: p for p in state.components[f"U{i}"].pads}
        cap = {p.pad_id: p for p in state.components[f"C{i}"].pads}
        assert owner["5"].pos.dist(cap["1"].pos) <= 2.5
        assert owner["2"].pos.dist(cap["2"].pos) <= 2.5



@pytest.mark.parametrize("recipe", [
    "stm32g030f6p6-minimal@1", "stm32g030k6t6-minimal@2",
])
def test_stm32g0_local_bypass_does_not_excuse_distant_bulk_capacitor(recipe):
    from types import SimpleNamespace

    from kicraft.autoplacer.brain.placement_solver import PlacementSolver
    from kicraft.design.models import BOM, RecipeSelection
    from kicraft.design.recipes.registry import expand_recipe
    from kicraft.design.synthesis.autoplacer import _recipe_local_bypasses

    selection = RecipeSelection(
        recipe=recipe, instance="core", sheets={"mcu": "MCU"},
        port_bindings={"vdd": "+3V3", "gnd": "GND"},
    )
    expansion = expand_recipe(selection)
    bom = BOM(parts=expansion.parts, connections=expansion.connections,
              recipe_ownership=[expansion.ownership])
    roles = {part.recipe_role: part.ref for part in expansion.parts}
    owner_ref = roles["mcu"]
    owner = _ic(owner_ref, 20, 20.5, [
        _pad(owner_ref, "4", 23, 20, "+3V3"),
        _pad(owner_ref, "5", 23, 21, "GND"),
    ])
    caps = []
    for role, x in (("local_decoupling", 25), ("bulk_decoupling", 70)):
        ref = roles[role]
        caps.append(_cap(ref, x, 20.5, [
            _pad(ref, "1", x, 20, "+3V3"),
            _pad(ref, "2", x, 21, "GND"),
        ]))
    state = _state([owner, *caps])
    cfg = {
        "local_bypasses": _recipe_local_bypasses(
            SimpleNamespace(recipe_selections=[selection]), bom),
        "legality_clearance_mm": 0,
        "courtyard_overlap_min_gap_mm": 0.15,
    }
    PlacementSolver(state, cfg)._place_local_bypasses(state.components)
    owner_pads = {pad.pad_id: pad for pad in state.components[owner_ref].pads}
    for cap in caps:
        pads = {pad.pad_id: pad for pad in state.components[cap.ref].pads}
        assert owner_pads["4"].pos.dist(pads["1"].pos) <= 3
        assert owner_pads["5"].pos.dist(pads["2"].pos) <= 3


def test_single_edge_header_does_not_strand_a_feasible_owned_mcu_group():
    from kicraft.autoplacer.brain.placement_solver import PlacementSolver

    center = 26.29 / 2
    owner = Component(
        ref="U1", value="MCU", pos=Point(center, center), rotation=0,
        layer=Layer.FRONT, width_mm=10.45, height_mm=10.45,
        kind="ic", body_center=Point(center, center),
        pads=[_pad("U1", "4", center - 4.4, center - 0.4, "+3V3"),
              _pad("U1", "5", center - 4.4, center + 0.4, "GND")],
    )
    header = Component(
        ref="J1", value="SWD", pos=Point(center, center), rotation=0,
        layer=Layer.FRONT, width_mm=2.59, height_mm=15.33,
        kind="connector", body_center=Point(center, center),
        is_through_hole=True, mating_axis="board_normal",
        pads=[_pad("J1", str(i + 1), center, center + (i - 2.5) * 2.54, net)
              for i, net in enumerate(("+3V3", "GND", "SWDIO", "SWCLK", "NRST", ""))],
    )
    caps = [
        Component(
            ref=ref, value=value, pos=Point(18, y), rotation=0,
            layer=Layer.FRONT, width_mm=3.05, height_mm=1.55,
            kind="passive", body_center=Point(18, y),
            pads=[_pad(ref, "1", 17.225, y, "+3V3"),
                  _pad(ref, "2", 18.775, y, "GND")],
        )
        for ref, value, y in (("C1", "100nF", 5), ("C2", "4.7uF", 8))
    ]
    state = _state([owner, header, *caps])
    state.board_outline = (Point(0, 0), Point(26.29, 26.29))
    solver = PlacementSolver(state, {
        "component_zones": {"J1": {"edge": "bottom"}},
        "connector_perp_orientation": True,
        "connector_edge_inset_mm": 2,
        "edge_margin_mm": 2,
        "placement_clearance_mm": 2.84,
        "legality_clearance_mm": 0.2,
        "courtyard_overlap_min_gap_mm": 0.2,
        "local_bypasses": [
            {"owner_ref": "U1", "bypass_ref": cap.ref,
             "power_pins": ["4", "1"], "ground_pins": [["5"], "2"],
             "max_distance_mm": 3}
            for cap in caps
        ],
    }, seed=9)
    solver._pin_edge_components(state.components)
    solver._reserve_local_bypasses(state.components)
    owner_pads = {pad.pad_id: pad for pad in state.components["U1"].pads}
    for cap in caps:
        pads = {pad.pad_id: pad for pad in state.components[cap.ref].pads}
        assert owner_pads["4"].pos.dist(pads["1"].pos) <= 3
        assert owner_pads["5"].pos.dist(pads["2"].pos) <= 3
    header = state.components["J1"]
    assert header.mating_axis == "board_normal"
    assert header.bbox()[1].y == pytest.approx(24.29)

@pytest.mark.parametrize("rotations", [(0, 0, 0, 0), (0, 90, 180, 270)])
def test_four_hall_owned_bypasses_fit_despite_overlapping_soft_poses(rotations):
    from itertools import combinations

    from kicraft.autoplacer.brain.placement_solver import PlacementSolver
    from kicraft.autoplacer.brain.geometry import rotate_component_in_place

    def group(i, x, y, cap_y):
        owner_ref, cap_ref = f"U{i}", f"C{i}"
        # AH49E manufacturer footprint pads and the native extracted 4.19 x
        # 3.19 courtyard; 0603 native geometry matches the helpers above.
        sensor = Component(
            ref=owner_ref, value="AH49ENTR-G1", pos=Point(x, y), rotation=0,
            layer=Layer.FRONT, width_mm=4.19, height_mm=3.19, kind="ic",
            body_center=Point(x, y),
            pads=[Pad(ref=owner_ref, pad_id=pin, pos=Point(x + dx, y + dy),
                      net=net, layer=Layer.FRONT, size_mm=Point(1.0, 0.7))
                  for pin, dx, dy, net in (
                      ("1", 1.3, 0.95, "+3V3"),
                      ("2", 1.3, -0.95, f"OUT{i}"),
                      ("3", -1.3, 0.0, "GND"),
                  )],
        )
        cap = Component(
            ref=cap_ref, value="100nF", pos=Point(x, cap_y), rotation=180,
            layer=Layer.FRONT, width_mm=3.05, height_mm=1.55, kind="passive",
            body_center=Point(x, cap_y),
            pads=[Pad(ref=cap_ref, pad_id=pin, pos=Point(x + dx, cap_y),
                      net=net, layer=Layer.FRONT, size_mm=Point(0.9, 0.95))
                  for pin, dx, net in (
                      ("1", 0.775, "+3V3"), ("2", -0.775, "GND"),
                  )],
        )
        return sensor, cap

    specs = [
        {"owner_ref": f"U{i}", "bypass_ref": f"C{i}",
         "power_pins": ["1", "1"], "ground_pins": ["3", "2"],
         "max_distance_mm": 3.0}
        for i in range(1, 5)
    ]

    def assert_legal(comps):
        for component in comps.values():
            tl, br = component.physical_bbox()
            assert tl.x >= 2.0 and tl.y >= 2.0
            assert br.x <= 15.72 and br.y <= 15.72
        for i in range(1, 5):
            sensor = {pad.pad_id: pad for pad in comps[f"U{i}"].pads}
            cap = {pad.pad_id: pad for pad in comps[f"C{i}"].pads}
            assert sensor["1"].pos.dist(cap["1"].pos) <= 3.0
            assert sensor["3"].pos.dist(cap["2"].pos) <= 3.0
        for a, b in combinations(comps.values(), 2):
            # Native solver applies the placement clearance across reserved
            # groups, but the legality gap within an owner/bypass pair.
            gap = 0.2 if a.ref[1:] == b.ref[1:] else 2.84
            at, ab = a.bbox(gap / 2)
            bt, bb = b.bbox(gap / 2)
            assert ab.x <= bt.x + 1e-9 or bb.x <= at.x + 1e-9 or (
                ab.y <= bt.y + 1e-9 or bb.y <= at.y + 1e-9
            ), (a.ref, b.ref)

    # Concrete feasible placement, independently checked before the solver:
    # owner centers (4.1,3.7), (12.2,3.7), (4.1,11.8), (12.2,11.8);
    # each cap is 2.8 mm below its owner with 180-degree rotation.
    witness = [component
               for i, (x, y) in enumerate(
                   ((4.1, 3.7), (12.2, 3.7), (4.1, 11.8), (12.2, 11.8)), 1)
               for component in group(i, x, y, y + 2.8)]
    assert_legal({component.ref: component for component in witness})

    components = [component for i in range(1, 5)
                  for component in group(i, 8.86, 8.86, 8.86)]
    for i, rotation in enumerate(rotations):
        rotate_component_in_place(components[2 * i], rotation)
    state = BoardState(
        components={component.ref: component for component in components},
        board_outline=(Point(0, 0), Point(17.72, 17.72)),
    )
    solver = PlacementSolver(state, {
        "local_bypasses": specs, "edge_margin_mm": 2.0,
        "placement_clearance_mm": 2.84, "legality_clearance_mm": 0.2,
        "courtyard_overlap_min_gap_mm": 0.2,
    })
    solver._reserve_local_bypasses(state.components)
    assert_legal(state.components)

def test_owned_bypass_refuses_when_legal_courtyard_cannot_fit():
    from kicraft.autoplacer.brain.placement_solver import PlacementSolver
    import json

    owner = _ic("U1", 20, 20, [
        _pad("U1", "5", 20, 20, "3V3"), _pad("U1", "2", 20, 21, "GND"),
    ])
    cap = _cap("C1", 40, 40, [
        _pad("C1", "1", 40, 40, "3V3"), _pad("C1", "2", 40, 41, "GND"),
    ])
    state = _state([owner, cap])
    solver = PlacementSolver(state, {"local_bypasses": [{
        "owner_ref": "U1", "bypass_ref": "C1",
        "power_pins": ["5", "1"], "ground_pins": ["2", "2"],
        "max_distance_mm": 0.5,
    }]})
    with pytest.raises(ValueError, match="no legal placement") as exc:
        solver._place_local_bypasses(state.components)
    evidence = json.loads(str(exc.value).split("evidence=", 1)[1])
    assert evidence["owner"]["ref"] == "U1" and evidence["cap"]["ref"] == "C1"
    assert evidence["power_distance_mm"] > evidence["max_distance_mm"]
    assert {p["net"] for p in evidence["cap"]["pads"]} == {"3V3", "GND"}
    assert evidence["board_bbox_mm"] == [0.0, 0.0, 100.0, 100.0]


def test_two_owned_caps_find_joint_slots_and_nearest_reviewed_ground():
    from kicraft.autoplacer.brain.placement_solver import PlacementSolver

    owner = Component(
        ref="U1", value="S3", pos=Point(20, 20), rotation=0, layer=Layer.FRONT,
        width_mm=15.49, height_mm=15.49, body_center=Point(20, 20),
        pads=[_pad("U1", "3", 15.75, 27, "3V3"),
              _pad("U1", "2", 14.9, 27, "GND"),
              _pad("U1", "43", 15.75, 13, "GND")],
    )
    caps = [
        # Both initially satisfy pin distances but overlap one another.
        Component(ref=f"C{i}", value="100nF", pos=Point(15.3, 28.8),
                  rotation=0, layer=Layer.FRONT, width_mm=3.05, height_mm=1.55,
                  body_center=Point(15.3, 28.8),
                  pads=[_pad(f"C{i}", "1", 16.075, 28.8, "3V3"),
                        _pad(f"C{i}", "2", 14.525, 28.8, "GND")])
        for i in (1, 2)
    ]
    state = _state([owner, *caps])
    specs = [{"owner_ref": "U1", "bypass_ref": f"C{i}",
              "power_pins": ["3", "1"], "ground_pins": [["2", "43"], "2"],
              "max_distance_mm": 3.0} for i in (1, 2)]
    solver = PlacementSolver(state, {"local_bypasses": specs,
                                    "legality_clearance_mm": 0.2})
    solver._reserve_local_bypasses(state.components)
    placed = [state.components[f"C{i}"] for i in (1, 2)]
    for cap in placed:
        pads = {p.pad_id: p for p in cap.pads}
        assert owner.pads[0].pos.dist(pads["1"].pos) <= 3.0
        assert min(p.pos.dist(pads["2"].pos) for p in owner.pads[1:]) <= 3.0
    at, ab = placed[0].bbox(0.1)
    bt, bb = placed[1].bbox(0.1)
    assert not (at.x < bb.x and ab.x > bt.x and at.y < bb.y and ab.y > bt.y)


def test_converter_input_and_output_caps_fit_together_outside_body():
    from kicraft.autoplacer.brain.placement_solver import PlacementSolver

    converter = Component(
        ref="U1", value="K7803M", pos=Point(20, 20), rotation=0, layer=Layer.FRONT,
        width_mm=9.24, height_mm=13.69, body_center=Point(17.975, 20),
        is_through_hole=True,
        pads=[_pad("U1", "1", 20, 22.54, "VIN"),
              _pad("U1", "2", 20, 20, "GND"),
              _pad("U1", "3", 20, 17.46, "3V3")],
    )
    caps = [
        Component(ref=f"C{i}", value="1206", pos=Point(40 + 5*i, 40), rotation=0,
                  layer=Layer.FRONT, width_mm=2.39, height_mm=4.69,
                  body_center=Point(40 + 5*i, 40),
                  pads=[_pad(f"C{i}", "1", 40 + 5*i, 38.525, net),
                        _pad(f"C{i}", "2", 40 + 5*i, 41.475, "GND")])
        for i, net in ((1, "VIN"), (2, "3V3"))
    ]
    state = _state([converter, *caps])
    specs = [{"owner_ref": "U1", "bypass_ref": f"C{i}",
              "power_pins": [pin, "1"], "ground_pins": [["2"], "2"],
              "max_distance_mm": 4.5} for i, pin in ((1, "1"), (2, "3"))]
    solver = PlacementSolver(state, {"local_bypasses": specs,
                                    "legality_clearance_mm": 0.2})
    solver._place_local_bypasses(state.components)
    owner_pads = {p.pad_id: p for p in converter.pads}
    for i, pin in ((1, "1"), (2, "3")):
        cap = state.components[f"C{i}"]
        pads = {p.pad_id: p for p in cap.pads}
        assert owner_pads[pin].pos.dist(pads["1"].pos) <= 4.5
        assert owner_pads["2"].pos.dist(pads["2"].pos) <= 4.5
        ct, cb = converter.bbox(0.1)
        pt, pb = cap.bbox(0.1)
        assert not (ct.x < pb.x and cb.x > pt.x and ct.y < pb.y and cb.y > pt.y)


def test_reservation_precedes_free_passive_packing_and_survives_legalizer():
    from kicraft.autoplacer.brain.placement_solver import PlacementSolver

    sensor = Component(
        ref="U1", value="TMP102", pos=Point(20, 20), rotation=0, layer=Layer.FRONT,
        width_mm=2.79, height_mm=2.29, body_center=Point(20, 20),
        pads=[_pad("U1", "5", 19.26, 20, "3V3"),
              _pad("U1", "2", 20.74, 20, "GND")],
    )
    cap = Component(
        ref="C1", value="100nF", pos=Point(40, 40), rotation=0, layer=Layer.FRONT,
        width_mm=1.55, height_mm=3.05, body_center=Point(40, 40),
        pads=[_pad("C1", "1", 40, 40.775, "3V3"),
              _pad("C1", "2", 40, 39.225, "GND")],
    )
    # An unrelated, movable component occupies the whole local search region.
    crowd = _cap("R1", 20, 20, [_pad("R1", "1", 20, 20, "OTHER")])
    crowd.width_mm = crowd.height_mm = 10
    state = _state([sensor, cap, crowd])
    cfg = {"local_bypasses": [{
        "owner_ref": "U1", "bypass_ref": "C1", "power_pins": ["5", "1"],
        "ground_pins": ["2", "2"], "max_distance_mm": 2.5,
    }], "legality_clearance_mm": 0.2, "placement_clearance_mm": 2.84}
    solver = PlacementSolver(state, cfg)
    solver._reserve_local_bypasses(state.components)
    assert state.components["U1"].locked and state.components["C1"].locked
    before = {r: (state.components[r].pos.x, state.components[r].pos.y)
              for r in ("U1", "C1")}
    solver._resolve_overlaps(state.components)
    assert {r: (state.components[r].pos.x, state.components[r].pos.y)
            for r in ("U1", "C1")} == before
    solver._place_local_bypasses(state.components)


def test_rf_overhang_owner_remains_pinned_while_local_cap_reserves_space():
    from kicraft.autoplacer.brain.placement_solver import PlacementSolver
    from kicraft.autoplacer.brain.types import AntennaEdgeIntent, KeepoutRect

    owner = Component(
        ref="U1", value="C3", pos=Point(12.621892300815766, 5.6),
        rotation=0, layer=Layer.FRONT, width_mm=13.29, height_mm=16.69,
        body_center=Point(12.621892300815766, 2.9),
        pads=[_pad("U1", "3", 6.721892300815766, 3.2, "3V3"),
              _pad("U1", "2", 6.721892300815766, 2.4, "GND"),
              Pad(ref="U1", pad_id="36", pos=Point(10, 0.6999999999999975),
                  net="GND", layer=Layer.FRONT, size_mm=Point(0.5, 0.8))],
    )
    cap = Component(
        ref="C1", value="100nF", pos=Point(40, 40), rotation=0, layer=Layer.FRONT,
        width_mm=1.55, height_mm=3.05, body_center=Point(40, 40),
        pads=[_pad("C1", "1", 40, 39.225, "3V3"),
              _pad("C1", "2", 40, 40.775, "GND")],
    )
    state = _state([owner, cap])
    state.antenna_edge_intents = [AntennaEdgeIntent(
        owner_ref="U1", source="explicit", source_id="rf", local_direction="top",
        local_anchor_mm=-5.6, local_anchor_midpoint=Point(0, -5.6),
        local_polygon=(Point(-6.6, -11), Point(6.6, -11),
                       Point(6.6, -5.6), Point(-6.6, -5.6)),
        target_edge="top",
    )]
    state.keepout_rects = [KeepoutRect(Point(6.02, -5.4), Point(19.22, 0), "U1")]
    solver = PlacementSolver(state, {"local_bypasses": [{
        "owner_ref": "U1", "bypass_ref": "C1", "power_pins": ["3", "1"],
        "ground_pins": [["2"], "2"], "max_distance_mm": 3.0,
    }], "legality_clearance_mm": 0.2})
    solver._reserve_local_bypasses(state.components)
    assert (owner.pos.x, owner.pos.y, owner.rotation) == (12.621892300815766, 5.6, 0)
    assert owner.physical_bbox()[0].y < 0  # antenna body overhang is intentional
    placed = state.components["C1"]
    assert placed.physical_bbox()[0].y >= 2.0
    assert owner.pads[0].pos.dist(placed.pads[0].pos) <= 3.0
    assert owner.pads[1].pos.dist(placed.pads[1].pos) <= 3.0
    # A real one-micrometre inset violation remains a hard refusal.
    owner.pads[2].pos.y = 0.699
    with pytest.raises(ValueError, match="antenna_pad_inset"):
        solver._reserve_local_bypasses(state.components)


def test_courtyard_repair_does_not_slide_reserved_or_other_locked_parts():
    from kicraft.autoplacer.brain.placement_solver import PlacementSolver

    owner = _ic("U1", 20, 20, [])
    owner.locked = True
    other = _cap("J1", 20, 20, [])
    other.locked = True
    cap = _cap("C1", 40, 40, [])
    cap.locked = True
    state = _state([owner, other, cap])
    solver = PlacementSolver(state, {"local_bypasses": [{
        "owner_ref": "U1", "bypass_ref": "C1",
        "power_pins": ["5", "1"], "ground_pins": ["2", "2"], "max_distance_mm": 2.5,
    }]})
    before = {ref: (c.pos.x, c.pos.y) for ref, c in state.components.items()}
    assert solver._resolve_courtyard_overlaps(state.components) > 0
    assert {ref: (c.pos.x, c.pos.y) for ref, c in state.components.items()} == before


def test_c3_rf_edge_two_front_0603_bypasses_refuse_no_joint_legal_slots():
    import json
    from kicraft.autoplacer.brain.placement_solver import PlacementSolver
    from kicraft.autoplacer.brain.types import AntennaEdgeIntent

    x = 12.429128042743468
    owner = Component(
        ref="U1", value="C3", pos=Point(x, 5.6), rotation=0, layer=Layer.FRONT,
        width_mm=13.29, height_mm=16.69, body_center=Point(x, 2.9),
        pads=[_pad("U1", "3", x - 5.9, 3.2, "3V3"),
              _pad("U1", "2", x - 5.9, 2.4, "GND"),
              _pad("U1", "1", x - 5.9, 1.6, "GND")],
    )
    caps = [
        Component(ref=f"C{i}", value="0603", pos=Point(40 + 5*i, 40), rotation=0,
                  layer=Layer.FRONT, width_mm=3.05, height_mm=1.55,
                  body_center=Point(40 + 5*i, 40),
                  pads=[_pad(f"C{i}", "1", 40 + 5*i - 0.775, 40, "3V3"),
                        _pad(f"C{i}", "2", 40 + 5*i + 0.775, 40, "GND")])
        for i in (1, 2)
    ]
    state = _state([owner, *caps])
    state.board_outline = (Point(0, 0), Point(75.01, 75.01))
    state.antenna_edge_intents = [AntennaEdgeIntent(
        owner_ref="U1", source="explicit", source_id="rf", local_direction="top",
        local_anchor_mm=-5.6, local_anchor_midpoint=Point(0, -5.6),
        local_polygon=(Point(-6.6, -11), Point(6.6, -11),
                       Point(6.6, -5.6), Point(-6.6, -5.6)), target_edge="top",
    )]
    specs = [{"owner_ref": "U1", "bypass_ref": f"C{i}",
              "power_pins": ["3", "1"], "ground_pins": [["1", "2"], "2"],
              "max_distance_mm": 3.0} for i in (1, 2)]
    solver = PlacementSolver(state, {"local_bypasses": specs,
                                    "legality_clearance_mm": 0.2})
    with pytest.raises(ValueError, match="no joint legal placement") as exc:
        solver._reserve_local_bypasses(state.components)
    # Nested owner evidence follows the inner joint JSON in the refusal.
    raw = str(exc.value).split("joint_evidence=", 1)[1].split("; owner_evidence=", 1)[0]
    evidence = json.loads(raw)
    assert all(c["count"] > 0 for c in evidence["candidate_geometry"])
    assert evidence["gap_mm"] == 0.2
