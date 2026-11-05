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


def test_buck_input_capacitor_cannot_remain_far_from_its_actual_supply_pins():
    from types import SimpleNamespace

    from kicraft.autoplacer.brain.placement_solver import PlacementSolver
    from kicraft.design.models import BOM, RecipeSelection
    from kicraft.design.recipes.registry import expand_recipe
    from kicraft.design.synthesis.autoplacer import _recipe_local_bypasses

    selection = RecipeSelection(
        recipe="tlv62569-3v3@3", instance="buck", sheets={"power": "POWER"},
        port_bindings={"input": "VIN", "gnd": "GND", "output": "+3V3"},
    )
    expansion = expand_recipe(selection)
    bom = BOM(parts=expansion.parts, connections=expansion.connections,
              recipe_ownership=[expansion.ownership])
    owner_ref = next(p.ref for p in expansion.parts if p.recipe_role == "converter")
    owner = Component(
        ref=owner_ref, value="buck", pos=Point(20, 20), rotation=0,
        layer=Layer.FRONT, width_mm=3.2, height_mm=3.7,
        kind="ic", body_center=Point(20, 20),
        pads=[_pad(owner_ref, "4", 20.95, 18.7, "VIN"),
              _pad(owner_ref, "2", 20, 21.3, "GND")],
    )
    caps = [
        Component(
            ref=part.ref, value=part.value, pos=Point(70, 20), rotation=0,
            layer=Layer.FRONT, width_mm=3.4, height_mm=1.9,
            kind="passive", body_center=Point(70, 20),
            pads=[_pad(part.ref, "1", 69.05, 20, "VIN"),
                  _pad(part.ref, "2", 70.95, 20, "GND")],
        )
        for part in expansion.parts if part.recipe_role == "input_cap"
    ]
    state = _state([owner, *caps])
    cfg = {
        "local_bypasses": _recipe_local_bypasses(
            SimpleNamespace(recipe_selections=[selection]), bom),
        "legality_clearance_mm": 0.2,
    }
    PlacementSolver(state, cfg)._place_local_bypasses(state.components)
    owner_pads = {pad.pad_id: pad for pad in state.components[owner_ref].pads}
    for cap in caps:
        placed = state.components[cap.ref]
        pads = {pad.pad_id: pad for pad in placed.pads}
        assert owner_pads["4"].pos.dist(pads["1"].pos) <= 3
        assert owner_pads["2"].pos.dist(pads["2"].pos) <= 3
        assert owner.physical_bbox()[1].x + 0.2 <= placed.physical_bbox()[0].x or (
            placed.physical_bbox()[1].x + 0.2 <= owner.physical_bbox()[0].x
        ) or owner.physical_bbox()[1].y + 0.2 <= placed.physical_bbox()[0].y or (
            placed.physical_bbox()[1].y + 0.2 <= owner.physical_bbox()[0].y
        )


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


@pytest.mark.parametrize(("recipe", "vin", "ground", "vout"), [
    ("me6211-3v3@2", "1", "2", "5"),
    ("mcp1700-3v3@2", "3", "1", "2"),
    ("ams1117-3v3@2", "3", "1", "2"),
])
def test_ldo_owned_bypasses_compile_from_actual_recipe_and_reject_bad_wiring(
    recipe, vin, ground, vout,
):
    from types import SimpleNamespace

    from kicraft.design.models import BOM, RecipeSelection
    from kicraft.design.recipes.registry import expand_recipe
    from kicraft.design.synthesis.autoplacer import _recipe_local_bypasses

    selection = RecipeSelection(
        recipe=recipe, instance="supply", sheets={"power": "POWER"},
        port_bindings={"input": "HOST_5V", "gnd": "RETURN", "output": "RAIL_3V3"},
    )
    expansion = expand_recipe(selection)
    bom = BOM(parts=expansion.parts, connections=expansion.connections,
              recipe_ownership=[expansion.ownership])
    architecture = SimpleNamespace(recipe_selections=[selection])
    roles = {part.recipe_role: part.ref for part in expansion.parts}
    rules = _recipe_local_bypasses(architecture, bom)
    assert len(rules) == 2
    by_cap = {rule["bypass_ref"]: rule for rule in rules}
    for role, pin in (("input_cap", vin), ("output_cap", vout)):
        rule = by_cap[roles[role]]
        assert rule["owner_ref"] == roles["regulator"]
        assert rule["power_pins"] == [pin, "1"]
        assert rule["ground_pins"] == [[ground], "2"]
        assert rule["max_distance_mm"] == 3.0

    # Corrupt the actual output-cap connection, not the recipe declaration.
    output = next(net for net in bom.connections if any(
        ep.ref == roles["output_cap"] and ep.pin == "1" for ep in net.endpoints
    ))
    output.net_name = "WRONG_RAIL"
    with pytest.raises(ValueError, match="unproven bypass pin"):
        _recipe_local_bypasses(architecture, bom)


def _reverse_blocked_compiler_fixture():
    from types import SimpleNamespace

    from kicraft.design.models import BOM, RecipeSelection
    from kicraft.design.recipes.registry import expand_selections

    selections = [
        RecipeSelection(
            recipe="k7803m-3v3-reverse-blocked@1", instance=instance,
            sheets={"power": "POWER"},
            port_bindings={"input": "VMOTOR", "gnd": "GND", "output": "DRIVER_RAIL"},
            requirement_ids=[f"req_{instance}"],
        )
        for instance in ("left", "right")
    ]
    expansions = expand_selections(selections)
    bom = BOM(
        parts=[part for expansion in expansions for part in expansion.parts],
        connections=[net for expansion in expansions for net in expansion.connections],
        recipe_ownership=[expansion.ownership for expansion in expansions],
    )
    architecture = SimpleNamespace(
        recipe_selections=selections, power_nets=["VMOTOR", "GND", "DRIVER_RAIL"],
        requirements=[],
    )
    roles = [{p.recipe_role: p.ref for p in expansion.parts} for expansion in expansions]
    return architecture, bom, roles


def test_reverse_blocked_scoped_bypasses_compile_to_native_placement(tmp_path):
    import json

    from kicraft.design.synthesis.autoplacer import write_autoplacer_json

    architecture, bom, roles = _reverse_blocked_compiler_fixture()
    path = write_autoplacer_json(tmp_path, "motor", architecture, bom)
    rules = json.loads(path.read_text())["local_bypasses"]
    assert len(rules) == 6
    by_cap = {rule["bypass_ref"]: rule for rule in rules}
    for refs in roles:
        for cap, owner, power, grounds, distance in (
            ("input_cap", "converter", "1", ["2"], 4.5),
            ("output_cap", "converter", "3", ["2"], 4.5),
            ("reverse_decoupling", "reverse_controller", "1", ["2", "3"], 3.0),
        ):
            assert by_cap[refs[cap]] == {
                "owner_ref": refs[owner], "bypass_ref": refs[cap],
                "power_pins": [power, "1"], "ground_pins": [grounds, "2"],
                "max_distance_mm": distance,
            }
    raw_nets = []
    for refs in roles:
        nets = {
            net.net_name for net in bom.connections for ep in net.endpoints
            if (ep.ref, ep.pin) in {
                (refs["converter"], "3"), (refs["output_cap"], "1"),
                (refs["reverse_controller"], "1"), (refs["reverse_decoupling"], "1"),
            }
        }
        assert len(nets) == 1
        raw_nets.append(nets.pop())
    assert raw_nets[0] != raw_nets[1]
    assert not set(raw_nets) & {"VMOTOR", "GND", "DRIVER_RAIL"}


@pytest.mark.parametrize("damage", [
    "cross_instance_graph_and_manifest", "input_graph", "ground_graph",
    "raw_output_cap_graph", "raw_reverse_controller_graph", "manifest_net",
    "manifest_owner", "manifest_requirement", "manifest_binding",
    "duplicate_graph_pin", "duplicate_manifest_pin", "part_identity",
])
def test_reverse_blocked_compiler_rejects_malformed_graph_and_ownership(tmp_path, damage):
    from kicraft.design.synthesis.autoplacer import write_autoplacer_json

    architecture, bom, roles = _reverse_blocked_compiler_fixture()
    left, right = roles
    manifest = bom.recipe_ownership[0]

    def connection(ref, pin):
        return next(net for net in bom.connections if any(
            (ep.ref, ep.pin) == (ref, pin) for ep in net.endpoints
        ))

    def owned(ref, pin):
        return next(row for row in manifest.pins if (row.ref, row.pin) == (ref, pin))

    if damage == "cross_instance_graph_and_manifest":
        target = connection(right["converter"], "3").net_name
        original = connection(left["converter"], "3").net_name
        connection(left["converter"], "3").net_name = target
        for row in manifest.pins:
            if row.net == original:
                row.net = target
    elif damage == "input_graph":
        connection(left["input_cap"], "1").net_name = "WRONG_INPUT"
    elif damage == "ground_graph":
        connection(left["reverse_controller"], "2").net_name = "WRONG_GROUND"
    elif damage in {"raw_output_cap_graph", "raw_reverse_controller_graph"}:
        ref = left["output_cap" if damage == "raw_output_cap_graph" else "reverse_controller"]
        net = connection(ref, "1")
        endpoint = next(ep for ep in net.endpoints if (ep.ref, ep.pin) == (ref, "1"))
        net.endpoints = [ep for ep in net.endpoints if ep is not endpoint]
        bom.connections.append(net.model_copy(update={"net_name": "WRONG_RAW", "endpoints": [endpoint]}))
    elif damage == "manifest_net":
        owned(left["reverse_decoupling"], "1").net = "WRONG_RAW"
    elif damage == "manifest_owner":
        owned(left["output_cap"], "1").owner_id = "wrong@1"
    elif damage == "manifest_requirement":
        owned(left["reverse_decoupling"], "1").requirement_id = "wrong"
    elif damage == "manifest_binding":
        manifest.port_bindings = {**manifest.port_bindings, "input": "WRONG_INPUT"}
    elif damage == "duplicate_graph_pin":
        net = connection(left["output_cap"], "1")
        bom.connections.append(net.model_copy(deep=True))
    elif damage == "duplicate_manifest_pin":
        manifest.pins.append(owned(left["output_cap"], "1").model_copy(deep=True))
    else:
        next(part for part in bom.parts if part.ref == left["converter"]).mpn = "WRONG"
    with pytest.raises(ValueError):
        write_autoplacer_json(tmp_path, "motor", architecture, bom)


def test_reverse_blocked_compiler_rejects_unknown_reviewed_logical_net(monkeypatch):
    from kicraft.design.recipes import registry
    from kicraft.design.synthesis.autoplacer import _recipe_local_bypasses

    architecture, bom, _ = _reverse_blocked_compiler_fixture()
    definition = registry.get_recipe(architecture.recipe_selections[0].recipe)
    bad = definition.model_copy(update={
        "pins": tuple(
            pin.model_copy(update={"net": "unknown_raw"})
            if pin.net == "raw_output" else pin for pin in definition.pins
        ),
        "placement_constraints": tuple(
            rule.model_copy(update={"parameters": {**rule.parameters, "power_net": "unknown_raw"}})
            if rule.kind == "owned_bypass" and rule.parameters["power_net"] == "raw_output"
            else rule for rule in definition.placement_constraints
        ),
    })
    monkeypatch.setattr(registry, "get_recipe", lambda _recipe: bad)
    with pytest.raises(ValueError, match="unknown bypass net"):
        _recipe_local_bypasses(architecture, bom)


def test_me6211_recipe_compiler_and_solver_place_both_native_caps_together():
    from itertools import combinations
    from types import SimpleNamespace

    from kicraft.autoplacer.brain.placement_solver import PlacementSolver
    from kicraft.design.models import BOM, RecipeSelection
    from kicraft.design.recipes.registry import expand_recipe
    from kicraft.design.synthesis.autoplacer import _recipe_local_bypasses

    selection = RecipeSelection(
        recipe="me6211-3v3@2", instance="supply", sheets={"power": "POWER"},
        port_bindings={"input": "HOST_5V", "gnd": "RETURN", "output": "RAIL_3V3"},
    )
    expansion = expand_recipe(selection)
    bom = BOM(parts=expansion.parts, connections=expansion.connections,
              recipe_ownership=[expansion.ownership])
    roles = {part.recipe_role: part.ref for part in expansion.parts}
    rules = _recipe_local_bypasses(SimpleNamespace(recipe_selections=[selection]), bom)
    owner_ref = roles["regulator"]
    # Actual 1150 SOT-23-5 and 0603 courtyard extents include line width.
    # The 1150 C2 center was 3.732 mm away and its GND pad 4.870 mm away.
    owner = Component(
        ref=owner_ref, value="ME6211", pos=Point(20, 20), rotation=0,
        layer=Layer.FRONT, width_mm=2.95, height_mm=1.61, kind="ic",
        body_center=Point(20, 20),
        pads=[_pad(owner_ref, pin, 20 + dx, 20 + dy, net)
              for pin, dx, dy, net in (
                  ("1", -0.95, 1.15, "HOST_5V"),
                  ("2", 0, 1.15, "RETURN"),
                  ("5", -0.95, -1.15, "RAIL_3V3"),
              )],
    )

    def capacitor(role, y):
        ref = roles[role]
        net = "HOST_5V" if role == "input_cap" else "RAIL_3V3"
        return Component(
            ref=ref, value="1uF", pos=Point(19.45, y), rotation=0,
            layer=Layer.FRONT, width_mm=3.01, height_mm=1.51, kind="passive",
            body_center=Point(19.45, y),
            pads=[_pad(ref, "1", 18.675, y, net),
                  _pad(ref, "2", 20.225, y, "RETURN")],
        )

    def assert_group(comps):
        for rule in rules:
            op = {p.pad_id: p for p in comps[owner_ref].pads}
            cp = {p.pad_id: p for p in comps[rule["bypass_ref"]].pads}
            assert op[rule["power_pins"][0]].pos.dist(cp["1"].pos) <= 3
            assert op["2"].pos.dist(cp["2"].pos) <= 3
        for a, b in combinations(comps.values(), 2):
            at, ab = a.bbox(0.1)
            bt, bb = b.bbox(0.1)
            assert ab.x <= bt.x + 1e-9 or bb.x <= at.x + 1e-9 or (
                ab.y <= bt.y + 1e-9 or bb.y <= at.y + 1e-9
            )

    # Independently established joint witness at the hard 0.2 mm gap.
    witness = [owner, capacitor("input_cap", 21.8), capacitor("output_cap", 18.2)]
    assert_group({c.ref: c for c in witness})
    obstacle = _cap("R99", 30, 20, [])
    obstacle.locked = True
    state = _state([owner, capacitor("input_cap", 30), capacitor("output_cap", 30), obstacle])
    solver = PlacementSolver(state, {
        "local_bypasses": rules, "edge_margin_mm": 2.0,
        "placement_clearance_mm": 2.84, "legality_clearance_mm": 0.2,
        "courtyard_overlap_min_gap_mm": 0.2,
    })
    solver._reserve_local_bypasses(state.components)
    group = {ref: component for ref, component in state.components.items() if ref != "R99"}
    assert_group(group)
    for component in group.values():
        at, ab = component.bbox(2.84 / 2)
        bt, bb = obstacle.bbox(2.84 / 2)
        assert ab.x <= bt.x + 1e-9 or bb.x <= at.x + 1e-9 or (
            ab.y <= bt.y + 1e-9 or bb.y <= at.y + 1e-9
        )


def test_ams1117_recipe_rejects_a_board_with_no_legal_cap_space():
    from types import SimpleNamespace

    from kicraft.autoplacer.brain.placement_solver import PlacementSolver
    from kicraft.design.models import BOM, RecipeSelection
    from kicraft.design.recipes.registry import expand_recipe
    from kicraft.design.synthesis.autoplacer import _recipe_local_bypasses

    selection = RecipeSelection(
        recipe="ams1117-3v3@2", instance="supply", sheets={"power": "POWER"},
        port_bindings={"input": "VIN", "gnd": "GND", "output": "VOUT"},
    )
    expansion = expand_recipe(selection)
    bom = BOM(parts=expansion.parts, connections=expansion.connections,
              recipe_ownership=[expansion.ownership])
    roles = {part.recipe_role: part.ref for part in expansion.parts}
    rules = _recipe_local_bypasses(SimpleNamespace(recipe_selections=[selection]), bom)
    ref = roles["regulator"]
    # Native SOT-223-3_TabPin2 courtyard and pin centers; 0603 caps retain
    # the recipe footprint, not a hypothetical smaller substitute.
    owner = Component(
        ref=ref, value="AMS1117", pos=Point(5, 4.5), rotation=0, layer=Layer.FRONT,
        width_mm=8.85, height_mm=7.25, kind="ic", body_center=Point(5, 4.5),
        pads=[_pad(ref, pin, x, y, net) for pin, x, y, net in (
            ("1", 1.85, 2.2, "GND"), ("2", 1.85, 4.5, "VOUT"),
            ("3", 1.85, 6.8, "VIN"),
        )],
    )
    caps = []
    for role, net in (("input_cap", "VIN"), ("output_cap", "VOUT")):
        cap_ref = roles[role]
        caps.append(Component(
            ref=cap_ref, value="C", pos=Point(5, 4.5), rotation=0, layer=Layer.FRONT,
            width_mm=3.01, height_mm=1.51, kind="passive", body_center=Point(5, 4.5),
            pads=[_pad(cap_ref, "1", 4.225, 4.5, net),
                  _pad(cap_ref, "2", 5.775, 4.5, "GND")],
        ))
    state = BoardState(components={c.ref: c for c in (owner, *caps)},
                       board_outline=(Point(0, 0), Point(10, 9)))
    solver = PlacementSolver(state, {
        "local_bypasses": rules, "edge_margin_mm": 0,
        "placement_clearance_mm": 2.84, "legality_clearance_mm": 0.2,
        "courtyard_overlap_min_gap_mm": 0.2,
    })
    with pytest.raises(ValueError, match="no (joint )?legal placement"):
        solver._place_local_bypasses(state.components)


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
