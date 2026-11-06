import copy
import json
from pathlib import Path

from types import SimpleNamespace

import pytest
from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.leaf_geometry import repair_leaf_placement_legality
from kicraft.autoplacer.brain.leaf_routing import _place_fabricated_edge_interfaces
from kicraft.autoplacer.brain.types import BoardState, Component, Layer, Pad, Point

from kicraft.design.lowering import lower_requirement
from kicraft.design.models import BOM, RecipeSelection
from kicraft.design.recipes import expand_recipe, expand_selections, migrate_recipe_selection
from kicraft.design.recipes.pin_allocator import PinAllocationError, allocate_pins
from kicraft.design.recipes.registry import get_recipe, locked_pin_assignments
from kicraft.design.synthesis.validation import check_mcu_programming_access, check_net_coverage
from kicraft.server.stage_contracts import (
    StageSchemaError,
    _normalize_stage_response,
)


def _selection(**parameters):
    return RecipeSelection(
        recipe="rp2040-minimal@3",
        instance="main",
        sheets={"mcu": "MCU", "io": "CASTELLATED IO"},
        port_bindings={
            "vdd": "+3V3",
            "gnd": "GND",
            "usb_dm": "USB_D_N",
            "usb_dp": "USB_D_P",
        },
        parameters=parameters,
    )


def test_recipe_expansion_is_deterministic_and_provenanced():
    first = expand_recipe(_selection())
    second = expand_recipe(_selection())
    assert first == second
    assert all(part.recipe_id == "rp2040-minimal@3" for part in first.parts)
    assert [edge.side for edge in first.edge_interfaces] == ["left", "right"]

    bom = BOM(
        parts=first.parts,
        connections=first.connections,
        no_connect_pins=first.no_connect_pins,
        edge_interfaces=first.edge_interfaces,
    )
    assert check_net_coverage(bom).ok
    # A native USB pair without a physical receptacle must not be represented as
    # a programming path by its recipe labels alone.
    access = check_mcu_programming_access(bom)
    assert not access.ok
    assert any(
        offender.startswith("native_usb_programming_required:") for offender in access.offenders
    )


def test_multiple_recipe_instances_allocate_disjoint_references():
    second = _selection().model_copy(update={"instance": "aux"})
    expansions = expand_selections([_selection(), second])
    refs = [[part.ref for part in expansion.parts] for expansion in expansions]
    assert set(refs[0]).isdisjoint(refs[1])


def test_castellation_solver_pins_pad_centers_to_declared_edge_and_pitch():
    components = {}
    for ref in ("TP1", "TP2", "TP3"):
        components[ref] = Component(
            ref=ref,
            value="Castellated pad",
            pos=Point(5.0, 5.0),
            rotation=0.0,
            layer=Layer.FRONT,
            width_mm=2.2,
            height_mm=2.2,
            pads=[Pad(ref=ref, pad_id="1", pos=Point(5.0, 5.0), net="N", layer=Layer.FRONT)],
        )
    state = BoardState(components=components, nets={})
    state.board_outline = (Point(0.0, 0.0), Point(20.0, 20.0))
    solver = PlacementSolver(
        state,
        {
            "edge_interfaces": [
                {
                    "name": "left-bank",
                    "refs": list(components),
                    "side": "left",
                    "pitch_mm": 2.54,
                }
            ]
        },
        seed=0,
    )
    placed = solver.solve(max_iterations=10)
    centers = [placed[ref].pads[0].pos for ref in components]
    assert [center.x for center in centers] == [0.0, 0.0, 0.0]
    assert [centers[i + 1].y - centers[i].y for i in range(2)] == pytest.approx([2.54, 2.54])
    assert all(component.locked for component in placed.values())


def test_leaf_legality_repair_preserves_solver_owned_castellation_datums():
    components = {
        ref: Component(
            ref=ref,
            value="Castellated pad",
            pos=Point(0.0, 5.0 + index * 2.54),
            rotation=0.0,
            layer=Layer.FRONT,
            width_mm=2.2,
            height_mm=2.2,
            pads=[
                Pad(
                    ref=ref,
                    pad_id="1",
                    pos=Point(0.0, 5.0 + index * 2.54),
                    net=f"N{index}",
                    layer=Layer.FRONT,
                )
            ],
        )
        for index, ref in enumerate(("TP1", "TP2", "TP3"))
    }
    state = BoardState(components=components, nets={})
    state.board_outline = (Point(0.0, 0.0), Point(20.0, 20.0))
    cfg = {
        "edge_interfaces": [
            {
                "name": "left-bank",
                "refs": list(components),
                "side": "left",
                "pitch_mm": 2.54,
                "exact_edge_datum": True,
            }
        ]
    }

    repaired, _diagnostics = repair_leaf_placement_legality(
        SimpleNamespace(local_state=state),
        components,
        cfg,
    )

    centers = [repaired[ref].pads[0].pos for ref in components]
    assert [center.x for center in centers] == [0.0, 0.0, 0.0]
    assert not _diagnostics["resolved"]
    assert _diagnostics["remaining_overlaps"]
    assert [centers[i + 1].y - centers[i].y for i in range(2)] == pytest.approx([2.54, 2.54])


def test_trivial_leaf_reframe_reapplies_exact_castellation_datums():
    components = {
        ref: Component(
            ref=ref,
            value="Castellated pad",
            pos=Point(4.0, 3.0 + index * 2.86),
            rotation=0.0,
            layer=Layer.FRONT,
            width_mm=2.2,
            height_mm=2.2,
            pads=[
                Pad(
                    ref=ref,
                    pad_id="1",
                    pos=Point(4.0, 3.0 + index * 2.86),
                    net=f"N{index}",
                    layer=Layer.FRONT,
                )
            ],
        )
        for index, ref in enumerate(("TP1", "TP2", "TP3"))
    }
    state = BoardState(components=components, nets={})
    state.board_outline = (Point(1.0, 1.0), Point(20.0, 20.0))
    _place_fabricated_edge_interfaces(
        state,
        {
            "edge_interfaces": [
                {
                    "refs": list(components),
                    "side": "left",
                    "pitch_mm": 2.54,
                }
            ]
        },
    )
    centers = [components[ref].pads[0].pos for ref in components]
    assert [center.x for center in centers] == [1.0, 1.0, 1.0]
    assert [centers[i + 1].y - centers[i].y for i in range(2)] == pytest.approx([2.54, 2.54])


def _edge_bank_state(sides):
    components = {}
    interfaces = []
    for side, count in sides:
        refs = [f"{side}{index}" for index in range(count)]
        interfaces.append({
            "refs": refs,
            "side": side,
            "pitch_mm": 2.54,
            "exact_edge_datum": True,
        })
        for ref in refs:
            components[ref] = Component(
                ref=ref, value="Castellated pad", pos=Point(5.0, 5.0),
                rotation=0.0, layer=Layer.FRONT, width_mm=1.0, height_mm=1.0,
                pads=[Pad(ref=ref, pad_id="1", pos=Point(5.2, 5.3), net=ref, layer=Layer.FRONT)],
            )
    state = BoardState(components=components, nets={})
    state.board_outline = (Point(0.0, 0.0), Point(20.0, 20.0))
    return state, interfaces


@pytest.mark.parametrize("placement", ["solver", "routing"])
@pytest.mark.parametrize("sides", [
    [("left", 3), ("top", 10)],
    [("right", 3), ("bottom", 10)],
    [("left", 10), ("right", 10), ("top", 10), ("bottom", 10)],
])
def test_mixed_edge_banks_use_final_outline_independent_of_order(placement, sides):
    results = []
    for reverse in (False, True):
        state, interfaces = _edge_bank_state(sides)
        cfg = {"edge_interfaces": list(reversed(interfaces)) if reverse else interfaces}
        if placement == "solver":
            if sides == [("left", 3), ("top", 10)]:
                state.components["H1"] = Component(
                    ref="H1", value="corner", pos=Point(10.0, 10.0), rotation=0.0,
                    layer=Layer.FRONT, width_mm=1.0, height_mm=1.0, pads=[],
                )
                cfg.update({"component_zones": {"H1": {"corner": "bottom-right"}},
                            "edge_jitter_mm": 0.0})
            solver = PlacementSolver(state, cfg, seed=0)
            solver._pin_edge_components(state.components)
            if "H1" in state.components:
                corner = state.components["H1"].pos
                tl, br = state.board_outline
                assert (corner.x, corner.y) == pytest.approx((br.x - 2.0, br.y - 2.0))
        else:
            _place_fabricated_edge_interfaces(state, cfg)
        tl, br = state.board_outline
        if any(count == 10 and side in ("top", "bottom") for side, count in sides):
            assert br.x - tl.x == pytest.approx(9 * 2.54 + 4.0)
        if any(count == 10 and side in ("left", "right") for side, count in sides):
            assert br.y - tl.y == pytest.approx(9 * 2.54 + 4.0)
        if placement == "solver":
            assert (tl.x, tl.y) == (0.0, 0.0)
        else:
            assert ((tl.x + br.x) / 2, (tl.y + br.y) / 2) == pytest.approx((10.0, 10.0))
        for interface in interfaces:
            side = interface["side"]
            centers = [state.components[ref].pads[0].pos for ref in interface["refs"]]
            perpendicular = [p.x if side in ("left", "right") else p.y for p in centers]
            datum = {"left": tl.x, "right": br.x, "top": tl.y, "bottom": br.y}[side]
            assert perpendicular == pytest.approx([datum] * len(centers))
            parallel = [p.y if side in ("left", "right") else p.x for p in centers]
            assert [b - a for a, b in zip(parallel, parallel[1:])] == pytest.approx(
                [2.54] * (len(centers) - 1)
            )
        results.append((
            (tl.x, tl.y, br.x, br.y),
            {ref: (c.pos.x, c.pos.y, c.pads[0].pos.x, c.pads[0].pos.y)
             for ref, c in state.components.items() if c.pads},
        ))
    assert results[0][0] == pytest.approx(results[1][0])
    assert results[0][1].keys() == results[1][1].keys()
    for ref in results[0][1]:
        assert results[0][1][ref] == pytest.approx(results[1][1][ref])


@pytest.mark.parametrize("placement", ["solver", "routing"])
@pytest.mark.parametrize("within_group", [True, False])
def test_duplicate_usable_edge_members_rejected_before_mutation(placement, within_group):
    state, interfaces = _edge_bank_state([("left", 3), ("top", 3)])
    if within_group:
        interfaces[0]["refs"] = ["left0", "left1", "left0", "left1"]
        duplicate = "left0"
    else:
        interfaces[1]["refs"] = ["top0", "left1", "left0"]
        duplicate = "left1"
    cfg = {"edge_interfaces": interfaces}
    solver = PlacementSolver(state, cfg, seed=0)
    solver._pinned_targets = {"sentinel": Point(9.0, 9.0)}
    solver._pinned_rotations = {"sentinel": 42.0}
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match=f"^edge interface reference declared more than once: {duplicate}$"):
        if placement == "solver":
            solver._pin_edge_components(state.components)
        else:
            _place_fabricated_edge_interfaces(state, cfg)
    assert state == before
    assert solver._pinned_targets == {"sentinel": Point(9.0, 9.0)}
    assert solver._pinned_rotations == {"sentinel": 42.0}


@pytest.mark.parametrize("placement", ["solver", "routing"])
def test_incomplete_edge_bank_is_skipped_without_datum_exemption(placement):
    state, interfaces = _edge_bank_state([("left", 1)])
    interfaces[0]["refs"].append("missing")
    component = state.components["left0"]
    component.pos = Point(0.0, 10.0)
    component.pads[0].pos = Point(0.0, 10.0)
    cfg = {"edge_interfaces": interfaces}
    solver = PlacementSolver(state, cfg, seed=0)
    if placement == "solver":
        solver._pin_edge_components(state.components)
    else:
        _place_fabricated_edge_interfaces(state, cfg)
    assert component.pads[0].pos == Point(0.0, 10.0)
    assert not component.locked
    assert solver.legality_diagnostics(state.components)["pad_outside_count"] == 1


def test_adjacent_edge_banks_reject_real_locked_overlap():
    state, interfaces = _edge_bank_state([("left", 1), ("top", 1)])
    for comp in state.components.values():
        comp.width_mm = comp.height_mm = 22.0
        comp.pads[0].size_mm = Point(22.0, 22.0)
    cfg = {"edge_interfaces": interfaces}
    solver = PlacementSolver(state, cfg, seed=0)
    solver._pin_edge_components(state.components)
    diagnostics = solver.legality_diagnostics(state.components)
    assert diagnostics["overlap_count"] == 1
    assert diagnostics["locked_overlap_count"] == 1


def test_recipe_rejects_unknown_version_parameter_and_sheet_role():
    with pytest.raises(ValueError, match="unknown circuit recipe"):
        expand_recipe(_selection().model_copy(update={"recipe": "rp2040-minimal@4"}))
    with pytest.raises(ValueError, match="unknown parameters"):
        expand_recipe(_selection(arbitrary_part="x"))
    with pytest.raises(ValueError, match="sheet roles mismatch"):
        expand_recipe(_selection().model_copy(update={"sheets": {"mcu": "MCU"}}))


def test_saved_rp2040_legacy_selection_migrates_usb_and_four_gpio_contacts():
    saved = _selection().model_copy(update={
        "recipe": "rp2040-minimal@2",
        "parameters": {"usb": True, "gpio_exposed": 4, "programming": "bootsel"},
    })
    migrated = migrate_recipe_selection(saved)
    expansion = expand_recipe(saved)
    assert saved.recipe == "rp2040-minimal@2"
    assert migrated.recipe == expansion.selection.recipe == "rp2040-minimal@3"
    assert migrated.parameters["flash_mbit"] == 16
    assert migrated.parameters["programming"] == "native_usb"
    mcu = next(part for part in expansion.parts if part.recipe_role == "mcu")
    connected = {
        endpoint.pin
        for connection in expansion.connections
        for endpoint in connection.endpoints
        if endpoint.ref == mcu.ref
    }
    disconnected = {pin.pin for pin in expansion.no_connect_pins if pin.ref == mcu.ref}
    assert {"2", "3", "4", "5", "46", "47"} <= connected
    assert {"6", "7", "8", "9", "11", "12"} <= disconnected
    assert any(part.mpn == "W25Q16JVSS" for part in expansion.parts)


def test_saved_rp2040_v1_migrates_original_net_names():
    saved = _selection().model_copy(update={
        "recipe": "rp2040-minimal@1", "port_bindings": {}, "parameters": {},
    })
    expansion = expand_recipe(saved)
    assert expansion.selection.port_bindings["vdd"] == "3V3"
    assert expansion.selection.port_bindings["usb_dp"] == "USB_DP"
    assert expansion.selection.port_bindings["gpio29"] == "GPIO29"
    assert any(connection.net_name == "GPIO29" for connection in expansion.connections)


def test_saved_rp2040_swd_contract_and_invalid_legacy_intent():
    saved = _selection().model_copy(update={
        "recipe": "rp2040-minimal@2",
        "parameters": {"native_usb": False, "programming": "swd"},
        "port_bindings": {"vdd": "+3V3", "gnd": "GND"},
    })
    expansion = expand_recipe(saved)
    assert any(part.recipe_role == "swd_header" for part in expansion.parts)
    for invalid in (
        {"usb": False, "programming": "bootsel"},
        {"usb": True, "programming": "swd"},
        {"usb": True, "flash_mbit": 8},
        {"usb": True, "gpio_exposed": 31},
        {"usb": True, "native_usb": False},
        {"usb": True, "other": 1},
    ):
        with pytest.raises(ValueError):
            expand_recipe(saved.model_copy(update={"parameters": invalid}))
    with pytest.raises(ValueError, match="unexposed GPIOs"):
        expand_recipe(_selection(gpio_exposed=4).model_copy(update={
            "port_bindings": {**_selection().port_bindings, "gpio5": "GPIO5"},
        }))
    for ambiguous in ({}, {"flash_mbit": 16}, {"logic_voltage_v": 3.3}):
        with pytest.raises(ValueError, match="ambiguous saved rp2040-minimal@2"):
            migrate_recipe_selection(saved.model_copy(update={"parameters": ambiguous}))
    with pytest.raises(ValueError, match="unknown parameters"):
        migrate_recipe_selection(saved.model_copy(update={
            "parameters": {"native_usb": True, "unexpected": 1},
        }))



@pytest.mark.parametrize("recipe,parameters", [
    ("rp2040-minimal@2", {"usb": True, "programming": "bootsel"}),
    ("rp2040-minimal@3", {"native_usb": True, "programming": "native_usb"}),
])
@pytest.mark.parametrize("dm,dp", [("SHORT", "SHORT"), ("GND", "DP"), ("DM", "+3V3")])
def test_rp2040_rejects_shorted_usb_pair_or_usb_to_supply(recipe, parameters, dm, dp):
    selection = _selection().model_copy(update={
        "recipe": recipe, "parameters": parameters,
        "port_bindings": {"vdd": "+3V3", "gnd": "GND", "usb_dm": dm, "usb_dp": dp},
    })
    with pytest.raises(ValueError, match="shorted USB"):
        expand_recipe(selection)


def test_saved_rp2040_resolution_preserves_legacy_parameters_ports_and_sheet_ownership():
    from kicraft.design.recipes import apply_architecture_recipe_resolution

    parameters = {"usb": True, "programming": "bootsel", "gpio_exposed": 4}
    ports = {"vdd": "+3V3", "gnd": "GND", "usb_dm": "DM", "usb_dp": "DP", "gpio0": "A"}
    architecture = {
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "RP2040 MCU"},
            {"name": "IO", "stem": "IO", "function": "Castellated GPIO"},
        ],
        "mcu_present": True,
        "power_nets": ["+3V3", "GND"],
        "rail_voltages": {"+3V3": 3.3},
        "inter_sheet_nets": [],
        "requirements": [{
            "id": "core", "sheet": "MCU", "role": "mcu_core", "family": "rp2040",
            "parameters": parameters, "ports": ports,
        }],
        "recipe_selections": [{
            "recipe": "rp2040-minimal@2", "instance": "main",
            "sheets": {"mcu": "MCU", "io": "IO"}, "requirement_ids": ["core"],
            "parameters": parameters, "port_bindings": ports,
        }],
    }
    original = copy.deepcopy(architecture)
    resolved = apply_architecture_recipe_resolution(architecture)
    selection = resolved.recipe_selections[0]
    assert selection.recipe == "rp2040-minimal@3"
    assert selection.sheets == {"mcu": "MCU", "io": "IO"}
    assert selection.parameters["gpio_exposed"] == 4
    assert selection.parameters["programming"] == "native_usb"
    assert selection.port_bindings == ports
    assert architecture == original
    assert apply_architecture_recipe_resolution(resolved).recipe_selections == [selection]


def test_rp2040_v3_scopes_internal_nets_and_binds_power_ports():
    selection = RecipeSelection(
        recipe="rp2040-minimal@3",
        instance="main",
        sheets={"mcu": "MCU", "io": "IO"},
        port_bindings={
            "vdd": "+3V3",
            "gnd": "GND",
            "usb_dm": "USB_D_N",
            "usb_dp": "USB_D_P",
        },
        requirement_ids=["mcu_core"],
    )
    first = expand_recipe(selection)
    second = expand_recipe(
        selection.model_copy(update={"instance": "aux", "requirement_ids": ["aux_core"]})
    )
    first_nets = {connection.net_name for connection in first.connections}
    assert "+3V3" in first_nets
    assert "QSPI_CS" not in first_nets
    assert set(first.ownership.internal_nets).isdisjoint(second.ownership.internal_nets)


def test_recipe_locked_wiring_rejects_model_overwrite():
    expansion = expand_recipe(_selection())
    bom = {"parts": [part.model_dump() for part in expansion.parts]}
    locked = locked_pin_assignments(bom)
    (ref, pin), net = next(iter(locked.items()))
    with pytest.raises(Exception, match="recipe-owned"):
        _normalize_stage_response(
            "wiring", {"pins": [{"ref": ref, "pin": pin, "net": net}]}, {"bom": bom}
        )
    merged = _normalize_stage_response("wiring", {"pins": []}, {"bom": bom})[0]
    assert {"ref": "U1", "pin": "26"} not in merged["no_connect_pins"]


def test_allocator_owned_wiring_pin_rejects_model_overwrite():
    from kicraft.design.models import RecipePinAllocation

    selection = RecipeSelection(
        recipe="esp32-s3-mini-1-minimal@1",
        instance="main",
        sheets={"mcu": "MCU"},
        parameters={"native_usb": True},
        port_bindings={
            "vdd": "+3V3",
            "gnd": "GND",
            "usb_dm": "USB_D_N",
            "usb_dp": "USB_D_P",
        },
        requirement_ids=["mcu_core"],
        pin_allocations=[RecipePinAllocation(net="APP_OUT", pin="5", capability="output")],
    )
    expansion = expand_recipe(selection)
    bom = {
        "parts": [part.model_dump(mode="json") for part in expansion.parts],
        "connections": [connection.model_dump(mode="json") for connection in expansion.connections],
        "no_connect_pins": [pin.model_dump(mode="json") for pin in expansion.no_connect_pins],
        "recipe_ownership": [expansion.ownership.model_dump(mode="json")],
    }
    with pytest.raises(StageSchemaError, match="recipe-owned"):
        _normalize_stage_response(
            "wiring",
            {"pins": [{"ref": "U1", "pin": "5", "net": "APP_OUT"}]},
            {"bom": bom},
        )


def _esp32_architecture_payload():
    return {
        "topologies": {"MCU": "ESP32-S3 controller with native USB recovery"},
        "rail_voltages": {"+3V3": 3.3},
        "comms_protocols": ["USB"],
        "mcu_present": True,
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "ESP32-S3 controller"},
        ],
        "power_nets": ["+3V3"],
        "inter_sheet_nets": [],
        "assumptions": ["3V3 is supplied externally at 500mA"],
    }


def test_architecture_exact_esp32_part_resolves_without_model_selection():
    payload = _add_native_usb(_esp32_architecture_payload())
    canonical, _expanded = _normalize_stage_response(
        "architecture",
        payload,
        {"intent": {"named_parts": ["ESP32-S3-MINI-1-N8"]}},
    )
    mcu = next(row for row in canonical["requirements"] if row["role"] == "mcu_core")
    assert mcu["exact_part"] == "ESP32-S3-MINI-1-N8"
    selection = next(
        row
        for row in canonical["recipe_selections"]
        if row["recipe"] == "esp32-s3-mini-1-minimal@1"
    )
    assert selection["port_bindings"] == {
        "gnd": "GND",
        "vdd": "+3V3",
        "usb_dm": "USB_D_N",
        "usb_dp": "USB_D_P",
    }
    assert any(row["recipe"] == selection["recipe"] for row in canonical["recipe_resolution"])


def test_architecture_unique_family_prefix_selects_registered_exact_part():
    payload = {
        "topologies": {"MCU": "STM32F103 controller"},
        "rail_voltages": {"+3V3": 3.3},
        "comms_protocols": ["SWD"],
        "mcu_present": True,
        "sheets": [{"name": "MCU", "stem": "MCU", "function": "STM32 controller"}],
        "power_nets": ["+3V3"],
        "inter_sheet_nets": [],
        "assumptions": [],
    }

    canonical, _expanded = _normalize_stage_response(
        "architecture",
        payload,
        {"intent": {"named_parts": ["STM32F103"]}},
    )

    assert canonical["requirements"][0]["exact_part"] == "STM32F103C8T6"
    assert canonical["recipe_selections"][0]["recipe"] == "stm32f103c8t6-minimal@1"


def test_stm32_recipe_does_not_claim_unowned_crystal_by_value_overlap():
    payload = {
        "topologies": {"MCU": "STM32 controller", "CRYSTAL": "8 MHz HSE crystal"},
        "rail_voltages": {"+3V3": 3.3},
        "comms_protocols": ["SWD"],
        "mcu_present": True,
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "STM32 controller"},
            {"name": "CRYSTAL", "stem": "CRYSTAL", "function": "8 MHz HSE crystal"},
        ],
        "power_nets": ["+3V3"],
        "inter_sheet_nets": [],
        "assumptions": [],
        "requirements": [
            {
                "id": "mcu_core",
                "sheet": "MCU",
                "functional_blocks": ["CONTROLLER"],
                "role": "mcu_core",
                "family": "stm32f103",
                "exact_part": "STM32F103C8T6",
                "ports": {"vdd": "+3V3", "gnd": "GND"},
            },
            {
                "id": "crystal_8mhz",
                "sheet": "CRYSTAL",
                "functional_blocks": ["OSCILLATOR"],
                "role": "analog_block",
                "family": "crystal",
                "ports": {"osc_in": "OSC_IN", "osc_out": "OSC_OUT"},
            },
        ],
        "recipe_selections": [
            {
                "recipe": "stm32f103c8t6-minimal@1",
                "instance": "mcu_core",
                "sheets": {"mcu": "MCU"},
                "requirement_ids": ["mcu_core"],
                "port_bindings": {"gnd": "GND", "vdd": "+3V3"},
            }
        ],
    }

    canonical, _expanded = _normalize_stage_response("architecture", payload, {})

    assert [sheet["name"] for sheet in canonical["sheets"]] == ["MCU", "CRYSTAL"]
    selection = canonical["recipe_selections"][0]
    assert selection["requirement_ids"] == ["mcu_core"]
    requirements = {row["id"]: row for row in canonical["requirements"]}
    assert requirements["crystal_8mhz"]["sheet"] == "CRYSTAL"
    assert requirements["crystal_8mhz"]["functional_blocks"] == ["OSCILLATOR"]
    assert requirements["mcu_core"]["functional_blocks"] == ["CONTROLLER"]
    assert "crystal_8mhz" in canonical["unresolved_requirement_ids"]


def test_usb_c_breakout_pin_ports_are_refused_against_the_published_contract():
    """A known lowerer's port vocabulary is executable, not advisory.

    Physical pin numbers are not the published `usb-c-breakout` contract, so the
    claim is refused at architecture ownership with those alternatives named. A
    genuinely model-owned breakout family remains model-owned work.
    """
    payload = {
        "topologies": {"USB": "USB-C receptacle breakout"},
        "rail_voltages": {"VBUS": 5.0},
        "comms_protocols": ["USB"],
        "mcu_present": False,
        "sheets": [{"name": "USB", "stem": "USB", "function": "USB-C receptacle breakout"}],
        "power_nets": ["VBUS", "GND"],
        "inter_sheet_nets": [],
        "assumptions": [],
        "requirements": [
            {
                "id": "usb_receptacle",
                "sheet": "USB",
                "role": "connector",
                "family": "usb-c-breakout",
                "ports": {"PIN1": "VBUS", "PIN2": "GND", "PIN3": "CC1"},
            }
        ],
    }

    with pytest.raises(StageSchemaError, match="unsupported_lowerer_contract") as rejected:
        _normalize_stage_response("architecture", payload, {})

    message = str(rejected.value)
    assert "usb-c-breakout@1" in message
    assert "PIN1" in message and "published port contract" in message

    payload["requirements"][0]["family"] = "custom-usb-breakout"
    canonical, _expanded = _normalize_stage_response("architecture", payload, {})

    assert canonical["recipe_selections"] == []
    assert canonical["unresolved_requirement_ids"] == ["usb_receptacle"]


def test_named_peripheral_cannot_disappear_when_mcu_requirement_exists():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = {
        "topologies": {"MCU": "STM32 controller", "CAN": "CAN transceiver"},
        "rail_voltages": {"+3V3": 3.3},
        "comms_protocols": ["CAN"],
        "mcu_present": True,
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "STM32 controller"},
            {"name": "CAN", "stem": "CAN", "function": "CAN transceiver"},
        ],
        "power_nets": ["+3V3"],
        "inter_sheet_nets": [],
        "assumptions": [],
        "requirements": [
            {
                "id": "mcu",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": "stm32f103c8",
                "exact_part": "STM32F103",
                "ports": {"vdd": "+3V3", "gnd": "GND"},
            }
        ],
    }

    result = resolve_architecture_recipes(payload, {"named_parts": ["SN65HVD230"]})

    assert result.selections[0].recipe == "stm32f103c8t6-minimal@1"
    assert "missing_recipe_requirement" in {row.code for row in result.blocking}
    assert "conflicting_exact_variant" not in {row.code for row in result.blocking}


def test_recipe_ports_bind_unique_declared_net_suffixes():
    payload = {
        "topologies": {"CAN": "SN65HVD230 transceiver"},
        "rail_voltages": {"+3V3": 3.3},
        "comms_protocols": ["CAN"],
        "mcu_present": False,
        "sheets": [
            {"name": "CAN", "stem": "CAN", "function": "SN65HVD230 CAN transceiver"},
            {"name": "HOST", "stem": "HOST", "function": "host interface"},
        ],
        "power_nets": ["+3V3"],
        "inter_sheet_nets": [
            {
                "name": name,
                "endpoints": [
                    {"sheet": "CAN", "direction": "bidirectional"},
                    {"sheet": "HOST", "direction": "bidirectional"},
                ],
            }
            for name in ("CAN_TX", "CAN_RX", "CAN_STBY", "TERM_EN", "CANH", "CANL")
        ],
        "assumptions": [],
        "requirements": [
            {
                "id": "can",
                "sheet": "CAN",
                "role": "bus_interface",
                "family": "can_transceiver",
                "exact_part": None,
                "ports": {},
            },
            {
                "id": "termination_switch",
                "sheet": "CAN",
                "role": "driver",
                "family": "termination_switch",
                "ports": {"enable": "TERM_EN"},
            },
        ],
    }

    canonical, _expanded = _normalize_stage_response("architecture", payload, {})

    selection = canonical["recipe_selections"][0]
    assert selection["port_bindings"] == {
        "canh": "CANH",
        "canl": "CANL",
        "gnd": "GND",
        "rx": "CAN_RX",
        "stby": "CAN_STBY",
        "tx": "CAN_TX",
        "vdd": "+3V3",
    }
    assert "TERM_EN" in {net["name"] for net in canonical["inter_sheet_nets"]}
    assert "termination_switch" in canonical["unresolved_requirement_ids"]


def _shared_net(expansion, ref: str, *pins: str) -> list[str]:
    """Net names whose endpoints include every one of `pins` on `ref`."""
    wanted = {(ref, pin) for pin in pins}
    return [
        connection.net_name
        for connection in expansion.connections
        if wanted <= {(endpoint.ref, endpoint.pin) for endpoint in connection.endpoints}
    ]


def test_a_declared_enable_binding_is_not_overwritten():
    """A brief that wires `enable` explicitly keeps its net: the recipe default tie is a fallback.

    The reviewed TLV62569 composition publishes an optional `enable` port tied to `input` by default. A design that
    names a net for it (the enable-jumper brief) must keep that binding, and the tie must not turn
    the binding into a `recipe_signal_in_power_nets` refusal.
    """
    from kicraft.design.models import Architecture
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = {
        "topologies": {"POWER": "12 V to 3.3 V synchronous buck converter"},
        "rail_voltages": {"+3V3": 3.3},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {"name": "POWER", "stem": "POWER", "function": "12 V to 3.3 V buck"},
            {"name": "CONN", "stem": "CONN", "function": "enable header"},
        ],
        "power_nets": ["VIN_12V", "+3V3", "GND"],
        "inter_sheet_nets": [
            {
                "name": "EN_JUMPER",
                "endpoints": [
                    {"sheet": "POWER", "direction": "bidirectional"},
                    {"sheet": "CONN", "direction": "bidirectional"},
                ],
            }
        ],
        "assumptions": [],
        "requirements": [
            {
                "id": "reg",
                "sheet": "POWER",
                "role": "regulator",
                "family": "tlv62569-3v3",
                "exact_part": None,
                "ports": {
                    "input": "VIN_12V",
                    "gnd": "GND",
                    "output": "+3V3",
                    "enable": "EN_JUMPER",
                },
            }
        ],
    }

    canonical, _expanded = _normalize_stage_response("architecture", payload, {})

    selection = canonical["recipe_selections"][0]
    assert selection["port_bindings"]["enable"] == "EN_JUMPER"
    assert selection["port_bindings"]["input"] == "VIN_12V"

    result = resolve_architecture_recipes(Architecture.model_validate(payload), {})
    assert not any(row.code == "recipe_signal_in_power_nets" for row in result.blocking)




def test_an_unbound_enable_port_expands_onto_the_input_rail():
    """The pre-change selection shape — no `enable` key — still expands onto the input rail."""
    selection = RecipeSelection(
        recipe="tlv62569-3v3@3",
        instance="buck",
        sheets={role: "POWER" for role in get_recipe("tlv62569-3v3@3").required_sheet_roles},
        port_bindings={"input": "VIN_12V", "gnd": "GND", "output": "+5V"},
    )

    expansion = expand_recipe(selection)

    # VIN (pin 4) and EN (pin 1) sit on one net; no separate `enable` net is created.
    assert _shared_net(expansion, "U1", "4", "1") == ["VIN_12V"]
    assert not any(connection.net_name == "enable" for connection in expansion.connections)


@pytest.mark.parametrize("obsolete", ["tlv62569-3v3@1", "tlv62569-3v3@2"])
def test_tlv62569_old_derived_graph_requires_upstream_regeneration(obsolete):
    from kicraft.design.stage_state import legacy_recipe_resume_stage

    old = RecipeSelection(
        recipe=obsolete, instance="buck", sheets={"power": "POWER"},
        port_bindings={"input": "VIN", "gnd": "GND", "output": "VOUT"},
        requirement_ids=["buck"],
    )
    with pytest.raises(ValueError, match="architecture regeneration"):
        migrate_recipe_selection(old)
    assert legacy_recipe_resume_stage({
        "architecture": {"recipe_selections": [old.model_dump()]}, "bom": {"parts": []},
    }) == "architecture"


@pytest.mark.parametrize(("old", "new"), [
    ("me6211-3v3@1", "me6211-3v3@2"),
    ("mcp1700-3v3@1", "mcp1700-3v3@2"),
    ("ams1117-3v3@1", "ams1117-3v3@2"),
])
def test_old_ldo_graph_requires_architecture_regeneration(old, new):
    from kicraft.design.recipe_versions import legacy_recipe_replacement
    from kicraft.design.stage_state import legacy_recipe_resume_stage

    saved = RecipeSelection(
        recipe=old, instance="ldo", sheets={"power": "POWER"},
        port_bindings={"input": "VIN", "gnd": "GND", "output": "VOUT"},
    )
    assert legacy_recipe_replacement(old) == new
    with pytest.raises(ValueError, match="architecture regeneration"):
        migrate_recipe_selection(saved)
    with pytest.raises(ValueError, match="architecture regeneration"):
        expand_recipe(saved)
    assert saved.recipe == old
    assert legacy_recipe_resume_stage({
        "architecture": {"recipe_selections": [saved.model_dump()]}, "bom": {"parts": []},
    }) == "architecture"
    assert legacy_recipe_resume_stage({
        "bom": {"recipe_ownership": [{"recipe": old}]},
    }) == "bom"


def test_raw_legacy_recipe_ids_identify_artifacts_needing_regeneration():
    from kicraft.design.recipe_versions import is_legacy_recipe_id, legacy_recipe_replacement

    assert legacy_recipe_replacement("rp2040-minimal@1") == "rp2040-minimal@3"
    assert legacy_recipe_replacement("rp2040-minimal@2") == "rp2040-minimal@3"
    assert is_legacy_recipe_id("drv8833-dual-motor@1")
    assert is_legacy_recipe_id("tps54331-adjustable@1")
    assert is_legacy_recipe_id("tps54331-12v-1a@1")
    assert not is_legacy_recipe_id("rp2040-minimal@3")
    assert not is_legacy_recipe_id({"recipe": "rp2040-minimal@2"})


@pytest.mark.parametrize(("old", "new"), [
    ("esp32-c3-mini-1-minimal@1", "esp32-c3-mini-1-minimal@2"),
    ("esp32-c3-mini-1-h4x-minimal@1", "esp32-c3-mini-1-h4x-minimal@2"),
    ("tps54331-adjustable@1", "tps54331-adjustable@2"),
    ("tps54331-12v-1a@1", "tps54331-12v-1a@2"),
])
def test_electrical_fix_versions_migrate_saved_selection_without_mutation(old, new):
    definition = get_recipe(new)
    saved = RecipeSelection(
        recipe=old, instance="saved",
        sheets={role: "SHEET" for role in definition.required_sheet_roles},
        port_bindings={port.name: port.name.upper() for port in definition.ports if port.required},
    )
    upgraded = migrate_recipe_selection(saved)
    assert saved.recipe == old
    assert upgraded.recipe == new
    assert upgraded.port_bindings == saved.port_bindings
    assert expand_recipe(saved).selection.recipe == new
    with pytest.raises(ValueError, match="unknown parameters"):
        migrate_recipe_selection(saved.model_copy(update={"parameters": {"unexpected": 1}}))

@pytest.mark.parametrize("old", ["drv8833-dual-motor@1", "drv8833-dual-motor@2", "drv8833-dual-motor@3", "drv8833-dual-motor@4", "drv8833-dual-motor@5"])
@pytest.mark.parametrize("complete", [False, True])
def test_legacy_motor_requires_architecture_regeneration(old, complete):
    from kicraft.design.stage_state import legacy_recipe_resume_stage

    definition = get_recipe("drv8833-dual-motor@6")
    selection = RecipeSelection(
        recipe=old, instance="motor", sheets={"interface": "MOTOR"},
        port_bindings={port.name: port.name.upper() for port in definition.ports
                       if port.required and (complete or port.name != "logic_vdd")},
    )
    with pytest.raises(ValueError, match="architecture regeneration"):
        migrate_recipe_selection(selection)
    assert legacy_recipe_resume_stage({
        "architecture": {"recipe_selections": [
            selection.model_dump(),
            {"recipe": "esp32-c3-mini-1-h4x-minimal@2", "instance": "core",
             "sheets": {"mcu": "MCU"}, "pin_allocations": [
                 {"net": "SLEEP_N", "pin": "20", "capability": "output"}
             ]},
        ]},
        "bom": {"parts": []},
    }) == "architecture"


def test_drv8833_charge_pump_capacitor_is_between_vcp_and_vm():
    definition = get_recipe("drv8833-dual-motor@6")
    selection = RecipeSelection(
        recipe=definition.recipe, instance="motor", sheets={"interface": "MOTOR"},
        port_bindings={port.name: port.name.upper() for port in definition.ports if port.required},
    )
    expansion = expand_recipe(selection)
    driver = next(part.ref for part in expansion.parts if part.recipe_role == "driver")
    capacitor = next(part.ref for part in expansion.parts if part.recipe_role == "vcp_cap")
    nets = {
        (pin.ref, pin.pin): connection.net_name
        for connection in expansion.connections for pin in connection.endpoints
    }
    assert nets[capacitor, "1"] == nets[driver, "11"]
    assert nets[capacitor, "2"] == nets[driver, "12"] == "VM"
    assert nets[capacitor, "1"] != nets[capacitor, "2"]
    pullup = next(part.ref for part in expansion.parts if part.recipe_role == "fault_pullup")
    assert nets[pullup, "1"] == nets[driver, "8"] == "FAULT_N"
    assert nets[pullup, "2"] == "LOGIC_VDD"
    assert nets[pullup, "2"] != nets[driver, "12"]
    assert nets[driver, "3"] == nets[driver, "6"] == nets[driver, "13"] == nets[driver, "17"] == "GND"
    assert not any(part.recipe_role == "sense" for part in expansion.parts)


def _protected_motor_evidence():
    from kicraft.design.models import Architecture

    motor = get_recipe("drv8833-dual-motor@6")
    selections = [
        RecipeSelection(recipe="k7803m-3v3-reverse-blocked@1", instance="supply",
                        sheets={"power": "POWER"}, requirement_ids=["supply"],
                        port_bindings={"input": "VIN", "gnd": "GND", "output": "VM"}),
        RecipeSelection(recipe=motor.recipe, instance="motor", sheets={"interface": "MOTOR"},
                        requirement_ids=["motor"],
                        port_bindings={p.name: ("VM" if p.name == "vm" else p.name.upper())
                                       for p in motor.ports if p.required}),
    ]
    expansions = expand_selections(selections)
    bom = BOM(parts=[p for e in expansions for p in e.parts],
              connections=[c for e in expansions for c in e.connections],
              no_connect_pins=[p for e in expansions for p in e.no_connect_pins],
              recipe_ownership=[e.ownership for e in expansions])
    architecture = Architecture.model_validate({
        "sheets": [{"name": "POWER", "stem": "POWER", "function": "supply"},
                   {"name": "MOTOR", "stem": "MOTOR", "function": "motor"}],
        "power_nets": ["VIN", "VM", "GND"],
        "inter_sheet_nets": [],
        "rail_voltages": {"VIN": 24.0, "VM": 3.3, "GND": 0.0},
        "recipe_selections": selections,
        "requirements": [
            {"id": "supply", "sheet": "POWER", "role": "regulator",
             "family": "k7803m-3v3-reverse-blocked",
             "ports": {"input": "VIN", "output": "VM", "gnd": "GND"}},
            {"id": "motor", "sheet": "MOTOR", "role": "driver",
             "family": "dual-dc-motor-driver", "exact_part": "DRV8833PWPR",
             "ports": {"vm": "VM", "gnd": "GND"}},
        ],
    })
    return architecture, bom


def _move_evidence_pin(bom, ref, pin, target):
    from kicraft.design.models import NetConnection, PinEndpoint

    for connection in bom.connections:
        connection.endpoints = [ep for ep in connection.endpoints
                                if (ep.ref, ep.pin) != (ref, pin)]
    bom.connections.append(NetConnection(sheet="POWER", net_name=target,
                                         endpoints=[PinEndpoint(ref=ref, pin=pin)]))


@pytest.mark.parametrize("damage", ["missing", "upstream", "small", "polarized"])
def test_motor_requires_qualified_local_vm_ceramic_on_driver_rail(damage):
    from kicraft.design.synthesis.validation import (
        _reviewed_identity_for_bom_part, check_reviewed_device_support_networks,
    )

    _, bom = _protected_motor_evidence()
    assert check_reviewed_device_support_networks(bom).ok
    ceramic = next(part for part in bom.parts if part.recipe_role == "vm_ceramic")
    record = _reviewed_identity_for_bom_part(ceramic)
    assert record.operating_limits["capacitance_f"] >= 10e-6
    if damage == "missing":
        bom.parts.remove(ceramic)
    elif damage == "upstream":
        converter = next(part for part in bom.parts if part.recipe_role == "converter")
        raw = next(connection.net_name for connection in bom.connections
                   if any(pin.ref == converter.ref and pin.pin == "3"
                          for pin in connection.endpoints))
        _move_evidence_pin(bom, ceramic.ref, "1", raw)
    elif damage == "small":
        ceramic.mpn = "CL10B104KO8NNNC"
        ceramic.value = "100nF 16V X7R"
        ceramic.footprint = "Capacitor_SMD:C_0603_1608Metric"
    else:
        bulk = next(part for part in bom.parts if part.recipe_role == "vm_bulk")
        ceramic.mpn, ceramic.value = bulk.mpn, bulk.value
        ceramic.symbol, ceramic.footprint = bulk.symbol, bulk.footprint
    assert not check_reviewed_device_support_networks(bom).ok


@pytest.mark.parametrize(("role", "pin", "target"), [
    ("reverse_fet", "2", "VIN"), ("reverse_fet", "3", "VM"),
    ("reverse_controller", "6", "VIN"), ("reverse_controller", "5", "GND"),
])
def test_protected_motor_source_rejects_broken_actual_fet_and_control_graph(role, pin, target):
    from kicraft.design.synthesis.validation import check_reviewed_power_transfer

    architecture, bom = _protected_motor_evidence()
    assert check_reviewed_power_transfer(architecture, bom).ok
    part = next(p for p in bom.parts if p.recipe_role == role)
    _move_evidence_pin(bom, part.ref, pin, target)
    assert not check_reviewed_power_transfer(architecture, bom).ok


def test_protected_motor_rejects_conductive_bypass_and_underrated_bulk():
    from kicraft.design.models import NetConnection, PinEndpoint
    from kicraft.design.synthesis.validation import (
        check_reviewed_device_support_networks, check_reviewed_power_transfer,
    )

    architecture, bom = _protected_motor_evidence()
    assert check_reviewed_power_transfer(architecture, bom).ok
    assert check_reviewed_device_support_networks(bom).ok
    converter = next(p for p in bom.parts if p.recipe_role == "converter")
    raw = next(c.net_name for c in bom.connections
               if any(ep.ref == converter.ref and ep.pin == "3" for ep in c.endpoints))
    resistor = next(p for p in bom.parts if p.recipe_role == "fault_pullup")
    bypass = resistor.model_copy(update={"ref": "R999", "recipe_id": None,
                                         "recipe_role": None, "recipe_instance": None})
    bom.parts.append(bypass)
    bom.connections.extend([
        NetConnection(sheet="POWER", net_name=raw, endpoints=[PinEndpoint(ref="R999", pin="1")]),
        NetConnection(sheet="POWER", net_name="VM", endpoints=[PinEndpoint(ref="R999", pin="2")]),
    ])
    assert not check_reviewed_power_transfer(architecture, bom).ok
    bulk = next(p for p in bom.parts if p.recipe_role == "vm_bulk")
    bulk.mpn = "CL31A476MQHNNNE"  # 47 uF, 6.3 V: the audited failure.
    assert not check_reviewed_device_support_networks(bom).ok

@pytest.mark.parametrize(("binding", "expected"), [(None, "VIN"), ("CONTROL", "CONTROL")])
def test_me6211_enable_binding_or_default_tracks_pin_three(binding, expected):
    ports = {"input": "VIN", "gnd": "GND", "output": "VOUT"}
    if binding is not None:
        ports["enable"] = binding
    expansion = expand_recipe(RecipeSelection(
        recipe="me6211-3v3@2", instance="ldo",
        sheets={"power": "POWER"}, port_bindings=ports,
    ))
    regulator = next(part.ref for part in expansion.parts if part.recipe_role == "regulator")
    nets = {
        (pin.ref, pin.pin): connection.net_name
        for connection in expansion.connections for pin in connection.endpoints
    }
    assert nets[regulator, "3"] == expected


def test_tps54331_external_enable_is_connected_without_local_shunt():
    selection = RecipeSelection(
        recipe="tps54331-adjustable@2",
        instance="buck",
        sheets={"power": "POWER"},
        port_bindings={"input": "VIN", "gnd": "GND", "output": "VOUT", "enable": "CTRL"},
    )
    expansion = expand_recipe(selection)
    converter = next(part.ref for part in expansion.parts if part.recipe_role == "converter")
    assert any(
        connection.net_name == "CTRL"
        and any(pin.ref == converter and pin.pin == "3" for pin in connection.endpoints)
        for connection in expansion.connections
    )
    assert not any(pin.ref == converter and pin.pin == "3" for pin in expansion.no_connect_pins)
    assert not any(part.recipe_role == "enable_jumper" for part in expansion.parts)


def test_an_unbound_feedback_port_expands_onto_the_output():
    """The follower's inverting input (pin 4) ties to its output (pin 1) for unity gain."""
    selection = RecipeSelection(
        recipe="mcp6001-follower@1",
        instance="buffer",
        sheets={role: "OPAMP" for role in get_recipe("mcp6001-follower@1").required_sheet_roles},
        port_bindings={"input": "AIN", "output": "AOUT", "vdd": "VCC", "gnd": "GND"},
    )

    expansion = expand_recipe(selection)

    assert _shared_net(expansion, "U1", "1", "4") == ["AOUT"]


def _stm32_without_usb_architecture() -> dict:
    """The held-out shape: a recipe publishes usb_dm/usb_dp, no design wires them.

    Mirrors the committed architecture of the held-out STM32F103C8T6 board: the
    MCU requirement binds only its supply and programming ports, the brief asks
    for no USB data, and no other requirement names those nets.
    """
    return {
        "topologies": {"CONTROLLER": "STM32F103C8T6 controller"},
        "rail_voltages": {"GND": 0.0, "+3V3": 3.3},
        "comms_protocols": ["SWD"],
        "mcu_present": True,
        "sheets": [
            {"name": "CONTROLLER", "stem": "CONTROLLER", "function": "STM32F103 controller"},
            {"name": "SWD HEADER", "stem": "SWD_HEADER", "function": "SWD programming header"},
            {"name": "RESET", "stem": "RESET", "function": "Physical reset button"},
        ],
        "power_nets": ["GND", "+3V3"],
        "inter_sheet_nets": [
            {
                "name": "GND",
                "endpoints": [
                    {"sheet": "CONTROLLER", "direction": "bidirectional"},
                    {"sheet": "SWD HEADER", "direction": "bidirectional"},
                ],
            },
            {
                "name": "+3V3",
                "endpoints": [
                    {"sheet": "CONTROLLER", "direction": "input"},
                    {"sheet": "SWD HEADER", "direction": "input"},
                ],
            },
            {
                "name": "SWDIO",
                "endpoints": [
                    {"sheet": "CONTROLLER", "direction": "bidirectional"},
                    {"sheet": "SWD HEADER", "direction": "bidirectional"},
                ],
            },
            {
                "name": "SWCLK",
                "endpoints": [
                    {"sheet": "CONTROLLER", "direction": "bidirectional"},
                    {"sheet": "SWD HEADER", "direction": "bidirectional"},
                ],
            },
            {
                "name": "NRST",
                "endpoints": [
                    {"sheet": "CONTROLLER", "direction": "bidirectional"},
                    {"sheet": "SWD HEADER", "direction": "bidirectional"},
                ],
            },
        ],
        "recipe_selections": [
            {
                "recipe": "stm32f103c8t6-minimal@1",
                "instance": "mcu",
                "sheets": {"mcu": "CONTROLLER"},
                "parameters": {"can_remap": "pb8-pb9"},
                "port_bindings": {
                    "gnd": "GND",
                    "vdd": "+3V3",
                    "swdio": "SWDIO",
                    "swclk": "SWCLK",
                    "nrst": "NRST",
                },
                "requirement_ids": ["mcu"],
            }
        ],
        "requirements": [
            {
                "id": "mcu",
                "sheet": "CONTROLLER",
                "role": "mcu_core",
                "family": "stm32f103c8",
                "exact_part": "STM32F103C8T6",
                "parameters": {"can_remap": "pb8-pb9"},
                "ports": {
                    "gnd": "GND",
                    "vdd": "+3V3",
                    "swdio": "SWDIO",
                    "swclk": "SWCLK",
                    "nrst": "NRST",
                },
            }
        ],
    }


def _stm32_usb_free_board(groups=()):
    """Compose the recorded shape through the real BOM stage normalizer."""
    from kicraft.design.models import Architecture
    from kicraft.design.synthesis.validation import check_no_dangling_signal_nets

    architecture = _stm32_without_usb_architecture()
    canonical, _count = _normalize_stage_response(
        "bom",
        {"groups": list(groups), "arrays": [], "assumptions": [], "substitutions": []},
        {"architecture": architecture},
    )
    bom = BOM.model_validate(canonical)
    verdict = check_no_dangling_signal_nets(Architecture.model_validate(architecture), bom)
    return canonical, bom, verdict


def test_unused_published_recipe_pins_compose_as_no_connects():
    """The held-out blocker: PA11/PA12 published as usb_dm/usb_dp, bound by nothing.

    An unused published pin is unused, not a signal that connects to nothing:
    the compiler completes it as an explicit no-connect, so §9.15 passes where
    the recorded run was refused for two dangling single-pin nets.
    """
    canonical, bom, verdict = _stm32_usb_free_board()

    assert verdict.ok, verdict.offenders
    assert not {"usb_dm", "usb_dp"} & {c.net_name for c in bom.connections}
    assert {"ref": "U1", "pin": "32"} in canonical["no_connect_pins"]
    assert {"ref": "U1", "pin": "33"} in canonical["no_connect_pins"]
    # The recipe still owns those pins -- as no-connects, so the wiring stage cannot
    # re-open them and §9.11 coverage still sees them.
    manifest = next(row for row in canonical["recipe_ownership"] if row["instance"] == "mcu")
    owned = {(row["ref"], row["pin"]): row.get("net") for row in manifest["pins"]}
    assert owned[("U1", "32")] is None and owned[("U1", "33")] is None
    # The pins the design does use keep their wiring.
    assert owned[("U1", "34")] == "SWDIO" and owned[("U1", "7")] == "NRST"


@pytest.mark.parametrize("claim", ["inter_sheet_net", "another_requirement_port"])
def test_published_pin_a_design_references_is_never_dropped(claim):
    """Rule: only a pin nothing references may complete as a no-connect.

    `usb_dm` is claimed by the design two ways a recipe cannot see, so its pin
    must stay wired; `usb_dp`, which nothing claims, still completes as unused.
    """
    from kicraft.design.recipes import complete_unused_published_ports

    architecture = _stm32_without_usb_architecture()
    if claim == "inter_sheet_net":
        architecture["inter_sheet_nets"].append(
            {
                "name": "usb_dm",
                "endpoints": [
                    {"sheet": "CONTROLLER", "direction": "bidirectional"},
                    {"sheet": "SWD HEADER", "direction": "bidirectional"},
                ],
            }
        )
    else:
        architecture["requirements"].append(
            {
                "id": "usb_monitor",
                "sheet": "SWD HEADER",
                "role": "sensor",
                "family": "usb-monitor",
                "ports": {"signal": "usb_dm"},
            }
        )
    expansions = complete_unused_published_ports(
        expand_selections(architecture["recipe_selections"]),
        architecture,
    )
    (expansion,) = expansions

    usb = [c for c in expansion.connections if c.net_name == "usb_dm"]
    assert [f"{e.ref}.{e.pin}" for c in usb for e in c.endpoints] == ["U1.32"]
    unconnected = {pin.pin for pin in expansion.no_connect_pins if pin.ref == "U1"}
    assert "32" not in unconnected  # referenced by the design: the pin stays wired
    assert "33" in unconnected  # nobody references it: the pin completes as unused


def test_a_published_port_the_recipe_ties_itself_stays_wired():
    """A port several of the recipe's own pins share is its internal tie, not a dangle."""
    from kicraft.design.recipes import complete_unused_published_ports

    architecture = _stm32_without_usb_architecture()  # boot0 is bound by nothing
    (expansion,) = complete_unused_published_ports(
        expand_selections(architecture["recipe_selections"]),
        architecture,
    )

    # BOOT0 still ties the MCU pin, its pulldown and the programming header together:
    # the design did not bind the port, but the recipe's own wiring is not an orphan.
    boot0 = next(c for c in expansion.connections if c.net_name == "boot0")
    assert sorted(f"{endpoint.ref}.{endpoint.pin}" for endpoint in boot0.endpoints) == [
        "J1.4",
        "R1.1",
        "U1.44",
    ]
    assert {
        f"{pin.ref}.{pin.pin}" for pin in expansion.no_connect_pins
    }.isdisjoint({"U1.44", "J1.4", "R1.1"})


def test_an_authored_orphan_signal_is_still_refused():
    """The negative control: model-authored wiring that connects to nothing stays a defect."""
    from kicraft.design.models import Architecture
    from kicraft.design.synthesis.validation import check_no_dangling_signal_nets
    from kicraft.server.stage_contracts import _normalize_wiring_stage_response

    canonical, _bom, verdict = _stm32_usb_free_board(
        groups=[
            {
                "id": "reset_button",
                "reference_prefix": "SW",
                "quantity": 1,
                "value": "TL3342",
                "symbol": "kicad-tl3342-button:TL3342",
                "footprint": "Button_Switch_SMD:SW_SPST_TL3342",
                "sheet": "RESET",
            }
        ]
    )
    assert verdict.ok, verdict.offenders

    wired = _normalize_wiring_stage_response(
        {
            "pins": [
                {"ref": "SW1", "pin": "1", "net": "NRST"},
                {"ref": "SW1", "pin": "2", "net": "SPARE"},
            ]
        },
        {"bom": canonical},
    )
    bom = BOM.model_validate({**canonical, **wired})

    orphan = check_no_dangling_signal_nets(Architecture.model_validate(
        _stm32_without_usb_architecture()
    ), bom)
    assert not orphan.ok
    assert any("'SPARE'" in offender for offender in orphan.offenders), orphan.offenders
    # The recipe-owned unused pins stayed no-connects, not nets.
    assert not {"usb_dm", "usb_dp"} & {c.net_name for c in bom.connections}


def test_a_default_tie_must_name_another_port_of_its_recipe():
    """A typo'd or self-referential tie is refused at definition time, never a dangling net."""
    from kicraft.design.recipes.models import RecipeDefinition, RecipePort

    def definition(*ports):
        return RecipeDefinition(
            recipe="toy@1",
            required_sheet_roles=("POWER",),
            ports=tuple(ports),
            parts=(),
            pins=(),
        )

    with pytest.raises(ValueError, match="must name another port of the same recipe"):
        definition(
            RecipePort(name="input", direction="power"),
            RecipePort(name="enable", direction="input", required=False, default_tie="nope"),
        )
    with pytest.raises(ValueError, match="must name another port of the same recipe"):
        definition(
            RecipePort(name="input", direction="power"),
            RecipePort(name="enable", direction="input", required=False, default_tie="enable"),
        )


def test_architecture_discards_unknown_provider_unresolved_ids_before_resolution():
    payload = _add_native_usb(_esp32_architecture_payload())
    payload["unresolved_requirement_ids"] = ["req_mcu", "req_power"]

    canonical, _expanded = _normalize_stage_response("architecture", payload, {})

    assert canonical["unresolved_requirement_ids"] == []




def test_architecture_rejects_n8r8_when_n16r8_was_requested():
    with pytest.raises(StageSchemaError, match="conflicting_exact_variant"):
        _normalize_stage_response(
            "architecture",
            _typed_esp32_architecture(
                family="esp32-s3-wroom-1-module", exact_part="ESP32-S3-WROOM-1-N8R8"
            ),
            {"intent": {"named_parts": ["ESP32-S3-WROOM-1-N16R8"]}},
        )


_NATIVE_FIXTURE_FAMILIES = {
    "esp32-s3-module",
    "esp32-s3-wroom-1-module",
    "esp32-c3-mini-1-module",
    "rp2040",
}


def _add_native_usb(payload, *, dm="USB_D_N", dp="USB_D_P", vbus="VBUS"):
    """Append the mandatory USB data companion for a native-USB MCU fixture."""
    payload["power_nets"] = list(dict.fromkeys([*(payload.get("power_nets") or []), vbus, "GND"]))
    payload.setdefault("rail_voltages", {})[vbus] = 5.0
    payload["sheets"].append({"name": "USB", "stem": "USB", "function": "USB connector"})
    payload.setdefault("requirements", []).append(
        {
            "id": "usb_connector",
            "sheet": "USB",
            "role": "connector",
            "family": "usb-c-usb2-device",
            "exact_part": "USB-C-USB2-DEVICE",
            "ports": {"vbus": vbus, "gnd": "GND", "usb_dm": dm, "usb_dp": dp},
        }
    )
    payload.setdefault("inter_sheet_nets", []).extend(
        {
            "name": net,
            "endpoints": [
                {"sheet": "MCU", "direction": "bidirectional"},
                {"sheet": "USB", "direction": "bidirectional"},
            ],
        }
        for net in (dm, dp)
    )
    return payload


def _typed_esp32_architecture(**requirement_overrides):
    payload = _esp32_architecture_payload()
    include_usb = requirement_overrides.pop("usb", None)
    requirement = {
        "id": "mcu_core",
        "sheet": "MCU",
        "role": "mcu_core",
        "family": "esp32-s3-module",
        "exact_part": "ESP32-S3-MINI-1-N8",
        "parameters": {},
        "ports": {"vdd": "+3V3", "gnd": "GND"},
        "interfaces": [],
        **requirement_overrides,
    }
    payload["requirements"] = [requirement]
    if include_usb is None:
        include_usb = requirement["family"] in _NATIVE_FIXTURE_FAMILIES
    if include_usb:
        _add_native_usb(payload)
    return payload


@pytest.mark.parametrize(
    ("dm", "dp"),
    [("USB_D-", "USB_D+"), ("D-", "D+"), ("USB_D_N", "USB_D_P")],
)
def test_recipe_usb_aliases_preserve_polarity_and_supply_sign(dm, dp):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(ports={}, interfaces=["usb_device"], usb=False)
    payload["power_nets"] = ["-3V3", "+3V3", "GND"]
    _add_native_usb(payload, dm=dm, dp=dp)

    result = resolve_architecture_recipes(payload)

    assert not result.blocking
    assert result.selections[0].port_bindings == {
        "vdd": "+3V3",
        "gnd": "GND",
        "usb_dm": dm,
        "usb_dp": dp,
    }


def test_native_usb_pair_preserves_distinct_polarity_contracts():
    from kicraft.design.architecture_intent import derive_architecture
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    architecture = derive_architecture(
        {
            "mcu_present": True,
            "sheets": [
                {"name": "MCU", "stem": "MCU", "role": "mcu", "function": "ESP32-S3 controller"}
            ],
            "requirements": [
                {
                    "id": "mcu_core",
                    "sheet": "MCU",
                    "role": "mcu_core",
                    "family": "esp32-s3-module",
                    "exact_part": "ESP32-S3-MINI-1-N8",
                    "supply": "+3V3",
                    "programming": "native_usb",
                }
            ],
            "power": {"rails": {"+3V3": {"voltage": 3.3}, "VBUS": {"voltage": 5.0}}},
            "signals": [
                {"name": "HOST_NEGATIVE", "from": "mcu_core.usb_dm", "to": "edge:USB_DATA"},
                {"name": "HOST_POSITIVE", "from": "mcu_core.usb_dp", "to": "edge:USB_DATA"},
            ],
        }
    )

    result = resolve_architecture_recipes(architecture)

    assert not result.blocking
    core = next(row for row in result.selections if row.requirement_ids == ["mcu_core"])
    connector = next(row for row in result.selections if row.recipe == "usb-c-usb2-device@1")
    # The socket is the connector the native-USB pair opens; the two conductors keep the
    # polarity the signals named on both ends, never one shared label.
    assert core.port_bindings["usb_dp"] == "HOST_POSITIVE"
    assert core.port_bindings["usb_dm"] == "HOST_NEGATIVE"
    assert connector.port_bindings["usb_dp"] == "HOST_POSITIVE"
    assert connector.port_bindings["usb_dm"] == "HOST_NEGATIVE"


def _c3_native_usb_architecture():
    payload = _typed_esp32_architecture(
        family="esp32-c3-mini-1-module",
        exact_part="ESP32-C3-MINI-1-N4",
    )
    payload["sheets"][0]["function"] = "ESP32-C3 MCU"
    payload["topologies"] = {"MCU": "ESP32-C3"}
    payload["sheets"].append(
        {"name": "BRIDGE", "stem": "BRIDGE", "function": "external UART bridge"}
    )
    payload["requirements"].append(
        {
            "id": "bridge",
            "sheet": "BRIDGE",
            "role": "bus_interface",
            "family": "external-uart-bridge",
            "ports": {"usb_dp": "USB_D_P", "usb_dm": "USB_D_N"},
        }
    )
    for net in payload["inter_sheet_nets"]:
        if net["name"] in ("USB_D_N", "USB_D_P"):
            net["endpoints"].append({"sheet": "BRIDGE", "direction": "bidirectional"})
    return payload



def test_remote_usb_does_not_hide_missing_c3_uart_application_contract():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_native_usb_architecture()
    payload["inter_sheet_nets"].extend(
        [
            {
                "name": net,
                "endpoints": [
                    {"sheet": "BRIDGE", "direction": bridge_direction},
                    {"sheet": "MCU", "direction": mcu_direction},
                ],
            }
            for net, bridge_direction, mcu_direction in [
                ("UART_TXD", "output", "input"),
                ("UART_RXD", "input", "output"),
            ]
        ]
    )
    result = resolve_architecture_recipes(payload)

    diagnostic = next(row for row in result.blocking if row.requirement_id == "mcu_core")
    assert diagnostic.code == "missing_mcu_application_contract"
    assert {"UART_TXD", "UART_RXD"} <= set(diagnostic.evidence)
    assert not any(row.requirement_ids == ["mcu_core"] for row in result.selections)


def test_explicit_unknown_usb_binding_is_not_replaced_by_owned_alias():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(usb=False)
    payload["requirements"][0]["ports"].update(usb_dp="MISSPELLED_DP", usb_dm="MISSPELLED_DM")
    payload["power_nets"] = ["+3V3", "VBUS", "GND"]
    payload["sheets"].append({"name": "USB", "stem": "USB", "function": "USB connector"})
    payload["requirements"].append(
        {
            "id": "connector",
            "sheet": "USB",
            "role": "connector",
            "family": "usb-c-usb2-device",
            "exact_part": "USB-C-USB2-DEVICE",
            "ports": {"vbus": "VBUS", "gnd": "GND", "usb_dp": "USB_D_P", "usb_dm": "USB_D_N"},
        }
    )
    payload["inter_sheet_nets"] = [
        {
            "name": net,
            "endpoints": [
                {"sheet": "USB", "direction": "bidirectional"},
                {"sheet": "MCU", "direction": "bidirectional"},
            ],
        }
        for net in ("USB_D_P", "USB_D_N", "MISSPELLED_DP", "MISSPELLED_DM")
    ]
    result = resolve_architecture_recipes(payload)

    # The explicit (wrong) bindings stand: the MCU never picks up the owned `usb_dm`/
    # `usb_dp` nets, and the socket is left with conductors no recipe port owns.
    diagnostic = next(row for row in result.blocking if row.requirement_id == "mcu_core")
    assert diagnostic.code == "missing_mcu_application_contract"
    assert {"USB_D_N", "USB_D_P"} <= set(diagnostic.evidence)
    orphaned = next(row for row in result.blocking if row.requirement_id == "connector")
    assert orphaned.code == "unsupported_recipe_endpoint"
    assert {"MISSPELLED_DP", "MISSPELLED_DM"} <= set(orphaned.evidence)
    assert not any(row.requirement_ids == ["mcu_core"] for row in result.selections)


def _grounded_usb_shield_architecture():
    payload = _typed_esp32_architecture(
        family="stm32f103c8",
        exact_part="STM32F103C8T6",
        ports={"vdd": "+3V3", "gnd": "GND", "usb_dp": "USB_D_P", "usb_dm": "USB_D_N"},
        interfaces=["usb_device"],
    )
    payload["sheets"][0]["function"] = "STM32F103C8T6 MCU"
    payload["topologies"] = {"MCU": "STM32F103C8T6"}
    payload["power_nets"] = ["+3V3", "VBUS", "GND"]
    payload["sheets"].append({"name": "USB C", "stem": "USB_C", "function": "USB-C input"})
    payload["requirements"].append(
        {
            "id": "usb_c_connector",
            "sheet": "USB C",
            "role": "connector",
            "family": "usb-c-usb2-device",
            "exact_part": "USB-C-USB2-DEVICE",
            "ports": {
                "vbus": "VBUS",
                "gnd": "GND",
                "shield": "GND",
                "usb_dp": "USB_D_P",
                "usb_dm": "USB_D_N",
            },
        }
    )
    payload["inter_sheet_nets"] = [
        {
            "name": net,
            "endpoints": [
                {"sheet": "USB C", "direction": "bidirectional"},
                {"sheet": "MCU", "direction": "bidirectional"},
            ],
        }
        for net in ("USB_D_P", "USB_D_N")
    ]
    return payload


@pytest.mark.parametrize("intent", [
    {"goal": "Use UMW USBLC6-2SC6 for the USB connector"},
    {"named_parts": ["UMW USBLC6-2SC6"]},
    {"constraints": ["USB ESD shall be LCSC C2687116"]},
])
def test_usb_connector_refuses_explicit_umw_not_st(intent):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    result = resolve_architecture_recipes(_grounded_usb_shield_architecture(), intent)
    assert any(row.code == "unsupported_protected_variant" for row in result.blocking)
    assert not any(row.recipe == "usb-c-usb2-device@1" for row in result.selections)


@pytest.mark.parametrize("family", ["usb-c-usb2-device", "usb-c-power-sink"])
@pytest.mark.parametrize("exact_component", [False, True])
def test_reviewed_usb_shield_ground_binding_expands_to_all_shell_pads(family, exact_component):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _grounded_usb_shield_architecture()
    connector = payload["requirements"][1]
    if family == "usb-c-power-sink":
        connector.update(family=family, exact_part="USB-C-5V-SINK")
        connector["ports"] = {"vbus": "VBUS", "gnd": "GND", "shield": "GND"}
        payload["requirements"] = [connector]
        payload["sheets"] = [payload["sheets"][1]]
        payload.update(mcu_present=False, topologies={}, inter_sheet_nets=[])
    if exact_component:
        connector["exact_part"] = "TYPE-C-31-M-12"
    result = resolve_architecture_recipes(
        payload, {"named_parts": ["TYPE-C-31-M-12"]} if exact_component else None
    )

    assert not result.blocking
    selection = next(row for row in result.selections if row.requirement_ids == ["usb_c_connector"])
    expansion = expand_recipe(selection)
    connector_ref = next(part.ref for part in expansion.parts if part.mpn == "TYPE-C-31-M-12")
    assert {
        endpoint.pin
        for connection in expansion.connections
        if connection.net_name == "GND"
        for endpoint in connection.endpoints
        if endpoint.ref == connector_ref
    } == {"1", "2", "3", "4", "A1B12", "B1A12"}


@pytest.mark.parametrize("ambiguous_family", [False, True])
def test_shared_connector_mpn_requires_one_explicit_reviewed_family(ambiguous_family):
    from kicraft.design.recipes.models import RegisteredRecipe
    from kicraft.design.recipes.registry import registered_recipes
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _grounded_usb_shield_architecture()
    connector = payload["requirements"][1]
    connector["exact_part"] = "TYPE-C-31-M-12"
    recipes = registered_recipes()
    if ambiguous_family:
        original = next(row for row in recipes if row.definition.family == "usb-c-usb2-device")
        duplicate = RegisteredRecipe(
            definition=original.definition.model_copy(
                update={"recipe": "other-usb-device@1", "exact_part": "OTHER-USB-DEVICE"}
            ),
            expand=original.expand,
        )
        recipes = (*recipes, duplicate)
    else:
        connector["family"] = "unspecified-usb-connector"
    result = resolve_architecture_recipes(payload, registry=recipes)
    assert any(
        row.code == "unsupported_protected_variant" and row.requirement_id == "usb_c_connector"
        for row in result.blocking
    )
    assert not any(row.requirement_ids == ["usb_c_connector"] for row in result.selections)


@pytest.mark.parametrize(
    ("port", "net", "code"),
    [
        ("usb_dp", "GND", "recipe_signal_in_power_nets"),
        ("shield", "VBUS", "recipe_signal_in_power_nets"),
    ],
)
def test_groundable_shield_preserves_signal_power_and_ownership_errors(port, net, code):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _grounded_usb_shield_architecture()
    payload["requirements"][1]["ports"][port] = net
    result = resolve_architecture_recipes(payload)

    assert any(
        row.code == code and row.requirement_id == "usb_c_connector" and net in row.evidence
        for row in result.blocking
    )
    assert not any(row.requirement_ids == ["usb_c_connector"] for row in result.selections)


def test_legacy_named_peripheral_cannot_reassign_later_mcu_sheet():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _esp32_architecture_payload()
    payload["sheets"].append(
        {
            "name": "CAN",
            "stem": "CAN",
            "function": "SN65HVD230 CAN transceiver",
        }
    )
    payload["inter_sheet_nets"] = [
        {
            "name": name,
            "endpoints": [
                {"sheet": "CAN", "direction": "bidirectional"},
                {"sheet": "MCU", "direction": "bidirectional"},
            ],
        }
        for name in ("CAN_TX", "CAN_RX", "CANH", "CANL")
    ]
    _add_native_usb(payload)
    payload["requirements"].append(
        {
            "id": "can",
            "sheet": "CAN",
            "role": "bus_interface",
            "family": "sn65hvd230-can-node",
            "exact_part": "SN65HVD230",
        }
    )

    result = resolve_architecture_recipes(
        payload, {"named_parts": ["SN65HVD230", "ESP32-S3-MINI-1-N8"]}
    )

    diagnostic = next(
        row for row in result.blocking if row.code == "missing_mcu_application_contract"
    )
    assert diagnostic.sheet == "MCU"
    assert {"CAN_TX", "CAN_RX", "CANH", "CANL"} <= set(diagnostic.evidence)
    assert next(row for row in result.requirements if row.role == "mcu_core").sheet == "MCU"
    selections = {selection.recipe: selection for selection in result.selections}
    assert selections["sn65hvd230-can-node@1"].sheets == {"interface": "CAN"}


def test_coin_cell_mcu_requires_explicit_supply_binding_without_inventing_rail():
    from kicraft.design.architecture_intent import ArchitectureIntentError, derive_architecture
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    intent = {
        "mcu_present": True,
        "sheets": [
            {"name": "MCU", "stem": "MCU", "role": "mcu", "function": "ATtiny1614 microcontroller"},
            {
                "name": "BATTERY INPUT",
                "stem": "BATTERY_INPUT",
                "role": "power_input",
                "function": "Direct CR2032 supply",
            },
        ],
        "requirements": [
            {
                "id": "mcu_core",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": "attiny1614",
                "exact_part": "ATTINY1614-SSNR",
                "programming": "updi",
            },
            {
                "id": "battery_holder",
                "sheet": "BATTERY INPUT",
                "role": "power_input",
                "family": "coin-cell-holder",
                "parameters": {"cell_format": "CR2032"},
                "ties": {"negative": "GND"},
            },
        ],
        "power": {"rails": {"VBAT": {"voltage": 3.0, "from": "battery_holder.positive"}}},
        "comms_protocols": ["UPDI"],
    }

    # A 3.0 V coin cell is not the 3.3 V logic rail: nothing may complete `vdd` for the
    # MCU, and a supply the design never stated is a refusal, never a guessed rail.
    with pytest.raises(ArchitectureIntentError) as rejected:
        derive_architecture(intent)
    diagnostic = next(
        row for row in rejected.value.diagnostics if row.code == "unbound_required_port"
    )
    assert diagnostic.requirement_id == "mcu_core"
    assert "vdd" in diagnostic.message

    intent["requirements"][0]["supply"] = "VBAT"
    architecture = derive_architecture(intent)
    mcu = next(row for row in architecture.requirements if row.id == "mcu_core")
    assert mcu.ports == {"gnd": "GND", "vdd": "VBAT"}
    net = next(row for row in architecture.inter_sheet_nets if row.name == "VBAT")
    assert [(endpoint.sheet, endpoint.direction) for endpoint in net.endpoints] == [
        ("BATTERY INPUT", "output"),
        ("MCU", "input"),
    ]
    resolved = resolve_architecture_recipes(architecture)
    assert not resolved.blocking
    assert next(
        row for row in resolved.selections if row.requirement_ids == ["mcu_core"]
    ).port_bindings == {"gnd": "GND", "vdd": "VBAT"}


def _round2_architecture_failure(name):
    frozen = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "stage_reliability"
            / "architecture_round2_20260911.json"
        ).read_text()
    )
    return frozen[name]


def test_frozen_badge_false_mcu_flag_cannot_bypass_named_ownership():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    frozen = _round2_architecture_failure("badge")
    result = resolve_architecture_recipes(frozen["architecture"], frozen["intent"])
    requirement = next(row for row in result.requirements if row.role == "mcu_core")
    assert requirement.sheet == "MCU"
    assert requirement.exact_part == "ATTINY1614-SSNR"
    # The coin-cell rail is 3.0 V, not the completed 3.3 V logic rail: without the design
    # stating its supply, the owned MCU keeps no `vdd` at all. Nothing is invented.
    assert "vdd" not in requirement.ports
    assert not result.selections

    # Supplying the actual coin-cell rail exposes, rather than hides, the next
    # missing contract: neither LED prose nor touch labels allocate MCU pins.
    frozen["architecture"]["requirements"] = [
        requirement.model_copy(update={"ports": {"vdd": "VBAT", "gnd": "GND"}}).model_dump()
    ]
    repaired_supply = resolve_architecture_recipes(frozen["architecture"], frozen["intent"])
    diagnostic = next(
        row for row in repaired_supply.blocking if row.code == "missing_mcu_application_contract"
    )
    assert {"LED_0", "LED_6", "TOUCH_0", "TOUCH_1"} <= set(diagnostic.evidence)
    assert "UPDI" not in diagnostic.evidence
    assert not repaired_supply.selections


def test_frozen_can_empty_application_contract_and_phantom_termination_fail_at_origin():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    frozen = _round2_architecture_failure("can")
    result = resolve_architecture_recipes(frozen["architecture"], frozen["intent"])
    diagnostics = {row.code: row for row in result.blocking}
    assert {"CAN_TX", "CAN_RX"} <= set(diagnostics["missing_mcu_application_contract"].evidence)
    assert diagnostics["unsupported_recipe_endpoint"].sheet == "CAN TRANSCEIVER"
    assert "CAN_TERM" in diagnostics["unsupported_recipe_endpoint"].evidence
    assert not result.selections
    assert "CAN_TERM" in {net["name"] for net in frozen["architecture"]["inter_sheet_nets"]}


def _typed_can_architecture():
    return {
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "STM32F103C8T6 microcontroller"},
            {"name": "CAN", "stem": "CAN", "function": "SN65HVD230 transceiver"},
            {"name": "BUS", "stem": "BUS", "function": "CAN bus connector"},
        ],
        "topologies": {"MCU": "STM32F103C8T6", "CAN": "SN65HVD230"},
        "rail_voltages": {"+3V3": 3.3},
        "power_nets": ["+3V3", "GND"],
        "comms_protocols": ["CAN"],
        "mcu_present": True,
        "inter_sheet_nets": [
            {
                "name": name,
                "endpoints": [
                    {"sheet": first, "direction": "bidirectional"},
                    {"sheet": second, "direction": "bidirectional"},
                ],
            }
            for name, first, second in [
                ("HOST_DRIVE", "MCU", "CAN"),
                ("HOST_SENSE", "MCU", "CAN"),
                ("TRUNK_HIGH", "CAN", "BUS"),
                ("TRUNK_LOW", "CAN", "BUS"),
            ]
        ],
        "requirements": [
            {
                "id": "core",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": "stm32f103c8",
                "ports": {"vdd": "+3V3", "gnd": "GND"},
            },
            {
                "id": "phy",
                "sheet": "CAN",
                "role": "bus_interface",
                "family": "sn65hvd230",
                "exact_part": "SN65HVD230",
                "ports": {
                    "vdd": "+3V3",
                    "gnd": "GND",
                    "tx": "HOST_DRIVE",
                    "rx": "HOST_SENSE",
                    "canh": "TRUNK_HIGH",
                    "canl": "TRUNK_LOW",
                },
            },
        ],
        "assumptions": [],
    }


@pytest.mark.parametrize("typed_peer", [False, True])
@pytest.mark.parametrize("reversed_direction", [False, True])
def test_owned_port_inference_respects_local_signal_direction(typed_peer, reversed_direction):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_can_architecture()
    phy = payload["requirements"][1]
    del phy["ports"]["tx"]
    del phy["ports"]["rx"]
    for net, name, direction in zip(
        payload["inter_sheet_nets"][:2], ("CAN_TX", "CAN_RX"), ("input", "output"), strict=True
    ):
        net["name"] = name
        local_direction = (
            ("output" if direction == "input" else "input") if reversed_direction else direction
        )
        net["endpoints"] = [
            {"sheet": "CAN", "direction": local_direction},
            {"sheet": "MCU", "direction": "output" if local_direction == "input" else "input"},
        ]
    if typed_peer:
        payload["requirements"].append(
            {
                "id": "host_connector",
                "sheet": "MCU",
                "role": "connector",
                "family": "application-connector",
                "ports": {"tx": "CAN_TX", "rx": "CAN_RX"},
            }
        )
    result = resolve_architecture_recipes(payload)

    if reversed_direction:
        # A mis-directed endpoint funds no port of the peer either: the conductor stays
        # unowned and the transceiver is refused, never wired by label.
        diagnostic = next(row for row in result.blocking if row.requirement_id == "phy")
        assert diagnostic.code == "unsupported_recipe_endpoint"
        assert {"CAN_TX", "CAN_RX"} <= set(diagnostic.evidence)
        assert not any(row.requirement_ids == ["phy"] for row in result.selections)
    else:
        selection = next(row for row in result.selections if row.requirement_ids == ["phy"])
        assert selection.port_bindings["tx"] == "CAN_TX"
        assert selection.port_bindings["rx"] == "CAN_RX"


def test_explicit_unknown_signal_is_not_replaced_by_owned_suffix():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_can_architecture()
    payload["requirements"][1]["ports"]["stby"] = "STBY"
    payload["inter_sheet_nets"].append(
        {
            "name": "CAN_STBY",
            "endpoints": [
                {"sheet": "CAN", "direction": "input"},
                {"sheet": "BUS", "direction": "output"},
            ],
        }
    )
    result = resolve_architecture_recipes(payload)

    diagnostic = next(row for row in result.blocking if row.requirement_id == "phy")
    # The explicitly bound net is not the conductor the sheet carries: the transceiver's
    # `stby` is never silently rebound to the owned CAN_STBY label.
    assert diagnostic.code == "unsupported_recipe_endpoint"
    assert "CAN_STBY" in diagnostic.evidence
    assert not any(row.requirement_ids == ["phy"] for row in result.selections)


def test_typed_can_peer_allocates_real_remapped_stm32_package_pins_and_persists_contract():
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution

    resolved = apply_architecture_recipe_resolution(_typed_can_architecture())
    core = next(row for row in resolved.requirements if row.id == "core")
    assert core.interfaces == ["can_controller"]
    assert core.ports["can_tx"] == "HOST_DRIVE"
    assert core.ports["can_rx"] == "HOST_SENSE"
    selection = next(row for row in resolved.recipe_selections if row.requirement_ids == ["core"])
    assert {(row.net, row.pin, row.capability) for row in selection.pin_allocations} == {
        ("HOST_DRIVE", "46", "can-tx"),
        ("HOST_SENSE", "45", "can-rx"),
    }
    assert selection.parameters["can_remap"] == "pb8-pb9"
    repeated = apply_architecture_recipe_resolution(resolved)
    assert repeated.requirements == resolved.requirements
    assert repeated.recipe_selections == resolved.recipe_selections
    expansion = expand_recipe(selection)
    mcu = next(part.ref for part in expansion.parts if part.mpn == "STM32F103C8T6")
    assert {
        (connection.net_name, ep.pin)
        for connection in expansion.connections
        for ep in connection.endpoints
        if ep.ref == mcu and connection.net_name in {"HOST_DRIVE", "HOST_SENSE"}
    } == {("HOST_DRIVE", "46"), ("HOST_SENSE", "45")}


def _direct_touch_architecture(*, touch_count=2, output_count=6, family="attiny1614"):
    return {
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "Microcontroller"},
            {"name": "TOUCH PADS", "stem": "TOUCH_PADS", "function": "Capacitive electrodes"},
            {"name": "LED ARRAY", "stem": "LED_ARRAY", "function": "Indicators"},
            {"name": "BATTERY INPUT", "stem": "BATTERY_INPUT", "function": "Battery source"},
            {"name": "PROGRAMMING", "stem": "PROGRAMMING", "function": "UPDI access"},
        ],
        "topologies": {},
        "rail_voltages": {"VBAT": 3.0},
        "power_nets": ["VBAT", "GND"],
        "comms_protocols": [],
        "mcu_present": True,
        "inter_sheet_nets": [
            {
                "name": net,
                "endpoints": [
                    {"sheet": "MCU", "direction": "bidirectional"},
                    {"sheet": sheet, "direction": "passive"},
                ],
            }
            for net, sheet in [
                ("UPDI", "PROGRAMMING"),
                *((f"TOUCH{index}", "TOUCH PADS") for index in range(touch_count)),
                *((f"LED{index}", "LED ARRAY") for index in range(output_count)),
            ]
        ],
        "requirements": [
            {
                "id": "mcu_core",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": family,
                "functional_blocks": ["MCU"],
                "ports": {
                    "vdd": "VBAT",
                    "gnd": "GND",
                    "updi": "UPDI",
                    **{f"input_touch{index}": f"TOUCH{index}" for index in range(touch_count)},
                    **{f"output_led{index}": f"LED{index}" for index in range(output_count)},
                },
            },
            {
                "id": "battery_holder",
                "sheet": "BATTERY INPUT",
                "role": "power_input",
                "family": "coin-cell-holder",
                "exact_part": None,
                "functional_blocks": ["BATTERY_INPUT"],
                "parameters": {"cell_format": "CR2032"},
                "ports": {"positive": "VBAT", "negative": "GND"},
            },
            *(
                {
                    "id": f"touch_pad{index}",
                    "sheet": "TOUCH PADS",
                    "role": "sensor",
                    "family": "touch-pad",
                    "functional_blocks": ["CAP_TOUCH_PAD"],
                    "ports": {"signal": f"TOUCH{index}"},
                }
                for index in range(touch_count)
            ),
            {
                "id": "updi_header",
                "sheet": "PROGRAMMING",
                "role": "programming",
                "family": "pin-header",
                "parameters": {"rows": 1},
                "ports": {"pin1": "UPDI", "pin2": "GND", "pin3": "VBAT"},
            },
        ],
        "assumptions": [],
    }


@pytest.mark.parametrize(("touch_count", "output_count"), [(2, 6), (6, 5)])
def test_direct_typed_pads_strengthen_input_requests_and_compile_reviewed_ptc_pins(
    touch_count, output_count
):
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution

    architecture = _direct_touch_architecture(touch_count=touch_count, output_count=output_count)
    intent = {"named_parts": ["CR2032", "BS-07-A1BJ001", "ATTINY1614-SSNR"]}
    resolved = apply_architecture_recipe_resolution(architecture, intent)
    (selection,) = resolved.recipe_selections
    expected = {
        **{f"TOUCH{index}": "touch" for index in range(touch_count)},
        **{f"LED{index}": "output" for index in range(output_count)},
    }
    allocations = {row.net: row for row in selection.pin_allocations}
    assert {net: row.capability for net, row in allocations.items()} == expected
    assert len({row.pin for row in allocations.values()}) == len(expected)
    assert {row.pin for row in allocations.values() if row.capability == "touch"} <= {
        "2",
        "3",
        "4",
        "5",
        "8",
        "9",
    }
    expansion = expand_recipe(selection)
    mcu = next(part for part in expansion.parts if part.mpn == "ATTINY1614-SSNR")
    assert {
        (connection.net_name, endpoint.pin)
        for connection in expansion.connections
        for endpoint in connection.endpoints
        if endpoint.ref == mcu.ref and connection.net_name in expected
    } == {(net, row.pin) for net, row in allocations.items()}
    holder = lower_requirement(
        next(row for row in resolved.requirements if row.id == "battery_holder")
    )
    assert [(group.mpn, group.value) for group in holder.groups] == [
        ("BS-07-A1BJ001", "CR2032 holder")
    ]
    assert {(row.pin, row.net) for row in holder.pins} == {("1", "VBAT"), ("2", "GND")}
    repeated = apply_architecture_recipe_resolution(resolved, intent)
    assert repeated.recipe_selections == resolved.recipe_selections
    assert repeated.requirements == resolved.requirements
    assert repeated.inter_sheet_nets == resolved.inter_sheet_nets
    assert {net["name"] for net in architecture["inter_sheet_nets"]} == {
        net.name for net in repeated.inter_sheet_nets
    }


@pytest.mark.parametrize(("family", "touch_count"), [("attiny402", 1), ("attiny1614", 7)])
def test_direct_typed_pads_fail_without_enough_reviewed_touch_pins(family, touch_count):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    result = resolve_architecture_recipes(
        _direct_touch_architecture(family=family, touch_count=touch_count, output_count=0)
    )
    assert not result.selections
    diagnostic = next(row for row in result.blocking if row.code == "unsatisfied_pin_capability")
    assert "touch" in diagnostic.evidence


@pytest.mark.parametrize(
    "conflict", ["output", "fixed_port", "fixed_gpio", "manual_pin", "manual_output"]
)
def test_direct_typed_pad_cannot_reassign_an_incompatible_explicit_contract(conflict):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    architecture = _direct_touch_architecture(touch_count=1, output_count=0)
    core = architecture["requirements"][0]
    if conflict == "output":
        core["ports"]["output_pad"] = core["ports"].pop("input_touch0")
    elif conflict == "fixed_port":
        core["ports"]["updi"] = core["ports"].pop("input_touch0")
    elif conflict == "fixed_gpio":
        core["family"] = "esp32-s3-module"
        core["ports"].pop("updi")
        core["ports"]["gpio21"] = "TOUCH0"  # No reviewed touch channel on GPIO21.
        _add_native_usb(architecture)
    else:
        architecture["recipe_selections"] = [
            {
                "recipe": "attiny1614-updi-minimal@1",
                "instance": "mcu_core",
                "sheets": {"mcu": "MCU"},
                "requirement_ids": ["mcu_core"],
                "pin_allocations": [
                    {
                        "net": "TOUCH0",
                        "pin": "8" if conflict == "manual_output" else "11",
                        "capability": "output" if conflict == "manual_output" else "input",
                    }
                ],
            }
        ]
    result = resolve_architecture_recipes(architecture)
    assert not any(row.requirement_ids == ["mcu_core"] for row in result.selections)
    diagnostic = next(row for row in result.blocking if row.code == "conflicting_pin_capability")
    assert "touch" in diagnostic.evidence


def test_direct_pad_strengthening_preserves_a_compatible_manual_physical_pin():
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution

    architecture = _direct_touch_architecture(touch_count=1, output_count=0)
    core = architecture["requirements"][0]
    core["ports"]["input_touch0"] = "ELECTRODE"
    architecture["requirements"][2]["ports"]["signal"] = "ELECTRODE"
    next(net for net in architecture["inter_sheet_nets"] if net["name"] == "TOUCH0")["name"] = (
        "ELECTRODE"
    )
    architecture["requirements"].append(
        {
            "id": "electrode_probe",
            "sheet": "TOUCH PADS",
            "role": "connector",
            "family": "test-points",
            "ports": {"electrode": "ELECTRODE"},
        }
    )
    architecture["recipe_selections"] = [
        {
            "recipe": "attiny1614-updi-minimal@1",
            "instance": "mcu_core",
            "sheets": {"mcu": "MCU"},
            "requirement_ids": ["mcu_core"],
            "pin_allocations": [{"net": "ELECTRODE", "pin": "8", "capability": "input"}],
        }
    ]
    resolved = apply_architecture_recipe_resolution(architecture)
    (selection,) = resolved.recipe_selections
    assert {(row.net, row.pin, row.capability) for row in selection.pin_allocations} == {
        ("ELECTRODE", "8", "touch")
    }
    expansion = expand_recipe(selection)
    mcu = next(part for part in expansion.parts if part.mpn == "ATTINY1614-SSNR")
    assert {
        (connection.net_name, endpoint.pin)
        for connection in expansion.connections
        for endpoint in connection.endpoints
        if endpoint.ref == mcu.ref and connection.net_name == "ELECTRODE"
    } == {("ELECTRODE", "8")}
    assert apply_architecture_recipe_resolution(resolved).recipe_selections == [selection]


@pytest.mark.parametrize("kind", ["label_only", "external_frontend"])
def test_touch_labels_and_external_frontend_outputs_do_not_require_mcu_ptc(kind):
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution

    architecture = _direct_touch_architecture(touch_count=1, output_count=0, family="attiny402")
    pad = architecture["requirements"][2]
    if kind == "label_only":
        pad["family"] = "digital-sensor"
    else:
        pad["ports"]["signal"] = "ELECTRODE"
        architecture["requirements"].append(
            {
                "id": "frontend",
                "sheet": "TOUCH PADS",
                "role": "sensor",
                "family": "capacitive-touch-controller",
                "ports": {"sense": "ELECTRODE", "output": "TOUCH0"},
            }
        )
    resolved = apply_architecture_recipe_resolution(architecture)
    (selection,) = resolved.recipe_selections
    expansion = expand_recipe(selection)
    mcu = next(part for part in expansion.parts if (part.mpn or "").startswith("ATTINY402-"))
    (allocation,) = selection.pin_allocations
    assert allocation.capability == "input"
    assert (mcu.ref, allocation.pin) in {
        (endpoint.ref, endpoint.pin)
        for connection in expansion.connections
        if connection.net_name == "TOUCH0"
        for endpoint in connection.endpoints
    }


@pytest.mark.parametrize("named", ["CR2032", "ATTINY1614-SSNR"])
def test_primitive_value_labels_cannot_claim_named_hardware_ownership(named):
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    architecture = {
        "sheets": [
            {"name": "BATTERY INPUT", "stem": "BATTERY_INPUT", "function": "Passive components"}
        ],
        "topologies": {},
        "rail_voltages": {"VBAT": 3.0},
        "power_nets": ["VBAT", "GND"],
        "comms_protocols": [],
        "mcu_present": False,
        "inter_sheet_nets": [],
        "assumptions": [],
    }
    architecture["requirements"] = [
        {
            "id": "decoupling",
            "sheet": "BATTERY INPUT",
            "role": "analog_block",
            "family": "explicit-decoupling",
            "parameters": {"count": 1, "value": named},
            "ports": {"vdd": "VBAT", "gnd": "GND"},
        }
    ]
    artifact = lower_requirement(architecture["requirements"][0])
    assert [group.symbol for group in artifact.groups] == ["Device:C"]
    result = resolve_architecture_recipes(architecture, {"named_parts": [named]})
    assert "missing_recipe_requirement" in {row.code for row in result.blocking}


@pytest.mark.parametrize("requested", ["LM35", "LM358", "NE555"])
def test_short_named_part_cannot_be_replaced_by_a_different_amplifier(requested):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    architecture = {
        "sheets": [{"name": "ANALOG", "stem": "ANALOG", "function": "Signal conditioning"}],
        "topologies": {},
        "rail_voltages": {"VCC": 5.0},
        "power_nets": ["VCC", "GND"],
        "comms_protocols": [],
        "mcu_present": False,
        "inter_sheet_nets": [],
        "assumptions": [],
        "requirements": [{
            "id": requested.lower(),
            "sheet": "ANALOG",
            "role": "analog_block",
            "family": "mcp6001-follower",
            "exact_part": "MCP6001T-I/OT",
            "ports": {"vdd": "VCC", "gnd": "GND", "input": "AIN", "output": "AOUT",
                      "feedback": "AOUT"},
        }],
    }
    result = resolve_architecture_recipes(architecture, {"named_parts": [requested]})
    assert any(
        row.code == "missing_recipe_requirement" and requested in row.evidence
        for row in result.blocking
    )


def test_can_capability_cannot_fall_back_to_generic_gpio():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins

    requirement = CircuitRequirement(
        id="core",
        sheet="MCU",
        role="mcu_core",
        family="attiny1614",
        interfaces=["can_controller"],
        ports={"can_tx": "TX", "can_rx": "RX"},
    )
    with pytest.raises(PinAllocationError, match="unsatisfied_pin_capability") as rejected:
        allocate_requirement_pins(get_recipe("attiny1614-updi-minimal@1"), requirement)
    assert rejected.value.capability in {"can-rx", "can-tx"}


def test_stm32_can_rejects_unreviewed_remap_at_architecture():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    architecture = _typed_can_architecture()
    architecture["requirements"][0]["parameters"] = {"can_remap": "pd0-pd1"}
    result = resolve_architecture_recipes(architecture)
    diagnostic = next(row for row in result.blocking if row.code == "conflicting_recipe_parameter")
    assert diagnostic.requirement_id == "core"
    assert "pd0-pd1" in diagnostic.evidence[0]





def test_non_numbered_mcu_advertises_allocatable_output_and_reviewed_touch():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins
    from kicraft.server.stage_runtime import _architecture_recipe_summaries

    summary = next(
        row
        for row in _architecture_recipe_summaries({"named_parts": ["ATTINY1614-SSNR"]})
        if row["recipe"] == "attiny1614-updi-minimal@1"
    )
    assert summary["allocatable_gpios"] == []
    capacities = summary["allocatable_capabilities"]
    assert capacities["output"] == 11
    assert capacities["touch"] == 6
    requirement = CircuitRequirement(
        id="badge",
        sheet="MCU",
        role="mcu_core",
        family=summary["family"],
        exact_part=summary["exact_part"],
        ports={
            "vdd": "VBAT",
            "gnd": "GND",
            "output_led0": "LED0",
            **{f"touch_pad{index}": f"TOUCH{index}" for index in range(capacities["touch"])},
        },
    )
    definition = get_recipe(summary["recipe"])
    allocations = allocate_requirement_pins(definition, requirement)
    assert {(row.net, row.capability) for row in allocations} == {
        ("LED0", "output"),
        *((f"TOUCH{index}", "touch") for index in range(6)),
    }
    assert len({row.pin for row in allocations}) == 7
    assert {row.pin for row in allocations if row.capability == "touch"} == {
        "2",
        "3",
        "4",
        "5",
        "8",
        "9",
    }
    assert {row.pin for row in allocations}.isdisjoint({"1", "10", "14"})
    with pytest.raises(PinAllocationError, match="unsatisfied_pin_capability"):
        allocate_requirement_pins(
            definition,
            requirement.model_copy(
                update={"ports": {**requirement.ports, "touch_extra": "TOUCH_EXTRA"}}
            ),
        )


def test_recipe_capability_counts_exclude_reserved_strapping_and_input_only_outputs(monkeypatch):
    from kicraft.design.recipes import registry
    from kicraft.design.recipes.models import RecipeAllocatablePin, RegisteredRecipe

    pins = (
        RecipeAllocatablePin(role="mcu", pin="1", capabilities=("gpio",)),
        RecipeAllocatablePin(
            role="mcu",
            pin="2",
            capabilities=("gpio", "output", "UART_TX", "adc"),
            input_only=True,
        ),
        RecipeAllocatablePin(
            role="mcu", pin="3", capabilities=("touch", "output", "adc"), reserved=True
        ),
        RecipeAllocatablePin(role="mcu", pin="4", capabilities=("touch", "output"), strapping=True),
        RecipeAllocatablePin(
            role="mcu", pin="5", capabilities=("output", "pwm", "can-tx"), input_only=True
        ),
    )
    definition = get_recipe("attiny1614-updi-minimal@1").model_copy(
        update={"allocatable_pins": pins}
    )
    monkeypatch.setattr(
        registry, "registered_recipes", lambda: (RegisteredRecipe(definition=definition),)
    )
    assert registry.recipe_summaries()[0]["allocatable_capabilities"] == {
        "adc": 1,
        "gpio": 1,
        "input": 2,
        "output": 1,
    }


def test_unsupported_mcu_touch_requests_cannot_consume_generic_digital_pins():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins
    from kicraft.design.recipes.registry import recipe_summaries

    requirement = CircuitRequirement(
        id="badge",
        sheet="MCU",
        role="mcu_core",
        family="attiny402",
        ports={
            "output_led": "LED",
            "touch_0": "TOUCH_0",
            "touch_1": "TOUCH_1",
        },
    )
    summary = next(row for row in recipe_summaries() if row["recipe"] == "attiny402-updi-minimal@1")
    assert "touch" not in summary["allocatable_capabilities"]
    with pytest.raises(PinAllocationError, match="unsatisfied_pin_capability") as rejected:
        allocate_requirement_pins(get_recipe("attiny402-updi-minimal@1"), requirement)
    assert rejected.value.capability == "touch"


def test_frozen_can_logic_signals_cannot_be_relabelled_as_generic_gpio():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    frozen = _round2_architecture_failure("can")
    core = next(row for row in frozen["architecture"]["requirements"] if row["role"] == "mcu_core")
    core["ports"] = {"output_tx": "CAN_TX", "input_rx": "CAN_RX"}
    result = resolve_architecture_recipes(frozen["architecture"], frozen["intent"])
    diagnostic = next(
        row for row in result.blocking if row.code == "missing_mcu_application_contract"
    )
    assert {"CAN_TX: requires can-tx", "CAN_RX: requires can-rx"} <= set(diagnostic.evidence)
    assert not result.selections


def test_mcu_recipe_preserves_valid_gpio_and_rejects_unavailable_contract():
    payload = _typed_esp32_architecture()
    payload["sheets"].append({"name": "HEADER", "stem": "HEADER", "function": "GPIO header"})
    payload["inter_sheet_nets"].extend(
        {
            "name": name,
            "endpoints": [
                {"sheet": "MCU", "direction": "bidirectional"},
                {"sheet": "HEADER", "direction": "bidirectional"},
            ],
        }
        for name in ("GPIO1", "GPIO99")
    )

    with pytest.raises(StageSchemaError, match="unavailable_recipe_gpio") as rejected:
        _normalize_stage_response("architecture", payload, {})
    assert rejected.value.diagnostic["requirement_id"] == "mcu_core"
    assert "GPIO99" in rejected.value.diagnostic["evidence"]
    assert {net["name"] for net in payload["inter_sheet_nets"]} == {
        "USB_D_N",
        "USB_D_P",
        "GPIO1",
        "GPIO99",
    }

    payload["inter_sheet_nets"] = [
        net for net in payload["inter_sheet_nets"] if net["name"] != "GPIO99"
    ]
    canonical, _expanded = _normalize_stage_response("architecture", payload, {})
    assert {net["name"] for net in canonical["inter_sheet_nets"]} == {
        "USB_D_N",
        "USB_D_P",
        "GPIO1",
    }
    assert canonical["recipe_selections"][0]["pin_allocations"][0]["net"] == "GPIO1"







def test_typed_named_mcu_cannot_be_dropped_by_false_presence_flag():
    payload = _add_native_usb(_esp32_architecture_payload())
    payload["mcu_present"] = False
    canonical, _expanded = _normalize_stage_response(
        "architecture", payload, {"intent": {"named_parts": ["ESP32-S3-MINI-1-N8"]}}
    )
    assert canonical["mcu_present"] is True
    mcu = next(row for row in canonical["requirements"] if row["role"] == "mcu_core")
    assert mcu["exact_part"] == "ESP32-S3-MINI-1-N8"
    assert any(mcu["id"] in row["requirement_ids"] for row in canonical["recipe_selections"])


def test_unregistered_mcu_requirement_stays_model_owned_and_keeps_required_nets():
    payload = _typed_esp32_architecture(family="unregistered-controller", exact_part=None)
    payload["topologies"] = {}
    payload["sheets"][0]["function"] = "Application processor"
    payload["sheets"].append({"name": "IO", "stem": "IO", "function": "host interface"})
    payload["inter_sheet_nets"] = [
        {
            "name": "GPIO99",
            "endpoints": [
                {"sheet": "MCU", "direction": "bidirectional"},
                {"sheet": "IO", "direction": "passive"},
            ],
        }
    ]
    canonical, _expanded = _normalize_stage_response("architecture", payload, {})
    assert canonical["recipe_selections"] == []
    assert canonical["unresolved_requirement_ids"] == ["mcu_core"]
    assert {net["name"] for net in canonical["inter_sheet_nets"]} == {"GPIO99"}


@pytest.mark.parametrize(
    ("requested", "candidate", "owned"),
    [
        ("nRF52840", "NRF52840-QIAA-R7", True),
        ("nRF52840", "NRF52832-QFAA-R7", False),
        ("NRF52840-QIAA-R7", "nRF52840", False),
    ],
)
def test_unregistered_named_owner_accepts_only_reviewed_directional_identity(
    requested, candidate, owned
):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(family="unregistered-controller", exact_part=candidate)
    payload["topologies"] = {}
    payload["sheets"][0]["function"] = "Application processor"
    result = resolve_architecture_recipes(payload, {"named_parts": [requested]})
    ownership_errors = [row for row in result.blocking if row.code == "missing_recipe_requirement"]
    assert bool(ownership_errors) is not owned
    assert result.unresolved_requirements == ["mcu_core"]
    assert not result.selections
    assert result.requirements[0].exact_part == candidate


@pytest.mark.parametrize("exact_part", [None, "STM32L0"])
def test_named_family_owns_architecture_without_becoming_a_physical_sku(exact_part):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(family="stm32l0", exact_part=exact_part)
    payload["topologies"] = {}
    payload["sheets"][0]["function"] = "Low-power application processor"
    result = resolve_architecture_recipes(payload, {"named_parts": ["STM32L0"]})
    ownership_errors = [row for row in result.blocking if row.code == "missing_recipe_requirement"]
    assert bool(ownership_errors) is (exact_part is not None)
    assert result.requirements[0].exact_part == exact_part
    assert not result.selections
    if exact_part is None:
        assert not result.blocking
        assert result.unresolved_requirements == ["mcu_core"]


def test_recipe_resolver_blocks_explicit_parameter_and_identity_conflicts():
    with pytest.raises(StageSchemaError, match="conflicting_recipe_parameter"):
        _normalize_stage_response(
            "architecture",
            _typed_esp32_architecture(
                parameters={"native_usb": False},
                interfaces=["usb_device"],
            ),
            {},
        )
    with pytest.raises(StageSchemaError, match="conflicting_exact_variant"):
        _normalize_stage_response(
            "architecture",
            _typed_esp32_architecture(),
            {"intent": {"named_parts": ["RP2040"]}},
        )


def test_esp32_native_usb_and_internal_nets_are_instance_scoped():
    base = RecipeSelection(
        recipe="esp32-s3-mini-1-minimal@1",
        instance="main",
        sheets={"mcu": "MCU"},
        parameters={"native_usb": True},
        port_bindings={
            "vdd": "+3V3",
            "gnd": "GND",
            "usb_dm": "USB_D_N",
            "usb_dp": "USB_D_P",
        },
        requirement_ids=["mcu_core"],
    )
    with_usb = expand_recipe(base)
    second = expand_recipe(
        base.model_copy(update={"instance": "aux", "requirement_ids": ["aux_core"]})
    )
    assert all(part.recipe_role != "program_header" for part in with_usb.parts)
    assert {part.recipe_role for part in with_usb.parts if part.value == "22R"} == {
        "usb_dm_series",
        "usb_dp_series",
    }
    assert set(with_usb.ownership.internal_nets).isdisjoint(second.ownership.internal_nets)
    assert {pin.pin for pin in with_usb.no_connect_pins} >= {"39", "40"}
    assert {connection.net_name for connection in with_usb.connections} >= {
        "USB_D_N",
        "USB_D_P",
    }


def test_pin_allocator_is_order_stable_and_reports_impossible_parallel_bus():
    from kicraft.design.recipes.pin_allocator import PinAllocationRequest

    pins = get_recipe("esp32-s3-mini-1-minimal@1").allocatable_pins
    requests = [
        PinAllocationRequest(
            id=f"d_{index}",
            net=f"D{index}",
            capability="output",
            group="parallel",
            contiguous=True,
        )
        for index in range(13)
    ]
    first = allocate_pins(pins, requests)
    second = allocate_pins(pins, list(reversed(requests)))
    assert first == second
    pin_by_number = {pin.pin: pin.gpio for pin in pins}
    by_net = {allocation.net: pin_by_number[allocation.pin] for allocation in first}
    assert [by_net[f"D{index}"] for index in range(13)] == list(
        range(by_net["D0"], by_net["D0"] + 13)
    )
    assert allocate_pins(pins, requests, existing=first) == first
    with pytest.raises(PinAllocationError, match="unsatisfied_pin_capability"):
        allocate_pins(
            pins,
            [
                PinAllocationRequest(
                    id=f"d_{index}",
                    net=f"D{index}",
                    capability="output",
                    group="parallel",
                    contiguous=True,
                )
                for index in range(32)
            ],
        )


def test_pin_allocator_excludes_reserved_input_only_and_strapping_pins():
    from kicraft.design.models import RecipePinAllocation
    from kicraft.design.recipes.models import RecipeAllocatablePin
    from kicraft.design.recipes.pin_allocator import PinAllocationRequest

    pins = (
        RecipeAllocatablePin(
            role="mcu", pin="1", gpio=0, capabilities=("gpio", "output"), strapping=True
        ),
        RecipeAllocatablePin(role="mcu", pin="2", gpio=1, capabilities=("gpio", "output")),
        RecipeAllocatablePin(
            role="mcu",
            pin="3",
            gpio=2,
            capabilities=("gpio", "input"),
            input_only=True,
        ),
        RecipeAllocatablePin(
            role="mcu", pin="4", gpio=3, capabilities=("gpio", "output"), reserved=True
        ),
    )
    output = PinAllocationRequest(id="out", net="OUT", capability="output")
    assert allocate_pins(pins, [output])[0].pin == "2"
    assert (
        allocate_pins(
            (pins[0],),
            [output.model_copy(update={"allow_strapping": True})],
        )[0].pin
        == "1"
    )
    duplicate_existing = [
        RecipePinAllocation(net="A", pin="2", capability="output"),
        RecipePinAllocation(net="B", pin="2", capability="output"),
    ]
    with pytest.raises(PinAllocationError, match="unsatisfied_pin_capability"):
        allocate_pins(
            pins,
            [
                output.model_copy(update={"id": "a", "net": "A", "group": "bus"}),
                output.model_copy(update={"id": "b", "net": "B", "group": "bus"}),
            ],
            existing=duplicate_existing,
        )


@pytest.mark.parametrize("spelling", ["bare", "prefixed", "identical_duplicates"])
@pytest.mark.parametrize(
    ("interface", "ports", "capabilities"),
    [
        (
            "i2c_controller",
            {"sda": "I2C_SDA", "scl": "I2C_SCL"},
            {"i2c-sda", "i2c-scl"},
        ),
        (
            "spi_controller",
            {"sclk": "SCLK", "mosi": "MOSI", "miso": "MISO", "cs": "CS"},
            {"spi-sclk", "spi-mosi", "spi-miso", "spi-cs"},
        ),
        ("uart", {"tx": "TX", "rx": "RX"}, {"uart-tx", "uart-rx"}),
    ],
)
def test_pin_allocator_allocates_controller_buses_atomically(
    interface, ports, capabilities, spelling
):
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins

    prefix = interface.split("_")[0]
    if spelling != "bare":
        prefixed = {f"{prefix}_{key}": net for key, net in ports.items()}
        ports = prefixed if spelling == "prefixed" else {**ports, **prefixed}

    requirement = CircuitRequirement(
        id="mcu_core",
        sheet="MCU",
        role="mcu_core",
        family="esp32-s3-module",
        ports=ports,
        interfaces=[interface],
    )
    allocations = allocate_requirement_pins(
        get_recipe("esp32-s3-mini-1-minimal@1"),
        requirement,
    )
    assert {allocation.capability for allocation in allocations} == capabilities
    assert len({allocation.pin for allocation in allocations}) == len(allocations)
    assert {allocation.net for allocation in allocations} == set(ports.values())


@pytest.mark.parametrize("defect", [None, "conflict", "missing_miso", "unknown_clock"])
def test_recipe_spi_clock_spelling_allocates_one_physical_bus_or_refuses(defect):
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import PinAllocationError, allocate_requirement_pins

    ports = {"spi_sck": "CLK", "spi_mosi": "MOSI", "spi_miso": "MISO", "spi_cs0": "CS"}
    if defect == "conflict":
        ports["spi_sclk"] = "OTHER_CLK"
    elif defect == "missing_miso":
        del ports["spi_miso"]
    elif defect == "unknown_clock":
        ports["spi_clock"] = ports.pop("spi_sck")
    requirement = CircuitRequirement(
        id="mcu", sheet="MCU", role="mcu_core", family="esp32-s3-module",
        ports=ports, interfaces=["spi_controller"],
    )
    definition = get_recipe("esp32-s3-mini-1-minimal@1")
    if defect is not None:
        with pytest.raises(PinAllocationError) as error:
            allocate_requirement_pins(definition, requirement)
        assert error.value.code == (
            "conflicting_interface_port" if defect == "conflict" else "missing_interface_port"
        )
        return
    allocations = allocate_requirement_pins(definition, requirement)
    canonical = requirement.model_copy(update={"ports": {
        "spi_sclk": "CLK", "spi_mosi": "MOSI", "spi_miso": "MISO", "spi_cs0": "CS",
    }})
    expected = allocate_requirement_pins(definition, canonical)
    assert {(row.net, row.pin, row.capability) for row in allocations} == {
        (row.net, row.pin, row.capability) for row in expected
    }
    assert len({row.pin for row in allocations}) == 4


def test_s3_prefixed_i2c_bindings_own_real_intersheet_nets():
    from kicraft.design.architecture_intent import derive_architecture
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    intent = {
        "mcu_present": True,
        "sheets": [
            {"name": "MCU", "stem": "MCU", "role": "mcu", "function": "ESP32-S3 controller"},
            {"name": "DAQ", "stem": "DAQ", "role": "sensor", "function": "I2C acquisition"},
        ],
        "requirements": [
            {
                "id": "mcu_core",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": "esp32-s3-module",
                "exact_part": "ESP32-S3-MINI-1-N8",
                "supply": "+3V3",
                "programming": "native_usb",
            },
            {
                "id": "daq",
                "sheet": "DAQ",
                "role": "sensor",
                "family": "custom-acquisition",
                "declared_ports": [
                    {"key": "sda", "direction": "bidirectional", "function": "I2C data"},
                    {"key": "scl", "direction": "bidirectional", "function": "I2C clock"},
                ],
            },
        ],
        "power": {"rails": {"+3V3": {"voltage": 3.3}, "VBUS": {"voltage": 5.0}}},
        "signals": [
            {"name": "I2C_SDA", "from": "mcu_core.i2c_sda", "to": "daq.sda"},
            {"name": "I2C_SCL", "from": "mcu_core.i2c_scl", "to": "daq.scl"},
            {"name": "USB_D_N", "from": "mcu_core.usb_dm", "to": "edge:USB_DATA"},
            {"name": "USB_D_P", "from": "mcu_core.usb_dp", "to": "edge:USB_DATA"},
        ],
    }
    architecture = derive_architecture(intent)
    core_requirement = next(row for row in architecture.requirements if row.id == "mcu_core")
    assert core_requirement.ports["i2c_sda"] == "I2C_SDA"
    assert core_requirement.ports["i2c_scl"] == "I2C_SCL"
    assert core_requirement.interfaces == ["i2c_controller"]

    result = resolve_architecture_recipes(architecture)
    assert not result.blocking
    core = next(row for row in result.selections if row.requirement_ids == ["mcu_core"])
    expansion = expand_recipe(core)
    mcu = next(part.ref for part in expansion.parts if part.recipe_role == "mcu")
    for allocation in core.pin_allocations:
        assert any(
            net.net_name == allocation.net
            and any(pin.ref == mcu and pin.pin == allocation.pin for pin in net.endpoints)
            for net in expansion.connections
        )
    assert {row.net for row in core.pin_allocations} == {"I2C_SDA", "I2C_SCL"}


@pytest.mark.parametrize(
    ("ports", "code"),
    [
        ({"sda": "SDA", "i2c_sda": "OTHER", "scl": "SCL"}, "conflicting_interface_port"),
        ({"sensor_sda": "SDA", "i2c_scl": "SCL"}, "missing_interface_port"),
        ({"i2c1_sda": "SDA", "i2c_scl": "SCL"}, "missing_interface_port"),
    ],
)
def test_interface_alias_failures_preserve_required_and_actual_bindings(ports, code):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(
        interfaces=["i2c_controller"], ports={"vdd": "+3V3", "gnd": "GND", **ports}
    )
    result = resolve_architecture_recipes(payload)
    diagnostic = next(row for row in result.blocking if row.code == code)
    assert "sda" in diagnostic.message and "i2c_sda" in diagnostic.message
    for key, net in ports.items():
        assert f"{key!r}: {net!r}" in diagnostic.message
    assert any("required_keys=" in value for value in diagnostic.evidence)
    assert any("actual_bindings=" in value for value in diagnostic.evidence)
    assert not any(row.requirement_ids == ["mcu_core"] for row in result.selections)


def test_advertised_interface_keys_allocate_their_bound_nets():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins
    from kicraft.design.recipes.registry import recipe_summaries

    definition = get_recipe("esp32-s3-mini-1-minimal@1")
    summary = next(row for row in recipe_summaries() if row["recipe"] == definition.recipe)
    for interface, members in summary["interfaces"].items():
        requirement = CircuitRequirement(
            id="core",
            sheet="MCU",
            role="mcu_core",
            family=definition.family,
            interfaces=[interface],
            ports={key: member["capability"] for member in members for key in member["keys"]},
        )
        allocations = allocate_requirement_pins(definition, requirement)
        assert {(row.net, row.capability) for row in allocations} == {
            (member["capability"], member["capability"]) for member in members
        }


def test_fixed_programming_uart_is_not_reallocated_as_an_application_interface():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins

    definition = get_recipe("esp32-c3-mini-1-minimal@2")
    requirement = CircuitRequirement(
        id="core",
        sheet="MCU",
        role="mcu_core",
        family=definition.family,
        interfaces=["uart"],
        ports={"uart_tx": "PROGRAM_TX", "uart_rx": "PROGRAM_RX"},
    )
    assert allocate_requirement_pins(definition, requirement) == []
    application = requirement.model_copy(
        update={"ports": {**requirement.ports, "tx": "APP_TX", "rx": "APP_RX"}}
    )
    allocations = allocate_requirement_pins(definition, application)
    assert {row.net for row in allocations} == {"APP_TX", "APP_RX"}
    assert not {row.pin for row in allocations} & {"30", "31"}
    incomplete = requirement.model_copy(update={"ports": {**requirement.ports, "tx": "APP_TX"}})
    with pytest.raises(PinAllocationError, match="missing_interface_port"):
        allocate_requirement_pins(definition, incomplete)


def test_pin_allocator_honors_explicit_gpio_number_ports():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins

    definition = get_recipe("esp32-c3-mini-1-minimal@2")
    requirement = CircuitRequirement(
        id="mcu_core",
        sheet="MCU",
        role="mcu_core",
        family="esp32-c3-mini-1-module",
        ports={"gpio0": "GPIO0", "gpio10": "GPIO10"},
    )

    allocations = allocate_requirement_pins(definition, requirement)
    by_net = {allocation.net: allocation.pin for allocation in allocations}
    expected = {
        f"GPIO{pin.gpio}": pin.pin for pin in definition.allocatable_pins if pin.gpio in {0, 10}
    }
    assert by_net == expected


def test_explicit_gpio_conflicting_with_saved_allocation_never_moves_to_another_pin():
    from kicraft.design.models import CircuitRequirement, RecipePinAllocation
    from kicraft.design.recipes.models import RecipeAllocatablePin, RecipeDefinition
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins

    pins = tuple(
        RecipeAllocatablePin(role="mcu", pin=str(i + 1), gpio=i, capabilities=("gpio", "adc"))
        for i in range(2)
    )
    definition = RecipeDefinition(
        recipe="example@1", required_sheet_roles=("mcu",), parts=(), pins=(), allocatable_pins=pins,
    )
    requirement = CircuitRequirement(
        id="mcu", sheet="MCU", role="mcu_core", family="example",
        interfaces=["adc"], ports={"gpio0": "A", "adc_0": "B"},
    )
    with pytest.raises(PinAllocationError, match="explicitly requested pin"):
        allocate_requirement_pins(
            definition, requirement,
            existing=[RecipePinAllocation(net="B", pin="1", capability="adc")],
        )


def test_explicit_gpio_and_adc_on_same_net_cannot_move_to_a_capable_neighbor():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.models import RecipeAllocatablePin, RecipeDefinition
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins

    definition = RecipeDefinition(
        recipe="example@1", required_sheet_roles=("mcu",), parts=(), pins=(),
        allocatable_pins=(
            RecipeAllocatablePin(role="mcu", pin="1", gpio=0, capabilities=("gpio",)),
            RecipeAllocatablePin(role="mcu", pin="2", gpio=1, capabilities=("gpio", "adc")),
        ),
    )
    requirement = CircuitRequirement(
        id="mcu", sheet="MCU", role="mcu_core", family="example",
        interfaces=["adc"], ports={"gpio0": "A", "adc_0": "A"},
    )
    with pytest.raises(PinAllocationError, match="explicitly requested pin"):
        allocate_requirement_pins(definition, requirement)


@pytest.mark.parametrize(
    "recipe",
    [
        "esp32-s3-wroom-1-minimal@1",
        "esp32-s3-wroom-1-n16r8-minimal@1",
        "esp32-wroom-32e-minimal@1",
        "esp32-c3-mini-1-minimal@2",
        "stm32f103c8t6-minimal@1",
        "attiny402-updi-minimal@1",
        "attiny412-updi-minimal@1",
        "attiny1614-updi-minimal@1",
        "ch32v003f4p6-minimal@1",
        "ch32v003f4u6-minimal@1",
        "ch32v003j4m6-minimal@1",
    ],
)
def test_wave_a_mcu_recipes_expand_with_complete_ownership(recipe):
    native_usb_recipes = {
        "esp32-s3-wroom-1-minimal@1", "esp32-s3-wroom-1-n16r8-minimal@1",
        "esp32-c3-mini-1-minimal@2",
    }
    definition = get_recipe(recipe)
    bindings = {"vdd": "+3V3", "gnd": "GND"}
    if recipe in native_usb_recipes:
        bindings |= {"usb_dm": "USB_D_N", "usb_dp": "USB_D_P"}
    expansion = expand_recipe(
        RecipeSelection(
            recipe=recipe,
            instance="main",
            sheets={"mcu": "MCU"},
            port_bindings=bindings,
        )
    )
    assert definition.maturity == "production"
    assert definition.exact_part
    assert definition.source_documents
    assert definition.electrical_assertions
    assert all(part.resolution_source == "recipe" for part in expansion.parts)
    owned_pins = {(pin.ref, pin.pin) for pin in expansion.ownership.pins}
    assert len(owned_pins) == len(expansion.ownership.pins)
    if recipe in native_usb_recipes:
        # Native-USB modules ship no UART programming header.
        assert all(part.recipe_role != "program_header" for part in expansion.parts)
    else:
        assert any(part.ref.startswith("J") for part in expansion.parts)
    if recipe in {"ch32v003f4p6-minimal@1", "ch32v003f4u6-minimal@1"}:
        # The WCH-Link path is recipe-owned even when no model adds a public
        # SWIO port binding.
        net_by_endpoint = {
            (pin.ref, pin.pin): pin.net for pin in expansion.ownership.pins
        }
        swio_pin = "18" if recipe.endswith("f4p6-minimal@1") else "15"
        assert net_by_endpoint[("U1", swio_pin)] == net_by_endpoint[("J1", "3")]
    from pathlib import Path

    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    for part in expansion.parts:
        expected = {
            str(pin["number"]) for pin in lookup_pins(part.symbol, project_root=Path("."))["pins"]
        }
        actual = {pin.pin for pin in expansion.ownership.pins if pin.ref == part.ref}
        assert actual == expected, (
            definition.recipe,
            part.ref,
            sorted(expected - actual),
            sorted(actual - expected),
        )


def test_n16r8_expansion_preserves_exact_assets_and_psram_reservations():
    definition = get_recipe("esp32-s3-wroom-1-n16r8-minimal@1")
    expansion = expand_recipe(RecipeSelection(
        recipe=definition.recipe, instance="main", sheets={"mcu": "MCU"},
        port_bindings={"vdd": "+3V3", "gnd": "GND", "usb_dm": "DM", "usb_dp": "DP"},
    ))
    module = next(part for part in expansion.parts if part.mpn == definition.exact_part)
    assert module.value == module.mpn == "ESP32-S3-WROOM-1-N16R8"
    assert module.symbol == "esp32-s3-wroom-1-n16r8:ESP32-S3-WROOM-1"
    assert module.footprint == "esp32-s3-wroom-1-n16r8:WIRELM-SMD_ESP32-S3-WROOM-1"
    assert not {35, 36, 37} & {pin.gpio for pin in definition.allocatable_pins}
    ownership = {pin.pin: pin.net for pin in expansion.ownership.pins if pin.ref == module.ref}
    assert all(ownership[pin] is None for pin in ("28", "29", "30"))
    for pin, role, external in (("13", "usb_dm_series", "DM"), ("14", "usb_dp_series", "DP")):
        resistor = next(part for part in expansion.parts if part.recipe_role == role)
        nets = {p.net for p in expansion.ownership.pins if p.ref == resistor.ref}
        assert nets == {ownership[pin], external}
    assert any(rule.kind == "antenna_keepout" for rule in definition.placement_constraints)


def test_ch32_wch_link_recipe_requires_complete_physical_wiring():
    expansion = expand_recipe(
        RecipeSelection(
            recipe="ch32v003f4p6-minimal@1",
            instance="main",
            sheets={"mcu": "MCU"},
            port_bindings={"vdd": "+3V3", "gnd": "GND"},
        )
    )
    bom = BOM(
        parts=expansion.parts,
        connections=expansion.connections,
        no_connect_pins=expansion.no_connect_pins,
    )
    assert check_mcu_programming_access(bom).ok

    swio = next(
        connection.net_name
        for connection in expansion.connections
        if any(endpoint.ref == "U1" and endpoint.pin == "18" for endpoint in connection.endpoints)
    )
    missing_swio = BOM(
        parts=expansion.parts,
        connections=[
            connection for connection in expansion.connections if connection.net_name != swio
        ],
        no_connect_pins=expansion.no_connect_pins,
    )
    result = check_mcu_programming_access(missing_swio)
    assert not result.ok
    assert any("WCH-Link" in offender and "SWIO" in offender for offender in result.offenders)


def _expand_ch32v003f4u6_probe(requirement):
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins

    definition = get_recipe("ch32v003f4u6-minimal@1")
    allocations = allocate_requirement_pins(definition, requirement)
    expansion = expand_recipe(
        RecipeSelection(
            recipe=definition.recipe,
            instance="ch32_probe",
            sheets={"mcu": "MCU"},
            port_bindings={"vdd": "+3V3", "gnd": "GND", **requirement.ports},
            pin_allocations=allocations,
        )
    )
    mcu_pins = {
        pin.net: pin.pin
        for pin in expansion.ownership.pins
        if pin.ref == "U1"
    }
    return definition, allocations, mcu_pins


def test_ch32v003f4u6_native_expansion_probes_seven_independent_outputs():
    """A MAX7219 serial control bundle plus four indicators fits without pad reuse."""
    from kicraft.design.models import CircuitRequirement

    outputs = {
        "output_0": "DISPLAY_DIN",
        "output_1": "DISPLAY_CLK",
        "output_2": "DISPLAY_LOAD",
        "output_3": "INDICATOR_0",
        "output_4": "INDICATOR_1",
        "output_5": "INDICATOR_2",
        "output_6": "INDICATOR_3",
    }
    _definition, allocations, mcu_pins = _expand_ch32v003f4u6_probe(
        CircuitRequirement(
            id="mcu_core",
            sheet="MCU",
            role="mcu_core",
            family="ch32v003",
            ports=outputs,
        )
    )

    assert len(allocations) == len(outputs)
    assert len({allocation.pin for allocation in allocations}) == len(outputs)
    assert {mcu_pins[net] for net in outputs.values()} == {
        allocation.pin for allocation in allocations
    }
    assert not {mcu_pins[net] for net in outputs.values()} & {"1", "15"}


def test_ch32v003f4u6_native_expansion_probes_adc_i2c_and_buttons_without_reuse():
    """Four light ADCs, Qwiic I²C, and three button inputs use distinct legal pads."""
    from kicraft.design.models import CircuitRequirement

    ports = {
        "adc_0": "LIGHT_ADC_0",
        "adc_1": "LIGHT_ADC_1",
        "adc_2": "LIGHT_ADC_2",
        "adc_3": "LIGHT_ADC_3",
        "i2c_sda": "QWIIC_SDA",
        "i2c_scl": "QWIIC_SCL",
        "input_0": "BUTTON_0",
        "input_1": "BUTTON_1",
        "input_2": "BUTTON_2",
    }
    definition, allocations, mcu_pins = _expand_ch32v003f4u6_probe(
        CircuitRequirement(
            id="mcu_core",
            sheet="MCU",
            role="mcu_core",
            family="ch32v003",
            interfaces=["adc", "i2c_controller"],
            ports=ports,
        )
    )

    assert len({allocation.pin for allocation in allocations}) == len(ports)
    assert mcu_pins["QWIIC_SDA"] == "8"
    assert mcu_pins["QWIIC_SCL"] == "9"
    assert definition.fixed_port_pins == {
        "uart_tx": "19",
        "uart_rx": "20",
        "i2c_sda": "8",
        "i2c_scl": "9",
        "spi_cs0": "8",
        "spi_sck": "12",
        "spi_mosi": "13",
        "spi_miso": "14",
    }
    adc_pins = {
        allocation.pin
        for allocation in allocations
        if allocation.net.startswith("LIGHT_ADC_")
    }
    reviewed_adc_pins = {
        pin.pin for pin in definition.allocatable_pins if "adc" in pin.capabilities
    }
    assert len(adc_pins) == 4
    assert adc_pins <= reviewed_adc_pins
    assert {mcu_pins[net] for net in ports.values()} == {
        allocation.pin for allocation in allocations
    }


def test_ch32v003_family_default_expands_capacity_without_replacing_exact_j4m6():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = {
        "topologies": {"MCU": "CH32V003 controller"},
        "rail_voltages": {"+3V3": 3.3},
        "comms_protocols": [],
        "mcu_present": True,
        "sheets": [{"name": "MCU", "stem": "MCU", "function": "CH32V003 controller"}],
        "power_nets": ["+3V3", "GND"],
        "inter_sheet_nets": [],
        "assumptions": [],
        "requirements": [
            {
                "id": "mcu_core",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": "ch32v003",
                "ports": {"vdd": "+3V3", "gnd": "GND"},
            }
        ],
    }

    generic = resolve_architecture_recipes(payload, {"named_parts": ["CH32V003"]})
    assert generic.selections[0].recipe == "ch32v003f4u6-minimal@1"
    assert generic.exact_parts["mcu_core"] == "CH32V003F4U6"

    exact_j4m6 = resolve_architecture_recipes(
        payload, {"named_parts": ["CH32V003J4M6"]}
    )
    assert exact_j4m6.selections[0].recipe == "ch32v003j4m6-minimal@1"
    assert exact_j4m6.exact_parts["mcu_core"] == "CH32V003J4M6"

def test_wave_b_power_recipes_own_every_symbol_pin():
    from pathlib import Path

    from kicraft.design.recipes.wave_b_power import WAVE_B_POWER_RECIPES
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    for definition in WAVE_B_POWER_RECIPES:
        assert definition.source_documents
        for group in definition.parts:
            expected = {
                str(pin["number"])
                for pin in lookup_pins(group.symbol, project_root=Path("."))["pins"]
            }
            for index in range(group.quantity):
                owned = {
                    pin.pin
                    for pin in (*definition.pins, *definition.no_connects)
                    if pin.role == group.role and pin.index == index
                }
                assert owned == expected, (
                    definition.recipe,
                    group.role,
                    index,
                    sorted(expected - owned),
                    sorted(owned - expected),
                )


def test_tps54331_recipe_uses_a_freewheel_diode_and_floating_enable():
    definition = get_recipe("tps54331-adjustable@2")
    expansion = expand_recipe(
        RecipeSelection(
            recipe=definition.recipe,
            instance="usb_5v_buck",
            sheets={"power": "POWER"},
            port_bindings={"input": "USB5V", "gnd": "GND", "output": "+3V3"},
        )
    )
    parts = {part.recipe_role: part for part in expansion.parts}
    diode = parts["catch_diode"]
    converter = parts["converter"]
    nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }

    assert nets[(diode.ref, "1")] == nets[(converter.ref, "8")]
    assert nets[(diode.ref, "2")] == nets[(converter.ref, "7")] == "GND"
    assert (converter.ref, "3") in {
        (endpoint.ref, endpoint.pin) for endpoint in expansion.no_connect_pins
    }

def test_wave_c_interface_recipes_own_every_symbol_pin():
    from pathlib import Path

    from kicraft.design.recipes.wave_c_interfaces import WAVE_C_INTERFACE_RECIPES
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    for definition in WAVE_C_INTERFACE_RECIPES:
        assert definition.maturity == "production"
        assert definition.source_documents
        for group in definition.parts:
            expected = {
                str(pin["number"])
                for pin in lookup_pins(group.symbol, project_root=Path("."))["pins"]
            }
            for index in range(group.quantity):
                owned = {
                    pin.pin
                    for pin in (*definition.pins, *definition.no_connects)
                    if pin.role == group.role and pin.index == index
                }
                assert owned == expected, (
                    definition.recipe,
                    group.role,
                    index,
                    sorted(expected - owned),
                    sorted(owned - expected),
                )


@pytest.mark.parametrize("supply_voltage", [3.3, 5.0])
def test_ch340c_supply_modes_follow_physical_v3_topology(supply_voltage):
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    bindings = {
        "vbus": "USB5V",
        "vdd": "BRIDGE_SUPPLY",
        "gnd": "GND",
        "usb_dp": "USB_DP",
        "usb_dm": "USB_DM",
        "tx": "HOST_TX",
        "rx": "HOST_RX",
        "dtr_n": "HOST_DTR_N",
        "rts_n": "HOST_RTS_N",
    }
    expansion = expand_recipe(
        RecipeSelection(
            recipe="ch340c-usb-uart@1",
            instance="bridge",
            sheets={"interface": "USB"},
            port_bindings=bindings,
            parameters={"supply_voltage": supply_voltage},
        )
    )
    parts = {part.recipe_role: part for part in expansion.parts}
    bridge = parts["bridge"]
    names = {
        str(pin["number"]): pin["name"]
        for pin in lookup_pins(bridge.symbol, project_root=Path("."))["pins"]
    }
    assert {pin: names[pin] for pin in ("2", "3", "4", "13", "14", "16")} == {
        "2": "TXD",
        "3": "RXD",
        "4": "V3",
        "13": "~{DTR}",
        "14": "~{RTS}",
        "16": "VCC",
    }
    endpoints = [
        (endpoint.ref, endpoint.pin)
        for connection in expansion.connections
        for endpoint in connection.endpoints
    ]
    assert len(endpoints) == len(set(endpoints))
    nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }
    assert {pin: nets[bridge.ref, pin] for pin in ("1", "2", "3", "5", "6", "13", "14", "16")} == {
        "1": "GND",
        "2": "HOST_TX",
        "3": "HOST_RX",
        "5": "USB_DP",
        "6": "USB_DM",
        "13": "HOST_DTR_N",
        "14": "HOST_RTS_N",
        "16": "BRIDGE_SUPPLY",
    }
    v3 = nets[bridge.ref, "4"]
    assert nets[parts["decoupling"].ref, "1"] == "BRIDGE_SUPPLY"
    assert nets[parts["v3_cap"].ref, "1"] == v3
    for role in ("decoupling", "v3_cap"):
        assert parts[role].value == "100nF"
        assert nets[parts[role].ref, "2"] == "GND"
    if supply_voltage == 3.3:
        assert v3 == "BRIDGE_SUPPLY"
    else:
        assert v3 not in bindings.values()
        assert {endpoint for endpoint, net in nets.items() if net == v3} == {
            (bridge.ref, "4"),
            (parts["v3_cap"].ref, "1"),
        }
    connector = parts["connector"]
    assert nets[connector.ref, "A4B9"] == nets[connector.ref, "B4A9"] == "USB5V"
    assert nets[connector.ref, "A6"] == nets[connector.ref, "B6"]
    assert nets[connector.ref, "A7"] == nets[connector.ref, "B7"]
    no_connects = {(pin.ref, pin.pin) for pin in expansion.no_connect_pins}
    assert no_connects.isdisjoint(nets)
    ownership = [(pin.ref, pin.pin) for pin in expansion.ownership.pins]
    assert len(ownership) == len(set(ownership))
    assert set(ownership) == set(nets) | no_connects
    for part in expansion.parts:
        physical = {
            str(pin["number"]) for pin in lookup_pins(part.symbol, project_root=Path("."))["pins"]
        }
        assert {pin for ref, pin in ownership if ref == part.ref} == physical


@pytest.mark.parametrize("supply_voltage", [1.8, float("nan"), "3.3"])
def test_ch340c_rejects_unsupported_supply_modes(supply_voltage):
    with pytest.raises(ValueError, match="supply_voltage"):
        expand_recipe(
            RecipeSelection(
                recipe="ch340c-usb-uart@1",
                instance="bridge",
                sheets={"interface": "USB"},
                parameters={"supply_voltage": supply_voltage},
                port_bindings={
                    "vbus": "USB5V",
                    "vdd": "BRIDGE_SUPPLY",
                    "gnd": "GND",
                    "usb_dp": "USB_DP",
                    "usb_dm": "USB_DM",
                    "tx": "HOST_TX",
                    "rx": "HOST_RX",
                },
            )
        )


def test_unused_can_reference_output_is_isolated_not_grounded():
    expanded = expand_recipe(
        RecipeSelection(
            recipe="sn65hvd230-can-node@1",
            instance="can",
            sheets={"interface": "CAN"},
            port_bindings={
                "vdd": "+3V3",
                "gnd": "GND",
                "tx": "CAN_TX",
                "rx": "CAN_RX",
                "canh": "CANH",
                "canl": "CANL",
            },
        )
    )
    transceiver = next(part.ref for part in expanded.parts if part.recipe_role == "transceiver")
    assert (transceiver, "5") in {(pin.ref, pin.pin) for pin in expanded.no_connect_pins}
    assert not any(
        pin.ref == transceiver and pin.pin == "5"
        for net in expanded.connections
        for pin in net.endpoints
    )


def test_user_mcu_identity_survives_generic_sheet_and_false_presence_flag():
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution
    from kicraft.design.stage_semantics import complete_intent_classification

    brief = "An ATtiny1614 drives an indicator."
    intent = complete_intent_classification(brief, {"goal": brief, "named_parts": []})
    payload = _esp32_architecture_payload()
    payload.update(topologies={}, mcu_present=False, comms_protocols=["UPDI"])
    payload["sheets"][0]["function"] = "mcu"
    result = apply_architecture_recipe_resolution(payload, intent)

    assert result.mcu_present
    assert result.requirements[0].exact_part == "ATTINY1614-SSNR"
    assert result.requirements[0].sheet == "MCU"
    assert result.requirements[0].ports["vdd"] == "+3V3"
    assert result.recipe_selections[0].recipe == "attiny1614-updi-minimal@1"


def test_old_intent_classification_cannot_hide_badge_phantom_endpoint():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _esp32_architecture_payload()
    payload.update(topologies={}, mcu_present=False, rail_voltages={})
    payload["sheets"][0]["function"] = "mcu"
    payload["power_nets"] = [".name", "VCC_3V0", "GND", "VCC_5V"]
    payload["sheets"].append({"name": "LED", "stem": "LED", "function": "indicator"})
    payload["inter_sheet_nets"] = [
        {
            "name": "LED_CTRL",
            "endpoints": [
                {"sheet": "MCU", "direction": "output"},
                {"sheet": "LED", "direction": "input"},
            ],
        }
    ]
    intent = {"goal": "ATtiny1614 badge powered by CR2032", "named_parts": ["CR2032"]}
    result = resolve_architecture_recipes(payload, intent)

    core = next(row for row in result.requirements if row.role == "mcu_core")
    assert core.exact_part == "ATTINY1614-SSNR"
    # The generic classification invents neither the MCU's supply nor an owner for the
    # conductor the design left dangling: the phantom endpoint is reported on the MCU.
    assert "vdd" not in core.ports
    assert (
        "LED_CTRL"
        in next(
            row for row in result.blocking if row.code == "missing_mcu_application_contract"
        ).evidence
    )
    assert not result.selections
    payload["requirements"] = [core.model_copy(update={"ports": {"vdd": "VCC_3V0"}}).model_dump()]
    bound = resolve_architecture_recipes(payload, intent)
    diagnostic = next(
        row for row in bound.blocking if row.code == "missing_mcu_application_contract"
    )
    assert "LED_CTRL" in diagnostic.evidence
    assert not bound.selections


def test_named_mcu_cannot_be_assigned_to_an_arbitrary_sheet():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _esp32_architecture_payload()
    payload.update(topologies={}, mcu_present=False)
    payload["sheets"] = [{"name": "POWER", "stem": "POWER", "function": "power input"}]
    result = resolve_architecture_recipes(payload, {"named_parts": ["ATtiny1614"]})
    assert "missing_mcu_requirement" in {row.code for row in result.blocking}
    assert not result.selections


@pytest.mark.parametrize("part", ["ATtiny1616", "ATtiny1614-OTHER", "STM32F103CBT6"])
def test_negative_named_mcu_variant_cannot_use_existing_generic_requirement(part):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    result = resolve_architecture_recipes(_typed_esp32_architecture(), {"named_parts": [part]})
    assert result.blocking
    assert any(part in row.evidence for row in result.blocking)


def test_c3_supply_completion_uses_unique_typed_rail_and_persists_binding():
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution

    payload = _typed_esp32_architecture(
        family="esp32-c3-mini-1-module", exact_part="ESP32-C3-MINI-1-N4", ports={}, usb=False
    )
    payload["sheets"][0]["function"] = "ESP32-C3 MCU"
    payload["topologies"] = {"MCU": "ESP32-C3"}
    payload["rail_voltages"] = {"VCC_3V3": 3.3, "VCC_5V": 5.0}
    payload["power_nets"] = ["VCC_3V3", "VCC_5V", "GND"]
    _add_native_usb(payload)
    result = apply_architecture_recipe_resolution(payload)
    assert result.requirements[0].ports["vdd"] == "VCC_3V3"
    assert result.recipe_selections[0].port_bindings["vdd"] == "VCC_3V3"


@pytest.mark.parametrize(
    "rails",
    [
        {"LOGIC_A": 3.3, "LOGIC_B": 3.3},
        {"VCC_5V": 5.0},
        {"VDD": 5.0},
        {"VCC_3V3": 5.0},
    ],
)
def test_supply_completion_rejects_ambiguous_or_incompatible_rails(rails):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(family="stm32f103c8", exact_part="STM32F103C8T6", ports={})
    payload["sheets"][0]["function"] = "STM32 MCU"
    payload["topologies"] = {"MCU": "STM32"}
    payload["rail_voltages"] = rails
    payload["power_nets"] = [*rails, "GND"]
    result = resolve_architecture_recipes(payload)
    # Two equal-voltage rails, a 5 V rail, or a name that only reads like 3V3 complete
    # nothing: the MCU is left with no supply binding rather than a guessed rail.
    selection = next(row for row in result.selections if row.requirement_ids == ["mcu_core"])
    assert "vdd" not in selection.port_bindings


def test_supply_completion_does_not_harvest_voltage_substrings_from_signals():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(ports={})
    payload["rail_voltages"] = {}
    payload["power_nets"] = ["ENABLE_VDD", "SENSE_3V3", "GND"]
    result = resolve_architecture_recipes(payload)
    # A signal whose label merely contains a rail name is not a supply: no vdd binding.
    selection = next(row for row in result.selections if row.requirement_ids == ["mcu_core"])
    assert "vdd" not in selection.port_bindings


@pytest.mark.parametrize("exact_part", [None, "CH224K"])
def test_composite_pd_normalization_preserves_physical_identity_and_expansion(exact_part):
    payload = {
        "topologies": {"PD": "CH224K USB-PD sink with selectable output"},
        "rail_voltages": {"VBUS": 5.0},
        "power_nets": ["VBUS", "GND"],
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {"name": "PD", "stem": "PD", "function": "CH224K PD controller"},
            {"name": "USB", "stem": "USB", "function": "USB connector"},
        ],
        "inter_sheet_nets": [
            {
                "name": name,
                "endpoints": [
                    {"sheet": "PD", "direction": "bidirectional"},
                    {"sheet": "USB", "direction": "bidirectional"},
                ],
            }
            for name in ("CC1", "CC2")
        ],
        "requirements": [
            {
                "id": "pd",
                "sheet": "PD",
                "role": "power_input",
                "family": "usb-pd-selectable-trigger",
                "exact_part": exact_part,
                "ports": {"vbus": "VBUS", "gnd": "GND", "cc1": "CC1", "cc2": "CC2"},
            }
        ],
        "assumptions": [],
    }
    state = {"intent": {"named_parts": ["CH224K"]}}
    first, _ = _normalize_stage_response("architecture", payload, state)
    second, _ = _normalize_stage_response("architecture", first, state)

    assert second == first
    pd = next(row for row in second["requirements"] if row["id"] == "pd")
    assert pd.get("exact_part") == exact_part
    selection = next(row for row in second["recipe_selections"] if "pd" in row["requirement_ids"])
    assert selection["recipe"] == "ch224k-pd-selectable@1"
    expanded = expand_recipe(RecipeSelection.model_validate(selection))
    controller = next(part for part in expanded.parts if part.mpn == "CH224K")
    controller_nets = {
        endpoint.pin: net.net_name
        for net in expanded.connections
        for endpoint in net.endpoints
        if endpoint.ref == controller.ref
    }
    assert controller_nets["7"] == "CC1"
    assert controller_nets["6"] == "CC2"


def test_c3_supply_endpoint_disambiguates_equal_voltage_rails_without_guessing():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(
        family="esp32-c3-mini-1-module", exact_part="ESP32-C3-MINI-1-N4", ports={}, usb=False
    )
    payload["sheets"][0]["function"] = "ESP32-C3 MCU"
    payload["topologies"] = {"MCU": "ESP32-C3"}
    payload["rail_voltages"] = {"CORE_RAIL": 3.3, "OTHER_RAIL": 3.3}
    payload["power_nets"] = ["CORE_RAIL", "OTHER_RAIL", "GND"]
    _add_native_usb(payload)
    ambiguous = resolve_architecture_recipes(payload)
    # Two equal 3.3 V rails name no supply: the MCU keeps no vdd binding.
    core = next(row for row in ambiguous.selections if row.requirement_ids == ["mcu_core"])
    assert "vdd" not in core.port_bindings

    # The sheet endpoint the design wired disambiguates them, without guessing.
    payload["sheets"].append({"name": "POWER", "stem": "POWER", "function": "power supply"})
    payload["inter_sheet_nets"].append(
        {
            "name": "CORE_RAIL",
            "endpoints": [
                {"sheet": "POWER", "direction": "output"},
                {"sheet": "MCU", "direction": "input"},
            ],
        }
    )
    bound = resolve_architecture_recipes(payload)
    assert not bound.blocking
    assert bound.requirements[0].ports["vdd"] == "CORE_RAIL"

    # A 5 V rail is not a 3.3 V logic supply, endpoint or not.
    payload["rail_voltages"] = {"CORE_RAIL": 5.0, "OTHER_RAIL": 5.0}
    incompatible = resolve_architecture_recipes(payload)
    core = next(row for row in incompatible.selections if row.requirement_ids == ["mcu_core"])
    assert "vdd" not in core.port_bindings


def _c3_typed_programming_architecture(*, handshakes=True):
    payload = _c3_native_usb_architecture()
    bridge = payload["requirements"][2]
    bridge.update(family="usb-uart-bridge", exact_part="CH340C")
    bridge["ports"].update(
        vbus="VBUS", vdd="+3V3", gnd="GND", tx="HOST_TO_MCU", rx="MCU_TO_HOST"
    )
    # A complete bridge recipe owns its USB-C receptacle.  This fixture models
    # UART programming, not an illegal second PHY sharing the native connector.
    payload["requirements"] = [
        requirement for requirement in payload["requirements"]
        if requirement["id"] != "usb_connector"
    ]
    payload["sheets"] = [sheet for sheet in payload["sheets"] if sheet["name"] != "USB"]
    payload["inter_sheet_nets"] = [
        net for net in payload["inter_sheet_nets"]
        if net["name"] not in {"USB_D_N", "USB_D_P"}
    ]
    signals = [
        ("HOST_TO_MCU", "output", "input"),
        ("MCU_TO_HOST", "input", "output"),
    ]
    if handshakes:
        bridge["ports"].update(dtr_n="HOST_DTR_N", rts_n="HOST_RTS_N")
        signals.extend([("HOST_DTR_N", "output", "input"), ("HOST_RTS_N", "output", "input")])
    payload["inter_sheet_nets"].extend(
        {
            "name": name,
            "endpoints": [
                {"sheet": "BRIDGE", "direction": bridge_direction},
                {"sheet": "MCU", "direction": mcu_direction},
            ],
        }
        for name, bridge_direction, mcu_direction in signals
    )
    payload["sheets"].append({"name": "APP", "stem": "APP", "function": "GPIO header"})
    payload["inter_sheet_nets"].extend(
        {
            "name": f"GPIO{gpio}",
            "endpoints": [
                {"sheet": "MCU", "direction": "bidirectional"},
                {"sheet": "APP", "direction": "passive"},
            ],
        }
        for gpio in (*range(9), 10)
    )
    return payload
@pytest.mark.parametrize("shared", ["both", "dp", "dm"])
def test_usb_uart_bridge_rejects_shared_native_usb_phy_pair(shared):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_native_usb_architecture()
    bridge = payload["requirements"][2]
    bridge.update(family="usb-uart-bridge", exact_part="CH340C")
    bridge["ports"].update(
        vbus="VBUS", vdd="+3V3", gnd="GND", tx="HOST_TO_MCU", rx="MCU_TO_HOST"
    )
    if shared == "dp":
        bridge["ports"]["usb_dm"] = "BRIDGE_ONLY_DM"
    elif shared == "dm":
        bridge["ports"]["usb_dp"] = "BRIDGE_ONLY_DP"
    payload["inter_sheet_nets"].extend(
        {
            "name": net,
            "endpoints": [
                {"sheet": "BRIDGE", "direction": bridge_direction},
                {"sheet": "MCU", "direction": mcu_direction},
            ],
        }
        for net, bridge_direction, mcu_direction in (
            ("HOST_TO_MCU", "output", "input"),
            ("MCU_TO_HOST", "input", "output"),
        )
    )

    result = resolve_architecture_recipes(payload)

    assert result.blocking



def test_direct_ch340c_power_domain_stays_blocked_after_uart_repair():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_typed_programming_architecture()
    core, bridge = payload["requirements"][0], payload["requirements"][1]
    payload["rail_voltages"] = {"VBUS": 5.0, "+3V3": 3.3}
    payload["power_nets"].append("VBUS")
    bridge["ports"]["vdd"] = "VBUS"
    core["ports"].update(uart_tx="HOST_TO_MCU", uart_rx="MCU_TO_HOST")
    rejected = resolve_architecture_recipes(payload)
    assert {"conflicting_programming_contract", "incompatible_programming_power_domain"} <= {
        row.code for row in rejected.blocking
    }
    with pytest.raises(StageSchemaError) as normalized:
        _normalize_stage_response("architecture", payload, {})
    assert {"conflicting_programming_contract", "incompatible_programming_power_domain"} <= {
        row["code"] for row in normalized.value.diagnostic["evidence"]
    }

    core["ports"].update(uart_tx="MCU_TO_HOST", uart_rx="HOST_TO_MCU")
    still_unsafe = resolve_architecture_recipes(payload)
    assert any(row.code == "incompatible_programming_power_domain" for row in still_unsafe.blocking)
    assert bridge["ports"]["vdd"] == "VBUS"

    bridge["ports"]["vdd"] = "+3V3"
    safe = resolve_architecture_recipes(payload)
    assert not safe.blocking
    mcu_selection = next(row for row in safe.selections if row.requirement_ids == ["mcu_core"])
    bridge_selection = next(row for row in safe.selections if row.requirement_ids == ["bridge"])
    assert len(mcu_selection.pin_allocations) == 10
    expansion = expand_recipe(bridge_selection)
    ref = next(part.ref for part in expansion.parts if part.mpn == "CH340C")
    nets = {
        pin.pin: connection.net_name
        for connection in expansion.connections
        for pin in connection.endpoints
        if pin.ref == ref
    }
    assert nets["4"] == nets["16"] == "+3V3"

    bridge["parameters"] = {"supply_voltage": 5.0}
    mismatched_mode = resolve_architecture_recipes(payload)
    assert any(row.code == "conflicting_recipe_supply" for row in mismatched_mode.blocking)
    assert bridge["ports"]["vdd"] == "+3V3"


def test_cp2102n_refuses_five_volt_uart_domain():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_typed_programming_architecture(handshakes=False)
    bridge = payload["requirements"][1]
    bridge.update(family="usb-uart-bridge", exact_part="CP2102N")
    bridge["ports"]["vdd"] = "VBUS"

    result = resolve_architecture_recipes(payload)

    assert any(row.code == "conflicting_recipe_supply" for row in result.blocking)


def test_ch340c_supply_requires_evidence_and_honors_authoritative_rail_voltage():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    ports = {"vbus": "VBUS", "vdd": "VBUS", "gnd": "GND", "usb_dp": "DP", "usb_dm": "DM", "tx": "TX", "rx": "RX"}
    payload = {
        "sheets": [{"name": name, "stem": name, "function": name} for name in ("BRIDGE", "HOST")],
        "power_nets": ["VBUS", "GND"],
        "inter_sheet_nets": [
            {
                "name": name,
                "endpoints": [
                    {"sheet": "BRIDGE", "direction": "bidirectional"},
                    {"sheet": "HOST", "direction": "passive"},
                ],
            }
            for name in ("DP", "DM", "TX", "RX")
        ],
        "requirements": [
            {
                "id": "bridge",
                "sheet": "BRIDGE",
                "role": "bus_interface",
                "family": "usb-uart-bridge",
                "exact_part": "CH340C",
                "ports": ports,
                "parameters": {"supply_voltage": 5.0},
            }
        ],
    }
    unknown = resolve_architecture_recipes(payload)
    assert any(row.code == "conflicting_recipe_supply" for row in unknown.blocking)
    payload["rail_voltages"] = {"VBUS": 5.0}
    standalone = resolve_architecture_recipes(payload)
    assert not standalone.blocking
    expansion = expand_recipe(standalone.selections[0])
    ref = next(part.ref for part in expansion.parts if part.mpn == "CH340C")
    nets = {
        pin.pin: connection.net_name
        for connection in expansion.connections
        for pin in connection.endpoints
        if pin.ref == ref
    }
    assert nets["16"] == "VBUS" and nets["4"] != "VBUS"

    ports["vdd"] = "+5V"
    ports["vbus"] = "USB5V"
    payload["power_nets"] = ["USB5V", "+5V", "GND"]
    payload["rail_voltages"] = {"USB5V": 5.0, "+5V": 3.3}
    conflict = resolve_architecture_recipes(payload)
    assert any(row.code == "conflicting_recipe_supply" for row in conflict.blocking)
    payload["requirements"][0]["parameters"] = {}
    authoritative = resolve_architecture_recipes(payload)
    assert not authoritative.blocking
    assert authoritative.selections[0].parameters["supply_voltage"] == 3.3
    assert authoritative.selections[0].port_bindings["vdd"] == "+5V"


def test_distinct_translator_conductors_are_not_a_direct_ch340c_domain():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_typed_programming_architecture(handshakes=False)
    core, bridge = payload["requirements"][0], payload["requirements"][1]
    bridge["ports"]["vdd"] = "VBUS"
    core["ports"].update(uart_tx="MCU_TX_3V3", uart_rx="MCU_RX_3V3")
    payload["rail_voltages"] = {"VBUS": 5.0, "+3V3": 3.3}
    payload["power_nets"].append("VBUS")
    payload["sheets"].append(
        {"name": "TRANSLATOR", "stem": "TRANSLATOR", "function": "level translation"}
    )
    for net in payload["inter_sheet_nets"]:
        if net["name"] in ("HOST_TO_MCU", "MCU_TO_HOST"):
            for endpoint in net["endpoints"]:
                if endpoint["sheet"] == "MCU":
                    endpoint["sheet"] = "TRANSLATOR"
    payload["inter_sheet_nets"].extend(
        {
            "name": name,
            "endpoints": [
                {"sheet": "MCU", "direction": direction},
                {
                    "sheet": "TRANSLATOR",
                    "direction": "input" if direction == "output" else "output",
                },
            ],
        }
        for name, direction in (("MCU_TX_3V3", "output"), ("MCU_RX_3V3", "input"))
    )
    payload["requirements"].append(
        {
            "id": "translator",
            "sheet": "TRANSLATOR",
            "role": "bus_interface",
            "family": "uart-level-translator",
            "ports": {
                "low_tx": "MCU_TX_3V3",
                "low_rx": "MCU_RX_3V3",
                "high_rx": "MCU_TO_HOST",
                "high_tx": "HOST_TO_MCU",
                "vdd_low": "+3V3",
                "vdd_high": "VBUS",
                "gnd": "GND",
            },
        }
    )
    result = resolve_architecture_recipes(payload)
    assert not result.blocking
    selection = next(row for row in result.selections if row.requirement_ids == ["mcu_core"])
    assert selection.port_bindings["uart_tx"] == "MCU_TX_3V3"
    assert selection.port_bindings["uart_rx"] == "MCU_RX_3V3"


@pytest.mark.parametrize("handshakes", [False, True])
@pytest.mark.parametrize("explicit_contract", [False, True])
@pytest.mark.parametrize("peer_role", ["bus_interface", "programming"])
def test_c3_typed_uart_bridge_uses_fixed_pins_and_preserves_all_ten_gpios(
    handshakes, explicit_contract, peer_role
):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_typed_programming_architecture(handshakes=handshakes)
    payload["requirements"][1]["role"] = peer_role
    if explicit_contract:
        core = payload["requirements"][0]
        bridge = payload["requirements"][1]
        core["interfaces"] = bridge["interfaces"] = ["uart"]
        core["ports"].update(uart_tx="MCU_TO_HOST", uart_rx="HOST_TO_MCU")
        if handshakes:
            core["ports"].update(dtr_n="HOST_DTR_N", rts_n="HOST_RTS_N")
        core["ports"].update({f"gpio{gpio}": f"GPIO{gpio}" for gpio in (*range(9), 10)})
        header_nets = [f"GPIO{gpio}" for gpio in (*range(9), 10)] + ["+3V3", "GND"] * 5
        payload["requirements"].append(
            {
                "id": "gpio_header",
                "sheet": "APP",
                "role": "connector",
                "family": "pin-header",
                "parameters": {"rows": 2},
                "ports": {f"pin{index}": net for index, net in enumerate(header_nets, 1)},
            }
        )
    result = resolve_architecture_recipes(payload)
    assert not result.blocking
    core = next(row for row in result.selections if row.requirement_ids == ["mcu_core"])
    assert core.port_bindings["uart_rx"] == "HOST_TO_MCU"
    assert core.port_bindings["uart_tx"] == "MCU_TO_HOST"
    assert {row.net for row in core.pin_allocations} == {f"GPIO{gpio}" for gpio in (*range(9), 10)}
    assert not {row.pin for row in core.pin_allocations} & {"8", "23", "26", "27", "30", "31"}
    expansion = expand_recipe(core)
    refs = {part.recipe_role: part.ref for part in expansion.parts}
    assert "program_header" not in refs
    nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }
    mcu = refs["mcu"]
    assert nets[mcu, "30"] == "HOST_TO_MCU"
    assert nets[mcu, "31"] == "MCU_TO_HOST"
    assert {
        pin: net
        for (ref, pin), net in nets.items()
        if ref == refs["mcu"] and net.startswith("GPIO")
    } == {
        "12": "GPIO0",
        "13": "GPIO1",
        "5": "GPIO2",
        "6": "GPIO3",
        "18": "GPIO4",
        "19": "GPIO5",
        "20": "GPIO6",
        "21": "GPIO7",
        "22": "GPIO8",
        "16": "GPIO10",
    }
    bridge = next(row for row in result.selections if row.requirement_ids == ["bridge"])
    bridge_expansion = expand_recipe(bridge)
    bridge_ref = next(part.ref for part in bridge_expansion.parts if part.mpn == "CH340C")
    bridge_nets = {
        endpoint.pin: connection.net_name
        for connection in bridge_expansion.connections
        for endpoint in connection.endpoints
        if endpoint.ref == bridge_ref
    }
    assert bridge_nets["2"] == nets[mcu, "30"] == "HOST_TO_MCU"
    assert bridge_nets["3"] == nets[mcu, "31"] == "MCU_TO_HOST"
    assert nets[mcu, "8"] == nets[refs["reset_button"], "1"]
    assert nets[mcu, "23"] == nets[refs["boot_button"], "1"]
    assert {(pin.ref, pin.pin) for pin in expansion.no_connect_pins}.isdisjoint(nets)
    ownership = [(pin.ref, pin.pin) for pin in expansion.ownership.pins]
    assert len(ownership) == len(set(ownership))
    assert (
        nets[refs["download_boot_high_pullup_22"], "2"] == nets[refs["mcu"], "22"] == "GPIO8"
    )
    if handshakes:
        assert core.port_bindings["dtr_n"] == "HOST_DTR_N"
        assert core.port_bindings["rts_n"] == "HOST_RTS_N"
    else:
        assert not any(part.ref.startswith("Q") for part in expansion.parts)
        assert not any(part.recipe_role.startswith("auto_reset") for part in expansion.parts)


@pytest.mark.parametrize(
    ("dtr", "rts", "en", "boot"),
    [(0, 0, 1, 1), (1, 1, 1, 1), (1, 0, 0, 1), (0, 1, 1, 0)],
)
def test_esp32_auto_reset_expansion_has_reference_electrical_truth_table(dtr, rts, en, boot):
    expansion = expand_recipe(
        RecipeSelection(
            recipe="esp32-c3-mini-1-minimal@2",
            instance="auto",
            sheets={"mcu": "MCU"},
            parameters={"auto_reset": True},
            port_bindings={
                "vdd": "+3V3",
                "gnd": "GND",
                "usb_dm": "USB_D_N",
                "usb_dp": "USB_D_P",
                "uart_tx": "SERIAL_OUT",
                "uart_rx": "SERIAL_IN",
                "dtr_n": "HOST_DTR_N",
                "rts_n": "HOST_RTS_N",
            },
        )
    )
    parts = {part.recipe_role: part for part in expansion.parts}
    nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }
    mcu = parts["mcu"].ref
    levels = {"HOST_DTR_N": dtr, "HOST_RTS_N": rts}
    outputs = {nets[mcu, "8"]: 1, nets[mcu, "23"]: 1}
    for target in ("en", "boot"):
        transistor = parts[f"auto_reset_{target}"]
        resistor = parts[f"auto_reset_{target}_base_resistor"]
        assert transistor.mpn == "MMBT3904LT1G"
        assert transistor.symbol == "Transistor_BJT:MMBT3904"
        assert transistor.footprint == "Package_TO_SOT_SMD:SOT-23"
        assert resistor.value == "10k"
        # Datasheet/SOT-23: physical base 1, emitter 2, collector 3.
        assert nets[transistor.ref, "1"] == nets[resistor.ref, "2"]
        base = levels[nets[resistor.ref, "1"]]
        emitter = levels[nets[transistor.ref, "2"]]
        if base > emitter:
            outputs[nets[transistor.ref, "3"]] = emitter
    assert (outputs[nets[mcu, "8"]], outputs[nets[mcu, "23"]]) == (en, boot)
    assert parts["en_capacitor"].value == "1uF"
    assert nets[parts["en_capacitor"].ref, "1"] == nets[mcu, "8"]
    assert nets[parts["en_capacitor"].ref, "2"] == "GND"
    assert parts["en_pullup"].value == parts["boot_pullup"].value == "10k"
    assert nets[parts["download_boot_high_pullup_22"].ref, "1"] == "+3V3"
    assert nets[parts["download_boot_high_pullup_22"].ref, "2"] == nets[mcu, "22"]
    assert (mcu, "22") not in {(pin.ref, pin.pin) for pin in expansion.no_connect_pins}
    ownership = [(pin.ref, pin.pin) for pin in expansion.ownership.pins]
    assert len(ownership) == len(set(ownership))


def test_unused_fixed_programming_ports_are_private_and_add_no_support_parts():
    selections = [
        RecipeSelection(
            recipe="esp32-c3-mini-1-minimal@2",
            instance=instance,
            sheets={"mcu": "MCU"},
            port_bindings={
                "vdd": "+3V3",
                "gnd": "GND",
                "usb_dm": "USB_D_N",
                "usb_dp": "USB_D_P",
            },
        )
        for instance in ("first", "second")
    ]
    expansions = expand_selections(selections)
    private = []
    for expansion in expansions:
        mcu = next(part.ref for part in expansion.parts if part.recipe_role == "mcu")
        private.append(
            {
                connection.net_name
                for connection in expansion.connections
                if any(
                    endpoint.ref == mcu and endpoint.pin in {"8", "23"}
                    for endpoint in connection.endpoints
                )
            }
        )
        ncs = {(pin.ref, pin.pin) for pin in expansion.no_connect_pins}
        assert (mcu, "30") in ncs and (mcu, "31") in ncs  # unused UART0 is NC
        assert not any(part.recipe_role.startswith("auto_reset") for part in expansion.parts)
        assert not {"dtr_n", "rts_n"} & {
            connection.net_name for connection in expansion.connections
        }
    assert len(private[0]) == len(private[1]) == 2
    assert private[0].isdisjoint(private[1])


@pytest.mark.parametrize(
    "conflict",
    ["direction", "missing_rts", "disabled", "generic_gpio", "duplicate_peer", "aliased_controls"],
)
def test_c3_programming_contract_conflicts_block_without_dropping_gpio_or_handshakes(conflict):
    from copy import deepcopy

    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_typed_programming_architecture()
    core = payload["requirements"][0]
    bridge = payload["requirements"][1]
    if conflict == "direction":
        # The frozen R11 candidate typed bridge RX as a sheet output.
        bridge["ports"]["tx"], bridge["ports"]["rx"] = bridge["ports"]["rx"], bridge["ports"]["tx"]
    elif conflict == "missing_rts":
        del bridge["ports"]["rts_n"]
    elif conflict == "disabled":
        core["parameters"]["auto_reset"] = False
    elif conflict == "generic_gpio":
        core["ports"]["input_reset"] = "HOST_RTS_N"
    elif conflict == "duplicate_peer":
        duplicate = deepcopy(bridge)
        duplicate["id"] = "second_bridge"
        payload["requirements"].append(duplicate)
    else:
        bridge["ports"]["rts_n"] = bridge["ports"]["dtr_n"]
    result = resolve_architecture_recipes(payload)
    assert any(
        row.code == "conflicting_programming_contract" and row.requirement_id == "mcu_core"
        for row in result.blocking
    )
    assert not any(row.requirement_ids == ["mcu_core"] for row in result.selections)
    assert {f"GPIO{gpio}" for gpio in (*range(9), 10)} <= {
        net["name"] for net in payload["inter_sheet_nets"]
    }


@pytest.mark.parametrize(
    ("first", "second", "peer_first", "peer_second", "first_net", "second_net"),
    [
        ("uart_tx", "uart_rx", "rx", "tx", "MCU_TO_HOST", "HOST_TO_MCU"),
        ("dtr_n", "rts_n", "dtr_n", "rts_n", "HOST_DTR_N", "HOST_RTS_N"),
    ],
)
@pytest.mark.parametrize("peer_role", ["bus_interface", "programming"])
def test_programming_conflict_reports_current_peer_and_required_nets_without_swapping(
    first, second, peer_first, peer_second, first_net, second_net, peer_role
):
    from copy import deepcopy

    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_typed_programming_architecture()
    payload["requirements"][1]["role"] = peer_role
    core = payload["requirements"][0]
    core["ports"].update(
        uart_tx="MCU_TO_HOST",
        uart_rx="HOST_TO_MCU",
        dtr_n="HOST_DTR_N",
        rts_n="HOST_RTS_N",
    )
    core["ports"].update({first: second_net, second: first_net})
    before = deepcopy(payload)
    result = resolve_architecture_recipes(payload)
    diagnostic = next(
        row for row in result.blocking if row.code == "conflicting_programming_contract"
    )
    for port, current, peer, required in (
        (first, second_net, peer_first, first_net),
        (second, first_net, peer_second, second_net),
    ):
        assert f"mcu_core.ports.{port}={current!r}" in diagnostic.message
        assert f"bridge.ports.{peer}={required!r}" in diagnostic.message
        assert f"required mcu_core.ports.{port}={required!r}" in diagnostic.message
    assert "(output) -> " in diagnostic.message and "(input)" in diagnostic.message
    assert payload == before
    assert not any(row.requirement_ids == ["mcu_core"] for row in result.selections)

    # The repair is explicit and preserves every unrelated conductor.
    core["ports"].update({first: first_net, second: second_net})
    repaired = resolve_architecture_recipes(payload)
    assert not repaired.blocking
    selection = next(row for row in repaired.selections if row.requirement_ids == ["mcu_core"])
    assert {row.net for row in selection.pin_allocations} == {
        f"GPIO{gpio}" for gpio in (*range(9), 10)
    }


def test_programming_peer_cannot_claim_remote_only_nets_even_with_matching_labels():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_typed_programming_architecture()
    for net in payload["inter_sheet_nets"]:
        if net["name"].startswith(("HOST_", "MCU_TO_")):
            net["endpoints"] = [
                {**endpoint, "sheet": "APP"} if endpoint["sheet"] == "MCU" else endpoint
                for endpoint in net["endpoints"]
            ]
    result = resolve_architecture_recipes(payload)
    assert not result.blocking
    core = next(row for row in result.selections if row.requirement_ids == ["mcu_core"])
    assert core.port_bindings == {"vdd": "+3V3", "gnd": "GND"}
    expansion = expand_recipe(core)
    assert {"26", "27"} <= {pin.pin for pin in expansion.no_connect_pins if pin.ref.startswith("U")}
    assert not any(part.ref.startswith("Q") for part in expand_recipe(core).parts)


@pytest.mark.parametrize(
    ("enabled", "ports"),
    [
        (True, {"uart_tx": "TX", "uart_rx": "RX", "dtr_n": "DTR"}),
        (False, {"dtr_n": "DTR", "rts_n": "RTS"}),
        (True, {"uart_tx": "TX", "uart_rx": "RX", "dtr_n": "CONTROL", "rts_n": "CONTROL"}),
    ],
)
def test_direct_expansion_rejects_incomplete_disabled_or_aliased_auto_reset(enabled, ports):
    with pytest.raises(ValueError):
        expand_recipe(
            RecipeSelection(
                recipe="esp32-c3-mini-1-minimal@2",
                instance="invalid",
                sheets={"mcu": "MCU"},
                parameters={"auto_reset": enabled},
                port_bindings={"vdd": "+3V3", "gnd": "GND", **ports},
            )
        )






def test_rp2040_family_default_requires_and_binds_a_native_usb_connector():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = {
        "topologies": {"MCU": "RP2040 modules"},
        "rail_voltages": {"+3V3": 3.3, "VBUS": 5.0},
        "comms_protocols": ["USB"],
        "mcu_present": True,
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "RP2040 microcontroller"},
            {"name": "IO", "stem": "IO", "function": "castellated IO"},
            {"name": "USB", "stem": "USB", "function": "USB connector"},
        ],
        "power_nets": ["+3V3", "VBUS", "GND"],
        "inter_sheet_nets": [
            {
                "name": net,
                "endpoints": [
                    {"sheet": "MCU", "direction": "bidirectional"},
                    {"sheet": "USB", "direction": "bidirectional"},
                ],
            }
            for net in ("USB_D_N", "USB_D_P")
        ],
        "requirements": [
            {
                "id": "mcu_core",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": "rp2040",
                "exact_part": "RP2040",
                "ports": {
                    "vdd": "+3V3",
                    "gnd": "GND",
                    "usb_dm": "USB_D_N",
                    "usb_dp": "USB_D_P",
                },
            },
            {
                "id": "usb_connector",
                "sheet": "USB",
                "role": "connector",
                "family": "usb-c-usb2-device",
                "exact_part": "USB-C-USB2-DEVICE",
                "ports": {
                    "vbus": "VBUS",
                    "gnd": "GND",
                    "usb_dm": "USB_D_N",
                    "usb_dp": "USB_D_P",
                },
            },
        ],
        "assumptions": [],
    }
    result = resolve_architecture_recipes(payload)
    assert not result.blocking
    core = next(row for row in result.selections if row.requirement_ids == ["mcu_core"])
    connector = next(row for row in result.selections if row.requirement_ids == ["usb_connector"])
    assert core.recipe == "rp2040-minimal@3"
    assert core.port_bindings["usb_dm"] == "USB_D_N"
    assert core.port_bindings["usb_dp"] == "USB_D_P"
    # The RP2040 owns its 27-ohm MCU-side pair; the connector must not add another.
    assert connector.parameters["series_resistors"] is False
    expansion = expand_recipe(core)
    assert all(part.recipe_role != "swd" for part in expansion.parts)
    assert {"24", "25"} <= {pin.pin for pin in expansion.no_connect_pins}
    payload["requirements"][1]["parameters"] = {"series_resistors": True}
    conflict = resolve_architecture_recipes(payload)
    assert any(
        diagnostic.code == "conflicting_recipe_parameter"
        and diagnostic.requirement_id == "usb_connector"
        for diagnostic in conflict.blocking
    )

def test_rp2040_ch340c_mode_crosses_uart_and_uses_physical_swd_recovery():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = {
        "rail_voltages": {"+3V3": 3.3, "VBUS": 5.0},
        "power_nets": ["+3V3", "VBUS", "GND"],
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "RP2040 microcontroller"},
            {"name": "IO", "stem": "IO", "function": "GPIO breakout"},
            {"name": "BRIDGE", "stem": "BRIDGE", "function": "CH340C USB-UART bridge"},
        ],
        "inter_sheet_nets": [
            {
                "name": "HOST_TO_MCU",
                "endpoints": [
                    {"sheet": "BRIDGE", "direction": "output"},
                    {"sheet": "MCU", "direction": "input"},
                ],
            },
            {
                "name": "MCU_TO_HOST",
                "endpoints": [
                    {"sheet": "MCU", "direction": "output"},
                    {"sheet": "BRIDGE", "direction": "input"},
                ],
            },
        ],
        "requirements": [
            {
                "id": "mcu",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": "rp2040",
                "exact_part": "RP2040",
                "parameters": {"native_usb": False, "programming": "swd"},
                "ports": {"vdd": "+3V3", "gnd": "GND"},
            },
            {
                "id": "bridge",
                "sheet": "BRIDGE",
                "role": "bus_interface",
                "family": "usb-uart-bridge",
                "exact_part": "CH340C",
                "ports": {
                    "vbus": "VBUS",
                    "vdd": "+3V3",
                    "gnd": "GND",
                    "usb_dm": "USB_D_N",
                    "usb_dp": "USB_D_P",
                    "tx": "HOST_TO_MCU",
                    "rx": "MCU_TO_HOST",
                },
            },
        ],
    }

    result = resolve_architecture_recipes(payload)

    assert not result.blocking
    core = next(row for row in result.selections if row.requirement_ids == ["mcu"])
    bridge = next(row for row in result.selections if row.requirement_ids == ["bridge"])
    assert core.port_bindings["uart_tx"] == bridge.port_bindings["rx"] == "MCU_TO_HOST"
    assert core.port_bindings["uart_rx"] == bridge.port_bindings["tx"] == "HOST_TO_MCU"
    expansion = expand_recipe(core)
    nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }
    mcu = next(part for part in expansion.parts if part.recipe_role == "mcu")
    header = next(part for part in expansion.parts if part.recipe_role == "swd_header")
    assert nets[mcu.ref, "2"] == "MCU_TO_HOST"
    assert nets[mcu.ref, "3"] == "HOST_TO_MCU"
    assert nets[mcu.ref, "24"] == nets[header.ref, "4"]
    assert nets[mcu.ref, "25"] == nets[header.ref, "3"]
    assert nets[mcu.ref, "26"] == nets[header.ref, "5"]
    assert {(mcu.ref, "46"), (mcu.ref, "47")} <= {
        (pin.ref, pin.pin) for pin in expansion.no_connect_pins
    }
    assert "gpio0" not in nets.values()
    assert "gpio1" not in nets.values()
    assert "usb_dm" not in nets.values()
    assert "usb_dp" not in nets.values()


def test_rp2040_native_usb_keeps_uart_bridge_contacts_connected():
    core = RecipeSelection(
        recipe="rp2040-minimal@3",
        instance="rp",
        sheets={"mcu": "MCU", "io": "IO"},
        parameters={"native_usb": True, "programming": "native_usb"},
        port_bindings={
            "vdd": "+3V3", "gnd": "GND",
            "usb_dp": "NATIVE_DP", "usb_dm": "NATIVE_DM",
            "uart_tx": "BRIDGE_RX", "uart_rx": "BRIDGE_TX",
        },
    )
    expansion = expand_recipe(core)
    mcu = next(part for part in expansion.parts if part.recipe_role == "mcu")
    nets = {
        endpoint.pin: connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
        if endpoint.ref == mcu.ref
    }
    assert {pin: nets[pin] for pin in ("2", "3")} == {
        "2": "BRIDGE_RX", "3": "BRIDGE_TX",
    }
    from kicraft.design.synthesis.validation import _resistance_ohms

    endpoints = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }
    termination = {
        frozenset((endpoints[part.ref, "1"], endpoints[part.ref, "2"]))
        for part in expansion.parts
        if part.symbol == "Device:R" and _resistance_ohms(part.value) == 27.0
    }
    assert nets["46"] != "NATIVE_DM" and nets["47"] != "NATIVE_DP"
    assert termination == {
        frozenset((nets["46"], "NATIVE_DM")),
        frozenset((nets["47"], "NATIVE_DP")),
    }
    for missing in ("uart_tx", "uart_rx"):
        with pytest.raises(ValueError, match="requires uart_tx and uart_rx together"):
            expand_recipe(core.model_copy(update={
                "port_bindings": {
                    key: value for key, value in core.port_bindings.items() if key != missing
                },
            }))


def test_rp2040_rejects_half_native_pair_and_conflicting_recovery_mode():
    base = {
        "recipe": "rp2040-minimal@3",
        "instance": "rp",
        "sheets": {"mcu": "MCU", "io": "IO"},
        "port_bindings": {"vdd": "+3V3", "gnd": "GND"},
    }
    with pytest.raises(ValueError, match="requires usb_dm and usb_dp together"):
        expand_recipe(
            RecipeSelection(
                **{
                    **base,
                    "port_bindings": {**base["port_bindings"], "usb_dm": "USB_D_N"},
                }
            )
        )
    with pytest.raises(ValueError, match="native_usb=False requires programming='swd'"):
        expand_recipe(
            RecipeSelection(
                **{
                    **base,
                    "parameters": {"native_usb": False, "programming": "native_usb"},
                    "port_bindings": {
                        **base["port_bindings"],
                        "uart_tx": "MCU_TO_HOST",
                        "uart_rx": "HOST_TO_MCU",
                    },
                }
            )
        )


def _rp2040_support_architecture():
    return {
        "topologies": {"MCU": "RP2040 with external clock and QSPI flash"},
        "rail_voltages": {"+3V3": 3.3},
        "comms_protocols": [],
        "mcu_present": True,
        "sheets": [{"name": "MCU", "stem": "MCU", "function": "RP2040 controller"}],
        "power_nets": ["+3V3", "GND"],
        "inter_sheet_nets": [],
        "requirements": [
            {
                "id": identity,
                "sheet": "MCU",
                "role": role,
                "family": "rp2040",
                "exact_part": "RP2040",
                "ports": {
                    "vdd": "+3V3",
                    "gnd": "GND",
                    "usb_dm": "USB_DM",
                    "usb_dp": "USB_DP",
                },
                "obligations": [
                    {
                        "kind": "physical",
                        "original_obligation_id": identity,
                        "component_class": component_class,
                    }
                ],
            }
            for identity, role, component_class in (
                ("processor", "mcu_core", "microcontroller"),
                ("clock", "analog_block", "crystal"),
                ("flash", "bus_interface", "flash-memory"),
            )
        ],
    }


def test_mcu_support_requirements_realize_one_complete_physical_circuit():
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution
    from kicraft.design.synthesis.validation import check_requirement_physical_realization

    architecture = apply_architecture_recipe_resolution(_rp2040_support_architecture())
    payload, _ = _normalize_stage_response(
        "bom", {"groups": [], "arrays": []}, {"architecture": architecture.model_dump()}
    )
    bom = BOM.model_validate(payload)
    verdict = check_requirement_physical_realization(architecture, bom)
    assert verdict.ok, verdict.offenders
    assert sum(part.mpn == "RP2040" for part in bom.parts) == 1
    assert sum(part.mpn == "W25Q16JVSS" for part in bom.parts) == 1
    assert sum(part.mpn == "ABM8-272-T3" for part in bom.parts) == 1

    # SS is Winbond's 208-mil package, not the smaller 150-mil SN package.
    # A resolvable but physically wrong footprint must fail realization.
    flash = next(part for part in bom.parts if part.mpn == "W25Q16JVSS")
    flash.footprint = "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"
    assert not check_requirement_physical_realization(architecture, bom).ok


def test_the_rp2040_recipes_bootsel_button_realizes_a_pushbutton_obligation():
    """The recipe-owned BOOTSEL button realizes the pushbutton obligation."""
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution
    from kicraft.design.synthesis.validation import check_requirement_physical_realization

    payload = _rp2040_support_architecture()
    processor = payload["requirements"][0]
    processor["obligations"].append(
        {"kind": "physical", "original_obligation_id": "bootsel_button",
         "component_class": "pushbutton"}
    )
    architecture = apply_architecture_recipe_resolution(payload)
    bom_payload, _ = _normalize_stage_response(
        "bom", {"groups": [], "arrays": []}, {"architecture": architecture.model_dump()}
    )
    bom = BOM.model_validate(bom_payload)
    verdict = check_requirement_physical_realization(architecture, bom)
    assert verdict.ok, verdict.offenders
    bootsel = next(part for part in bom.parts if part.value == "BOOTSEL")
    assert bootsel.mpn == "KMR232NGULCLFS"
    from kicraft.design.part_identity import physical_inventory_record
    assert physical_inventory_record(
        mpn=bootsel.mpn, symbol=bootsel.symbol, footprint="Button_Switch_SMD:SW_SPST_TL3342"
    ) is None


@pytest.mark.parametrize("conflict", ["second_core", "different_usb_net", "different_sheet"])
def test_mcu_support_cannot_choose_an_ambiguous_or_different_core(conflict):
    from copy import deepcopy

    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _rp2040_support_architecture()
    if conflict == "second_core":
        second = deepcopy(payload["requirements"][0])
        second["id"] = "second_processor"
        payload["requirements"].append(second)
    elif conflict == "different_usb_net":
        payload["requirements"][1]["ports"]["usb_dp"] = "OTHER_USB_DP"
    else:
        payload["sheets"].append(
            {"name": "CLOCK", "stem": "CLOCK", "function": "External clock"}
        )
        payload["requirements"][1]["sheet"] = "CLOCK"
    result = resolve_architecture_recipes(payload)
    assert any(row.code == "mcu_support_owner_unproven" for row in result.blocking)


def test_two_explicit_mcu_cores_are_not_coalesced():
    from copy import deepcopy

    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _rp2040_support_architecture()
    second = deepcopy(payload["requirements"][0])
    second["id"] = "second_processor"
    payload["requirements"] = [payload["requirements"][0], second]
    result = resolve_architecture_recipes(payload)
    assert not result.blocking
    expansions = expand_selections(result.selections)
    assert sum(part.mpn == "RP2040" for row in expansions for part in row.parts) == 2




def test_the_pd_selectable_recipes_own_selector_realizes_a_switch_demand():
    """`ch224k-pd-selectable@1` emits the reviewed SS13D07VG4, so its `switch` demand is met.

    Live walkthrough (usb-pd-trigger, 2026-09-29): §9.42 refused the requirement with
    "requires 1 reviewed 'switch' physical part(s), found 0" although the bound recipe's own
    selector is that switch and the class resolves to the selector's reviewed features.
    """
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution
    from kicraft.design.synthesis.validation import check_requirement_physical_realization

    payload = {
        "topologies": {"PD": "CH224K USB-PD sink with selectable output"},
        "rail_voltages": {"VBUS": 5.0},
        "power_nets": ["VBUS", "GND"],
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [{"name": "PD", "stem": "PD", "function": "CH224K PD controller"}],
        "inter_sheet_nets": [],
        "requirements": [
            {
                "id": "pd_trigger",
                "sheet": "PD",
                "role": "power_input",
                "family": "usb-pd-selectable-trigger",
                "exact_part": "CH224K",
                "ports": {"vbus": "VBUS", "gnd": "GND", "cc1": "CC1", "cc2": "CC2"},
                "obligations": [
                    {"kind": "physical", "original_obligation_id": "selector",
                     "component_class": "switch"}
                ],
            }
        ],
        "recipe_selections": [
            {
                "recipe": "ch224k-pd-selectable@1",
                "instance": "pd",
                "sheets": {"power": "PD"},
                "requirement_ids": ["pd_trigger"],
                "port_bindings": {"vbus": "VBUS", "gnd": "GND", "cc1": "CC1", "cc2": "CC2"},
            }
        ],
        "assumptions": [],
    }
    architecture = apply_architecture_recipe_resolution(payload)
    bom_payload, _ = _normalize_stage_response(
        "bom", {"groups": [], "arrays": []}, {"architecture": architecture.model_dump()}
    )
    bom = BOM.model_validate(bom_payload)
    assert any(part.mpn == "SS13D07VG4" for part in bom.parts)
    verdict = check_requirement_physical_realization(architecture, bom)
    assert not any("'switch'" in offender for offender in verdict.offenders), verdict.offenders


def test_stm32_recipe_crystal_has_realizable_reviewed_identity():
    from kicraft.design.models import Architecture
    from kicraft.design.synthesis.validation import check_requirement_physical_realization

    payload = _rp2040_support_architecture()
    core = payload["requirements"][0]
    core.update(family="stm32f103c8", exact_part="STM32F103C8T6")
    core["obligations"].append(
        {"kind": "physical", "original_obligation_id": "hse", "component_class": "crystal"}
    )
    payload["requirements"] = [core]
    payload["topologies"] = {"MCU": "STM32F103C8T6 with 8 MHz HSE"}
    payload["sheets"][0]["function"] = "STM32F103C8T6 controller"
    payload["recipe_selections"] = [
        {
            "recipe": "stm32f103c8t6-minimal@1",
            "instance": "processor",
            "sheets": {"mcu": "MCU"},
            "requirement_ids": ["processor"],
            "port_bindings": {"vdd": "+3V3", "gnd": "GND"},
        }
    ]
    architecture = Architecture.model_validate(payload)
    bom_payload, _ = _normalize_stage_response(
        "bom", {"groups": [], "arrays": []}, {"architecture": architecture.model_dump()}
    )
    bom = BOM.model_validate(bom_payload)
    verdict = check_requirement_physical_realization(architecture, bom)
    assert verdict.ok, verdict.offenders


def test_recipe_castellation_edges_replace_the_synthetic_header_unit():
    """RP2040 recipe pads, not an assembly header, realize a declared edge."""
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution
    from kicraft.server.stage_work_units import plan_stage_work_units

    architecture = {
        "topologies": {"MCU": "RP2040 controller"},
        "rail_voltages": {"+3V3": 3.3},
        "comms_protocols": [],
        "mcu_present": True,
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "RP2040 controller"},
            {"name": "CASTELLATED", "stem": "CASTELLATED", "function": "GPIO edge pads"},
        ],
        "power_nets": ["+3V3", "GND"],
        "inter_sheet_nets": [],
        "requirements": [
            {
                "id": "rp2040",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": "rp2040",
                "exact_part": "RP2040",
                "ports": {
                    "vdd": "+3V3",
                    "gnd": "GND",
                    "usb_dm": "USB_DM",
                    "usb_dp": "USB_DP",
                },
                "functional_blocks": ["RP2040_PROCESSOR", "CASTELLATED_GPIO"],
            },
            {
                "id": "rp2040_castellated_gpio",
                "sheet": "CASTELLATED",
                "role": "connector",
                "family": "pin-header",
                "compiler_origin": "edge_connector",
                "ports": {"pin1": "GPIO0", "pin2": "GPIO1", "pin3": "GND"},
                "functional_blocks": ["RP2040_PROCESSOR", "CASTELLATED_GPIO"],
            },
        ],
    }

    resolved = apply_architecture_recipe_resolution(architecture)

    (selection,) = resolved.recipe_selections
    assert selection.requirement_ids == ["rp2040", "rp2040_castellated_gpio"]
    assert selection.sheets["io"] == "CASTELLATED"
    assert resolved.unresolved_requirement_ids == []
    assert plan_stage_work_units("bom", {"architecture": resolved.model_dump()}, {}) == ()

    bom, _ = _normalize_stage_response(
        "bom",
        {"groups": [], "arrays": []},
        {"architecture": resolved.model_dump()},
    )
    edge_parts = [part for part in bom["parts"] if part["sheet"] == "CASTELLATED"]
    assert len(edge_parts) == 34
    assert all(part["value"] == "Castellated pad" and not part["assembly"] for part in edge_parts)
    assert not any(part["ref"].startswith("J") for part in bom["parts"])


def test_real_pin_header_is_not_consumed_by_matching_recipe_edges():
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution
    from kicraft.server.stage_work_units import plan_stage_work_units

    architecture = {
        "topologies": {"MCU": "RP2040 controller"},
        "rail_voltages": {"+3V3": 3.3},
        "comms_protocols": [],
        "mcu_present": True,
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "RP2040 controller"},
            {"name": "HEADER", "stem": "HEADER", "function": "Physical GPIO header"},
        ],
        "power_nets": ["+3V3", "GND"],
        "inter_sheet_nets": [],
        "requirements": [
            {
                "id": "rp2040",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": "rp2040",
                "exact_part": "RP2040",
                "parameters": {"native_usb": False, "programming": "swd"},
                "ports": {"vdd": "+3V3", "gnd": "GND"},
                "functional_blocks": ["RP2040_PROCESSOR", "CASTELLATED_GPIO"],
            },
            {
                "id": "physical_gpio_header",
                "sheet": "HEADER",
                "role": "connector",
                "family": "pin-header",
                "parameters": {"rows": 1, "gender": "male"},
                "ports": {"pin1": "GPIO0", "pin2": "GPIO1", "pin3": "GND"},
                "functional_blocks": ["RP2040_PROCESSOR", "CASTELLATED_GPIO"],
            },
        ],
    }

    resolved = apply_architecture_recipe_resolution(architecture)

    (selection,) = resolved.recipe_selections
    assert selection.requirement_ids == ["rp2040"]
    (header_unit,) = plan_stage_work_units("bom", {"architecture": resolved.model_dump()}, {})
    assert header_unit.requirement_ids == ("physical_gpio_header",)


# ---------------------------------------------------------------------------
# order-code variants and semantic recipe ports (dfc1582 follow-ups)
# ---------------------------------------------------------------------------


def test_n16r8_exact_recipe_selection_and_conflicting_sibling_rejection():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    intent = {"named_parts": ["ESP32-S3-WROOM-1-N16R8"]}
    payload = _typed_esp32_architecture(
        family="esp32-s3-wroom-1-module", exact_part="ESP32-S3-WROOM-1-N16R8"
    )
    result = resolve_architecture_recipes(payload, intent)
    assert not result.blocking
    assert result.exact_parts["mcu_core"] == "ESP32-S3-WROOM-1-N16R8"
    assert any(row.recipe == "esp32-s3-wroom-1-n16r8-minimal@1" for row in result.selections)

    sibling = _typed_esp32_architecture(
        family="esp32-s3-wroom-1-module", exact_part="ESP32-S3-WROOM-1-N8R8"
    )
    rejected = resolve_architecture_recipes(sibling, intent)
    assert any(row.code == "conflicting_exact_variant" for row in rejected.blocking)
    assert not any(row.requirement_ids == ["mcu_core"] for row in rejected.selections)



def test_exact_esp32_c3_variant_is_not_accepted_as_a_generic_module_label():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    result = resolve_architecture_recipes(
        _typed_esp32_architecture(
            family="esp32-c3-mini-1-module",
            exact_part="ESP32-C3-MINI-1-N8",
        ),
        {"named_parts": ["ESP32-C3-MINI-1-N8"]},
    )

    assert any(
        row.code == "unsupported_protected_variant"
        and "ESP32-C3-MINI-1-N8" in row.evidence
        for row in result.blocking
    )


def test_unreviewed_protected_variant_still_blocks_and_names_the_choice():
    """A protected identity with no reviewed member must still block — and the
    correction must name a binding the model can actually make."""
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(family="stm32f103c8", exact_part="STM32H743VIT6")
    result = resolve_architecture_recipes(payload, {"named_parts": ["STM32H743VIT6"]})

    diagnostic = next(row for row in result.blocking if row.code == "unsupported_protected_variant")
    assert "STM32H743VIT6" in diagnostic.evidence
    assert any(row.startswith("canonical choice:") for row in diagnostic.evidence)
    assert not any(row.requirement_ids == ["mcu_core"] for row in result.selections)


def test_named_order_code_prefers_exact_recipe_per_named_mcu():
    """Exact identities take precedence without dropping other named MCUs."""
    from kicraft.server.stage_runtime import _architecture_recipe_summaries

    def mcus(named):
        rows = _architecture_recipe_summaries({"named_parts": [named]})
        return {row["recipe"] for row in rows if "mcu" in row["required_sheet_roles"]}

    assert mcus("ESP32-S3-WROOM-1-N16R8") == {"esp32-s3-wroom-1-n16r8-minimal@1"}
    assert mcus("ESP32-S3-MINI-1-N8") == {"esp32-s3-mini-1-minimal@1"}
    # An unrecognised MCU name must not starve the stage of every alternative.
    assert len(mcus("ACME-WIDGET-9000")) > 1
    rows = _architecture_recipe_summaries({
        "named_parts": ["ESP32-S3-WROOM-1-N16R8", "CH32V003J4M6"],
    })
    assert {row["recipe"] for row in rows if "mcu" in row["required_sheet_roles"]} == {
        "esp32-s3-wroom-1-n16r8-minimal@1", "ch32v003j4m6-minimal@1",
    }


def test_ads1115_address_strap_is_selectable_per_instance():
    """Two ADS1115s on one bus need distinct straps, not a hard-wired ground.

    A grounded ADDR pin on every instance silently put both converters at 0x48,
    so the eight-channel brief was unrealizable even though it committed.
    """
    from kicraft.design.recipes.models import RecipeSelection, ResolvedRecipeSelection
    from kicraft.design.recipes.registry import expand_static_definition

    definition = get_recipe("ads1115-i2c-adc@1")
    assert definition.parameter_defaults["address_strap"] == "gnd"
    assert definition.allowed_parameters["address_strap"] == ("gnd", "vdd", "sda", "scl")
    sheets = {role: "A" for role in definition.required_sheet_roles}

    def adc_address_net(strap):
        selection = RecipeSelection(
            recipe=definition.recipe,
            instance=f"adc_{strap}",
            sheets=sheets,
            parameters={"address_strap": strap},
            port_bindings={"vdd": "+3V3", "gnd": "GND", "sda": "SDA", "scl": "SCL"},
            requirement_ids=["adc"],
        )
        expansion = expand_static_definition(
            definition,
            ResolvedRecipeSelection(selection=selection, parameters={"address_strap": strap}),
        )
        return {
            own.net
            for own in expansion.ownership.pins
            if own.pin == "1" and own.ref.startswith("U")
        }

    # ADDR (pin 1) follows the declared strap, which is exactly what decides the
    # part's I2C address; the default keeps the historic grounded behaviour.
    assert adc_address_net("gnd") == {"GND"}
    assert adc_address_net("vdd") == {"+3V3"}
    assert adc_address_net("sda") == {"SDA"}
    assert adc_address_net("scl") == {"SCL"}


def test_ads1115_record_and_recipe_agree_on_the_address_straps():
    """§9.43 counts the reviewed record's straps; the recipe must not offer more.

    A recipe strap outside the record's list would let a fifth converter commit
    while the capacity gate computed against the reviewed four.
    """
    from kicraft.design.part_identity import reviewed_part

    record = reviewed_part("ADS1115IDGSR")
    definition = get_recipe("ads1115-i2c-adc@1")
    assert tuple(sorted(record.support_network["address_strap"]["straps"])) == tuple(
        sorted(definition.allowed_parameters["address_strap"])
    )


def test_hub75_recipe_exposes_named_channels_including_oe_and_d():
    """The HUB75 interface must declare its real signals.

    Opaque `input0..input11` ports made the interface unbindable by name, and pin
    12 (the D address line) was grounded while OE had no port of its own.
    """
    definition = get_recipe("hub75-sn74hct245-interface@1")
    names = [port.name for port in definition.ports]

    assert names[:2] == ["vdd_5v", "gnd"]
    assert names[2:] == [
        "r0",
        "g0",
        "b0",
        "r1",
        "g1",
        "b1",
        "addr_a",
        "addr_b",
        "addr_c",
        "addr_d",
        "clk",
        "lat",
        "oe",
    ]
    assert not any(name.startswith("input") for name in names)

    connector = {pin.pin: pin.net for pin in definition.pins if pin.role == "connector"}
    assert connector["15"] == "shifted12"  # OE is buffered, not direct
    assert connector["12"] == "shifted9"  # D is a channel, not ground
    assert {pin for pin, net in connector.items() if net == "gnd"} == {"4", "8", "16"}
    # Every logic channel has a distinct level-shifted net; each spare A input
    # is grounded and each spare B output is a no-connect.
    assert len({net for net in connector.values() if net.startswith("shifted")}) == 13
    level_shifter = {
        (pin.index, pin.pin): pin.net
        for pin in definition.pins
        if pin.role == "level_shifter"
    }
    assert {level_shifter[(1, pin)] for pin in ("7", "8", "9")} == {"gnd"}
    assert {row.pin for row in definition.no_connects} == {"11", "12", "13"}


@pytest.mark.parametrize(
    ("recipe", "index", "unused_a_inputs", "unused_b_outputs"),
    (
        ("ws2812-external-strip-interface@1", 0, ("3", "4", "5", "6", "7", "8", "9"), ("11", "12", "13", "14", "15", "16", "17")),
        ("max7219-four-digit-seven-segment@1", 0, ("5", "6", "7", "8", "9"), ("11", "12", "13", "14", "15")),
        ("led-matrix-hub75-interface@1", 1, ("7", "8", "9"), ("11", "12", "13")),
        ("hub75-sn74hct245-interface@1", 1, ("7", "8", "9"), ("11", "12", "13")),
    ),
)
def test_hct245_recipes_ground_unused_a_inputs_and_no_connect_b_outputs(
    recipe, index, unused_a_inputs, unused_b_outputs,
):
    definition = get_recipe(recipe)
    level_shifter_pins = {
        (pin.index, pin.pin): pin.net
        for pin in definition.pins
        if pin.role == "level_shifter"
    }
    no_connects = {
        (row.index, row.pin)
        for row in definition.no_connects
        if row.role == "level_shifter"
    }

    assert {level_shifter_pins[index, pin] for pin in unused_a_inputs} == {"gnd"}
    assert not {(index, pin) for pin in unused_a_inputs} & no_connects
    assert {(index, pin) for pin in unused_b_outputs} <= no_connects


def test_hub75_channels_bind_by_signal_name():
    """A declared HUB75 interface binds its channels to the recipe's semantic
    ports (HUB75_OE → oe), which is what the architecture stage could not do."""
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    signals = {
        "R0": "r0",
        "G0": "g0",
        "B0": "b0",
        "R1": "r1",
        "G1": "g1",
        "B1": "b1",
        "A": "addr_a",
        "B": "addr_b",
        "C": "addr_c",
        "D": "addr_d",
        "CLK": "clk",
        "LAT": "lat",
        "OE": "oe",
    }
    payload = {
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "microcontroller"},
            {"name": "HUB75", "stem": "HUB75", "function": "HUB75 display connector"},
            {"name": "POWER", "stem": "POWER", "function": "5V input"},
        ],
        "requirements": [
            {
                "id": "display",
                "sheet": "HUB75",
                "role": "connector",
                "family": "hub75-level-shift-interface",
                "exact_part": "HUB75-SN74HCT245",
                "ports": {"gnd": "GND", "vdd_5v": "+5V"},
            },
        ],
        "power_nets": ["GND", "+5V"],
        "inter_sheet_nets": [
            {
                "name": f"HUB75_{signal}",
                "endpoints": [
                    {"sheet": "HUB75", "direction": "input"},
                    {"sheet": "MCU", "direction": "output"},
                ],
            }
            for signal in signals
        ],
        "mcu_present": False,
        "topologies": {},
    }
    result = resolve_architecture_recipes(payload)

    assert not any(row.code == "unsupported_recipe_endpoint" for row in result.blocking)
    display = next(row for row in result.selections if row.requirement_ids == ["display"])
    assert display.recipe == "hub75-sn74hct245-interface@1"
    assert {port: display.port_bindings[port] for port in signals.values()} == {
        port: f"HUB75_{signal}" for signal, port in signals.items()
    }


def test_unused_groundable_hub75_addr_d_is_tied_low_not_demanded():
    """A board whose panel does not need HUB75's D line must not be blocked.

    `addr_d` is the 13th channel: a 1/32-scan panel uses it, a 1/8 or 1/16 panel
    does not. The derivation ties the spare '245 input low (never floating, the
    connector position still driven) instead of demanding an undeclared HUB75_D net.
    """
    from kicraft.design.architecture_intent import derive_architecture
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    channels = {
        "R0": "r0",
        "G0": "g0",
        "B0": "b0",
        "R1": "r1",
        "G1": "g1",
        "B1": "b1",
        "A": "addr_a",
        "B": "addr_b",
        "C": "addr_c",
        "CLK": "clk",
        "LAT": "lat",
        "OE": "oe",
    }
    architecture = derive_architecture(
        {
            "mcu_present": True,
            "sheets": [
                {"name": "MCU", "stem": "MCU", "role": "mcu", "function": "microcontroller"},
                {
                    "name": "HUB75",
                    "stem": "HUB75",
                    "role": "display",
                    "function": "HUB75 display connector",
                },
            ],
            "requirements": [
                {
                    "id": "mcu_core",
                    "sheet": "MCU",
                    "role": "mcu_core",
                    "family": "stm32f103c8",
                    "exact_part": "STM32F103C8T6",
                    "supply": "+3V3",
                },
                {
                    "id": "display",
                    "sheet": "HUB75",
                    "role": "connector",
                    "family": "hub75-level-shift-interface",
                    "exact_part": "HUB75-SN74HCT245",
                    "supply": "+5V",
                },
            ],
            "power": {"rails": {"+3V3": {"voltage": 3.3}, "+5V": {"voltage": 5.0}}},
            "signals": [
                {
                    "name": f"HUB75_{signal}",
                    "from": f"mcu_core.output_hub75_{port}",
                    "to": f"display.{port}",
                }
                for signal, port in channels.items()
            ],
        }
    )
    display_requirement = next(row for row in architecture.requirements if row.id == "display")
    assert display_requirement.ports["addr_d"] == "GND"
    assert any("addr_d tied to GND" in row for row in architecture.assumptions)

    result = resolve_architecture_recipes(architecture)
    assert not result.blocking
    display = next(row for row in result.selections if row.requirement_ids == ["display"])
    assert display.port_bindings["addr_d"] == "GND"
    # The spare '245 A-side input sits on GND and the connector position is still
    # driven by the buffered channel: no floating input, no one-pin net.
    expansion = expand_recipe(display)
    shifted9 = [
        endpoint
        for connection in expansion.connections
        if connection.net_name.endswith("shifted9")
        for endpoint in connection.endpoints
    ]
    assert {endpoint.pin for endpoint in shifted9} == {"17", "12"}
    grounded = [
        endpoint
        for connection in expansion.connections
        if connection.net_name == "GND"
        for endpoint in connection.endpoints
    ]
    assert any(endpoint.ref == "U2" and endpoint.pin == "3" for endpoint in grounded)


def _native_usb_board(*, usb_connector: bool, bind_usb: bool, five_volt_rail: bool = True):
    """An RP2040 board whose brief never mentions USB (live run 904's shape)."""
    def sheet(name, stem, function, role):
        return {"name": name, "stem": stem, "function": function, "role": role}

    requirements = [
        {
            "id": "rp2040",
            "sheet": "RP2040",
            "role": "mcu_core",
            "family": "rp2040",
            "parameters": {"package": "QFN-56"},
            "supply": "3V3",
        },
        {
            "id": "buck",
            "sheet": "POWER",
            "role": "regulator",
            "family": "tps54331-adjustable",
            "parameters": {"output_voltage": 3.3},
            "supply": "VBUS" if five_volt_rail else "+12V",
        },
    ]
    rails = {"3V3": {"voltage": 3.3, "from": "buck.output"}}
    if five_volt_rail:
        requirements.append(
            {
                "id": "usb_in",
                "sheet": "USB POWER",
                "role": "power_input",
                "family": "usb-c-power-sink",
                "exact_part": "USB-C-5V-SINK",
            }
        )
        rails["VBUS"] = {"voltage": 5.0, "from": "usb_in.vbus"}
    else:
        requirements.append(
            {
                "id": "jack",
                "sheet": "POWER",
                "role": "power_input",
                "family": "barrel-jack",
                "exact_part": "DC005",
                "declared_ports": [
                    {"key": "positive", "direction": "power", "function": "12 V DC input"},
                    {"key": "gnd", "direction": "power", "function": "supply return"},
                ],
            }
        )
        rails["+12V"] = {"voltage": 12.0, "from": "jack.positive"}
    sheets = [
        sheet("RP2040", "RP2040", "RP2040 core", "mcu"),
        sheet("POWER", "POWER", "Power input and rails", "power"),
    ]
    signals = []
    if five_volt_rail:
        sheets.append(sheet("USB POWER", "USB_POWER", "USB-C 5 V power input", "power"))
    if usb_connector:
        sheets.append(sheet("USB", "USB", "USB-C USB 2.0 device receptacle", "interface"))
        requirements.append(
            {
                "id": "usb_dev",
                "sheet": "USB",
                "role": "connector",
                "family": "usb-c-usb2-device",
                "exact_part": "USB-C-USB2-DEVICE",
                "supply": "VBUS",
            }
        )
    if bind_usb:
        signals = [
            {"name": "USB_DM", "from": "rp2040.usb_dm", "to": "usb_dev.usb_dm"},
            {"name": "USB_DP", "from": "rp2040.usb_dp", "to": "usb_dev.usb_dp"},
        ]
    return {
        "mcu_present": True,
        "sheets": sheets,
        "requirements": requirements,
        "power": {"rails": rails},
        "signals": signals,
    }

def test_architecture_decode_completes_usb_host_rail_before_derivation():
    payload = _native_usb_board(
        usb_connector=False, bind_usb=False, five_volt_rail=False,
    )
    payload["signals"] = [
        {"name": "USB_DM", "from": "rp2040.usb_dm", "to": "edge:USB_DATA"},
        {"name": "USB_DP", "from": "rp2040.usb_dp", "to": "edge:USB_DATA"},
    ]
    canonical, _ = _normalize_stage_response("architecture", payload, {})
    requirements = {row["id"]: row for row in canonical["requirements"]}
    socket = next(row for row in requirements.values() if row["family"] == "usb-c-usb2-device")
    assert socket["ports"]["vbus"] == "VBUS"
    assert canonical["rail_voltages"]["VBUS"] == 5.0
    assert requirements["buck"]["ports"]["input"] == "+12V"
    assert requirements["rp2040"]["ports"]["vdd"] == "3V3"
    assert socket["ports"]["usb_dm"] == requirements["rp2040"]["ports"]["usb_dm"]
    assert socket["ports"]["usb_dp"] == requirements["rp2040"]["ports"]["usb_dp"]
    assert "VBUS" not in payload["power"]["rails"]

@pytest.mark.parametrize("usb_rail", ["VBUS", "+5V"])
def test_usb_edge_rail_disambiguates_two_independent_five_volt_supplies(usb_rail):
    payload = _native_usb_board(
        usb_connector=False, bind_usb=False, five_volt_rail=False,
    )
    payload["power"]["rails"].update({
        "VBUS": {"voltage": 5.0, "from": None},
        "+5V": {"voltage": 5.0, "from": None},
    })
    payload["signals"] = [
        {"name": "USB_DM", "from": "rp2040.usb_dm", "to": "edge:USB", "rails": [usb_rail]},
        {"name": "USB_DP", "from": "rp2040.usb_dp", "to": "edge:USB", "rails": [usb_rail]},
    ]
    canonical, _ = _normalize_stage_response("architecture", payload, {})
    socket = next(row for row in canonical["requirements"] if row["family"] == "usb-c-usb2-device")
    assert socket["ports"]["vbus"] == usb_rail
    assert {"+5V", "VBUS", "+12V"} <= set(canonical["power_nets"])
    for signal in payload["signals"]:
        signal["rails"] = ["+12V"]
    with pytest.raises(StageSchemaError, match="usb_connector_supply_unknown"):
        _normalize_stage_response("architecture", payload, {})




def test_native_usb_mcu_gains_exactly_one_derived_data_connector():
    """Live run 904 (KC-SGYXF5) died on a port the brief never asked for.

    `unbound_required_port: requirement 'rp2040' port 'usb_dm' is required by
    rp2040-minimal@2 and nothing wires it` refused an RP2040 current-sense brief that
    never mentioned USB. The reviewed satisfier already exists (`usb-c-usb2-device@1`),
    so the compiler completes it the way it completes a published supply contact and says
    why the part is on the board.
    """
    from kicraft.design.architecture_intent import derive_architecture

    architecture = derive_architecture(_native_usb_board(usb_connector=False, bind_usb=False))

    connectors = [row for row in architecture.requirements if row.family == "usb-c-usb2-device"]
    assert len(connectors) == 1
    assert connectors[0].ports["usb_dm"] == "USB_DM"
    assert connectors[0].ports["usb_dp"] == "USB_DP"
    assert connectors[0].ports["vbus"] == "VBUS"
    assert connectors[0].exact_part == "USB-C-USB2-DEVICE"
    mcu = next(row for row in architecture.requirements if row.id == "rp2040")
    assert (mcu.ports["usb_dm"], mcu.ports["usb_dp"]) == ("USB_DM", "USB_DP")
    assert any(
        "native-USB MCU needs a data connector" in note for note in architecture.assumptions
    )


def test_native_usb_mcu_that_wires_its_own_connector_is_left_alone():
    """A design that declares its USB data connector keeps exactly that one."""
    from kicraft.design.architecture_intent import derive_architecture

    architecture = derive_architecture(_native_usb_board(usb_connector=True, bind_usb=True))

    connectors = [row for row in architecture.requirements if row.family == "usb-c-usb2-device"]
    assert [row.id for row in connectors] == ["usb_dev"]
    assert connectors[0].ports["usb_dm"] == "USB_DM"
    assert not any(
        "native-USB MCU needs a data connector" in note for note in architecture.assumptions
    )


def test_native_usb_completion_refuses_a_rail_that_is_not_a_usb_supply():
    """The completion never puts a barrel jack's rail on a USB socket's VBUS.

    A 12 V-powered RP2040 board (run 904's brief) must state the rail the socket exposes;
    silently binding VBUS to `+12V` would ship a socket that cannot be plugged into. The
    refusal names the rails the design did declare, so the next draft can add the 5 V rail.
    """
    from kicraft.design.architecture_intent import ArchitectureIntentError, derive_architecture

    with pytest.raises(ArchitectureIntentError) as rejected:
        derive_architecture(_native_usb_board(usb_connector=False, bind_usb=False, five_volt_rail=False))

    codes = {row.code for row in rejected.value.diagnostics}
    assert "usb_connector_supply_unknown" in codes


@pytest.mark.parametrize("quantity", [1, 3])
@pytest.mark.parametrize("export_output", [False, True])
def test_ws2812_pixels_form_a_complete_cascade_with_optional_final_output(quantity, export_output):
    bindings = {"vdd": "+5V", "gnd": "GND", "data_in": "PIXEL_DATA"}
    if export_output:
        bindings["data_out"] = "NEXT_BANK"
    selection = RecipeSelection(
        recipe="ws2812-output@1",
        instance="pixels",
        sheets={"interface": "LEDS"},
        parameters={"quantity": quantity},
        port_bindings=bindings,
    )
    expanded = expand_recipe(selection)
    leds = [part.ref for part in expanded.parts if part.recipe_role == "led"]
    capacitors = [part.ref for part in expanded.parts if part.recipe_role == "decoupling"]
    assert len(leds) == len(capacitors) == quantity
    nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expanded.connections
        for endpoint in connection.endpoints
    }
    endpoints = {
        connection.net_name: {(endpoint.ref, endpoint.pin) for endpoint in connection.endpoints}
        for connection in expanded.connections
    }
    series = next(part.ref for part in expanded.parts if part.recipe_role == "series_resistor")
    assert endpoints[nets[series, "2"]] == {(series, "2"), (leds[0], "4")}
    for ref in (*leds, *capacitors):
        assert nets[ref, "1"] == "+5V"
        assert nets[ref, "3" if ref in leds else "2"] == "GND"
    for previous, following in zip(leds, leds[1:]):
        assert endpoints[nets[previous, "2"]] == {(previous, "2"), (following, "4")}
    no_connects = {(endpoint.ref, endpoint.pin) for endpoint in expanded.no_connect_pins}
    if export_output:
        assert nets[leds[-1], "2"] == "NEXT_BANK"
        assert not no_connects
    else:
        assert (leds[-1], "2") not in nets
        assert no_connects == {(leds[-1], "2")}
    other = expand_recipe(selection.model_copy(update={"instance": "other"}))
    assert not (
        {connection.net_name for connection in expanded.connections} - set(bindings.values())
    ) & ({connection.net_name for connection in other.connections} - set(bindings.values()))


@pytest.mark.parametrize("quantity", [0, 501, True, 1.5])
def test_ws2812_rejects_unrealizable_pixel_quantities(quantity):
    with pytest.raises(ValueError):
        expand_recipe(
            RecipeSelection(
                recipe="ws2812-output@1",
                instance="pixels",
                sheets={"interface": "LEDS"},
                parameters={"quantity": quantity},
                port_bindings={"vdd": "+5V", "gnd": "GND", "data_in": "DATA"},
            )
        )





def test_reverse_protection_uses_drain_input_and_source_referenced_gate_clamp() -> None:
    """Reverse input must block the body diode; forward input must not overdrive Vgs."""
    expansion = expand_recipe(
        RecipeSelection(
            recipe="reverse-polarity-pmos@1",
            instance="protection",
            sheets={"power": "POWER INPUT"},
            port_bindings={"input": "HOST_5V", "output": "PROTECTED_5V", "gnd": "GND"},
        )
    )
    nets = {connection.net_name: {f"{e.ref}.{e.pin}" for e in connection.endpoints}
            for connection in expansion.connections}
    # Every drain contact faces the raw input and every source contact the protected rail.
    assert nets["HOST_5V"] == {"Q1.5", "Q1.6", "Q1.7", "Q1.8", "Q1.9"}
    assert nets["PROTECTED_5V"] == {"Q1.1", "Q1.2", "Q1.3", "D1.1"}
    assert nets["GND"] == {"R1.2"}
    assert {"Q1.4", "R1.1", "D1.2"} in nets.values()


def _mcu_recipe_nets(expansion):
    return {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }


def test_stm32g030_complete_artifact_has_swd_reset_boot_and_bypass():
    definition = get_recipe("stm32g030f6p6-minimal@2")
    expansion = expand_recipe(
        RecipeSelection(
            recipe=definition.recipe,
            instance="g0",
            sheets={"mcu": "MCU"},
            port_bindings={"vdd": "+3V3", "gnd": "GND"},
        )
    )
    refs = {part.recipe_role: part.ref for part in expansion.parts}
    nets = _mcu_recipe_nets(expansion)
    mcu = refs["mcu"]
    assert definition.exact_part == "STM32G030F6P6"
    assert (next(part for part in expansion.parts if part.ref == mcu).symbol,
            next(part for part in expansion.parts if part.ref == mcu).footprint) == (
        "MCU_ST_STM32G0:STM32G030F6Px",
        "Package_SO:TSSOP-20_4.4x6.5mm_P0.65mm",
    )
    assert {nets[mcu, pin] for pin in ("4", "5", "6", "18", "19")} == {
        "+3V3",
        "GND",
        nets[refs["reset_button"], "1"],
        nets[refs["swd_header"], "3"],
        nets[refs["swd_header"], "4"],
    }
    assert nets[mcu, "6"] == nets[refs["reset_pullup"], "2"] == nets[refs["reset_capacitor"], "1"]
    assert nets[refs["reset_pullup"], "1"] == "+3V3"
    assert nets[refs["reset_capacitor"], "2"] == "GND"
    assert nets[mcu, "19"] == nets[refs["boot0_pulldown"], "1"]
    assert nets[refs["boot0_pulldown"], "2"] == "GND"
    for role in ("local_decoupling", "bulk_decoupling"):
        assert (nets[refs[role], "1"], nets[refs[role], "2"]) == ("+3V3", "GND")
    assert check_net_coverage(
        BOM(parts=expansion.parts, connections=expansion.connections, no_connect_pins=expansion.no_connect_pins)
    ).ok


def test_stm32g030k6_native_graph_and_four_adc_resources():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins
    from kicraft.design.part_identity import reviewed_part

    definition = get_recipe("stm32g030k6t6-minimal@3")
    f6 = get_recipe("stm32g030f6p6-minimal@2")
    assert sum("adc" in pin.capabilities for pin in definition.allocatable_pins) == 14
    assert sum("adc" in pin.capabilities for pin in f6.allocatable_pins) == 8
    assert {"19", "21"}.isdisjoint({pin.pin for pin in definition.allocatable_pins})
    assert all(pin.input_only for pin in definition.allocatable_pins if pin.pin in {"2", "3"})
    ports = {f"adc_{index}": f"ADC_{index}" for index in range(4)}
    requirement = CircuitRequirement(
        id="g0", sheet="MCU", role="mcu_core", family="stm32g0",
        interfaces=["adc"], ports=ports,
    )
    allocations = allocate_requirement_pins(definition, requirement)
    assert len({row.pin for row in allocations}) == 4
    assert {row.pin for row in allocations} <= {
        "7", "8", "9", "10", "11", "12", "13", "14",
        "15", "16", "17", "22", "23", "31",
    }
    expansion = expand_recipe(RecipeSelection(
        recipe=definition.recipe, instance="g0", sheets={"mcu": "MCU"},
        port_bindings={"vdd": "+3V3", "gnd": "GND", **ports},
        pin_allocations=allocations,
    ))
    refs = {part.recipe_role: part.ref for part in expansion.parts}
    nets = _mcu_recipe_nets(expansion)
    mcu = refs["mcu"]
    assert {(mcu, "19"), (mcu, "21")} <= {
        (endpoint.ref, endpoint.pin) for endpoint in expansion.no_connect_pins
    }
    part = next(part for part in expansion.parts if part.ref == mcu)
    record = reviewed_part(part.mpn)
    assert (part.mpn, part.symbol, part.footprint) == (
        "STM32G030K6T6", record.symbol, record.footprint,
    )
    assert record.lcsc == "C529331" and len(record.contacts) == 32
    swd = next(part for part in expansion.parts if part.recipe_role == "swd_header")
    assert (swd.mpn, swd.symbol, swd.footprint) == (
        "68000-406HLF", "Connector_Generic:Conn_01x06",
        "amphenol-68000-406hlf:HDR-TH_6P-P2.54-V-M-1",
    )
    assert next(part for part in f6.parts if part.role == "swd_header").mpn is None
    assert nets[mcu, "4"] == "+3V3" and nets[mcu, "5"] == "GND"
    assert nets[mcu, "6"] == nets[refs["swd_header"], "5"] == nets[refs["reset_button"], "1"]
    assert nets[mcu, "24"] == nets[refs["swd_header"], "3"]
    assert nets[mcu, "25"] == nets[refs["swd_header"], "4"] == nets[refs["boot0_pulldown"], "1"]
    assert nets[refs["boot0_pulldown"], "2"] == "GND"
    assert nets[refs["reset_pullup"], "1"] == "+3V3"
    assert nets[refs["reset_pullup"], "2"] == nets[mcu, "6"] == nets[refs["reset_capacitor"], "1"]
    assert nets[refs["reset_capacitor"], "2"] == "GND"
    for role in ("local_decoupling", "bulk_decoupling"):
        assert (nets[refs[role], "1"], nets[refs[role], "2"]) == ("+3V3", "GND")
    for allocation in allocations:
        assert nets[mcu, allocation.pin] == allocation.net
    assert check_net_coverage(BOM(parts=expansion.parts, connections=expansion.connections,
                                  no_connect_pins=expansion.no_connect_pins)).ok


def test_stm32g0_family_resolution_preserves_exact_and_saved_f6():
    from kicraft.design.part_identity import matches_part_identity
    from kicraft.design.stage_state import legacy_recipe_resume_stage
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = {
        "sheets": [{"name": "MCU", "stem": "MCU", "function": "MCU core"}],
        "power_nets": ["+3V3", "GND"], "rail_voltages": {"+3V3": 3.3},
        "inter_sheet_nets": [],
        "requirements": [{
            "id": "mcu", "sheet": "MCU", "role": "mcu_core", "family": "stm32g0",
            "ports": {"vdd": "+3V3", "gnd": "GND"},
        }],
    }
    for family in ("STM32G0", "STM32G030"):
        result = resolve_architecture_recipes(payload, {"named_parts": [family]})
        assert not result.blocking
        assert result.exact_parts["mcu"] == "STM32G030K6T6"
        assert next(part for part in expand_recipe(result.selections[0]).parts
                    if part.recipe_role == "mcu").mpn == "STM32G030K6T6"
    exact = resolve_architecture_recipes(payload, {"named_parts": ["STM32G030F6P6"]})
    assert not exact.blocking
    assert exact.exact_parts["mcu"] == "STM32G030F6P6"
    assert next(part for part in expand_recipe(exact.selections[0]).parts
                if part.recipe_role == "mcu").footprint == "Package_SO:TSSOP-20_4.4x6.5mm_P0.65mm"
    saved = RecipeSelection(recipe="stm32g030f6p6-minimal@1", instance="old",
                            sheets={"mcu": "MCU"}, port_bindings={"vdd": "+3V3", "gnd": "GND"})
    with pytest.raises(ValueError):
        expand_recipe(saved)
    assert legacy_recipe_resume_stage({
        "architecture": {"recipe_selections": [saved.model_dump()]},
    }) == "architecture"
    assert not matches_part_identity("STM32G030F6P6", "STM32G030K6T6")


def test_attiny1604_complete_artifact_preserves_updi_and_bypass():
    definition = get_recipe("attiny1604-ssn-minimal@1")
    expansion = expand_recipe(
        RecipeSelection(
            recipe=definition.recipe,
            instance="tiny",
            sheets={"mcu": "MCU"},
            port_bindings={"vdd": "+3V3", "gnd": "GND"},
        )
    )
    refs = {part.recipe_role: part.ref for part in expansion.parts}
    nets = _mcu_recipe_nets(expansion)
    mcu = refs["mcu"]
    assert (next(part for part in expansion.parts if part.ref == mcu).mpn,
            next(part for part in expansion.parts if part.ref == mcu).symbol,
            next(part for part in expansion.parts if part.ref == mcu).footprint) == (
        "ATTINY1604-SSN",
        "MCU_Microchip_ATtiny:ATtiny1604-SS",
        "Package_SO:SOIC-14_3.9x8.7mm_P1.27mm",
    )
    assert nets[mcu, "1"] == "+3V3" and nets[mcu, "14"] == "GND"
    assert nets[mcu, "10"] == nets[refs["updi_header"], "2"] == nets[refs["updi_pullup"], "2"]
    assert nets[refs["updi_pullup"], "1"] == "+3V3"
    assert (refs["updi_header"], "4") in {
        (endpoint.ref, endpoint.pin) for endpoint in expansion.no_connect_pins
    }
    for role in ("local_decoupling", "bulk_decoupling"):
        assert (nets[refs[role], "1"], nets[refs[role], "2"]) == ("+3V3", "GND")
    assert (mcu, "10") not in {(endpoint.ref, endpoint.pin) for endpoint in expansion.no_connect_pins}


def test_stm32g030_mux_allocator_reserves_only_bound_documented_peripherals():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins

    definition = get_recipe("stm32g030f6p6-minimal@2")
    adc_ports = {f"adc_{index}": f"ADC_{index}" for index in range(8)}
    direct = CircuitRequirement(
        id="g0", sheet="MCU", role="mcu_core", family="stm32g0",
        interfaces=["adc"], ports=adc_ports,
    )
    assert {row.pin for row in allocate_requirement_pins(definition, direct)} == {
        str(pin) for pin in range(7, 15)
    }
    uart_and_adc = direct.model_copy(
        update={
            "interfaces": ["uart", "adc"],
            "ports": {**adc_ports, "uart_tx": "UART_TX", "uart_rx": "UART_RX"},
        }
    )
    with pytest.raises(PinAllocationError, match="unsatisfied_pin_capability"):
        allocate_requirement_pins(definition, uart_and_adc)


def test_attiny1604_mux_and_updi_boundaries_are_explicit():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins

    definition = get_recipe("attiny1604-ssn-minimal@1")
    collision = CircuitRequirement(
        id="tiny", sheet="MCU", role="mcu_core", family="attiny1604",
        interfaces=["uart", "i2c_controller"],
        ports={"uart_tx": "UART_TX", "uart_rx": "UART_RX", "i2c_sda": "I2C_SDA", "i2c_scl": "I2C_SCL"},
    )
    with pytest.raises(PinAllocationError, match="conflicting_pin_capability"):
        allocate_requirement_pins(definition, collision)
    overcommitted = CircuitRequirement(
        id="tiny", sheet="MCU", role="mcu_core", family="attiny1604",
        interfaces=["adc"],
        ports={f"adc_{index}": f"ADC_{index}" for index in range(8)},
    )
    with pytest.raises(PinAllocationError, match="unsatisfied_pin_capability"):
        allocate_requirement_pins(definition, overcommitted)
    assert "10" not in {row.pin for row in definition.allocatable_pins}




@pytest.mark.parametrize(
    "recipe,expected_pins",
    [
        ("stm32g030f6p6-minimal@2", {"i2c_scl": "1", "i2c_sda": "2"}),
        ("stm32g030k6t6-minimal@3", {"i2c_scl": "32", "i2c_sda": "1"}),
        ("attiny1604-ssn-minimal@1", {"i2c_scl": "13", "i2c_sda": "12"}),
    ],
)
def test_new_mcu_i2c_artifact_owns_one_pullup_per_documented_line(recipe, expected_pins):
    expansion = expand_recipe(
        RecipeSelection(
            recipe=recipe,
            instance="i2c",
            sheets={"mcu": "MCU"},
            port_bindings={
                "vdd": "+3V3",
                "gnd": "GND",
                "i2c_scl": "I2C_SCL",
                "i2c_sda": "I2C_SDA",
            },
        )
    )
    refs = {part.recipe_role: part.ref for part in expansion.parts}
    nets = _mcu_recipe_nets(expansion)
    mcu = refs["mcu"]
    for port, pin in expected_pins.items():
        assert nets[mcu, pin] == nets[refs[f"{port}_pullup"], "2"] == port.upper()
        assert nets[refs[f"{port}_pullup"], "1"] == "+3V3"
    assert {part.recipe_role for part in expansion.parts if part.recipe_role.startswith("i2c_")} == {
        "i2c_scl_pullup",
        "i2c_sda_pullup",
    }


@pytest.mark.parametrize("recipe", ["stm32g030f6p6-minimal@2", "stm32g030k6t6-minimal@3", "attiny1604-ssn-minimal@1"])
def test_new_mcu_conditional_ads1115_pair_has_distinct_addresses_and_eight_endpoints(recipe):
    expansion = expand_recipe(
        RecipeSelection(
            recipe=recipe, instance="adc", sheets={"mcu": "MCU"},
            parameters={"adc_expansion": "ads1115x2"},
            port_bindings={"vdd": "+3V3", "gnd": "GND", **{f"adc_{index}": f"ADC_{index}" for index in range(8)}},
        )
    )
    nets = _mcu_recipe_nets(expansion)
    ads = [part.ref for part in expansion.parts if part.mpn == "ADS1115IDGSR"]
    assert len(ads) == 2
    assert {nets[ref, "1"] for ref in ads} == {"+3V3", "GND"}
    assert {nets[ref, pin] for ref in ads for pin in ("4", "5", "6", "7")} == {f"ADC_{index}" for index in range(8)}
    mcu = next(part.ref for part in expansion.parts if part.recipe_role == "mcu")
    assert all(
        nets[ref, "9"] == nets[mcu, get_recipe(recipe).fixed_port_pins["i2c_sda"]]
        and nets[ref, "10"] == nets[mcu, get_recipe(recipe).fixed_port_pins["i2c_scl"]]
        for ref in ads
    )


@pytest.mark.parametrize("recipe", ["stm32g030f6p6-minimal@2", "stm32g030k6t6-minimal@3", "attiny1604-ssn-minimal@1"])
def test_new_mcu_conditional_mcp23017_owns_sixteen_explicit_outputs(recipe):
    expansion = expand_recipe(
        RecipeSelection(
            recipe=recipe, instance="gpio", sheets={"mcu": "MCU"},
            parameters={"gpio_expansion": "mcp23017"},
            port_bindings={"vdd": "+3V3", "gnd": "GND", **{f"output_{index}": f"OUT_{index}" for index in range(16)}},
        )
    )
    nets = _mcu_recipe_nets(expansion)
    mcp = next(part.ref for part in expansion.parts if part.mpn == "MCP23017-E/SO")
    assert {nets[mcp, str(pin)] for pin in (*range(21, 29), *range(1, 9))} == {f"OUT_{index}" for index in range(16)}
    assert {nets[mcp, pin] for pin in ("15", "16", "17")} == {"GND"}


def test_new_mcu_i2c_rejects_single_wire_pseudo_bus():
    with pytest.raises(ValueError, match="requires both i2c_scl and i2c_sda"):
        expand_recipe(
            RecipeSelection(
                recipe="stm32g030f6p6-minimal@2",
                instance="bad_i2c",
                sheets={"mcu": "MCU"},
                port_bindings={"vdd": "+3V3", "gnd": "GND", "i2c_scl": "I2C_SCL"},
            )
        )


@pytest.mark.parametrize(
    ("family", "exact_part"),
    [("stm32g0", "STM32G030F6P6"), ("stm32g0", "STM32G030K6T6"), ("attiny1604", "ATTINY1604-SSN")],
)
def test_new_mcu_cores_require_their_reviewed_3v3_logic_rail(family, exact_part):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = {
        "sheets": [{"name": "MCU", "stem": "MCU", "function": "MCU core"}],
        "power_nets": ["+3V3", "GND"],
        "rail_voltages": {"+3V3": 3.3},
        "inter_sheet_nets": [],
        "requirements": [{
            "id": "mcu", "sheet": "MCU", "role": "mcu_core",
            "family": family, "exact_part": exact_part,
            "ports": {"vdd": "+3V3", "gnd": "GND"},
        }],
    }
    assert not resolve_architecture_recipes(payload).blocking
    payload["power_nets"] = ["+5V", "GND"]
    payload["rail_voltages"] = {"+5V": 5.0}
    payload["requirements"][0]["ports"]["vdd"] = "+5V"
    rejected = resolve_architecture_recipes(payload)
    assert any(row.code == "conflicting_recipe_supply" for row in rejected.blocking)


def test_external_display_recipes_keep_physical_load_topologies():
    ws = expand_recipe(
        RecipeSelection(
            recipe="ws2812-external-strip-interface@1",
            instance="strip",
            sheets={"interface": "DISPLAY"},
            port_bindings={"vdd_5v": "+5V", "gnd": "GND", "data_in": "MCU_DATA"},
        )
    )
    ws_nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in ws.connections
        for endpoint in connection.endpoints
    }
    connector = next(part for part in ws.parts if part.recipe_role == "connector")
    resistor = next(part for part in ws.parts if part.recipe_role == "series_resistor")
    level_shifter = next(part for part in ws.parts if part.recipe_role == "level_shifter")
    assert ws_nets[connector.ref, "1"] == "+5V"
    assert ws_nets[connector.ref, "3"] == "GND"
    assert ws_nets[connector.ref, "2"] == ws_nets[resistor.ref, "2"]
    assert ws_nets[resistor.ref, "1"] == ws_nets[level_shifter.ref, "18"]
    assert ws_nets[level_shifter.ref, "2"] == "MCU_DATA"

    power = expand_recipe(
        RecipeSelection(
            recipe="al8860-3w-power-led@1",
            instance="lamp",
            sheets={"interface": "DISPLAY"},
            port_bindings={"vin": "+12V", "gnd": "GND", "enable": "MCU_ENABLE"},
        )
    )
    power_nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in power.connections
        for endpoint in connection.endpoints
    }
    driver = next(part for part in power.parts if part.recipe_role == "driver")
    emitter = next(part for part in power.parts if part.recipe_role == "emitter")
    inductor = next(part for part in power.parts if part.recipe_role == "inductor")
    assert power_nets[driver.ref, "1"] != power_nets[driver.ref, "4"]
    assert power_nets[driver.ref, "4"] == "MCU_ENABLE"
    assert power_nets[emitter.ref, "2"] == power_nets[driver.ref, "1"]
    assert power_nets[emitter.ref, "2"] != "+12V"
    assert power_nets[emitter.ref, "1"] == power_nets[emitter.ref, "3"] == power_nets[inductor.ref, "1"]


def test_four_digit_recipe_has_four_independent_common_cathode_sinks():
    expansion = expand_recipe(
        RecipeSelection(
            recipe="max7219-four-digit-seven-segment@1",
            instance="digits",
            sheets={"interface": "DISPLAY"},
            port_bindings={"vdd_5v": "+5V", "gnd": "GND", "din": "DIN", "clk": "CLK", "load": "LOAD"},
        )
    )
    nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }
    driver = next(part for part in expansion.parts if part.recipe_role == "driver")
    level_shifter = next(part for part in expansion.parts if part.recipe_role == "level_shifter")
    assert [nets[level_shifter.ref, pin] for pin in ("18", "17", "16")] == [
        nets[driver.ref, pin] for pin in ("1", "13", "12")
    ]
    assert len({nets[level_shifter.ref, pin] for pin in ("2", "3", "4")}) == 3
    digits = [part for part in expansion.parts if part.recipe_role == "digit"]
    sinks = {nets[driver.ref, pin] for pin in ("2", "11", "6", "7")}
    assert len(digits) == 4
    assert len(sinks) == 4
    assert {nets[digit.ref, "3"] for digit in digits} == sinks
    assert {(pin.ref, pin.pin) for pin in expansion.no_connect_pins} >= {(driver.ref, "24")}


def test_usb_uart_recipes_expose_header_and_refuse_unbound_modem_contacts():
    for recipe, vdd in (("ch340c-usb-uart@1", "+3V3"), ("cp2102n-usb-uart@1", "+3V3")):
        expansion = expand_recipe(
            RecipeSelection(
                recipe=recipe,
                instance="bridge",
                sheets={"interface": "USB"},
                port_bindings={
                    "vbus": "USB5V", "vdd": vdd, "gnd": "GND",
                    "usb_dp": "D+", "usb_dm": "D-", "tx": "BRIDGE_TX", "rx": "BRIDGE_RX",
                },
                parameters={"supply_voltage": 3.3} if recipe.startswith("ch340") else {},
            )
        )
        header = next(part for part in expansion.parts if part.recipe_role == "program_header")
        owned = {(pin.ref, pin.pin) for pin in expansion.ownership.pins}
        no_connects = {(pin.ref, pin.pin) for pin in expansion.no_connect_pins}
        assert {(header.ref, str(pin)) for pin in range(1, 7)} <= owned
        assert {(header.ref, "5"), (header.ref, "6")} <= no_connects


@pytest.mark.parametrize(
    ("recipe", "gpio", "expected_pin"),
    [
        ("esp32-c3-mini-1-minimal@2", 0, "12"),
        ("esp32-c3-mini-1-minimal@2", 5, None),
        ("esp32-c3-mini-1-minimal@2", 6, None),
        ("esp32-wroom-32e-minimal@1", 16, None),
        ("esp32-wroom-32e-minimal@1", 34, "6"),
        ("esp32-s3-wroom-1-minimal@1", 14, "22"),
        ("esp32-s3-wroom-1-minimal@1", 21, None),
    ],
)
def test_module_adc_requests_use_real_supported_analog_contacts(recipe, gpio, expected_pin):
    from kicraft.design.recipes.pin_allocator import PinAllocationRequest

    pins = tuple(pin for pin in get_recipe(recipe).allocatable_pins if pin.gpio == gpio)
    request = PinAllocationRequest(id="sample", net="SENSOR", capability="adc")
    if expected_pin is None:
        with pytest.raises(PinAllocationError):
            allocate_pins(pins, [request])
    else:
        assert allocate_pins(pins, [request])[0].pin == expected_pin


def test_s3_n8r8_internal_psram_contacts_cannot_be_assigned_to_application_outputs():
    from kicraft.design.recipes.pin_allocator import PinAllocationRequest

    definition = get_recipe("esp32-s3-wroom-1-minimal@1")
    pins = tuple(pin for pin in definition.allocatable_pins if pin.gpio in {35, 36, 37})
    with pytest.raises(PinAllocationError):
        allocate_pins(pins, [PinAllocationRequest(id="load", net="LOAD", capability="output")])
    expanded = expand_recipe(RecipeSelection(
        recipe=definition.recipe, instance="core", sheets={"mcu": "MCU"},
        port_bindings={"vdd": "+3V3", "gnd": "GND"},
    ))
    mcu = next(part.ref for part in expanded.parts if part.recipe_role == "mcu")
    assert {(mcu, pin) for pin in ("28", "29", "30")} <= {
        (pin.ref, pin.pin) for pin in expanded.no_connect_pins
    }
    assert not any(
        pin.ref == mcu and pin.pin in {"28", "29", "30"}
        for connection in expanded.connections for pin in connection.endpoints
    )


def _current_conditioner_adc_architecture():
    from kicraft.design.models import Architecture, CircuitRequirement

    ports = {"vplus": "+5V", "vminus": "GND", "analog_out_return": "GND"}
    for index in (1, 2):
        ports.update({
            f"sensor_{index}_signal": f"SHUNT_{index}",
            f"sensor_{index}_return": "GND",
            f"load_return_in_{index}": f"SHUNT_{index}",
            f"load_return_out_{index}": "GND",
            f"conditioned_{index}": f"ADC_{index}",
            f"analog_out_{index}": f"ADC_{index}",
        })
    return Architecture(
        topologies={"AFE": "Two current-sense channels into MCU ADC"},
        mcu_present=True,
        rail_voltages={"+5V": 5.0, "+3V3": 3.3, "GND": 0.0},
        power_nets=["+5V", "+3V3", "GND"],
        sheets=[
            {"name": "AFE", "stem": "AFE", "function": "Current sensing"},
            {"name": "MCU", "stem": "MCU", "function": "Analog acquisition"},
        ],
        inter_sheet_nets=[
            {"name": f"ADC_{index}", "endpoints": [
                {"sheet": "AFE", "direction": "output"},
                {"sheet": "MCU", "direction": "input"},
            ]}
            for index in (1, 2)
        ],
        requirements=[
            CircuitRequirement(
                id="afe", sheet="AFE", role="analog_block", family="analog-conditioner",
                exact_part="TLV9062IDR",
                parameters={"channel_count": 2, "sensor_type": "current-sense"}, ports=ports,
            ),
            CircuitRequirement(
                id="mcu", sheet="MCU", role="mcu_core", family="stm32g0",
                exact_part="STM32G030F6P6", interfaces=["adc"],
                ports={"vdd": "+3V3", "gnd": "GND", "adc_0": "ADC_1", "adc_1": "ADC_2"},
            ),
            CircuitRequirement(
                id="loads", sheet="AFE", role="connector", family="screw-terminal",
                ports={"pin1": "SHUNT_1", "pin2": "GND", "pin3": "SHUNT_2", "pin4": "GND"},
            ),
        ],
    )


@pytest.mark.parametrize("stated_limit", [None, 5.0, 3.0])
def test_analog_output_respects_connected_adc_without_duplicating_explicit_inputs(stated_limit):
    from kicraft.design.recipes import apply_architecture_recipe_resolution
    from kicraft.design.synthesis.validation import _resistance_ohms

    architecture = _current_conditioner_adc_architecture()
    if stated_limit is not None:
        architecture.requirements[0].parameters["consumer_input_max_v"] = stated_limit
    original = architecture.model_dump()
    resolved = apply_architecture_recipe_resolution(architecture)
    conditioner = next(row for row in resolved.requirements if row.id == "afe")
    artifact = lower_requirement(conditioner)
    resistances = {
        group.role: _resistance_ohms(group.value)
        for group in artifact.groups if group.role in {"gain_feedback", "gain_ground"}
    }
    actual_endpoint = 0.1 * (1 + resistances["gain_feedback"] / resistances["gain_ground"])
    assert actual_endpoint <= min(3.3, stated_limit if stated_limit is not None else 3.3)
    mcu = next(row for row in resolved.recipe_selections if "mcu" in row.requirement_ids)
    assert [(row.net, row.capability) for row in mcu.pin_allocations] == [
        ("ADC_1", "adc"), ("ADC_2", "adc"),
    ]
    assert architecture.model_dump() == original


def test_explicit_analog_gain_is_refused_rather_than_rewritten_for_adc():
    from kicraft.design.recipes import apply_architecture_recipe_resolution, RecipeResolutionError

    architecture = _current_conditioner_adc_architecture()
    architecture.requirements[0].parameters["gain"] = 49.5
    original = architecture.model_dump()
    with pytest.raises(RecipeResolutionError) as refused:
        apply_architecture_recipe_resolution(architecture)
    assert any(row.requirement_id == "afe" for row in refused.value.diagnostics)
    assert architecture.model_dump() == original


def test_constructed_analog_circuit_remains_observable_after_library_normalization():
    from kicraft.design.recipes import apply_architecture_recipe_resolution
    from kicraft.eval.analog_artifact_evidence import analog_conditioner_facts
    from kicraft.server.stage_work_units import probe_architecture_construction

    architecture = apply_architecture_recipe_resolution(_current_conditioner_adc_architecture())
    bom = probe_architecture_construction({"architecture": architecture.model_dump()})
    assert bom is not None
    pad_nets = {
        (endpoint["ref"], endpoint["pin"]): connection["net_name"]
        for connection in bom["connections"]
        for endpoint in connection["endpoints"]
    }
    facts = analog_conditioner_facts(
        bom["parts"], {}, pad_nets, state={"architecture": architecture.model_dump()},
        supply_voltage=lambda rail, ground: 5.0 if (rail, ground) == ("+5V", "GND") else None,
    )
    assert facts["channel_counts"]["analog_conditioner"] == 2
    assert facts["analog_output_max_v"] <= 3.3

@pytest.mark.parametrize(
    ("shared_data", "shared_vbus", "blocked"),
    [(True, True, True), (True, False, True), (False, True, True), (False, False, False)],
)
def test_integrated_usb_bridge_does_not_parallel_device_receptacles(shared_data, shared_vbus, blocked):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = {
        "rail_voltages": {"+3V3": 3.3, "VBUS": 5.0, "VBUS2": 5.0},
        "power_nets": ["+3V3", "VBUS", "VBUS2", "GND"],
        "inter_sheet_nets": [],
        "sheets": [
            {"name": "BRIDGE", "stem": "BRIDGE", "function": "USB serial bridge"},
            {"name": "USB2", "stem": "USB2", "function": "Separate USB interface"},
        ],
        "requirements": [
            {
                "id": "bridge", "sheet": "BRIDGE", "role": "bus_interface",
                "family": "usb-uart-bridge", "exact_part": "CH340C",
                "ports": {
                    "vbus": "VBUS", "vdd": "+3V3", "gnd": "GND",
                    "usb_dp": "BRIDGE_DP", "usb_dm": "BRIDGE_DM",
                    "tx": "UART_TX", "rx": "UART_RX",
                },
            },
            {
                "id": "socket", "sheet": "USB2", "role": "connector",
                "family": "usb-c-usb2-device",
                "ports": {
                    "vbus": "VBUS" if shared_vbus else "VBUS2", "gnd": "GND",
                    "usb_dp": "BRIDGE_DP" if shared_data else "SECOND_DP",
                    "usb_dm": "SECOND_DM",
                },
            },
        ],
    }
    result = resolve_architecture_recipes(payload)
    collisions = [row for row in result.blocking if row.code == "usb_receptacle_ownership_collision"]
    assert bool(collisions) is blocked


def test_exact_c3_n4_identity_is_not_replaced_by_stocked_family_variant():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes


    explicit = _typed_esp32_architecture(
        family="esp32-c3-mini-1-module", exact_part="ESP32-C3-MINI-1-N4", usb=True
    )
    result = resolve_architecture_recipes(explicit)
    assert not result.blocking
    core = next(row for row in result.selections if "mcu_core" in row.requirement_ids)
    assert core.recipe == "esp32-c3-mini-1-minimal@2"


@pytest.mark.parametrize("missing", ("usb_dm", "usb_dp"))
def test_c3_h4x_requires_both_native_usb_ports(missing):
    selection = RecipeSelection(
        recipe="esp32-c3-mini-1-h4x-minimal@2",
        instance="main",
        sheets={"mcu": "MCU"},
        port_bindings={"vdd": "+3V3", "gnd": "GND", "usb_dm": "DM", "usb_dp": "DP"},
    )
    missing_port = selection.model_copy(
        update={"port_bindings": {
            name: net for name, net in selection.port_bindings.items() if name != missing
        }}
    )
    with pytest.raises(ValueError, match="port bindings mismatch"):
        expand_recipe(missing_port)


@pytest.mark.parametrize("missing", ("usb_dm", "usb_dp"))
def test_s3u_requires_native_usb(missing):
    selection = RecipeSelection(
        recipe="esp32-s3-mini-1u-n4r2-minimal@1",
        instance="main",
        sheets={"mcu": "MCU"},
        port_bindings={"vdd": "+3V3", "gnd": "GND", "usb_dm": "DM", "usb_dp": "DP"},
    )
    bindings = {name: net for name, net in selection.port_bindings.items() if name != missing}
    with pytest.raises(ValueError, match="port bindings mismatch"):
        expand_recipe(selection.model_copy(update={"port_bindings": bindings}))


def test_s3u_memory_straps_and_radio_identity():
    from kicraft.design.part_identity import reviewed_part
    from kicraft.design.recipes.wave_a_mcus import ESP32_S3_MINI_1U_N4R2_MINIMAL

    record = reviewed_part("esp32-s3-mini-1u-n4r2")
    assert record.operating_limits["flash_mb"] == 4
    assert record.operating_limits["psram_mb"] == 2
    assert not {"wifi-module", "bluetooth-le-soc"} & record.physical_features
    recipe = ESP32_S3_MINI_1U_N4R2_MINIMAL
    assert {"26", "41"} <= {pin.pin for pin in recipe.no_connects}
    assert not {"26", "41"} & {pin.pin for pin in recipe.allocatable_pins}


def test_exact_s3_n8_is_not_replaced_by_u_variant():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    explicit = _typed_esp32_architecture(
        family="esp32-s3-module", exact_part="ESP32-S3-MINI-1-N8", usb=True
    )
    result = resolve_architecture_recipes(explicit)
    assert not result.blocking
    core = next(row for row in result.selections if "mcu_core" in row.requirement_ids)
    assert core.recipe == "esp32-s3-mini-1-minimal@1"


def test_generic_s3_selects_available_u_module():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    architecture = _typed_esp32_architecture(exact_part=None, usb=True)
    result = resolve_architecture_recipes(architecture)
    assert not result.blocking
    core = next(row for row in result.selections if "mcu_core" in row.requirement_ids)
    assert core.recipe == "esp32-s3-mini-1u-n4r2-minimal@1"


def test_generic_s3_name_is_not_a_protected_exact_variant():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    architecture = _typed_esp32_architecture(exact_part="ESP32-S3", usb=True)
    result = resolve_architecture_recipes(architecture)
    assert not result.blocking
    core = next(row for row in result.selections if "mcu_core" in row.requirement_ids)
    assert core.recipe == "esp32-s3-mini-1u-n4r2-minimal@1"


@pytest.mark.parametrize("constraint", ["8 MB flash", "Wi-Fi", "PCB antenna"])
def test_generic_s3_u_rejects_unfulfilled_capability(constraint):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    architecture = _typed_esp32_architecture(exact_part=None, usb=True)
    result = resolve_architecture_recipes(architecture, {"constraints": [constraint]})
    assert any(row.code == "s3u_requested_capability_unproven" for row in result.blocking)


@pytest.mark.parametrize("recipe", [
    "esp32-s3-mini-1u-n4r2-uart-minimal@1",
    "esp32-c3-mini-1-h4x-uart-minimal@1",
])
def test_uart_recipe_cannot_silently_consume_usb_bindings(recipe):
    selection = RecipeSelection(
        recipe=recipe, instance="core", sheets={"mcu": "MCU"},
        port_bindings={"vdd": "+3V3", "gnd": "GND",
                       "usb_dm": "DM", "usb_dp": "DP"},
    )
    with pytest.raises(ValueError, match="port bindings mismatch"):
        expand_recipe(selection)

@pytest.mark.parametrize(("architecture_factory", "recipe"), [
    (_typed_esp32_architecture, "esp32-s3-mini-1u-n4r2-uart-minimal@1"),
    (_c3_native_usb_architecture, "esp32-c3-mini-1-h4x-uart-minimal@1"),
])
def test_explicit_uart_rejects_native_usb_contract(architecture_factory, recipe):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    architecture = architecture_factory()
    core = next(row for row in architecture["requirements"] if row["id"] == "mcu_core")
    core.update(exact_part=None, interfaces=["usb_device"])
    architecture["recipe_selections"] = [RecipeSelection(
        recipe=recipe, instance="core", sheets={"mcu": "MCU"},
        port_bindings={"vdd": "+3V3", "gnd": "GND"},
        requirement_ids=["mcu_core"],
    ).model_dump()]
    result = resolve_architecture_recipes(
        architecture, allowed_maturities=frozenset({"production", "canary"}),
    )
    assert any(row.code == "conflicting_programming_contract" for row in result.blocking)


@pytest.mark.parametrize("mcu_recipe", [
    "esp32-s3-mini-1u-n4r2-minimal@1",
    "esp32-c3-mini-1-minimal@2",
    "esp32-c3-mini-1-h4x-minimal@2",
])
def test_reviewed_recipe_supply_locality_rejects_swapped_owned_roles(mcu_recipe):
    from kicraft.design.synthesis.autoplacer import _recipe_local_bypasses

    selections = [
        RecipeSelection(recipe=mcu_recipe, instance="core", sheets={"mcu": "MAIN"},
                        port_bindings={"vdd": "+3V3", "gnd": "GND",
                                       "usb_dm": "DM", "usb_dp": "DP"},
                        requirement_ids=["core"]),
        RecipeSelection(recipe="ylptec-k7803m-1000r3@1", instance="supply",
                        sheets={"power": "MAIN"},
                        port_bindings={"input": "VIN", "output": "+3V3", "gnd": "GND"},
                        requirement_ids=["supply"]),
    ]
    expansions = expand_selections(selections)
    bom = BOM(parts=[p for e in expansions for p in e.parts],
              connections=[c for e in expansions for c in e.connections],
              recipe_ownership=[e.ownership for e in expansions])
    # Swapping role claims on a shared rail cannot steal another owner's cap.
    cap = next(p for p in bom.parts
               if p.recipe_instance == "core" and p.recipe_role == "local_decoupling")
    cap.recipe_role = "bulk_decoupling"
    with pytest.raises(ValueError, match="recipe role"):
        _recipe_local_bypasses(SimpleNamespace(recipe_selections=selections), bom)

@pytest.mark.parametrize("recipe", [
    "esp32-c3-mini-1-minimal@2", "esp32-c3-mini-1-h4x-minimal@2",
])
def test_c3_bypass_expansion_keeps_both_caps_on_mcu_supply_and_return(recipe):
    expansion = expand_recipe(RecipeSelection(
        recipe=recipe, instance="core", sheets={"mcu": "MCU"},
        port_bindings={"vdd": "LOGIC", "gnd": "GND", "usb_dm": "DM", "usb_dp": "DP"},
    ))
    nets = {
        (pin.ref, pin.pin): connection.net_name
        for connection in expansion.connections for pin in connection.endpoints
    }
    for role in ("local_decoupling", "bulk_decoupling"):
        capacitor = next(part for part in expansion.parts if part.recipe_role == role)
        assert capacitor.footprint == "Capacitor_SMD:C_0402_1005Metric"
        assert {nets[capacitor.ref, "1"], nets[capacitor.ref, "2"]} == {"LOGIC", "GND"}


def test_reset_control_bound_ports_and_reallocation_use_documented_c3_pads():
    from kicraft.design.models import Architecture, CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins
    from kicraft.design.recipes.registry import registered_recipes
    from kicraft.design.recipes.resolver import _complete_reset_control_peers

    signals = {
        "ain1": "BRIDGE_A1", "ain2": "BRIDGE_A2", "bin1": "BRIDGE_B1",
        "bin2": "BRIDGE_B2", "sleep_n": "ENABLE", "fault_n": "FAULT",
    }
    core = CircuitRequirement(
        id="core", sheet="MCU", role="mcu_core", family="esp32-c3-mini-1-module",
        ports={**{f"output_{i}": net for i, net in enumerate(list(signals.values())[:5])},
               "input_0": "FAULT"},
    )
    motor = CircuitRequirement(
        id="motor", sheet="MOTOR", role="driver", family="dual-dc-motor-driver",
        exact_part="DRV8833PWPR", ports=signals,
    )
    architecture = Architecture.model_validate({
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "controller"},
            {"name": "MOTOR", "stem": "MOTOR", "function": "motor driver"},
        ],
        "power_nets": ["GND"],
        "inter_sheet_nets": [
            {"name": net, "endpoints": [
                {"sheet": "MCU", "direction": "input" if name == "fault_n" else "output"},
                {"sheet": "MOTOR", "direction": "output" if name == "fault_n" else "input"},
            ]}
            for name, net in signals.items()
        ],
        "requirements": [core.model_dump(), motor.model_dump()],
    })
    completed, controls = _complete_reset_control_peers(core, architecture, registered_recipes())
    assert set(controls) == set(signals.values()) - {"FAULT"}
    assert set(completed.ports.values()) == set(signals.values())
    definition = get_recipe("esp32-c3-mini-1-h4x-minimal@2")
    old = allocate_requirement_pins(definition, completed)
    result = allocate_requirement_pins(
        definition, completed, existing=old, required_reset_controls=controls,
    )
    gpio_by_pin = {pin.pin: pin.gpio for pin in definition.allocatable_pins}
    mapped = {row.net: gpio_by_pin[row.pin] for row in result}
    assert {mapped[net] for net in controls} == {0, 1, 3, 4, 5}
    assert mapped["FAULT"] == 6
    assert allocate_requirement_pins(
        definition, completed, existing=result, required_reset_controls=controls,
    ) == result
    # A co-sheet motor output is not an MCU input merely because the sheet
    # exports it, and missing MCU ports are not manufactured from sheet labels.
    unbound = core.model_copy(update={"ports": {}})
    cosheet_motor = motor.model_copy(update={
        "sheet": "MCU", "ports": {**signals, "aout1": "MOTOR_POWER"},
    })
    cosheet = architecture.model_copy(update={"requirements": [unbound, cosheet_motor]})
    unchanged, no_controls = _complete_reset_control_peers(unbound, cosheet, registered_recipes())
    assert unchanged.ports == {}
    assert no_controls == {}

    from kicraft.design.recipes.resolver import resolve_architecture_recipes
    unknown = core.model_copy(update={"family": "unreviewed-mcu"})
    unknown_arch = architecture.model_copy(update={"requirements": [unknown, motor]})
    unresolved = resolve_architecture_recipes(unknown_arch)
    assert any(row.code == "unsupported_reset_control" and row.requirement_id == "core"
               for row in unresolved.blocking)
    second = core.model_copy(update={"id": "core2"})
    shared = architecture.model_copy(update={"requirements": [core, second, motor]})
    conflicts = resolve_architecture_recipes(shared)
    assert {row.requirement_id for row in conflicts.blocking
            if row.code == "conflicting_reset_control_owner"} == {"core", "core2"}


def test_reset_control_explicit_conflict_and_unknown_mcu_fail_closed():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins

    core = CircuitRequirement(
        id="core", sheet="MCU", role="mcu_core", family="esp32-c3-mini-1-module",
        ports={"gpio6": "ENABLE"},
    )
    controls = {"ENABLE": "external_low"}
    with pytest.raises(PinAllocationError, match="conflicting_reset_control"):
        allocate_requirement_pins(
            get_recipe("esp32-c3-mini-1-h4x-minimal@2"), core,
            required_reset_controls=controls,
        )
    good = core.model_copy(update={"ports": {"gpio4": "ENABLE"}})
    assert allocate_requirement_pins(
        get_recipe("esp32-c3-mini-1-h4x-minimal@2"), good,
        required_reset_controls=controls,
    )[0].pin == "18"
    neutral = core.model_copy(update={"ports": {"output_0": "ENABLE"}})
    with pytest.raises(PinAllocationError, match="unsupported_reset_control"):
        allocate_requirement_pins(
            get_recipe("attiny402-updi-minimal@1"), neutral,
            required_reset_controls=controls,
        )


def _adc_ownership_fixture(*, onboard=False, monitor=False, typed_source=False):
    """Four-channel 1152-shaped connector/ADC graph, with optional producers."""
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.resolver import ResolutionResult, _external_adc_contracts
    from kicraft.design.recipes.wave_a_mcus import STM32G030K6T6_MINIMAL
    from kicraft.design.recipes.wave_f_sensing import AH49ENTR_G1_ANALOG_HALL

    mcu = RecipeSelection(
        recipe=STM32G030K6T6_MINIMAL.recipe, instance="mcu", sheets={"mcu": "MCU"},
        port_bindings={"vdd": "+3V3", "gnd": "GND", **{
            f"adc_{i}": f"HALL_{i + 1}" for i in range(4)
        }},
        pin_allocations=[
            {"net": f"HALL_{i + 1}", "pin": str(i + 7), "capability": "adc"}
            for i in range(4)
        ],
    )
    connectors = [
        CircuitRequirement(
            id=f"terminal{i}", sheet="CONNECTORS", role="connector", family="screw-terminal",
            ports={"positive": f"HALL_{i}", "negative": "GND"},
            declared_interface={"ports": [
                {"key": "positive", "direction": "output" if monitor else "passive",
                 "function": "signal monitor" if monitor else "sensor signal", "pin": "1"},
                {"key": "negative", "direction": "passive", "function": "return", "pin": "2"},
            ]},
            obligations=([{"kind": "external_analog_source",
                           "original_obligation_id": f"source-{i}"}] if typed_source else []),
        )
        for i in range(1, 5)
    ]
    selections = [mcu]
    definitions = {mcu.recipe: STM32G030K6T6_MINIMAL}
    if onboard:
        definitions[AH49ENTR_G1_ANALOG_HALL.recipe] = AH49ENTR_G1_ANALOG_HALL
        selections.extend(
            RecipeSelection(
                recipe=AH49ENTR_G1_ANALOG_HALL.recipe, instance=f"sensor{i}",
                sheets={"sensor": "SENSORS"},
                port_bindings={"vdd": "+3V3", "gnd": "GND", "out": f"HALL_{i}"},
            )
            for i in range(1, 5)
        )
    result = ResolutionResult(requirements=connectors, selections=selections)
    _external_adc_contracts(result, definitions)
    return result, definitions


def test_external_adc_rejects_four_onboard_driver_conflicts_and_keeps_source_facts():
    result, _ = _adc_ownership_fixture(onboard=True, typed_source=True)
    conflicts = [row for row in result.blocking
                 if row.code == "external_adc_driver_ownership_conflict"]
    assert len(conflicts) == 4
    for i, conflict in enumerate(conflicts, 1):
        assert conflict.requirement_id == f"terminal{i}"
        assert f"HALL_{i}" in conflict.message
        assert any(f"sensor{i}.out" in item for item in conflict.evidence)
        assert result.requirements[i - 1].obligations[0].original_obligation_id == f"source-{i}"
    assert not result.external_adc_contracts
    ambiguous, _ = _adc_ownership_fixture(onboard=True)
    assert len([row for row in ambiguous.blocking
                if row.code == "external_adc_driver_ownership_conflict"]) == 4


def test_external_adc_accepts_external_only_and_explicit_onboard_monitor():
    external, _ = _adc_ownership_fixture(typed_source=True)
    assert not external.blocking
    assert [row.source_obligation_ids for row in external.external_adc_contracts] == [
        [f"source-{i}"] for i in range(1, 5)
    ]
    monitor, _ = _adc_ownership_fixture(onboard=True, monitor=True)
    assert not monitor.blocking
    assert not monitor.external_adc_contracts
    contradictory, _ = _adc_ownership_fixture(onboard=True, monitor=True, typed_source=True)
    assert any(row.code == "external_adc_scope_conflict" for row in contradictory.blocking)


def test_external_adc_refuses_relabel_of_preexisting_contract():
    from kicraft.design.recipes.resolver import _external_adc_contracts

    external, _ = _adc_ownership_fixture()
    monitor, definitions = _adc_ownership_fixture(onboard=True, monitor=True)
    _external_adc_contracts(monitor, definitions, external.external_adc_contracts)
    assert any(row.code == "external_adc_scope_conflict" for row in monitor.blocking)
    assert any(row.code == "external_adc_contract_not_preserved" for row in monitor.blocking)


def test_external_adc_graph_gate_includes_model_owned_driver_and_not_passive_or_input(monkeypatch):
    from kicraft.design.models import Architecture, BomPart, NetConnection
    from kicraft.design.synthesis import validation

    result, _ = _adc_ownership_fixture()
    architecture = Architecture(
        sheets=[{"name": "SENSORS", "stem": "SENSORS", "function": "Acquisition"}],
        power_nets=["GND"], inter_sheet_nets=[],
        external_adc_contracts=result.external_adc_contracts,
    )
    parts = [
        BomPart(ref=ref, value=ref, symbol=f"Example:{ref}", footprint="Test:Test", sheet="SENSORS")
        for ref in ("U1", "U2", "R1", "U3")
    ]
    bom = BOM.model_construct(parts=parts, connections=[
        NetConnection(net_name="HALL_1", sheet="SENSORS", endpoints=[
            {"ref": "U1", "pin": "1"}, {"ref": "U2", "pin": "1"},
            {"ref": "R1", "pin": "1"},
        ]),
        NetConnection(net_name="BUFFER_OUT", sheet="SENSORS", endpoints=[{"ref": "U3", "pin": "1"}]),
    ])
    monkeypatch.setattr(validation, "_pin_info_by_ref", lambda _: ({
        "U1": {"1": {"type": "output", "name": "OUT"}},
        "U2": {"1": {"type": "input", "name": "IN"}},
        "R1": {"1": {"type": "passive", "name": "~"}},
        "U3": {"1": {"type": "output", "name": "OUT"}},
    }, {}))
    check = validation.check_external_adc_driver_ownership(architecture, bom)
    assert not check.ok and len(check.offenders) == 1
    assert all(token in check.offenders[0] for token in ("terminal1.positive", "HALL_1", "U1.1"))
    bom.connections[0].endpoints.pop(0)
    assert validation.check_external_adc_driver_ownership(architecture, bom).ok


def test_public_resolver_refuses_onboard_producers_on_four_external_adc_nets():
    from kicraft.design.models import Architecture, CircuitRequirement
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution, RecipeResolutionError

    sheets = [
        {"name": name, "stem": name, "function": function}
        for name, function in (("CONNECTORS", "Sensor terminals"),
                               ("MCU", "Acquisition"), ("SENSORS", "Onboard sensors"))
    ]
    terminals = [
        CircuitRequirement(id=f"term{i}", sheet="CONNECTORS", role="connector",
                           family="screw-terminal",
                           ports={"positive": f"HALL_{i}", "negative": "GND"})
        for i in range(1, 5)
    ]
    mcu = CircuitRequirement(
        id="mcu", sheet="MCU", role="mcu_core", family="stm32g0",
        exact_part="STM32G030K6T6", interfaces=["adc"],
        ports={"vdd": "+3V3", "gnd": "GND",
               **{f"adc_{i - 1}": f"HALL_{i}" for i in range(1, 5)}},
    )
    sensors = [
        CircuitRequirement(id=f"sensor{i}", sheet="SENSORS", role="sensor",
                           family="hall-effect", exact_part="AH49ENTR-G1",
                           ports={"vdd": "+3V3", "gnd": "GND", "out": f"HALL_{i}"})
        for i in range(1, 5)
    ]
    nets = [{"name": f"HALL_{i}", "endpoints": [
        {"sheet": "CONNECTORS", "direction": "bidirectional"},
        {"sheet": "MCU", "direction": "input"},
        {"sheet": "SENSORS", "direction": "output"},
    ]} for i in range(1, 5)]
    architecture = Architecture(
        mcu_present=True, rail_voltages={"+3V3": 3.3, "GND": 0.0},
        power_nets=["+3V3", "GND"], sheets=sheets, inter_sheet_nets=nets,
        requirements=[*terminals, mcu, *sensors],
    )
    with pytest.raises(RecipeResolutionError) as exc:
        apply_architecture_recipe_resolution(architecture)
    conflicts = [row for row in exc.value.diagnostics
                 if row.code == "external_adc_driver_ownership_conflict"]
    assert {row.requirement_id for row in conflicts} == {f"term{i}" for i in range(1, 5)}
    external = architecture.model_copy(update={
        "sheets": architecture.sheets[:2],
        "requirements": [*terminals, mcu],
        "inter_sheet_nets": [
            net.model_copy(update={"endpoints": net.endpoints[:2]})
            for net in architecture.inter_sheet_nets
        ],
    })
    resolved = apply_architecture_recipe_resolution(external)
    assert {row.net for row in resolved.external_adc_contracts} == {f"HALL_{i}" for i in range(1, 5)}


def test_declared_buffer_input_does_not_drive_its_external_source_net():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.resolver import ResolutionResult, _external_adc_contracts

    external, definitions = _adc_ownership_fixture(typed_source=True)
    buffer = CircuitRequirement(
        id="buffer", sheet="BUFFER", role="sensor", family="custom-buffer",
        ports={"input": "HALL_1", "output": "BUFFER_OUT"},
        declared_interface={"ports": [
            {"key": "input", "direction": "input", "function": "buffer input"},
            {"key": "output", "direction": "output", "function": "buffer output"},
        ]},
    )
    result = ResolutionResult(requirements=[*external.requirements, buffer],
                              selections=external.selections)
    _external_adc_contracts(result, definitions)
    assert not result.blocking and len(result.external_adc_contracts) == 4
    buffer.ports["output"] = "HALL_1"
    conflict = ResolutionResult(requirements=[*external.requirements, buffer],
                                selections=external.selections)
    _external_adc_contracts(conflict, definitions)
    assert any(row.code == "external_adc_driver_ownership_conflict"
               and "requirement buffer.output" in row.evidence for row in conflict.blocking)
