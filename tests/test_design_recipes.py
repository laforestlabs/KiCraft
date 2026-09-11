import json
from pathlib import Path

from types import SimpleNamespace

import pytest
from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.leaf_geometry import repair_leaf_placement_legality
from kicraft.autoplacer.brain.leaf_routing import _place_fabricated_edge_interfaces
from kicraft.autoplacer.brain.types import BoardState, Component, Layer, Pad, Point

from kicraft.design.models import BOM, RecipeSelection
from kicraft.design.recipes import expand_recipe, expand_selections
from kicraft.design.recipes.pin_allocator import PinAllocationError, allocate_pins
from kicraft.design.recipes.registry import get_recipe, locked_pin_assignments
from kicraft.design.synthesis.validation import check_mcu_programming_access, check_net_coverage
from kicraft.server.stage_contracts import (
    StageSchemaError,
    _normalize_architecture_sheet_aliases,
    _normalize_stage_response,
)


def _selection(**parameters):
    return RecipeSelection(
        recipe="rp2040-minimal@1",
        instance="main",
        sheets={"mcu": "MCU", "io": "CASTELLATED IO"},
        parameters=parameters,
    )


def test_recipe_expansion_is_deterministic_and_provenanced():
    first = expand_recipe(_selection())
    second = expand_recipe(_selection())
    assert first == second
    assert all(part.recipe_id == "rp2040-minimal@1" for part in first.parts)
    assert len([part for part in first.parts if not part.assembly]) == 34
    assert [edge.side for edge in first.edge_interfaces] == ["left", "right"]

    assert len(first.connections) >= 35
    bom = BOM(
        parts=first.parts,
        connections=first.connections,
        no_connect_pins=first.no_connect_pins,
        edge_interfaces=first.edge_interfaces,
    )
    assert check_net_coverage(bom).ok
    assert check_mcu_programming_access(bom).ok


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


def test_recipe_rejects_unknown_version_parameter_and_sheet_role():
    with pytest.raises(ValueError, match="unknown circuit recipe"):
        expand_recipe(_selection().model_copy(update={"recipe": "rp2040-minimal@3"}))
    with pytest.raises(ValueError, match="unknown parameters"):
        expand_recipe(_selection(arbitrary_part="x"))
    with pytest.raises(ValueError, match="sheet roles mismatch"):
        expand_recipe(_selection().model_copy(update={"sheets": {"mcu": "MCU"}}))


def test_rp2040_v2_scopes_internal_nets_and_binds_power_ports():
    selection = RecipeSelection(
        recipe="rp2040-minimal@2",
        instance="main",
        sheets={"mcu": "MCU", "io": "IO"},
        port_bindings={"vdd": "+3V3", "gnd": "GND"},
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
    assert {"ref": "U1", "pin": "26"} in merged["no_connect_pins"]


def test_allocator_owned_wiring_pin_rejects_model_overwrite():
    from kicraft.design.models import RecipePinAllocation

    selection = RecipeSelection(
        recipe="esp32-s3-mini-1-minimal@1",
        instance="main",
        sheets={"mcu": "MCU"},
        parameters={"native_usb": False},
        port_bindings={"vdd": "+3V3", "gnd": "GND"},
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


def test_compact_architecture_range_expands_without_downstream_shape():
    payload = {
        "topologies": {},
        "rail_voltages": {},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "controller"},
            {"name": "CASTELLATED IO", "stem": "CASTELLATED_IO", "function": "edge IO"},
        ],
        "power_nets": [],
        "inter_sheet_nets": [],
        "inter_sheet_net_ranges": [
            {
                "name_pattern": "GPIO{n}",
                "start": 0,
                "end": 29,
                "endpoints": [
                    {"sheet": "MCU", "direction": "bidirectional"},
                    {"sheet": "CASTELLATED IO", "direction": "bidirectional"},
                ],
            }
        ],
    }
    canonical, expanded = _normalize_stage_response("architecture", payload, {})
    assert expanded == 30
    assert [net["name"] for net in canonical["inter_sheet_nets"]] == [f"GPIO{i}" for i in range(30)]
    assert "inter_sheet_net_ranges" not in canonical


def test_architecture_normalizes_sheet_stems_used_as_endpoint_names():
    payload = {
        "topologies": {},
        "rail_voltages": {},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {
                "name": "DIGITAL_INPUT_HEADER",
                "stem": "DIGITAL_INPUT_HEADER",
                "function": "logic input",
            },
            {"name": "R2R LADDER", "stem": "R2R_LADDER", "function": "DAC ladder"},
        ],
        "power_nets": [],
        "inter_sheet_nets": [
            {
                "name": "D0",
                "endpoints": [
                    {"sheet": "DIGITAL_INPUT_HEADER", "direction": "output"},
                    {"sheet": "R2R_LADDER", "direction": "input"},
                ],
            }
        ],
    }

    canonical, expanded = _normalize_stage_response("architecture", payload, {})

    assert expanded == 0
    assert [sheet["name"] for sheet in canonical["sheets"]] == [
        "DIGITAL INPUT HEADER",
        "R2R LADDER",
    ]
    assert [endpoint["sheet"] for endpoint in canonical["inter_sheet_nets"][0]["endpoints"]] == [
        "DIGITAL INPUT HEADER",
        "R2R LADDER",
    ]


def test_architecture_normalizes_sheet_identifier_case_and_spacing():
    payload = {
        "topologies": {},
        "rail_voltages": {},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {
                "name": "power_input_sheet",
                "stem": "power input sheet",
                "function": "power input",
            },
            {
                "name": "Sensor-Sheet",
                "stem": "sensor-sheet",
                "function": "sensor",
            },
        ],
        "power_nets": [],
        "inter_sheet_nets": [
            {
                "name": "POWER_OK",
                "endpoints": [
                    {"sheet": "POWER_INPUT_SHEET", "direction": "output"},
                    {"sheet": "sensor-sheet", "direction": "input"},
                ],
            }
        ],
    }

    canonical, expanded = _normalize_stage_response("architecture", payload, {})

    assert expanded == 0
    assert [(sheet["name"], sheet["stem"]) for sheet in canonical["sheets"]] == [
        ("POWER INPUT SHEET", "POWER_INPUT_SHEET"),
        ("SENSOR SHEET", "SENSOR_SHEET"),
    ]
    assert [endpoint["sheet"] for endpoint in canonical["inter_sheet_nets"][0]["endpoints"]] == [
        "POWER INPUT SHEET",
        "SENSOR SHEET",
    ]


def test_architecture_rejects_colliding_lossless_sheet_aliases():
    with pytest.raises(ValueError, match="duplicate canonical sheet name"):
        _normalize_architecture_sheet_aliases(
            {
                "sheets": [
                    {"name": "Power Input", "stem": "POWER_INPUT"},
                    {"name": "POWER_INPUT", "stem": "POWER_INPUT_2"},
                ]
            }
        )

    with pytest.raises(ValueError, match="ambiguously names"):
        _normalize_architecture_sheet_aliases(
            {
                "sheets": [
                    {"name": "POWER", "stem": "CONTROL"},
                    {"name": "CONTROL", "stem": "CONTROL_2"},
                ]
            }
        )


def test_architecture_normalizes_requirement_sheet_aliases():
    payload = {
        "topologies": {"crossover": "passive two-way crossover"},
        "rail_voltages": {},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [{"name": "crossover", "stem": "crossover", "function": "passive crossover"}],
        "power_nets": [],
        "inter_sheet_nets": [],
        "requirements": [
            {
                "id": "inductor_lowpass",
                "sheet": "crossover",
                "role": "analog_block",
                "family": "passive-crossover",
            }
        ],
    }

    canonical, _expanded = _normalize_stage_response("architecture", payload, {})

    assert canonical["sheets"][0]["name"] == "CROSSOVER"
    assert canonical["requirements"][0]["sheet"] == "CROSSOVER"


def test_architecture_auto_binds_connector_ports_from_sheet_interface():
    payload = {
        "topologies": {},
        "rail_voltages": {},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {"name": "SPI HEADER", "stem": "SPI_HEADER", "function": "spi header"},
            {"name": "AMPLIFIER", "stem": "AMPLIFIER", "function": "amp"},
            {"name": "TC INPUT", "stem": "TC_INPUT", "function": "screw terminal"},
        ],
        "power_nets": ["GND"],
        "inter_sheet_nets": [
            {
                "name": "SPI_MISO",
                "endpoints": [
                    {"sheet": "AMPLIFIER", "direction": "output"},
                    {"sheet": "SPI HEADER", "direction": "input"},
                ],
            },
            {
                "name": "SPI_SCK",
                "endpoints": [
                    {"sheet": "AMPLIFIER", "direction": "input"},
                    {"sheet": "SPI HEADER", "direction": "output"},
                ],
            },
            {
                "name": "SPI_CS",
                "endpoints": [
                    {"sheet": "AMPLIFIER", "direction": "input"},
                    {"sheet": "SPI HEADER", "direction": "output"},
                ],
            },
            {
                "name": "TC_P",
                "endpoints": [
                    {"sheet": "TC INPUT", "direction": "output"},
                    {"sheet": "AMPLIFIER", "direction": "input"},
                ],
            },
            {
                "name": "TC_N",
                "endpoints": [
                    {"sheet": "TC INPUT", "direction": "output"},
                    {"sheet": "AMPLIFIER", "direction": "input"},
                ],
            },
        ],
        "requirements": [
            {
                "id": "req_spi_header",
                "sheet": "SPI HEADER",
                "role": "connector",
                "family": "spi_header",
            },
            {
                "id": "req_screw",
                "sheet": "TC INPUT",
                "role": "connector",
                "family": "screw_terminal",
            },
            {
                "id": "req_amp",
                "sheet": "AMPLIFIER",
                "role": "sensor",
                "family": "thermocouple_converter",
                "exact_part": "MAX31855",
            },
        ],
    }

    canonical, _expanded = _normalize_stage_response("architecture", payload, {})

    by_id = {req["id"]: req for req in canonical["requirements"]}
    assert list(by_id["req_spi_header"]["ports"].values()) == [
        "SPI_MISO",
        "SPI_SCK",
        "SPI_CS",
        "GND",
    ]
    assert list(by_id["req_screw"]["ports"].values()) == ["TC_P", "TC_N"]


def test_architecture_normalizes_recipe_selection_sheet_aliases():
    normalized = _normalize_architecture_sheet_aliases(
        {
            "sheets": [{"name": "controller", "stem": "controller", "function": "controller"}],
            "recipe_selections": [
                {
                    "recipe": "example@1",
                    "instance": "main",
                    "sheets": {"mcu": "Controller"},
                }
            ],
        }
    )

    assert normalized["recipe_selections"][0]["sheets"] == {"mcu": "CONTROLLER"}


def test_architecture_derives_missing_or_empty_sheet_stems():
    for raw_stem in (None, ""):
        sheet = {"name": "sensor interface", "function": "sensor"}
        if raw_stem is not None:
            sheet["stem"] = raw_stem
        normalized = _normalize_architecture_sheet_aliases({"sheets": [sheet]})
        assert normalized["sheets"][0] == {
            "name": "SENSOR INTERFACE",
            "stem": "SENSOR_INTERFACE",
            "function": "sensor",
        }


def _range_architecture_payload(explicit_nets, ranges):
    return {
        "topologies": {},
        "rail_voltages": {},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {"name": "MOTOR 1", "stem": "MOTOR1", "function": "motor stage"},
            {"name": "MOTOR 2", "stem": "MOTOR2", "function": "motor stage"},
            {"name": "MCU", "stem": "MCU", "function": "controller"},
        ],
        "power_nets": [],
        "inter_sheet_nets": explicit_nets,
        "inter_sheet_net_ranges": ranges,
    }


_MOTOR1_MCU = [
    {"sheet": "MOTOR 1", "direction": "bidirectional"},
    {"sheet": "MCU", "direction": "bidirectional"},
]


def test_compact_architecture_range_deduplicates_identical_explicit_net():
    # MOTOR1_A is declared explicitly and again through MOTOR{n}_A 1..2 with the
    # same endpoints in reversed order: endpoint order is not semantically
    # meaningful, so the redundant range expansion is dropped losslessly.
    payload = _range_architecture_payload(
        explicit_nets=[{"name": "MOTOR1_A", "endpoints": _MOTOR1_MCU}],
        ranges=[
            {
                "name_pattern": "MOTOR{n}_A",
                "start": 1,
                "end": 2,
                "endpoints": list(reversed(_MOTOR1_MCU)),
            }
        ],
    )
    canonical, expanded = _normalize_stage_response("architecture", payload, {})
    names = [net["name"] for net in canonical["inter_sheet_nets"]]
    assert names.count("MOTOR1_A") == 1
    assert names == ["MOTOR1_A", "MOTOR2_A"]
    assert expanded == 1
    assert "inter_sheet_net_ranges" not in canonical


def test_compact_architecture_range_rejects_conflicting_explicit_net():
    # Same generated name exists explicitly with different endpoint semantics.
    payload = _range_architecture_payload(
        explicit_nets=[
            {
                "name": "MOTOR1_A",
                "endpoints": [
                    {"sheet": "MOTOR 1", "direction": "output"},
                    {"sheet": "MCU", "direction": "input"},
                ],
            }
        ],
        ranges=[
            {
                "name_pattern": "MOTOR{n}_A",
                "start": 1,
                "end": 2,
                "endpoints": _MOTOR1_MCU,
            }
        ],
    )
    with pytest.raises(StageSchemaError, match="duplicate/overlapping inter-sheet net 'MOTOR1_A'"):
        _normalize_stage_response("architecture", payload, {})

    # Same generated name exists explicitly over a different sheet set.
    payload = _range_architecture_payload(
        explicit_nets=[
            {
                "name": "MOTOR1_A",
                "endpoints": [
                    {"sheet": "MOTOR 2", "direction": "bidirectional"},
                    {"sheet": "MCU", "direction": "bidirectional"},
                ],
            }
        ],
        ranges=[
            {
                "name_pattern": "MOTOR{n}_A",
                "start": 1,
                "end": 2,
                "endpoints": _MOTOR1_MCU,
            }
        ],
    )
    with pytest.raises(StageSchemaError, match="duplicate/overlapping inter-sheet net 'MOTOR1_A'"):
        _normalize_stage_response("architecture", payload, {})

    # A second range covering an already generated name stays rejected even
    # when both ranges carry identical endpoints.
    payload = _range_architecture_payload(
        explicit_nets=[],
        ranges=[
            {"name_pattern": "GPIO{n}", "start": 0, "end": 4, "endpoints": _MOTOR1_MCU},
            {"name_pattern": "GPIO{n}", "start": 3, "end": 9, "endpoints": _MOTOR1_MCU},
        ],
    )
    with pytest.raises(StageSchemaError, match="duplicate/overlapping inter-sheet net 'GPIO3'"):
        _normalize_stage_response("architecture", payload, {})


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
    canonical, _expanded = _normalize_stage_response(
        "architecture",
        _esp32_architecture_payload(),
        {"intent": {"named_parts": ["ESP32-S3-MINI-1-N8"]}},
    )
    assert canonical["requirements"][0]["exact_part"] == "ESP32-S3-MINI-1-N8"
    selection = canonical["recipe_selections"][0]
    assert selection["recipe"] == "esp32-s3-mini-1-minimal@1"
    assert selection["port_bindings"] == {"gnd": "GND", "vdd": "+3V3"}
    assert canonical["recipe_resolution"][0]["recipe"] == selection["recipe"]


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


def test_usb_c_breakout_pin_ports_stay_model_owned():
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

    canonical, _expanded = _normalize_stage_response("architecture", payload, {})

    assert canonical["recipe_selections"] == []
    assert canonical["unresolved_requirement_ids"] == ["usb_receptacle"]


def test_usb_c_sheet_with_exposed_cc_signal_avoids_sink_recipe():
    payload = {
        "topologies": {"USB C": "USB-C receptacle breakout", "HEADER": "pin header"},
        "rail_voltages": {"VBUS": 5.0},
        "comms_protocols": ["USB"],
        "mcu_present": False,
        "sheets": [
            {"name": "USB C", "stem": "USB_C", "function": "USB-C receptacle breakout"},
            {"name": "HEADER", "stem": "HEADER", "function": "pin header"},
        ],
        "power_nets": ["VBUS", "GND"],
        "inter_sheet_nets": [
            {
                "name": "CC1",
                "endpoints": [
                    {"sheet": "USB C", "direction": "output"},
                    {"sheet": "HEADER", "direction": "input"},
                ],
            }
        ],
        "assumptions": [],
    }

    canonical, _expanded = _normalize_stage_response("architecture", payload, {})

    assert canonical["recipe_selections"] == []
    requirement = canonical["requirements"][0]
    assert requirement["family"] == "usb-c-breakout"
    assert requirement["ports"]["cc1"] == "CC1"


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


def test_architecture_discards_unknown_provider_unresolved_ids_before_resolution():
    payload = _esp32_architecture_payload()
    payload["unresolved_requirement_ids"] = ["req_mcu", "req_power"]

    canonical, _expanded = _normalize_stage_response("architecture", payload, {})

    assert canonical["unresolved_requirement_ids"] == []
    assert canonical["requirements"][0]["exact_part"] == "ESP32-S3-MINI-1-N8"


def test_architecture_family_default_records_typed_exact_part():
    canonical, _expanded = _normalize_stage_response(
        "architecture",
        _esp32_architecture_payload(),
        {"intent": {}},
    )

    assert canonical["requirements"][0]["family"] == "esp32-s3-module"
    assert canonical["requirements"][0]["exact_part"] == "ESP32-S3-MINI-1-N8"


def test_architecture_unsupported_protected_esp32_variant_blocks():
    with pytest.raises(
        StageSchemaError,
        match="unsupported_protected_variant",
    ):
        _normalize_stage_response(
            "architecture",
            _esp32_architecture_payload(),
            {"intent": {"named_parts": ["ESP32-S3-WROOM-1-N16R8"]}},
        )


def _typed_esp32_architecture(**requirement_overrides):
    payload = _esp32_architecture_payload()
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
    return payload


@pytest.mark.parametrize(
    ("dm", "dp"),
    [("USB_D-", "USB_D+"), ("D-", "D+"), ("USB_D_N", "USB_D_P")],
)
def test_recipe_usb_aliases_preserve_polarity_and_supply_sign(dm, dp):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(ports={}, interfaces=["usb_device"])
    payload["power_nets"] = ["-3V3", "+3V3", "GND"]
    payload["sheets"].append({"name": "USB", "stem": "USB", "function": "USB connector"})
    payload["inter_sheet_nets"] = [
        {
            "name": name,
            "endpoints": [
                {"sheet": "USB", "direction": "bidirectional"},
                {"sheet": "MCU", "direction": "bidirectional"},
            ],
        }
        for name in (dp, dm)
    ]

    result = resolve_architecture_recipes(payload)

    assert not result.blocking
    assert result.selections[0].port_bindings == {
        "vdd": "+3V3",
        "gnd": "GND",
        "usb_dm": dm,
        "usb_dp": dp,
    }


def test_typed_usb_connector_peers_complete_distinct_polarity_contracts():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(interfaces=["usb_device"])
    payload["sheets"].append({"name": "USB", "stem": "USB", "function": "USB connector"})
    payload["requirements"].append(
        {
            "id": "connector",
            "sheet": "USB",
            "role": "connector",
            "family": "application-connector",
            "ports": {"D+": "HOST_POSITIVE", "D-": "HOST_NEGATIVE"},
        }
    )

    result = resolve_architecture_recipes(payload)

    assert not result.blocking
    assert result.selections[0].port_bindings["usb_dp"] == "HOST_POSITIVE"
    assert result.selections[0].port_bindings["usb_dm"] == "HOST_NEGATIVE"
    nets = {net.name: net for net in result.completed_nets}
    assert set(nets) == {"HOST_POSITIVE", "HOST_NEGATIVE"}
    assert all(
        {endpoint.sheet for endpoint in net.endpoints} == {"MCU", "USB"} for net in nets.values()
    )


def _c3_remote_usb_architecture():
    payload = _typed_esp32_architecture(
        family="esp32-c3-mini-1-module",
        exact_part="ESP32-C3-MINI-1-N4",
        parameters={"native_usb": False},
    )
    payload["sheets"][0]["function"] = "ESP32-C3 MCU"
    payload["topologies"] = {"MCU": "ESP32-C3"}
    payload["sheets"].extend(
        [
            {"name": "USB", "stem": "USB", "function": "USB connector"},
            {"name": "BRIDGE", "stem": "BRIDGE", "function": "external UART bridge"},
        ]
    )
    payload["requirements"].extend(
        [
            {
                "id": "connector",
                "sheet": "USB",
                "role": "connector",
                "family": "application-connector",
                "ports": {"usb_dp": "USB_D_P", "usb_dm": "USB_D_N"},
            },
            {
                "id": "bridge",
                "sheet": "BRIDGE",
                "role": "bus_interface",
                "family": "external-uart-bridge",
                "ports": {"usb_dp": "USB_D_P", "usb_dm": "USB_D_N"},
            },
        ]
    )
    payload["inter_sheet_nets"] = [
        {
            "name": net,
            "endpoints": [
                {"sheet": "USB", "direction": "bidirectional"},
                {"sheet": "BRIDGE", "direction": "bidirectional"},
            ],
        }
        for net in ("USB_D_P", "USB_D_N")
    ]
    return payload


@pytest.mark.parametrize("declared_contract", [False, True])
def test_remote_usb_connector_does_not_enable_or_wire_c3_native_usb(declared_contract):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_remote_usb_architecture()
    if not declared_contract:
        payload["inter_sheet_nets"] = []
    result = resolve_architecture_recipes(payload)

    assert not result.blocking
    core = next(row for row in result.selections if row.requirement_ids == ["mcu_core"])
    assert core.parameters["native_usb"] is False
    assert core.port_bindings == {"vdd": "+3V3", "gnd": "GND"}
    expansion = expand_recipe(core)
    assert not {"USB_D_P", "USB_D_N"} & {
        connection.net_name for connection in expansion.connections
    }
    assert not result.completed_nets


def test_remote_usb_does_not_hide_missing_c3_uart_application_contract():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_remote_usb_architecture()
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
    assert not result.completed_nets


def test_explicit_remote_usb_binding_still_requires_mcu_ownership():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_remote_usb_architecture()
    payload["requirements"][0]["ports"].update(usb_dp="USB_D_P", usb_dm="USB_D_N")
    result = resolve_architecture_recipes(payload)

    diagnostic = next(row for row in result.blocking if row.requirement_id == "mcu_core")
    assert diagnostic.code == "missing_recipe_port_contract"
    assert {"USB_D_P", "USB_D_N"} <= set(diagnostic.evidence)
    assert not result.completed_nets


def test_explicit_unknown_usb_binding_is_not_replaced_by_owned_alias():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_remote_usb_architecture()
    payload["requirements"][0]["ports"].update(usb_dp="MISSPELLED_DP", usb_dm="MISSPELLED_DM")
    for net in payload["inter_sheet_nets"]:
        net["endpoints"].append({"sheet": "MCU", "direction": "bidirectional"})
    result = resolve_architecture_recipes(payload)

    diagnostic = next(row for row in result.blocking if row.requirement_id == "mcu_core")
    assert diagnostic.code == "unknown_recipe_port_net"
    assert {"MISSPELLED_DP", "MISSPELLED_DM"} <= set(diagnostic.evidence)
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
        original = next(
            row for row in recipes if row.definition.family == "usb-c-usb2-device"
        )
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
        row.code == "unsupported_protected_variant"
        and row.requirement_id == "usb_c_connector"
        for row in result.blocking
    )
    assert not any(
        row.requirement_ids == ["usb_c_connector"] for row in result.selections
    )


@pytest.mark.parametrize(
    ("port", "net", "code"),
    [
        ("usb_dp", "GND", "recipe_signal_in_power_nets"),
        ("shield", "VBUS", "recipe_signal_in_power_nets"),
        ("shield", "UNDECLARED_SHIELD", "unknown_recipe_port_net"),
        ("shield", "REMOTE_SHIELD", "missing_recipe_port_contract"),
    ],
)
def test_groundable_shield_preserves_signal_power_and_ownership_errors(port, net, code):
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _grounded_usb_shield_architecture()
    payload["requirements"][1]["ports"][port] = net
    if net == "REMOTE_SHIELD":
        payload["sheets"].append({"name": "CASE", "stem": "CASE", "function": "case"})
        payload["inter_sheet_nets"].append(
            {
                "name": net,
                "endpoints": [
                    {"sheet": "MCU", "direction": "passive"},
                    {"sheet": "CASE", "direction": "passive"},
                ],
            }
        )
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
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(
        family="attiny1614",
        exact_part="ATTINY1614-SSNR",
        ports={},
    )
    payload["topologies"] = {"MCU": "ATtiny1614", "POWER": "Direct CR2032 supply"}
    payload["sheets"][0]["function"] = "ATtiny1614 microcontroller"
    payload["rail_voltages"] = {"VBAT": 3.0}
    payload["power_nets"] = ["VBAT", "GND"]
    payload["comms_protocols"] = ["UPDI"]

    rejected = resolve_architecture_recipes(payload)
    diagnostic = next(row for row in rejected.blocking if row.code == "missing_recipe_port")
    assert diagnostic.requirement_id == "mcu_core"
    assert diagnostic.recipe == "attiny1614-updi-minimal@1"
    assert "vdd" in diagnostic.evidence
    assert not rejected.selections

    payload["requirements"][0]["ports"] = {"vdd": "VBAT", "gnd": "GND"}
    resolved = resolve_architecture_recipes(payload)
    assert not resolved.blocking
    assert resolved.selections[0].port_bindings == {"vdd": "VBAT", "gnd": "GND"}


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
    diagnostic = next(row for row in result.blocking if row.code == "missing_recipe_port")
    assert diagnostic.requirement_id == requirement.id
    assert "vdd" in diagnostic.evidence
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
        diagnostic = next(row for row in result.blocking if row.requirement_id == "phy")
        assert diagnostic.code == "missing_recipe_port"
        assert {"tx", "rx"} <= set(diagnostic.evidence)
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
    assert diagnostic.code == "unknown_recipe_port_net"
    assert "STBY" in diagnostic.evidence
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
                "parameters": {"rows": 3},
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
    assert not result.selections
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
    mcu = next(part for part in expansion.parts if part.mpn == "ATTINY402-SSN")
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
    assert {net["name"] for net in payload["inter_sheet_nets"]} == {"GPIO1", "GPIO99"}

    payload["inter_sheet_nets"] = payload["inter_sheet_nets"][:1]
    canonical, _expanded = _normalize_stage_response("architecture", payload, {})
    assert {net["name"] for net in canonical["inter_sheet_nets"]} == {"GPIO1"}
    assert canonical["recipe_selections"][0]["pin_allocations"][0]["net"] == "GPIO1"


def _frozen_architecture_recovery():
    return json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "stage_reliability"
            / "architecture_recovery_20260911.json"
        ).read_text()
    )


def test_frozen_can_missing_connector_ports_fails_at_owning_requirement():
    frozen = _frozen_architecture_recovery()["can"]
    candidate = frozen["candidates"][0]
    with pytest.raises(StageSchemaError, match="missing_recipe_port") as rejected:
        _normalize_stage_response("architecture", candidate, {"intent": frozen["intent"]})
    assert rejected.value.diagnostic["requirement_id"] == "can_transceiver"
    assert rejected.value.diagnostic["recipe"] == "sn65hvd230-can-node@1"
    assert rejected.value.diagnostic["sheet"] == "CAN TRANSCEIVER"
    assert {"canh", "canl"} <= set(rejected.value.diagnostic["evidence"])
    assert not {"CANH", "CANL"} & {net["name"] for net in candidate["inter_sheet_nets"]}


def test_typed_can_connector_does_not_hide_other_unowned_endpoints():
    frozen = _frozen_architecture_recovery()["can"]
    candidate = frozen["candidates"][0]
    connector = next(row for row in candidate["requirements"] if row["id"] == "can_connector")
    connector["ports"] = {"canh": "TRUNK_HIGH", "canl": "TRUNK_LOW"}
    with pytest.raises(StageSchemaError, match="unsupported_recipe_endpoint") as rejected:
        _normalize_stage_response("architecture", candidate, {"intent": frozen["intent"]})
    assert rejected.value.diagnostic["sheet"] == "CAN TRANSCEIVER"
    assert {"CAN_TX_0", "CAN_TX_1", "CAN_RX_0", "CAN_RX_1"} <= set(
        rejected.value.diagnostic["evidence"]
    )


def test_typed_peer_conflict_does_not_choose_a_connector_bus():
    frozen = _frozen_architecture_recovery()["can"]
    candidate = frozen["candidates"][0]
    connector = next(row for row in candidate["requirements"] if row["id"] == "can_connector")
    connector["ports"] = {"canh": "BUS_A", "canl": "BUS_LOW"}
    candidate["requirements"].append(
        {
            "id": "second_bus",
            "sheet": "CAN TERMINATION",
            "role": "connector",
            "family": "db9",
            "ports": {"canh": "BUS_B"},
        }
    )
    with pytest.raises(StageSchemaError, match="missing_recipe_port"):
        _normalize_stage_response("architecture", candidate, {"intent": frozen["intent"]})
    assert not {"BUS_A", "BUS_B"} & {net["name"] for net in candidate["inter_sheet_nets"]}


def test_frozen_can_power_misclassification_is_not_an_mcu_recipe():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    frozen = _frozen_architecture_recovery()["can"]
    result = resolve_architecture_recipes(frozen["candidates"][1], frozen["intent"])
    diagnostic = next(
        row
        for row in result.blocking
        if row.sheet == "CAN TRANSCEIVER" and row.code == "missing_recipe_port"
    )
    assert diagnostic.recipe == "sn65hvd230-can-node@1"
    assert {"canh", "canl"} <= set(diagnostic.evidence)
    assert not result.selections


def test_frozen_c3_connector_requirement_cannot_hide_missing_mcu_ownership():
    frozen = _frozen_architecture_recovery()["c3"]
    with pytest.raises(StageSchemaError, match="unavailable_recipe_gpio") as rejected:
        _normalize_stage_response(
            "architecture", frozen["architecture"], {"intent": frozen["intent"]}
        )
    assert rejected.value.diagnostic["recipe"] == "esp32-c3-mini-1-minimal@1"
    assert rejected.value.diagnostic["sheet"] == "ESP32 C3 MODULE"
    assert "GPIO9" in rejected.value.diagnostic["evidence"]


def test_typed_named_mcu_cannot_be_dropped_by_false_presence_flag():
    payload = _esp32_architecture_payload()
    payload["mcu_present"] = False
    payload["requirements"] = [
        {
            "id": "application_connector",
            "sheet": "MCU",
            "role": "connector",
            "family": "application-connector",
            "ports": {},
        }
    ]
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

    payload = _typed_esp32_architecture(
        family="unregistered-controller", exact_part=candidate
    )
    payload["topologies"] = {}
    payload["sheets"][0]["function"] = "Application processor"
    result = resolve_architecture_recipes(payload, {"named_parts": [requested]})
    ownership_errors = [
        row for row in result.blocking if row.code == "missing_recipe_requirement"
    ]
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
    ownership_errors = [
        row for row in result.blocking if row.code == "missing_recipe_requirement"
    ]
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


def test_recipe_resolver_requires_conditional_usb_ports_and_declared_nets():
    with pytest.raises(StageSchemaError, match="missing_recipe_port"):
        _normalize_stage_response(
            "architecture",
            _typed_esp32_architecture(parameters={"native_usb": True}),
            {},
        )
    with pytest.raises(StageSchemaError, match="unknown_recipe_port_net"):
        _normalize_stage_response(
            "architecture",
            _typed_esp32_architecture(
                ports={
                    "vdd": "+3V3",
                    "gnd": "GND",
                    "usb_dm": "UNDECLARED_DM",
                    "usb_dp": "UNDECLARED_DP",
                }
            ),
            {},
        )


def test_esp32_optional_usb_and_internal_nets_are_instance_scoped():
    base = RecipeSelection(
        recipe="esp32-s3-mini-1-minimal@1",
        instance="main",
        sheets={"mcu": "MCU"},
        parameters={"native_usb": False},
        port_bindings={"vdd": "+3V3", "gnd": "GND"},
        requirement_ids=["mcu_core"],
    )
    without_usb = expand_recipe(base)
    with_usb = expand_recipe(
        base.model_copy(
            update={
                "parameters": {"native_usb": True},
                "port_bindings": {
                    "vdd": "+3V3",
                    "gnd": "GND",
                    "usb_dm": "USB_DM",
                    "usb_dp": "USB_DP",
                },
            }
        )
    )
    second = expand_recipe(
        base.model_copy(update={"instance": "aux", "requirement_ids": ["aux_core"]})
    )
    assert len(with_usb.parts) == len(without_usb.parts) + 2
    assert set(without_usb.ownership.internal_nets).isdisjoint(second.ownership.internal_nets)
    assert {pin.pin for pin in without_usb.no_connect_pins} >= {"23", "24"}
    assert {connection.net_name for connection in with_usb.connections} >= {
        "USB_DM",
        "USB_DP",
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


def test_s3_prefixed_i2c_bindings_own_real_intersheet_nets():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(
        interfaces=["i2c_controller"],
        ports={"vdd": "+3V3", "gnd": "GND", "i2c_sda": "I2C_SDA", "i2c_scl": "I2C_SCL"},
    )
    payload["sheets"].append({"name": "DAQ", "stem": "DAQ", "function": "I2C acquisition"})
    payload["inter_sheet_nets"] = [
        {
            "name": name,
            "endpoints": [
                {"sheet": "MCU", "direction": "bidirectional"},
                {"sheet": "DAQ", "direction": "bidirectional"},
            ],
        }
        for name in ("I2C_SDA", "I2C_SCL")
    ]
    payload["requirements"].append({
        "id": "daq", "sheet": "DAQ", "role": "sensor", "family": "custom-acquisition",
        "ports": {"sda": "I2C_SDA", "scl": "I2C_SCL"},
    })
    result = resolve_architecture_recipes(payload)
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
    assert not result.selections


def test_advertised_interface_keys_allocate_their_bound_nets():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins
    from kicraft.design.recipes.registry import recipe_summaries

    definition = get_recipe("esp32-s3-mini-1-minimal@1")
    summary = next(row for row in recipe_summaries() if row["recipe"] == definition.recipe)
    for interface, members in summary["interfaces"].items():
        requirement = CircuitRequirement(
            id="core", sheet="MCU", role="mcu_core", family=definition.family,
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

    definition = get_recipe("esp32-c3-mini-1-minimal@1")
    requirement = CircuitRequirement(
        id="core", sheet="MCU", role="mcu_core", family=definition.family,
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
    incomplete = requirement.model_copy(
        update={"ports": {**requirement.ports, "tx": "APP_TX"}}
    )
    with pytest.raises(PinAllocationError, match="missing_interface_port"):
        allocate_requirement_pins(definition, incomplete)


def test_pin_allocator_honors_explicit_gpio_number_ports():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.recipes.pin_allocator import allocate_requirement_pins

    definition = get_recipe("esp32-c3-mini-1-minimal@1")
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


@pytest.mark.parametrize(
    "recipe",
    [
        "esp32-s3-wroom-1-minimal@1",
        "esp32-wroom-32e-minimal@1",
        "esp32-c3-mini-1-minimal@1",
        "stm32f103c8t6-minimal@1",
        "attiny402-updi-minimal@1",
        "attiny412-updi-minimal@1",
        "attiny1614-updi-minimal@1",
        "ch32v003j4m6-minimal@1",
    ],
)
def test_wave_a_mcu_recipes_expand_with_complete_ownership(recipe):
    definition = get_recipe(recipe)
    expansion = expand_recipe(
        RecipeSelection(
            recipe=recipe,
            instance="main",
            sheets={"mcu": "MCU"},
            port_bindings={"vdd": "+3V3", "gnd": "GND"},
        )
    )
    assert definition.maturity == "production"
    assert definition.exact_part
    assert definition.source_documents
    assert definition.electrical_assertions
    assert all(part.resolution_source == "recipe" for part in expansion.parts)
    owned_pins = {(pin.ref, pin.pin) for pin in expansion.ownership.pins}
    assert len(owned_pins) == len(expansion.ownership.pins)
    assert any(part.ref.startswith("J") for part in expansion.parts)
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


def test_wave_b_power_recipes_own_every_symbol_pin():
    from pathlib import Path

    from kicraft.design.recipes.wave_b_power import WAVE_B_POWER_RECIPES
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    for definition in WAVE_B_POWER_RECIPES:
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
        assert expansion.ownership.internal_nets == []
        assert set(nets.values()) == set(bindings.values())
    else:
        assert v3 not in bindings.values()
        assert expansion.ownership.internal_nets == [v3]
        assert {endpoint for endpoint, net in nets.items() if net == v3} == {
            (bridge.ref, "4"),
            (parts["v3_cap"].ref, "1"),
        }
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
                    "vdd": "BRIDGE_SUPPLY",
                    "gnd": "GND",
                    "usb_dp": "USB_DP",
                    "usb_dm": "USB_DM",
                    "tx": "HOST_TX",
                    "rx": "HOST_RX",
                },
            )
        )


def test_expanded_recipe_wave_assets_resolve():
    from pathlib import Path

    from kicraft.design.recipes.wave_a_mcus import WAVE_A_MCU_RECIPES
    from kicraft.design.recipes.wave_b_power import WAVE_B_POWER_RECIPES
    from kicraft.design.recipes.wave_c_interfaces import WAVE_C_INTERFACE_RECIPES
    from kicraft.design.synthesis.footprint_library import lookup_footprint
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    for definition in (
        *WAVE_A_MCU_RECIPES,
        *WAVE_B_POWER_RECIPES,
        *WAVE_C_INTERFACE_RECIPES,
    ):
        for group in definition.parts:
            lookup_pins(group.symbol, project_root=Path("."))
            lookup_footprint(group.footprint, project_root=Path("."))


def test_promoted_recipes_are_advertised_and_protected_in_production():
    from kicraft.design.recipes.registry import (
        protected_identity_matches,
        recipe_summaries,
    )

    production_ids = {row["recipe"] for row in recipe_summaries()}
    assert "esp32-s3-mini-1-minimal@1" in production_ids
    assert "tp4056-1s-charger@1" in production_ids
    assert protected_identity_matches("TP4056") == ("tp4056",)
    assert recipe_summaries(frozenset({"canary"})) == []


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
    assert any(
        row.code == "missing_recipe_port" and row.requirement_id == core.id
        for row in result.blocking
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
        family="esp32-c3-mini-1-module", exact_part="ESP32-C3-MINI-1-N4", ports={}
    )
    payload["sheets"][0]["function"] = "ESP32-C3 MCU"
    payload["topologies"] = {"MCU": "ESP32-C3"}
    payload["rail_voltages"] = {"VCC_3V3": 3.3, "VCC_5V": 5.0}
    payload["power_nets"] = ["VCC_3V3", "VCC_5V", "GND"]
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
    assert any(
        row.code == "missing_recipe_port" and "vdd" in row.evidence for row in result.blocking
    )
    assert not result.selections


def test_supply_completion_does_not_harvest_voltage_substrings_from_signals():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _typed_esp32_architecture(ports={})
    payload["rail_voltages"] = {}
    payload["power_nets"] = ["ENABLE_VDD", "SENSE_3V3", "GND"]
    result = resolve_architecture_recipes(payload)
    assert any(
        row.code == "missing_recipe_port" and "vdd" in row.evidence for row in result.blocking
    )
    assert not result.selections


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
        family="esp32-c3-mini-1-module", exact_part="ESP32-C3-MINI-1-N4", ports={}
    )
    payload["sheets"][0]["function"] = "ESP32-C3 MCU"
    payload["topologies"] = {"MCU": "ESP32-C3"}
    payload["rail_voltages"] = {"CORE_RAIL": 3.3, "OTHER_RAIL": 3.3}
    payload["power_nets"] = ["CORE_RAIL", "OTHER_RAIL", "GND"]
    ambiguous = resolve_architecture_recipes(payload)
    assert any(
        row.code == "missing_recipe_port" and "vdd" in row.evidence for row in ambiguous.blocking
    )
    assert not ambiguous.selections

    payload["sheets"].append({"name": "POWER", "stem": "POWER", "function": "power supply"})
    payload["inter_sheet_nets"] = [
        {
            "name": "CORE_RAIL",
            "endpoints": [
                {"sheet": "POWER", "direction": "output"},
                {"sheet": "MCU", "direction": "input"},
            ],
        }
    ]
    bound = resolve_architecture_recipes(payload)
    assert not bound.blocking
    assert bound.requirements[0].ports["vdd"] == "CORE_RAIL"

    payload["rail_voltages"] = {"CORE_RAIL": 5.0, "OTHER_RAIL": 5.0}
    incompatible = resolve_architecture_recipes(payload)
    assert any(
        row.code == "missing_recipe_port" and "vdd" in row.evidence for row in incompatible.blocking
    )
    assert not incompatible.selections


def _c3_typed_programming_architecture(*, handshakes=True):
    payload = _c3_remote_usb_architecture()
    bridge = payload["requirements"][2]
    bridge.update(family="usb-uart-bridge", exact_part="CH340C")
    bridge["ports"].update(vdd="+3V3", gnd="GND", tx="HOST_TO_MCU", rx="MCU_TO_HOST")
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


def test_direct_ch340c_power_domain_stays_blocked_after_uart_repair():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    payload = _c3_typed_programming_architecture()
    core, bridge = payload["requirements"][0], payload["requirements"][2]
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


def test_ch340c_supply_requires_evidence_and_honors_authoritative_rail_voltage():
    from kicraft.design.recipes.resolver import resolve_architecture_recipes

    ports = {"vdd": "VBUS", "gnd": "GND", "usb_dp": "DP", "usb_dm": "DM", "tx": "TX", "rx": "RX"}
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
    payload["power_nets"] = ["+5V", "GND"]
    payload["rail_voltages"] = {"+5V": 3.3}
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
    core, bridge = payload["requirements"][0], payload["requirements"][2]
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
    payload["requirements"][2]["role"] = peer_role
    if explicit_contract:
        core = payload["requirements"][0]
        bridge = payload["requirements"][2]
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
    assert core.parameters["native_usb"] is False
    assert {row.net for row in core.pin_allocations} == {f"GPIO{gpio}" for gpio in (*range(9), 10)}
    assert not {row.pin for row in core.pin_allocations} & {"8", "23", "26", "27", "30", "31"}
    expansion = expand_recipe(core)
    refs = {part.recipe_role: part.ref for part in expansion.parts}
    nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }
    assert nets[refs["mcu"], "30"] == nets[refs["program_header"], "4"] == "HOST_TO_MCU"
    assert nets[refs["mcu"], "31"] == nets[refs["program_header"], "3"] == "MCU_TO_HOST"
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
    assert bridge_nets["2"] == nets[refs["mcu"], "30"] == "HOST_TO_MCU"
    assert bridge_nets["3"] == nets[refs["mcu"], "31"] == "MCU_TO_HOST"
    assert (
        nets[refs["mcu"], "8"]
        == nets[refs["reset_button"], "1"]
        == nets[refs["program_header"], "5"]
    )
    assert (
        nets[refs["mcu"], "23"]
        == nets[refs["boot_button"], "1"]
        == nets[refs["program_header"], "6"]
    )
    assert {(pin.ref, pin.pin) for pin in expansion.no_connect_pins}.isdisjoint(nets)
    ownership = [(pin.ref, pin.pin) for pin in expansion.ownership.pins]
    assert len(ownership) == len(set(ownership))
    if handshakes:
        assert (
            nets[refs["auto_reset_boot_high_pullup_22"], "2"] == nets[refs["mcu"], "22"] == "GPIO8"
        )
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
            recipe="esp32-c3-mini-1-minimal@1",
            instance="auto",
            sheets={"mcu": "MCU"},
            parameters={"auto_reset": True},
            port_bindings={
                "vdd": "+3V3",
                "gnd": "GND",
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
    assert nets[parts["auto_reset_boot_high_pullup_22"].ref, "1"] == "+3V3"
    assert nets[parts["auto_reset_boot_high_pullup_22"].ref, "2"] == nets[mcu, "22"]
    assert (mcu, "22") not in {(pin.ref, pin.pin) for pin in expansion.no_connect_pins}
    ownership = [(pin.ref, pin.pin) for pin in expansion.ownership.pins]
    assert len(ownership) == len(set(ownership))


def test_unused_fixed_programming_ports_are_private_and_add_no_support_parts():
    selections = [
        RecipeSelection(
            recipe="esp32-c3-mini-1-minimal@1",
            instance=instance,
            sheets={"mcu": "MCU"},
            port_bindings={"vdd": "+3V3", "gnd": "GND"},
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
                    endpoint.ref == mcu and endpoint.pin in {"8", "23", "30", "31"}
                    for endpoint in connection.endpoints
                )
            }
        )
        assert not any(part.recipe_role.startswith("auto_reset") for part in expansion.parts)
        assert not {"dtr_n", "rts_n"} & {
            connection.net_name for connection in expansion.connections
        }
    assert len(private[0]) == len(private[1]) == 4
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
    bridge = payload["requirements"][2]
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
    payload["requirements"][2]["role"] = peer_role
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


def test_c3_uart_recovery_still_rejects_gpio_range_including_fixed_boot_pin():
    from kicraft.server.stage_contracts import StageSchemaError

    payload = _c3_typed_programming_architecture()
    for requirement in (payload["requirements"][0], payload["requirements"][2]):
        requirement["interfaces"] = ["uart"]
    payload["inter_sheet_nets"] = [
        net for net in payload["inter_sheet_nets"] if not net["name"].startswith("GPIO")
    ]
    payload["inter_sheet_net_ranges"] = [
        {
            "name_pattern": "GPIO{n}",
            "start": 0,
            "end": 10,
            "endpoints": [
                {"sheet": "MCU", "direction": "bidirectional"},
                {"sheet": "APP", "direction": "passive"},
            ],
        }
    ]
    with pytest.raises(StageSchemaError) as rejected:
        _normalize_stage_response("architecture", payload, {})
    assert rejected.value.diagnostic["code"] == "unavailable_recipe_gpio"
    assert "GPIO9" in rejected.value.diagnostic["evidence"]
    assert "allocatable_gpios=0,1,2,3,4,5,6,7,8,10" in rejected.value.diagnostic["evidence"]
    assert payload["inter_sheet_net_ranges"][0]["end"] == 10


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
                recipe="esp32-c3-mini-1-minimal@1",
                instance="invalid",
                sheets={"mcu": "MCU"},
                parameters={"auto_reset": enabled},
                port_bindings={"vdd": "+3V3", "gnd": "GND", **ports},
            )
        )
