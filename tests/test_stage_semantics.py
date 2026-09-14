import json
from pathlib import Path

import pytest

from kicraft.design.stage_semantics import (
    complete_unsourced_external_rails,
    diagnose_stage,
    remove_mislabeled_architecture_defaults,
    remove_mislabeled_functional_defaults,
)

FIXTURES = Path(__file__).parent / "fixtures" / "stage_reliability"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def _codes(stage, candidate, upstream=None):
    brief = _load("rp2040_brief.json")["brief"]
    return {
        d.code
        for d in diagnose_stage(
            stage, brief=brief, upstream_state=upstream or {}, candidate=candidate
        )
    }


def test_live_intent_candidate_reports_classification_defects():
    codes = _codes("intent", _load("rp2040_intent_candidate.json"))
    assert {
        "intent_named_part_omitted",
        "intent_constraints_empty",
        "intent_unclassified_copy",
    } <= codes
    diagnostics = diagnose_stage(
        "intent",
        brief=_load("rp2040_brief.json")["brief"],
        upstream_state={},
        candidate=_load("rp2040_intent_candidate.json"),
    )
    named_part = next(d for d in diagnostics if d.code == "intent_named_part_omitted")
    assert named_part.severity == "repair_required"


def test_vague_intent_may_leave_classification_empty():
    diagnostics = diagnose_stage(
        "intent",
        brief="A small sensor board",
        upstream_state={},
        candidate={"goal": "A small sensor board", "constraints": [], "named_parts": []},
    )
    assert diagnostics == []


def test_functional_spec_rejects_hidden_topology_and_explicit_defaults():
    brief = "USB C PD power configured for 5V to an ESP32-S3-WROOM-1-N16R8 with a speaker output"
    candidate = {
        "blocks": [
            {
                "name": "SPEAKER",
                "category": "drive",
                "purpose": "Provides an amplified PWM or DAC-driven speaker output.",
            },
            {
                "name": "POWER_DISTRIBUTION",
                "category": "power",
                "purpose": "Distributes the 5V rail.",
            },
        ],
        "connections": [
            {
                "from_block": "POWER_DISTRIBUTION",
                "to_block": "SPEAKER",
                "signal_type": "power",
                "description": "5V supply to the MCU",
            }
        ],
        "assumptions": [
            "USB-C PD input is configured to negotiate a fixed 5V output (defaulted).",
            "The display is a single 64x64 panel driven directly by the MCU (defaulted).",
            "The LED string uses a 5V WS2812-style single-wire protocol (defaulted).",
            "The speaker uses an analog audio output (defaulted).",
            "The ESP32-S3-WROOM-1-N16R8 is the sole processor (defaulted).",
            "The board supplies 5V power to both external loads (defaulted).",
            "The LED string is a single data-line type (defaulted).",
        ],
    }

    diagnostics = diagnose_stage(
        "functional_spec",
        brief=brief,
        upstream_state={
            "intent": {
                "constraints": ["USB C PD power configured for 5V"],
                "named_parts": ["ESP32-S3-WROOM-1-N16R8"],
            },
            "_stage_answers": [{"answer": "Board supplies power to both loads"}],
        },
        candidate=candidate,
    )
    by_code = {diagnostic.code: diagnostic for diagnostic in diagnostics}
    topology_evidence = by_code["functional_spec_premature_topology"].evidence
    assert "pwm" in topology_evidence
    assert "dac-driven" in topology_evidence
    assert "amplified" in topology_evidence
    assert "64x64" in topology_evidence
    assert "ws2812-style" in topology_evidence
    assert "driven directly" in topology_evidence
    assert "single-wire" in topology_evidence
    assert "analog audio" in topology_evidence
    assert "single data-line" in topology_evidence
    assert "5v supply to the mcu" in topology_evidence
    assert "functional_spec_nonfunctional_block" in by_code
    assert len(by_code["functional_spec_explicit_fact_defaulted"].evidence) == 3
    cleaned = remove_mislabeled_functional_defaults(
        brief,
        {
            "intent": {
                "constraints": ["USB C PD power configured for 5V"],
                "named_parts": ["ESP32-S3-WROOM-1-N16R8"],
            },
            "_stage_answers": [{"answer": "Board supplies power to both loads"}],
        },
        candidate,
    )
    assert cleaned["assumptions"] == [
        "The display is a single 64x64 panel driven directly by the MCU (defaulted).",
        "The LED string uses a 5V WS2812-style single-wire protocol (defaulted).",
        "The speaker uses an analog audio output (defaulted).",
        "The LED string is a single data-line type (defaulted).",
    ]


def test_functional_spec_requires_external_load_power_decision():
    candidate = {
        "blocks": [
            {"name": "USB_INPUT", "category": "power", "purpose": "Board power"},
            {"name": "HUB75_OUTPUT", "category": "drive", "purpose": "Display output"},
            {
                "name": "ADDRESSABLE_LED_OUTPUT",
                "category": "drive",
                "purpose": "LED output",
            },
        ],
        "connections": [
            {
                "from_block": "USB_INPUT",
                "to_block": "HUB75_OUTPUT",
                "signal_type": "power",
                "description": "Panel power",
            },
            {
                "from_block": "USB_INPUT",
                "to_block": "ADDRESSABLE_LED_OUTPUT",
                "signal_type": "power",
                "description": "LED power",
            },
        ],
        "assumptions": [],
    }

    ambiguous = diagnose_stage(
        "functional_spec",
        brief="Drive a HUB75 panel and an addressable LED string",
        upstream_state={},
        candidate=candidate,
    )
    diagnostic = next(
        item for item in ambiguous if item.code == "functional_spec_external_load_power_assumed"
    )
    assert diagnostic.evidence == ["addressable_led_output", "hub75_output"]

    explicit = diagnose_stage(
        "functional_spec",
        brief="Power the HUB75 panel and LED string from the board",
        upstream_state={},
        candidate=candidate,
    )
    assert "functional_spec_external_load_power_assumed" not in {item.code for item in explicit}

    answered = diagnose_stage(
        "functional_spec",
        brief="Drive a HUB75 panel and an addressable LED string",
        upstream_state={
            "_stage_answers": [
                {
                    "text": "Should the board power the display and LED string?",
                    "answer": "Board supplies power to both loads",
                }
            ]
        },
        candidate=candidate,
    )
    assert "functional_spec_external_load_power_assumed" not in {item.code for item in answered}


def test_functional_spec_requires_power_and_consistent_ground_for_drives():
    candidate = {
        "blocks": [
            {"name": "POWER", "category": "power", "purpose": "Board power"},
            {"name": "MCU", "category": "process", "purpose": "Controller"},
            {"name": "SPEAKER", "category": "drive", "purpose": "Speaker output"},
        ],
        "connections": [
            {
                "from_block": "POWER",
                "to_block": "MCU",
                "signal_type": "power",
                "description": "MCU power",
            },
            {
                "from_block": "MCU",
                "to_block": "SPEAKER",
                "signal_type": "other",
                "description": "Audio",
            },
            {
                "from_block": "POWER",
                "to_block": "MCU",
                "signal_type": "ground",
                "description": "Ground",
            },
        ],
        "assumptions": [],
    }

    diagnostics = diagnose_stage(
        "functional_spec",
        brief="Controller with speaker output",
        upstream_state={},
        candidate=candidate,
    )
    by_code = {item.code: item for item in diagnostics}
    assert by_code["functional_spec_drive_missing_power"].evidence == ["speaker"]
    assert by_code["functional_spec_partial_ground_flow"].evidence == ["speaker"]


def test_live_functional_spec_reports_premature_topology():
    candidate = _load("rp2040_functional_spec_candidate.json")
    codes = _codes("functional_spec", candidate, {"intent": _load("rp2040_intent_candidate.json")})
    assert "functional_spec_premature_topology" in codes


def test_live_architecture_reports_graph_and_domain_defects():
    upstream = {"functional_spec": _load("rp2040_functional_spec_candidate.json")}
    codes = _codes("architecture", _load("rp2040_architecture_candidate.json"), upstream)
    assert "architecture_power_block_as_sheet" in codes
    assert "architecture_fragmented_physical_domain" in codes
    assert "architecture_wrong_signal_direction" in codes


def _stm32_physical_domains():
    # Domain declarations from the round9 STM32 architecture repair response.
    return {
        "sheets": [
            {
                "name": "MCU",
                "stem": "MCU",
                "function": (
                    "STM32F103C8T6 MCU with 8 MHz HSE crystal, decoupling, and programming support."
                ),
            },
            {
                "name": "RESET CONTROL",
                "stem": "RESET_CONTROL",
                "function": "Manual reset button with pullup resistor.",
            },
        ],
        "requirements": [
            {
                "id": "mcu_core",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": "stm32f103c8",
                "exact_part": "STM32F103C8T6",
                "functional_blocks": ["MCU", "CLOCK_SOURCE"],
                "interfaces": ["usb_device", "swd"],
                "parameters": {"can_remap": "pb8-pb9"},
                "ports": {
                    "boot0": "BOOT0",
                    "gnd": "GND",
                    "nrst": "NRST",
                    "swclk": "SWCLK",
                    "swdio": "SWDIO",
                    "usb_dm": "USB_D_N",
                    "usb_dp": "USB_D_P",
                    "vdd": "+3V3",
                },
            },
            {
                "id": "reset_control",
                "sheet": "RESET CONTROL",
                "role": "user_io",
                "family": "switch-input",
                "functional_blocks": ["RESET_CONTROL"],
                "parameters": {
                    "active_level": "low",
                    "pull_policy": "pull_up",
                    "resistance": 10000,
                },
                "ports": {"gnd": "GND", "signal": "NRST", "vdd": "+3V3"},
            },
        ],
        "inter_sheet_nets": [
            {
                "name": "NRST",
                "endpoints": [
                    {"sheet": "MCU", "direction": "bidirectional"},
                    {"sheet": "RESET CONTROL", "direction": "bidirectional"},
                ],
            },
        ],
    }


def test_architecture_keeps_integrated_ic_support_and_manual_control_domains():
    candidate = _stm32_physical_domains()
    assert "architecture_fragmented_physical_domain" not in _codes("architecture", candidate)

    # Before typed ownership was available, the physical function still
    # distinguished an owning IC and a real button from their support parts.
    del candidate["requirements"]
    assert "architecture_fragmented_physical_domain" not in _codes("architecture", candidate)


@pytest.mark.parametrize(
    ("name", "function", "family", "ports"),
    [
        (
            "HSE SUPPORT",
            "8 MHz HSE crystal with load capacitors for the MCU.",
            "crystal",
            {"in": "OSC_IN", "out": "OSC_OUT"},
        ),
        (
            "MCU DECOUPLING",
            "Decoupling capacitors for the STM32F103C8T6 MCU.",
            "capacitor",
            {"a": "+3V3", "b": "GND"},
        ),
        (
            "RESET BIAS",
            "NRST pullup resistor for the MCU.",
            "resistor",
            {"a": "+3V3", "b": "NRST"},
        ),
    ],
)
def test_architecture_rejects_support_only_sheets_separated_from_ic(name, function, family, ports):
    candidate = _stm32_physical_domains()
    candidate["sheets"].append({"name": name, "stem": name.replace(" ", "_"), "function": function})
    candidate["requirements"].append(
        {
            "id": "separated_support",
            "sheet": name,
            "role": "analog_block",
            "family": family,
            "functional_blocks": ["CLOCK_SOURCE" if family == "crystal" else "MCU"],
            "ports": ports,
        }
    )
    candidate["requirements"][0]["ports"].update(ports)
    candidate["inter_sheet_nets"].extend(
        {
            "name": net,
            "endpoints": [
                {"sheet": "MCU", "direction": "passive"},
                {"sheet": name, "direction": "passive"},
            ],
        }
        for net in ports.values()
    )
    diagnostics = diagnose_stage(
        "architecture", brief="STM32F103C8T6 board", upstream_state={}, candidate=candidate
    )
    fragments = [
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.code == "architecture_fragmented_physical_domain"
    ]
    assert [diagnostic.evidence for diagnostic in fragments] == [[name.lower()]]
    assert all(diagnostic.severity == "repair_required" for diagnostic in fragments)


def test_power_category_keeps_physical_connectors_and_holders_not_bare_rails():
    sheets = [
        {"name": "OUTPUT POWER", "stem": "OUTPUT_POWER", "function": "Load connector"},
        {"name": "ENERGY STORAGE", "stem": "ENERGY_STORAGE", "function": "CR2032 cell holder"},
        {"name": "GND", "stem": "GND", "function": "Ground reference net"},
    ]
    diagnostics = diagnose_stage(
        "architecture",
        brief="A battery holder and load connector.",
        upstream_state={
            "functional_spec": {
                "blocks": [
                    {"name": sheet["stem"], "category": "power", "purpose": sheet["function"]}
                    for sheet in sheets
                ]
            }
        },
        candidate={"sheets": sheets, "power_nets": ["VBAT", "GND"], "inter_sheet_nets": []},
    )
    assert [
        evidence.upper()
        for diagnostic in diagnostics
        if diagnostic.code == "architecture_power_block_as_sheet"
        for evidence in diagnostic.evidence
    ] == ["GND"]


def test_power_input_conversion_uses_owned_port_voltages_not_sheet_prose():
    candidate = {
        "sheets": [
            {"name": "POWER", "stem": "POWER", "function": "External input and board power"}
        ],
        "rail_voltages": {"VIN": 5.0, "VOUT": 3.3, "GND": 0.0},
        "requirements": [
            {
                "id": "supply",
                "sheet": "POWER",
                "role": "power_input",
                "family": "power-input",
                "ports": {"input": "VIN", "output": "VOUT", "gnd": "GND"},
            }
        ],
    }
    assert "architecture_unowned_power_conversion" in _codes("architecture", candidate)

    # No conversion is claimed by a connector carrying one unchanged rail.
    candidate["requirements"][0]["ports"]["output"] = "VIN"
    assert "architecture_unowned_power_conversion" not in _codes("architecture", candidate)

    # A standalone physical regulator may remain model-owned; this gate does
    # not demand a registered recipe or treat ordinary support as fragmentation.
    candidate["requirements"][0].update(
        role="regulator",
        family="ldo",
        exact_part="MCP1700T-3302E/TT",
        ports={"input": "VIN", "output": "VOUT", "gnd": "GND"},
    )
    assert not (
        {"architecture_unowned_power_conversion", "architecture_unowned_power_support"}
        & _codes("architecture", candidate)
    )


def test_architecture_rejects_false_pd_dac_provenance_and_missing_load_budget():
    candidate = {
        "topologies": {
            "USB_PD_INPUT": "USB-C PD sink via CC resistors, no PD negotiation IC",
            "MCU": "ESP32-S3-WROOM-1 with native USB",
            "SPEAKER": "Class-D amplifier driven by ESP32-S3 DAC",
        },
        "rail_voltages": {"VBUS": 5.0, "+3V3": 3.3},
        "comms_protocols": ["USB 2.0 FS"],
        "mcu_present": True,
        "sheets": [
            {
                "name": "MCU",
                "stem": "MCU",
                "function": "ESP32-S3 flashed over native USB with boot and reset controls",
            },
            {
                "name": "HUB75 OUTPUT",
                "stem": "HUB75_OUTPUT",
                "function": "HUB75 data and clock output",
            },
            {
                "name": "POWER INPUT",
                "stem": "POWER_INPUT",
                "function": "USB-C PD sink controller and protected 5V supply",
            },
        ],
        "power_nets": ["VBUS", "+3V3", "GND"],
        "inter_sheet_nets": [],
        "assumptions": ["LDO 3.3V <=500mA: ME6211C33 per core defaults (defaulted)"],
    }
    diagnostics = diagnose_stage(
        "architecture",
        brief="USB-C PD 5V ESP32-S3 HUB75 controller",
        upstream_state={
            "intent": {
                "goal": "USB-C PD 5V ESP32-S3 HUB75 controller",
                "named_parts": ["ESP32-S3-WROOM-1-N16R8"],
            },
            "_stage_answers": [{"answer": "Board supplies power to both loads"}],
            "_stage_extras": {},
        },
        candidate=candidate,
    )

    codes = {item.code for item in diagnostics}
    assert "architecture_usb_pd_without_controller" in codes
    assert "architecture_unsupported_esp32s3_dac" in codes
    assert "architecture_unavailable_core_default" in codes
    assert "architecture_external_load_current_unspecified" in codes
    assert "architecture_programming_decision_incomplete" not in codes
    assert "architecture_fragmented_physical_domain" not in codes
    assert "architecture_power_block_as_sheet" not in codes
    assert "architecture_missing_power_endpoint" not in codes
    assert "architecture_rail_source_unspecified" not in codes
    assert "architecture_mcu_regulator_incomplete" in codes

    analog_audio = {
        **candidate,
        "topologies": {
            **candidate["topologies"],
            "SPEAKER": "Class-D amplifier driven by an analog audio signal from the MCU",
        },
    }
    analog_codes = {
        item.code
        for item in diagnose_stage(
            "architecture",
            brief="USB-C PD 5V ESP32-S3 HUB75 controller",
            upstream_state={
                "intent": {
                    "goal": "USB-C PD 5V ESP32-S3 HUB75 controller",
                    "named_parts": ["ESP32-S3-WROOM-1-N16R8"],
                },
                "_stage_answers": [{"answer": "5A"}],
                "_stage_extras": {},
            },
            candidate=analog_audio,
        )
    }
    assert "architecture_unsupported_esp32s3_dac" in analog_codes

    # The check reads the registry, not the model's prose: a topology line that
    # says "rated 1A" proves nothing, a reviewed 2A recipe does.
    upstream = {
        "intent": {
            "goal": "USB-C PD 5V ESP32-S3 HUB75 controller",
            "named_parts": ["ESP32-S3-WROOM-1-N16R8"],
        },
        "_stage_answers": [{"answer": "5A"}],
        "_stage_extras": {},
    }

    def _regulator_codes(candidate: dict) -> set[str]:
        return {
            item.code
            for item in diagnose_stage(
                "architecture",
                brief="USB-C PD 5V ESP32-S3 HUB75 controller",
                upstream_state=upstream,
                candidate=candidate,
            )
        }

    prose_only = {
        **candidate,
        "topologies": {
            **candidate["topologies"],
            "REGULATOR_3V3": "5V-to-3.3V synchronous buck regulator rated 1A",
        },
        "sheets": [
            *candidate["sheets"],
            {
                "name": "REGULATOR 3V3",
                "stem": "REGULATOR_3V3",
                "function": "Dedicated 5V-to-3.3V regulator IC and support parts",
            },
        ],
    }
    assert "architecture_mcu_regulator_incomplete" in _regulator_codes(prose_only)

    def _with_3v3_producer(recipe: str) -> dict:
        return {
            **prose_only,
            "requirements": [
                {
                    "id": "buck",
                    "sheet": "REGULATOR 3V3",
                    "role": "regulator",
                    "family": recipe.split("@")[0],
                    "ports": {"input": "VBUS", "output": "+3V3", "gnd": "GND"},
                }
            ],
            "recipe_resolution": [{"requirement_id": "buck", "recipe": recipe}],
        }

    # A 0.5A LDO cannot be the MCU's 3.3V source, however the prose is phrased.
    weak = _with_3v3_producer("me6211-3v3@1")
    assert "architecture_mcu_regulator_incomplete" in _regulator_codes(weak)

    # A 2A reviewed buck is enough, even with the topology prose removed.
    complete_regulator = {
        **_with_3v3_producer("tlv62569-3v3@1"),
        "topologies": {"MCU": "ESP32-S3-WROOM-1 with native USB"},
    }
    complete_codes = _regulator_codes(complete_regulator)
    assert "architecture_mcu_regulator_incomplete" not in complete_codes
    assert "architecture_power_block_as_sheet" not in complete_codes

    answered_candidate = {
        **candidate,
        "assumptions": [
            *candidate["assumptions"],
            "External-load budget: 5A total as specified by user (defaulted)",
        ],
    }
    cleaned = remove_mislabeled_architecture_defaults(
        {"_stage_answers": [{"answer": "5A"}]},
        answered_candidate,
    )
    assert cleaned["assumptions"] == candidate["assumptions"]

    missing_rail = {**candidate, "rail_voltages": {"VBUS": 5.0}}
    missing_rail_codes = {
        item.code
        for item in diagnose_stage(
            "architecture",
            brief="USB-C PD 5V ESP32-S3 HUB75 controller",
            upstream_state={
                "intent": {
                    "goal": "USB-C PD 5V ESP32-S3 HUB75 controller",
                    "named_parts": ["ESP32-S3-WROOM-1-N16R8"],
                },
                "_stage_answers": [{"answer": "5A"}],
                "_stage_extras": {},
            },
            candidate=missing_rail,
        )
    }
    assert "architecture_mcu_supply_rail_missing" in missing_rail_codes


def test_architecture_accepts_an_explicit_externally_supplied_rail():
    candidate = {
        "topologies": {
            "INPUT_HEADER": "Logic input header",
            "DAC": "R-2R ladder with an op-amp buffer",
        },
        "rail_voltages": {"+3V3": 3.3},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {"name": "INPUT HEADER", "stem": "INPUT_HEADER", "function": "Logic input"},
            {"name": "DAC", "stem": "DAC", "function": "R-2R DAC and op-amp buffer"},
        ],
        "power_nets": ["+3V3", "GND"],
        "inter_sheet_nets": [],
        "assumptions": [],
    }
    upstream = {"intent": {"goal": "R-2R DAC breakout with external logic inputs"}}

    missing_source = _codes("architecture", candidate, upstream)
    candidate["assumptions"] = [
        "Power is 3.3V and provided externally via the input header (defaulted)."
    ]
    external_source = _codes("architecture", candidate, upstream)
    candidate["rail_voltages"] = {"VCC": 3.3}
    candidate["power_nets"] = ["VCC", "GND"]
    candidate["assumptions"] = [
        "Power is supplied via a 3.3V rail from an external source (defaulted)."
    ]
    external_source_via_voltage = _codes("architecture", candidate, upstream)

    assert "architecture_rail_source_unspecified" in missing_source
    assert "architecture_rail_source_unspecified" not in external_source
    assert "architecture_rail_source_unspecified" not in external_source_via_voltage


def test_architecture_defaults_unsourced_rail_to_external_power_input():
    candidate = {
        "topologies": {"DAC": "R-2R ladder"},
        "rail_voltages": {"+3V3": 3.3},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [{"name": "DAC", "stem": "DAC", "function": "R-2R DAC"}],
        "power_nets": ["+3V3", "GND"],
        "inter_sheet_nets": [],
        "assumptions": [],
    }
    upstream = {"intent": {"goal": "R-2R DAC breakout"}}
    diagnostics = diagnose_stage(
        "architecture",
        brief="R-2R DAC breakout",
        upstream_state=upstream,
        candidate=candidate,
    )

    completed = complete_unsourced_external_rails(candidate, diagnostics)

    assert candidate["sheets"] == [{"name": "DAC", "stem": "DAC", "function": "R-2R DAC"}]
    assert completed["topologies"]["POWER INPUT"] == ("2-pin header for external +3V3 and GND")
    assert completed["sheets"][-1]["name"] == "POWER INPUT"
    assert completed["assumptions"] == [
        "+3V3 is supplied externally through the power input (defaulted)"
    ]
    assert "architecture_rail_source_unspecified" not in _codes(
        "architecture",
        completed,
        upstream,
    )


def test_architecture_rejects_source_without_headroom_and_unrelated_equal_rails():
    candidate = {
        "topologies": {
            "USB_PD_INPUT": "USB-C PD sink controller requesting 5V/5A",
            "REGULATOR_3V3": "5V-to-3.3V synchronous buck regulator rated 1A",
        },
        "rail_voltages": {"VBUS": 5.0, "+5V": 5.0, "+3V3": 3.3},
        "comms_protocols": ["USB 2.0 FS"],
        "mcu_present": True,
        "sheets": [
            {
                "name": "USB PD INPUT",
                "stem": "USB_PD_INPUT",
                "function": "USB-C PD sink controller and connector",
            },
            {
                "name": "MCU",
                "stem": "MCU",
                "function": "ESP32-S3 with native USB flashing and boot controls",
            },
            {
                "name": "REGULATOR 3V3",
                "stem": "REGULATOR_3V3",
                "function": "Dedicated 5V-to-3.3V buck regulator IC supplying the ESP32 MCU",
            },
            {
                "name": "HUB75 OUTPUT",
                "stem": "HUB75_OUTPUT",
                "function": "HUB75 connector and level shifter",
            },
            {
                "name": "LED STRING OUTPUT",
                "stem": "LED_STRING_OUTPUT",
                "function": "Addressable LED output connector",
            },
        ],
        "power_nets": ["VBUS", "+5V", "+3V3", "GND"],
        "inter_sheet_nets": [
            {
                "name": "VBUS",
                "endpoints": [
                    {"sheet": "USB PD INPUT", "direction": "bidirectional"},
                    {"sheet": "HUB75 OUTPUT", "direction": "bidirectional"},
                ],
            },
            {
                "name": "+5V",
                "endpoints": [
                    {"sheet": "USB PD INPUT", "direction": "bidirectional"},
                    {"sheet": "LED STRING OUTPUT", "direction": "bidirectional"},
                ],
            },
        ],
        "assumptions": [],
    }
    upstream = {
        "intent": {"goal": "USB-C PD powered ESP32-S3 HUB75 and LED-string controller"},
        "functional_spec": {
            "connections": [
                {
                    "to_block": "HUB75_OUTPUT",
                    "signal_type": "power",
                },
                {
                    "to_block": "LED_STRING_OUTPUT",
                    "signal_type": "power",
                },
            ]
        },
        "_stage_answers": [{"answer": "5A"}],
        "_stage_extras": {},
    }

    codes = {
        diagnostic.code
        for diagnostic in diagnose_stage(
            "architecture",
            brief="USB-C PD controller with regulated 5V load outputs",
            upstream_state=upstream,
            candidate=candidate,
        )
    }

    assert "architecture_external_load_source_has_no_headroom" in codes
    assert "architecture_duplicate_voltage_rails_unrelated" in codes

    overcurrent_candidate = {
        **candidate,
        "topologies": {
            **candidate["topologies"],
            "USB_PD_INPUT": "USB-C PD sink controller with 5V/6A output",
        },
    }
    overcurrent_codes = {
        diagnostic.code
        for diagnostic in diagnose_stage(
            "architecture",
            brief="USB-C PD controller with regulated 5V load outputs",
            upstream_state=upstream,
            candidate=overcurrent_candidate,
        )
    }
    assert "architecture_usb_pd_current_exceeds_standard" in overcurrent_codes

    corrected = {
        **candidate,
        "topologies": {
            **candidate["topologies"],
            "USB_PD_INPUT": "USB-C PD sink using a 20V/5A PD contract, buck-converted to a regulated 5V/6A load output",
            "5V_BUCK": "Synchronous buck converter from a 20V PD rail to regulated 5V/6A",
        },
        "rail_voltages": {"VBUS": 20.0, "+5V": 5.0, "+3V3": 3.3},
        "requirements": [
            {
                "id": "buck",
                "sheet": "REGULATOR 3V3",
                "role": "regulator",
                "family": "tlv62569-3v3",
                "ports": {"input": "VBUS", "output": "+3V3", "gnd": "GND"},
            }
        ],
        "recipe_resolution": [{"requirement_id": "buck", "recipe": "tlv62569-3v3@1"}],
        "inter_sheet_nets": [
            {
                "name": "VBUS",
                "endpoints": [
                    {"sheet": "USB PD INPUT", "direction": "output"},
                    {"sheet": "REGULATOR 3V3", "direction": "input"},
                ],
            },
            {
                "name": "+5V",
                "endpoints": [
                    {"sheet": "REGULATOR 3V3", "direction": "output"},
                    {"sheet": "HUB75 OUTPUT", "direction": "bidirectional"},
                    {"sheet": "LED STRING OUTPUT", "direction": "bidirectional"},
                ],
            },
        ],
    }
    corrected_codes = {
        diagnostic.code
        for diagnostic in diagnose_stage(
            "architecture",
            brief="USB-C PD controller with regulated 5V load outputs",
            upstream_state=upstream,
            candidate=corrected,
        )
    }

    assert "architecture_external_load_source_has_no_headroom" not in corrected_codes
    assert "architecture_duplicate_voltage_rails_unrelated" not in corrected_codes
    assert "architecture_usb_pd_current_exceeds_standard" not in corrected_codes
    assert "architecture_5v_converter_capacity_unspecified" not in corrected_codes
    assert "architecture_5v_converter_has_no_headroom" not in corrected_codes

    assert "architecture_mcu_regulator_incomplete" not in corrected_codes

    merged_3v3 = {
        **corrected,
        "topologies": {
            name: description
            for name, description in corrected["topologies"].items()
            if name != "REGULATOR_3V3"
        },
        "sheets": [sheet for sheet in corrected["sheets"] if sheet["name"] != "REGULATOR 3V3"],
    }
    merged_3v3_codes = {
        diagnostic.code
        for diagnostic in diagnose_stage(
            "architecture",
            brief="USB-C PD controller with regulated 5V load outputs",
            upstream_state=upstream,
            candidate=merged_3v3,
        )
    }
    assert "architecture_mcu_regulator_incomplete" in merged_3v3_codes


def _switched_rail_architecture():
    return {
        "rail_voltages": {"VBUS": 5.0, "PORT1_5V": 5.0, "PORT2_5V": 5.0},
        "topologies": {
            "CURRENT_LIMIT": "Independent high-side current-limited power switch"
        },
        "sheets": [
            {"name": f"CHANNEL {index}", "function": "Independent current limiting"}
            for index in (1, 2)
        ],
        "requirements": [
            {
                "id": f"limit_{index}",
                "sheet": f"CHANNEL {index}",
                "role": "power_input",
                "family": "usb-current-limiter",
                "ports": {"input": "VBUS", "output": f"PORT{index}_5V"},
                "functional_blocks": ["CURRENT_LIMIT"],
            }
            for index in (1, 2)
        ],
        "inter_sheet_nets": [
            {
                "name": "VBUS",
                "endpoints": [
                    {"sheet": f"CHANNEL {index}", "direction": "input"}
                    for index in (1, 2)
                ],
            },
            *(
                {
                    "name": f"PORT{index}_5V",
                    "endpoints": [{"sheet": f"CHANNEL {index}", "direction": "output"}],
                }
                for index in (1, 2)
            ),
        ],
    }


def test_architecture_accepts_typed_switched_fanout_without_rail_names_in_prose():
    candidate = _switched_rail_architecture()

    assert "architecture_duplicate_voltage_rails_unrelated" not in _codes(
        "architecture", candidate
    )
    assert candidate["rail_voltages"] == {"VBUS": 5.0, "PORT1_5V": 5.0, "PORT2_5V": 5.0}
    assert [row["ports"]["output"] for row in candidate["requirements"]] == [
        "PORT1_5V",
        "PORT2_5V",
    ]


@pytest.mark.parametrize(
    "missing_proof",
    ["component_ports", "output_direction", "circuit_meaning", "owned_topology"],
)
def test_architecture_rejects_equal_rails_without_owned_directional_component(missing_proof):
    candidate = _switched_rail_architecture()
    first = candidate["requirements"][0]
    if missing_proof == "component_ports":
        first["ports"] = {}
    elif missing_proof == "output_direction":
        candidate["inter_sheet_nets"][1]["endpoints"][0]["direction"] = "input"
    elif missing_proof == "circuit_meaning":
        first["family"] = "power-monitor"
        first["functional_blocks"] = []
    else:
        first["family"] = "power-monitor"
        first["functional_blocks"] = ["MONITOR"]
        candidate["topologies"]["MONITOR"] = "Voltage measurement"
    # A same-sheet endpoint or arbitrary prose co-mention cannot replace proof.
    candidate["sheets"][0]["function"] = "VBUS and PORT1_5V switch/filter connections"
    diagnostics = diagnose_stage(
        "architecture", brief="", upstream_state={}, candidate=candidate
    )
    unrelated = [
        item
        for item in diagnostics
        if item.code == "architecture_duplicate_voltage_rails_unrelated"
    ]
    evidence = {entry for item in unrelated for entry in item.evidence}
    assert "vbus/port1_5v: 5v" in evidence
    assert "port1_5v/port2_5v: 5v" in evidence
    assert "vbus/port2_5v: 5v" not in evidence


def test_architecture_rejects_independent_equal_rails_without_common_upstream():
    candidate = _switched_rail_architecture()
    candidate["rail_voltages"]["OTHER_SUPPLY"] = 12.0
    candidate["requirements"][1]["ports"]["input"] = "OTHER_SUPPLY"
    candidate["inter_sheet_nets"][0]["endpoints"].pop()
    candidate["inter_sheet_nets"].append(
        {
            "name": "OTHER_SUPPLY",
            "endpoints": [{"sheet": "CHANNEL 2", "direction": "input"}],
        }
    )

    diagnostics = diagnose_stage(
        "architecture", brief="", upstream_state={}, candidate=candidate
    )
    assert any(
        item.code == "architecture_duplicate_voltage_rails_unrelated"
        and "port1_5v/port2_5v: 5v" in item.evidence
        for item in diagnostics
    )


def test_live_bom_and_wiring_report_fabrication_gates():
    bom = _load("rp2040_bom_candidate.json")
    assert "bom_castellation_placeholder" in _codes("bom", bom)
    wiring = _load("rp2040_wiring_candidate.json")
    codes = _codes("wiring", wiring, {"bom": bom})
    assert "wiring_bootsel_unreachable" in codes
    assert "wiring_special_pin_no_connect" in codes


def test_architecture_power_contract_fixtures_preserve_dual_buck_and_right_size():
    from kicraft.design.models import Architecture

    overprovisioned = _load("architecture_20v_5a_overprovisioned.json")
    right_sized = _load("architecture_15v_3a_right_sized.json")
    Architecture.model_validate(overprovisioned)
    Architecture.model_validate(right_sized)
    upstream = {
        "intent": {"goal": "USB-C PD powered ESP32-S3 HUB75 and LED-string controller"},
        "functional_spec": {
            "connections": [
                {"to_block": "HUB75_OUTPUT", "signal_type": "power"},
                {"to_block": "LED_STRING_OUTPUT", "signal_type": "power"},
            ]
        },
        "_stage_answers": [{"answer": "Board supplies both loads, 5 A maximum"}],
        "_stage_extras": {},
    }
    oversized_diagnostics = diagnose_stage(
        "architecture",
        brief="Regulate 5 V / 5 A for HUB75 and LED loads",
        upstream_state=upstream,
        candidate=overprovisioned,
    )
    oversized = {item.code: item for item in oversized_diagnostics}
    corrected_codes = {
        item.code
        for item in diagnose_stage(
            "architecture",
            brief="Regulate 5 V / 5 A for HUB75 and LED loads",
            upstream_state=upstream,
            candidate=right_sized,
        )
    }

    assert oversized["architecture_input_power_grossly_overprovisioned"].severity == "advisory"
    assert "architecture_input_power_grossly_overprovisioned" not in corrected_codes
    assert {
        "architecture_external_load_source_has_no_headroom",
        "architecture_5v_converter_capacity_unspecified",
        "architecture_5v_converter_has_no_headroom",
        "architecture_duplicate_voltage_rails_unrelated",
        "architecture_usb_pd_current_exceeds_standard",
    }.isdisjoint(corrected_codes)
    assert {"5V_BUCK", "3V3_BUCK"} <= right_sized["topologies"].keys()
    assert [
        endpoint["direction"]
        for net in right_sized["inter_sheet_nets"]
        if net["name"] == "VBUS"
        for endpoint in net["endpoints"]
    ] == ["output", "input"]


def test_bom_requires_architecture_ic_roles_on_their_declared_sheets():
    architecture = _load("architecture_15v_3a_right_sized.json")
    parts = [
        {"ref": "U1", "sheet": "USB PD INPUT"},
        {"ref": "U2", "sheet": "5V BUCK"},
        {"ref": "U3", "sheet": "MCU"},
        {"ref": "U4", "sheet": "HUB75 OUTPUT"},
    ]
    diagnostics = diagnose_stage(
        "bom",
        brief="USB PD controller",
        upstream_state={"architecture": architecture},
        candidate={"parts": parts},
    )
    by_code = {item.code: item for item in diagnostics}
    assert by_code["bom_architecture_role_unsupported"].evidence == ["3v3 buck"]

    supported = {**parts[2], "ref": "U5", "sheet": "3V3 BUCK"}
    clean = diagnose_stage(
        "bom",
        brief="USB PD controller",
        upstream_state={"architecture": architecture},
        candidate={"parts": [*parts, supported]},
    )
    assert "bom_architecture_role_unsupported" not in {item.code for item in clean}


def test_bom_does_not_infer_amplifier_ic_from_typed_input_connector_title():
    architecture = {
        "sheets": [
            {
                "name": "AMPLIFIER INPUT",
                "function": "Binding-post terminal pair accepting a speaker-level input signal.",
            }
        ],
        "requirements": [
            {
                "id": "input_terminals",
                "sheet": "AMPLIFIER INPUT",
                "role": "connector",
                "family": "binding-post-terminal",
                "ports": {"negative": "GND", "positive": "IN_POS"},
            }
        ],
    }
    candidate = {"parts": [{"ref": "J1", "sheet": "AMPLIFIER INPUT"}]}

    assert "bom_architecture_role_unsupported" not in _codes(
        "bom", candidate, {"architecture": architecture}
    )


@pytest.mark.parametrize(
    ("function", "active_requirement"),
    [
        ("Connector and on-board amplifier", None),
        ("Control interface", {"role": "mcu_core", "family": "microcontroller"}),
        ("Control interface", {"role": "analog_block", "family": "audio-amplifier"}),
    ],
)
def test_bom_connector_does_not_hide_missing_active_implementation(function, active_requirement):
    architecture = {
        "sheets": [{"name": "CONTROL", "function": function}],
        "requirements": [
            {
                "id": "connector",
                "sheet": "CONTROL",
                "role": "connector",
                "family": "pin-header",
                "ports": {"signal": "SIGNAL"},
            }
        ],
    }
    if active_requirement is not None:
        architecture["requirements"].append(
            {"id": "active", "sheet": "CONTROL", **active_requirement}
        )
    candidate = {"parts": [{"ref": "J1", "sheet": "CONTROL"}]}

    assert "bom_architecture_role_unsupported" in _codes(
        "bom", candidate, {"architecture": architecture}
    )
    candidate["parts"].append({"ref": "U1", "sheet": "CONTROL"})
    assert "bom_architecture_role_unsupported" not in _codes(
        "bom", candidate, {"architecture": architecture}
    )


def test_mixed_case_and_short_mcu_families_survive_intent_classification():
    from kicraft.design.stage_semantics import complete_intent_classification

    brief = "An ATtiny1614 badge; alternatives are atmega328p, STM32, esp32 C3 and nRF52."
    completed = complete_intent_classification(brief, {"named_parts": ["ATtiny1614"]})
    assert completed["named_parts"] == ["ATtiny1614", "atmega328p", "STM32", "esp32 C3", "nRF52"]
    assert not any(
        row.code == "intent_named_part_omitted"
        for row in diagnose_stage("intent", brief=brief, upstream_state={}, candidate=completed)
    )


def test_intent_tokens_preserve_exact_variants_without_harvesting_counts_or_packages():
    from kicraft.design.synthesis.validation import named_part_tokens

    tokens = named_part_tokens(
        [
            "ATtiny1614 ATtiny1616 ESP32-S3-WROOM-1-N16R8 ESP32-S3-WROOM-1-N8R8 "
            "six 0805 LEDs at 3.3V, 5V, 500mA with GPIO12, QFN32, UART115200, "
            "pins16, LED0805, VCC33 and USB20."
        ]
    )
    assert set(tokens) == {
        "attiny1614",
        "attiny1616",
        "esp32-s3-wroom-1-n16r8",
        "esp32-s3-wroom-1-n8r8",
    }


def test_intent_family_separator_spellings_do_not_duplicate_named_parts():
    from kicraft.design.stage_semantics import complete_intent_classification

    completed = complete_intent_classification(
        "ESP32-C3 or esp32 C3; ATtiny1614 or ATtiny 1614", {"named_parts": [], "constraints": []}
    )
    assert completed["named_parts"] == ["ESP32-C3", "ATtiny1614"]
