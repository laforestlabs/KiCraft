import json
from pathlib import Path

from kicraft.design.stage_semantics import (
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
    brief = (
        "USB C PD power configured for 5V to an ESP32-S3-WROOM-1-N16R8 "
        "with a speaker output"
    )
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
            "_stage_answers": [
                {"answer": "Board supplies power to both loads"}
            ],
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
        item
        for item in ambiguous
        if item.code == "functional_spec_external_load_power_assumed"
    )
    assert diagnostic.evidence == ["addressable_led_output", "hub75_output"]

    explicit = diagnose_stage(
        "functional_spec",
        brief="Power the HUB75 panel and LED string from the board",
        upstream_state={},
        candidate=candidate,
    )
    assert "functional_spec_external_load_power_assumed" not in {
        item.code for item in explicit
    }


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
    assert "functional_spec_external_load_power_assumed" not in {
        item.code for item in answered
    }

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
        "assumptions": [
            "LDO 3.3V <=500mA: ME6211C33 per core defaults (defaulted)"
        ],
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

    complete_regulator = {
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
    complete_codes = {
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
            candidate=complete_regulator,
        )
    }
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

def test_live_bom_and_wiring_report_fabrication_gates():
    bom = _load("rp2040_bom_candidate.json")
    assert "bom_castellation_placeholder" in _codes("bom", bom)
    wiring = _load("rp2040_wiring_candidate.json")
    codes = _codes("wiring", wiring, {"bom": bom})
    assert "wiring_bootsel_unreachable" in codes
    assert "wiring_special_pin_no_connect" in codes
