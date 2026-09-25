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


def test_intent_rejects_physical_obligation_no_reviewed_part_implements():
    """A board-format or qualifier-wording class is unsatisfiable at every later gate.

    Reproduces the live proto-shield draft, whose stacking/regulator/format rows are the
    wording no reviewed record carries: `physical-obligation-unfulfilled` at BOM, not a
    repairable intent defect (see part_identity._REVIEWED_FEATURE_VOCABULARY).
    """
    brief = (
        "An Arduino-Uno-format prototyping shield with stacking through-hole headers "
        "and an onboard SMT 3.3 V regulator."
    )
    diagnostics = diagnose_stage(
        "intent",
        brief=brief,
        upstream_state={},
        candidate={
            "goal": "An Arduino Uno-format prototyping shield.",
            "constraints": ["Arduino Uno-format board"],
            "named_parts": [],
            "obligations": [
                {"kind": "physical", "original_obligation_id": "uno_format",
                 "component_class": "arduino-uno-format-board"},
                {"kind": "physical", "original_obligation_id": "stacking_headers",
                 "component_class": "stacking-through-hole-header"},
                {"kind": "physical", "original_obligation_id": "regulator",
                 "component_class": "smt-voltage-regulator"},
                {"kind": "quantitative", "original_obligation_id": "vout",
                 "quantity": "output voltage", "relation": "equal", "value": 3.3, "unit": "V"},
            ],
        },
    )
    unrealizable = [d for d in diagnostics if d.code == "intent_obligation_class_unrealizable"]
    assert [d.severity for d in unrealizable] == ["repair_required"] * 3
    # The repair must be actionable: the reviewed spelling travels in the evidence.
    stacking = next(
        d for d in unrealizable if "stacking-through-hole-header" in " ".join(d.evidence)
    )
    assert "stacking-header" in " ".join(stacking.evidence)


def test_intent_accepts_reviewed_physical_classes():
    """The reviewed spellings of the same requirements raise nothing."""
    diagnostics = diagnose_stage(
        "intent",
        brief="A shield with stacking headers and a 3.3 V regulator.",
        upstream_state={},
        candidate={
            "goal": "A shield with stacking headers and a regulator.",
            "constraints": ["3.3 V regulated output"],
            "named_parts": [],
            "obligations": [
                {"kind": "physical", "original_obligation_id": "stacking_headers",
                 "component_class": "stacking-header"},
                {"kind": "physical", "original_obligation_id": "regulator",
                 "component_class": "voltage-regulator"},
                {"kind": "physical", "original_obligation_id": "terminals",
                 "component_class": "power-screw-terminal"},
            ],
        },
    )
    assert [d.code for d in diagnostics if "obligation" in d.code] == []


def test_intent_leaves_a_new_part_category_alone():
    """A class the reviewed library does not cover yet is legitimate, not a defect.

    Regression guard: an earlier version of this gate flagged any class with no reviewed
    coverage and suggested the nearest reviewed name, which made a GPS brief draft a
    *wifi* module (`gps-module` shares only "module" with `wifi-module`). A new category
    must pass through untouched and keep its own name.
    """
    diagnostics = diagnose_stage(
        "intent",
        brief="A GPS tracker board with a u-blox NEO-6M GPS module and USB-C power.",
        upstream_state={},
        candidate={
            "goal": "A GPS tracker board with a GPS module and USB-C power.",
            "constraints": ["USB-C power"],
            "named_parts": ["u-blox NEO-6M"],
            "obligations": [
                {"kind": "physical", "original_obligation_id": "gnss",
                 "component_class": "gps-module"},
                {"kind": "physical", "original_obligation_id": "air",
                 "component_class": "air-quality-sensor"},
            ],
        },
    )
    assert [d.code for d in diagnostics if "obligation" in d.code] == []


def test_intent_flags_interface_and_printed_board_rows_as_not_a_part():
    """A bus or a copper feature is not a class, and the evidence says so."""
    diagnostics = diagnose_stage(
        "intent",
        brief="A panel board with an I2C interface and a thermal-via copper pour.",
        upstream_state={},
        candidate={
            "goal": "A panel board with an I2C interface and a copper heat spreader.",
            "constraints": [],
            "named_parts": [],
            "obligations": [
                {"kind": "physical", "original_obligation_id": "i2c",
                 "component_class": "i2c-interface"},
                {"kind": "physical", "original_obligation_id": "pour",
                 "component_class": "thermal-via-copper-pour"},
            ],
        },
    )
    flagged = [d for d in diagnostics if d.code == "intent_obligation_class_unrealizable"]
    assert len(flagged) == 2
    assert all("not a part class" in " ".join(d.evidence) for d in flagged)


def test_intent_records_a_prototyping_area_as_a_board_feature():
    """The defining feature of a prototyping shield must survive as a typed row.

    Regression guard for the proto-shield brief: "prototyping" is not a component class,
    owns no pin and draws no net, so if the intent drops it nothing downstream can build
    the pad field (no sheet, no requirement, no pads) and the delivered board fails the
    `prototyping_area` acceptance gate.
    """
    brief = (
        "An Arduino-Uno-format prototyping shield with stacking through-hole headers "
        "and an onboard SMT 3.3 V regulator."
    )
    candidate = {
        "goal": brief,
        "constraints": ["Arduino Uno format/outline", "Stacking through-hole headers"],
        "named_parts": [],
        "obligations": [
            {"kind": "physical", "original_obligation_id": "stacking_headers",
             "component_class": "stacking-header"},
        ],
    }
    diagnostics = diagnose_stage("intent", brief=brief, upstream_state={}, candidate=candidate)
    omitted = [d for d in diagnostics if d.code == "intent_prototyping_area_omitted"]
    assert len(omitted) == 1
    assert omitted[0].severity == "repair_required"
    assert "prototyping-area" in " ".join(omitted[0].evidence)

    # Recorded as a fabrication row the diagnostic clears, and nothing else changes.
    candidate["obligations"].append(
        {"kind": "fabrication", "original_obligation_id": "prototyping_area",
         "feature": "prototyping-area"}
    )
    assert [
        d.code
        for d in diagnose_stage("intent", brief=brief, upstream_state={}, candidate=candidate)
        if "prototyping" in d.code
    ] == []


def test_intent_ignores_a_plain_prototype_mention():
    """A one-off "prototype" build is not a pad field."""
    brief = "A prototype board with an ESP32-S3 and a USB-C connector."
    diagnostics = diagnose_stage(
        "intent",
        brief=brief,
        upstream_state={},
        candidate={"goal": brief, "constraints": ["USB-C"], "named_parts": [], "obligations": []},
    )
    assert [d.code for d in diagnostics if "prototyping" in d.code] == []


def test_functional_spec_rejects_a_block_for_the_prototyping_pad_field():
    """A pad field is a board feature, not a user-visible function: it is not a block.

    Regression guard for the proto-shield run that declared PROTOTYPING_AREA and then tried to
    route rails into it (`malformed_signal_ref`, `unsupported_lowerer_contract`). The sheet the
    compiler derives owns no port -- its pads are bare copper with no net -- so a block for it
    asks the architecture for a part that cannot exist; the field is derived from the intent's
    `fabrication` row instead.
    """
    code = "functional_spec_board_feature_block"

    def block(name: str) -> dict:
        return {"name": name, "category": "interface", "purpose": "One function.", "count": 1}

    def codes(blocks: list[dict]) -> set[str]:
        return _codes("functional_spec", {"blocks": blocks, "connections": [], "assumptions": []})

    diagnostics = diagnose_stage(
        "functional_spec",
        brief=_load("rp2040_brief.json")["brief"],
        upstream_state={},
        candidate={
            "blocks": [block("ARDUINO_HEADERS"), block("PROTOTYPING_AREA")],
            "connections": [
                {
                    "from_block": "ARDUINO_HEADERS",
                    "to_block": "PROTOTYPING_AREA",
                    "signal_type": "power",
                }
            ],
            "assumptions": [],
        },
    )
    flagged = next(d for d in diagnostics if d.code == code)
    assert flagged.severity == "repair_required"
    # The repair is a delete, and it names the block it is about.
    assert len(flagged.evidence) == 1
    assert "prototyping_area" in flagged.evidence[0]
    assert "not a functional block" in flagged.evidence[0]

    # The wording the model chose does not matter; a real function that merely starts with
    # "proto" is not the field.
    assert code in codes([block("ARDUINO_HEADERS"), block("PAD_FIELD")])
    assert code not in codes([block("ARDUINO_HEADERS"), block("PROTOCOL_BRIDGE")])
    assert code not in codes([block("ARDUINO_HEADERS"), block("POWER_CONVERSION")])


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

    # The production policy disables clarifying questions at this stage, so the writer is told
    # to record the choice instead. A recorded, disclosed default is a resolution -- refusing it
    # would make the stage unsatisfiable (no question is representable in that schema).
    disclosed = diagnose_stage(
        "functional_spec",
        brief="Drive a HUB75 panel and an addressable LED string",
        upstream_state={},
        candidate={
            **candidate,
            "assumptions": [
                "The board supplies power to the hub75 panel and the addressable led "
                "string from the input supply (defaulted)"
            ],
        },
    )
    assert "functional_spec_external_load_power_assumed" not in {item.code for item in disclosed}

    # An undisclosed claim (or one that names an unrelated part) does not clear the refusal:
    # the disclosure is what the contract requires, not merely any assumption row.
    for row in (
        "The board supplies power to the panel",
        "The panel is a 64x64 module (defaulted)",
    ):
        still_open = diagnose_stage(
            "functional_spec",
            brief="Drive a HUB75 panel and an addressable LED string",
            upstream_state={},
            candidate={**candidate, "assumptions": [row]},
        )
        assert "functional_spec_external_load_power_assumed" in {
            item.code for item in still_open
        }, row


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


def test_functional_spec_names_a_self_loop_connection():
    """Run 6's `DISPLAY_DRIVE -> DISPLAY_DRIVE` must be a named defect, not a bare schema error.

    The candidate is schema-valid (`BlockConnection` has no self-loop rule), so the only
    feedback the model got was the commit gate's unclassified `self-loop connection: ...`
    line. A repairable semantic defect belongs on the stage's own diagnostic surface, with
    the offending pair as evidence.
    """
    candidate = {
        "blocks": [
            {"name": "MCU", "category": "process", "purpose": "Controller"},
            {"name": "DISPLAY_DRIVE", "category": "drive", "purpose": "Segment drive"},
        ],
        "connections": [
            {
                "from_block": "MCU",
                "to_block": "DISPLAY_DRIVE",
                "signal_type": "other",
                "description": "Segment data",
            },
            {
                "from_block": "DISPLAY_DRIVE",
                "to_block": "DISPLAY_DRIVE",
                "signal_type": "power",
                "description": "Loop",
            },
        ],
        "assumptions": [],
    }

    diagnostics = diagnose_stage(
        "functional_spec",
        brief="A 7-segment display driver",
        upstream_state={},
        candidate=candidate,
    )
    by_code = {item.code: item for item in diagnostics}
    # `_diag` lowercases evidence, exactly as it does for every other functional-spec finding.
    assert by_code["functional_spec_self_loop"].evidence == ["'display_drive' -> 'display_drive'"]
    assert by_code["functional_spec_self_loop"].severity == "repair_required"


def test_live_functional_spec_reports_premature_topology():
    candidate = _load("rp2040_functional_spec_candidate.json")
    codes = _codes("functional_spec", candidate, {"intent": _load("rp2040_intent_candidate.json")})
    assert "functional_spec_premature_topology" in codes


def test_live_architecture_reports_power_block_as_sheet():
    upstream = {"functional_spec": _load("rp2040_functional_spec_candidate.json")}
    codes = _codes("architecture", _load("rp2040_architecture_candidate.json"), upstream)
    assert "architecture_power_block_as_sheet" in codes


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
    assert "architecture_power_block_as_sheet" not in codes
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


def test_architecture_rejects_source_without_headroom_for_external_loads():
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


def test_board_feature_block_is_removed_with_its_connections():
    """A block for a board feature is dropped deterministically, connections and all.

    A live run (proto-shield r2) kept its `PROTOTYPING_AREA` block through the one semantic
    repair round, and the architecture stage then refused twice with "crosses sheets but
    has no inter_sheet_net": those connections can never be mapped, because the pad-field
    requirement owns no port. The pad field survives as the intent's `fabrication` row and
    the sheet is derived from it, so removing the block loses nothing.
    """
    from kicraft.design.stage_semantics import remove_board_feature_blocks

    candidate = {
        "blocks": [
            {"name": "ARDUINO_SHIELD_INTERFACE", "category": "interface", "purpose": "x", "count": 1},
            {"name": "POWER_INPUT", "category": "power", "purpose": "y", "count": 1},
            {"name": "PROTOTYPING_AREA", "category": "interface", "purpose": "z", "count": 1},
        ],
        "connections": [
            {"from_block": "ARDUINO_SHIELD_INTERFACE", "to_block": "POWER_INPUT",
             "signal_type": "power", "description": "supply"},
            {"from_block": "ARDUINO_SHIELD_INTERFACE", "to_block": "PROTOTYPING_AREA",
             "signal_type": "digital", "description": "breakout"},
            {"from_block": "POWER_INPUT", "to_block": "PROTOTYPING_AREA",
             "signal_type": "power", "description": "rail to the field"},
        ],
        "assumptions": [],
    }
    cleaned = remove_board_feature_blocks(candidate)
    assert [b["name"] for b in cleaned["blocks"]] == ["ARDUINO_SHIELD_INTERFACE", "POWER_INPUT"]
    assert [c["to_block"] for c in cleaned["connections"]] == ["POWER_INPUT"]
    # The input is never mutated, and a spec without such a block is returned unchanged.
    assert len(candidate["blocks"]) == 3
    assert remove_board_feature_blocks({"blocks": [{"name": "POWER"}], "connections": []}) == {
        "blocks": [{"name": "POWER"}],
        "connections": [],
    }


def test_a_power_source_obligation_is_named_not_demanded():
    """The brief's power source is not a part the board places, and the demand must say so.

    Live run KC-5CNKJ3 (seed 26) turned "2S Li-ion battery pack" into a physical obligation for
    `battery-pack`. No placed part can implement it -- the pack is off-board, and the connector
    that honestly implements the input carries no MPN to prove an uncovered class with -- so the
    BOM unit burned all four attempts and died `unit_repair_exhausted`. The corpus's battery-input
    designs that pass BOM carry no such obligation: the connector requirement alone.
    """
    source = _codes(
        "intent",
        {"obligations": [{"kind": "physical", "component_class": "li-ion-battery-pack"}]},
    )
    assert "intent_obligation_class_unrealizable" in source

    # The mate the board does carry is a real demand, and an unplaceable-but-real category the
    # library has never covered stays unflagged.
    clean = _codes(
        "intent",
        {
            "obligations": [
                {"kind": "physical", "component_class": "coin-cell-holder"},
                {"kind": "physical", "component_class": "gps-module"},
            ]
        },
    )
    assert "intent_obligation_class_unrealizable" not in clean


def test_quantity_row_subject_must_name_the_class_it_counts():
    """A count binds in the writer's own spelling; a class-shaped count with no class is refused.

    The model writes counts in prose ("two JST-XH connectors", "2 BNC connectors"). Before the
    binding rule existed the gate compared the prose to the class character by character, so 613
    of 634 committed intents carried a count no gate could enforce -- and when the count was the
    only row naming its class, no demand for that class at all.
    """
    bound = _codes(
        "intent",
        {
            "obligations": [
                {"kind": "physical", "component_class": "jst-xh-connector"},
                {"kind": "quantity", "subject": "JST-XH connectors", "minimum": 2},
            ]
        },
    )
    assert "intent_quantity_subject_unbound" not in bound

    # A role spelling resolves through the library's aliases, so no repair is asked for.
    role = _codes(
        "intent",
        {
            "obligations": [
                {"kind": "physical", "component_class": "led"},
                {"kind": "quantity", "subject": "status led", "minimum": 2},
            ]
        },
    )
    assert "intent_quantity_subject_unbound" not in role

    # A count of a PROPERTY of one part is the writer's own business: eight pins on one header
    # must never be nudged towards eight headers.
    for subject in ("pins on the 0.1 inch header", "relay channels"):
        property_count = _codes(
            "intent",
            {
                "obligations": [
                    {"kind": "physical", "component_class": "pin-header"},
                    {"kind": "quantity", "subject": subject, "minimum": 8},
                ]
            },
        )
        assert "intent_quantity_subject_unbound" not in property_count

    # A count that names a part class the slot never carries: the shape that silently dropped
    # both connectors of a motor driver from the checklist.
    orphan = _codes(
        "intent",
        {"obligations": [{"kind": "quantity", "subject": "jst-xh connectors", "minimum": 2}]},
    )
    assert "intent_quantity_subject_unbound" in orphan


def test_unstated_dc_input_gets_a_defaulted_two_position_screw_terminal():
    """A supply voltage with no entry path gets one connector and says so in assumptions."""
    from kicraft.design.stage_semantics import complete_unstated_power_input

    brief = (
        "An ESP32-C3 module actuator driver: a DRV8833 dual H-bridge, an 18 V DC input, "
        "two JST-XH connectors, and a secondary status LED. Use a two-layer stack-up."
    )
    candidate = {
        "obligations": [
            {"kind": "physical", "component_class": "jst-xh-connector"},
            {
                "kind": "quantitative",
                "quantity": "input voltage",
                "relation": "equal",
                "value": 18.0,
                "unit": "V DC",
            },
        ]
    }

    completed = complete_unstated_power_input(brief, candidate)

    assert [row["component_class"] for row in completed["obligations"] if row["kind"] == "physical"] == [
        "jst-xh-connector",
        "screw-terminal",
    ]
    assert completed["assumptions"] == [
        "Power input: 2-position screw terminal for the 18 V DC supply (defaulted)"
    ]
    # The signal connector the brief names is not an entry path, and the completion is idempotent.
    assert complete_unstated_power_input(brief, completed) == completed


@pytest.mark.parametrize(
    ("brief", "obligations"),
    [
        ("A board with a 12 V DC barrel jack input", []),
        ("A logger powered from a 2S Li-ion battery pack", []),
        ("A USB-C 5 V input sensor node", []),
        ("A 5 V header from the host board", []),
        (
            "An 18 V DC input board",
            [{"kind": "physical", "component_class": "screw-terminal"}],
        ),
    ],
)
def test_power_entry_default_stays_silent_when_the_entry_is_already_described(
    brief, obligations
):
    """Never invent a carrier over one the brief or the writer already named."""
    from kicraft.design.stage_semantics import complete_unstated_power_input

    candidate = {"obligations": list(obligations)}
    assert complete_unstated_power_input(brief, candidate) == candidate


def test_rail_above_the_reviewed_part_rating_is_refused_at_architecture():
    """The 18 V rail on a DRV8833's VM pin is refused where it is written, not at build time.

    The reviewed record carries ``motor_supply_max_v`` 10.8 V, so the architecture stage can say
    so and the writer can make the rail a regulated one (or pick a part rated for the input).
    Surprise-me seed 37 declared ``VIN_18V`` 18 V straight onto the driver's supply and nothing
    objected until this check existed.
    """
    over = {
        "power": {"rails": {"VIN_18V": {"voltage": 18.0, "from": "input.positive"}}},
        "requirements": [
            {
                "id": "driver",
                "role": "driver",
                "family": "dual-dc-motor-driver",
                "exact_part": "DRV8833PWPR",
                "supply": "VIN_18V",
            }
        ],
    }
    diagnostics = diagnose_stage(
        "architecture", brief="an 18 V motor driver", upstream_state={}, candidate=over
    )
    flagged = [d for d in diagnostics if d.code == "architecture_supply_exceeds_part_rating"]
    assert flagged
    assert "10.8" in flagged[0].evidence[0]
    assert "steps this rail down" in flagged[0].evidence[0]

    # A regulated rail inside the part's rating is what the design should have said.
    in_range = {
        **over,
        "power": {"rails": {"VIN_9V": {"voltage": 9.0, "from": "reg.output"}}},
        "requirements": [{**over["requirements"][0], "supply": "VIN_9V"}],
    }
    assert "architecture_supply_exceeds_part_rating" not in {
        d.code
        for d in diagnose_stage(
            "architecture", brief="an 18 V motor driver", upstream_state={}, candidate=in_range
        )
    }


def test_usb_data_edge_without_a_five_volt_rail_gets_one_declared():
    """A native-USB socket carries VBUS: state the rail rather than lose the socket.

    Two refusals on the seed-37 drafts had this one cause — the compiler writes the socket for a
    `usb_dm`/`usb_dp` edge, and with no declared ~5 V rail it drops the edge
    (`usb_connector_supply_unknown`), after which the MCU's USB pins read as unwired.
    """
    from kicraft.design.stage_semantics import complete_usb_socket_rail

    candidate = {
        "power": {"rails": {"+3V3": {"voltage": 3.3, "from": "buck.output"}}},
        "signals": [
            {"name": "USB_DM", "from": "mcu.usb_dm", "to": "edge:USB"},
            {"name": "USB_DP", "from": "mcu.usb_dp", "to": "edge:USB"},
        ],
        "assumptions": ["native USB programming (defaulted)"],
    }

    completed = complete_usb_socket_rail(candidate)

    assert completed["power"]["rails"]["VBUS"] == {"voltage": 5.0, "from": None}
    assert completed["assumptions"][-1].endswith("(defaulted)")
    assert "USB" in completed["assumptions"][-1]
    assert complete_usb_socket_rail(completed) == completed  # idempotent

    # A rail the draft sources from a requirement it never declared cannot resolve — the socket
    # is compiler-created, so naming it is the refusal that repeated in every round of the last
    # live draft. The completion re-states the rail as the host's.
    broken = {
        **candidate,
        "power": {"rails": {"VBUS": {"voltage": 5.0, "from": "usb.vbus"}}},
    }
    repaired = complete_usb_socket_rail(broken)
    assert repaired["power"]["rails"]["VBUS"] == {"voltage": 5.0, "from": None}

    # A rail sourced from a requirement the draft DOES declare is left exactly as written.
    declared = {
        **candidate,
        "requirements": [{"id": "usb", "family": "usb-c-receptacle", "sheet": "MAIN"}],
        "power": {"rails": {"VBUS": {"voltage": 5.0, "from": "usb.vbus"}}},
    }
    assert complete_usb_socket_rail(declared) == declared
    no_usb = {
        "power": {"rails": {"+3V3": {"voltage": 3.3, "from": "buck.output"}}},
        "signals": [{"name": "LED", "from": "mcu.io1", "to": "led.anode"}],
    }
    assert complete_usb_socket_rail(no_usb) == no_usb


def test_derivation_refusals_reach_the_repair_path_as_diagnostics():
    """A design-contract refusal is a design defect, not a dead end.

    `derive_architecture` used to be the only place these contracts were enforced, and the loop
    answers a refusal there with a single from-scratch retry and then fails the stage (four live
    drafts, no candidate). Reported as repairable diagnostics they join the ordinary correction
    path, which keeps the rest of the draft and carries the whole defect list at once.
    """
    candidate = {
        "power": {"rails": {"VBUS": {"voltage": 5.0, "from": "ghost.vbus"}}},
        "sheets": [{"name": "MAIN", "stem": "MAIN", "role": "mcu", "function": "the board"}],
        "requirements": [
            {
                "id": "mcu",
                "sheet": "MAIN",
                "role": "mcu_core",
                "family": "generic-header",
                "parameters": {"rows": 1, "gender": "male"},
                "functional_blocks": [],
            }
        ],
        "signals": [{"name": "GPIO", "from": "mcu.pin1", "to": "edge:IO"}],
    }
    diagnostics = diagnose_stage(
        "architecture", brief="a header breakout", upstream_state={}, candidate=candidate
    )
    assert [d.code for d in diagnostics] == ["unknown_signal_requirement"]
    assert diagnostics[0].severity == "repair_required"
    assert "undeclared requirement" in diagnostics[0].message


def test_supply_over_rating_reads_the_derived_architecture_shape():
    """The response contract derives the slot before diagnosis, so the check must read that shape.

    Live walkthrough (2026-09-25, seed 37): an 18 V rail landed on a DRV8833 whose reviewed
    ``vm`` maximum is 10.8 V, and the stage reported zero diagnostics -- the check read
    ``power.rails`` and ``requirement.supply``, both of which the derivation removes.
    """
    from kicraft.design.stage_semantics import _architecture_supply_over_rating

    derived = {
        "rail_voltages": {"VIN18": 18.0, "+3V3": 3.3, "GND": 0.0},
        "requirements": [
            {
                "id": "driver",
                "role": "driver",
                "family": "dual-dc-motor-driver",
                "exact_part": "DRV8833PWPR",
                "ports": {"vm": "VIN18", "gnd": "GND"},
            },
            {
                "id": "regulator",
                "role": "regulator",
                "family": "tps54331-adjustable",
                "exact_part": "TPS54331DDAR",
                "ports": {"input": "VIN18", "output": "+3V3", "gnd": "GND"},
            },
        ],
    }
    rows = _architecture_supply_over_rating(derived)
    assert [row.code for row in rows] == ["architecture_supply_exceeds_part_rating"]
    assert "vin18" in rows[0].evidence[0].casefold() and "10.8" in rows[0].evidence[0]

    # The model's own shape still works: `supply` against `power.rails`.
    stated = {
        "power": {"rails": {"VIN18": {"voltage": 18.0}}},
        "requirements": [
            {
                "id": "driver",
                "family": "dual-dc-motor-driver",
                "exact_part": "DRV8833PWPR",
                "supply": "VIN18",
            }
        ],
    }
    assert [row.code for row in _architecture_supply_over_rating(stated)] == [
        "architecture_supply_exceeds_part_rating"
    ]

    # A part whose reviewed input range covers the rail (TPS54331: 3.5-28 V) is not flagged,
    # and neither is a rail that is simply not bound to any supply port.
    within = {
        "rail_voltages": {"VIN18": 18.0},
        "requirements": [
            {
                "id": "regulator",
                "role": "regulator",
                "exact_part": "TPS54331DDAR",
                "ports": {"input": "VIN18"},
            }
        ],
    }
    assert _architecture_supply_over_rating(within) == []
