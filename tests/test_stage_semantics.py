import json
from pathlib import Path

import pytest

from kicraft.design.stage_semantics import (
    complete_unsourced_external_rails,
    diagnose_stage,
    external_load_budget_stated,
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


@pytest.mark.parametrize("fault", ["missing_source", "unconnected_source", "signal_only", "denied_operation", None])
def test_battery_operation_requires_a_power_path_to_a_load(fault):
    blocks = [
        {"name": "HOST", "category": "power", "purpose": "Accept host 5 V"},
        {"name": "MCU", "category": "process", "purpose": "Process measurements"},
        {"name": "REGULATION", "category": "power", "purpose": "Supply the logic rail"},
    ]
    connections = [
        {"from_block": "HOST", "to_block": "REGULATION", "signal_type": "power"},
        {"from_block": "REGULATION", "to_block": "MCU", "signal_type": "power"},
    ]
    if fault != "missing_source":
        blocks.append(
            {"name": "PACK", "category": "power", "purpose": "Accept battery power"}
        )
    if fault not in {"missing_source", "unconnected_source"}:
        connections.append({
            "from_block": "PACK", "to_block": "REGULATION",
            "signal_type": "digital" if fault == "signal_only" else "power",
        })
    candidate = {
        "blocks": blocks, "connections": connections,
        "assumptions": ["Battery chemistry and charging details are unspecified (defaulted)"],
    }
    if fault == "denied_operation":
        candidate["assumptions"].append(
            "No battery power path is included because charging is unspecified (defaulted)"
        )
    diagnostics = diagnose_stage(
        "functional_spec", brief="A battery-powered measurement node with a host 5 V input",
        upstream_state={}, candidate=candidate,
    )
    omitted = {
        diagnostic.code for diagnostic in diagnostics
        if diagnostic.severity == "repair_required"
    }
    assert ("functional_spec_battery_operation_omitted" in omitted) == (fault is not None)


@pytest.mark.parametrize("brief", [
    "A host-powered battery voltage monitor",
    "A host-powered monitor, not battery-powered",
    "A non-battery-powered monitor",
    "A monitor not powered from a battery",
])
def test_battery_monitoring_does_not_imply_battery_powered_operation(brief):
    candidate = {
        "blocks": [
            {"name": "HOST", "category": "power", "purpose": "Accept host 5 V"},
            {"name": "SENSOR", "category": "sense", "purpose": "Measure external battery voltage"},
        ],
        "connections": [
            {"from_block": "HOST", "to_block": "SENSOR", "signal_type": "power"},
        ],
    }
    diagnostics = diagnose_stage(
        "functional_spec", brief=brief,
        upstream_state={}, candidate=candidate,
    )
    assert "functional_spec_battery_operation_omitted" not in {d.code for d in diagnostics}


@pytest.mark.parametrize("fault", ["missing_sensor", "unconnected_sensor", "reversed", "power_only", None])
def test_ambient_light_node_requires_measurement_acquisition(fault):
    blocks = [
        {"name": "MCU", "category": "process", "purpose": "Process ambient-light-node functions"}
    ]
    connections = []
    if fault != "missing_sensor":
        blocks.append({
            "name": "ACQUISITION", "category": "sense",
            "purpose": "Acquire ambient light measurements",
        })
    if fault not in {"missing_sensor", "unconnected_sensor"}:
        connections.append({
            "from_block": "MCU" if fault == "reversed" else "ACQUISITION",
            "to_block": "ACQUISITION" if fault == "reversed" else "MCU",
            "signal_type": "power" if fault == "power_only" else "analog",
        })
    diagnostics = diagnose_stage(
        "functional_spec", brief="An ambient-light node",
        upstream_state={}, candidate={"blocks": blocks, "connections": connections},
    )
    assert (
        "functional_spec_ambient_light_acquisition_omitted" in {d.code for d in diagnostics}
    ) == (fault is not None)


def test_onboard_power_led_is_not_two_external_five_volt_loads():
    upstream = {
        "intent": {"goal": "A 12 V board driving a 3 W power LED"},
        "functional_spec": {
            "blocks": [{
                "name": "POWER_LED_DISPLAY", "category": "drive",
                "purpose": "Regulate and drive the onboard 3 W LED",
            }],
            "connections": [{
                "from_block": "DC_INPUT", "to_block": "POWER_LED_DISPLAY",
                "signal_type": "power",
            }],
        },
    }
    assert "architecture_external_load_current_unspecified" not in _codes(
        "architecture", {"rail_voltages": {"VIN": 12.0}}, upstream
    )


def test_distinct_external_loads_are_identified_by_function_not_block_spelling():
    upstream = {
        "intent": {"goal": "Power a HUB75 panel and an LED strip"},
        "functional_spec": {
            "blocks": [
                {"name": "LOAD_A", "category": "drive", "purpose": "HUB75 panel power"},
                {"name": "LOAD_B", "category": "drive", "purpose": "WS2812 LED strip power"},
            ],
            "connections": [
                {"from_block": "INPUT", "to_block": target, "signal_type": "power"}
                for target in ("LOAD_A", "LOAD_B")
            ],
        },
    }
    assert "architecture_external_load_current_unspecified" in _codes(
        "architecture", {"rail_voltages": {"VBUS": 5.0}}, upstream
    )
    answered = {
        **upstream,
        "_stage_answers": [{
            "text": "How much current must the board supply to the external 5 V loads?",
            "answer": "Up to 2 A",
        }],
    }
    assert "architecture_external_load_current_unspecified" not in _codes(
        "architecture", {"rail_voltages": {"VBUS": 5.0}}, answered
    )


def test_external_load_budget_does_not_treat_logic_regulator_rating_as_load_budget():
    candidate = {
        "topologies": {"POWER": "Buck VBUS to 3.3 V at 1 A for logic"},
        "requirements": [{"id": "hub75", "family": "hub75-level-shift-interface"}],
    }
    assert not external_load_budget_stated(candidate)
    assert external_load_budget_stated("The HUB75 panel and LED string draw up to 2 A at 5 V.")


@pytest.mark.parametrize("current", ["1.5 A", "250.5 mA"])
def test_external_load_budget_preserves_decimal_current(current):
    assert external_load_budget_stated(f"The HUB75 panel draws {current}.")
    assert not external_load_budget_stated(
        f"The HUB75 panel is external. The logic regulator supplies {current}."
    )


@pytest.mark.parametrize(
    ("answer", "source_current", "no_headroom"),
    [("1 A", "2 A", False), ("500 mA", "0.5 A", True)],
)
def test_external_load_capacity_uses_only_relevant_answer_current(
    answer, source_current, no_headroom
):
    upstream = {
        "functional_spec": {
            "blocks": [
                {"name": "PANEL", "purpose": "HUB75 panel power"},
                {"name": "STRIP", "purpose": "LED strip power"},
            ],
            "connections": [
                {"to_block": target, "signal_type": "power"}
                for target in ("PANEL", "STRIP")
            ],
        },
        "_stage_answers": [
            {
                "text": "Is the external load current 20 A? State the actual current.",
                "answer": answer,
            },
            {"text": "What current does the motor draw?", "answer": "20 A"},
        ],
    }
    candidate = {"topologies": {"INPUT": f"USB input 5 V / {source_current}"}}
    codes = _codes("architecture", candidate, upstream)
    assert "architecture_external_load_current_unspecified" not in codes
    assert ("architecture_external_load_source_has_no_headroom" in codes) is no_headroom
    assert "architecture_external_load_source_capacity_unspecified" not in codes


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


def test_functional_spec_flags_explicit_defaults_and_reads_no_topology_prose():
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
    # Behavioural prose is not a technology commitment. This candidate names its technologies
    # only in `purpose`/`description`/`assumptions`; the check reads the block names the writer
    # undertakes to realize, so it stays silent here (replay 2026-09-26: 72 of 73 firings were on
    # boards that shipped).
    assert "functional_spec_premature_topology" not in by_code
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


def _ground_flow_candidate(ground_connections):
    return {
        "blocks": [
            {"name": "BNC_INPUT", "category": "interface", "purpose": "Input connector"},
            {"name": "RC_FILTER", "category": "process", "purpose": "Adjustable low-pass"},
            {"name": "BNC_OUTPUT", "category": "interface", "purpose": "Output connector"},
        ],
        "connections": [
            {
                "from_block": "BNC_INPUT",
                "to_block": "RC_FILTER",
                "signal_type": "analog",
                "description": "Filtered input",
            },
            *ground_connections,
        ],
        "assumptions": [],
    }


def test_ground_flow_accepts_a_block_that_sources_the_ground_reference():
    """A block that *originates* the ground net is grounded; it cannot also be its own sink.

    The live false refusal (2026-09-25 passive RC filter, which shipped): the spec listed
    `BNC_INPUT -> BNC_OUTPUT` and `BNC_INPUT -> RC_FILTER` as ground and recorded the common
    ground as an assumption. Reading only the `to_block` endpoints made `BNC_INPUT` the one
    block with no ground flow.
    """
    candidate = _ground_flow_candidate(
        [
            {
                "from_block": "BNC_INPUT",
                "to_block": "BNC_OUTPUT",
                "signal_type": "ground",
                "description": "Common ground",
            },
            {
                "from_block": "BNC_INPUT",
                "to_block": "RC_FILTER",
                "signal_type": "ground",
                "description": "Filter ground",
            },
        ]
    )
    codes = _codes("functional_spec", candidate)
    assert "functional_spec_partial_ground_flow" not in codes


def test_ground_flow_still_refuses_a_block_with_no_ground_participation():
    """The defect the check exists for: a functional block in no ground connection at all."""
    candidate = _ground_flow_candidate(
        [
            {
                "from_block": "BNC_INPUT",
                "to_block": "RC_FILTER",
                "signal_type": "ground",
                "description": "Filter ground",
            }
        ]
    )
    diagnostics = diagnose_stage(
        "functional_spec",
        brief="A passive RC low-pass filter breakout with two BNC connectors",
        upstream_state={},
        candidate=candidate,
    )
    by_code = {item.code: item for item in diagnostics}
    assert by_code["functional_spec_partial_ground_flow"].evidence == ["bnc_output"]


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


def _retained_same_voltage_conversion():
    obligation = {
        "kind": "conversion",
        "original_obligation_id": "voltage-conversion",
        "input_kind": "5 V USB-C power",
        "output_kind": "5 V DC power",
        "behavior": "Provide a 5 V output from the 5 V input.",
    }
    upstream = {
        "intent": {"obligations": [obligation]},
        "functional_spec": {
            "obligations": [dict(obligation)],
            "blocks": [{"name": "FIVE_VOLT_POWER_PATH", "purpose": "Provide 5 V"}],
        },
    }
    candidate = {
        "rail_voltages": {"GND": 0.0},
        "requirements": [{
            "id": "five_volt_path",
            "sheet": "POWER",
            "role": "power_input",
            "family": "usb-c-power-sink",
            "exact_part": "USB-C-5V-SINK",
            "parameters": {},
            "ports": {"gnd": "GND", "vbus": "PROTECTED_5V"},
            "obligations": [dict(obligation)],
            "functional_blocks": ["FIVE_VOLT_POWER_PATH"],
        }],
    }
    return upstream, candidate


def _power_owner_findings(upstream, candidate):
    return [
        diagnostic for diagnostic in diagnose_stage(
            "architecture", brief="", upstream_state=upstream, candidate=candidate
        )
        if diagnostic.code == "architecture_unowned_power_conversion"
    ]


def test_retained_same_voltage_conversion_rejects_connector_owner_before_bom():
    upstream, candidate = _retained_same_voltage_conversion()
    findings = _power_owner_findings(upstream, candidate)
    assert len(findings) == 1
    assert findings[0].severity == "repair_required"


@pytest.mark.parametrize("defect", ["missing_input", "shorted_output", "distribution", "connector"])
def test_committed_conversion_needs_a_distinct_physical_power_path(defect):
    upstream, candidate = _retained_same_voltage_conversion()
    owner = candidate["requirements"][0]
    owner.update(
        role="regulator", family="buck-boost-regulator", exact_part=None,
        ports={"input": "USB_VBUS", "output": "PROTECTED_5V", "gnd": "GND"},
    )
    if defect == "missing_input":
        owner["ports"].pop("input")
    elif defect == "shorted_output":
        owner["ports"]["output"] = "USB_VBUS"
    elif defect == "distribution":
        owner["family"] = "power-distribution"
    else:
        owner["family"] = "usb-c-power-sink"
    assert _power_owner_findings(upstream, candidate)


def test_conversion_can_be_owned_by_separate_model_owned_regulator():
    upstream, candidate = _retained_same_voltage_conversion()
    connector = candidate["requirements"][0]
    obligation = connector.pop("obligations")
    connector["functional_blocks"] = ["USB_INPUT"]
    candidate["requirements"].append({
        "id": "converter",
        "sheet": "POWER",
        "role": "regulator",
        "family": "buck-boost-regulator",
        "ports": {"vin": "USB_VBUS", "vout": "PROTECTED_5V", "gnd": "GND"},
        "obligations": obligation,
        "functional_blocks": ["FIVE_VOLT_POWER_PATH"],
    })
    assert not _power_owner_findings(upstream, candidate)


@pytest.mark.parametrize("input_voltage", [5, 12])
def test_same_voltage_path_can_be_protection_but_voltage_change_needs_regulator(input_voltage):
    upstream, candidate = _retained_same_voltage_conversion()
    owner = candidate["requirements"][0]
    owner.update(
        role="power_input", family="reverse-polarity-pmos", exact_part="AONR21357",
        ports={"input": "USB_VBUS", "output": "PROTECTED_5V", "gnd": "GND"},
    )
    for obligation in (
        upstream["intent"]["obligations"][0],
        upstream["functional_spec"]["obligations"][0],
        owner["obligations"][0],
    ):
        obligation["input_kind"] = f"{input_voltage} V DC power"
    assert bool(_power_owner_findings(upstream, candidate)) is (input_voltage != 5)


@pytest.mark.parametrize("shorted", [False, True])
def test_same_voltage_fused_distribution_owns_its_real_branch_ports(shorted):
    upstream, candidate = _retained_same_voltage_conversion()
    owner = candidate["requirements"][0]
    owner.update(
        family="fused-power-distribution", exact_part=None,
        ports={
            "input": "USB_VBUS", "gnd": "GND", "output1": "LOAD1",
            "output2": "GND" if shorted else "LOAD2",
        },
    )
    assert bool(_power_owner_findings(upstream, candidate)) is shorted


def test_signal_power_conversion_does_not_impose_a_regulator_role():
    upstream, candidate = _retained_same_voltage_conversion()
    owner = candidate["requirements"][0]
    owner.update(
        role="analog_block", family="rf-amplifier", exact_part=None,
        ports={"input": "RF_IN", "output": "RF_OUT", "gnd": "GND"},
    )
    for obligation in (
        upstream["intent"]["obligations"][0],
        upstream["functional_spec"]["obligations"][0],
        owner["obligations"][0],
    ):
        obligation.update(input_kind="RF power", output_kind="amplified RF power")
    assert not _power_owner_findings(upstream, candidate)


@pytest.mark.parametrize("conversion", [None, ("analog voltage", "digital I2C data")])
def test_connector_passthrough_and_nonpower_conversion_do_not_imply_regulator(conversion):
    upstream, candidate = _retained_same_voltage_conversion()
    if conversion is None:
        upstream["intent"]["obligations"] = []
        upstream["functional_spec"]["obligations"] = []
        candidate["requirements"][0]["obligations"] = []
    else:
        for obligation in (
            upstream["intent"]["obligations"][0],
            upstream["functional_spec"]["obligations"][0],
            candidate["requirements"][0]["obligations"][0],
        ):
            obligation["input_kind"], obligation["output_kind"] = conversion
    assert not _power_owner_findings(upstream, candidate)


def _relay_rail_contract_candidate():
    # The retained E6 requirement binds the on-board coil supply to +24V
    # despite selecting the lowerer's 5 V motor-voltage contract.
    return {
        "rail_voltages": {"+24V": 24.0, "+3V3": 3.3, "GND": 0.0},
        "requirements": [{
            "id": "relay_driver", "sheet": "RELAY OUTPUTS", "role": "driver",
            "family": "relay-output", "exact_part": "ULN2003ADR",
            "parameters": {"channel_count": 2, "motor_voltage_v": 5.0, "load_current_a": 0.1},
            "ports": {
                "gnd": "GND", "vmotor": "+24V",
                "control_1": "RELAY_CONTROL_1", "control_2": "RELAY_CONTROL_2",
                "relay_1_common": "RELAY_1_COMMON",
                "relay_1_normally_open": "RELAY_1_NO",
                "relay_1_normally_closed": "RELAY_1_NC",
                "relay_2_common": "RELAY_2_COMMON",
                "relay_2_normally_open": "RELAY_2_NO",
                "relay_2_normally_closed": "RELAY_2_NC",
            },
        }],
    }


def test_relay_coil_rail_contract_rejects_retained_contradiction_at_consumer():
    from kicraft.server.stage_contracts import StageSchemaError, _validate_power_requirement_contracts

    candidate = _relay_rail_contract_candidate()
    with pytest.raises(StageSchemaError) as exc:
        _validate_power_requirement_contracts(candidate, {})
    assert exc.value.diagnostic["code"] == "unrealizable_power_requirement"
    finding = exc.value.diagnostic["evidence"][0]
    assert finding["code"] == "architecture_relay_coil_supply_mismatch"
    assert candidate["requirements"][0]["ports"]["vmotor"] == "+24V"
    assert candidate["rail_voltages"]["+24V"] == 24.0


@pytest.mark.parametrize("case", ["valid_5v", "contact_24v", "offboard", "unknown"])
def test_relay_coil_rail_contract_does_not_conflate_contacts_or_unknowns(case):
    from kicraft.design.stage_semantics import architecture_power_requirement_diagnostics

    candidate = _relay_rail_contract_candidate()
    requirement = candidate["requirements"][0]
    if case in {"valid_5v", "contact_24v"}:
        candidate["rail_voltages"]["+5V"] = 5.0
        requirement["ports"]["vmotor"] = "+5V"
    if case == "contact_24v":
        requirement["ports"]["relay_1_common"] = "+24V"
    elif case == "offboard":
        requirement["family"] = "uln2003-actuator"
        requirement["ports"] = {
            "vmotor": "+24V", "gnd": "GND",
            "control_1": "RELAY_CONTROL_1", "control_2": "RELAY_CONTROL_2",
            "load_1_positive": "+24V", "load_1_negative": "COIL_1_LOW",
            "load_2_positive": "+24V", "load_2_negative": "COIL_2_LOW",
        }
    elif case == "unknown":
        candidate["rail_voltages"].pop("+24V")
    assert architecture_power_requirement_diagnostics({}, candidate) == []
    if case == "unknown":
        assert "+24V" not in candidate["rail_voltages"]
        assert requirement["ports"]["vmotor"] == "+24V"


def test_relay_coil_conflict_survives_independently_invalid_lowering_contract():
    from kicraft.server.stage_contracts import StageSchemaError, _validate_power_requirement_contracts

    candidate = _relay_rail_contract_candidate()
    candidate["requirements"][0]["parameters"]["channel_count"] = 9
    with pytest.raises(StageSchemaError) as exc:
        _validate_power_requirement_contracts(candidate, {})
    assert exc.value.diagnostic["evidence"][0]["code"] == "architecture_relay_coil_supply_mismatch"


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

    # The check needs a reviewed regulator; topology prose alone proves nothing.
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


    # The bounded reviewed buck is enough for this producer check.
    complete_regulator = {
        **_with_3v3_producer("tlv62569-3v3@2"),
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
        "_stage_answers": [{
            "text": "What is the external load current budget?",
            "answer": "5A",
        }],
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
        "recipe_resolution": [{"requirement_id": "buck", "recipe": "tlv62569-3v3@2"}],
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
        "_stage_answers": [{
            "text": "What is the maximum external load current?",
            "answer": "Board supplies both loads, 5 A maximum",
        }],
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


def test_bom_prose_cannot_refuse_a_sheet_that_states_what_it_implements():
    """A sheet carrying a realized requirement is not refused for a word in its function prose.

    Live run 1109 (brief 2, 2026-09-30): the POWER INDICATOR sheet's function reads "a visible
    low-current indication that 5 V controller power is present" -- "controller" names the MCU's work
    on another sheet -- and the sheet's own requirement (the reviewed LED build) was realized, yet the
    BOM stage refused the run with `bom_architecture_role_unsupported`. Prose is an effect a sheet
    names, never a role it promises once the sheet has stated a non-connector requirement of its own;
    the title path and the requirement-less prose path both still refuse.
    """
    architecture = {
        "sheets": [
            {"name": "ESP32 S3", "function": "Run controller logic and drive the LED matrix."},
            {
                "name": "POWER INDICATOR",
                "function": (
                    "Provide a visible low-current indication that 5 V controller power is present."
                ),
            },
        ],
        "requirements": [
            {
                "id": "mcu",
                "sheet": "ESP32 S3",
                "role": "mcu_core",
                "family": "esp32-s3-module",
                "ports": {"vdd": "+3V3", "gnd": "GND"},
            },
            {
                "id": "power_led",
                "sheet": "POWER INDICATOR",
                "role": "user_io",
                "family": "led-current-resistor",
                "ports": {"drive": "VBUS", "gnd": "GND"},
            },
        ],
    }
    parts = [
        {"ref": "U1", "sheet": "ESP32 S3"},
        {"ref": "D1", "sheet": "POWER INDICATOR"},
        {"ref": "R1", "sheet": "POWER INDICATOR"},
    ]
    assert "bom_architecture_role_unsupported" not in _codes(
        "bom", {"parts": parts}, {"architecture": architecture}
    )

    # The sheet's own *title* still promises its role: a sheet named for an IC role with nothing
    # built on it is refused, and so is prose when the sheet states no requirement at all.
    titled = {
        "sheets": [{"name": "MOTOR DRIVER", "function": "Drive one motor."}],
        "requirements": [],
    }
    assert _codes("bom", {"parts": []}, {"architecture": titled}) == {
        "bom_architecture_role_unsupported"
    }
    prose_only = {
        "sheets": [{"name": "LED DRIVE", "function": "Provide a controller for the matrix."}],
        "requirements": [],
    }
    assert _codes("bom", {"parts": []}, {"architecture": prose_only}) == {
        "bom_architecture_role_unsupported"
    }


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


@pytest.mark.parametrize("part", ["LM35", "LM358", "NE555"])
def test_short_named_parts_are_retained_when_the_writer_omits_them(part):
    from kicraft.design.stage_semantics import complete_intent_classification

    brief = f"A board using {part}, with GPIO12 and a QFN32 package."
    completed = complete_intent_classification(brief, {"named_parts": []})
    assert completed["named_parts"] == [part]


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


def test_prose_modifier_is_not_glued_onto_the_part_identity():
    """``<part>-based`` names the part, not a second one the library cannot own.

    Live cohort 2026-09-26, seeds 52/53/56: the token tail ran under ``re.IGNORECASE`` and
    swallowed the hyphenated English word after a part name, so ``named_part_tokens`` returned
    ``"LM358-based"`` / ``"CH32V003-based"`` as exact identities. Intent classification then
    recorded a phantom named part, and the architecture gate refused the board with
    ``missing_recipe_requirement`` / ``unsupported_protected_variant`` for a part that does not
    exist. The upper/digit order-code tail is genuinely part of the identity and stays whole.
    """
    from kicraft.design.stage_semantics import complete_intent_classification
    from kicraft.design.synthesis.validation import named_part_tokens

    assert named_part_tokens(["An LM358-based hall-effect amplifier board"]) == {"lm358": "LM358"}
    assert named_part_tokens(["Build a CH32V003-based LED driver board"]) == {
        "ch32v003": "CH32V003"
    }
    assert named_part_tokens(["An ESP32-S3-WROOM-1-N16R8 module"]) == {
        "esp32-s3-wroom-1-n16r8": "ESP32-S3-WROOM-1-N16R8"
    }
    completed = complete_intent_classification(
        "An LM358-based hall-effect amplifier board", {"named_parts": ["LM358"]}
    )
    assert completed["named_parts"] == ["LM358"]


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


def test_a_drive_part_may_not_be_powered_from_the_logic_rail():
    """The rating refusal has a cheap escape: move the drive's supply onto the logic rail.

    Live walkthrough (2026-09-25, seed 37): after the 18 V-on-DRV8833 refusal, the correction
    bound the bridge's `vm` to the 3.3 V rail that powers the ESP32-C3 and reported zero
    diagnostics -- the actuators would run from the MCU's regulator. A deliberate shared rail
    is a real design, so a disclosed load-current budget clears it; silence does not.
    """
    from kicraft.design.stage_semantics import _architecture_drive_on_logic_rail

    def candidate(driver_rail: str) -> dict:
        return {
            "rail_voltages": {"+3V3": 3.3, "VMOT": 9.0},
            "requirements": [
                {
                    "id": "mcu",
                    "role": "mcu_core",
                    "exact_part": "ESP32-C3-MINI-1-N4",
                    "ports": {"vdd": "+3V3", "gnd": "GND"},
                },
                {
                    "id": "bridge",
                    "role": "driver",
                    "exact_part": "DRV8833PWPR",
                    "ports": {"vm": driver_rail, "gnd": "GND"},
                },
                {
                    "id": "reg",
                    "role": "regulator",
                    "exact_part": "TPS54331DDAR",
                    "ports": {"input": "VIN18", "output": "+3V3"},
                },
            ],
            "assumptions": [],
        }

    shared = _architecture_drive_on_logic_rail({}, candidate("+3V3"))
    assert [row.code for row in shared] == ["architecture_drive_shares_logic_rail"]
    assert "'+3v3'" in shared[0].evidence[0].casefold()

    # Its own regulated rail: clean.
    assert _architecture_drive_on_logic_rail({}, candidate("VMOT")) == []

    # A deliberate shared rail is allowed once the load's current is disclosed.
    disclosed = candidate("+3V3")
    disclosed["assumptions"] = ["The +3V3 rail feeds the bridge with up to 1.2 A (defaulted)"]
    assert _architecture_drive_on_logic_rail({}, disclosed) == []


def test_a_logic_rail_driver_that_needs_no_load_rail_is_not_flagged():
    """A display/level driver on the logic rail is a real design; only a load-rail part is."""
    from kicraft.design.stage_semantics import _architecture_drive_on_logic_rail

    def candidate(exact_part: str, role: str = "driver") -> dict:
        return {
            "rail_voltages": {"+3V3": 3.3},
            "requirements": [
                {"id": "mcu", "role": "mcu_core", "exact_part": "ESP32-C3-MINI-1-N4",
                 "ports": {"vdd": "+3V3"}},
                {"id": "panel", "role": role, "exact_part": exact_part, "ports": {"vm": "+3V3"}},
            ],
            "assumptions": [],
        }

    # Reviewed limits that name no load domain (a logic-rail part): not this check's business.
    assert _architecture_drive_on_logic_rail({}, candidate("TPS54331DDAR")) == []


def test_an_over_rated_load_gets_its_own_regulated_rail_before_diagnosis():
    """Catching the fault is not enough: the pipeline adds the rail the part can run on.

    Live walkthrough (2026-09-25, seed 37): 18 V DC in, a DRV8833 (`vm` 2.7-10.8 V) driving the
    actuators, and no reviewed dual-H-bridge rated for 18 V -- three drafts restated the
    impossibility and the stage parked. The completion adds a reviewed converter, sets a
    rail inside the part's range, rebinds the part, and discloses it as a default.
    """
    from kicraft.design.stage_semantics import (
        _architecture_supply_over_rating,
        complete_over_rated_supply,
    )
    from kicraft.server.stage_runtime import _normalize_candidate_for_diagnostics

    candidate = {
        "rail_voltages": {"+18V": 18.0, "+3V3": 3.3, "GND": 0.0},
        "power_nets": ["GND", "+3V3", "+18V"],
        "sheets": [{"name": "H BRIDGE", "stem": "H_BRIDGE", "role": "driver", "function": "Drive loads"}],
        "requirements": [
            {
                "id": "bridge",
                "sheet": "H BRIDGE",
                "role": "driver",
                "family": "dual-dc-motor-driver",
                "exact_part": "DRV8833PWPR",
                "parameters": {},
                "ports": {"vm": "+18V", "gnd": "GND", "aout1": "MOTOR_A"},
                "functional_blocks": ["DUAL H-BRIDGE"],
            }
        ],
        "assumptions": [],
    }
    assert [row.code for row in _architecture_supply_over_rating(candidate)] == [
        "architecture_supply_exceeds_part_rating"
    ]

    fixed = complete_over_rated_supply(candidate)

    # A rail inside the part's rated range, sourced from a reviewed converter on the input.
    assert 2.7 <= fixed["rail_voltages"]["BRIDGE_RAIL"] <= 10.8
    assert "+18V" in fixed["rail_voltages"] and fixed["rail_voltages"]["+18V"] == 18.0
    converter = next(row for row in fixed["requirements"] if row["id"] == "bridge_regulator")
    assert converter["parameters"]["output_voltage"] == fixed["rail_voltages"]["BRIDGE_RAIL"]
    assert {
        key: converter["ports"][key] for key in ("input", "output", "gnd")
    } == {"input": "+18V", "output": "BRIDGE_RAIL", "gnd": "GND"}
    bridge = next(row for row in fixed["requirements"] if row["id"] == "bridge")
    assert bridge["ports"]["vm"] == "BRIDGE_RAIL"
    assert any(row["name"] == "BRIDGE REGULATOR" for row in fixed["sheets"])
    assert any("(defaulted)" in row and "BRIDGE_RAIL" in row for row in fixed["assumptions"])

    # The fault is gone from the candidate the checks and the commit see.
    assert _architecture_supply_over_rating(fixed) == []

    # The normalize step (the pipeline's own path) applies it, so diagnosis never sees the fault.
    normalized = _normalize_candidate_for_diagnostics("architecture", candidate, "18 V actuator board", {})
    assert _architecture_supply_over_rating(normalized) == []

    # A design already inside its rating is left exactly as it is.
    within = {**candidate, "rail_voltages": {"+9V": 9.0, "GND": 0.0},
              "requirements": [{**candidate["requirements"][0], "ports": {"vm": "+9V"}}]}
    assert complete_over_rated_supply(within) == within



def test_undervoltage_rgb_supply_uses_its_battery_ancestry_not_usb_vbus():
    """An unsupported 5 V load is not silently put on a 3.3 V rail or borrowed USB."""
    from kicraft.design.stage_semantics import _complete_load_supply_rails

    candidate = {
        "rail_voltages": {
            "BATT_2S": 7.4,
            "+3V3": 3.3,
            "USB_VBUS": 5.0,
            "GND": 0.0,
        },
        "requirements": [
            {
                "id": "battery_input",
                "role": "power_input",
                "family": "jst-xh-connector",
                "ports": {"positive": "BATT_2S", "negative": "GND"},
            },
            {
                "id": "mcu_regulator",
                "role": "regulator",
                "family": "ap63203-3v3",
                "exact_part": "AP63203WU-7",
                "parameters": {"output_voltage": 3.3},
                "ports": {"input": "BATT_2S", "output": "+3V3", "gnd": "GND"},
                "obligation_ids": ["battery-to-3v3"],
            },
            {
                "id": "rp2040",
                "role": "mcu_core",
                "family": "rp2040",
                "exact_part": "RP2040",
                "ports": {"vdd": "+3V3", "gnd": "GND"},
            },
            {
                "id": "usb",
                "role": "connector",
                "family": "usb-c-receptacle",
                "ports": {"vbus": "USB_VBUS", "gnd": "GND"},
            },
            {
                "id": "rgb",
                "role": "user_io",
                "family": "ws2812b",
                "exact_part": "WS2812B-B/T",
                "ports": {"vdd": "+3V3", "gnd": "GND", "din": "RGB_DATA"},
                "obligation_ids": ["rgb-status"],
            },
        ],
        "assumptions": [],
    }

    fixed = _complete_load_supply_rails(json.loads(json.dumps(candidate)))

    assert fixed == candidate

    # Neither a missing nor a branched source lineage may be repaired by borrowing USB VBUS.
    missing = json.loads(json.dumps(candidate))
    missing["requirements"] = [
        row for row in missing["requirements"] if row["id"] != "mcu_regulator"
    ]
    assert _complete_load_supply_rails(missing) == missing
    ambiguous = json.loads(json.dumps(candidate))
    ambiguous["requirements"].append(
        {
            "id": "usb_regulator",
            "role": "regulator",
            "family": "ap63203-3v3",
            "ports": {"input": "USB_VBUS", "output": "+3V3", "gnd": "GND"},
        }
    )
    assert _complete_load_supply_rails(ambiguous) == ambiguous

@pytest.mark.parametrize(
    ("exact_part", "host_rail"), [("CH340C", "HOST_5V"), ("CP2102N", "USB_SUPPLY")]
)
def test_integrated_reviewed_usb_bridge_vbus_is_an_external_source_not_a_load(
    exact_part, host_rail
):
    """A reviewed bridge receptacle keeps arbitrary host VBUS names out of fallback generation."""
    from kicraft.design.stage_semantics import _complete_load_supply_rails

    candidate = {
        "rail_voltages": {"VBAT": 7.4, "+3V3": 3.3, host_rail: 5.0, "GND": 0.0},
        "requirements": [
            {
                "id": "battery_input",
                "role": "power_input",
                "family": "pin-header",
                "ports": {"pin1": "VBAT", "pin2": "GND"},
            },
            {
                "id": "mcu_regulator",
                "role": "regulator",
                "family": "ap63203-3v3",
                "exact_part": "AP63203WU-7",
                "parameters": {"output_voltage": 3.3},
                "ports": {"input": "VBAT", "output": "+3V3", "gnd": "GND"},
            },
            {
                "id": "usb_bridge",
                "role": "bus_interface",
                "family": "usb-uart-bridge",
                "exact_part": exact_part,
                "ports": {"vbus": host_rail, "vdd": "+3V3", "gnd": "GND"},
            },
            {
                "id": "rgb",
                "role": "user_io",
                "family": "ws2812b",
                "exact_part": "WS2812B-B/T",
                "ports": {"vdd": "+3V3", "gnd": "GND", "din": "RGB_DATA"},
            },
        ],
        "assumptions": [],
    }

    fixed = _complete_load_supply_rails(candidate)

    assert next(row for row in fixed["requirements"] if row["id"] == "rgb")["ports"]["vdd"] == "+3V3"
    assert not any(
        row.get("role") == "regulator" and row.get("ports", {}).get("output") == host_rail
        for row in fixed["requirements"]
    )
    assert _complete_load_supply_rails(fixed) == fixed

def test_the_rail_fix_corrects_the_statements_it_invalidates():
    """A candidate whose prose contradicts its bindings is not reviewable.

    Live walkthrough (2026-09-25): after the pipeline rebound the bridge's `vm` to a 10 V rail,
    the writer's own rows still said "the DRV8833 motor supply is connected to the 18 V input",
    and the sheet said "from the 18 V motor supply" -- a review that has to notice the binding
    was 10 V was reading two contradictory stories.
    """
    from kicraft.design.stage_semantics import complete_over_rated_supply

    candidate = {
        "rail_voltages": {"+18V": 18.0, "GND": 0.0},
        "topologies": {
            "DRIVE": "DRV8833 dual H-bridge fed from the 18 V input",
            "POWER": "18 V input converted to 3.3 V by an adjustable buck regulator",
        },
        "sheets": [
            {
                "name": "H BRIDGE",
                "stem": "H_BRIDGE",
                "role": "driver",
                "function": "Drive two external actuators from the 18 V motor supply",
            },
            {
                "name": "POWER CONVERSION",
                "stem": "POWER_CONVERSION",
                "role": "regulator",
                "function": "Convert the 18 V input to a 3.3 V rail for the ESP32-C3 and the "
                "DRV8833 logic side",
            }
        ],
        "requirements": [
            {
                "id": "bridge",
                "sheet": "H BRIDGE",
                "role": "driver",
                "family": "dual-dc-motor-driver",
                "exact_part": "DRV8833PWPR",
                "parameters": {},
                "ports": {"vm": "+18V", "gnd": "GND"},
                "functional_blocks": ["DUAL H-BRIDGE"],
            }
        ],
        "assumptions": [
            "The DRV8833 motor supply is connected to the 18 V input (defaulted)",
            "The 18 V DC input is treated as the actuator supply and the input to a dedicated "
            "3.3 V buck regulator (defaulted)",
            "A green status LED is operated at approximately 2 mA (defaulted)",
        ],
    }

    fixed = complete_over_rated_supply(candidate)
    rows = [str(row).casefold() for row in fixed["assumptions"]]
    text = " ".join(rows)

    # No writer row says the part runs on the 18 V rail any more (the pipeline's own disclosure
    # names both rails on purpose -- it is where the whole picture is stated), and the row that
    # mixed the input voltage with the load's supply is gone.
    disclosure_rows = [row for row in rows if "runs from a regulated" in row]
    assert len(disclosure_rows) == 1, rows
    assert not any(
        "drv8833" in row and "18 v" in row for row in rows if row not in disclosure_rows
    ), rows
    assert not any("actuator supply" in row for row in rows), rows
    # Statements the fix does not invalidate are left alone.
    assert any("status led" in row for row in rows)
    assert not any("18 v" in str(row).casefold() for row in [fixed["topologies"]["DRIVE"]])
    assert "BRIDGE_RAIL" in fixed["topologies"]["DRIVE"]
    sheet = next(row for row in fixed["sheets"] if row["name"] == "H BRIDGE")
    assert "18 V" not in sheet["function"] and "BRIDGE_RAIL" in sheet["function"]
    # And the pipeline states the whole picture in one disclosed row.
    disclosure = next(row for row in fixed["assumptions"] if "(defaulted)" in row and "regulator" in row)
    assert "BRIDGE_RAIL" in disclosure and "10.8 V" in disclosure and "18 V" in disclosure
    # The statements about the board input survive: the fix regulates the *load* rail, and the
    # converter still takes 18 V in.
    conversion_sheet = next(
        row for row in fixed["sheets"] if row["name"] == "POWER CONVERSION"
    )
    # The conversion sentence is intact -- not rewritten into "regulated rail … to a 3.3 V rail".
    assert "Convert the 18 V input to a 3.3 V rail" in conversion_sheet["function"], (
        conversion_sheet["function"]
    )
    assert "18 V input" in fixed["topologies"]["POWER"], fixed["topologies"]["POWER"]


def test_a_drive_on_the_logic_rail_gets_its_own_rail_too():
    """The escape has a deterministic fix as well, so it cannot park either."""
    from kicraft.design.stage_semantics import (
        _architecture_drive_on_logic_rail,
        complete_over_rated_supply,
    )

    candidate = {
        "rail_voltages": {"+18V": 18.0, "+3V3": 3.3, "GND": 0.0},
        "requirements": [
            {"id": "mcu", "role": "mcu_core", "exact_part": "ESP32-C3-MINI-1-N4",
             "ports": {"vdd": "+3V3", "gnd": "GND"}},
            {"id": "bridge", "role": "driver", "family": "dual-dc-motor-driver",
             "exact_part": "DRV8833PWPR", "ports": {"vm": "+3V3", "gnd": "GND"},
             "functional_blocks": ["DUAL H-BRIDGE"]},
        ],
        "sheets": [{"name": "H BRIDGE", "stem": "H_BRIDGE", "role": "driver",
                    "function": "Drive the actuators from the 3.3 V rail"}],
        "assumptions": [],
    }
    assert [row.code for row in _architecture_drive_on_logic_rail({}, candidate)] == [
        "architecture_drive_shares_logic_rail"
    ]

    fixed = complete_over_rated_supply(candidate)

    # Its own rail fits the load and is sourced from the board input.
    assert 2.7 <= fixed["rail_voltages"]["BRIDGE_RAIL"] <= 10.8
    assert next(r for r in fixed["requirements"] if r["id"] == "bridge")["ports"]["vm"] == "BRIDGE_RAIL"
    converter = next(r for r in fixed["requirements"] if r["id"] == "bridge_regulator")
    assert converter["ports"] == {"input": "+18V", "output": "BRIDGE_RAIL", "gnd": "GND"}
    assert _architecture_drive_on_logic_rail({}, fixed) == []


def test_the_rail_rewrite_neither_mangles_words_nor_skips_the_part():
    """Two defects the live draft exposed (2026-09-25).

    The rail name "VIN" was substituted inside the word "driving" ("driVINg"), and the part was
    written as "DRV8833" while the token list held only the order code "DRV8833PWPR", so its own
    false claim ("supplied from the 18 V input") survived the correction.
    """
    from kicraft.design.stage_semantics import complete_over_rated_supply

    candidate = {
        "rail_voltages": {"VIN": 18.0, "+3V3": 3.3, "GND": 0.0},
        "topologies": {
            "DRIVER": "DRV8833 dual H-bridge driving two actuator connectors",
            "POWER": "18 V DC input with buck conversion to 3.3 V; DRV8833 supplied from the 18 V input",
            "CONTROL": "Run actuator control and provide H-bridge control signals from the 18 V input",
        },
        "sheets": [
            {"name": "DUAL H BRIDGE", "stem": "DUAL_H_BRIDGE", "role": "driver",
             "function": "Drive two actuator outputs from the 18 V input"},
        ],
        "requirements": [
            {"id": "hbridge", "sheet": "DUAL H BRIDGE", "role": "driver",
             "family": "dual-dc-motor-driver", "exact_part": "DRV8833PWPR",
             "ports": {"vm": "VIN", "gnd": "GND"}, "functional_blocks": ["DUAL H-BRIDGE"]},
        ],
        "assumptions": [],
    }

    fixed = complete_over_rated_supply(candidate)
    bridge_rail = "HBRIDGE_RAIL"
    driver = next(r for r in fixed["requirements"] if r["id"] == "hbridge")
    assert driver["ports"]["vm"] == bridge_rail

    # No mangled word: the rail name never lands inside "driving".
    assert "driving two actuator connectors" in fixed["topologies"]["DRIVER"]
    assert "dri" + bridge_rail not in fixed["topologies"]["DRIVER"]
    # The part's own claim is corrected, even though the writer shortened the order code.
    assert "DRV8833 supplied from the 18 V input" not in fixed["topologies"]["POWER"]
    assert bridge_rail in fixed["topologies"]["POWER"]
    # A sentence about control signals keeps the board input's voltage: it is not a supply claim.
    assert "18 V input" in fixed["topologies"]["CONTROL"]
    # And the bridge's own sheet now names the rail it runs on.
    assert "18 V" not in fixed["sheets"][0]["function"]
    assert bridge_rail in fixed["sheets"][0]["function"]


@pytest.mark.parametrize("source_name", ["led_output", "motor-driver"])
@pytest.mark.parametrize("declared_rail", [False, True])
def test_generated_regulator_sheet_is_valid_and_owns_its_connections(source_name, declared_rail):
    from kicraft.design.models import Sheet
    from kicraft.design.stage_semantics import _complete_load_supply_rails

    rail = source_name.upper() if declared_rail else "VIN"
    candidate = {
        "rail_voltages": {"VIN": 18.0, "GND": 0.0, **({rail: 3.3} if declared_rail else {})},
        "sheets": [{"name": "LOAD", "stem": "LOAD", "function": "Drive load"}],
        "requirements": [{
            "id": source_name, "sheet": "LOAD", "role": "driver",
            "family": "dual-dc-motor-driver", "exact_part": "DRV8833PWPR",
            "ports": {"vm": rail, "gnd": "GND"}, "functional_blocks": [],
        }, {
            "id": "input", "sheet": "LOAD", "role": "power_input",
            "family": "screw-terminal",
            "ports": {"positive": "VIN", "negative": "GND"},
        }],
        "assumptions": [],
    }
    fixed = _complete_load_supply_rails(candidate)
    sheets = {
        row["name"]: Sheet.model_validate({k: row[k] for k in ("name", "stem", "function")})
        for row in fixed["sheets"]
    }
    converter = next(row for row in fixed["requirements"] if row["role"] == "regulator")
    assert converter["sheet"] in sheets
    assert converter["ports"]["input"] == "VIN"
    assert converter["ports"]["output"] == fixed["requirements"][0]["ports"]["vm"]
    assert 2.7 <= fixed["rail_voltages"][converter["ports"]["output"]] <= 10.8


def test_declared_load_rail_preserves_unsupported_voltage_and_completes_exact_voltage():
    """Fallback generation never retargets a declared rail to a nearby reviewed output."""
    from kicraft.design.stage_semantics import _complete_load_supply_rails

    candidate = {
        "rail_voltages": {"VIN": 18.0, "MOTOR_VIN": 10.8, "+3V3": 3.3, "GND": 0.0},
        "requirements": [
            {
                "id": "input",
                "role": "power_input",
                "family": "screw-terminal",
                "ports": {"positive": "VIN", "negative": "GND"},
            },
            {
                "id": "reg",
                "role": "regulator",
                "family": "tps54331-adjustable",
                "exact_part": "TPS54331DDAR",
                "parameters": {"output_voltage": 3.3},
                "ports": {"input": "VIN", "output": "+3V3"},
            },
            {
                "id": "driver",
                "role": "driver",
                "family": "dual-dc-motor-driver",
                "exact_part": "DRV8833PWPR",
                "ports": {"vm": "MOTOR_VIN", "gnd": "GND"},
                "functional_blocks": ["DUAL H-BRIDGE"],
            },
        ],
        "assumptions": [],
    }

    assert _complete_load_supply_rails(candidate) == candidate

    supported = json.loads(json.dumps(candidate))
    supported["rail_voltages"]["MOTOR_VIN"] = 3.3
    fixed = _complete_load_supply_rails(supported)
    converter = next(
        row for row in fixed["requirements"] if row["id"] == "motor_vin_regulator"
    )
    assert converter["ports"] == {"input": "VIN", "output": "MOTOR_VIN", "gnd": "GND"}
    assert converter["parameters"]["output_voltage"] == 3.3
    assert converter["family"] == "ylptec-k7803m-1000r3"
    assert fixed["rail_voltages"]["MOTOR_VIN"] == 3.3

    ambiguous = json.loads(json.dumps(supported))
    ambiguous["rail_voltages"]["ALT_INPUT"] = 5.0
    ambiguous["requirements"].append(
        {
            "id": "alternate_input",
            "role": "power_input",
            "family": "screw-terminal",
            "ports": {"positive": "ALT_INPUT", "negative": "GND"},
        }
    )
    assert _complete_load_supply_rails(ambiguous) == ambiguous


def test_compiler_generated_regulator_id_normalizes_numeric_rail_and_avoids_collision():
    """A declared rail name is an electrical net, not a requirement identifier."""
    from kicraft.design.models import Architecture
    from kicraft.design.stage_semantics import _complete_load_supply_rails

    rail = "+3.3 V/Logic"
    candidate = {
        "topologies": {},
        "rail_voltages": {"VIN": 18.0, rail: 3.3, "GND": 0.0},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {"name": "POWER INPUT", "stem": "POWER_INPUT", "role": "power_input", "function": "Input"},
            {"name": "LOAD", "stem": "LOAD", "role": "driver", "function": "Drive"},
            {"name": "RESERVED", "stem": "RESERVED", "role": "connector", "function": "Reserve"},
        ],
        "power_nets": ["VIN", rail, "GND"],
        "inter_sheet_nets": [
            {"name": "VIN", "endpoints": [{"sheet": "POWER INPUT", "direction": "output"}]},
            {"name": rail, "endpoints": [{"sheet": "LOAD", "direction": "input"}]},
            {
                "name": "GND",
                "endpoints": [
                    {"sheet": "POWER INPUT", "direction": "bidirectional"},
                    {"sheet": "LOAD", "direction": "bidirectional"},
                ],
            },
        ],
        "requirements": [
            {
                "id": "input",
                "sheet": "POWER INPUT",
                "role": "power_input",
                "family": "screw-terminal",
                "parameters": {"rows": 1},
                "ports": {"positive": "VIN", "negative": "GND"},
            },
            {
                "id": "load",
                "sheet": "LOAD",
                "role": "driver",
                "family": "dual-dc-motor-driver",
                "exact_part": "DRV8833PWPR",
                "ports": {"vm": rail, "gnd": "GND"},
            },
            {
                "id": "rail_3_3_v_logic_regulator",
                "sheet": "RESERVED",
                "role": "connector",
                "family": "screw-terminal",
                "parameters": {"rows": 1},
                "ports": {"positive": "RESERVED", "negative": "GND"},
            },
        ],
        "assumptions": [],
    }
    source_requirements = json.loads(json.dumps(candidate["requirements"]))
    source_nets = {row["name"] for row in candidate["inter_sheet_nets"]}

    fixed = _complete_load_supply_rails(candidate)
    architecture = Architecture.model_validate(fixed)

    converter = next(
        row for row in fixed["requirements"] if row["id"] == "rail_3_3_v_logic_regulator_2"
    )
    assert converter["ports"] == {"input": "VIN", "output": rail, "gnd": "GND"}
    assert fixed["requirements"][:len(source_requirements)] == source_requirements
    assert source_nets <= {net.name for net in architecture.inter_sheet_nets}
    assert rail in architecture.power_nets
    assert fixed["rail_voltages"][rail] == 3.3
    assert len({requirement.id for requirement in architecture.requirements}) == len(
        architecture.requirements
    )


def test_compiler_generated_regulator_id_is_bounded_for_long_requirement_id():
    from kicraft.design.models import CircuitRequirement
    from kicraft.design.stage_semantics import _complete_load_supply_rails

    source_id = "load_" + "x" * 59
    colliding_id = "load_" + "x" * 49 + "_regulator"
    candidate = {
        "rail_voltages": {"VIN": 18.0, "GND": 0.0},
        "sheets": [{"name": "LOAD", "stem": "LOAD", "role": "driver", "function": "Drive"}],
        "requirements": [
            {
                "id": source_id,
                "sheet": "LOAD",
                "role": "driver",
                "family": "dual-dc-motor-driver",
                "exact_part": "DRV8833PWPR",
                "ports": {"vm": "VIN", "gnd": "GND"},
            },
            {
                "id": colliding_id,
                "sheet": "LOAD",
                "role": "connector",
                "family": "screw-terminal",
                "parameters": {"rows": 1},
                "ports": {"positive": "RESERVED", "negative": "GND"},
            },
        ],
        "assumptions": [],
    }

    fixed = _complete_load_supply_rails(candidate)
    converter = fixed["requirements"][-1]

    assert converter["id"].endswith("_2")
    assert len(converter["id"]) == 64
    assert converter["id"] not in {source_id, colliding_id}
    assert len({row["id"] for row in fixed["requirements"]}) == len(fixed["requirements"])
    CircuitRequirement.model_validate(converter)


def test_the_rail_completion_keeps_the_derived_nets_consistent():
    """The commit gate reads the derived nets, so a rebound rail must not leave a stale endpoint.

    Live walkthrough (2026-09-25): the commit refused the architecture candidate with
    "inter-sheet net '+18V' endpoint on sheet 'DUAL H BRIDGE' has no requirement.ports value
    bound to that exact net name" -- the completion had rebound the bridge's supply but the
    derived net list is computed at decode, before that. This runs the gate that refused.
    """
    from kicraft.design import models
    from kicraft.design.stage_semantics import complete_over_rated_supply
    from kicraft.design.synthesis.validation import check_fs_connections_mapped

    candidate = {
        "topologies": {},
        "rail_voltages": {"VIN": 18.0, "+3V3": 3.3, "GND": 0.0},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {"name": "POWER INPUT", "stem": "POWER_INPUT", "role": "power_input", "function": "Input"},
            {"name": "H BRIDGE", "stem": "H_BRIDGE", "role": "driver", "function": "Drive"},
        ],
        "power_nets": ["GND", "VIN"],
        "inter_sheet_nets": [
            {"name": "VIN", "endpoints": [{"sheet": "POWER INPUT", "direction": "output"},
                                          {"sheet": "H BRIDGE", "direction": "input"}]},
            {"name": "GND", "endpoints": [{"sheet": "POWER INPUT", "direction": "bidirectional"},
                                          {"sheet": "H BRIDGE", "direction": "bidirectional"}]},
        ],
        "requirements": [
            {"id": "input", "sheet": "POWER INPUT", "role": "power_input", "family": "screw-terminal",
             "parameters": {"rows": 1}, "ports": {"positive": "VIN", "negative": "GND"},
             "functional_blocks": ["INPUT"]},
            {"id": "hbridge", "sheet": "H BRIDGE", "role": "driver",
             "family": "dual-dc-motor-driver", "exact_part": "DRV8833PWPR", "parameters": {},
             "ports": {"vm": "VIN", "gnd": "GND"}, "functional_blocks": ["DRIVE"]},
        ],
        "assumptions": [],
    }
    spec = models.FunctionalSpec.model_validate(
        {
            "blocks": [
                {"name": "INPUT", "category": "power", "purpose": "Input"},
                {"name": "DRIVE", "category": "drive", "purpose": "Drive"},
            ],
            "connections": [
                {"from_block": "INPUT", "to_block": "DRIVE", "signal_type": "power",
                 "description": "18 V"}
            ],
            "assumptions": [],
        }
    )

    fixed = complete_over_rated_supply(candidate)
    architecture = models.Architecture.model_validate(fixed)
    result = check_fs_connections_mapped(spec, architecture)

    assert result.ok, result.offenders
    sheets_by_rail = {
        net.name: {endpoint.sheet for endpoint in net.endpoints}
        for net in architecture.inter_sheet_nets
    }
    # The bridge sheet no longer speaks for the input rail; the new rail is a two-ended net.
    assert "H BRIDGE" not in sheets_by_rail["VIN"], sheets_by_rail["VIN"]
    assert sheets_by_rail["HBRIDGE_RAIL"] == {"H BRIDGE", "HBRIDGE REGULATOR"}
    # The untouched ground net keeps the derivation's own rows.
    assert sheets_by_rail["GND"] == {"POWER INPUT", "H BRIDGE"}


def test_a_requirement_family_that_cannot_implement_its_class_is_refused_here():
    """The parts stage may not reopen a family, so the mismatch is refused where it is chosen.

    Live walkthrough (2026-09-25): the architecture gave the JST-XH connector requirements the
    generic `pin-header` family while they carried the `jst-xh-connector` obligation; the BOM then
    failed four rounds with "missing-requirement-implementation=['motor_a']".
    """
    from kicraft.design.stage_semantics import _architecture_obligation_family_mismatch

    obligation = {"kind": "physical", "original_obligation_id": "xh",
                  "component_class": "jst-xh-connector"}

    def candidate(family: str, component_class: str = "jst-xh-connector") -> dict:
        return {
            "requirements": [
                {
                    "id": "motor_a",
                    "sheet": "ACTUATOR CONNECTOR 1",
                    "role": "connector",
                    "family": family,
                    "exact_part": None,
                    "parameters": {},
                    "ports": {},
                    "obligations": [{**obligation, "component_class": component_class}],
                }
            ]
        }

    rows = _architecture_obligation_family_mismatch(candidate("pin-header"))
    assert [row.code for row in rows] == ["architecture_obligation_family_mismatch"]
    assert "jst-xh-connector" in rows[0].evidence[0]

    # The carrier's own family, or its exact part, is the fix; an uncovered class stays legitimate.
    assert _architecture_obligation_family_mismatch(candidate("jst-xh-connector")) == []
    named = candidate("pin-header")
    named["requirements"][0]["exact_part"] = "B2B-XH-A(LF)(SN)"
    assert _architecture_obligation_family_mismatch(named) == []
    assert _architecture_obligation_family_mismatch(
        candidate("pin-header", component_class="gps-module")
    ) == []


def test_a_pinned_reviewed_part_realizes_its_demanded_class():
    """The requirement's own reviewed exact part decides, whatever the lowerer family is.

    Live walkthrough (stepper-a4988, 2026-09-29): a requirement pinned A4988SETTR-T in family
    `a4988-stepper-driver` and claimed `motor-driver`. The library's only raw `motor-driver`
    carrier was the DRV8833, so the audit refused the board and pushed a family the writer had
    not chosen -- though the pinned reviewed part IS a motor driver.
    """
    from kicraft.design.stage_semantics import _architecture_obligation_family_mismatch

    def requirement(exact_part):
        return {
            "id": "driver",
            "sheet": "DRIVER",
            "role": "driver",
            "family": "a4988-stepper-driver",
            "exact_part": exact_part,
            "parameters": {},
            "ports": {},
            "obligations": [
                {"kind": "physical", "original_obligation_id": "motor-driver",
                 "component_class": "motor-driver"}
            ],
        }

    assert _architecture_obligation_family_mismatch(
        {"requirements": [requirement("A4988SETTR-T")]}
    ) == []
    # An exact part that does not implement the class is still refused here.
    assert [
        row.code
        for row in _architecture_obligation_family_mismatch(
            {"requirements": [requirement("SS13D07VG4")]}
        )
    ] == ["architecture_obligation_family_mismatch"]


def test_the_selected_recipes_own_reviewed_part_realizes_the_demand():
    """A bound recipe that emits a reviewed part of the class is a realization, not a substitute.

    Live walkthrough (usb-pd-trigger, 2026-09-29): the `pd_trigger` requirement claimed a
    `switch` and the audit refused its bound `ch224k-pd-selectable@1` family, although the
    recipe's own reviewed SS13D07VG4 selector is that switch. Only the recipe the architecture
    actually selected counts, and only its own exact parts.
    """
    from kicraft.design.stage_semantics import _architecture_obligation_family_mismatch

    def requirement(component_class):
        return {
            "id": "pd_trigger",
            "sheet": "PD",
            "role": "power_input",
            "family": "usb-pd-selectable-trigger",
            "exact_part": "CH224K",
            "parameters": {},
            "ports": {},
            "obligations": [
                {"kind": "physical", "original_obligation_id": "selector",
                 "component_class": component_class}
            ],
        }

    selection = {
        "recipe": "ch224k-pd-selectable@1",
        "instance": "pd",
        "requirement_ids": ["pd_trigger"],
    }
    assert _architecture_obligation_family_mismatch(
        {"requirements": [requirement("switch")], "recipe_selections": [selection]}
    ) == []
    # The same recipe does not emit a relay, so the class it truly cannot realize is refused.
    assert [
        row.code
        for row in _architecture_obligation_family_mismatch(
            {"requirements": [requirement("relay")], "recipe_selections": [selection]}
        )
    ] == ["architecture_obligation_family_mismatch"]


def test_a_lowerer_that_builds_the_class_itself_is_not_a_family_mismatch(monkeypatch):
    """`switch-input` builds the reset button, so it satisfies a `pushbutton` demand.

    Live seed-43 run (2026-09-25): the architecture stage refused the reset requirement's
    `switch-input` family -- the `pushbutton` class had just been researched and its carrier's
    family was `pushbutton` -- then spent its repair rounds asking for that carrier. The parts
    stage went on to bind exactly this lowerer's SW1 (`Switch:SW_Push` on the TL3342 button
    footprint) and the board was right. Only a family whose own graph proves the class is exempt:
    the generic `pin-header` family is still refused for a demanded `jst-xh-connector` (below).
    """
    from types import SimpleNamespace

    from kicraft.design import part_identity
    from kicraft.design.stage_semantics import _architecture_obligation_family_mismatch

    carrier = SimpleNamespace(family="pushbutton", identity="k2-1109df-e4sw-04")
    monkeypatch.setattr(
        part_identity, "reviewed_parts_for_feature", lambda _feature: (carrier,)
    )
    requirement = {
        "id": "reset",
        "sheet": "RESET INPUT",
        "role": "user_io",
        "family": "switch-input",
        "exact_part": None,
        "parameters": {"pull_policy": "external"},
        "ports": {"signal": "RESET", "vdd": "+3V3", "gnd": "GND"},
        "obligations": [
            {"kind": "physical", "original_obligation_id": "reset-button",
             "component_class": "pushbutton"}
        ],
    }

    assert _architecture_obligation_family_mismatch({"requirements": [requirement]}) == []
    # A family that does not build the class is still refused.
    assert _architecture_obligation_family_mismatch(
        {"requirements": [{**requirement, "family": "pin-header"}]}
    ) != []


@pytest.mark.parametrize(("component_class", "enable", "accepted"), [
    ("fuse", False, True),
    ("jumper", True, True),
    ("jumper", False, False),
    ("screw-terminal", True, False),
])
def test_distribution_claims_only_parts_its_selected_circuit_builds(component_class, enable, accepted):
    from kicraft.design.stage_semantics import _architecture_obligation_family_mismatch

    requirement = {
        "id": "distribution",
        "sheet": "POWER",
        "role": "power_input",
        "family": "fused-power-distribution",
        "parameters": {
            "channels": 4,
            "input_voltage_v": 5.0,
            "source_current_a": 0.5,
            "enable_jumper": enable,
        },
        "ports": {
            "input": "VBUS",
            "gnd": "GND",
            **{f"output{index}": f"LOAD{index}" for index in range(1, 5)},
        },
        "obligations": [{
            "kind": "physical",
            "original_obligation_id": "requested-hardware",
            "component_class": component_class,
        }],
    }

    diagnostics = _architecture_obligation_family_mismatch({"requirements": [requirement]})
    assert (diagnostics == []) is accepted


def test_the_buck_converter_class_spelling_resolves_to_the_reviewed_buck_regulator():
    """The brief's TLV62569DBVR buck is a reviewed `buck-regulator`; `buck-converter` is its alias.

    Held-out rp1 `stm32-four-adc-usb` (2026-09-30): the brief names the TLV62569DBVR buck, the
    intent classed it `buck-converter`, and the requirement pinned the reviewed part. The
    architecture audit passed but the BOM commit then died with `E_PHYSICAL_REALIZATION 'buck':
    requires 1 reviewed 'buck-converter' physical part(s), found 0` -- unrepairable, because the
    obligation class is committed from intent. The two spellings are one reviewed class (the
    library's own `buck-converter-ic` alias already equates them), so the demand now resolves to
    the record the brief actually names, and a part that is not a buck is still refused.
    """
    from kicraft.design.part_identity import (
        canonical_physical_features,
        reviewed_part,
        reviewed_record_realizes_class,
    )
    from kicraft.design.stage_semantics import _architecture_obligation_family_mismatch

    assert canonical_physical_features("buck-converter") == frozenset(
        {"buck-converter", "buck-regulator"}
    )
    assert reviewed_record_realizes_class(reviewed_part("TLV62569DBVR"), "buck-converter")

    def requirement(exact_part, family="tlv62569-3v3"):
        return {
            "id": "buck",
            "sheet": "POWER",
            "role": "regulator",
            "family": family,
            "exact_part": exact_part,
            "parameters": {},
            "ports": {},
            "obligations": [
                {"kind": "physical", "original_obligation_id": "buck",
                 "component_class": "buck-converter"}
            ],
        }

    assert _architecture_obligation_family_mismatch(
        {"requirements": [requirement("TLV62569DBVR")]}
    ) == []
    # Negative control: a pinned part that is not a buck converter is still refused.
    assert [
        row.code
        for row in _architecture_obligation_family_mismatch(
            {"requirements": [requirement("SS13D07VG4")]}
        )
    ] == ["architecture_obligation_family_mismatch"]


def test_the_rp2040_recipes_own_bootsel_button_realizes_a_pushbutton_demand():
    """Recipe-owned buttons satisfy a demand; unrelated physical owners do not."""
    from kicraft.design.stage_semantics import (
        _architecture_obligation_family_mismatch,
        _selected_recipe_realizes_class,
    )

    requirement = {
        "id": "rp2040",
        "sheet": "RP2040",
        "role": "mcu_core",
        "family": "rp2040",
        "exact_part": "RP2040",
        "parameters": {},
        "ports": {"vdd": "+3V3", "gnd": "GND"},
        "obligations": [
            {"kind": "physical", "original_obligation_id": "rp2040",
             "component_class": "microcontroller"},
            {"kind": "physical", "original_obligation_id": "bootsel_button",
             "component_class": "pushbutton"},
        ],
    }
    candidate = {
        "requirements": [requirement],
        "recipe_selections": [
            {"recipe": "rp2040-minimal@3", "instance": "rp2040",
             "requirement_ids": ["rp2040"]}
        ],
    }

    assert _selected_recipe_realizes_class(candidate, requirement, "pushbutton")
    assert _architecture_obligation_family_mismatch(candidate) == []

    # Negative control: a selected recipe that owns no momentary switch is still refused.
    assert any(
        row.code == "architecture_obligation_family_mismatch"
        for row in _architecture_obligation_family_mismatch(
            {
                "requirements": [{
                    **requirement, "role": "regulator",
                    "family": "tlv62569-3v3", "exact_part": "TLV62569DBVR",
                }],
                "recipe_selections": [
                    {"recipe": "tlv62569-3v3@2", "instance": "rp2040",
                     "requirement_ids": ["rp2040"]}
                ],
            }
        )
    )


def test_unbuildable_regulator_targets_and_exact_identity_are_preserved():
    """Unsupported outputs remain the request for the owning contract to refuse."""
    from kicraft.design.stage_semantics import _retarget_unbuildable_regulators

    candidate = {
        "rail_voltages": {"+18V": 18.0, "+12V": 12.0, "HBRIDGE_RAIL": 10.0},
        "requirements": [
            {
                "id": "boost",
                "role": "regulator",
                "family": "ap63205-5v",
                "exact_part": "AP63205WU-7",
                "parameters": {"output_voltage": 12.0},
                "ports": {"input": "+18V", "output": "+12V", "gnd": "GND"},
            },
            {
                "id": "hbridge_regulator",
                "role": "regulator",
                "family": "tps54331-adjustable",
                "parameters": {"output_voltage": 10.0},
                "ports": {"input": "+18V", "output": "HBRIDGE_RAIL", "gnd": "GND"},
            },
        ],
    }
    before = json.loads(json.dumps(candidate))

    assert _retarget_unbuildable_regulators(candidate) == []
    assert candidate == before


def test_unbuildable_regulator_reads_rail_target_without_rewriting_it():
    """A parameter-less derived regulator has the same no-drift behavior as an explicit target."""
    from kicraft.design.stage_semantics import _retarget_unbuildable_regulators

    candidate = {
        "rail_voltages": {"+18V": 18.0, "HBRIDGE_RAIL": 10.0},
        "requirements": [
            {
                "id": "hbridge_regulator",
                "role": "regulator",
                "family": "tps54331-adjustable",
                "ports": {"input": "+18V", "output": "HBRIDGE_RAIL", "gnd": "GND"},
            }
        ],
    }
    before = json.loads(json.dumps(candidate))

    assert _retarget_unbuildable_regulators(candidate) == []
    assert candidate == before


def test_recipe_resolution_never_discards_an_unsupported_protected_identity(monkeypatch):
    from kicraft.design.recipes import resolver
    from kicraft.design.stage_semantics import _resolve_added_requirements

    candidate = {
        "requirements": [
            {
                "id": "regulator",
                "role": "regulator",
                "family": "ap63205-5v",
                "exact_part": "ESP32-S3-WROOM-1-N16R8",
            }
        ]
    }

    def reject(*_args, **_kwargs):
        raise resolver.RecipeResolutionError(
            [
                resolver.ResolutionDiagnostic(
                    code="unsupported_protected_variant",
                    requirement_id="regulator",
                    message="protected variant has no verified recipe",
                )
            ]
        )

    monkeypatch.setattr(resolver, "apply_architecture_recipe_resolution", reject)

    assert _resolve_added_requirements(candidate) is candidate
    assert candidate["requirements"][0]["exact_part"] == "ESP32-S3-WROOM-1-N16R8"


def test_retarget_only_swaps_an_unprotected_family_at_the_same_voltage():
    from kicraft.design.stage_semantics import _retarget_unbuildable_regulators

    candidate = {
        "rail_voltages": {"+18V": 18.0, "+3V3": 3.3},
        "requirements": [
            {
                "id": "replaceable",
                "role": "regulator",
                "family": "ap63205-5v",
                "parameters": {"output_voltage": 3.3},
                "ports": {"input": "+18V", "output": "+3V3", "gnd": "GND"},
            },
            {
                "id": "protected",
                "role": "regulator",
                "family": "ap63205-5v",
                "exact_part": "AP63205WU-7",
                "parameters": {"output_voltage": 3.3},
                "ports": {"input": "+18V", "output": "+3V3", "gnd": "GND"},
            },
        ],
    }

    assert _retarget_unbuildable_regulators(candidate) == ["replaceable"]
    assert candidate["requirements"][0]["family"] == "ylptec-k7803m-1000r3"
    assert candidate["requirements"][1]["family"] == "ap63205-5v"
    assert candidate["requirements"][1]["exact_part"] == "AP63205WU-7"
    assert candidate["rail_voltages"]["+3V3"] == 3.3
    assert candidate["requirements"][0]["parameters"]["output_voltage"] == 3.3


@pytest.mark.parametrize("input_voltage", [6.0, 18.0, 24.0])
@pytest.mark.parametrize("model_shape", [False, True])
def test_declared_3v3_rail_uses_qualified_input_and_typed_ports(input_voltage, model_shape):
    from kicraft.design.stage_semantics import _complete_load_supply_rails

    candidate = {
        "requirements": [
            {"id": "entry", "role": "power_input", "ports": {"positive": "VIN", "negative": "GND"}},
            {"id": "load", "role": "mcu_core", "family": "esp32-c3",
             "ports": {"vdd": "LOGIC", "gnd": "GND"}},
        ],
        "assumptions": [],
    }
    if model_shape:
        candidate["power"] = {"rails": {
            "VIN": {"voltage": input_voltage, "from": None},
            "LOGIC": {"voltage": 3.3},
            "GND": {"voltage": 0.0},
        }}
    else:
        candidate["rail_voltages"] = {"VIN": input_voltage, "LOGIC": 3.3, "GND": 0.0}
    fixed = _complete_load_supply_rails(candidate)
    converter = next(row for row in fixed["requirements"] if row["role"] == "regulator")
    assert converter["family"] == "ylptec-k7803m-1000r3"
    assert converter["parameters"]["output_voltage"] == 3.3
    assert converter["ports"] == {"input": "VIN", "output": "LOGIC", "gnd": "GND"}


@pytest.mark.parametrize("input_voltage", [18.0, 24.0])
def test_overrated_load_gets_qualified_converter_from_same_input(input_voltage):
    from kicraft.design.stage_semantics import _complete_load_supply_rails

    candidate = {
        "rail_voltages": {"VIN": input_voltage, "GND": 0.0},
        "requirements": [
            {"id": "entry", "role": "power_input", "ports": {"positive": "VIN", "negative": "GND"}},
            {"id": "bridge", "role": "driver", "family": "dual-dc-motor-driver",
             "exact_part": "DRV8833PWPR", "ports": {"vm": "VIN", "gnd": "GND"}},
        ],
        "sheets": [], "assumptions": [],
    }
    fixed = _complete_load_supply_rails(candidate)
    converter = next(row for row in fixed["requirements"] if row["role"] == "regulator")
    assert converter["family"] == "ylptec-k7803m-1000r3"
    assert converter["ports"] == {"input": "VIN", "output": "BRIDGE_RAIL", "gnd": "GND"}
    assert fixed["rail_voltages"]["BRIDGE_RAIL"] == 3.3
    assert fixed["requirements"][1]["ports"]["vm"] == "BRIDGE_RAIL"


@pytest.mark.parametrize("input_voltage", [6.0, 18.0, 24.0])
def test_underrated_load_regulator_keeps_declared_upstream_source(input_voltage):
    from kicraft.design.stage_semantics import _complete_load_supply_rails

    candidate = {
        "rail_voltages": {"VIN": input_voltage, "LOW": 2.5, "GND": 0.0},
        "requirements": [
            {"id": "entry", "role": "power_input", "ports": {"positive": "VIN", "negative": "GND"}},
            {"id": "existing", "role": "regulator", "family": "other",
             "ports": {"input": "VIN", "output": "LOW", "gnd": "GND"}},
            {"id": "bridge", "role": "driver", "family": "dual-dc-motor-driver",
             "exact_part": "DRV8833PWPR", "ports": {"vm": "LOW", "gnd": "GND"}},
        ],
        "sheets": [], "assumptions": [],
    }
    fixed = _complete_load_supply_rails(candidate)
    converter = next(row for row in fixed["requirements"] if row["id"] == "bridge_regulator")
    assert converter["family"] == "ylptec-k7803m-1000r3"
    assert converter["ports"] == {"input": "VIN", "output": "BRIDGE_RAIL", "gnd": "GND"}
    assert fixed["requirements"][2]["ports"]["vm"] == "BRIDGE_RAIL"
    assert fixed["rail_voltages"]["LOW"] == 2.5


@pytest.mark.parametrize("input_voltage", [None, 5.0, 30.0, 36.0])
@pytest.mark.parametrize("target", [3.3, 5.0, 10.0, 12.0])
def test_unqualified_input_or_output_never_creates_module(input_voltage, target):
    from kicraft.design.stage_semantics import _complete_load_supply_rails

    candidate = {
        "rail_voltages": {"VIN": input_voltage, "LOAD": target, "GND": 0.0},
        "requirements": [
            {"id": "entry", "role": "power_input", "ports": {"positive": "VIN", "negative": "GND"}},
            {"id": "load", "role": "sensor", "ports": {"vdd": "LOAD", "gnd": "GND"}},
        ],
    }
    assert _complete_load_supply_rails(candidate) == candidate


@pytest.mark.parametrize("target", [5.0, 10.0, 12.0])
def test_qualified_input_does_not_rewrite_other_requested_outputs(target):
    from kicraft.design.stage_semantics import _complete_load_supply_rails

    candidate = {
        "rail_voltages": {"VIN": 18.0, "LOAD": target, "GND": 0.0},
        "requirements": [
            {"id": "entry", "role": "power_input", "ports": {"positive": "VIN", "negative": "GND"}},
            {"id": "load", "role": "sensor", "ports": {"vdd": "LOAD", "gnd": "GND"}},
        ],
    }
    assert _complete_load_supply_rails(candidate) == candidate


@pytest.mark.parametrize("input_voltage", [None, 5.0, 30.0])
def test_retarget_requires_a_known_qualified_input(input_voltage):
    from kicraft.design.stage_semantics import _retarget_unbuildable_regulators

    candidate = {
        "rail_voltages": {"VIN": input_voltage, "OUT": 3.3},
        "requirements": [{"id": "reg", "role": "regulator", "family": "ap63205-5v",
                          "parameters": {"output_voltage": 3.3},
                          "ports": {"input": "VIN", "output": "OUT", "gnd": "GND"}}],
    }
    before = json.loads(json.dumps(candidate))
    assert _retarget_unbuildable_regulators(candidate) == []
    assert candidate == before


def test_retarget_refuses_manufacturer_and_ambiguous_inputs():
    from kicraft.design.stage_semantics import _retarget_unbuildable_regulators

    candidate = {
        "rail_voltages": {"VIN": 18.0, "OTHER": 24.0, "OUT": 3.3},
        "requirements": [
            {"id": "protected", "role": "regulator", "family": "ap63205-5v",
             "manufacturer": "Diodes Incorporated", "parameters": {"output_voltage": 3.3},
             "ports": {"input": "VIN", "output": "OUT"}},
            {"id": "ambiguous", "role": "regulator", "family": "ap63205-5v",
             "parameters": {"output_voltage": 3.3}, "supply": "OTHER",
             "ports": {"input": "VIN", "output": "OUT"}},
        ],
    }
    before = json.loads(json.dumps(candidate))
    assert _retarget_unbuildable_regulators(candidate) == []
    assert candidate == before


def test_conversion_obligation_refuses_a_wrong_fixed_output_after_normalization():
    """The runtime pre-diagnosis path retains the original 12 V conversion request."""
    from kicraft.server.stage_runtime import _normalize_candidate_for_diagnostics

    conversion = {
        "kind": "conversion",
        "original_obligation_id": "battery-to-12v",
        "input_kind": "2S Li-ion battery-pack DC",
        "output_kind": "12 V DC",
        "behavior": "Convert the battery pack voltage to 12 V DC.",
    }
    candidate = {
        "topologies": {"POWER": "Battery converter"},
        "rail_voltages": {"BATT": 7.4, "+12V": 5.0, "GND": 0.0},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {"name": "POWER", "stem": "POWER", "function": "battery converter"}
        ],
        "power_nets": ["BATT", "+12V", "GND"],
        "inter_sheet_nets": [],
        "assumptions": [],
        "requirements": [
            {
                "id": "boost",
                "sheet": "POWER",
                "role": "regulator",
                "family": "ap63205-5v",
                "exact_part": "AP63205WU-7",
                "parameters": {"output_voltage": 5.0},
                "ports": {"input": "BATT", "output": "+12V", "gnd": "GND"},
                "obligation_ids": ["battery-to-12v"],
            }
        ],
    }
    upstream = {
        "intent": {"obligations": [conversion]},
        "functional_spec": {"obligations": [conversion]},
    }

    normalized = _normalize_candidate_for_diagnostics(
        "architecture", candidate, "2S Li-ion battery pack to 12 V converter", upstream
    )
    diagnostics = diagnose_stage(
        "architecture",
        brief="2S Li-ion battery pack to 12 V converter",
        upstream_state=upstream,
        candidate=normalized,
    )

    assert normalized["rail_voltages"]["+12V"] == 5.0
    boost = normalized["requirements"][0]
    assert boost["parameters"]["output_voltage"] == 5.0
    assert boost["exact_part"] == "AP63205WU-7"
    mismatch = next(
        row for row in diagnostics if row.code == "architecture_conversion_output_mismatch"
    )
    assert "battery-to-12v" in mismatch.evidence[0]


def _conversion_acceptance_payload(
    *,
    requested_voltage: float,
    rail_voltage: float,
    family: str,
    parameters: dict | None = None,
    exact_part: str | None = None,
) -> tuple[dict, dict]:
    identifier = f"battery-to-{requested_voltage:g}v".replace(".", "_")
    conversion = {
        "kind": "conversion",
        "original_obligation_id": identifier,
        "input_kind": "2S Li-ion battery-pack DC",
        "output_kind": f"{requested_voltage:g} V DC",
        "behavior": f"Convert the battery pack voltage to {requested_voltage:g} V DC.",
    }
    payload = {
        "topologies": {"POWER": "Battery converter"},
        "rail_voltages": {"BATT": 7.4, "VOUT": rail_voltage, "GND": 0.0},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [{"name": "POWER", "stem": "POWER", "function": "battery converter"}],
        "power_nets": ["BATT", "VOUT", "GND"],
        "inter_sheet_nets": [],
        "assumptions": [],
        "obligations": [conversion],
        "requirements": [
            {
                "id": "converter",
                "sheet": "POWER",
                "role": "regulator",
                "family": family,
                "exact_part": exact_part,
                "parameters": parameters or {},
                "ports": {"input": "BATT", "output": "VOUT", "gnd": "GND"},
                "obligations": [conversion],
            }
        ],
    }
    upstream = {
        "intent": {"obligations": [conversion]},
        "functional_spec": {"obligations": [conversion]},
    }
    return payload, upstream


@pytest.mark.parametrize(
    ("requested_voltage", "rail_voltage", "family", "parameters", "exact_part"),
    [
        (12.0, 5.0, "ap63205-5v", {"output_voltage": 5.0}, "AP63205WU-7"),
        (10.0, 5.0, "ap63205-5v", {"output_voltage": 5.0}, None),
        (12.0, 12.0, "ap63205-5v", {}, None),
        (10.0, 10.0, "tps54331-adjustable", {}, None),
    ],
)
def test_architecture_contract_rejects_resolved_output_drift(
    requested_voltage,
    rail_voltage,
    family,
    parameters,
    exact_part,
):
    """The final architecture contract checks recipe defaults, not only raw draft rails."""
    from kicraft.server.stage_contracts import StageSchemaError, _normalize_stage_response

    payload, upstream = _conversion_acceptance_payload(
        requested_voltage=requested_voltage,
        rail_voltage=rail_voltage,
        family=family,
        parameters=parameters,
        exact_part=exact_part,
    )

    with pytest.raises(StageSchemaError) as rejected:
        _normalize_stage_response("architecture", payload, upstream)

    assert rejected.value.diagnostic["code"] == "architecture_conversion_output_mismatch"


def test_supported_3v3_regulator_still_passes_architecture_normalization():
    """A real covered regulator still satisfies its original typed conversion request."""
    from kicraft.server.stage_contracts import _normalize_stage_response

    payload, upstream = _conversion_acceptance_payload(
        requested_voltage=3.3,
        rail_voltage=3.3,
        family="tps54331-adjustable",
    )

    normalized, _expanded = _normalize_stage_response("architecture", payload, upstream)

    assert normalized["rail_voltages"]["VOUT"] == 3.3
    regulator = next(row for row in normalized["requirements"] if row["id"] == "converter")
    assert regulator["family"] == "tps54331-adjustable"
    assert regulator["parameters"]["output_voltage"] == 3.3


def test_a_rail_named_sheet_that_holds_a_circuit_is_not_a_distribution_sheet():
    """A sheet may be named for the rail it serves and still carry a real circuit.

    Live architecture draft 2026-09-25: the "POWER INDICATOR" sheet holds the power LED, and the
    word "power" in its name alone was enough to refuse it -- the stage spent two repair rounds
    trying to delete a sheet its own functional block requires. A bare rail sheet with nothing on
    it (or only a converter) is still refused.
    """
    diagnostics = diagnose_stage(
        "architecture",
        brief="A 5 V to 3.3 V converter with a power LED.",
        upstream_state={},
        candidate={
            "sheets": [
                {"name": "POWER INDICATOR", "stem": "POWER_INDICATOR",
                 "function": "Show that the 3.3 V rail is up"},
                {"name": "+3V3", "stem": "3V3", "function": "3.3 V distribution"},
            ],
            "requirements": [
                {"id": "power_led", "sheet": "POWER INDICATOR", "role": "user_io",
                 "family": "led-0603"},
                {"id": "regulator", "sheet": "+3V3", "role": "regulator",
                 "family": "me6211-3v3"},
            ],
        },
    )
    flagged = [
        evidence
        for diagnostic in diagnostics
        if diagnostic.code == "architecture_power_block_as_sheet"
        for evidence in diagnostic.evidence
    ]
    assert flagged == ["+3v3"]


def test_functional_spec_reads_no_topology_from_behaviour_prose():
    """The two live false alarms: prose that describes the world, not a committed technology.

    A passive crossover fed by an external amplifier ("Accept the amplifier input ..."), and a
    buffer describing its signal domain ("analog audio"). Both shipped; both were refused by the
    whole-candidate scan (replay 2026-09-26).
    """
    crossover = {
        "blocks": [
            {
                "name": "INPUT_TERMINAL",
                "category": "interface",
                "purpose": "Accept the amplifier input through positive and negative "
                "binding-post terminals.",
            }
        ],
        "connections": [
            {
                "from_block": "INPUT_TERMINAL",
                "to_block": "INPUT_TERMINAL",
                "signal_type": "analog",
                "description": "Amplifier input audio signal to the crossover branches",
            }
        ],
        "assumptions": ["The amplifier is external to the board and is not included (defaulted)."],
    }
    codes = _codes("functional_spec", crossover)
    assert "functional_spec_premature_topology" not in codes

    buffer = {
        "blocks": [
            {
                "name": "POWER_INPUT",
                "category": "power",
                "purpose": "Accepts the external supply connections required by the analog audio "
                "circuitry.",
            }
        ],
        "connections": [],
        "assumptions": ["Single-supply analog audio operation assumed (defaulted)."],
    }
    assert "functional_spec_premature_topology" not in _codes("functional_spec", buffer)


def test_functional_spec_rejects_a_technology_committed_by_a_block_name():
    """The defect the check still owes: the writer undertakes to build a technology unasked."""
    candidate = {
        "blocks": [
            {"name": "LDO_3V3", "category": "power", "purpose": "Regulates 5 V to 3.3 V."},
            {"name": "MCU", "category": "process", "purpose": "Runs the firmware."},
        ],
        "connections": [],
        "assumptions": [],
    }
    diagnostics = diagnose_stage(
        "functional_spec",
        brief="A 5 V USB input to a 3.3 V ESP32-S3 board.",
        upstream_state={},
        candidate=candidate,
    )
    by_code = {item.code: item for item in diagnostics}
    assert by_code["functional_spec_premature_topology"].evidence == ["ldo"]


def test_prototyping_area_is_only_recorded_when_the_brief_asks_for_the_field():
    """A purpose mention is not a request for a pad field; a named field or its geometry is."""
    from kicraft.design.synthesis.board_features import prototyping_area_requested

    assert prototyping_area_requested("…a header row for easy prototyping.") is None
    assert prototyping_area_requested("A prototyping shield with stacking headers.") == (
        "prototyping shield"
    )
    assert prototyping_area_requested("A prototyping area for soldering.") == "prototyping area"
    assert prototyping_area_requested("A grid of 2.54 mm holes for soldering.") == (
        "grid of 2.54 mm holes"
    )
    assert prototyping_area_requested("A perfboard section.") == "perfboard"


def test_a_sheet_that_only_references_another_sheets_ic_is_not_role_unsupported(tmp_path):
    """Live seed-43 replay: naming the MCU on a connector sheet is not claiming to host it.

    The shipped CH32V003 board was refused for its `UART INTERFACE` and `RESET INPUT` sheets,
    whose prose names the MCU that lives (with U2) on the MCU sheet.
    """
    architecture = {
        "sheets": [
            {"name": "MCU", "function": "Provide CH32V003 processing, UART and reset input."},
            {
                "name": "UART INTERFACE",
                "function": "Expose the MCU UART transmit and receive signals with ground on a header.",
            },
            {
                "name": "RESET INPUT",
                "function": "Provide a momentary pushbutton that pulls the MCU reset input low.",
            },
        ],
        "requirements": [
            {"id": "mcu", "sheet": "MCU", "role": "mcu_core", "family": "ch32v003"},
            {
                "id": "uart_header",
                "sheet": "UART INTERFACE",
                "role": "connector",
                "family": "pin-header",
                "ports": {"tx": "UART_TX", "rx": "UART_RX", "gnd": "GND"},
            },
            {"id": "reset", "sheet": "RESET INPUT", "role": "user_io", "family": "switch-input"},
        ],
    }
    parts = [
        {"ref": "U2", "sheet": "MCU"},
        {"ref": "J3", "sheet": "UART INTERFACE"},
        {"ref": "SW1", "sheet": "RESET INPUT"},
    ]
    assert "bom_architecture_role_unsupported" not in _codes(
        "bom", {"parts": parts}, {"architecture": architecture}
    )


def test_the_adjustable_rc_lowerer_that_builds_the_trimmer_is_not_a_family_mismatch():
    """`adjustable-rc-lowpass` emits the reviewed 3296W trimmer, so it realizes the class.

    Live replay 2026-09-26: the shipped passive RC low-pass breakout was refused because its
    `rc_filter` requirement claimed `trim-potentiometer` under an `adjustable-rc-lowpass` family.
    The lowerer's own graph contains the reviewed 3296W-1-103LF, whose feature is exactly that
    class. A family that does not build the class is still refused.
    """
    from kicraft.design.stage_semantics import _architecture_obligation_family_mismatch

    def candidate(family: str) -> dict:
        return {
            "requirements": [
                {
                    "id": "rc_filter",
                    "sheet": "RC FILTER",
                    "role": "analog_block",
                    "family": family,
                    "exact_part": None,
                    "parameters": {
                        "capacitance_f": 10e-9,
                        "capacitor_exact_part": "C0805C103J5GACTU",
                    },
                    "ports": {"input": "SIGNAL_IN", "output": "FILTERED", "gnd": "GND"},
                    "obligations": [
                        {
                            "kind": "physical",
                            "original_obligation_id": "trim",
                            "component_class": "trim-potentiometer",
                        }
                    ],
                }
            ]
        }

    assert _architecture_obligation_family_mismatch(candidate("adjustable-rc-lowpass")) == []
    assert [
        row.code for row in _architecture_obligation_family_mismatch(candidate("pin-header"))
    ] == ["architecture_obligation_family_mismatch"]


def test_a_crystal_block_is_functional_while_a_bare_net_block_is_not():
    """A brief asks for a crystal; a bare net is not a functional block.

    The shipped STM32 dev board was refused because `crystal` sat in a list of mechanical board
    features (replay 2026-09-26). `power_distribution` stays on the list.
    """
    crystal = {
        "blocks": [
            {"name": "CRYSTAL", "category": "process", "purpose": "Provide the 8 MHz MCU clock."}
        ],
        "connections": [],
        "assumptions": [],
    }
    assert "functional_spec_nonfunctional_block" not in _codes("functional_spec", crystal)

    bare_net = {
        "blocks": [
            {"name": "POWER_DISTRIBUTION", "category": "power", "purpose": "Distributes the rail."}
        ],
        "connections": [],
        "assumptions": [],
    }
    assert "functional_spec_nonfunctional_block" in _codes("functional_spec", bare_net)


def test_a_connection_to_a_board_realized_block_is_refused_before_the_spec_is_frozen():
    """A block the board itself realizes has no sheet, so a connection naming it never maps.

    Live cohort 2026-09-30, brief 3: the committed functional spec carried a PCB_STACKUP block
    (obligation `pcb-copper-layers`, a board-wide stack-up measurement) and a connection
    PCB_STACKUP -> POWER_INPUT. No architecture sheet can implement a board-wide block, so every
    architecture draft was refused with an unmapped block endpoint while the frozen spec kept the
    connection. The refusal belongs to the stage that authored it.
    """
    candidate = {
        "blocks": [
            {
                "name": "POWER_INPUT",
                "category": "power",
                "purpose": "Accept the 18 V DC input.",
                "obligation_ids": ["input-voltage"],
            },
            {
                "name": "PCB_STACKUP",
                "category": "interface",
                "purpose": "Implement the specified two-layer PCB copper stack-up.",
                "obligation_ids": ["pcb-copper-layers"],
            },
        ],
        "connections": [
            {
                "from_block": "PCB_STACKUP",
                "to_block": "POWER_INPUT",
                "signal_type": "other",
                "description": "Two-layer copper stack-up.",
            }
        ],
        "assumptions": [],
    }
    upstream = {
        "intent": {
            "obligations": [
                {
                    "kind": "quantitative",
                    "original_obligation_id": "input-voltage",
                    "quantity": "DC input voltage",
                    "relation": "equal",
                    "value": 18.0,
                    "unit": "V",
                },
                {
                    "kind": "quantitative",
                    "original_obligation_id": "pcb-copper-layers",
                    "quantity": "PCB copper layers",
                    "relation": "equal",
                    "value": 2.0,
                    "unit": "layers",
                },
            ]
        }
    }
    diagnostics = diagnose_stage(
        "functional_spec",
        brief="An 18 V DC input to a two-layer board.",
        upstream_state=upstream,
        candidate=candidate,
    )
    by_code = {item.code: item for item in diagnostics}
    assert by_code["functional_spec_board_realized_block_connected"].evidence == [
        "'pcb_stackup' -> 'power_input'"
    ]

    # Dropping the connection leaves the board-wide block alone: the block itself is legal, and the
    # check stays silent, while a block carrying an obligation that needs a requirement owner is
    # never treated as board-realized.
    candidate["connections"] = []
    candidate["blocks"][1]["obligation_ids"] = ["input-voltage"]
    assert "functional_spec_board_realized_block_connected" not in _codes(
        "functional_spec", candidate, upstream
    )


def test_ground_flow_does_not_demand_a_return_from_a_mechanical_block():
    """A mounting-features block draws no current and owns no return.

    Live replay 2026-09-26: a shipped USB-UART bridge was refused with `mounting_features` as the
    only ungrounded block.
    """
    candidate = {
        "blocks": [
            {"name": "USB_INPUT", "category": "interface", "purpose": "USB-C receptacle"},
            {"name": "BRIDGE", "category": "process", "purpose": "USB to UART"},
            {"name": "MOUNTING_FEATURES", "category": "mechanical", "purpose": "Mounting holes"},
        ],
        "connections": [
            {
                "from_block": "USB_INPUT",
                "to_block": "BRIDGE",
                "signal_type": "ground",
                "description": "Signal ground",
            }
        ],
        "assumptions": [],
    }
    assert "functional_spec_partial_ground_flow" not in _codes("functional_spec", candidate)

    # A functional block still in no ground connection is still refused.
    candidate["blocks"].append(
        {"name": "STATUS_LED", "category": "drive", "purpose": "Show activity"}
    )
    diagnostics = diagnose_stage(
        "functional_spec",
        brief="A USB to UART bridge",
        upstream_state={},
        candidate=candidate,
    )
    by_code = {item.code: item for item in diagnostics}
    assert by_code["functional_spec_partial_ground_flow"].evidence == ["status_led"]


def test_a_rail_named_sheet_that_regulates_is_not_a_distribution_sheet():
    """The escape hatch must read "regulate", not only "regulator"/"regulation".

    Live replay 2026-09-26: a shipped Arduino-shield board was asked to delete its POWER sheet,
    whose function is "Accept host power rails and regulate VIN to a 3.3 V rail". A sheet that
    only distributes a rail is still refused.
    """
    def architecture(function):
        return {
            "sheets": [{"name": "POWER", "stem": "POWER", "function": function}],
            "requirements": [
                {"id": "reg", "sheet": "POWER", "role": "regulator", "family": "ams1117-3v3"}
            ],
        }

    regulating = _codes("architecture", architecture("Accept host power rails and regulate VIN "
                                                     "to a 3.3 V rail for shield circuitry."))
    assert "architecture_power_block_as_sheet" not in regulating

    distributing = _codes("architecture", architecture("Distribute the shield 5 V rail."))
    assert "architecture_power_block_as_sheet" in distributing


def test_a_power_named_sheet_that_implements_a_device_is_not_distribution_only():
    """The escape hatch must read "implements", not only connector/holder/regulator.

    Live replay 2026-09-26: a shipped hex environmental sensor was asked to delete its POWER
    INDICATOR sheet, whose function is "Implements the 3.3 V power-status LED and its series
    current-limiting resistor." A sheet that only distributes a rail is still refused.
    """
    def architecture(function):
        return {
            "sheets": [{"name": "POWER INDICATOR", "stem": "POWER_INDICATOR", "function": function}],
            "requirements": [],
        }

    implementing = _codes(
        "architecture",
        architecture("Implements the 3.3 V power-status LED and its series current-limiting resistor."),
    )
    assert "architecture_power_block_as_sheet" not in implementing

    distributing = _codes("architecture", architecture("Distribute the 3.3 V rail and ground."))
    assert "architecture_power_block_as_sheet" in distributing


def test_a_relay_or_led_sheet_is_not_a_missing_ic_implementation():
    """`driver` is the pipeline's role for relays, LED strings and transistor stages.

    Live replay 2026-09-26: shipped relay-quad and LED-ring boards were asked to add a
    U-reference for a relay (K1) and a WS2812 string (D1..D12). A sheet *titled* for an IC
    still needs one.
    """
    def candidate(sheet, role, family, ref, function):
        return {
            "architecture": {
                "sheets": [{"name": sheet, "stem": sheet.replace(" ", "_"), "function": function}],
                "requirements": [
                    {"id": "r1", "sheet": sheet, "role": role, "family": family}
                ],
            },
            "parts": [{"ref": ref, "sheet": sheet}],
        }

    def codes(payload):
        return _codes("bom", {"parts": payload["parts"]}, {"architecture": payload["architecture"]})

    assert "bom_architecture_role_unsupported" not in codes(
        candidate("RELAY CHANNEL 1", "driver", "relay-spdt-through-hole", "K1",
                  "Switch the through-hole relay coil for channel 1.")
    )
    assert "bom_architecture_role_unsupported" not in codes(
        candidate("LED RING", "driver", "ws2812-output", "D1",
                  "Implement twelve addressable WS2812B LEDs in a ring.")
    )
    assert "bom_architecture_role_unsupported" in codes(
        candidate("MOTOR DRIVER", "driver", "dual-dc-motor-driver", "J1",
                  "Drive the motors.")
    )


def test_a_brief_that_powers_its_motors_discharges_the_external_load_rule():
    """The proof vocabulary must cover the load the connection names.

    Live replay 2026-09-26: the rule had a `motor` branch but only LED/display terms, so a brief
    that explicitly powers its motors could never satisfy it.
    """
    candidate = {
        "blocks": [{"name": "MOTOR_DRIVER", "category": "drive", "purpose": "Drive the motors"}],
        "connections": [
            {
                "from_block": "BATTERY_INPUT",
                "to_block": "MOTOR_DRIVER",
                "signal_type": "power",
                "description": "Battery motor supply",
            }
        ],
        "assumptions": [],
    }

    def codes(brief):
        return {
            item.code
            for item in diagnose_stage(
                "functional_spec", brief=brief, upstream_state={}, candidate=dict(candidate)
            )
        }

    assert "functional_spec_external_load_power_assumed" not in codes(
        "A robot controller that powers the two motors from a 2S battery, with DRV8833 drivers."
    )
    assert "functional_spec_external_load_power_assumed" in codes(
        "A robot controller with two DRV8833 motor drivers and motor screw terminals."
    )


def test_a_longer_spelling_of_a_reviewed_class_is_renamed_not_refused():
    """The check's own evidence names the repair; the pipeline makes it instead of asking.

    Live replay 2026-09-26: 150 of 215 `intent_obligation_class_unrealizable` findings are this
    shape, across 154 runs whose class then reached the parts stage unresolved.
    """
    from kicraft.design.stage_semantics import complete_class_spellings

    candidate = {
        "obligations": [{"kind": "physical", "component_class": "smt-voltage-regulator"}],
        "assumptions": [],
    }
    completed = complete_class_spellings(candidate)
    assert completed["obligations"][0]["component_class"] == "voltage-regulator"
    assert any(
        "read as reviewed class 'voltage-regulator' (defaulted)" in row
        for row in completed["assumptions"]
    )
    assert "intent_obligation_class_unrealizable" not in _codes("intent", completed)
    # the input is untouched
    assert candidate["obligations"][0]["component_class"] == "smt-voltage-regulator"


def test_a_class_spelling_is_never_renamed_when_it_would_invert_or_distort():
    """A negation inverts the token-superset relation; a source and a new class have other fixes."""
    from kicraft.design.stage_semantics import complete_class_spellings

    for demanded in ("no-microcontroller", "coin-cell-battery", "gps-module"):
        candidate = {"obligations": [{"kind": "physical", "component_class": demanded}], "assumptions": []}
        assert complete_class_spellings(candidate)["obligations"][0]["component_class"] == demanded

    # The negated class is still reported: the writer must restate or drop it.
    assert "intent_obligation_class_unrealizable" in _codes(
        "intent", {"obligations": [{"kind": "physical", "component_class": "no-microcontroller"}],
                   "assumptions": []}
    )
    # A class the library already covers, and a quantity row, are left alone.
    covered = complete_class_spellings(
        {"obligations": [
            {"kind": "physical", "component_class": "pin-header"},
            {"kind": "quantity", "subject": "pin-header", "minimum": 2},
        ], "assumptions": []}
    )
    assert [row.get("component_class") or row.get("subject") for row in covered["obligations"]] == [
        "pin-header",
        "pin-header",
    ]


def test_a_declared_interface_that_names_no_contact_is_refused_at_architecture():
    """An unnamed declared contact is unverifiable, and committing it dooms the run.

    Seven architectures in the live store carried an unnamed declared contact and every one died at
    BOM with `unit_repair_exhausted` (projects 866, 971, 1019, 1034, 1057, 1074, 1105); no board
    that shipped has one. The BOM gate can name the condition ("claim states no pin number; name the
    actual contact of <symbol>") but cannot be acted on there -- the claim is frozen with the
    architecture -- so the refusal belongs to the stage that authored it.
    """
    unnamed = {
        "sheets": [{"name": "OUTPUT 1", "stem": "OUTPUT_1", "function": "Fused 5 V output"}],
        "requirements": [
            {
                "id": "fuse1",
                "sheet": "OUTPUT 1",
                "role": "connector",
                "family": "fused-output",
                "declared_ports": [
                    {"key": "input", "pin": None, "direction": "input", "function": "5 V feed"},
                    {"key": "output", "pin": None, "direction": "output", "function": "fused out"},
                ],
            }
        ],
    }
    diagnostics = diagnose_stage(
        "architecture", brief="A fused 5 V distribution board.", upstream_state={}, candidate=unnamed
    )
    row = next(d for d in diagnostics if d.code == "architecture_declared_contact_unnamed")
    assert row.evidence == ["fuse1: input, output"]

    # The canonical shape is read too, so a committed claim is caught as well as a draft.
    canonical = {
        "sheets": unnamed["sheets"],
        "requirements": [
            {
                "id": "mcu",
                "sheet": "OUTPUT 1",
                "role": "mcu_core",
                "family": "stm32g0",
                "declared_interface": {
                    "ports": [
                        {"key": "vdd", "pin": None, "direction": "power_in", "function": "3V3"},
                        {"key": "gnd", "pin": "2", "direction": "power_in", "function": "return"},
                    ]
                },
            }
        ],
    }
    assert "architecture_declared_contact_unnamed" in _codes("architecture", canonical)

    # A claim that names a contact for every port is untouched.
    named = {
        "sheets": unnamed["sheets"],
        "requirements": [
            {
                "id": "fuse1",
                "sheet": "OUTPUT 1",
                "role": "connector",
                "family": "fused-output",
                "declared_ports": [
                    {"key": "input", "pin": "1", "direction": "passive", "function": "5 V feed"},
                    {"key": "output", "pin": "2", "direction": "passive", "function": "fused out"},
                ],
            }
        ],
    }
    assert "architecture_declared_contact_unnamed" not in _codes("architecture", named)


def test_a_class_the_brief_brands_with_a_participle_must_be_recorded():
    """`fused outputs` demands a fuse; nothing downstream can place one the intent never records.

    Live cohort 2026-09-30, project 1108: this exact brief committed an intent with no fuse
    obligation, the board built and exported a fab package carrying no fuse at all, and it would
    have counted as a successful run. Only the participle form is read: the bare and plural forms
    fire on 30 boards in the live store that are correct as built.
    """
    brief = (
        "A 5 V header from the host board power distribution board with two fused outputs on screw "
        "terminals and an RGB status LED. Keep it under 100 x 100 mm."
    )
    recorded_without_fuse = {
        "obligations": [
            {"kind": "physical", "component_class": "pin-header"},
            {"kind": "physical", "component_class": "screw-terminal"},
            {"kind": "quantity", "subject": "screw-terminal", "minimum": 2},
            {"kind": "physical", "component_class": "led"},
        ],
        "named_parts": [],
        "constraints": [],
        "assumptions": [],
        "goal": "Distribute 5 V from a host board through two fused outputs.",
    }
    diagnostics = diagnose_stage(
        "intent", brief=brief, upstream_state={}, candidate=recorded_without_fuse
    )
    row = next(d for d in diagnostics if d.code == "intent_demanded_class_omitted")
    assert row.evidence == ["'fused' -> fuse"]

    # Recording the class silences it.
    recorded_with_fuse = {
        **recorded_without_fuse,
        "obligations": [
            *recorded_without_fuse["obligations"],
            {"kind": "physical", "component_class": "fuse"},
        ],
    }
    assert "intent_demanded_class_omitted" not in {
        d.code
        for d in diagnose_stage(
            "intent", brief=brief, upstream_state={}, candidate=recorded_with_fuse
        )
    }

    # The bare and plural forms are compounds and counts, not a second class: a brief that says
    # "screw terminals" or "servo headers" and records them under another spelling stays silent.
    plain = {
        "obligations": [{"kind": "physical", "component_class": "pin-header"}],
        "named_parts": [],
        "constraints": [],
        "assumptions": [],
        "goal": "Break out host power.",
    }
    assert "intent_demanded_class_omitted" not in {
        d.code
        for d in diagnose_stage(
            "intent",
            brief="A power breakout with screw terminals and two servo headers.",
            upstream_state={},
            candidate=plain,
        )
    }

def test_external_load_intent_is_not_an_onboard_led_and_preserves_source_id():
    from kicraft.design.stage_semantics import complete_intent_classification

    candidate = {"obligations": [{
        "kind": "physical", "original_obligation_id": "load",
        "component_class": "addressable-led-strip",
    }]}
    brief = "A driver board for a WS2812 addressable LED strip."
    completed = complete_intent_classification(brief, candidate)
    assert completed["obligations"] == [{
        "kind": "external_interface", "original_obligation_id": "load", "protocol": "ws2812",
    }]
    onboard = complete_intent_classification(
        "A driver board for a WS2812 strip with onboard LEDs.", candidate,
    )
    assert onboard["obligations"][0]["kind"] == "physical"


def test_physical_enable_jumper_retains_physical_and_control_demands():
    from kicraft.design.stage_semantics import complete_intent_classification

    completed = complete_intent_classification(
        "A regulator board with an enable jumper.",
        {"obligations": [{"kind": "physical", "original_obligation_id": "en",
                           "component_class": "jumper"}]},
    )
    assert completed["obligations"] == [
        {"kind": "physical", "original_obligation_id": "en", "component_class": "jumper"},
        {"kind": "adjustability", "original_obligation_id": "en-control",
         "parameter": "enable", "mechanism": "jumper"},
    ]

def test_linear_regulator_absolute_dissipation_refuses_proven_overload():
    from kicraft.design.stage_semantics import architecture_power_requirement_diagnostics

    candidate = {
        "rail_voltages": {"VIN": 5.0, "VOUT": 3.3},
        "requirements": [{
            "id": "reg", "sheet": "POWER", "role": "regulator",
            "family": "me6211-3v3", "exact_part": "ME6211C33M5G-N",
            "ports": {"input": "VIN", "output": "VOUT", "gnd": "GND"},
            "obligations": [{"kind": "quantitative", "original_obligation_id": "current",
                             "quantity": "output current rating", "relation": "equal",
                             "value": 0.5, "unit": "A"}],
        }],
    }
    issues = architecture_power_requirement_diagnostics({}, candidate)
    assert any(issue.code == "architecture_linear_regulator_dissipation_exceeded" for issue in issues)
    candidate["requirements"][0]["obligations"][0]["value"] = 0.1
    issues = architecture_power_requirement_diagnostics({}, candidate)
    assert not any(issue.code == "architecture_linear_regulator_dissipation_exceeded" for issue in issues)
    candidate["rail_voltages"]["VIN"] = 3.5
    candidate["requirements"][0]["obligations"][0]["value"] = 0.5
    candidate["requirements"].append({
        "id": "source", "sheet": "POWER", "role": "power_input",
        "family": "pin-header", "ports": {"pin1": "VIN"},
        "obligations": [{"kind": "quantitative", "original_obligation_id": "vin-range",
                         "quantity": "input voltage", "relation": "range",
                         "minimum": 3.5, "maximum": 5.0, "unit": "V"}],
    })
    issues = architecture_power_requirement_diagnostics({}, candidate)
    assert any(issue.code == "architecture_linear_regulator_dissipation_exceeded" for issue in issues)
    candidate["requirements"][0]["exact_part"] = "TLV62569DBVR"
    candidate["requirements"][0]["obligations"][0]["value"] = 0.5
    issues = architecture_power_requirement_diagnostics({}, candidate)
    assert not any(issue.code == "architecture_linear_regulator_dissipation_exceeded" for issue in issues)


@pytest.mark.parametrize(("voltage", "rejected"), [(3.3, False), (5.0, False), (5.1, True), (9.0, True)])
def test_native_drv_motor_tvs_standoff_constrains_architecture_vm(voltage, rejected):
    from kicraft.design.stage_semantics import _architecture_motor_clamp_standoff

    candidate = {
        "rail_voltages": {"VM": voltage},
        "requirements": [{"id": "motor", "role": "driver", "family": "dual-dc-motor-driver",
                          "exact_part": "DRV8833PWPR", "ports": {"vm": "VM"}}],
    }
    assert bool(_architecture_motor_clamp_standoff(candidate)) is rejected
