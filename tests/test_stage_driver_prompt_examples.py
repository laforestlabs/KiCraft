"""The worked examples embedded in the stage prompts MUST validate against the
real slot models (2026-07-19 review §7.1) — a schema change that breaks an
example must fail here, not teach the production model a guaranteed bounce.
"""

from __future__ import annotations

import json
import pytest

from kicraft.server.client import _StreamingCollectionGuard
from kicraft.design import models
from kicraft.design.architecture_intent import derive_architecture
from kicraft.design.stage_state import DESIGN_STAGES
from kicraft.design.synthesis.validation import check_spec_named_mpn_substitutions
from kicraft.server.config import STAGE_COLLECTION_BOUNDS, CollectionBound
from kicraft.server.stage_contracts import (
    ArchitectureStageResponse,
    IntentStageResponse,
    _is_free_form_object,
    _normalize_bom_stage_response,
    _strict_provider_schema,
    _fold_recipe_covered_sheets,
    _normalize_usb_c_requirements,
    _normalize_stage_response,
    _normalize_wiring_stage_response,
    apply_collection_bounds,
    build_stage_response_contract,
)
from kicraft.server.stage_prompts import _WORKED_EXAMPLES, build_system as _build_system


def build_system(stage: str, collection_bounds=None) -> str:
    state = {"architecture": {"sheets": [{"name": "POWER"}]}} if stage == "bom" else {}
    return _build_system(build_stage_response_contract(stage, state), collection_bounds)


def test_bom_example_validates_against_the_model_contract():
    slot = json.loads(_WORKED_EXAMPLES["bom"])
    canonical, expanded = _normalize_bom_stage_response(slot)
    assert expanded == 4
    assert [part["ref"] for part in canonical["parts"]] == [
        "U1",
        "C1",
        "C2",
        "J1",
    ]


def test_wiring_example_normalizes_to_canonical_wiring():
    bom, _ = _normalize_bom_stage_response(json.loads(_WORKED_EXAMPLES["bom"]))
    canonical = _normalize_wiring_stage_response(
        json.loads(_WORKED_EXAMPLES["wiring"]), {"bom": bom}
    )
    nets = {connection["net_name"] for connection in canonical["connections"]}
    assert nets == {"VIN", "+3V3", "GND"}
    endpoints_by_net = {
        connection["net_name"]: {
            (endpoint["ref"], endpoint["pin"]) for endpoint in connection["endpoints"]
        }
        for connection in canonical["connections"]
    }
    assert endpoints_by_net["VIN"] == {("J1", "1"), ("U1", "3"), ("C1", "1")}
    assert endpoints_by_net["+3V3"] == {("U1", "1"), ("C2", "1")}


def test_intent_example_validates_against_the_model_contract():
    slot = json.loads(_WORKED_EXAMPLES["intent"])
    IntentStageResponse.model_validate(slot)


def test_functional_spec_example_validates_against_the_model_contract():
    models.FunctionalSpec.model_validate(json.loads(_WORKED_EXAMPLES["functional_spec"]))


def test_architecture_example_validates_and_carries_a_requirement():
    slot = json.loads(_WORKED_EXAMPLES["architecture"])
    architecture = ArchitectureStageResponse.model_validate(slot)
    assert architecture.requirements
    assert architecture.requirements[0].functional_blocks


def test_architecture_intent_example_derives_and_normalizes():
    """The example the model is shown must itself survive derivation and the real reader."""
    slot = json.loads(_WORKED_EXAMPLES["architecture_intent"])
    architecture = derive_architecture(slot)
    assert architecture.requirements
    normalized, _ = _normalize_stage_response("architecture", slot, {})
    assert normalized["requirements"]
    assert normalized["power_nets"] == ["GND", "+3V3", "+5V"]
    assert normalized["rail_voltages"] == {"+5V": 5.0, "+3V3": 3.3, "GND": 0.0}


def test_recipe_covered_bom_records_the_curated_identity():
    """A curated recipe's identity is not always the part it ships; §9.33 must see the pairing.

    Live 2026-09-14: a board whose sheets are all recipe- or lowerer-covered has no provider
    call in which to write a substitution ledger, so a recipe identity its shipped part does
    not spell out failed §9.33 outright with zero attempts and no way to recover.
    """
    from kicraft.design.recipes.registry import get_recipe

    definition = get_recipe("hub75-sn74hct245-interface@1")
    bindings = {
        port.name: (
            "+5V"
            if port.name == "vdd_5v"
            else "GND"
            if port.name in {"gnd", "addr_d"}
            else f"HUB75_{port.name.upper()}"
        )
        for port in definition.ports
        if port.required
    }
    architecture = {
        "topologies": {},
        "rail_voltages": {"+5V": 5.0},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [{"name": "HUB75", "stem": "HUB75", "function": "HUB75 level-shift interface"}],
        "power_nets": ["+5V", "GND"],
        "inter_sheet_nets": [],
        "requirements": [
            {
                "id": "hub75",
                "sheet": "HUB75",
                "role": "bus_interface",
                "family": "hub75-level-shift-interface",
                "exact_part": "HUB75-SN74HCT245",
                "ports": bindings,
                "functional_blocks": ["DISPLAY"],
            }
        ],
        "recipe_selections": [
            {
                "recipe": "hub75-sn74hct245-interface@1",
                "instance": "hub75",
                "sheets": {"interface": "HUB75"},
                "requirement_ids": ["hub75"],
                "port_bindings": bindings,
            }
        ],
    }
    bom, _ = _normalize_bom_stage_response({"groups": []}, {"architecture": architecture})
    carrier = next(part for part in bom["parts"] if part["value"].startswith("SN74HCT245"))
    assert "HUB75-SN74HCT245" in carrier["sourcing_note"]
    result = check_spec_named_mpn_substitutions(
        None,
        models.Architecture.model_validate(architecture),
        models.BOM.model_validate(bom),
    )
    assert result.ok, result.offenders


def test_every_stage_has_a_worked_example_riding_the_system_prompt():
    for stage in DESIGN_STAGES:
        assert stage in _WORKED_EXAMPLES, stage
        json.loads(_WORKED_EXAMPLES[stage])  # a concrete JSON instance, not a schema
        assert "Worked example" in build_system(stage)
        assert _WORKED_EXAMPLES[stage] in build_system(stage)


@pytest.mark.parametrize("stage", list(DESIGN_STAGES))
def test_deepseek_json_object_prompts_name_json(stage):
    # DeepSeek's json_object response format requires the word "json" in the
    # prompt (case-insensitive); the deepseek profile has no json_schema mode.
    assert "json" in build_system(stage).lower()


def test_strict_provider_schema_is_openai_compatible():
    schema = build_stage_response_contract("architecture", {}).schema
    strict = _strict_provider_schema(schema)

    assert strict is not schema
    assert strict["type"] == "object"
    assert strict["additionalProperties"] is False
    # Every fixed-shape root property is required; free-form maps are not.
    fixed = {
        name
        for name, subschema in strict["properties"].items()
        if not _is_free_form_object(subschema)
    }
    assert set(strict["required"]) == fixed
    assert "rail_voltages" in strict["properties"]
    assert "rail_voltages" not in strict["required"]
    nested = strict["$defs"]["Sheet"]
    assert nested["additionalProperties"] is False
    assert set(nested["required"]) == set(nested["properties"])
    # The canonical schema is untouched.
    assert "additionalProperties" not in schema["$defs"]["Sheet"]


def test_interactive_contract_merges_questions_and_empty_list_is_a_slot():
    contract = build_stage_response_contract("intent", {}, allow_questions=True)
    assert contract.allow_questions is True
    assert "anyOf" not in contract.schema
    questions = contract.schema["properties"]["questions"]
    assert questions["minItems"] == 0 and questions["maxItems"] == 5
    assert "questions" in contract.schema["required"]

    slot = json.loads(_WORKED_EXAMPLES["intent"])
    # The strict decoder always emits `questions`; an empty list is not a question.
    normalized, _ = _normalize_stage_response("intent", {**slot, "questions": []}, {})
    assert normalized["project_stem"] == slot["project_stem"]

    asked = {
        "goal": "x",
        "questions": [
            {
                "text": "Q?",
                "options": ["a", "b"],
                "blocking": True,
                "material": True,
                "stage": "intent",
            }
        ],
    }
    normalized, _ = _normalize_stage_response("intent", asked, {})
    assert set(normalized) == {"questions"}


def test_noninteractive_contract_drops_questions_and_tells_the_model():
    contract = build_stage_response_contract("intent", {}, allow_questions=False)
    assert contract.allow_questions is False
    assert "questions" not in contract.schema["properties"]
    assert "CLARIFYING QUESTIONS ARE DISABLED" in _build_system(contract)


@pytest.mark.parametrize("stage", ["intent", "functional_spec", "architecture", "bom"])
@pytest.mark.parametrize("allow_questions", [True, False])
def test_provider_schema_and_stream_collection_limits_agree(stage, allow_questions) -> None:
    state = {"architecture": {"sheets": [{"name": "POWER"}]}} if stage == "bom" else {}
    contract = build_stage_response_contract(stage, state, allow_questions=allow_questions)
    schema = contract.response_format["json_schema"]["schema"]
    properties = schema["properties"]
    for bound in STAGE_COLLECTION_BOUNDS[stage]:
        if bound.field not in properties:
            # A stage's bounds cover more than one slot shape (the intent-shaped
            # architecture slot has `signals`, the explicit one does not); a field
            # the contract does not carry has nothing to advertise or enforce.
            continue
        advertised_limit = properties[bound.field]["maxItems"]
        guard = _StreamingCollectionGuard((bound,))
        at_limit = json.dumps({bound.field: [{}] * advertised_limit})
        assert guard.consume(at_limit) == (at_limit, None)
        guard = _StreamingCollectionGuard((bound,))
        _accepted, overflow = guard.consume(
            json.dumps({bound.field: [{}] * (advertised_limit + 1)})
        )
        assert overflow["observed_count"] == advertised_limit + 1
        assert overflow["configured_total"] == advertised_limit
    if allow_questions:
        # The interactive contract is one object: slot + `questions`, where an
        # empty array is the normal "no question" answer.
        assert properties["questions"]["maxItems"] == 5
        assert properties["questions"]["minItems"] == 0
    else:
        assert "questions" not in properties


def test_unit_schema_bounds_remain_authoritative_and_invocation_local() -> None:
    state = {"architecture": {"sheets": [{"name": "POWER"}]}}
    unit = build_stage_response_contract("bom", state, bom_sheet="POWER")
    apply_collection_bounds(
        unit.schema,
        (CollectionBound(field="groups", total=21, per_group=21, group_key="sheet"),),
    )
    # A later global application may never relax the unit's smaller ceiling.
    apply_collection_bounds(unit.schema, STAGE_COLLECTION_BOUNDS["bom"])
    groups = unit.response_format["json_schema"]["schema"]["properties"]["groups"]
    assert groups["minItems"] == 1
    assert groups["maxItems"] == 21
    fresh = build_stage_response_contract("bom", state)
    assert fresh.schema["properties"]["groups"]["maxItems"] == 64

    wiring = build_stage_response_contract("wiring", {}, wiring_refs=("J1",))
    apply_collection_bounds(wiring.schema, (CollectionBound(field="pins", total=2),))
    pins = wiring.response_format["json_schema"]["schema"]["properties"]["pins"]
    assert pins["maxItems"] == 2


def test_bom_contract_closes_group_sheet_and_reuses_schema_object():
    names = ["ADDRESSABLE LED OUTPUT", "SPEAKER OUTPUT"]
    state = {"architecture": {"sheets": [{"name": name} for name in names]}}
    contract = build_stage_response_contract("bom", state)

    definitions = contract.schema["$defs"]
    assert definitions["BomComponentGroup"]["properties"]["sheet"]["enum"] == names
    provider_schema = contract.response_format["json_schema"]["schema"]
    # The provider envelope is a strictified copy: same shape, but every object
    # closed and every fixed-shape property required for OpenAI structured outputs.
    assert provider_schema is not contract.schema
    assert set(provider_schema["properties"]) == set(contract.schema["properties"])
    assert provider_schema["additionalProperties"] is False
    assert set(provider_schema["required"]) == set(provider_schema["properties"])
    assert "ADDRESSABLE LED OTPUT" not in names
    assert "SPEAKER OTPUT" not in names

    prompt = _build_system(contract)
    encoded = prompt.split("string patterns are strict):\n", 1)[1].split("\nWorked example", 1)[0]
    assert json.loads(encoded) == contract.schema
    assert "SHEET NAMES ARE CLOSED" in prompt


def test_generic_usb_c_requirements_select_verified_recipe_families():
    base = {
        "power_nets": ["VBUS", "GND"],
        "inter_sheet_nets": [
            {"name": "usb_dp", "endpoints": [{"sheet": "USB", "direction": "bidirectional"}]},
            {"name": "usb_dm", "endpoints": [{"sheet": "USB", "direction": "bidirectional"}]},
        ],
    }
    data = {
        **base,
        "requirements": [{"id": "usb", "sheet": "USB", "role": "connector", "family": "usb-c"}],
    }
    device = _normalize_usb_c_requirements(data)["requirements"][0]
    sink = _normalize_usb_c_requirements(
        {
            "power_nets": ["VBUS", "GND"],
            "requirements": [
                {
                    "id": "input",
                    "sheet": "POWER",
                    "role": "power_input",
                    "family": "usb-c-receptacle",
                }
            ],
        }
    )["requirements"][0]

    assert device["family"] == "usb-c-usb2-device"
    assert device["ports"] == {
        "gnd": "GND",
        "vbus": "VBUS",
        "usb_dp": "usb_dp",
        "usb_dm": "usb_dm",
    }
    assert sink["family"] == "usb-c-power-sink"
    assert sink["ports"] == {"gnd": "GND", "vbus": "VBUS"}
    synthesized = _normalize_usb_c_requirements(
        {
            "power_nets": ["VBUS", "GND"],
            "inter_sheet_nets": [],
            "sheets": [
                {
                    "name": "USB C INPUT",
                    "function": "USB-C receptacle with CC pull-downs",
                }
            ],
            "requirements": [],
        }
    )["requirements"][0]
    assert synthesized["family"] == "usb-c-power-sink"
    assert synthesized["sheet"] == "USB C INPUT"


def test_rounded_c3_connector_binding_survives_downstream_five_volt_rail():
    # Connector/power projection of round10's final rounded-c3-devboard candidate.
    ports = {
        "gnd": "GND",
        "usb_dm": "USB_D_N",
        "usb_dp": "USB_D_P",
        "vbus": "VBUS",
    }
    connector = {
        "family": "usb-c-usb2-device",
        "functional_blocks": ["USB_C_CONNECTOR"],
        "id": "usb_c_connector",
        "parameters": {},
        "ports": ports,
        "role": "connector",
        "sheet": "USB C CONNECTOR",
    }
    payload = {
        "power_nets": ["VBUS", "+5V", "+3V3", "GND"],
        "rail_voltages": {"+3V3": 3.3, "+5V": 5.0, "VBUS": 5.0},
        "requirements": [connector],
        "inter_sheet_nets": [
            {
                "name": name,
                "endpoints": [{"sheet": sheet, "direction": "bidirectional"} for sheet in sheets],
            }
            for name, sheets in [
                ("USB_D_P", ["USB C CONNECTOR", "USB SERIAL BRIDGE"]),
                ("USB_D_N", ["USB C CONNECTOR", "USB SERIAL BRIDGE"]),
                ("VBUS", ["USB C CONNECTOR", "POWER INPUT"]),
                (
                    "GND",
                    [
                        "USB C CONNECTOR",
                        "POWER INPUT",
                        "USB SERIAL BRIDGE",
                        "VOLTAGE REGULATION",
                        "ESP32 C3 MODULE",
                        "GPIO HEADER",
                    ],
                ),
                ("+5V", ["POWER INPUT", "VOLTAGE REGULATION"]),
                ("+3V3", ["VOLTAGE REGULATION", "ESP32 C3 MODULE", "GPIO HEADER"]),
            ]
        ],
    }

    normalized = _normalize_usb_c_requirements(payload)

    assert normalized["requirements"] == [connector]
    assert normalized["power_nets"] == ["VBUS", "+5V", "+3V3", "GND"]
    assert normalized["inter_sheet_nets"] == payload["inter_sheet_nets"]
    assert normalized["rail_voltages"] == payload["rail_voltages"]


def test_usb_connector_inference_scopes_rails_and_auxiliary_signals_to_owner():
    payload = {
        "power_nets": ["VBUS", "+5V", "GND"],
        "sheets": [
            {"name": "USB C BREAKOUT", "function": "USB-C receptacle exposing all pins"},
            {"name": "USB C INPUT", "function": "USB-C receptacle with CC pull-downs"},
            {"name": "HEADER", "function": "Header for USB-C breakout signals"},
        ],
        "requirements": [
            {"id": "input", "sheet": "USB C INPUT", "role": "connector", "family": "usb-c"}
        ],
        "inter_sheet_nets": [
            {
                "name": net,
                "endpoints": [{"sheet": "USB C BREAKOUT"}, {"sheet": "HEADER"}],
            }
            for net in ("VBUS", "D+", "D-", "CC1", "CC2", "SBU1", "SBU2")
        ]
        + [{"name": "+5V", "endpoints": [{"sheet": "USB C INPUT"}, {"sheet": "POWER"}]}],
    }

    requirements = {
        row["sheet"]: row for row in _normalize_usb_c_requirements(payload)["requirements"]
    }

    assert set(requirements) == {"USB C BREAKOUT", "USB C INPUT"}
    assert requirements["USB C BREAKOUT"]["family"] == "usb-c-breakout"
    assert requirements["USB C BREAKOUT"]["ports"] == {
        "gnd": "GND",
        "vbus": "VBUS",
        "usb_dp": "D+",
        "usb_dm": "D-",
        "cc1": "CC1",
        "cc2": "CC2",
        "sbu1": "SBU1",
        "sbu2": "SBU2",
    }
    assert requirements["USB C INPUT"]["family"] == "usb-c-power-sink"
    assert requirements["USB C INPUT"]["ports"] == {"gnd": "GND", "vbus": "+5V"}


@pytest.fixture
def round10_breakout_contract():
    # Saved run_06_usb-c-full-breakout final candidate: D_P/D_N are bound by
    # both owners but absent from inter_sheet_nets. Keep the composite ownership
    # and real header pin order, not a guessed USB-to-header functional edge.
    receptacle_blocks = [
        "USB_C_RECEPTACLE",
        "VBUS_POWER_RAIL",
        "GND_REFERENCE",
        "CC_CONFIGURATION",
        "SBU_SIDEBAND",
        "HIGH_SPEED_DIFFERENTIAL",
    ]
    header_nets = [
        "VBUS",
        "GND",
        "CC1",
        "CC2",
        "SBU1",
        "SBU2",
        "TX1P",
        "TX1N",
        "RX1P",
        "RX1N",
        "TX2P",
        "TX2N",
        "RX2P",
        "RX2N",
        "D_P",
        "D_N",
    ]
    sheets = [
        {
            "name": "USB C RECEPTACLE",
            "stem": "USB_C_RECEPTACLE",
            "function": "Physical USB-C receptacle exposing VBUS, GND, CC1/CC2, SBU1/SBU2, and the high-speed differential pairs.",
        },
        {
            "name": "BREAKOUT HEADER",
            "stem": "BREAKOUT_HEADER",
            "function": "0.1-inch pin header breakout carrying all exposed USB-C signals to external wiring.",
        },
    ]
    payload = {
        "sheets": sheets,
        "power_nets": ["VBUS", "GND"],
        "rail_voltages": {"VBUS": 5.0, "GND": 0.0},
        "inter_sheet_nets": [
            {
                "name": name,
                "endpoints": [
                    {"sheet": sheet["name"], "direction": "bidirectional"} for sheet in sheets
                ],
            }
            for name in header_nets[:-2]
        ],
        "requirements": [
            {
                "id": "usb_c_receptacle_breakout",
                "sheet": "USB C RECEPTACLE",
                "role": "connector",
                "family": "usb-c-breakout",
                "functional_blocks": receptacle_blocks,
                "ports": {
                    "cc1": "CC1",
                    "cc2": "CC2",
                    "gnd": "GND",
                    "rx1n": "RX1N",
                    "rx1p": "RX1P",
                    "rx2n": "RX2N",
                    "rx2p": "RX2P",
                    "sbu1": "SBU1",
                    "sbu2": "SBU2",
                    "tx1n": "TX1N",
                    "tx1p": "TX1P",
                    "tx2n": "TX2N",
                    "tx2p": "TX2P",
                    "usb_dm": "D_N",
                    "usb_dp": "D_P",
                    "vbus": "VBUS",
                },
            },
            {
                "id": "breakout_header",
                "sheet": "BREAKOUT HEADER",
                "role": "connector",
                "family": "pin-header",
                "functional_blocks": ["HEADER_BREAKOUT"],
                "parameters": {"rows": 16},
                "ports": {f"pin{index}": net for index, net in enumerate(header_nets, 1)},
            },
        ],
    }
    # The saved functional graph routes signals through logical subfunctions
    # that the receptacle requirement explicitly owns.
    functions = [
        ("VBUS_POWER_RAIL", "power"),
        ("GND_REFERENCE", "ground"),
        ("CC_CONFIGURATION", "digital"),
        ("SBU_SIDEBAND", "analog"),
        ("HIGH_SPEED_DIFFERENTIAL", "rf"),
    ]
    state = {
        "functional_spec": {
            "blocks": [
                {
                    "name": name,
                    "category": "power"
                    if name in {"VBUS_POWER_RAIL", "GND_REFERENCE"}
                    else "interface",
                    "purpose": name,
                }
                for name in [*receptacle_blocks, "HEADER_BREAKOUT"]
            ],
            "connections": [
                {"from_block": source, "to_block": target, "signal_type": signal_type}
                for block, signal in functions
                for source, target in [("USB_C_RECEPTACLE", block), (block, "HEADER_BREAKOUT")]
                for signal_type in (
                    [signal, "ground"] if signal not in {"power", "ground"} else [signal]
                )
            ],
        }
    }
    return payload, state


def test_saved_breakout_rejects_unsupported_rows_before_model_bom(round10_breakout_contract):
    from kicraft.server.stage_contracts import StageSchemaError

    payload, state = round10_breakout_contract
    with pytest.raises(StageSchemaError) as rejected:
        _normalize_stage_response("architecture", payload, state)

    diagnostic = rejected.value.diagnostic
    assert diagnostic["code"] == "invalid_lowerer_parameters"
    assert len(diagnostic["evidence"]) == 1
    evidence = diagnostic["evidence"][0]
    assert evidence["requirement_id"] == "breakout_header"
    assert evidence["sheet"] == "BREAKOUT HEADER"
    assert evidence["parameter"] == "rows"
    assert evidence["value"] == 16
    assert evidence["choices"] == [1, 2]
    assert payload["requirements"][1]["parameters"] == {"rows": 16}


@pytest.fixture(params=["can", "badge"])
def round12_power_contract(request):
    # Exact power-domain requirements and purposes from the frozen round12 states.
    # Unrelated MCU/interface domains are omitted from this local contract fixture.
    if request.param == "can":
        requirements = [
            {
                "id": "power_input",
                "sheet": "POWER INPUT",
                "role": "power_input",
                "family": "power-input",
                "parameters": {"input_voltage": 5.0, "output_voltage": 3.3},
                "ports": {"gnd": "GND", "output": "+3V3"},
                "functional_blocks": ["POWER_INPUT"],
            }
        ]
        blocks = [
            {
                "name": "POWER_INPUT",
                "category": "power",
                "purpose": "Accepts external power and conditions it for the board.",
            }
        ]
        function = "External 5V power input with 3.3V regulator for the board."
        rails = {"+3V3": 3.3, "GND": 0.0}
    else:
        requirements = [
            {
                "id": "battery_holder",
                "sheet": "BATTERY INPUT",
                "role": "power_input",
                "family": "coin-cell-holder",
                "parameters": {"cell_format": "CR2032"},
                "ports": {"negative": "GND", "positive": "VBAT"},
                "functional_blocks": ["BATTERY_INPUT"],
            },
            {
                "id": "power_distribution",
                "sheet": "BATTERY INPUT",
                "role": "power_input",
                "family": "power-distribution",
                "parameters": {},
                "ports": {"gnd": "GND", "vbat": "VBAT"},
                "functional_blocks": ["POWER_CONVERSION"],
            },
        ]
        blocks = [
            {
                "name": "BATTERY_INPUT",
                "category": "power",
                "purpose": "CR2032 coin-cell holder providing the board's primary power source.",
            },
            {
                "name": "POWER_CONVERSION",
                "category": "power",
                "purpose": "Conditions and distributes battery power to the MCU and LEDs.",
            },
        ]
        function = (
            "CR2032 coin-cell holder providing the primary battery input "
            "and direct power distribution."
        )
        rails = {"VBAT": 3.0, "GND": 0.0}
    sheet = requirements[0]["sheet"]
    architecture = {
        "sheets": [{"name": sheet, "stem": sheet.replace(" ", "_"), "function": function}],
        "requirements": requirements,
        "rail_voltages": rails,
        "power_nets": list(rails),
        "inter_sheet_nets": [],
    }
    return architecture, {"functional_spec": {"blocks": blocks}}


@pytest.mark.parametrize("entry_point", ["architecture", "bom"])
def test_saved_power_contract_rejected_before_immutable_bom(round12_power_contract, entry_point):
    from kicraft.design.stage_semantics import diagnose_stage
    from kicraft.server.stage_contracts import StageSchemaError
    from kicraft.server.stage_work_units import plan_stage_work_units

    payload, state = round12_power_contract
    original = json.loads(json.dumps(payload))
    with pytest.raises(StageSchemaError) as rejected:
        if entry_point == "architecture":
            _normalize_stage_response("architecture", payload, state)
        else:
            plan_stage_work_units("bom", {**state, "architecture": payload}, {})
    diagnostic = rejected.value.diagnostic
    assert diagnostic["code"] == "unrealizable_power_requirement"
    owned = original["requirements"][-1]
    message = diagnostic["message"]
    assert owned["id"] in message
    assert owned["sheet"] in message
    assert all(f"ports.{key}={net!r}" in message for key, net in owned["ports"].items())
    assert all(block in message for block in owned["functional_blocks"])
    if owned["id"] == "power_input":
        assert "input_voltage=5.0" in message and "output_voltage=3.3" in message
        assert "input net=None" in message
    else:
        assert "battery_holder" in message
        assert "Conditions and distributes battery power to the MCU and LEDs." in message
    assert payload["requirements"] == original["requirements"]
    semantic_codes = {
        finding.code
        for finding in diagnose_stage(
            "architecture", brief="", upstream_state=state, candidate=original
        )
    }
    assert {row["code"] for row in diagnostic["evidence"]} <= semantic_codes


def test_explicit_power_repairs_preserve_functions_nets_and_real_hardware(round12_power_contract):
    from kicraft.design.lowering import lower_requirement
    from kicraft.server.stage_work_units import deterministic_bom_candidate, plan_stage_work_units

    payload, state = round12_power_contract
    original_nets = set(payload["power_nets"])
    repair = payload["requirements"][-1]
    if repair["id"] == "power_input":
        repair.update(
            role="regulator",
            family="ams1117-3v3",
            exact_part="AMS1117-3.3",
            parameters={},
            ports={"input": "VIN5", "output": "+3V3", "gnd": "GND"},
        )
        payload["rail_voltages"]["VIN5"] = 5.0
        payload["power_nets"].append("VIN5")
        payload["requirements"].append(
            {
                "id": "power_connector",
                "sheet": "POWER INPUT",
                "role": "connector",
                "family": "pin-header",
                "parameters": {"rows": 1},
                "ports": {"pin1": "VIN5", "pin2": "GND"},
                "functional_blocks": ["POWER_INPUT"],
            }
        )
    else:
        # An explicit capacitor provides physical conditioning on the same rail;
        # no implicit fold or change to the committed functional purpose.
        repair.update(
            role="analog_block",
            family="explicit-decoupling",
            parameters={"count": 1, "value": "100nF"},
            ports={"vdd": "VBAT", "gnd": "GND"},
        )
    canonical, _ = _normalize_stage_response("architecture", payload, state)
    planned_state = {**state, "architecture": canonical}
    units = plan_stage_work_units("bom", planned_state, {})
    owned_blocks = {
        block for row in canonical["requirements"] for block in row["functional_blocks"]
    }
    assert owned_blocks == {block["name"] for block in state["functional_spec"]["blocks"]}
    assert original_nets <= set(canonical["power_nets"])
    if repair["id"] == "power_input":
        (selection,) = canonical["recipe_selections"]
        assert selection["recipe"] == "ams1117-3v3@1"
        assert selection["requirement_ids"] == ["power_input"]
        assert selection["port_bindings"] == {"input": "VIN5", "output": "+3V3", "gnd": "GND"}
        assert [unit.requirement_ids for unit in units] == [("power_connector",)]
        connector = deterministic_bom_candidate(units[0], planned_state)
        assert connector["groups"][0]["symbol"] == "Connector_Generic:Conn_01x02"
    else:
        assert {unit.requirement_ids for unit in units} == {
            ("battery_holder",),
            ("power_distribution",),
        }
        artifacts = {row["id"]: lower_requirement(row) for row in canonical["requirements"]}
        assert {pin.net for pin in artifacts["power_distribution"].pins} == {"VBAT", "GND"}
        groups = [
            group
            for unit in units
            for group in deterministic_bom_candidate(unit, planned_state)["groups"]
        ]
        assert sorted(group["symbol"] for group in groups) == ["Device:Battery_Cell", "Device:C"]


@pytest.fixture
def round11_switch_contract():
    return {
        "sheets": [
            {"name": name, "stem": name.replace(" ", "_"), "function": name}
            for name in ("BOOT BUTTON", "RESET BUTTON")
        ],
        "power_nets": ["GND", "+3V3"],
        "inter_sheet_nets": [],
        "requirements": [
            {
                "id": f"req_{name}_button",
                "sheet": f"{name.upper()} BUTTON",
                "role": "user_io",
                "family": "switch-input",
                "functional_blocks": [f"{name.upper()}_BUTTON"],
                "parameters": {
                    "pull_policy": policy,
                    "active_level": active_level,
                    "resistance": 10000,
                },
                "ports": {"gnd": "GND", "signal": signal, "vdd": "+3V3"},
            }
            for name, policy, active_level, signal in (
                ("boot", "pull_down", "high", "BOOT0"),
                ("reset", "pull_up", "low", "NRST"),
            )
        ],
    }


def test_saved_switch_policies_reject_all_owners_then_accept_explicit_repair(
    round11_switch_contract,
):
    from kicraft.design.lowering import lower_requirement
    from kicraft.server.stage_contracts import StageSchemaError

    payload = round11_switch_contract
    with pytest.raises(StageSchemaError) as rejected:
        _normalize_stage_response("architecture", payload, {})

    diagnostic = rejected.value.diagnostic
    assert diagnostic["code"] == "invalid_lowerer_parameters"
    assert {
        (row["requirement_id"], row["sheet"], row["parameter"], row["value"])
        for row in diagnostic["evidence"]
    } == {
        ("req_boot_button", "BOOT BUTTON", "pull_policy", "pull_down"),
        ("req_reset_button", "RESET BUTTON", "pull_policy", "pull_up"),
    }
    for row in diagnostic["evidence"]:
        assert row["lowerer_id"] == "switch-input@1"
        assert row["choices"] == ["internal", "external"]
        assert row["missing"] is False
    assert [row["parameters"]["pull_policy"] for row in payload["requirements"]] == [
        "pull_down",
        "pull_up",
    ]

    for row in payload["requirements"]:
        row["parameters"]["pull_policy"] = "external"
    canonical, _ = _normalize_stage_response("architecture", payload, {})
    artifacts = {row["id"]: lower_requirement(row) for row in canonical["requirements"]}
    assert {(pin.role, pin.pin): pin.net for pin in artifacts["req_boot_button"].pins} == {
        ("switch", "1"): "BOOT0",
        ("switch", "2"): "+3V3",
        ("pulldown", "1"): "GND",
        ("pulldown", "2"): "BOOT0",
    }
    assert {(pin.role, pin.pin): pin.net for pin in artifacts["req_reset_button"].pins} == {
        ("switch", "1"): "NRST",
        ("switch", "2"): "GND",
        ("pullup", "1"): "+3V3",
        ("pullup", "2"): "NRST",
    }


@pytest.mark.parametrize(
    ("parameters", "key", "value", "choices", "missing"),
    [
        ({"active_level": "high"}, "pull_policy", None, ["internal", "external"], True),
        (
            {"pull_policy": "external", "active_level": "toggle"},
            "active_level",
            "toggle",
            ["low", "high"],
            False,
        ),
    ],
)
def test_architecture_requires_switch_implementation_and_finite_actuation(
    round11_switch_contract, parameters, key, value, choices, missing
):
    from kicraft.server.stage_contracts import StageSchemaError

    payload = round11_switch_contract
    payload["requirements"] = payload["requirements"][:1]
    payload["requirements"][0]["parameters"] = parameters
    with pytest.raises(StageSchemaError) as rejected:
        _normalize_stage_response("architecture", payload, {})
    diagnostic = rejected.value.diagnostic
    assert diagnostic["code"] == "invalid_lowerer_parameters"
    assert len(diagnostic["evidence"]) == 1
    evidence = diagnostic["evidence"][0]
    assert evidence["requirement_id"] == "req_boot_button"
    assert evidence["sheet"] == "BOOT BUTTON"
    assert evidence["parameter"] == key
    assert evidence["value"] == value
    assert evidence["choices"] == choices
    assert evidence["missing"] is missing


def test_parameter_validation_preserves_unknown_families_and_unproven_exact_parts(
    round11_switch_contract,
):
    from kicraft.design.lowering import lower_requirement

    payload = round11_switch_contract
    unknown, exact = payload["requirements"]
    unknown["family"] = "custom-switch"
    exact["parameters"]["pull_policy"] = "external"
    exact["exact_part"] = "PTS645SL50SMTR92 LFS"
    canonical, _ = _normalize_stage_response("architecture", payload, {})
    assert set(canonical["unresolved_requirement_ids"]) == {"req_boot_button", "req_reset_button"}
    by_id = {row["id"]: row for row in canonical["requirements"]}
    assert by_id["req_boot_button"]["parameters"]["pull_policy"] == "pull_down"
    assert by_id["req_reset_button"]["exact_part"] == "PTS645SL50SMTR92 LFS"
    assert lower_requirement(by_id["req_reset_button"]) is None


def test_saved_breakout_requires_both_explicit_data_boundaries(round10_breakout_contract):
    from kicraft.server.stage_contracts import StageSchemaError

    payload, state = round10_breakout_contract
    # Repair the independent invalid rows=16 contract before testing net ownership.
    payload["requirements"][1]["parameters"]["rows"] = 2
    original_ports = {row["id"]: dict(row["ports"]) for row in payload["requirements"]}
    with pytest.raises(StageSchemaError) as rejected:
        _normalize_stage_response("architecture", payload, state)

    diagnostic = rejected.value.diagnostic
    assert diagnostic["code"] == "missing_typed_inter_sheet_contract"
    for net in ("D_P", "D_N"):
        evidence = next(row for row in diagnostic["evidence"] if f"net {net!r}:" in row)
        for owner in (
            "usb_c_receptacle_breakout",
            "breakout_header",
            "USB C RECEPTACLE",
            "BREAKOUT HEADER",
        ):
            assert owner in evidence

    # Architecture must reject, not guess a wire or delete a binding. Explicit
    # provider repair supplies the actual boundary and permits normalization.
    assert {row["name"] for row in payload["inter_sheet_nets"]}.isdisjoint({"D_P", "D_N"})
    for net in ("D_P", "D_N"):
        payload["inter_sheet_nets"].append(
            {
                "name": net,
                "endpoints": [
                    {"sheet": name, "direction": "bidirectional"}
                    for name in ("USB C RECEPTACLE", "BREAKOUT HEADER")
                ],
            }
        )
    normalized, _ = _normalize_stage_response("architecture", payload, state)
    assert {row["id"]: row["ports"] for row in normalized["requirements"]} == original_ports
    assert {
        row["name"]: {endpoint["sheet"] for endpoint in row["endpoints"]}
        for row in normalized["inter_sheet_nets"]
        if row["name"] in {"D_P", "D_N"}
    } == {net: {"USB C RECEPTACLE", "BREAKOUT HEADER"} for net in ("D_P", "D_N")}


def test_saved_breakout_same_sheet_data_bindings_remain_local(round10_breakout_contract):
    payload, state = round10_breakout_contract
    payload["requirements"][1]["parameters"]["rows"] = 2
    payload["sheets"] = payload["sheets"][:1]
    payload["inter_sheet_nets"] = []
    payload["requirements"][1]["sheet"] = "USB C RECEPTACLE"
    original_ports = {row["id"]: dict(row["ports"]) for row in payload["requirements"]}

    normalized, _ = _normalize_stage_response("architecture", payload, state)

    assert normalized["inter_sheet_nets"] == []
    assert {row["id"]: row["ports"] for row in normalized["requirements"]} == original_ports


@pytest.mark.parametrize("linked", [False, True])
def test_existing_boundary_requires_only_signal_linked_typed_owners(linked):
    from kicraft.server.stage_contracts import StageSchemaError

    names = ("SOURCE", "SINK", "LOCAL")
    payload = {
        "sheets": [{"name": name, "stem": name, "function": name} for name in names],
        "power_nets": [],
        "inter_sheet_nets": [
            {
                "name": "DATA",
                "endpoints": [
                    {"sheet": name, "direction": "passive"} for name in ("SOURCE", "SINK")
                ],
            }
        ],
        "requirements": [
            {
                "id": name.lower(),
                "sheet": name,
                "role": "connector",
                "family": "pin-header",
                "functional_blocks": [name],
                "ports": {"pin1": "DATA", "pin2": "GND"},
            }
            for name in names
        ],
    }
    state = {
        "functional_spec": {
            "blocks": [{"name": name, "category": "interface", "purpose": name} for name in names],
            "connections": [
                {"from_block": "SOURCE", "to_block": "SINK", "signal_type": "digital"},
                {
                    "from_block": "SOURCE",
                    "to_block": "LOCAL",
                    "signal_type": "digital" if linked else "ground",
                },
            ],
        }
    }
    if linked:
        with pytest.raises(StageSchemaError) as rejected:
            _normalize_stage_response("architecture", payload, state)
        diagnostic = rejected.value.diagnostic
        assert diagnostic["code"] == "missing_typed_inter_sheet_contract"
        evidence = diagnostic["evidence"]
        assert len(evidence) == 1
        assert all(identity in evidence[0] for identity in ("DATA", "source", "local", "LOCAL"))
    else:
        normalized, _ = _normalize_stage_response("architecture", payload, state)
        assert normalized["requirements"][2]["ports"] == {"pin1": "DATA", "pin2": "GND"}
        assert normalized["inter_sheet_nets"] == payload["inter_sheet_nets"]


def test_typed_signal_boundary_check_runs_after_existing_peer_completion():
    payload = {
        "sheets": [
            {"name": "MCU", "stem": "MCU", "function": "ESP32-S3 controller"},
            {"name": "USB", "stem": "USB", "function": "USB-C device connector"},
        ],
        "rail_voltages": {"+3V3": 3.3, "VBUS": 5.0},
        "power_nets": ["+3V3", "VBUS", "GND"],
        "inter_sheet_nets": [],
        "requirements": [
            {
                "id": "mcu_core",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": "esp32-s3-module",
                "exact_part": "ESP32-S3-MINI-1-N8",
                "functional_blocks": ["CONTROLLER"],
                "ports": {"vdd": "+3V3", "gnd": "GND", "usb_dp": "D_P", "usb_dm": "D_N"},
                "interfaces": ["usb_device"],
            },
            {
                # A native-family MCU's USB pair must reach a physical USB data
                # connector: a bare header no longer satisfies the contract.
                "id": "usb_connector",
                "sheet": "USB",
                "role": "connector",
                "family": "usb-c-usb2-device",
                "exact_part": "USB-C-USB2-DEVICE",
                "functional_blocks": ["USB"],
                "ports": {"vbus": "VBUS", "gnd": "GND", "usb_dp": "D_P", "usb_dm": "D_N"},
            },
        ],
    }
    state = {
        "functional_spec": {
            "blocks": [
                {"name": "CONTROLLER", "category": "process", "purpose": "Native USB device"},
                {"name": "USB", "category": "interface", "purpose": "Expose native USB"},
            ],
            "connections": [{"from_block": "CONTROLLER", "to_block": "USB", "signal_type": "bus"}],
        }
    }

    normalized, _ = _normalize_stage_response("architecture", payload, state)

    assert {
        row["name"]: {endpoint["sheet"] for endpoint in row["endpoints"]}
        for row in normalized["inter_sheet_nets"]
        if row["name"] in ("D_P", "D_N")
    } == {net: {"MCU", "USB"} for net in ("D_P", "D_N")}
    assert normalized["recipe_selections"][0]["port_bindings"]["usb_dp"] == "D_P"
    assert normalized["recipe_selections"][0]["port_bindings"]["usb_dm"] == "D_N"


@pytest.mark.parametrize("local_rails", [[], ["VBUS", "+5V"]])
def test_usb_connector_requires_explicit_binding_for_ambiguous_supply(local_rails):
    from kicraft.server.stage_contracts import StageSchemaError

    connector = {
        "id": "usb",
        "sheet": "USB",
        "role": "connector",
        "family": "usb-c",
    }
    payload = {
        "power_nets": ["VBUS", "+5V", "GND"],
        "requirements": [connector],
        "inter_sheet_nets": [
            {"name": net, "endpoints": [{"sheet": "USB"}, {"sheet": "POWER"}]}
            for net in local_rails
        ],
    }

    with pytest.raises(StageSchemaError, match="ambiguous connector net aliases"):
        _normalize_usb_c_requirements(payload)

    connector["ports"] = {"vbus": "VBUS"}
    requirement = _normalize_usb_c_requirements(payload)["requirements"][0]
    assert requirement["ports"] == {"vbus": "VBUS", "gnd": "GND"}
    assert payload["power_nets"] == ["VBUS", "+5V", "GND"]


def test_typed_mcp6001_requirement_resolves_to_verified_follower_recipe():
    payload = {
        "topologies": {"BUFFER": "voltage follower"},
        "rail_voltages": {"VCC": 5.0},
        "comms_protocols": [],
        "mcu_present": False,
        "sheets": [
            {
                "name": "OP AMP BUFFER",
                "stem": "OP_AMP_BUFFER",
                "function": "Op-amp voltage buffer",
            },
            {"name": "HEADER", "stem": "HEADER", "function": "Analog input and output"},
        ],
        "power_nets": ["VCC", "GND"],
        "inter_sheet_nets": [
            {
                "name": name,
                "endpoints": [
                    {"sheet": "OP AMP BUFFER", "direction": direction},
                    {"sheet": "HEADER", "direction": "passive"},
                ],
            }
            for name, direction in (("AIN", "input"), ("AOUT", "output"))
        ],
        "assumptions": ["Use a rail-to-rail voltage follower"],
        "requirements": [
            {
                "id": "buffer",
                "sheet": "OP AMP BUFFER",
                "role": "analog_block",
                "family": "mcp6001-follower",
                "exact_part": "MCP6001T-I/OT",
                "parameters": {},
                "ports": {
                    "input": "AIN",
                    "output": "AOUT",
                    "vdd": "VCC",
                    "gnd": "GND",
                },
                "interfaces": [],
                "functional_blocks": ["BUFFER"],
            }
        ],
    }

    canonical, _ = _normalize_stage_response("architecture", payload, {"intent": {}})

    assert canonical["recipe_selections"][0]["recipe"] == "mcp6001-follower@1"
    assert canonical["requirements"][0]["functional_blocks"] == ["BUFFER"]


def test_usb_connector_completion_preserves_polarity_and_header_hardware():
    payload = {
        "power_nets": ["VBUS", "GND"],
        "sheets": [
            {"name": "USB C RECEPTACLE", "function": "USB-C receptacle exposing CC and SBU"},
            {
                "name": "HEADER BREAKOUT",
                "function": "Header breakout for all exposed USB-C signals",
            },
        ],
        "requirements": [],
        "inter_sheet_nets": [
            {
                "name": name,
                "endpoints": [
                    {"sheet": sheet} for sheet in ("USB C RECEPTACLE", "HEADER BREAKOUT")
                ],
            }
            for name in (
                "D+",
                "D-",
                "CC1",
                "CC2",
                "SBU1",
                "SBU2",
                "TX1+",
                "TX1-",
                "TX2+",
                "TX2-",
                "RX1+",
                "RX1-",
                "RX2+",
                "RX2-",
            )
        ],
    }
    requirements = _normalize_usb_c_requirements(payload)["requirements"]
    assert [row["sheet"] for row in requirements] == ["USB C RECEPTACLE"]
    assert requirements[0]["family"] == "usb-c-breakout"
    assert requirements[0]["ports"]["usb_dp"] == "D+"
    assert requirements[0]["ports"]["usb_dm"] == "D-"
    assert set(requirements[0]["ports"].values()) == {
        "VBUS",
        "GND",
        "D+",
        "D-",
        "CC1",
        "CC2",
        "SBU1",
        "SBU2",
        "TX1+",
        "TX1-",
        "TX2+",
        "TX2-",
        "RX1+",
        "RX1-",
        "RX2+",
        "RX2-",
    }


def test_external_pd_cc_contract_never_lowers_to_parallel_sink_resistors():
    from kicraft.design.lowering import lower_requirement
    from kicraft.server.stage_contracts import StageSchemaError

    connector = {
        "id": "usb",
        "sheet": "USB",
        "role": "connector",
        "family": "usb-c-connector",
        "ports": {"vbus": "VBUS", "gnd": "GND"},
    }
    payload = {
        "power_nets": ["VBUS", "+5V", "GND"],
        "requirements": [connector],
        "inter_sheet_nets": [
            {"name": net, "endpoints": [{"sheet": "USB"}, {"sheet": "PD"}]}
            for net in ("CC1", "CC2")
        ],
    }
    normalized = _normalize_usb_c_requirements(payload)["requirements"][0]
    artifact = lower_requirement(normalized)
    assert artifact is not None
    assert {group.reference_prefix for group in artifact.groups} == {"J"}
    assignments = {pin.pin: pin.net for pin in artifact.pins}
    assert assignments["A5"] == "CC1"
    assert assignments["B5"] == "CC2"

    connector["exact_part"] = "USB-C-5V-SINK"
    with pytest.raises(StageSchemaError, match="active sink"):
        _normalize_usb_c_requirements(payload)


@pytest.mark.parametrize(
    ("recipe_id", "explicit_requirement"),
    [("usb-c-usb2-device@1", True), ("usb-c-5v-sink@1", False)],
)
def test_usb_shield_preserves_active_sink_hardware(recipe_id, explicit_requirement):
    from kicraft.design.recipes import expand_recipe, get_recipe

    definition = get_recipe(recipe_id)
    ports = {"vbus": "VBUS", "gnd": "GND", "shield": "SHIELD"}
    if explicit_requirement:
        ports.update(usb_dp="USB_DP", usb_dm="USB_DM")
    payload = {
        "power_nets": ["VBUS", "GND"],
        "sheets": [{"name": "USB", "function": "USB-C connector"}],
        "requirements": [
            {
                "id": "usb",
                "sheet": "USB",
                "role": "connector",
                "family": definition.family,
                "exact_part": definition.exact_part,
                "ports": ports,
            }
        ]
        if explicit_requirement
        else [],
        "inter_sheet_nets": [
            {"name": "SHIELD", "endpoints": [{"sheet": "USB"}, {"sheet": "CHASSIS"}]}
        ],
    }
    requirement = _normalize_usb_c_requirements(payload)["requirements"][0]
    assert requirement["family"] == definition.family
    assert requirement["ports"] == ports
    expansion = expand_recipe(
        models.RecipeSelection(
            recipe=recipe_id,
            instance="usb",
            sheets={"power": "USB"},
            port_bindings=requirement["ports"],
        )
    )
    connector = next(part for part in expansion.parts if part.recipe_role == "connector")
    pull_refs = {part.ref for part in expansion.parts if part.recipe_role == "cc_pulldown"}
    assert len(pull_refs) == 2
    shield = next(net for net in expansion.connections if net.net_name == "SHIELD")
    assert {(pin.ref, pin.pin) for pin in shield.endpoints} == {
        (connector.ref, str(pin)) for pin in range(1, 5)
    }
    ground = next(net for net in expansion.connections if net.net_name == "GND")
    assert {(ref, "2") for ref in pull_refs} <= {(pin.ref, pin.pin) for pin in ground.endpoints}


def test_usb_pd_requirement_is_not_downgraded_to_fixed_sink():
    requirement = {
        "id": "pd",
        "sheet": "INPUT",
        "role": "power_input",
        "family": "usb-c-pd-trigger",
        "parameters": {"selectable": True},
        "ports": {"vbus": "VBUS", "gnd": "GND"},
    }
    result = _normalize_usb_c_requirements(
        {
            "power_nets": ["VBUS", "GND"],
            "requirements": [requirement],
        }
    )
    assert result["requirements"] == [requirement]


def test_model_owned_pd_controller_matches_typed_family_by_real_identity():
    from kicraft.server.stage_contracts import BomComponentGroup, _requirement_owns_protected_group

    group = BomComponentGroup(
        id="pd_controller",
        reference_prefix="U",
        quantity=1,
        value="CH224K",
        mpn="CH224K",
        symbol="ch224k:CH224K",
        footprint="ch224k:ESSOP-10_L4.9-W3.9-P1.0-LS6.0-TL-EP",
        sheet="INPUT",
    )
    assert _requirement_owns_protected_group(
        group,
        [
            {
                "id": "negotiation",
                "family": "usb-pd-trigger",
                "role": "power_input",
            }
        ],
    )
    assert not _requirement_owns_protected_group(
        group,
        [
            {
                "id": "status",
                "family": "led-resistor",
                "role": "user_io",
            }
        ],
    )


def test_protected_connector_ownership_cannot_override_an_explicit_mpn():
    from kicraft.server.stage_contracts import BomComponentGroup, _requirement_owns_protected_group

    group = BomComponentGroup(
        id="usb_connector",
        reference_prefix="J",
        quantity=1,
        value="TYPE-C-31-M-12",
        mpn="TYPE-C-31-M-12",
        symbol="Connector:USB_C_Receptacle_USB2.0",
        footprint="Connector_USB:USB_C_Receptacle_HRO_TYPE-C-31-M-12",
        sheet="INPUT",
    )
    requirement = {
        "id": "usb_connector",
        "family": "usb-c-breakout",
        "exact_part": "USB4085-GF-A",
    }
    assert not _requirement_owns_protected_group(group, [requirement])
    requirement["exact_part"] = "TYPE-C-31-M-12"
    assert _requirement_owns_protected_group(group, [requirement])
    misleading_value = group.model_copy(update={"value": "USB4085-GF-A"})
    requirement["exact_part"] = "USB4085-GF-A"
    assert not _requirement_owns_protected_group(misleading_value, [requirement])


def test_noninteractive_contract_requires_a_complete_slot():
    contract = build_stage_response_contract("architecture", {}, allow_questions=False)

    assert "anyOf" not in contract.schema
    assert "sheets" in contract.schema["required"]
    assert (
        contract.response_format["json_schema"]["name"]
        == "kicraft_architecture_response_v2_noninteractive"
    )


@pytest.mark.parametrize(
    "architecture",
    [
        None,
        {},
        {"sheets": []},
        {"sheets": [{"name": ""}]},
        {"sheets": [{"name": "POWER"}, {"name": "POWER"}]},
        {"sheets": ["POWER"]},
        {"sheets": [{"name": 7}]},
    ],
)
def test_bom_contract_rejects_missing_duplicate_and_malformed_architecture(architecture):
    with pytest.raises(ValueError):
        build_stage_response_contract("bom", {"architecture": architecture})


def test_work_unit_contracts_are_v3_scoped_and_prompt_examples_are_unit_shaped():
    state = {
        "architecture": {
            "sheets": [{"name": "POWER"}, {"name": "MCU"}],
        }
    }
    bom = build_stage_response_contract("bom", state, bom_sheet="MCU")
    wiring = build_stage_response_contract("wiring", state, wiring_refs=("U1", "R1"))

    assert bom.response_format["json_schema"]["name"] == "kicraft_bom_response_v3"
    assert bom.schema["$defs"]["BomComponentGroup"]["properties"]["sheet"]["enum"] == ["MCU"]
    assert bom.schema["properties"]["groups"]["minItems"] == 1
    assert "groups" in bom.schema["required"]
    assert wiring.response_format["json_schema"]["name"] == "kicraft_wiring_response_v3"
    for definition in ("ConnectedPinAssignment", "NoConnectPinAssignment"):
        assert wiring.schema["$defs"][definition]["properties"]["ref"]["enum"] == [
            "U1",
            "R1",
        ]
    prompt = _build_system(
        wiring,
        work_unit_instructions=(
            '{"unit_id":"wiring-u000","target_sheet":"MCU","owned_refs":["U1","R1"]}'
        ),
    )
    assert "=== WORK UNIT ===" in prompt
    assert "Prior accepted-unit summaries are immutable" in prompt
    assert "VALID work unit" in prompt


def test_recipe_only_bom_work_unit_may_return_no_additional_groups():
    state = {
        "architecture": {
            "sheets": [{"name": "MCU"}],
            "requirements": [{"id": "mcu", "sheet": "MCU"}],
            "recipe_selections": [
                {
                    "recipe": "rp2040-minimal@2",
                    "instance": "mcu",
                    "sheets": {"mcu": "MCU", "io": "MCU"},
                    "requirement_ids": ["mcu"],
                }
            ],
        }
    }

    contract = build_stage_response_contract("bom", state, bom_sheet="MCU")

    assert "minItems" not in contract.schema["properties"]["groups"]


@pytest.mark.parametrize("allow_questions", [True, False])
def test_architecture_provider_requires_explicit_nonempty_implementation_contracts(allow_questions):
    contract = build_stage_response_contract("architecture", {}, allow_questions=allow_questions)
    architecture = contract.schema
    assert "requirements" in architecture["required"]
    assert architecture["properties"]["requirements"]["minItems"] == 1
    requirement = contract.schema["$defs"]["CircuitRequirement"]
    assert "functional_blocks" in requirement["required"]
    assert requirement["properties"]["functional_blocks"]["minItems"] == 1
    for field in ("recipe_resolution", "unresolved_requirement_ids", "protected_identities"):
        assert field not in architecture["properties"]
        assert field in models.Architecture.model_fields


@pytest.mark.parametrize("allow_questions", [True, False])
def test_architecture_ownership_schema_uses_committed_block_names(allow_questions):
    state = {
        "functional_spec": {
            "blocks": [
                {"name": "USB_PD_CONTROL", "category": "power", "purpose": "Negotiate power"},
                {"name": "OUTPUT", "category": "interface", "purpose": "Expose power"},
            ]
        }
    }
    contract = build_stage_response_contract("architecture", state, allow_questions=allow_questions)
    requirement = contract.schema["$defs"]["CircuitRequirement"]
    ownership = requirement["properties"]["functional_blocks"]
    assert ownership["items"]["enum"] == ["USB_PD_CONTROL", "OUTPUT"]
    assert "PD controller" not in ownership["items"]["enum"]
    assert ownership["minItems"] == 1
    assert "functional_blocks" in requirement["required"]
    state["functional_spec"]["blocks"][0]["name"] = "OTHER_CONTROL"
    other = build_stage_response_contract("architecture", state, allow_questions=allow_questions)
    assert other.schema["$defs"]["CircuitRequirement"]["properties"]["functional_blocks"]["items"][
        "enum"
    ] == ["OTHER_CONTROL", "OUTPUT"]
    assert ownership["items"]["enum"] == ["USB_PD_CONTROL", "OUTPUT"]


@pytest.mark.parametrize("selection_sheet", ["MAIN", "OTHER"])
def test_scoped_recipe_sheet_cannot_excuse_unresolved_typed_work(selection_sheet):
    state = {
        "architecture": {
            "sheets": [{"name": "MAIN"}, {"name": "OTHER"}],
            "requirements": [{"id": "pd", "sheet": "MAIN", "family": "usb-pd-trigger"}],
            "unresolved_requirement_ids": ["pd"],
            "recipe_selections": [
                {
                    "recipe": "usb-c-5v-sink@1",
                    "instance": "usb",
                    "sheets": {"power": selection_sheet},
                    "requirement_ids": ["pd"],
                }
            ],
        }
    }
    contract = build_stage_response_contract("bom", state, bom_sheet="MAIN", allow_questions=False)
    assert contract.schema["properties"]["groups"]["minItems"] == 1
    assert "groups" in contract.schema["required"]


def test_folding_preserves_explicit_membership_and_unimplemented_sheets():
    architecture = models.Architecture(
        sheets=[
            models.Sheet(name=name, stem=name, function=name) for name in ("POWER", "USB", "HEADER")
        ],
        power_nets=["VBUS", "GND"],
        inter_sheet_nets=[
            models.InterSheetNet(
                name="DATA",
                endpoints=[
                    models.SheetPin(sheet="USB", direction="passive"),
                    models.SheetPin(sheet="HEADER", direction="passive"),
                ],
            )
        ],
        requirements=[
            models.CircuitRequirement(
                id="usb",
                sheet="USB",
                role="power_input",
                family="usb-c-power-sink",
                functional_blocks=["USB_INPUT"],
            ),
            models.CircuitRequirement(
                id="header",
                sheet="HEADER",
                role="connector",
                family="header",
                functional_blocks=["BREAKOUT"],
            ),
        ],
        recipe_selections=[
            models.RecipeSelection(
                recipe="usb-c-5v-sink@1",
                instance="usb",
                sheets={"power": "POWER"},
                requirement_ids=["usb"],
            )
        ],
        unresolved_requirement_ids=["header"],
    )
    folded = _fold_recipe_covered_sheets(architecture)
    assert {sheet.name for sheet in folded.sheets} == {"POWER", "HEADER"}
    assert [(row.sheet, row.functional_blocks) for row in folded.requirements] == [
        ("POWER", ["USB_INPUT"]),
        ("HEADER", ["BREAKOUT"]),
    ]
    assert {endpoint.sheet for endpoint in folded.inter_sheet_nets[0].endpoints} == {
        "POWER",
        "HEADER",
    }
    assert folded.unresolved_requirement_ids == ["header"]
    assert folded.recipe_selections[0].requirement_ids == ["usb"]

    architecture.unresolved_requirement_ids.append("usb")
    unfolded = _fold_recipe_covered_sheets(architecture)
    assert {sheet.name for sheet in unfolded.sheets} == {"POWER", "USB", "HEADER"}


def test_passive_led_headroom_is_required_before_model_owned_array_bom():
    from kicraft.server.stage_contracts import StageSchemaError

    payload = {
        "sheets": [{"name": "LED ARRAY", "stem": "LED_ARRAY", "function": "indicators"}],
        "power_nets": ["VBAT", "GND"],
        "inter_sheet_nets": [],
        "requirements": [
            {
                "id": "led_array",
                "sheet": "LED ARRAY",
                "role": "driver",
                "family": "led-current-resistor",
                "parameters": {"rail_voltage": 3.0, "led_vf": 3.0, "target_current_ma": 1},
                "ports": {
                    "drive": "LED1",
                    "led1": "LED1",
                    "led2": "LED2",
                    "gnd": "GND",
                    "vdd": "VBAT",
                },
            }
        ],
    }
    with pytest.raises(StageSchemaError) as rejected:
        _normalize_stage_response("architecture", payload, {})
    diagnostic = rejected.value.diagnostic
    assert diagnostic["code"] == "invalid_lowerer_parameters"
    assert diagnostic["evidence"][0]["requirement_id"] == "led_array"
    assert diagnostic["evidence"][0]["parameter"] == "rail_voltage"

    payload["requirements"][0]["parameters"]["led_vf"] = 2.0
    normalized, _ = _normalize_stage_response("architecture", payload, {})
    assert normalized["requirements"][0]["ports"] == payload["requirements"][0]["ports"]
