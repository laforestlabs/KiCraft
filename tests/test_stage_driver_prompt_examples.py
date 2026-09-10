"""The worked examples embedded in the stage prompts MUST validate against the
real slot models (2026-07-19 review §7.1) — a schema change that breaks an
example must fail here, not teach the production model a guaranteed bounce.
"""

from __future__ import annotations

import json
import pytest

from kicraft.server.stage_contracts import (
    StageQuestionResponse,
    _normalize_bom_stage_response,
    _normalize_usb_c_requirements,
    _normalize_stage_response,
    _normalize_wiring_stage_response,
    _response_schema,
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


def test_examples_ride_the_system_prompt():
    assert _WORKED_EXAMPLES["bom"] in build_system("bom")
    assert _WORKED_EXAMPLES["wiring"] in build_system("wiring")
    assert "Worked example" not in build_system("intent")


def test_bom_system_prompt_carries_collection_bounds() -> None:
    prompt = build_system("bom")
    assert "`groups` collection must contain at most 64 items total" in prompt
    assert "at most 64 items per `sheet`" in prompt
    assert "`arrays` collection must contain at most 100 items total" in prompt
    assert "`assumptions` collection must contain at most 32 items total" in prompt
    assert "`substitutions` collection must contain at most 32 items total" in prompt
    assert "BOUNDED OUTPUT POLICY" not in build_system("wiring")
    assert "BOUNDED OUTPUT POLICY" not in build_system("bom", ())


def test_structured_stage_prompts_carry_degeneracy_bounds() -> None:
    functional = build_system("functional_spec")
    architecture = build_system("architecture")

    assert "`blocks` collection must contain at most 32 items total" in functional
    assert "every `name` value must be unique" in functional
    assert "`sheets` collection must contain at most 32 items total" in architecture
    assert "`inter_sheet_nets` collection must contain at most 128 items total" in architecture


def test_bom_contract_closes_group_sheet_and_reuses_schema_object():
    names = ["ADDRESSABLE LED OUTPUT", "SPEAKER OUTPUT"]
    state = {"architecture": {"sheets": [{"name": name} for name in names]}}
    contract = build_stage_response_contract("bom", state)

    definitions = contract.schema["$defs"]
    assert definitions["BomComponentGroup"]["properties"]["sheet"]["enum"] == names
    assert contract.response_format["json_schema"]["schema"] is contract.schema
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
            {"name": "usb_dp", "endpoints": []},
            {"name": "usb_dm", "endpoints": []},
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


def test_mcp6001_architecture_default_resolves_to_verified_follower_recipe():
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
            }
        ],
        "power_nets": ["VCC", "GND"],
        "inter_sheet_nets": [],
        "assumptions": ["Use MCP6001 as the rail-to-rail voltage follower (defaulted)"],
        "requirements": [
            {
                "id": "buffer",
                "sheet": "OP AMP BUFFER",
                "role": "analog_block",
                "family": "opamp-buffer",
                "parameters": {},
                "ports": {"input": "VCC", "output": "VCC"},
                "interfaces": [],
            }
        ],
    }

    canonical, _ = _normalize_stage_response("architecture", payload, {"intent": {}})

    assert canonical["requirements"][0]["exact_part"] == "MCP6001T-I/OT"
    assert canonical["recipe_selections"][0]["recipe"] == "mcp6001-follower@1"


def test_question_branch_and_non_bom_contracts_are_unchanged():
    question = StageQuestionResponse.model_json_schema()
    contract = build_stage_response_contract("architecture", {})
    expected = _response_schema("architecture")
    assert contract.schema == expected
    assert contract.schema["anyOf"][1] == {
        key: value for key, value in question.items() if key != "$defs"
    }


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
    bom_variant = next(
        variant for variant in bom.schema["anyOf"] if "groups" in variant.get("properties", {})
    )
    assert bom_variant["properties"]["groups"]["minItems"] == 1
    assert "groups" in bom_variant["required"]
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
    bom_variant = next(
        variant for variant in contract.schema["anyOf"] if "groups" in variant.get("properties", {})
    )

    assert "minItems" not in bom_variant["properties"]["groups"]
