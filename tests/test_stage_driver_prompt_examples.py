"""The worked examples embedded in the stage prompts MUST validate against the
real slot models (2026-07-19 review §7.1) — a schema change that breaks an
example must fail here, not teach the production model a guaranteed bounce.
"""

from __future__ import annotations

import json
import pytest

from kicraft.design import models
from kicraft.server.stage_contracts import (
    StageQuestionResponse,
    _response_schema,
    build_stage_response_contract,
)
from kicraft.server.stage_prompts import _WORKED_EXAMPLES, build_system as _build_system


def build_system(stage: str, collection_bounds=None) -> str:
    state = {"architecture": {"sheets": [{"name": "POWER"}]}} if stage == "bom" else {}
    return _build_system(build_stage_response_contract(stage, state), collection_bounds)


def test_bom_example_validates_against_the_model():
    slot = json.loads(_WORKED_EXAMPLES["bom"])
    bom = models.BOM.model_validate(slot)
    assert [p.ref for p in bom.parts] == ["U1", "C1", "C2", "R1", "R2", "J1"]
    assert bom.ic_groups["U1"] == ["C1", "C2"]


def test_wiring_example_validates_as_bom_wiring_fields():
    slot = json.loads(_WORKED_EXAMPLES["wiring"])
    # Wiring commits into bom.connections/no_connect_pins; validate through
    # the BOM model carrying the same parts as the bom example.
    base = json.loads(_WORKED_EXAMPLES["bom"])
    base["connections"] = slot["connections"]
    base["no_connect_pins"] = slot["no_connect_pins"]
    bom = models.BOM.model_validate(base)
    nets = {c.net_name for c in bom.connections}
    assert nets == {"VIN", "+3V3", "GND", "NRST", "BOOT0"}


def test_examples_ride_the_system_prompt():
    assert _WORKED_EXAMPLES["bom"] in build_system("bom")
    assert _WORKED_EXAMPLES["wiring"] in build_system("wiring")
    assert "Worked example" not in build_system("intent")


def test_bom_system_prompt_carries_collection_bounds() -> None:
    prompt = build_system("bom")
    assert "`parts` collection must contain at most 500 items total" in prompt
    assert "at most 450 items per `sheet`" in prompt
    assert "BOUNDED OUTPUT POLICY" not in build_system("wiring")
    assert "BOUNDED OUTPUT POLICY" not in build_system("bom", ())


def test_bom_contract_closes_both_sheet_fields_and_reuses_schema_object():
    names = ["ADDRESSABLE LED OUTPUT", "SPEAKER OUTPUT"]
    state = {"architecture": {"sheets": [{"name": name} for name in names]}}
    contract = build_stage_response_contract("bom", state)

    definitions = contract.schema["$defs"]
    assert definitions["BomPart"]["properties"]["sheet"]["enum"] == names
    assert definitions["BomPartRun"]["properties"]["sheet"]["enum"] == names
    assert contract.response_format["json_schema"]["schema"] is contract.schema
    assert "ADDRESSABLE LED OTPUT" not in names
    assert "SPEAKER OTPUT" not in names

    prompt = _build_system(contract)
    encoded = prompt.split("string patterns are strict):\n", 1)[1].split(
        "\nWorked example", 1
    )[0]
    assert json.loads(encoded) == contract.schema
    assert "SHEET NAMES ARE CLOSED" in prompt


def test_question_branch_and_non_bom_contracts_are_unchanged():
    question = StageQuestionResponse.model_json_schema()
    contract = build_stage_response_contract("architecture", {})
    expected = _response_schema("architecture")
    assert contract.schema == expected
    assert contract.schema["anyOf"][1] == {
        key: value for key, value in question.items() if key != "$defs"
    }


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
