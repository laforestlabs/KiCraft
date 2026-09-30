"""A design-data refusal of an obligation row stays a repairable contract defect.

Measured live on the held-out brief `stm32-four-adc-usb`
(`logs/self_eval/pipeline_consolidation_20260928/held_out_{r1,r2,r3}/run_02_stm32-four-adc-usb/
events.jsonl`): every trial died at stage 1 with

    stage_done intent ok=false failure_kind=invalid_schema
    1 validation error for IntentStageResponse
    obligations.13.quantitative
      Value error, scalar quantitative obligation needs value only
      input_value={'kind': 'quantitative', ...0x48, 0x49, 0x4A, 0x4B'}, input_type=dict

The brief names four I2C addresses (0x48, 0x49, 0x4A, 0x4B), so the model stated them as a
`quantitative` obligation with no numeric value -- an address set is not a scalar measurement --
and the pydantic ``ValidationError`` reached the driver as unusable provider output. It answered
with a generic re-emit, so a brief whose design data is fine never got the one restatement only
its writer can make. The same run's first draft failed the same way one row over
(`board_max_dimensions`: `relation` `maximum` with both `value` and `maximum` set).

`_obligation_shape_refusal` translates exactly those refusals into the repairable
`invalid_obligation_shape` / `repair_required` diagnostic, quoting the offending row (the four
addresses included) and naming the slot's home for a non-measurement: `constraints`. Every other
schema error is left alone, and `decode_stage_response` still labels it `invalid_schema`.
"""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from kicraft.server import stage_contracts as sc
from kicraft.server.stage_runtime import (
    AttemptOutcome,
    PreparedStage,
    ProviderFacts,
    decode_stage_response,
)

#: The terminal intent draft of `stm32-four-adc-usb` (held_out_r1, run_02), verbatim from the
#: answer deltas. Row 13 is the one the recorded `stage_done` names.
RECORDED_DRAFT = """
{
 "goal": "Build a two-layer USB-C-powered STM32F103C8T6 data-acquisition board with sixteen analog inputs and four shared-I2C ADCs.",
 "constraints": [
  "Two-layer PCB",
  "USB-C powered",
  "STM32F103C8T6",
  "Four ADS1115 converters sharing I2C",
  "ADS1115 I2C addresses: 0x48, 0x49, 0x4A, and 0x4B",
  "Expose all sixteen single-ended analog inputs and ground on screw terminals",
  "Analog input range: ground to 3.0 V",
  "ADS1115 supply voltage: 3.3 V",
  "3.3 V buck supply sized for at least 200 mA",
  "No external power loads",
  "Board size: within 100 x 100 mm",
  "Physical SWD programming header",
  "Reset button"
 ],
 "named_parts": ["STM32F103C8T6", "TLV62569DBVR", "ADS1115IDGSR", "USB-C"],
 "inferred_expertise": "intermediate",
 "assumptions": [
  "Ground reference is shared across analog input terminals and board ground (defaulted)",
  "USB-C power input uses the standard 5 V VBUS supply (defaulted)"
 ],
 "form_factor": null,
 "project_stem": "STM32_DATA_ACQUISITION",
 "questions": [],
 "obligations": [
  {"kind": "physical", "original_obligation_id": "usb-c-connector", "component_class": "usb-c-connector"},
  {"kind": "physical", "original_obligation_id": "microcontroller", "component_class": "microcontroller"},
  {"kind": "physical", "original_obligation_id": "buck-converter", "component_class": "buck-converter"},
  {"kind": "physical", "original_obligation_id": "ads1115", "component_class": "adc"},
  {"kind": "quantity", "original_obligation_id": "four-adcs", "subject": "adc", "minimum": 4},
  {"kind": "physical", "original_obligation_id": "screw-terminal", "component_class": "screw-terminal"},
  {"kind": "quantity", "original_obligation_id": "sixteen-analog-inputs", "subject": "screw-terminal", "minimum": 17},
  {"kind": "physical", "original_obligation_id": "swd-header", "component_class": "programming-header"},
  {"kind": "physical", "original_obligation_id": "reset-button", "component_class": "pushbutton"},
  {"kind": "quantitative", "original_obligation_id": "board-maximum-size", "quantity": "board maximum dimension", "relation": "maximum", "value": 100, "minimum": null, "maximum": null, "unit": "mm"},
  {"kind": "quantitative", "original_obligation_id": "supply-output-voltage", "quantity": "3.3 V supply output", "relation": "equal", "value": 3.3, "minimum": null, "maximum": null, "unit": "V"},
  {"kind": "quantitative", "original_obligation_id": "supply-current", "quantity": "3.3 V supply current capacity", "relation": "minimum", "value": 200, "minimum": null, "maximum": null, "unit": "mA"},
  {"kind": "quantitative", "original_obligation_id": "analog-input-voltage", "quantity": "analog input voltage range", "relation": "range", "value": null, "minimum": 0, "maximum": 3.0, "unit": "V"},
  {"kind": "quantitative", "original_obligation_id": "adc-i2c-addresses", "quantity": "ADS1115 I2C addresses", "relation": "equal", "value": null, "minimum": null, "maximum": null, "unit": "addresses 0x48, 0x49, 0x4A, 0x4B"},
  {"kind": "negative", "original_obligation_id": "no-external-power-loads", "absent_class": "external-power-load"}
 ]
}
"""

ADDRESSES = "0x48, 0x49, 0x4A, 0x4B"


def recorded_draft() -> dict:
    return json.loads(RECORDED_DRAFT)


def decode(raw: str, *, stage: str = "intent") -> AttemptOutcome:
    """One provider reply through the site that labels the failure the release gate reads.

    `decode_stage_response` labels a refusal `contract_rejected` (a schema-clean candidate a
    deterministic design contract refused) exactly when it carries a diagnostic, and
    `invalid_schema` (unusable provider output) when it does not.
    """
    prepared = PreparedStage(
        stage=stage,
        prompt_state={},
        extras={},
        base_messages=(),
        contract=None,
        policy=None,
        tools=None,
        executor=None,
    )
    facts = ProviderFacts(
        raw=raw,
        finish="stop",
        rounds=None,
        tool_calls=None,
        cost_usd=0.0,
        had_content=True,
        loop_detected=False,
        collection_limit=None,
        loop_abort_reason=None,
        collection_counts={},
    )
    return decode_stage_response(prepared, facts)


def test_the_recorded_brief_is_refused_as_a_repairable_design_defect():
    """The four-address row is a design defect with an itemized repair, not unusable output."""
    with pytest.raises(sc.StageSchemaError) as caught:
        sc._normalize_stage_response("intent", recorded_draft(), {})
    diagnostic = caught.value.diagnostic
    assert diagnostic is not None
    assert diagnostic["code"] == "invalid_obligation_shape"
    assert diagnostic["severity"] == "repair_required"
    assert "adc-i2c-addresses" in diagnostic["message"]
    # The correct home for an address set is named, so the repair cannot drop the datum.
    assert "`constraints`" in diagnostic["message"]
    assert "Never invent a value" in diagnostic["message"]
    # The refusal names the row the recorded `stage_done` blamed, and quotes it whole.
    assert diagnostic["evidence"][0].startswith("obligations.13.quantitative: ")
    row = json.loads(diagnostic["evidence"][0].split(": ", 1)[1])
    assert row["original_obligation_id"] == "adc-i2c-addresses"
    assert row["unit"] == f"addresses {ADDRESSES}"
    assert row["value"] is None  # reported as stated: no measurement is invented for it
    assert isinstance(caught.value.__cause__, ValidationError)


def _states_every_address(text: str) -> bool:
    return all(address in text for address in ("0x48", "0x49", "0x4A", "0x4B"))


def test_the_refusal_keeps_the_draft_and_every_address_it_stated():
    """Nothing is dropped or rewritten: the correction the driver asks for starts from the draft."""
    payload = recorded_draft()
    before = copy.deepcopy(payload)
    with pytest.raises(sc.StageSchemaError):
        sc._normalize_stage_response("intent", payload, {})
    assert payload == before
    assert any(_states_every_address(constraint) for constraint in payload["constraints"])


def test_the_repaired_draft_commits_with_the_four_addresses_still_in_the_slot():
    """The repair the diagnostic asks for is reachable and lossless."""
    payload = recorded_draft()
    payload["obligations"] = [
        row
        for row in payload["obligations"]
        if row["original_obligation_id"] != "adc-i2c-addresses"
    ]
    committed, _expanded = sc._normalize_stage_response("intent", payload, {})
    assert [row["original_obligation_id"] for row in committed["obligations"]] == [
        "usb-c-connector",
        "microcontroller",
        "buck-converter",
        "ads1115",
        "four-adcs",
        "screw-terminal",
        "sixteen-analog-inputs",
        "swd-header",
        "reset-button",
        "board-maximum-size",
        "supply-output-voltage",
        "supply-current",
        "analog-input-voltage",
        "no-external-power-loads",
    ]
    assert any(_states_every_address(constraint) for constraint in committed["constraints"])


def test_the_other_recorded_shape_defect_is_refused_the_same_way():
    """The first draft's row: `relation` `maximum` with both `value` and `maximum` set."""
    payload = recorded_draft()
    payload["obligations"][13] = {
        "kind": "quantitative",
        "original_obligation_id": "board_max_dimensions",
        "quantity": "board maximum dimensions",
        "relation": "maximum",
        "value": 100,
        "minimum": None,
        "maximum": 100,
        "unit": "mm",
    }
    with pytest.raises(sc.StageSchemaError) as caught:
        sc._normalize_stage_response("intent", payload, {})
    diagnostic = caught.value.diagnostic
    assert diagnostic["code"] == "invalid_obligation_shape"
    assert diagnostic["severity"] == "repair_required"
    assert "board_max_dimensions" in diagnostic["message"]
    assert "`value` alone" in diagnostic["message"]


def test_a_fabrication_row_stating_a_unit_without_a_limit_is_refused_the_same_way():
    """The same defect class one row kind over: design data in a shape its `kind` cannot carry."""
    payload = {
        "goal": "A printed copper heatsink area on the board",
        "project_stem": "COPPER_HEATSINK",
        "obligations": [
            {
                "kind": "fabrication",
                "original_obligation_id": "copper-heatsink-area",
                "feature": "copper-area",
                "unit": "mm2",
            }
        ],
    }
    with pytest.raises(sc.StageSchemaError) as caught:
        sc._normalize_stage_response("intent", payload, {})
    diagnostic = caught.value.diagnostic
    assert diagnostic["code"] == "invalid_obligation_shape"
    assert diagnostic["severity"] == "repair_required"
    assert "copper-heatsink-area" in diagnostic["message"]
    assert "drop the `unit`" in diagnostic["message"]
    assert decode(json.dumps(payload)).payload["failure_kind"] == "contract_rejected"


def test_the_driver_labels_the_refusal_contract_rejected_not_invalid_schema():
    outcome = decode(RECORDED_DRAFT)
    assert outcome.kind == "recoverable_failure"
    assert outcome.payload["failure_kind"] == "contract_rejected"
    assert outcome.payload["diagnostic"]["code"] == "invalid_obligation_shape"
    assert outcome.payload["diagnostic"]["severity"] == "repair_required"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"project_stem": "STM32_DATA_ACQUISITION"}, id="missing-goal"),
        pytest.param({"goal": 5, "project_stem": "A_BOARD"}, id="goal-not-a-string"),
        pytest.param({"goal": "A board"}, id="project-stem-missing"),
        pytest.param(
            {"goal": "A board", "project_stem": "lower-case", "obligations": []},
            id="project-stem-not-upper-snake",
        ),
        pytest.param(
            {"goal": "A board", "project_stem": "A_BOARD", "constraints": "not a list"},
            id="constraints-not-a-list",
        ),
        pytest.param(
            {"goal": "A board", "project_stem": "A_BOARD", "invented_key": 1},
            id="unknown-slot-field",
        ),
        pytest.param(
            {
                "goal": "A board",
                "project_stem": "A_BOARD",
                "obligations": [
                    {
                        "kind": "quantitative",
                        "original_obligation_id": "board-size",
                        "quantity": "board size",
                        "relation": "maximum",
                        "value": 100,
                        "unit": "mm",
                        "invented_key": 1,
                    }
                ],
            },
            id="unknown-obligation-field",
        ),
    ],
)
def test_a_non_design_schema_error_still_classifies_as_a_schema_error(payload):
    with pytest.raises(sc.StageSchemaError) as caught:
        sc._normalize_stage_response("intent", payload, {})
    assert caught.value.diagnostic is None
    assert decode(json.dumps(payload)).payload["failure_kind"] == "invalid_schema"


def test_a_matching_validator_message_outside_an_obligation_row_is_left_alone():
    """The translation is keyed on the row, so a lookalike refusal elsewhere stays a schema error."""
    error = ValidationError.from_exception_data(
        "IntentStageResponse",
        [
            {
                "type": "value_error",
                "loc": ("goal",),
                "input": {},
                "ctx": {
                    "error": ValueError("scalar quantitative obligation needs value only")
                },
            }
        ],
    )
    assert sc._obligation_shape_refusal(error, {"goal": "A board"}) is None
    assert sc._obligation_shape_refusal(ValueError("something else"), recorded_draft()) is None
