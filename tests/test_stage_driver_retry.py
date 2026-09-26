"""Tests for the stage driver's self-correction feedback and per-stage retries.

Pure functions, no OpenRouter / network: exercises the retry-message construction
and the per-stage retry budget that help the wiring stage converge.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import requests

from kicraft.design.models import StageDiagnostic
from kicraft.server import stage_runtime as stage_driver_mod
from kicraft.server import stage_pipeline
from kicraft.server.config import DESIGN_PROFILES, Settings
from kicraft.server.stage_bom_tools import BOM_TOOLS
from kicraft.server.stage_contracts import (
    _extract_json,
    _normalize_bom_stage_response,
    _normalize_wiring_stage_response,
    build_stage_response_contract,
)
from kicraft.server.stage_prompts import build_system as _build_system
from kicraft.server.stage_runtime import (
    EXTERNAL_LOAD_CURRENT_QUESTION,
    _classify_parse_failure,
    _commit_rejection_signature,
    _design_reasoning,
    _normalize_questions,
    _redacted_rejection_facts,
    _retry_feedback,
    _stage_max_retries,
    _stage_max_tokens,
)
from kicraft.server.stage_state_io import attach_questions as _attach_questions
from kicraft.server.stage_work_units import StageWorkUnit
from kicraft.server.session import run_session


def test_wiring_prompt_state_exposes_only_owned_bom_refs_and_local_contracts():
    state = {
        "architecture": {
            "sheets": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            "requirements": [
                {"id": "local", "sheet": "A"},
                {"id": "remote", "sheet": "C"},
            ],
            "inter_sheet_nets": [
                {
                    "name": "A_TO_B",
                    "endpoints": [{"sheet": "A"}, {"sheet": "B"}],
                },
                {
                    "name": "C_ONLY",
                    "endpoints": [{"sheet": "C"}],
                },
            ],
            "recipe_selections": [
                {"recipe": "local@1", "sheets": {"main": "A"}},
                {"recipe": "remote@1", "sheets": {"main": "C"}},
            ],
            "recipe_resolution": [
                {"requirement_id": "local", "recipe": "local@1"},
                {"requirement_id": "remote", "recipe": "remote@1"},
            ],
            "unresolved_requirement_ids": ["local", "remote"],
        },
        "bom": {
            "parts": [{"ref": "R1"}, {"ref": "U1"}],
            "connections": [
                {"endpoints": [{"ref": "R1"}, {"ref": "R1"}]},
                {"endpoints": [{"ref": "R1"}, {"ref": "U1"}]},
            ],
            "no_connect_pins": [{"ref": "R1"}, {"ref": "U1"}],
        },
    }
    unit = StageWorkUnit(
        "wiring-u000",
        "wiring",
        "A",
        refs=("R1",),
        requirement_ids=("local",),
    )

    visible = stage_driver_mod._work_unit_prompt_state(unit, state)

    assert [part["ref"] for part in visible["bom"]["parts"]] == ["R1"]
    assert len(visible["bom"]["connections"]) == 1
    assert visible["bom"]["no_connect_pins"] == [{"ref": "R1"}]
    assert [sheet["name"] for sheet in visible["architecture"]["sheets"]] == ["A", "B"]
    assert [net["name"] for net in visible["architecture"]["inter_sheet_nets"]] == ["A_TO_B"]
    assert [row["id"] for row in visible["architecture"]["requirements"]] == ["local"]
    assert [row["recipe"] for row in visible["architecture"]["recipe_selections"]] == ["local@1"]
    assert [row["requirement_id"] for row in visible["architecture"]["recipe_resolution"]] == [
        "local"
    ]
    assert visible["architecture"]["unresolved_requirement_ids"] == ["local"]


def test_bom_prompt_keeps_owned_contract_and_read_only_exact_part_exclusions():
    owned = {
        "id": "button",
        "sheet": "USER IO",
        "role": "user_io",
        "family": "button",
        "parameters": {"active_low": True},
        "ports": {"out": "BOOT0"},
        "interfaces": [{"kind": "digital", "voltage": 3.3}],
    }
    sibling = {
        "id": "mcu",
        "sheet": "USER IO",
        "role": "mcu",
        "exact_part": "STM32F103C8T6",
    }
    recipe = {
        "recipe": "stm32f103c8t6-minimal@1",
        "instance": "mcu",
        "sheets": {"mcu": "USER IO"},
        "requirement_ids": ["mcu"],
    }
    state = {
        "intent": {
            "named_parts": ["STM32F103C8T6"],
            "constraints": ["hand solderable"],
            "inferred_expertise": "expert",
        },
        "functional_spec": {"blocks": [{"name": "USB"}, {"name": "MCU"}]},
        "architecture": {
            "sheets": [
                {"name": "USER IO", "stem": "USER_IO", "function": "MCU and button"},
                {"name": "USB", "function": "USB receptacle"},
            ],
            "topologies": {"USER_IO": "MCU with button", "USB": "USB device"},
            "requirements": [owned, sibling],
            "recipe_selections": [recipe],
            "recipe_resolution": [{"requirement_id": "mcu", "recipe": recipe["recipe"]}],
            "unresolved_requirement_ids": ["button"],
            "protected_identities": ["stm32f103c8t6"],
            "rail_voltages": {"+3V3": 3.3},
            "power_nets": ["+3V3", "GND"],
            "inter_sheet_nets": [
                {"name": "BOOT0", "endpoints": [{"sheet": "USER IO"}, {"sheet": "USB"}]},
                {"name": "VBUS", "endpoints": [{"sheet": "USB"}]},
            ],
        },
    }
    original = json.loads(json.dumps(state))
    unit = StageWorkUnit(
        "bom-s000", "bom", "USER IO", requirement_ids=("button",), owned_roles=("user_io",)
    )

    visible = stage_driver_mod._work_unit_prompt_state(unit, state)
    boundary = json.loads(stage_driver_mod._work_unit_instructions(unit, 1, 2, state))

    assert visible["intent"] == state["intent"]
    assert visible["architecture"]["requirements"] == [owned]
    assert [sheet["name"] for sheet in visible["architecture"]["sheets"]] == ["USER IO"]
    assert visible["architecture"]["topologies"] == {"USER_IO": "MCU with button"}
    assert visible["architecture"]["inter_sheet_nets"] == [
        state["architecture"]["inter_sheet_nets"][0]
    ]
    assert visible["architecture"]["rail_voltages"] == {"+3V3": 3.3}
    assert visible["architecture"]["protected_identities"] == ["stm32f103c8t6"]
    assert visible["read_only_exclusions"] == {"requirements": [sibling], "recipes": [recipe]}
    assert boundary["owned_requirements"] == [owned]
    assert state == original


def build_system(stage: str, collection_bounds=None) -> str:
    state = {"architecture": {"sheets": [{"name": "POWER"}]}} if stage == "bom" else {}
    contract = build_stage_response_contract(stage, state)
    return _build_system(contract, collection_bounds)


def _group_payload(**overrides):
    payload = {
        "id": "capacitors",
        "reference_prefix": "C",
        "quantity": 2,
        "value": "100nF",
        "symbol": "Device:C",
        "footprint": "Capacitor_SMD:C_0603_1608Metric",
        "sheet": "MAIN",
    }
    payload.update(overrides)
    return payload


def test_group_bom_expands_large_array_to_canonical_parts():
    payload = {
        "groups": [
            _group_payload(
                id="leds",
                reference_prefix="D",
                quantity=400,
                value="LED",
                symbol="Device:LED",
                footprint="LED_SMD:LED_0805_2012Metric",
            )
        ],
        "arrays": [{"group_id": "leds", "pattern": "grid", "rows": 20, "cols": 20}],
    }
    canonical, expanded = _normalize_bom_stage_response(payload)
    assert expanded == 400
    assert [part["ref"] for part in canonical["parts"][:2]] == ["D1", "D2"]
    assert canonical["parts"][-1]["ref"] == "D400"
    assert canonical["arrays"][0]["refs"] == [f"D{number}" for number in range(1, 401)]
    assert "groups" not in canonical


def test_group_bom_allocates_repeated_prefixes_in_response_order():
    canonical, _ = _normalize_bom_stage_response(
        {
            "groups": [
                _group_payload(id="input_caps", quantity=2),
                _group_payload(id="output_cap", quantity=1, value="1uF"),
            ]
        }
    )
    assert [part["ref"] for part in canonical["parts"]] == ["C1", "C2", "C3"]


def test_group_bom_accepts_installed_phoenix_footprint_with_comma():
    footprint = (
        "TerminalBlock_Phoenix:TerminalBlock_Phoenix_MKDS-1,5-3-5.08_1x03_P5.08mm_Horizontal"
    )
    canonical, total = _normalize_bom_stage_response(
        {
            "groups": [
                _group_payload(
                    id="output_terminal",
                    reference_prefix="J",
                    quantity=1,
                    value="3-pin terminal block",
                    symbol="Connector_Generic:Conn_01x03",
                    footprint=footprint,
                    sheet="POWER",
                )
            ]
        }
    )
    assert total == 1
    assert canonical["parts"][0]["footprint"] == footprint


@pytest.mark.parametrize(
    "payload",
    [
        {
            "groups": [
                _group_payload(quantity=300),
                _group_payload(id="resistors", reference_prefix="R", quantity=201),
            ]
        },
        {"groups": [_group_payload(quantity=450), _group_payload(id="extra", quantity=1)]},
        {"groups": [_group_payload(), _group_payload()]},
        {"groups": [_group_payload()], "arrays": [{"group_id": "missing", "rows": 1, "cols": 2}]},
        {"parts": []},
        {"part_runs": []},
    ],
)
def test_group_bom_rejects_limits_duplicates_unknown_arrays_and_legacy_shapes(payload):
    with pytest.raises(ValueError):
        _normalize_bom_stage_response(payload)


def _wiring_prompt_state():
    return {
        "bom": {
            "parts": [
                {"ref": "U1", "sheet": "CONTROL"},
                {"ref": "R1", "sheet": "CONTROL"},
                {"ref": "J1", "sheet": "IO"},
            ]
        }
    }


def test_final_pin_assignments_derive_canonical_connections_and_sheets():
    canonical = _normalize_wiring_stage_response(
        {
            "pins": [
                {"ref": "U1", "pin": "1", "net": "SIG"},
                {"ref": "R1", "pin": "1", "net": "SIG"},
                {"ref": "J1", "pin": "1", "net": "SIG"},
                {"ref": "U1", "pin": "2", "no_connect": True},
            ]
        },
        _wiring_prompt_state(),
    )
    assert canonical == {
        "connections": [
            {
                "net_name": "SIG",
                "endpoints": [
                    {"ref": "U1", "pin": "1"},
                    {"ref": "R1", "pin": "1"},
                ],
                "sheet": "CONTROL",
            },
            {
                "net_name": "SIG",
                "endpoints": [{"ref": "J1", "pin": "1"}],
                "sheet": "IO",
            },
        ],
        "no_connect_pins": [{"ref": "U1", "pin": "2"}],
    }


@pytest.mark.parametrize(
    "pins",
    [
        [
            {"ref": "U1", "pin": "1", "net": "A"},
            {"ref": "U1", "pin": "1", "net": "B"},
        ],
        [{"ref": "X9", "pin": "1", "net": "A"}],
        [{"ref": "U1", "pin": "1", "net": "A", "no_connect": True}],
        [{"ref": "U1", "pin": "1"}],
    ],
)
def test_final_pin_assignments_reject_ambiguous_and_unknown_endpoints(pins):
    with pytest.raises(ValueError):
        _normalize_wiring_stage_response({"pins": pins}, _wiring_prompt_state())


def test_retry_feedback_includes_errors_and_offenders():
    out = {
        "ok": False,
        "errors": ["9.11 net coverage: uncovered pin(s)"],
        "offenders": ["U2.4 ('EN', unspecified) not in connections or no_connect_pins"],
    }
    msg = _retry_feedback(out)
    assert "9.11 net coverage" in msg  # the rule that failed
    assert "U2.4" in msg  # the exact pin the model must fix
    assert "preserv" in msg.lower()  # patch, do not redraft
    assert "only the slot" in msg.lower()


def test_retry_feedback_without_offenders_omits_that_line():
    msg = _retry_feedback({"ok": False, "errors": ["some other error"]})
    assert "some other error" in msg
    assert "offenders" not in msg  # no offenders line when none present


def test_architecture_retry_feedback_explains_missing_inter_sheet_nets():
    message = _retry_feedback(
        {
            "errors": ["fs-connection mapping: 2 connection(s) not mapped"],
            "offenders": [
                "connection 'MCU'→'LED' (digital) crosses sheets but has no inter_sheet_net"
            ],
        },
        stage="architecture",
    )
    assert "add every missing cross-sheet signal" in message
    assert "at least two endpoints" in message
    assert "exact canonical `sheets[].name`" in message


def test_retry_feedback_explains_series_path_for_dangling_terminal():
    msg = _retry_feedback(
        {
            "ok": False,
            "errors": ["9.15 no dangling signal nets: 1 signal net wires a single pin"],
            "offenders": ["net 'CAN_TX_MCU' on sheet 'CAN TRANSCEIVER' wires only R3.2"],
        },
        stage="wiring",
    )
    assert "each side has the resistor terminal plus a non-resistor endpoint" in msg
    assert "Do not assume the populated side is the source" in msg
    assert "moving the destination pin" not in msg


def test_retry_feedback_explains_complete_self_short_repair():
    msg = _retry_feedback(
        {
            "ok": False,
            "errors": ["9.17 two-terminal self-short: 1 part shorted"],
            "offenders": [
                "R3 (Device:R) has both terminals on net 'CAN_TX' -- the part is shorted"
            ],
        },
        stage="wiring",
    )
    assert "complete three-item change" in msg
    assert "MOVE the intended destination" in msg
    assert "Do not merely rename one part terminal" in msg


def test_parse_failure_classification_distinguishes_the_three_kinds():
    # finish=length with NO content is provider exhaustion (reasoning_loop),
    # finish=length WITH content is a truncated answer, any normal stop is
    # invalid_json — the durable taxonomy, never collapsed into one label.
    assert _classify_parse_failure("length", had_content=False) == "reasoning_loop"
    assert _classify_parse_failure("length", had_content=True) == "truncated_json"
    assert _classify_parse_failure("stop", had_content=True) == "invalid_json"
    assert _classify_parse_failure("stop", had_content=False) == "invalid_json"
    assert _classify_parse_failure(None, had_content=True) == "invalid_json"


def test_extract_json_rejects_trailing_prose_and_second_object():
    # A complete object followed by non-whitespace is invalid_json, not a
    # silent success that drops content (bom-stage-json-gaps plan).
    with pytest.raises(ValueError):
        _extract_json('{"a": 1} some prose')
    with pytest.raises(ValueError):
        _extract_json('{"a": 1} {"b": 2}')
    # fences and a leading preamble are still tolerated; braces inside strings
    # and nested objects parse correctly
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('here is the slot: {"a": 1}') == {"a": 1}
    assert _extract_json('{"a": "}", "b": {"c": 1}}') == {"a": "}", "b": {"c": 1}}


def test_wiring_gets_more_retries_than_the_simple_stages():
    assert _stage_max_retries("wiring", 2) == 7
    assert _stage_max_retries("intent", 2) == 2
    assert _stage_max_retries("functional_spec", 2) == 2


def test_caller_default_wins_when_higher_than_the_floor():
    assert _stage_max_retries("wiring", 8) == 8
    assert _stage_max_retries("intent", 8) == 8


def test_wiring_gets_a_larger_token_budget():
    assert _stage_max_tokens("wiring", 4096) >= 8192  # wiring floors higher
    assert _stage_max_tokens("intent", 4096) == 4096  # simple stages keep default
    assert _stage_max_tokens("wiring", 16000) == 16000  # a higher caller default wins


def test_complex_stages_get_larger_retry_budgets():
    assert _stage_max_retries("bom", 2) == 4
    assert _stage_max_retries("architecture", 2) == 3


def test_bom_has_a_symbol_search_tool():
    names = {t["function"]["name"] for t in BOM_TOOLS}
    assert "search_symbols" in names  # discover, do not guess
    assert {"list_parts", "lookup_symbol", "lookup_lcsc_id", "add_part_from_lcsc"} <= names


def test_bom_has_a_footprint_search_tool():
    names = {t["function"]["name"] for t in BOM_TOOLS}
    assert "search_footprints" in names  # footprint discovery, do not guess
    assert "lookup_footprint" in names  # verify a footprint exists + pad count


# ---- clarifying questions -------------------------------------------------


def test_normalize_questions_preserves_recommended_choices_and_drops_junk():
    qs = _normalize_questions(
        [
            {"text": "Battery chemistry?", "options": ["LiPo", "18650"], "blocking": True},
            {"text": "   ", "blocking": True},  # dropped: blank text
            {"nope": 1},
        ],  # dropped: not a question
        "intent",
    )
    assert len(qs) == 1
    q = qs[0]
    assert q["stage"] == "intent" and q["blocking"] is True
    assert q["options"] == ["LiPo", "18650"] and q["answer"] is None
    normalized = _normalize_questions(
        [
            {"text": "Supply voltage?", "blocking": True},
            {
                "text": "Choose an op amp. (Default: rail-to-rail.)",
                "options": ["Rail-to-rail", "Dual supply"],
                "blocking": True,
            },
        ],
        "architecture",
    )
    assert [question["blocking"] for question in normalized] == [False, True]
    from kicraft.server.stage_runtime import _questions_need_input

    assert _questions_need_input(
        [normalized[1]],
        "architecture",
        auto_default=False,
        answers=None,
        instruction=None,
        auto_default_questions=False,
    )
    capped = _normalize_questions(
        [
            {
                "text": "Choose a sourced part?",
                "options": ["A", "B", "C", "D", "E", "F"],
                "blocking": True,
            }
        ],
        "bom",
    )
    assert capped[0]["options"] == ["A", "B", "C", "D"]


def test_normalize_questions_carries_and_whitelists_reconcile_target():
    # A "bom" target is preserved (the pipeline can self-repair it); an unknown
    # target is dropped to None so a park can't route to an arbitrary/looping
    # stage, and an untagged question stays a plain user question.
    qs = _normalize_questions(
        [
            {
                "text": "Add 3 more 100nF for U1 DEC pins",
                "blocking": True,
                "reconcile_target": "bom",
            },
            {"text": "Active-high or active-low button?", "blocking": True},
            {"text": "route to nowhere", "blocking": True, "reconcile_target": "wiring"},
        ],
        "wiring",
    )
    assert [q["reconcile_target"] for q in qs] == ["bom", None, None]
    assert [q["blocking"] for q in qs] == [True, False, False]
    # the normalized dicts still validate as Question (schema-safe for state.json)
    from kicraft.design.models import Question

    for q in qs:
        Question.model_validate(q)


def test_explicit_question_policy_overrides_legacy_stage_and_answer_rules():
    """The persisted bool, unlike legacy None, controls all ordinary questions."""
    from kicraft.server.stage_runtime import (
        _auto_default_questions_enabled as auto_enabled,
        _questions_need_input as parks,
    )

    blocking = [{"text": "Which LoRa band?", "blocking": True}]

    # Legacy callers retain production's early-stage-only defaulting behavior.
    assert not parks(
        blocking,
        "architecture",
        auto_default=auto_enabled(
            "architecture", review_before_commit=False, auto_default_questions=None
        ),
        answers=None,
        instruction=None,
    )
    assert parks(
        blocking,
        "wiring",
        auto_default=auto_enabled("wiring", review_before_commit=False, auto_default_questions=None),
        answers=None,
        instruction=None,
    )

    # Explicit settings apply to every stage and do not lose False after a resume.
    assert not parks(
        blocking,
        "wiring",
        auto_default=True,
        answers=[{"text": "prior", "answer": "yes"}],
        instruction="Re-drive this stage.",
        auto_default_questions=True,
    )
    assert parks(
        blocking,
        "intent",
        auto_default=False,
        answers=[{"text": "prior", "answer": "yes"}],
        instruction="Re-drive this stage.",
        auto_default_questions=False,
    )

    # Reconciliation is internal pipeline control and always wins over either
    # ordinary-question policy.
    assert parks(
        [{"text": "add caps", "blocking": True, "reconcile_target": "bom"}],
        "architecture",
        auto_default=True,
        answers=None,
        instruction=None,
        auto_default_questions=True,
    )


def test_external_current_question_recommends_finding_the_fact_before_a_rating():
    questions = _normalize_questions([EXTERNAL_LOAD_CURRENT_QUESTION], "architecture")
    assert questions[0]["options"] == [
        "Determine the load current from the load datasheets first",
        "Up to 1 A",
        "Up to 2 A",
        "Another current requirement (enter amperes)",
    ]
    from kicraft.server.stage_runtime import _questions_need_input

    assert not _questions_need_input(
        questions,
        "architecture",
        auto_default=True,
        answers=None,
        instruction=None,
        auto_default_questions=True,
    )
    # This policy gate is intentionally pure: a missing user-owned physical fact
    # parks an interactive run before another provider call can invent a rating.
    assert _questions_need_input(
        questions,
        "architecture",
        auto_default=False,
        answers=None,
        instruction=None,
        auto_default_questions=False,
    )


def test_wiring_prompt_tells_model_to_self_repair_a_bom_shortfall():
    sysmsg = build_system("wiring")
    # wiring must be told to tag a BOM parts shortfall for automatic repair
    # instead of asking the user.
    assert "reconcile_target" in sysmsg
    assert '"bom"' in sysmsg


def test_bom_prompt_closes_architecture_decisions_and_demands_decoupling():
    sysmsg = build_system("bom")
    assert "ARCHITECTURE DECISIONS ARE CLOSED" in sysmsg
    assert "Never ask the user to choose an MCU" in sysmsg
    assert "emit only the named requirement and owned role" in sysmsg
    assert "Decoupling completeness" in sysmsg
    assert "per dedicated supply/decoupling pin" in sysmsg


def test_bom_reconcile_instruction_lists_the_missing_parts():
    from kicraft.server.web import _bom_reconcile_instruction

    instr = _bom_reconcile_instruction(
        [
            {"text": "Add three 100nF caps for U1 DEC3-DEC5", "reconcile_target": "bom"},
            {"text": "", "reconcile_target": "bom"},
        ]
    )  # blank text is skipped
    assert "Add three 100nF caps for U1 DEC3-DEC5" in instr
    assert "Do NOT ask the user" in instr
    assert instr.count("\n- ") == 1  # only the one non-blank deficit line


def test_retry_feedback_unknown_ref_in_wiring_points_at_reconcile(tmp_path):
    # WS6: an unknown-ref rejection in wiring must tell the model it cannot add
    # parts and to park with reconcile_target=bom, and list the real refs -- so it
    # stops inventing refs and burning its retry budget.
    from kicraft.server.stage_runtime import _retry_feedback

    out = {"errors": ["NetConnection 'PWR' references unknown ref 'Q99'"]}
    msg = _retry_feedback(out, stage="wiring", valid_refs=["C1", "U1"])
    assert "CANNOT add parts" in msg
    assert "reconcile_target" in msg and '"bom"' in msg
    assert "C1" in msg and "U1" in msg


def test_retry_feedback_no_reconcile_note_for_non_wiring_stage():
    from kicraft.server.stage_runtime import _retry_feedback

    out = {"errors": ["some symbol not found"]}
    msg = _retry_feedback(out, stage="bom")
    assert "reconcile_target" not in msg


def test_retry_feedback_power_name_as_ref_teaches_net_name_shape():
    # KC-6DCV66: the model wrote '+3V3'/'GND' as an endpoint.ref and got a raw

    # Pydantic regex dump. Feedback must name the fix (rails are net_name values,
    # not component refs), not just echo the regex.
    err = (
        "slot validation failed: 2 validation errors for BOM\n"
        "connections.7.endpoints.1.ref\n"
        "  Value error, PinEndpoint.ref '+3V3' must match ^[A-Z]+[0-9]+[A-Z0-9_-]*$ "
        "[type=value_error, input_value='+3V3', input_type=str]\n"
        "connections.9.endpoints.1.ref\n"
        "  Value error, PinEndpoint.ref 'GND' must match ^[A-Z]+[0-9]+[A-Z0-9_-]*$ "
        "[type=value_error, input_value='GND', input_type=str]"
    )
    msg = _retry_feedback({"ok": False, "errors": [err]}, stage="wiring")
    assert "is a net name" in msg
    assert "+3V3" in msg and "GND" in msg


def test_semantic_repair_explains_missing_esp32_supply_source():
    message = stage_driver_mod._semantic_repair_message(
        "architecture",
        [
            StageDiagnostic(
                code="architecture_rail_source_unspecified",
                severity="repair_required",
                message="missing source",
                evidence=["+3V3"],
                detector_version=1,
            )
        ],
    )
    assert "5V-to-3.3V regulator" in message
    assert "at least 1A" in message


def test_retry_feedback_power_name_as_ref_skipped_for_other_stages():
    err = "PinEndpoint.ref '+3V3' must match ^[A-Z]+[0-9]+[A-Z0-9_-]*$"
    msg = _retry_feedback({"ok": False, "errors": [err]}, stage="bom")
    assert "not a component ref" not in msg  # wiring-only guidance


def test_attach_questions_writes_open_questions(tmp_path):
    sp = tmp_path / ".kicraft" / "state.json"  # not yet created (a first-stage question)
    qs = _normalize_questions([{"text": "Q1", "blocking": True}], "intent")
    _attach_questions(sp, "intent", qs)
    sj = json.loads(sp.read_text())
    assert sj["open_questions"][0]["text"] == "Q1"
    assert sj["open_questions"][0]["stage"] == "intent"


def test_attach_questions_replaces_only_that_stage(tmp_path):
    sp = tmp_path / ".kicraft" / "state.json"
    sp.parent.mkdir(parents=True)
    sp.write_text(
        json.dumps(
            {
                "open_questions": [
                    {"text": "old-intent", "stage": "intent"},
                    {"text": "keep-arch", "stage": "architecture"},
                ]
            }
        )
    )
    _attach_questions(sp, "intent", _normalize_questions([{"text": "new-intent"}], "intent"))
    texts = {q["text"] for q in json.loads(sp.read_text())["open_questions"]}
    assert texts == {"new-intent", "keep-arch"}  # intent replaced, architecture kept


def test_build_system_offers_clarifying_questions():
    sysmsg = build_system("intent")
    assert '"questions"' in sysmsg  # the model is told it may ask
    assert "blocking" in sysmsg
    assert "2-4 concise suggested answers" in sysmsg
    assert "Never ask to confirm a default" in sysmsg


@pytest.mark.parametrize(
    ("stage", "code", "expected_option"),
    [
        (
            "functional_spec",
            "functional_spec_external_load_power_assumed",
            "Board supplies the external loads",
        ),
        (
            "architecture",
            "architecture_external_load_current_unspecified",
            "1 A",
        ),
    ],
)
def test_semantic_blocking_questions_include_concrete_options(stage, code, expected_option):
    message = stage_driver_mod._semantic_repair_message(
        stage,
        [
            StageDiagnostic(
                code=code,
                severity="repair_required",
                message="x",
                evidence=[],
                detector_version=1,
            )
        ],
    )
    assert expected_option in message


def test_bom_part_hints_extracts_pasted_lcsc_ids():
    from kicraft.server.stage_prompts import _bom_part_hints

    brief = (
        "ToF breakout with the sensor at "
        "https://www.lcsc.com/product-detail/C7386355.html and an LDO C6186"
    )
    hint = _bom_part_hints(brief, "also use c2924337 please")
    assert "C7386355" in hint and "C6186" in hint and "C2924337" in hint
    assert "add_part_from_lcsc" in hint


def test_bom_part_hints_ignores_refdes_and_embedded_runs():
    from kicraft.server.stage_prompts import _bom_part_hints

    # C1/C104 are refdes/values, C8051F320 is an MPN — none are LCSC ids.
    assert _bom_part_hints("decouple C1 with 100nF, C104 pattern, MCU C8051F320") == ""
    assert _bom_part_hints("", None) == ""


def test_bom_prompt_mentions_search_budget():
    msg = build_system("bom")
    assert "SEARCH BUDGET" in msg
    assert "STOP searching" in msg


# ---- in-stream reasoning-loop breakout (KC-VWW5X7) -------------------------


class _LoopGuard:
    def status(self):
        return {"spent_total_usd": 0.0}


class _LoopClient:
    """First N chat replies are a reasoning-loop abort; the next is a valid
    intent slot. Records the `reasoning` policy each call received."""

    def __init__(self, loop_replies, ok_json):
        self.loop_replies = loop_replies
        self.ok_json = ok_json
        self.reasoning_seen = []
        self.guard = _LoopGuard()
        self._n = 0

    def chat(
        self,
        messages,
        max_tokens=4096,
        temperature=0.2,
        progress=None,
        meta_ctx=None,
        reasoning=None,
        reasoning_guard=None,
        collection_bounds=(),
        response_format=None,
    ):
        self.reasoning_seen.append(reasoning)
        self._n += 1
        if self._n <= self.loop_replies:
            return {
                "text": "",
                "reasoning": "x" * 600,
                "finish_reason": "reasoning_loop",
                "loop_detected": True,
                "cost_usd": 0.0,
            }
        return {
            "text": self.ok_json,
            "reasoning": "",
            "finish_reason": "stop",
            "loop_detected": False,
            "cost_usd": 0.0,
        }


_OK_INTENT = json.dumps(
    {
        "goal": "a USB-powered LED",
        "constraints": [],
        "named_parts": [],
        "inferred_expertise": "intermediate",
        "assumptions": [],
        "project_stem": "USB_LED",
    }
)


def test_loop_detected_retries_reasoning_disabled_then_commits(tmp_path):
    client = _LoopClient(loop_replies=1, ok_json=_OK_INTENT)
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "ok"
    assert len(client.reasoning_seen) == 2
    assert client.reasoning_seen[1] == {"enabled": False}  # loop retry drops reasoning


def test_second_loop_fails_with_reasoning_loop_label(tmp_path):
    client = _LoopClient(loop_replies=99, ok_json=_OK_INTENT)
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "failed"
    last = res["results"][-1]
    assert last.get("error") == "reasoning_loop"
    assert len(client.reasoning_seen) == 2  # one initial + one anti-loop retry, then give up


def test_design_reasoning_policy_selection():
    class _S:
        def design_reasoning(self, stage):
            if stage in ("intent", "functional_spec"):
                return {"enabled": False}
            return {"max_tokens": 2048}

    class _C:
        s = _S()

    assert _design_reasoning(_C(), "intent") == {"enabled": False}
    assert _design_reasoning(_C(), "functional_spec") == {"enabled": False}
    assert _design_reasoning(_C(), "architecture") == {"max_tokens": 2048}
    defaults = Settings(api_key="test")
    assert defaults.design_reasoning_tokens == 0
    assert defaults.design_reasoning("architecture") == {"enabled": False}
    assert _design_reasoning(object(), "intent") is None  # mock: no .s policy


# ---- serialization recovery (bom-stage-programming-and-json-gaps) ---------


class _ScriptedClient:
    """Replies in order, one dict per completion; records (max_tokens,
    reasoning, serialization-flag) per call so the tests can assert the exact
    recovery contract: one plain tool-free call at the fixed cap."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

        class _G:
            def status(self):
                return {"spent_total_usd": 0.0}

        self.guard = _G()

    def with_design_profile(self, profile_name):
        profile = DESIGN_PROFILES[profile_name]
        clone = object.__new__(type(self))
        clone.replies = self.replies
        clone.calls = self.calls
        clone.guard = self.guard
        clone.s = replace(
            self.s,
            model=str(profile["model"]),
            design_profile=profile_name,
            provider_order=list(profile["provider_order"]),
            max_price_prompt=float(profile["max_price_prompt"]),
            max_price_completion=float(profile["max_price_completion"]),
        )
        return clone

    def _next_reply(self):
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return dict(reply)

    def chat(
        self,
        messages,
        max_tokens=4096,
        temperature=0.2,
        progress=None,
        meta_ctx=None,
        reasoning=None,
        reasoning_guard=None,
        collection_bounds=(),
        response_format=None,
    ):
        self.calls.append(
            {
                "temperature": temperature,
                "collection_bounds": collection_bounds,
                "max_tokens": max_tokens,
                "reasoning": reasoning,
                "serialization": bool((meta_ctx or {}).get("serialization")),
                "messages": list(messages),
                "response_format": response_format,
                "model": getattr(getattr(self, "s", None), "model", None),
                "guard": self.guard,
            }
        )
        return self._next_reply()

    def chat_with_tools(
        self,
        messages,
        tools,
        executor,
        max_tokens=4096,
        temperature=0.2,
        max_rounds=6,
        progress=None,
        meta_ctx=None,
        reasoning=None,
        reasoning_guard=None,
        collection_bounds=(),
        response_format=None,
    ):
        self.calls.append(
            {
                "temperature": temperature,
                "collection_bounds": collection_bounds,
                "max_tokens": max_tokens,
                "reasoning": reasoning,
                "serialization": bool((meta_ctx or {}).get("serialization")),
                "messages": list(messages),
                "response_format": response_format,
                "model": getattr(getattr(self, "s", None), "model", None),
                "guard": self.guard,
            }
        )
        r = self._next_reply()
        r.setdefault("rounds", 1)
        r.setdefault("tool_calls", 0)
        return r


def _ok_intent_reply():
    return {"text": _OK_INTENT, "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0}


def test_default_semantic_mode_repairs_explicit_intent_classification(tmp_path):
    brief = (
        "make a board with USB C PD power input (configured for 5V) to an "
        "ESP32-S3-WROOM-1-N16R8 that drives a HUB75 display"
    )
    initial = {
        "goal": brief,
        "constraints": [],
        "named_parts": [],
        "inferred_expertise": "intermediate",
        "assumptions": [],
        "project_stem": "ESP32S3_HUB75_DRIVER_BOARD_5V_US",
    }
    client = _ScriptedClient(
        [
            {
                "text": json.dumps(initial),
                "reasoning": "",
                "finish_reason": "stop",
                "cost_usd": 0.0,
            }
        ]
    )
    client.s = Settings(api_key="test")

    result = run_session(tmp_path, brief, ["intent"], client=client)

    stage = result["results"][0]
    assert result["status"] == "ok"
    assert stage["repair_attempted"] is False
    schema = json.dumps(client.calls[0]["response_format"])
    assert "Every explicit package" in schema
    assert "Every exact MPN" in schema
    assert stage["diagnostics"] == []
    assert stage["slot"]["constraints"] == [brief]
    assert stage["slot"]["named_parts"] == ["ESP32-S3-WROOM-1-N16R8"]
    state = json.loads((tmp_path / ".kicraft" / "state.json").read_text())
    assert state["project_stem"] == "ESP32S3_HUB75_DRIVER"
    assert len(client.calls) == 1


def test_commit_rejection_rows_validate_as_durable_diagnostics():
    """The row ``_commit_rejection_diagnostics`` writes must reload.

    ``StageDiagnostic`` forbids extra keys and needs a detector version; a row that fails it makes
    the saved state unloadable, so ``replay`` cannot reopen the very run the row was written for
    (the block-sheet mapping failure reproduced on the beacon brief did exactly that).
    """
    rows = stage_driver_mod._commit_rejection_diagnostics(
        {
            "errors": ["9.11 net coverage: unconnected net USB_D+"],
            "offenders": ["net USB_D+ has 1 pin(s)"],
        }
    )
    assert len(rows) == 1
    diag = StageDiagnostic.model_validate(rows[0])
    assert diag.code == "commit_gate_rejected"
    assert diag.severity == "fab_gate"
    assert diag.gate_codes == ["9.11"]
    assert diag.detector_version == stage_driver_mod.DETECTOR_VERSION


def test_semantic_repair_is_bounded_to_one_correction(tmp_path):
    brief = "USB-C 5V controller with a speaker output"
    intent = {
        "goal": brief,
        "constraints": ["USB-C input", "5V input", "speaker output"],
        "named_parts": [],
        "inferred_expertise": "intermediate",
        "assumptions": [],
        "project_stem": "USB_SPEAKER",
    }
    blocks = [
        {
            # The committed technology lives in the block NAME: the topology check reads the
            # names the writer undertakes to realize, not the behaviour prose it used to scan.
            "name": "PWM_CONTROLLER",
            "category": "process",
            "purpose": "Generates an amplified PWM audio output.",
        },
        {"name": "SPEAKER", "category": "drive", "purpose": "Drives the speaker output."},
        {"name": "POWER", "category": "power", "purpose": "Powers the controller"},
    ]
    connection = {
        "from_block": "PWM_CONTROLLER",
        "to_block": "SPEAKER",
        "signal_type": "analog",
        "description": "Audio output",
    }
    power_connection = {
        "from_block": "POWER",
        "to_block": "SPEAKER",
        "signal_type": "power",
        "description": "Speaker power",
    }
    initial = {
        "blocks": blocks,
        "connections": [connection, power_connection],
        "assumptions": ["USB-C input is configured for 5V (defaulted)."],
    }
    first_repair = {
        **initial,
        "blocks": [
            {
                "name": "CONTROLLER",
                "category": "process",
                "purpose": "Generates the speaker control signal.",
            },
            blocks[1],
            blocks[2],
        ],
        "connections": [{**connection, "from_block": "CONTROLLER"}, power_connection],
    }
    second_repair = {
        **first_repair,
        "connections": [
            {
                **connection,
                "from_block": "CONTROLLER",
                "signal_type": "other",
                "description": "Speaker control signal",
            },
            power_connection,
        ],
        "assumptions": [],
    }

    def reply(payload):
        return {
            "text": json.dumps(payload),
            "reasoning": "",
            "finish_reason": "stop",
            "cost_usd": 0.0,
        }

    client = _ScriptedClient(
        [reply(intent), reply(initial), reply(first_repair), reply(second_repair)]
    )
    client.s = Settings(api_key="test")

    events = []
    result = run_session(
        tmp_path,
        brief,
        ["intent", "functional_spec"],
        client=client,
        progress=events.append,
    )

    functional = result["results"][1]
    assert result["status"] == "ok"
    assert functional["attempts"] == 2
    assert functional["repair_attempted"] is True
    assert functional["repair_adopted"] is True
    # The adopted candidate is clean, so the stage's final row set is empty; the defect itself is
    # recorded on the attempt that raised it (asserted from the events below).
    assert functional["semantic_clean"] is True
    assert functional["diagnostics"] == []
    assert len(client.calls) == 3
    assert any(
        event.get("kind") == "stage_diagnostic"
        and event.get("code") == "functional_spec_premature_topology"
        and event.get("attempt") == 1
        for event in events
    )


def test_semantic_repair_review_names_the_adopted_attempt(tmp_path):
    """An adopted semantic repair is reviewed as the call that produced it.

    The review path (``review_before_commit``) stamps the ``candidate_review`` event and
    the review's stage_attempts row from the current_* context. The repair round advances
    ``attempts`` without syncing that context, so a walkthrough saw
    ``result["attempts"] == 2`` beside ``candidate_review attempt=1`` and a review row
    carrying the superseded first call's facts (Surprise-me seed 37, 2026-09-24).
    """
    brief = "USB-C 5V controller with a speaker output"
    intent = {
        "goal": brief,
        "constraints": ["USB-C input", "5V input", "speaker output"],
        "named_parts": [],
        "inferred_expertise": "intermediate",
        "assumptions": [],
        "project_stem": "USB_SPEAKER",
    }
    blocks = [
        {
            # The committed technology lives in the block NAME: the topology check reads the
            # names the writer undertakes to realize, not the behaviour prose it used to scan.
            "name": "PWM_CONTROLLER",
            "category": "process",
            "purpose": "Generates an amplified PWM audio output.",
        },
        {"name": "SPEAKER", "category": "drive", "purpose": "Drives the speaker output."},
        {"name": "POWER", "category": "power", "purpose": "Powers the controller"},
    ]
    connection = {
        "from_block": "PWM_CONTROLLER",
        "to_block": "SPEAKER",
        "signal_type": "analog",
        "description": "Audio output",
    }
    power_connection = {
        "from_block": "POWER",
        "to_block": "SPEAKER",
        "signal_type": "power",
        "description": "Speaker power",
    }
    initial = {
        "blocks": blocks,
        "connections": [connection, power_connection],
        "assumptions": ["USB-C input is configured for 5V (defaulted)."],
    }
    repaired = {
        **initial,
        # The repair drops the committed technology from the block NAME; the topology check
        # reads names, so this is the change that clears the diagnostic.
        "blocks": [
            {
                "name": "CONTROLLER",
                "category": "process",
                "purpose": "Generates the speaker control signal.",
            },
            blocks[1],
            blocks[2],
        ],
        "connections": [
            {
                **connection,
                "from_block": "CONTROLLER",
                "signal_type": "other",
                "description": "Speaker control signal",
            },
            power_connection,
        ],
    }

    def reply(payload):
        return {
            "text": json.dumps(payload),
            "reasoning": "",
            "finish_reason": "stop",
            "cost_usd": 0.0,
        }

    client = _ScriptedClient([reply(intent), reply(initial), reply(repaired)])
    client.s = Settings(api_key="test")
    state_path = tmp_path / ".kicraft" / "state.json"

    assert run_session(tmp_path, brief, ["intent"], client=client)["status"] == "ok"

    events: list[dict] = []
    result = stage_driver_mod.drive_stage(
        client,
        "functional_spec",
        brief,
        state_path,
        tmp_path,
        progress=events.append,
        review_before_commit=True,
    )

    assert result["repair_adopted"] is True
    assert result["attempts"] == 2
    assert [
        event["attempt"] for event in events if event["kind"] == "candidate_review"
    ] == [2]
    assert json.loads(state_path.read_text())["functional_spec"] is None


def test_intent_repair_is_scored_under_first_candidate_normalization(tmp_path):
    """A semantic repair is normalized exactly like the first candidate.

    ``complete_intent_classification`` adds the exact brief token ("ESP32-C3") to
    the first candidate's ``named_parts``. A repair that fixes an unrealizable
    obligation class but does not repeat that token used to be charged a spurious
    ``intent_named_part_omitted``; the defect scores tied, the adoption guard
    discarded the repair, and the committed slot kept the flagged class.

    The class is the off-board power source, which the deterministic spelling repair
    deliberately leaves alone (only the writer can choose the mate class or drop the
    obligation), so this still exercises the repair path.
    """
    brief = (
        "An ESP32-C3 module controller board powered from a 24 V DC screw terminal, "
        "with the input reverse-polarity protected."
    )
    flagged = {
        "goal": brief,
        "constraints": ["Power input: 24 V DC", "Input reverse-polarity protection required"],
        "named_parts": ["ESP32-C3 module"],
        "inferred_expertise": "intermediate",
        "assumptions": [],
        "obligations": [
            {
                "kind": "physical",
                "original_obligation_id": "coin-cell",
                "component_class": "coin-cell-battery",
            },
            {
                "kind": "physical",
                "original_obligation_id": "power-terminal",
                "component_class": "screw-terminal",
            },
        ],
        "project_stem": "ESP32C3_CONTROLLER",
    }
    # The repair drops the source obligation -- the remedy the diagnostic names -- and omits
    # `named_parts`, which normalization must re-add so the scores stay comparable.
    repaired = {
        **{key: value for key, value in flagged.items() if key != "named_parts"},
        "obligations": [flagged["obligations"][1]],
    }

    def reply(payload):
        return {
            "text": json.dumps(payload),
            "reasoning": "",
            "finish_reason": "stop",
            "cost_usd": 0.0,
        }

    client = _ScriptedClient([reply(flagged), reply(repaired)])
    client.s = Settings(api_key="test")

    result = run_session(tmp_path, brief, ["intent"], client=client)

    stage = result["results"][0]
    assert result["status"] == "ok"
    assert stage["repair_attempted"] is True
    assert stage["repair_adopted"] is True
    assert stage["repair_required"] is False
    assert stage["diagnostics"] == []
    classes = [
        row.get("component_class")
        for row in stage["slot"]["obligations"]
        if row.get("kind") == "physical"
    ]
    assert "coin-cell-battery" not in classes
    assert "screw-terminal" in classes


def test_rate_limit_falls_back_once_with_shared_guard_and_pristine_messages(tmp_path):
    events = []
    client = _ScriptedClient(
        [
            requests.exceptions.HTTPError("429 Too Many Requests"),
            _ok_intent_reply(),
        ]
    )
    client.s = Settings(
        api_key="test",
        design_profile="deepseek",
        model=str(DESIGN_PROFILES["deepseek"]["model"]),
        provider_order=list(DESIGN_PROFILES["deepseek"]["provider_order"]),
        provider_fallback_profile="luna",
        escalation_profile="",
    )
    result = run_session(
        tmp_path,
        "a USB-powered LED",
        ["intent"],
        client=client,
        progress=events.append,
    )
    assert result["status"] == "ok"
    assert result["results"][0]["attempts"] == 2
    assert [call["model"] for call in client.calls] == [
        DESIGN_PROFILES["deepseek"]["model"],
        DESIGN_PROFILES["luna"]["model"],
    ]
    assert all(call["guard"] is client.guard for call in client.calls)
    assert client.calls[1]["messages"] == client.calls[0]["messages"]
    assert [
        event["kind"] for event in events if event["kind"] in {"retry", "provider_fallback"}
    ] == [
        "retry",
        "provider_fallback",
    ]
    fallback = next(event for event in events if event["kind"] == "provider_fallback")
    assert fallback == {
        "kind": "provider_fallback",
        "stage": "intent",
        "from": DESIGN_PROFILES["deepseek"]["model"],
        "to": DESIGN_PROFILES["luna"]["model"],
        "from_profile": "deepseek",
        "to_profile": "luna",
        "from_providers": DESIGN_PROFILES["deepseek"]["provider_order"],
        "to_providers": DESIGN_PROFILES["luna"]["provider_order"],
        "attempt": 2,
        "reason": "provider_rate_limited",
    }


def test_provider_fallback_is_one_shot_and_only_for_rate_limits(tmp_path):
    rate_limited = _ScriptedClient(
        [
            requests.exceptions.HTTPError("429 Too Many Requests"),
            requests.exceptions.HTTPError("429 Too Many Requests"),
            _ok_intent_reply(),
        ]
    )
    rate_limited.s = Settings(
        api_key="test",
        design_profile="deepseek",
        model=str(DESIGN_PROFILES["deepseek"]["model"]),
        provider_fallback_profile="luna",
    )
    result = run_session(
        tmp_path / "rate-limited",
        "a USB-powered LED",
        ["intent"],
        client=rate_limited,
    )
    assert result["status"] == "failed"
    assert result["results"][0]["attempts"] == 2
    assert len(rate_limited.calls) == 2

    upstream = _ScriptedClient(
        [
            requests.exceptions.HTTPError("500 Upstream"),
            _ok_intent_reply(),
        ]
    )
    upstream.s = Settings(
        api_key="test",
        design_profile="deepseek",
        model=str(DESIGN_PROFILES["deepseek"]["model"]),
        provider_fallback_profile="luna",
    )
    result = run_session(
        tmp_path / "upstream",
        "a USB-powered LED",
        ["intent"],
        client=upstream,
    )
    assert result["status"] == "failed"
    assert len(upstream.calls) == 1


def test_empty_provider_fallback_profile_preserves_terminal_rate_limit(tmp_path):
    client = _ScriptedClient(
        [
            requests.exceptions.HTTPError("429 Too Many Requests"),
            _ok_intent_reply(),
        ]
    )
    client.s = Settings(
        api_key="test",
        design_profile="deepseek",
        model=str(DESIGN_PROFILES["deepseek"]["model"]),
        provider_fallback_profile="",
    )
    result = run_session(
        tmp_path,
        "a USB-powered LED",
        ["intent"],
        client=client,
    )
    assert result["status"] == "failed"
    assert len(client.calls) == 1
    assert result["failure_kind"] == "provider_rate_limited"
    assert result["retryable"] is True
    assert result["retry_action"] == "retry_stage"


def test_provider_fallback_budget_refusal_does_not_return_to_initial_route(tmp_path):
    from kicraft.server.spend_guard import BudgetExceeded

    class RefusingFallbackClient(_ScriptedClient):
        def chat(self, *args, **kwargs):
            if self.s.design_profile == "luna":
                self.calls.append(
                    {
                        "model": self.s.model,
                        "guard": self.guard,
                        "messages": list(args[0]),
                    }
                )
                raise BudgetExceeded("run budget exhausted")
            return super().chat(*args, **kwargs)

    client = RefusingFallbackClient([requests.exceptions.HTTPError("429 Too Many Requests")])
    client.s = Settings(
        api_key="test",
        design_profile="deepseek",
        model=str(DESIGN_PROFILES["deepseek"]["model"]),
        provider_fallback_profile="luna",
    )
    with pytest.raises(BudgetExceeded):
        run_session(
            tmp_path,
            "a USB-powered LED",
            ["intent"],
            client=client,
        )
    assert [call["model"] for call in client.calls] == [
        DESIGN_PROFILES["deepseek"]["model"],
        DESIGN_PROFILES["luna"]["model"],
    ]
    assert all(call["guard"] is client.guard for call in client.calls)


def test_truncated_json_triggers_one_plain_tool_free_serialization_call(tmp_path):
    # finish=length WITH content -> truncated_json -> exactly ONE serialization
    # call: tool-free (plain chat), reasoning disabled, FIXED cap (never the
    # old cap-doubling), then the parseable result commits.
    client = _ScriptedClient(
        [
            {
                "text": '{"goal": "x", truncated',
                "reasoning": "",
                "finish_reason": "length",
                "cost_usd": 0.0,
            },
            _ok_intent_reply(),
        ]
    )
    events = []
    res = run_session(
        tmp_path,
        "a USB-powered LED",
        ["intent"],
        client=client,
        progress=events.append,
    )
    assert res["status"] == "ok"
    assert len(client.calls) == 2
    first, serial = client.calls
    assert first["serialization"] is False
    assert serial["serialization"] is True
    assert serial["reasoning"] == {"enabled": False}
    assert serial["max_tokens"] == 8192  # fixed serialization cap for intent
    assert serial["max_tokens"] == 2 * first["max_tokens"]  # ... which is 2x the 4096 normal
    # the cap is the policy's fixed value, never doubled AGAIN: a truncated
    # serialization result would go terminal, not raise to 16384.
    recovery = next(event for event in events if event["kind"] == "serialization_recovery")
    assert recovery == {
        "kind": "serialization_recovery",
        "stage": "intent",
        "failure_kind": "truncated_json",
        "resolution_ledger_entries": 0,
    }
    decoded = next(event for event in events if event["kind"] == "candidate_decoded")
    assert decoded["attempt"] == 2
    assert decoded["serialization_recovery"] is True
    assert decoded["clean_slate"] is False
    assert decoded["expanded_component_count"] == 0
    assert decoded["unknown_sheet_references"] == []


def test_repeated_truncated_replies_use_the_full_bounded_budget(tmp_path):
    client = _ScriptedClient(
        [
            {
                "text": '{"goal": "x", truncated',
                "reasoning": "",
                "finish_reason": "length",
                "cost_usd": 0.0,
            },
            {
                "text": '{"goal": "y", still',
                "reasoning": "",
                "finish_reason": "length",
                "cost_usd": 0.0,
            },
            {
                "text": '{"goal": "z", still',
                "reasoning": "",
                "finish_reason": "length",
                "cost_usd": 0.0,
            },
        ]
    )
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "failed"
    last = res["results"][-1]
    assert len(client.calls) == 3
    assert sum(1 for call in client.calls if call["serialization"]) == 1
    assert last["failure_kind"] == "truncated_json"
    assert last["attempts"] == 3


def test_repeated_invalid_json_uses_the_full_bounded_budget(tmp_path):
    client = _ScriptedClient(
        [
            {"text": "not json at all", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
            {"text": "still not json", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
            {"text": "still not json", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
        ]
    )
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "failed"
    last = res["results"][-1]
    assert len(client.calls) == 3
    assert last["failure_kind"] == "invalid_json"
    assert last["attempts"] == 3


def test_serialization_schema_failure_gets_final_normal_correction(tmp_path):
    client = _ScriptedClient(
        [
            {
                "text": '{"goal": "x", truncated',
                "reasoning": "",
                "finish_reason": "length",
                "cost_usd": 0.0,
            },
            {"text": "{}", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
            _ok_intent_reply(),
        ]
    )
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "ok"
    assert len(client.calls) == 3
    assert sum(1 for call in client.calls if call["serialization"]) == 1
    assert client.calls[2]["serialization"] is False


def test_serialization_goes_through_chat_even_for_bom(tmp_path, monkeypatch):
    # Serialization recovery must route through plain client.chat() for the BOM
    # stage too — never chat_with_tools, so no tool rounds and no transcript
    # resend. The normal attempt is the tool loop (chat_with_tools).
    client = _ScriptedClient(
        [
            {
                "text": '{"parts": [trunc',
                "reasoning": "",
                "finish_reason": "length",
                "cost_usd": 0.0,
            },
            {"text": "also bad", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
            {"text": "still bad", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
            {"text": "bad again", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
            {"text": "final bad", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
            {"text": "bad once more", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
            {"text": "last bad", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
        ]
    )
    prep = {
        "state": {"architecture": {"sheets": [{"name": "POWER"}]}},
        "extras": {},
    }
    monkeypatch.setattr(
        stage_driver_mod,
        "prepare_stage",
        lambda *args, **kwargs: type(
            "Proc", (), {"returncode": 0, "stdout": json.dumps(prep), "stderr": ""}
        )(),
    )
    res = run_session(tmp_path, "a USB-powered LED", ["bom"], client=client)
    assert res["status"] == "failed"
    assert len(client.calls) == 3
    assert client.calls[0]["serialization"] is False  # chat_with_tools (tool loop)
    assert client.calls[1]["serialization"] is True  # plain chat for serialization
    assert client.calls[2]["serialization"] is False  # one bounded clean-slate call
    assert client.calls[2]["reasoning"] == {"enabled": False}
    assert client.calls[2]["max_tokens"] == 2048
    assert client.calls[1]["max_tokens"] == 2048  # affordable BOM-unit serialization cap
    assert client.calls[1]["reasoning"] == {"enabled": False}
    assert res["results"][-1]["failure_kind"] == "invalid_json"


def test_bom_ownership_conflict_stops_after_one_repair_attempt(
    tmp_path,
    monkeypatch,
):
    response = {
        "groups": [
            {
                "id": "unowned_esp32",
                "reference_prefix": "U",
                "quantity": 1,
                "value": "ESP32-S3-MINI-1-N8",
                "symbol": "esp32-s3-mini-1:ESP32-S3-MINI-1-N8",
                "footprint": "esp32-s3-mini-1:BULETM-SMD_ESP32-S3-MINI-1-N8",
                "mpn": "ESP32-S3-MINI-1-N8",
                "sheet": "POWER",
            }
        ],
        "arrays": [],
        "assumptions": [],
        "substitutions": [],
    }
    reply = {
        "text": json.dumps(response),
        "reasoning": "",
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }
    # One repair attempt is granted (with drop-the-duplicate guidance); an
    # identical repeat is terminal so the model can never override determinism.
    client = _ScriptedClient([reply, dict(reply)])
    prep = {
        "state": {
            "architecture": {
                "sheets": [{"name": "POWER"}],
                "recipe_selections": [],
            }
        },
        "extras": {},
    }
    monkeypatch.setattr(
        stage_driver_mod,
        "prepare_stage",
        lambda *args, **kwargs: type(
            "Proc",
            (),
            {"returncode": 0, "stdout": json.dumps(prep), "stderr": ""},
        )(),
    )

    res = run_session(tmp_path, "a power board", ["bom"], client=client)

    assert res["status"] == "failed"
    assert res["results"][-1]["failure_kind"] == "unit_ownership_conflict"
    assert len(client.calls) == 2


def test_typed_bom_lowerer_skips_provider_call(tmp_path, monkeypatch):
    state = {
        "architecture": {
            "topologies": {"INPUT": "9-pin header"},
            "rail_voltages": {},
            "sheets": [
                {
                    "name": "INPUT",
                    "stem": "INPUT",
                    "function": "Eight digital inputs",
                }
            ],
            "power_nets": ["GND"],
            "inter_sheet_nets": [],
            "recipe_selections": [],
            "requirements": [
                {
                    "id": "input_header",
                    "sheet": "INPUT",
                    "role": "connector",
                    "family": "pin-header",
                    "parameters": {"rows": 1},
                    "ports": {
                        **{f"pin{index + 1}": f"D{index}" for index in range(8)},
                        "pin9": "GND",
                    },
                }
            ],
        }
    }
    prep = {"state": state, "extras": {}}
    monkeypatch.setattr(
        stage_driver_mod,
        "prepare_stage",
        lambda *args, **kwargs: type(
            "Proc", (), {"returncode": 0, "stdout": json.dumps(prep), "stderr": ""}
        )(),
    )
    commits = []
    monkeypatch.setattr(
        stage_driver_mod,
        "commit_stage",
        lambda stage, slot, *args, **kwargs: (commits.append(slot) or True, {"ok": True}),
    )
    progress = []
    client = _unit_client([])

    result = stage_driver_mod.drive_stage(
        client,
        "bom",
        "eight digital inputs",
        tmp_path / "state.json",
        tmp_path,
        progress=progress.append,
    )

    assert result["commit_ok"] is True, result
    assert result["attempts"] == 0
    assert client.calls == []
    state["bom"] = commits[0]
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    prep["extras"] = {
        "symbol_pinouts": {
            part["ref"]: lookup_pins(part["symbol"], all_units=True)
            for part in state["bom"]["parts"]
        }
    }
    wiring = stage_driver_mod.drive_stage(
        client,
        "wiring",
        "eight digital inputs",
        tmp_path / "state.json",
        tmp_path,
    )
    assert wiring["commit_ok"] is True, wiring
    assert client.calls == []
    assert {connection["net_name"] for connection in commits[-1]["connections"]} == {
        *(f"D{index}" for index in range(8)),
        "GND",
    }


def test_recipe_complete_mcu_bom_and_wiring_skip_provider_calls(tmp_path, monkeypatch):
    fixture = Path(__file__).parent / "fixtures" / "recipe_coverage" / "mcu_only_architecture.json"
    state = {
        "intent": {"named_parts": ["ESP32-S3-MINI-1-N8"]},
        "architecture": json.loads(fixture.read_text(encoding="utf-8")),
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    def prepare(*args, **kwargs):
        parts = (state.get("bom") or {}).get("parts") or []
        extras = {
            "symbol_pinouts": {
                part["ref"]: {"symbol": part["symbol"], "pins": []} for part in parts
            }
        }
        return type(
            "Proc",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps({"state": state, "extras": extras}),
                "stderr": "",
            },
        )()

    committed = {}

    def commit(stage, slot, *args, **kwargs):
        committed[stage] = slot
        state[stage] = slot
        return True, {"ok": True}

    monkeypatch.setattr(stage_driver_mod, "prepare_stage", prepare)
    monkeypatch.setattr(stage_driver_mod, "commit_stage", commit)
    progress = []
    client = _unit_client([])

    bom_result = stage_driver_mod.drive_stage(
        client,
        "bom",
        "ESP32-S3-MINI-1-N8 native USB controller",
        state_path,
        tmp_path,
        progress=progress.append,
    )
    wiring_result = stage_driver_mod.drive_stage(
        client,
        "wiring",
        "ESP32-S3-MINI-1-N8 native USB controller",
        state_path,
        tmp_path,
        progress=progress.append,
    )

    assert bom_result["commit_ok"] is wiring_result["commit_ok"] is True
    assert bom_result["work_units"] == wiring_result["work_units"] == 0
    assert bom_result["attempts"] == wiring_result["attempts"] == 0
    assert client.calls == []
    assert {
        part["recipe_id"] for part in committed["bom"]["parts"]
    } == {"esp32-s3-mini-1-minimal@1", "usb-c-usb2-device@1"}
    assert committed["bom"]["recipe_ownership"][0]["pins"]
    assert committed["wiring"]["connections"]
    assert committed["wiring"]["no_connect_pins"]
    recipe_events = [event for event in progress if event.get("kind") == "recipe_selected"]
    assert [event["stage"] for event in recipe_events] == ["bom", "bom", "wiring", "wiring"]
    assert all(event["owned_call_count"] == 0 for event in recipe_events)
    assert all(
        event["resolution"][0]["requirement_id"]
        in {"auto_esp32_s3_module", "usb_connector"}
        for event in recipe_events
    )


def test_mixed_recipe_sheet_calls_provider_once_for_only_novel_role(tmp_path, monkeypatch):
    fixture = Path(__file__).parent / "fixtures" / "recipe_coverage" / "mixed_architecture.json"
    state = {
        "intent": {"named_parts": ["ESP32-S3-MINI-1-N8"]},
        "architecture": json.loads(fixture.read_text(encoding="utf-8")),
    }
    prep = {"state": state, "extras": {}}
    monkeypatch.setattr(
        stage_driver_mod,
        "prepare_stage",
        lambda *args, **kwargs: type(
            "Proc", (), {"returncode": 0, "stdout": json.dumps(prep), "stderr": ""}
        )(),
    )
    committed = []
    monkeypatch.setattr(
        stage_driver_mod,
        "commit_stage",
        lambda stage, slot, *args, **kwargs: (
            committed.append(slot) or True,
            {"ok": True},
        ),
    )
    reply = {
        "text": json.dumps(
            {
                "groups": [
                    {
                        "id": "threshold_resistor",
                        "reference_prefix": "R",
                        "quantity": 1,
                        "value": "100k",
                        "symbol": "Device:R",
                        "footprint": "Resistor_SMD:R_0603_1608Metric",
                        "sheet": "MCU",
                    }
                ],
                "arrays": [],
                "assumptions": [],
                "substitutions": [],
            }
        ),
        "reasoning": "",
        "finish_reason": "stop",
        "cost_usd": 0.001,
    }
    client = _unit_client([reply])
    result = stage_driver_mod.drive_stage(
        client,
        "bom",
        "ESP32-S3-MINI-1-N8 with a novel analog threshold detector",
        tmp_path / "state.json",
        tmp_path,
    )

    assert result["commit_ok"] is True
    assert result["work_units"] == result["attempts"] == len(client.calls) == 1
    system_prompt = client.calls[0]["messages"][0]["content"]
    assert '"owned_roles":["analog_block"]' in system_prompt
    assert '"recipe_ids":["esp32-s3-mini-1-minimal@1"]' in system_prompt
    assert any(part.get("recipe_id") for part in committed[0]["parts"])
    assert [
        part["resolution_source"] for part in committed[0]["parts"] if not part.get("recipe_id")
    ] == ["llm"]


def test_invalid_bom_architecture_fails_before_provider_call(tmp_path, monkeypatch):
    prep = {"state": {"architecture": {"sheets": []}}, "extras": {}}
    monkeypatch.setattr(
        stage_driver_mod,
        "prepare_stage",
        lambda *args, **kwargs: type(
            "Proc", (), {"returncode": 0, "stdout": json.dumps(prep), "stderr": ""}
        )(),
    )
    client = _ScriptedClient([])
    result = stage_driver_mod.drive_stage(
        client,
        "bom",
        "test",
        tmp_path / "state.json",
        tmp_path,
    )
    assert client.calls == []
    assert result["attempts"] == 0
    assert result["cost_usd"] == 0.0
    assert result["error"].startswith("stage contract failed:")


# ---- a refused deterministic lowering is re-driven, not fatal (§4 item 1) ----


def _refused_lowering_architecture():
    """One unit whose demand its own deterministic lowering cannot satisfy.

    ``input_condition`` (family ``rc-lowpass``) demands a reviewed
    ``coupling-capacitor`` and ``input-protection``, but the lowerer emits a bare
    R/C pair with no orderable identity, so ``_validate_bom_unit`` refuses the
    lowering with ``physical-obligation-unfulfilled``.
    """
    return {
        "architecture": {
            "topologies": {},
            "rail_voltages": {},
            "sheets": [
                {
                    "name": "INPUT",
                    "stem": "INPUT",
                    "function": "Line-level input conditioning",
                }
            ],
            "power_nets": ["GND"],
            "inter_sheet_nets": [],
            "recipe_selections": [],
            "requirements": [
                {
                    "id": "input_condition",
                    "sheet": "INPUT",
                    "role": "analog_block",
                    "family": "rc-lowpass",
                    "parameters": {
                        "cutoff_hz": 20000.0,
                        "resistance_ohm": 1000.0,
                        "max_error_percent": 5.0,
                    },
                    "obligations": [
                        {
                            "kind": "physical",
                            "original_obligation_id": "ac_coupling",
                            "component_class": "coupling-capacitor",
                        },
                        {
                            "kind": "physical",
                            "original_obligation_id": "input_protection",
                            "component_class": "input-protection",
                        },
                    ],
                    "ports": {"gnd": "GND", "input": "AUDIO_IN", "output": "CONDITIONED"},
                }
            ],
        }
    }


def _generic_unit_payload():
    """What the lowering (and so a model that just repeats it) emits: a bare R/C pair."""
    return {
        "groups": [
            {
                "id": "resistor",
                "reference_prefix": "R",
                "quantity": 1,
                "value": "1k",
                "symbol": "Device:R",
                "footprint": "Resistor_SMD:R_0603_1608Metric",
                "sheet": "INPUT",
            },
            {
                "id": "capacitor",
                "reference_prefix": "C",
                "quantity": 1,
                "value": "8.2nF",
                "symbol": "Device:C",
                "footprint": "Capacitor_SMD:C_0603_1608Metric",
                "sheet": "INPUT",
            },
        ],
        "arrays": [],
        "assumptions": [],
        "substitutions": [],
    }


def _satisfying_unit_payload():
    """The same unit with each demanded class carried by an orderable real part."""
    return {
        "groups": [
            {
                "id": "ac_coupling",
                "reference_prefix": "C",
                "quantity": 1,
                "value": "1uF",
                "symbol": "Device:C",
                "footprint": "Capacitor_SMD:C_0603_1608Metric",
                "mpn": "CL10B105KB8NNNC",
                "sheet": "INPUT",
            },
            {
                "id": "input_protection",
                "reference_prefix": "D",
                "quantity": 2,
                "value": "BAV99",
                "symbol": "Diode:BAV99",
                "footprint": "Package_TO_SOT_SMD:SOT-363_SC-70-6",
                "mpn": "BAV99,215",
                "sheet": "INPUT",
            },
            {
                "id": "filter_series",
                "reference_prefix": "R",
                "quantity": 1,
                "value": "1k",
                "symbol": "Device:R",
                "footprint": "Resistor_SMD:R_0603_1608Metric",
                "mpn": "RC0603FR-071KL",
                "sheet": "INPUT",
            },
        ],
        "arrays": [],
        "assumptions": [],
        "substitutions": [],
    }


def _bom_unit_reply(payload):
    return {
        "text": json.dumps(payload),
        "reasoning": "",
        "finish_reason": "stop",
        "cost_usd": 0.001,
    }


def _bom_prep(monkeypatch, state):
    prep = {"state": state, "extras": {}}
    monkeypatch.setattr(
        stage_driver_mod,
        "prepare_stage",
        lambda *args, **kwargs: type(
            "Proc", (), {"returncode": 0, "stdout": json.dumps(prep), "stderr": ""}
        )(),
    )


def test_refused_deterministic_lowering_is_redriven_with_the_refusal_as_feedback(
    tmp_path, monkeypatch
):
    _bom_prep(monkeypatch, _refused_lowering_architecture())
    commits = []
    monkeypatch.setattr(
        stage_driver_mod,
        "commit_stage",
        lambda stage, slot, *args, **kwargs: (commits.append(slot) or True, {"ok": True}),
    )
    client = _unit_client([_bom_unit_reply(_satisfying_unit_payload())])

    result = stage_driver_mod.drive_stage(
        client, "bom", "a line-level input stage", tmp_path / "state.json", tmp_path
    )

    assert result["commit_ok"] is True, result
    assert len(client.calls) == 1
    prompt = client.calls[0]["messages"][1]["content"]
    assert "AGGREGATE OR LOCAL DEFECTS TO REPAIR" in prompt
    assert "requires 1 real coupling-capacitor" in prompt
    assert "requires 1 real input-protection" in prompt
    assert "the unit emitted: resistor=Device:R mpn=1k | capacitor=Device:C mpn=8.2nF" in prompt
    # The committed BOM carries the model's real parts, not the refused lowering.
    assert {part["mpn"] for part in commits[0]["parts"]} == {
        "CL10B105KB8NNNC",
        "BAV99,215",
        "RC0603FR-071KL",
    }


def _wrong_sheet_unit_payload():
    """The satisfying answer emitted on the wrong sheet: defective for the model's own reason."""
    payload = _satisfying_unit_payload()
    payload["groups"][0]["sheet"] = "POWER"
    return payload


def test_refused_deterministic_lowering_ends_in_unit_repair_exhausted(tmp_path, monkeypatch):
    _bom_prep(monkeypatch, _refused_lowering_architecture())
    client = _unit_client([_bom_unit_reply(_generic_unit_payload())] * 8)

    result = stage_driver_mod.drive_stage(
        client, "bom", "a line-level input stage", tmp_path / "state.json", tmp_path
    )

    assert result["failure_kind"] == "unit_repair_exhausted"
    assert result["attempts"] == len(client.calls) == 3
    assert not result["error"].startswith("stage contract failed:")


def test_refused_deterministic_lowering_reports_the_models_own_defect(tmp_path, monkeypatch):
    """The repair loop must learn what *it* got wrong, not what the lowering got wrong.

    A unit whose lowering is already refused must not fall back to that lowering:
    the adoption can only re-raise the lowering's defect, which hides the model's
    and makes every repair round repeat the same text (the bounded identical
    signature then ends the unit). Measured on the frozen line receiver: the
    power-LED unit failed three times with the lowering's leftover, never with
    the model's own defect.
    """
    _bom_prep(monkeypatch, _refused_lowering_architecture())
    client = _unit_client([_bom_unit_reply(_wrong_sheet_unit_payload())] * 4)

    result = stage_driver_mod.drive_stage(
        client, "bom", "a line-level input stage", tmp_path / "state.json", tmp_path
    )

    assert result["failure_kind"] == "unit_repair_exhausted"
    assert "wrong-sheet" in result["error"]
    assert "physical-obligation-unfulfilled" not in result["error"]
    assert "wrong-sheet" in client.calls[1]["messages"][1]["content"]


def test_empty_length_takes_reasoning_recovery_not_invalid_json(tmp_path):
    # finish=length with NO content is provider exhaustion (even without the
    # client loop detector firing): reasoning is disabled for the retry, and
    # the failure is NEVER mislabeled invalid_json.
    client = _ScriptedClient(
        [
            {"text": "", "reasoning": "x" * 600, "finish_reason": "length", "cost_usd": 0.0},
            _ok_intent_reply(),
        ]
    )
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "ok"
    assert client.calls[0]["serialization"] is False
    assert client.calls[1]["reasoning"] == {"enabled": False}  # reasoning-disabled retry
    assert sum(1 for c in client.calls if c["serialization"]) == 0  # no serialization call


def test_empty_length_twice_fails_as_reasoning_loop(tmp_path):
    client = _ScriptedClient(
        [
            {"text": "", "reasoning": "x" * 600, "finish_reason": "length", "cost_usd": 0.0},
            {"text": "", "reasoning": "y" * 600, "finish_reason": "length", "cost_usd": 0.0},
        ]
    )
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "failed"
    last = res["results"][-1]
    assert last["failure_kind"] == "reasoning_loop"  # never invalid_json
    assert last["error"] == "reasoning_loop"
    assert len(client.calls) == 2


def test_failure_kind_reaches_stage_status(tmp_path):
    client = _ScriptedClient(
        [
            {"text": "not json", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
            {"text": "not json either", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
            {"text": "still not json", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
        ]
    )
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    last = res["results"][-1]
    sp = tmp_path / ".kicraft" / "state.json"
    sj = json.loads(sp.read_text(encoding="utf-8"))
    entry = sj["stage_status"]["intent"]
    assert entry["failure_kind"] == "invalid_json"
    assert entry["attempts"] == last["attempts"] == 3


# ---- provider/transport failures never enter JSON recovery -----------------


def test_collection_limit_uses_one_escape_serialization_call(tmp_path):
    client = _ScriptedClient(
        [
            {
                "text": '{"goal":',
                "finish_reason": "collection_limit",
                "collection_limit": {
                    "field": "parts",
                    "observed_count": 501,
                    "configured_total": 500,
                    "emitted_content_chars": 82000,
                },
                "cost_usd": 0.01,
            },
            _ok_intent_reply(),
        ]
    )
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "ok"
    assert len(client.calls) == 2
    assert client.calls[1]["temperature"] == 0.4
    retry = client.calls[1]["messages"][-1]["content"]
    assert "item 501" in retry and "configured total limit is 500" in retry
    assert "82000 content characters" in retry


def test_repeated_collection_limit_uses_the_full_bounded_budget(tmp_path):
    overflow = {
        "text": '{"goal":',
        "finish_reason": "collection_limit",
        "collection_limit": {
            "field": "parts",
            "observed_count": 501,
            "configured_total": 500,
            "emitted_content_chars": 82000,
        },
        "cost_usd": 0.01,
    }
    client = _ScriptedClient([overflow, overflow, overflow])
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    last = res["results"][-1]
    assert last["failure_kind"] == "collection_limit"
    assert last["attempts"] == 3


@pytest.mark.parametrize("recovers", [False, True])
def test_syntax_stopped_stream_uses_truthful_feedback_and_existing_retry_ceiling(
    tmp_path, recovers
):
    syntax_error = "Expected ',' or ']' after a value"
    stopped = {
        # Even a parseable saved prefix is not a candidate after a syntax abort.
        "text": _OK_INTENT,
        "finish_reason": "collection_limit",
        "collection_limit": {
            "limit_scope": "syntax",
            "syntax_error": syntax_error,
            "field": "$.inter_sheet_nets",
            "character_offset": 3034,
            "line": 51,
            "column": 129,
            "emitted_content_chars": 3034,
        },
        "cost_usd": 0.01,
    }
    client = _ScriptedClient([stopped, _ok_intent_reply()] if recovers else [stopped] * 3)
    result = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert result["status"] == ("ok" if recovers else "failed")
    assert len(client.calls) == (2 if recovers else 3)
    feedback = client.calls[1]["messages"][-1]["content"]
    assert syntax_error in feedback
    assert "3034" in feedback and "$.inter_sheet_nets" in feedback
    assert "configured" not in feedback and "item unknown" not in feedback
    if not recovers:
        assert result["results"][-1]["failure_kind"] == "collection_limit"
        assert result["results"][-1]["attempts"] == 3


def test_commit_rejection_signature_normalizes_gate_ids_and_offenders():
    first = {
        "errors": ["§9.15  multi-net pin short", "9.17 dangling net"],
        "offenders": [" U1.2   on A/B ", "R1.1"],
    }
    second = {
        "errors": ["9.15 changed explanation", "9.17 another explanation"],
        "offenders": ["R1.1 remains shorted", "pin U1.2 still appears on two nets"],
    }
    assert _commit_rejection_signature(first) == _commit_rejection_signature(second)


def test_commit_rejection_facts_keep_gates_and_hash_part_identifiers():
    facts = _redacted_rejection_facts(
        {
            "ok": False,
            "errors": ["9.14 inter-sheet net missing", "9.29 programming path absent"],
            "offenders": ["U7.12 on CUSTOMER_NET", "J4.3"],
        },
        candidate_retained=True,
    )
    assert facts["commit_gate_codes"] == ["9.14", "9.29"]
    assert facts["offender_count"] == 2
    assert len(facts["rejection_signature"]) == 64
    assert "U7" not in facts["rejection_signature"]


def test_repeated_commit_rejection_stops_without_pristine_escape(tmp_path, monkeypatch):
    rejected = {
        "ok": False,
        "errors": ["9.15 multi-net pin short"],
        "offenders": ["R1.1 on SIG_A and SIG_B"],
    }
    monkeypatch.setattr(stage_driver_mod, "commit_stage", lambda *args, **kwargs: (False, rejected))
    client = _ScriptedClient([_ok_intent_reply(), _ok_intent_reply(), _ok_intent_reply()])
    client.s = Settings(api_key="test")
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    last = res["results"][-1]
    assert last["failure_kind"] == "commit_rejected"
    assert last["attempts"] == 2
    assert len(client.calls) == 2
    assert {call["model"] for call in client.calls} == {client.s.model}


def test_a1_enriched_915_offender_keeps_legacy_commit_signature(monkeypatch):
    """Cross-module contract (validation.py owns the formatting rule): the
    REAL enriched §9.15 offender must signature-match the lead-only offender,
    so clean-slate arming/terminating logic is unaffected by A1."""
    import kicraft.design.synthesis.symbol_pinout as sp
    from kicraft.design.models import (
        BOM,
        Architecture,
        BomPart,
        InterSheetNet,
        NetConnection,
        PinEndpoint,
        Sheet,
        SheetPin,
    )
    from kicraft.design.synthesis.validation import check_no_dangling_signal_nets

    pinmap = {
        "A:R": [("1", "1", "passive"), ("2", "2", "passive")],
        "A:IC": [("5", "OUT", "output")],
    }

    def _lookup(lib_id, *a, **k):
        if lib_id not in pinmap:
            raise sp.SymbolNotFoundError(lib_id)
        return {
            "symbol": lib_id,
            "unit_count": 1,
            "pins": [{"number": n, "name": m, "electrical_type": t} for n, m, t in pinmap[lib_id]],
        }

    monkeypatch.setattr(sp, "lookup_pins", _lookup)
    arch = Architecture(
        sheets=[
            Sheet(name="MCU", stem="MCU", function="mcu"),
            Sheet(name="IO", stem="IO", function="io"),
        ],
        power_nets=[],
        inter_sheet_nets=[
            InterSheetNet(
                name="X_NET",
                endpoints=[
                    SheetPin(sheet="MCU", direction="output"),
                    SheetPin(sheet="IO", direction="input"),
                ],
            )
        ],
    )
    bom = BOM(
        parts=[
            BomPart(ref="R3", value="1k", symbol="A:R", footprint="F:F", sheet="MCU"),
            BomPart(ref="U1", value="x", symbol="A:IC", footprint="F:F", sheet="MCU"),
        ],
        connections=[
            NetConnection(
                net_name="SIG_IN",
                sheet="MCU",
                endpoints=[
                    PinEndpoint(ref="U1", pin="5"),
                    PinEndpoint(ref="R3", pin="1"),
                ],
            ),
            NetConnection(
                net_name="SIG_OUT", sheet="MCU", endpoints=[PinEndpoint(ref="R3", pin="2")]
            ),
        ],
    )
    offenders = check_no_dangling_signal_nets(arch, bom).offenders
    assert len(offenders) == 1
    enriched = offenders[0]
    assert " -- " in enriched  # the topology context really fired
    lead = enriched.split(" -- ")[0]
    errors = ["9.15 no dangling signal nets: 1 signal net(s) wire a single pin"]
    legacy = _commit_rejection_signature({"errors": errors, "offenders": [lead]})
    modern = _commit_rejection_signature({"errors": errors, "offenders": [enriched]})
    assert legacy == modern


# ---------- KC-VKUT5H A3: one-clean-slate bounded-continuation state machine ----------


def _a3_sig(letter: str) -> dict:
    # Offender identity IS the canonical pin, so a distinct signature must
    # move a distinct pin (A1.2, B1.2, ...) — renaming only the net is not
    # enough for _offender_identity.
    return {
        "ok": False,
        "errors": ["9.15 no dangling signal nets: 1 signal net(s) wire a single pin"],
        "offenders": [f"net 'SIG_{letter}' on sheet 'MCU' wires only {letter}1.2 and is dangling"],
    }


def _a3_wiring_state(tmp_path, monkeypatch):
    state = {
        "architecture": {
            "sheets": [{"name": "MAIN"}],
            "power_nets": [],
            "inter_sheet_nets": [],
        },
        "bom": {
            "parts": [
                {"ref": "U1", "sheet": "MAIN", "symbol": "Test:U", "value": "IC"},
                {"ref": "R1", "sheet": "MAIN", "symbol": "Test:R", "value": "1k"},
            ],
            "connections": [],
            "no_connect_pins": [],
        },
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    prep = {
        "state": state,
        "extras": {
            "symbol_pinouts": {
                "Test:U": {"pins": [{"number": "1"}, {"number": "2"}]},
                "Test:R": {"pins": [{"number": "1"}, {"number": "2"}]},
            }
        },
    }
    monkeypatch.setattr(
        stage_driver_mod,
        "prepare_stage",
        lambda *args, **kwargs: type(
            "Proc", (), {"returncode": 0, "stdout": json.dumps(prep), "stderr": ""}
        )(),
    )
    return state_path


def _a3_run(
    tmp_path,
    monkeypatch,
    rejects,
    max_retries=7,
    extra_ok_reply=True,
    *,
    design_profile="deepseek",
    escalation_profile="luna",
    progress=None,
    attempt_observer=None,
    vary_replies=False,
):
    """Drive wiring; commit rejects with _a3_sig(letter) per entry, then OK."""
    state_path = _a3_wiring_state(tmp_path, monkeypatch)
    n = {"i": 0}

    def fake_commit(stage, slot, *args, **kwargs):
        i = n["i"]
        n["i"] += 1
        if i < len(rejects):
            return False, _a3_sig(rejects[i])
        return True, {"ok": True}

    monkeypatch.setattr(stage_driver_mod, "commit_stage", fake_commit)
    reply = {
        "text": json.dumps(
            {
                "pins": [
                    {"ref": "U1", "pin": "1", "net": "A"},
                    {"ref": "R1", "pin": "1", "net": "A"},
                    {"ref": "R1", "pin": "2", "net": "B"},
                    {"ref": "U1", "pin": "2", "net": "B"},
                ]
            }
        ),
        "reasoning": "",
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }
    replies = [dict(reply) for _ in rejects] + ([dict(reply)] if extra_ok_reply else [])
    if vary_replies:
        for index, item in enumerate(replies):
            payload = json.loads(item["text"])
            payload["pins"][0]["net"] = f"A{index}"
            payload["pins"][1]["net"] = f"A{index}"
            item["text"] = json.dumps(payload)
    client = _ScriptedClient(replies)
    client.s = Settings(
        api_key="test",
        design_profile=design_profile,
        model=str(DESIGN_PROFILES[design_profile]["model"]),
        escalation_profile=escalation_profile,
    )
    result = stage_driver_mod.drive_stage(
        client,
        "wiring",
        "test",
        state_path,
        tmp_path,
        max_retries=max_retries,
        progress=progress,
        attempt_observer=attempt_observer,
    )
    return result, client


def test_work_unit_repeated_signature_stops_after_one_aggregate_repair(
    tmp_path,
    monkeypatch,
):
    result, client = _a3_run(
        tmp_path,
        monkeypatch,
        ["A", "A"],
        max_retries=99,
        extra_ok_reply=False,
    )
    assert result["commit_ok"] is False
    assert result["failure_kind"] == "commit_rejected"
    assert result["aggregate_repair_rounds"] == 1
    assert result["attempts"] == 2
    assert len(client.calls) == 2


def test_work_unit_same_gate_continues_when_candidate_changes(
    tmp_path,
    monkeypatch,
):
    result, client = _a3_run(
        tmp_path,
        monkeypatch,
        ["A", "A"],
        max_retries=99,
        extra_ok_reply=True,
        vary_replies=True,
    )
    assert result["commit_ok"] is True
    assert result["aggregate_repair_rounds"] == 2
    assert result["attempts"] == 3
    assert len(client.calls) == 3


def test_work_unit_changed_signatures_get_third_aggregate_repair(
    tmp_path,
    monkeypatch,
):
    result, client = _a3_run(
        tmp_path,
        monkeypatch,
        ["A", "B", "C"],
        max_retries=99,
        extra_ok_reply=True,
    )
    assert result["commit_ok"] is True
    assert result["aggregate_repair_rounds"] == 3
    assert result["attempts"] == 4
    assert len(client.calls) == 4


def test_work_unit_commit_process_failure_has_no_followup_call(tmp_path, monkeypatch):
    state_path = _a3_wiring_state(tmp_path, monkeypatch)
    monkeypatch.setattr(
        stage_driver_mod,
        "commit_stage",
        lambda *args, **kwargs: (
            False,
            {
                "ok": False,
                "failure_kind": "commit_process_failed",
                "errors": ["stage-commit exited 7"],
            },
        ),
    )
    reply = {
        "text": json.dumps(
            {
                "pins": [
                    {"ref": "U1", "pin": "1", "net": "A"},
                    {"ref": "R1", "pin": "1", "net": "A"},
                    {"ref": "R1", "pin": "2", "net": "B"},
                    {"ref": "U1", "pin": "2", "net": "B"},
                ]
            }
        ),
        "reasoning": "",
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }
    client = _ScriptedClient([reply])
    client.s = Settings(api_key="test")
    result = stage_driver_mod.drive_stage(
        client,
        "wiring",
        "test",
        state_path,
        tmp_path,
        max_retries=99,
    )
    assert result["failure_kind"] == "commit_process_failed"
    assert result["attempts"] == 1
    assert len(client.calls) == 1


def test_wiring_rejection_uses_complete_same_schema_correction(tmp_path, monkeypatch):
    state = {
        "architecture": {
            "sheets": [{"name": "MAIN"}],
            "power_nets": [],
            "inter_sheet_nets": [],
        },
        "bom": {
            "parts": [
                {"ref": "U1", "sheet": "MAIN", "symbol": "Test:U", "value": "IC"},
                {"ref": "R1", "sheet": "MAIN", "symbol": "Test:R", "value": "1k"},
            ],
            "connections": [],
            "no_connect_pins": [],
        },
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    prep = {
        "state": state,
        "extras": {
            "symbol_pinouts": {
                "Test:U": {"pins": [{"number": "1"}, {"number": "2"}]},
                "Test:R": {"pins": [{"number": "1"}]},
            }
        },
    }
    monkeypatch.setattr(
        stage_driver_mod,
        "prepare_stage",
        lambda *args, **kwargs: type(
            "Proc", (), {"returncode": 0, "stdout": json.dumps(prep), "stderr": ""}
        )(),
    )
    commits = []
    rejected = {
        "ok": False,
        "errors": ["9.15 multi-net pin short on net 'A'"],
        "offenders": ["U1.1"],
    }

    def fake_commit(stage, slot, *args, **kwargs):
        commits.append(json.loads(json.dumps(slot)))
        if len(commits) == 1:
            return False, rejected
        by_net = {connection["net_name"]: connection for connection in commits[1]["connections"]}
        assert by_net["A"]["endpoints"] == [{"ref": "R1", "pin": "1"}]
        assert by_net["B"]["endpoints"] == [
            {"ref": "U1", "pin": "1"},
            {"ref": "U1", "pin": "2"},
        ]
        return True, {"ok": True}

    monkeypatch.setattr(stage_driver_mod, "commit_stage", fake_commit)
    first = {
        "pins": [
            {"ref": "U1", "pin": "1", "net": "A"},
            {"ref": "R1", "pin": "1", "net": "A"},
            {"ref": "U1", "pin": "2", "net": "B"},
        ]
    }
    corrected = {
        "pins": [
            {"ref": "U1", "pin": "1", "net": "B"},
            {"ref": "R1", "pin": "1", "net": "A"},
            {"ref": "U1", "pin": "2", "net": "B"},
        ]
    }
    client = _ScriptedClient(
        [
            {"text": json.dumps(first), "finish_reason": "stop", "cost_usd": 0.0},
            {"text": json.dumps(corrected), "finish_reason": "stop", "cost_usd": 0.0},
        ]
    )
    client.s = Settings(api_key="test")
    result = stage_driver_mod.drive_stage(
        client,
        "wiring",
        "test",
        state_path,
        tmp_path,
        max_retries=4,
    )
    assert result["commit_ok"] is True
    assert result["expanded_component_count"] == 0
    assert client.calls[0]["reasoning"] == {"enabled": False}
    assert all(
        call["response_format"]["json_schema"]["name"] == "kicraft_wiring_response_v3"
        for call in client.calls
    )


class _RaisingClient:
    """Raises the configured exception on every completion call."""

    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

        class _G:
            def status(self):
                return {"spent_total_usd": 0.0}

        self.guard = _G()

    def chat(
        self,
        messages,
        max_tokens=4096,
        temperature=0.2,
        progress=None,
        meta_ctx=None,
        reasoning=None,
        reasoning_guard=None,
        collection_bounds=(),
        response_format=None,
    ):
        self.calls += 1
        raise self.exc


def test_provider_failure_is_terminal_and_not_sent_through_json_recovery(tmp_path):
    client = _RaisingClient(requests.exceptions.HTTPError("402 Payment Required"))
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "failed"
    last = res["results"][-1]
    assert last["failure_kind"] == "provider_request_rejected"
    assert last["attempts"] == 1
    assert client.calls == 1  # terminal: NO serialization retry, NO re-parse


def test_transport_failure_is_terminal_and_not_sent_through_json_recovery(tmp_path):
    client = _RaisingClient(requests.exceptions.ConnectionError("connection reset"))
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "failed"
    last = res["results"][-1]
    assert last["failure_kind"] == "transport_connection"
    assert client.calls == 1


def test_budget_exceeded_propagates_and_is_not_classified(tmp_path):
    from kicraft.server.spend_guard import BudgetExceeded

    class _BrokeClient(_RaisingClient):
        def __init__(self):
            super().__init__(BudgetExceeded("run budget exhausted"))

    with pytest.raises(BudgetExceeded):
        run_session(tmp_path, "a USB-powered LED", ["intent"], client=_BrokeClient())


def test_response_policy_falls_back_for_mock_clients():
    from kicraft.server.config import StageResponsePolicy
    from kicraft.server.stage_runtime import _response_policy

    # a settings-less mock client: floored normal cap, no reasoning control
    # (no design_reasoning), the fixed serialization cap, one serialization retry
    pol = _response_policy(object(), "bom", 4096)
    assert isinstance(pol, StageResponsePolicy)
    assert pol.normal_max_tokens == 16384  # caller 4096 floored up for bom
    assert pol.normal_reasoning is None  # no .s -> no reasoning control
    assert pol.serialization_max_tokens == 32768
    assert pol.serialization_retries == 1
    assert [bound.field for bound in pol.collection_bounds] == [
        "groups",
        "arrays",
        "assumptions",
        "substitutions",
    ]
    assert pol.collection_bounds[0].total == 64
    assert pol.collection_bounds[0].per_group == 64
    assert [bound.total for bound in pol.collection_bounds[1:]] == [100, 32, 32]
    # a HIGHER caller cap is preserved (never floored down)
    assert _response_policy(object(), "bom", 20000).normal_max_tokens == 20000

    # a settings object WITH the policy method drives the values
    class _S:
        def design_stage_policy(self, stage, normal_max_tokens):
            return StageResponsePolicy(normal_max_tokens, {"enabled": False}, 12345, 1)

    class _C:
        s = _S()

    pol2 = _response_policy(_C(), "bom", 4096)
    assert pol2.serialization_max_tokens == 12345
    assert pol2.normal_reasoning == {"enabled": False}


def test_review_candidate_captures_forensics_without_committing(tmp_path):
    state_path = tmp_path / ".kicraft" / "state.json"
    events = []
    result = stage_driver_mod.drive_stage(
        _ScriptedClient([_ok_intent_reply()]),
        "intent",
        "a USB-powered LED",
        state_path,
        tmp_path,
        progress=events.append,
        review_before_commit=True,
    )

    assert result["needs_review"] is True
    assert result["commit_ok"] is False
    assert result["slot"]["project_stem"] == "USB_LED"
    assert result["debug_context"]["raw_response"] == _OK_INTENT
    assert result["debug_context"]["base_messages"][1]["role"] == "user"
    assert result["debug_context"]["response_schema"]
    assert result["debug_context"]["response_format"]
    assert [event["kind"] for event in events][-1] == "candidate_review"
    assert not state_path.exists()


def test_early_design_question_defaults_without_parking(tmp_path):
    question = {
        "text": json.dumps(
            {
                "questions": [
                    {
                        "text": "Which op amp?",
                        "options": ["Rail-to-rail", "Dual supply"],
                        "blocking": True,
                    }
                ]
            }
        ),
        "reasoning": "",
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }
    client = _ScriptedClient([question, _ok_intent_reply()])

    result = run_session(tmp_path, "an op-amp board", ["intent"], client=client)

    assert result["status"] == "ok"
    assert len(client.calls) == 2
    assert "Do not ask more questions" in client.calls[1]["messages"][-1]["content"]
    assert not json.loads((tmp_path / ".kicraft" / "state.json").read_text())["open_questions"]


def test_explicit_false_parks_an_early_stage_despite_a_redrive_instruction(tmp_path):
    question = {
        "text": json.dumps(
            {
                "questions": [
                    {
                        "text": "Which op amp?",
                        "options": ["Rail-to-rail", "Dual supply"],
                        "blocking": True,
                    }
                ]
            }
        ),
        "reasoning": "",
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }
    client = _ScriptedClient([question])

    result = run_session(
        tmp_path,
        "an op-amp board",
        ["intent"],
        client=client,
        answers=[{"text": "Earlier decision", "answer": "retained"}],
        instruction="Re-drive the stage.",
        auto_default_questions=False,
    )

    assert result["status"] == "awaiting_input"
    assert result["questions"][0]["text"] == "Which op amp?"
    assert len(client.calls) == 1


def test_complete_instruction_suppresses_repeated_blocking_question(tmp_path):
    question = {
        "text": json.dumps(
            {
                "questions": [
                    {
                        "text": "Which input contract?",
                        "options": ["15 V / 3 A", "20 V / 5 A"],
                        "blocking": True,
                    }
                ]
            }
        ),
        "reasoning": "",
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }
    client = _ScriptedClient([question, _ok_intent_reply()])
    result = run_session(
        tmp_path,
        "USB-PD input",
        ["intent"],
        client=client,
        instruction="Use a 15 V / 3 A USB-PD input contract.",
    )
    assert result["status"] == "ok"
    assert len(client.calls) == 2
    assert "Do not ask more questions" in client.calls[1]["messages"][-1]["content"]


def test_review_question_does_not_persist_open_questions(tmp_path):
    state_path = tmp_path / ".kicraft" / "state.json"
    raw = json.dumps(
        {
            "questions": [
                {
                    "text": "Which supply voltage?",
                    "stage": "intent",
                    "blocking": True,
                    "material": True,
                    "options": ["3.3 V", "5 V"],
                }
            ]
        }
    )
    result = stage_driver_mod.drive_stage(
        _ScriptedClient([{"text": raw, "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0}]),
        "intent",
        "an LED board",
        state_path,
        tmp_path,
        review_before_commit=True,
    )

    assert result["needs_input"] is True
    assert result["questions"][0]["text"] == "Which supply voltage?"
    assert result["debug_context"]["raw_response"] == raw
    assert not state_path.exists()


@pytest.mark.parametrize(
    "review,answers,instruction,reconcile,needs_input",
    [
        pytest.param(False, None, None, False, False, id="noninteractive-default"),
        pytest.param(
            True,
            [{"text": "Which supply voltage?", "answer": "5 V"}],
            None,
            False,
            False,
            id="answered-review",
        ),
        pytest.param(
            True, None, "Use 5 V; do not ask again.", False, False, id="instructed-review"
        ),
        pytest.param(True, None, None, False, True, id="interactive-clarification"),
        pytest.param(
            False,
            [{"text": "Which supply voltage?", "answer": "5 V"}],
            "Use the supplied answers.",
            True,
            True,
            id="reconcile-after-answer",
        ),
        pytest.param(True, None, "Use defaults.", True, True, id="reconcile-review"),
    ],
)
def test_semantic_repair_questions_follow_primary_disposition(
    tmp_path, monkeypatch, review, answers, instruction, reconcile, needs_input
):
    # Isolate the question transition from individual semantic detectors: a valid
    # candidate needs repair, and the repair provider asks rather than repairing.
    monkeypatch.setattr(
        stage_driver_mod,
        "diagnose_stage",
        lambda *args, **kwargs: [
            StageDiagnostic(
                code="intent_named_part_omitted",
                severity="repair_required",
                message="An explicit part needs clarification.",
                evidence=["LED"],
                detector_version=1,
            )
        ],
    )
    question = {
        "text": "Which supply voltage?",
        "blocking": True,
        "options": ["3.3 V", "5 V"],
    }
    if reconcile:
        question["reconcile_target"] = "bom"
    client = _ScriptedClient(
        [
            _ok_intent_reply(),
            {
                "text": json.dumps({"questions": [question]}),
                "reasoning": "",
                "finish_reason": "stop",
                "cost_usd": 0.0,
            },
        ]
    )
    client.s = Settings(api_key="test", stage_semantics="repair")
    state_path = tmp_path / ".kicraft" / "state.json"
    events = []
    result = stage_driver_mod.drive_stage(
        client,
        "intent",
        "a USB-powered LED",
        state_path,
        tmp_path,
        answers=answers,
        instruction=instruction,
        review_before_commit=review,
        progress=events.append,
    )

    assert result.get("needs_input", False) is needs_input
    assert bool([event for event in events if event["kind"] == "question"]) is needs_input
    if needs_input:
        assert result["questions"][0]["text"] == question["text"]
        assert result["questions"][0]["reconcile_target"] == ("bom" if reconcile else None)
        assert result["commit_ok"] is False
    else:
        # The recoverable question is not a replacement slot or an extra repair.
        assert "questions" not in result["slot"]
        assert result["commit_ok"] is (not review)
    if review:
        assert not state_path.exists()
    elif not needs_input:
        state = json.loads(state_path.read_text())
        assert not state.get("open_questions")


def test_attempt_trace_associates_candidates_with_one_bounded_repair(tmp_path, monkeypatch):
    records = []
    progress = []
    result, _client = _a3_run(
        tmp_path,
        monkeypatch,
        ["A", "A", "A"],
        extra_ok_reply=False,
        progress=progress.append,
        attempt_observer=records.append,
    )

    assert result["failure_kind"] == "commit_rejected"
    assert [row["provider_attempt"] for row in records] == [1, 2]
    assert [row["call_mode"] for row in records] == ["normal", "normal"]
    assert [row["outcome"] for row in records] == ["candidate", "candidate"]
    assert [row["version"] for row in records] == [2, 2]
    assert [row["unit_id"] for row in records] == ["wiring-u000"] * 2
    assert [row["aggregate_round"] for row in records] == [None, 1]
    assert [row["design_profile"] for row in records] == ["deepseek", "luna"]
    assert records[0]["aggregate_signature"] is None
    assert records[1]["aggregate_signature"] is not None
    for row in records:
        assert "raw" not in row and "messages" not in row and "reasoning" not in row
    plans = [event for event in progress if event.get("kind") == "work_unit_plan"]
    attempts = [event for event in progress if event.get("kind") == "work_unit_attempt"]
    accepted = [event for event in progress if event.get("kind") == "work_unit_done"]
    assert [(event["unit_id"], event["source"]) for event in plans] == [("wiring-u000", "llm")]
    assert [event["outcome"] for event in attempts] == ["candidate", "candidate"]
    assert [event["aggregate_round"] for event in attempts] == [None, 1]
    assert all(event["response_chars"] > 0 for event in attempts)
    assert [event["response_format_mode"] for event in attempts] == ["json_schema"] * 2
    assert [event["source"] for event in accepted] == ["llm", "llm"]
    for event in attempts:
        assert not ({"raw", "messages", "reasoning", "candidate"} & event.keys())


def test_neutral_series_feedback_allows_commit_progression(tmp_path, monkeypatch):
    state_path = _a3_wiring_state(tmp_path, monkeypatch)
    commits = iter(
        [
            (
                False,
                {
                    "ok": False,
                    "errors": ["9.15 no dangling signal nets: 1 signal net wires a single pin"],
                    "offenders": ["net 'B' on sheet 'MAIN' wires only R1.2"],
                    "offenders_total": 1,
                },
            ),
            (True, {"ok": True}),
        ]
    )
    monkeypatch.setattr(stage_driver_mod, "commit_stage", lambda *args, **kwargs: next(commits))
    first = {
        "pins": [
            {"ref": "U1", "pin": "1", "net": "A"},
            {"ref": "R1", "pin": "1", "net": "A"},
            {"ref": "R1", "pin": "2", "net": "B"},
            {"ref": "U1", "pin": "2", "net": "B"},
        ]
    }
    corrected = {
        "pins": [
            {"ref": "U1", "pin": "1", "net": "A"},
            {"ref": "R1", "pin": "1", "net": "A"},
            {"ref": "R1", "pin": "2", "net": "B"},
            {"ref": "U1", "pin": "2", "net": "B"},
        ]
    }
    replies = [
        {"text": json.dumps(slot), "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0}
        for slot in (first, corrected)
    ]
    client = _ScriptedClient(replies)
    client.s = Settings(api_key="test", design_profile="deepseek")
    records = []

    result = stage_driver_mod.drive_stage(
        client,
        "wiring",
        "test",
        state_path,
        tmp_path,
        max_retries=2,
        attempt_observer=records.append,
    )

    assert result["commit_ok"] is True
    feedback = client.calls[1]["messages"][-1]["content"]
    assert "Do not assume the populated side is the source" in feedback
    assert "moving the destination pin" not in feedback
    assert [row["outcome"] for row in records] == [
        "candidate",
        "candidate",
    ]
    assert records[0]["aggregate_signature"] is None
    assert records[1]["aggregate_signature"] is not None


# ---------- bounded BOM/wiring work-unit orchestration ----------


def _work_unit_state(tmp_path, monkeypatch):
    state = {
        "architecture": {
            "sheets": [{"name": "A"}, {"name": "B"}],
            "power_nets": [],
            "inter_sheet_nets": [],
        },
        "bom": {
            "parts": [
                {"ref": "U1", "sheet": "A", "symbol": "Test:U", "value": "IC"},
                {"ref": "R1", "sheet": "B", "symbol": "Test:R", "value": "1k"},
            ]
        },
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    prep = {
        "state": state,
        "extras": {
            "symbol_pinouts": {
                "U1": {"symbol": "Test:U", "pins": [{"number": "1"}]},
                "R1": {"symbol": "Test:R", "pins": [{"number": "1"}]},
            }
        },
    }
    monkeypatch.setattr(
        stage_driver_mod,
        "prepare_stage",
        lambda *args, **kwargs: type(
            "Proc", (), {"returncode": 0, "stdout": json.dumps(prep), "stderr": ""}
        )(),
    )
    return state_path


def _unit_reply(ref, net="N"):
    return {
        "text": json.dumps({"pins": [{"ref": ref, "pin": "1", "net": net}]}),
        "reasoning": "",
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }


def _unit_client(replies):
    client = _ScriptedClient(replies)
    client.s = Settings(
        api_key="test",
        design_profile="deepseek",
        model=str(DESIGN_PROFILES["deepseek"]["model"]),
        provider_order=list(DESIGN_PROFILES["deepseek"]["provider_order"]),
        stage_semantics="observe",
        provider_fallback_profile="",
    )
    return client


def test_work_units_make_one_initial_call_each_before_one_full_commit(tmp_path, monkeypatch):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    client = _unit_client([_unit_reply("U1"), _unit_reply("R1")])
    commits = []

    def commit(stage, slot, *args, **kwargs):
        commits.append(slot)
        assert len(client.calls) == 2
        return True, {"ok": True}

    monkeypatch.setattr(stage_driver_mod, "commit_stage", commit)
    result = stage_driver_mod.drive_stage(client, "wiring", "test", state_path, tmp_path)

    assert result["commit_ok"] is True
    assert result["work_units"] == 2
    assert len(client.calls) == 2
    assert [call["collection_bounds"][0].total for call in client.calls] == [1, 1]
    # Wiring keeps the question affordance: a missing support part must be able
    # to request one BOM reconcile round.
    for call in client.calls:
        properties = call["response_format"]["json_schema"]["schema"]["properties"]
        assert "questions" in properties and properties["questions"]["minItems"] == 0
    assert len(commits) == 1


def test_deterministic_endpoint_assertion_ignores_sheets_without_pins():
    from kicraft.server.stage_work_units import _assert_deterministic_endpoints

    prompt_state = {
        "bom": {
            "parts": [
                {"ref": "H1", "sheet": "MOUNTING STRUCTURE"},
                {"ref": "U1", "sheet": "MAIN"},
            ]
        },
        "architecture": {
            "inter_sheet_nets": [
                {
                    "name": "GND",
                    "endpoints": [{"sheet": "MOUNTING STRUCTURE", "direction": "passive"}],
                },
                {"name": "SIG", "endpoints": [{"sheet": "MAIN", "direction": "output"}]},
            ]
        },
    }
    # A pinless mechanical sheet cannot carry an assignment, so its endpoint is
    # ignored; a sheet with wireable pins and no assignment still fails.
    with pytest.raises(ValueError, match="MAIN:SIG") as excinfo:
        _assert_deterministic_endpoints((), prompt_state, {}, {"H1": (), "U1": ("1",)})
    assert "MOUNTING STRUCTURE" not in str(excinfo.value)


def test_endpoint_assertion_ignores_power_nets():
    from kicraft.server.stage_work_units import _assert_deterministic_endpoints

    prompt_state = {
        "bom": {"parts": [{"ref": "U1", "sheet": "MAIN"}]},
        "architecture": {
            "inter_sheet_nets": [
                {"name": "VCC_5V", "endpoints": [{"sheet": "MAIN", "direction": "output"}]},
                {"name": "SIG", "endpoints": [{"sheet": "MAIN", "direction": "output"}]},
            ]
        },
    }
    # Power crosses via global symbols, not a wiring assignment (§9.14 exempts it
    # too); a signal net with no carrying assignment still fails.
    with pytest.raises(ValueError, match="MAIN:SIG") as excinfo:
        _assert_deterministic_endpoints((), prompt_state, {}, {"U1": ("1",)})
    assert "VCC_5V" not in str(excinfo.value)


def test_missing_requirement_defect_names_the_required_identity():
    from kicraft.server.stage_runtime import _requirement_identity

    identity = _requirement_identity(
        {
            "id": "relay_driver",
            "role": "driver",
            "family": "uln2003-smt",
            "exact_part": "ULN2003",
        }
    )
    assert identity.startswith("relay_driver role=driver family=uln2003-smt exact_part=ULN2003")
    # A requirement that names a device without an order code must be told the
    # reviewed concrete parts it could name instead (the boundary's own
    # alternatives), otherwise the defect names a problem with no remedy.
    assert "accepted_concrete_parts=['uln2003a', 'uln2003adr']" in identity
    assert _requirement_identity({"id": "x"}) == "x"


@pytest.mark.parametrize("recover", [True, False])
def test_bom_unit_collection_overflow_gets_one_focused_recovery(tmp_path, monkeypatch, recover):
    state = {
        "architecture": {
            "sheets": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            "recipe_selections": [],
        }
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(
        stage_driver_mod,
        "prepare_stage",
        lambda *args, **kwargs: type(
            "Proc",
            (),
            {"returncode": 0, "stdout": json.dumps({"state": state, "extras": {}}), "stderr": ""},
        )(),
    )

    def reply(sheet):
        return {
            "text": json.dumps({"groups": [_group_payload(sheet=sheet)]}),
            "finish_reason": "stop",
            "cost_usd": 0.0,
        }

    overflow = {
        "text": '{"groups":[',
        "finish_reason": "collection_limit",
        "collection_limit": {
            "field": "groups",
            "configured_total": 21,
            "observed_count": 22,
            "emitted_content_chars": 3726,
        },
        "cost_usd": 0.0,
    }
    correction = (
        reply("B")
        if recover
        else {
            **overflow,
            "collection_limit": {**overflow["collection_limit"], "emitted_content_chars": 4767},
        }
    )
    replies = [reply("A"), overflow, correction]
    if recover:
        replies.append(reply("C"))
    else:
        # One identical defect may repeat before it is terminal; the pristine
        # clean-slate retry still ends the unit.
        replies.append(dict(correction))
    client = _unit_client(replies)
    monkeypatch.setattr(
        stage_driver_mod, "commit_stage", lambda *args, **kwargs: (True, {"ok": True})
    )

    result = stage_driver_mod.drive_stage(client, "bom", "passive board", state_path, tmp_path)

    assert result["commit_ok"] is recover
    assert [call["serialization"] for call in client.calls[:3]] == [False, False, True]
    for call in client.calls:
        schema = call["response_format"]["json_schema"]["schema"]
        groups = schema["properties"]["groups"]
        assert groups["maxItems"] == 21
        assert call["max_tokens"] == 2048
        # BOM units are non-interactive: a question can never park the stage, so
        # the affordance must not be offered (it only burns repair turns).
        assert "questions" not in schema["properties"]
    if recover:
        assert [(part["ref"], part["sheet"]) for part in result["slot"]["parts"]] == [
            ("C1", "A"),
            ("C2", "A"),
            ("C3", "B"),
            ("C4", "B"),
            ("C5", "C"),
            ("C6", "C"),
        ]
        assert len(client.calls) == 4
    else:
        assert result["failure_kind"] == "collection_limit"
        assert len(client.calls) == 4
        saved = json.loads((tmp_path / "drafts" / "bom-units.json").read_text())
        # The normalized group carries the contract's explicit assembly default.
        assert saved["candidates"]["bom-s000"]["groups"] == [
            {"assembly": True, **_group_payload(sheet="A")}
        ]


def test_work_unit_nonconsecutive_failure_retains_accepted_sibling(tmp_path, monkeypatch):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    client = _unit_client(
        [
            _unit_reply("U1"),
            _unit_reply("X1"),
            _unit_reply("Y1"),
            _unit_reply("X1", "RENAMED"),
            # One identical defect may repeat before it is terminal.
            _unit_reply("X1", "RENAMED"),
        ]
    )
    events = []

    result = stage_driver_mod.drive_stage(
        client, "wiring", "test", state_path, tmp_path, progress=events.append
    )

    assert result["failure_kind"] == "unit_repair_exhausted"
    assert len(client.calls) == 5
    saved = json.loads((tmp_path / "drafts" / "wiring-units.json").read_text())
    assert saved["candidates"] == {"wiring-u000": {"pins": [{"ref": "U1", "pin": "1", "net": "N"}]}}
    done = next(event for event in events if event["kind"] == "stage_done")
    assert done["diagnostic"]["unit_id"] == "wiring-u001"
    assert done["diagnostic"]["defects"]["unknown-ref"] == ["X1"]
    assert done["diagnostic"]["defects"]["unexpected"] == ["X1.1"]
    assert "X1.1" in done["schema_error"]


def test_invalid_wiring_unit_escalates_after_first_attempt(tmp_path, monkeypatch):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    client = _unit_client([_unit_reply("R1", "WRONG"), _unit_reply("U1"), _unit_reply("R1")])
    client.s = replace(client.s, escalation_profile="luna")
    monkeypatch.setattr(
        stage_driver_mod,
        "commit_stage",
        lambda *args, **kwargs: (True, {"ok": True}),
    )

    result = stage_driver_mod.drive_stage(client, "wiring", "test", state_path, tmp_path)

    assert result["commit_ok"] is True
    assert [call["model"] for call in client.calls] == [
        str(DESIGN_PROFILES["deepseek"]["model"]),
        str(DESIGN_PROFILES["luna"]["model"]),
        str(DESIGN_PROFILES["deepseek"]["model"]),
    ]


def test_empty_escalation_profile_never_switches_models(tmp_path, monkeypatch):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    client = _unit_client([_unit_reply("R1", "WRONG"), _unit_reply("U1"), _unit_reply("R1")])
    assert client.s.escalation_profile == ""
    monkeypatch.setattr(
        stage_driver_mod,
        "commit_stage",
        lambda *args, **kwargs: (True, {"ok": True}),
    )

    result = stage_driver_mod.drive_stage(client, "wiring", "test", state_path, tmp_path)

    assert result["commit_ok"] is True
    assert {call["model"] for call in client.calls} == {
        str(DESIGN_PROFILES["deepseek"]["model"])
    }

def test_explicit_false_allows_bom_work_units_to_park_on_a_question(tmp_path, monkeypatch):
    state = {
        "architecture": {
            "sheets": [{"name": "ANALOG"}],
            "recipe_selections": [],
            "requirements": [
                {
                    "id": "sensor_front_end",
                    "sheet": "ANALOG",
                    "role": "sensor",
                    "family": "novel_sensor",
                    "ports": {"out": "SENSE"},
                }
            ],
        }
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(
        stage_driver_mod,
        "prepare_stage",
        lambda *args, **kwargs: type(
            "Proc",
            (),
            {"returncode": 0, "stdout": json.dumps({"state": state, "extras": {}}), "stderr": ""},
        )(),
    )
    client = _unit_client(
        [
            {
                "text": json.dumps(
                    {
                        "questions": [
                            {
                                "text": "Which input connector style?",
                                "options": ["Screw terminal", "Barrel jack"],
                                "blocking": True,
                            }
                        ]
                    }
                ),
                "finish_reason": "stop",
                "cost_usd": 0.0,
            }
        ]
    )
    monkeypatch.setattr(
        stage_driver_mod,
        "commit_stage",
        lambda *args, **kwargs: pytest.fail("interactive BOM question must park before commit"),
    )

    result = stage_driver_mod.drive_stage(
        client,
        "bom",
        "adjustable power board",
        state_path,
        tmp_path,
        auto_default_questions=False,
    )

    assert result["needs_input"] is True
    assert result["questions"][0]["options"] == ["Screw terminal", "Barrel jack"]
    properties = client.calls[0]["response_format"]["json_schema"]["schema"]["properties"]
    assert "questions" in properties


def test_rate_limited_escalated_unit_returns_to_primary_profile(tmp_path, monkeypatch):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    client = _unit_client(
        [
            _unit_reply("R1", "WRONG"),
            requests.exceptions.HTTPError("429 Too Many Requests"),
            _unit_reply("U1"),
            _unit_reply("R1"),
        ]
    )
    client.s = replace(
        client.s,
        escalation_profile="luna",
        provider_fallback_profile="luna",
    )
    monkeypatch.setattr(
        stage_driver_mod,
        "commit_stage",
        lambda *args, **kwargs: (True, {"ok": True}),
    )

    result = stage_driver_mod.drive_stage(client, "wiring", "test", state_path, tmp_path)

    assert result["commit_ok"] is True
    assert [call["model"] for call in client.calls] == [
        str(DESIGN_PROFILES["deepseek"]["model"]),
        str(DESIGN_PROFILES["luna"]["model"]),
        str(DESIGN_PROFILES["deepseek"]["model"]),
        str(DESIGN_PROFILES["deepseek"]["model"]),
    ]


def test_bom_work_unit_provider_question_defaults_without_parking(tmp_path, monkeypatch):
    state = {
        "architecture": {
            "sheets": [{"name": "PD TRIGGER"}],
            "power_nets": ["VBUS", "GND"],
            "inter_sheet_nets": [],
            "recipe_selections": [],
            "requirements": [
                {
                    "id": "pd_trigger_controller",
                    "sheet": "PD TRIGGER",
                    "role": "bus_interface",
                    "family": "pd_trigger_controller",
                    "exact_part": "HUSB238",
                    "parameters": {"supported_pdos": "9V,12V,20V"},
                    "ports": {"VBUS": "VBUS"},
                },
                {
                    "id": "usb_c_receptacle",
                    "sheet": "PD TRIGGER",
                    "role": "connector",
                    "family": "usb_c_receptacle",
                    # A protected class is owned by naming reviewed hardware, so the
                    # typed requirement carries the reviewed order code.
                    "exact_part": "TYPE-C-31-M-12",
                    "ports": {"VBUS": "VBUS", "GND": "GND"},
                },
            ],
        }
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    monkeypatch.setattr(
        stage_driver_mod,
        "prepare_stage",
        lambda *args, **kwargs: type(
            "Proc",
            (),
            {"returncode": 0, "stdout": json.dumps({"state": state, "extras": {}}), "stderr": ""},
        )(),
    )
    question = {
        "text": json.dumps(
            {
                "questions": [
                    {
                        "text": "Which MCU/control architecture should the BOM reflect?",
                        "options": ["Dedicated PD trigger", "ESP32-S3", "ESP32 + CH340C"],
                        "blocking": True,
                        "material": True,
                    }
                ]
            }
        ),
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }
    candidate = {
        "text": json.dumps(
            {
                "groups": [
                    _group_payload(
                        id="pd_controller",
                        reference_prefix="U",
                        quantity=1,
                        value="HUSB238",
                        symbol="Test:HUSB238",
                        footprint="Test:HUSB238",
                        sheet="PD TRIGGER",
                    )
                ],
                "arrays": [],
                "assumptions": ["Selected a dedicated PD controller (defaulted)"],
                "substitutions": [],
            }
        ),
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }
    protected_candidate = {
        "text": json.dumps(
            {
                **json.loads(candidate["text"]),
                "groups": [
                    *json.loads(candidate["text"])["groups"],
                    # A protected class must be realized by its reviewed part, so
                    # this group names the reviewed USB-C bundle's own pair instead
                    # of an unreviewed stock symbol (which the boundary refuses).
                    _group_payload(
                        id="usb_c_receptacle",
                        reference_prefix="J",
                        quantity=1,
                        value="TYPE-C-31-M-12",
                        symbol="usb-c-16p:TYPE-C-31-M-12",
                        footprint="usb-c-16p:USB-C_SMD-TYPE-C-31-M-12_1",
                        mpn="TYPE-C-31-M-12",
                        sheet="PD TRIGGER",
                    ),
                ],
            }
        ),
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }
    client = _unit_client([question, protected_candidate])
    client.s = replace(client.s, escalation_profile="luna")
    committed = []

    def commit(_stage, slot, *args, **kwargs):
        committed.append(slot)
        return True, {"ok": True}

    monkeypatch.setattr(stage_driver_mod, "commit_stage", commit)

    result = stage_driver_mod.drive_stage(
        client,
        "bom",
        "USB-C PD trigger",
        state_path,
        tmp_path,
        max_tokens=16384,
    )

    assert result["commit_ok"] is True, result.get("error") or result
    assert result.get("needs_input") is not True
    assert {part["value"] for part in committed[0]["parts"]} == {
        "HUSB238",
        "TYPE-C-31-M-12",
    }


@pytest.mark.parametrize(
    ("rejection", "repair_refs"),
    [
        ({"ok": False, "errors": ["gate"], "offenders": ["U1.1"]}, ["U1"]),
        ({"ok": False, "errors": ["unscoped"], "offenders": []}, ["U1", "R1"]),
    ],
)
def test_work_unit_commit_repair_redrafts_only_mapped_owners(
    tmp_path, monkeypatch, rejection, repair_refs
):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    replies = [_unit_reply("U1"), _unit_reply("R1")] + [
        _unit_reply(ref, "FIXED") for ref in repair_refs
    ]
    client = _unit_client(replies)
    commits = {"count": 0}

    def commit(*args, **kwargs):
        commits["count"] += 1
        return (False, rejection) if commits["count"] == 1 else (True, {"ok": True})

    monkeypatch.setattr(stage_driver_mod, "commit_stage", commit)
    result = stage_driver_mod.drive_stage(client, "wiring", "test", state_path, tmp_path)

    assert result["commit_ok"] is True
    assert len(client.calls) == 2 + len(repair_refs)
    for call, ref in zip(client.calls[2:], repair_refs, strict=True):
        assert f'"owned_refs":["{ref}"]' in call["messages"][0]["content"]


def test_work_unit_question_discards_checkpoint_but_provider_failure_retains_it(
    tmp_path, monkeypatch
):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    question_client = _unit_client(
        [
            _unit_reply("U1"),
            {
                "text": json.dumps(
                    {
                        "questions": [
                            {
                                "text": "Choose?",
                                "options": ["A", "B"],
                                "blocking": True,
                            }
                        ]
                    }
                ),
                "finish_reason": "stop",
                "cost_usd": 0.0,
            },
        ]
    )
    result = stage_driver_mod.drive_stage(question_client, "wiring", "test", state_path, tmp_path)
    draft_path = tmp_path / "drafts" / "wiring-units.json"
    assert result["needs_input"] is True
    assert not draft_path.exists()

    state_path = _work_unit_state(tmp_path, monkeypatch)
    failure_client = _unit_client([_unit_reply("U1"), requests.exceptions.Timeout("down")])
    result = stage_driver_mod.drive_stage(failure_client, "wiring", "test", state_path, tmp_path)
    assert result["failure_kind"] == "transport_timeout"
    assert draft_path.exists()
    saved = json.loads(draft_path.read_text())
    assert list(saved["candidates"]) == ["wiring-u000"]


def test_explicit_true_defaults_an_ordinary_wiring_question_but_not_reconciliation(
    tmp_path, monkeypatch
):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    question = {
        "text": json.dumps(
            {
                "questions": [
                    {
                        "text": "Choose?",
                        "options": ["Recommended route", "Alternative route"],
                        "blocking": True,
                    }
                ]
            }
        ),
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }
    client = _unit_client([_unit_reply("U1"), question, _unit_reply("R1")])
    monkeypatch.setattr(
        stage_driver_mod, "commit_stage", lambda *args, **kwargs: (True, {"ok": True})
    )

    result = stage_driver_mod.drive_stage(
        client,
        "wiring",
        "test",
        state_path,
        tmp_path,
        auto_default_questions=True,
    )

    assert result["commit_ok"] is True
    assert "Do not ask more questions" in client.calls[2]["messages"][-1]["content"]


def test_work_unit_debug_review_and_observer_v2_are_redacted(tmp_path, monkeypatch):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    client = _unit_client([_unit_reply("U1"), _unit_reply("R1")])
    monkeypatch.setattr(
        stage_driver_mod,
        "commit_stage",
        lambda *args, **kwargs: pytest.fail("review mode must not commit"),
    )
    observed = []
    result = stage_driver_mod.drive_stage(
        client,
        "wiring",
        "test",
        state_path,
        tmp_path,
        review_before_commit=True,
        attempt_observer=observed.append,
    )

    assert result["needs_review"] is True
    assert result["work_units"] == 2
    assert [row["unit_id"] for row in observed] == ["wiring-u000", "wiring-u001"]
    assert all(row["version"] == 2 for row in observed)
    assert all(
        forbidden not in row
        for row in observed
        for forbidden in ("prompt", "brief", "reasoning", "raw_reply", "tool_output")
    )


def test_work_unit_repeated_commit_signature_stops_when_candidate_also_repeats(
    tmp_path,
    monkeypatch,
):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    client = _unit_client(
        [
            _unit_reply("U1"),
            _unit_reply("R1"),
            _unit_reply("U1", "PRESERVED"),
            _unit_reply("U1", "PRESERVED"),
        ]
    )
    commits = {"count": 0}

    def reject(*args, **kwargs):
        commits["count"] += 1
        return False, {"ok": False, "errors": ["same gate"], "offenders": ["U1.1"]}

    monkeypatch.setattr(stage_driver_mod, "commit_stage", reject)
    result = stage_driver_mod.drive_stage(
        client, "wiring", "test", state_path, tmp_path, max_retries=99
    )

    assert result["failure_kind"] == "commit_rejected"
    assert result["aggregate_repair_rounds"] == 2
    assert len(client.calls) == 4
    assert commits["count"] == 3


def test_work_unit_nonconsecutive_commit_failure_stops_before_another_redraft(
    tmp_path, monkeypatch
):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    client = _unit_client(
        [_unit_reply("U1"), _unit_reply("R1"), _unit_reply("U1", "OTHER"), _unit_reply("U1")]
    )
    committed = []

    def reject(_stage, slot, *args, **kwargs):
        committed.append(slot)
        return False, {"ok": False, "errors": ["same gate"], "offenders": ["U1.1"]}

    monkeypatch.setattr(stage_driver_mod, "commit_stage", reject)
    result = stage_driver_mod.drive_stage(
        client, "wiring", "test", state_path, tmp_path, max_retries=99
    )

    assert result["failure_kind"] == "commit_rejected"
    assert len(client.calls) == 4
    assert len(committed) == 3
    assert committed[0] == committed[2]
    assert committed[0] != committed[1]


def test_repeated_model_singleton_repairs_locally_and_retains_sibling(
    tmp_path,
    monkeypatch,
):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    client = _unit_client(
        [
            _unit_reply("U1"),
            _unit_reply("R1"),
            _unit_reply("U1", "PRESERVED"),
            _unit_reply("U1", "REPAIRED"),
        ]
    )
    commits = []
    rejection = {
        "ok": False,
        "errors": ["9.15 no dangling signal nets"],
        "offenders": [
            "net 'CTRL' on sheet 'A' wires only U1.1 and is neither a power net "
            "nor a declared inter-sheet net"
        ],
    }

    def reject(_stage, slot, *args, **kwargs):
        commits.append(slot)
        return (False, rejection) if len(commits) < 3 else (True, {"ok": True})

    monkeypatch.setattr(stage_driver_mod, "commit_stage", reject)
    result = stage_driver_mod.drive_stage(
        client,
        "wiring",
        "test",
        state_path,
        tmp_path,
        max_retries=99,
    )

    assert result["commit_ok"] is True
    assert result["aggregate_repair_rounds"] == 2
    assert len(client.calls) == 4
    assert len(commits) == 3
    sibling = lambda slot: [
        row for row in slot["connections"]
        if any(endpoint["ref"] == "R1" for endpoint in row["endpoints"])
    ]
    assert sibling(commits[0]) == sibling(commits[1]) == sibling(commits[2])


@pytest.mark.parametrize("offender_ref", ["U1", "R1"])
def test_singleton_reconciliation_respects_exact_immutable_pin_owner(
    tmp_path, monkeypatch, offender_ref
):
    from kicraft.server import stage_runtime

    state_path = _work_unit_state(tmp_path, monkeypatch)
    monkeypatch.setattr(
        stage_runtime,
        "deterministic_wiring_candidate",
        lambda unit, *_: (
            {"pins": [{"ref": "U1", "pin": "1", "net": "CTRL"}]} if unit.refs == ("U1",) else None
        ),
    )
    client = _unit_client([_unit_reply("R1"), _unit_reply("R1", "OTHER"), _unit_reply("R1", "OTHER")])
    rejection = {
        "ok": False,
        "errors": ["9.15 no dangling signal nets"],
        "offenders": [
            f"net 'CTRL' wires only {offender_ref}.1 and is neither a power net "
            "nor a declared inter-sheet net"
        ],
    }
    monkeypatch.setattr(
        stage_driver_mod, "commit_stage", lambda *args, **kwargs: (False, rejection)
    )

    result = stage_driver_mod.drive_stage(
        client, "wiring", "test", state_path, tmp_path, max_retries=99
    )

    assert result["failure_kind"] == (
        "architecture_reconciliation_required" if offender_ref == "U1" else "commit_rejected"
    )
    assert result["aggregate_repair_rounds"] == (0 if offender_ref == "U1" else 2)
    assert len(client.calls) == (1 if offender_ref == "U1" else 3)


def test_work_unit_bom_reconcile_question_keeps_pipeline_park_shape(tmp_path, monkeypatch):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    client = _unit_client(
        [
            {
                "text": json.dumps(
                    {
                        "questions": [
                            {
                                "text": "Add C1 before wiring",
                                "options": ["Add it"],
                                "blocking": True,
                                "reconcile_target": "bom",
                            }
                        ]
                    }
                ),
                "finish_reason": "stop",
                "cost_usd": 0.0,
            }
        ]
    )
    monkeypatch.setattr(
        stage_driver_mod,
        "commit_stage",
        lambda *args, **kwargs: pytest.fail("reconcile park must not commit"),
    )

    result = stage_driver_mod.drive_stage(
        client,
        "wiring",
        "test",
        state_path,
        tmp_path,
        answers=[{"text": "prior", "answer": "yes"}],
        auto_default_questions=True,
    )

    assert result["needs_input"] is True
    assert result["questions"][0]["reconcile_target"] == "bom"


def test_noninteractive_default_policy_applies_to_every_stage(tmp_path, monkeypatch):
    seen = []

    def fake_drive_stage(_client, stage, _brief, _state_path, _workspace, *args, **kwargs):
        seen.append((stage, kwargs["instruction"]))
        return {"stage": stage, "commit_ok": True, "cost_usd": 0.0}

    class FakeGuard:
        @staticmethod
        def status():
            return {}

    class FakeClient:
        guard = FakeGuard()

    monkeypatch.setattr(stage_pipeline, "drive_stage", fake_drive_stage)
    stages = ["intent", "functional_spec", "architecture"]
    stage_pipeline.drive_chain(
        stages,
        "brief",
        tmp_path,
        client=FakeClient(),
        instruction=stage_driver_mod.NONINTERACTIVE_DEFAULTS_INSTRUCTION,
    )

    assert seen == [
        (stage, stage_driver_mod.NONINTERACTIVE_DEFAULTS_INSTRUCTION) for stage in stages
    ]


def test_contract_rejection_gets_the_schema_correction_not_a_reserialize():
    """A semantic/recipe contract rejection must be corrected, not reserialized.

    `invalid_schema` and `contract_rejected` share the StageSchemaError path, so
    a new kind that misses the template branch silently hands the model the
    "your reply was not a single complete JSON object" instruction — which is
    both false and useless for a candidate that parsed fine.
    """
    common = dict(raw='{"sheets": []}', bounds_sentence="", schema_error="unsupported_recipe_endpoint: ...")
    contract = stage_driver_mod._stage_recovery_message("contract_rejected", **common)
    schema = stage_driver_mod._stage_recovery_message("invalid_schema", **common)

    assert contract == schema
    assert "failed KiCraft's local slot validation" in contract
    assert "not a single complete JSON object" not in contract


@pytest.mark.parametrize("diagnostic,expected", [
    ({"code": "unsupported_recipe_endpoint"}, "contract_rejected"),
    (None, "invalid_schema"),
])
def test_decode_splits_contract_rejections_from_bad_provider_json(
    monkeypatch, diagnostic, expected
):
    """The label split is exactly 'did a contract attach a diagnostic'.

    A recipe/semantic contract refusing a schema-clean candidate is not the
    provider failing to produce JSON, and the durable failure_kind must say so.
    """
    from types import SimpleNamespace

    from kicraft.server.stage_contracts import StageSchemaError

    def boom(*_args, **_kwargs):
        raise StageSchemaError("unsupported_recipe_endpoint: no compatible port",
                               diagnostic=diagnostic)

    monkeypatch.setattr(stage_driver_mod, "_normalize_stage_response", boom)
    facts = stage_driver_mod.ProviderFacts(
        raw='{"sheets": []}', finish="stop", rounds=None, tool_calls=None,
        cost_usd=0.0, had_content=True, loop_detected=False,
        collection_limit=None, loop_abort_reason=None, collection_counts={},
    )
    outcome = stage_driver_mod.decode_stage_response(
        SimpleNamespace(stage="architecture", prompt_state={}), facts
    )
    assert outcome.payload["failure_kind"] == expected
    assert outcome.payload["schema_error"].startswith("unsupported_recipe_endpoint")


# --------------------------------------------------------------------------- #
# Correction-ladder arms (docs/plans/architecture-contract-correction-ladder.md)
# --------------------------------------------------------------------------- #
def _ladder_client(replies, **overrides):
    """A scripted client whose settings carry one KICRAFT_CONTRACT_LADDER arm."""
    client = _ScriptedClient(replies)
    client.s = Settings(api_key="test", **overrides)
    return client


def _bad_reply(text, finish_reason="stop"):
    return {"text": text, "reasoning": "", "finish_reason": finish_reason, "cost_usd": 0.0}


def _rungs(client):
    """The call modes the drive actually spent, in order."""
    return ["serialization" if call["serialization"] else "normal" for call in client.calls]


def test_ladder_stock_spends_serialization_then_one_terminal_clean_slate(tmp_path):
    client = _ladder_client([_bad_reply("not json at all")] * 4)
    result = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    # 3 calls: draft, dedicated serialization, one clean-slate escape whose
    # rejection is terminal by policy whatever the nominal budget allows.
    assert _rungs(client) == ["normal", "serialization", "normal"]
    assert result["results"][-1]["attempts"] == 3
    assert result["results"][-1]["failure_kind"] == "invalid_json"


def test_ladder_no_serialization_spends_only_preserving_corrections(tmp_path):
    client = _ladder_client([_bad_reply("not json at all")] * 4, contract_ladder="no_serialization")
    result = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert _rungs(client) == ["normal", "normal", "normal"]
    assert all(call["reasoning"] == {"enabled": False} for call in client.calls[1:])
    # The cleaner is told its own draft was malformed, never to start fresh.
    assert "not a single complete JSON object" in client.calls[1]["messages"][-1]["content"]
    assert result["results"][-1]["attempts"] == 3
    assert result["results"][-1]["failure_kind"] == "invalid_json"


def _contract_rejecting_normalizer(codes):
    """Patch the normalizer to reject each reply with the next diagnostic code."""
    from kicraft.server.stage_contracts import StageSchemaError

    seen = {"n": 0}

    def boom(*_args, **_kwargs):
        code = codes[min(seen["n"], len(codes) - 1)]
        seen["n"] += 1
        raise StageSchemaError(
            f"{code}: deterministic contract refused the candidate",
            diagnostic={"code": code, "message": f"{code} at rung {seen['n']}", "evidence": []},
        )

    return boom


def test_ladder_signature_continues_when_the_contract_defect_changed(tmp_path, monkeypatch):
    replies = [_bad_reply("{}")] * 5

    stock = _ladder_client(replies)
    monkeypatch.setattr(
        stage_driver_mod, "_normalize_stage_response", _contract_rejecting_normalizer(["a", "b", "c"])
    )
    stage_pipeline.drive_chain(["intent"], "a USB-powered LED", tmp_path / "stock",
                               max_retries=3, client=stock)
    # A design-contract refusal whose defect set changed now earns preserving corrections on its
    # own: this replay changes once (a -> c) and then repeats c, so it earns one round and stops —
    # 4 calls where it used to stop at 3. Every other failure lane keeps its stock ceiling.
    assert _rungs(stock) == ["normal", "serialization", "normal", "normal"]
    assert len(stock.calls) == 4

    monkeypatch.setattr(
        stage_driver_mod,
        "_normalize_stage_response",
        _contract_rejecting_normalizer(["a", "b", "c", "d"]),
    )
    signature = _ladder_client(replies, contract_ladder="signature")
    _results, _guard, state_path = stage_pipeline.drive_chain(
        ["intent"], "a USB-powered LED", tmp_path / "sig", max_retries=3, client=signature
    )
    # A clean-slate response that failed on a different diagnostic is progress:
    # the drive keeps correcting (never a second escape), to the call budget.
    assert _rungs(signature) == ["normal", "serialization", "normal", "normal"]
    assert state_path.endswith("state.json")


def test_ladder_signature_keeps_a_repeated_contract_defect_terminal(tmp_path, monkeypatch):
    client = _ladder_client([_bad_reply("{}")] * 5, contract_ladder="signature")
    monkeypatch.setattr(
        stage_driver_mod, "_normalize_stage_response", _contract_rejecting_normalizer(["a", "b", "b"])
    )
    result = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert len(client.calls) == 3
    assert result["results"][-1]["failure_kind"] == "contract_rejected"


def test_ladder_signature_never_rescues_a_diagnostic_free_parse_failure(tmp_path):
    # invalid_json/truncated_json carry no diagnostic identity, so the escape
    # stays terminal whatever the modality.
    client = _ladder_client([_bad_reply("not json at all")] * 4, contract_ladder="signature")
    result = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert len(client.calls) == 3
    assert result["results"][-1]["failure_kind"] == "invalid_json"


def test_ladder_preserving_clean_slate_carries_the_previous_candidate(tmp_path):
    replies = [
        _bad_reply("not json at all"),
        _bad_reply("still not json"),
        _bad_reply("also not json"),
    ]
    stock = _ladder_client(list(replies))
    run_session(tmp_path / "stock", "a USB-powered LED", ["intent"], client=stock)
    stock_clean_slate = stock.calls[-1]["messages"]
    assert [row["role"] for row in stock_clean_slate][-1:] == ["user"]

    preserving = _ladder_client(list(replies), contract_ladder="preserving")
    run_session(tmp_path / "preserving", "a USB-powered LED", ["intent"], client=preserving)
    messages = preserving.calls[-1]["messages"]
    assert messages[-2] == {"role": "assistant", "content": "still not json"}
    assert "Preserve every already-valid net" in messages[-1]["content"]
    assert "Start from the binding state" not in messages[-1]["content"]


def test_ladder_dropped_gate_names_content_the_revision_dropped(tmp_path):
    def arch_like(include_d) -> str:
        # Parseable JSON that the intent schema rejects (unknown keys) — the
        # same shape the dropped-content gate reads its identities from.
        payload = {
            "goal": "a HUB75 board",
            "inter_sheet_nets": [
                {"name": name, "endpoints": [{"sheet": "MCU"}, {"sheet": "HUB75"}]}
                for name in (("HUB75_D", "HUB75_OE") if include_d else ("HUB75_OE",))
            ],
            "requirements": [{"id": "hub75", "ports": {"addr_d": "HUB75_D"} if include_d else {}}],
        }
        return json.dumps(payload)

    replies = [_bad_reply(arch_like(True)), _bad_reply(arch_like(False)), _bad_reply(arch_like(False))]

    gate = _ladder_client(list(replies), contract_ladder="dropped_gate")
    run_session(tmp_path / "gate", "a HUB75 board", ["intent"], client=gate)
    # The loss is detectable only after the revision that dropped it: the note
    # rides the correction built from rung 2's rejection (the clean-slate call).
    clean_slate_message = gate.calls[2]["messages"][-1]["content"]
    assert "REGRESSION FIX" in clean_slate_message
    assert "'HUB75_D'" in clean_slate_message

    stock = _ladder_client(list(replies))
    run_session(tmp_path / "stock", "a HUB75 board", ["intent"], client=stock)
    assert "REGRESSION FIX" not in stock.calls[2]["messages"][-1]["content"]


def test_ladder_full_feedback_repeats_every_diagnostic_from_the_drive(tmp_path, monkeypatch):
    replies = [_ok_intent_reply(), _ok_intent_reply(), _ok_intent_reply()]
    client = _ladder_client(replies, contract_ladder="full_feedback")
    calls = {"n": 0}

    def fake_commit(stage, slot, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return False, {"errors": ["§9.15 dangling signal nets"], "offenders": []}
        if calls["n"] == 2:
            return False, {"errors": ["§9.17 two-terminal self-short"], "offenders": []}
        return True, {"ok": True}

    monkeypatch.setattr(stage_driver_mod, "commit_stage", fake_commit)
    result = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert result["status"] == "ok"
    # The third correction carries BOTH gate errors, not only the latest rung's.
    feedback = client.calls[2]["messages"][-1]["content"]
    assert "ALL BLOCKING DEFECTS THIS STAGE HAS REPORTED" in feedback
    assert "§9.15 dangling signal nets" in feedback
    assert "§9.17 two-terminal self-short" in feedback



def test_an_aggregate_refusal_is_identified_by_its_member_defects():
    """A refusal naming several defects compares by its members, whichever shape wrote them.

    `_derive_intent_payload` stores an aggregate's members under `findings`; artifacts written
    before that field existed carry the same rows in `evidence` as bare dicts. The rejection
    signature, the per-class counts and the ``full_feedback`` text all read through
    `_diagnostic_member_rows`, so the two shapes must give one identity -- otherwise a *changed*
    defect set looks like a repeated one and the `signature` arm stops correcting exactly when
    the draft is making progress (live runs KC-HPD3YF, KC-P4E2PH).
    """
    from kicraft.server.stage_runtime import (
        _diagnostic_codes,
        _diagnostic_member_rows,
        _schema_rejection_signature,
    )

    members = [
        {"code": "unknown_part_refused", "message": "requirement 'jack' (dc005) declares no interface"},
        {"code": "unsupported_lowerer_contract", "message": "led does not implement 'LTST-C190KGKT'"},
    ]
    written_now = {
        "code": "multiple_intent_contracts",
        "severity": "repair_required",
        "message": "two defects",
        "evidence": [f"{m['code']}: {m['message']}" for m in members],
        "findings": members,
    }
    written_before = {
        "code": "multiple_intent_contracts",
        "message": "two defects",
        "evidence": members,
    }
    for row in (written_now, written_before):
        assert _diagnostic_codes(row) == [
            "multiple_intent_contracts",
            "unknown_part_refused",
            "unsupported_lowerer_contract",
        ]
    assert _schema_rejection_signature("contract_rejected", written_now) == (
        _schema_rejection_signature("contract_rejected", written_before)
    )

    # One defect traded for another is progress, and the identity has to say so.
    changed = {
        "code": "multiple_intent_contracts",
        "severity": "repair_required",
        "message": "one defect",
        "findings": [
            {"code": "conflicting_port_binding", "message": "port 'gnd' of 'reg' already bound"},
        ],
    }
    assert _schema_rejection_signature("contract_rejected", changed) != (
        _schema_rejection_signature("contract_rejected", written_now)
    )
    assert len(_diagnostic_member_rows(written_now)) == 3


def test_unstated_dc_input_gets_a_defaulted_screw_terminal(tmp_path):
    """A supply voltage with no entry path is completed before the candidate is reviewed.

    The live product auto-answers the question rather than parking, so the walkthrough has to
    see the same outcome: one reviewed 2-position screw terminal, with the assumption that says
    it was defaulted, and no repair round spent on the omission.
    """
    brief = (
        "An ESP32-C3 module actuator driver: a DRV8833 dual H-bridge, an 18 V DC input, "
        "two JST-XH connectors, and a secondary status LED."
    )
    intent = {
        "goal": brief,
        "constraints": ["18 V DC input", "two JST-XH connectors"],
        "named_parts": ["ESP32-C3", "DRV8833", "JST-XH"],
        "inferred_expertise": "intermediate",
        "assumptions": [],
        "obligations": [
            {
                "kind": "physical",
                "original_obligation_id": "mcu",
                "component_class": "microcontroller",
            },
            {
                "kind": "physical",
                "original_obligation_id": "jst",
                "component_class": "jst-xh-connector",
            },
            {
                "kind": "quantity",
                "original_obligation_id": "jst_count",
                "subject": "JST-XH connectors",
                "minimum": 2,
            },
            {
                "kind": "quantitative",
                "original_obligation_id": "vin",
                "quantity": "input voltage",
                "relation": "equal",
                "value": 18.0,
                "unit": "V DC",
            },
        ],
        "project_stem": "ESP32_ACTUATOR_DRIVER",
    }
    client = _ScriptedClient(
        [{"text": json.dumps(intent), "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0}]
    )
    client.s = Settings(api_key="test")

    result = stage_driver_mod.drive_stage(
        client,
        "intent",
        brief,
        tmp_path / ".kicraft" / "state.json",
        tmp_path,
        review_before_commit=True,
    )

    slot = result["slot"]
    assert [row["component_class"] for row in slot["obligations"] if row["kind"] == "physical"] == [
        "microcontroller",
        "jst-xh-connector",
        "screw-terminal",
    ]
    assert slot["assumptions"] == [
        "Power input: 2-position screw terminal for the 18 V DC supply (defaulted)"
    ]
    assert result["attempts"] == 1  # the bound count and the completed entry need no repair
    assert result["diagnostics"] == []


def _header_board(rails: dict) -> dict:
    """A minimal architecture answer that derives, for contract-refusal loop tests."""
    return {
        "power": {"rails": rails},
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


def test_a_changing_design_defect_earns_bounded_preserving_corrections(tmp_path):
    """A design-contract refusal is a design defect: keep the draft while the defect set changes.

    Stock behaviour answered the first refusal with one serialization pass and one from-scratch
    rewrite, then failed the stage outright — which is how four live architecture drafts died
    without ever producing a candidate (seed-37 walkthrough). A *changed* defect set is progress,
    the same rule the commit path and the optional ladder already used, so it now earns up to two
    preserving corrections on its own. An unchanged defect set still terminates immediately.
    """
    def reply(payload):
        return {
            "text": json.dumps(payload),
            "reasoning": "",
            "finish_reason": "stop",
            "cost_usd": 0.0,
        }

    defects = [
        _header_board({"VBUS": {"voltage": 5.0, "from": "ghost.vbus"}}),
        _header_board({"VBUS": {"voltage": 5.0, "from": "missing_rail.vbus"}}),
        _header_board({"VBUS": {"voltage": 5.0, "from": "phantom.vbus"}}),
        _header_board({"VBUS": {"voltage": 5.0, "from": "absent_rail.vbus"}}),
    ]
    good = _header_board({"VBUS": {"voltage": 5.0, "from": None}})

    client = _ScriptedClient([reply(payload) for payload in [*defects, good]])
    client.s = Settings(api_key="test")
    state_path = tmp_path / ".kicraft" / "state.json"

    result = stage_driver_mod.drive_stage(
        client, "architecture", "a header breakout", state_path, tmp_path
    )

    assert result["commit_ok"] is True, result
    assert len(client.calls) == 5  # normal, serialization, escape, then two design-contract rounds
    assert json.loads(state_path.read_text())["architecture"] is not None


def test_a_repeated_design_defect_stays_terminal(tmp_path):
    """No progress, no extra rounds: the same defect twice ends the stage as before."""
    def reply(payload):
        return {
            "text": json.dumps(payload),
            "reasoning": "",
            "finish_reason": "stop",
            "cost_usd": 0.0,
        }

    same = _header_board({"VBUS": {"voltage": 5.0, "from": "ghost.vbus"}})
    client = _ScriptedClient([reply(same) for _ in range(4)])
    client.s = Settings(api_key="test")

    result = stage_driver_mod.drive_stage(
        client, "architecture", "a header breakout", tmp_path / ".kicraft" / "state.json", tmp_path
    )

    assert result["commit_ok"] is False
    assert len(client.calls) == 3  # unchanged defect set: nothing is earned
