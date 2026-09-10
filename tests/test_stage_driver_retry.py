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
    _classify_parse_failure,
    _commit_rejection_signature,
    _redacted_rejection_facts,
    _design_reasoning,
    _normalize_questions,
    _retry_feedback,
    _stage_max_retries,
    _stage_max_tokens,
)
from kicraft.server.stage_state_io import attach_questions as _attach_questions
from kicraft.server.session import run_session


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
    assert _stage_max_retries("architecture", 2) == 6


def test_bom_has_a_symbol_search_tool():
    names = {t["function"]["name"] for t in BOM_TOOLS}
    assert "search_symbols" in names  # discover, do not guess
    assert {"list_parts", "lookup_symbol", "lookup_lcsc_id", "add_part_from_lcsc"} <= names


def test_bom_has_a_footprint_search_tool():
    names = {t["function"]["name"] for t in BOM_TOOLS}
    assert "search_footprints" in names  # footprint discovery, do not guess
    assert "lookup_footprint" in names  # verify a footprint exists + pad count


# ---- clarifying questions -------------------------------------------------


def test_normalize_questions_shapes_and_drops_junk():
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
    demoted = _normalize_questions(
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
    assert [question["blocking"] for question in demoted] == [False, False]
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


def test_architecture_spec_declares_power_rails_are_not_sheets():
    # The architecture spec must resolve the power-block contradiction: a
    # functional-spec block whose category is `power` is a net, never a sheet.
    sysmsg = build_system("architecture")
    assert "power/ground NETS" in sysmsg
    assert "never emit a Sheet" in sysmsg


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
            "name": "CONTROLLER",
            "category": "process",
            "purpose": "Generates an amplified PWM audio output.",
        },
        {"name": "SPEAKER", "category": "drive", "purpose": "Drives the speaker output."},
        {"name": "POWER", "category": "power", "purpose": "Powers the controller"},
    ]
    connection = {
        "from_block": "CONTROLLER",
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
    }
    second_repair = {
        **first_repair,
        "connections": [
            {
                **connection,
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
    assert [item["code"] for item in functional["diagnostics"]] == [
        "functional_spec_premature_topology"
    ]
    assert len(client.calls) == 3
    assert any(
        event.get("kind") == "stage_diagnostic"
        and event.get("code") == "functional_spec_premature_topology"
        and event.get("attempt") == 1
        for event in events
    )


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
        design_profile="flash",
        model=str(DESIGN_PROFILES["flash"]["model"]),
        provider_fallback_profile="pro",
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
        DESIGN_PROFILES["flash"]["model"],
        DESIGN_PROFILES["pro"]["model"],
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
        "from": DESIGN_PROFILES["flash"]["model"],
        "to": DESIGN_PROFILES["pro"]["model"],
        "from_profile": "flash",
        "to_profile": "pro",
        "from_providers": DESIGN_PROFILES["flash"]["provider_order"],
        "to_providers": DESIGN_PROFILES["pro"]["provider_order"],
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
        design_profile="flash",
        model=str(DESIGN_PROFILES["flash"]["model"]),
        provider_fallback_profile="pro",
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
        design_profile="flash",
        model=str(DESIGN_PROFILES["flash"]["model"]),
        provider_fallback_profile="pro",
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
        design_profile="flash",
        model=str(DESIGN_PROFILES["flash"]["model"]),
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
            if self.s.design_profile == "pro":
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
        design_profile="flash",
        model=str(DESIGN_PROFILES["flash"]["model"]),
        provider_fallback_profile="pro",
    )
    with pytest.raises(BudgetExceeded):
        run_session(
            tmp_path,
            "a USB-powered LED",
            ["intent"],
            client=client,
        )
    assert [call["model"] for call in client.calls] == [
        DESIGN_PROFILES["flash"]["model"],
        DESIGN_PROFILES["pro"]["model"],
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
    retry_message = serial["messages"][-1]["content"]
    assert "about 23 characters" in retry_message
    assert "`constraints` collection must contain at most 64 items total" in retry_message
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
    assert "Field required" in client.calls[2]["messages"][-1]["content"]


def test_schema_recovery_reports_local_validation_error(tmp_path):
    client = _ScriptedClient(
        [
            {"text": "{}", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
            _ok_intent_reply(),
        ]
    )
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "ok"
    retry_message = client.calls[1]["messages"][-1]["content"]
    assert "valid JSON but failed KiCraft's local slot validation" in retry_message
    assert "Field required" in retry_message


def test_answers_survive_schema_recovery(tmp_path):
    client = _ScriptedClient(
        [
            {"text": "{}", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
            _ok_intent_reply(),
        ]
    )
    result = run_session(
        tmp_path,
        "a powered LED",
        ["intent"],
        client=client,
        answers=[{"text": "Supply voltage?", "answer": "12 V"}],
    )
    assert result["status"] == "ok"
    assert "Q: Supply voltage?\nA: 12 V" in client.calls[1]["messages"][1]["content"]


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
    assert len(client.calls) == 7
    assert client.calls[0]["serialization"] is False  # chat_with_tools (tool loop)
    assert client.calls[1]["serialization"] is True  # plain chat for serialization
    assert all(call["serialization"] is False for call in client.calls[2:])
    assert client.calls[1]["response_format"] is client.calls[0]["response_format"]
    assert client.calls[1]["max_tokens"] == 32768  # bom serialization cap
    assert client.calls[1]["reasoning"] == {"enabled": False}
    retry_message = client.calls[1]["messages"][-1]["content"]
    assert "about 16 characters" in retry_message
    assert client.calls[1]["response_format"]["json_schema"]["name"] == "kicraft_bom_response_v3"
    assert res["results"][-1]["failure_kind"] == "unit_repair_exhausted"


def test_typed_bom_lowerer_skips_provider_call(tmp_path, monkeypatch):
    state = {
        "architecture": {
            "topologies": {"INPUT": "8-pin header"},
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
                        **{f"d{index}": f"D{index}" for index in range(8)},
                        "gnd": "GND",
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
    assert len(commits[0]["parts"]) == 1
    provenance = [
        event for event in progress if event.get("source") == "deterministic_architecture_lowering"
    ]
    assert [event["kind"] for event in provenance] == [
        "work_unit_plan",
        "work_unit_done",
    ]


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
    assert all(
        part["recipe_id"] == "esp32-s3-mini-1-minimal@1" for part in committed["bom"]["parts"]
    )
    assert committed["bom"]["recipe_ownership"][0]["pins"]
    assert committed["wiring"]["connections"]
    assert committed["wiring"]["no_connect_pins"]
    recipe_events = [event for event in progress if event.get("kind") == "recipe_selected"]
    assert [event["stage"] for event in recipe_events] == ["bom", "wiring"]
    assert all(event["owned_call_count"] == 0 for event in recipe_events)
    assert all(
        event["resolution"][0]["requirement_id"] == "auto_esp32_s3_module"
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
    design_profile="flash",
    escalation_profile="pro",
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
    assert [row["design_profile"] for row in records] == ["flash", "pro"]
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
    client.s = Settings(api_key="test", design_profile="flash")
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
        design_profile="flash",
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
    assert len(commits) == 1


def test_invalid_wiring_unit_escalates_after_first_attempt(tmp_path, monkeypatch):
    state_path = _work_unit_state(tmp_path, monkeypatch)
    client = _unit_client([_unit_reply("R1", "WRONG"), _unit_reply("U1"), _unit_reply("R1")])
    client.s = replace(client.s, escalation_profile="pro")
    monkeypatch.setattr(
        stage_driver_mod,
        "commit_stage",
        lambda *args, **kwargs: (True, {"ok": True}),
    )

    result = stage_driver_mod.drive_stage(client, "wiring", "test", state_path, tmp_path)

    assert result["commit_ok"] is True
    assert [call["model"] for call in client.calls] == [
        str(DESIGN_PROFILES["flash"]["model"]),
        str(DESIGN_PROFILES["pro"]["model"]),
        str(DESIGN_PROFILES["flash"]["model"]),
    ]


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
        escalation_profile="pro",
        provider_fallback_profile="pro",
    )
    monkeypatch.setattr(
        stage_driver_mod,
        "commit_stage",
        lambda *args, **kwargs: (True, {"ok": True}),
    )

    result = stage_driver_mod.drive_stage(client, "wiring", "test", state_path, tmp_path)

    assert result["commit_ok"] is True
    assert [call["model"] for call in client.calls] == [
        str(DESIGN_PROFILES["flash"]["model"]),
        str(DESIGN_PROFILES["pro"]["model"]),
        str(DESIGN_PROFILES["flash"]["model"]),
        str(DESIGN_PROFILES["flash"]["model"]),
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
                    "parameters": {"supported_pdos": "9V,12V,20V"},
                    "ports": {"VBUS": "VBUS"},
                },
                {
                    "id": "usb_c_receptacle",
                    "sheet": "PD TRIGGER",
                    "role": "connector",
                    "family": "usb_c_receptacle",
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
                    _group_payload(
                        id="usb_c_receptacle",
                        reference_prefix="J",
                        quantity=1,
                        value="USB_C_Receptacle_HRO_TYPE-C-31-M-12",
                        symbol="Connector_USB:USB_C_Receptacle_HRO_TYPE-C-31-M-12",
                        footprint=("Connector_USB:USB_C_Receptacle_HRO_TYPE-C-31-M-12"),
                        sheet="PD TRIGGER",
                    ),
                ],
            }
        ),
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }
    usb_candidate = {
        "text": json.dumps(
            {
                "groups": [json.loads(protected_candidate["text"])["groups"][1]],
                "arrays": [],
                "assumptions": [],
                "substitutions": [],
            }
        ),
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }
    collection_limit = {
        "text": "{}",
        "finish_reason": "collection_limit",
        "cost_usd": 0.0,
    }
    client = _unit_client(
        [question, collection_limit, collection_limit, protected_candidate, usb_candidate]
    )
    client.s = replace(client.s, escalation_profile="pro")
    committed = []

    def commit(_stage, slot, *args, **kwargs):
        committed.append(slot)
        return True, {"ok": True}

    monkeypatch.setattr(stage_driver_mod, "commit_stage", commit)

    result = stage_driver_mod.drive_stage(client, "bom", "USB-C PD trigger", state_path, tmp_path)

    assert result["commit_ok"] is True, result.get("error") or result
    assert result.get("needs_input") is not True
    assert len(client.calls) == 5
    feedback = client.calls[1]["messages"][-1]["content"]
    assert "committed architecture is binding" in feedback
    assert "requirement_ids=['pd_trigger_controller']" in feedback
    assert "do not emit components owned by another requirement" in feedback
    assert {part["value"] for part in committed[0]["parts"]} == {
        "HUSB238",
        "USB_C_Receptacle_HRO_TYPE-C-31-M-12",
    }
    assert [call["collection_bounds"][0].total for call in client.calls] == [32] * 5
    assert [call["serialization"] for call in client.calls] == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert [call["model"] for call in client.calls] == [str(DESIGN_PROFILES["pro"]["model"])] * 5


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
