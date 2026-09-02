"""Tests for the stage driver's self-correction feedback and per-stage retries.

Pure functions, no OpenRouter / network: exercises the retry-message construction
and the per-stage retry budget that help the wiring stage converge.
"""

from __future__ import annotations

import json

import pytest
import requests

from kicraft.server import stage_runtime as stage_driver_mod
from kicraft.server.config import Settings
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


def test_retry_feedback_explains_series_path_for_dangling_terminal():
    msg = _retry_feedback(
        {
            "ok": False,
            "errors": ["9.15 no dangling signal nets: 1 signal net wires a single pin"],
            "offenders": ["net 'CAN_TX_MCU' on sheet 'CAN TRANSCEIVER' wires only R3.2"],
        },
        stage="wiring",
    )
    assert "source + Rn.1 = SIG_IN" in msg
    assert "Rn.2 + destination = SIG_OUT" in msg
    assert "moving the destination pin" in msg


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
    assert _stage_max_retries("wiring", 2) >= 4  # wiring floors higher than default
    assert _stage_max_retries("intent", 2) == 2  # simple stages keep the default
    assert _stage_max_retries("functional_spec", 2) == 2


def test_caller_default_wins_when_higher_than_the_floor():
    assert _stage_max_retries("wiring", 6) == 6
    assert _stage_max_retries("intent", 6) == 6


def test_wiring_gets_a_larger_token_budget():
    assert _stage_max_tokens("wiring", 4096) >= 8192  # wiring floors higher
    assert _stage_max_tokens("intent", 4096) == 4096  # simple stages keep default
    assert _stage_max_tokens("wiring", 16000) == 16000  # a higher caller default wins


def test_bom_gets_more_retries_for_symbol_resolution():
    assert _stage_max_retries("bom", 2) >= 4  # bom floors higher now
    assert _stage_max_retries("architecture", 2) == 2


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


def test_bom_prompt_demands_decoupling_completeness():
    # The requirement rides the bom.md spec block (single source; the
    # _stage_extra restatement was deduped 2026-07-19 review §7.2).
    sysmsg = build_system("bom")
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
            }
        )
        return dict(self.replies.pop(0))

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
            }
        )
        r = dict(self.replies.pop(0))
        r.setdefault("rounds", 1)
        r.setdefault("tool_calls", 0)
        return r


def _ok_intent_reply():
    return {"text": _OK_INTENT, "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0}


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
    assert "collection must contain" not in retry_message
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


def test_second_truncated_serialization_result_terminates_as_truncated_json(tmp_path):
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
        ]
    )
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "failed"
    last = res["results"][-1]
    assert len(client.calls) == 2  # normal + ONE serialization, no more
    assert last["failure_kind"] == "truncated_json"  # classified by its own signature
    assert last["attempts"] == 2  # actual calls, not max_retries+1


def test_malformed_normal_stop_terminates_as_invalid_json_after_one_serialization(tmp_path):
    client = _ScriptedClient(
        [
            {"text": "not json at all", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
            {"text": "still not json", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
        ]
    )
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "failed"
    last = res["results"][-1]
    assert len(client.calls) == 2
    assert last["failure_kind"] == "invalid_json"
    assert last["attempts"] == 2


def test_serialization_schema_failure_terminates_without_commit_correction(tmp_path):
    # A parseable object that violates the requested provider schema is not a
    # commit candidate. The one schema-bound serialization recovery is terminal
    # when it repeats the schema defect.
    client = _ScriptedClient(
        [
            {
                "text": '{"goal": "x", truncated',
                "reasoning": "",
                "finish_reason": "length",
                "cost_usd": 0.0,
            },
            {"text": "{}", "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0},
        ]
    )
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "failed"
    last = res["results"][-1]
    assert len(client.calls) == 2
    assert sum(1 for call in client.calls if call["serialization"]) == 1
    assert last["failure_kind"] == "invalid_schema"


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
    assert len(client.calls) == 2
    assert client.calls[0]["serialization"] is False  # chat_with_tools (tool loop)
    assert client.calls[1]["serialization"] is True  # plain chat for serialization
    assert client.calls[1]["response_format"] is client.calls[0]["response_format"]
    assert client.calls[1]["max_tokens"] == 32768  # bom serialization cap
    assert client.calls[1]["reasoning"] == {"enabled": False}
    retry_message = client.calls[1]["messages"][-1]["content"]
    assert "about 16 characters" in retry_message
    assert client.calls[1]["response_format"]["json_schema"]["name"] == "kicraft_bom_response_v2"
    assert res["results"][-1]["failure_kind"] == "invalid_json"


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
        ]
    )
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    last = res["results"][-1]
    sp = tmp_path / ".kicraft" / "state.json"
    sj = json.loads(sp.read_text(encoding="utf-8"))
    entry = sj["stage_status"]["intent"]
    assert entry["failure_kind"] == "invalid_json"
    assert entry["attempts"] == last["attempts"] == 2


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


def test_second_collection_limit_is_terminal(tmp_path):
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
    client = _ScriptedClient([overflow, overflow])
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    last = res["results"][-1]
    assert last["failure_kind"] == "collection_limit"
    assert last["attempts"] == 2


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


def test_repeated_commit_rejection_gets_one_pristine_escape_then_stops(tmp_path, monkeypatch):
    rejected = {
        "ok": False,
        "errors": ["9.15 multi-net pin short"],
        "offenders": ["R1.1 on SIG_A and SIG_B"],
    }
    monkeypatch.setattr(stage_driver_mod, "commit_stage", lambda *args, **kwargs: (False, rejected))
    client = _ScriptedClient([_ok_intent_reply(), _ok_intent_reply(), _ok_intent_reply()])
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    last = res["results"][-1]
    assert last["failure_kind"] == "commit_rejected"
    assert last["attempts"] == 3
    assert client.calls[2]["reasoning"] == {"enabled": False}
    assert client.calls[2]["temperature"] == 0.4
    roles = [message["role"] for message in client.calls[2]["messages"]]
    assert roles == ["system", "user", "user"]
    assert all(
        call["response_format"] is client.calls[0]["response_format"] for call in client.calls
    )


def test_wiring_rejection_uses_complete_same_schema_correction(tmp_path, monkeypatch):
    state = {
        "architecture": {
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
    assert result["attempts"] == 2
    assert result["expanded_component_count"] == 0
    assert client.calls[0]["reasoning"] == {"enabled": False}
    assert all(
        call["response_format"]["json_schema"]["name"] == "kicraft_wiring_response_v2"
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
    assert len(pol.collection_bounds) == 1
    assert pol.collection_bounds[0].field == "groups"
    assert pol.collection_bounds[0].total == 500
    assert pol.collection_bounds[0].per_group == 450
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
                }
            ]
        }
    )
    result = stage_driver_mod.drive_stage(
        _ScriptedClient(
            [{"text": raw, "reasoning": "", "finish_reason": "stop", "cost_usd": 0.0}]
        ),
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
