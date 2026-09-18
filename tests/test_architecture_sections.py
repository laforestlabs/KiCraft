"""The architecture stage's section rung (Phase A of the design-completion follow-up).

The architecture stage used to answer one large document for ~thirty rules, and its refusals were
interactions between sections: a repair that fixed the contact list broke the net binding. These
tests drive the real stage driver with a scripted provider and assert the observable contract:

- a whole-slot answer that validates is still one call, unchanged (the reference replay's path);
- a refused whole-slot answer is answered section by section, with the section's own small contract,
  and the assembled document is what commits;
- a section answer that does not satisfy its own refusal is re-asked alone, not the whole document;
- an unparseable answer is rebuilt from the sections in dependency order.
"""

from __future__ import annotations

import json

from kicraft.server.config import Settings

from test_architecture_intent import _half_usb_pair_intent, _hub75_intent
from test_stage_driver_retry import _OK_INTENT, _ScriptedClient


def _reply(payload) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "text": text,
        "reasoning": "",
        "finish_reason": "stop",
        "cost_usd": 0.0,
    }


def _section_names(client) -> list[str]:
    """The architecture calls' contract names: the intent draft is the other stage's call."""
    return [
        call["response_format"]["json_schema"]["name"]
        for call in client.calls
        if call["response_format"]["json_schema"]["name"].startswith("kicraft_architecture")
    ]


def _driver(tmp_path, replies: list[dict]):
    """One intent + architecture drive over the real stage driver, provider scripted."""
    from kicraft.server.session import run_session

    client = _ScriptedClient([_reply(json.loads(_OK_INTENT)), *replies])
    client.s = Settings(api_key="test", architecture_sections=True)
    events: list[dict] = []
    result = run_session(
        tmp_path,
        "a USB-C ESP32-S3 HUB75 controller",
        ["intent", "architecture"],
        client=client,
        progress=events.append,
    )
    return client, events, next(
        row for row in result["results"] if row["stage"] == "architecture"
    )


def test_the_section_rung_is_off_unless_the_operator_enables_it(tmp_path):
    """The measured cost increase is not paid by default: a refusal takes the ordinary ladder."""
    from kicraft.server.session import run_session

    client = _ScriptedClient(
        [
            _reply(json.loads(_OK_INTENT)),
            _reply(_half_usb_pair_intent()),
            _reply(_hub75_intent()),
        ]
    )
    client.s = Settings(api_key="test")  # architecture_sections defaults to False
    result = run_session(
        tmp_path,
        "a USB-C ESP32-S3 HUB75 controller",
        ["intent", "architecture"],
        client=client,
    )
    architecture = next(row for row in result["results"] if row["stage"] == "architecture")
    assert architecture["commit_ok"] is True
    # Two whole-slot calls (draft, corrected draft) — never a section contract.
    assert _section_names(client) == [
        "kicraft_architecture_response_v2",
        "kicraft_architecture_response_v2",
    ]


def _fixed_signals() -> list[dict]:
    """The reference board's signals, with the half USB pair completed onto the socket.

    `USB_D_N` reaches `edge:USB_DATA` beside `USB_D_P`, and the test header the broken draft wired
    it to keeps a data-line net of its own: the draft's second sheet is not what the refusal was
    about, so the section answer must not orphan it.
    """
    signals = [dict(row) for row in _hub75_intent()["signals"]]
    signals.append({"name": "USB_DATA_TEST", "from": "esp32.usb_dm", "to": "usb_test.pin1"})
    return signals


def test_whole_slot_answer_still_commits_in_one_call(tmp_path):
    """The fast path is untouched: a valid whole-slot draft never reaches the section rung."""
    client, events, architecture = _driver(tmp_path, [_reply(_hub75_intent())])

    assert architecture["commit_ok"] is True, architecture
    assert architecture["attempts"] == 1
    assert _section_names(client) == ["kicraft_architecture_response_v2"]
    assert [event["kind"] for event in events if event["kind"] == "architecture_section"] == []


def test_refused_whole_slot_answer_is_answered_by_the_section_the_refusal_names(tmp_path):
    """One refusal, one small section call, and the merged document is what commits."""
    client, events, architecture = _driver(
        tmp_path,
        [
            _reply(_half_usb_pair_intent()),
            _reply({"signals": _fixed_signals()}),
        ],
    )

    assert architecture["commit_ok"] is True, architecture
    assert architecture["attempts"] == 2
    assert _section_names(client) == [
        "kicraft_architecture_response_v2",
        "kicraft_architecture_signals_response_v1",
    ]
    section_call = client.calls[-1]
    # The section call is scoped: its own section spec and schema, the refusal to fix, and the
    # already-fixed requirement ids the signals may reference.
    system = section_call["messages"][0]["content"]
    assert "Draft ONLY the 'signals' section" in system
    assert "Architecture section: cross-part wiring." in system
    assert "boards" not in system  # the whole-slot spec is not re-sent
    user = section_call["messages"][-1]["content"]
    assert "incomplete_usb_edge" in user
    assert '"id":"esp32"' in user
    assert section_call["response_format"]["json_schema"]["name"].endswith("_signals_response_v1")
    # The measurement instrument still names the class the reader refused.
    done = next(
        event
        for event in events
        if event.get("kind") == "stage_done" and event.get("stage") == "architecture"
    )
    assert done["defect_codes"] == ["incomplete_usb_edge"]
    assert done["contract_rejections"] == 1
    assert done["first_draft_accepted"] is False
    section_events = [event for event in events if event["kind"] == "architecture_section"]
    assert [(event["section"], event["accepted"]) for event in section_events] == [("signals", True)]
    # The committed design is the derived reference board with the section's repair in it.
    committed = json.loads((tmp_path / ".kicraft" / "state.json").read_text(encoding="utf-8"))
    assert architecture["slot"]["inter_sheet_nets"] == committed["architecture"]["inter_sheet_nets"]


def test_section_answer_that_does_not_fix_its_refusal_is_asked_again_alone(tmp_path):
    """A useless section answer is repaired as that section, never by re-asking the whole slot."""
    client, events, architecture = _driver(
        tmp_path,
        [
            _reply(_half_usb_pair_intent()),
            # The same incomplete pair again: the merge changes nothing, so the refusal stands.
            _reply({"signals": _half_usb_pair_intent()["signals"]}),
            _reply({"signals": _fixed_signals()}),
        ],
    )

    assert architecture["commit_ok"] is True, architecture
    assert architecture["attempts"] == 3
    assert _section_names(client) == [
        "kicraft_architecture_response_v2",
        "kicraft_architecture_signals_response_v1",
        "kicraft_architecture_signals_response_v1",
    ]
    assert [event["section"] for event in events if event["kind"] == "architecture_section"] == [
        "signals",
        "signals",
    ]
    second = client.calls[2]
    assert "incomplete_usb_edge" in second["messages"][-1]["content"]


def test_unparseable_answer_is_rebuilt_from_the_sections_in_dependency_order(tmp_path):
    """A truncated whole-slot answer falls back to the four small answers, assembled as one doc."""
    missing = _hub75_intent()
    client, events, architecture = _driver(
        tmp_path,
        [
            _reply("{not json"),
            _reply({key: missing[key] for key in ("topologies", "comms_protocols", "mcu_present", "power", "sheets", "assumptions")}),
            _reply({"requirements": missing["requirements"]}),
            _reply({"signals": missing["signals"]}),
        ],
    )

    assert architecture["commit_ok"] is True, architecture
    assert _section_names(client) == [
        "kicraft_architecture_response_v2",
        "kicraft_architecture_sheets_response_v1",
        "kicraft_architecture_parts_response_v1",
        "kicraft_architecture_signals_response_v1",
    ]
    assert [event["section"] for event in events if event["kind"] == "architecture_section"] == [
        "sheets",
        "parts",
        "signals",
    ]
    committed = json.loads((tmp_path / ".kicraft" / "state.json").read_text(encoding="utf-8"))
    declared_sheets = [sheet["name"] for sheet in missing["sheets"]]
    committed_sheets = [sheet["name"] for sheet in committed["architecture"]["sheets"]]
    # The declared sheets come first, in order; the compiler then adds a sheet per `edge:` peer.
    assert committed_sheets[: len(declared_sheets)] == declared_sheets
    assert {row["id"] for row in missing["requirements"]} <= {
        row["id"] for row in committed["architecture"]["requirements"]
    }


def test_section_selection_follows_the_refusal_codes_not_its_prose():
    """`declared_signal_port_tied` is a parts defect, whatever words the message contains.

    The measured bug this guards: the aggregate `multiple_intent_contracts` message names ports,
    nets and signals, so a keyword scan asked the signals section three times for a requirement's
    own `declared_ports` defect and never reached the section that owns it.
    """
    from kicraft.server.stage_contracts import architecture_pending_sections

    assembled = {
        "sheets": [{"name": "POWER"}],
        "requirements": [{"id": "mcp23017"}],
        "signals": [{"name": "GPIO_0", "from": "a.b", "to": "c.d"}],
    }
    diagnostic = {
        "code": "multiple_intent_contracts",
        "message": "declared port 'vdd' of 'mcp23017' carries a signal; conflicting_port_binding",
        "evidence": [
            {"code": "declared_signal_port_tied"},
            {"code": "unbound_required_port"},
        ],
    }
    assert architecture_pending_sections(assembled, {}, diagnostic) == ("parts", "signals")
    # A code the table does not name is not guessed at: a complete answer with no owed section
    # stays off the rung even when its refusal text is full of section words.
    assert architecture_pending_sections(assembled, {}, {"code": "some_unknown_code"}) == ()
    assert (
        architecture_pending_sections(
            assembled, {}, {"code": "source_obligation_not_retained"}
        )
        == ("obligations",)
    )


def test_section_calls_stay_inside_the_stage_provider_budget(tmp_path):
    """A section that never answers spends the rung's bounded budget, not the drive's.

    The first callback is the whole slot; every later one is a section call. Nothing may exceed the
    stage budget, and the drive must still end with a terminal outcome rather than an endless rung.
    """
    from kicraft.server import stage_runtime

    replies = [_reply("{not json")] * (
        1 + stage_runtime._ARCHITECTURE_SECTION_CALL_BUDGET + 4
    )
    client, _events, architecture = _driver(tmp_path, replies)

    assert architecture["commit_ok"] is False
    section_calls = [
        name for name in _section_names(client) if name != "kicraft_architecture_response_v2"
    ]
    assert len(section_calls) <= stage_runtime._ARCHITECTURE_SECTION_CALL_BUDGET
    assert architecture["attempts"] <= (
        stage_runtime._STAGE_MIN_RETRIES["architecture"]
        + 1
        + stage_runtime._ARCHITECTURE_SECTION_CALL_BUDGET
    )
