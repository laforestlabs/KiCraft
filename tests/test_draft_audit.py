"""The draft audit: which questions a draft earns, and which answers become findings."""

from __future__ import annotations

from kicraft.server import draft_audit
from kicraft.server.decision_layer import Answer, DecisionUnavailable


def _draft(**overrides):
    draft = {
        "power": {"rails": {"+18V": {"voltage": 18.0, "from": "input.pin1"}}},
        "requirements": [
            {
                "id": "driver",
                "family": "dual-dc-motor-driver",
                "exact_part": "DRV8833PWPR",
                "supply": "+18V",
            },
            {"id": "jst_a", "family": "jst-xh-connector", "exact_part": None, "supply": None},
        ],
        "signals": [{"name": "USB_DM", "from": "mcu.usb_dm", "to": "edge:USB"}],
    }
    draft.update(overrides)
    return draft


def test_a_draft_earns_a_question_for_each_rule_it_touches():
    keys = [question.key for question in draft_audit.architecture_questions(_draft())]
    assert "usb_socket_rail" in keys          # a USB data edge
    assert "rated_driver" in keys             # 18 V rail on a 10.8 V part
    # No part-choice question for the driver: the catalogue holds no other motor driver, so the
    # deterministic refusal states the remedy instead of the audit asking a degenerate question.
    assert "part_choice_driver" not in keys
    # `jst-xh-connector` is a reviewed family, so the compiler builds it and the audit has nothing
    # to ask: asking anyway produced a confident "no carrier in catalogue" finding on a board the
    # pipeline resolves (58 such findings per measured campaign, 26 of 28 provably false).
    assert "identity_jst_a" not in keys
    assert "carrier_exists_jst_a" not in keys
    # A family the pipeline really cannot build still earns both questions.
    unknown = _draft()
    unknown["requirements"][1] = {
        "id": "jst_a",
        "family": "unobtainium-sensor",
        "exact_part": None,
        "supply": None,
    }
    unknown_keys = [question.key for question in draft_audit.architecture_questions(unknown)]
    assert "identity_jst_a" in unknown_keys
    assert "carrier_exists_jst_a" in unknown_keys
    # A draft that breaks nothing gets no questions, so the audit costs nothing.
    clean = {
        "power": {"rails": {"+3V3": {"voltage": 3.3, "from": "reg.output"}}},
        "requirements": [
            {"id": "reg", "family": "ap63203-3v3", "exact_part": "AP63203WU-7", "supply": None},
            {"id": "mcu", "family": "esp32-c3-mini-1-module", "exact_part": "ESP32-C3-MINI-1-N4", "supply": "+3V3"},
        ],
        "signals": [{"name": "GPIO", "from": "mcu.io1", "to": "led.anode"}],
    }
    assert draft_audit.architecture_questions(clean) == []


def test_findings_follow_the_answers_and_only_above_the_threshold(monkeypatch):
    scripted = [
        Answer(key="rated_driver", kind="noul", value=False, confidence=0.9, probability=0.1),
        Answer(key="carrier_exists_jst_a", kind="noul", value=False, confidence=0.8, probability=0.2),
        Answer(
            key="usb_socket_rail",
            kind="noul",
            value=False,
            confidence=0.4,  # below the threshold: a guess is not reported
            probability=0.6,
        ),
    ]
    monkeypatch.setattr(draft_audit, "decide", lambda *a, **k: scripted)

    codes = [finding.code for finding in draft_audit.audit_architecture(_draft())]

    # The carrier answer is a claim about catalogue coverage the compiler settles itself: the
    # reviewed JST-XH family builds `jst_a`, so no finding is reported whatever the auditor says.
    assert codes == ["audit_part_over_rating"]

    unknown = _draft()
    unknown["requirements"][1] = {
        "id": "jst_a",
        "family": "unobtainium-sensor",
        "exact_part": None,
        "supply": None,
    }
    assert [f.code for f in draft_audit.audit_architecture(unknown)] == [
        "audit_part_over_rating",
        "audit_no_carrier_in_catalogue",
    ]


def test_an_unavailable_auditor_yields_no_findings(monkeypatch):
    def boom(*args, **kwargs):
        raise DecisionUnavailable("no key")

    monkeypatch.setattr(draft_audit, "decide", boom)
    assert draft_audit.audit_architecture(_draft()) == []


def test_driver_runs_the_audit_and_folds_its_findings_into_the_round(tmp_path, monkeypatch):
    """A Jev finding reaches the correction round; an unavailable auditor is fail-soft."""
    import json

    from kicraft.server import stage_runtime
    from kicraft.server.config import Settings
    from kicraft.server.decision_layer import Answer

    finding = stage_runtime.models.StageDiagnostic(
        code="audit_part_over_rating",
        severity="repair_required",
        message="the driver runs above its part rating",
        evidence=["confidence=0.90", "full evidence " + "x" * 180],
    )

    def fake_audit(candidate, **kwargs):
        assert kwargs["stage"] == "architecture"
        assert kwargs["guard"] is not None
        return [finding]

    monkeypatch.setattr("kicraft.server.draft_audit.audit_architecture", fake_audit)

    class _Guard:
        pass

    class _Client:
        def __init__(self, replies):
            self.replies = list(replies)
            self.calls = []
            self.guard = _Guard()
            self.s = Settings(api_key="test")

        def chat(self, messages=None, **kwargs):
            self.calls.append({**kwargs, "messages": messages})
            return {
                "text": json.dumps(self.replies.pop(0)),
                "reasoning": "",
                "finish_reason": "stop",
                "cost_usd": 0.0,
            }

    draft = {
        "power": {"rails": {"+3V3": {"voltage": 3.3, "from": None}}},
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
    # The finding is `repair_required`, so the driver asks for a correction round: one draft
    # per scheduled call, and the audit runs on each candidate it sees.
    client = _Client([draft] * 4)

    result = stage_runtime.drive_stage(
        client, "architecture", "a header breakout", tmp_path / ".kicraft/state.json", tmp_path
    )

    assert any(f["code"] == "audit_part_over_rating" for f in result["diagnostics"])
    correction = client.calls[1]["messages"]
    assert any(m["role"] == "assistant" and json.loads(m["content"]) == draft for m in correction)
    assert "full evidence " + "x" * 180 in correction[-1]["content"]


def test_part_alternatives_are_offered_only_when_the_catalogue_has_them():
    from kicraft.server.draft_audit import part_alternatives

    # A class spelling resolves to its reviewed parts, so real alternatives can be offered.
    leds = part_alternatives("status-led", "ltst-c190kgkt")
    assert leds and "ltst-c190kgkt" not in leds

    # A curated recipe family owns its parts; nothing else can stand in for it.
    assert part_alternatives("dual-dc-motor-driver", "DRV8833PWPR") == ()


def test_construction_refusal_is_not_audited_and_replacement_is(tmp_path, monkeypatch):
    import copy
    import json

    from kicraft.server import stage_runtime
    from kicraft.server.config import Settings

    controls = ("enable_n", "ms1", "ms2", "ms3", "reset_n", "sleep_n", "step", "dir")
    coils = ("coil_1a", "coil_1b", "coil_2a", "coil_2b")
    bad = {
        "power": {"rails": {"+3V3": {"voltage": 3.3, "from": None},
                             "VMOTOR": {"voltage": 12.0, "from": None}}},
        "sheets": [{"name": "MAIN", "stem": "MAIN", "role": "analog_block",
                    "function": "Stepper driver and interfaces"}],
        "requirements": [
            {"id": "driver", "sheet": "MAIN", "role": "driver",
             "family": "a4988-driver-socket", "ties": {"vdd": "+3V3", "gnd": "GND"},
             "functional_blocks": []},
            {"id": "controls", "sheet": "MAIN", "role": "connector", "family": "pin-header",
             "parameters": {"rows": 1, "gender": "male"}, "functional_blocks": []},
            {"id": "motor", "sheet": "MAIN", "role": "connector", "family": "pin-header",
             "parameters": {"rows": 1, "gender": "male"}, "functional_blocks": []},
        ],
        "signals": [
            *[{"name": port.upper(), "from": f"controls.pin{index}", "to": f"driver.{port}"}
              for index, port in enumerate(controls, 1)],
            *[{"name": port.upper(), "from": f"driver.{port}", "to": f"motor.pin{index}"}
              for index, port in enumerate(coils, 1)],
        ],
    }
    good = copy.deepcopy(bad)
    good["requirements"][0]["ties"]["vmotor"] = "VMOTOR"
    seen = []

    def audit(candidate, **kwargs):
        assert kwargs["reserve_call"]()
        seen.append(copy.deepcopy(candidate))
        return []

    monkeypatch.setattr("kicraft.server.draft_audit.audit_architecture", audit)

    class Client:
        def __init__(self):
            self.replies = [bad, good]
            self.calls = []
            self.s = Settings(api_key="test", stage_semantics="observe")

        def chat(self, messages=None, **kwargs):
            self.calls.append({"messages": messages, **kwargs})
            return {"text": json.dumps(self.replies.pop(0)), "reasoning": "",
                    "finish_reason": "stop", "cost_usd": 0.0}

    client = Client()
    result = stage_runtime.drive_stage(
        client, "architecture", "a stepper driver", tmp_path / ".kicraft/state.json",
        tmp_path, review_before_commit=True,
    )
    assert result.get("needs_review")
    assert result["attempts"] == 3
    assert seen == [good]
    correction = client.calls[1]["messages"]
    assert any(m["role"] == "assistant" and json.loads(m["content"]) == bad for m in correction)
    assert "vmotor" in correction[-1]["content"].lower()
