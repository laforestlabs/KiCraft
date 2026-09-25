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
    assert "identity_jst_a" in keys           # a class no curated family carries
    assert "carrier_exists_jst_a" in keys
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

    assert codes == ["audit_part_over_rating", "audit_no_carrier_in_catalogue"]


def test_an_unavailable_auditor_yields_no_findings(monkeypatch):
    def boom(*args, **kwargs):
        raise DecisionUnavailable("no key")

    monkeypatch.setattr(draft_audit, "decide", boom)
    assert draft_audit.audit_architecture(_draft()) == []


def test_driver_runs_the_audit_and_folds_its_findings_into_the_round(tmp_path, monkeypatch):
    """A Jev finding reaches the correction round, and its call is billed.

    The audit exists so a rule the compiler would refuse outright is instead stated to the
    corrector, in the same round as the deterministic findings. It is fail-soft: an unreachable
    auditor contributes nothing.
    """
    import json

    from kicraft.server import stage_runtime
    from kicraft.server.config import Settings
    from kicraft.server.decision_layer import Answer

    finding = stage_runtime.models.StageDiagnostic(
        code="audit_part_over_rating",
        severity="repair_required",
        message="the driver runs above its part rating",
        evidence=["confidence=0.90"],
    )
    billed: list[tuple] = []

    def fake_audit(candidate, *, model, confidence, recorder):
        recorder({"input_tokens": 100, "output_tokens": 20, "cost": 0.0001})
        return [finding]

    monkeypatch.setattr("kicraft.server.draft_audit.audit_architecture", fake_audit)

    class _Guard:
        def record(self, model, input_tokens, output_tokens, cost_usd, meta=""):
            billed.append((model, cost_usd))

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
    assert billed, "the audit's call was never billed"
    assert {model for model, _cost in billed} == {"typesafe/jev-1.13"}
    assert {cost for _model, cost in billed} == {0.0001}


def test_part_alternatives_are_offered_only_when_the_catalogue_has_them():
    from kicraft.server.draft_audit import part_alternatives

    # A class spelling resolves to its reviewed parts, so real alternatives can be offered.
    leds = part_alternatives("status-led", "ltst-c190kgkt")
    assert leds and "ltst-c190kgkt" not in leds

    # A curated recipe family owns its parts; nothing else can stand in for it.
    assert part_alternatives("dual-dc-motor-driver", "DRV8833PWPR") == ()


def test_a_draft_the_compiler_refuses_is_still_audited(tmp_path, monkeypatch):
    """The audit sees the parsed draft *before* the derivation, so a refusal cannot hide it.

    The derivation runs inside decode, so an audit placed after it never saw the drafts that
    actually fail (measured on the seed-37 walkthrough). Here the compiler refuses the draft's
    rail reference, and the audit still runs and still reaches the correction message.
    """
    import json

    from kicraft.server import stage_runtime
    from kicraft.server.config import Settings

    finding = stage_runtime.models.StageDiagnostic(
        code="audit_identity_unresolved",
        severity="repair_required",
        message="Requirement 'mcu' does not name a curated family",
        evidence=["set the requirement's family or exact_part to 'esp32-c3-mini-1-module'"],
    )
    seen: list[dict] = []

    def fake_pre_audit(parsed):
        seen.append(parsed)
        return [finding]

    monkeypatch.setattr(stage_runtime, "_pre_audit_hook", lambda client: fake_pre_audit)

    class _Client:
        def __init__(self):
            self.replies = [refused_draft] * 4
            self.calls = []
            self.s = Settings(api_key="test")

        def chat(self, messages=None, **kwargs):
            self.calls.append({"messages": messages, **kwargs})
            return {
                "text": json.dumps(self.replies.pop(0)),
                "reasoning": "",
                "finish_reason": "stop",
                "cost_usd": 0.0,
            }

    refused_draft = {
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
    client = _Client()

    result = stage_runtime.drive_stage(
        client, "architecture", "a header breakout", tmp_path / ".kicraft/state.json", tmp_path
    )

    assert seen, "the audit never saw the parsed draft"
    assert result["failure_kind"] == "contract_rejected"
    correction = client.calls[1]["messages"][-1]["content"]
    assert "audit_identity_unresolved" in correction
    assert "A Jev audit of this same draft" in correction
