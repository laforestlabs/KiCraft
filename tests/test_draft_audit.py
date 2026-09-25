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
    assert "part_choice_driver" in keys       # ... with a suggested alternative
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
