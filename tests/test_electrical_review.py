"""Tests for the Layer-3 LLM electrical-review module (offline, fake client)."""
from __future__ import annotations

import json

import kicraft.design.synthesis.symbol_pinout as _sp
from kicraft.design.models import (
    BOM,
    BomPart,
    ConversationState,
    IntentSlot,
    NetConnection,
    PinEndpoint,
)
from kicraft.design.synthesis.electrical_review import (
    _categorize,
    build_design_digest,
    clamp_findings,
    has_blocker,
    review_design,
    review_design_corroborated,
)


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.last_meta = None

    def chat(self, messages, **kw):
        text = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        self.last_meta = kw.get("meta_ctx")
        self.last_reasoning = kw.get("reasoning")
        self.last_model = kw.get("model")
        return {"text": text, "cost_usd": 0.002}


def _fake_lookup(pinmap):
    from kicraft.design.synthesis.symbol_pinout import SymbolNotFoundError

    def _lookup(lib_id, *a, **k):
        if lib_id not in pinmap:
            raise SymbolNotFoundError(lib_id)
        return {"symbol": lib_id, "unit_count": 1,
                "pins": [{"number": n, "name": nm, "electrical_type": "passive"}
                         for n, nm in pinmap[lib_id]]}

    return _lookup


def _state():
    return ConversationState(
        intent=IntentSlot(goal="A K-type thermocouple amp", constraints=["SPI output"],
                          named_parts=["MAX31855"]),
        bom=BOM(
            parts=[BomPart(ref="U1", value="MAX31855", symbol="Fake:MAX31855",
                           footprint="Package_SO:SOIC-8", sheet="AMP")],
            connections=[
                NetConnection(net_name="+3V3", sheet="AMP",
                              endpoints=[PinEndpoint(ref="U1", pin="1")]),
                NetConnection(net_name="GND", sheet="AMP",
                              endpoints=[PinEndpoint(ref="U1", pin="2")]),
                NetConnection(net_name="SPI_SCK", sheet="AMP",
                              endpoints=[PinEndpoint(ref="U1", pin="5")]),
            ],
        ),
    )


def test_digest_uses_pin_function_names(monkeypatch):
    monkeypatch.setattr(_sp, "lookup_pins", _fake_lookup({"Fake:MAX31855": [
        ("1", "VCC"), ("2", "GND"), ("5", "SCK")]}))
    digest = build_design_digest(_state())
    assert "GOAL: A K-type thermocouple amp" in digest
    assert "MAX31855" in digest          # named part surfaced
    assert "NETLIST" in digest
    assert "U1.1(VCC)" in digest         # pin FUNCTION name, not just the number
    assert "U1.5(SCK)" in digest
    # geometry must never leak in
    assert "position" not in digest.lower()


def test_review_parses_valid_findings():
    reply = json.dumps({"findings": [
        {"severity": "Blocker", "area": "current-limit",
         "issue": "U1 SENSE tied to GND -- no current limit", "suggestion": "add Rsense"},
        {"severity": "note", "area": "layout", "issue": "minor", "suggestion": ""},
    ]})
    r = review_design(FakeClient(reply), "digest")
    assert r["ok"] and r["error"] is None
    assert len(r["findings"]) == 2
    assert r["findings"][0]["severity"] == "blocker"   # blocker-eligible, kept
    assert r["findings"][0]["category"] == "current-limit"
    assert r["findings"][0]["refs"] == ["U1"]          # refdes recovered from issue
    assert has_blocker(r["findings"])
    assert round(r["cost_usd"], 4) == 0.002


def test_clamp_demotes_warning_max_category():
    # KC-PN2YUC: a decoupling-SIZING critique the model called a blocker -> warning.
    reply = json.dumps({"findings": [
        {"severity": "blocker", "area": "decoupling",
         "issue": "only 2x 100nF for 45 LEDs", "suggestion": "add caps"}]})
    r = review_design(FakeClient(reply), "digest")
    f = r["findings"][0]
    assert f["severity"] == "warning" and f["severity_raw"] == "blocker"
    assert f["clamped"] is True and f["category"] == "other"
    assert not has_blocker(r["findings"])


def test_categorize_blocker_eligible_phrasings():
    cases = {
        ("R2R_ladder", "R-2R inputs drive the ladder nodes directly"): "ladder-topology",
        ("regulator-feedback", "feedback divider R1/R2 sets 5.08V not 3.3V"): "regulator-feedback",
        ("current-sense", "SENSE1 tied to GND with no current-sense resistor"): "current-limit",
        ("programming", "no firmware-flash path; cannot be programmed"): "programming-path",
        ("isolation", "signal isolation compromised across the optocouplers"): "isolation",
        ("input-connector", "control inputs have no input connector"): "missing-input",
    }
    for (area, issue), cat in cases.items():
        assert _categorize(area, issue) == cat, (area, issue)


def test_categorize_margin_intent_defaults_to_other():
    # Every known over-block class must fall to 'other' (warning ceiling).
    for area, issue in [
        ("decoupling", "only 2x 100nF for 45 LEDs"),
        ("protection", "no TVS / no input protection on the exposed line"),
        ("intent-mismatch", "screw terminals instead of binding posts"),
        ("input-range", "Vin 5V below the 5.5V minimum input"),
        ("thermal", "regulator runs hot at full load"),
        ("crystal-load", "32MHz load caps may be low"),
        ("overvoltage", "VDD at 6V exceeds the 5.5V max"),
        ("value-tolerance", "1% resistor where 5% suffices"),
    ]:
        assert _categorize(area, issue) == "other", (area, issue)


def test_clamp_is_additive_and_pure():
    src = [{"severity": "blocker", "area": "decoupling", "issue": "x", "suggestion": "y"}]
    out = clamp_findings(src)
    assert src[0]["severity"] == "blocker"             # input not mutated
    assert out[0]["severity"] == "warning"
    assert {"category", "refs", "severity_raw", "clamped"} <= out[0].keys()


def test_review_empty_findings_is_sound():
    r = review_design(FakeClient(json.dumps({"findings": []})), "digest")
    assert r["ok"] and r["findings"] == [] and not has_blocker(r["findings"])


def test_review_retries_then_succeeds():
    good = json.dumps({"findings": [
        {"severity": "warning", "area": "thermal", "issue": "marginal", "suggestion": "x"}]})
    client = FakeClient("not json at all", good)
    r = review_design(client, "digest")
    assert r["ok"] and client.calls == 2
    assert round(r["cost_usd"], 4) == 0.004     # both attempts billed


def test_review_fails_closed_on_bad_output():
    client = FakeClient("garbage", "still {bad")
    r = review_design(client, "digest")
    assert not r["ok"] and r["findings"] == [] and r["error"]


def test_review_rejects_bad_severity():
    bad = json.dumps({"findings": [{"severity": "critical", "issue": "x"}]})
    r = review_design(FakeClient(bad, bad), "digest")
    assert not r["ok"]      # 'critical' is not in blocker|warning|note


def test_review_meta_ctx_tags_phase():
    client = FakeClient(json.dumps({"findings": []}))
    review_design(client, "digest")
    assert client.last_meta["phase"] == "electrical_review"


def test_review_forwards_thinking_budget():
    client = FakeClient(json.dumps({"findings": []}))
    review_design(client, "digest", model="deepseek/deepseek-v4-flash",
                  reasoning={"max_tokens": 8000})
    assert client.last_model == "deepseek/deepseek-v4-flash"
    assert client.last_reasoning == {"max_tokens": 8000}


# --- lazy corroboration ------------------------------------------------------
_CLEAN = json.dumps({"findings": []})


def _blk(area="current-limit", ref="U1"):
    return json.dumps({"findings": [
        {"severity": "blocker", "area": area,
         "issue": f"{ref} SENSE tied to GND -- no current limit", "suggestion": "fix"}]})


def test_corroborate_clean_costs_one_pass():
    c = FakeClient(_CLEAN)
    r = review_design_corroborated(c, "digest", corroboration=2)
    assert r["ok"] and r["blocked"] is False
    assert c.calls == 1                                  # pass 2 never runs
    assert round(r["cost_usd"], 4) == 0.002


def test_corroborate_blocker_agrees_blocks():
    c = FakeClient(_blk())                               # same blocker both passes
    r = review_design_corroborated(c, "digest", corroboration=2)
    assert r["blocked"] is True and c.calls == 2
    assert r["findings"][0]["corroborated"] is True
    assert round(r["cost_usd"], 4) == 0.004             # cost only grows on pass 2


def test_corroborate_blocker_then_clean_demotes():
    c = FakeClient(_blk(), _CLEAN)
    r = review_design_corroborated(c, "digest", corroboration=2)
    assert r["blocked"] is False and c.calls == 2
    f = r["findings"][0]
    assert f["severity"] == "warning" and f["demoted_from"] == "blocker"
    assert f["corroborated"] is False                    # kept, never dropped


def test_corroborate_disagree_on_refdes_demotes():
    c = FakeClient(_blk(ref="U1"), _blk(ref="U2"))
    r = review_design_corroborated(c, "digest", corroboration=2)
    assert r["blocked"] is False and c.calls == 2
    assert r["findings"][0]["demoted_from"] == "blocker"


def test_corroborate_single_pass_knob():
    c = FakeClient(_blk())
    r = review_design_corroborated(c, "digest", corroboration=1)
    assert r["blocked"] is True and c.calls == 1         # legacy single-pass gate


def test_corroborate_warning_max_never_reaches_pass2():
    # A decoupling 'blocker' is clamped to warning in pass 1 -> no candidate.
    c = FakeClient(json.dumps({"findings": [
        {"severity": "blocker", "area": "decoupling", "issue": "thin", "suggestion": "x"}]}))
    r = review_design_corroborated(c, "digest", corroboration=2)
    assert r["blocked"] is False and c.calls == 1


def test_corroborate_fails_closed_when_pass1_unparseable():
    c = FakeClient("garbage")
    r = review_design_corroborated(c, "digest", corroboration=2)
    assert r["ok"] is False and r["blocked"] is False

