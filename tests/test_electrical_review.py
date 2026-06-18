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
    build_design_digest,
    has_blocker,
    review_design,
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
        {"severity": "Blocker", "area": "decoupling",
         "issue": "U1 has no 100nF bypass cap", "suggestion": "add 100nF VCC-GND"},
        {"severity": "note", "area": "layout", "issue": "minor", "suggestion": ""},
    ]})
    r = review_design(FakeClient(reply), "digest")
    assert r["ok"] and r["error"] is None
    assert len(r["findings"]) == 2
    assert r["findings"][0]["severity"] == "blocker"   # normalized lowercase
    assert has_blocker(r["findings"])
    assert round(r["cost_usd"], 4) == 0.002


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

