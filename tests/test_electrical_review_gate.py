"""Layer-4 fab gate: _maybe_electrical_review (cost-gated, fail-soft)."""
from __future__ import annotations

import json

import kicraft.server.client as _client_mod
import kicraft.server.config as _config_mod
from kicraft.design.cli_app import _maybe_electrical_review
from kicraft.design.models import (
    BOM,
    BomPart,
    ConversationState,
    NetConnection,
    PinEndpoint,
)
from kicraft.server.config import Settings


class _Fake:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, messages, **kw):
        return {"text": self.reply, "cost_usd": 0.01}


def _state():
    return ConversationState(bom=BOM(
        parts=[BomPart(ref="U1", value="x", symbol="Fake:X",
                       footprint="Package_SO:SOIC-8", sheet="A")],
        connections=[NetConnection(net_name="+3V3", sheet="A",
                                   endpoints=[PinEndpoint(ref="U1", pin="1")])],
    ))


def _enable(monkeypatch, reply):
    monkeypatch.setenv("KICRAFT_ELECTRICAL_REVIEW", "1")
    monkeypatch.setattr(_config_mod.Settings, "from_env",
                        classmethod(lambda cls, **k: Settings(api_key="x")))
    monkeypatch.setattr(_client_mod, "make_client", lambda s=None: _Fake(reply))


def test_gate_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("KICRAFT_ELECTRICAL_REVIEW", raising=False)
    r = _maybe_electrical_review(_state(), tmp_path)
    assert r["ran"] is False and r["blocked"] is False


def test_gate_skips_when_no_connections(monkeypatch, tmp_path):
    monkeypatch.setenv("KICRAFT_ELECTRICAL_REVIEW", "1")
    empty = ConversationState(bom=BOM(parts=[], connections=[]))
    r = _maybe_electrical_review(empty, tmp_path)
    assert r["ran"] is False


def test_gate_blocks_on_blocker(monkeypatch, tmp_path):
    _enable(monkeypatch, json.dumps({"findings": [
        {"severity": "blocker", "area": "filter-math",
         "issue": "wrong values", "suggestion": "fix"}]}))
    r = _maybe_electrical_review(_state(), tmp_path)
    assert r["ran"] is True and r["blocked"] is True
    assert r["findings"][0]["severity"] == "blocker"


def test_gate_passes_on_clean_or_warnings(monkeypatch, tmp_path):
    _enable(monkeypatch, json.dumps({"findings": [
        {"severity": "warning", "area": "esd", "issue": "no TVS", "suggestion": "add"}]}))
    r = _maybe_electrical_review(_state(), tmp_path)
    assert r["ran"] is True and r["blocked"] is False


def test_gate_fail_soft_on_infra_error(monkeypatch, tmp_path):
    # Enabled, but the client/settings blow up -> must NOT block (ran False).
    monkeypatch.setenv("KICRAFT_ELECTRICAL_REVIEW", "1")

    def _boom(cls, **k):
        raise RuntimeError("no API key")

    monkeypatch.setattr(_config_mod.Settings, "from_env", classmethod(_boom))
    r = _maybe_electrical_review(_state(), tmp_path)
    assert r["ran"] is False and r["blocked"] is False


def test_gate_fail_soft_on_bad_model_output(monkeypatch, tmp_path):
    _enable(monkeypatch, "not json at all")   # review_design fails closed (ok=False)
    r = _maybe_electrical_review(_state(), tmp_path)
    assert r["ran"] is False and r["blocked"] is False
