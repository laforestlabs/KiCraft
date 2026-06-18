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


def test_gate_disabled_when_env_off(monkeypatch, tmp_path):
    # Explicit opt-out short-circuits before any client/Settings access.
    monkeypatch.setenv("KICRAFT_ELECTRICAL_REVIEW", "0")
    r = _maybe_electrical_review(_state(), tmp_path)
    assert r["ran"] is False and r["blocked"] is False


def test_gate_on_by_default(monkeypatch, tmp_path):
    # No env var set -> the gate runs (on by default).
    monkeypatch.delenv("KICRAFT_ELECTRICAL_REVIEW", raising=False)
    monkeypatch.setattr(_config_mod.Settings, "from_env",
                        classmethod(lambda cls, **k: Settings(api_key="x")))
    monkeypatch.setattr(_client_mod, "make_client",
                        lambda s=None: _Fake(json.dumps({"findings": []})))
    r = _maybe_electrical_review(_state(), tmp_path)
    assert r["ran"] is True and r["blocked"] is False


def test_gate_skips_when_no_connections(monkeypatch, tmp_path):
    empty = ConversationState(bom=BOM(parts=[], connections=[]))
    r = _maybe_electrical_review(empty, tmp_path)
    assert r["ran"] is False


def test_gate_fail_soft_without_api_key(monkeypatch, tmp_path):
    # On by default, but no API key -> Settings.from_env raises SystemExit, which
    # MUST be caught so a keyless build never crashes.
    monkeypatch.delenv("KICRAFT_ELECTRICAL_REVIEW", raising=False)

    def _no_key(cls, **k):
        raise SystemExit("OPENROUTER_API_KEY is not set")

    monkeypatch.setattr(_config_mod.Settings, "from_env", classmethod(_no_key))
    r = _maybe_electrical_review(_state(), tmp_path)
    assert r["ran"] is False and r["blocked"] is False


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
