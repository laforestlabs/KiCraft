"""Cost-safety + telemetry tests for the capped OpenRouter client and the spend
report. Network is mocked (a fake SSE stream), so nothing here spends tokens.

Covers the changes that cut web-KiCraft LLM cost:
- the OpenRouter `provider` routing block (allowlist + price cap) is built from
  settings and sent on every call,
- the prompt-cache breakpoint is applied to the system prompt (and gated by the
  setting),
- internal `_meta*` control keys never leak into the request body,
- the real billed cost + cached-token count + resolved provider are recorded as
  structured meta, which the web-cost-report then attributes per run/stage,
- the stage driver raises max_tokens (instead of blindly redrafting) when a reply
  is truncated at the output cap.
"""
from __future__ import annotations

import json

import pytest
import requests

from kicraft.server import client as client_mod
from kicraft.server.client import CappedOpenRouterClient
from kicraft.server.config import Settings
from kicraft.server.session import run_session
from kicraft.server.spend_guard import SpendGuard
from kicraft.cli import web_cost_report


# ---- fakes ----------------------------------------------------------------

class _FakeResp:
    """A minimal stand-in for requests' streaming Response (context manager)."""
    def __init__(self, chunks, status_code=200, reason="OK"):
        self._lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
        self.status_code = status_code
        self.reason = reason

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} {self.reason}", response=self)

    def close(self):
        pass

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _RecordingGuard:
    def __init__(self):
        self.records = []

    def preflight(self):
        pass

    def record(self, model, intok, outtok, cost, meta=""):
        self.records.append({"model": model, "in": intok, "out": outtok,
                             "cost": cost, "meta": meta})

    def status(self):
        return {"spent_total_usd": 0.0}


def _usage_chunk(cached=0, cost=0.001, intok=1000, outtok=50):
    return {"provider": "DeepSeek",
            "usage": {"prompt_tokens": intok, "completion_tokens": outtok, "cost": cost,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


# ---- provider block + cache control (pure) --------------------------------

def test_provider_block_from_settings():
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    pb = c._provider_block()
    assert pb["order"] == ["novita/fp8", "siliconflow/fp8", "streamlake"]
    assert pb["allow_fallbacks"] is True
    assert pb["max_price"] == {"prompt": 0.18, "completion": 0.35}


def test_provider_block_omits_zero_price_cap():
    s = Settings(api_key="k", max_price_prompt=0.0, max_price_completion=0.0)
    pb = CappedOpenRouterClient(s, guard=_RecordingGuard())._provider_block()
    assert "max_price" not in pb


def test_apply_cache_control_marks_system_and_is_idempotent():
    msgs = [{"role": "system", "content": "BIG STABLE PREFIX"},
            {"role": "user", "content": "hi"}]
    CappedOpenRouterClient._apply_cache_control(msgs)
    blk = msgs[0]["content"]
    assert isinstance(blk, list) and blk[0]["cache_control"] == {"type": "ephemeral"}
    assert blk[0]["text"] == "BIG STABLE PREFIX"
    CappedOpenRouterClient._apply_cache_control(msgs)            # second pass
    assert len(msgs[0]["content"]) == 1                          # not double-wrapped
    assert msgs[1]["content"] == "hi"                            # user untouched


# ---- _stream: payload shape + structured recording ------------------------

def test_stream_sends_provider_block_and_records_structured_meta(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        captured["payload"] = json
        return _FakeResp([
            {"choices": [{"delta": {"content": '{"x":1}'}}]},
            {"choices": [{"finish_reason": "stop", "delta": {}}]},
            _usage_chunk(cached=800),
        ])

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    msg, cost = c._stream({
        "messages": [{"role": "system", "content": "SYS"},
                     {"role": "user", "content": "hi"}],
        "_meta": "tools", "_meta_ctx": {"run_id": "r1", "stage": "bom", "attempt": 0}})

    p = captured["payload"]
    assert p["provider"]["order"] == ["novita/fp8", "siliconflow/fp8", "streamlake"]  # routing pinned
    assert p["provider"]["max_price"]["prompt"] == 0.18          # spike ceiling
    assert "_meta" not in p and "_meta_ctx" not in p             # control keys stripped
    assert isinstance(p["messages"][0]["content"], list)        # cache breakpoint applied
    assert p["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    rec = c.guard.records[-1]
    assert cost == 0.001
    assert rec["meta"]["run_id"] == "r1" and rec["meta"]["stage"] == "bom"
    assert rec["meta"]["cached_tokens"] == 800
    assert rec["meta"]["provider"] == "DeepSeek"
    assert rec["meta"]["finish_reason"] == "stop"
    assert rec["meta"]["phase"] == "tools"


def test_stream_cache_control_gated_off(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        fake_post.payload = json
        return _FakeResp([{"choices": [{"finish_reason": "stop", "delta": {"content": "{}"}}]},
                          _usage_chunk()])

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    s = Settings(api_key="k", enable_prompt_cache=False)
    CappedOpenRouterClient(s, guard=_RecordingGuard())._stream(
        {"messages": [{"role": "system", "content": "SYS"}]})
    assert fake_post.payload["messages"][0]["content"] == "SYS"   # left as a plain string


# ---- transient-failure retry (D5) -----------------------------------------

def _ok_chunks():
    return [{"choices": [{"delta": {"content": "{}"}}]},
            {"choices": [{"finish_reason": "stop", "delta": {}}]},
            _usage_chunk()]


def test_open_stream_retries_transient_5xx_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        calls["n"] += 1
        if calls["n"] <= 2:                      # two 503s, then a good stream
            return _FakeResp([], status_code=503, reason="Service Unavailable")
        return _FakeResp(_ok_chunks())

    sleeps = []
    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: sleeps.append(s))
    s = Settings(api_key="k", llm_max_retries=3, llm_retry_backoff_s=0.5)
    c = CappedOpenRouterClient(s, guard=_RecordingGuard())
    msg, cost = c._stream({"messages": [{"role": "user", "content": "hi"}]})
    assert calls["n"] == 3                        # 2 failures + 1 success
    assert sleeps == [0.5, 1.0]                   # exponential backoff between attempts
    assert cost == 0.001


def test_open_stream_retries_connection_error(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ConnectionError("reset by peer")
        return _FakeResp(_ok_chunks())

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    s = Settings(api_key="k", llm_max_retries=2)
    c = CappedOpenRouterClient(s, guard=_RecordingGuard())
    c._stream({"messages": [{"role": "user", "content": "hi"}]})
    assert calls["n"] == 2


def test_open_stream_does_not_retry_4xx(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        calls["n"] += 1
        return _FakeResp([], status_code=400, reason="Bad Request")

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    s = Settings(api_key="k", llm_max_retries=3)
    c = CappedOpenRouterClient(s, guard=_RecordingGuard())
    with pytest.raises(requests.exceptions.HTTPError):
        c._stream({"messages": [{"role": "user", "content": "hi"}]})
    assert calls["n"] == 1                        # client error: no retry


def test_open_stream_raises_after_exhausting_retries(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        calls["n"] += 1
        return _FakeResp([], status_code=503, reason="Service Unavailable")

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    s = Settings(api_key="k", llm_max_retries=2)
    c = CappedOpenRouterClient(s, guard=_RecordingGuard())
    with pytest.raises(requests.exceptions.HTTPError):
        c._stream({"messages": [{"role": "user", "content": "hi"}]})
    assert calls["n"] == 3                        # 1 initial + 2 retries, then give up


# ---- design temperature (D3) ----------------------------------------------

def test_design_temperature_defaults_to_zero_and_is_configurable():
    from kicraft.server.stage_driver import _design_temperature
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    assert c.s.design_temperature == 0.0                # new default cuts variance
    assert _design_temperature(c) == 0.0
    c2 = CappedOpenRouterClient(Settings(api_key="k", design_temperature=0.3),
                                guard=_RecordingGuard())
    assert _design_temperature(c2) == 0.3

    class _NoSettings:           # mock-style client without .s -> historical 0.2
        pass
    assert _design_temperature(_NoSettings()) == 0.2


# ---- spend ledger: structured meta round-trips ----------------------------

def test_spend_guard_serializes_dict_meta(tmp_path):
    s = Settings(api_key="k", ledger_path=tmp_path / "ledger.db",
                 daily_usd_ceiling=100, total_usd_ceiling=100)
    g = SpendGuard(s)
    g.record("deepseek/deepseek-v4-flash", 1000, 50, 0.001,
             meta={"run_id": "r1", "stage": "bom", "cached_tokens": 800})
    g.record("deepseek/deepseek-v4-flash", 10, 5, 0.0, meta="legacy-tag")  # bare string still ok
    import sqlite3
    rows = sqlite3.connect(str(s.ledger_path)).execute(
        "SELECT meta FROM spend ORDER BY id").fetchall()
    assert json.loads(rows[0][0])["run_id"] == "r1"               # dict -> JSON
    assert rows[1][0] == "legacy-tag"                             # str -> verbatim


# ---- web cost report: attribution + cache + spikes ------------------------

def test_web_cost_report_attributes_and_flags(tmp_path):
    s = Settings(api_key="k", ledger_path=tmp_path / "ledger.db",
                 daily_usd_ceiling=100, total_usd_ceiling=100)
    g = SpendGuard(s)
    # one normal cached call, one routing spike (huge $/Mtok, tiny output)
    g.record("m", 1000, 50, 0.0001, meta={"run_id": "r1", "stage": "bom",
             "cached_tokens": 900, "provider": "DeepSeek"})
    g.record("m", 1000, 20, 0.002, meta={"run_id": "r1", "stage": "bom",
             "cached_tokens": 0, "provider": "Expensive"})

    rows = web_cost_report.load_rows(str(s.ledger_path))
    summary = web_cost_report.summarize(rows, spike_threshold=0.50)
    assert summary["total"]["calls"] == 2
    assert summary["total"]["spikes"] == 1                        # the $2/Mtok call
    assert "r1" in summary["runs"]
    # cache hit-rate = cached/input over the run = 900 / 2000 = 45%
    run = summary["runs"]["r1"]
    assert round(run["cached"] / run["input"] * 100) == 45
    assert "bom" in summary["run_stage"]["r1"]
    # report renders without error
    assert "cache hit-rate" in web_cost_report.format_report(summary, by="stage")


def test_web_cost_report_legacy_rows_cluster_by_time(tmp_path):
    s = Settings(api_key="k", ledger_path=tmp_path / "ledger.db",
                 daily_usd_ceiling=100, total_usd_ceiling=100)
    g = SpendGuard(s)
    g.record("m", 100, 10, 0.0001, meta="tools")                 # legacy bare-string rows
    g.record("m", 100, 10, 0.0001, meta="tools")
    rows = web_cost_report.load_rows(str(s.ledger_path))
    summary = web_cost_report.summarize(rows)
    assert summary["total"]["calls"] == 2
    assert all(r.startswith("legacy#") for r in summary["runs"])  # no run_id -> legacy cluster


# ---- truncation-aware retry (stage driver) --------------------------------

class _TruncThenOkClient:
    """First reply is truncated at the output cap; second is a valid intent slot.
    Records the max_tokens it was asked for on each call."""
    def __init__(self, ok_reply):
        self.max_tokens_seen = []
        self._ok = ok_reply
        self._n = 0

        class _G:
            def status(self_inner):
                return {"spent_total_usd": 0.0, "daily_remaining_usd": 5.0,
                        "daily_ceiling_usd": 5.0}
        self.guard = _G()

    def chat(self, messages, max_tokens=4096, temperature=0.2, progress=None, meta_ctx=None):
        self.max_tokens_seen.append(max_tokens)
        self._n += 1
        if self._n == 1:
            return {"text": '{ "goal": "x", truncated', "cost_usd": 0.0,
                    "reasoning": "", "finish_reason": "length"}
        return {"text": self._ok, "cost_usd": 0.0, "reasoning": "", "finish_reason": "stop"}


def test_truncated_reply_raises_max_tokens_then_commits(tmp_path):
    ok = json.dumps({"goal": "a USB-powered LED", "constraints": [], "named_parts": [],
                     "inferred_expertise": "intermediate", "assumptions": [],
                     "project_stem": "USB_LED"})
    client = _TruncThenOkClient(ok)
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "ok"                                 # recovered, committed
    assert len(client.max_tokens_seen) == 2
    assert client.max_tokens_seen[1] > client.max_tokens_seen[0]  # cap was raised, not re-tried flat


# ---- ERC-recovery offender parsing (web) ----------------------------------

def _write_synth_check(tmp_path, checks):
    (tmp_path / ".kicraft").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".kicraft" / "synthesis_check.json").write_text(
        json.dumps({"status": "failed", "checks": checks}), encoding="utf-8")


def test_erc_offenders_returns_failed_erc_errors(tmp_path):
    from kicraft.server.web import _erc_offenders
    _write_synth_check(tmp_path, [
        {"name": "9.2 footprints non-empty", "ok": True, "offenders": []},
        {"name": "9.12 ERC", "ok": False,
         "offenders": ["root: Pin U1.3 not connected", "MCU: conflicting outputs"]}])
    assert _erc_offenders(tmp_path) == ["root: Pin U1.3 not connected",
                                        "MCU: conflicting outputs"]


def test_erc_offenders_empty_when_erc_clean(tmp_path):
    from kicraft.server.web import _erc_offenders
    _write_synth_check(tmp_path, [{"name": "9.12 ERC", "ok": True, "offenders": []}])
    assert _erc_offenders(tmp_path) == []                        # nothing to recover
    assert _erc_offenders(tmp_path / "nope") == []               # missing file -> []
