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
import types

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
    Records the max_tokens and reasoning policy it was asked for on each call."""
    def __init__(self, ok_reply):
        self.max_tokens_seen = []
        self.reasoning_seen = []
        self._ok = ok_reply
        self._n = 0

        class _G:
            def status(self_inner):
                return {"spent_total_usd": 0.0, "daily_remaining_usd": 5.0,
                        "daily_ceiling_usd": 5.0}
        self.guard = _G()

    def chat(self, messages, max_tokens=4096, temperature=0.2, progress=None, meta_ctx=None,
             reasoning=None):
        self.max_tokens_seen.append(max_tokens)
        self.reasoning_seen.append(reasoning)
        self._n += 1
        if self._n == 1:
            return {"text": '{ "goal": "x", truncated', "cost_usd": 0.0,
                    "reasoning": "", "finish_reason": "length"}
        return {"text": self._ok, "cost_usd": 0.0, "reasoning": "", "finish_reason": "stop"}


def test_truncated_reply_triggers_one_fixed_cap_serialization_call(tmp_path):
    ok = json.dumps({"goal": "a USB-powered LED", "constraints": [], "named_parts": [],
                     "inferred_expertise": "intermediate", "assumptions": [],
                     "project_stem": "USB_LED"})
    client = _TruncThenOkClient(ok)
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "ok"                                 # recovered, committed
    assert len(client.max_tokens_seen) == 2
    # the serialization retry uses the policy's FIXED cap (never the old
    # cap-doubling) and disables reasoning
    assert client.max_tokens_seen[1] == 8192
    assert client.max_tokens_seen[1] > client.max_tokens_seen[0]  # still more headroom
    assert client.reasoning_seen[1] == {"enabled": False}


# ---- completion metadata flows through chat / tool rounds / forced final ----

def test_stream_records_cap_and_reasoning_policy_in_ledger_meta(monkeypatch):
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
    c.chat([{"role": "user", "content": "hi"}], max_tokens=8192,
           reasoning={"enabled": False})
    rec = c.guard.records[-1]
    assert rec["meta"]["max_tokens"] == 8192
    assert rec["meta"]["reasoning_policy"] == {"enabled": False}
    assert rec["meta"]["content_chars"] == len('{"x":1}')
    # the payload carries the control keys only, never _meta/_meta_ctx
    assert "_meta" not in captured["payload"] and "_meta_ctx" not in captured["payload"]


def test_chat_returns_completion_telemetry(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        return _FakeResp([
            {"choices": [{"delta": {"content": '{"x":1}'}}]},
            {"choices": [{"finish_reason": "stop", "delta": {}}]},
            _usage_chunk(cached=800),
        ])

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    r = c.chat([{"role": "user", "content": "hi"}], max_tokens=4096,
               reasoning={"max_tokens": 2048})
    assert r["provider"] == "DeepSeek"
    assert r["usage"]["prompt_tokens"] == 1000
    assert r["usage"]["completion_tokens"] == 50
    assert r["max_tokens"] == 4096
    assert r["reasoning_policy"] == {"max_tokens": 2048}
    assert r["content_chars"] == len('{"x":1}')
    assert r["finish_reason"] == "stop"


def test_chat_with_tools_rounds_carry_telemetry(monkeypatch):
    client = CappedOpenRouterClient(
        settings=types.SimpleNamespace(),
        guard=types.SimpleNamespace(status=lambda: {}))

    def fake_stream(body, on_delta=None):
        return ({"role": "assistant", "content": '{"ok": true}',
                 "finish_reason": "stop", "provider": "DeepSeek",
                 "usage": {"prompt_tokens": 5},
                 "requested_max_tokens": body.get("max_tokens"),
                 "reasoning_policy": body.get("reasoning")}, 0.0)

    monkeypatch.setattr(client, "_stream", fake_stream)
    r = client.chat_with_tools([{"role": "user", "content": "go"}], tools=[],
                               executor=lambda n, a: "ok", max_rounds=1,
                               max_tokens=16384, reasoning={"enabled": False})
    assert r["provider"] == "DeepSeek"
    assert r["usage"] == {"prompt_tokens": 5}
    assert r["max_tokens"] == 16384
    assert r["reasoning_policy"] == {"enabled": False}
    assert r["finish_reason"] == "stop"


def test_chat_with_tools_forced_final_carries_telemetry(monkeypatch):
    client = CappedOpenRouterClient(
        settings=types.SimpleNamespace(),
        guard=types.SimpleNamespace(status=lambda: {}))

    def fake_stream(body, on_delta=None):
        return ({"role": "assistant", "content": None, "finish_reason": "tool_calls",
                 "tool_calls": [{"id": "t1", "type": "function",
                                 "function": {"name": "list_parts", "arguments": "{}"}}],
                 "requested_max_tokens": body.get("max_tokens"),
                 "reasoning_policy": body.get("reasoning")}, 0.0)

    monkeypatch.setattr(client, "_stream", fake_stream)
    r = client.chat_with_tools([{"role": "user", "content": "go"}], tools=[],
                               executor=lambda n, a: "ok", max_rounds=1,
                               max_tokens=16384, reasoning=None)
    assert r.get("forced_final") is True          # budget exhausted -> cold final
    assert r["max_tokens"] == 16384               # the final round still carries the cap
    assert r["reasoning_policy"] is None
    assert r["finish_reason"] == "tool_calls"


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


# ---- UTF-8 SSE decoding regression (KC-U2VAA8 "12 ÂµF" mojibake) -----------


class _ByteStreamResp:
    """Fake streaming Response that decodes exactly like ``requests``:
    ``iter_lines(decode_unicode=True)`` uses ``self.encoding``, which requests
    defaults to ISO-8859-1 for a ``text/event-stream`` body with no charset --
    the bug that turned a UTF-8 ``µ`` (0xC2 0xB5) into ``Âµ``."""

    def __init__(self, chunks, status_code=200, reason="OK"):
        lines = [f"data: {json.dumps(c, ensure_ascii=False)}" for c in chunks]
        lines.append("data: [DONE]")
        self._raw = [ln.encode("utf-8") for ln in lines]
        self.status_code = status_code
        self.reason = reason
        self.headers = {"content-type": "text/event-stream"}
        self.encoding = "ISO-8859-1"  # requests' buggy default for text/* w/o charset

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code} {self.reason}", response=self)

    def close(self):
        pass

    def iter_lines(self, decode_unicode=True):
        for b in self._raw:
            yield b.decode(self.encoding) if decode_unicode else b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_open_stream_pins_utf8_and_avoids_mojibake(monkeypatch):
    # The BOM value "12 µF" arrived over the SSE stream as bytes; without the
    # encoding pin, requests decoded them as Latin-1 -> "12 ÂµF" landed in
    # state.json. _open_stream must set resp.encoding = "utf-8".
    chunks = [
        {"choices": [{"delta": {"content": "value: 12 µF, 4.7 kΩ, 10 °C"}}]},
        {"choices": [{"finish_reason": "stop", "delta": {}}]},
        _usage_chunk(),
    ]
    holder = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        r = _ByteStreamResp(chunks)
        holder["resp"] = r
        return r

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    msg, _cost = c._stream({"messages": [{"role": "user", "content": "hi"}]})

    assert holder["resp"].encoding == "utf-8"          # _open_stream pinned it
    assert "12 µF" in msg["content"]                   # decoded on the wire bytes
    assert "4.7 kΩ" in msg["content"]
    assert "10 °C" in msg["content"]
    assert "Â" not in msg["content"]                   # no double-encoding


# ---- _stream: mid-stream disconnect retry (2026-07-19 review §4.1) --------

class _BrokenMidStreamResp(_FakeResp):
    """Streams a couple of deltas, then dies like board 625's
    "Connection broken: InvalidChunkLength" ChunkedEncodingError."""

    def iter_lines(self, decode_unicode=True):
        yield self._lines[0]
        raise requests.exceptions.ChunkedEncodingError(
            "Connection broken: InvalidChunkLength(got length b'', 0 bytes read)"
        )


def test_stream_retries_mid_stream_disconnect(monkeypatch):
    s = Settings(api_key="k", llm_max_retries=2, llm_retry_backoff_s=0.0)
    guard = _RecordingGuard()
    c = CappedOpenRouterClient(s, guard=guard)
    good_chunks = [
        {"choices": [{"delta": {"content": "hello"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        _usage_chunk(),
    ]
    attempts = []

    def _fake_open(payload):
        attempts.append(1)
        if len(attempts) == 1:
            return _BrokenMidStreamResp(
                [{"choices": [{"delta": {"content": "par"}}]}]
            )
        return _FakeResp(good_chunks)

    monkeypatch.setattr(c, "_open_stream", _fake_open)
    msg, cost = c._stream({"messages": [{"role": "user", "content": "x"}]})
    assert len(attempts) == 2
    # The partial "par" from the aborted attempt was discarded, not prepended.
    assert msg["content"] == "hello"
    assert msg["finish_reason"] == "stop"
    assert cost > 0.0


def test_stream_gives_up_after_max_retries(monkeypatch):
    s = Settings(api_key="k", llm_max_retries=1, llm_retry_backoff_s=0.0)
    c = CappedOpenRouterClient(s, guard=_RecordingGuard())
    monkeypatch.setattr(
        c,
        "_open_stream",
        lambda payload: _BrokenMidStreamResp(
            [{"choices": [{"delta": {"content": "par"}}]}]
        ),
    )
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        c._stream({"messages": [{"role": "user", "content": "x"}]})


# ---- in-stream reasoning-loop breaker (KC-VWW5X7) --------------------------

def test_reasoning_loop_ceiling_signal():
    # Reasoning-only stream over the token ceiling with no content -> loop.
    s = Settings(api_key="k", reasoning_repeat_window=1_000_000)  # isolate ceiling
    c = CappedOpenRouterClient(s, guard=_RecordingGuard())
    assert c._reasoning_loop(20_000, 0, "x" * 20_000, client_mod.time.monotonic()) is True
    assert c._reasoning_loop(1_000, 0, "x" * 1_000, client_mod.time.monotonic()) is False


def test_reasoning_loop_repetition_signal():
    # A 256-char block repeated 4x -> repetition, even below the ceiling.
    s = Settings(api_key="k", reasoning_max_tokens=1_000_000)  # isolate repetition
    c = CappedOpenRouterClient(s, guard=_RecordingGuard())
    block = "a" * 255 + "Z"
    assert c._reasoning_loop(len(block) * 4, 0, block * 4,
                             client_mod.time.monotonic()) is True
    assert c._reasoning_loop(len(block), 0, block,
                             client_mod.time.monotonic()) is False


def test_reasoning_loop_stall_signal():
    # Reasoning flowing, no content, wall clock past the timeout -> stall.
    s = Settings(api_key="k", reasoning_max_tokens=1_000_000,
                 reasoning_repeat_window=1_000_000)
    c = CappedOpenRouterClient(s, guard=_RecordingGuard())
    assert c._reasoning_loop(100, 0, "x" * 100,
                             client_mod.time.monotonic() - 999) is True


def test_reasoning_loop_never_fires_with_content():
    # Any content token gates every signal off.
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    assert c._reasoning_loop(99_999, 1, "y" * 99_999,
                             client_mod.time.monotonic() - 999) is False


def test_stream_aborts_reasoning_ceiling(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        return _FakeResp([{"choices": [{"delta": {"reasoning": "x" * 20_000}}]}])

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    msg, cost = c._stream({"messages": [{"role": "user", "content": "x"}]})
    assert msg["loop_detected"] is True
    assert msg["finish_reason"] == "reasoning_loop"
    assert msg["content"] is None
    assert cost > 0.0  # partial stream still recorded against the guard


def test_stream_aborts_reasoning_repetition(monkeypatch):
    block = "a" * 255 + "Z"

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        return _FakeResp([{"choices": [{"delta": {"reasoning": block * 4}}]}])

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    msg, _ = c._stream({"messages": [{"role": "user", "content": "x"}]})
    assert msg["loop_detected"] is True
    assert msg["finish_reason"] == "reasoning_loop"


def test_stream_content_stream_is_not_aborted(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        return _FakeResp([
            {"choices": [{"delta": {"reasoning": "thinking here"}}]},
            {"choices": [{"delta": {"content": '{"x":1}'}}]},
            {"choices": [{"finish_reason": "stop", "delta": {}}]},
            _usage_chunk(),
        ])

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    msg, _ = c._stream({"messages": [{"role": "user", "content": "x"}]})
    assert not msg.get("loop_detected")
    assert msg["finish_reason"] == "stop"
    assert msg["content"] == '{"x":1}'
