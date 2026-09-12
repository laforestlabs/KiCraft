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
- normal and recovery calls honor explicit stage output ceilings without
  relaxing the project's monetary budget.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import requests

from kicraft.server import client as client_mod
from kicraft.server.client import CappedOpenRouterClient, _StreamingCollectionGuard
from kicraft.server.config import (
    CollectionBound,
    DESIGN_PROFILES,
    STAGE_COLLECTION_BOUNDS,
    Settings,
)
from kicraft.cli.model_preflight import preflight_role
from kicraft.server.session import run_session
from kicraft.server.spend_guard import SpendGuard
from kicraft.cli import web_cost_report


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (requests.exceptions.HTTPError("429 Too Many Requests"), "provider_rate_limited"),
        (requests.exceptions.HTTPError("500 Upstream"), "provider_upstream_5xx"),
        (requests.exceptions.HTTPError("401 Unauthorized"), "provider_auth"),
        (
            requests.exceptions.HTTPError("400 response_format unsupported"),
            "provider_response_format_rejected",
        ),
        (requests.exceptions.Timeout("timed out"), "transport_timeout"),
        (
            requests.exceptions.ChunkedEncodingError("stream interrupted"),
            "transport_stream_interrupted",
        ),
    ],
)
def test_provider_failures_have_stable_detailed_kinds(exc, kind):
    facts = client_mod.classify_provider_exception(exc)
    assert facts["failure_kind"] == kind
    assert not ({"messages", "payload", "response_body"} & facts.keys())


# ---- fakes ----------------------------------------------------------------


class _FakeResp:
    """A minimal stand-in for requests' streaming Response (context manager)."""

    def __init__(self, chunks, status_code=200, reason="OK"):
        self._lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]
        self.status_code = status_code
        self.reason = reason
        self.closed = False
        self.lines_read = 0

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} {self.reason}", response=self)

    def close(self):
        self.closed = True

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            self.lines_read += 1
            yield line

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


class _RecordingGuard:
    def __init__(self):
        self.records = []
        self.preflights = []

    def preflight(self, call_ceiling_usd=0.0, run_id=None):
        self.preflights.append((call_ceiling_usd, run_id))

    def record(self, model, intok, outtok, cost, meta=""):
        self.records.append(
            {"model": model, "in": intok, "out": outtok, "cost": cost, "meta": meta}
        )

    def status(self):
        return {"spent_total_usd": 0.0}


def _usage_chunk(cached=0, cost=0.001, intok=1000, outtok=50):
    return {
        "provider": "DeepSeek",
        "usage": {
            "prompt_tokens": intok,
            "completion_tokens": outtok,
            "cost": cost,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


# ---- provider block + cache control (pure) --------------------------------


def test_provider_block_from_settings():
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    pb = c._provider_block()
    assert pb["order"] == ["open-inference/fp8"]
    assert pb["allow_fallbacks"] is False
    assert pb["max_price"] == {"prompt": 0.11, "completion": 0.24}


def test_design_profile_clone_reuses_guard_and_changes_only_route():
    guard = _RecordingGuard()
    settings = Settings(
        api_key="k",
        design_profile="deepseek",
        escalation_profile="luna",
        request_timeout_s=17,
    )
    original = CappedOpenRouterClient(settings, guard=guard)
    escalated = original.with_design_profile("luna")
    profile = DESIGN_PROFILES["luna"]
    assert escalated is not original
    assert escalated.guard is guard
    assert escalated.s.model == profile["model"]
    assert escalated.s.provider_order == profile["provider_order"]
    assert escalated.s.max_price_prompt == profile["max_price_prompt"]
    assert escalated.s.max_price_completion == profile["max_price_completion"]
    assert escalated.s.request_timeout_s == 17
    assert original.s.design_profile == "deepseek"


def test_provider_block_omits_zero_price_cap():
    s = Settings(api_key="k", max_price_prompt=0.0, max_price_completion=0.0)
    pb = CappedOpenRouterClient(s, guard=_RecordingGuard())._provider_block()
    assert "max_price" not in pb


def test_apply_cache_control_marks_system_and_is_idempotent():
    msgs = [{"role": "system", "content": "BIG STABLE PREFIX"}, {"role": "user", "content": "hi"}]
    CappedOpenRouterClient._apply_cache_control(msgs)
    blk = msgs[0]["content"]
    assert isinstance(blk, list) and blk[0]["cache_control"] == {"type": "ephemeral"}
    assert blk[0]["text"] == "BIG STABLE PREFIX"
    CappedOpenRouterClient._apply_cache_control(msgs)  # second pass
    assert len(msgs[0]["content"]) == 1  # not double-wrapped
    assert msgs[1]["content"] == "hi"  # user untouched


# ---- _stream: payload shape + structured recording ------------------------


def test_stream_sends_provider_block_and_records_structured_meta(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        captured["payload"] = json
        return _FakeResp(
            [
                {"choices": [{"delta": {"content": '{"x":1}'}}]},
                {"choices": [{"finish_reason": "stop", "delta": {}}]},
                _usage_chunk(cached=800),
            ]
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    msg, cost = c._stream(
        {
            "messages": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "hi"},
            ],
            "_meta": "tools",
            "_meta_ctx": {"run_id": "r1", "stage": "bom", "attempt": 0},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test_schema_v1",
                    "strict": True,
                    "schema": {"type": "object"},
                },
            },
        }
    )
    p = captured["payload"]
    assert p["provider"]["order"] == ["open-inference/fp8"]  # dated Flash route pinned
    assert p["provider"]["max_price"]["prompt"] == 0.11
    assert "_meta" not in p and "_meta_ctx" not in p  # control keys stripped
    assert isinstance(p["messages"][0]["content"], list)  # cache breakpoint applied
    assert p["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    rec = c.guard.records[-1]
    assert cost == 0.001
    assert rec["meta"]["run_id"] == "r1" and rec["meta"]["stage"] == "bom"
    assert rec["meta"]["cached_tokens"] == 800
    assert rec["meta"]["provider"] == "DeepSeek"
    assert rec["meta"]["finish_reason"] == "stop"
    assert rec["meta"]["phase"] == "tools"
    assert rec["meta"]["response_policy_name"] == "test_schema_v1"


def test_stream_cache_control_gated_off(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        fake_post.payload = json
        return _FakeResp(
            [{"choices": [{"finish_reason": "stop", "delta": {"content": "{}"}}]}, _usage_chunk()]
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    s = Settings(api_key="k", enable_prompt_cache=False)
    CappedOpenRouterClient(s, guard=_RecordingGuard())._stream(
        {"messages": [{"role": "system", "content": "SYS"}]}
    )
    assert fake_post.payload["messages"][0]["content"] == "SYS"  # left as a plain string


# ---- transient-failure retry (D5) -----------------------------------------


def _ok_chunks():
    return [
        {"choices": [{"delta": {"content": "{}"}}]},
        {"choices": [{"finish_reason": "stop", "delta": {}}]},
        _usage_chunk(),
    ]


def test_open_stream_retries_transient_5xx_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        calls["n"] += 1
        if calls["n"] <= 2:  # two 503s, then a good stream
            return _FakeResp([], status_code=503, reason="Service Unavailable")
        return _FakeResp(_ok_chunks())

    sleeps = []
    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: sleeps.append(s))
    s = Settings(api_key="k", llm_max_retries=3, llm_retry_backoff_s=0.5)
    c = CappedOpenRouterClient(s, guard=_RecordingGuard())
    msg, cost = c._stream({"messages": [{"role": "user", "content": "hi"}]})
    assert calls["n"] == 3  # 2 failures + 1 success
    assert sleeps == [0.5, 1.0]  # exponential backoff between attempts
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
    assert calls["n"] == 1  # client error: no retry


def test_open_stream_returns_429_to_top_level_without_retry(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        calls["n"] += 1
        return _FakeResp([], status_code=429, reason="Too Many Requests")

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: pytest.fail("unexpected sleep"))
    client = CappedOpenRouterClient(
        Settings(api_key="k", llm_max_retries=3),
        guard=_RecordingGuard(),
    )
    with pytest.raises(requests.exceptions.HTTPError):
        client._stream({"messages": [{"role": "user", "content": "hi"}]})
    assert calls["n"] == 1


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
    assert calls["n"] == 3  # 1 initial + 2 retries, then give up


def test_stream_preflight_reserves_configured_call_ceiling_and_run(monkeypatch):
    monkeypatch.setattr(
        client_mod.requests,
        "post",
        lambda *a, **k: _FakeResp(
            [
                {"choices": [{"delta": {"content": "{}"}, "finish_reason": "stop"}]},
                _usage_chunk(cost=0.001),
            ]
        ),
    )
    settings = Settings(
        api_key="k",
        llm_max_retries=0,
        max_tokens_per_call=1_000,
        max_price_prompt=1.0,
        max_price_completion=2.0,
    )
    guard = _RecordingGuard()
    CappedOpenRouterClient(settings, guard=guard)._stream(
        {
            "messages": [{"role": "user", "content": "reserve this call"}],
            "_meta_ctx": {"run_id": "p7-run"},
        }
    )
    [(reserved, run_id)] = guard.preflights
    assert run_id == "p7-run"
    assert 0.002 < reserved < 0.0021


# ---- design temperature (D3) ----------------------------------------------


def test_design_temperature_defaults_to_zero_and_is_configurable():
    from kicraft.server.stage_runtime import _design_temperature

    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    assert c.s.design_temperature == 0.0  # new default cuts variance
    assert _design_temperature(c) == 0.0
    c2 = CappedOpenRouterClient(
        Settings(api_key="k", design_temperature=0.3), guard=_RecordingGuard()
    )
    assert _design_temperature(c2) == 0.3

    class _NoSettings:  # mock-style client without .s -> historical 0.2
        pass

    assert _design_temperature(_NoSettings()) == 0.2


# ---- spend ledger: structured meta round-trips ----------------------------


def test_spend_guard_serializes_dict_meta(tmp_path):
    s = Settings(
        api_key="k",
        ledger_path=tmp_path / "ledger.db",
        daily_usd_ceiling=100,
        total_usd_ceiling=100,
    )
    g = SpendGuard(s)
    g.record(
        "deepseek/deepseek-v4-flash",
        1000,
        50,
        0.001,
        meta={"run_id": "r1", "stage": "bom", "cached_tokens": 800},
    )
    g.record("deepseek/deepseek-v4-flash", 10, 5, 0.0, meta="legacy-tag")  # bare string still ok
    import sqlite3

    rows = (
        sqlite3.connect(str(s.ledger_path)).execute("SELECT meta FROM spend ORDER BY id").fetchall()
    )
    assert json.loads(rows[0][0])["run_id"] == "r1"  # dict -> JSON
    assert rows[1][0] == "legacy-tag"  # str -> verbatim


# ---- web cost report: attribution + cache + spikes ------------------------


def test_web_cost_report_attributes_and_flags(tmp_path):
    s = Settings(
        api_key="k",
        ledger_path=tmp_path / "ledger.db",
        daily_usd_ceiling=100,
        total_usd_ceiling=100,
    )
    g = SpendGuard(s)
    # one normal cached call, one routing spike (huge $/Mtok, tiny output)
    g.record(
        "m",
        1000,
        50,
        0.0001,
        meta={"run_id": "r1", "stage": "bom", "cached_tokens": 900, "provider": "DeepSeek"},
    )
    g.record(
        "m",
        1000,
        20,
        0.002,
        meta={"run_id": "r1", "stage": "bom", "cached_tokens": 0, "provider": "Expensive"},
    )

    rows = web_cost_report.load_rows(str(s.ledger_path))
    summary = web_cost_report.summarize(rows, spike_threshold=0.50)
    assert summary["total"]["calls"] == 2
    assert summary["total"]["spikes"] == 1  # the $2/Mtok call
    assert "r1" in summary["runs"]
    # cache hit-rate = cached/input over the run = 900 / 2000 = 45%
    run = summary["runs"]["r1"]
    assert round(run["cached"] / run["input"] * 100) == 45
    assert "bom" in summary["run_stage"]["r1"]
    # report renders without error
    assert "cache hit-rate" in web_cost_report.format_report(summary, by="stage")


def test_web_cost_report_legacy_rows_cluster_by_time(tmp_path):
    s = Settings(
        api_key="k",
        ledger_path=tmp_path / "ledger.db",
        daily_usd_ceiling=100,
        total_usd_ceiling=100,
    )
    g = SpendGuard(s)
    g.record("m", 100, 10, 0.0001, meta="tools")  # legacy bare-string rows
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
                return {
                    "spent_total_usd": 0.0,
                    "daily_remaining_usd": 5.0,
                    "daily_ceiling_usd": 5.0,
                }

        self.guard = _G()

    def chat(
        self,
        messages,
        max_tokens=4096,
        temperature=0.2,
        progress=None,
        meta_ctx=None,
        reasoning=None,
        reasoning_guard=None,
        collection_bounds=(),
        response_format=None,
    ):
        self.max_tokens_seen.append(max_tokens)
        self.reasoning_seen.append(reasoning)
        self._n += 1
        if self._n == 1:
            return {
                "text": '{ "goal": "x", truncated',
                "cost_usd": 0.0,
                "reasoning": "",
                "finish_reason": "length",
            }
        return {"text": self._ok, "cost_usd": 0.0, "reasoning": "", "finish_reason": "stop"}


def test_truncated_reply_triggers_one_fixed_cap_serialization_call(tmp_path):
    ok = json.dumps(
        {
            "goal": "a USB-powered LED",
            "constraints": [],
            "named_parts": [],
            "inferred_expertise": "intermediate",
            "assumptions": [],
            "project_stem": "USB_LED",
        }
    )
    client = _TruncThenOkClient(ok)
    res = run_session(tmp_path, "a USB-powered LED", ["intent"], client=client)
    assert res["status"] == "ok"  # recovered, committed
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
        return _FakeResp(
            [
                {"choices": [{"delta": {"content": '{"x":1}'}}]},
                {"choices": [{"finish_reason": "stop", "delta": {}}]},
                _usage_chunk(cached=800),
            ]
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    c.chat([{"role": "user", "content": "hi"}], max_tokens=8192, reasoning={"enabled": False})
    rec = c.guard.records[-1]
    assert rec["meta"]["max_tokens"] == 8192
    assert rec["meta"]["reasoning_policy"] == {"enabled": False}
    assert rec["meta"]["content_chars"] == len('{"x":1}')
    # the payload carries the control keys only, never _meta/_meta_ctx
    assert "_meta" not in captured["payload"] and "_meta_ctx" not in captured["payload"]


def test_chat_returns_completion_telemetry(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        return _FakeResp(
            [
                {"choices": [{"delta": {"content": '{"x":1}'}}]},
                {"choices": [{"finish_reason": "stop", "delta": {}}]},
                _usage_chunk(cached=800),
            ]
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    r = c.chat([{"role": "user", "content": "hi"}], max_tokens=4096, reasoning={"max_tokens": 2048})
    assert r["provider"] == "DeepSeek"
    assert r["usage"]["prompt_tokens"] == 1000
    assert r["usage"]["completion_tokens"] == 50
    assert r["max_tokens"] == 4096
    assert r["reasoning_policy"] == {"max_tokens": 2048}
    assert r["content_chars"] == len('{"x":1}')
    assert r["finish_reason"] == "stop"


def test_chat_with_tools_rounds_carry_telemetry(monkeypatch):
    client = CappedOpenRouterClient(
        settings=types.SimpleNamespace(), guard=types.SimpleNamespace(status=lambda: {})
    )

    def fake_stream(body, on_delta=None):
        return (
            {
                "role": "assistant",
                "content": '{"ok": true}',
                "finish_reason": "stop",
                "provider": "DeepSeek",
                "usage": {"prompt_tokens": 5},
                "requested_max_tokens": body.get("max_tokens"),
                "reasoning_policy": body.get("reasoning"),
            },
            0.0,
        )

    monkeypatch.setattr(client, "_stream", fake_stream)
    r = client.chat_with_tools(
        [{"role": "user", "content": "go"}],
        tools=[],
        executor=lambda n, a: "ok",
        max_rounds=1,
        max_tokens=16384,
        reasoning={"enabled": False},
    )
    assert r["provider"] == "DeepSeek"
    assert r["usage"] == {"prompt_tokens": 5}
    assert r["max_tokens"] == 16384
    assert r["reasoning_policy"] == {"enabled": False}
    assert r["finish_reason"] == "stop"


def test_tool_capable_first_response_is_schema_bound_and_can_finish_directly(monkeypatch):
    client = CappedOpenRouterClient(
        settings=types.SimpleNamespace(), guard=types.SimpleNamespace(status=lambda: {})
    )
    calls = []

    def fake_stream(body, on_delta=None):
        calls.append(body)
        return (
            {
                "role": "assistant",
                "content": '{"groups": []}',
                "finish_reason": "stop",
            },
            0.0,
        )

    monkeypatch.setattr(client, "_stream", fake_stream)
    response_format = {"type": "json_schema", "json_schema": {"name": "bom"}}
    result = client.chat_with_tools(
        [{"role": "user", "content": "go"}],
        tools=[],
        executor=lambda n, a: "ok",
        max_rounds=2,
        response_format=response_format,
    )

    assert result["rounds"] == 1
    assert calls[0]["tool_choice"] == "auto"
    assert calls[0]["response_format"] is response_format


def test_chat_with_tools_forced_final_carries_telemetry(monkeypatch):
    client = CappedOpenRouterClient(
        settings=types.SimpleNamespace(), guard=types.SimpleNamespace(status=lambda: {})
    )

    def fake_stream(body, on_delta=None):
        return (
            {
                "role": "assistant",
                "content": None,
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "list_parts", "arguments": "{}"},
                    }
                ],
                "requested_max_tokens": body.get("max_tokens"),
                "reasoning_policy": body.get("reasoning"),
            },
            0.0,
        )

    monkeypatch.setattr(client, "_stream", fake_stream)
    r = client.chat_with_tools(
        [{"role": "user", "content": "go"}],
        tools=[],
        executor=lambda n, a: "ok",
        max_rounds=1,
        max_tokens=16384,
        reasoning=None,
    )
    assert r.get("forced_final") is True  # budget exhausted -> cold final
    assert r["max_tokens"] == 16384  # the final round still carries the cap
    assert r["reasoning_policy"] is None
    assert r["finish_reason"] == "tool_calls"


# ---- ERC-recovery offender parsing (web) ----------------------------------


def _write_synth_check(tmp_path, checks):
    (tmp_path / ".kicraft").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".kicraft" / "synthesis_check.json").write_text(
        json.dumps({"status": "failed", "checks": checks}), encoding="utf-8"
    )


def test_erc_offenders_returns_failed_erc_errors(tmp_path):
    from kicraft.server.web import _erc_offenders

    _write_synth_check(
        tmp_path,
        [
            {"name": "9.2 footprints non-empty", "ok": True, "offenders": []},
            {
                "name": "9.12 ERC",
                "ok": False,
                "offenders": ["root: Pin U1.3 not connected", "MCU: conflicting outputs"],
            },
        ],
    )
    assert _erc_offenders(tmp_path) == ["root: Pin U1.3 not connected", "MCU: conflicting outputs"]


def test_erc_offenders_empty_when_erc_clean(tmp_path):
    from kicraft.server.web import _erc_offenders

    _write_synth_check(tmp_path, [{"name": "9.12 ERC", "ok": True, "offenders": []}])
    assert _erc_offenders(tmp_path) == []  # nothing to recover
    assert _erc_offenders(tmp_path / "nope") == []  # missing file -> []


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
            raise requests.exceptions.HTTPError(f"{self.status_code} {self.reason}", response=self)

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

    assert holder["resp"].encoding == "utf-8"  # _open_stream pinned it
    assert "12 µF" in msg["content"]  # decoded on the wire bytes
    assert "4.7 kΩ" in msg["content"]
    assert "10 °C" in msg["content"]
    assert "Â" not in msg["content"]  # no double-encoding


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

    def _fake_open(payload, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            return _BrokenMidStreamResp([{"choices": [{"delta": {"content": "par"}}]}])
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
        lambda payload, **kwargs: _BrokenMidStreamResp(
            [{"choices": [{"delta": {"content": "par"}}]}]
        ),
    )
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        c._stream({"messages": [{"role": "user", "content": "x"}]})


def test_stream_returns_in_band_429_to_top_level_without_retry(monkeypatch):
    settings = Settings(api_key="k", llm_max_retries=1, llm_retry_backoff_s=0.0)
    guard = _RecordingGuard()
    client = CappedOpenRouterClient(settings, guard=guard)
    attempts = []

    def open_stream(payload, **kwargs):
        attempts.append(1)
        return _FakeResp(
            [
                {"choices": [{"delta": {"content": "partial"}}]},
                {"error": {"code": 429, "message": "rate limited"}},
            ]
        )

    monkeypatch.setattr(client, "_open_stream", open_stream)
    with pytest.raises(requests.exceptions.HTTPError):
        client._stream({"messages": [{"role": "user", "content": "x"}]})
    assert len(attempts) == 1
    assert guard.records == []


@pytest.mark.parametrize("status", [400, 401, 403])
def test_stream_does_not_retry_in_band_authenticated_or_other_4xx(monkeypatch, status):
    client = CappedOpenRouterClient(
        Settings(api_key="k", llm_max_retries=2, llm_retry_backoff_s=0.0),
        guard=_RecordingGuard(),
    )
    attempts = []

    def open_stream(payload, **kwargs):
        attempts.append(1)
        return _FakeResp([{"error": {"code": status, "message": "rejected"}}])

    monkeypatch.setattr(client, "_open_stream", open_stream)
    with pytest.raises(requests.exceptions.HTTPError):
        client._stream({"messages": [{"role": "user", "content": "x"}]})
    assert len(attempts) == 1


# ---- in-stream reasoning-loop breaker (KC-VWW5X7) --------------------------


def test_reasoning_guard_reports_distinct_abort_reasons():
    s = Settings(api_key="k", reasoning_repeat_window=256, reasoning_repeat_threshold=3)
    policy = s.design_reasoning_guard()
    now = client_mod.time.monotonic()
    block = "a" * 255 + "Z"

    assert (
        CappedOpenRouterClient._reasoning_abort_reason(policy, 20_000, 0, "x" * 4096, now)
        == "hard_ceiling"
    )
    assert (
        CappedOpenRouterClient._reasoning_abort_reason(policy, len(block) * 4, 0, block * 4, now)
        == "repetition"
    )
    assert (
        CappedOpenRouterClient._reasoning_abort_reason(policy, 100, 0, "x" * 100, now - 999)
        == "wall_stall"
    )
    assert (
        CappedOpenRouterClient._reasoning_abort_reason(policy, 99_999, 1, "y" * 4096, now - 999)
        is None
    )


def test_judge_policy_allows_documented_reasoning_range_without_repetition_abort():
    s = Settings(api_key="k")
    policy = s.judge_reasoning_guard()
    assert policy.repetition_enabled is False
    assert (
        CappedOpenRouterClient._reasoning_abort_reason(
            policy,
            23_000 * 4,
            0,
            "repeated judge analysis" * 200,
            client_mod.time.monotonic(),
        )
        is None
    )


def test_stream_aborts_reasoning_ceiling(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        return _FakeResp([{"choices": [{"delta": {"reasoning": "x" * 20_000}}]}])

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    msg, cost = c._stream(
        {
            "messages": [{"role": "user", "content": "x"}],
            "_reasoning_guard": c.s.design_reasoning_guard(),
        }
    )
    assert msg["loop_detected"] is True
    assert msg["finish_reason"] == "reasoning_loop"
    assert msg["loop_abort_reason"] == "hard_ceiling"
    assert msg["reasoning_policy_name"] == "design"
    assert msg["content"] is None
    assert cost > 0.0  # partial stream still recorded against the guard


def test_stream_aborts_reasoning_repetition(monkeypatch):
    block = "a" * 255 + "Z"

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        return _FakeResp([{"choices": [{"delta": {"reasoning": block * 4}}]}])

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    msg, _ = c._stream(
        {
            "messages": [{"role": "user", "content": "x"}],
            "_reasoning_guard": c.s.design_reasoning_guard(),
        }
    )
    assert msg["loop_detected"] is True
    assert msg["finish_reason"] == "reasoning_loop"
    assert msg["loop_abort_reason"] == "repetition"


def test_stream_content_stream_is_not_aborted(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        return _FakeResp(
            [
                {"choices": [{"delta": {"reasoning": "thinking here"}}]},
                {"choices": [{"delta": {"content": '{"x":1}'}}]},
                {"choices": [{"finish_reason": "stop", "delta": {}}]},
                _usage_chunk(),
            ]
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    msg, _ = c._stream({"messages": [{"role": "user", "content": "x"}]})
    assert not msg.get("loop_detected")
    assert msg["finish_reason"] == "stop"
    assert msg["content"] == '{"x":1}'


# ---- incremental collection bounds ----------------------------------------


def test_collection_guard_counts_direct_members_across_chunks_and_nested_json():
    guard = _StreamingCollectionGuard((CollectionBound(field="parts", total=2),))
    chunks = [
        '{"pa',
        'rts":[{"ref":"R1","value":"escaped \\" [, }"},',
        '{"nested":{"items":[1,2,3]}},',
        '{"ref":"R3"}],"other":[]}',
    ]
    accepted = ""
    overflow = None
    for chunk in chunks:
        piece, overflow = guard.consume(chunk)
        accepted += piece
        if overflow:
            break
    assert overflow == {
        "field": "parts",
        "observed_count": 3,
        "configured_total": 2,
    }
    assert '"ref":"R3"' not in accepted


def test_collection_guard_accepts_empty_array_and_ignores_malformed_string_tail():
    empty = _StreamingCollectionGuard((CollectionBound(field="parts", total=1),))
    accepted, overflow = empty.consume('{"parts":[]}')
    assert accepted == '{"parts":[]}' and overflow is None

    malformed = _StreamingCollectionGuard((CollectionBound(field="parts", total=1),))
    text = '{"parts":[{"value":"unterminated }, [ ,'
    accepted, overflow = malformed.consume(text)
    assert accepted == text and overflow is None


def test_stream_collection_guard_ignores_tool_call_arguments(monkeypatch):
    arguments = '{"parts":[1,2,3]}'

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        return _FakeResp(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "function": {"name": "lookup", "arguments": arguments},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {"choices": [{"finish_reason": "tool_calls", "delta": {}}]},
                _usage_chunk(),
            ]
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    msg, _ = c._stream(
        {
            "messages": [{"role": "user", "content": "x"}],
            "_collection_bounds": (CollectionBound(field="parts", total=2),),
        }
    )
    assert msg["finish_reason"] == "tool_calls"
    assert msg.get("collection_limit") is None
    assert msg["tool_calls"][0]["function"]["arguments"] == arguments


def test_collection_guard_enforces_per_group_bound_without_full_response_copy():
    guard = _StreamingCollectionGuard(
        (CollectionBound(field="parts", total=5, per_group=2, group_key="sheet"),)
    )
    text = (
        '{"parts":['
        '{"ref":"R1","sheet":"ARRAY"},'
        '{"ref":"R2","sheet":"ARRAY"},'
        '{"ref":"R3","sheet":"ARRAY"}'
        "]}"
    )
    accepted, overflow = guard.consume(text)
    assert overflow == {
        "field": "parts",
        "observed_count": 3,
        "configured_total": 2,
        "limit_scope": "group",
        "group_key": "sheet",
        "group_value": "ARRAY",
    }
    assert accepted.endswith('{"ref":"R3","sheet":"ARRAY"}')


def test_collection_guard_stops_on_duplicate_single_key():
    guard = _StreamingCollectionGuard(
        (CollectionBound(field="groups", total=500, unique_keys=("id",)),)
    )
    text = '{"groups":[{"id":"header"},{"id":"header"},{"id":"unreachable"}]}'

    accepted, overflow = guard.consume(text)

    assert overflow == {
        "field": "groups",
        "observed_count": 2,
        "configured_total": 500,
        "limit_scope": "duplicate",
        "unique_keys": ["id"],
        "duplicate_values": ["header"],
    }
    assert accepted.endswith('{"id":"header"}')


def test_collection_guard_without_identity_allows_repeated_members():
    text = '{"connections":[{"from_block":"A"},{"from_block":"A"}]}'
    guard = _StreamingCollectionGuard((CollectionBound(field="connections", total=2),))
    assert guard.consume(text) == (text, None)


@pytest.mark.parametrize("reply_index", [0, 1])
@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_collection_guard_stops_captured_functional_repetition(reply_index, chunk_size):
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures/stage_reliability/functional_spec_round3_20260911.json"
        ).read_text()
    )["replies"][reply_index]
    guard = _StreamingCollectionGuard(STAGE_COLLECTION_BOUNDS["functional_spec"])
    text = fixture["text"]
    pieces = []
    overflow = None
    for start in range(0, len(text), chunk_size):
        accepted, overflow = guard.consume(text[start : start + chunk_size])
        pieces.append(accepted)
        if overflow is not None:
            break

    assert overflow == {
        "field": "connections",
        "observed_count": fixture["first_duplicate_count"],
        "configured_total": 128,
        "limit_scope": "duplicate",
        "unique_keys": ["from_block", "to_block", "signal_type"],
        "duplicate_values": fixture["duplicate_values"],
    }
    assert "".join(pieces).rstrip() == text[: fixture["duplicate_end_chars"]]
    assert overflow["observed_count"] < 128


@pytest.mark.parametrize("chunk_size", [1, 11, 4096])
def test_compound_identity_preserves_fanout_direction_type_and_escaped_strings(chunk_size):
    text = (
        r'{"nested":{"connections":[{"from_block":"A","to_block":"B","signal_type":"digital"}]},'
        r'"connec\u0074ions":['
        r'{"from_block":"A","to_block":"B","signal_type":"digital","description":"escaped \" [, }"},'
        r'{"from_block":"A","to_block":"C","signal_type":"digital"},'
        r'{"from_block":"B","to_block":"A","signal_type":"digital"},'
        r'{"from_block":"A","to_block":"B","signal_type":"power"},'
        r'{"from_block":"A\\B","to_block":"C","signal_type":"bus"},'
        r'{"from_block":"A","to_block":"B\\C","signal_type":"bus"},'
        r'{"from_block":"A\"","to_block":"B","signal_type":"digital"}'
    )
    guard = _StreamingCollectionGuard(STAGE_COLLECTION_BOUNDS["functional_spec"])
    for start in range(0, len(text), chunk_size):
        chunk = text[start : start + chunk_size]
        accepted, overflow = guard.consume(chunk)
        assert accepted == chunk
        assert overflow is None
    # Key escapes, value escapes, reordered fields and a different description
    # must still identify the original A -> B digital flow.
    duplicate = (
        r',{"description":"second physical signal","to_block":"\u0042",'
        r'"signal_type":"dig\u0069tal","from_\u0062lock":"\u0041"}]}'
    )
    overflow = None
    for start in range(0, len(duplicate), chunk_size):
        _accepted, overflow = guard.consume(duplicate[start : start + chunk_size])
        if overflow is not None:
            break
    assert overflow["observed_count"] == 8
    assert overflow["limit_scope"] == "duplicate"
    assert overflow["duplicate_values"] == ["A", "B", "digital"]


def test_compound_identity_does_not_invent_missing_member_keys():
    text = (
        '{"connections":[{"from_block":"A"},{"from_block":"A"},'
        '{"from_block":"A","to_block":"B"},'
        '{"from_block":"A","to_block":"B","signal_type":"digital"}]}'
    )
    guard = _StreamingCollectionGuard(STAGE_COLLECTION_BOUNDS["functional_spec"])
    assert guard.consume(text) == (text, None)
    # Structural validation, not the lexer, owns these malformed members.
    from kicraft.server.stage_contracts import StageSchemaError, _normalize_stage_response

    payload = json.loads(text)
    payload["blocks"] = [
        {"name": name, "category": "process", "purpose": "Signal processing"} for name in ("A", "B")
    ]
    with pytest.raises(StageSchemaError):
        _normalize_stage_response("functional_spec", payload, {})


def test_stream_duplicate_abort_is_charged_and_never_completed_or_pruned(monkeypatch):
    text = (
        '{"connections":['
        '{"from_block":"A","to_block":"B","signal_type":"digital","description":"first"},'
        '{"from_block":"A","to_block":"B","signal_type":"digital","description":"second"}]}'
    )
    response = _FakeResp([{"choices": [{"delta": {"content": text}}]}])
    monkeypatch.setattr(client_mod.requests, "post", lambda *a, **kw: response)
    spend = _RecordingGuard()
    client = CappedOpenRouterClient(Settings(api_key="k"), guard=spend)
    msg, cost = client._stream(
        {
            "messages": [{"role": "user", "content": "Return functional flows"}],
            "_collection_bounds": STAGE_COLLECTION_BOUNDS["functional_spec"],
        }
    )

    assert msg["finish_reason"] == "collection_limit"
    assert msg["collection_limit"]["limit_scope"] == "duplicate"
    assert msg["collection_limit"]["unique_keys"] == ["from_block", "to_block", "signal_type"]
    assert msg["collection_limit"]["duplicate_values"] == ["A", "B", "digital"]
    assert '"description":"second"' in msg["content"]
    with pytest.raises(json.JSONDecodeError):
        json.loads(msg["content"])
    assert len(spend.records) == 1
    charged = spend.records[0]
    assert charged["cost"] == cost > 0
    assert charged["in"] > 0 and charged["out"] > 0
    assert charged["meta"]["collection_limit"] == msg["collection_limit"]
    assert charged["meta"]["bounded_collection_completed"] is False


def _property_response_format():
    return {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {
                    "inter_sheet_net_ranges": {"type": "array", "items": {"$ref": "#/$defs/range"}},
                    "dictionary": {"type": "object", "additionalProperties": True},
                    "choice": {
                        "anyOf": [
                            {"properties": {"left": True}, "additionalProperties": False},
                            {"properties": {"right": True}, "additionalProperties": False},
                        ]
                    },
                },
                "additionalProperties": False,
                "$defs": {
                    "range": {
                        "properties": {"net_class": True, "sheet": True},
                        "additionalProperties": False,
                    }
                },
            }
        },
    }


@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_property_guard_aborts_recorded_nested_unknown_key(chunk_size):
    text = '{"inter_sheet_net_ranges":[{"start_sheet":"root","end_sheet":"leds",'
    text += '"net_class_net_class":"x",' * 200
    guard = _StreamingCollectionGuard((), _property_response_format())
    accepted_parts = []
    violation = None
    for offset in range(0, len(text), chunk_size):
        accepted, violation = guard.consume(text[offset : offset + chunk_size])
        accepted_parts.append(accepted)
        if violation:
            break
    assert violation == {
        "limit_scope": "property",
        "field": "$.inter_sheet_net_ranges[0]",
        "property": "start_sheet",
    }
    assert "".join(accepted_parts) == '{"inter_sheet_net_ranges":[{"start_sheet'


@pytest.mark.parametrize("chunk_size", [1, 5, 4096])
def test_property_guard_preserves_refs_escapes_dictionaries_and_union_branches(chunk_size):
    text = (
        r'{"inter_sheet_net_ranges":[{"net_\u0063lass":"escaped \" } [","sheet":1}],'
        r'"dictionary":{"arbitrary":{"also\"arbitrary":true}},'
        r'"choice":{"right":{"unconstrained":null}}}'
    )
    guard = _StreamingCollectionGuard((), _property_response_format())
    for offset in range(0, len(text), chunk_size):
        chunk = text[offset : offset + chunk_size]
        assert guard.consume(chunk) == (chunk, None)


def test_property_guard_leaves_no_schema_calls_permissive():
    text = '{"unknown":[{"anything":true}]}'
    assert _StreamingCollectionGuard(()).consume(text) == (text, None)


@pytest.mark.parametrize("with_tools", [False, True])
def test_stream_property_abort_is_paid_and_never_completed(monkeypatch, with_tools):
    text = '{"inter_sheet_net_ranges":[{"start_sheet":"root","net_class_net_class":1}]}'
    response = _FakeResp([{"choices": [{"delta": {"content": text}}]}])
    monkeypatch.setattr(client_mod.requests, "post", lambda *a, **kw: response)
    spend = _RecordingGuard()
    client = CappedOpenRouterClient(Settings(api_key="k"), guard=spend)
    call = client.chat_with_tools if with_tools else client.chat
    kwargs = {"tools": [], "executor": lambda name, args: ""} if with_tools else {}
    result = call(
        [{"role": "user", "content": "Return ranges"}],
        response_format=_property_response_format(),
        **kwargs,
    )
    assert result["finish_reason"] == "collection_limit"
    assert result["collection_limit"]["property"] == "start_sheet"
    assert "net_class_net_class" not in result["text"]
    with pytest.raises(json.JSONDecodeError):
        json.loads(result["text"])
    assert spend.records[0]["cost"] > 0
    assert spend.records[0]["in"] > 0 and spend.records[0]["out"] > 0
    assert spend.records[0]["meta"]["bounded_collection_completed"] is False


def _guarded_json_stream(monkeypatch, text, chunk_size, with_tools=False, **kwargs):
    chunks = [text[offset : offset + chunk_size] for offset in range(0, len(text), chunk_size)]
    response = _FakeResp(
        [{"choices": [{"delta": {"content": chunk}}]} for chunk in chunks]
        + [{"choices": [{"delta": {}, "finish_reason": "stop"}]}, _usage_chunk()]
    )
    monkeypatch.setattr(client_mod.requests, "post", lambda *a, **kw: response)
    spend = _RecordingGuard()
    client = CappedOpenRouterClient(Settings(api_key="k"), guard=spend)
    call = client.chat_with_tools if with_tools else client.chat
    if with_tools:
        kwargs.update(tools=[], executor=lambda name, args: "")
    deltas = []
    result = call(
        [{"role": "user", "content": "Return one JSON object"}],
        response_format=kwargs.pop("response_format", {"type": "json_object"}),
        progress=deltas.append,
        **kwargs,
    )
    assert response.closed
    assert len(spend.records) == 1
    return result, spend.records[0], response, deltas


@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 17, 4096])
def test_stream_rejects_captured_can_syntax_at_first_irreparable_character(
    monkeypatch, with_tools, chunk_size
):
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures/stage_reliability/architecture_can_syntax_round8_20260911.json"
        ).read_text()
    )
    prefix = fixture["prefix"]
    text = prefix + "UNREACHABLE" * 500
    result, paid, response, deltas = _guarded_json_stream(
        monkeypatch, text, chunk_size, with_tools
    )
    offset = fixture["character_offset"]
    assert result["finish_reason"] == "collection_limit"
    limit = result["collection_limit"]
    assert limit["limit_scope"] == "syntax"
    assert limit["character_offset"] == offset == 3034
    assert limit["field"] == "$.inter_sheet_nets"
    assert limit["syntax_error"].startswith("Expected ','")
    assert result["text"] == prefix[:offset]
    assert "".join(
        delta["text"] for delta in deltas if delta["kind"] == "answer_delta"
    ) == prefix[:offset]
    assert response.lines_read == offset // chunk_size + 1
    received_chars = min(len(text), response.lines_read * chunk_size)
    assert paid["out"] == max(1, received_chars // 4)
    assert paid["in"] > 0 and paid["cost"] == result["cost_usd"] > 0
    assert paid["meta"]["collection_limit"] == limit
    assert paid["meta"]["bounded_collection_completed"] is False


@pytest.mark.parametrize(
    ("prefix", "bad", "error"),
    [
        ('{"a":{}', "{", "Expected ','"),
        ('{"a":1 ', '"', "Expected ','"),
        ('{"a"', "1", "Expected ':'"),
        ('{"a":', "}", "Expected a JSON value"),
        ('{"a":1,', "}", "Expected a quoted object key"),
        ('{"a":[1,', "]", "Expected a JSON value"),
        ('{"a":[1 ', "2", "Expected ','"),
        ('{"a":[', "}", "Expected a JSON value"),
        ('{"a":tru', "x", "Invalid JSON literal"),
        ('{"a":fals', "}", "Invalid JSON literal"),
        ('{"a":nul', " ", "Invalid JSON literal"),
        ('{"a":0', "1", "Invalid JSON number"),
        ('{"a":-', "}", "Invalid JSON number"),
        ('{"a":1.', "}", "Invalid JSON number"),
        ('{"a":1e+', "]", "Invalid JSON number"),
        ('{"a":1', "x", "Invalid JSON number"),
        ('{"a":"\\', "x", "Invalid JSON string escape"),
        ('{"a":"\\u12', "x", "Expected a hexadecimal digit"),
        ('{"a":"', "\n", "Unescaped control character"),
        ('{"a":1}', "{", "Unexpected content"),
        ('{"a":true,"a', '"', "Duplicate object key"),
        (r'{"a":1,"\u0061', '"', "Duplicate object key"),
        (r'{"\u0061":1,"a', '"', "Duplicate object key"),
        (r'{"😀":1,"\ud83d\ude00', '"', "Duplicate object key"),
        ('{"dictionary":{"x":1,"x', '"', "Duplicate object key"),
    ],
)
@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
def test_stream_syntax_abort_preserves_only_repairable_prefix(
    monkeypatch, prefix, bad, error, chunk_size
):
    result, paid, response, _ = _guarded_json_stream(
        monkeypatch, prefix + bad + "unreachable" * 500, chunk_size
    )
    assert result["finish_reason"] == "collection_limit"
    assert result["text"] == prefix
    assert result["collection_limit"]["limit_scope"] == "syntax"
    assert result["collection_limit"]["syntax_error"].startswith(error)
    assert result["collection_limit"]["character_offset"] == len(prefix)
    assert response.lines_read == len(prefix) // chunk_size + 1
    assert paid["cost"] > 0 and paid["in"] > 0 and paid["out"] > 0
    assert paid["meta"]["bounded_collection_completed"] is False


def test_syntax_abort_does_not_execute_previously_streamed_tool_calls(monkeypatch):
    tool_call = {
        "index": 0,
        "id": "pending",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"query":"STM32"}'},
    }
    response = _FakeResp(
        [
            {"choices": [{"delta": {"tool_calls": [tool_call]}}]},
            {"choices": [{"delta": {"content": '{"a":1}{}'}}]},
            _usage_chunk(),
        ]
    )
    monkeypatch.setattr(client_mod.requests, "post", lambda *a, **kw: response)
    spend = _RecordingGuard()
    client = CappedOpenRouterClient(Settings(api_key="k"), guard=spend)
    result = client.chat_with_tools(
        [{"role": "user", "content": "Return JSON"}],
        tools=[],
        executor=lambda *a: pytest.fail("A syntax-stopped draft must not execute tools"),
        response_format={"type": "json_object"},
    )
    assert result["finish_reason"] == "collection_limit"
    assert result["collection_limit"]["limit_scope"] == "syntax"
    assert result["rounds"] == 1
    assert response.closed and response.lines_read == 2
    assert len(spend.records) == 1 and spend.records[0]["cost"] > 0


@pytest.mark.parametrize("chunk_size", [1, 5, 4096])
@pytest.mark.parametrize(
    "wrapper",
    [
        "{}",
        " \n{}\t",
        "```json\n{}\n```",
        "```{}\n```",
        "Here is the result:\n{}",
        "Here is the result:\n```json\n{}\n```\nEnd of result.",
    ],
)
def test_stream_accepts_chunked_nested_json_and_fences(monkeypatch, chunk_size, wrapper):
    value = (
        r'{"dictionary":{"arbitrary":[{},[],true,false,null,0,-0,123,-42,'
        r'0.25,-1.25e-10,1E+2,1e0,"quote\"slash\\solidus\/\b\f\n\r\t\u0041",'
        r'{"\u0061":1},{"a":2}]},"choice":{"right":{"unconstrained":null}}}'
    )
    text = wrapper.format(value)
    result, paid, response, _ = _guarded_json_stream(
        monkeypatch, text, chunk_size, response_format=_property_response_format()
    )
    assert result["finish_reason"] == "stop"
    assert result.get("collection_limit") is None
    assert result["text"] == text
    from kicraft.server.stage_contracts import _extract_json

    assert _extract_json(result["text"]) == json.loads(value)
    assert response.lines_read == (len(text) + chunk_size - 1) // chunk_size + 3
    assert paid["cost"] == 0.001


@pytest.mark.parametrize(
    "text",
    [
        '{"a":tru',
        '{"a":fals',
        '{"a":nul',
        '{"a":-',
        '{"a":1.',
        '{"a":1e',
        '{"a":1e+',
        '{"a":"\\',
        '{"a":"\\u12',
        '{"a":[{"b":',
        '```j',
        '```json\n{"a":1}\n``',
    ],
)
def test_stream_does_not_abort_incomplete_but_repairable_json(monkeypatch, text):
    result, paid, _, _ = _guarded_json_stream(monkeypatch, text, 1)
    assert result["finish_reason"] == "stop"
    assert result.get("collection_limit") is None
    assert result["text"] == text
    assert paid["cost"] == 0.001


def test_stream_oversized_keys_remain_syntax_checked_and_conservative(monkeypatch):
    key = "x" * 5000
    text = json.dumps({key: {key: [True, None, 1.2e-3]}})
    result, _, _, _ = _guarded_json_stream(
        monkeypatch, text, 7, response_format=_property_response_format()
    )
    assert result["finish_reason"] == "stop"
    assert result["text"] == text
    # Dropping the oversized key buffer must not disable string escape grammar.
    malformed = '{"' + key + "\\u12"
    result, paid, _, _ = _guarded_json_stream(monkeypatch, malformed + "X", 7)
    assert result["collection_limit"]["limit_scope"] == "syntax"
    assert result["text"] == malformed
    assert paid["cost"] > 0


def test_syntax_abort_never_salvages_a_bounded_wiring_prefix(monkeypatch):
    text = '{"pins":[{"ref":"J1","pin":"1","net":"GND"}]{}'
    result, paid, _, _ = _guarded_json_stream(
        monkeypatch, text, 1, collection_bounds=(CollectionBound(field="pins", total=1),)
    )
    assert result["finish_reason"] == "collection_limit"
    assert result["collection_limit"]["limit_scope"] == "syntax"
    assert result["text"] == '{"pins":[{"ref":"J1","pin":"1","net":"GND"}]'
    assert paid["meta"]["bounded_collection_completed"] is False


def test_collection_guard_accepts_empty_and_large_in_bound_arrays():
    guard = _StreamingCollectionGuard((CollectionBound(field="parts", total=500),))
    text = '{"parts":[' + ",".join(f'{{"ref":"D{i}"}}' for i in range(400)) + "]}"
    accepted, overflow = guard.consume(text)
    assert overflow is None
    assert accepted == text


def test_stream_collection_limit_stops_before_overflow_object(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        return _FakeResp(
            [
                {"choices": [{"delta": {"content": '{"parts":[{"ref":"R1"},'}}]},
                {"choices": [{"delta": {"content": '{"ref":"R2"},{"ref":"R3"}]}'}}]},
            ]
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())
    msg, cost = c._stream(
        {
            "messages": [{"role": "user", "content": "x"}],
            "_collection_bounds": (CollectionBound(field="parts", total=2),),
        }
    )
    assert msg["finish_reason"] == "collection_limit"
    assert msg["collection_limit"] == {
        "field": "parts",
        "observed_count": 3,
        "configured_total": 2,
        "emitted_content_chars": len('{"parts":[{"ref":"R1"},{"ref":"R2"},'),
    }
    assert '"ref":"R3"' not in msg["content"]
    assert cost > 0.0


def test_stream_closes_exact_wiring_unit_prefix_at_collection_bound(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        return _FakeResp(
            [
                {
                    "choices": [
                        {
                            "delta": {
                                "content": (
                                    '{"pins":[{"ref":"J1","pin":"1","net":"D0"},'
                                    '{"ref":"J1","pin":"2","net":"D1"},'
                                )
                            }
                        }
                    ]
                },
                {"choices": [{"delta": {"content": '{"ref":"J1","pin":"1","net":"D0"}]}'}}]},
            ]
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(Settings(api_key="k"), guard=_RecordingGuard())

    msg, _cost = c._stream(
        {
            "messages": [{"role": "user", "content": "x"}],
            "_collection_bounds": (CollectionBound(field="pins", total=2),),
        }
    )

    assert msg["finish_reason"] == "stop"
    assert json.loads(msg["content"]) == {
        "pins": [
            {"ref": "J1", "pin": "1", "net": "D0"},
            {"ref": "J1", "pin": "2", "net": "D1"},
        ]
    }


def test_stream_cost_estimate_uses_selected_provider_prices():
    c = CappedOpenRouterClient(
        Settings(
            api_key="k",
            max_price_prompt=0.05,
            max_price_completion=0.16,
        ),
        guard=_RecordingGuard(),
    )

    assert c._estimated_cost("unknown", 10_000, 1_000) == pytest.approx(0.00066)


def _clear_profile_env(monkeypatch):
    for name in (
        "KICRAFT_DESIGN_PROFILE",
        "KICRAFT_ESCALATION_PROFILE",
        "KICRAFT_PROVIDER_FALLBACK_PROFILE",
        "KICRAFT_MODEL",
        "KICRAFT_PROVIDER_ORDER",
        "KICRAFT_MAX_PRICE_PROMPT",
        "KICRAFT_MAX_PRICE_COMPLETION",
        "KICRAFT_STAGE_OUTPUT_LIMITS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")


@pytest.mark.parametrize("recovery", [False, True])
def test_stage_output_ceiling_admits_bounded_call_without_raising_project_budget(
    tmp_path, monkeypatch, recovery
):
    from kicraft.server.spend_guard import BudgetExceeded
    from kicraft.server.stage_runtime import _response_policy

    _clear_profile_env(monkeypatch)
    monkeypatch.setenv("KICRAFT_DESIGN_PROFILE", "luna")
    monkeypatch.setenv("KICRAFT_STAGE_OUTPUT_LIMITS", '{"architecture":8192}')
    settings = Settings.from_env(dotenv=False)
    settings.ledger_path = tmp_path / "ledger.db"
    # A project budget that sits between the two call ceilings under the active
    # profile's price caps (ceilings for 60k chars + 16384/8192 out are
    # $0.02266 / $0.01283): the caller's unbounded 16384 request must be refused,
    # the policy-bounded (8192) call must still fit.
    settings.project_llm_budget_usd = 0.025
    guard = SpendGuard(settings)
    guard.record(settings.model, 100, 100, 0.007, meta={"run_id": "bounded", "stage": "intent"})
    client = CappedOpenRouterClient(settings, guard=guard)
    messages = [{"role": "user", "content": "x" * 60000}]
    monkeypatch.setattr(
        client_mod.requests,
        "post",
        lambda *a, **k: _FakeResp(
            [
                {"choices": [{"delta": {"content": "{}"}, "finish_reason": "stop"}]},
                _usage_chunk(cost=0.001),
            ]
        ),
    )
    with pytest.raises(BudgetExceeded):
        client.chat(messages, max_tokens=16384, meta_ctx={"run_id": "bounded"})
    policy = _response_policy(client, "architecture", 16384)
    output_cap = policy.serialization_max_tokens if recovery else policy.normal_max_tokens
    client.chat(messages, max_tokens=output_cap, meta_ctx={"run_id": "bounded"})

    assert guard.spent_for_run("bounded") == pytest.approx(0.008)
    with pytest.raises(BudgetExceeded) as refused:
        guard.preflight(call_ceiling_usd=0.02, run_id="bounded")
    assert refused.value.limit_usd == 0.025


@pytest.mark.parametrize(
    "limits",
    [
        {"architecture": True},
        {"unknown_stage": 8192},
        {"architecture": 0},
        {"architecture": 16385},
    ],
)
def test_stage_output_ceiling_rejects_nonbinding_configuration(monkeypatch, limits):
    _clear_profile_env(monkeypatch)
    monkeypatch.setenv("KICRAFT_STAGE_OUTPUT_LIMITS", json.dumps(limits))
    with pytest.raises(SystemExit, match="KICRAFT_STAGE_OUTPUT_LIMITS"):
        Settings.from_env(dotenv=False)


def test_design_profiles_resolve_dated_models_and_finite_caps(monkeypatch):
    _clear_profile_env(monkeypatch)
    for name, expected in DESIGN_PROFILES.items():
        monkeypatch.setenv("KICRAFT_DESIGN_PROFILE", name)
        settings = Settings.from_env(dotenv=False)
        assert settings.model == expected["model"]
        assert settings.provider_order == expected["provider_order"]
        assert settings.max_price_prompt > 0
        assert settings.max_price_completion > 0
        assert settings.provider_allow_fallbacks is False


def test_escalation_profile_defaults_disabled_and_can_be_enabled(monkeypatch):
    _clear_profile_env(monkeypatch)
    settings = Settings.from_env(dotenv=False)
    assert settings.design_profile == "luna"
    assert settings.escalation_profile == ""
    monkeypatch.setenv("KICRAFT_ESCALATION_PROFILE", "deepseek")
    assert Settings.from_env(dotenv=False).escalation_profile == "deepseek"


def test_provider_fallback_profile_defaults_disabled_and_can_be_enabled(monkeypatch):
    _clear_profile_env(monkeypatch)
    settings = Settings.from_env(dotenv=False)
    assert settings.provider_fallback_profile == ""
    monkeypatch.setenv("KICRAFT_PROVIDER_FALLBACK_PROFILE", "deepseek")
    assert Settings.from_env(dotenv=False).provider_fallback_profile == "deepseek"


def test_escalation_profile_same_route_disables_and_unknown_fails(monkeypatch):
    _clear_profile_env(monkeypatch)
    monkeypatch.setenv("KICRAFT_DESIGN_PROFILE", "deepseek")
    assert Settings.from_env(dotenv=False).escalation_profile == ""
    monkeypatch.setenv("KICRAFT_ESCALATION_PROFILE", "unknown")
    with pytest.raises(SystemExit, match="KICRAFT_ESCALATION_PROFILE"):
        Settings.from_env(dotenv=False)


def test_provider_fallback_same_route_disables_and_unknown_fails(monkeypatch):
    _clear_profile_env(monkeypatch)
    monkeypatch.setenv("KICRAFT_DESIGN_PROFILE", "deepseek")
    assert Settings.from_env(dotenv=False).provider_fallback_profile == ""
    monkeypatch.setenv("KICRAFT_PROVIDER_FALLBACK_PROFILE", "unknown")
    with pytest.raises(SystemExit, match="KICRAFT_PROVIDER_FALLBACK_PROFILE"):
        Settings.from_env(dotenv=False)


def test_known_profile_rejects_mixed_model_or_price_cap(monkeypatch):
    _clear_profile_env(monkeypatch)
    monkeypatch.setenv("KICRAFT_DESIGN_PROFILE", "deepseek")
    monkeypatch.setenv("KICRAFT_MAX_PRICE_COMPLETION", "0.14")
    with pytest.raises(SystemExit, match="conflicts with design profile"):
        Settings.from_env(dotenv=False)


# ---- DeepSeek direct backend (deepseek profile) ---------------------------


def test_deepseek_profile_routes_designer_to_deepseek_backend(monkeypatch):
    _clear_profile_env(monkeypatch)
    monkeypatch.setenv("KICRAFT_DESIGN_PROFILE", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-key")
    settings = Settings.from_env(dotenv=False)
    assert settings.design_profile == "deepseek"
    assert settings.backend == "deepseek"
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.model == "deepseek-flash"
    assert settings.provider_order == []
    assert settings.deepseek_api_key == "ds-key"
    assert settings.api_key == "test"  # OpenRouter key still resolved for review/judge


def test_luna_profile_is_the_default_openrouter_route(monkeypatch):
    _clear_profile_env(monkeypatch)
    settings = Settings.from_env(dotenv=False)
    assert settings.design_profile == "luna"
    assert settings.backend == "openrouter"
    assert settings.base_url == "https://openrouter.ai/api/v1"
    assert settings.model == "openai/gpt-5.6-luna"
    assert settings.provider_order == ["openai"]


def test_deepseek_profile_requires_deepseek_api_key(monkeypatch):
    _clear_profile_env(monkeypatch)
    monkeypatch.setenv("KICRAFT_DESIGN_PROFILE", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="DEEPSEEK_API_KEY is not set"):
        Settings.from_env(dotenv=False)


def test_review_and_judge_routes_reset_to_openrouter(monkeypatch):
    _clear_profile_env(monkeypatch)
    monkeypatch.setenv("KICRAFT_DESIGN_PROFILE", "deepseek")
    settings = Settings.from_env(dotenv=False)  # deepseek -> DeepSeek API
    assert settings.backend == "deepseek"
    for routed in (settings.for_review(), settings.for_judge()):
        assert routed.backend == "openrouter"
        assert routed.base_url == "https://openrouter.ai/api/v1"


def test_with_design_profile_switches_backend():
    guard = _RecordingGuard()
    original = CappedOpenRouterClient(
        Settings(
            api_key="or-key",
            deepseek_api_key="ds-key",
            backend="deepseek",
            base_url="https://api.deepseek.com",
            design_profile="deepseek",
            escalation_profile="luna",
        ),
        guard=guard,
    )
    escalated = original.with_design_profile("luna")
    assert escalated.guard is guard
    assert escalated.s.backend == "openrouter"
    assert escalated.s.base_url == "https://openrouter.ai/api/v1"
    assert escalated.s.model == DESIGN_PROFILES["luna"]["model"]


def test_stream_deepseek_omits_openrouter_fields_and_disables_thinking(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = json
        return _FakeResp(
            [
                {"choices": [{"delta": {"content": '{"x":1}'}}]},
                {"choices": [{"finish_reason": "stop", "delta": {}}]},
                {
                    "usage": {
                        "prompt_tokens": 1000,
                        "completion_tokens": 50,
                        "prompt_cache_hit_tokens": 800,
                    }
                },
            ]
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(
        Settings(
            api_key="or-key",
            backend="deepseek",
            deepseek_api_key="ds-key",
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
            provider_order=[],
            max_price_prompt=0.30,
            max_price_completion=1.20,
        ),
        guard=_RecordingGuard(),
    )
    msg, cost = c._stream(
        {
            "messages": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "hi"},
            ],
            "_meta": "stream",
            "_meta_ctx": {"run_id": "r1"},
            "reasoning": {"enabled": False},
        }
    )
    p = captured["payload"]
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer ds-key"
    assert "X-Title" not in captured["headers"]
    assert "provider" not in p
    assert "usage" not in p  # OpenRouter's usage.include field is not sent
    assert "reasoning" not in p
    assert p["thinking"] == {"type": "disabled"}  # DeepSeek defaults to ON; must be explicit
    assert p["messages"][0]["content"] == "SYS"  # no cache_control breakpoint applied
    # No usage.cost from DeepSeek -> estimate from peak caps (1000 in, 50 out).
    assert cost == pytest.approx((1000 * 0.30 + 50 * 1.20) / 1_000_000)
    rec = c.guard.records[-1]
    assert rec["meta"]["cached_tokens"] == 800  # DeepSeek prompt_cache_hit_tokens


def test_stream_deepseek_enables_thinking_when_reasoning_enabled(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        captured["payload"] = json
        return _FakeResp(
            [
                {"choices": [{"delta": {"content": '{"x":1}'}}]},
                {"choices": [{"finish_reason": "stop", "delta": {}}]},
                _usage_chunk(cost=0.0),
            ]
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(
        Settings(api_key="k", backend="deepseek", deepseek_api_key="dk", model="deepseek-flash"),
        guard=_RecordingGuard(),
    )
    c._stream(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"max_tokens": 8000},
        }
    )
    p = captured["payload"]
    assert p["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in p  # no effort hint -> DeepSeek default high
    assert "reasoning" not in p


def test_deepseek_effort_mapping():
    assert CappedOpenRouterClient._deepseek_effort("low") == "low"
    assert CappedOpenRouterClient._deepseek_effort("medium") == "high"
    assert CappedOpenRouterClient._deepseek_effort("max") == "max"
    assert CappedOpenRouterClient._deepseek_effort("bogus") is None


@pytest.mark.parametrize("backend", ["deepseek", "openrouter"])
def test_json_schema_is_translated_only_for_the_deepseek_backend(monkeypatch, backend):
    # DeepSeek has no OpenAI structured-outputs mode: the request carries the
    # looser json_object form (the schema still rides verbatim in the prompt and
    # the in-stream guard). OpenRouter keeps the strict json_schema envelope.
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, stream=None):
        captured["payload"] = json
        return _FakeResp(
            [
                {"choices": [{"delta": {"content": '{"ok": true}'}}]},
                {"choices": [{"finish_reason": "stop", "delta": {}}]},
                _usage_chunk(cost=0.0),
            ]
        )

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    c = CappedOpenRouterClient(
        Settings(
            api_key="or-key",
            backend=backend,
            deepseek_api_key="ds-key",
            base_url=(
                "https://api.deepseek.com"
                if backend == "deepseek"
                else "https://openrouter.ai/api/v1"
            ),
            model="deepseek-flash" if backend == "deepseek" else "openai/gpt-5.6-luna",
            max_price_prompt=0.30,
            max_price_completion=1.20,
        ),
        guard=_RecordingGuard(),
    )
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "kicraft_test",
            "strict": True,
            "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
        },
    }
    c._stream(
        {
            "messages": [{"role": "user", "content": "return json"}],
            "response_format": dict(schema),
        }
    )
    if backend == "deepseek":
        assert captured["payload"]["response_format"] == {"type": "json_object"}
    else:
        assert captured["payload"]["response_format"] == schema


class _EndpointResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _EndpointHttp:
    def __init__(self, payload):
        self.payload = payload

    def get(self, *args, **kwargs):
        return _EndpointResponse(self.payload)


def test_model_preflight_rejects_missing_schema_capability_before_smoke():
    settings = Settings(api_key="k")
    endpoint = {
        "data": {
            "id": settings.model,
            "endpoints": [
                {
                    "model_id": settings.model,
                    "provider_name": "OpenInference",
                    "tag": "open-inference/fp8",
                    "pricing": {"prompt": "0.00000005", "completion": "0.00000016"},
                    "supported_parameters": ["reasoning", "tools", "tool_choice"],
                }
            ],
        }
    }
    result = preflight_role(
        settings,
        role="designer",
        model=settings.model,
        smoke=False,
        http=_EndpointHttp(endpoint),
    )
    assert result["ok"] is False
    assert result["endpoints"][0]["missing_parameters"] == ["response_format"]


def test_model_preflight_merges_campaign_metadata_into_fixed_role_context():
    settings = Settings(api_key="k")
    endpoint = {
        "data": {
            "id": settings.model,
            "endpoints": [
                {
                    "model_id": settings.model,
                    "provider_name": "OpenInference",
                    "tag": "open-inference/fp8",
                    "pricing": {"prompt": "0.00000005", "completion": "0.00000016"},
                    "supported_parameters": [
                        "reasoning",
                        "response_format",
                        "tools",
                        "tool_choice",
                    ],
                }
            ],
        }
    }
    seen = {}

    class Client:
        def __init__(self, _settings):
            pass

        def chat_with_tools(self, *args, **kwargs):
            seen.update(kwargs["meta_ctx"])
            return {
                "text": '{"ok": true}',
                "cost_usd": 0.001,
                "provider": "OpenInference",
                "model": settings.model,
                "finish_reason": "stop",
            }

    result = preflight_role(
        settings,
        role="designer",
        model=settings.model,
        http=_EndpointHttp(endpoint),
        client_factory=Client,
        meta_ctx={"campaign_id": "canary-1", "phase": "wrong", "role": "wrong"},
    )
    assert result["ok"] is True
    assert seen == {
        "campaign_id": "canary-1",
        "phase": "model_preflight",
        "role": "designer",
    }
