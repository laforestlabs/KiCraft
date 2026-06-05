"""The chat_with_tools loop: the last-round nudge that keeps the final answer
cache-warm, and the forced-final fallback when the model never stops.

No network/ledger: `chat_with_tools` never reads `self.s`, so a stub settings +
guard is enough, and `_stream` is replaced with a scripted fake.
"""
from __future__ import annotations

import json
import types

from kicraft.server.client import CappedOpenRouterClient


def _client() -> CappedOpenRouterClient:
    return CappedOpenRouterClient(
        settings=types.SimpleNamespace(),
        guard=types.SimpleNamespace(status=lambda: {}),
    )


def _tool_msg() -> dict:
    return {"role": "assistant", "content": None, "finish_reason": "tool_calls",
            "tool_calls": [{"id": "t1", "type": "function",
                            "function": {"name": "list_parts", "arguments": "{}"}}]}


def _text_msg(text: str = '{"ok": true}') -> dict:
    return {"role": "assistant", "content": text, "finish_reason": "stop"}


def test_last_round_nudge_returns_warm_without_forced_final(monkeypatch):
    """Model calls a tool every round but emits JSON on the nudged final round:
    we return via the warm path and never make the cold tool_choice=none call."""
    client = _client()
    metas: list[str] = []

    def fake_stream(body, on_delta=None):
        metas.append(body["_meta"])
        if body["_meta_ctx"]["round"] == 2:  # final round (max_rounds-1)
            return _text_msg(), 0.0
        return _tool_msg(), 0.0

    monkeypatch.setattr(client, "_stream", fake_stream)
    messages = [{"role": "user", "content": "go"}]
    r = client.chat_with_tools(messages, tools=[], executor=lambda n, a: "ok", max_rounds=3)

    assert r["text"] == '{"ok": true}'
    assert r.get("forced_final") is not True       # warm path, not the cold final
    assert "tools-final" not in metas              # the cache-busting call never happened
    assert r["rounds"] == 3
    assert any(m["role"] == "user" and "FINAL tool round" in (m.get("content") or "")
               for m in messages)                  # the nudge was injected


def test_forced_final_still_fires_when_model_never_stops(monkeypatch):
    """If the model keeps calling tools even on the nudged final round, the
    forced-final fallback still runs so we always get a parseable answer."""
    client = _client()
    metas: list[str] = []

    def fake_stream(body, on_delta=None):
        metas.append(body["_meta"])
        if body["_meta"] == "tools-final":
            return _text_msg(), 0.0
        return _tool_msg(), 0.0

    monkeypatch.setattr(client, "_stream", fake_stream)
    messages = [{"role": "user", "content": "go"}]
    r = client.chat_with_tools(messages, tools=[], executor=lambda n, a: "ok", max_rounds=3)

    assert r.get("forced_final") is True
    assert metas[-1] == "tools-final"
    assert r["text"] == '{"ok": true}'


def test_identical_calls_are_cached_and_force_an_early_final(monkeypatch):
    """A model that repeats the exact same tool call must not re-run the tool
    every round, nor burn the whole budget: identical calls are served from
    cache, and after a few the loop hard-stops tools (tool_choice=none) so the
    model commits to an answer well before max_rounds."""
    client = _client()
    choices: list[str] = []
    executor_calls = {"n": 0}

    def fake_stream(body, on_delta=None):
        choices.append(body["tool_choice"])
        if body["tool_choice"] == "none":  # tools hard-stopped -> model answers
            return _text_msg(), 0.0
        return _tool_msg(), 0.0            # always the SAME identical call

    def executor(name, args):
        executor_calls["n"] += 1
        return "parts table"

    monkeypatch.setattr(client, "_stream", fake_stream)
    messages = [{"role": "user", "content": "go"}]
    r = client.chat_with_tools(messages, tools=[], executor=executor, max_rounds=20)

    assert executor_calls["n"] == 1     # identical calls reused the cached result
    assert "none" in choices            # thrash detection hard-stopped the tools
    assert r["text"] == '{"ok": true}'
    assert r["rounds"] < 20             # converged early, did not burn the budget


def test_distinct_tool_calls_are_not_force_stopped(monkeypatch):
    """Genuine, distinct lookups each run and must not trip the convergence cap:
    the loop only hard-stops on *redundant* repeats, not normal multi-tool use."""
    client = _client()
    choices: list[str] = []
    executor_calls = {"n": 0}

    def fake_stream(body, on_delta=None):
        choices.append(body["tool_choice"])
        rnd = body["_meta_ctx"]["round"]
        if rnd < 2:  # two DIFFERENT calls, then commit
            return {"role": "assistant", "content": None, "finish_reason": "tool_calls",
                    "tool_calls": [{"id": f"t{rnd}", "type": "function",
                                    "function": {"name": "lookup_footprint",
                                                 "arguments": json.dumps({"footprint": f"F{rnd}"})}}]}, 0.0
        return _text_msg(), 0.0

    def executor(name, args):
        executor_calls["n"] += 1
        return "ok"

    monkeypatch.setattr(client, "_stream", fake_stream)
    r = client.chat_with_tools(messages=[{"role": "user", "content": "go"}],
                               tools=[], executor=executor, max_rounds=20)

    assert executor_calls["n"] == 2     # both distinct calls executed
    assert "none" not in choices        # never force-stopped
    assert r["text"] == '{"ok": true}'
