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


# ---- tool event contract (call identity, outcome, full evidence) -------------


def _progress_sink():
    events: list[dict] = []
    return events, events.append


def test_repeated_provider_ids_still_pair_uniquely(monkeypatch):
    """A provider reuses the same call id across rounds; the events must carry
    ids this loop minted, so each result pairs with ITS OWN call."""
    client = _client()
    events, progress = _progress_sink()

    def fake_stream(body, on_delta=None):
        rnd = body["_meta_ctx"]["round"]
        if rnd < 2:
            return {"role": "assistant", "content": None, "finish_reason": "tool_calls",
                    "tool_calls": [{"id": "call_1", "type": "function",  # SAME id twice
                                    "function": {"name": "lookup_footprint",
                                                 "arguments": json.dumps({"footprint": f"F{rnd}"})}}]}, 0.0
        return _text_msg(), 0.0

    monkeypatch.setattr(client, "_stream", fake_stream)
    client.chat_with_tools(messages=[{"role": "user", "content": "go"}], tools=[],
                           executor=lambda n, a: f"result for {a['footprint']}",
                           max_rounds=20, progress=progress)

    calls = [e for e in events if e["kind"] == "tool"]
    results = [e for e in events if e["kind"] == "tool_result"]
    assert [c["name"] for c in calls] == ["lookup_footprint"] * 2
    assert len({c["call_id"] for c in calls}) == 2      # minted locally, unique
    assert [r["call_id"] for r in results] == [c["call_id"] for c in calls]
    assert [r["output"] for r in results] == ["result for F0", "result for F1"]


def test_result_carries_outcome_duration_and_complete_evidence(monkeypatch):
    """A 2,000-character result keeps the evidence past character 600, and an
    explicit error contract is classified as a failure."""
    client = _client()
    events, progress = _progress_sink()
    long_evidence = "x" * 700 + "IMPORTANT-EVIDENCE"

    def executor(name, args):
        return long_evidence if name == "read_all" else "lookup_footprint exit=7"

    def fake_stream(body, on_delta=None):
        rnd = body["_meta_ctx"]["round"]
        if rnd < 2:
            name = "read_all" if rnd == 0 else "lookup_footprint"
            return {"role": "assistant", "content": None, "finish_reason": "tool_calls",
                    "tool_calls": [{"id": f"t{rnd}", "type": "function",
                                    "function": {"name": name, "arguments": "{}"}}]}, 0.0
        return _text_msg(), 0.0

    monkeypatch.setattr(client, "_stream", fake_stream)
    client.chat_with_tools(messages=[{"role": "user", "content": "go"}], tools=[],
                           executor=executor, max_rounds=20, progress=progress)

    first, second = [e for e in events if e["kind"] == "tool_result"]
    assert "IMPORTANT-EVIDENCE" in first["output"]   # nothing silently dropped
    assert first["output_chars"] == len(long_evidence)
    assert first["output_truncated"] is False
    assert first["ok"] is None                        # plain text is unverified
    assert first["cached"] is False
    assert first["duration_ms"] >= 0
    assert second["ok"] is False                      # nonzero exit marker
    # The model still sees the bounded message it always did.
    assert len(second["output"]) > 0


def test_over_limit_output_is_bounded_and_marked(monkeypatch):
    from kicraft.server.client import _MAX_TOOL_OUTPUT_CHARS

    client = _client()
    events, progress = _progress_sink()
    huge = "y" * (_MAX_TOOL_OUTPUT_CHARS + 500)

    def fake_stream(body, on_delta=None):
        rnd = body["_meta_ctx"]["round"]
        if rnd == 0:
            return _tool_msg(), 0.0
        return _text_msg(), 0.0

    monkeypatch.setattr(client, "_stream", fake_stream)
    client.chat_with_tools(messages=[{"role": "user", "content": "go"}], tools=[],
                           executor=lambda n, a: huge, max_rounds=5, progress=progress)

    result = next(e for e in events if e["kind"] == "tool_result")
    assert len(result["output"]) == _MAX_TOOL_OUTPUT_CHARS
    assert result["output_truncated"] is True
    assert result["output_chars"] == len(huge)


def test_cached_result_keeps_its_classification(monkeypatch):
    """A repeated identical call reuses the cached outcome -- including whether it
    failed -- and is marked as cached."""
    client = _client()
    events, progress = _progress_sink()

    def fake_stream(body, on_delta=None):
        if body["tool_choice"] == "none":
            return _text_msg(), 0.0
        return _tool_msg(), 0.0  # always the same call

    monkeypatch.setattr(client, "_stream", fake_stream)
    client.chat_with_tools(messages=[{"role": "user", "content": "go"}], tools=[],
                           executor=lambda n, a: "unknown tool: list_parts",
                           max_rounds=20, progress=progress)

    results = [e for e in events if e["kind"] == "tool_result"]
    assert len(results) >= 2
    assert all(e["ok"] is False for e in results)      # error survives the cache
    assert results[0]["cached"] is False and results[1]["cached"] is True


def test_consumer_side_sanitization_redacts_and_bounds(monkeypatch):
    """What the workspace persists and shows: credentials are redacted, URLs lose
    their userinfo/secret query, the project path becomes <project>, and oversized
    text is bounded with an explicit marker. The model message policy is untouched
    (asserted separately by the 4,000-character contract tests)."""
    from kicraft.server import activity

    client = _client()
    events, progress = _progress_sink()
    payload = (
        '{"api_key": "sk-live-abcdefghijklmnop", "cookie": "session=abc", '
        '"note": "see https://user:pw@example.com/x?token=secret123 and '
        'Bearer sk-abcdefghijklmnop"}'
    )

    def fake_stream(body, on_delta=None):
        rnd = body["_meta_ctx"]["round"]
        if rnd == 0:
            return _tool_msg(), 0.0
        return _text_msg(), 0.0

    monkeypatch.setattr(client, "_stream", fake_stream)
    client.chat_with_tools(messages=[{"role": "user", "content": "go"}], tools=[],
                           executor=lambda n, a: payload, max_rounds=5, progress=progress)

    result = next(e for e in events if e["kind"] == "tool_result")
    assert "sk-live-abcdefghijklmnop" in result["output"]      # raw is the transport's
    safe = activity.sanitize_activity_event(result, workspace="/tmp/run/ws")
    text = json.dumps(safe)
    assert "sk-live-abcdefghijklmnop" not in text
    assert "secret123" not in text
    assert "user:pw@" not in text
    assert "Bearer sk-abcdefghijklmnop" not in text
    assert "[redacted]" in text
    # Oversized technical text is bounded, and it SAYS so instead of clipping.
    bounded = activity.sanitize_activity_event(
        {"kind": "tool_result", "output": "z" * (activity.MAX_TECHNICAL_TEXT + 900)})
    assert len(bounded["output"]) < activity.MAX_TECHNICAL_TEXT + 120
    assert "truncated 900 characters" in bounded["output"]
    # A workspace path never leaks into a persisted event.
    pathy = activity.sanitize_activity_event(
        {"kind": "run_error", "message": "failed in /tmp/run/ws/generated/X"},
        workspace="/tmp/run/ws")
    assert pathy["message"] == "failed in <project>/generated/X"
