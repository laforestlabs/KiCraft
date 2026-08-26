"""Mock + replay LLM client: exercise the whole design pipeline at $0.

The real ``CappedOpenRouterClient`` streams from OpenRouter and costs money. For
load/stress testing we want to drive everything AROUND the model -- the
stage-driver loop, the subprocess stage-prep / stage-commit calls, the daemon
worker threads, the build queue, the SQLite writes -- WITHOUT spend and
deterministically. ``MockClient`` implements the exact surface ``drive_stage``
consumes (``chat`` / ``chat_with_tools`` / ``.s`` / ``.guard``) and replays a
recorded per-stage transcript so every stage commits on the first try.

Transcript shape::

    {"stem": "A_USB_C", "stages": {"intent": "<json>", "functional_spec": "<json>",
                                   "architecture": "<json>", "bom": "<json>",
                                   "wiring": "<json>"}}

The cheapest way to make one is :func:`transcript_from_state`: any frozen
``state.json`` from a prior successful run already holds each stage's committed
slot, and the slot->state ownership (``cli_app._apply_slot``) is invertible, so we
reconstruct the exact slot JSON the model must emit -- $0, no real call.

Selected by ``KICRAFT_LLM_MODE`` (``mock``/``replay`` -> MockClient; anything else,
including unset, -> the real client via :func:`kicraft.server.client.make_client`).
``KICRAFT_MOCK_TRANSCRIPT`` points at the transcript JSON; ``KICRAFT_MOCK_LATENCY_MS``
adds simulated think-time so the concurrency profile resembles real traffic;
``KICRAFT_MOCK_BOM_TOOLS=1`` makes the BOM stage actually invoke the parts-lookup
subprocess once (exercises that load path) before returning the recorded slot.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Stages whose committed slot lives at a single top-level state key. ``intent``
# additionally carries ``project_stem`` (the driver pops it before commit), and
# ``wiring`` is derived from the bom's connection fields -- both handled below.
_TOP_LEVEL_STAGES = ("intent", "functional_spec", "architecture")
# bom-owned fields set by the wiring stage, not the bom stage; stripped from the
# reconstructed bom slot so replay mirrors the real pipeline order.
_WIRING_OWNED = ("connections", "no_connect_pins")


def transcript_from_state(state: dict) -> dict:
    """Reconstruct a stage->slot-text transcript from a frozen ``state.json`` dict.

    Inverts ``cli_app._apply_slot``: each stage's slot is the subset of state it
    owns. Returns ``{"stem", "stages"}``; stages absent from the state are
    omitted (so a partially-committed state yields a partial transcript).
    """
    stem = state.get("project_stem") or "MOCK_BOARD"
    stages: dict[str, str] = {}
    for stg in _TOP_LEVEL_STAGES:
        val = state.get(stg)
        if val is None:
            continue
        if stg == "intent":
            stages[stg] = json.dumps({**val, "project_stem": stem})
        else:
            stages[stg] = json.dumps(val)
    bom = state.get("bom")
    if bom is not None:
        bom_slot = {k: v for k, v in bom.items() if k not in _WIRING_OWNED}
        stages["bom"] = json.dumps(bom_slot)
        stages["wiring"] = json.dumps(
            {
                "connections": bom.get("connections") or [],
                "no_connect_pins": bom.get("no_connect_pins") or [],
            }
        )
    return {"stem": stem, "stages": stages}


def transcript_from_state_file(path: str | Path) -> dict:
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    return transcript_from_state(state)


def write_transcript(transcript: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(transcript, indent=2), encoding="utf-8")
    return p


def load_transcript(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class _NullGuard:
    """Spend guard stand-in: never touches the ledger, always reports $0.

    Lets a load test assert spend is exactly unchanged (the mock cannot spend),
    while satisfying ``client.guard.status()`` callers (drive_chain return value).
    """

    def status(self) -> dict:
        return {
            "spent_total_usd": 0.0,
            "spent_today_usd": 0.0,
            "daily_remaining_usd": 0.0,
            "daily_ceiling_usd": 0.0,
            "total_ceiling_usd": 0.0,
            "kill_switch": False,
            "mock": True,
        }

    def record(self, *a, **k) -> None:  # pragma: no cover - trivial no-op
        pass

    def preflight(self) -> None:  # pragma: no cover - trivial no-op
        pass


class _StubSettings:
    """Minimal stand-in for Settings so MockClient needs no OPENROUTER_API_KEY."""

    def __init__(self, model: str = "mock") -> None:
        self.model = model


class MockClient:
    """Drop-in for ``CappedOpenRouterClient`` that replays a recorded transcript.

    Reads the active stage from ``meta_ctx['stage']`` (set by ``drive_stage``) and
    returns that stage's recorded slot text, so the deterministic stage-commit
    accepts it first try. Cost is always 0; ``.guard`` is a no-op so the spend
    ledger is never touched.
    """

    def __init__(
        self,
        settings=None,
        *,
        transcript: dict | None = None,
        latency_ms: int | None = None,
        run_bom_tools: bool | None = None,
    ) -> None:
        self.s = settings or _StubSettings(os.environ.get("KICRAFT_MODEL", "mock"))
        self.guard = _NullGuard()
        self._transcript = transcript
        if latency_ms is None:
            latency_ms = int(os.environ.get("KICRAFT_MOCK_LATENCY_MS", "0") or 0)
        self._latency_s = max(0.0, latency_ms / 1000.0)
        if run_bom_tools is None:
            run_bom_tools = os.environ.get("KICRAFT_MOCK_BOM_TOOLS", "").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        self._run_bom_tools = run_bom_tools

    # -- transcript -----------------------------------------------------------
    def _stages(self) -> dict:
        if self._transcript is None:
            path = os.environ.get("KICRAFT_MOCK_TRANSCRIPT", "").strip()
            if not path:
                raise RuntimeError(
                    "MockClient has no transcript: pass transcript=... or set "
                    "KICRAFT_MOCK_TRANSCRIPT to a transcript JSON file "
                    "(make one with transcript_from_state)."
                )
            self._transcript = load_transcript(path)
        return self._transcript.get("stages") or {}

    def _text_for(self, meta_ctx: dict | None) -> str:
        stage = (meta_ctx or {}).get("stage")
        stages = self._stages()
        if stage in stages:
            return stages[stage]
        # No recorded slot for this stage: return an empty object so the commit
        # rejects it and the harness records a stage failure (loud, not silent).
        return "{}"

    def _settle(self, progress, text: str) -> None:
        if self._latency_s:
            time.sleep(self._latency_s)
        if progress:
            # Emit a single answer event so the web render path (which reads the
            # event stream) is exercised under load, mirroring a streamed reply.
            progress({"kind": "answer_delta", "text": text[:80]})

    # -- client surface -------------------------------------------------------
    def chat(
        self,
        messages,
        model=None,
        max_tokens=None,
        temperature=0.2,
        progress=None,
        meta_ctx=None,
        reasoning=None,
        reasoning_guard=None,
        collection_bounds=(),
        response_format=None,
    ) -> dict:
        text = self._text_for(meta_ctx)
        self._settle(progress, text)
        return {
            "text": text,
            "reasoning": None,
            "finish_reason": "stop",
            "model": getattr(self.s, "model", "mock"),
            "usage": {},
            "cost_usd": 0.0,
            "guard": self.guard.status(),
        }

    def chat_with_tools(
        self,
        messages,
        tools,
        executor,
        model=None,
        max_tokens=None,
        temperature=0.2,
        max_rounds=12,
        progress=None,
        meta_ctx=None,
        reasoning=None,
        reasoning_guard=None,
        collection_bounds=(),
        response_format=None,
    ) -> dict:
        text = self._text_for(meta_ctx)
        # Optionally exercise the real parts-lookup subprocess once so the BOM
        # tool path contributes to load (off by default for max throughput).
        if self._run_bom_tools and executor is not None:
            try:
                executor("list_parts", {})
            except Exception:  # pragma: no cover - tool errors are not the SUT here
                pass
        self._settle(progress, text)
        return {
            "text": text,
            "cost_usd": 0.0,
            "rounds": 1,
            "tool_calls": 0,
            "finish_reason": "stop",
            "guard": self.guard.status(),
        }


class RecordingClient:
    """Wrap a real client and capture each stage's committed slot into a transcript.

    Use to seed a transcript from one real ``self_eval --limit 1`` run when you want
    the more-realistic recorded text rather than the frozen-state reconstruction.
    Delegates every call to ``inner`` and records the assistant text keyed by the
    stage in ``meta_ctx`` (last write wins -> the text that finally committed).
    """

    def __init__(self, inner) -> None:
        self.inner = inner
        self.s = getattr(inner, "s", None)
        self.guard = getattr(inner, "guard", None)
        self.transcript: dict = {"stem": None, "stages": {}}

    def _capture(self, meta_ctx, res) -> None:
        stage = (meta_ctx or {}).get("stage")
        if stage and res.get("text"):
            self.transcript["stages"][stage] = res["text"]

    def chat(self, messages, **kw):
        res = self.inner.chat(messages, **kw)
        self._capture(kw.get("meta_ctx"), res)
        return res

    def chat_with_tools(self, messages, tools, executor, **kw):
        res = self.inner.chat_with_tools(messages, tools, executor, **kw)
        self._capture(kw.get("meta_ctx"), res)
        return res
