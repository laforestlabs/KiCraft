"""Tests for the self-eval full-fidelity event sink."""
from __future__ import annotations

import json

from kicraft.eval.self_eval import _event_writer




def test_event_writer_lean_drops_token_deltas(tmp_path):
    p = tmp_path / "lean.jsonl"
    w = _event_writer(p, full=False)
    w({"kind": "stage_done", "ok": True})
    w({"kind": "answer_delta", "text": "x"})
    w({"kind": "reasoning_delta", "text": "y"})
    kinds = [json.loads(l)["kind"] for l in p.read_text().splitlines()]
    assert kinds == ["stage_done"]


def test_event_writer_full_keeps_everything(tmp_path):
    # The full sink keeps reasoning/answer deltas -- this is the data the eval
    # detail page's Thinking stream replays, so it must not regress.
    p = tmp_path / "full.jsonl"
    w = _event_writer(p, full=True)
    w({"kind": "stage_done", "ok": True})
    w({"kind": "answer_delta", "text": "x"})
    w({"kind": "reasoning_delta", "text": "y"})
    kinds = [json.loads(l)["kind"] for l in p.read_text().splitlines()]
    assert kinds == ["stage_done", "answer_delta", "reasoning_delta"]
