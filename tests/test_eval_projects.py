"""Tests for the self-eval full-fidelity event sink, and a guard that self-eval
runs are NOT registered as user projects.

Self-eval runs used to be inserted into the ``projects`` table (EV- board codes)
so they would open in the standard project viewer. That conflated eval artifacts
with real user boards; they now live on their own page (``/admin/self-eval``),
browsable to full depth there, and are never projects. The registration path
(``AccountStore.sync_eval_projects`` and friends) was removed."""
from __future__ import annotations

import json

from kicraft.server.accounts import AccountStore
from kicraft.eval.self_eval import _event_writer


def test_eval_runs_are_not_registered_as_projects():
    # The decoupling: there is no longer any path that turns a self-eval run into
    # a projects-table row. Guard it so the coupling can't silently come back.
    assert not hasattr(AccountStore, "sync_eval_projects")


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
