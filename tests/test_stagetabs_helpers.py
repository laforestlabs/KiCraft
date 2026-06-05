"""Pure helpers behind the live per-stage feed.

The NiceGUI rendering in ``StagePanel``/``StageTabs`` needs a UI context, so it is
exercised via the ``KICRAFT_WEB_DEMO`` ``/demo`` replay page. Here we cover the
parts that are plain functions: the partial-JSON pretty-printer that powers the
live Project-state draft, and the demo event stream that drives the new windows.
"""
from __future__ import annotations

import json

from kicraft.server.stagetabs import _close_json, _loose_pretty, demo_events


def test_loose_pretty_full_object():
    out = _loose_pretty('{"a": 1, "b": [2, 3]}')
    assert out is not None
    assert json.loads(out) == {"a": 1, "b": [2, 3]}
    assert "\n" in out  # indented


def test_loose_pretty_strips_code_fence():
    out = _loose_pretty('```json\n{"a": 1}\n```')
    assert out is not None and json.loads(out) == {"a": 1}


def test_loose_pretty_skips_leading_prose():
    out = _loose_pretty('Here is the slot:\n{"a": 1}')
    assert out is not None and json.loads(out) == {"a": 1}


def test_loose_pretty_truncated_object_recovers():
    # Cut off after a complete value, with a dangling comma + open containers.
    out = _loose_pretty('{"goal": "x", "parts": [1, 2,')
    assert out is not None
    assert json.loads(out) == {"goal": "x", "parts": [1, 2]}


def test_loose_pretty_truncated_string_recovers():
    out = _loose_pretty('{"goal": "USB-C flash')
    assert out is not None
    assert json.loads(out) == {"goal": "USB-C flash"}


def test_loose_pretty_non_json_is_none():
    assert _loose_pretty("just some reasoning text") is None
    assert _loose_pretty("") is None


def test_loose_pretty_never_raises():
    # Mid-key / mid-number / unbalanced fragments must fall back gracefully.
    for s in ("{", "[", '{"a"', '{"a":', '{"a": 1.', '"', '{"a": [}', "}{", "```json"):
        _loose_pretty(s)  # must not raise


def test_close_json_balanced_returns_none():
    # Nothing open -> caller keeps the raw text instead of re-deriving valid JSON.
    assert _close_json('{"a": 1}') is None


# ------------------------------------------------------------------ demo stream

def test_demo_events_drive_the_new_windows():
    evs = demo_events()
    # answer_delta feeds the live Project-state draft (Change 3).
    assert any(e.get("kind") == "answer_delta" for e in evs)
    # every LLM stage announces its model so the activity diagnostics show it (Change 4).
    starts = [e for e in evs if e.get("kind") == "stage_start"]
    assert starts and all(e.get("model") for e in starts)


def test_demo_answer_deltas_assemble_into_valid_slots():
    # Concatenated answer_delta text per stage should be (loosely) parseable JSON,
    # i.e. the live draft has something real to preview.
    evs = demo_events()
    buf, stage = "", None
    seen = 0
    for e in evs:
        k = e.get("kind")
        if k == "stage_start":
            buf, stage = "", e.get("stage")
        elif k == "answer_delta":
            buf += e.get("text", "")
        elif k == "stage_done" and buf:
            assert _loose_pretty(buf) is not None, f"unparseable draft for {stage}"
            seen += 1
    assert seen >= 3  # several stages exercise the draft path
