"""Pure helpers behind the live per-stage feed.

The NiceGUI rendering in ``StagePanel``/``StageTabs`` needs a UI context, so it is
exercised via the ``KICRAFT_WEB_DEMO`` ``/demo`` replay page. Here we cover the
parts that are plain functions: the partial-JSON pretty-printer that powers the
live Project-state draft, and the demo event stream that drives the new windows.
"""
from __future__ import annotations

import json

from kicraft.server.stagetabs import (
    _cell_html,
    _close_json,
    _loose_pretty,
    _table_html,
    demo_events,
)


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


# --------------------------------------------------- inspector tables (kc-table)

def test_table_html_is_one_aligned_table():
    # A single real <table> (so columns line up) with a header + one <tr> per row;
    # None renders as an empty cell, not the literal "None".
    h = _table_html(["name", "purpose"], [["MCU", "brains"], ["PWR", None]])
    assert h.startswith('<table class="kc-table">')
    assert h.count("<table") == 1
    assert h.count("<th>") == 2 and h.count("<tr>") == 3  # 1 header + 2 body
    assert "<td></td>" in h and "None" not in h


def test_table_html_escapes_cell_and_header_text():
    h = _table_html(["<col>"], [["<b>x</b>"]])
    assert "&lt;b&gt;x&lt;/b&gt;" in h and "<b>" not in h
    assert "&lt;col&gt;" in h


def test_cell_html_renders_https_link_new_tab():
    cell = {"text": "C2687116",
            "href": "https://www.lcsc.com/product-detail/C2687116.html"}
    out = _cell_html(cell)
    assert out.startswith('<a href="https://www.lcsc.com/product-detail/C2687116.html"')
    assert 'target="_blank"' in out and 'rel="noopener noreferrer"' in out
    assert ">C2687116</a>" in out


def test_cell_html_drops_non_http_scheme():
    # javascript:/data: hrefs must never become a clickable link; text stays escaped.
    for bad in ("javascript:alert(1)", "data:text/html,<b>", "", "/relative"):
        out = _cell_html({"text": "<x>", "href": bad})
        assert "<a " not in out
        assert out == "&lt;x&gt;"


def test_cell_html_scalar_and_none():
    assert _cell_html(None) == ""
    assert _cell_html("R_0805") == "R_0805"
    assert _cell_html("<b>") == "&lt;b&gt;"


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
