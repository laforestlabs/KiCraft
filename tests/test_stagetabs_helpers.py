"""Pure helpers behind the live per-stage feed.

The NiceGUI rendering in ``StagePanel``/``StageTabs`` needs a UI context, so it is
exercised via the ``KICRAFT_WEB_DEMO`` ``/demo`` replay page. Here we cover the
parts that are plain functions: the partial-JSON pretty-printer that powers the
live Project-state draft, and the demo event stream that drives the new windows.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from kicraft.server.stagetabs import (
    StageTabs,
    _cell_html,
    _close_json,
    _loose_pretty,
    _parse_draft,
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


# ------------------------------------------------- live-draft structured preview

def test_parse_draft_returns_object_for_partial_stream():
    obj = _parse_draft('{"parts": [{"ref": "U1", "value": "TP40')
    assert obj == {"parts": [{"ref": "U1", "value": "TP40"}]}
    assert _parse_draft("no json here") is None


def test_draft_sections_bom_renders_table_mid_stream():
    """The BOM draft renders as the real Parts table (same columns as the
    committed view) while the JSON is still streaming, not as raw text."""
    from kicraft.server.web import _draft_sections

    obj = _parse_draft(
        '{"parts": [{"ref": "U1", "value": "TP4056", "symbol": "tp4056:TP4056_C725790",'
        ' "footprint": "tp4056:ESOP-8", "sheet": "MAIN"}, {"ref": "J1", "value": "USB')
    secs = _draft_sections("bom", obj, prices={})
    tables = [s for s in secs if s.get("type") == "table"]
    assert tables and tables[0]["columns"][:2] == ["ref", "value"]
    assert [r[0] for r in tables[0]["rows"]] == ["U1", "J1"]  # half-streamed row included


def test_draft_sections_wiring_maps_into_bom_slot():
    # The wiring slot commits into state.json's "bom" key; the draft wrapper must
    # place the parsed buffer there for _inspector_spec to see the connections.
    from kicraft.server.web import _draft_sections

    obj = {"connections": [{"net_name": "VBUS", "sheet": "MAIN",
                            "endpoints": [{"ref": "J1", "pin": "A4"}]}]}
    secs = _draft_sections("wiring", obj)
    assert any(s.get("title") == "Connections" for s in secs)


def test_draft_sections_drops_graph_sections():
    # Rebuilding an echart every flush tick is heavy + flickery, so the draft
    # keeps only the cheap section types.
    from kicraft.server.web import _draft_sections

    obj = {"blocks": [{"name": "CHARGER", "category": "power", "purpose": "charge"}],
           "connections": [{"from_block": "A", "to_block": "B", "signal_type": "power"}]}
    secs = _draft_sections("functional_spec", obj)
    assert secs and all(s.get("type") != "graph" for s in secs)


def test_draft_sections_unshapeable_returns_empty():
    # [] tells the panel to fall back to the pretty-JSON text draft.
    from kicraft.server.web import _draft_sections

    assert _draft_sections("bom", ["not", "a", "dict"]) == []
    assert _draft_sections("synthesize", {"a": 1}) == []  # build stage: no draft
    assert _draft_sections("bom", {"parts": []}) == []    # nothing to show yet


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


def test_table_html_renders_footer_rows():
    # The BOM total lives in a <tfoot> so it sits below the body, set off by CSS.
    h = _table_html(["ref", "cost"], [["R1", "$0.01"]],
                    foot=[["TOTAL", "$2.41"]])
    assert "<tfoot>" in h and "<td>TOTAL</td>" in h and "<td>$2.41</td>" in h
    assert h.index("<tbody>") < h.index("<tfoot>")  # footer after the body
    # no footer -> no <tfoot>
    assert "<tfoot>" not in _table_html(["a"], [["1"]])


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


# ------------------------------------------------- tab-reveal hook (KiCanvas re-fit)

def test_on_tab_change_runs_show_hook_and_toggles_follow():
    """Revealing a tab runs its registered on_show hook (used to re-fit a KiCanvas
    view built while its tab was hidden, which would otherwise stay a blank panel),
    and auto-follow resumes only on the live stage. Driven on a stub self so the
    method's logic is covered without a UI context."""
    fired = []
    stub = SimpleNamespace(
        _current="bom",
        _auto_follow=True,
        _on_show={"synthesize": lambda: fired.append("synthesize")},
    )
    # Reveal a non-live tab that has a hook: the hook fires, auto-follow turns off.
    StageTabs._on_tab_change(stub, SimpleNamespace(value="synthesize"))
    assert fired == ["synthesize"]
    assert stub._auto_follow is False
    # Reveal a hookless tab that IS the live stage: no-op hook, auto-follow resumes.
    StageTabs._on_tab_change(stub, SimpleNamespace(value="bom"))
    assert fired == ["synthesize"]
    assert stub._auto_follow is True


# ------------------------------------------------- build-log tab classifier

from kicraft.server.stagetabs import _build_substage


def test_build_substage_step_markers_anchor_to_line_head():
    assert _build_substage("[build] 1/5 synthesize (schematic + seed PCB + ERC) ...") == "synthesize"
    assert _build_substage("[build]     synthesized /x/generated/FOO (ERC clean)") == "synthesize"
    assert _build_substage("[build] 2/5 place + route (quality=good, seed=auto) ...") == "place_route"
    assert _build_substage("[build] 3/5 promoted routed parent -> FOO.kicad_pcb") == "place_route"
    assert _build_substage("[build] 4/5 verify: shorts=0 unconnected=0 ...") == "place_route"
    assert _build_substage("[build] 5/5 export fab package (Gerbers + drill) ...") == "fab"


def test_build_substage_project_path_is_not_a_step_marker():
    """Regression: '1/5' matched as a bare substring, so any line carrying a
    project path like /projects/1/550/ flipped the tab machine back to
    synthesize mid-build (every project id starting with 5 was affected)."""
    for line in (
        "Log:        /home/k/.kicraft/projects/1/550/generated/X/.experiments/experiments.jsonl",
        "[timing] round 1 solve_subcircuits_total=37.218s",
        "[round 2] --leaves-only: skipping parent compose",
        "BUILD COMPLETE: TPS5430_BUCK",
        "  routed PCB : /home/k/.kicraft/projects/1/550/generated/X/X.kicad_pcb",
        "[build]   leaf phase: 3x3 designs/leaf + auto-pin best ...",
    ):
        assert _build_substage(line) is None, line


def test_build_substage_review_markers():
    assert _build_substage(
        "[build]     electrical review: scanning design for electrical defects ..."
    ) == "electrical_review"
    assert _build_substage(
        "[build]     review BLOCKER: [power] VSENSE divider swapped"
    ) == "electrical_review"
    assert _build_substage(
        "[build]     review WARNING: [esd] no TVS on USB"
    ) == "electrical_review"
    assert _build_substage(
        "[build]     electrical review found a blocker; re-driving wiring once to fix"
    ) == "electrical_review"


def test_build_lines_for_splits_a_real_stream():
    """web._build_lines_for shares the classifier: a run-550-shaped stream (paths
    containing '1/5' everywhere) must keep place/route lines out of synthesize."""
    from kicraft.server.web import _build_lines_for

    lines = [
        "[build] 1/5 synthesize (schematic + seed PCB + ERC) ...",
        "[build]     synthesized /home/k/.kicraft/projects/1/550/generated/X (ERC clean)",
        "[build] 2/5 place + route (quality=good, seed=auto) ...",
        "[timing] round 1 solve_subcircuits_total=37.218s",
        "Log:        /home/k/.kicraft/projects/1/550/generated/X/.experiments/experiments.jsonl",
        "[build] 4/5 verify: shorts=0 unconnected=0 courtyard=0",
        "[build] 5/5 export fab package ...",
        "  routed PCB : /home/k/.kicraft/projects/1/550/generated/X/X.kicad_pcb",
    ]
    assert _build_lines_for("synthesize", lines) == lines[:2]
    assert _build_lines_for("place_route", lines) == lines[2:6]
    assert _build_lines_for("fab", lines) == lines[6:]
