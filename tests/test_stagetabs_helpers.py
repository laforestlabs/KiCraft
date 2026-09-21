"""Pure helpers behind the live per-stage feed.

The NiceGUI rendering in ``StagePanel``/``StageTabs`` needs a UI context, so it is
exercised via the ``KICRAFT_WEB_DEMO`` ``/demo`` replay page. Here we cover the
parts that are plain functions: the partial-JSON pretty-printer that powers the
live Project-state draft, and the demo event stream that drives the new windows.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from nicegui.testing.user_simulation import user_simulation

from kicraft.server.web import _draft_sections, _inspector_spec



from kicraft.server.stagetabs import (
    StageTabs,
    _cell_html,
    _close_json,
    _loose_pretty,
    _parse_draft,
    _table_html,
    demo_events,
)

@pytest.fixture
def anyio_backend():
    return "asyncio"


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



@pytest.mark.anyio
async def test_stage_tabs_bom_preview_survives_invalid_partial():
    holder = {}

    def root():
        holder["tabs"] = StageTabs(
            draft_spec=lambda stg, obj: _draft_sections(stg, obj, prices={}))

    async with user_simulation(root=root) as u:
        await u.open("/")
        tabs = holder["tabs"]

        with u:
            tabs.push({"kind": "stage_start", "stage": "bom", "model": "test"})
            tabs.push({"kind": "reasoning_delta", "text": "reason"})
            tabs.push({"kind": "answer_delta", "text": '{"parts": ['})
            tabs.flush()
        await u.should_see('"parts"')
        await u.should_not_see("Parts")

        with u:
            tabs.push({"kind": "answer_delta", "text":
                       '{"ref": "U1", "value": "TP4056", "symbol": '
                       '"tp4056:TP4056_C725790", "footprint": "tp4056:ESOP-8", '
                       '"sheet": "MAIN"}, {'})
            tabs.flush()
        await u.should_see("Parts")
        await u.should_see("U1")

        with u:
            tabs.push({"kind": "answer_delta", "text": '"'})
            tabs.flush()
        await u.should_see("Parts")
        await u.should_see("U1")
        await u.should_not_see('"parts"')

        with u:
            tabs.push({"kind": "answer_delta", "text":
                       'ref": "J1", "value": "USB-C", "symbol": '
                       '"usb-c-16p:TYPE-C-31-M-12", "footprint": "usb-c-16p:TYPE-C", '
                       '"sheet": "MAIN"}]}'})
            tabs.flush()
        await u.should_see("J1")

        parts = [
            {"ref": "U1", "value": "TP4056", "symbol": "tp4056:TP4056_C725790",
             "footprint": "tp4056:ESOP-8", "sheet": "MAIN"},
            {"ref": "J1", "value": "USB-C", "symbol": "usb-c-16p:TYPE-C-31-M-12",
             "footprint": "usb-c-16p:TYPE-C", "sheet": "MAIN"},
        ]
        with u:
            tabs.set_inspector(
                "bom",
                _inspector_spec("bom", {"bom": {"parts": parts}}, {}, None, [], prices={}),
            )
        await u.should_not_see("writing bom slot")
        await u.should_see("U1")
        await u.should_see("J1")



@pytest.mark.anyio
async def test_stage_tabs_render_execution_provenance():
    holder = {}

    def root():
        holder["tabs"] = StageTabs()

    async with user_simulation(root=root) as u:
        await u.open("/")
        tabs = holder["tabs"]
        with u:
            tabs.push({"kind": "stage_start", "stage": "bom", "model": "test"})
            tabs.push(
                {
                    "kind": "recipe_selected",
                    "stage": "bom",
                    "recipe": "rp2040-minimal@1",
                    "instance": "controller",
                    "sheets": {"mcu": "MCU"},
                    "parameters": {"usb": True},
                }
            )
            tabs.push(
                {
                    "kind": "work_unit_plan",
                    "stage": "bom",
                    "unit_id": "bom-s000",
                    "unit_sheet": "BUFFER",
                    "source": "deterministic_architecture_lowering",
                }
            )
            tabs.push(
                {
                    "kind": "work_unit_attempt",
                    "stage": "bom",
                    "unit_id": "bom-s001",
                    "unit_sheet": "POWER",
                    "unit_attempt": 2,
                    "source": "llm",
                    "provider": "openrouter",
                    "model": "test/model",
                    "outcome": "invalid_work_unit",
                    "input_tokens": 120,
                    "output_tokens": 40,
                }
            )
            tabs.push(
                {
                    "kind": "work_unit_done",
                    "stage": "bom",
                    "unit_id": "bom-s000",
                    "unit_sheet": "BUFFER",
                    "source": "deterministic_architecture_lowering",
                }
            )

        await u.should_see("Circuit recipe")
        await u.should_see("rp2040-minimal@1")
        await u.should_see("Deterministic")
        await u.should_see("LLM")
        await u.should_see("invalid_work_unit")
        await u.should_see("Candidate validated and retained")


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


def test_unknown_events_are_not_blamed_on_intent():
    """A tool/retry event with no stage and no current stage is dropped, not
    attributed to Intent (which used to make the first tab light up for work that
    belonged elsewhere)."""

    class Panel:
        def __init__(self):
            self.events = []

        def push(self, event):
            self.events.append(event)

    intent = Panel()
    stub = SimpleNamespace(_current=None, panels={"intent": intent}, _replaying=False,
                           _auto_follow=True, _set_follow_ui=lambda: None,
                           _follow_btn=SimpleNamespace(set_visibility=lambda _v: None),
                           tabs=SimpleNamespace(value=None, set_value=lambda _v: None))

    StageTabs.push(stub, {"kind": "tool", "call_id": "t-1", "name": "lookup"})

    assert intent.events == []


def test_stage_stamped_events_route_to_their_own_panel():
    """A recorded event carries the stage it happened in, so a replay lands it in
    the right tab regardless of what finished last."""

    class Panel:
        def __init__(self):
            self.events = []

        def push(self, event):
            self.events.append(event)

    intent = Panel()
    bom = Panel()
    stub = SimpleNamespace(_current="intent", panels={"intent": intent, "bom": bom},
                           _replaying=True, _auto_follow=False,
                           _set_follow_ui=lambda: None,
                           _follow_btn=SimpleNamespace(set_visibility=lambda _v: None),
                           tabs=SimpleNamespace(value=None, set_value=lambda _v: None))
    event = {"kind": "work_unit_plan", "stage": "bom", "unit_id": "bom-s000",
             "source": "llm"}

    StageTabs.push(stub, event)

    assert intent.events == []
    assert bom.events == [event]


# ------------------------------------------------- reopened / live stage tabs


def _wiring_failure_events():
    """A Wiring run that failed after two same-name tool calls and a retry."""
    return [
        {"kind": "run_started", "run_id": "p1-abc", "seq": 1,
         "ts": "2026-01-01T00:00:00+00:00"},
        {"kind": "stage_start", "stage": "intent", "seq": 2,
         "ts": "2026-01-01T00:00:01+00:00"},
        {"kind": "stage_done", "stage": "intent", "ok": True, "seq": 3,
         "ts": "2026-01-01T00:00:02+00:00"},
        {"kind": "stage_start", "stage": "wiring", "seq": 4,
         "ts": "2026-01-01T00:00:03+00:00"},
        {"kind": "tool", "call_id": "tok-1", "name": "lookup_lcsc",
         "args": {"mpn": "TPS54331"}, "stage": "wiring", "seq": 5,
         "ts": "2026-01-01T00:00:04+00:00"},
        {"kind": "tool", "call_id": "tok-2", "name": "lookup_lcsc",
         "args": {"mpn": "TPS54331"}, "stage": "wiring", "seq": 6,
         "ts": "2026-01-01T00:00:05+00:00"},
        {"kind": "tool_result", "call_id": "tok-1", "name": "lookup_lcsc",
         "output": "found TPS54331", "ok": True, "duration_ms": 41,
         "output_chars": 14, "output_truncated": False, "stage": "wiring", "seq": 7,
         "ts": "2026-01-01T00:00:06+00:00"},
        {"kind": "tool_result", "call_id": "tok-2", "name": "lookup_lcsc",
         "output": "unknown tool: lookup_lcsc", "ok": False, "duration_ms": 3,
         "output_chars": 24, "output_truncated": False, "stage": "wiring", "seq": 8,
         "ts": "2026-01-01T00:00:07+00:00"},
        {"kind": "retry", "stage": "wiring", "failure_kind": "commit_rejected",
         "errors": ["U1 pin 7 unconnected", "net VBUS has one endpoint"], "seq": 9,
         "ts": "2026-01-01T00:00:08+00:00"},
        {"kind": "run_error", "stage": "wiring", "failure_kind": "unexpected_error",
         "exception_type": "TimeoutError", "message": "TimeoutError: provider slow",
         "seq": 10, "ts": "2026-01-01T00:00:09+00:00"},
        {"kind": "run_finished", "status": "failed", "stage": "wiring", "seq": 11,
         "ts": "2026-01-01T00:00:10+00:00"},
    ]


@pytest.mark.anyio
async def test_reopened_failure_names_wiring_and_shows_its_evidence():
    """Reopening a failed Wiring run: the current stage is selected, the details
    expand to the named call/result pair, and the retry errors stay readable."""
    holder = {}

    def root():
        holder["tabs"] = StageTabs()

    async with user_simulation(root=root) as u:
        await u.open("/")
        tabs = holder["tabs"]

        with u:
            tabs.begin_replay()
            for event in _wiring_failure_events():
                tabs.push(event)
            tabs.flush()
        with u:
            tabs.end_replay("wiring")
        tabs.flush()

        assert tabs._current == "wiring"
        await u.should_see("failed")
        assert tabs.active() == "wiring"
        # Both call/result pairs are inspectable by name, with their outcome.
        await u.should_see("Returned")
        await u.should_see("Failed")
        await u.should_see("found TPS54331")
        await u.should_see("unknown tool: lookup_lcsc")
        await u.should_see("0.04s")  # 41 ms, rendered as a duration
        # A retry is readable prose plus the complete error items, not clipped JSON.
        await u.should_see("Retrying Wiring")
        await u.should_see("U1 pin 7 unconnected")
        await u.should_see("net VBUS has one endpoint")
        await u.should_see("TimeoutError")


@pytest.mark.anyio
async def test_manual_tab_choice_survives_later_activity():
    """Once the user picks a tab, live activity must not yank them elsewhere; the
    explicit follow control brings them back."""
    holder = {}

    def root():
        holder["tabs"] = StageTabs()

    async with user_simulation(root=root) as u:
        await u.open("/")
        tabs = holder["tabs"]

        with u:
            tabs.push({"kind": "stage_start", "stage": "intent"})
            tabs.push({"kind": "stage_done", "stage": "intent", "ok": True})
            tabs.push({"kind": "stage_start", "stage": "functional_spec"})
        tabs.flush()
        with u:
            tabs.select("intent")  # the user looks back at an earlier stage
        assert tabs.active() == "intent"

        with u:
            tabs.push({"kind": "stage_done", "stage": "functional_spec", "ok": True})
            tabs.push({"kind": "stage_start", "stage": "architecture"})
        tabs.flush()
        assert tabs.active() == "intent"  # not dragged along

        with u:
            tabs.follow_live()
        assert tabs.active() == "architecture"
        assert tabs.active() is not None


@pytest.mark.anyio
async def test_corrected_stage_is_not_blocked_by_historical_diagnostics():
    """A historical failure in the same stage must not outlive its own correction:
    the tab paints the LATEST attempt's outcome, while the durable derive keeps
    committed-with-findings a warning."""
    holder = {}

    def root():
        holder["tabs"] = StageTabs()

    async with user_simulation(root=root) as u:
        await u.open("/")
        tabs = holder["tabs"]

        with u:
            tabs.push({"kind": "stage_start", "stage": "wiring"})
            tabs.push({"kind": "stage_diagnostic", "stage": "wiring",
                       "code": "unresolved_pin", "message": "U1 pin 7 dangling"})
            tabs.push({"kind": "stage_done", "stage": "wiring", "ok": False,
                       "failure_kind": "commit_rejected"})
        tabs.flush()
        with u:
            tabs.set_statuses({"wiring": "warning"},
                              {"wiring": {"ok": True, "semantic_clean": True}})
        tabs.flush()

        # The corrected commit is a warning (with findings), never a red failure.
        await u.should_see("committed with findings")
        await u.should_not_see("failed")


@pytest.mark.anyio
async def test_awaiting_input_is_not_shown_as_running():
    holder = {}

    def root():
        holder["tabs"] = StageTabs()

    async with user_simulation(root=root) as u:
        await u.open("/")
        tabs = holder["tabs"]

        with u:
            tabs.push({"kind": "stage_start", "stage": "wiring"})
            tabs.push({"kind": "question", "stage": "wiring",
                       "questions": [{"stage": "wiring", "text": "which USB con?"}]})
        tabs.flush()

        await u.should_see("waiting for your answer")
        # The status line the summary reads, and the tab's accessible name, both
        # say parked -- never running.
        assert tabs._tab_el["wiring"]._props["aria-label"] == "Wiring: parked"


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

@pytest.mark.anyio
async def test_on_tab_change_runs_show_hook_and_toggles_follow():
    """Revealing a tab runs its registered on_show hook (used to re-fit a KiCanvas
    view built while its tab was hidden, which would otherwise stay blank), and
    auto-follow resumes only on the live stage."""
    holder = {}

    def root():
        holder["tabs"] = StageTabs()

    async with user_simulation(root=root) as u:
        await u.open("/")
        tabs = holder["tabs"]
        fired: list[str] = []
        tabs.on_show("synthesize", lambda: fired.append("synthesize"))

        with u:
            tabs.push({"kind": "stage_start", "stage": "bom"})
        tabs.flush()
        assert tabs._current == "bom"
        assert tabs._auto_follow is True  # following the live stage

        with u:
            tabs.select("synthesize")  # the user reveals another tab
        assert fired == ["synthesize"]
        assert tabs._auto_follow is False

        with u:
            tabs.follow_live()
        assert tabs._auto_follow is True
        assert tabs._current == "bom"
        await u.should_see("▶ BOM started")


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
