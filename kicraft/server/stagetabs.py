"""Per-stage tabbed view of the live design feed.

The web app streams one event per token/tool/stage from the agent loop (see
``stage_driver``/``client``). The old ``FeedView`` rendered them all into one
long scrolling column. ``StageTabs`` instead gives every pipeline phase its own
tab, and inside each tab lays out three windows (the user-chosen
inspector-left / stream-right arrangement):

  * PROJECT STATE (left) - the structured data this stage committed (the parts
    list, the nets, the sheets, ...), rebuilt from ``state.json`` for inspection,
    plus the native KiCad view / download for the build phases.
  * THINKING (top-right) - the model's reasoning stream for the stage, in
    collapsible runs that auto-fold as the stage moves on.
  * ACTIVITY / LOG (bottom-right) - tool calls, tool results, retries, and (for
    the build phases) the build-log lines.

The caller drives it from the page's 0.2s timer exactly like before:
``push(event)`` per new event, then ``flush()`` once. The committed slot data is
supplied separately via ``set_inspector(stage, spec)`` (the page reads
``state.json`` and builds the spec; this module only renders it).

Event kinds handled (shapes unchanged from the agent loop):
  stage_start{stage} reasoning_delta{text} answer_delta{text} tool{name,args}
  tool_result{output} retry{stage,errors} stage_done{stage,ok,cost}
  build_start build_log{text} build_done{ok}
Both ``reasoning_delta`` (the model's reasoning channel) and ``answer_delta`` (its
content draft) stream into the Thinking window so it fills live even for models /
tool-free stages that only emit content; the committed result still lands,
structured, in the Project State window.
"""
from __future__ import annotations

import json

from nicegui import ui

# Phase identity: (key, label, Material icon, accent hex). Order is the pipeline
# order and drives both the tab row and each panel's accenting. The first five
# are the LLM design stages (DESIGN_STAGES); the last three are the deterministic
# build sub-phases, fed from the single `kicraft build` log stream.
PHASES: list[tuple[str, str, str, str]] = [
    ("intent",          "Intent",        "lightbulb",       "#38bdf8"),  # sky
    ("functional_spec", "Functional",    "list_alt",        "#a78bfa"),  # violet
    ("architecture",    "Architecture",  "account_tree",    "#22d3ee"),  # cyan
    ("bom",             "BOM",           "inventory_2",     "#fbbf24"),  # amber
    ("wiring",          "Wiring",         "cable",          "#34d399"),  # emerald
    ("synthesize",      "Synthesize",    "bolt",            "#f472b6"),  # pink
    ("place_route",     "Place/Route",   "developer_board", "#60a5fa"),  # blue
    ("fab",             "Fab",           "inventory",       "#818cf8"),  # indigo
]
_META = {k: (label, icon, accent) for k, label, icon, accent in PHASES}
_BUILD_STAGES = ("synthesize", "place_route", "fab")

_OK = "#34d399"
_FAIL = "#f87171"
_DIM = "#94a3b8"       # slate-400, secondary text
_DIMMER = "#64748b"    # slate-500, tertiary / glyphs
_STATUS_COLOR = {
    "pending": "#64748b",
    "active": "#fbbf24",
    "done": "#34d399",
    "failed": "#f87171",
}
_RESULT_FOLD_OVER = 300  # tool results longer than this fold into an expansion


class _Run:
    """A streaming, collapsible text block (a reasoning run or the build log): its
    expansion, the label tokens stream into, and the growing text buffer. Coalesced
    once per tick by ``StagePanel.flush`` so a burst of deltas is one DOM write."""

    __slots__ = ("exp", "label", "buf", "head", "mode")

    def __init__(self, exp, label, head="Thinking", mode="chars"):
        self.exp = exp
        self.label = label
        self.buf = ""
        self.head = head      # header title prefix
        self.mode = mode      # "chars" (reasoning) or "lines" (build log)


class StagePanel:
    """One phase's tab body: inspector (left) over thinking + activity (right).

    Built once into the surrounding ``ui.tab_panel`` context. Streaming events are
    fed via ``push``; ``flush`` coalesces the growing text blocks; ``set_inspector``
    (re)renders the structured project-state. ``view_slot`` is an empty column at
    the top of the inspector column for the page to drop a KiCanvas view / download
    button into (the build phases), kept separate from the data area so refreshing
    the data never clobbers the view.
    """

    def __init__(self, key: str, label: str, icon: str, accent: str) -> None:
        self.key = key
        self.accent = accent
        self._active_run: _Run | None = None
        self._open_run: _Run | None = None
        self._build_log: _Run | None = None
        self._dirty: set[_Run] = set()

        with ui.column().classes("w-full gap-2 p-2"):
            # Status bar: a spinner while the stage runs, a result pill when done.
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon(icon).style(f"color:{accent};font-size:1.2rem")
                ui.label(label).classes("text-sm font-bold uppercase tracking-wide") \
                    .style(f"color:{accent}")
                self._status_slot = ui.row().classes("items-center gap-2")

            with ui.row().classes("w-full no-wrap gap-3").style("height:62vh"):
                # LEFT: project-state inspector (+ view slot for KiCanvas/download).
                with ui.column().classes("gap-1").style(
                        "width:42%;min-width:300px;height:100%"):
                    ui.label("Project state").classes(
                        "text-xs font-bold uppercase tracking-wide").style(f"color:{_DIM}")
                    insp = ui.scroll_area().classes("w-full rounded").style(
                        "flex:1;min-height:0;background:#0f172a;border:1px solid #1e293b")
                    with insp:
                        self.view_slot = ui.column().classes("w-full p-2 gap-2")
                        self._insp = ui.column().classes("w-full p-2 gap-3")

                # RIGHT: thinking (top) over activity/log (bottom).
                with ui.column().classes("gap-1").style("flex:1;min-width:0;height:100%"):
                    ui.label("Thinking").classes(
                        "text-xs font-bold uppercase tracking-wide").style(f"color:{_DIM}")
                    think = ui.scroll_area().classes("w-full rounded").style(
                        "height:38%;background:#0f172a;border:1px solid #1e293b")
                    with think:
                        self._think = ui.column().classes("w-full p-2 gap-0")
                    self._think_scroll = think

                    ui.label("Activity / log").classes(
                        "text-xs font-bold uppercase tracking-wide mt-1").style(f"color:{_DIM}")
                    act = ui.scroll_area().classes("w-full rounded").style(
                        "flex:1;min-height:0;background:#0f172a;border:1px solid #1e293b")
                    with act:
                        self._act = ui.column().classes("w-full p-2 gap-1")
                    self._act_scroll = act

        self.clear()

    # ---- lifecycle ----------------------------------------------------------
    def clear(self) -> None:
        """Reset all three windows to idle placeholders for a fresh run."""
        self._active_run = None
        self._open_run = None
        self._build_log = None
        self._dirty.clear()
        self._status_slot.clear()
        self.view_slot.clear()
        self._insp.clear()
        self._think.clear()
        self._act.clear()
        with self._insp:
            ui.label("No data committed for this stage yet.") \
                .classes("text-xs italic").style(f"color:{_DIMMER}")
        with self._think:
            self._think_ph = ui.label("Reasoning will stream here.") \
                .classes("text-xs italic").style(f"color:{_DIMMER}")
        with self._act:
            self._act_ph = ui.label("Tool calls and log output will appear here.") \
                .classes("text-xs italic").style(f"color:{_DIMMER}")

    def push(self, e: dict) -> None:
        k = e.get("kind")
        if k in ("reasoning_delta", "answer_delta"):
            # Both the model's reasoning channel AND its content draft stream into
            # the Thinking window. Many models (e.g. deepseek-v4-flash) and the
            # tool-free stages emit only `content` (answer_delta), so without this
            # the window stays empty for the whole stage; streaming the draft keeps
            # it filling live with the work in progress.
            self._on_reasoning(e.get("text", ""))
        elif k == "tool":
            self._on_tool(e.get("name", ""), e.get("args") or {})
        elif k == "tool_result":
            self._on_tool_result(str(e.get("output", "")))
        elif k == "retry":
            self._on_retry(e.get("errors"))
        elif k == "build_log":
            self._on_build_log(e.get("text", ""))
        # stage_start/stage_done/build_* are handled by StageTabs (tab status).

    def flush(self) -> None:
        """Write coalesced streamed text once per tick (one DOM update per growing
        block instead of one per token)."""
        for run in self._dirty:
            run.label.set_text(run.buf)
            n = run.buf.count("\n") if run.mode == "lines" else len(run.buf)
            unit = "lines" if run.mode == "lines" else "chars"
            run.exp.set_text(f"{run.head} · {n:,} {unit}")
        self._dirty.clear()

    # ---- status -------------------------------------------------------------
    def mark_running(self) -> None:
        self._status_slot.clear()
        with self._status_slot:
            ui.spinner(size="sm").style(f"color:{self.accent}")

    def set_status(self, ok: bool, cost=None) -> None:
        self._status_slot.clear()
        with self._status_slot:
            if ok:
                ui.icon("check_circle").style(f"color:{_OK};font-size:1.1rem")
                if isinstance(cost, (int, float)):
                    ui.label(f"${cost:.4f}").classes("text-xs font-mono") \
                        .style(f"color:{_DIM}")
            else:
                ui.icon("cancel").style(f"color:{_FAIL};font-size:1.1rem")
                ui.label("failed").classes("text-xs").style(f"color:{_FAIL}")

    # ---- thinking -----------------------------------------------------------
    def _on_reasoning(self, text: str) -> None:
        if not text:
            return
        if self._think_ph is not None:
            self._think_ph.delete()
            self._think_ph = None
        if self._active_run is None:
            self._fold_open()
            with self._think:
                exp = ui.expansion("Thinking", icon="psychology", value=True) \
                    .classes("w-full").props('dense expand-separator '
                                             'header-class="text-xs text-grey-5"')
                with exp:
                    lab = ui.label("").classes(
                        "text-xs font-mono whitespace-pre-wrap leading-relaxed") \
                        .style(f"color:{_DIM}")
            self._active_run = _Run(exp, lab)
            self._open_run = self._active_run
        self._active_run.buf += text
        self._dirty.add(self._active_run)

    def _fold_open(self) -> None:
        if self._open_run is not None:
            self._open_run.exp.value = False
            self._open_run = None

    def end_runs(self) -> None:
        """Fold any open reasoning/build-log block (stage finished)."""
        self._fold_open()
        self._active_run = None

    # ---- activity / log -----------------------------------------------------
    def _act_ready(self) -> None:
        if self._act_ph is not None:
            self._act_ph.delete()
            self._act_ph = None

    def _on_tool(self, name: str, args: dict) -> None:
        self._active_run = None  # a tool ends the current thinking run
        self._act_ready()
        preview = json.dumps(args)[:140] if args else ""
        with self._act:
            with ui.row().classes("items-center gap-2 flex-nowrap min-w-0 pt-0.5"):
                ui.icon("terminal").style(f"color:{_DIMMER};font-size:1rem")
                ui.label(name).classes("text-xs font-mono px-1.5 py-0.5 rounded shrink-0") \
                    .style("background:rgba(56,189,248,0.14);color:#7dd3fc")
                if preview:
                    ui.label(preview).classes("text-xs font-mono truncate min-w-0") \
                        .style(f"color:{_DIMMER}")

    def _on_tool_result(self, output: str) -> None:
        self._active_run = None
        self._act_ready()
        with self._act:
            if len(output) > _RESULT_FOLD_OVER:
                exp = ui.expansion(f"result · {len(output):,} chars",
                                   icon="subdirectory_arrow_right") \
                    .classes("w-full").props('dense header-class="text-xs text-grey-5"')
                with exp:
                    ui.label(output).classes(
                        "text-xs font-mono whitespace-pre-wrap").style(f"color:{_DIM}")
            else:
                with ui.row().classes("items-start gap-1 flex-nowrap min-w-0"):
                    ui.icon("subdirectory_arrow_right") \
                        .style(f"color:{_DIMMER};font-size:0.95rem")
                    ui.label(output).classes(
                        "text-xs font-mono whitespace-pre-wrap min-w-0") \
                        .style(f"color:{_DIM}")

    def _on_retry(self, errors) -> None:
        self._active_run = None
        self._act_ready()
        msg = json.dumps(errors)[:200] if errors is not None else ""
        with self._act:
            with ui.row().classes("items-center gap-2 flex-nowrap min-w-0 px-2 py-1 rounded") \
                    .style("background:rgba(251,191,36,0.10)"):
                ui.icon("warning").style("color:#fbbf24;font-size:1rem")
                ui.label(f"retry: {msg}").classes("text-xs font-mono truncate min-w-0") \
                    .style("color:#fcd34d")

    def _on_build_log(self, text: str) -> None:
        self._act_ready()
        if self._build_log is None:
            with self._act:
                exp = ui.expansion("Build log", icon="terminal", value=True) \
                    .classes("w-full").props('dense expand-separator '
                                             'header-class="text-xs text-grey-5"')
                with exp:
                    lab = ui.label("").classes(
                        "text-xs font-mono whitespace-pre-wrap leading-relaxed") \
                        .style(f"color:{_DIM}")
            self._build_log = _Run(exp, lab, head="Build log", mode="lines")
        self._build_log.buf += text + "\n"
        self._dirty.add(self._build_log)

    # ---- inspector (structured project-state) -------------------------------
    def set_inspector(self, sections: list[dict]) -> None:
        """Render the committed project-state for this stage.

        `sections` is a list of dicts produced by the page from state.json:
          {"type": "kv",    "title": str, "rows": [(k, v), ...]}
          {"type": "list",  "title": str, "items": [str, ...]}
          {"type": "table", "title": str, "columns": [str, ...], "rows": [[...], ...]}
        """
        self._insp.clear()
        with self._insp:
            if not sections:
                ui.label("No data committed for this stage yet.") \
                    .classes("text-xs italic").style(f"color:{_DIMMER}")
                return
            for sec in sections:
                _render_section(sec, self.accent)


def _render_section(sec: dict, accent: str) -> None:
    title = sec.get("title", "")
    if title:
        ui.label(title).classes("text-xs font-semibold uppercase tracking-wide") \
            .style(f"color:{accent}")
    kind = sec.get("type")
    if kind == "kv":
        with ui.column().classes("w-full gap-0.5"):
            for k, v in sec.get("rows", []):
                with ui.row().classes("w-full no-wrap gap-2 items-start"):
                    ui.label(str(k)).classes("text-xs font-mono shrink-0") \
                        .style(f"color:{_DIMMER};min-width:9rem")
                    ui.label(str(v)).classes("text-xs font-mono whitespace-pre-wrap min-w-0") \
                        .style(f"color:{_DIM}")
    elif kind == "list":
        items = sec.get("items", [])
        if not items:
            ui.label("(none)").classes("text-xs italic").style(f"color:{_DIMMER}")
        with ui.column().classes("w-full gap-0.5"):
            for it in items:
                with ui.row().classes("w-full no-wrap gap-1 items-start"):
                    ui.label("•").style(f"color:{_DIMMER}")
                    ui.label(str(it)).classes(
                        "text-xs font-mono whitespace-pre-wrap min-w-0").style(f"color:{_DIM}")
    elif kind == "table":
        cols = sec.get("columns", [])
        rows = sec.get("rows", [])
        with ui.element("div").classes("w-full").style("overflow-x:auto"):
            with ui.column().classes("gap-0").style("min-width:max-content"):
                with ui.row().classes("no-wrap gap-3 px-1 py-0.5 rounded") \
                        .style("background:rgba(148,163,184,0.08)"):
                    for c in cols:
                        ui.label(str(c)).classes("text-xs font-mono font-bold") \
                            .style(f"color:{_DIM}")
                for r in rows:
                    with ui.row().classes("no-wrap gap-3 px-1"):
                        for cell in r:
                            ui.label(str(cell)).classes("text-xs font-mono whitespace-nowrap") \
                                .style(f"color:{_DIM}")


class StageTabs:
    """The tab row + tab panels, with event routing and status-coloured tabs.

    Built once inside a page layout. The page feeds it streaming events with
    ``push`` / ``flush`` and supplies committed data with ``set_inspector``. The
    active tab auto-follows the running stage until the user clicks a different
    tab (then it stays put until they click back to the live one).
    """

    def __init__(self) -> None:
        self.panels: dict[str, StagePanel] = {}
        self._tab_el: dict[str, ui.tab] = {}
        self._current: str | None = None
        self._auto_follow = True

        with ui.tabs().classes("w-full").props("dense inline-label") as self.tabs:
            for key, label, icon, accent in PHASES:
                t = ui.tab(key, label=label, icon=icon)
                t.style(f"color:{_STATUS_COLOR['pending']}")
                self._tab_el[key] = t
        self.tabs.on_value_change(self._on_tab_change)

        with ui.tab_panels(self.tabs, value=PHASES[0][0]).classes("w-full") \
                .style("background:transparent"):
            for key, label, icon, accent in PHASES:
                with ui.tab_panel(key).classes("p-0"):
                    self.panels[key] = StagePanel(key, label, icon, accent)

    # ---- tab status / follow ------------------------------------------------
    def _set_tab_status(self, key: str, status: str) -> None:
        t = self._tab_el.get(key)
        if t is not None:
            t.style(f"color:{_STATUS_COLOR[status]}")

    def _on_tab_change(self, e) -> None:
        # Resume auto-follow only while the user is parked on the live stage.
        self._auto_follow = (getattr(e, "value", None) == self._current)

    def _set_current(self, key: str | None) -> None:
        if key is None or key not in self.panels:
            return
        # Finishing one stage and entering the next: fold the previous panel's runs.
        if self._current and self._current != key:
            self.panels[self._current].end_runs()
        self._current = key
        self._set_tab_status(key, "active")
        self.panels[key].mark_running()
        if self._auto_follow:
            self.tabs.set_value(key)

    # ---- event routing ------------------------------------------------------
    def push(self, e: dict) -> None:
        k = e.get("kind")
        if k == "stage_start":
            self._set_current(e.get("stage"))
        elif k == "stage_done":
            self._finish(e.get("stage") or self._current, bool(e.get("ok")), e.get("cost"))
        elif k == "build_start":
            self._set_current("synthesize")
        elif k == "build_log":
            sub = _build_substage(e.get("text", ""))
            if sub and sub != self._current:
                # The previous build sub-phase completed when the next one logs.
                if self._current in _BUILD_STAGES:
                    self._finish(self._current, True, None)
                self._set_current(sub)
            if self._current:
                self.panels[self._current].push(e)
        elif k == "build_done":
            cur = self._current if self._current in _BUILD_STAGES else "fab"
            self._finish(cur, bool(e.get("ok")), None)
        else:  # reasoning_delta / tool / tool_result / retry: implicit current stage
            if self._current is None:
                self._set_current("intent")
            self.panels[self._current].push(e)

    def _finish(self, key: str | None, ok: bool, cost) -> None:
        if key is None or key not in self.panels:
            return
        self.panels[key].end_runs()
        self.panels[key].set_status(ok, cost)
        self._set_tab_status(key, "done" if ok else "failed")

    def flush(self) -> None:
        for p in self.panels.values():
            p.flush()

    # ---- data + reset (driven by the page) ----------------------------------
    def set_inspector(self, key: str, sections: list[dict]) -> None:
        p = self.panels.get(key)
        if p is not None:
            p.set_inspector(sections)

    def view_slot(self, key: str):
        """The empty column at the top of a panel's inspector for KiCanvas/download."""
        return self.panels[key].view_slot

    def scroll_active_to_bottom(self) -> None:
        if self._current and self._current in self.panels:
            self.panels[self._current]._act_scroll.scroll_to(percent=1.0)
            self.panels[self._current]._think_scroll.scroll_to(percent=1.0)

    def reset(self) -> None:
        self._current = None
        self._auto_follow = True
        for key in self.panels:
            self.panels[key].clear()
            self._set_tab_status(key, "pending")
        self.tabs.set_value(PHASES[0][0])


def _build_substage(text: str) -> str | None:
    """Map a `kicraft build` log line to its tab (markers from cli_app build)."""
    if "1/5" in text or "synthesized " in text:
        return "synthesize"
    if "2/5" in text or "3/5" in text or "4/5" in text:
        return "place_route"
    if "5/5" in text:
        return "fab"
    return None  # unmarked continuation: keep the current sub-stage


def demo_events() -> list[dict]:
    """A realistic canned event stream (the flashlight brief) for offline preview.

    Used by the KICRAFT_WEB_DEMO replay page so the styling can be screenshotted
    without spending or network. Exercises every branch: reasoning, multi-tool BOM,
    a wiring retry, and the deterministic build with its `[build] N/5` markers
    routed across the Synthesize / Place-Route / Fab tabs.
    """
    def think(*chunks: str) -> list[dict]:
        return [{"kind": "reasoning_delta", "text": c} for c in chunks]

    ev: list[dict] = []
    ev.append({"kind": "stage_start", "stage": "intent"})
    ev += think("The brief is a flashlight powered by an 18650 cell with USB-C ",
                "recharging. Core functions: USB-C 5V input, a Li-ion charger, the ",
                "18650 cell, a high-power white LED with a constant-current driver, ",
                "and a push-button to cycle modes. No microcontroller is required.")
    ev.append({"kind": "stage_done", "stage": "intent", "ok": True, "cost": 0.0021})

    ev.append({"kind": "stage_start", "stage": "functional_spec"})
    ev += think("Blocks: USB_C_INPUT, CHARGER, BATTERY, LED_DRIVER, LED, CONTROL. ",
                "Rails: VBUS 5.0V from USB, VBAT ~4.2V max from the cell. The driver ",
                "boosts VBAT to the LED forward voltage under constant current.")
    ev.append({"kind": "stage_done", "stage": "functional_spec", "ok": True, "cost": 0.0034})

    ev.append({"kind": "stage_start", "stage": "architecture"})
    ev += think("Single sheet is fine for this part count. Power nets: VBUS, VBAT, ",
                "GND, plus the switched LED node. TP4056 for the charger, a boost ",
                "constant-current driver for the LED, debounced push-button on CONTROL.")
    ev.append({"kind": "stage_done", "stage": "architecture", "ok": True, "cost": 0.0048})

    ev.append({"kind": "stage_start", "stage": "bom"})
    ev += think("I need real symbols and footprints. Start from the curated library, ",
                "then resolve the charger and USB-C connector from LCSC.")
    ev.append({"kind": "tool", "name": "list_parts", "args": {}})
    ev.append({"kind": "tool_result", "name": "list_parts",
               "output": "usb-c-16p   TYPE-C-31-M-12   USB-C receptacle, 16-pin\n"
                         "Device:R    Resistor_SMD:R_0603_1608Metric\n"
                         "Device:C    Capacitor_SMD:C_0603_1608Metric\n"
                         "Device:LED  LED_SMD:LED_0603_1608Metric\n... 14 bundles"})
    ev += think("USB-C receptacle is covered by usb-c-16p. The TP4056 charger is not ",
                "in the library, so resolve it from LCSC and fetch the bundle.")
    ev.append({"kind": "tool", "name": "lookup_lcsc_id", "args": {"mpn": "TP4056"}})
    ev.append({"kind": "tool_result", "name": "lookup_lcsc_id",
               "output": '{"ok": true, "lcsc": "C16581", "desc": "TP4056 1A Li-ion charger, SOP-8"}'})
    ev.append({"kind": "tool", "name": "add_part_from_lcsc",
               "args": {"lcsc_id": "C16581", "name": "tp4056"}})
    ev.append({"kind": "tool_result", "name": "add_part_from_lcsc",
               "output": "add-part exit=0\nfetched tp4056 (symbol + footprint SOP-8)\n\n"
                         "CURRENT PARTS LIBRARY:\n  tp4056:TP4056   tp4056:SOP-8_3.9x4.9\n"
                         "  usb-c-16p:TYPE-C-31-M-12 ...   (15 bundles total)"})
    ev += think("Good, all symbols and footprints resolve. Emit the BOM slot JSON.")
    ev.append({"kind": "stage_done", "stage": "bom", "ok": True, "cost": 0.0431, "attempts": 1})

    ev.append({"kind": "stage_start", "stage": "wiring"})
    ev += think("Connect VBUS from USB-C to the charger input, VBAT to the cell and ",
                "the driver input, the LED node through the driver, and CONTROL to the ",
                "button. Tie unused USB-C pins (SBU1/SBU2, shield) to no_connect.")
    ev.append({"kind": "retry", "stage": "wiring",
               "errors": ["pin (U1,4) of TP4056 not covered by a connection or no_connect"]})
    ev += think("Missed the TP4056 PROG pin. Add R_prog from PROG to GND to set the ",
                "charge current, which also covers that pin.")
    ev.append({"kind": "stage_done", "stage": "wiring", "ok": True, "cost": 0.0508, "attempts": 2})

    ev.append({"kind": "build_start"})
    for line in (
        "[build] 1/5 synthesize (schematic + seed PCB + ERC) ...",
        "[build]     synthesized generated/FLASHLIGHT (ERC clean)",
        "[build] 2/5 place + route (quality=balanced) -- may take minutes ...",
        "[build] 3/5 promoted routed parent -> FLASHLIGHT.kicad_pcb",
        "[build] 4/5 verify: shorts=0 unconnected=0 drc=0",
        "[build] 5/5 export fab package (Gerbers + drill + CPL + BOM) ...",
    ):
        ev.append({"kind": "build_log", "text": line})
    ev.append({"kind": "build_done", "ok": True})
    return ev
