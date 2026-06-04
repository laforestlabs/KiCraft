"""Structured, themed render of the live design feed.

The web app streams one event per token/tool/stage from the agent loop (see
``stage_driver``/``client``). Rendering them as a single growing string makes a
flat, unreadable wall of text. ``FeedView`` instead builds real DOM per event:
each design stage becomes a colored, titled section; the model's reasoning streams
into a collapsible "Thinking" block that auto-folds once its stage finishes; tool
calls, tool results and retries each get their own distinct row. The caller drives
it from the existing 0.2s timer: ``push(event)`` per new event, then ``flush()``.

Event kinds handled (shapes unchanged from the agent loop):
  stage_start{stage} reasoning_delta{text} tool{name,args} tool_result{name,output}
  retry{stage,errors} stage_done{stage,ok,cost,attempts}
``answer_delta`` (the JSON draft) is intentionally ignored: it is the result, not
the thinking, and belongs in the downloaded project rather than the live log.
"""
from __future__ import annotations

import json

from nicegui import ui

# Per-stage identity: (human label, Material icon, accent hex). The accent drives
# the section's left border, header wash, icon and title so each phase transition
# is visible at a glance. Order mirrors DESIGN_STAGES + the deterministic
# synthesize step + the exception path.
STAGE_META: dict[str, tuple[str, str, str]] = {
    "intent":          ("Intent",            "lightbulb",    "#38bdf8"),  # sky
    "functional_spec": ("Functional Spec",   "list_alt",     "#a78bfa"),  # violet
    "architecture":    ("Architecture",      "account_tree", "#22d3ee"),  # cyan
    "bom":             ("Bill of Materials", "inventory_2",  "#fbbf24"),  # amber
    "wiring":          ("Wiring",            "cable",        "#34d399"),  # emerald
    "synthesize":      ("Synthesize",        "bolt",         "#f472b6"),  # pink
    "build":           ("Build",             "memory",       "#818cf8"),  # indigo
    "error":           ("Error",             "error",        "#f87171"),  # red
}
_DEFAULT_META = ("Stage", "tune", "#94a3b8")

_OK = "#34d399"
_FAIL = "#f87171"
_DIM = "#94a3b8"      # slate-400, secondary text
_DIMMER = "#64748b"   # slate-500, tertiary / glyphs
_RESULT_FOLD_OVER = 300  # tool results longer than this fold into an expansion


class _Run:
    """A streaming, collapsible text block (a reasoning run or the build log): its
    expansion, the label tokens stream into, and the growing text buffer."""

    __slots__ = ("exp", "label", "buf", "head", "mode")

    def __init__(self, exp, label, head="Thinking", mode="chars"):
        self.exp = exp
        self.label = label
        self.buf = ""
        self.head = head      # header title prefix
        self.mode = mode      # "chars" (reasoning) or "lines" (build log)


class FeedView:
    """Incrementally renders design events into a NiceGUI column.

    `container` is the column (inside the page's scroll area) that sections are
    appended to. All element creation happens lazily as events arrive, so this is
    safe to drive from a `ui.timer` callback (same client context as the page).
    """

    def __init__(self, container) -> None:
        self.col = container
        self._current_stage: str | None = None
        self._body = None            # column for the active stage's rows
        self._status_slot = None     # right-hand header slot (spinner -> pill)
        self._active_run: _Run | None = None        # current append target
        self._open_reasoning = None  # the one expansion currently left open
        self._build_log: _Run | None = None         # the build-phase log block
        self._dirty: set[_Run] = set()
        self._placeholder = None
        self.clear()

    # ---- lifecycle ----------------------------------------------------------
    def clear(self) -> None:
        """Reset for a fresh run and show an idle placeholder."""
        self.col.clear()
        self._current_stage = None
        self._body = None
        self._status_slot = None
        self._active_run = None
        self._open_reasoning = None
        self._build_log = None
        self._dirty.clear()
        with self.col:
            self._placeholder = ui.label("Your design log will appear here as each "
                                         "stage streams in.").classes("text-sm italic") \
                .style(f"color:{_DIMMER}")

    def push(self, e: dict) -> None:
        k = e.get("kind")
        if k == "reasoning_delta":
            self._on_reasoning(e.get("text", ""))
        elif k == "stage_start":
            self._start_stage(e.get("stage"))
        elif k == "tool":
            self._on_tool(e.get("name", ""), e.get("args") or {})
        elif k == "tool_result":
            self._on_tool_result(str(e.get("output", "")))
        elif k == "retry":
            self._on_retry(e.get("errors"))
        elif k == "stage_done":
            self._on_stage_done(e.get("stage"), bool(e.get("ok")), e.get("cost"))
        elif k == "build_start":
            self._on_build_start()
        elif k == "build_log":
            self._on_build_log(e.get("text", ""))
        elif k == "build_done":
            self._on_build_done(bool(e.get("ok")))
        # answer_delta and any unknown kind: ignored on purpose.

    def flush(self) -> None:
        """Write coalesced streamed text once per tick (cheap; one DOM update per
        growing block instead of one per token)."""
        for run in self._dirty:
            run.label.set_text(run.buf)
            n = run.buf.count("\n") if run.mode == "lines" else len(run.buf)
            unit = "lines" if run.mode == "lines" else "chars"
            run.exp.set_text(f"{run.head} · {n:,} {unit}")
        self._dirty.clear()

    # ---- stage sections -----------------------------------------------------
    def _start_stage(self, stage: str | None) -> None:
        self._collapse_open()
        self._active_run = None
        if self._placeholder is not None:
            self._placeholder.delete()
            self._placeholder = None
        label, icon, accent = STAGE_META.get(stage, _DEFAULT_META)
        with self.col:
            sec = ui.column().classes("w-full gap-0 rounded-lg overflow-hidden mb-3") \
                .style(f"border-left:3px solid {accent};background:rgba(148,163,184,0.04)")
            with sec:
                hdr = ui.row().classes("w-full items-center justify-between px-3 py-2 gap-2") \
                    .style(f"background:linear-gradient(90deg,{accent}24,transparent)")
                with hdr:
                    with ui.row().classes("items-center gap-2 min-w-0"):
                        ui.icon(icon).style(f"color:{accent};font-size:1.3rem")
                        ui.label(label).classes(
                            "text-sm font-bold uppercase tracking-wide truncate") \
                            .style(f"color:{accent}")
                    self._status_slot = ui.row().classes("items-center gap-2 shrink-0")
                    with self._status_slot:
                        ui.spinner(size="sm").style(f"color:{accent}")
                self._body = ui.column().classes("w-full gap-1 px-3 pb-2 pt-1")
        self._current_stage = stage

    def _ensure_stage(self, stage: str | None) -> None:
        """Open a section for `stage` if it is not already the current one (covers
        the exception path, which emits stage_done without a matching start)."""
        if stage and (self._current_stage != stage or self._body is None):
            self._start_stage(stage)

    def _on_stage_done(self, stage, ok: bool, cost) -> None:
        self._ensure_stage(stage)
        self._collapse_open()
        self._active_run = None
        self._set_status(ok, cost)

    def _set_status(self, ok: bool, cost=None) -> None:
        """Replace the active section's header spinner with a result pill."""
        if self._status_slot is None:
            return
        self._status_slot.clear()
        with self._status_slot:
            if ok:
                ui.icon("check_circle").style(f"color:{_OK};font-size:1.15rem")
                if isinstance(cost, (int, float)):
                    ui.label(f"${cost:.4f}").classes("text-xs font-mono") \
                        .style(f"color:{_DIM}")
            else:
                ui.icon("cancel").style(f"color:{_FAIL};font-size:1.15rem")
                ui.label("failed").classes("text-xs").style(f"color:{_FAIL}")

    # ---- build phase (deterministic synthesize -> place -> route -> fab) -----
    def _on_build_start(self) -> None:
        self._start_stage("build")
        self._build_log = None

    def _on_build_log(self, text: str) -> None:
        if self._body is None or self._current_stage != "build":
            self._start_stage("build")
            self._build_log = None
        if self._build_log is None:
            with self._body:
                exp = ui.expansion("Build log", icon="terminal", value=True) \
                    .classes("w-full").props('dense expand-separator '
                                             'header-class="text-xs text-grey-5"')
                with exp:
                    lab = ui.label("").classes(
                        "text-xs font-mono whitespace-pre-wrap leading-relaxed") \
                        .style(f"color:{_DIM}")
            self._build_log = _Run(exp, lab, head="Build log", mode="lines")
            self._open_reasoning = exp  # fold it on build_done, like a reasoning run
        self._build_log.buf += text + "\n"
        self._dirty.add(self._build_log)

    def _on_build_done(self, ok: bool) -> None:
        self._collapse_open()
        self._build_log = None
        self._set_status(ok)

    # ---- reasoning ----------------------------------------------------------
    def _on_reasoning(self, text: str) -> None:
        if not text:
            return
        if self._body is None:
            self._start_stage(self._current_stage or "intent")
        if self._active_run is None:
            self._open_run()
        self._active_run.buf += text
        self._dirty.add(self._active_run)

    def _open_run(self) -> None:
        # one open at a time: fold the previous thinking block as a new one starts.
        self._collapse_open()
        with self._body:
            exp = ui.expansion("Thinking", icon="psychology", value=True) \
                .classes("w-full").props('dense expand-separator '
                                         'header-class="text-xs text-grey-5"')
            with exp:
                lab = ui.label("").classes(
                    "text-xs font-mono whitespace-pre-wrap leading-relaxed") \
                    .style(f"color:{_DIM}")
        self._active_run = _Run(exp, lab)
        self._open_reasoning = exp

    def _collapse_open(self) -> None:
        if self._open_reasoning is not None:
            self._open_reasoning.value = False
            self._open_reasoning = None

    # ---- tools / retries ----------------------------------------------------
    def _on_tool(self, name: str, args: dict) -> None:
        self._active_run = None  # a tool ends the current thinking run
        if self._body is None:
            self._start_stage(self._current_stage or "bom")
        preview = json.dumps(args)[:140] if args else ""
        with self._body:
            with ui.row().classes("items-center gap-2 flex-nowrap min-w-0 pt-0.5"):
                ui.icon("terminal").style(f"color:{_DIMMER};font-size:1rem")
                ui.label(name).classes("text-xs font-mono px-1.5 py-0.5 rounded shrink-0") \
                    .style("background:rgba(56,189,248,0.14);color:#7dd3fc")
                if preview:
                    ui.label(preview).classes("text-xs font-mono truncate min-w-0") \
                        .style(f"color:{_DIMMER}")

    def _on_tool_result(self, output: str) -> None:
        self._active_run = None
        if self._body is None:
            self._start_stage(self._current_stage or "bom")
        with self._body:
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
        if self._body is None:
            self._start_stage(self._current_stage or "intent")
        msg = json.dumps(errors)[:200] if errors is not None else ""
        with self._body:
            with ui.row().classes("items-center gap-2 flex-nowrap min-w-0 px-2 py-1 rounded") \
                    .style("background:rgba(251,191,36,0.10)"):
                ui.icon("warning").style("color:#fbbf24;font-size:1rem")
                ui.label(f"retry: {msg}").classes("text-xs font-mono truncate min-w-0") \
                    .style("color:#fcd34d")


def demo_events() -> list[dict]:
    """A realistic canned event stream (the flashlight brief) for offline preview.

    Used by the KICRAFT_WEB_DEMO replay page so the styling can be screenshotted
    without spending or network. Exercises every branch: reasoning, multi-tool BOM,
    a wiring retry, and the deterministic synthesize step.
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
        "synthesize: wrote FLASHLIGHT.kicad_sch (6 symbols, 5 nets)",
        "place: 9 footprints placed, antenna keep-outs honored",
        "route: 18/18 nets routed, 0 DRC violations",
        "verify: ERC clean, DRC clean",
        "fab: gerbers + drill + BOM + centroid -> generated/FLASHLIGHT/fab",
    ):
        ev.append({"kind": "build_log", "text": line})
    ev.append({"kind": "build_done", "ok": True})
    return ev
