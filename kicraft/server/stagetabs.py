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
  build_start queue{position,depth,eta_s} build_log{text} build_done{ok}
Both ``reasoning_delta`` (the model's reasoning channel) and ``answer_delta`` (its
content draft) stream into the Thinking window so it fills live even for models /
tool-free stages that only emit content; the committed result still lands,
structured, in the Project State window.
"""
from __future__ import annotations

import json
import time
from html import escape

from nicegui import app, ui

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
_WARN = "#eab308"      # yellow-500: succeeded-with-a-caution (e.g. minor courtyard clip)
_STATUS_COLOR = {
    "pending": "#64748b",
    "active": "#fbbf24",
    "parked": "#fbbf24",   # waiting on the user's answer (amber, like active)
    "done": "#34d399",
    "warning": _WARN,      # build succeeded but carries a non-blocking warning
    "failed": "#f87171",
}
_RESULT_FOLD_OVER = 300  # tool results longer than this fold into an expansion


def _follow_head() -> None:
    """Inject the tail-follow assets (static/kc_follow.*) once per client.

    Idempotent within a client connection via app.storage.client, so building a
    StageTabs adds a single <link>/<script>. Must be called inside a @ui.page
    handler (where the client context exists)."""
    try:
        flag = app.storage.client
    except Exception:
        flag = None
    if flag is not None:
        if flag.get("_kc_follow_head"):
            return
        flag["_kc_follow_head"] = True
    ui.add_head_html('<link rel="stylesheet" href="/static/kc_follow.css">')
    ui.add_head_html('<script src="/static/kc_follow.js" defer></script>')


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

    def __init__(self, key: str, label: str, icon: str, accent: str,
                 show_cost: bool = False) -> None:
        self.key = key
        self.label = label
        self.accent = accent
        # Per-stage LLM cost is admin-only telemetry; regular users never see a
        # dollar figure for a design round (the spend is still tracked server-side).
        self._show_cost = show_cost
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

            # Fill the viewport under the tab row: the windows used to be a short
            # 62vh band with a large empty area below. min-height keeps them usable
            # on short screens.
            #
            # The column split depends on the phase. The build phases (synthesize /
            # place_route / fab) render the native KiCad view (KiCanvas) as their
            # project state, and that view is the artifact the user wants to inspect,
            # so the inspector column takes most of the width there. The LLM stages
            # have no view and lead with the reasoning stream, so they keep Thinking
            # as the larger pane.
            left_w, left_min = ("72%", "440px") if self.key in _BUILD_STAGES \
                else ("42%", "300px")
            with ui.row().classes("w-full no-wrap gap-3 kc-stage-body").style(
                    "height:calc(100vh - 320px);min-height:540px"):
                # LEFT: project-state inspector (+ view slot for KiCanvas/download).
                with ui.column().classes("gap-1 kc-stage-left").style(
                        f"width:{left_w};min-width:{left_min};height:100%"):
                    ui.label("Project state").classes(
                        "text-xs font-bold uppercase tracking-wide").style(f"color:{_DIM}")
                    insp = ui.scroll_area().classes("w-full rounded kc-stage-insp").style(
                        "flex:1;min-height:0;background:#0f172a;border:1px solid #1e293b")
                    with insp:
                        self.view_slot = ui.column().classes("w-full p-2 gap-2")
                        self._insp = ui.column().classes("w-full p-2 gap-3")

                # RIGHT: thinking (top, the star of the show) over activity/log.
                # Plain overflow containers (not ui.scroll_area) so native scroll
                # events fire only on real position changes; tail-follow to the
                # bottom is handled client-side by kc_follow.js (.kc-follow).
                with ui.column().classes("gap-1 kc-stage-right").style(
                        "flex:1;min-width:0;height:100%"):
                    ui.label("Thinking").classes(
                        "text-xs font-bold uppercase tracking-wide").style(f"color:{_DIM}")
                    with ui.element("div").classes(
                            "w-full rounded kc-follow kc-stage-think").style(
                            "height:58%;overflow-y:auto;"
                            "background:#0f172a;border:1px solid #1e293b"):
                        self._think = ui.column().classes("w-full p-2 gap-0")

                    ui.label("Activity / log").classes(
                        "text-xs font-bold uppercase tracking-wide mt-1").style(f"color:{_DIM}")
                    with ui.element("div").classes(
                            "w-full rounded kc-follow kc-stage-act").style(
                            "flex:1;min-height:0;overflow-y:auto;"
                            "background:#0f172a;border:1px solid #1e293b"):
                        self._act = ui.column().classes("w-full p-2 gap-1")

        self.clear()

    # ---- lifecycle ----------------------------------------------------------
    def clear(self) -> None:
        """Reset all three windows to idle placeholders for a fresh run."""
        self._active_run = None
        self._open_run = None
        self._build_log = None
        self._dirty.clear()
        # Live project-state draft (the slot JSON the model is writing).
        self._draft_buf = ""
        self._draft_dirty = False
        self._committed = False
        self._saw_reasoning = False
        # Activity diagnostic line (model / elapsed / chars / tool calls).
        self._live = None
        self._live_done = False
        self._t0 = None
        self._chars = 0
        self._tools = 0
        self._model = None
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
        if k == "reasoning_delta":
            # The model's reasoning channel -> Thinking window.
            self._on_reasoning(e.get("text", ""))
        elif k == "answer_delta":
            # The model's content draft = the slot JSON it is writing -> a live
            # preview in Project state (and Thinking too for content-only models
            # that emit no reasoning, so that window never stays empty).
            self._on_answer(e.get("text", ""))
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
        block instead of one per token), refresh the live project-state draft, and
        tick the activity diagnostic line."""
        for run in self._dirty:
            run.label.set_text(run.buf)
            n = run.buf.count("\n") if run.mode == "lines" else len(run.buf)
            unit = "lines" if run.mode == "lines" else "chars"
            run.exp.set_text(f"{run.head} · {n:,} {unit}")
        self._dirty.clear()
        if self._draft_dirty and not self._committed:
            self._draft_dirty = False
            self._render_draft()
        if self._live is not None and not self._live_done:
            self._live.set_text(self._live_text())

    # ---- status -------------------------------------------------------------
    def mark_running(self, model: str | None = None) -> None:
        self._status_slot.clear()
        with self._status_slot:
            ui.spinner(size="sm").style(f"color:{self.accent}")
        # Build sub-phases stream a build log instead of per-token diagnostics.
        if self.key in _BUILD_STAGES:
            return
        # Seed the activity diagnostic so the (tool-free) early stages are never an
        # empty pane: a start line + a live "streaming ..." line updated each flush.
        self._model = model
        self._t0 = time.monotonic()
        self._chars = 0
        self._tools = 0
        self._live_done = False
        self._act_ready()
        with self._act:
            head = f"▶ {self.label} started"
            if model:
                head += f"  ·  {model}"
            ui.label(head).classes("text-xs font-mono").style(f"color:{self.accent}")
            self._live = ui.label("streaming…").classes("text-xs font-mono") \
                .style(f"color:{_DIMMER}")

    def set_queued(self, position: int, depth: int, eta_s=None) -> None:
        """Queue pill: the run's deterministic build is waiting for a host build
        slot (other users' builds are ahead). Replaced by the normal running
        spinner as soon as the first build log line lands."""
        self._status_slot.clear()
        with self._status_slot:
            ui.spinner("hourglass", size="sm").style(f"color:{_DIM}")
            msg = ("Queued: next up" if position <= 0
                   else f"Queued: {position} build{'s' if position != 1 else ''} ahead")
            if isinstance(eta_s, (int, float)) and eta_s > 0:
                msg += f" · est. ~{max(1, round(eta_s / 60))} min"
            ui.label(msg).classes("text-xs").style(f"color:{_DIM}")

    def set_status(self, ok: bool, cost=None, attempts=None) -> None:
        self._status_slot.clear()
        with self._status_slot:
            if ok:
                ui.icon("check_circle").style(f"color:{_OK};font-size:1.1rem")
                if self._show_cost and isinstance(cost, (int, float)):
                    ui.label(f"${cost:.4f}").classes("text-xs font-mono") \
                        .style(f"color:{_DIM}")
            else:
                ui.icon("cancel").style(f"color:{_FAIL};font-size:1.1rem")
                ui.label("failed").classes("text-xs").style(f"color:{_FAIL}")
        # Freeze the activity diagnostic line into a final per-stage summary.
        if self._live is not None and not self._live_done:
            self._live_done = True
            self._live.set_text(self._live_text(done=True, ok=ok, cost=cost, attempts=attempts))
            self._live.style(f"color:{_OK if ok else _FAIL}")

    def set_parked(self) -> None:
        """Pill for a stage parked on a clarifying question: nothing is running,
        the user owes an answer (a live park or a reopened parked project)."""
        self._status_slot.clear()
        with self._status_slot:
            ui.icon("help").style("color:#fbbf24;font-size:1.1rem")
            ui.label("waiting for your answer").classes("text-xs").style("color:#fbbf24")

    def set_pending(self) -> None:
        """Drop any result pill: the stage's outcome was invalidated (an upstream
        edit cleared its slot) and it has not run again yet."""
        self._status_slot.clear()

    def _live_text(self, done=False, ok=True, cost=None, attempts=None) -> str:
        elapsed = (time.monotonic() - self._t0) if self._t0 else 0.0
        head = ("✓ committed" if ok else "✗ failed") if done else "streaming"
        parts = [head, f"{elapsed:.1f}s"]
        if done and self._show_cost and isinstance(cost, (int, float)):
            parts.append(f"${cost:.4f}")
        parts.append(f"{self._chars:,} chars")
        if self._tools:
            parts.append(f"{self._tools} tool calls")
        if done and isinstance(attempts, int) and attempts > 1:
            parts.append(f"{attempts} attempts")
        return "  ·  ".join(parts)

    # ---- thinking -----------------------------------------------------------
    def _on_reasoning(self, text: str) -> None:
        if not text:
            return
        self._saw_reasoning = True
        self._append_thinking(text)

    def _on_answer(self, text: str) -> None:
        """The model's content draft is the slot JSON being written. Feed it to the
        live Project-state preview; mirror it into Thinking only when the model
        emitted no reasoning channel (so content-only models still fill that pane)."""
        if not text:
            return
        self._draft_buf += text
        self._draft_dirty = True
        self._chars += len(text)
        if not self._saw_reasoning:
            self._append_thinking(text, count=False)

    def _append_thinking(self, text: str, count: bool = True) -> None:
        if not text:
            return
        if count:
            self._chars += len(text)
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
                        "text-sm font-mono whitespace-pre-wrap leading-relaxed") \
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
        self._tools += 1
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
          {"type": "graph", "title": str, "option": <echarts option dict>}

        Empty `sections` means "nothing committed yet": keep an in-progress live
        draft on screen rather than wiping it back to the placeholder (the draft
        owns the pane until this stage commits). Non-empty sections are the
        validated result and supersede the draft.
        """
        if not sections:
            if self._draft_buf and not self._committed:
                return
            self._committed = False  # slot cleared (an upstream edit): a new
            self._insp.clear()       # run's draft may own the pane again
            with self._insp:
                ui.label("No data committed for this stage yet.") \
                    .classes("text-xs italic").style(f"color:{_DIMMER}")
            return
        self._committed = True
        self._draft_buf = ""
        self._draft_dirty = False
        self._insp.clear()
        with self._insp:
            for sec in sections:
                _render_section(sec, self.accent)

    def _render_draft(self) -> None:
        """Show the slot JSON the model is currently writing as a live, uncommitted
        preview in the Project-state window (pretty-printed when it parses, else the
        raw streaming text). Replaced by the validated view once the stage commits."""
        pretty = _loose_pretty(self._draft_buf)
        body = pretty if pretty is not None else self._draft_buf
        self._insp.clear()
        with self._insp:
            ui.label(f"● writing {self.key} slot…  ·  {len(self._draft_buf):,} chars") \
                .classes("text-xs font-semibold").style(f"color:{self.accent}")
            ui.label(body).classes(
                "text-xs font-mono whitespace-pre-wrap").style(f"color:{_DIM}")

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
            ui.html(_table_html(cols, rows, sec.get("foot")), sanitize=False)
        note = sec.get("note")
        if note:
            ui.label(str(note)).classes("text-xs italic mt-0.5").style(f"color:{_DIMMER}")
    elif kind == "graph":
        ui.echart(sec.get("option", {})).classes("w-full").style("height:360px")


def _cell_html(cell) -> str:
    """One <td> body. A cell is either a scalar (rendered as escaped text) or a
    ``{"text", "href"}`` dict (rendered as a new-tab link, e.g. a vendor lookup)."""
    if isinstance(cell, dict):
        text = escape(str(cell.get("text", "")))
        href = str(cell.get("href") or "")
        # Only emit real web links; anything else (javascript:, data:, ...) falls
        # back to plain text so a cell can never inject a clickable script URL.
        if href[:7].lower() == "http://" or href[:8].lower() == "https://":
            return (f'<a href="{escape(href)}" target="_blank" '
                    f'rel="noopener noreferrer">{text}</a>')
        return text
    return escape("" if cell is None else str(cell))


def _table_html(cols: list, rows: list, foot: list | None = None) -> str:
    """A real <table> for an inspector section so columns line up across every row
    (the prior per-row flex layout sized each row independently, so they didn't).
    Optional ``foot`` rows render in a <tfoot> (e.g. the BOM total). All dynamic
    text is HTML-escaped; only hrefs the page supplies -- vendor URLs we build
    ourselves in web._vendor_cell -- become links."""
    def trow(r):
        return "<tr>" + "".join(f"<td>{_cell_html(c)}</td>" for c in r) + "</tr>"
    head = "".join(f"<th>{escape(str(c))}</th>" for c in cols)
    body = "".join(trow(r) for r in rows)
    tfoot = f"<tfoot>{''.join(trow(r) for r in foot)}</tfoot>" if foot else ""
    return (f'<table class="kc-table"><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody>{tfoot}</table>")


def _loose_pretty(buf: str) -> str | None:
    """Best-effort pretty-print of a partial slot-JSON draft (the model's streaming
    content). Returns indented JSON when the buffer parses, whole or after closing
    its unbalanced brackets/strings; else None so the caller shows the raw text.
    Never raises."""
    if not buf:
        return None
    try:
        s = buf.strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[1] if "\n" in s else s.lstrip("`")
            if s.endswith("```"):
                s = s[:-3]
        i = s.find("{")
        if i < 0:
            return None
        s = s[i:]
        try:
            return json.dumps(json.loads(s), indent=2)
        except (json.JSONDecodeError, ValueError):
            pass
        repaired = _close_json(s)
        if repaired is None:
            return None
        return json.dumps(json.loads(repaired), indent=2)
    except Exception:
        return None


def _close_json(s: str) -> str | None:
    """Append the minimal closers to make a truncated JSON object parseable: finish
    an open string, drop a dangling trailing comma, then close open `[`/`{` in
    stack order. Returns None when nothing is open (so a balanced-but-invalid buffer
    falls through to the raw-text path)."""
    stack: list[str] = []
    in_str = esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append(ch)
        elif ch in "]}" and stack:
            stack.pop()
    if not stack and not in_str:
        return None
    out = s + ('"' if in_str else "")
    tail = out.rstrip()
    if tail.endswith(","):
        out = tail[:-1]
    closers = {"[": "]", "{": "}"}
    out += "".join(closers[c] for c in reversed(stack))
    return out


class StageTabs:
    """The tab row + tab panels, with event routing and status-coloured tabs.

    Built once inside a page layout. The page feeds it streaming events with
    ``push`` / ``flush`` and supplies committed data with ``set_inspector``. The
    active tab auto-follows the running stage until the user clicks a different
    tab (then it stays put until they click back to the live one).
    """

    def __init__(self, show_cost: bool = False) -> None:
        _follow_head()
        self.show_cost = show_cost
        self.panels: dict[str, StagePanel] = {}
        self._tab_el: dict[str, ui.tab] = {}
        self._current: str | None = None
        self._auto_follow = True
        self._on_show: dict[str, object] = {}

        with ui.tabs().classes("w-full kc-stage-tabs") \
                .props("dense inline-label mobile-arrows") as self.tabs:
            for key, label, icon, accent in PHASES:
                t = ui.tab(key, label=label, icon=icon)
                t.style(f"color:{_STATUS_COLOR['pending']}")
                self._tab_el[key] = t
        self.tabs.on_value_change(self._on_tab_change)

        with ui.tab_panels(self.tabs, value=PHASES[0][0]).classes("w-full") \
                .style("background:transparent"):
            for key, label, icon, accent in PHASES:
                with ui.tab_panel(key).classes("p-0"):
                    self.panels[key] = StagePanel(key, label, icon, accent,
                                                  show_cost)

    # ---- tab status / follow ------------------------------------------------
    def _set_tab_status(self, key: str, status: str) -> None:
        t = self._tab_el.get(key)
        if t is not None:
            t.style(f"color:{_STATUS_COLOR[status]}")

    def _on_tab_change(self, e) -> None:
        val = getattr(e, "value", None)
        # Resume auto-follow only while the user is parked on the live stage.
        self._auto_follow = (val == self._current)
        # Re-fit any view built while this tab was hidden: a hidden KiCanvas WebGL
        # canvas sizes to zero and never repaints, so it would show blank otherwise.
        cb = self._on_show.get(val)
        if cb is not None:
            cb()

    def _set_current(self, key: str | None, model: str | None = None) -> None:
        if key is None or key not in self.panels:
            return
        # Finishing one stage and entering the next: fold the previous panel's runs.
        if self._current and self._current != key:
            self.panels[self._current].end_runs()
        self._current = key
        self._set_tab_status(key, "active")
        self.panels[key].mark_running(model)
        if self._auto_follow:
            self.tabs.set_value(key)

    # ---- event routing ------------------------------------------------------
    def push(self, e: dict) -> None:
        k = e.get("kind")
        if k == "stage_start":
            self._set_current(e.get("stage"), e.get("model"))
        elif k == "stage_done":
            self._finish(e.get("stage") or self._current, bool(e.get("ok")),
                         e.get("cost"), e.get("attempts"))
        elif k == "question":
            # The stage parked on a clarifying question: stop the spinner, say
            # the run is waiting on the user (it is not failed, not running).
            stg = e.get("stage") or self._current
            p = self.panels.get(stg)
            if p is not None:
                p.end_runs()
                p.set_parked()
                self._set_tab_status(stg, "parked")
        elif k == "build_start":
            self._set_current("synthesize")
        elif k == "queue":
            # The whole deterministic build is parked in the host build queue;
            # surface position/ETA on the tab build_start just activated.
            self._set_current("synthesize")
            self.panels["synthesize"].set_queued(
                int(e.get("position") or 0), int(e.get("depth") or 0), e.get("eta_s"))
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

    def _finish(self, key: str | None, ok: bool, cost, attempts=None) -> None:
        if key is None or key not in self.panels:
            return
        self.panels[key].end_runs()
        self.panels[key].set_status(ok, cost, attempts)
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

    def on_show(self, key: str, fn) -> None:
        """Run `fn` when `key`'s tab becomes the visible one (see _on_tab_change)."""
        self._on_show[key] = fn

    def active(self) -> str | None:
        """The currently selected tab key."""
        return self.tabs.value

    def set_statuses(self, statuses: dict[str, str],
                     stage_status: dict | None = None) -> None:
        """Paint each tab's durable status on a reopened (or edited) project:
        the tab color plus the panel's result pill. `statuses` comes from
        session.derive_stage_statuses; `stage_status` is state.json's persisted
        per-stage outcome block, read for the done pill's cost/attempts. Live
        streaming events layered on top afterwards win (mark_running and
        _finish each repaint the same status slot)."""
        meta = stage_status or {}
        for key, st in statuses.items():
            p = self.panels.get(key)
            if p is None or st not in _STATUS_COLOR:
                continue
            self._set_tab_status(key, st)
            e = meta.get(key) if isinstance(meta.get(key), dict) else {}
            if st == "done":
                p.set_status(True, cost=e.get("cost_usd"), attempts=e.get("attempts"))
            elif st == "warning":
                # Succeeded with a non-blocking caution: keep the panel content
                # (fab package + 3D model) visible; the yellow tab flags the gap.
                p.set_status(True, cost=e.get("cost_usd"), attempts=e.get("attempts"))
            elif st == "failed":
                p.set_status(False)
            elif st == "parked":
                p.set_parked()
            elif st == "pending":
                p.set_pending()

    def reset_stage(self, key: str) -> None:
        """Clear ONE phase back to a pending placeholder (its data was
        invalidated by an upstream edit and it will re-run)."""
        p = self.panels.get(key)
        if p is None:
            return
        p.clear()
        self._set_tab_status(key, "pending")
        if self._current == key:
            self._current = None

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

    def answer(*chunks: str) -> list[dict]:
        return [{"kind": "answer_delta", "text": c} for c in chunks]

    _MODEL = "demo/opus-preview"
    ev: list[dict] = []
    ev.append({"kind": "stage_start", "stage": "intent", "model": _MODEL})
    ev += think("The brief is a flashlight powered by an 18650 cell with USB-C ",
                "recharging. Core functions: USB-C 5V input, a Li-ion charger, the ",
                "18650 cell, a high-power white LED with a constant-current driver, ",
                "and a push-button to cycle modes. No microcontroller is required.")
    ev += answer('{"goal": "USB-C rechargeable 18650 flashlight, no microcontroller",',
                 ' "inferred_expertise": "intermediate",',
                 ' "named_parts": ["TP4056", "18650 cell"],',
                 ' "assumptions": ["USB-C 5V input (defaulted)"],',
                 ' "project_stem": "FLASHLIGHT"}')
    ev.append({"kind": "stage_done", "stage": "intent", "ok": True, "cost": 0.0021})

    ev.append({"kind": "stage_start", "stage": "functional_spec", "model": _MODEL})
    ev += think("Blocks: USB_C_INPUT, CHARGER, BATTERY, LED_DRIVER, LED, CONTROL. ",
                "Rails: VBUS 5.0V from USB, VBAT ~4.2V max from the cell. The driver ",
                "boosts VBAT to the LED forward voltage under constant current.")
    ev += answer('{"blocks": [{"name": "USB_C_INPUT", "category": "power", ',
                 '"purpose": "5V from USB-C"}, {"name": "CHARGER", "category": "power", ',
                 '"purpose": "TP4056 Li-ion charger"}], ',
                 '"assumptions": ["1A charge current (defaulted)"]}')
    ev.append({"kind": "stage_done", "stage": "functional_spec", "ok": True, "cost": 0.0034})

    ev.append({"kind": "stage_start", "stage": "architecture", "model": _MODEL})
    ev += think("Single sheet is fine for this part count. Power nets: VBUS, VBAT, ",
                "GND, plus the switched LED node. TP4056 for the charger, a boost ",
                "constant-current driver for the LED, debounced push-button on CONTROL.")
    ev += answer('{"sheets": [{"name": "MAIN", "stem": "FLASHLIGHT", "function": "all"}], ',
                 '"power_nets": ["VBUS", "VBAT", "GND"], ',
                 '"rail_voltages": {"VBUS": 5.0, "VBAT": 4.2}, "mcu_present": false}')
    ev.append({"kind": "stage_done", "stage": "architecture", "ok": True, "cost": 0.0048})

    ev.append({"kind": "stage_start", "stage": "bom", "model": _MODEL})
    ev += think("I need real symbols and footprints. Start from the curated library, ",
                "then resolve the charger and USB-C connector from LCSC.")
    ev.append({"kind": "tool", "name": "list_parts", "args": {}})
    ev.append({"kind": "tool_result", "name": "list_parts",
               "output": "usb-c-16p   TYPE-C-31-M-12   USB-C receptacle, 16-pin\n"
                         "tp4056      TP4056_C725790   1A Li-ion charger, ESOP-8\n"
                         "Device:R    Resistor_SMD:R_0603_1608Metric\n"
                         "Device:C    Capacitor_SMD:C_0603_1608Metric\n"
                         "Device:LED  LED_SMD:LED_0603_1608Metric\n... 55 bundles"})
    ev += think("Both the USB-C receptacle and the charger are vendored bundles ",
                "(tp4056 is the lipo-charger-1s core default), so nothing needs ",
                "fetching: take the exact ids from list_parts verbatim.")
    ev += think("All symbols and footprints resolve. Emit the BOM slot JSON.")
    ev += answer('{"parts": [{"ref": "U1", "value": "TP4056", ',
                 '"symbol": "tp4056:TP4056_C725790", "footprint": "tp4056:ESOP-8", "sheet": "MAIN"}, ',
                 '{"ref": "J1", "value": "USB-C", "symbol": "usb-c-16p:TYPE-C-31-M-12", ',
                 '"footprint": "usb-c-16p:TYPE-C", "sheet": "MAIN"}]}')
    ev.append({"kind": "stage_done", "stage": "bom", "ok": True, "cost": 0.0431, "attempts": 1})

    ev.append({"kind": "stage_start", "stage": "wiring", "model": _MODEL})
    ev += think("Connect VBUS from USB-C to the charger input, VBAT to the cell and ",
                "the driver input, the LED node through the driver, and CONTROL to the ",
                "button. Tie unused USB-C pins (SBU1/SBU2, shield) to no_connect.")
    ev.append({"kind": "retry", "stage": "wiring",
               "errors": ["pin (U1,4) of TP4056 not covered by a connection or no_connect"]})
    ev += think("Missed the TP4056 PROG pin. Add R_prog from PROG to GND to set the ",
                "charge current, which also covers that pin.")
    ev += answer('{"connections": [{"net_name": "VBUS", "sheet": "MAIN", ',
                 '"endpoints": [{"ref": "J1", "pin": "A4"}, {"ref": "U1", "pin": "4"}]}], ',
                 '"no_connect_pins": [{"ref": "J1", "pin": "A8"}]}')
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
