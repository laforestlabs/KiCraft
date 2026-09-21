"""Per-stage tabbed view of the live design feed.

The web app streams one event per token/tool/stage from the agent loop (see
``stage_driver``/``client``). The old ``FeedView`` rendered them all into one
long scrolling column. ``StageTabs`` instead gives every pipeline phase its own
tab, and inside each tab lays out a stage body:

  * PROJECT STATE - the structured data this stage committed (the parts
    list, the nets, the sheets, ...), rebuilt from ``state.json`` for inspection,
    plus the native KiCad view / download for the build phases. This is what the
    user came to inspect, so it leads and gets the full width.
  * A concise OUTCOME / current-activity line under it, always on screen.
  * TECHNICAL DETAILS - collapsed by default, holding the EXECUTION / LOG pane
    (tool cards, retries, diagnostics, work-unit provenance, build log) and, in
    its own further-collapsed expansion, the model's REASONING stream.

Collapsing the streams by default is deliberate: the previous fixed 42%/58%
inspector/thinking split left half the screen empty whenever a stage had nothing
to stream, and buried the artifact under a thinking pane before it existed.

The caller drives it from the page's timer exactly like before: ``push(event)``
per new event, then ``flush()`` once. The committed slot data is supplied
separately via ``set_inspector(stage, spec)`` (the page reads ``state.json`` and
builds the spec; this module only renders it).

Event kinds handled include:
  stage_start{stage} reasoning_delta{text} answer_delta{text}
  tool{call_id,name,args} tool_result{call_id,name,output,duration_ms,cached,ok}
  retry{stage,errors,failure_kind} stage_diagnostic{code,message,evidence}
  run_error{failure_kind,exception_type,message}
  recipe_selected work_unit_plan work_unit_attempt work_unit_done
  build_start queue{position,depth,eta_s} build_log{text} build_done{ok}
Both ``reasoning_delta`` (the model's reasoning channel) and ``answer_delta`` (its
content draft) fill the reasoning pane; the committed result still lands,
structured, in the Project State window. An unknown kind is ignored rather than
being attributed to whichever stage happened to run last.
"""

from __future__ import annotations

import json
import time
from html import escape

from nicegui import app, ui

from . import activity as _activity

# Phase identity: (key, label, Material icon, accent hex). Order is the pipeline
# order and drives both the tab row and each panel's accenting. The first five
# are the LLM design stages (DESIGN_STAGES); the last four are the deterministic
# build sub-phases, fed from the single `kicraft build` log stream.
PHASES: list[tuple[str, str, str, str]] = [
    ("intent", "Intent", "lightbulb", "#38bdf8"),  # sky
    ("functional_spec", "Functional", "list_alt", "#a78bfa"),  # violet
    ("architecture", "Architecture", "account_tree", "#22d3ee"),  # cyan
    ("bom", "BOM", "inventory_2", "#fbbf24"),  # amber
    ("wiring", "Wiring", "cable", "#34d399"),  # emerald
    ("synthesize", "Synthesize", "bolt", "#f472b6"),  # pink
    ("place_route", "Place/Route", "developer_board", "#60a5fa"),  # blue
    ("electrical_review", "Elec Review", "plagiarism", "#f59e0b"),  # amber-yellow
    ("fab", "Fab", "inventory", "#818cf8"),  # indigo
]
_BUILD_STAGES = ("synthesize", "place_route", "electrical_review", "fab")

_OK = "#34d399"
_FAIL = "#f87171"
_DIM = "#94a3b8"  # slate-400, secondary text
_DIMMER = "#64748b"  # slate-500, tertiary / glyphs
_WARN = "#eab308"  # yellow-500: succeeded-with-a-caution (e.g. minor courtyard clip)
_STATUS_COLOR = {
    "pending": "#64748b",
    "active": "#fbbf24",
    "parked": "#fbbf24",  # waiting on the user's answer (amber, like active)
    "queued": "#fbbf24",  # waiting for a host build slot
    "done": "#34d399",
    "warning": _WARN,  # build succeeded but carries a non-blocking warning
    "failed": "#f87171",
    "interrupted": _DIMMER,  # lost to a restart: neither failed nor live
}
# Status is never colour-only: every tab also carries this icon and an accessible
# name containing the phase and its status (see StageTabs._set_tab_status).
_STATUS_ICON = {
    "pending": "radio_button_unchecked",
    "active": "autorenew",
    "parked": "help",
    "queued": "hourglass_top",
    "done": "check_circle",
    "warning": "warning",
    "failed": "cancel",
    "interrupted": "link_off",
}
_RESULT_FOLD_OVER = 300  # tool results longer than this fold into an expansion
_PROVENANCE_SOURCE = {
    "recipe": ("Circuit recipe", "verified", "#c084fc"),
    "deterministic_architecture_lowering": ("Deterministic", "functions", "#22d3ee"),
    "reused_validated_draft": ("Reused", "cached", "#60a5fa"),
    "recipe_plus_llm": ("Recipe + LLM", "schema", "#c084fc"),
    "llm": ("LLM", "smart_toy", "#fbbf24"),
}


def _provenance_source(source: object) -> tuple[str, str, str]:
    """Stable user-facing label, icon, and color for an execution source."""
    return _PROVENANCE_SOURCE.get(str(source), (str(source or "Unknown"), "help", _DIM))


_MAX_ERROR_ITEM = 2000


def _error_text(entry) -> str:
    """One readable error item from whatever shape the producer used."""
    if isinstance(entry, dict):
        parts = [
            str(entry.get(key)) for key in
            ("message", "error", "reason", "detail", "code", "ref", "pin", "net")
            if entry.get(key)
        ]
        text = " · ".join(parts) or json.dumps(entry, ensure_ascii=False, default=str)
    else:
        text = str(entry)
    text = " ".join(text.split())
    if len(text) > _MAX_ERROR_ITEM:
        text = text[:_MAX_ERROR_ITEM] + f" …[truncated {len(text) - _MAX_ERROR_ITEM} characters]"
    return text


def _error_items(event: dict) -> list[str]:
    """Every readable error item on a retry event, in a stable order."""
    items: list[str] = []
    for key in ("errors", "offenders"):
        value = event.get(key)
        if isinstance(value, (list, tuple)):
            items.extend(_error_text(entry) for entry in value if entry is not None)
        elif value:
            items.append(_error_text(value))
    for key in ("commit_gate_codes", "work_unit_ids", "declared_identities"):
        value = event.get(key)
        if isinstance(value, (list, tuple)) and value:
            items.append(f"{key.replace('_', ' ')}: " + ", ".join(str(v) for v in value))
    if not items and event.get("message"):
        items.append(_error_text(event["message"]))
    return items


def _render_error_item(text: str) -> None:
    """A complete, bounded error item: short ones inline, long ones expandable."""
    if len(text) <= 200:
        ui.label(text).classes("text-xs whitespace-pre-wrap break-all") \
            .style(f"color:{_DIM}")
        return
    with ui.expansion(f"Error detail · {len(text):,} chars", icon="report") \
            .classes("w-full").props('dense header-class="text-xs text-grey-5"'):
        ui.label(text).classes("text-xs font-mono whitespace-pre-wrap break-all") \
            .style(f"color:{_DIM}")


def _render_tool_output(output: str, event: dict) -> None:
    """A tool result: wrapped, folded when long, and NEVER silently clipped --
    an over-limit result states how much was withheld."""
    chars = event.get("output_chars")
    total = int(chars) if isinstance(chars, int) and chars >= len(output) else len(output)
    truncated = bool(event.get("output_truncated")) or total > len(output)
    note = (f"Showing first {len(output):,} of {total:,} characters"
            if truncated else "")
    if len(output) > _RESULT_FOLD_OVER or note:
        exp = ui.expansion(f"Result · {total:,} chars",
                           icon="subdirectory_arrow_right") \
            .classes("w-full").props('dense header-class="text-xs text-grey-5"')
        with exp:
            if note:
                ui.label(note).classes("text-xs").style(f"color:{_WARN}")
            ui.label(output).classes(
                "text-xs font-mono whitespace-pre-wrap break-all") \
                .style(f"color:{_DIM}")
        return
    ui.label(output).classes("text-xs font-mono whitespace-pre-wrap break-all") \
        .style(f"color:{_DIM}")


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
        self.head = head  # header title prefix
        self.mode = mode  # "chars" (reasoning) or "lines" (build log)


class StagePanel:
    """One phase's tab body: the inspector plus the thinking + activity streams
    (LLM stages: inspector left, streams right; build stages: full-width
    artifact on top, streams underneath — see the module docstring).

    Built once into the surrounding ``ui.tab_panel`` context. Streaming events are
    fed via ``push``; ``flush`` coalesces the growing text blocks; ``set_inspector``
    (re)renders the structured project-state. ``view_slot`` is an empty column at
    the top of the inspector column for the page to drop a KiCanvas view / download
    button into (the build phases), kept separate from the data area so refreshing
    the data never clobbers the view.
    """

    def __init__(
        self, key: str, label: str, icon: str, accent: str, show_cost: bool = False, draft_spec=None
    ) -> None:
        self.key = key
        self.label = label
        self.accent = accent
        # Per-stage LLM cost is admin-only telemetry; regular users never see a
        # dollar figure for a design round (the spend is still tracked server-side).
        self._show_cost = show_cost
        # Page-supplied `(stage_key, parsed_draft) -> sections` builder: lets the
        # live draft render through the same structured sections as the committed
        # view (the BOM table fills in row by row instead of showing raw JSON).
        self._draft_spec = draft_spec
        self._active_run: _Run | None = None
        self._open_run: _Run | None = None
        self._build_log: _Run | None = None
        self._dirty: set[_Run] = set()

        with ui.column().classes("w-full gap-2 p-2"):
            # Status bar: a spinner while the stage runs, a result pill when done.
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon(icon).style(f"color:{accent};font-size:1.2rem")
                ui.label(label).classes("text-sm font-bold uppercase tracking-wide").style(
                    f"color:{accent}"
                )
                self._status_slot = ui.row().classes("items-center gap-2")

            # The stage body: the artifact / project state first (it is what the
            # user came to inspect), then a concise outcome line, then the
            # reasoning + execution streams folded under "Technical details".
            # Collapsed by default: a quiet stage must not leave half the screen
            # empty the way the old fixed 42%/58% Thinking-first split did.
            def _inspector() -> None:
                ui.label("Project state").classes(
                    "text-xs font-bold uppercase tracking-wide"
                ).style(f"color:{_DIM}")
                insp = (
                    ui.scroll_area()
                    .classes("w-full rounded kc-stage-insp")
                    .style(
                        "flex:1;min-height:0;background:var(--kc-surface);border:1px solid var(--kc-border)"
                    )
                )
                with insp:
                    self.view_slot = ui.column().classes("w-full p-2 gap-2")
                    # Persistent issues for THIS stage (severity, message, code,
                    # evidence, jump-to-stage). Kept out of _insp so rebuilding the
                    # committed project state never wipes the findings.
                    self.issues_slot = ui.column().classes("w-full p-2 gap-2")
                    self.issues_slot.set_visibility(False)
                    self._issues_sig: tuple = ()
                    self._insp = ui.column().classes("w-full p-2 gap-3")

            def _thinking() -> None:
                with (
                    ui.element("div")
                    .classes("w-full rounded kc-follow kc-stage-think")
                    .style(
                        "min-height:120px;max-height:320px;overflow-y:auto;"
                        "background:var(--kc-surface);border:1px solid var(--kc-border)"
                    )
                ):
                    self._think = ui.column().classes("w-full p-2 gap-0")

            def _activity() -> None:
                ui.label("Execution / log").classes(
                    "text-xs font-bold uppercase tracking-wide"
                ).style(f"color:{_DIM}")
                with (
                    ui.element("div")
                    .classes("w-full rounded kc-follow kc-stage-act")
                    .style(
                        "min-height:120px;max-height:360px;overflow-y:auto;"
                        "background:var(--kc-surface);border:1px solid var(--kc-border)"
                    )
                ):
                    self._act = ui.column().classes("w-full p-2 gap-1")

            with ui.column().classes("w-full gap-2 kc-stage-body"):
                with (
                    ui.column()
                    .classes("w-full gap-1 kc-stage-left")
                    .style("height:calc(100vh - 330px);min-height:520px")
                ):
                    _inspector()
                # The concise stage outcome / current activity, always on screen.
                self._outcome = ui.label("").classes("text-xs font-mono") \
                    .style(f"color:{_DIM}")
                with ui.expansion("Technical details", icon="terminal") \
                        .classes("w-full") \
                        .props('dense header-class="text-xs text-grey-5"'):
                    with ui.column().classes("w-full gap-2 p-1"):
                        _activity()
                        with ui.expansion("Model reasoning", icon="psychology") \
                                .classes("w-full") \
                                .props('dense header-class="text-xs text-grey-5"'):
                            _thinking()

        self.clear()

    # ---- lifecycle ----------------------------------------------------------
    def clear(self) -> None:
        """Reset the stage body to idle placeholders for a fresh run."""
        self._active_run = None
        self._open_run = None
        self._build_log = None
        self._dirty.clear()
        # Live project-state draft (the slot JSON the model is writing).
        self._draft_buf = ""
        self._draft_dirty = False
        self._committed = False
        self._draft_structured = False
        self._saw_reasoning = False
        # Activity diagnostic line (model / elapsed / chars / tool calls).
        self._live = None
        self._live_done = False
        self._t0 = None
        self._chars = 0
        self._tools = 0
        self._model = None
        # Tool cards keyed by their LOCAL call id (minted by the client), so a
        # call and its result become one card instead of two anonymous lines.
        self._tool_cards: dict[str, dict] = {}
        self._last_unpaired: list[dict] = []
        self._status_slot.clear()
        self._outcome.set_text("")
        self._outcome.style(f"color:{_DIM}")
        self.issues_slot.clear()
        self.issues_slot.set_visibility(False)
        self._issues_sig = ()
        self.view_slot.clear()
        self._insp.clear()
        self._think.clear()
        self._act.clear()
        with self._insp:
            ui.label("No data committed for this stage yet.").classes("text-xs italic").style(
                f"color:{_DIMMER}"
            )
        with self._think:
            self._think_ph = (
                ui.label("Reasoning will stream here.")
                .classes("text-xs italic")
                .style(f"color:{_DIMMER}")
            )
        with self._act:
            self._act_ph = (
                ui.label("Tool calls and log output will appear here.")
                .classes("text-xs italic")
                .style(f"color:{_DIMMER}")
            )

    def push(self, e: dict) -> None:
        k = e.get("kind")
        if k == "reasoning_delta":
            # The model's reasoning channel -> the (collapsed) reasoning pane.
            self._on_reasoning(e.get("text", ""))
        elif k == "answer_delta":
            # The model's content draft = the slot JSON it is writing -> a live
            # preview in Project state (and the reasoning pane too for
            # content-only models that emit no reasoning).
            self._on_answer(e.get("text", ""))
        elif k == "tool":
            self._on_tool(e)
        elif k == "tool_result":
            self._on_tool_result(e)
        elif k == "retry":
            self._on_retry(e)
        elif k == "stage_diagnostic":
            self._on_stage_diagnostic(e)
        elif k == "run_error":
            self._on_run_error(e)
        elif k in ("provider_fallback", "escalation", "serialization_recovery",
                   "candidate_decoded"):
            self._on_activity_note(e)
        elif k == "recipe_selected":
            self._on_recipe_selected(e)
        elif k == "work_unit_plan":
            self._on_work_unit_plan(e)
        elif k == "work_unit_attempt":
            self._on_work_unit_attempt(e)
        elif k == "work_unit_done":
            self._on_work_unit_done(e)
        elif k == "build_log":
            self._on_build_log(e.get("text", ""))
        # stage_start/stage_done/build_start/queue/build_done/run_finished are
        # handled by StageTabs (tab status); an unknown kind is ignored here
        # rather than being attributed to some other stage.

    def flush(self) -> None:
        """Write coalesced streamed text once per tick (one DOM update per growing
        block instead of one per token), refresh the live project-state draft, and
        tick the concise outcome line."""
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
            text = self._live_text()
            self._live.set_text(text)
            self._outcome.set_text(text)

    # ---- status -------------------------------------------------------------
    def mark_running(self, model: str | None = None) -> None:
        self._status_slot.clear()
        with self._status_slot:
            ui.spinner(size="sm").style(f"color:{self.accent}")
        # Seed the concise outcome line (and, for LLM stages, the activity pane) so
        # even a tool-free stage is never blank.
        self._model = model
        self._t0 = time.monotonic()
        self._chars = 0
        self._tools = 0
        self._live_done = False
        head = f"▶ {self.label} started"
        if model:
            head += f"  ·  {model}"
        self._outcome.set_text(head)
        self._outcome.style(f"color:{self.accent}")
        # Build sub-phases stream a build log instead of per-token diagnostics.
        if self.key in _BUILD_STAGES:
            return
        self._act_ready()
        with self._act:
            ui.label(head).classes("text-xs font-mono").style(f"color:{self.accent}")
            self._live = (
                ui.label("streaming…").classes("text-xs font-mono").style(f"color:{_DIMMER}")
            )

    def set_outcome(self, text: str, color: str | None = None) -> None:
        """Set the always-visible outcome/current-activity line."""
        self._outcome.set_text(text)
        self._outcome.style(f"color:{color or _DIM}")

    def set_queued_note(self, note: str) -> None:
        """Queue pill from an already-written sentence (a build this page only
        observes: it has no queue EVENT to react to)."""
        self._status_slot.clear()
        with self._status_slot:
            ui.spinner("hourglass_top", size="sm").style(f"color:{_DIM}")
            ui.label(note or "Queued").classes("text-xs").style(f"color:{_DIM}")
        self.set_outcome(note or "Queued for board build", _DIM)

    def set_queued(self, position: int, depth: int, eta_s=None) -> None:
        """Queue pill: the run's deterministic build is waiting for a host build
        slot (other users' builds are ahead). Replaced by the normal running
        spinner as soon as the first build log line lands."""
        self._status_slot.clear()
        with self._status_slot:
            ui.spinner("hourglass_top", size="sm").style(f"color:{_DIM}")
            msg = (
                "Queued: next up"
                if position <= 0
                else f"Queued: {position} build{'s' if position != 1 else ''} ahead"
            )
            if isinstance(eta_s, (int, float)) and eta_s > 0:
                msg += f" · est. ~{max(1, round(eta_s / 60))} min"
            ui.label(msg).classes("text-xs").style(f"color:{_DIM}")
        self._outcome.set_text(msg)
        self._outcome.style(f"color:{_DIM}")

    def set_status(
        self,
        ok: bool,
        cost=None,
        attempts=None,
        *,
        warning=False,
        failure_kind=None,
        retryable=False,
        work_units=None,
        reused_work_units=None,
    ) -> None:
        self._status_slot.clear()
        color = _WARN if retryable or warning else _OK if ok else _FAIL
        with self._status_slot:
            if ok:
                ui.icon("warning" if warning else "check_circle").style(
                    f"color:{color};font-size:1.1rem"
                )
                if warning:
                    ui.label("committed with findings").classes("text-xs").style(f"color:{_WARN}")
                if self._show_cost and isinstance(cost, (int, float)):
                    ui.label(f"${cost:.4f}").classes("text-xs font-mono").style(f"color:{_DIM}")
                if isinstance(work_units, int) and work_units > 0:
                    unit_text = f"{work_units} unit{'s' if work_units != 1 else ''}"
                    if isinstance(reused_work_units, int) and reused_work_units > 0:
                        unit_text += f" · {reused_work_units} reused"
                    ui.label(unit_text).classes("text-xs font-mono").style(f"color:{_DIM}")
            elif retryable and failure_kind == "provider_rate_limited":
                ui.icon("schedule").style(f"color:{_WARN};font-size:1.1rem")
                ui.label("provider busy — Retry").classes("text-xs").style(f"color:{_WARN}")
            else:
                ui.icon("cancel").style(f"color:{_FAIL};font-size:1.1rem")
                ui.label("failed").classes("text-xs").style(f"color:{_FAIL}")
        if ok:
            self._outcome.set_text(
                f"{self.label} committed with findings" if warning
                else f"{self.label} committed")
        elif retryable and failure_kind == "provider_rate_limited":
            self._outcome.set_text(f"{self.label}: provider busy — retry")
        else:
            self._outcome.set_text(f"{self.label} failed")
        self._outcome.style(f"color:{color}")
        self._settle_live(self._live_text(done=True, ok=ok, cost=cost, attempts=attempts),
                          color)
        # A call that never returned is not still running: say so instead of
        # leaving an eternal spinner behind.
        self._close_open_tool_cards()

    def _settle_live(self, text: str, color: str) -> None:
        """Finish the activity pane's live line.

        Repainting a stage's outcome (a corrected commit after a retry) must also
        settle the line the FIRST outcome wrote -- otherwise a stale "✗ failed"
        stays on screen behind a successful result."""
        self._live_done = True
        if self._live is not None:
            self._live.set_text(text)
            self._live.style(f"color:{color}")

    def set_parked(self) -> None:
        """Pill for a stage parked on a clarifying question: nothing is running,
        the user owes an answer (a live park or a reopened parked project)."""
        self._status_slot.clear()
        with self._status_slot:
            ui.icon("help").style("color:#fbbf24;font-size:1.1rem")
            ui.label("waiting for your answer").classes("text-xs").style("color:#fbbf24")
        self._outcome.set_text("Waiting for your answer")
        self._outcome.style("color:#fbbf24")
        self._settle_live("waiting for your answer", "#fbbf24")
        self._close_open_tool_cards()

    def set_pending(self) -> None:
        """Drop any result pill: the stage's outcome was invalidated (an upstream
        edit cleared its slot) and it has not run again yet."""
        self._status_slot.clear()
        self._outcome.set_text("")
        self._outcome.style(f"color:{_DIM}")
        # Nothing is streaming: an old stage_start from a lost attempt must not
        # leave a stage looking like it is running.
        self._settle_live("", _DIMMER)

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
                exp = (
                    ui.expansion("Thinking", icon="psychology", value=True)
                    .classes("w-full")
                    .props('dense expand-separator header-class="text-xs text-grey-5"')
                )
                with exp:
                    lab = (
                        ui.label("")
                        .classes("text-sm font-mono whitespace-pre-wrap leading-relaxed")
                        .style(f"color:{_DIM}")
                    )
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

    def _on_tool(self, event: dict) -> None:
        """One card per tool CALL: name, state, elapsed, cache marker, then its
        arguments and result expandable beneath it (paired by the client's local
        call id -- never by the provider's id, which can repeat across rounds)."""
        self._active_run = None  # a tool ends the current reasoning run
        self._tools += 1
        self._act_ready()
        name = str(event.get("name") or "tool")
        call_id = str(event.get("call_id") or "")
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        card: dict = {}
        with self._act:
            with (
                ui.column()
                .classes("w-full gap-1 px-2 py-1 rounded")
                .style("background:rgba(56,189,248,0.08);"
                       "border:1px solid rgba(56,189,248,0.28)")
            ):
                header = ui.row().classes("items-center gap-2 flex-nowrap min-w-0")
                with header:
                    card["icon"] = ui.icon("progress_activity") \
                        .style(f"color:{_DIMMER};font-size:1rem")
                    ui.label(name).classes(
                        "text-xs font-mono px-1.5 py-0.5 rounded shrink-0").style(
                        "background:rgba(56,189,248,0.14);color:#7dd3fc")
                    card["state_label"] = ui.label("Running").classes("text-xs shrink-0") \
                        .style(f"color:{_DIMMER}")
                    card["meta"] = ui.label("").classes("text-xs font-mono").style(
                        f"color:{_DIMMER}")
                if args:
                    with ui.expansion("Arguments", icon="data_object") \
                            .classes("w-full") \
                            .props('dense header-class="text-xs text-grey-5"'):
                        ui.label(json.dumps(args, indent=2, ensure_ascii=False)) \
                            .classes("text-xs font-mono whitespace-pre-wrap") \
                            .style(f"color:{_DIM}")
                card["result_slot"] = ui.column().classes("w-full gap-0")
        card["name"] = name
        card["state"] = "Running"
        card["started"] = time.monotonic()
        card["args"] = args
        if call_id:
            self._tool_cards[call_id] = card
        self._last_unpaired.append(card)

    def _on_tool_result(self, event: dict) -> None:
        """Attach a result to its call's card, or render a standalone named card
        when the call itself was never recorded (a result whose call predates the
        durable window) -- the evidence is never dropped."""
        self._active_run = None
        self._act_ready()
        name = str(event.get("name") or "")
        call_id = str(event.get("call_id") or "")
        card = self._tool_cards.pop(call_id, None) if call_id else None
        if card is None:
            card = next((c for c in self._last_unpaired
                         if c.get("name") == name and c.get("state") == "Running"), None) \
                if name else None
            if card is not None:
                self._last_unpaired.remove(card)
        if card is None:
            card = self._orphan_card(name)
        ok = event.get("ok")
        self._last_unpaired = [c for c in self._last_unpaired if c is not card]
        duration = event.get("duration_ms")
        cached = bool(event.get("cached"))
        if ok is False:
            state, color, icon = "Failed", _FAIL, "error"
        else:
            state, color, icon = "Returned", _OK, "check_circle"
        card["state"] = state
        card["state_label"].text = state
        card["icon"].name = icon
        card["icon"].style(f"color:{color};font-size:1rem")
        card["state_label"].style(f"color:{color}")
        bits = []
        if isinstance(duration, (int, float)):
            bits.append(f"{float(duration) / 1000:.2f}s")
        if cached:
            bits.append("cached")
        if ok is None:
            bits.append("unverified")
        card["meta"].text = "  ·  ".join(bits)
        output = str(event.get("output") or "")
        with card["result_slot"]:
            _render_tool_output(output, event)

    def _orphan_card(self, name: str) -> dict:
        """A named card for a result whose call was never recorded."""
        card: dict = {}
        with self._act:
            with (
                ui.column()
                .classes("w-full gap-1 px-2 py-1 rounded")
                .style("background:rgba(148,163,184,0.08);"
                       "border:1px solid rgba(148,163,184,0.28)")
            ):
                with ui.row().classes("items-center gap-2 flex-nowrap min-w-0"):
                    card["icon"] = ui.icon("help").style(
                        f"color:{_DIMMER};font-size:1rem")
                    ui.label(name or "tool").classes(
                        "text-xs font-mono px-1.5 py-0.5 rounded shrink-0").style(
                        "background:rgba(148,163,184,0.14);color:#cbd5e1")
                    card["state_label"] = ui.label("Call details unavailable").classes(
                        "text-xs shrink-0").style(f"color:{_DIMMER}")
                    card["meta"] = ui.label("").classes("text-xs font-mono").style(
                        f"color:{_DIMMER}")
                card["result_slot"] = ui.column().classes("w-full gap-0")
        return card

    def _close_open_tool_cards(self) -> None:
        """A call with no result at termination says so, instead of spinning."""
        for card in list(self._last_unpaired):
            if card.get("state") == "Running":
                card["state"] = "No result recorded"
                card["icon"].name = "help"
                card["icon"].style(f"color:{_DIMMER};font-size:1rem")
                card["state_label"].text = "No result recorded"
                card["state_label"].style(f"color:{_DIMMER}")
        self._last_unpaired = []

    def _on_retry(self, event: dict) -> None:
        """A retry is a human sentence plus the complete (bounded, expandable)
        error items -- never a clipped JSON blob."""
        self._active_run = None
        self._act_ready()
        stage = _activity.stage_label(event.get("stage")) or self.label
        kind = str(event.get("failure_kind") or "")
        summary = f"Retrying {stage}" + (f" after {kind.replace('_', ' ')}" if kind else "")
        items = _error_items(event)
        with self._act:
            with (
                ui.column()
                .classes("w-full gap-1 px-2 py-1 rounded")
                .style("background:rgba(251,191,36,0.10)")
            ):
                with ui.row().classes("items-center gap-2 flex-nowrap min-w-0"):
                    ui.icon("warning").style("color:#fbbf24;font-size:1rem")
                    ui.label(summary).classes("text-xs min-w-0").style("color:#fcd34d")
                for item in items[:20]:
                    _render_error_item(item)
                if len(items) > 20:
                    ui.label(f"…and {len(items) - 20} more").classes("text-xs") \
                        .style(f"color:{_DIMMER}")

    def _on_run_error(self, event: dict) -> None:
        """The terminal technical record of a failed run, on its stage."""
        self._active_run = None
        self._act_ready()
        kind = str(event.get("failure_kind") or "run_error")
        with self._act:
            with (
                ui.column()
                .classes("w-full gap-1 px-2 py-1 rounded")
                .style("background:rgba(248,113,113,0.10);"
                       "border:1px solid rgba(248,113,113,0.35)")
            ):
                with ui.row().classes("items-center gap-2 flex-nowrap min-w-0"):
                    ui.icon("cancel").style(f"color:{_FAIL};font-size:1rem")
                    ui.label(kind.replace("_", " ")).classes("text-xs font-bold") \
                        .style(f"color:{_FAIL}")
                    if event.get("exception_type"):
                        ui.label(str(event["exception_type"])).classes(
                            "text-xs font-mono").style(f"color:{_DIMMER}")
                if event.get("message"):
                    ui.label(str(event["message"])).classes(
                        "text-xs whitespace-pre-wrap").style(f"color:{_DIM}")
                if event.get("retryable"):
                    ui.label("This failure is retryable.").classes("text-xs") \
                        .style(f"color:{_DIM}")

    def _on_activity_note(self, event: dict) -> None:
        """A short, plain-language line for a lifecycle event that has no richer
        card of its own (provider fallback, escalation, format recovery, decode)."""
        kind = str(event.get("kind") or "")
        if kind == "provider_fallback":
            summary = (f"Falling back to {event.get('to') or 'another provider'}"
                       + (f" (from {event['from']})" if event.get("from") else ""))
        elif kind == "escalation":
            summary = (f"Escalating to {event.get('to') or 'a stronger model'}"
                       + (f" after {event['reason']}" if event.get("reason") else ""))
        elif kind == "serialization_recovery":
            summary = ("Recovered the model's response format"
                       + (f" ({event['failure_kind']})"
                          if event.get("failure_kind") else ""))
        else:
            summary = ("Candidate decoded"
                       + (f" (attempt {event['attempt']})"
                          if event.get("attempt") else ""))
        self._active_run = None
        self._act_ready()
        with self._act:
            with ui.row().classes("items-center gap-2 flex-nowrap min-w-0 pt-0.5"):
                ui.icon("bolt").style(f"color:{_DIMMER};font-size:1rem")
                ui.label(summary).classes("text-xs min-w-0 whitespace-normal") \
                    .style(f"color:{_DIM}")

    def set_interrupted(self) -> None:
        """Pill for a run that was lost (a restart mid-run), which is neither a
        failure of the design nor a live run."""
        self._status_slot.clear()
        with self._status_slot:
            ui.icon("link_off").style(f"color:{_DIM};font-size:1.1rem")
            ui.label("interrupted").classes("text-xs").style(f"color:{_DIM}")
        self._outcome.set_text("Run was interrupted")
        self._outcome.style(f"color:{_DIM}")
        self._settle_live("✗ interrupted", _DIM)
        self._close_open_tool_cards()

    def _on_stage_diagnostic(self, diagnostic: dict) -> None:
        self._active_run = None
        self._act_ready()
        code = str(diagnostic.get("code") or "semantic_finding")
        evidence = ", ".join(str(item) for item in diagnostic.get("evidence") or [])
        message = str(diagnostic.get("message") or "")
        with self._act:
            with (
                ui.column()
                .classes("w-full gap-0 px-2 py-1 rounded")
                .style("background:rgba(234,179,8,0.10);border:1px solid rgba(234,179,8,0.35)")
            ):
                ui.label(code).classes("text-xs font-mono font-bold").style(f"color:{_WARN}")
                ui.label(message).classes("text-xs").style(f"color:{_DIM}")
                if evidence:
                    ui.label(evidence).classes("text-xs font-mono").style(f"color:{_DIMMER}")

    def _provenance_card(
        self,
        source: object,
        title: str,
        detail: str,
        *,
        outcome: str | None = None,
        failed: bool = False,
    ) -> None:
        self._active_run = None
        self._act_ready()
        label, icon, color = _provenance_source(source)
        border = _FAIL if failed else color
        with self._act:
            with (
                ui.column()
                .classes("w-full gap-1 px-2 py-1.5 rounded")
                .style(
                    f"background:{border}12;border:1px solid {border}55"
                )
            ):
                with ui.row().classes("w-full items-center gap-2 flex-nowrap"):
                    ui.icon(icon).style(f"color:{color};font-size:1rem")
                    ui.label(label).classes(
                        "text-xs font-bold uppercase tracking-wide shrink-0"
                    ).style(f"color:{color}")
                    ui.label(title).classes(
                        "text-xs font-mono truncate min-w-0"
                    ).style(f"color:{_DIM}")
                    if outcome:
                        ui.label(outcome).classes(
                            "text-xs font-mono ml-auto shrink-0"
                        ).style(f"color:{_FAIL if failed else _OK}")
                if detail:
                    ui.label(detail).classes(
                        "text-xs font-mono whitespace-normal"
                    ).style(f"color:{_DIMMER}")

    def _on_recipe_selected(self, event: dict) -> None:
        sheets = ", ".join(
            f"{role}→{sheet}" for role, sheet in (event.get("sheets") or {}).items()
        )
        parameters = ", ".join(
            f"{name}={value}" for name, value in (event.get("parameters") or {}).items()
        )
        detail = " · ".join(value for value in (sheets, parameters) if value)
        self._provenance_card(
            "recipe",
            f"{event.get('recipe') or 'unknown'} · {event.get('instance') or 'instance'}",
            detail,
        )

    def _on_work_unit_plan(self, event: dict) -> None:
        refs = event.get("refs") or []
        pin_count = int(event.get("expected_pin_count") or 0)
        facts = []
        if refs:
            facts.append(f"{len(refs)} refs: {', '.join(str(ref) for ref in refs)}")
        if pin_count:
            facts.append(f"{pin_count} unresolved pins")
        self._provenance_card(
            event.get("source"),
            f"{event.get('unit_id') or 'unit'} · {event.get('unit_sheet') or 'unknown sheet'}",
            " · ".join(facts) or "sheet-local generation unit",
            outcome="planned",
        )

    def _on_work_unit_attempt(self, event: dict) -> None:
        outcome = str(event.get("outcome") or "unknown")
        failed = outcome not in {"candidate", "ok"}
        provider = " / ".join(
            str(value) for value in (event.get("provider"), event.get("model")) if value
        )
        metrics = []
        if event.get("wall_s") is not None:
            metrics.append(f"{float(event['wall_s']):.1f}s")
        if event.get("input_tokens") is not None or event.get("output_tokens") is not None:
            metrics.append(
                f"{int(event.get('input_tokens') or 0):,} in / "
                f"{int(event.get('output_tokens') or 0):,} out"
            )
        if self._show_cost and event.get("cost_usd") is not None:
            metrics.append(f"${float(event['cost_usd']):.4f}")
        if event.get("error_code"):
            metrics.append(f"error {event['error_code']}")
        if event.get("request_id"):
            metrics.append(f"request {event['request_id']}")
        self._provenance_card(
            "llm",
            (
                f"{event.get('unit_id') or 'unit'} · attempt "
                f"{int(event.get('unit_attempt') or 1)}"
            ),
            " · ".join(value for value in (provider, *metrics) if value),
            outcome=outcome,
            failed=failed,
        )

    def _on_work_unit_done(self, event: dict) -> None:
        self._provenance_card(
            event.get("source"),
            f"{event.get('unit_id') or 'unit'} · {event.get('unit_sheet') or 'unknown sheet'}",
            "Candidate validated and retained for aggregate commit.",
            outcome="accepted",
        )

    def _on_build_log(self, text: str) -> None:
        self._act_ready()
        if self._build_log is None:
            with self._act:
                exp = (
                    ui.expansion("Build log", icon="terminal", value=True)
                    .classes("w-full")
                    .props('dense expand-separator header-class="text-xs text-grey-5"')
                )
                with exp:
                    lab = (
                        ui.label("")
                        .classes("text-xs font-mono whitespace-pre-wrap leading-relaxed")
                        .style(f"color:{_DIM}")
                    )
            self._build_log = _Run(exp, lab, head="Build log", mode="lines")
        self._build_log.buf += text + "\n"
        self._dirty.add(self._build_log)

    # ---- issues -------------------------------------------------------------
    def set_issues(self, items: list[dict], *, on_view=None) -> None:
        """Render (or clear) this stage's own outstanding issues.

        Rebuilt only when the issue set actually changes, so a per-second poll
        does not churn the DOM."""
        items = list(items or [])
        sig = tuple((i.get("severity"), i.get("code"), i.get("message"),
                     tuple(i.get("evidence") or [])) for i in items)
        if sig == self._issues_sig:
            return
        self._issues_sig = sig
        self.issues_slot.clear()
        self.issues_slot.set_visibility(bool(items))
        if items:
            with self.issues_slot:
                _render_issues(issues_section(items, title="Needs attention",
                                              on_view=on_view))

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
            self._draft_structured = False
            self._insp.clear()  # run's draft may own the pane again
            with self._insp:
                ui.label("No data committed for this stage yet.").classes("text-xs italic").style(
                    f"color:{_DIMMER}"
                )
            return
        self._committed = True
        self._draft_buf = ""
        self._draft_dirty = False
        self._draft_structured = False
        self._insp.clear()
        with self._insp:
            for sec in sections:
                _render_section(sec, self.accent)

    def _render_draft(self) -> None:
        """Show the slot JSON the model is currently writing as a live, uncommitted
        preview in the Project-state window. When the partial buffer parses (whole,
        or after closing its unbalanced brackets/strings) and the page supplied a
        draft-spec builder, render it through the same structured sections as the
        committed view — so e.g. the BOM table fills in row by row while streaming.
        Otherwise fall back to pretty-printed (or raw) text. Replaced by the
        validated view once the stage commits."""
        obj = _parse_draft(self._draft_buf)
        secs: list[dict] = []
        if obj is not None and self._draft_spec is not None:
            try:
                secs = self._draft_spec(self.key, obj) or []
            except Exception:
                secs = []  # partial data the builder can't shape yet: text fallback
        if secs:
            self._draft_structured = True
        elif self._draft_structured:
            return
        self._insp.clear()
        with self._insp:
            ui.label(f"● writing {self.key} slot…  ·  {len(self._draft_buf):,} chars").classes(
                "text-xs font-semibold"
            ).style(f"color:{self.accent}")
            if secs:
                for sec in secs:
                    _render_section(sec, self.accent)
            else:
                body = json.dumps(obj, indent=2) if obj is not None else self._draft_buf
                ui.label(body).classes("text-xs font-mono whitespace-pre-wrap").style(
                    f"color:{_DIM}"
                )


def _render_section(sec: dict, accent: str) -> None:
    title = sec.get("title", "")
    if title:
        ui.label(title).classes("text-xs font-semibold uppercase tracking-wide").style(
            f"color:{accent}"
        )
    kind = sec.get("type")
    if kind == "pcb_error":
        with (
            ui.element("div")
            .classes("w-full rounded p-3")
            .style("border:2px solid #b91c1c;background:#450a0a")
        ):
            ui.label(str(sec.get("title") or "PCB failure")).classes("text-base font-bold").style(
                "color:#fecaca"
            )
            ui.label(str(sec.get("explanation") or "")).classes(
                "text-sm leading-snug whitespace-pre-wrap mt-1"
            ).style("color:#fca5a5")
            details = sec.get("details") or []
            if details:
                with ui.column().classes("w-full gap-0.5 mt-2"):
                    for detail in details:
                        ui.label(f"• {detail}").classes("text-xs whitespace-pre-wrap").style(
                            "color:#fecaca"
                        )
            rows = []
            if sec.get("counts"):
                rows.append(("counts", ", ".join(f"{k}={v}" for k, v in sec["counts"])))
            if sec.get("nets"):
                rows.append(("nets", ", ".join(sec["nets"])))
            if sec.get("footprint_refs"):
                rows.append(("footprints", ", ".join(sec["footprint_refs"])))
            if rows:
                with ui.column().classes("w-full gap-0.5 mt-2"):
                    for key, value in rows:
                        with ui.row().classes("w-full no-wrap gap-2 items-start"):
                            ui.label(key).classes("text-xs font-mono shrink-0").style(
                                "color:#fca5a5;min-width:6rem"
                            )
                            ui.label(value).classes(
                                "text-xs font-mono whitespace-pre-wrap min-w-0"
                            ).style("color:#fecaca")
            violations = sec.get("violations") or []
            if violations:
                ui.html(
                    _table_html(
                        ["type", "location", "nets", "footprints", "description"],
                        violations,
                    ),
                    sanitize=False,
                ).classes("w-full mt-2")
            if sec.get("next_action"):
                ui.label(f"Next action: {sec['next_action']}").classes(
                    "text-xs italic mt-2 whitespace-pre-wrap"
                ).style("color:#fcd34d")
    elif kind == "kv":
        with ui.column().classes("w-full gap-0.5"):
            for k, v in sec.get("rows", []):
                with ui.row().classes("w-full no-wrap gap-2 items-start"):
                    ui.label(str(k)).classes("text-xs font-mono shrink-0").style(
                        f"color:{_DIMMER};min-width:9rem"
                    )
                    ui.label(str(v)).classes("text-xs font-mono whitespace-pre-wrap min-w-0").style(
                        f"color:{_DIM}"
                    )
    elif kind == "list":
        items = sec.get("items", [])
        if not items:
            ui.label("(none)").classes("text-xs italic").style(f"color:{_DIMMER}")
        with ui.column().classes("w-full gap-0.5"):
            for it in items:
                with ui.row().classes("w-full no-wrap gap-1 items-start"):
                    ui.label("•").style(f"color:{_DIMMER}")
                    ui.label(str(it)).classes(
                        "text-xs font-mono whitespace-pre-wrap min-w-0"
                    ).style(f"color:{_DIM}")
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
    elif kind == "findings":
        items = sec.get("items", [])
        if not items:
            ui.label("No findings — design passed review.").classes("text-xs italic").style(
                f"color:{_DIMMER}"
            )
        _SEV_COLOR = {"blocker": _FAIL, "warning": _WARN, "note": _DIMMER}
        _SEV_BG = {
            "blocker": "rgba(248,113,113,0.12)",
            "warning": "rgba(234,179,8,0.10)",
            "note": "rgba(100,116,139,0.08)",
        }
        _SEV_BORDER = {
            "blocker": "rgba(248,113,113,0.45)",
            "warning": "rgba(234,179,8,0.40)",
            "note": "rgba(100,116,139,0.30)",
        }
        with ui.column().classes("w-full gap-2"):
            for f in items:
                sev = f.get("severity", "note")
                area = f.get("area", "")
                issue = f.get("issue", "")
                suggestion = f.get("suggestion", "")
                color = _SEV_COLOR.get(sev, _DIMMER)
                bg = _SEV_BG.get(sev, "rgba(100,116,139,0.08)")
                border = _SEV_BORDER.get(sev, "rgba(100,116,139,0.30)")
                with (
                    ui.element("div")
                    .classes("w-full rounded p-2")
                    .style(f"border-left:3px solid {border};background:{bg}")
                ):
                    with ui.row().classes("w-full no-wrap gap-2 items-center mb-1"):
                        ui.label(sev.upper()).classes(
                            "text-xs font-bold px-1.5 py-0.5 rounded"
                        ).style(f"color:{color};background:rgba(0,0,0,0.2)")
                        if area:
                            ui.label(f"[{area}]").classes("text-xs font-mono").style(
                                f"color:{_DIMMER}"
                            )
                    ui.label(issue).classes("text-sm leading-snug whitespace-pre-wrap").style(
                        f"color:{_DIM}"
                    )
                    if suggestion:
                        ui.label(f"→ {suggestion}").classes("text-xs italic mt-1").style(
                            f"color:{_DIMMER}"
                        )

    elif kind == "issues":
        _render_issues(sec)

    elif kind == "progress":
        pct = max(0, min(100, int(float(sec.get("percent", 0)))))
        phase = sec.get("phase", "")
        action = sec.get("action", "")
        elapsed = sec.get("elapsed", "")
        eta = sec.get("eta", "")
        bar_color = sec.get("bar_color", accent)
        with ui.column().classes("w-full gap-1.5"):
            # Phase + action label
            if phase:
                with ui.row().classes("w-full no-wrap gap-2 items-center"):
                    ui.label(phase).classes("text-sm font-semibold").style(f"color:{bar_color}")
                    if action:
                        ui.label(action).classes("text-xs").style(f"color:{_DIMMER}")
            # Progress bar (pure CSS, no JS)
            with (
                ui.element("div")
                .classes("w-full")
                .style("height:8px;border-radius:4px;background:var(--kc-border);overflow:hidden")
            ):
                with ui.element("div").style(
                    f"height:100%;width:{pct}%;border-radius:4px;"
                    f"background:{bar_color};transition:width 0.5s"
                ):
                    pass
            ui.label(f"{pct}%").classes("text-xs font-mono").style(f"color:{_DIMMER}")
            # Elapsed / ETA line
            if elapsed or eta:
                parts = []
                if elapsed:
                    parts.append(f"Elapsed: {elapsed}")
                if eta:
                    parts.append(f"ETA: {eta}")
                ui.label(" · ".join(parts)).classes("text-xs").style(f"color:{_DIMMER}")
        # Leaf status chips (if present)
        items = sec.get("items") or []
        if items:
            with ui.row().classes("w-full flex-wrap gap-1.5 mt-1"):
                for it in items:
                    label = it.get("label", "?")
                    done = it.get("done", False)
                    active = it.get("active", False)
                    if done:
                        chip_color = _OK
                        bg = "rgba(52,211,153,0.15)"
                    elif active:
                        chip_color = bar_color
                        bg = "rgba(96,165,250,0.15)"
                    else:
                        chip_color = _DIMMER
                        bg = "rgba(100,116,139,0.08)"
                    ui.label(label).classes("text-xs px-1.5 py-0.5 rounded").style(
                        f"color:{chip_color};background:{bg}"
                    )


_SEV_STYLE = {
    "error": (_FAIL, "rgba(248,113,113,0.12)", "rgba(248,113,113,0.45)", "cancel"),
    "warning": (_WARN, "rgba(234,179,8,0.10)", "rgba(234,179,8,0.40)", "warning"),
    "info": (_DIMMER, "rgba(100,116,139,0.08)", "rgba(100,116,139,0.30)", "info"),
}


def issues_section(items: list[dict], *, title: str = "Issues",
                   on_view=None) -> dict:
    """Spec for one bounded, deduplicated issue list.

    `on_view(stage)` is what the "View stage" control calls (the page selects
    that stage's tab); pass None where there is nowhere to jump to.
    """
    return {"type": "issues", "title": title, "items": list(items or []),
            "on_view": on_view}


def _render_issues(sec: dict) -> None:
    """Severity, stage, readable message, code, evidence, and a jump to the stage."""
    items = sec.get("items") or []
    if not items:
        ui.label("No current issues.").classes("text-xs italic").style(f"color:{_DIMMER}")
        return
    on_view = sec.get("on_view")
    with ui.column().classes("w-full gap-2"):
        for issue in items:
            sev = str(issue.get("severity") or "warning")
            color, bg, border, icon = _SEV_STYLE.get(sev, _SEV_STYLE["warning"])
            with (
                ui.element("div")
                .classes("w-full rounded p-2")
                .style(f"border-left:3px solid {border};background:{bg}")
            ):
                stage = issue.get("stage")
                with ui.row().classes("w-full no-wrap gap-2 items-center"):
                    ui.icon(icon).style(f"color:{color};font-size:1rem")
                    ui.label(sev.upper()).classes(
                        "text-xs font-bold px-1.5 py-0.5 rounded"
                    ).style(f"color:{color};background:rgba(0,0,0,0.2)")
                    if stage:
                        ui.label(_activity.stage_label(stage)).classes(
                            "text-xs font-mono").style(f"color:{_DIMMER}")
                    if issue.get("code"):
                        ui.label(str(issue["code"])).classes("text-xs font-mono") \
                            .style(f"color:{_DIMMER}")
                    ui.space()
                    if stage and on_view is not None:
                        ui.button("View stage", icon="arrow_forward",
                                  on_click=lambda s=stage: on_view(s)) \
                            .props("flat dense no-caps").classes("text-xs")
                if issue.get("message"):
                    ui.label(str(issue["message"])).classes(
                        "text-xs whitespace-pre-wrap break-all").style(f"color:{_DIM}")
                for item in issue.get("evidence") or []:
                    text = str(item)
                    if len(text) <= 200:
                        ui.label(text).classes(
                            "text-xs font-mono whitespace-pre-wrap break-all") \
                            .style(f"color:{_DIMMER}")
                    else:
                        with ui.expansion(f"Evidence · {len(text):,} chars",
                                          icon="notes").classes("w-full") \
                                .props('dense header-class="text-xs text-grey-5"'):
                            ui.label(text).classes(
                                "text-xs font-mono whitespace-pre-wrap break-all") \
                                .style(f"color:{_DIMMER}")


def _cell_html(cell) -> str:
    """One <td> body. A cell is either a scalar (rendered as escaped text) or a
    ``{"text", "href"}`` dict (rendered as a new-tab link, e.g. a vendor lookup)."""
    if isinstance(cell, dict):
        text = escape(str(cell.get("text", "")))
        href = str(cell.get("href") or "")
        # Only emit real web links; anything else (javascript:, data:, ...) falls
        # back to plain text so a cell can never inject a clickable script URL.
        if href[:7].lower() == "http://" or href[:8].lower() == "https://":
            return f'<a href="{escape(href)}" target="_blank" rel="noopener noreferrer">{text}</a>'
        return text
    return escape("" if cell is None else str(cell))


def _table_html(cols: list, rows: list, foot: list | None = None) -> str:
    """A real <table> for an inspector section so columns line up across every row
    (the prior per-row flex layout sized each row independently, so they didn't).
    Optional ``foot`` rows render in a <tfoot> (e.g. the BOM total). All dynamic
    text is HTML-escaped; only hrefs the page supplies -- vendor URLs we build
    ourselves in web._vendor_cell -- become links."""

    def trow(r):
        warn = any(isinstance(c, dict) and c.get("warn") for c in r)
        style = ' style="background:rgba(234,179,8,0.12)"' if warn else ""
        return f"<tr{style}>" + "".join(f"<td>{_cell_html(c)}</td>" for c in r) + "</tr>"

    head = "".join(f"<th>{escape(str(c))}</th>" for c in cols)
    body = "".join(trow(r) for r in rows)
    tfoot = f"<tfoot>{''.join(trow(r) for r in foot)}</tfoot>" if foot else ""
    return (
        f'<table class="kc-table"><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody>{tfoot}</table>"
    )


def _parse_draft(buf: str):
    """Best-effort parse of a partial slot-JSON draft (the model's streaming
    content). Returns the parsed object when the buffer parses, whole or after
    closing its unbalanced brackets/strings; else None so the caller shows the
    raw text. Never raises."""
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
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            pass
        repaired = _close_json(s)
        if repaired is None:
            return None
        return json.loads(repaired)
    except Exception:
        return None


def _loose_pretty(buf: str) -> str | None:
    """Indented-JSON text of a partial draft (see _parse_draft), or None."""
    obj = _parse_draft(buf)
    return None if obj is None else json.dumps(obj, indent=2)


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

    def __init__(self, show_cost: bool = False, draft_spec=None) -> None:
        _follow_head()
        self.show_cost = show_cost
        self.panels: dict[str, StagePanel] = {}
        self._tab_el: dict[str, ui.tab] = {}
        self._tab_tip: dict[str, object] = {}
        self._current: str | None = None
        self._auto_follow = True
        # While replaying persisted history the selection must not chase events;
        # `end_replay` then selects the authoritative stage exactly once.
        self._replaying = False
        self._pending_select: str | None = None
        self._select_attempts = 0
        # Which state a build driven OUTSIDE this page last painted (so the per
        # second observer does not churn the DOM).
        self._foreign_state: str | None = None
        # True while the open design is genuinely running (so the follow control
        # is only offered when there is something to follow).
        self._something_live = False
        self._on_show: dict[str, object] = {}

        # Explicit escape hatch back to live following, shown only while the user
        # is looking at a stage other than the live one.
        with ui.row().classes("w-full items-center justify-end gap-2") as _follow_row:
            self._follow_btn = ui.button(
                "Follow live stage", icon="my_location",
                on_click=lambda: self.follow_live()) \
                .props("flat dense no-caps").classes("text-xs") \
                .tooltip("Jump back to the stage that is running and follow it")
            self._follow_btn.set_visibility(False)

        with (
            ui.tabs()
            .classes("w-full kc-stage-tabs")
            .props("dense inline-label mobile-arrows") as self.tabs
        ):
            for key, label, icon, accent in PHASES:
                t = ui.tab(key, label=label, icon=icon)
                t.style(f"color:{_STATUS_COLOR['pending']}")
                self._tab_el[key] = t
                # ONE tooltip per tab, updated in place: Element.tooltip() appends a
                # new QTooltip on every call, so repeated status changes used to
                # leave every earlier status' tooltip attached to the same tab.
                with t:
                    self._tab_tip[key] = ui.tooltip(f"{label}: pending")
                self._set_tab_status(key, "pending")
        self.tabs.on_value_change(self._on_tab_change)

        with (
            ui.tab_panels(self.tabs, value=PHASES[0][0])
            .classes("w-full")
            .style("background:transparent") as self._tab_panels
        ):
            for key, label, icon, accent in PHASES:
                with ui.tab_panel(key).classes("p-0"):
                    self.panels[key] = StagePanel(key, label, icon, accent, show_cost, draft_spec)

    # ---- selection ----------------------------------------------------------
    def _show(self, key: str) -> None:
        """Select a tab AND the panel it drives.

        The tab row is bound to the panels element, so the PANELS are what the
        client actually renders: setting only the tabs value is silently ignored
        (measured live -- a 'View stage' click left the old tab on screen)."""
        if key not in self.panels:
            return
        self.tabs.set_value(key)
        if self._tab_panels.value != key:
            self._tab_panels.set_value(key)

    def _shown(self) -> str | None:
        """The tab the client is showing (the tabs element is the value the client
        reports back; the panels element is what renders)."""
        return self.tabs.value or self._tab_panels.value

    def set_live(self, live: bool) -> None:
        """Whether anything is actually running right now.

        The follow control is only meaningful while something is live, so a
        finished project does not offer to 'follow' a run that has ended."""
        if self._something_live != live:
            self._something_live = live
            self._set_follow_ui()

    # ---- follow / replay ----------------------------------------------------
    def _set_follow_ui(self) -> None:
        off_live = bool(self._current) and self._shown() != self._current
        self._follow_btn.set_visibility(
            off_live and self._something_live and not self._auto_follow
            and not self._replaying)

    def begin_replay(self) -> None:
        """Filling historical panels: suspend auto-follow so the tab row is not
        yanked around by events that are already over."""
        self._replaying = True
        self._auto_follow = False
        self._set_follow_ui()

    def end_replay(self, select: str | None = None) -> None:
        """Leave replay, selecting the authoritative stage exactly once.

        The selection is RE-ASSERTED for a few render ticks (`settle`): the
        browser echoes the tab panels' initial value back, and that echo must not
        win over the stage the user needs to see."""
        self._replaying = False
        if select:
            self._pending_select = select
            self._select_attempts = 0
            self._assert_selection()
        else:
            self._auto_follow = True
            self._set_follow_ui()

    def _assert_selection(self) -> None:
        if self._pending_select and self._shown() != self._pending_select:
            self._show(self._pending_select)
        self._set_follow_ui()

    def settle(self) -> None:
        """One render tick's worth of follow bookkeeping (cheap, no-op when done).

        The pending selection is re-asserted for a short window rather than until
        it first appears applied: the browser's own initial value for the tab
        panels arrives a moment AFTER the page is built and would otherwise
        silently restore the first tab (measured live)."""
        if not self._pending_select:
            return
        self._select_attempts += 1
        if self._select_attempts <= 8:
            self._show(self._pending_select)
            return
        self._auto_follow = self._pending_select == self._current
        self._pending_select = None
        self._set_follow_ui()

    def select(self, key: str | None) -> None:
        """Show `key`'s tab; auto-follow resumes only if it IS the live stage."""
        if key and key in self.panels:
            self._show(key)
            self._auto_follow = key == self._current
        self._set_follow_ui()

    def follow_live(self) -> None:
        """Resume auto-follow and jump back to the live stage."""
        self._auto_follow = True
        if self._current:
            self._show(self._current)
        self._set_follow_ui()

    # ---- tab status / follow ------------------------------------------------
    def _set_tab_status(self, key: str, status: str) -> None:
        t = self._tab_el.get(key)
        if t is None:
            return
        t.style(f"color:{_STATUS_COLOR.get(status, _DIMMER)}")
        # Status is not colour-only: the tab carries an icon and an accessible
        # name containing the phase AND its status (for screen readers and for the
        # phone layout, which hides the tab labels).
        label = next((lbl for k, lbl, _i, _a in PHASES if k == key), key)
        word = status.replace("_", " ")
        icon = _STATUS_ICON.get(status)
        if icon and t._props.get("icon") != icon:
            t._props["icon"] = icon
            t.update()
        aria = f"{label}: {word}"
        if t._props.get("aria-label") != aria:
            t._props["aria-label"] = aria
            t.update()
        tip = self._tab_tip.get(key)
        if tip is not None:
            tip.text = f"{label} — {word}"

    def _on_tab_change(self, e) -> None:
        val = getattr(e, "value", None)
        # Resume auto-follow only while the user is parked on the live stage. With
        # no live stage yet there is nothing to follow: the client's initial panel
        # value (and any other echo) must not switch following off before the run
        # has even announced a stage.
        if self._current is not None and not self._replaying:
            self._auto_follow = val == self._current
        self._set_follow_ui()
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
        if self._auto_follow and not self._replaying:
            self._show(key)
        self._set_follow_ui()

    # ---- event routing ------------------------------------------------------
    def push(self, e: dict) -> None:
        k = e.get("kind")
        if k == "stage_start":
            self._set_current(e.get("stage"), e.get("model"))
        elif k == "stage_done":
            self._finish(
                e.get("stage") or self._current,
                bool(e.get("ok")),
                e.get("cost"),
                e.get("attempts"),
                warning=bool(e.get("warning")),
                failure_kind=e.get("failure_kind"),
                retryable=bool(e.get("retryable")),
                work_units=e.get("work_units"),
                reused_work_units=e.get("reused_work_units"),
            )
        elif k == "question":
            # The stage parked on a clarifying question: stop the spinner, say
            # the run is waiting on the user (it is not failed, not running).
            stg = e.get("stage") or self._current
            p = self.panels.get(stg)
            if p is not None:
                p.end_runs()
                p.set_parked()
                self._set_tab_status(stg, "parked")
        elif k in _activity.PROVENANCE_EVENT_KINDS:
            # Crash-journal events can be replayed without their in-memory
            # stage_start. Route by the event's durable stage identity instead
            # of whichever historical tab happened to finish last.
            stg = e.get("stage") or self._current
            panel = self.panels.get(stg)
            if panel is not None:
                panel.push(e)
        elif k == "build_start":
            self._set_current("synthesize")
        elif k == "queue":
            # The whole deterministic build is parked in the host build queue;
            # surface position/ETA on the tab build_start just activated.
            self._set_current("synthesize")
            self._set_tab_status("synthesize", "queued")
            self.panels["synthesize"].set_queued(
                int(e.get("position") or 0), int(e.get("depth") or 0), e.get("eta_s")
            )
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
        elif k == "run_finished":
            self._on_run_finished(e)
        elif k == "run_error":
            # The terminal technical record of an unexpected exception: show it on
            # its stage AND settle that stage's status, so a failure is never left
            # looking like a run that is still streaming.
            stg = e.get("stage") or self._current
            panel = self.panels.get(stg) if stg else None
            if panel is not None:
                panel.push(e)
            self._finish(stg, False, None, failure_kind=e.get("failure_kind"),
                         retryable=bool(e.get("retryable")))
        else:
            # Deltas / tool cards / retries / diagnostics: route by the event's OWN
            # stage (stamped when it was recorded from the most recent announced
            # stage), else the current one. An unattributed event is DROPPED rather
            # than blamed on whichever stage happened to be selected.
            stg = e.get("stage") or self._current
            panel = self.panels.get(stg) if stg else None
            if panel is not None:
                panel.push(e)

    def _on_run_finished(self, event: dict) -> None:
        """The attempt's terminal event: stop whatever was running and paint the
        attempt's outcome on the stage it reached."""
        status = str(event.get("status") or "")
        stage = event.get("stage") or self._current
        panel = self.panels.get(stage) if stage else None
        if panel is None:
            return
        if status == "ok":
            panel.end_runs()
            if stage in _BUILD_STAGES:
                self._finish(stage, True, None)
            return
        if status == "awaiting_input":
            panel.end_runs()
            panel.set_parked()
            self._set_tab_status(stage, "parked")
            return
        if status == "interrupted":
            panel.end_runs()
            panel.set_interrupted()
            self._set_tab_status(stage, "interrupted")
            return
        self._finish(
            stage, False, None,
            failure_kind=event.get("failure_kind"),
            retryable=bool(event.get("retryable")))

    def _finish(
        self,
        key: str | None,
        ok: bool,
        cost,
        attempts=None,
        *,
        warning=False,
        failure_kind=None,
        retryable=False,
        work_units=None,
        reused_work_units=None,
    ) -> None:
        if key is None or key not in self.panels:
            return
        self.panels[key].end_runs()
        self.panels[key].set_status(
            ok,
            cost,
            attempts,
            warning=warning,
            failure_kind=failure_kind,
            retryable=retryable,
            work_units=work_units,
            reused_work_units=reused_work_units,
        )
        self._set_tab_status(
            key,
            "warning" if retryable or ok and warning else "done" if ok else "failed",
        )

    def flush(self) -> None:
        for p in self.panels.values():
            p.flush()

    # ---- data + reset (driven by the page) ----------------------------------
    def set_inspector(self, key: str, sections: list[dict]) -> None:
        p = self.panels.get(key)
        if p is not None:
            p.set_inspector(sections)

    def set_issues(self, key: str, items: list[dict], *, on_view=None) -> None:
        """Paint one stage's outstanding issues into its own panel."""
        p = self.panels.get(key)
        if p is not None:
            p.set_issues(items, on_view=on_view)

    def mark_parked(self, key: str | None) -> None:
        """Show `key` as the stage that is waiting on the user.

        A durable parked run may already have COMMITTED the stage it raised the
        question in (the artifact-derived status then says 'done'), so the parked
        presentation is layered on top: the distinct help icon and the violet
        accent, never a green 'done' that hides an unanswered question."""
        p = self.panels.get(key) if key else None
        if p is None:
            return
        p.end_runs()
        p.set_parked()
        self._set_tab_status(key, "parked")

    def mark_queued(self, note: str = "") -> None:
        """Reflect a build waiting in the host queue that THIS page does not drive.

        Display only: nothing is enqueued, registered, or persisted."""
        if self._foreign_state == "queued":
            if note:
                panel = self.panels.get("synthesize")
                if panel is not None:
                    panel.set_outcome(note, _STATUS_COLOR["queued"])
            return
        self._foreign_state = "queued"
        panel = self.panels.get("synthesize")
        if panel is not None:
            panel.set_queued_note(note)
        self._set_tab_status("synthesize", "queued")

    def mark_building(self, stage: str, note: str = "") -> None:
        """A build another process is driving has reached `stage` (observed from
        its log tail). Display only -- no live driver is registered."""
        if not stage or stage not in self.panels:
            return
        key = f"active:{stage}"
        if self._foreign_state == key:
            if note:
                self.panels[stage].set_outcome(note, _STATUS_COLOR["active"])
            return
        self._foreign_state = key
        self.panels[stage].set_outcome(
            note or f"▶ {_activity.stage_label(stage)} running", _STATUS_COLOR["active"])
        self._set_tab_status(stage, "active")

    def view_slot(self, key: str):
        """The empty column at the top of a panel's inspector for KiCanvas/download."""
        return self.panels[key].view_slot

    def on_show(self, key: str, fn) -> None:
        """Run `fn` when `key`'s tab becomes the visible one (see _on_tab_change)."""
        self._on_show[key] = fn

    def active(self) -> str | None:
        """The currently selected tab key."""
        return self._shown()

    def set_statuses(self, statuses: dict[str, str], stage_status: dict | None = None) -> None:
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
                p.set_status(
                    True,
                    cost=e.get("cost_usd"),
                    attempts=e.get("attempts"),
                    work_units=e.get("work_units"),
                    reused_work_units=e.get("reused_work_units"),
                )
            elif st == "warning":
                p.set_status(
                    True,
                    cost=e.get("cost_usd"),
                    attempts=e.get("attempts"),
                    warning=True,
                    work_units=e.get("work_units"),
                    reused_work_units=e.get("reused_work_units"),
                )
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
        self._replaying = False
        for key in self.panels:
            self.panels[key].clear()
            self._set_tab_status(key, "pending")
        self._show(PHASES[0][0])
        self._set_follow_ui()


def _build_substage(text: str) -> str | None:
    """Map a `kicraft build` log line to its tab (markers from cli_app build).

    The step markers are anchored to the "[build] N/5" line head: a bare
    substring match ("1/5" in text) misfired on any line carrying a path like
    /projects/1/550/... — every project whose id starts with 5 flip-flopped
    the tab state machine mid-build.
    """
    t = text.lower()
    # Electrical review first: finding lines + the stage heading (these carry
    # no [build] N/5 marker of their own on the re-review path).
    if any(
        m in t for m in ("review blocker", "review warning", "review note", "electrical review")
    ):
        return "electrical_review"
    if not text.startswith("[build]"):
        return None  # timing/round/tool output: keep the current sub-stage
    if text.startswith("[build] 1/5") or "synthesized " in text:
        return "synthesize"
    if (
        text.startswith("[build] 2/5")
        or text.startswith("[build] 3/5")
        or text.startswith("[build] 4/5")
    ):
        return "place_route"
    if text.startswith("[build] 5/5"):
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
    ev += think(
        "The brief is a flashlight powered by an 18650 cell with USB-C ",
        "recharging. Core functions: USB-C 5V input, a Li-ion charger, the ",
        "18650 cell, a high-power white LED with a constant-current driver, ",
        "and a push-button to cycle modes. No microcontroller is required.",
    )
    ev += answer(
        '{"goal": "USB-C rechargeable 18650 flashlight, no microcontroller",',
        ' "inferred_expertise": "intermediate",',
        ' "named_parts": ["TP4056", "18650 cell"],',
        ' "assumptions": ["USB-C 5V input (defaulted)"],',
        ' "project_stem": "FLASHLIGHT"}',
    )
    ev.append({"kind": "stage_done", "stage": "intent", "ok": True, "cost": 0.0021})

    ev.append({"kind": "stage_start", "stage": "functional_spec", "model": _MODEL})
    ev += think(
        "Blocks: USB_C_INPUT, CHARGER, BATTERY, LED_DRIVER, LED, CONTROL. ",
        "Rails: VBUS 5.0V from USB, VBAT ~4.2V max from the cell. The driver ",
        "boosts VBAT to the LED forward voltage under constant current.",
    )
    ev += answer(
        '{"blocks": [{"name": "USB_C_INPUT", "category": "power", ',
        '"purpose": "5V from USB-C"}, {"name": "CHARGER", "category": "power", ',
        '"purpose": "TP4056 Li-ion charger"}], ',
        '"assumptions": ["1A charge current (defaulted)"]}',
    )
    ev.append({"kind": "stage_done", "stage": "functional_spec", "ok": True, "cost": 0.0034})

    ev.append({"kind": "stage_start", "stage": "architecture", "model": _MODEL})
    ev += think(
        "Single sheet is fine for this part count. Power nets: VBUS, VBAT, ",
        "GND, plus the switched LED node. TP4056 for the charger, a boost ",
        "constant-current driver for the LED, debounced push-button on CONTROL.",
    )
    ev += answer(
        '{"sheets": [{"name": "MAIN", "stem": "FLASHLIGHT", "function": "all"}], ',
        '"power_nets": ["VBUS", "VBAT", "GND"], ',
        '"rail_voltages": {"VBUS": 5.0, "VBAT": 4.2}, "mcu_present": false}',
    )
    ev.append({"kind": "stage_done", "stage": "architecture", "ok": True, "cost": 0.0048})

    ev.append({"kind": "stage_start", "stage": "bom", "model": _MODEL})
    ev.append(
        {
            "kind": "work_unit_plan",
            "stage": "bom",
            "unit_id": "bom-s000",
            "unit_sheet": "POWER INPUT",
            "source": "deterministic_architecture_lowering",
            "refs": [],
            "expected_pin_count": 0,
        }
    )
    ev.append(
        {
            "kind": "work_unit_done",
            "stage": "bom",
            "unit_id": "bom-s000",
            "unit_sheet": "POWER INPUT",
            "source": "deterministic_architecture_lowering",
        }
    )
    ev.append(
        {
            "kind": "work_unit_plan",
            "stage": "bom",
            "unit_id": "bom-s001",
            "unit_sheet": "MAIN",
            "source": "llm",
            "refs": [],
            "expected_pin_count": 0,
        }
    )
    ev += think(
        "I need real symbols and footprints. Start from the curated library, ",
        "then resolve the charger and USB-C connector from LCSC.",
    )
    ev.append({"kind": "tool", "name": "list_parts", "args": {}})
    ev.append(
        {
            "kind": "tool_result",
            "name": "list_parts",
            "output": "usb-c-16p   TYPE-C-31-M-12   USB-C receptacle, 16-pin\n"
            "tp4056      TP4056_C725790   1A Li-ion charger, ESOP-8\n"
            "Device:R    Resistor_SMD:R_0603_1608Metric\n"
            "Device:C    Capacitor_SMD:C_0603_1608Metric\n"
            "Device:LED  LED_SMD:LED_0603_1608Metric\n... 55 bundles",
        }
    )
    ev += think(
        "Both the USB-C receptacle and the charger are vendored bundles ",
        "(tp4056 is the lipo-charger-1s core default), so nothing needs ",
        "fetching: take the exact ids from list_parts verbatim.",
    )
    ev += think("All symbols and footprints resolve. Emit the BOM slot JSON.")
    ev += answer(
        '{"parts": [{"ref": "U1", "value": "TP4056", ',
        '"symbol": "tp4056:TP4056_C725790", "footprint": "tp4056:ESOP-8", "sheet": "MAIN"}, ',
        '{"ref": "J1", "value": "USB-C", "symbol": "usb-c-16p:TYPE-C-31-M-12", ',
        '"footprint": "usb-c-16p:TYPE-C", "sheet": "MAIN"}]}',
    )
    ev.append(
        {
            "kind": "work_unit_attempt",
            "stage": "bom",
            "unit_id": "bom-s001",
            "unit_sheet": "MAIN",
            "unit_attempt": 1,
            "source": "llm",
            "provider": "demo",
            "model": _MODEL,
            "outcome": "candidate",
            "wall_s": 4.2,
            "input_tokens": 1800,
            "output_tokens": 640,
            "cost_usd": 0.0431,
        }
    )
    ev.append(
        {
            "kind": "work_unit_done",
            "stage": "bom",
            "unit_id": "bom-s001",
            "unit_sheet": "MAIN",
            "source": "llm",
        }
    )
    ev.append(
        {
            "kind": "stage_done",
            "stage": "bom",
            "ok": True,
            "cost": 0.0431,
            "attempts": 1,
            "work_units": 2,
            "reused_work_units": 0,
        }
    )

    ev.append({"kind": "stage_start", "stage": "wiring", "model": _MODEL})
    ev += think(
        "Connect VBUS from USB-C to the charger input, VBAT to the cell and ",
        "the driver input, the LED node through the driver, and CONTROL to the ",
        "button. Tie unused USB-C pins (SBU1/SBU2, shield) to no_connect.",
    )
    ev.append(
        {
            "kind": "retry",
            "stage": "wiring",
            "errors": ["pin (U1,4) of TP4056 not covered by a connection or no_connect"],
        }
    )
    ev += think(
        "Missed the TP4056 PROG pin. Add R_prog from PROG to GND to set the ",
        "charge current, which also covers that pin.",
    )
    ev += answer(
        '{"connections": [{"net_name": "VBUS", "sheet": "MAIN", ',
        '"endpoints": [{"ref": "J1", "pin": "A4"}, {"ref": "U1", "pin": "4"}]}], ',
        '"no_connect_pins": [{"ref": "J1", "pin": "A8"}]}',
    )
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
