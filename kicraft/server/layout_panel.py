"""Web manual-layout editor for the place/route tab.

Renders the shared layout canvas (kicraft.layout_editor) inside the
place/route view slot so a paid user can drag the solved leaves into
their own arrangement, pick the board size and shape, declare mounting
holes, and stamp the result for an instant DRC preview. Routing the
saved layout is a build-queue job (PR: manual_route) and not handled
here.

Concurrency: the stamp runs compose_subcircuits as a subprocess inside
the web process (10-20 s incl. kicad-cli DRC). A module-level
semaphore caps concurrent stamps across all sessions and a timeout
bounds a hung subprocess; the build queue is NOT involved (stamping is
cheap and interactive).
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from nicegui import background_tasks, run, ui

from kicraft.layout_editor import (
    build_canvas_html,
    build_canvas_init_script,
    discover_leaves,
    load_initial_layout,
    load_last_route_result,
    load_parent_local_components,
    log_manual_event,
    run_manual_compose,
    save_manual_layout_json,
)
from kicraft.layout_editor.leaves import discover_missing_leaves
from kicraft.layout_editor.canvas import DEFAULT_ASSET_MOUNT
from kicraft.layout_editor.leaves import LeafUrlFor
from kicraft.layout_editor.ratsnest import build_ratsnest
from kicraft.layout_editor.nicegui_panels import (
    compose_result_panel,
    mounting_hole_panel,
    outline_controls,
    selected_block_controls,
    view_options_panel,
)
from kicraft.server.kicanvas import KiCanvasSource, KiCanvasView

# At most this many concurrent stamp subprocesses across every session
# on the box (each spawns pcbnew + kicad-cli DRC). Excess saves queue
# on the semaphore; the button shows a spinner meanwhile.
_STAMP_SEMAPHORE = asyncio.Semaphore(2)
# A stamp that exceeds this is reported as failed; the subprocess is
# killed with the page's await (run_manual_compose awaits proc.wait).
_STAMP_TIMEOUT_S = 180.0

EDITOR_TIERS = ("pro", "max")

# Stamp-DRC violation types worth a canvas marker: the ones the fab
# gate cares about. Silkscreen/mask cosmetics would bury the signal
# (a repeated-channel board can carry dozens of silk_overlap warnings).
_MARKER_TYPES = frozenset({
    "shorting_items", "clearance", "hole_clearance",
    "copper_edge_clearance", "courtyards_overlap",
    "items_not_allowed", "tracks_crossing", "unconnected_items",
})


def user_may_edit_layout(user) -> bool:
    """Pro/max (or admin) may use the layout editor; checked again at
    save time, never only in the UI."""
    if user is None:
        return False
    if getattr(user, "role", "") == "admin":
        return True
    return getattr(user, "tier", None) in EDITOR_TIERS


def leaf_artifacts_exist(project_dir: Path) -> bool:
    """Whether the editor has anything to edit: at least one routed
    leaf artifact under the project's .experiments tree."""
    sub = project_dir / ".experiments" / "subcircuits"
    if not sub.is_dir():
        return False
    return any(d.is_dir() and (d / "leaf_routed.kicad_pcb").is_file()
               for d in sub.iterdir())


def _project_render_url_for(project_dir: Path, token: str) -> LeafUrlFor:
    """Map a leaf canvas PNG to the tokened render route."""

    def url_for(png_path: Path) -> str | None:
        try:
            rel = png_path.relative_to(project_dir)
        except ValueError:
            return None
        try:
            v = int(png_path.stat().st_mtime)
        except OSError:
            v = 0
        return f"/project/{token}/render/{rel.as_posix()}?v={v}"

    return url_for


def manual_preview_name(stem: str) -> str:
    return f"{stem}_manual_preview.kicad_pcb"


class LayoutEditorPanel:
    """One open editor: canvas + controls + stamp preview.

    Built fresh each time the user opens the editor (leaf discovery and
    PNG rendering happen at construction); render() paints into the
    caller's current container slot.
    """

    def __init__(
        self,
        *,
        project_dir: Path,
        stem: str,
        token: str,
        user,
        on_exit,
        is_run_active,
        on_route=None,
    ) -> None:
        self.project_dir = project_dir
        self.experiments_dir = project_dir / ".experiments"
        self.stem = stem
        self.token = token
        self.user = user
        self.on_exit = on_exit
        # Callable: True while a build/design run owns this project's
        # workspace; saving is refused then (mutual exclusion with the
        # build queue).
        self.is_run_active = is_run_active
        # Host callback that enqueues the manual_route job for this
        # project (the queue/state plumbing lives with the page).
        self.on_route = on_route
        self.canvas_id = "web-layout-canvas"
        self._body = None
        self._missing_leaves: list[str] = []
        self._preview_view: KiCanvasView | None = None
        self._preview_slot = None
        self._status = None
        self._drc_card = None
        self._save_btn = None
        self._route_btn = None

    def saved_layout_path(self) -> Path:
        return self.experiments_dir / "manual" / "manual_layout.json"

    def _on_route_clicked(self) -> None:
        if not self.saved_layout_path().is_file():
            ui.notify("Save the layout first.", color="warning")
            return
        if self.on_route is not None:
            log_manual_event(self.experiments_dir, "route_enqueued")
            self.on_route()

    # -- UI -----------------------------------------------------------------

    def render(self) -> None:
        with ui.row().classes("w-full items-center gap-3"):
            ui.button("Back to board", icon="arrow_back",
                      on_click=self.on_exit).props("flat dense")
            ui.label("Manual layout").classes("text-sm font-medium") \
                .style("color:#e2e8f0")
            ui.label("drag · R rotate · arrows nudge (Shift = 1 mm) · "
                     "wheel zoom · Ctrl+Z undo") \
                .classes("text-xs").style("color:#64748b")
            # Live readouts the canvas controller writes into by DOM id
            # (updateCoordsLabel in layout_canvas.js): selected leaf's
            # sheet name, its x/y/rot, and the live outline W x H.
            ui.html(
                f'<span class="text-xs" style="color:#64748b">Selected: '
                f'<span id="{self.canvas_id}-selected" '
                f'style="color:#e2e8f0">none</span> '
                f'<span id="{self.canvas_id}-coords" '
                f'style="color:#94a3b8">--</span></span>',
                sanitize=False,
            )
            ui.html(
                f'<span class="text-xs" style="color:#64748b">Board: '
                f'<span id="{self.canvas_id}-outline" '
                f'style="color:#e2e8f0"></span></span>',
                sanitize=False,
            )
            self._status = ui.label("").classes("text-xs ml-auto") \
                .style("color:#94a3b8")

        # Leaf discovery renders a PNG per leaf (kicad-cli subprocess when
        # the build-tail pre-render didn't already warm the cache), so it
        # must NOT run on the UI event loop -- a cold open would stall
        # every session on the host. Build a shell synchronously, then
        # fill it from a background task that does the heavy work in the
        # executor. A background task (NOT an awaited continuation of the
        # caller): the host opens this panel from a ui.timer that lives
        # inside the very slot it clears, so an await in the caller's
        # task gets CANCELLED by that clear -- and nicegui's run.io_bound
        # swallows the CancelledError, silently dropping the body.
        self._body = ui.column().classes("w-full")
        with self._body:
            with ui.row().classes("items-center gap-2 mt-2"):
                ui.spinner(size="sm")
                ui.label("Preparing circuit blocks…").classes("text-xs") \
                    .style("color:#94a3b8")
        background_tasks.create(
            self._populate_body(),
            name=f"layout-editor-populate-{self.canvas_id}",
        )

    async def _populate_body(self) -> None:
        try:
            url_for = _project_render_url_for(self.project_dir, self.token)
            leaves = await run.io_bound(
                discover_leaves, self.experiments_dir, url_for=url_for
            )
            initial = await run.io_bound(
                load_initial_layout, self.experiments_dir, leaves
            )
            ratsnest = await run.io_bound(build_ratsnest, leaves)
            missing = await run.io_bound(
                discover_missing_leaves, self.experiments_dir
            )
            last_route = await run.io_bound(
                load_last_route_result, self.experiments_dir
            )
            parent_components = await run.io_bound(
                load_parent_local_components, self.experiments_dir
            )
            log_manual_event(self.experiments_dir, "editor_opened",
                             leaves=len(leaves), missing=len(missing))
        except Exception as exc:  # noqa: BLE001 - surface, don't vanish
            if self._body.is_deleted:
                return
            self._body.clear()
            with self._body:
                ui.label(f"Failed to load circuit blocks: {exc}") \
                    .classes("text-sm text-red-300")
            return
        if self._body.is_deleted:  # the user left the editor meanwhile
            return
        self._body.clear()
        with self._body:
            self._render_body(leaves, initial, ratsnest, missing, last_route,
                              parent_components)

    def _render_body(self, leaves, initial, ratsnest=None, missing=None,
                     last_route=None, parent_components=None) -> None:
        if not leaves:
            ui.label(
                "No solved leaves found for this project; run a build first."
            ).classes("text-sm text-amber-300")
            return

        # Honesty gate: the manual composer refuses a layout that does
        # not place EVERY expected leaf, so when some blocks have no
        # routed artifact, say which and block Save -- nothing the user
        # does on the canvas can fix a missing block.
        self._missing_leaves = list(missing or [])
        if self._missing_leaves:
            with ui.card().classes("w-full p-2 mb-1").style(
                    "border:1px solid #b45309;background:#451a03"):
                ui.label(
                    f"{len(leaves)} of {len(leaves) + len(self._missing_leaves)} "
                    "circuit blocks are available. Missing (their routing "
                    "failed): " + ", ".join(sorted(self._missing_leaves))
                    + ". Saving is disabled -- re-run the build (or fix the "
                    "design) so every block has a routed board."
                ).classes("text-xs").style("color:#fcd34d")

        # Round trip from a failed manual-route job: say WHY it failed and
        # (below, once the canvas has mounted) mark the failure locations,
        # instead of leaving the diagnosis in the generic failed-build view.
        route_markers: list = []
        if last_route and last_route.get("rc") != 0:
            v = last_route.get("verify") or {}
            with ui.card().classes("w-full p-2 mb-1").style(
                    "border:1px solid #b91c1c;background:#450a0a"):
                if last_route.get("stage") == "route":
                    ui.label(
                        "Last routing attempt failed: FreeRouting could not "
                        "route this arrangement. Give the connections more "
                        "room -- spread blocks apart along their net lines, "
                        "or grow the board outline."
                    ).classes("text-xs").style("color:#fca5a5")
                else:
                    bits = []
                    if v.get("shorts"):
                        bits.append(f"{v['shorts']} short(s)")
                    if v.get("unconnected"):
                        bits.append(f"{v['unconnected']} unconnected item(s)")
                    if v.get("courtyard"):
                        bits.append(f"{v['courtyard']} courtyard overlap(s)")
                    if v.get("keepout"):
                        bits.append(f"{v['keepout']} keep-out intrusion(s)")
                    detail = ", ".join(bits) \
                        or ", ".join(str(r) for r in v.get("reasons", [])) \
                        or f"rc={last_route.get('rc')}"
                    ui.label(
                        f"Last routing attempt failed verification: {detail}."
                    ).classes("text-xs").style("color:#fca5a5")
                    nets = v.get("unconnected_nets") or []
                    if nets:
                        shown = ", ".join(nets[:12])
                        more = "…" if len(nets) > 12 else ""
                        ui.label(f"Unconnected nets: {shown}{more}") \
                            .classes("text-xs").style("color:#fca5a5")
                route_markers = [
                    x for x in (v.get("violations") or [])
                    if x.get("type") in _MARKER_TYPES
                ]
                if route_markers:
                    ui.label(
                        "The failure locations are marked on the canvas "
                        "(hover a red pin for details)."
                    ).classes("text-xs").style("color:#fca5a5")

        ui.html(build_canvas_html(leaves, initial, self.canvas_id),
                sanitize=False).classes("w-full")
        ui.run_javascript(build_canvas_init_script(
            leaves, initial, self.canvas_id,
            asset_url=f"{DEFAULT_ASSET_MOUNT}/layout_canvas.js",
            ratsnest=ratsnest,
            parent_components=parent_components,
        ))
        if route_markers:
            # The controller registers asynchronously (asset load); defer
            # the push. The timer lives in the editor body, so leaving the
            # editor cancels it instead of pushing at a dead canvas.
            ui.timer(0.8, lambda: self._push_drc_markers(
                {"violations": route_markers}), once=True)

        outline_controls(self.canvas_id, initial)
        selected_block_controls(self.canvas_id)
        mounting_hole_panel(self.canvas_id, initial.get("mounting_holes") or [])
        view_options_panel(self.canvas_id)

        with ui.row().classes("w-full items-center gap-3 mt-2"):
            self._save_btn = ui.button(
                "Save & stamp preview", icon="save", color="primary",
                on_click=self._on_save,
            )
            if self._missing_leaves:
                self._save_btn.disable()
                self._save_btn.tooltip(
                    "Some circuit blocks have no routed board; the "
                    "composer cannot stamp a partial layout."
                )
            if self.on_route is not None:
                self._route_btn = ui.button(
                    "Route this layout", icon="bolt", color="secondary",
                    on_click=self._on_route_clicked,
                )
                if self._missing_leaves or not self.saved_layout_path().is_file():
                    self._route_btn.disable()
                self._route_btn.tooltip(
                    "Routes the saved layout with FreeRouting through the "
                    "build queue (minutes), then verifies it and rebuilds "
                    "the fab package."
                )
            ui.button("Reset", icon="refresh", on_click=lambda: ui.run_javascript(
                f"window.manualLayoutCanvases['{self.canvas_id}'].reset()"
            )).props("flat")
            ui.label(
                "Stamping places your layout on the real board and runs "
                "DRC (~20 s); routing commits it through the build queue."
            ).classes("text-xs").style("color:#64748b")

        self._drc_card = ui.card().classes("w-full mt-3 hidden")
        self._preview_slot = ui.column().classes("w-full mt-3")

    # -- Save & stamp ---------------------------------------------------------

    async def _on_save(self) -> None:
        if not user_may_edit_layout(self.user):
            ui.notify("The layout editor needs a Pro or Max plan.",
                      color="warning")
            return
        if self.is_run_active():
            ui.notify("A build is running for this project; wait for it "
                      "to finish before editing the layout.", color="warning")
            return
        if self._missing_leaves:
            ui.notify("Some circuit blocks have no routed board; saving "
                      "is disabled.", color="warning")
            return

        self._save_btn.props("loading")
        self._set_status("reading canvas state…")
        try:
            raw = await ui.run_javascript(
                f"JSON.stringify(window.manualLayoutCanvases"
                f"['{self.canvas_id}'].getState())",
                timeout=5.0,
            )
            payload = json.loads(raw) if raw else {}
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"error reading canvas: {exc}")
            self._save_btn.props(remove="loading")
            return

        try:
            leaves = await run.io_bound(
                discover_leaves,
                self.experiments_dir,
                url_for=_project_render_url_for(self.project_dir, self.token),
            )
            ml_path = save_manual_layout_json(
                self.experiments_dir, payload, leaves
            )
            if self._route_btn is not None:
                self._route_btn.enable()
            log_manual_event(self.experiments_dir, "layout_saved")
            self._set_status("saved; stamping + running DRC…")
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"save failed: {exc}")
            self._save_btn.props(remove="loading")
            return

        try:
            async with _STAMP_SEMAPHORE:
                result = await asyncio.wait_for(
                    run_manual_compose(
                        project_root=self.project_dir,
                        experiments_dir=self.experiments_dir,
                        manual_layout_path=ml_path,
                        pcb_file=f"{self.stem}.kicad_pcb",
                        parent="/",
                        route=False,
                    ),
                    timeout=_STAMP_TIMEOUT_S,
                )
        except asyncio.TimeoutError:
            self._set_status(
                f"stamp timed out after {int(_STAMP_TIMEOUT_S)} s; see "
                ".experiments/manual/stamp.log"
            )
            self._save_btn.props(remove="loading")
            return
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"stamp failed: {exc}")
            self._save_btn.props(remove="loading")
            return
        finally:
            self._save_btn.props(remove="loading")

        compose_result_panel(self._drc_card, result)
        _drc = result.get("stamp_drc") or {}
        log_manual_event(
            self.experiments_dir, "stamp_result",
            rc=result.get("rc"),
            shorts=_drc.get("shorts"),
            clearance=_drc.get("clearance"),
        )
        self._push_drc_markers(_drc)
        if result.get("rc") == 0:
            self._set_status("stamp ok; preview below reflects your layout")
            self._show_preview()
        else:
            self._set_status(
                f"stamp returned rc={result.get('rc')}; review the DRC "
                "counts and log below"
            )

    def _push_drc_markers(self, stamp_drc: dict) -> None:
        """Draw the stamp's gate-relevant DRC violations as markers at
        their board positions -- 'fix the thing under the red pin'
        instead of reading a report tail. Cosmetic types (silk overlap
        etc.) are filtered; the canvas clears markers on the next drag
        since they describe the just-stamped arrangement."""
        markers = [
            v for v in stamp_drc.get("violations") or []
            if v.get("type") in _MARKER_TYPES
        ]
        ui.run_javascript(
            f"window.manualLayoutCanvases['{self.canvas_id}'] && "
            f"window.manualLayoutCanvases['{self.canvas_id}']"
            f".setDrcMarkers({json.dumps(markers)})"
        )
        if markers:
            self._set_status(
                f"{len(markers)} DRC violation(s) marked on the canvas -- "
                "hover a red pin for details"
            )

    def _show_preview(self) -> None:
        """Copy the stamped board next to the project files (so the
        tokened file route can serve it) and (re)build the KiCanvas."""
        stamped = self._latest_stamped_board()
        if stamped is None:
            return
        preview = self.project_dir / manual_preview_name(self.stem)
        try:
            shutil.copy2(stamped, preview)
        except OSError as exc:
            self._set_status(f"preview copy failed: {exc}")
            return
        url = f"/project/{self.token}/{preview.name}"
        if self._preview_view is not None:
            self._preview_view.refresh()
            return
        with self._preview_slot:
            ui.label("Stamped preview (unrouted)").classes("text-xs font-medium") \
                .style("color:#94a3b8")
            self._preview_view = KiCanvasView(
                [KiCanvasSource(url, preview.name)],
                height="", style="height:46vh;min-height:320px",
            )

    def _latest_stamped_board(self) -> Path | None:
        sub = self.experiments_dir / "subcircuits"
        if not sub.is_dir():
            return None
        candidates = [
            f for d in sub.iterdir() if d.is_dir()
            for f in [d / "parent_pre_freerouting.kicad_pcb"] if f.is_file()
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _set_status(self, text: str) -> None:
        if self._status is not None:
            self._status.set_text(text)
