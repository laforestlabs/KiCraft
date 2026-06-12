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

from nicegui import ui

from kicraft.layout_editor import (
    build_canvas_html,
    build_canvas_init_script,
    discover_leaves,
    load_initial_layout,
    run_manual_compose,
    save_manual_layout_json,
)
from kicraft.layout_editor.canvas import DEFAULT_ASSET_MOUNT
from kicraft.layout_editor.leaves import LeafUrlFor
from kicraft.layout_editor.nicegui_panels import (
    compose_result_panel,
    mounting_hole_panel,
    outline_controls,
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
        self.canvas_id = "web-layout-canvas"
        self._preview_view: KiCanvasView | None = None
        self._preview_slot = None
        self._status = None
        self._drc_card = None
        self._save_btn = None

    # -- UI -----------------------------------------------------------------

    def render(self) -> None:
        leaves = discover_leaves(
            self.experiments_dir,
            url_for=_project_render_url_for(self.project_dir, self.token),
        )
        initial = load_initial_layout(self.experiments_dir, leaves)

        with ui.row().classes("w-full items-center gap-3"):
            ui.button("Back to board", icon="arrow_back",
                      on_click=self.on_exit).props("flat dense")
            ui.label("Manual layout").classes("text-sm font-medium") \
                .style("color:#e2e8f0")
            ui.label("drag · R / right-click rotate · edge handles resize") \
                .classes("text-xs").style("color:#64748b")
            self._status = ui.label("").classes("text-xs ml-auto") \
                .style("color:#94a3b8")

        if not leaves:
            ui.label(
                "No solved leaves found for this project; run a build first."
            ).classes("text-sm text-amber-300")
            return

        ui.html(build_canvas_html(leaves, initial, self.canvas_id),
                sanitize=False).classes("w-full")
        ui.run_javascript(build_canvas_init_script(
            leaves, initial, self.canvas_id,
            asset_url=f"{DEFAULT_ASSET_MOUNT}/layout_canvas.js",
        ))

        outline_controls(self.canvas_id, initial)
        mounting_hole_panel(self.canvas_id, initial.get("mounting_holes") or [])
        view_options_panel(self.canvas_id)

        with ui.row().classes("w-full items-center gap-3 mt-2"):
            self._save_btn = ui.button(
                "Save & stamp preview", icon="save", color="primary",
                on_click=self._on_save,
            )
            ui.button("Reset", icon="refresh", on_click=lambda: ui.run_javascript(
                f"window.manualLayoutCanvases['{self.canvas_id}'].reset()"
            )).props("flat")
            ui.label(
                "Stamping places your layout on the real board and runs "
                "DRC (~20 s). Routing the saved layout comes next."
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
            leaves = discover_leaves(
                self.experiments_dir,
                url_for=_project_render_url_for(self.project_dir, self.token),
            )
            ml_path = save_manual_layout_json(
                self.experiments_dir, payload, leaves
            )
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
        if result.get("rc") == 0:
            self._set_status("stamp ok; preview below reflects your layout")
            self._show_preview()
        else:
            self._set_status(
                f"stamp returned rc={result.get('rc')}; review the DRC "
                "counts and log below"
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
