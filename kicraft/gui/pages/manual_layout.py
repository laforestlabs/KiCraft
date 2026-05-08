"""Manual layout page -- drag/rotate leaves into a user-chosen parent layout.

The user discovers solved leaves from .experiments/subcircuits/, drags
them onto an interactive canvas, optionally resizes the parent outline,
saves, sees stamp_drc results, and triggers FreeRouting on the chosen
layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nicegui import ui

from ..state import get_state
from .manual_layout_canvas import (
    build_canvas_html,
    build_canvas_init_script,
)
from .manual_layout_runner import (
    LeafInfo,
    discover_leaves,
    load_initial_layout,
    run_manual_compose,
    save_manual_layout_json,
)


def manual_layout_page() -> None:
    """Render the Manual Layout tab."""
    state = get_state()

    ui.label("Manual Layout").classes("text-2xl font-bold mb-1")
    ui.label(
        "Drag each leaf onto the parent canvas, rotate (R or right-click for "
        "90°; hold Shift while rotating for free angle), and pull the four "
        "edge handles to resize the outline. Click Save to stamp the layout "
        "and run a fast DRC pass; click Route Now to commit to FreeRouting."
    ).classes("text-sm text-gray-400 mb-4")

    # --- Discover leaves ---
    leaves = discover_leaves(state.experiments_dir)
    if not leaves:
        with ui.card().classes("p-4 bg-amber-900/20 border border-amber-600"):
            ui.label("No solved leaves found.").classes("text-amber-300 font-bold")
            ui.label(
                "Run a leaves-only experiment first (Setup → Start, or use "
                "the Monitor tab) so manual layout has something to place."
            ).classes("text-sm text-gray-400")
        return

    initial = load_initial_layout(state.experiments_dir, leaves)

    # --- Canvas + controls ---
    canvas_id = "manual-layout-canvas"
    status_label = ui.label("").classes("text-sm text-gray-300 ml-2")
    drc_card = ui.card().classes("w-full mt-4 hidden")
    drc_card.props("id=manual-drc-card")

    with ui.row().classes("w-full items-stretch gap-4"):
        with ui.column().classes("flex-1"):
            # sanitize=False is REQUIRED: NiceGUI 3.x defaults to
            # DOMPurify which strips <svg> and <style> blocks. Without
            # this the canvas renders as plain text and the JS init
            # script can't find the SVG container.
            ui.html(
                build_canvas_html(leaves, initial, canvas_id),
                sanitize=False,
            ).classes("w-full")
            # Inject the canvas controller as a body-level script. The
            # script polls for the SVG element to appear because the
            # Manual Layout tab panel is mounted lazily by Quasar --
            # the script runs at page-parse time, before the user
            # switches to this tab.
            init_js = build_canvas_init_script(leaves, initial, canvas_id)
            ui.add_body_html(f"<script>{init_js}</script>")
        with ui.column().classes("w-72 gap-3"):
            _legend(leaves)
            with ui.card().classes("p-3"):
                ui.label("Selected").classes("text-xs uppercase text-gray-400")
                ui.html(
                    f'<div id="{canvas_id}-selected" '
                    'class="text-sm text-gray-200">none</div>'
                )
                ui.html(
                    f'<div id="{canvas_id}-coords" '
                    'class="text-xs text-gray-400 mt-1">--</div>'
                )
                ui.label("Outline").classes("text-xs uppercase text-gray-400 mt-2")
                ui.html(
                    f'<div id="{canvas_id}-outline" '
                    'class="text-xs text-gray-400">--</div>'
                )

    with ui.row().classes("w-full items-center gap-3 mt-4"):
        save_btn = ui.button("Save Layout", icon="save", color="primary")
        route_btn = ui.button("Route Now", icon="bolt", color="secondary")
        route_btn.props("disable")
        ui.button("Reset", icon="refresh", on_click=lambda: ui.run_javascript(
            f"window.manualLayoutCanvases['{canvas_id}'].reset()"
        )).props("flat")

    saved_path: dict[str, Path | None] = {"path": None}

    async def _on_save() -> None:
        save_btn.props("loading")
        route_btn.props("disable")
        status_label.set_text("reading canvas state…")
        try:
            raw = await ui.run_javascript(
                f"JSON.stringify(window.manualLayoutCanvases['{canvas_id}'].getState())",
                timeout=5.0,
            )
            payload = json.loads(raw) if raw else {}
        except Exception as exc:  # noqa: BLE001
            status_label.set_text(f"error reading canvas: {exc}")
            save_btn.props(remove="loading")
            return

        try:
            ml_path = save_manual_layout_json(
                state.experiments_dir, payload, leaves
            )
            saved_path["path"] = ml_path
            status_label.set_text(
                f"saved {ml_path.relative_to(state.project_root)}; "
                "stamping + running DRC…"
            )
        except Exception as exc:  # noqa: BLE001
            status_label.set_text(f"save failed: {exc}")
            save_btn.props(remove="loading")
            return

        # Run compose --stamp (no route) to produce stamped board + stamp_drc.
        try:
            result = await run_manual_compose(
                project_root=state.project_root,
                experiments_dir=state.experiments_dir,
                manual_layout_path=ml_path,
                pcb_file=state.strategy["pcb_file"],
                parent="/",
                route=False,
            )
        except Exception as exc:  # noqa: BLE001
            status_label.set_text(f"stamp failed: {exc}")
            save_btn.props(remove="loading")
            return
        finally:
            save_btn.props(remove="loading")

        _render_drc_panel(drc_card, result)
        if result.get("rc") == 0:
            status_label.set_text("save + stamp ok; ready to route")
            route_btn.props(remove="disable")
        else:
            status_label.set_text(
                f"stamp returned rc={result.get('rc')}; review log + DRC"
            )

    async def _on_route() -> None:
        if saved_path["path"] is None:
            status_label.set_text("save the layout first")
            return
        route_btn.props("loading disable")
        save_btn.props("disable")
        status_label.set_text(
            "running FreeRouting (5–7 min) -- watch the Monitor tab for live progress"
        )
        try:
            result = await run_manual_compose(
                project_root=state.project_root,
                experiments_dir=state.experiments_dir,
                manual_layout_path=saved_path["path"],
                pcb_file=state.strategy["pcb_file"],
                parent="/",
                route=True,
            )
        except Exception as exc:  # noqa: BLE001
            status_label.set_text(f"route failed: {exc}")
            return
        finally:
            route_btn.props(remove="loading disable")
            save_btn.props(remove="disable")

        _render_drc_panel(drc_card, result)
        if result.get("rc") == 0:
            status_label.set_text(
                f"route ok in {result.get('elapsed_s', 0):.1f}s; "
                f"output: {result.get('output_json', '?')}"
            )
        else:
            status_label.set_text(
                f"route returned rc={result.get('rc')}; review log"
            )

    save_btn.on_click(_on_save)
    route_btn.on_click(_on_route)


def _legend(leaves: list[LeafInfo]) -> None:
    with ui.card().classes("p-3"):
        ui.label("Leaves").classes("text-xs uppercase text-gray-400")
        for leaf in leaves:
            with ui.row().classes("items-center gap-2"):
                swatch = ui.html(
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'background:{leaf.color};border-radius:2px"></span>'
                )
                ui.label(f"{leaf.sheet_name}").classes("text-sm")
                ui.label(
                    f"{leaf.width_mm:.1f}×{leaf.height_mm:.1f}mm"
                ).classes("text-xs text-gray-500 ml-auto")


def _render_drc_panel(card: ui.card, result: dict[str, Any]) -> None:
    """Show stamp_drc / routing summary in the panel below the canvas."""
    card.clear()
    card.classes(remove="hidden")
    with card:
        ui.label("Compose Result").classes("text-sm font-bold mb-1")
        rc = result.get("rc")
        rc_color = "text-green-400" if rc == 0 else "text-red-400"
        ui.label(f"rc={rc}, elapsed={result.get('elapsed_s', 0):.1f}s").classes(
            f"text-xs {rc_color}"
        )

        drc = result.get("stamp_drc") or {}
        if drc:
            with ui.row().classes("gap-4 mt-2"):
                _drc_pill("shorts", int(drc.get("shorts", 0)))
                _drc_pill("clearance", int(drc.get("clearance", 0)))
                _drc_pill("unconnected", int(drc.get("unconnected", 0)))
                _drc_pill("copper-edge", int(drc.get("copper_edge_clearance", 0)))

        log_tail = result.get("log_tail") or ""
        if log_tail:
            ui.label("Log (tail)").classes("text-xs uppercase text-gray-400 mt-3")
            ui.html(
                f'<pre class="text-[11px] text-gray-400 bg-slate-900/50 '
                f'p-2 rounded overflow-x-auto whitespace-pre-wrap">'
                f'{_html_escape(log_tail)}</pre>'
            )


def _drc_pill(label: str, count: int) -> None:
    color = "bg-green-900/40 text-green-300" if count == 0 else "bg-red-900/40 text-red-300"
    ui.html(
        f'<span class="px-2 py-1 rounded text-xs {color}">'
        f'{label}: {count}</span>'
    )


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
