"""Manual layout page -- drag/rotate leaves into a user-chosen parent layout.

The user discovers solved leaves from .experiments/subcircuits/, drags
them onto an interactive canvas, optionally resizes the parent outline,
saves, sees stamp_drc results, and triggers FreeRouting on the chosen
layout.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from nicegui import ui

from kicraft.layout_editor import (
    LeafInfo,
    build_canvas_html,
    build_canvas_init_script,
    discover_leaves,
    find_latest_parent_pcb,
    load_initial_layout,
    run_manual_compose,
    save_manual_layout_json,
)

from ..state import get_state
from .manual_layout_runner import open_in_pcbnew


def manual_layout_page() -> None:
    """Render the Manual Layout tab.

    Built once at page load. A 2 s watcher detects new leaf solves on
    disk and pushes a fresh canvas init script via
    ``ui.run_javascript``; the canvas controller's version sentinel +
    ``render()`` clear-and-repaint handle the in-place swap so the SVG
    repaints with the new leaf set without any Python element churn.

    The legend on the right gets a surgical clear+repopulate when
    leaves change (a few dozen messages). Everything else -- canvas
    wrapper, outline inputs, save/route buttons, mounting holes panel,
    view options panel -- is mounted once and survives untouched.

    This replaces an earlier ``@ui.refreshable`` body that rebuilt the
    whole panel per leaf landing. That sledgehammer + a 39 KB
    per-refresh ``<script>`` appended via ``ui.add_body_html`` produced
    a message storm and DOM growth that destabilised the Socket.IO
    connection, and a brief client reconnect would land on NiceGUI's
    hard-reload recovery (``outbox.try_rewind`` ->
    ``window.location.reload()``), snapping the active tab back to
    Setup mid-run.
    """
    state = get_state()
    rendered_at = {"ts": _max_leaf_mtime(state.experiments_dir)}
    canvas_id = "manual-layout-canvas"

    leaves = discover_leaves(state.experiments_dir)
    initial = load_initial_layout(state.experiments_dir, leaves)

    # Compact header: the canvas should claim the rest of the viewport
    # vertically. Detailed instructions live in a tooltip on the title
    # so they don't burn rows on regular use.
    with ui.row().classes("w-full items-center gap-3 mb-2"):
        ui.label("Manual Layout").classes("text-xl font-bold").tooltip(
            "Drag each leaf onto the parent canvas. Rotate with R or "
            "right-click (90° snap; hold Shift on the rotation handle "
            "for free angle). Pull any of the four outline-edge handles "
            "to resize. Save Layout stamps + runs DRC; Route Now commits "
            "to FreeRouting."
        )
        ui.label("drag · R / right-click rotate · edge handles resize").classes(
            "text-xs text-gray-500"
        )

    # Waiting banner: visible only when zero leaves have been solved
    # yet. Stays mounted so the watcher only has to toggle visibility,
    # never insert/remove DOM. _seeded_grid([]) already returns a
    # default outline so the canvas renders fine with no leaves.
    waiting_label = ui.label(
        "No solved leaves yet -- start a leaves-only run from the Monitor "
        "tab. The canvas refreshes automatically as each leaf completes."
    ).classes(
        "text-sm text-amber-300 mb-2 px-3 py-2 bg-amber-900/20 "
        "rounded border border-amber-600/40"
    )
    waiting_label.set_visibility(not leaves)

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

            # Outline size inputs -- two-way bound to the canvas state.
            # Drag updates the JS state; a 600ms timer poll mirrors it
            # back into these inputs, but only when the field isn't
            # focused (so it doesn't clobber what the user is typing).
            initial_w = float(initial["board_outline"]["max"]["x"]) - float(
                initial["board_outline"]["min"]["x"]
            )
            initial_h = float(initial["board_outline"]["max"]["y"]) - float(
                initial["board_outline"]["min"]["y"]
            )
            initial_shape = dict(
                initial.get("outline_shape")
                or {"shape": "rect", "corner_radius_mm": 0.0, "chamfer_mm": 0.0}
            )
            with ui.row().classes("items-center gap-2 mt-1"):
                ui.label("Outline").classes("text-xs text-gray-400")
                width_input = ui.number(
                    "W (mm)",
                    value=round(initial_w, 2),
                    min=10,
                    step=0.5,
                    format="%.2f",
                ).classes("w-24").props("dense")
                ui.label("×").classes("text-xs text-gray-500")
                height_input = ui.number(
                    "H (mm)",
                    value=round(initial_h, 2),
                    min=10,
                    step=0.5,
                    format="%.2f",
                ).classes("w-24").props("dense")
                shape_select = ui.select(
                    options={
                        "rect": "Rectangle",
                        "rounded_rect": "Rounded",
                        "chamfered_rect": "Chamfered",
                        "circle": "Circle",
                    },
                    value=initial_shape.get("shape", "rect"),
                    label="Shape",
                ).classes("w-36").props("dense options-dense")
                shape_param_input = ui.number(
                    "Radius/chamfer (mm)",
                    value=float(
                        initial_shape.get("corner_radius_mm")
                        or initial_shape.get("chamfer_mm")
                        or 3.0
                    ),
                    min=0.5,
                    step=0.5,
                    format="%.1f",
                ).classes("w-36").props("dense")
                if initial_shape.get("shape") not in ("rounded_rect", "chamfered_rect"):
                    shape_param_input.set_visibility(False)

            def _push_shape_to_canvas() -> None:
                shape = str(shape_select.value or "rect")
                param = float(shape_param_input.value or 0.0)
                spec = {
                    "shape": shape,
                    "corner_radius_mm": param if shape == "rounded_rect" else 0.0,
                    "chamfer_mm": param if shape == "chamfered_rect" else 0.0,
                }
                shape_param_input.set_visibility(
                    shape in ("rounded_rect", "chamfered_rect")
                )
                ui.run_javascript(
                    f"window.manualLayoutCanvases['{canvas_id}'] && "
                    f"window.manualLayoutCanvases['{canvas_id}']"
                    f".setOutlineShape({json.dumps(spec)})"
                )

            shape_select.on_value_change(lambda _e: _push_shape_to_canvas())
            shape_param_input.on_value_change(lambda _e: _push_shape_to_canvas())

            def _push_size_to_canvas() -> None:
                w = float(width_input.value or 0)
                h = float(height_input.value or 0)
                if w >= 10 and h >= 10:
                    ui.run_javascript(
                        f"window.manualLayoutCanvases['{canvas_id}']"
                        f".setOutlineSize({w}, {h})"
                    )

            width_input.on("blur", lambda _e: _push_size_to_canvas())
            height_input.on("blur", lambda _e: _push_size_to_canvas())
            width_input.on("keydown.enter", lambda _e: _push_size_to_canvas())
            height_input.on("keydown.enter", lambda _e: _push_size_to_canvas())

            async def _pull_size_from_canvas() -> None:
                # Don't trample mid-edit. The JS hasFocus check returns
                # true if either input is focused; in that case skip
                # the update so the user can finish typing.
                try:
                    has_focus = await ui.run_javascript(
                        "(document.activeElement && "
                        "document.activeElement.tagName === 'INPUT')",
                        timeout=1.0,
                    )
                    if has_focus:
                        return
                    size = await ui.run_javascript(
                        f"window.manualLayoutCanvases['{canvas_id}'] "
                        f"&& window.manualLayoutCanvases['{canvas_id}']"
                        f".getOutlineSize()",
                        timeout=1.0,
                    )
                except Exception:  # noqa: BLE001
                    return
                if not isinstance(size, dict):
                    return
                w = float(size.get("width", 0))
                h = float(size.get("height", 0))
                if w > 0 and abs(w - float(width_input.value or 0)) > 0.01:
                    width_input.value = round(w, 2)
                if h > 0 and abs(h - float(height_input.value or 0)) > 0.01:
                    height_input.value = round(h, 2)

            pull_size_timer = ui.timer(0.6, _pull_size_from_canvas)

            # Canvas controller injection. ALWAYS via run_javascript --
            # add_body_html would append a fresh <script> tag to the
            # body on every push, accumulating ~39 KB per leaf landing
            # in the DOM until the browser fell behind and NiceGUI's
            # reconnect-recovery hard-reloaded the page. The IIFE's
            # tryInit MutationObserver waits for the SVG to mount
            # (Quasar lazy-mounts inactive tabs, see commit 992c03c)
            # and its version sentinel lets later pushes supersede
            # earlier ones cleanly.
            ui.run_javascript(
                build_canvas_init_script(leaves, initial, canvas_id)
            )

        with ui.column().classes("w-72 gap-3"):
            # Stable container so the watcher can clear + repopulate
            # the legend rows without churning the surrounding column.
            legend_card = ui.card().classes("p-3")
            _legend(legend_card, leaves)
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

    with ui.row().classes("w-full items-center gap-3 mt-2"):
        save_btn = ui.button("Save Layout", icon="save", color="primary")
        route_btn = ui.button("Route Now", icon="bolt", color="secondary")
        route_btn.props("disable")
        open_btn = ui.button("Open in KiCad", icon="open_in_new")
        open_btn.props("flat")
        ui.button("Reset", icon="refresh", on_click=lambda: ui.run_javascript(
            f"window.manualLayoutCanvases['{canvas_id}'].reset()"
        )).props("flat")

    _mounting_hole_panel(canvas_id, initial.get("mounting_holes") or [])

    _view_options_panel(canvas_id)

    def _on_open_in_kicad() -> None:
        pcb = find_latest_parent_pcb(state.experiments_dir)
        if pcb is None:
            status_label.set_text(
                "no parent board on disk yet -- save the layout first"
            )
            return
        try:
            open_in_pcbnew(pcb)
        except FileNotFoundError:
            status_label.set_text(
                "pcbnew not found on PATH; install kicad or open manually: " + str(pcb)
            )
            return
        status_label.set_text(f"launched pcbnew on {pcb.name}")

    open_btn.on_click(_on_open_in_kicad)

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
            # Re-discover so a save after the watcher has picked up
            # additional leaves still matches the canvas's current
            # placements -- save_manual_layout_json filters out any
            # placement whose instance_path isn't in this list.
            current_leaves = discover_leaves(state.experiments_dir)
            ml_path = save_manual_layout_json(
                state.experiments_dir, payload, current_leaves
            )
            saved_path["path"] = ml_path
            # Drop any prior parent_routed.kicad_pcb so the
            # find_latest_parent_pcb helper used by "Open in KiCad"
            # doesn't return a stale routed board from a previous
            # session -- the user just changed the layout, the
            # last-routed result no longer matches their canvas.
            for sub in (state.experiments_dir / "subcircuits").glob(
                "*/parent_routed.kicad_pcb"
            ):
                try:
                    sub.unlink()
                except OSError:
                    pass
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

    def _watch() -> None:
        latest = _max_leaf_mtime(state.experiments_dir)
        if latest == rendered_at["ts"]:
            return
        # Let bursts settle: if the latest write is younger than 3 s,
        # wait for the run to finish dumping its current leaf before
        # rebuilding. Keeps the canvas from flickering through
        # half-written solved_layout.json files.
        if latest > 0 and (time.time() - latest) < 3.0:
            return
        rendered_at["ts"] = latest
        new_leaves = discover_leaves(state.experiments_dir)
        new_initial = load_initial_layout(state.experiments_dir, new_leaves)
        waiting_label.set_visibility(not new_leaves)
        _legend(legend_card, new_leaves)
        ui.run_javascript(
            build_canvas_init_script(new_leaves, new_initial, canvas_id)
        )

    watch_timer = ui.timer(2.0, _watch)

    # Cancel both timers on client disconnect so they don't tick into a
    # deleted parent_slot after a page reload (NiceGUI's _should_stop
    # check runs AFTER _get_context() inside the loop, so a fresh tick
    # whose parent was GC'd between _can_start() and _get_context()
    # always raises -- see the patch in app.py for the same reason).
    def _on_disconnect() -> None:
        for t in (watch_timer, pull_size_timer):
            try:
                t.cancel()
            except Exception:  # noqa: BLE001
                pass

    ui.context.client.on_disconnect(_on_disconnect)


def _legend(card: ui.card, leaves: list[LeafInfo]) -> None:
    """Clear ``card`` and repopulate it with one row per leaf.

    Takes the container so the caller can keep its identity stable
    across watcher updates -- only the rows inside churn, not the
    surrounding column.
    """
    card.clear()
    with card:
        ui.label("Leaves").classes("text-xs uppercase text-gray-400")
        for leaf in leaves:
            silk_w = leaf.silk_max_x - leaf.silk_min_x
            silk_h = leaf.silk_max_y - leaf.silk_min_y
            with ui.row().classes("items-center gap-2"):
                ui.html(
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'background:{leaf.color};border-radius:2px"></span>'
                )
                ui.label(f"{leaf.sheet_name}").classes("text-sm")
                ui.label(
                    f"{silk_w:.1f}×{silk_h:.1f}mm"
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


def _max_leaf_mtime(experiments_dir: Path) -> float:
    """Latest mtime of any leaf's solved_layout.json or routed pcb.

    Used by the canvas auto-refresh watcher: when a fresh leaves-only
    run lands new files, this number jumps and the watcher fires
    once the writes have settled.
    """
    sub_root = experiments_dir / "subcircuits"
    if not sub_root.is_dir():
        return 0.0
    latest = 0.0
    for d in sub_root.iterdir():
        if not d.is_dir():
            continue
        for name in ("solved_layout.json", "leaf_routed.kicad_pcb"):
            f = d / name
            if f.is_file():
                try:
                    mt = f.stat().st_mtime
                    if mt > latest:
                        latest = mt
                except OSError:
                    pass
    return latest


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_MOUNTING_HOLE_CORNER_OPTIONS = {
    "": "None (unpinned)",
    "top-left": "Top-Left",
    "top-right": "Top-Right",
    "bottom-left": "Bottom-Left",
    "bottom-right": "Bottom-Right",
}
_MOUNTING_HOLE_SCREW_OPTIONS = ("M2", "M2.5", "M3", "M4")
# Default cycling order so a new hole picks up a sensible corner.
_DEFAULT_CORNER_CYCLE = (
    "top-left",
    "bottom-right",
    "top-right",
    "bottom-left",
)


def _mounting_hole_panel(canvas_id: str, initial_holes: list[dict]) -> None:
    """Collapsible expander to declare N mounting holes and their corners.

    State lives in a Python list mirrored to the canvas via
    ``setMountingHoles``. The canvas re-pegs each pinned hole to the
    outline's corner + inset on every render, so the user can resize
    the outline and see the holes follow.
    """
    state: list[dict] = []
    for i, h in enumerate(initial_holes):
        corner = h.get("corner")
        if corner not in _MOUNTING_HOLE_CORNER_OPTIONS:
            corner = None
        screw = h.get("screw")
        if screw not in _MOUNTING_HOLE_SCREW_OPTIONS:
            screw = "M3"
        state.append(
            {
                "index": int(h.get("index", i)),
                "corner": corner,
                "inset_mm": float(h.get("inset_mm", 5.0)),
                "screw": screw,
            }
        )

    with ui.expansion(
        "Mounting Holes",
        icon="circle",
        value=bool(state),
    ).classes("w-full mt-4 bg-slate-800/40 rounded"):
        with ui.row().classes("items-center gap-3 px-2 pt-2"):
            count_input = ui.number(
                "Count",
                value=len(state),
                min=0,
                max=8,
                step=1,
                format="%d",
            ).classes("w-24")
            ui.label(
                "Each hole pegs to the named corner with the given inset; "
                "set corner to 'None' to skip the peg."
            ).classes("text-xs text-gray-400")
        rows_container = ui.column().classes("w-full px-2 pb-2 gap-1")

        def _push_to_canvas() -> None:
            ui.run_javascript(
                f"window.manualLayoutCanvases['{canvas_id}'] && "
                f"window.manualLayoutCanvases['{canvas_id}']"
                f".setMountingHoles({json.dumps(state)})"
            )

        def _rebuild() -> None:
            rows_container.clear()
            with rows_container:
                for i, h in enumerate(state):
                    _build_hole_row(i, h, _push_to_canvas)
            _push_to_canvas()

        def _on_count_change(e: Any) -> None:
            n = max(0, min(8, int(e.value or 0)))
            while len(state) < n:
                idx = len(state)
                state.append(
                    {
                        "index": idx,
                        "corner": _DEFAULT_CORNER_CYCLE[
                            idx % len(_DEFAULT_CORNER_CYCLE)
                        ],
                        "inset_mm": 5.0,
                        "screw": "M3",
                    }
                )
            while len(state) > n:
                state.pop()
            _rebuild()

        count_input.on_value_change(_on_count_change)
        _rebuild()


def _view_options_panel(canvas_id: str) -> None:
    """Collapsible View options panel at the bottom of the manual layout tab.

    Toggles plus a spacing input that propagate to the canvas controller
    via ``setViewOptions(opts)``. Defaults match the historical canvas
    behavior (grid on, snap on, 0 mm gap) so flipping the expansion open
    without changing anything is a no-op. The spacing field defaults to
    1 mm so flipping it on visibly opens up gaps between snapped leaves.
    """
    options: dict[str, Any] = {
        "show_grid": True,
        "snap_enabled": True,
        "snap_spacing_mm": 1.0,
    }

    def _push() -> None:
        ui.run_javascript(
            f"window.manualLayoutCanvases['{canvas_id}'] && "
            f"window.manualLayoutCanvases['{canvas_id}']"
            f".setViewOptions({json.dumps(options)})"
        )

    def _on_grid_change(e: Any) -> None:
        options["show_grid"] = bool(e.value)
        _push()

    def _on_snap_change(e: Any) -> None:
        options["snap_enabled"] = bool(e.value)
        _push()

    def _on_spacing_change(e: Any) -> None:
        try:
            options["snap_spacing_mm"] = max(0.0, float(e.value or 0))
        except (TypeError, ValueError):
            options["snap_spacing_mm"] = 0.0
        _push()

    with ui.expansion(
        "View options",
        icon="visibility",
        value=False,
    ).classes("w-full mt-4 bg-slate-800/40 rounded"):
        with ui.column().classes("w-full px-2 py-2 gap-2"):
            ui.switch(
                "Show grid",
                value=options["show_grid"],
                on_change=_on_grid_change,
            )
            with ui.row().classes("items-center gap-3"):
                ui.switch(
                    "Snap leafs together",
                    value=options["snap_enabled"],
                    on_change=_on_snap_change,
                )
                ui.number(
                    "Spacing (mm)",
                    value=options["snap_spacing_mm"],
                    min=0,
                    step=0.1,
                    format="%.2f",
                    on_change=_on_spacing_change,
                ).classes("w-32")

    # Push the initial values once the canvas JS has had a chance to
    # mount -- the panel renders before the IIFE registers
    # window.manualLayoutCanvases[canvas_id], so a synchronous push
    # would silently no-op via the `&& ...` guard. once=True so it
    # self-destructs after firing; no cleanup needed.
    ui.timer(0.3, _push, once=True)


def _build_hole_row(i: int, hole: dict, on_change) -> None:
    """One H{N+1} row: corner dropdown + screw size + inset input."""
    with ui.row().classes("items-center gap-2"):
        ui.label(f"H{i + 1}").classes("text-xs text-gray-400 w-8")
        sel = ui.select(
            options=_MOUNTING_HOLE_CORNER_OPTIONS,
            value=hole["corner"] or "",
            label="Corner",
        ).classes("w-44")

        def _on_corner(e: Any, _hole=hole) -> None:
            v = str(e.value or "")
            _hole["corner"] = v if v in _MOUNTING_HOLE_CORNER_OPTIONS and v else None
            on_change()

        sel.on_value_change(_on_corner)

        screw_sel = ui.select(
            options=list(_MOUNTING_HOLE_SCREW_OPTIONS),
            value=hole.get("screw", "M3"),
            label="Screw",
        ).classes("w-24")

        def _on_screw(e: Any, _hole=hole) -> None:
            v = str(e.value or "M3")
            _hole["screw"] = v if v in _MOUNTING_HOLE_SCREW_OPTIONS else "M3"
            on_change()

        screw_sel.on_value_change(_on_screw)

        inset = ui.number(
            "Inset (mm)",
            value=hole["inset_mm"],
            min=0,
            step=0.5,
            format="%.2f",
        ).classes("w-32")

        def _on_inset(e: Any, _hole=hole) -> None:
            try:
                _hole["inset_mm"] = max(0.0, float(e.value))
            except (TypeError, ValueError):
                return
            on_change()

        inset.on_value_change(_on_inset)
