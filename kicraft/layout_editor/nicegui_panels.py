"""Shared NiceGUI control panels for the manual layout canvas.

Both hosts (offline GUI page, web place/route panel) are NiceGUI apps
driving the same canvas controller (``static/layout_canvas.js``), so
the control panels that talk to it over ``ui.run_javascript`` live
here once: outline size + shape controls, the mounting-holes panel,
view options, and the stamp/route result card.

Every function takes the ``canvas_id`` the host passed to
``build_canvas_html`` / ``build_canvas_init_script`` and pushes state
through ``window.manualLayoutCanvases[canvas_id]``; the ``&&`` guard
makes pushes a no-op until the controller has registered.
"""

from __future__ import annotations

import json
from typing import Any

from nicegui import ui

MOUNTING_HOLE_CORNER_OPTIONS = {
    "": "None (unpinned)",
    "top-left": "Top-Left",
    "top-right": "Top-Right",
    "bottom-left": "Bottom-Left",
    "bottom-right": "Bottom-Right",
}
MOUNTING_HOLE_SCREW_OPTIONS = ("M2", "M2.5", "M3", "M4")
# Default cycling order so a new hole picks up a sensible corner.
DEFAULT_CORNER_CYCLE = (
    "top-left",
    "bottom-right",
    "top-right",
    "bottom-left",
)

_SHAPE_OPTIONS = {
    "rect": "Rectangle",
    "rounded_rect": "Rounded",
    "chamfered_rect": "Chamfered",
    "circle": "Circle",
}


def _canvas_call(canvas_id: str, method: str, arg_js: str = "") -> None:
    ui.run_javascript(
        f"window.manualLayoutCanvases['{canvas_id}'] && "
        f"window.manualLayoutCanvases['{canvas_id}'].{method}({arg_js})"
    )


def outline_controls(canvas_id: str, initial: dict[str, Any]) -> None:
    """One row: W/H inputs + shape select + radius/chamfer parameter.

    Pushes are one-way (Python -> canvas). Hosts that also want the
    canvas's edge-handle drags mirrored back into the inputs add their
    own poll loop (the offline page does; the web panel reads the
    authoritative size from ``getState()`` at save time instead).
    """
    bo = initial.get("board_outline") or {}
    try:
        initial_w = float(bo["max"]["x"]) - float(bo["min"]["x"])
        initial_h = float(bo["max"]["y"]) - float(bo["min"]["y"])
    except (KeyError, TypeError, ValueError):
        initial_w, initial_h = 80.0, 60.0
    shape0 = dict(
        initial.get("outline_shape")
        or {"shape": "rect", "corner_radius_mm": 0.0, "chamfer_mm": 0.0}
    )

    with ui.row().classes("items-center gap-2 mt-1"):
        ui.label("Outline").classes("text-xs text-gray-400")
        width_input = ui.number(
            "W (mm)", value=round(initial_w, 2), min=10, step=0.5, format="%.2f",
        ).classes("w-24").props("dense")
        ui.label("×").classes("text-xs text-gray-500")
        height_input = ui.number(
            "H (mm)", value=round(initial_h, 2), min=10, step=0.5, format="%.2f",
        ).classes("w-24").props("dense")
        shape_select = ui.select(
            options=_SHAPE_OPTIONS,
            value=shape0.get("shape", "rect"),
            label="Shape",
        ).classes("w-36").props("dense options-dense")
        shape_param_input = ui.number(
            "Radius/chamfer (mm)",
            value=float(
                shape0.get("corner_radius_mm") or shape0.get("chamfer_mm") or 3.0
            ),
            min=0.5,
            step=0.5,
            format="%.1f",
        ).classes("w-36").props("dense")
        if shape0.get("shape") not in ("rounded_rect", "chamfered_rect"):
            shape_param_input.set_visibility(False)

    def _push_size() -> None:
        w = float(width_input.value or 0)
        h = float(height_input.value or 0)
        if w >= 10 and h >= 10:
            _canvas_call(canvas_id, "setOutlineSize", f"{w}, {h}")

    width_input.on("blur", lambda _e: _push_size())
    height_input.on("blur", lambda _e: _push_size())
    width_input.on("keydown.enter", lambda _e: _push_size())
    height_input.on("keydown.enter", lambda _e: _push_size())

    def _on_canvas_outline(e: Any) -> None:
        """Mirror canvas-side outline changes (edge-handle drags, circle
        squaring) back into the W/H inputs so the numbers on screen never
        go stale. Only writes values that actually differ -- that is what
        breaks the input->canvas->event->input cycle."""
        data = e.args if isinstance(e.args, dict) else {}
        if data.get("canvas_id") != canvas_id:
            return
        try:
            w = float(data["width"])
            h = float(data["height"])
        except (KeyError, TypeError, ValueError):
            return
        try:
            if abs(float(width_input.value or 0) - w) > 0.005:
                width_input.set_value(round(w, 2))
            if abs(float(height_input.value or 0) - h) > 0.005:
                height_input.set_value(round(h, 2))
        except (TypeError, ValueError):
            pass

    ui.on("kicraft-ml-outline", _on_canvas_outline)

    def _push_shape() -> None:
        shape = str(shape_select.value or "rect")
        try:
            param = float(shape_param_input.value)
        except (TypeError, ValueError):
            param = 0.0
        if shape in ("rounded_rect", "chamfered_rect") and param < 0.5:
            # A cleared field arrives as None (bypassing the widget's min=0.5),
            # and OutlineSpec.from_dict rejects rounded/chamfered with
            # param <= 0 at save -- clamp and reflect it in the input so the
            # canvas never shows an outline the pipeline refuses to persist.
            param = 0.5
            shape_param_input.set_value(param)
        spec = {
            "shape": shape,
            "corner_radius_mm": param if shape == "rounded_rect" else 0.0,
            "chamfer_mm": param if shape == "chamfered_rect" else 0.0,
        }
        shape_param_input.set_visibility(shape in ("rounded_rect", "chamfered_rect"))
        _canvas_call(canvas_id, "setOutlineShape", json.dumps(spec))

    shape_select.on_value_change(lambda _e: _push_shape())
    shape_param_input.on_value_change(lambda _e: _push_shape())


def mounting_hole_panel(canvas_id: str, initial_holes: list[dict]) -> None:
    """Collapsible expander to declare N mounting holes (corner / screw /
    inset per hole). State lives in a Python list mirrored to the canvas
    via ``setMountingHoles``; the canvas re-pegs pinned holes to the
    outline shape on every render."""
    state: list[dict] = []
    for i, h in enumerate(initial_holes):
        corner = h.get("corner")
        if corner not in MOUNTING_HOLE_CORNER_OPTIONS:
            corner = None
        screw = h.get("screw")
        if screw not in MOUNTING_HOLE_SCREW_OPTIONS:
            screw = "M3"
        entry = {
            "index": int(h.get("index", i)),
            "corner": corner,
            "inset_mm": float(h.get("inset_mm", 5.0)),
            "screw": screw,
        }
        # Keep the saved position: an unpinned (corner=None) hole's pos is
        # authoritative from manual_layout.json, and the mount-time push used
        # to drop it -- the canvas then reset the hole to the board's top-left
        # corner, which Save persisted and the composer stamped half off-board.
        pos = h.get("pos")
        if isinstance(pos, dict) and "x" in pos and "y" in pos:
            entry["pos"] = {"x": float(pos["x"]), "y": float(pos["y"])}
        state.append(entry)

    with ui.expansion(
        "Mounting Holes",
        icon="circle",
        value=bool(state),
    ).classes("w-full mt-4 bg-slate-800/40 rounded"):
        with ui.row().classes("items-center gap-3 px-2 pt-2"):
            count_input = ui.number(
                "Count", value=len(state), min=0, max=8, step=1, format="%d",
            ).classes("w-24")
            ui.label(
                "Each hole pegs to the named corner with the given inset; "
                "set corner to 'None' to skip the peg. Holes without a "
                "matching footprint in the schematic are synthesized from "
                "KiCad's stock MountingHole library at stamp time."
            ).classes("text-xs text-gray-400")
        rows_container = ui.column().classes("w-full px-2 pb-2 gap-1")

        def _push_to_canvas() -> None:
            _canvas_call(canvas_id, "setMountingHoles", json.dumps(state))

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
                        "corner": DEFAULT_CORNER_CYCLE[idx % len(DEFAULT_CORNER_CYCLE)],
                        "inset_mm": 5.0,
                        "screw": "M3",
                    }
                )
            while len(state) > n:
                state.pop()
            _rebuild()

        count_input.on_value_change(_on_count_change)
        _rebuild()


def _build_hole_row(i: int, hole: dict, on_change) -> None:
    """One H{N+1} row: corner dropdown + screw size + inset input."""
    with ui.row().classes("items-center gap-2"):
        ui.label(f"H{i + 1}").classes("text-xs text-gray-400 w-8")
        sel = ui.select(
            options=MOUNTING_HOLE_CORNER_OPTIONS,
            value=hole["corner"] or "",
            label="Corner",
        ).classes("w-44")

        def _on_corner(e: Any, _hole=hole) -> None:
            v = str(e.value or "")
            _hole["corner"] = v if v in MOUNTING_HOLE_CORNER_OPTIONS and v else None
            on_change()

        sel.on_value_change(_on_corner)

        screw_sel = ui.select(
            options=list(MOUNTING_HOLE_SCREW_OPTIONS),
            value=hole.get("screw", "M3"),
            label="Screw",
        ).classes("w-24")

        def _on_screw(e: Any, _hole=hole) -> None:
            v = str(e.value or "M3")
            _hole["screw"] = v if v in MOUNTING_HOLE_SCREW_OPTIONS else "M3"
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


def view_options_panel(canvas_id: str) -> None:
    """Collapsible View options: grid toggle, snap toggle, snap spacing.

    Defaults match the historical canvas behavior (grid on, snap on,
    0 mm gap) so opening the expansion without changing anything is a
    no-op -- the mount-time push below sends these values, so a nonzero
    spacing default here would silently break flush edge-to-edge snapping
    (and grow every stamped board) for users who never touched the field.
    """
    options: dict[str, Any] = {
        "show_grid": True,
        "snap_enabled": True,
        "snap_spacing_mm": 0.0,
        "show_ratsnest": True,
    }

    def _push() -> None:
        _canvas_call(canvas_id, "setViewOptions", json.dumps(options))

    def _on_grid_change(e: Any) -> None:
        options["show_grid"] = bool(e.value)
        _push()

    def _on_ratsnest_change(e: Any) -> None:
        options["show_ratsnest"] = bool(e.value)
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
            ui.switch(
                "Show connections (ratsnest)",
                value=options["show_ratsnest"],
                on_change=_on_ratsnest_change,
            ).tooltip(
                "Dashed lines join circuit blocks that share a net -- "
                "keep connected blocks close so routing has short paths. "
                "GND is plane-connected and not drawn."
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
    # mount -- the panel renders before the controller registers
    # window.manualLayoutCanvases[canvas_id], so a synchronous push
    # would silently no-op via the `&& ...` guard. once=True so it
    # self-destructs after firing; no cleanup needed.
    ui.timer(0.3, _push, once=True)


def compose_result_panel(card, result: dict[str, Any]) -> None:
    """Show stamp_drc / routing summary in a card below the canvas."""
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
