"""Node detail panel -- shows full PCB render, score plot, and round timeline."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import plotly.graph_objects as go
from nicegui import ui

from .pipeline_graph import NodeStatus, RoundInfo


def node_detail_panel(
    node: NodeStatus,
    experiments_dir: Path | None = None,
    on_pins_changed: Callable[[], None] | None = None,
) -> None:
    """Render the detail panel for a selected pipeline node.

    When `experiments_dir` is provided and the selected node is a leaf,
    appends a "Snapshots" picker showing every prior experiment-round
    snapshot for this leaf with Pin/Unpin controls. Pin/unpin actions
    fire `on_pins_changed` so the caller (Monitor) can refresh its
    pinned-leaves summary.
    """
    # Mutable container for the "maximized" image path so clicking a round
    # thumbnail below can swap the big render above without rebuilding the
    # whole panel.
    if node.status == "routing_failed":
        initial_label = "Pre-route (routing failed)"
    else:
        initial_label = "Best round"
    maximized = {"src": node.best_render, "label": initial_label}

    with ui.column().classes("w-full gap-3"):
        _header(node)
        if node.status == "routing_failed":
            _rejection_reason_panel(node)
        main_image_host = ui.column().classes("w-full")
        _render_main_image(main_image_host, maximized)
        if node.is_leaf and node.rounds:
            _score_plot(node)
            _round_timeline(node, main_image_host, maximized)
        if node.is_leaf and experiments_dir is not None and node.artifact_dir:
            _snapshot_picker(
                node,
                Path(experiments_dir),
                on_pins_changed=on_pins_changed,
            )


def _snapshot_picker(
    node: NodeStatus,
    experiments_dir: Path,
    on_pins_changed: Callable[[], None] | None = None,
) -> None:
    """Per-leaf round snapshot picker with Pin/Unpin controls.

    Reads .experiments/pins.json + the leaf's round_NNNN_* snapshot files
    via the kicraft.autoplacer.brain.pins module. Replaces the per-leaf
    pin section that used to live in the Analysis tab's leaf gallery.
    """
    from kicraft.autoplacer.brain import pins as pins_module

    leaf_key = Path(node.artifact_dir).name
    leaf_dir = experiments_dir / "subcircuits" / leaf_key
    renders_dir = leaf_dir / "renders"

    section = ui.column().classes("w-full gap-2 mt-3")

    def _redraw() -> None:
        section.clear()
        available = pins_module.list_available_rounds(experiments_dir, leaf_key)
        current_pin = pins_module.is_pinned(experiments_dir, leaf_key)
        with section:
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon("push_pin", color="amber").classes("text-lg")
                ui.label("Snapshots").classes("text-base font-bold")
                if current_pin is not None:
                    ui.badge(
                        f"PINNED to round {current_pin}", color="amber"
                    ).classes("text-xs")

                    def _unpin() -> None:
                        pins_module.unpin_leaf(experiments_dir, leaf_key)
                        ui.notify(f"Unpinned {leaf_key}", color="amber")
                        if on_pins_changed:
                            on_pins_changed()
                        _redraw()

                    ui.space()
                    ui.button("Unpin", icon="link_off", on_click=_unpin).props(
                        "flat dense"
                    ).classes("text-amber-300 text-xs")
                else:
                    ui.space()
                    ui.label(
                        f"{len(available)} pickable snapshot{'s' if len(available) != 1 else ''}"
                    ).classes("text-xs text-gray-400")

            if not available:
                # Distinguish "trivial leaf with no PCB" (e.g. BT1, a
                # battery with no internal nets) from "haven't run an
                # experiment yet". Trivial leaves never produce a
                # leaf_routed.kicad_pcb because there's nothing to route.
                has_any_pcb = (
                    leaf_dir.exists()
                    and any(leaf_dir.glob("*_leaf_routed.kicad_pcb"))
                )
                if leaf_dir.exists() and not has_any_pcb:
                    ui.label(
                        "This leaf has no internal nets to route -- there's "
                        "no PCB layout to pin. The composer uses the leaf's "
                        "metadata directly, and the parent_only gate exempts "
                        "it from the pinning requirement."
                    ).classes("text-xs text-gray-400 italic")
                else:
                    ui.label(
                        "No round snapshots on disk yet for this leaf. Run "
                        "a leaves-only or complete experiment to generate "
                        "them (each successful placement attempt becomes a "
                        "pickable candidate)."
                    ).classes("text-xs text-gray-500 italic")
                return

            ui.label(
                "Each snapshot is a complete leaf state from a prior placement "
                "attempt. Pinning copies the snapshot over the canonical files "
                "so the next parent compose uses it."
            ).classes("text-xs text-gray-400")

            with ui.grid(columns=4).classes("w-full gap-2"):
                for round_num in available:
                    is_current = current_pin == round_num
                    card_classes = "w-full p-2 bg-slate-900/70"
                    if is_current:
                        card_classes += " border-2 border-amber-400"
                    with ui.card().classes(card_classes):
                        with ui.row().classes("w-full items-center gap-1"):
                            ui.badge(f"R{round_num}", color="blue").classes("text-xs")
                            if is_current:
                                ui.badge("PINNED", color="amber").classes("text-xs")

                        preview = renders_dir / (
                            f"round_{round_num:04d}_routed_front_all.png"
                        )
                        if not preview.exists():
                            preview = renders_dir / (
                                f"round_{round_num:04d}_pre_route_front_all.png"
                            )
                        if preview.exists():
                            ui.image(str(preview)).classes(
                                "w-full h-[140px] object-contain rounded "
                                "border border-slate-700 bg-slate-950"
                            )
                        else:
                            ui.label("(no preview)").classes(
                                "text-xs text-gray-500 italic"
                            )

                        def _make_pin(rn=round_num):
                            def _on_pin() -> None:
                                try:
                                    pins_module.pin_leaf(
                                        experiments_dir, leaf_key, rn
                                    )
                                    ui.notify(
                                        f"Pinned {node.name} to round {rn}",
                                        color="positive",
                                    )
                                    if on_pins_changed:
                                        on_pins_changed()
                                    _redraw()
                                except FileNotFoundError as exc:
                                    ui.notify(str(exc), color="negative")

                            return _on_pin

                        ui.button(
                            "Pin this round" if not is_current else "Re-apply pin",
                            icon="push_pin",
                            on_click=_make_pin(),
                        ).props("flat dense").classes(
                            "text-amber-300 text-xs w-full"
                        )

    _redraw()


def _header(node: NodeStatus) -> None:
    with ui.row().classes("w-full items-center gap-3"):
        ui.label(node.name).classes("text-xl font-bold")
        badge_text = (
            "FAILED TO ROUTE" if node.status == "routing_failed" else node.status.upper()
        )
        ui.badge(badge_text, color=_status_color(node.status))
        ui.space()
        if node.score is not None:
            ui.label(f"Score: {node.score:.2f}").classes(
                "text-lg font-mono text-green-400"
            )

    if node.is_leaf:
        with ui.row().classes("gap-4 text-sm text-gray-300"):
            ui.label(f"Components: {node.component_count}")
            ui.label(f"Traces: {node.traces}")
            ui.label(f"Vias: {node.vias}")
            ui.label(f"Rounds: {node.total_rounds_run}")


def _rejection_reason_panel(node: NodeStatus) -> None:
    """Banner explaining why the round(s) in view failed to route.

    Counts how often each rejection reason appears across this leaf's
    rounds in the current view. Shows the top 2 reasons so the user can
    quickly tell whether a leaf is consistently hitting one failure mode
    or churning through several.
    """
    reasons: dict[str, int] = {}
    for r in node.rounds:
        if r.rejection_reason:
            reasons[r.rejection_reason] = reasons.get(r.rejection_reason, 0) + 1

    with ui.card().classes("w-full bg-red-900/30 border border-red-700 p-2"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("warning", color="red-4").classes("text-red-400")
            ui.label("Routing rejected").classes(
                "text-sm font-medium text-red-300"
            )
        if not reasons:
            ui.label(
                "No rejection reason recorded. Check solve_subcircuits "
                "stderr or the leaf's debug.json."
            ).classes("text-xs text-red-200 mt-1")
            return
        sorted_reasons = sorted(reasons.items(), key=lambda kv: -kv[1])
        for reason, count in sorted_reasons[:3]:
            ui.label(f"{reason}  ({count}× in view)").classes(
                "text-xs text-red-200 font-mono"
            )


def _render_main_image(host, maximized: dict) -> None:
    """(Re)render the large image area from the `maximized` dict in-place."""
    host.clear()
    with host:
        src = maximized.get("src")
        label = maximized.get("label", "")
        if src:
            if label:
                ui.label(label).classes("text-xs text-gray-400")
            ui.image(src).classes(
                "w-full max-h-[400px] object-contain rounded bg-slate-950 border border-slate-700"
            )
        else:
            with ui.row().classes(
                "w-full h-[200px] items-center justify-center bg-slate-950 rounded border border-slate-700"
            ):
                ui.label("No render available").classes("text-gray-500 italic")


def _score_plot(node: NodeStatus) -> None:
    if not node.rounds:
        return

    x_rounds = [r.index for r in node.rounds]
    y_scores = [r.score for r in node.rounds]

    best_y: list[float] = []
    running_best = float("-inf")
    for s in y_scores:
        if s > running_best:
            running_best = s
        best_y.append(running_best)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_rounds,
        y=y_scores,
        mode="markers+lines",
        name="Score",
        line=dict(color="#4dabf7", width=2),
        marker=dict(size=7, color="#4dabf7"),
    ))
    fig.add_trace(go.Scatter(
        x=x_rounds,
        y=best_y,
        mode="lines",
        name="Best",
        line=dict(color="#51cf66", width=2, dash="dot"),
    ))
    fig.update_layout(
        template="plotly_dark",
        height=180,
        margin=dict(l=40, r=20, t=30, b=30),
        xaxis_title="Round",
        yaxis_title="Score",
        showlegend=False,
    )
    ui.plotly(fig).classes("w-full")


def _round_timeline(node: NodeStatus, main_image_host, maximized: dict) -> None:
    ui.label("Round Timeline").classes("text-sm font-medium text-gray-300 mt-2")
    ui.label("Click a round to maximize its render above.").classes(
        "text-[11px] text-gray-500"
    )

    with ui.scroll_area().classes("w-full").style("max-height: 280px"):
        with ui.row().classes("gap-2 flex-wrap"):
            for r in node.rounds:
                _round_thumbnail_card(r, main_image_host, maximized)


def _round_thumbnail_card(r: RoundInfo, main_image_host, maximized: dict) -> None:
    thumb = r.thumbnail or r.pre_route_thumbnail
    border_color = "border-green-500" if r.routed else "border-slate-600"

    def _on_click():
        if not thumb:
            return
        maximized["src"] = thumb
        maximized["label"] = (
            f"Round {r.index} — score {r.score:.2f}"
            if r.score is not None
            else f"Round {r.index}"
        )
        _render_main_image(main_image_host, maximized)

    card = ui.card().classes(
        f"p-1 w-[120px] {border_color} border bg-slate-900/80 cursor-pointer hover:border-blue-400"
    )
    card.on("click", lambda _e: _on_click())
    with card:
        ui.label(f"R{r.index}").classes("text-[10px] text-gray-400 font-mono")

        if thumb:
            ui.image(thumb).classes(
                "w-full h-[60px] object-contain rounded bg-slate-950"
            )
        else:
            with ui.row().classes("w-full h-[60px] items-center justify-center bg-slate-950 rounded"):
                ui.icon("image", size="xs").classes("text-gray-700")

        ui.label(f"{r.score:.1f}").classes("text-[11px] text-green-400 font-mono text-center w-full")


def _status_color(status: str) -> str:
    colors = {
        "pending": "grey",
        "solving": "blue",
        "routing": "orange",
        "accepted": "green",
        "failed": "red",
        "routing_failed": "red",
        "composing": "amber",
        "done": "green",
    }
    return colors.get(status, "grey")
