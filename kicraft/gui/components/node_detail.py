"""Node detail panel -- shows full PCB render, score plot, and round timeline."""

from __future__ import annotations

import plotly.graph_objects as go
from nicegui import ui

from .pipeline_graph import NodeStatus, RoundInfo


def node_detail_panel(
    node: NodeStatus,
) -> None:
    """Render the detail panel for a selected pipeline node."""
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
        main_image_host = ui.column().classes("w-full")
        _render_main_image(main_image_host, maximized)
        if node.is_leaf and node.rounds:
            _score_plot(node)
            _round_timeline(node, main_image_host, maximized)


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
