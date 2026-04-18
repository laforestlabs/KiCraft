"""Score-over-time charts for hierarchical experiment runs."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from nicegui import ui
from plotly.subplots import make_subplots


def create_score_chart(
    rounds: list[dict[str, Any]],
    title: str = "Hierarchical Score vs Round",
) -> ui.plotly:
    """Build a Plotly chart showing hierarchical score over rounds."""
    fig = build_score_figure(rounds, title)
    return ui.plotly(fig).classes("w-full h-96")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mode_color(mode: str) -> str:
    colors = {
        "hierarchical": "#4dabf7",
        "leaf": "#339af0",
        "compose": "#f59f00",
        "top": "#51cf66",
        "baseline": "#845ef7",
    }
    return colors.get(mode.lower(), "#adb5bd")


def _stage_color(stage: str) -> str:
    colors = {
        "solve_leafs": "#339af0",
        "compose_parent": "#f59f00",
        "route_parent": "#51cf66",
        "done": "#94d82d",
        "startup": "#868e96",
        "complete": "#20c997",
    }
    return colors.get(stage.lower(), "#adb5bd")


def build_score_figure(
    rounds: list[dict[str, Any]],
    title: str = "Hierarchical Score vs Round",
) -> go.Figure:
    """Build score chart for hierarchical experiment rounds."""
    fig = go.Figure()

    if not rounds:
        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_title="Round",
            yaxis_title="Score",
        )
        return fig

    mode_data: dict[str, dict[str, list[Any]]] = {}
    for r in rounds:
        mode = str(r.get("mode", "hierarchical"))
        if mode not in mode_data:
            mode_data[mode] = {
                "x": [],
                "y": [],
                "text": [],
                "marker_symbol": [],
                "marker_color": [],
            }

        round_num = _as_int(r.get("round_num", 0))
        score = _as_float(r.get("score", 0.0))
        kept = bool(r.get("kept", False))
        details = str(r.get("details", "") or "")
        leaf_total = _as_int(r.get("leaf_total", 0))
        leaf_accepted = _as_int(r.get("leaf_accepted", 0))
        parent_routed = bool(r.get("parent_routed", False))

        mode_data[mode]["x"].append(round_num)
        mode_data[mode]["y"].append(score)
        mode_data[mode]["marker_symbol"].append("circle" if kept else "x")
        mode_data[mode]["marker_color"].append(_mode_color(mode))
        mode_data[mode]["text"].append(
            f"R{round_num} | score={score:.2f}"
            f"<br>mode={mode}"
            f"<br>leafs={leaf_accepted}/{leaf_total}"
            f"<br>parent_routed={'yes' if parent_routed else 'no'}"
            + (f"<br>{details}" if details else "")
        )

    for mode, data in mode_data.items():
        fig.add_trace(
            go.Scatter(
                x=data["x"],
                y=data["y"],
                mode="markers",
                name=mode.upper(),
                text=data["text"],
                marker=dict(
                    color=data["marker_color"],
                    size=9,
                    symbol=data["marker_symbol"],
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    best_x: list[int] = []
    best_y: list[float] = []
    running_best = float("-inf")
    for r in sorted(rounds, key=lambda row: _as_int(row.get("round_num", 0))):
        score = _as_float(r.get("score", 0.0))
        if score > running_best:
            running_best = score
        best_x.append(_as_int(r.get("round_num", 0)))
        best_y.append(running_best)

    fig.add_trace(
        go.Scatter(
            x=best_x,
            y=best_y,
            mode="lines",
            name="BEST",
            line=dict(color="#51cf66", width=2),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title="Round",
        yaxis_title="Score",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=50, r=20, t=50, b=40),
        hovermode="closest",
    )
    return fig


def build_subscore_figure(
    rounds: list[dict[str, Any]],
    title: str = "Hierarchical Progress Breakdown",
) -> go.Figure:
    """Build multi-line chart for hierarchical progress metrics."""
    fig = go.Figure()

    if not rounds:
        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_title="Round",
            yaxis_title="Metric",
        )
        return fig

    sorted_rounds = sorted(rounds, key=lambda r: _as_int(r.get("round_num", 0)))

    x_rounds = [_as_int(r.get("round_num", 0)) for r in sorted_rounds]
    y_scores = [_as_float(r.get("score", 0.0)) for r in sorted_rounds]
    y_leaf_accept = []
    y_top_ready = []
    y_traces = []
    y_vias = []

    max_traces = max(
        [_as_int(r.get("accepted_trace_count", 0)) for r in sorted_rounds] or [1]
    )
    max_vias = max(
        [_as_int(r.get("accepted_via_count", 0)) for r in sorted_rounds] or [1]
    )

    for r in sorted_rounds:
        leaf_total = _as_int(r.get("leaf_total", 0))
        leaf_accepted = _as_int(r.get("leaf_accepted", 0))
        leaf_pct = (leaf_accepted / leaf_total * 100.0) if leaf_total > 0 else 0.0
        y_leaf_accept.append(leaf_pct)

        y_top_ready.append(100.0 if r.get("parent_routed", False) else 0.0)

        traces = _as_int(r.get("accepted_trace_count", 0))
        vias = _as_int(r.get("accepted_via_count", 0))
        y_traces.append((traces / max_traces * 100.0) if max_traces > 0 else 0.0)
        y_vias.append((vias / max_vias * 100.0) if max_vias > 0 else 0.0)

    metrics = [
        ("Score", y_scores, "#4dabf7"),
        ("Leaf Acceptance %", y_leaf_accept, "#51cf66"),
        ("Parent Routed", y_top_ready, "#f59f00"),
        ("Accepted Traces (norm)", y_traces, "#e599f7"),
        ("Accepted Vias (norm)", y_vias, "#ffd43b"),
    ]

    for name, y_values, color in metrics:
        fig.add_trace(
            go.Scatter(
                x=x_rounds,
                y=y_values,
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=5),
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title="Round",
        yaxis_title="Metric (0-100 or score)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def build_stage_figure(
    rounds: list[dict[str, Any]],
    title: str = "Pipeline Stage Timeline",
) -> go.Figure:
    """Build a stage-colored scatter timeline for hierarchical rounds."""
    fig = go.Figure()

    if not rounds:
        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_title="Round",
            yaxis_title="Stage",
        )
        return fig

    stage_order = {
        "startup": 0,
        "solve_leafs": 1,
        "compose_parent": 2,
        "route_parent": 3,
        "done": 4,
        "complete": 5,
    }

    x_vals: list[int] = []
    y_vals: list[int] = []
    colors: list[str] = []
    texts: list[str] = []

    for r in sorted(rounds, key=lambda row: _as_int(row.get("round_num", 0))):
        round_num = _as_int(r.get("round_num", 0))
        stage = str(r.get("latest_stage", r.get("stage", "done")) or "done")
        score = _as_float(r.get("score", 0.0))
        leaf_total = _as_int(r.get("leaf_total", 0))
        leaf_accepted = _as_int(r.get("leaf_accepted", 0))

        x_vals.append(round_num)
        y_vals.append(stage_order.get(stage, 4))
        colors.append(_stage_color(stage))
        texts.append(
            f"R{round_num}"
            f"<br>stage={stage}"
            f"<br>score={score:.2f}"
            f"<br>leafs={leaf_accepted}/{leaf_total}"
        )

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers+lines",
            name="Stage",
            text=texts,
            marker=dict(color=colors, size=10, line=dict(color="white", width=1)),
            line=dict(color="#495057", width=1, dash="dot"),
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title="Round",
        yaxis_title="Stage",
        yaxis=dict(
            tickmode="array",
            tickvals=list(stage_order.values()),
            ticktext=[
                label for label, _ in sorted(stage_order.items(), key=lambda x: x[1])
            ],
        ),
        margin=dict(l=80, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


def _timing_value(
    round_data: dict[str, Any],
    key: str,
    default: float = 0.0,
) -> float:
    timing = round_data.get("timing_breakdown", {})
    if not isinstance(timing, dict):
        timing = {}
    return _as_float(timing.get(key, default), default)


def build_timing_figure(
    rounds: list[dict[str, Any]],
    title: str = "Round Timing Breakdown",
) -> go.Figure:
    """Build a stacked bar chart for per-round timing breakdown."""
    fig = go.Figure()

    if not rounds:
        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_title="Round",
            yaxis_title="Seconds",
            barmode="stack",
        )
        return fig

    sorted_rounds = sorted(rounds, key=lambda r: _as_int(r.get("round_num", 0)))
    x_rounds = [_as_int(r.get("round_num", 0)) for r in sorted_rounds]

    timing_series = [
        ("solve_subcircuits_total", "#4dabf7"),
        ("compose_subcircuits_total", "#f59f00"),
        ("parent_route_total", "#51cf66"),
        ("score_round_total", "#e599f7"),
    ]

    for key, color in timing_series:
        y_values = [_timing_value(r, key) for r in sorted_rounds]
        fig.add_trace(
            go.Bar(
                x=x_rounds,
                y=y_values,
                name=key,
                marker_color=color,
                hovertemplate=(f"Round %{{x}}<br>{key}=%{{y:.3f}}s<extra></extra>"),
            )
        )

    round_totals = [
        _timing_value(r, "round_total", _as_float(r.get("duration_s", 0.0)))
        for r in sorted_rounds
    ]
    fig.add_trace(
        go.Scatter(
            x=x_rounds,
            y=round_totals,
            mode="lines+markers",
            name="round_total",
            line=dict(color="#ffffff", width=2),
            marker=dict(size=6),
            hovertemplate="Round %{x}<br>round_total=%{y:.3f}s<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title="Round",
        yaxis_title="Seconds",
        barmode="stack",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def build_leaf_timing_figure(
    rounds: list[dict[str, Any]],
    title: str = "Leaf Pipeline Timing Breakdown",
) -> go.Figure:
    """Build a multi-line chart for leaf-stage timing metrics."""
    fig = go.Figure()

    if not rounds:
        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_title="Round",
            yaxis_title="Seconds",
        )
        return fig

    sorted_rounds = sorted(rounds, key=lambda r: _as_int(r.get("round_num", 0)))
    x_rounds = [_as_int(r.get("round_num", 0)) for r in sorted_rounds]

    metrics = [
        ("placement_solve_s", "#4dabf7"),
        ("freerouting_s", "#51cf66"),
        ("pre_route_render_diagnostics_s", "#f59f00"),
        ("routed_render_diagnostics_s", "#ff922b"),
        ("persist_solution_s", "#e599f7"),
        ("leaf_total_s", "#ffffff"),
    ]

    for key, color in metrics:
        fig.add_trace(
            go.Scatter(
                x=x_rounds,
                y=[_timing_value(r, key) for r in sorted_rounds],
                mode="lines+markers",
                name=key,
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate=(f"Round %{{x}}<br>{key}=%{{y:.3f}}s<extra></extra>"),
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title="Round",
        yaxis_title="Seconds",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


def build_timing_summary_figure(
    rounds: list[dict[str, Any]],
    title: str = "Timing Summary",
) -> go.Figure:
    """Build a compact timing summary figure with totals and render share."""
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}]],
        subplot_titles=("Average Time Share", "Average Stage Time"),
    )

    if not rounds:
        fig.update_layout(
            title=title,
            template="plotly_dark",
            margin=dict(l=40, r=20, t=60, b=40),
        )
        return fig

    sorted_rounds = sorted(rounds, key=lambda r: _as_int(r.get("round_num", 0)))

    avg_solve = sum(
        _timing_value(r, "solve_subcircuits_total") for r in sorted_rounds
    ) / max(1, len(sorted_rounds))
    avg_compose = sum(
        _timing_value(r, "compose_subcircuits_total") for r in sorted_rounds
    ) / max(1, len(sorted_rounds))
    avg_parent = sum(
        _timing_value(r, "parent_route_total") for r in sorted_rounds
    ) / max(1, len(sorted_rounds))
    avg_score = sum(_timing_value(r, "score_round_total") for r in sorted_rounds) / max(
        1, len(sorted_rounds)
    )
    avg_render = sum(
        _timing_value(r, "pre_route_render_diagnostics_s")
        + _timing_value(r, "routed_render_diagnostics_s")
        for r in sorted_rounds
    ) / max(1, len(sorted_rounds))

    fig.add_trace(
        go.Pie(
            labels=[
                "solve_subcircuits_total",
                "compose_subcircuits_total",
                "parent_route_total",
                "score_round_total",
                "leaf_render_diagnostics",
            ],
            values=[avg_solve, avg_compose, avg_parent, avg_score, avg_render],
            hole=0.45,
            textinfo="label+percent",
            marker=dict(colors=["#4dabf7", "#f59f00", "#51cf66", "#e599f7", "#ff922b"]),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=[
                "solve",
                "compose",
                "parent route",
                "score",
                "render",
            ],
            y=[avg_solve, avg_compose, avg_parent, avg_score, avg_render],
            marker_color=["#4dabf7", "#f59f00", "#51cf66", "#e599f7", "#ff922b"],
            hovertemplate="%{x}<br>%{y:.3f}s<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_yaxes(title_text="Seconds", row=1, col=2)
    return fig
