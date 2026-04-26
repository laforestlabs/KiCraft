"""Parent-round score plot for the Monitor tab.

Renders a compact line+marker chart of parent scores per autoexperiment
round. Clicking a point fires `on_select(round_num)` so the monitor can
filter leaf renders below to only that round.
"""

from __future__ import annotations

from typing import Any, Callable

import plotly.graph_objects as go
from nicegui import ui


def _is_leaves_only_run(rounds: list[dict[str, Any]]) -> bool:
    """Detect a --leaves-only run.

    Heuristic: every completed round failed parent_routed AND has a
    leaf_score_summary attached. The score field in this case is the
    not_routed_penalty fallback (20.0), which is meaningless to plot.
    """
    if not rounds:
        return False
    for r in rounds:
        if r.get("parent_routed"):
            return False
        summary = r.get("leaf_score_summary") or {}
        if not isinstance(summary, dict) or "avg_score" not in summary:
            return False
    return True


def build_parent_round_figure(
    rounds: list[dict[str, Any]],
    selected_round: int | None = None,
    title: str = "Parent Score vs Round",
) -> go.Figure:
    """Build the parent-round plotly figure.

    `rounds` is a list of dicts as emitted to experiments.jsonl (one per
    completed parent round). We read `round_num`, `score`, `kept`, and a
    few summary fields for the hover tooltip. In --leaves-only mode the
    chart switches to plotting average leaf score per round, since the
    parent score is uniformly the 20.0 not_routed_penalty fallback.
    """
    fig = go.Figure()

    if not rounds:
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=200,
            margin=dict(l=40, r=20, t=40, b=30),
            xaxis_title="Experiment round",
            yaxis_title="Parent score",
        )
        return fig

    leaves_only = _is_leaves_only_run(rounds)
    if leaves_only and title.startswith("Parent Score"):
        title = "Avg Leaf Score vs Round (leaves-only)"

    sorted_rounds = sorted(rounds, key=lambda r: _as_int(r.get("round_num", 0)))

    xs: list[int] = []
    ys: list[float] = []
    marker_sizes: list[int] = []
    marker_colors: list[str] = []
    hover_texts: list[str] = []

    for r in sorted_rounds:
        round_num = _as_int(r.get("round_num", 0))
        score = _as_float(r.get("score", 0.0))
        kept = bool(r.get("kept", False))
        leaf_total = _as_int(r.get("leaf_total", 0))
        leaf_accepted = _as_int(r.get("leaf_accepted", 0))
        parent_routed = bool(r.get("parent_routed", False))
        leaf_summary = r.get("leaf_score_summary") or {}
        leaf_avg = _as_float(leaf_summary.get("avg_score", 0.0)) if isinstance(leaf_summary, dict) else 0.0
        leaf_min = _as_float(leaf_summary.get("min_score", 0.0)) if isinstance(leaf_summary, dict) else 0.0
        leaf_max = _as_float(leaf_summary.get("max_score", 0.0)) if isinstance(leaf_summary, dict) else 0.0

        xs.append(round_num)
        # In leaves-only mode, plot avg leaf score instead of the
        # uniformly-20.0 parent_score fallback.
        ys.append(leaf_avg if leaves_only else score)
        is_selected = selected_round is not None and round_num == selected_round
        marker_sizes.append(16 if is_selected else 10)
        marker_colors.append("#ffd43b" if is_selected else ("#51cf66" if kept else "#4dabf7"))
        if leaves_only:
            hover_texts.append(
                f"R{round_num} | avg leaf score={leaf_avg:.2f}"
                f"<br>min={leaf_min:.2f}  max={leaf_max:.2f}"
                f"<br>leafs accepted={leaf_accepted}/{leaf_total}"
                f"<br>(parent compose skipped)"
            )
        else:
            hover_texts.append(
                f"R{round_num} | score={score:.2f}"
                f"<br>kept={'yes' if kept else 'no'}"
                f"<br>leafs={leaf_accepted}/{leaf_total}"
                f"<br>parent_routed={'yes' if parent_routed else 'no'}"
            )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name="Avg leaf score" if leaves_only else "Parent score",
            line=dict(color="#4dabf7", width=2),
            marker=dict(
                color=marker_colors,
                size=marker_sizes,
                line=dict(width=1, color="white"),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    best_y: list[float] = []
    running_best = float("-inf")
    for score in ys:
        if score > running_best:
            running_best = score
        best_y.append(running_best)

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=best_y,
            mode="lines",
            name="Best so far",
            line=dict(color="#51cf66", width=1.5, dash="dot"),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=220,
        margin=dict(l=40, r=20, t=40, b=30),
        xaxis_title="Experiment round",
        yaxis_title="Avg leaf score" if leaves_only else "Parent score",
        showlegend=False,
    )
    fig.update_xaxes(dtick=1)
    return fig


def pick_best_round(rounds: list[dict[str, Any]]) -> int | None:
    """Return the round_num of the best-scoring completed round, or None.

    Uses leaf_score_summary.avg_score in leaves-only mode (where the
    parent score is a meaningless 20.0 fallback for every round).
    """
    if not rounds:
        return None
    if _is_leaves_only_run(rounds):
        scorer = lambda r: _as_float(  # noqa: E731
            (r.get("leaf_score_summary") or {}).get("avg_score", float("-inf"))
        )
    else:
        scorer = lambda r: _as_float(r.get("score", float("-inf")))  # noqa: E731
    best = max(rounds, key=scorer)
    rn = _as_int(best.get("round_num", 0))
    return rn if rn > 0 else None


def parent_score_chart(
    rounds: list[dict[str, Any]],
    selected_round: int | None,
    on_select: Callable[[int], None],
):
    """Render the plot as a nicegui element and wire a click handler.

    Returns the ui.plotly element so the caller can call .update() on it
    when new rounds arrive.
    """
    fig = build_parent_round_figure(rounds, selected_round=selected_round)
    plot = ui.plotly(fig).classes("w-full")

    def _handle_click(e) -> None:
        # nicegui wraps plotly click events in e.args with points list
        try:
            points = e.args.get("points", []) if hasattr(e, "args") else []
        except AttributeError:
            points = []
        if not points:
            return
        p = points[0]
        try:
            round_num = int(p.get("x"))
        except (TypeError, ValueError):
            return
        on_select(round_num)

    plot.on("plotly_click", _handle_click)
    return plot


def _as_int(val: Any, default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _as_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default
