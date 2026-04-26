"""Analysis page — hierarchical experiment statistics and progression."""

from __future__ import annotations

import json
from typing import Any

from nicegui import ui

from ..components.experiment_table import create_experiment_table
from ..components.score_chart import (
    build_leaf_timing_figure,
    build_score_figure,
    build_stage_figure,
    build_subscore_figure,
    build_timing_figure,
    build_timing_summary_figure,
)
from ..run_archive import list_runs, load_live_rounds, load_run_rounds
from ..state import get_state


def analysis_page() -> None:
    state = get_state()

    ui.label("Experiment Analysis").classes("text-2xl font-bold mb-2")
    ui.label(
        "Review hierarchical experiment runs: scores, timing, stage "
        "progression, and convergence. Per-leaf pinning and live "
        "renders are on the Monitor tab."
    ).classes("text-sm text-gray-400 mb-4")

    _experiment_data_panel(state)


def _experiment_data_panel(state) -> None:
    # Source of truth: ``.experiments/runs/<run_id>/`` archives plus the
    # currently-running ``.experiments/experiments.jsonl``. The "live"
    # pseudo-run is selectable so users can analyze a run as it progresses;
    # it gets a synthetic id "__live__" until autoexperiment finishes and
    # archives it.
    LIVE_ID = "__live__"

    def _live_run_summary() -> dict[str, Any] | None:
        rounds = load_live_rounds(state.experiments_dir)
        if not rounds:
            return None
        best = max((r.get("score", 0) or 0 for r in rounds), default=0)
        return {
            "id": LIVE_ID,
            "label": f"Live run — {len(rounds)}r, best={float(best):.1f}",
            "rounds": rounds,
        }

    def _build_options() -> dict[str, str]:
        opts: dict[str, str] = {}
        live = _live_run_summary()
        if live is not None:
            opts[LIVE_ID] = live["label"]
        for run in list_runs(state.experiments_dir):
            opts[run.run_id] = (
                f"{run.name} — {run.completed_rounds}r, best={run.best_score:.1f}"
            )
        return opts

    options = _build_options()
    if not options:
        ui.label(
            "No experiments found. Run a hierarchical experiment to populate "
            "this page."
        ).classes("text-gray-500 italic")
        return

    selected_exp = {"id": next(iter(options))}
    content = ui.column().classes("w-full")

    def _resolve_rounds(exp_id: str) -> list[dict[str, Any]]:
        if exp_id == LIVE_ID:
            return load_live_rounds(state.experiments_dir)
        return load_run_rounds(state.experiments_dir, exp_id)

    def _load_experiment(exp_id: str) -> None:
        selected_exp["id"] = exp_id
        content.clear()
        rounds = _resolve_rounds(exp_id)

        if not rounds:
            with content:
                ui.label("No round data for this experiment.").classes(
                    "text-gray-500 italic"
                )
            return

        with content:
            _summary_cards(rounds)

            with ui.tabs().classes("w-full") as tabs:
                scores_tab = ui.tab("Scores", icon="show_chart")
                stages_tab = ui.tab("Stages", icon="timeline")
                timing_tab = ui.tab("Timing", icon="schedule")
                scheduling_tab = ui.tab("Scheduling", icon="alt_route")
                table_tab = ui.tab("All Rounds", icon="table_chart")
                convergence_tab = ui.tab("Convergence", icon="trending_up")
                export_tab = ui.tab("Export", icon="download")

            with ui.tab_panels(tabs, value=scores_tab).classes("w-full"):
                with ui.tab_panel(scores_tab):
                    fig = build_score_figure(rounds, "Hierarchical Score vs Round")
                    ui.plotly(fig).classes("w-full h-96")

                    ui.separator()
                    fig2 = build_subscore_figure(
                        rounds, "Leaf / Parent / Top-Level Progress"
                    )
                    ui.plotly(fig2).classes("w-full h-80")

                with ui.tab_panel(stages_tab):
                    fig_stage = build_stage_figure(rounds, "Pipeline Stage Timeline")
                    ui.plotly(fig_stage).classes("w-full h-80")
                    _stage_summary(rounds)

                with ui.tab_panel(timing_tab):
                    fig_timing = build_timing_figure(rounds, "Round Timing Breakdown")
                    ui.plotly(fig_timing).classes("w-full h-96")

                    ui.separator()

                    fig_leaf_timing = build_leaf_timing_figure(
                        rounds, "Leaf Pipeline Timing Breakdown"
                    )
                    ui.plotly(fig_leaf_timing).classes("w-full h-80")

                    ui.separator()

                    fig_timing_summary = build_timing_summary_figure(
                        rounds, "Timing Summary"
                    )
                    ui.plotly(fig_timing_summary).classes("w-full h-[28rem]")

                    ui.separator()
                    _timing_summary_panel(rounds)

                with ui.tab_panel(scheduling_tab):
                    _scheduling_summary_panel(rounds)

                with ui.tab_panel(table_tab):
                    create_experiment_table(rounds)

                with ui.tab_panel(convergence_tab):
                    _convergence_panel(rounds)

                with ui.tab_panel(export_tab):
                    _export_panel(rounds, exp_id)

    with ui.row().classes("w-full items-center gap-4 mb-4"):
        exp_select = ui.select(
            options=options,
            value=selected_exp["id"],
            label="Select Experiment",
            on_change=lambda e: _load_experiment(e.value),
        ).classes("w-96")

        def _refresh() -> None:
            new_options = _build_options()
            exp_select.options = new_options
            if selected_exp["id"] not in new_options and new_options:
                selected_exp["id"] = next(iter(new_options))
                exp_select.value = selected_exp["id"]
            exp_select.update()
            if selected_exp["id"]:
                _load_experiment(selected_exp["id"])

        ui.button("Refresh", icon="refresh", on_click=_refresh).props("flat")

    def _auto_refresh() -> None:
        new_options = _build_options()
        exp_select.options = new_options
        if selected_exp["id"] not in new_options and new_options:
            selected_exp["id"] = next(iter(new_options))
            exp_select.value = selected_exp["id"]
        exp_select.update()

        # Only auto-reload contents when viewing the live run; archived runs
        # don't change.
        if selected_exp["id"] == LIVE_ID:
            _load_experiment(LIVE_ID)

    ui.timer(10.0, _auto_refresh)

    if selected_exp["id"]:
        _load_experiment(selected_exp["id"])


def _summary_cards(rounds: list[dict[str, Any]]) -> None:
    best_round = max(rounds, key=lambda r: _as_float(r.get("score", 0)))
    total_kept = sum(1 for r in rounds if r.get("kept"))
    avg_score = (
        sum(_as_float(r.get("score", 0)) for r in rounds) / len(rounds)
        if rounds
        else 0.0
    )
    total_duration = sum(_as_float(r.get("duration_s", 0)) for r in rounds)

    best_leaf_accept = 0.0
    top_ready_count = 0
    for r in rounds:
        leaf_total = _as_int(r.get("leaf_total", 0))
        leaf_accepted = _as_int(r.get("leaf_accepted", 0))
        if leaf_total > 0:
            best_leaf_accept = max(best_leaf_accept, leaf_accepted / leaf_total)
        if r.get("parent_routed"):
            top_ready_count += 1

    with ui.row().classes("w-full gap-4 mb-4"):
        _stat_card("Rounds", str(len(rounds)))
        _stat_card("Best Score", f"{_as_float(best_round.get('score', 0)):.2f}")
        _stat_card("Avg Score", f"{avg_score:.2f}")
        _stat_card("Kept", f"{total_kept} ({(total_kept / len(rounds)):.0%})")
        _stat_card("Best Leaf Acceptance", f"{best_leaf_accept:.0%}")
        _stat_card("Parent Routed", f"{top_ready_count}/{len(rounds)}")
        _stat_card("Total Time", f"{total_duration / 60:.0f}m")


def _stage_summary(rounds: list[dict[str, Any]]) -> None:
    stage_counts: dict[str, int] = {}
    for r in rounds:
        stage = str(r.get("latest_stage", r.get("stage", "done")) or "done")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    ui.label("Stage Summary").classes("text-lg font-bold mt-4 mb-2")
    if not stage_counts:
        ui.label("No stage data available").classes("text-gray-500 italic")
        return

    with ui.row().classes("gap-3 flex-wrap"):
        for stage, count in sorted(stage_counts.items()):
            color = _stage_badge_color(stage)
            ui.badge(f"{stage}: {count}", color=color)


def _timing_summary_panel(rounds: list[dict[str, Any]]) -> None:
    ui.label("Timing Summary").classes("text-lg font-bold mt-4 mb-2")

    if not rounds:
        ui.label("No timing data available").classes("text-gray-500 italic")
        return

    def _timing_value(round_data: dict[str, Any], key: str) -> float:
        timing = round_data.get("timing_breakdown", {})
        if not isinstance(timing, dict):
            timing = {}
        return _as_float(timing.get(key, 0.0))

    round_count = max(1, len(rounds))
    avg_solve = (
        sum(_timing_value(r, "solve_subcircuits_total") for r in rounds) / round_count
    )
    avg_compose = (
        sum(_timing_value(r, "compose_subcircuits_total") for r in rounds) / round_count
    )
    avg_parent_route = (
        sum(_timing_value(r, "parent_route_total") for r in rounds) / round_count
    )
    avg_score = sum(_timing_value(r, "score_round_total") for r in rounds) / round_count
    avg_render = (
        sum(
            _timing_value(r, "pre_route_render_diagnostics_s")
            + _timing_value(r, "routed_render_diagnostics_s")
            for r in rounds
        )
        / round_count
    )
    avg_leaf_total = sum(_timing_value(r, "leaf_total_s") for r in rounds) / round_count
    avg_round_total = (
        sum(
            _timing_value(r, "round_total") or _as_float(r.get("duration_s", 0.0))
            for r in rounds
        )
        / round_count
    )

    with ui.grid(columns=4).classes("w-full gap-4 mb-4"):
        _stat_card("Avg Solve", f"{avg_solve:.2f}s")
        _stat_card("Avg Compose", f"{avg_compose:.2f}s")
        _stat_card("Avg Parent Route", f"{avg_parent_route:.2f}s")
        _stat_card("Avg Score", f"{avg_score:.2f}s")
        _stat_card("Avg Render", f"{avg_render:.2f}s")
        _stat_card("Avg Leaf Total", f"{avg_leaf_total:.2f}s")
        _stat_card("Avg Round Total", f"{avg_round_total:.2f}s")

    render_heavy_rounds = sorted(
        rounds,
        key=lambda r: (
            _timing_value(r, "pre_route_render_diagnostics_s")
            + _timing_value(r, "routed_render_diagnostics_s")
        ),
        reverse=True,
    )[:5]

    ui.label("Most Render-Heavy Rounds").classes("text-md font-bold mt-3 mb-2")
    if not render_heavy_rounds:
        ui.label("No render timing data available").classes("text-gray-500 italic")
        return

    with ui.column().classes("w-full gap-2"):
        for round_data in render_heavy_rounds:
            round_num = _as_int(round_data.get("round_num", 0))
            render_total = _timing_value(
                round_data, "pre_route_render_diagnostics_s"
            ) + _timing_value(round_data, "routed_render_diagnostics_s")
            solve_total = _timing_value(round_data, "solve_subcircuits_total")
            round_total = _timing_value(round_data, "round_total") or _as_float(
                round_data.get("duration_s", 0.0)
            )
            ui.label(
                f"Round {round_num}: render={render_total:.2f}s | "
                f"solve={solve_total:.2f}s | total={round_total:.2f}s"
            ).classes("text-sm text-gray-300 font-mono")


def _scheduling_summary_panel(rounds: list[dict[str, Any]]) -> None:
    ui.label("Leaf Scheduling + Long-Pole Summary").classes(
        "text-lg font-bold mt-4 mb-2"
    )

    if not rounds:
        ui.label("No scheduling data available").classes("text-gray-500 italic")
        return

    def _leaf_timing_summary(round_data: dict[str, Any]) -> dict[str, Any]:
        summary = round_data.get("leaf_timing_summary", {})
        if not isinstance(summary, dict):
            summary = {}
        return summary

    def _scheduled_leafs(round_data: dict[str, Any]) -> list[dict[str, Any]]:
        summary = _leaf_timing_summary(round_data)
        rows = summary.get("scheduled_leafs", [])
        if not isinstance(rows, list):
            rows = []
        return [row for row in rows if isinstance(row, dict)]

    def _long_poles(round_data: dict[str, Any]) -> list[dict[str, Any]]:
        summary = _leaf_timing_summary(round_data)
        rows = summary.get("long_pole_leafs", [])
        if not isinstance(rows, list):
            rows = []
        return [row for row in rows if isinstance(row, dict)]

    rounds_with_scheduling = [
        r
        for r in rounds
        if _leaf_timing_summary(r) or _scheduled_leafs(r) or _long_poles(r)
    ]
    if not rounds_with_scheduling:
        ui.label(
            "No persisted scheduling or long-pole metadata found in these rounds."
        ).classes("text-gray-500 italic")
        return

    latest_round = max(
        rounds_with_scheduling,
        key=lambda r: _as_int(r.get("round_num", 0)),
    )
    latest_summary = _leaf_timing_summary(latest_round)
    latest_scheduled = _scheduled_leafs(latest_round)
    latest_long_poles = _long_poles(latest_round)

    with ui.grid(columns=4).classes("w-full gap-4 mb-4"):
        _stat_card(
            "Latest Leaf Count",
            str(_as_int(latest_summary.get("leaf_count", 0))),
        )
        _stat_card(
            "Latest Imbalance",
            f"{_as_float(latest_summary.get('imbalance_ratio', 0.0)):.2f}",
        )
        _stat_card(
            "Latest Max Leaf",
            f"{_as_float(latest_summary.get('max_leaf_time_s', 0.0)):.2f}s",
        )
        _stat_card(
            "Latest Total Leaf Time",
            f"{_as_float(latest_summary.get('total_leaf_time_s', 0.0)):.2f}s",
        )

    ui.label("Latest Recommended Order").classes("text-md font-bold mt-3 mb-2")
    if latest_scheduled:
        with ui.column().classes("w-full gap-2 mb-4"):
            for item in latest_scheduled[:8]:
                position = _as_int(item.get("scheduled_position", 0))
                name = str(
                    item.get("sheet_name", item.get("scheduled_selector", "")) or ""
                )
                score = _as_float(item.get("scheduling_score", 0.0))
                freerouting_s = _as_float(item.get("freerouting_s", 0.0))
                leaf_total_s = _as_float(item.get("leaf_total_s", 0.0))
                failed_round_count = _as_int(item.get("failed_round_count", 0))
                trivial = bool(item.get("historically_trivial_candidate", False))
                ui.label(
                    f"{position}. {name} | score={score:.2f} | "
                    f"leaf={leaf_total_s:.2f}s | freerouting={freerouting_s:.2f}s | "
                    f"failed_rounds={failed_round_count} | trivial={'yes' if trivial else 'no'}"
                ).classes("text-sm text-gray-300 font-mono")
    else:
        ui.label("No scheduled leaf ordering persisted for the latest round.").classes(
            "text-gray-500 italic mb-4"
        )

    ui.label("Latest Long-Pole Leafs").classes("text-md font-bold mt-3 mb-2")
    if latest_long_poles:
        with ui.column().classes("w-full gap-2 mb-4"):
            for item in latest_long_poles[:5]:
                name = str(item.get("sheet_name", "") or "")
                leaf_total_s = _as_float(item.get("leaf_total_s", 0.0))
                route_total_s = _as_float(item.get("route_total_s", 0.0))
                freerouting_s = _as_float(item.get("freerouting_s", 0.0))
                internal_net_count = _as_int(item.get("internal_net_count", 0))
                ui.label(
                    f"{name}: leaf={leaf_total_s:.2f}s | route={route_total_s:.2f}s | "
                    f"freerouting={freerouting_s:.2f}s | internal_nets={internal_net_count}"
                ).classes("text-sm text-gray-300 font-mono")
    else:
        ui.label("No long-pole leafs recorded for the latest round.").classes(
            "text-gray-500 italic mb-4"
        )

    ui.label("Scheduling Trend by Round").classes("text-md font-bold mt-3 mb-2")
    with ui.column().classes("w-full gap-2"):
        for round_data in sorted(
            rounds_with_scheduling,
            key=lambda r: _as_int(r.get("round_num", 0)),
            reverse=True,
        )[:8]:
            round_num = _as_int(round_data.get("round_num", 0))
            summary = _leaf_timing_summary(round_data)
            scheduled = _scheduled_leafs(round_data)
            long_poles = _long_poles(round_data)
            top_scheduled = (
                ", ".join(
                    str(
                        item.get("sheet_name", item.get("scheduled_selector", "")) or ""
                    )
                    for item in scheduled[:3]
                )
                or "none"
            )
            top_long_poles = (
                ", ".join(
                    str(item.get("sheet_name", "") or "") for item in long_poles[:3]
                )
                or "none"
            )
            ui.label(
                f"Round {round_num}: imbalance={_as_float(summary.get('imbalance_ratio', 0.0)):.2f} | "
                f"max_leaf={_as_float(summary.get('max_leaf_time_s', 0.0)):.2f}s | "
                f"scheduled={top_scheduled} | long_poles={top_long_poles}"
            ).classes("text-sm text-gray-300 font-mono")


def _convergence_panel(rounds: list[dict[str, Any]]) -> None:
    if len(rounds) < 2:
        ui.label("Need more rounds for convergence analysis").classes(
            "text-gray-500 italic"
        )
        return

    sorted_rounds = sorted(rounds, key=lambda r: _as_int(r.get("round_num", 0)))
    scores = [_as_float(r.get("score", 0)) for r in sorted_rounds]

    running_best: list[float] = []
    best = float("-inf")
    for score in scores:
        if score > best:
            best = score
        running_best.append(best)

    last_improvement = 0
    for i in range(1, len(running_best)):
        if running_best[i] > running_best[i - 1]:
            last_improvement = i

    useful_rounds = last_improvement + 1
    wasted = len(rounds) - useful_rounds
    initial = scores[0] if scores else 0.0
    improvement = best - initial if scores else 0.0

    top_ready = sum(1 for r in rounds if r.get("parent_routed"))
    parent_ok = sum(1 for r in rounds if r.get("parent_composed"))

    ui.label("Convergence Summary").classes("text-lg font-bold mb-2")

    with ui.grid(columns=4).classes("w-full gap-4 mb-4"):
        _stat_card("Last Improvement", f"Round {last_improvement + 1}")
        _stat_card("Useful Rounds", f"{useful_rounds}/{len(rounds)}")
        _stat_card("Wasted Tail", str(wasted))
        _stat_card("Total Improvement", f"+{improvement:.2f}")

    ui.label("Hierarchy Outcome Rates").classes("text-md font-bold mt-3")
    with ui.row().classes("gap-4 flex-wrap"):
        ui.badge(
            f"Parent composed: {parent_ok}/{len(rounds)} ({parent_ok / len(rounds):.0%})",
            color="orange",
        )
        ui.badge(
            f"Parent routed: {top_ready}/{len(rounds)} ({top_ready / len(rounds):.0%})",
            color="green",
        )

    stage_counts: dict[str, int] = {}
    for r in rounds:
        stage = str(r.get("latest_stage", r.get("stage", "done")) or "done")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    ui.label("Stage Distribution").classes("text-md font-bold mt-3")
    with ui.row().classes("gap-3 flex-wrap"):
        for stage, count in sorted(stage_counts.items()):
            ui.badge(f"{stage}: {count}", color=_stage_badge_color(stage))


def _export_panel(rounds: list[dict[str, Any]], exp_id: int) -> None:
    ui.label("Export experiment data").classes("text-md font-bold mb-2")

    def _download_csv() -> None:
        if not rounds:
            ui.notify("No data to export", type="warning")
            return

        keys = [
            "round_num",
            "score",
            "mode",
            "kept",
            "leaf_total",
            "leaf_accepted",
            "parent_composed",
            "parent_routed",
            "accepted_trace_count",
            "accepted_via_count",
            "latest_stage",
            "details",
        ]
        lines = [",".join(keys)]
        for r in rounds:
            lines.append(",".join(_csv_escape(r.get(k, "")) for k in keys))
        csv_data = "\n".join(lines)
        ui.download(csv_data.encode(), f"experiment_{exp_id}.csv")

    def _download_json() -> None:
        if not rounds:
            ui.notify("No data to export", type="warning")
            return
        data = json.dumps(rounds, indent=2)
        ui.download(data.encode(), f"experiment_{exp_id}.json")

    with ui.row().classes("gap-2"):
        ui.button("Download CSV", icon="download", on_click=_download_csv)
        ui.button("Download JSON", icon="download", on_click=_download_json)



def _stat_card(label: str, value: str) -> None:
    with ui.card().classes("p-3 flex-1 text-center"):
        ui.label(label).classes("text-xs text-gray-400")
        ui.label(value).classes("text-xl font-bold")


def _stage_badge_color(stage: str) -> str:
    stage = stage.lower()
    if stage == "solve_leafs":
        return "blue"
    if stage == "compose_parent":
        return "orange"
    if stage in {"route_parent", "done", "complete"}:
        return "green"
    if stage == "startup":
        return "gray"
    return "gray"


def _csv_escape(value: Any) -> str:
    text = str(value)
    if any(ch in text for ch in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


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
