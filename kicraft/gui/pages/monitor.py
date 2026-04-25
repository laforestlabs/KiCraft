"""Monitor page -- live hierarchical experiment dashboard with pipeline visualization."""

from __future__ import annotations

import time

from nicegui import ui

from ..components.node_detail import node_detail_panel
from ..components.parent_score_chart import (
    parent_score_chart,
    pick_best_round,
)
from ..components.pipeline_graph import (
    NodeStatus,
    PipelineState,
    _path_with_mtime,
    gather_pipeline_state,
    pipeline_graph,
)
from ..state import get_state


def _format_time(seconds: float) -> str:
    s = max(0, int(seconds or 0))
    return f"{s // 60}m{s % 60:02d}s"


def monitor_page():
    state = get_state()
    runner = state.runner

    selected_node: dict = {"value": None}
    live_rounds: list[dict] = []
    prev_phase = {"value": "idle"}
    prev_graph_fingerprint = {"value": ""}
    prev_detail_node_id = {"value": ""}
    latest_pipeline_state: dict = {"value": None}
    experiment_terminated = {"value": False}

    # Live timing: start_monotonic anchors to subprocess elapsed_s on each
    # status read, then a 1s ticker extrapolates to keep the label moving
    # between status writes (which only happen at round/stage transitions).
    run_timing: dict = {"start_monotonic": None}

    # Parent-round selection: None = auto-track best; int = user pinned a round.
    selected_parent_round: dict = {"value": None, "user_pinned": False}
    prev_parent_score_fp: dict = {"value": ""}

    with ui.row().classes("w-full items-center gap-3 mb-2 px-2"):
        start_btn = ui.button("Start", icon="play_arrow", color="green").props("dense")
        stop_btn = ui.button("Stop", icon="stop", color="red").props("dense")
        stop_btn.set_visibility(False)
        force_kill_btn = ui.button("Kill", icon="dangerous", color="deep-orange").props("dense")
        force_kill_btn.set_visibility(False)
        stopping_spinner = ui.spinner(size="sm")
        stopping_spinner.set_visibility(False)

        ui.separator().props("vertical")

        status_badge = ui.badge("IDLE", color="gray").classes("text-sm")
        phase_label = ui.label("--").classes("text-sm font-medium")
        ui.separator().props("vertical")
        timing_label = ui.label("--").classes("text-xs text-gray-400")
        ui.space()
        progress_label = ui.label("").classes("text-xs text-gray-400")

    parent_chart_container = ui.column().classes("w-full mb-2")
    with parent_chart_container:
        with ui.row().classes("w-full items-center gap-3 px-2"):
            parent_round_label = ui.label("Parent scores will appear here once rounds complete.").classes(
                "text-xs text-gray-400"
            )
            ui.space()
            reset_selection_btn = ui.button(
                "Auto-track best", icon="auto_awesome"
            ).props("flat dense")
            reset_selection_btn.set_visibility(False)
        parent_plot_host = ui.column().classes("w-full")

    with ui.row().classes("w-full gap-4 items-start min-h-[500px]"):
        graph_container = ui.column().classes("flex-1 min-w-[400px]")
        detail_container = ui.column().classes("flex-1 min-w-[350px] max-w-[600px]")

    def _on_node_select(node: NodeStatus | None) -> None:
        selected_node["value"] = node
        prev_detail_node_id["value"] = ""
        prev_graph_fingerprint["value"] = ""
        if latest_pipeline_state["value"] is not None:
            _rebuild_graph(latest_pipeline_state["value"])
        _rebuild_detail()

    def _on_parent_round_select(round_num: int) -> None:
        if round_num <= 0:
            return
        selected_parent_round["value"] = round_num
        selected_parent_round["user_pinned"] = True
        reset_selection_btn.set_visibility(True)
        prev_parent_score_fp["value"] = ""
        prev_graph_fingerprint["value"] = ""
        prev_detail_node_id["value"] = ""
        _update_status()

    def _reset_round_selection() -> None:
        selected_parent_round["user_pinned"] = False
        selected_parent_round["value"] = pick_best_round(live_rounds)
        reset_selection_btn.set_visibility(False)
        prev_parent_score_fp["value"] = ""
        prev_graph_fingerprint["value"] = ""
        prev_detail_node_id["value"] = ""
        _update_status()

    reset_selection_btn.on_click(_reset_round_selection)

    def _rebuild_parent_chart() -> None:
        parent_plot_host.clear()
        with parent_plot_host:
            if not live_rounds:
                return
            parent_score_chart(
                live_rounds,
                selected_round=selected_parent_round["value"],
                on_select=_on_parent_round_select,
            )
        best = pick_best_round(live_rounds)
        sel = selected_parent_round["value"]
        if sel is None:
            parent_round_label.set_text(
                f"{len(live_rounds)} round(s) complete — best so far: R{best}" if best else ""
            )
        elif selected_parent_round["user_pinned"]:
            suffix = f" (best is R{best})" if best and best != sel else ""
            parent_round_label.set_text(
                f"Viewing R{sel}{suffix}"
            )
        else:
            parent_round_label.set_text(
                f"Viewing best so far: R{sel}"
            )

    def _rebuild_graph(pipeline_state: PipelineState) -> None:
        graph_container.clear()
        with graph_container:
            pipeline_graph(
                pipeline_state,
                on_node_select=_on_node_select,
                selected_node_id=selected_node["value"].node_id if selected_node["value"] else None,
            )

    def _rebuild_detail() -> None:
        detail_container.clear()
        node = selected_node["value"]
        if node is None:
            with detail_container:
                with ui.column().classes("w-full items-center justify-center h-[300px]"):
                    ui.icon("touch_app", size="xl").classes("text-gray-600")
                    ui.label("Click a node to view details").classes("text-gray-500")
            return
        with detail_container:
            node_detail_panel(node)

    def _update_status():
        run_status = runner.read_status()
        phase = run_status.get("phase", "idle")

        # Suppress transient "error" from JSON parse races while process is alive
        if phase == "error" and runner.is_running:
            phase = prev_phase["value"] if prev_phase["value"] != "idle" else "running"
            run_status["phase"] = phase

        # Detect stopping: stop.now exists and process still alive
        stop_file = state.experiments_dir / "stop.now"
        if stop_file.exists() and runner.is_running:
            phase = "stopping"

        pipeline_state = gather_pipeline_state(
            state.experiments_dir,
            run_status,
            project_dir=state.project_root,
            project_name=state.project_name,
            selected_round=selected_parent_round["value"],
        )
        latest_pipeline_state["value"] = pipeline_state

        badge_colors = {
            "idle": "gray",
            "running": "blue",
            "stopping": "orange",
            "done": "green",
            "error": "red",
        }
        status_badge.set_text(phase.upper())
        status_badge._props["color"] = badge_colors.get(phase, "gray")
        status_badge.update()

        stage = run_status.get("stage") or run_status.get("pipeline_phase") or phase
        phase_label.set_text(stage.replace("_", " ").title())

        try:
            backend_elapsed = float(run_status.get("elapsed_s", 0))
        except (TypeError, ValueError):
            backend_elapsed = 0.0

        if phase == "running":
            # Re-anchor the local clock to the subprocess's elapsed_s so the
            # 1s ticker can extrapolate forward without drifting.
            run_timing["start_monotonic"] = time.monotonic() - backend_elapsed
        else:
            run_timing["start_monotonic"] = None
            timing_label.set_text(
                f"Elapsed: {_format_time(backend_elapsed)} | ETA: --"
            )

        try:
            rnd = int(run_status.get("round", 0))
        except (TypeError, ValueError):
            rnd = 0
        try:
            total = int(run_status.get("total_rounds", 0))
        except (TypeError, ValueError):
            total = 0
        accepted_count = sum(1 for l in pipeline_state.leaves if l.status == "accepted")
        total_leaves = len(pipeline_state.leaves)
        progress_label.set_text(
            f"Round {rnd}/{total} | Leaves {accepted_count}/{total_leaves}"
        )

        is_running = phase == "running" or runner.is_running
        is_stopping = phase == "stopping"
        start_btn.set_visibility(not is_running and not is_stopping)
        stop_btn.set_visibility(is_running and not is_stopping)
        force_kill_btn.set_visibility(is_stopping)
        stopping_spinner.set_visibility(is_stopping)

        # Full re-read every poll. experiments.jsonl is unconditionally
        # truncated by autoexperiment at startup, so reading the whole file
        # always reflects the current run. The previous incremental
        # `read_latest_rounds(since_round=N)` optimization saved ~50KB of I/O
        # but silently kept stale rounds from prior runs cached in
        # `live_rounds` whenever a new run started outside the GUI (CLI).
        fresh_rounds = runner.read_latest_rounds(0)

        # Detect new run via identity: round count or first-round seed
        # changed. On change, drop the in-memory cache and reset selection.
        cached_signature = (
            len(live_rounds),
            live_rounds[0].get("seed") if live_rounds else None,
        )
        fresh_signature = (
            len(fresh_rounds),
            fresh_rounds[0].get("seed") if fresh_rounds else None,
        )
        if cached_signature != fresh_signature:
            live_rounds.clear()
            live_rounds.extend(fresh_rounds)
            if not selected_parent_round["user_pinned"]:
                selected_parent_round["value"] = pick_best_round(live_rounds)
            # Force chart redraw next pass.
            prev_parent_score_fp["value"] = ""

        # Redraw the parent chart only when its contents changed (new rounds,
        # selection flip, or pin state change).
        parent_fp = (
            f"{len(live_rounds)}|"
            f"{live_rounds[0].get('seed') if live_rounds else None}|"
            f"{selected_parent_round['value']}|{selected_parent_round['user_pinned']}"
        )
        if parent_fp != prev_parent_score_fp["value"]:
            prev_parent_score_fp["value"] = parent_fp
            _rebuild_parent_chart()

        if (
            prev_phase["value"] in ("running", "stopping")
            and phase in ("done", "idle", "error")
            and not experiment_terminated["value"]
        ):
            # Guard: if phase is "error" but subprocess is still alive,
            # this is a transient read failure -- skip the transition.
            if phase == "error" and runner.is_running:
                pass  # transient JSON parse error; do not mark as failed
            else:
                experiment_terminated["value"] = True
                if phase == "error":
                    ui.notify("Experiment failed!", type="negative")
                else:
                    ui.notify("Experiment finished!", type="positive")
        prev_phase["value"] = phase

        sel_id = selected_node["value"].node_id if selected_node["value"] else ""
        graph_fp = pipeline_state.graph_fingerprint() + "|sel:" + sel_id
        if graph_fp != prev_graph_fingerprint["value"]:
            prev_graph_fingerprint["value"] = graph_fp
            _rebuild_graph(pipeline_state)

        if selected_node["value"]:
            sel_id = selected_node["value"].node_id
            updated = None
            if sel_id == "__root__":
                updated = NodeStatus(
                    name=pipeline_state.root_name,
                    node_id="__root__",
                    is_leaf=False,
                    status=pipeline_state.root_status,
                    best_render=pipeline_state.root_render,
                )
            else:
                for leaf in pipeline_state.leaves:
                    if leaf.node_id == sel_id:
                        updated = leaf
                        break
            if updated:
                detail_fp = (
                    f"{updated.node_id}|{updated.status}|{updated.score}|"
                    f"{_path_with_mtime(updated.best_render)}|"
                    f"{updated.total_rounds_run}|{updated.traces}|{updated.vias}"
                )
                if detail_fp != prev_detail_node_id["value"]:
                    prev_detail_node_id["value"] = detail_fp
                    selected_node["value"] = updated
                    _rebuild_detail()

    ui.timer(2.0, _update_status)

    def _update_timing() -> None:
        """Tick the elapsed/ETA label every second between status writes."""
        start_mono = run_timing["start_monotonic"]
        if start_mono is None:
            return
        live_elapsed = max(0.0, time.monotonic() - start_mono)

        # ETA from completed-round durations. Need >=1 finished round for a
        # meaningful estimate; otherwise show "--".
        completed = [
            float(r.get("duration_s", 0)) for r in live_rounds
            if r.get("duration_s")
        ]
        try:
            total_rounds = int(
                latest_pipeline_state["value"].total_rounds
                if latest_pipeline_state["value"]
                else 0
            )
        except (AttributeError, TypeError, ValueError):
            total_rounds = 0

        if completed and total_rounds > 0:
            avg_dur = sum(completed) / len(completed)
            current_round_elapsed = max(0.0, live_elapsed - sum(completed))
            remaining_after_current = max(0, total_rounds - len(completed) - 1)
            eta = (
                max(0.0, avg_dur - current_round_elapsed)
                + remaining_after_current * avg_dur
            )
            eta_text = _format_time(eta)
        else:
            eta_text = "--"

        timing_label.set_text(
            f"Elapsed: {_format_time(live_elapsed)} | ETA: {eta_text}"
        )

    ui.timer(1.0, _update_timing)

    async def _start():
        try:
            pid = runner.start(
                pcb_file=state.strategy["pcb_file"],
                rounds=state.strategy["rounds"],
                workers=state.strategy["workers"],
                seed=state.strategy.get("seed"),
                param_ranges=state.get_control_ranges(),
                score_weights=state.score_weights,
                extra_config={
                    "schematic_file": state.strategy["schematic_file"],
                    "parent": state.strategy.get("parent", "/"),
                    "only": state.strategy.get("only", []),
                    "leaf_rounds": state.strategy.get("leaf_rounds", 2),
                    "render_png": state.toggles.get("render_png", True),
                    "save_round_details": state.toggles.get("save_round_details", True),
                    "placement_config": {
                        **state.placement_config,
                        "freerouting_hide_window": bool(
                            state.toggles.get("freerouting_hide_window", True)
                        ),
                    },
                },
            )
            ui.notify(f"Started experiment (PID {pid})", type="positive")
            live_rounds.clear()
            experiment_terminated["value"] = False
        except Exception as e:
            ui.notify(f"Failed to start: {e}", type="negative")

    def _stop():
        runner.stop()
        ui.notify("Stop requested -- will stop at next safe checkpoint", type="info")

    def _force_kill():
        runner.kill()
        experiment_terminated["value"] = True
        ui.notify("Force killed experiment", type="warning")

    start_btn.on_click(_start)
    stop_btn.on_click(_stop)
    force_kill_btn.on_click(_force_kill)

    # Initial render -- don't wait 2s for the first timer tick
    _update_status()
    if selected_node["value"] is None:
        _rebuild_detail()
