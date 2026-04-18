"""Monitor page — live hierarchical experiment dashboard."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, cast

from nicegui import ui

from ..components.score_chart import create_score_chart
from ..state import get_state

_STAGE_LABELS = {
    "startup": "Startup",
    "solve_leafs": "Solve Leafs",
    "compose_parent": "Compose Parent",
    "route_parent": "Route Parent",
    "score_round": "Score Round",
    "done": "Done",
    "complete": "Complete",
    "failed": "Failed",
}


def monitor_page():
    state = get_state()
    runner = state.runner
    db = state.db
    if db is None:
        raise RuntimeError("Database is not initialized")

    ui.label("Experiment Manager Monitor").classes("text-2xl font-bold mb-4")
    ui.label(
        "Focused live visibility into what the experiment manager is doing now: "
        "run state, worker activity, accepted artifacts, recent events, and "
        "top-level progression."
    ).classes("text-sm text-gray-400 mb-4")

    # ── Controls ──
    with ui.row().classes("w-full items-center gap-4 mb-4"):
        start_btn = ui.button("Start Experiment", icon="play_arrow", color="green")
        stop_btn = ui.button("Stop", icon="stop", color="red")
        stop_btn.set_visibility(False)
        force_kill_btn = ui.button("Force Kill", icon="dangerous", color="deep-orange")
        force_kill_btn.set_visibility(False)
        stopping_spinner = ui.spinner(size="sm")
        stopping_spinner.set_visibility(False)
        stopping_label = ui.label("Stopping…")
        stopping_label.set_visibility(False)

    # ── Top status cards ──
    with ui.row().classes("w-full gap-4 mb-4"):
        with ui.card().classes("p-3 flex-1"):
            ui.label("Run Status").classes("text-xs text-gray-400")
            status_badge = ui.badge("IDLE", color="gray").classes("text-lg")

        with ui.card().classes("p-3 flex-1"):
            ui.label("Current Phase").classes("text-xs text-gray-400")
            phase_label = ui.label("—").classes("text-lg font-bold")

        with ui.card().classes("p-3 flex-1"):
            ui.label("Round Progress").classes("text-xs text-gray-400")
            progress_label = ui.label("0 / 0")
            progress_bar = ui.linear_progress(value=0).classes("w-full")

        with ui.card().classes("p-3 flex-1"):
            ui.label("Timing").classes("text-xs text-gray-400")
            timing_label = ui.label("Elapsed: — | ETA: —")

    # ── Hierarchical pipeline cards ──
    with ui.row().classes("w-full gap-4 mb-4"):
        with ui.card().classes("p-3 flex-1"):
            ui.label("Leaf Progress").classes("text-xs text-gray-400")
            leaves_label = ui.label("—")
            leaves_bar = ui.linear_progress(value=0).classes("w-full mt-2")

        with ui.card().classes("p-3 flex-1"):
            ui.label("Accepted Artifacts").classes("text-xs text-gray-400")
            artifacts_label = ui.label("—").classes("text-lg font-bold text-green-400")

        with ui.card().classes("p-3 flex-1"):
            ui.label("Current Target").classes("text-xs text-gray-400")
            current_node_label = ui.label("—")

        with ui.card().classes("p-3 flex-1"):
            ui.label("Parent Routing").classes("text-xs text-gray-400")
            top_level_label = ui.label("—")

    with ui.row().classes("w-full gap-4 mb-4"):
        with ui.card().classes("p-3 flex-1"):
            ui.label("Best Score").classes("text-xs text-gray-400")
            best_score_label = ui.label("—").classes(
                "text-2xl font-bold text-green-400"
            )

        with ui.card().classes("p-3 flex-1"):
            ui.label("Latest Event").classes("text-xs text-gray-400")
            latest_label = ui.label("—")

        with ui.card().classes("p-3 flex-1"):
            ui.label("Worker Activity").classes("text-xs text-gray-400")
            workers_label = ui.label("—")

        with ui.card().classes("p-3 flex-1"):
            ui.label("Run Health").classes("text-xs text-gray-400")
            health_label = ui.label("—")

    with ui.row().classes("w-full gap-4 mb-4"):
        with ui.card().classes("p-3 flex-1"):
            ui.label("Experiment History").classes("text-xs text-gray-400")
            history_summary_label = ui.label("—")

        with ui.card().classes("p-3 flex-1"):
            ui.label("Recent Best").classes("text-xs text-gray-400")
            recent_best_label = ui.label("—")

        with ui.card().classes("p-3 flex-1"):
            ui.label("Completed Runs").classes("text-xs text-gray-400")
            completed_runs_label = ui.label("—")

        with ui.card().classes("p-3 flex-1"):
            ui.label("Last Finished").classes("text-xs text-gray-400")
            last_finished_label = ui.label("—")

    # ── Score chart ──
    chart_container = ui.column().classes("w-full")
    with chart_container:
        create_score_chart([], "Hierarchical Score")

    # ── Pipeline detail panels ──
    with ui.row().classes("w-full gap-4 items-start"):
        with ui.card().classes("p-3 flex-1"):
            ui.label("Recent Pipeline Events").classes("text-lg font-bold mb-2")
            events_container = ui.column().classes("w-full gap-2")

        with ui.card().classes("p-3 flex-1"):
            ui.label("Accepted Leaf Artifacts").classes("text-lg font-bold mb-2")
            artifacts_container = ui.column().classes("w-full gap-2")

    with ui.row().classes("w-full gap-4 items-start mt-4"):
        with ui.card().classes("p-3 flex-1"):
            ui.label("Top-Level / Parent Outputs").classes("text-lg font-bold mb-2")
            top_outputs_container = ui.column().classes("w-full gap-2")

    # ── Board preview ──
    with ui.expansion(
        "Visual Progression / Latest Preview", icon="image", value=True
    ).classes("w-full mt-4"):
        board_container = ui.column().classes("w-full items-center")
        with board_container:
            ui.label("Start an experiment to see hierarchical previews").classes(
                "text-gray-500 italic"
            )

    # ── State tracking ──
    live_rounds: list[dict] = []
    last_round_seen = {"value": 0}
    prev_phase = {"value": "idle"}

    def _format_time(seconds: float) -> str:
        s = max(0, int(seconds or 0))
        return f"{s // 60}m{s % 60:02d}s"

    def _safe_read_json(path: Path) -> dict[str, Any] | list[Any] | None:
        try:
            if not path.exists():
                return None
            with open(path) as f:
                return cast(dict[str, Any] | list[Any], json.load(f))
        except (json.JSONDecodeError, OSError):
            return None

    def _read_jsonl_tail(path: Path, limit: int = 12) -> list[dict]:
        if not path.exists():
            return []
        rows: list[dict] = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return []
        return rows[-limit:]

    def _find_preview_candidates() -> list[Path]:
        exp = state.experiments_dir
        candidates = [
            exp / "best_preview.png",
            exp / "progress.gif",
            exp / "frames" / "frame_latest.png",
            exp / "frames" / f"frame_{last_round_seen['value']:04d}.png",
            exp / "best" / "best_preview.png",
            exp / "hierarchical_pipeline" / "parent_stamped.png",
            exp / "hierarchical_pipeline" / "parent_routed.png",
        ]

        auto_root = exp / "hierarchical_autoexperiment"
        if auto_root.exists():
            for round_dir in sorted(auto_root.glob("round_*"), reverse=True):
                candidates.extend(
                    [
                        round_dir / "parent_stamped.png",
                        round_dir / "parent_routed.png",
                    ]
                )

        return [p for p in candidates if p.exists()]

    def _render_events(events: list[dict], status: dict) -> None:
        events_container.clear()

        current_stage = (
            status.get("stage")
            or status.get("pipeline_phase")
            or (status.get("hierarchy", {}) or {}).get("current_stage")
            or status.get("phase")
            or "idle"
        )
        current_action = status.get("current_action") or (
            (status.get("hierarchy", {}) or {}).get("current_action")
        )
        current_command = status.get("current_command") or (
            (status.get("hierarchy", {}) or {}).get("current_command")
        )
        current_leaf = status.get("current_leaf") or (
            (status.get("hierarchy", {}) or {}).get("current_leaf")
        )
        current_parent = status.get("current_parent") or (
            (status.get("hierarchy", {}) or {}).get("current_parent")
        )
        round_num = status.get("round")
        latest_marker = status.get("latest_marker") or status.get("latest_event")

        synthesized_events: list[dict[str, Any]] = []
        if latest_marker or current_action:
            synthesized_events.append(
                {
                    "phase": current_stage,
                    "title": latest_marker or current_action or "Live status update",
                    "detail": " | ".join(
                        part
                        for part in [
                            f"round {round_num}" if round_num else "",
                            f"leaf={current_leaf}" if current_leaf else "",
                            f"parent={current_parent}" if current_parent else "",
                            current_action or "",
                        ]
                        if part
                    ),
                    "command": current_command or "",
                    "is_live": True,
                }
            )

        merged_events = list(events[-9:]) + synthesized_events
        if not merged_events:
            with events_container:
                ui.label("No pipeline events yet").classes("text-gray-500 italic")
            return

        with events_container:
            for event in reversed(merged_events[-10:]):
                phase = (
                    event.get("phase")
                    or event.get("stage")
                    or event.get("mode")
                    or event.get("event", "—")
                )
                title = event.get("title") or event.get("latest_marker") or phase
                detail = event.get("detail") or event.get("message") or ""
                command = event.get("command") or ""
                score = event.get("score")
                kept = event.get("kept")
                is_live = bool(event.get("is_live"))
                with ui.card().classes("w-full p-2 bg-slate-800/40"):
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.badge(
                            str(_STAGE_LABELS.get(str(phase), str(phase))).upper(),
                            color="purple" if is_live else "blue",
                        )
                        ui.label(str(title)).classes("font-medium")
                        ui.space()
                        if score is not None:
                            ui.label(f"{float(score):.2f}").classes(
                                "text-green-400 font-mono"
                            )
                        if kept is True:
                            ui.badge("KEPT", color="green")
                        elif kept is False and "kept" in event:
                            ui.badge("DISCARDED", color="red")
                        if is_live:
                            ui.badge("LIVE", color="orange")
                    if detail:
                        ui.label(str(detail)).classes("text-xs text-gray-400")
                    if command:
                        ui.label(str(command)).classes(
                            "text-[11px] text-amber-300 font-mono break-all"
                        )

    def _render_artifacts() -> None:
        artifacts_container.clear()
        sub_root = state.experiments_dir / "subcircuits"
        if not sub_root.exists():
            with artifacts_container:
                ui.label("No subcircuit artifacts found yet").classes(
                    "text-gray-500 italic"
                )
            return

        accepted: list[dict] = []
        for artifact_dir in sorted(sub_root.iterdir()):
            if not artifact_dir.is_dir():
                continue
            solved = _safe_read_json(artifact_dir / "solved_layout.json")
            meta = _safe_read_json(artifact_dir / "metadata.json")
            if not isinstance(solved, dict):
                continue
            validation = solved.get("validation", {})
            meta_dict = meta if isinstance(meta, dict) else {}
            if isinstance(validation, dict) and validation.get("accepted") is True:
                accepted.append(
                    {
                        "dir": artifact_dir.name,
                        "sheet_name": solved.get("sheet_name")
                        or meta_dict.get("sheet_name")
                        or artifact_dir.name,
                        "instance_path": solved.get("instance_path")
                        or meta_dict.get("instance_path")
                        or "",
                        "traces": len(solved.get("traces", [])),
                        "vias": len(solved.get("vias", [])),
                    }
                )

        if not accepted:
            with artifacts_container:
                ui.label("No accepted routed leaf artifacts yet").classes(
                    "text-gray-500 italic"
                )
            return

        with artifacts_container:
            for item in accepted[:12]:
                with ui.card().classes("w-full p-2 bg-slate-800/40"):
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.badge("LEAF", color="green")
                        ui.label(item["sheet_name"]).classes("font-medium")
                        ui.space()
                        ui.label(f"T{item['traces']}").classes("text-xs text-cyan-300")
                        ui.label(f"V{item['vias']}").classes("text-xs text-amber-300")
                    if item["instance_path"]:
                        ui.label(item["instance_path"]).classes(
                            "text-xs text-gray-400 font-mono"
                        )

    def _render_top_outputs(status: dict) -> None:
        top_outputs_container.clear()
        hp = state.experiments_dir / "hierarchical_pipeline"
        preview_paths = status.get("preview_paths", {})
        if not isinstance(preview_paths, dict):
            preview_paths = {}

        parent_artifact_dir = preview_paths.get("parent_artifact_dir")
        parent_artifact = Path(parent_artifact_dir) if parent_artifact_dir else None

        candidates = [
            hp / "parent_composition.json",
            hp / "parent_pipeline.json",
            state.experiments_dir / "report.html",
        ]
        if parent_artifact is not None:
            candidates.extend(
                [
                    parent_artifact / "parent_pre_freerouting.kicad_pcb",
                    parent_artifact / "parent_routed.kicad_pcb",
                    parent_artifact / "debug.json",
                    parent_artifact / "metadata.json",
                ]
            )

        existing = [p for p in candidates if p.exists()]
        board_source_paths = []
        for key in [
            "leaf_round_illegal_board",
            "leaf_round_pre_route_board",
            "leaf_round_routed_board",
            "parent_stamped_board",
            "parent_routed_board",
        ]:
            value = preview_paths.get(key)
            if value:
                board_source_paths.append((key, Path(str(value))))

        if not existing and not board_source_paths:
            with top_outputs_container:
                ui.label("No parent/top-level outputs yet").classes(
                    "text-gray-500 italic"
                )
            return

        with top_outputs_container:
            if board_source_paths:
                with ui.card().classes("w-full p-3 bg-slate-900/50"):
                    ui.label("KiCad board source paths").classes("font-medium")
                    ui.label(
                        "These are the actual `.kicad_pcb` files backing the current live previews."
                    ).classes("text-xs text-gray-400 mb-2")
                    for key, path in board_source_paths:
                        with ui.column().classes("w-full gap-0"):
                            ui.label(key).classes("text-xs text-cyan-300 font-mono")
                            try:
                                display_path = str(path.relative_to(state.project_root))
                            except ValueError:
                                display_path = str(path)
                            ui.label(display_path).classes(
                                "text-xs text-gray-400 font-mono break-all"
                            )

            for path in existing:
                with ui.card().classes("w-full p-2 bg-slate-800/40"):
                    ui.label(path.name).classes("font-medium")
                    ui.label(str(path.relative_to(state.project_root))).classes(
                        "text-xs text-gray-400 font-mono"
                    )

    def _render_status_json(status: dict) -> None:
        return

    def _update_board_preview(status: dict) -> None:
        board_container.clear()

        exp = state.experiments_dir
        hierarchy = status.get("hierarchy", {})
        if not isinstance(hierarchy, dict):
            hierarchy = {}

        preview_paths = status.get("preview_paths", {})
        if not isinstance(preview_paths, dict):
            preview_paths = {}
        hierarchy_preview_paths = hierarchy.get("preview_paths", {})
        if not isinstance(hierarchy_preview_paths, dict):
            hierarchy_preview_paths = {}

        merged_preview_paths = {
            **hierarchy_preview_paths,
            **preview_paths,
        }

        current_stage = (
            hierarchy.get("current_stage")
            or status.get("stage")
            or status.get("pipeline_phase")
            or status.get("phase")
            or "idle"
        )
        current_action = status.get("current_action") or hierarchy.get("current_action")
        current_command = status.get("current_command") or hierarchy.get(
            "current_command"
        )

        def _existing_path(value: Any) -> Path | None:
            if not value:
                return None
            try:
                candidate = Path(str(value))
            except (TypeError, ValueError):
                return None
            return candidate if candidate.exists() else None

        def _display_path(value: Any) -> str:
            existing = _existing_path(value)
            if existing is None:
                return ""
            try:
                return str(existing.relative_to(state.project_root))
            except ValueError:
                return str(existing)

        leaf_preview_candidates = [
            exp / "best_preview.png",
            exp / "frames" / "frame_latest.png",
            exp / "frames" / f"frame_{last_round_seen['value']:04d}.png",
        ]
        fallback_leaf_preview = next(
            (p for p in leaf_preview_candidates if p.exists()), None
        )

        fallback_parent_candidates = [
            exp / "hierarchical_pipeline" / "parent_routed.png",
            exp / "hierarchical_pipeline" / "parent_stamped.png",
        ]
        auto_root = exp / "hierarchical_autoexperiment"
        if auto_root.exists():
            for round_dir in sorted(auto_root.glob("round_*"), reverse=True):
                fallback_parent_candidates.extend(
                    [
                        round_dir / "parent_routed.png",
                        round_dir / "parent_stamped.png",
                    ]
                )

        fallback_parent_preview = next(
            (p for p in fallback_parent_candidates if p.exists()),
            None,
        )

        leaf_preview = _existing_path(merged_preview_paths.get("leaf_preview"))
        parent_stamped_preview = _existing_path(
            merged_preview_paths.get("parent_stamped_preview")
        )
        parent_routed_preview = _existing_path(
            merged_preview_paths.get("parent_routed_preview")
        )

        if parent_stamped_preview is None:
            parent_stamped_preview = next(
                (
                    p
                    for p in fallback_parent_candidates
                    if p.exists()
                    and p.name
                    in {
                        "parent_stamped.png",
                        "board.png",
                        "snapshot.png",
                    }
                ),
                None,
            )
        if parent_routed_preview is None:
            parent_routed_preview = next(
                (
                    p
                    for p in fallback_parent_candidates
                    if p.exists()
                    and p.name
                    in {
                        "parent_routed.png",
                        "board_routed.png",
                        "routed.png",
                    }
                ),
                None,
            )

        leaf_source = "status" if leaf_preview is not None else "fallback"
        if leaf_preview is None:
            leaf_preview = fallback_leaf_preview

        if current_stage == "route_parent":
            parent_preview = parent_routed_preview or parent_stamped_preview
            parent_label = (
                "Live routed parent preview"
                if parent_routed_preview is not None
                else "Live stamped parent preview"
            )
            parent_relevance = (
                "current-stage preview"
                if parent_routed_preview is not None
                else "using previous preview while current stage runs"
            )
        elif current_stage == "compose_parent":
            parent_preview = parent_stamped_preview or parent_routed_preview
            parent_label = (
                "Live stamped parent preview"
                if parent_stamped_preview is not None
                else "Live routed parent preview"
            )
            parent_relevance = (
                "current-stage preview"
                if parent_stamped_preview is not None
                else "using previous preview while current stage runs"
            )
        else:
            parent_preview = (
                parent_routed_preview
                or parent_stamped_preview
                or fallback_parent_preview
            )
            parent_label = (
                "Live routed parent preview"
                if parent_routed_preview is not None
                else "Live stamped parent preview"
            )
            parent_relevance = (
                "current-stage preview"
                if parent_preview is not None
                and parent_preview
                in {
                    parent_routed_preview,
                    parent_stamped_preview,
                }
                else "fallback preview"
            )

        parent_source = (
            "status"
            if parent_preview is not None
            and parent_preview in {parent_stamped_preview, parent_routed_preview}
            and (
                _existing_path(merged_preview_paths.get("parent_stamped_preview"))
                == parent_preview
                or _existing_path(merged_preview_paths.get("parent_routed_preview"))
                == parent_preview
            )
            else "fallback"
        )

        if leaf_preview is None and parent_preview is None:
            previews = _find_preview_candidates()
            if not previews:
                with board_container:
                    ui.label("No preview artifacts found yet").classes(
                        "text-gray-500 italic"
                    )
                    if current_action:
                        ui.label(f"Current action: {current_action}").classes(
                            "text-sm text-gray-400"
                        )
                    if current_command:
                        ui.label(f"Current command: {current_command}").classes(
                            "text-xs text-amber-300 font-mono break-all"
                        )
                return
            leaf_preview = previews[0]
            leaf_source = "fallback"

        leaf_artifact_dir = merged_preview_paths.get("leaf_artifact_dir")
        parent_artifact_dir = merged_preview_paths.get("parent_artifact_dir")
        if not parent_artifact_dir and parent_preview is not None:
            parent_artifact_dir = str(parent_preview.parent)

        leaf_round_illegal_board = merged_preview_paths.get("leaf_round_illegal_board")
        leaf_round_pre_route_board = merged_preview_paths.get(
            "leaf_round_pre_route_board"
        )
        leaf_round_routed_board = merged_preview_paths.get("leaf_round_routed_board")
        parent_stamped_board = merged_preview_paths.get("parent_stamped_board")
        parent_routed_board = merged_preview_paths.get("parent_routed_board")

        with board_container:
            with ui.row().classes("w-full gap-4 items-start"):
                with ui.card().classes("flex-1 p-3 bg-slate-900/60"):
                    ui.label("Live leaf / current best preview").classes(
                        "text-sm font-bold text-gray-200 mb-2"
                    )
                    if leaf_preview is not None:
                        ui.label(leaf_preview.name).classes(
                            "text-xs text-gray-400 mb-1 font-mono"
                        )
                        ui.label(
                            f"source={leaf_source} | relevance={'current-stage preview' if leaf_source == 'status' else 'fallback preview'}"
                        ).classes("text-[11px] text-cyan-300 mb-1")
                        if leaf_artifact_dir:
                            ui.label(str(leaf_artifact_dir)).classes(
                                "text-[11px] text-gray-500 mb-2 font-mono break-all"
                            )
                        if leaf_round_pre_route_board or leaf_round_routed_board:
                            with ui.column().classes("w-full gap-1 mb-2"):
                                ui.label("KiCad board sources").classes(
                                    "text-[11px] text-emerald-300 font-mono"
                                )
                                if leaf_round_illegal_board:
                                    ui.label(
                                        "illegal_pre_stamp="
                                        + _display_path(leaf_round_illegal_board)
                                    ).classes(
                                        "text-[11px] text-gray-400 font-mono break-all"
                                    )
                                if leaf_round_pre_route_board:
                                    ui.label(
                                        "pre_route="
                                        + _display_path(leaf_round_pre_route_board)
                                    ).classes(
                                        "text-[11px] text-gray-400 font-mono break-all"
                                    )
                                if leaf_round_routed_board:
                                    ui.label(
                                        "routed="
                                        + _display_path(leaf_round_routed_board)
                                    ).classes(
                                        "text-[11px] text-gray-400 font-mono break-all"
                                    )
                        ui.image(str(leaf_preview)).classes(
                            "w-full max-w-3xl max-h-[620px] object-contain rounded border border-slate-700 bg-slate-950"
                        )
                    else:
                        ui.label("No new leaf preview yet for this stage").classes(
                            "text-gray-500 italic"
                        )
                        if current_action:
                            ui.label(f"Current action: {current_action}").classes(
                                "text-sm text-gray-400 mt-2"
                            )
                        if current_command:
                            ui.label(f"Current command: {current_command}").classes(
                                "text-xs text-amber-300 font-mono break-all"
                            )

                with ui.card().classes("flex-1 p-3 bg-slate-900/60"):
                    ui.label(parent_label).classes(
                        "text-sm font-bold text-gray-200 mb-2"
                    )
                    if parent_preview is not None:
                        ui.label(parent_preview.name).classes(
                            "text-xs text-gray-400 mb-1 font-mono"
                        )
                        ui.label(
                            f"source={parent_source} | relevance={parent_relevance}"
                        ).classes("text-[11px] text-cyan-300 mb-1")
                        if parent_artifact_dir:
                            ui.label(str(parent_artifact_dir)).classes(
                                "text-[11px] text-gray-500 mb-2 font-mono break-all"
                            )
                        if parent_stamped_board or parent_routed_board:
                            with ui.column().classes("w-full gap-1 mb-2"):
                                ui.label("KiCad board sources").classes(
                                    "text-[11px] text-emerald-300 font-mono"
                                )
                                if parent_stamped_board:
                                    ui.label(
                                        "stamped=" + _display_path(parent_stamped_board)
                                    ).classes(
                                        "text-[11px] text-gray-400 font-mono break-all"
                                    )
                                if parent_routed_board:
                                    ui.label(
                                        "routed=" + _display_path(parent_routed_board)
                                    ).classes(
                                        "text-[11px] text-gray-400 font-mono break-all"
                                    )
                        ui.image(str(parent_preview)).classes(
                            "w-full max-w-3xl max-h-[620px] object-contain rounded border border-slate-700 bg-slate-950"
                        )
                    else:
                        ui.label("No new parent preview yet for this stage").classes(
                            "text-gray-500 italic"
                        )
                        ui.label(
                            "The pipeline may still be composing, stamping, or routing the parent."
                        ).classes("text-sm text-gray-400 mt-2")
                        if current_action:
                            ui.label(f"Current action: {current_action}").classes(
                                "text-sm text-gray-400 mt-2"
                            )
                        if current_command:
                            ui.label(f"Current command: {current_command}").classes(
                                "text-xs text-amber-300 font-mono break-all"
                            )

    def _extract_hierarchical_metrics(status: dict) -> tuple[int, int, int]:
        leaves = status.get("leaves", {}) if isinstance(status, dict) else {}
        hierarchy = status.get("hierarchy", {}) if isinstance(status, dict) else {}
        solved = int(leaves.get("solved", 0) or 0)
        total = int(leaves.get("total", 0) or 0)
        accepted = int(leaves.get("accepted", 0) or 0)

        if total == 0:
            total = int(hierarchy.get("leaf_total", 0) or 0)
        if total == 0:
            total = int(status.get("total_leaves", 0) or 0)

        if solved == 0:
            solved = int(hierarchy.get("leaf_accepted", 0) or 0)
        if solved == 0:
            solved = int(status.get("solved_leaves", 0) or 0)

        if accepted == 0:
            accepted = int(hierarchy.get("leaf_accepted", 0) or 0)
        if accepted == 0:
            accepted = int(status.get("accepted_artifacts", 0) or 0)

        return solved, total, accepted

    def _extract_leaf_scheduling(status: dict) -> tuple[list[dict], list[dict]]:
        hierarchy = status.get("hierarchy", {}) if isinstance(status, dict) else {}
        if not isinstance(hierarchy, dict):
            hierarchy = {}

        leaf_timing_summary = hierarchy.get("leaf_timing_summary", {})
        if not isinstance(leaf_timing_summary, dict):
            leaf_timing_summary = {}

        scheduled_leafs = leaf_timing_summary.get("scheduled_leafs", [])
        if not isinstance(scheduled_leafs, list):
            scheduled_leafs = []
        scheduled_leafs = [item for item in scheduled_leafs if isinstance(item, dict)]

        long_pole_leafs = leaf_timing_summary.get("long_pole_leafs", [])
        if not isinstance(long_pole_leafs, list):
            long_pole_leafs = []
        long_pole_leafs = [item for item in long_pole_leafs if isinstance(item, dict)]

        return scheduled_leafs, long_pole_leafs

    def _update_status():
        """Poll run_status.json and update UI."""
        status = runner.read_status()
        phase = status.get("phase", "idle")
        hierarchy = status.get("hierarchy", {})
        if not isinstance(hierarchy, dict):
            hierarchy = {}

        scheduled_leafs, long_pole_leafs = _extract_leaf_scheduling(status)

        experiments = db.get_experiments()
        total_runs = len(experiments)
        completed_runs = sum(
            1
            for exp in experiments
            if str(getattr(exp, "status", "") or "").lower() == "done"
        )
        running_runs = sum(
            1
            for exp in experiments
            if str(getattr(exp, "status", "") or "").lower() in {"running", "stopping"}
        )
        best_completed = max(
            (
                float(getattr(exp, "best_score", 0) or 0)
                for exp in experiments
                if str(getattr(exp, "status", "") or "").lower() == "done"
            ),
            default=0.0,
        )
        latest_finished = next(
            (
                exp
                for exp in experiments
                if str(getattr(exp, "status", "") or "").lower() == "done"
            ),
            None,
        )

        history_summary_label.set_text(
            f"{total_runs} total | {running_runs} active"
            if total_runs
            else "No runs recorded yet"
        )
        recent_best_label.set_text(f"{best_completed:.2f}" if best_completed else "—")
        completed_runs_label.set_text(str(completed_runs))
        if latest_finished and getattr(latest_finished, "created_at", None):
            last_finished_label.set_text(
                latest_finished.created_at.strftime("%Y-%m-%d %H:%M")
            )
        else:
            last_finished_label.set_text("—")

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

        current_stage = (
            hierarchy.get("current_stage")
            or status.get("stage")
            or status.get("pipeline_phase")
            or phase
        )

        if phase == "done":
            current_stage = "complete"
        elif phase == "error":
            current_stage = "failed"

        current_stage_label = _STAGE_LABELS.get(str(current_stage), str(current_stage))
        phase_label.set_text(current_stage_label)

        rnd = status.get("round", 0)
        total = status.get("total_rounds", 0)
        pct = status.get("progress_percent", 0)
        progress_label.set_text(f"{rnd} / {total}")
        progress_bar.set_value((pct or 0) / 100)

        elapsed = status.get("elapsed_s", 0)
        eta = status.get("eta_s", 0)
        timing_label.set_text(
            f"Elapsed: {_format_time(elapsed)} | ETA: {_format_time(eta)}"
        )

        solved_leaves, total_leaves, accepted_artifacts = _extract_hierarchical_metrics(
            status
        )
        leaves_label.set_text(
            f"{solved_leaves} / {total_leaves}" if total_leaves else f"{solved_leaves}"
        )
        leaves_bar.set_value((solved_leaves / total_leaves) if total_leaves else 0)
        artifacts_label.set_text(str(accepted_artifacts))

        current_leaf = (
            status.get("current_leaf") or hierarchy.get("current_leaf") or "—"
        )
        current_parent = (
            status.get("current_parent") or hierarchy.get("current_parent") or "—"
        )
        current_node = (
            status.get("current_node")
            or hierarchy.get("current_node")
            or status.get("current_leaf")
            or status.get("current_parent")
            or "—"
        )
        current_action = (
            status.get("current_action") or hierarchy.get("current_action") or "—"
        )

        current_node_label.set_text(
            f"stage={current_stage_label} | action={current_action} | "
            f"node={current_node} | leaf={current_leaf} | parent={current_parent}"
        )

        top_level = (
            status.get("top_level_status")
            or status.get("parent_status")
            or status.get("composition_status")
            or "—"
        )

        leaf_workers = hierarchy.get("leaf_workers", {})
        if not isinstance(leaf_workers, dict):
            leaf_workers = {}

        configured_workers = int(state.strategy.get("workers", 1) or 1)
        if configured_workers <= 0:
            configured_workers = max(1, mp.cpu_count() // 2)

        active_leaf_workers = int(leaf_workers.get("active", 0) or 0)
        total_leaf_workers = int(
            leaf_workers.get("total", configured_workers) or configured_workers
        )
        queued_leafs = int(leaf_workers.get("queued", 0) or 0)
        completed_leafs = int(
            leaf_workers.get("completed", solved_leaves) or solved_leaves
        )
        idle_leaf_workers = int(
            leaf_workers.get("idle", max(0, total_leaf_workers - active_leaf_workers))
            or 0
        )

        copper = hierarchy.get("copper_accounting", {})
        if not isinstance(copper, dict):
            copper = {}

        preserved_child_traces = int(copper.get("preserved_child_trace_count", 0) or 0)
        expected_child_traces = int(
            copper.get("expected_preserved_child_trace_count", 0) or 0
        )
        preserved_child_vias = int(copper.get("preserved_child_via_count", 0) or 0)
        expected_child_vias = int(
            copper.get("expected_preserved_child_via_count", 0) or 0
        )
        added_parent_traces = int(copper.get("added_parent_trace_count", 0) or 0)
        added_parent_vias = int(copper.get("added_parent_via_count", 0) or 0)

        if phase == "done":
            top_level = "ready" if top_level == "routing_top_level" else top_level

        top_scheduled = (
            ", ".join(
                str(item.get("sheet_name", item.get("scheduled_selector", "")) or "")
                for item in scheduled_leafs[:3]
            )
            or "n/a"
        )
        top_long_poles = (
            ", ".join(
                str(item.get("sheet_name", "") or "") for item in long_pole_leafs[:3]
            )
            or "n/a"
        )

        top_level_label.set_text(
            f"parent_status={top_level} | phase={phase} | stage={current_stage_label} "
            f"| leaf workers {active_leaf_workers}/{total_leaf_workers} "
            f"(idle {idle_leaf_workers}) | queued {queued_leafs} "
            f"| completed {completed_leafs} | child Cu "
            f"T {preserved_child_traces}/{expected_child_traces} "
            f"V {preserved_child_vias}/{expected_child_vias} "
            f"| parent Cu +T {added_parent_traces} +V {added_parent_vias} "
            f"| scheduled {top_scheduled} | long poles {top_long_poles}"
        )

        best = status.get("best_score", 0)
        best_score_label.set_text(f"{best:.2f}" if best else "—")

        latest = status.get("latest_score")
        marker = status.get("latest_marker", "") or status.get("latest_event", "")

        if phase == "done" and marker != "run complete":
            marker = "run complete"
        if phase == "done" and current_action in {"—", "", None}:
            current_action = "run complete"

        schedule_hint = ""
        if scheduled_leafs:
            first_scheduled = str(
                scheduled_leafs[0].get(
                    "sheet_name",
                    scheduled_leafs[0].get("scheduled_selector", ""),
                )
                or ""
            )
            if first_scheduled:
                schedule_hint = f" | next priority={first_scheduled}"

        long_pole_hint = ""
        if long_pole_leafs:
            first_long_pole = str(long_pole_leafs[0].get("sheet_name", "") or "")
            if first_long_pole:
                long_pole_hint = f" | long pole={first_long_pole}"

        if latest is not None:
            latest_label.set_text(
                f"{float(latest):.2f} ({marker}) | action={current_action} | leaf={current_leaf} | parent={current_parent}{schedule_hint}{long_pole_hint}"
            )
        else:
            latest_label.set_text(
                f"{marker or '—'} | action={current_action} | leaf={current_leaf} | parent={current_parent}{schedule_hint}{long_pole_hint}"
            )

        w = status.get("workers", {})
        hierarchy = status.get("hierarchy", {})
        if not isinstance(hierarchy, dict):
            hierarchy = {}
        leaf_workers = hierarchy.get("leaf_workers", {})
        if not isinstance(leaf_workers, dict):
            leaf_workers = {}

        run_active = 0 if phase in {"done", "idle", "error"} else w.get("in_flight", 0)
        run_idle = 1 if phase in {"done", "idle", "error"} else w.get("idle", 0)

        workers_label.set_text(
            f"Run: total={w.get('total', 0)} active={run_active} idle={run_idle}"
            f" | Leaf: total={leaf_workers.get('total', 0)} active={leaf_workers.get('active', 0)}"
            f" idle={leaf_workers.get('idle', 0)} queued={leaf_workers.get('queued', 0)}"
            f" completed={leaf_workers.get('completed', 0)}"
            f" | stage={current_stage_label}"
        )

        stuck = status.get("maybe_stuck", False)
        if stuck:
            health_label.set_text("⚠ POSSIBLY STUCK")
            health_label.classes(replace="text-red-400")
        elif phase == "running":
            health_label.set_text("✓ Active")
            health_label.classes(replace="text-green-400")
        else:
            health_label.set_text("—")
            health_label.classes(replace="")

        is_running = phase == "running" or runner.is_running
        is_stopping = phase == "stopping"

        start_btn.set_visibility(not is_running and not is_stopping)
        stop_btn.set_visibility(is_running and not is_stopping)
        force_kill_btn.set_visibility(is_stopping)
        stopping_spinner.set_visibility(is_stopping)
        stopping_label.set_visibility(is_stopping)

        new_rounds = runner.read_latest_rounds(last_round_seen["value"])
        if new_rounds:
            live_rounds.extend(new_rounds)
            last_round_seen["value"] = max(r.get("round_num", 0) for r in new_rounds)
            if state.active_experiment_id:
                for nr in new_rounds:
                    db.add_round(state.active_experiment_id, nr)
                best_so_far = max((r.get("score", 0) for r in live_rounds), default=0)
                db.update_experiment(
                    state.active_experiment_id,
                    completed_rounds=len(live_rounds),
                    best_score=best_so_far,
                )
            chart_container.clear()
            with chart_container:
                create_score_chart(live_rounds, "Hierarchical Score")

        if prev_phase["value"] in ("running", "stopping") and phase in ("done", "idle"):
            if state.active_experiment_id:
                best_so_far = max((r.get("score", 0) for r in live_rounds), default=0)
                db.update_experiment(
                    state.active_experiment_id,
                    status="done",
                    completed_rounds=len(live_rounds),
                    best_score=best_so_far,
                )
                ui.notify("Hierarchical experiment finished!", type="positive")
        prev_phase["value"] = phase

        events = _read_jsonl_tail(state.experiments_dir / "experiments.jsonl", limit=20)
        _render_events(events, status)
        _render_artifacts()
        _render_top_outputs(status)
        _update_board_preview(status)

    # Start timer for live updates
    ui.timer(2.0, _update_status)

    async def _start():
        try:
            pid = runner.start(
                pcb_file=state.strategy["pcb_file"],
                rounds=state.strategy["rounds"],
                workers=state.strategy["workers"],
                plateau=state.strategy["plateau_threshold"],
                seed=state.strategy.get("seed"),
                param_ranges=state.get_control_ranges(),
                score_weights=state.score_weights,
                extra_config={
                    "schematic_file": state.strategy["schematic_file"],
                    "parent": state.strategy.get("parent", "/"),
                    "only": state.strategy.get("only", []),
                    "leaf_rounds": state.strategy.get("leaf_rounds", 1),
                    "render_png": state.toggles.get("render_png", True),
                    "save_round_details": state.toggles.get("save_round_details", True),
                },
            )
            exp = db.create_experiment(
                name=f"Hierarchical Run {time.strftime('%Y-%m-%d %H:%M')}",
                pcb_file=state.strategy["pcb_file"],
                total_rounds=state.strategy["rounds"],
                config=state.to_config_dict(),
            )
            state.active_experiment_id = exp.id
            db.update_experiment(exp.id, status="running")

            ui.notify(f"Started hierarchical experiment (PID {pid})", type="positive")
            live_rounds.clear()
            last_round_seen["value"] = 0
        except Exception as e:
            ui.notify(f"Failed to start: {e}", type="negative")

    def _stop():
        runner.stop()
        ui.notify(
            "Stop requested — pipeline will stop after the current safe checkpoint",
            type="info",
        )
        if state.active_experiment_id:
            db.update_experiment(state.active_experiment_id, status="stopping")

    def _force_kill():
        runner.kill()
        ui.notify("Force killed experiment and all child processes", type="warning")
        if state.active_experiment_id:
            db.update_experiment(state.active_experiment_id, status="done")

    start_btn.on_click(_start)
    stop_btn.on_click(_stop)
    force_kill_btn.on_click(_force_kill)
