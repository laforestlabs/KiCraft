"""Analysis page — hierarchical experiment statistics and progression."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from nicegui import ui

from ..components.experiment_table import create_experiment_table
from ..components.progression_viewer import create_progression_viewer
from ..components.score_chart import (
    build_leaf_timing_figure,
    build_score_figure,
    build_stage_figure,
    build_subscore_figure,
    build_timing_figure,
    build_timing_summary_figure,
)
from ..state import get_state


def analysis_page() -> None:
    state = get_state()

    ui.label("Experiment Analysis").classes("text-2xl font-bold mb-2")
    ui.label(
        "Review hierarchical experiment runs: routed leaf acceptance, parent "
        "composition, top-level readiness, and visual progression."
    ).classes("text-sm text-gray-400 mb-4")

    with ui.tabs().classes("w-full") as top_tabs:
        exp_tab = ui.tab("Experiment Data", icon="analytics")
        prog_tab = ui.tab("Board Progression", icon="slideshow")

    with ui.tab_panels(top_tabs, value=exp_tab).classes("w-full"):
        with ui.tab_panel(exp_tab):
            _experiment_data_panel(state)

        with ui.tab_panel(prog_tab):
            _progression_panel(state)


def _experiment_data_panel(state) -> None:
    def _ensure_latest_running_experiment() -> None:
        active_id = state.active_experiment_id
        if not active_id:
            return

        exp = state.db.get_experiment(active_id)
        if exp is not None:
            return

        status = state.runner.read_status()
        total_rounds = int(
            status.get("total_rounds", 0) or state.strategy.get("rounds", 0) or 0
        )
        best_score = float(status.get("best_score", 0) or 0)
        completed_rounds = int(status.get("round", 0) or 0)
        phase = str(status.get("phase", "running") or "running")
        if phase not in {"running", "stopping", "done", "error"}:
            phase = "running"

        exp = state.db.create_experiment(
            name=f"Hierarchical Run {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            pcb_file=state.strategy["pcb_file"],
            total_rounds=total_rounds,
            config=state.to_config_dict(),
        )
        state.active_experiment_id = exp.id
        state.db.update_experiment(
            exp.id,
            status=phase,
            best_score=best_score,
            completed_rounds=completed_rounds,
        )

    _ensure_latest_running_experiment()

    experiments = state.db.get_experiments()
    if not experiments:
        ui.label(
            "No experiments found. Import JSONL data or run a hierarchical experiment."
        ).classes("text-gray-500 italic")

        async def _import() -> None:
            from ..migrations.init_db import import_all_jsonl

            ids = import_all_jsonl(state.db, state.experiments_dir)
            if ids:
                ui.notify(f"Imported {len(ids)} experiments", type="positive")
                ui.navigate.reload()
            else:
                ui.notify("No JSONL files found to import", type="warning")

        ui.button("Import from .experiments/", icon="upload", on_click=_import)
        return

    def _build_exp_options() -> dict[int, str]:
        exps = state.db.get_experiments()
        return {
            exp.id: (
                f"#{exp.id} — {exp.name} "
                f"({exp.completed_rounds}r, best={exp.best_score:.1f})"
            )
            for exp in exps
        }

    exp_options = _build_exp_options()
    best_default = (
        next((exp for exp in experiments if exp.id == state.active_experiment_id), None)
        or experiments[0]
    )
    selected_exp = {"id": best_default.id}

    content = ui.column().classes("w-full")

    def _load_experiment(exp_id: int) -> None:
        selected_exp["id"] = exp_id
        content.clear()
        rounds = state.db.get_round_dicts(exp_id)

        if not rounds:
            with content:
                ui.label("No round data for this experiment.").classes(
                    "text-gray-500 italic"
                )

                async def _sync() -> None:
                    from ..migrations.init_db import import_all_jsonl

                    ids = import_all_jsonl(state.db, state.experiments_dir)
                    if ids:
                        ui.notify(
                            f"Re-imported {len(ids)} experiments", type="positive"
                        )
                        ui.navigate.reload()
                    else:
                        ui.notify("No new data to import", type="info")

                ui.button("Sync from disk", icon="sync", on_click=_sync).classes("mt-2")
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
            options=exp_options,
            value=selected_exp["id"],
            label="Select Experiment",
            on_change=lambda e: _load_experiment(e.value),
        ).classes("w-96")

        async def _import() -> None:
            from ..migrations.init_db import import_all_jsonl

            ids = import_all_jsonl(state.db, state.experiments_dir)
            if ids:
                ui.notify(f"Imported {len(ids)} experiments", type="positive")
                ui.navigate.reload()
            else:
                ui.notify("No new JSONL files to import", type="info")

        ui.button("Import JSONL", icon="upload", on_click=_import).props("flat")

        def _refresh() -> None:
            _ensure_latest_running_experiment()
            new_options = _build_exp_options()
            exp_select.options = new_options

            active_id = state.active_experiment_id
            if active_id and active_id in new_options:
                selected_exp["id"] = active_id
                exp_select.value = active_id
            elif selected_exp["id"] not in new_options and new_options:
                selected_exp["id"] = next(iter(new_options))
                exp_select.value = selected_exp["id"]

            exp_select.update()
            if selected_exp["id"]:
                _load_experiment(selected_exp["id"])

        ui.button("Refresh", icon="refresh", on_click=_refresh).props("flat")

    def _auto_refresh() -> None:
        _ensure_latest_running_experiment()
        new_options = _build_exp_options()
        exp_select.options = new_options

        active_id = state.active_experiment_id
        if active_id and active_id in new_options:
            if selected_exp["id"] != active_id:
                selected_exp["id"] = active_id
                exp_select.value = active_id
        elif selected_exp["id"] not in new_options and new_options:
            selected_exp["id"] = next(iter(new_options))
            exp_select.value = selected_exp["id"]

        exp_select.update()

        if selected_exp["id"]:
            exp = state.db.get_experiment(selected_exp["id"])
            if exp and exp.status in {"running", "stopping"}:
                _load_experiment(selected_exp["id"])

    ui.timer(10.0, _auto_refresh)

    if selected_exp["id"]:
        _load_experiment(selected_exp["id"])


def _progression_panel(state) -> None:
    ui.label("Hierarchical Progression").classes("text-xl font-bold mb-2")
    ui.label(
        "Browse accepted leaf artifacts and parent/top-level previews separately. "
        "This view is intended to make the bottom-up flow visually inspectable."
    ).classes("text-sm text-gray-400 mb-4")

    with ui.tabs().classes("w-full") as prog_tabs:
        viewer_tab = ui.tab("Timeline Viewer", icon="slideshow")
        leaf_tab = ui.tab("Accepted Leaf Gallery", icon="view_module")
        parent_tab = ui.tab("Parent / Top-Level Previews", icon="dashboard")

    with ui.tab_panels(prog_tabs, value=viewer_tab).classes("w-full"):
        with ui.tab_panel(viewer_tab):
            create_progression_viewer(state.experiments_dir)

        with ui.tab_panel(leaf_tab):
            _leaf_gallery_panel(state)

        with ui.tab_panel(parent_tab):
            _parent_preview_panel(state)


def _leaf_gallery_panel(state) -> None:
    sub_root = state.experiments_dir / "subcircuits"
    if not sub_root.exists():
        ui.label("No subcircuit artifacts found yet.").classes("text-gray-500 italic")
        return

    accepted: list[dict[str, Any]] = []

    def _open_preview_dialog(title: str, preview_path, subtitle: str = "") -> None:
        with (
            ui.dialog().props("maximized") as dialog,
            ui.card().classes("w-full h-full bg-slate-950 text-white p-4"),
        ):
            with ui.row().classes("w-full items-center gap-3 mb-3"):
                ui.label(title).classes("text-xl font-bold")
                if subtitle:
                    ui.label(subtitle).classes("text-sm text-gray-400 font-mono")
                ui.space()
                ui.button("Close", icon="close", on_click=dialog.close).props(
                    "flat color=white"
                )
            with ui.column().classes("w-full h-full items-center justify-center"):
                ui.image(str(preview_path)).classes(
                    "w-full h-[88vh] object-contain rounded border border-slate-700 bg-slate-900"
                )
        dialog.open()

    def _round_preview_paths(round_payload: dict[str, Any]) -> dict[str, Any]:
        routing = round_payload.get("routing", {})
        if not isinstance(routing, dict):
            routing = {}

        preview_paths = round_payload.get("preview_paths", {})
        if not isinstance(preview_paths, dict):
            preview_paths = {}

        board_paths = round_payload.get("board_paths", {})
        if not isinstance(board_paths, dict):
            board_paths = {}

        log_summary = round_payload.get("log_summary", {})
        if not isinstance(log_summary, dict):
            log_summary = {}

        render_diagnostics = routing.get("render_diagnostics", {})
        if not isinstance(render_diagnostics, dict):
            render_diagnostics = {}

        def _path_or_none(value: Any) -> Path | None:
            if not value:
                return None
            try:
                return Path(str(value))
            except (TypeError, ValueError):
                return None

        def _paths_for(stage_key: str) -> dict[str, Path | None]:
            stage_payload = render_diagnostics.get(stage_key, {})
            if not isinstance(stage_payload, dict):
                stage_payload = {}
            board_views = stage_payload.get("board_views", {})
            if not isinstance(board_views, dict):
                board_views = {}
            paths = board_views.get("paths", {})
            if not isinstance(paths, dict):
                paths = {}
            return {
                "front": _path_or_none(paths.get("front_all")),
                "back": _path_or_none(paths.get("back_all")),
                "copper": _path_or_none(paths.get("copper_both")),
            }

        pre_route = _paths_for("pre_route")
        routed = _paths_for("routed")
        return {
            "pre_front": _path_or_none(preview_paths.get("pre_route_front"))
            or pre_route["front"],
            "pre_back": _path_or_none(preview_paths.get("pre_route_back"))
            or pre_route["back"],
            "pre_copper": _path_or_none(preview_paths.get("pre_route_copper"))
            or pre_route["copper"],
            "routed_front": _path_or_none(preview_paths.get("routed_front"))
            or routed["front"],
            "routed_back": _path_or_none(preview_paths.get("routed_back"))
            or routed["back"],
            "routed_copper": _path_or_none(preview_paths.get("routed_copper"))
            or routed["copper"],
            "illegal_board": _path_or_none(board_paths.get("illegal_pre_stamp")),
            "pre_route_board": _path_or_none(board_paths.get("pre_route")),
            "routed_board": _path_or_none(board_paths.get("routed")),
            "router": str(log_summary.get("router", "") or ""),
            "reason": str(log_summary.get("reason", "") or ""),
            "failed": bool(log_summary.get("failed", False)),
            "skipped": bool(log_summary.get("skipped", False)),
            "failed_internal_nets": list(
                log_summary.get("failed_internal_nets", []) or []
            ),
            "routed_internal_nets": list(
                log_summary.get("routed_internal_nets", []) or []
            ),
            "total_length_mm": float(log_summary.get("total_length_mm", 0.0) or 0.0),
        }

    for artifact_dir in sorted(sub_root.iterdir()):
        if not artifact_dir.is_dir():
            continue

        solved_path = artifact_dir / "solved_layout.json"
        metadata_path = artifact_dir / "metadata.json"
        debug_path = artifact_dir / "debug.json"
        if not solved_path.exists():
            continue

        try:
            with open(solved_path, encoding="utf-8") as f:
                solved = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(solved, dict):
            continue

        validation = solved.get("validation", {})
        if not isinstance(validation, dict) or validation.get("accepted") is not True:
            continue

        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    metadata = loaded
            except (OSError, json.JSONDecodeError):
                metadata = {}

        debug_payload: dict[str, Any] = {}
        if debug_path.exists():
            try:
                with open(debug_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    debug_payload = loaded
            except (OSError, json.JSONDecodeError):
                debug_payload = {}

        extra = debug_payload.get("extra", {})
        if not isinstance(extra, dict):
            extra = {}

        all_rounds = extra.get("all_rounds", [])
        if not isinstance(all_rounds, list):
            all_rounds = []

        best_round = extra.get("best_round", {})
        if not isinstance(best_round, dict):
            best_round = {}

        renders_dir = artifact_dir / "renders"
        front_preview = next(
            (
                p
                for p in [
                    renders_dir / "routed_front_all.png",
                    renders_dir / "pre_route_front_all.png",
                ]
                if p.exists()
            ),
            None,
        )
        back_preview = next(
            (
                p
                for p in [
                    renders_dir / "routed_back_all.png",
                    renders_dir / "pre_route_back_all.png",
                ]
                if p.exists()
            ),
            None,
        )
        copper_preview = next(
            (
                p
                for p in [
                    renders_dir / "routed_copper_both.png",
                    renders_dir / "pre_route_copper_both.png",
                ]
                if p.exists()
            ),
            None,
        )

        candidate_rounds: list[dict[str, Any]] = []
        for round_payload in all_rounds:
            if not isinstance(round_payload, dict):
                continue
            placement = round_payload.get("placement", {})
            if not isinstance(placement, dict):
                placement = {}
            routing = round_payload.get("routing", {})
            if not isinstance(routing, dict):
                routing = {}
            validation_payload = routing.get("validation", {})
            if not isinstance(validation_payload, dict):
                validation_payload = {}
            preview_paths = _round_preview_paths(round_payload)
            candidate_rounds.append(
                {
                    "round_index": int(round_payload.get("round_index", 0) or 0),
                    "seed": int(round_payload.get("seed", 0) or 0),
                    "score": float(round_payload.get("score", 0.0) or 0.0),
                    "routed": bool(round_payload.get("routed", False)),
                    "accepted": bool(validation_payload.get("accepted", False)),
                    "traces": int(routing.get("traces", 0) or 0),
                    "vias": int(routing.get("vias", 0) or 0),
                    "net_distance": float(placement.get("net_distance", 0.0) or 0.0),
                    "crossovers": int(placement.get("crossover_count", 0) or 0),
                    "compactness": float(placement.get("compactness", 0.0) or 0.0),
                    "pre_front": preview_paths["pre_front"],
                    "pre_back": preview_paths["pre_back"],
                    "pre_copper": preview_paths["pre_copper"],
                    "routed_front": preview_paths["routed_front"],
                    "routed_back": preview_paths["routed_back"],
                    "routed_copper": preview_paths["routed_copper"],
                    "illegal_board": preview_paths["illegal_board"],
                    "pre_route_board": preview_paths["pre_route_board"],
                    "routed_board": preview_paths["routed_board"],
                    "router": preview_paths["router"],
                    "reason": preview_paths["reason"],
                    "failed": preview_paths["failed"],
                    "skipped": preview_paths["skipped"],
                    "failed_internal_nets": preview_paths["failed_internal_nets"],
                    "routed_internal_nets": preview_paths["routed_internal_nets"],
                    "total_length_mm": preview_paths["total_length_mm"],
                }
            )

        candidate_rounds.sort(key=lambda item: item["score"], reverse=True)

        best_round_index = None
        if best_round:
            best_round_index = int(best_round.get("round_index", 0) or 0)

        accepted.append(
            {
                "sheet_name": solved.get("sheet_name")
                or metadata.get("sheet_name")
                or artifact_dir.name,
                "instance_path": solved.get("instance_path")
                or metadata.get("instance_path")
                or "",
                "trace_count": len(solved.get("traces", [])),
                "via_count": len(solved.get("vias", [])),
                "artifact_dir": artifact_dir.name,
                "front_preview": front_preview,
                "back_preview": back_preview,
                "copper_preview": copper_preview,
                "candidate_rounds": candidate_rounds,
                "best_round_index": best_round_index,
            }
        )

    if not accepted:
        ui.label("No accepted routed leaf artifacts yet.").classes(
            "text-gray-500 italic"
        )
        return

    ui.label(f"{len(accepted)} accepted routed leaf artifacts").classes(
        "text-sm text-gray-400 mb-3"
    )
    ui.label(
        "Each leaf now shows separate front, back, and combined-copper previews so "
        "silkscreen and back-layer routing are easier to inspect. Click any preview "
        "to open a larger inspection view."
    ).classes("text-sm text-gray-400 mb-2")
    ui.label(
        "Use the candidate-round inspector under each accepted leaf to see whether "
        "the solver actually explored multiple placements, which seed won, and how "
        "the attempted rounds scored against each other."
    ).classes("text-sm text-amber-300 mb-4")

    with ui.grid(columns=1).classes("w-full gap-4"):
        for item in accepted:
            with ui.card().classes("w-full p-4"):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.badge("LEAF", color="green")
                    ui.label(str(item["sheet_name"])).classes("font-bold text-lg")
                    ui.space()
                    ui.badge(f"{item['trace_count']} traces", color="cyan").classes(
                        "text-xs"
                    )
                    ui.badge(f"{item['via_count']} vias", color="amber").classes(
                        "text-xs"
                    )

                if item["instance_path"]:
                    ui.label(str(item["instance_path"])).classes(
                        "text-xs text-gray-400 font-mono"
                    )

                ui.label(str(item["artifact_dir"])).classes(
                    "text-xs text-gray-500 font-mono mb-2"
                )

                with ui.grid(columns=3).classes("w-full gap-3"):
                    for label, preview in [
                        ("Front (silkscreen + copper)", item["front_preview"]),
                        ("Back (silkscreen + copper)", item["back_preview"]),
                        ("Combined copper", item["copper_preview"]),
                    ]:
                        with ui.card().classes("w-full p-2 bg-slate-900"):
                            ui.label(label).classes(
                                "text-xs text-gray-300 font-medium mb-2"
                            )
                            if preview is not None:
                                ui.image(str(preview)).classes(
                                    "w-full h-[360px] object-contain rounded border border-slate-700 bg-slate-950 cursor-pointer"
                                ).on(
                                    "click",
                                    lambda _, p=preview, preview_label=label, item=item: (
                                        _open_preview_dialog(
                                            f"{item['sheet_name']} — {preview_label}",
                                            p,
                                            str(
                                                item["instance_path"]
                                                or item["artifact_dir"]
                                            ),
                                        )
                                    ),
                                )
                                ui.button(
                                    "Open full resolution",
                                    icon="open_in_full",
                                    on_click=lambda p=preview, preview_label=label, item=item: (
                                        _open_preview_dialog(
                                            f"{item['sheet_name']} — {preview_label}",
                                            p,
                                            str(
                                                item["instance_path"]
                                                or item["artifact_dir"]
                                            ),
                                        )
                                    ),
                                ).props("flat dense").classes("mt-2 text-cyan-300")
                            else:
                                ui.label("Preview not available").classes(
                                    "text-gray-500 italic text-sm"
                                )

                candidate_rounds = item["candidate_rounds"]
                if candidate_rounds:
                    with ui.expansion(
                        f"Candidate round inspector ({len(candidate_rounds)} attempted rounds)",
                        value=False,
                    ).classes("w-full mt-4"):
                        ui.label(
                            "These are the attempted placement/routing rounds for this leaf. "
                            "The winning round is marked so you can tell whether the solver "
                            "actually explored alternatives or just converged to one obvious answer."
                        ).classes("text-sm text-gray-400 mb-3")

                        with ui.column().classes("w-full gap-3"):
                            for round_item in candidate_rounds:
                                is_best = (
                                    item["best_round_index"] is not None
                                    and round_item["round_index"]
                                    == item["best_round_index"]
                                )
                                with ui.card().classes("w-full p-3 bg-slate-900/70"):
                                    with ui.row().classes("w-full items-center gap-2"):
                                        ui.badge(
                                            f"ROUND {round_item['round_index']}",
                                            color="blue",
                                        )
                                        if is_best:
                                            ui.badge("WINNER", color="green")
                                        if round_item["accepted"]:
                                            ui.badge("ACCEPTED", color="green")
                                        elif round_item["routed"]:
                                            ui.badge("ROUTED", color="orange")
                                        else:
                                            ui.badge("FAILED", color="red")
                                        ui.space()
                                        ui.label(
                                            f"score {round_item['score']:.2f}"
                                        ).classes("font-mono text-green-300")

                                    with ui.row().classes(
                                        "w-full gap-4 flex-wrap mt-2"
                                    ):
                                        ui.label(f"seed={round_item['seed']}").classes(
                                            "text-xs text-gray-400 font-mono"
                                        )
                                        ui.label(
                                            f"crossovers={round_item['crossovers']}"
                                        ).classes("text-xs text-cyan-300")
                                        ui.label(
                                            f"net_distance={round_item['net_distance']:.2f}"
                                        ).classes("text-xs text-cyan-300")
                                        ui.label(
                                            f"compactness={round_item['compactness']:.2f}"
                                        ).classes("text-xs text-cyan-300")
                                        ui.label(
                                            f"traces={round_item['traces']}"
                                        ).classes("text-xs text-amber-300")
                                        ui.label(f"vias={round_item['vias']}").classes(
                                            "text-xs text-amber-300"
                                        )
                                        if round_item["router"]:
                                            ui.label(
                                                f"router={round_item['router']}"
                                            ).classes(
                                                "text-xs text-purple-300 font-mono"
                                            )
                                        ui.label(
                                            f"length_mm={round_item['total_length_mm']:.2f}"
                                        ).classes("text-xs text-emerald-300")

                                    with ui.expansion(
                                        "Board paths and machine diagnostics",
                                        value=False,
                                    ).classes("w-full mt-3"):
                                        if round_item["reason"]:
                                            ui.label(
                                                f"reason={round_item['reason']}"
                                            ).classes(
                                                "text-xs text-amber-300 font-mono break-all mb-2"
                                            )

                                        with ui.column().classes("w-full gap-1"):
                                            for label, board_path in [
                                                (
                                                    "illegal_pre_stamp_board",
                                                    round_item["illegal_board"],
                                                ),
                                                (
                                                    "pre_route_board",
                                                    round_item["pre_route_board"],
                                                ),
                                                (
                                                    "routed_board",
                                                    round_item["routed_board"],
                                                ),
                                            ]:
                                                if board_path is not None:
                                                    ui.label(
                                                        f"{label}={board_path}"
                                                    ).classes(
                                                        "text-[11px] text-gray-400 font-mono break-all"
                                                    )

                                        if round_item["failed_internal_nets"]:
                                            ui.label(
                                                "failed_internal_nets="
                                                + ", ".join(
                                                    str(net)
                                                    for net in round_item[
                                                        "failed_internal_nets"
                                                    ]
                                                )
                                            ).classes(
                                                "text-[11px] text-red-300 font-mono break-all mt-2"
                                            )

                                        if round_item["routed_internal_nets"]:
                                            ui.label(
                                                "routed_internal_nets="
                                                + ", ".join(
                                                    str(net)
                                                    for net in round_item[
                                                        "routed_internal_nets"
                                                    ]
                                                )
                                            ).classes(
                                                "text-[11px] text-green-300 font-mono break-all"
                                            )

                                        ui.label(
                                            f"failed={round_item['failed']} | skipped={round_item['skipped']}"
                                        ).classes(
                                            "text-[11px] text-gray-500 font-mono mt-2"
                                        )

                                    with ui.grid(columns=3).classes(
                                        "w-full gap-3 mt-3"
                                    ):
                                        for preview_label, preview in [
                                            (
                                                "Pre-route front",
                                                round_item["pre_front"],
                                            ),
                                            (
                                                "Routed front",
                                                round_item["routed_front"],
                                            ),
                                            (
                                                "Routed copper",
                                                round_item["routed_copper"],
                                            ),
                                        ]:
                                            with ui.card().classes(
                                                "w-full p-2 bg-slate-950/70"
                                            ):
                                                ui.label(preview_label).classes(
                                                    "text-xs text-gray-300 font-medium mb-2"
                                                )
                                                if (
                                                    preview is not None
                                                    and preview.exists()
                                                ):
                                                    ui.image(str(preview)).classes(
                                                        "w-full h-[220px] object-contain rounded border border-slate-700 bg-slate-950 cursor-pointer"
                                                    ).on(
                                                        "click",
                                                        lambda _, p=preview, preview_label=preview_label, item=item, round_item=round_item: (
                                                            _open_preview_dialog(
                                                                f"{item['sheet_name']} — round {round_item['round_index']} — {preview_label}",
                                                                p,
                                                                f"{item['instance_path'] or item['artifact_dir']} | seed={round_item['seed']}",
                                                            )
                                                        ),
                                                    )
                                                else:
                                                    ui.label(
                                                        "No per-round preview available"
                                                    ).classes(
                                                        "text-gray-500 italic text-sm"
                                                    )
                else:
                    ui.label(
                        "No candidate-round metadata was found for this leaf."
                    ).classes("text-sm text-gray-500 italic mt-4")


def _parent_preview_panel(state) -> None:
    ui.label(
        "The parent previews below now come from the single unified parent pipeline. "
        "The stamped parent image is the source of truth for preserved routed child copper, "
        "and the routed parent image shows the result after parent interconnect routing. "
        "If the parent pipeline cannot produce valid geometry, it should fail explicitly "
        "instead of silently continuing to routing."
    ).classes("text-sm text-amber-300 mb-4")

    preview_sets: list[dict[str, Any]] = []

    def _open_preview_dialog(title: str, preview_path, subtitle: str = "") -> None:
        with (
            ui.dialog().props("maximized") as dialog,
            ui.card().classes("w-full h-full bg-slate-950 text-white p-4"),
        ):
            with ui.row().classes("w-full items-center gap-3 mb-3"):
                ui.label(title).classes("text-xl font-bold")
                if subtitle:
                    ui.label(subtitle).classes("text-sm text-gray-400 font-mono")
                ui.space()
                ui.button("Close", icon="close", on_click=dialog.close).props(
                    "flat color=white"
                )
            with ui.column().classes("w-full h-full items-center justify-center"):
                ui.image(str(preview_path)).classes(
                    "w-full h-[88vh] object-contain rounded border border-slate-700 bg-slate-900"
                )
        dialog.open()

    def _normalize_copper_accounting(payload: dict[str, Any]) -> dict[str, int]:
        return {
            "expected_preserved_child_trace_count": int(
                payload.get("expected_preserved_child_trace_count", 0) or 0
            ),
            "expected_preserved_child_via_count": int(
                payload.get("expected_preserved_child_via_count", 0) or 0
            ),
            "preserved_child_trace_count": int(
                payload.get("preserved_child_trace_count", 0) or 0
            ),
            "preserved_child_via_count": int(
                payload.get("preserved_child_via_count", 0) or 0
            ),
            "routed_total_trace_count": int(
                payload.get("routed_total_trace_count", 0) or 0
            ),
            "routed_total_via_count": int(
                payload.get("routed_total_via_count", 0) or 0
            ),
            "added_parent_trace_count": int(
                payload.get("added_parent_trace_count", 0) or 0
            ),
            "added_parent_via_count": int(
                payload.get("added_parent_via_count", 0) or 0
            ),
        }

    def _extract_copper_accounting(payload: dict[str, Any]) -> dict[str, int]:
        if not isinstance(payload, dict):
            return {}

        normalized = _normalize_copper_accounting(payload)
        if any(normalized.values()):
            return normalized

        routing_result = payload.get("routing_result", {})
        if isinstance(routing_result, dict):
            copper = routing_result.get("copper_accounting", {})
            if isinstance(copper, dict):
                normalized = _normalize_copper_accounting(copper)
                if any(normalized.values()):
                    return normalized

        hierarchy = payload.get("hierarchical_status", {})
        if isinstance(hierarchy, dict):
            copper = hierarchy.get("copper_accounting", {})
            if isinstance(copper, dict):
                normalized = _normalize_copper_accounting(copper)
                if any(normalized.values()):
                    return normalized

        composition = payload.get("composition", {})
        track_counts = payload.get("track_counts", {})
        if isinstance(composition, dict) and isinstance(track_counts, dict):
            preloaded = track_counts.get("preloaded", {})
            final = track_counts.get("final", {})
            if isinstance(preloaded, dict) and isinstance(final, dict):
                expected_traces = int(composition.get("trace_count", 0) or 0)
                expected_vias = int(composition.get("via_count", 0) or 0)
                preloaded_traces = int(preloaded.get("traces", 0) or 0)
                preloaded_vias = int(preloaded.get("vias", 0) or 0)
                final_traces = int(final.get("traces", 0) or 0)
                final_vias = int(final.get("vias", 0) or 0)
                return {
                    "expected_preserved_child_trace_count": expected_traces,
                    "expected_preserved_child_via_count": expected_vias,
                    "preserved_child_trace_count": min(
                        expected_traces, preloaded_traces
                    ),
                    "preserved_child_via_count": min(expected_vias, preloaded_vias),
                    "routed_total_trace_count": final_traces,
                    "routed_total_via_count": final_vias,
                    "added_parent_trace_count": max(0, final_traces - preloaded_traces),
                    "added_parent_via_count": max(0, final_vias - preloaded_vias),
                }

        return {}

    def _add_preview_set(base_dir, label: str) -> None:
        if not base_dir.exists():
            return

        preloaded = None
        routed = None
        metadata = None
        stamped_board = None
        routed_board = None

        search_dirs = [
            base_dir,
            base_dir / "renders",
        ]

        for search_dir in search_dirs:
            for candidate in [
                search_dir / "parent_stamped.png",
                search_dir / "board.png",
                search_dir / "snapshot.png",
            ]:
                if candidate.exists():
                    preloaded = candidate
                    break
            if preloaded is not None:
                break

        for search_dir in search_dirs:
            for candidate in [
                search_dir / "parent_routed.png",
                search_dir / "routed.png",
                search_dir / "board_routed.png",
            ]:
                if candidate.exists():
                    routed = candidate
                    break
            if routed is not None:
                break

        for candidate in [
            base_dir / "parent_pre_freerouting.kicad_pcb",
            base_dir / "parent_stamped.kicad_pcb",
        ]:
            if candidate.exists():
                stamped_board = candidate
                break

        for candidate in [
            base_dir / "parent_routed.kicad_pcb",
        ]:
            if candidate.exists():
                routed_board = candidate
                break

        for search_dir in search_dirs:
            for candidate in [
                search_dir / "debug.json",
                search_dir / "metadata.json",
                search_dir / "summary.json",
                search_dir / "parent_composition.json",
            ]:
                if candidate.exists():
                    metadata = candidate
                    break
            if metadata is not None:
                break

        if (
            preloaded is None
            and routed is None
            and metadata is None
            and stamped_board is None
            and routed_board is None
        ):
            return

        metadata_payload: dict[str, Any] = {}
        copper_accounting: dict[str, int] = {}
        if metadata is not None:
            try:
                with open(metadata, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    metadata_payload = loaded
                    copper_accounting = _extract_copper_accounting(loaded)
            except (OSError, json.JSONDecodeError):
                metadata_payload = {}
                copper_accounting = {}

        artifact_paths = metadata_payload.get("artifact_paths", {})
        if not isinstance(artifact_paths, dict):
            artifact_paths = {}

        if stamped_board is None and artifact_paths.get("parent_pre_freerouting_board"):
            stamped_board = Path(str(artifact_paths["parent_pre_freerouting_board"]))
        if routed_board is None and artifact_paths.get("parent_routed_board"):
            routed_board = Path(str(artifact_paths["parent_routed_board"]))

        preview_sets.append(
            {
                "label": label,
                "base_dir": base_dir,
                "preloaded": preloaded,
                "routed": routed,
                "stamped_board": stamped_board,
                "routed_board": routed_board,
                "metadata": metadata,
                "metadata_payload": metadata_payload,
                "copper_accounting": copper_accounting,
            }
        )

    auto_root = state.experiments_dir / "hierarchical_autoexperiment"
    if auto_root.exists():
        for round_dir in sorted(auto_root.glob("round_*"), reverse=True):
            _add_preview_set(round_dir, f"Autoexperiment {round_dir.name}")

    if not preview_sets:
        ui.label("No parent/top-level preview artifacts found yet.").classes(
            "text-gray-500 italic"
        )
        return

    with ui.column().classes("w-full gap-6"):
        for item in preview_sets:
            copper = item["copper_accounting"]
            with ui.card().classes("w-full p-4"):
                ui.label(str(item["label"])).classes("text-lg font-bold mb-1")
                ui.label(str(item["base_dir"])).classes(
                    "text-xs text-gray-500 font-mono mb-3"
                )
                ui.label(
                    "Compare the stamped parent against the routed parent from the unified "
                    "parent pipeline. The stamped image is the best visual proof that routed "
                    "child copper survived composition. The routed/final image is mainly for "
                    "checking what parent interconnect was added afterward."
                ).classes("text-sm text-gray-300 mb-4")

                if copper:
                    with ui.row().classes("w-full gap-2 mb-4 flex-wrap"):
                        ui.badge(
                            "Preserved child copper",
                            color="green",
                        )
                        ui.badge(
                            f"traces {copper.get('preserved_child_trace_count', 0)}/"
                            f"{copper.get('expected_preserved_child_trace_count', 0)}",
                            color="cyan",
                        )
                        ui.badge(
                            f"vias {copper.get('preserved_child_via_count', 0)}/"
                            f"{copper.get('expected_preserved_child_via_count', 0)}",
                            color="amber",
                        )
                        ui.badge(
                            f"added parent traces {copper.get('added_parent_trace_count', 0)}",
                            color="blue",
                        )
                        ui.badge(
                            f"added parent vias {copper.get('added_parent_via_count', 0)}",
                            color="purple",
                        )
                        ui.badge(
                            f"final traces {copper.get('routed_total_trace_count', 0)}",
                            color="teal",
                        )
                        ui.badge(
                            f"final vias {copper.get('routed_total_via_count', 0)}",
                            color="orange",
                        )

                with ui.grid(columns=2).classes("w-full gap-4"):
                    with ui.card().classes("w-full p-3 bg-slate-900"):
                        ui.label("Stamped parent").classes(
                            "text-sm font-bold text-gray-200 mb-2"
                        )
                        ui.label(
                            "Use this first when you want to verify preserved child copper, "
                            "module spacing, and the truth of the composed parent before parent "
                            "routing adds anything new. This image is generated by the same "
                            "single parent pipeline that performs stamping and routing."
                        ).classes("text-xs text-gray-400 mb-3")
                        if item["stamped_board"] is not None:
                            ui.label(f"board={item['stamped_board']}").classes(
                                "text-[11px] text-emerald-300 font-mono break-all mb-2"
                            )
                        if item["preloaded"] is not None:
                            ui.image(str(item["preloaded"])).classes(
                                "w-full h-[520px] object-contain rounded border border-slate-700 bg-slate-950 cursor-pointer"
                            ).on(
                                "click",
                                lambda _, p=item["preloaded"], label=item["label"]: (
                                    _open_preview_dialog(
                                        f"{label} — Stamped parent",
                                        p,
                                        str(item["base_dir"]),
                                    )
                                ),
                            )
                            ui.button(
                                "Open full resolution",
                                icon="open_in_full",
                                on_click=lambda p=item["preloaded"], label=item["label"]: (
                                    _open_preview_dialog(
                                        f"{label} — Stamped parent",
                                        p,
                                        str(item["base_dir"]),
                                    )
                                ),
                            ).props("flat dense").classes("mt-2 text-cyan-300")
                        else:
                            ui.label("No stamped preview found").classes(
                                "text-gray-500 italic"
                            )

                    with ui.card().classes("w-full p-3 bg-slate-900"):
                        ui.label("Routed parent").classes(
                            "text-sm font-bold text-gray-200 mb-2"
                        )
                        ui.label(
                            "Use this to inspect newly added parent interconnect. If child copper "
                            "looks visually understated here, compare against the stamped view "
                            "and the copper-accounting badges above."
                        ).classes("text-xs text-gray-400 mb-3")
                        if item["routed_board"] is not None:
                            ui.label(f"board={item['routed_board']}").classes(
                                "text-[11px] text-emerald-300 font-mono break-all mb-2"
                            )
                        if item["routed"] is not None:
                            ui.image(str(item["routed"])).classes(
                                "w-full h-[520px] object-contain rounded border border-slate-700 bg-slate-950 cursor-pointer"
                            ).on(
                                "click",
                                lambda _, p=item["routed"], label=item["label"]: (
                                    _open_preview_dialog(
                                        f"{label} — Routed parent",
                                        p,
                                        str(item["base_dir"]),
                                    )
                                ),
                            )
                            ui.button(
                                "Open full resolution",
                                icon="open_in_full",
                                on_click=lambda p=item["routed"], label=item["label"]: (
                                    _open_preview_dialog(
                                        f"{label} — Routed parent",
                                        p,
                                        str(item["base_dir"]),
                                    )
                                ),
                            ).props("flat dense").classes("mt-2 text-cyan-300")
                        else:
                            ui.label("No routed preview found").classes(
                                "text-gray-500 italic"
                            )

                if item["metadata"] is not None:
                    with ui.expansion("Metadata", value=False).classes("w-full mt-4"):
                        try:
                            payload = item["metadata_payload"]
                            ui.code(json.dumps(payload, indent=2)).classes(
                                "w-full text-xs"
                            )
                        except (OSError, json.JSONDecodeError, TypeError):
                            ui.label("Could not load metadata").classes(
                                "text-red-400 text-sm"
                            )


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
