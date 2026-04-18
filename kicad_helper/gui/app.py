"""NiceGUI application shell — main tabbed layout."""

from __future__ import annotations

import json
from pathlib import Path

from nicegui import ui

from .pages.analysis import analysis_page
from .pages.monitor import monitor_page
from .pages.setup import setup_page
from .state import get_state


def _load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _import_hierarchical_best_preset() -> None:
    """Import the best hierarchical run summary as a preset if available."""
    state = get_state()
    if not state.toggles.get("import_best_as_preset", True):
        return

    summary_path = state.experiments_dir / "best" / "best_hierarchical_round.json"
    summary = _load_json(summary_path)
    if not summary:
        return

    presets = state.db.get_presets()
    if any(p.name == "Best Hierarchical (imported)" for p in presets):
        return

    preset_config = {
        "_strategy": {
            "rounds": state.strategy.get("rounds", 50),
            "workers": state.strategy.get("workers", 0),
            "plateau_threshold": state.strategy.get("plateau_threshold", 1),
            "seed": summary.get("seed", 0),
            "pcb_file": state.strategy["pcb_file"],
        },
        "_hierarchical_best": summary,
    }

    notes = (
        "Auto-imported from hierarchical best summary "
        f"(round={summary.get('round_num', '?')}, score={summary.get('score', '?')})"
    )
    state.db.save_preset("Best Hierarchical (imported)", preset_config, notes)


def _auto_import_on_startup() -> None:
    """Import existing experiment data and hierarchical presets on first startup."""
    state = get_state()

    existing = state.db.get_experiments()
    if not existing:
        from .migrations.init_db import import_all_jsonl

        import_all_jsonl(state.db, state.experiments_dir)

    _import_hierarchical_best_preset()


_auto_import_on_startup()


@ui.page("/")
def index() -> None:
    """Main page with tabbed layout."""
    state = get_state()

    ui.dark_mode(True)
    ui.add_head_html(
        """
    <style>
        .nicegui-content { max-width: 1400px; margin: 0 auto; }
        .q-tab-panel { padding: 16px 0 !important; }
    </style>
    """
    )

    with ui.header().classes("items-center justify-between px-6"):
        ui.label(f"{state.project_name} Experiment Manager" if state.project_name != "project" else "KiCad Experiment Manager").classes("text-xl font-bold tracking-wide")
        with ui.row().classes("items-center gap-3"):
            ui.label(f"Project: {state.project_root.name}").classes(
                "text-sm text-gray-400"
            )
            ui.badge("Hierarchical Subcircuits", color="green").classes("text-xs")

    show_analysis_tab = bool(state.gui_cleanup.get("show_analysis_tab", True))

    with ui.tabs().classes("w-full") as tabs:
        setup_tab = ui.tab("Setup", icon="tune")
        monitor_tab = ui.tab("Monitor", icon="monitor")
        analysis_tab = None
        if show_analysis_tab:
            analysis_tab = ui.tab("Analysis", icon="analytics")

    with ui.tab_panels(tabs, value=setup_tab).classes("w-full px-4"):
        with ui.tab_panel(setup_tab):
            setup_page()
        with ui.tab_panel(monitor_tab):
            monitor_page()
        if analysis_tab is not None:
            with ui.tab_panel(analysis_tab):
                analysis_page()
