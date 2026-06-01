"""Design tab: read-only view of the CircuitChat conversation state.

Surfaces the upstream design stages from ``.kicraft/state.json`` (intent,
functional spec, architecture, BOM, wiring) inside the GUI, so the early design
is observable alongside the place/route experiments. Refreshes on a timer to
track the live conversation. Read-only: state is owned by ``stage-commit``.
"""
from __future__ import annotations

import json
from pathlib import Path

from nicegui import ui

from ..state import get_state


def _load_state(root: Path) -> dict | None:
    try:
        data = json.loads((root / ".kicraft" / "state.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _slot_summary(key: str, slot: dict) -> str:
    """One-line summary per slot so the tab is scannable without expanding."""
    if key == "intent":
        return str(slot.get("goal", ""))[:140]
    if key == "functional_spec":
        return f"{len(slot.get('blocks') or [])} blocks"
    if key == "architecture":
        return f"{len(slot.get('sheets') or [])} sheets"
    if key == "bom":
        return (
            f"{len(slot.get('parts') or [])} parts, "
            f"{len(slot.get('connections') or [])} connections"
        )
    return ""


def design_page() -> None:
    state = get_state()
    root = state.project_root

    @ui.refreshable
    def _body() -> None:
        d = _load_state(root)
        if not d:
            ui.label(
                "No CircuitChat design in this project yet "
                "(no .kicraft/state.json). Start a design with the circuitchat skill."
            ).classes("text-grey-6 p-4")
            return

        stem = d.get("project_stem") or "(unnamed)"
        ui.label(f"Project: {stem}").classes("text-lg font-bold")

        open_qs = d.get("open_questions") or []
        if open_qs:
            with ui.card().classes("w-full bg-amber-950"):
                ui.label(f"Open questions ({len(open_qs)})").classes(
                    "font-semibold text-amber-300"
                )
                for q in open_qs:
                    text = q.get("text", "") if isinstance(q, dict) else str(q)
                    ui.label(f"- {text}").classes("text-xs")

        for key, label in (
            ("intent", "Intent"),
            ("functional_spec", "Functional spec"),
            ("architecture", "Architecture"),
            ("bom", "BOM + wiring"),
        ):
            slot = d.get(key)
            present = bool(slot)
            icon = "check_circle" if present else "radio_button_unchecked"
            title = label if not present else f"{label} - {_slot_summary(key, slot)}"
            with ui.expansion(title, icon=icon).classes("w-full"):
                if not present:
                    ui.label("not yet captured").classes("text-grey-6 text-xs")
                else:
                    ui.code(json.dumps(slot, indent=2)).classes("w-full text-xs")

    _body()
    ui.timer(3.0, _body.refresh)
