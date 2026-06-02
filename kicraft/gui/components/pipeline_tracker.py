"""Whole-pipeline tracker bar for the app header.

A compact horizontal stepper across the full KiCraft pipeline
(Intent -> Spec -> Architecture -> BOM -> Wiring -> Synthesize -> Place & Route
-> Fab), driven by ``kicraft.gui.pipeline_state.pipeline_progress``. It refreshes
on a timer so it advances live as the design conversation and the build proceed,
making the early CircuitChat stages observable in the GUI from the start.
"""
from __future__ import annotations

from pathlib import Path

from nicegui import ui

from ..pipeline_state import pipeline_progress

# state -> (material icon, quasar text-color class)
_STATE_STYLE: dict[str, tuple[str, str]] = {
    "done": ("check_circle", "text-positive"),
    "active": ("radio_button_checked", "text-info"),
    "pending": ("radio_button_unchecked", "text-grey-6"),
}


def pipeline_tracker(project_root: Path, *, poll_s: float = 3.0) -> None:
    """Mount the live pipeline tracker bar. Call inside a page context."""
    root = Path(project_root)

    @ui.refreshable
    def _bar() -> None:
        stages = pipeline_progress(root)
        with ui.row().classes("items-center gap-1 w-full flex-wrap py-1"):
            for i, st in enumerate(stages):
                icon, color = _STATE_STYLE.get(st.state, _STATE_STYLE["pending"])
                with ui.row().classes("items-center gap-1"):
                    ui.icon(icon).classes(f"{color} text-base")
                    text = st.label + (f" ({st.detail})" if st.detail else "")
                    weight = "font-semibold" if st.state == "active" else ""
                    ui.label(text).classes(f"text-xs {color} {weight}")
                if i < len(stages) - 1:
                    ui.icon("chevron_right").classes("text-grey-7 text-sm")

    _bar()
    ui.timer(poll_s, _bar.refresh)
