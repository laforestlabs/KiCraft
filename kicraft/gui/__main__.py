"""Entry point: python -m kicraft.gui"""
# NiceGUI requires page definitions to exist before ui.run(),
# and this module must not use an `if __name__` guard.
from . import app  # noqa: F401 — registers @ui.page routes
from nicegui import ui

from .state import get_state

_state = get_state()
_title = (
    f"{_state.project_name} Experiment Manager"
    if _state.project_name != "project"
    else "KiCad Experiment Manager"
)

ui.run(
    title=_title,
    port=8080,
    reload=False,
    show=True,
)
