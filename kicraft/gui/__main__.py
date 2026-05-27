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
    # A heavy round-end tick on the Monitor page can block the event loop
    # for a few seconds. NiceGUI derives the Socket.IO ping timeout from
    # this value (ping_timeout = max(reconnect_timeout*0.4, 2)), so the
    # default 3s yields a ~2s timeout and such a tick trips a spurious
    # disconnect/reconnect. Widening the window keeps the socket alive
    # across those stalls; monitor.py's timer pause/resume makes any
    # reconnect that does happen non-fatal.
    reconnect_timeout=8.0,
)
