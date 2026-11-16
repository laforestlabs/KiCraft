"""Small manifest-only controls; the replay iframe is created on demand."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re

from nicegui import ui


class LayoutHistoryPanel:
    def __init__(self, project_dir: Path, token: str, *, generation=None):
        self.project_dir = Path(project_dir)
        self.token = token
        self.generation = generation
        self._signal = object()
        self._capture_id = None
        self._loaded_capture = None
        self._dialog = None
        self._frame = None
        with ui.row().classes("w-full items-center gap-3") as self.element:
            self._status = ui.label("").classes("text-sm")
            self._watch = ui.button("Watch layout history", icon="movie", on_click=self.open)
            self._download = ui.link("Download timelapse (.html)", "#")
        self.refresh()

    def refresh(self):
        path = self.project_dir / "layout-history.json"
        try:
            stat = path.stat()
            signal = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            signal = None
        if signal == self._signal:
            return
        self._signal = signal
        try:
            manifest = json.loads(path.read_text()) if signal else {}
        except (OSError, ValueError):
            return
        capture_id = manifest.get("capture_id", "")
        available = (manifest.get("schema_version") == 1
                     and re.fullmatch(r"[0-9a-f]{32}", capture_id) is not None
                     and manifest.get("status") in ("complete", "partial")
                     and manifest.get("html_path") == f"layout-history/{capture_id}/timelapse.html")
        current = self.generation is None or manifest.get("generation") == self.generation
        if not current:
            text = "Layout history wasn't recorded for this build."
            available = False
        elif manifest.get("status") == "recording":
            text = "Recording layout history"
        elif available:
            text = "Partial layout history — gaps are labelled in the replay." if manifest["status"] == "partial" else "Layout history"
        elif manifest.get("status") == "disabled":
            text = "Layout history recording is disabled."
        else:
            text = "Layout history wasn't recorded for this build."
        self._status.set_text(text)
        self._watch.set_visibility(bool(available))
        self._download.set_visibility(bool(available))
        self._capture_id = capture_id if available else None
        if available:
            self._download.props(f'href="/project/{self.token}/history/{capture_id}/download"')

    def open(self):
        if not self._capture_id:
            return
        if self._dialog is None:
            # Created outside the polling renderer; its iframe survives manifest polls.
            with ui.dialog().props("maximized") as self._dialog:
                with ui.card().classes("w-full h-full p-2"):
                    with ui.row().classes("w-full justify-between"):
                        ui.label("Placement and routing history")
                        ui.button("Close", on_click=self._dialog.close).props("flat")
                    self._frame = ui.html("", sanitize=False).classes("w-full grow")
        if self._loaded_capture != self._capture_id:
            url = escape(f"/project/{self.token}/history/{self._capture_id}", quote=True)
            self._frame.set_content(
                f'<iframe title="Layout history" src="{url}" sandbox="allow-scripts" '
                'style="width:100%;height:calc(100vh - 100px);border:0"></iframe>'
            )
            self._loaded_capture = self._capture_id
        self._dialog.open()

    def dispose(self):
        if self._dialog is not None:
            self._dialog.delete()
            self._dialog = None
