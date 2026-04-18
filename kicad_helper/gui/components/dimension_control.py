"""Reusable search dimension control — toggle + range + fixed value."""
from __future__ import annotations

from nicegui import ui


class DimensionControl:
    """A single search dimension with enable toggle, range bounds, and fixed value.

    When enabled: optimizer searches within [min, max].
    When disabled: optimizer uses the fixed `default` value.
    """

    def __init__(self, dim: dict, on_change=None):
        self.dim = dim
        self._on_change = on_change
        self._build()

    def _notify(self):
        if self._on_change:
            self._on_change(self.dim)

    def _build(self):
        with ui.card().classes("w-full p-3"):
            with ui.row().classes("w-full items-center gap-4"):
                # Enable toggle
                self._toggle = ui.switch(
                    self.dim["label"],
                    value=self.dim["enabled"],
                    on_change=self._on_toggle,
                ).classes("font-bold")

                # Fixed value (shown when disabled)
                self._fixed_container = ui.column().classes("gap-1")
                with self._fixed_container:
                    ui.label("Fixed value:").classes("text-xs text-gray-400")
                    self._fixed_input = ui.number(
                        value=self.dim["default"],
                        min=self.dim["min"],
                        max=self.dim["max"],
                        step=self.dim["step"],
                        format=f"%.{self._decimals()}f",
                        on_change=lambda e: self._set("default", e.value),
                    ).classes("w-28")

                # Range controls (shown when enabled)
                self._range_container = ui.column().classes("gap-1 flex-grow")
                with self._range_container:
                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.label("Min:").classes("text-xs text-gray-400 w-8")
                        self._min_input = ui.number(
                            value=self.dim["min"],
                            step=self.dim["step"],
                            format=f"%.{self._decimals()}f",
                            on_change=lambda e: self._set("min", e.value),
                        ).classes("w-24")
                        ui.label("Max:").classes("text-xs text-gray-400 w-8")
                        self._max_input = ui.number(
                            value=self.dim["max"],
                            step=self.dim["step"],
                            format=f"%.{self._decimals()}f",
                            on_change=lambda e: self._set("max", e.value),
                        ).classes("w-24")
                        ui.label("Default:").classes("text-xs text-gray-400 w-14")
                        self._default_input = ui.number(
                            value=self.dim["default"],
                            step=self.dim["step"],
                            format=f"%.{self._decimals()}f",
                            on_change=lambda e: self._set("default", e.value),
                        ).classes("w-24")

            self._update_visibility()

    def _decimals(self) -> int:
        step = self.dim["step"]
        if step >= 1:
            return 0
        s = str(step)
        if "." in s:
            return len(s.split(".")[-1])
        return 2

    def _set(self, key: str, value):
        if value is not None:
            self.dim[key] = float(value)
            self._notify()

    def _on_toggle(self, e):
        self.dim["enabled"] = e.value
        self._update_visibility()
        self._notify()

    def _update_visibility(self):
        if self.dim["enabled"]:
            self._range_container.set_visibility(True)
            self._fixed_container.set_visibility(False)
        else:
            self._range_container.set_visibility(False)
            self._fixed_container.set_visibility(True)
