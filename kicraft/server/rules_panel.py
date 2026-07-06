"""Web per-component placement-rules panel for the place/route tab.

Lists every component grouped by leaf sheet with editable anchor /
rotation / thermal overrides, sheet-level backside-THT flags, and fixed
board dimensions. Edits commit to the durable ``placement`` section of
state.json (`stage-commit placement`: deterministic, zero AI cost, no
upstream invalidation) and take effect through a normal rebuild, where
``write_autoplacer_json`` merges them over the LLM-derived hints with
the highest precedence.

Seeding reads the project's GENERATED ``<stem>_autoplacer.json`` (the
effective merged config) so the panel shows what the placer actually
used, including the LLM's own zones; committing the edited set as the
placement section simply makes those choices durable user rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nicegui import ui

from kicraft.layout_editor.rules import (
    ANCHOR_CHOICES,
    ANCHOR_VALUES,
    LeafEntry,
    _anchor_target,
    _anchor_value,
    _classify,
    _load_from_schematic,
    _load_overrides_into_state,
    _set_anchor,
    _set_backside_through_hole,
    _set_rotation,
    _set_thermal,
)
from kicraft.server import session


def _seed_state(project_dir: Path) -> SimpleNamespace:
    """The mutable staging holder the rules functions are duck-typed to."""
    holder = SimpleNamespace(
        project_root=project_dir,
        component_zone_overrides={},
        thermal_ref_overrides=set(),
        backside_through_hole_overrides=set(),
        per_component_loaded=False,
    )
    _load_overrides_into_state(holder)
    return holder


def _current_board_config(project_dir: Path, stem: str) -> dict[str, Any]:
    try:
        cfg = json.loads(
            (project_dir / f"{stem}_autoplacer.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        cfg = {}
    return {
        "width_mm": cfg.get("board_width_mm"),
        "height_mm": cfg.get("board_height_mm"),
        "size_search": bool(cfg.get("enable_board_size_search", True)),
    }


class PlacementRulesPanel:
    """One open rules panel; render() paints into the current container."""

    def __init__(self, *, ws: Path, project_dir: Path, stem: str, user,
                 on_exit, on_rebuild, is_run_active) -> None:
        self.ws = ws
        self.project_dir = project_dir
        self.stem = stem
        self.user = user
        self.on_exit = on_exit
        # Host callback that kicks the LLM-free rebuild (synthesize ->
        # place -> route -> fab) through the build queue.
        self.on_rebuild = on_rebuild
        self.is_run_active = is_run_active
        self.holder = _seed_state(project_dir)
        self.board = _current_board_config(project_dir, stem)
        self._status = None

    # -- UI -------------------------------------------------------------------

    def render(self) -> None:
        with ui.row().classes("w-full items-center gap-3"):
            ui.button("Back to board", icon="arrow_back",
                      on_click=self.on_exit).props("flat dense")
            ui.label("Placement rules").classes("text-sm font-medium") \
                .style("color:#e2e8f0")
            ui.button("Apply & re-place", icon="published_with_changes",
                      color="primary", on_click=self._confirm_apply) \
                .props("dense")
            self._status = ui.label("").classes("text-xs ml-auto") \
                .style("color:#94a3b8")
        ui.label(
            "Anchor pins a part to an edge, corner, or board region; thermal "
            "adds a keep-away radius. Rules persist with the project and are "
            "honored by every future re-place."
        ).classes("text-xs").style("color:#64748b")

        self._render_board_row()

        data = _load_from_schematic(self.project_dir, f"{self.stem}.kicad_sch")
        if data.error:
            ui.label(data.error).classes("text-sm text-red-400")
            return
        for leaf in sorted(data.leaves, key=lambda lf: lf.label.lower()):
            self._render_leaf(leaf)

    def _render_board_row(self) -> None:
        with ui.row().classes("w-full items-center gap-3 mt-1"):
            ui.label("Board").classes("text-xs text-gray-400")
            self._w_input = ui.number(
                "W (mm)", value=self.board.get("width_mm"), min=10, step=1.0,
                format="%.1f").classes("w-24").props("dense")
            self._h_input = ui.number(
                "H (mm)", value=self.board.get("height_mm"), min=10, step=1.0,
                format="%.1f").classes("w-24").props("dense")
            self._auto_size = ui.switch(
                "Auto size", value=self.board.get("size_search", True),
            ).props("dense")
            self._auto_size.tooltip(
                "On: the placer searches board sizes around the leaf areas. "
                "Off: it must fit your exact W x H."
            )

    def _render_leaf(self, leaf: LeafEntry) -> None:
        is_real_leaf = leaf.label not in ("Board-level", "(root sheet)")
        header = f"{leaf.label} [{len(leaf.refs)}]"
        with ui.expansion(header, icon="account_tree").classes("w-full"):
            if is_real_leaf:
                is_back = leaf.label in self.holder.backside_through_hole_overrides
                ui.switch(
                    "Backside THT anchor (SMT leaves may stack on top)",
                    value=is_back,
                    on_change=lambda e, s=leaf.label: _set_backside_through_hole(
                        self.holder, s, bool(e.value)),
                ).props("dense")
            with ui.grid(columns="90px 70px 130px 160px 100px 90px").classes(
                    "w-full gap-1 p-2 items-center text-xs"):
                for col in ("Ref", "Kind", "Anchor", "Value", "Rotation",
                            "Thermal"):
                    ui.label(col).classes("font-bold text-gray-300")
                for ref in leaf.refs:
                    self._render_row(ref)

    def _render_row(self, ref: str) -> None:
        override = self.holder.component_zone_overrides.get(ref)
        target = _anchor_target(override)
        value = _anchor_value(override, target)
        rotation = (override or {}).get("rotation")

        ui.label(ref).classes("font-mono")
        ui.label(_classify(ref)).classes("text-gray-400")

        anchor_select = ui.select(options=ANCHOR_CHOICES, value=target) \
            .props("dense options-dense")
        value_select = ui.select(
            options=ANCHOR_VALUES.get(target, []), value=value,
        ).props("dense options-dense")
        if target == "none":
            value_select.disable()

        def _on_anchor_change(e):
            new_target = e.value or "none"
            value_select.set_options(ANCHOR_VALUES.get(new_target, []), value=None)
            if new_target == "none":
                value_select.disable()
                _set_anchor(self.holder, ref, "none", None)
            else:
                value_select.enable()
                # Clear the PREVIOUS target's override now (the new one is
                # written once a value is picked): the UI shows the new target
                # with an empty value, so leaving e.g. the old {"edge":"left"}
                # in the holder would silently re-pin the part to the left
                # edge on Apply & re-place.
                _set_anchor(self.holder, ref, "none", None)

        anchor_select.on("update:model-value", _on_anchor_change)
        value_select.on(
            "update:model-value",
            lambda e: _set_anchor(self.holder, ref,
                                  anchor_select.value or "none", e.value),
        )

        rot_input = ui.number(
            value=float(rotation) if rotation is not None else None,
            min=0.0, max=360.0, step=90.0, format="%g",
        ).props("dense clearable").classes("w-full")
        rot_input.on(
            "update:model-value",
            lambda e: _set_rotation(
                self.holder, ref,
                None if e.value in (None, "") else float(e.value)),
        )

        ui.switch(
            value=ref in self.holder.thermal_ref_overrides,
            on_change=lambda e: _set_thermal(self.holder, ref, bool(e.value)),
        ).props("dense")

    # -- Apply ------------------------------------------------------------------

    def _build_slot(self) -> dict[str, Any]:
        slot: dict[str, Any] = {
            "component_zones": {
                ref: dict(spec)
                for ref, spec in self.holder.component_zone_overrides.items()
            },
            "thermal_refs": sorted(self.holder.thermal_ref_overrides),
            "backside_through_hole_leaves": sorted(
                self.holder.backside_through_hole_overrides),
        }
        w = self._w_input.value
        h = self._h_input.value
        auto = bool(self._auto_size.value)
        if w and h:
            slot["board"] = {"width_mm": float(w), "height_mm": float(h),
                             "size_search": auto}
        return slot

    def _confirm_apply(self) -> None:
        if self.is_run_active():
            ui.notify("A run is already in progress.", color="warning")
            return
        with ui.dialog() as dialog, ui.card():
            ui.label("Apply placement rules and re-place the board?") \
                .classes("text-sm font-medium")
            ui.label(
                "Re-runs place + route + fab through the build queue "
                "(takes minutes). No AI cost: the schematic is unchanged."
            ).classes("text-xs").style("color:#94a3b8")
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=dialog.close).props("flat")

                def _go():
                    dialog.close()
                    self._apply()

                ui.button("Apply & re-place", color="primary", on_click=_go)
        dialog.open()

    def _apply(self) -> None:
        slot = self._build_slot()
        ok, out = session.commit_slot(self.ws, "placement", slot)
        if not ok:
            errs = "; ".join(str(e) for e in (out.get("errors") or [])) or "rejected"
            ui.notify(f"Placement rules rejected: {errs[:300]}", color="negative")
            return
        for warning in out.get("warnings") or []:
            ui.notify(warning, color="warning")
        self._set_status("rules saved; re-placing…")
        self.on_rebuild()

    def _set_status(self, text: str) -> None:
        if self._status is not None:
            self._status.set_text(text)
