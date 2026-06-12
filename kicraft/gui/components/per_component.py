"""Per-component placement rules UI.

Lists every component in the project grouped by leaf (schematic sheet),
with editable anchor/rotation/overhang/thermal overrides. Writes back
to the project's autoplacer.json file via a diff-confirm modal.

The UI-free data layer (load/stage/diff/write) lives in
``kicraft.layout_editor.rules``; this module only renders it.
"""

from __future__ import annotations

import json
from typing import Any

from nicegui import ui

from kicraft.layout_editor.rules import (
    ANCHOR_CHOICES,
    ANCHOR_VALUES,
    LeafEntry,
    PerComponentData,
    _anchor_target,
    _anchor_value,
    _build_updated_config,
    _classify,
    _diff_dicts,
    _has_override,
    _load_from_schematic,
    _load_overrides_into_state,
    _project_config_path,
    _read_project_config,
    _set_anchor,
    _set_backside_through_hole,
    _set_rotation,
    _set_thermal,
    _write_config_with_backup,
)


def per_component_panel(state) -> None:
    _load_overrides_into_state(state)

    ui.label("Per-Component Placement Rules").classes("text-lg font-bold mb-2")
    ui.label(
        "Per-component overrides written to the project's autoplacer.json. "
        "Anchor pins a part to an edge, corner, or board region; rotation "
        "overrides the auto-orient for edge-pinned parts; overhang applies "
        "to connectors; thermal adds a keep-away radius. Each leaf header "
        "also exposes sheet-level options (e.g. backside-THT anchor for "
        "battery holders or screw terminals)."
    ).classes("text-sm text-gray-400 mb-3")

    container = ui.column().classes("w-full gap-3")
    filter_state: dict[str, Any] = {
        "query": "",
        "only_overridden": False,
        "expand_all": False,
    }

    def _refresh_and_render():
        container.clear()
        schematic_file = state.strategy.get("schematic_file", "")
        data = _load_from_schematic(state.project_root, schematic_file)
        with container:
            _render_toolbar(state, filter_state, lambda: _refresh_and_render())
            if data.error:
                ui.label(data.error).classes("text-red-400")
                return
            _render_leaves(state, data, filter_state)

    _refresh_and_render()


def _render_toolbar(state, filter_state: dict[str, Any], refresh_cb) -> None:
    with ui.row().classes("w-full items-center gap-2 mb-2"):
        ui.button(
            "Refresh from schematic",
            icon="refresh",
            on_click=refresh_cb,
        ).props("flat dense")

        ui.button(
            "Save to project JSON…",
            icon="save",
            on_click=lambda: _open_save_dialog(state),
        ).props("dense")

        ui.space()

        search = ui.input(placeholder="filter ref/leaf…").props("dense clearable")
        search.classes("w-48")

        def _on_search(e):
            filter_state["query"] = (e.value or "").strip().lower()
            refresh_cb()

        search.on("update:model-value", _on_search)

        ui.switch(
            "Only overridden",
            value=filter_state["only_overridden"],
            on_change=lambda e: (
                filter_state.update({"only_overridden": bool(e.value)}),
                refresh_cb(),
            ),
        ).props("dense")


def _render_leaves(state, data: PerComponentData, filter_state: dict[str, Any]) -> None:
    signal_flow = state.placement_config.get("signal_flow_order", [])

    def _leaf_sort_key(leaf: LeafEntry):
        if leaf.leader_ref and leaf.leader_ref in signal_flow:
            return (0, signal_flow.index(leaf.leader_ref))
        return (1, leaf.label.lower())

    leaves = sorted(data.leaves, key=_leaf_sort_key)

    # Board-level bucket: overridden refs that are not in any leaf
    schematic_refs = data.all_refs
    overridden_not_in_leaf = [
        r
        for r in sorted(state.component_zone_overrides.keys())
        if r not in schematic_refs
    ]
    if overridden_not_in_leaf:
        leaves.append(
            LeafEntry(
                label="Board-level",
                leader_ref=None,
                refs=overridden_not_in_leaf,
                instance_path="/",
            )
        )

    q = filter_state["query"]
    only_overridden = filter_state["only_overridden"]

    for leaf in leaves:
        refs = leaf.refs
        if only_overridden:
            refs = [r for r in refs if _has_override(state, r)]
        if q:
            if q in leaf.label.lower():
                pass  # keep all refs
            else:
                refs = [r for r in refs if q in r.lower()]
        if not refs:
            continue

        override_count = sum(1 for r in leaf.refs if _has_override(state, r))
        header_bits = [
            leaf.label,
            f"[{len(leaf.refs)}]",
        ]
        if leaf.leader_ref:
            header_bits.insert(1, f"· {leaf.leader_ref}")
        if override_count:
            header_bits.append(f"· {override_count} override{'s' if override_count != 1 else ''}")

        # The backside-THT toggle lives on the leaf header, not on
        # individual components, because it is a *sheet*-level override.
        # Skip synthetic buckets (board-level, root) -- they are not
        # subcircuit leaves and have no sheet name to register against.
        is_real_leaf = leaf.label not in ("Board-level", "(root sheet)")
        sheet_name = leaf.label if is_real_leaf else None
        if sheet_name and sheet_name in state.backside_through_hole_overrides:
            header_bits.append("· backside-THT")

        with ui.expansion(" ".join(header_bits), icon="account_tree").classes("w-full"):
            if sheet_name:
                _render_leaf_options(state, sheet_name)
            _render_leaf_table(state, refs)


def _render_leaf_options(state, sheet_name: str) -> None:
    """Sheet-level placement overrides shown above the per-component table.

    Currently exposes the backside-through-hole flag; future leaf-level
    knobs (rotation lock, side-bias, edge-only) can land in this row
    without touching the per-component grid below."""
    is_back = sheet_name in state.backside_through_hole_overrides
    with ui.row().classes("w-full items-center gap-3 px-2 pb-2"):
        ui.icon("layers").classes("text-gray-400")
        ui.label("Sheet-level options:").classes("text-xs text-gray-300")
        switch = ui.switch(
            "Backside THT anchor (allow SMT-front leaves to stack on top)",
            value=is_back,
            on_change=lambda e: _set_backside_through_hole(
                state, sheet_name, bool(e.value)
            ),
        ).props("dense")
        switch.tooltip(
            "Treat this leaf as a backside through-hole anchor (battery "
            "holder, screw terminal). Suppresses its F.Cu-shadow pads in "
            "the same-layer-overlap check so opposite-side SMT leaves "
            "may stack on top. Persists as "
            "parent_placement.backside_through_hole_leaves in the "
            "project autoplacer.json. Don't enable for leaves with real "
            "F.Cu routing (SMT-heavy regulators) -- it would re-introduce "
            "continuous-F.Cu stamping shorts."
        )


def _render_leaf_table(state, refs: list[str]) -> None:
    with ui.grid(columns="90px 70px 120px 150px 90px 110px 90px 90px 90px").classes(
        "w-full gap-1 p-2 items-center text-xs"
    ):
        for col in ("Ref", "Kind", "Anchor", "Value", "Rotation", "Thermal", "Side", "Lock"):
            ui.label(col).classes("font-bold text-gray-300")

        for ref in refs:
            _render_component_row(state, ref)


def _render_component_row(state, ref: str) -> None:
    kind = _classify(ref)
    override = state.component_zone_overrides.get(ref)
    target = _anchor_target(override)
    value = _anchor_value(override, target)
    rotation = (override or {}).get("rotation")
    thermal = ref in state.thermal_ref_overrides

    ui.label(ref).classes("font-mono")
    ui.label(kind).classes("text-gray-400")

    anchor_select = ui.select(
        options=ANCHOR_CHOICES,
        value=target,
    ).props("dense options-dense")

    value_select = ui.select(
        options=ANCHOR_VALUES.get(target, []),
        value=value,
    ).props("dense options-dense")
    if target == "none":
        value_select.disable()

    def _on_anchor_change(e):
        new_target = e.value or "none"
        new_options = ANCHOR_VALUES.get(new_target, [])
        value_select.set_options(new_options, value=None)
        if new_target == "none":
            value_select.disable()
            _set_anchor(state, ref, "none", None)
        else:
            value_select.enable()
            # Don't write until user picks a value

    anchor_select.on("update:model-value", _on_anchor_change)

    def _on_value_change(e):
        new_target = anchor_select.value or "none"
        _set_anchor(state, ref, new_target, e.value)

    value_select.on("update:model-value", _on_value_change)

    rot_input = ui.number(
        value=float(rotation) if rotation is not None else None,
        min=0.0,
        max=360.0,
        step=90.0,
        format="%g",
    ).props("dense clearable").classes("w-full")

    def _on_rot_change(e):
        v = e.value
        _set_rotation(state, ref, None if v in (None, "") else float(v))

    rot_input.on("update:model-value", _on_rot_change)

    thermal_switch = ui.switch(
        value=thermal,
        on_change=lambda e: _set_thermal(state, ref, bool(e.value)),
    ).props("dense")

    side_select = ui.select(
        options=["auto", "F.Cu", "B.Cu"], value="auto"
    ).props("dense options-dense").tooltip("Future — not yet wired in the engine")
    side_select.disable()

    lock_switch = ui.switch(value=False).props("dense").tooltip(
        "Future — not yet wired in the engine"
    )
    lock_switch.disable()


# ---------------------------------------------------------------------------
# Save dialog
# ---------------------------------------------------------------------------


def _open_save_dialog(state) -> None:
    path = _project_config_path(state.project_root)
    before = _read_project_config(path)
    after = _build_updated_config(state)
    changes = _diff_dicts(before, after)

    with ui.dialog() as dialog, ui.card().classes("w-[720px]"):
        ui.label("Save per-component overrides").classes("text-lg font-bold")
        ui.label(str(path)).classes("text-xs text-gray-400 font-mono mb-2")

        if not changes:
            ui.label("No changes to save.").classes("text-gray-400")
        else:
            with ui.column().classes("w-full gap-1 max-h-96 overflow-auto"):
                for change, key, before_val, after_val in changes:
                    _render_diff_row(change, key, before_val, after_val)

        with ui.row().classes("w-full justify-end gap-2 mt-3"):
            ui.button("Cancel", on_click=dialog.close).props("flat")
            if changes:
                def _confirm():
                    try:
                        _write_config_with_backup(path, after)
                    except OSError as exc:
                        ui.notify(f"Write failed: {exc}", type="negative")
                        return
                    dialog.close()
                    ui.notify(
                        f"Wrote {len(changes)} change(s) to {path.name}",
                        type="positive",
                    )

                ui.button("Write file", on_click=_confirm, color="primary")

    dialog.open()


def _render_diff_row(change: str, key: str, before_val: Any, after_val: Any) -> None:
    colors = {"add": "text-green-400", "remove": "text-red-400", "change": "text-amber-400"}
    glyphs = {"add": "+", "remove": "−", "change": "~"}
    with ui.row().classes(f"w-full items-start gap-2 text-xs font-mono {colors.get(change, '')}"):
        ui.label(glyphs.get(change, "?")).classes("w-4")
        ui.label(key).classes("w-64 truncate")
        if change == "change":
            ui.label(f"{json.dumps(before_val)} → {json.dumps(after_val)}").classes(
                "flex-1 break-all"
            )
        elif change == "add":
            ui.label(json.dumps(after_val)).classes("flex-1 break-all")
        else:
            ui.label(json.dumps(before_val)).classes("flex-1 break-all")
