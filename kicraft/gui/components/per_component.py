"""Per-component placement rules UI.

Lists every component in the project grouped by leaf (schematic sheet),
with editable anchor/rotation/overhang/thermal overrides. Writes back
to the project's autoplacer.json file via a diff-confirm modal.
"""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nicegui import ui

from kicraft.autoplacer.brain.hierarchy_parser import parse_hierarchy
from kicraft.autoplacer.config import discover_project_config


ANCHOR_VALUES: dict[str, list[str]] = {
    "edge": ["left", "right", "top", "bottom"],
    "corner": ["top-left", "top-right", "bottom-left", "bottom-right"],
    "zone": [
        "center", "top", "bottom", "left", "right",
        "center-top", "center-bottom", "center-left", "center-right",
        "top-left", "top-right", "bottom-left", "bottom-right",
    ],
}

ANCHOR_CHOICES = ["none", "edge", "corner", "zone"]

_KIND_PREFIXES: list[tuple[str, str]] = [
    ("BT", "battery"),
    ("RT", "passive"),
    ("J", "connector"),
    ("H", "mh"),
    ("U", "ic"),
    ("R", "passive"),
    ("C", "passive"),
    ("L", "passive"),
    ("D", "passive"),
    ("F", "passive"),
    ("Q", "passive"),
]


def _classify(ref: str) -> str:
    r = ref.upper()
    for prefix, kind in _KIND_PREFIXES:
        if r.startswith(prefix):
            return kind
    return "misc"


@dataclass
class LeafEntry:
    label: str
    leader_ref: str | None
    refs: list[str]
    instance_path: str


@dataclass
class PerComponentData:
    leaves: list[LeafEntry] = field(default_factory=list)
    board_level: list[str] = field(default_factory=list)
    ungrouped: list[str] = field(default_factory=list)
    all_refs: set[str] = field(default_factory=set)
    error: str | None = None


def _leaf_leader(refs: list[str]) -> str | None:
    for r in refs:
        if r.upper().startswith(("U", "BT")):
            return r
    return refs[0] if refs else None


def _load_from_schematic(project_root: Path, schematic_file: str | None) -> PerComponentData:
    data = PerComponentData()

    sch_override: Path | None = None
    if schematic_file:
        candidate = (project_root / schematic_file).resolve()
        if candidate.exists():
            sch_override = candidate

    try:
        graph = parse_hierarchy(project_root, sch_override)
    except Exception as exc:
        data.error = f"Failed to parse schematic: {exc}"
        return data

    assigned: set[str] = set()
    for node in graph.leaf_nodes():
        refs = [r for r in node.definition.component_refs if r not in assigned]
        if not refs:
            continue
        assigned.update(refs)
        leader = _leaf_leader(refs)
        label = node.id.sheet_name or leader or "(leaf)"
        data.leaves.append(
            LeafEntry(
                label=label,
                leader_ref=leader,
                refs=sorted(refs),
                instance_path=node.id.instance_path or "/",
            )
        )

    root_only = [r for r in graph.root.definition.component_refs if r not in assigned]
    if root_only:
        assigned.update(root_only)
        data.leaves.append(
            LeafEntry(
                label="(root sheet)",
                leader_ref=None,
                refs=sorted(root_only),
                instance_path="/",
            )
        )

    data.all_refs = set(assigned)
    return data


def _project_config_path(project_root: Path) -> Path:
    found = discover_project_config(project_root)
    if found:
        return found
    return project_root / f"{project_root.name}_autoplacer.json"


def _read_project_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _load_overrides_into_state(state) -> None:
    """Seed override staging dicts from the project JSON once per session."""
    if state.per_component_loaded:
        return
    cfg_path = _project_config_path(state.project_root)
    cfg = _read_project_config(cfg_path)

    zones = cfg.get("component_zones", {})
    if isinstance(zones, dict):
        for ref, spec in zones.items():
            if isinstance(spec, dict):
                state.component_zone_overrides[ref] = dict(spec)
            elif isinstance(spec, str):
                if ":" in spec:
                    tgt, val = spec.split(":", 1)
                else:
                    tgt, val = "edge", spec
                state.component_zone_overrides[ref] = {tgt: val}

    thermals = cfg.get("thermal_refs", [])
    if isinstance(thermals, list):
        state.thermal_ref_overrides.update(str(r) for r in thermals)

    backside = (
        cfg.get("parent_placement", {}).get("backside_through_hole_leaves", [])
    )
    if isinstance(backside, list):
        state.backside_through_hole_overrides.update(str(s) for s in backside)

    state.per_component_loaded = True


def _anchor_target(override: dict[str, Any] | None) -> str:
    if not override:
        return "none"
    for key in ("edge", "corner", "zone"):
        if key in override:
            return key
    return "none"


def _anchor_value(override: dict[str, Any] | None, target: str) -> str | None:
    if not override or target == "none":
        return None
    val = override.get(target)
    return str(val) if val is not None else None


def _set_anchor(state, ref: str, target: str, value: str | None) -> None:
    current = dict(state.component_zone_overrides.get(ref, {}))
    # Preserve non-anchor keys (e.g. "rotation")
    for k in ("edge", "corner", "zone"):
        current.pop(k, None)
    if target != "none" and value:
        current[target] = value
    if current:
        state.component_zone_overrides[ref] = current
    else:
        state.component_zone_overrides.pop(ref, None)


def _set_rotation(state, ref: str, rotation: float | None) -> None:
    current = dict(state.component_zone_overrides.get(ref, {}))
    if rotation is None:
        current.pop("rotation", None)
    else:
        current["rotation"] = float(rotation)
    if current:
        state.component_zone_overrides[ref] = current
    else:
        state.component_zone_overrides.pop(ref, None)


def _set_thermal(state, ref: str, enabled: bool) -> None:
    if enabled:
        state.thermal_ref_overrides.add(ref)
    else:
        state.thermal_ref_overrides.discard(ref)


def _set_backside_through_hole(state, sheet_name: str, enabled: bool) -> None:
    """Toggle a leaf as backside-through-hole-anchor for SMT stacking.

    Persisted under ``parent_placement.backside_through_hole_leaves``;
    keyed by sheet name so the override is portable across projects."""
    if enabled:
        state.backside_through_hole_overrides.add(sheet_name)
    else:
        state.backside_through_hole_overrides.discard(sheet_name)


def _has_override(state, ref: str) -> bool:
    return (
        ref in state.component_zone_overrides
        or ref in state.thermal_ref_overrides
    )


def _build_updated_config(state) -> dict[str, Any]:
    cfg = _read_project_config(_project_config_path(state.project_root))

    if state.component_zone_overrides:
        cfg["component_zones"] = {
            ref: dict(spec) for ref, spec in state.component_zone_overrides.items()
        }
    else:
        cfg.pop("component_zones", None)

    # parent_overhang_mm was a per-ref escape hatch for the bug fixed by
    # the Pad.size_mm + Component.physical_bbox refactor. Strip it from
    # any persisted project config so old overrides don't linger.
    cfg.pop("parent_overhang_mm", None)

    if state.thermal_ref_overrides:
        cfg["thermal_refs"] = sorted(state.thermal_ref_overrides)
    else:
        cfg.pop("thermal_refs", None)

    # parent_placement.backside_through_hole_leaves carries the THT-anchor
    # stacking override. Preserve any other parent_placement keys (e.g.
    # candidate_search) the user has hand-edited; only rewrite the
    # backside list. When empty, drop the key (don't strip the whole
    # parent_placement block).
    parent_placement = dict(cfg.get("parent_placement", {}) or {})
    if state.backside_through_hole_overrides:
        parent_placement["backside_through_hole_leaves"] = sorted(
            state.backside_through_hole_overrides
        )
    else:
        parent_placement.pop("backside_through_hole_leaves", None)
    if parent_placement:
        cfg["parent_placement"] = parent_placement
    else:
        cfg.pop("parent_placement", None)

    return cfg


def _diff_dicts(before: dict[str, Any], after: dict[str, Any]) -> list[tuple[str, str, Any, Any]]:
    """Return list of (change, key, before_val, after_val) for tracked keys."""
    rows: list[tuple[str, str, Any, Any]] = []
    for top_key in ("component_zones", "thermal_refs"):
        b = before.get(top_key)
        a = after.get(top_key)
        if b == a:
            continue
        if top_key == "thermal_refs":
            b_set = set(b or [])
            a_set = set(a or [])
            for r in sorted(a_set - b_set):
                rows.append(("add", f"{top_key}: {r}", None, True))
            for r in sorted(b_set - a_set):
                rows.append(("remove", f"{top_key}: {r}", True, None))
            continue
        b_map = b or {}
        a_map = a or {}
        for ref in sorted(set(a_map.keys()) | set(b_map.keys())):
            if ref not in b_map:
                rows.append(("add", f"{top_key}[{ref}]", None, a_map[ref]))
            elif ref not in a_map:
                rows.append(("remove", f"{top_key}[{ref}]", b_map[ref], None))
            elif a_map[ref] != b_map[ref]:
                rows.append(("change", f"{top_key}[{ref}]", b_map[ref], a_map[ref]))

    b_back = set(
        (before.get("parent_placement", {}) or {}).get(
            "backside_through_hole_leaves", []
        )
        or []
    )
    a_back = set(
        (after.get("parent_placement", {}) or {}).get(
            "backside_through_hole_leaves", []
        )
        or []
    )
    if b_back != a_back:
        for s in sorted(a_back - b_back):
            rows.append(("add", f"backside_through_hole_leaves: {s}", None, True))
        for s in sorted(b_back - a_back):
            rows.append(("remove", f"backside_through_hole_leaves: {s}", True, None))
    return rows


def _write_config_with_backup(path: Path, cfg: dict[str, Any]) -> None:
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


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
