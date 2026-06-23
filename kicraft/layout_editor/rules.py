"""Per-component placement rules: UI-free data layer.

Lists every component in the project grouped by leaf (schematic sheet)
and manages the anchor/rotation/thermal/backside-THT overrides that
persist to the project's autoplacer.json. The rendering half lives in
the host app (``kicraft.server`` rules panel); this module owns
load/stage/diff/write so any host can drive it.

The ``state`` argument throughout is duck-typed: any object with
``project_root`` plus the mutable staging fields
``component_zone_overrides`` (dict), ``thermal_ref_overrides`` (set),
``backside_through_hole_overrides`` (set), and
``per_component_loaded`` (bool) works.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kicraft.autoplacer.brain.hierarchy_parser import parse_hierarchy
from kicraft.autoplacer.config import discover_project_config

# Anchor vocabulary lives with the state schema (the placement section
# validates against it); re-exported here for the UI layers.
from kicraft.design.models import (  # noqa: F401  (re-export)
    PLACEMENT_ANCHOR_VALUES as ANCHOR_VALUES,
)

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
