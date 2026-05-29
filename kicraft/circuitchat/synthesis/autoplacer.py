"""Emit `<PROJECT>_autoplacer.json` from architecture + BOM slots.

Straight serialization of the state into the schema KiCraft already reads
(see `docs/circuitchat_schematic_prompt.md` §4 and the existing
`LLUPS_autoplacer.json` for shape).

When the architecture contains library-backed sheets, the synthesis
stage passes in the merged library fragments and a ``library_leaves``
map; both flow into the emitted JSON.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import Architecture, BOM


def write_autoplacer_json(
    project_dir: Path,
    project_stem: str,
    architecture: Architecture,
    bom: BOM,
    *,
    library_fragments: dict[str, Any] | None = None,
    library_leaves: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Write `<project_stem>_autoplacer.json` to project_dir. Returns the path.

    Args:
        library_fragments: dict already merged from every library leaf's
            ``autoplacer_fragment.json`` (renumbered). Keys like
            ``ic_groups``, ``thermal_refs`` are unioned with the
            project's BOM-derived values.
        library_leaves: ``{sheet_name: InstalledLeaf.to_library_leaves_entry()}``
            written as a top-level audit record.
    """
    out = project_dir / f"{project_stem}_autoplacer.json"
    body: dict[str, object] = {
        "project_name": project_stem,
        "pcb_file": f"{project_stem}.kicad_pcb",
        "power_nets": sorted(set(architecture.power_nets)),
    }

    # Start with BOM-derived values; fold in library fragments additively.
    ic_groups: dict[str, list[str]] = {
        ic: list(members) for ic, members in bom.ic_groups.items()
    }
    group_labels: dict[str, str] = dict(bom.group_labels)
    thermal_refs: list[str] = list(bom.thermal_refs)
    signal_flow_order: list[str] = list(bom.signal_flow_order)
    component_zones: dict[str, dict[str, str]] = {
        ref: dict(spec) for ref, spec in bom.component_zones.items()
    }

    if library_fragments:
        for k, v in (library_fragments.get("ic_groups") or {}).items():
            if k not in ic_groups:
                ic_groups[k] = list(v)
        for k, v in (library_fragments.get("group_labels") or {}).items():
            group_labels.setdefault(k, v)
        for r in library_fragments.get("thermal_refs", []) or []:
            if r not in thermal_refs:
                thermal_refs.append(r)
        for r in library_fragments.get("signal_flow_order", []) or []:
            if r not in signal_flow_order:
                signal_flow_order.append(r)
        for k, v in (library_fragments.get("component_zones") or {}).items():
            component_zones.setdefault(k, dict(v))

    if ic_groups:
        body["ic_groups"] = ic_groups
    if group_labels:
        body["group_labels"] = group_labels
    if thermal_refs:
        body["thermal_refs"] = thermal_refs
    if signal_flow_order:
        body["signal_flow_order"] = signal_flow_order
    if component_zones:
        body["component_zones"] = component_zones

    # Matrix/array placement hints. The autoplacer grids these members
    # programmatically (serpentine) instead of running force/SA over them.
    # autoplacer.json is the project config, so this key merges straight into
    # the solver cfg (see autoplacer/config.discover_project_config).
    arrays = [
        {
            "refs": list(spec.refs),
            "rows": spec.rows,
            "cols": spec.cols,
            "pitch_mm": spec.pitch_mm,
            "serpentine": spec.serpentine,
        }
        for spec in bom.arrays
    ]
    if arrays:
        body["arrays"] = arrays

    if library_leaves:
        body["library_leaves"] = library_leaves

    body["enable_board_size_search"] = True
    out.write_text(json.dumps(body, indent=2) + "\n")
    return out
