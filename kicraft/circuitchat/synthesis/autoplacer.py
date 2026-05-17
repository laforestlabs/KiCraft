"""Emit `<PROJECT>_autoplacer.json` from architecture + BOM slots.

Straight serialization of the state into the schema KiCraft already reads
(see `docs/circuitchat_schematic_prompt.md` §4 and the existing
`LLUPS_autoplacer.json` for shape).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..models import Architecture, BOM


def write_autoplacer_json(
    project_dir: Path,
    project_stem: str,
    architecture: Architecture,
    bom: BOM,
) -> Path:
    """Write `<project_stem>_autoplacer.json` to project_dir. Returns the path."""
    out = project_dir / f"{project_stem}_autoplacer.json"
    body: dict[str, object] = {
        "project_name": project_stem,
        "pcb_file": f"{project_stem}.kicad_pcb",
        "power_nets": sorted(set(architecture.power_nets)),
    }
    if bom.ic_groups:
        body["ic_groups"] = {ic: list(members) for ic, members in bom.ic_groups.items()}
    if bom.group_labels:
        body["group_labels"] = dict(bom.group_labels)
    if bom.thermal_refs:
        body["thermal_refs"] = list(bom.thermal_refs)
    if bom.signal_flow_order:
        body["signal_flow_order"] = list(bom.signal_flow_order)
    if bom.component_zones:
        body["component_zones"] = {
            ref: dict(spec) for ref, spec in bom.component_zones.items()
        }
    body["enable_board_size_search"] = True
    out.write_text(json.dumps(body, indent=2) + "\n")
    return out
