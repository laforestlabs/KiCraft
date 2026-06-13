"""Emit `<PROJECT>.kicad_pro` (KiCad project file).

The contract doc §3 only requires:
- `board.design_settings.rules` with the JLCPCB-friendly floor.
- `net_settings.classes` with at least Default + Power.

Anything else is GUI state; KiCad fills in defaults if absent. Keeping the
file minimal makes the emitter easy to read and avoids drift with KiCad
version updates.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..models import Architecture


# Floors sit at 0.153 mm, just ABOVE the true 6 mil (0.1524 mm) fab limit:
# 0.15 is under the limit outright, and 0.1524 itself is a rounding trap --
# the DSN export rounds to whole µm, so 0.1524 becomes 152 µm < 152.4 µm and
# every minimum-width track or clearance fails DRC. 0.153 rounds to 153 µm.
# (Same reasoning as freerouting_min_clearance_mm / _fine_pitch_track_mm in
# autoplacer/config.py.)
DEFAULT_RULES = {
    "min_clearance": 0.153,
    "min_track_width": 0.153,
    "min_via_diameter": 0.508,
    "min_via_annular_width": 0.127,
    "min_hole_to_hole": 0.127,
    "min_copper_edge_clearance": 0.381,
    "min_silk_clearance": 0.0,
    "min_text_height": 0.8,
    "min_text_thickness": 0.08,
    "use_height_for_length_calcs": True,
}

DEFAULT_NETCLASS = {
    "bus_width": 12,
    "clearance": 0.153,
    "diff_pair_gap": 0.25,
    "diff_pair_via_gap": 0.25,
    "diff_pair_width": 0.2,
    "line_style": 0,
    "microvia_diameter": 0.3,
    "microvia_drill": 0.1,
    "name": "Default",
    "pcb_color": "rgba(0, 0, 0, 0.000)",
    "priority": 2147483647,
    "schematic_color": "rgba(0, 0, 0, 0.000)",
    "track_width": 0.2,
    "via_diameter": 0.6,
    "via_drill": 0.3,
    "wire_width": 6,
}

# Power differs from Default ONLY in conductor sizing (wider tracks/vias for
# current capacity). Clearance deliberately matches Default: clearance is a
# voltage/fab parameter and ~6 mil is ample at logic-level voltages, while a
# wider Power clearance starves routing and zone fill around fine-pitch parts
# whose pad gap is below it (a 0.8 mm-pitch LGA has 0.28 mm gaps; the old
# 0.3 mm rule made its supply pads unreachable and un-DRC-able).
POWER_NETCLASS = {
    "bus_width": 12,
    "clearance": 0.153,
    "diff_pair_gap": 0.25,
    "diff_pair_via_gap": 0.25,
    "diff_pair_width": 0.2,
    "line_style": 0,
    "microvia_diameter": 0.3,
    "microvia_drill": 0.1,
    "name": "Power",
    "pcb_color": "rgba(0, 0, 0, 0.000)",
    "priority": 0,
    "schematic_color": "rgba(0, 0, 0, 0.000)",
    "track_width": 0.5,
    "via_diameter": 0.8,
    "via_drill": 0.4,
    "wire_width": 6,
}


def write_kicad_pro(
    project_dir: Path,
    project_stem: str,
    architecture: Architecture,
) -> Path:
    """Write `<project_stem>.kicad_pro` to project_dir. Returns the path."""
    out = project_dir / f"{project_stem}.kicad_pro"
    netclass_patterns = [
        {"netclass": "Power", "pattern": net} for net in architecture.power_nets
    ]
    body = {
        "board": {
            "design_settings": {
                "rules": dict(DEFAULT_RULES),
                "meta": {"version": 2},
                "track_widths": [0.0, 0.2, 0.5],
                "via_dimensions": [
                    {"diameter": 0.0, "drill": 0.0},
                    {"diameter": 0.6, "drill": 0.3},
                    {"diameter": 0.8, "drill": 0.4},
                ],
            },
            "layer_presets": [],
            "viewports": [],
        },
        "boards": [],
        "cvpcb": {"equivalence_files": []},
        "erc": {"meta": {"version": 0}, "rule_severities": {}},
        "libraries": {"pinned_footprint_libs": [], "pinned_symbol_libs": []},
        "meta": {"filename": f"{project_stem}.kicad_pro", "version": 1},
        "net_settings": {
            "classes": [dict(DEFAULT_NETCLASS), dict(POWER_NETCLASS)],
            "meta": {"version": 3},
            "net_colors": None,
            "netclass_assignments": None,
            "netclass_patterns": netclass_patterns,
        },
        "pcbnew": {
            "last_paths": {
                "gencad": "",
                "idf": "",
                "netlist": "",
                "plot": "",
                "pos_files": "",
                "specctra_dsn": "",
                "step": "",
                "svg": "",
                "vrml": "",
            },
            "page_layout_descr_file": "",
        },
        "schematic": {
            "annotate_start_num": 0,
            "drawing": {
                "default_line_thickness": 6.0,
                "default_text_size": 50.0,
            },
            "legacy_lib_dir": "",
            "legacy_lib_list": [],
            "meta": {"version": 1},
            "net_format_name": "",
            "page_layout_descr_file": "",
            "plot_directory": "",
            "spice_adjust_passive_values": False,
            "spice_external_command": "spice \"%I\"",
            "subpart_first_id": 65,
            "subpart_id_separator": 0,
        },
        "sheets": [],
        "text_variables": {},
    }
    out.write_text(json.dumps(body, indent=2) + "\n")
    return out
