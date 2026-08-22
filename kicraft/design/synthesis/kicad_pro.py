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

from kicraft.autoplacer.fab_profile import (
    NETCLASS_CLEARANCE_MM,
    fab_floors,
    fanout_via,
)

from ..models import Architecture


# The DRC *floors* mirror the fab capability profile (autoplacer/fab_profile.py),
# which is the single source of truth for what JLC can actually build. They used
# to encode OSH Park's 6 mil (0.1524 mm) limit at 0.153 -- the +0.5 µm dodged the
# router exchange input's whole-µm rounding, since 0.1524 exports as 152 µm and fails its own rule.
# 0.127 mm (5 mil) is both inside JLC's 2-layer capability with margin and
# exactly 127 µm, so no rounding dodge is needed. Lowering min_via_diameter to
# the 0.4/0.2 fanout class is what makes a dog-bone escape out of a fine-pitch
# inner ring legal at all; the netclasses below are unchanged, so ordinary
# routing still happens at 0.2 mm track / 0.153 mm clearance / 0.6 mm via.
_FLOORS = fab_floors()
_FANOUT_VIA_DIA, _FANOUT_VIA_DRILL = fanout_via()

DEFAULT_RULES = {
    "min_clearance": _FLOORS["clearance_mm"],
    "min_track_width": _FLOORS["track_mm"],
    "min_via_diameter": _FANOUT_VIA_DIA,
    "min_via_annular_width": round((_FANOUT_VIA_DIA - _FANOUT_VIA_DRILL) / 2.0, 4),
    # KiCad's "minimum through hole" gates VIA drills as well as PTH pads, and
    # its 0.3 mm default is the netclass via's drill -- it would fail every
    # fanout via by construction. Library PTH pads are held to the fab floor
    # separately by validate-part/add-part (check 6).
    "min_through_hole_diameter": _FANOUT_VIA_DRILL,
    "min_hole_to_hole": 0.127,
    # 0.2 mm = JLCPCB's routed board-edge-to-copper minimum. The old 0.381 mm
    # (15 mil) was overly conservative and failed boards whose routed tracks sit
    # 0.2-0.38 mm from the edge -- fab-fine copper. KiCad Routing Tools 1.9.0 ignores the
    # router exchange input boundary for wires, so this gate threshold (not a router setting) is the
    # only lever for those track-near-edge cases; genuinely-too-close copper
    # (< 0.2 mm or past the edge) still fails. Connector pad clearance is kept by
    # connector_edge_pad_clearance_mm in _repair_parent_outline.
    "min_copper_edge_clearance": 0.2,
    "min_silk_clearance": 0.0,
    "min_text_height": 0.8,
    "min_text_thickness": 0.08,
    "use_height_for_length_calcs": True,
}

DEFAULT_NETCLASS = {
    "bus_width": 12,
    "clearance": NETCLASS_CLEARANCE_MM,
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
    "clearance": NETCLASS_CLEARANCE_MM,
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
                    # The escape/dog-bone class. Present so a stamped fanout via
                    # round-trips through the router exchange and shows up in KiCad's via
                    # dropdown; KiCad Routing Tools still places the netclass via, which
                    # each class names in its own (use_via ...).
                    {"diameter": _FANOUT_VIA_DIA, "drill": _FANOUT_VIA_DRILL},
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
