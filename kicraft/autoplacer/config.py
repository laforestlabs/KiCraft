"""Default configuration — project-agnostic placement/routing engine defaults.

Project-specific overrides (ic_groups, component_zones, etc.) live in a
per-project JSON file (e.g. ``LLUPS_autoplacer.json``).  Use
``discover_project_config()`` to locate it automatically, then
``load_project_config()`` to parse it.
"""

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = {
    # Trace widths (5 mil = 0.127mm)
    "signal_width_mm": 0.127,
    "power_width_mm": 0.127,
    # Via
    "via_drill_mm": 0.3,
    "via_size_mm": 0.6,
    # Placement clearance — minimum gap between component bounding boxes.
    # 2.5mm leaves room for vias/traces, reduces courtyard overlaps.
    "placement_clearance_mm": 2.5,
    # Power nets (common names — projects override with their own)
    "power_nets": set(),
    # Placement (spread components — room to route, no courtyard overlaps)
    "placement_grid_mm": 1.0,
    "edge_margin_mm": 6.0,
    "force_attract_k": 0.02,
    "force_repel_k": 200.0,
    "cooling_factor": 0.97,
    "sa_refine_enabled": True,
    "sa_refine_iterations": 1000,
    "sa_refine_initial_temp": 5.0,
    "sa_refine_cooling_rate": 0.995,
    "sa_refine_move_radius_mm": 2.0,
    "sa_refine_swap_probability": 0.3,
    "sa_refine_rotation_probability": 0.2,
    # Placement solver iterations
    "max_placement_iterations": 300,
    "placement_convergence_threshold": 0.5,
    "placement_score_every_n": 1,
    "intra_cluster_iters": 80,
    # Placement diversity: "cluster" (centroid-based) or "random" (uniform scatter).
    # MINOR mode always uses "cluster"; MAJOR uses "random" 50% of the time;
    # EXPLORE always uses "random".  Set by autoexperiment per mutation mode.
    "scatter_mode": "cluster",
    # Temperature reheat: at 50% of max_iterations, apply a random perturbation
    # kick to escape local minima. 0 = disabled, 0.1 = moderate, 0.3 = aggressive.
    "reheat_strength": 0.1,
    # Randomize IC-group internal layout (radius spread + angular shuffle).
    # True for MAJOR/EXPLORE, False for MINOR.
    "randomize_group_layout": False,
    # Courtyard overlap padding — extra margin (mm) added when scoring
    # courtyard overlaps.  Drives the optimizer to leave breathing room.
    "courtyard_padding_mm": 0.5,
    # Pad inset margin — minimum distance (mm) all electrical pads must be
    # inside the board Edge.Cuts boundary.  Pads outside are unfabricatable.
    "pad_inset_margin_mm": 0.3,
    # Edge jitter — maximum random displacement (mm) along the assigned edge
    # for edge-pinned components (connectors, mounting holes).  Provides
    # placement diversity across rounds while keeping components on edges.
    "edge_jitter_mm": 5.0,
    # Connector gap — spacing (mm) between connectors grouped on the same edge.
    "connector_gap_mm": 2.0,
    # Connector edge inset — distance (mm) from board edge to the nearest
    # edge of the connector body.  0 = flush, positive = inset, negative =
    # overhang.  Only applies to edge-pinned connectors.
    "connector_edge_inset_mm": 1.0,
    # Connector pad margin -- extra margin (mm) added around each pad of
    # edge-pinned connectors when computing tight geometry bounds.
    # Compensates for pad copper extent beyond the pad center point.
    # Prevents copper_edge_clearance DRC violations after size reduction.
    "connector_pad_margin_mm": 1.0,
    # Orderedness — how strongly passives are snapped into neat rows/columns.
    # 0.0 = organic/force-directed layout, 1.0 = full grid alignment.
    # Intermediate values blend proportionally.  Searchable by autoexperiment.
    "orderedness": 0.3,
    # Through-hole backside threshold — THT components with bounding-box area
    # above this value (mm²) are placed on B.Cu so SMT parts can use F.Cu.
    # SMT passives always stay on F.Cu — IC group connectivity forces keep
    # them near their THT group leaders, achieving dual-sided board usage.
    "tht_backside_min_area_mm2": 50.0,
    # SMT opposite THT — when True, actively attract SMT components on F.Cu
    # toward XY regions occupied by large back-side THT components.  This
    # uses board space efficiently by placing SMT on the opposite side of
    # THT footprints.  Adds an attraction force (0.3× force_attract_k) and
    # a small scoring bonus (~5% weight) for SMT-over-THT overlap.
    "smt_opposite_tht": True,
    # Align large pairs — when True, detect pairs of large non-passive
    # components with similar footprints and force them to be placed
    # side-by-side (aligned on one axis).  Only applies to components
    # with area above tht_backside_min_area_mm2.
    "align_large_pairs": True,
    # Component zone constraints — per-reference placement rules.
    # Each key is a component reference; value is a dict with one of:
    #   {"edge": "left"|"right"|"top"|"bottom"}  — snap to named edge, lock
    #   {"zone": "center-bottom"|"top-left"|...}  — confine to board region
    #   {"corner": "top-left"|"top-right"|"bottom-left"|"bottom-right"} — pin
    # Unassigned connectors fall back to nearest-edge heuristic.
    "component_zones": {},
    # Signal flow order — ordered list of IC group leader references.
    # Biases cluster centroids along the X-axis (left-to-right) during
    # initial placement.  Gives the layout a natural signal-flow direction.
    "signal_flow_order": [],
    # Net priority overrides (higher = routed earlier among same class)
    "net_priority": {},
    # Thermal
    "thermal_refs": [],
    "thermal_radius_mm": 3.0,
    # FreeRouting
    "freerouting_jar": os.path.expanduser("~/.local/lib/freerouting-1.9.0.jar"),
    "freerouting_timeout_s": 60,
    "freerouting_max_passes": 40,
    # GND zone pour — automatically created/updated to cover full board.
    # Set gnd_zone_net to "" to disable automatic zone creation.
    "gnd_zone_net": "GND",
    "gnd_zone_layer": "B.Cu",
    "gnd_zone_margin_mm": 0.5,
    "zone_clearance_mm": 0.3,
    "zone_min_thickness_mm": 0.25,
    "zone_thermal_gap_mm": 0.5,
    "zone_thermal_spoke_mm": 0.5,
    # Ignorable DRC patterns — list of regex strings.  During post-route
    # DRC validation, if ALL significant violations match at least one
    # pattern (searched against the violation description text), they are
    # treated as ignorable.  This is in addition to the automatic
    # footprint-baseline clearance heuristic.
    "ignorable_drc_patterns": [],
    # --- Functional group settings ---
    # Group source: how to discover functional groups.
    #   "auto"      — try schematic sheets first, fall back to netlist analysis
    #   "schematic" — only use schematic hierarchical sheets
    #   "netlist"   — only use netlist community detection
    #   "manual"    — only use ic_groups from config
    # Manual ic_groups overrides are always applied on top of auto-detected
    # groups regardless of this setting.
    "group_source": "auto",
    # When True, use hierarchical group-based placement: place components
    # within each functional group first, then arrange groups on the board
    # as rigid blocks.  When False, use flat global placement (legacy).
    "hierarchical_placement": True,
    # Explicit IC groups (IC + supporting components that should stay together).
    # Each key is the group leader (typically an IC reference), value is a list
    # of supporting component references.  Optional — when group_source is
    # "auto" or "schematic", groups are auto-discovered from .kicad_sch files.
    "ic_groups": {},
    # Human-readable group labels for silkscreen annotation.
    "group_labels": {},
    # --- Search space flags ---
    # When True, batteries/connectors/mounting holes are NOT auto-locked;
    # edge_compliance scoring still incentivizes edge placement.
    "unlock_all_footprints": True,
    # When True, the autoexperiment loop can vary board_width_mm / board_height_mm.
    "enable_board_size_search": True,
    # Default board dimensions (mm) — overridden per-round when board size search is active.
    "board_width_mm": 90.0,
    "board_height_mm": 58.0,
    # Subcircuit margin — extra space (mm) added around the tight bounding
    # box of component positions when building a local subcircuit board.
    # Gives the solver room to rearrange components.
    "subcircuit_margin_mm": 5.0,
    # Parent spacing — gap (mm) between child subcircuit bounding boxes when
    # composing them into the parent board.  Searchable by autoexperiment.
    "parent_spacing_mm": 2.0,
}


CONFIG_SEARCH_SPACE = {
    "orderedness": {"min": 0.0, "max": 1.0, "sigma": 0.05, "type": "float"},
    "reheat_strength": {"min": 0.0, "max": 0.4, "sigma": 0.05, "type": "float"},
    "force_attract_k": {"min": 0.001, "max": 0.2, "sigma": 0.01, "type": "float"},
    "force_repel_k": {"min": 50.0, "max": 1000.0, "sigma": 50.0, "type": "float"},
    "placement_clearance_mm": {"min": 0.5, "max": 8.0, "sigma": 0.5, "type": "float"},
    "cooling_factor": {"min": 0.80, "max": 0.999, "sigma": 0.02, "type": "float"},
    "edge_margin_mm": {"min": 0.5, "max": 15.0, "sigma": 0.5, "type": "float"},
    "courtyard_padding_mm": {"min": 0.0, "max": 3.0, "sigma": 0.1, "type": "float"},
    "board_width_mm": {"min": 30.0, "max": 200.0, "sigma": 5.0, "type": "float"},
    "board_height_mm": {"min": 20.0, "max": 150.0, "sigma": 5.0, "type": "float"},
    "sa_refine_initial_temp": {"min": 0.5, "max": 30.0, "sigma": 2.0, "type": "float"},
    "sa_refine_move_radius_mm": {"min": 0.2, "max": 8.0, "sigma": 0.5, "type": "float"},
    "sa_refine_iterations": {"min": 100, "max": 10000, "sigma": 500, "type": "int"},
    "connector_gap_mm": {"min": 0.0, "max": 8.0, "sigma": 0.5, "type": "float"},
    "edge_jitter_mm": {"min": 0.0, "max": 15.0, "sigma": 1.0, "type": "float"},
    "intra_cluster_iters": {"min": 10, "max": 500, "sigma": 20, "type": "int"},
    "max_placement_iterations": {"min": 100, "max": 5000, "sigma": 300, "type": "int"},
    "subcircuit_margin_mm": {"min": 1.0, "max": 15.0, "sigma": 1.0, "type": "float"},
    "signal_width_mm": {"min": 0.05, "max": 2.0, "sigma": 0.02, "type": "float"},
    "power_width_mm": {"min": 0.05, "max": 5.0, "sigma": 0.05, "type": "float"},
    "via_drill_mm": {"min": 0.15, "max": 1.0, "sigma": 0.05, "type": "float"},
    "via_size_mm": {"min": 0.3, "max": 1.5, "sigma": 0.05, "type": "float"},
    "gnd_zone_margin_mm": {"min": 0.1, "max": 2.0, "sigma": 0.1, "type": "float"},
    "connector_edge_inset_mm": {"min": -2.0, "max": 5.0, "sigma": 0.25, "type": "float"},
    "connector_pad_margin_mm": {"min": 0.0, "max": 3.0, "sigma": 0.2, "type": "float"},
    "thermal_radius_mm": {"min": 1.0, "max": 10.0, "sigma": 0.5, "type": "float"},
    "sa_refine_cooling_rate": {"min": 0.9, "max": 0.9999, "sigma": 0.005, "type": "float"},
    "sa_refine_swap_probability": {"min": 0.0, "max": 1.0, "sigma": 0.05, "type": "float"},
    "sa_refine_rotation_probability": {"min": 0.0, "max": 1.0, "sigma": 0.05, "type": "float"},
    "placement_convergence_threshold": {"min": 0.01, "max": 2.0, "sigma": 0.1, "type": "float"},

    "placement_grid_mm": {"min": 0.1, "max": 2.54, "sigma": 0.1, "type": "float"},
    "pad_inset_margin_mm": {"min": 0.0, "max": 2.0, "sigma": 0.1, "type": "float"},
    "tht_backside_min_area_mm2": {"min": 10.0, "max": 200.0, "sigma": 10.0, "type": "float"},
    "freerouting_timeout_s": {"min": 10, "max": 600, "sigma": 20, "type": "int"},
    "freerouting_max_passes": {"min": 5, "max": 200, "sigma": 10, "type": "int"},
    "zone_clearance_mm": {"min": 0.1, "max": 1.0, "sigma": 0.05, "type": "float"},
    "zone_min_thickness_mm": {"min": 0.1, "max": 0.8, "sigma": 0.05, "type": "float"},
    "zone_thermal_gap_mm": {"min": 0.2, "max": 1.5, "sigma": 0.1, "type": "float"},
    "zone_thermal_spoke_mm": {"min": 0.2, "max": 1.5, "sigma": 0.1, "type": "float"},
    "parent_spacing_mm": {"min": 0.5, "max": 6.0, "sigma": 0.5, "type": "float"},
}


def normalize_bounds(
    key: str,
    lo: float,
    hi: float,
    spec: dict[str, Any] | None = None,
) -> tuple[float, float] | None:
    """Clamp and validate a (lo, hi) bound pair against a CONFIG_SEARCH_SPACE spec.

    Returns normalized (lo, hi) or None if the range is invalid (e.g. empty
    integer range after rounding, or non-finite inputs like NaN/Infinity).
    """
    import math as _math

    if spec is None:
        spec = CONFIG_SEARCH_SPACE.get(key)
    if spec is None:
        return None

    # Reject NaN/Infinity before any arithmetic (platform-dependent otherwise)
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return None
    if not (_math.isfinite(lo) and _math.isfinite(hi)):
        return None

    spec_min = float(spec["min"])
    spec_max = float(spec["max"])

    lo = max(spec_min, min(spec_max, lo))
    hi = max(spec_min, min(spec_max, hi))

    if lo > hi:
        lo, hi = hi, lo

    if spec.get("type") == "int":
        lo = _math.ceil(lo)
        hi = _math.floor(hi)
        if lo > hi:
            return None

    return (lo, hi)


PARAM_CONSTRAINTS = [
    ("via_drill_mm", "<", "via_size_mm"),
]


def enforce_param_constraints(config: dict[str, Any]) -> dict[str, Any]:
    """Fix cross-parameter constraint violations in a config dict.

    Modifies values in-place to ensure physical consistency:
    - via_drill_mm must be < via_size_mm (annular ring requirement)

    Returns the (possibly modified) config.
    """
    for key_a, op, key_b in PARAM_CONSTRAINTS:
        if key_a not in config or key_b not in config:
            continue
        a, b = float(config[key_a]), float(config[key_b])
        if op == "<" and a >= b:
            config[key_a] = b * 0.5
        elif op == "<=" and a > b:
            config[key_b] = a
    return config


def load_project_config(config_path: str | None = None) -> dict[str, Any]:
    """Load a project config from a JSON file.

    If config_path is None, looks for a *_config.json in the autoplacer
    directory. Returns empty dict if no file found.

    JSON values are converted: lists of strings in "power_nets" become sets.
    """
    if config_path is None:
        # Auto-discover config file next to this module
        module_dir = Path(__file__).parent
        candidates = sorted(module_dir.glob("*_config.json"))
        if not candidates:
            return {}
        config_path = str(candidates[0])

    with open(config_path) as f:
        cfg = json.load(f)

    # Convert power_nets list to set for efficient lookup
    if "power_nets" in cfg and isinstance(cfg["power_nets"], list):
        cfg["power_nets"] = set(cfg["power_nets"])

    return cfg



def discover_project_config(project_dir: str | Path) -> Path | None:
    """Auto-discover a project-specific config file in *project_dir*.

    Search order:
    1. ``autoplacer.json``
    2. <dir_stem>_autoplacer.json  (e.g. LLUPS_autoplacer.json)
    3. [autoplacer] section in a .kicad_pro file (not yet implemented)

    Returns the :class:`Path` to the first match, or ``None``.
    """
    project_dir = Path(project_dir)

    # 1. Generic name
    generic = project_dir / "autoplacer.json"
    if generic.is_file():
        return generic

    # 2. <stem>_autoplacer.json
    stem_cfg = project_dir / f"{project_dir.name}_autoplacer.json"
    if stem_cfg.is_file():
        return stem_cfg

    # 3. .kicad_pro [autoplacer] section -- not yet implemented
    return None
