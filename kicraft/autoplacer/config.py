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
    # 2.84mm gives the SA solver enough breathing room to avoid courtyard
    # overlaps while staying within typical 4-layer trace+via budgets.
    # Tuned via overnight parameter sweep (r=-0.59, top-quintile median).
    "placement_clearance_mm": 2.84,
    # Power nets (common names — projects override with their own)
    "power_nets": set(),
    # Placement (spread components — room to route, no courtyard overlaps)
    "placement_grid_mm": 1.20,
    "edge_margin_mm": 6.0,
    "force_attract_k": 0.02,
    "force_repel_k": 200.0,
    "cooling_factor": 0.97,
    "sa_refine_enabled": True,
    "sa_refine_iterations": 1000,
    "sa_refine_initial_temp": 5.0,
    # Faster cooling (0.952 vs 0.995) lets SA spend more iterations near the
    # target temperature instead of crawling through high-temp randomness.
    "sa_refine_cooling_rate": 0.952,
    # Larger SA move radius helps escape local minima found by force solve.
    "sa_refine_move_radius_mm": 5.63,
    "sa_refine_swap_probability": 0.3,
    # Higher rotation probability gives the solver more chances to find a
    # better orientation per component during refinement.
    "sa_refine_rotation_probability": 0.44,
    # Placement solver iterations -- raised from 300 to 3332 because the
    # sweep found measurable score gains for harder leaves at the higher
    # cap. Easy leaves still terminate at placement_convergence_threshold
    # well before this limit.
    "max_placement_iterations": 3332,
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
    # 1.30mm chosen via parameter sweep (top-quintile median).
    "courtyard_padding_mm": 1.30,
    # Pad inset margin — minimum distance (mm) all electrical pads must be
    # inside the board Edge.Cuts boundary.  Pads outside are unfabricatable.
    "pad_inset_margin_mm": 0.3,
    # Edge jitter — maximum random displacement (mm) along the assigned edge
    # for edge-pinned components (connectors, mounting holes).  Provides
    # placement diversity across rounds while keeping components on edges.
    "edge_jitter_mm": 5.0,
    # Connector gap — spacing (mm) between connectors grouped on the same edge.
    "connector_gap_mm": 3.58,
    # Connector edge inset — distance (mm) from board edge to the nearest
    # edge of the connector body.  0 = flush, positive = inset, negative =
    # overhang.  Only applies to edge-pinned connectors.
    "connector_edge_inset_mm": 2.5,
    # Mounting hole keep-in -- minimum distance (mm) from board edge to
    # the center of a mounting hole.
    "mounting_hole_keep_in_mm": 2.5,
    # Orderedness — how strongly passives are snapped into neat rows/columns.
    # 0.0 = organic/force-directed layout, 1.0 = full grid alignment.
    # Intermediate values blend proportionally.  Searchable by autoexperiment.
    "orderedness": 0.3,
    # Through-hole backside threshold — THT components with bounding-box area
    # above this value (mm²) are placed on B.Cu so SMT parts can use F.Cu.
    # SMT passives always stay on F.Cu — IC group connectivity forces keep
    # them near their THT group leaders, achieving dual-sided board usage.
    # 130mm² (vs former 50) keeps small-medium THT on the front side; only
    # genuinely large THT (battery holders, big connectors) move to B.Cu.
    "tht_backside_min_area_mm2": 130.0,
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
    "freerouting_max_passes": 20,
    # Separate pass cap for leaf routing. Leaves are smaller and need
    # less optimization than the parent board, so we keep this lower by
    # default. When unset, leaves fall back to freerouting_max_passes.
    "leaf_freerouting_max_passes": 12,
    # Hide the FreeRouting Swing window. For 2.x this is passed as
    # --gui.enabled=false. For 1.x the runner wraps the invocation in
    # xvfb-run when xvfb-run is on PATH (install xorg-x11-server-Xvfb).
    # If neither path is available the window still appears.
    "freerouting_hide_window": True,
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
    # Footprint refs whose internal clearance DRCs may be ignored when
    # the report parser cannot reliably extract refs from every violation.
    "ignorable_footprint_refs": [],
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
    # 10.82mm gives the SA solver enough slack to find good arrangements;
    # the previous 5mm was the strongest leaf-side bottleneck (r=+0.59).
    "subcircuit_margin_mm": 10.82,
    # Parent spacing — gap (mm) between child subcircuit bounding boxes when
    # composing them into the parent board.  1.17mm packs leaves tightly
    # without compromising routability (r=-0.41 in parents-only sweep).
    "parent_spacing_mm": 1.17,
}


# Only true layout heuristics belong here. Fab/circuit constraints
# (signal_width_mm, power_width_mm, via_drill_mm, via_size_mm, zone_*_mm,
# pad_inset_margin_mm, thermal_radius_mm) are dictated by the fab's minimum
# feature size and the schematic's current/voltage requirements -- they are
# NOT optimization knobs. Operational params (freerouting_timeout_s,
# freerouting_max_passes) are runtime budgets, not quality knobs. Board
# dimensions (board_width_mm, board_height_mm) are derived from leaf areas
# and enclosure constraints, not searched.
CONFIG_SEARCH_SPACE = {
    # --- Insensitive params: full original ranges retained ---
    # These showed |r| < 0.10 in both stages of the overnight sweep, so
    # narrowing their range would just constrain future searches without
    # gain. Widen if a future change makes them sensitive again.
    "orderedness": {"min": 0.0, "max": 1.0, "sigma": 0.05, "type": "float"},
    "reheat_strength": {"min": 0.0, "max": 0.4, "sigma": 0.05, "type": "float"},
    "force_attract_k": {"min": 0.001, "max": 0.2, "sigma": 0.01, "type": "float"},
    "force_repel_k": {"min": 50.0, "max": 1000.0, "sigma": 50.0, "type": "float"},
    "cooling_factor": {"min": 0.80, "max": 0.999, "sigma": 0.02, "type": "float"},
    "edge_margin_mm": {"min": 0.5, "max": 15.0, "sigma": 0.5, "type": "float"},
    "sa_refine_initial_temp": {"min": 0.5, "max": 30.0, "sigma": 2.0, "type": "float"},
    "sa_refine_iterations": {"min": 100, "max": 10000, "sigma": 500, "type": "int"},
    "edge_jitter_mm": {"min": 0.0, "max": 15.0, "sigma": 1.0, "type": "float"},
    "intra_cluster_iters": {"min": 10, "max": 500, "sigma": 20, "type": "int"},
    "gnd_zone_margin_mm": {"min": 0.1, "max": 2.0, "sigma": 0.1, "type": "float"},
    "sa_refine_swap_probability": {"min": 0.0, "max": 1.0, "sigma": 0.05, "type": "float"},
    "placement_convergence_threshold": {"min": 0.01, "max": 2.0, "sigma": 0.1, "type": "float"},
    # --- Sensitive params: ranges narrowed to top-quintile [P10, P90] ---
    # min/max replaced from .experiments/param_sweep/proposed_param_ranges.json
    # Sigma roughly = 10% of new range so Gaussian mutation steps at a
    # sensible scale for the narrower interval.
    "placement_clearance_mm": {"min": 1.10, "max": 5.38, "sigma": 0.4, "type": "float"},
    "courtyard_padding_mm": {"min": 0.65, "max": 2.57, "sigma": 0.2, "type": "float"},
    "sa_refine_move_radius_mm": {"min": 1.52, "max": 7.41, "sigma": 0.6, "type": "float"},
    "connector_gap_mm": {"min": 0.46, "max": 7.50, "sigma": 0.7, "type": "float"},
    "max_placement_iterations": {"min": 829, "max": 4551, "sigma": 370, "type": "int"},
    "subcircuit_margin_mm": {"min": 6.97, "max": 13.38, "sigma": 0.6, "type": "float"},
    "connector_edge_inset_mm": {"min": 0.47, "max": 4.32, "sigma": 0.4, "type": "float"},
    "sa_refine_cooling_rate": {"min": 0.9076, "max": 0.99, "sigma": 0.008, "type": "float"},
    "sa_refine_rotation_probability": {"min": 0.05, "max": 0.85, "sigma": 0.08, "type": "float"},
    "placement_grid_mm": {"min": 0.34, "max": 2.13, "sigma": 0.18, "type": "float"},
    "tht_backside_min_area_mm2": {"min": 80.0, "max": 176.0, "sigma": 10.0, "type": "float"},
    "parent_spacing_mm": {"min": 0.66, "max": 1.60, "sigma": 0.1, "type": "float"},
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
