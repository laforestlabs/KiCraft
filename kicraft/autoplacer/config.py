"""Default configuration — project-agnostic placement/routing engine defaults.

Project-specific overrides (ic_groups, component_zones, etc.) live in a
per-project JSON file (e.g. ``LLUPS_autoplacer.json``).  Use
``discover_project_config()`` to locate it automatically, then
``load_project_config()`` to parse it.
"""

import json
import os
from pathlib import Path

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
    # Minimum placement score to proceed to routing.
    # Below this threshold routing is skipped (saves 15-30s on degenerate layouts).
    "min_placement_score": 20.0,
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
    # Skip GND from net counting (routed as zones)
    "skip_gnd_routing": True,
    # Net priority overrides (higher = routed earlier among same class)
    "net_priority": {},
    # Thermal
    "thermal_refs": [],
    "thermal_radius_mm": 3.0,
    # FreeRouting
    "freerouting_jar": os.path.expanduser("~/.local/lib/freerouting-1.9.0.jar"),
    "freerouting_timeout_s": 60,
    "freerouting_max_passes": 40,
    # Nets excluded from trace routing (use copper zones instead).
    # Add power nets that have zone fills (e.g. "GND") to reduce
    # spurious "unconnected" DRC violations.
    "freerouting_ignore_nets": ["GND"],
    # GND zone pour — automatically created/updated to cover full board.
    # Set gnd_zone_net to "" to disable automatic zone creation.
    "gnd_zone_net": "GND",
    "gnd_zone_layer": "B.Cu",
    "gnd_zone_margin_mm": 0.5,
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
    # Board size overhead factor — minimum board area is estimated as
    # total_component_area * overhead_factor.  Larger = more routing room.
    "board_size_overhead_factor": 2.0,
    # Subcircuit margin — extra space (mm) added around the tight bounding
    # box of component positions when building a local subcircuit board.
    # Gives the solver room to rearrange components.
    "subcircuit_margin_mm": 5.0,
}


def load_project_config(config_path: str = None) -> dict:
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
