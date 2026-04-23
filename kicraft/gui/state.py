"""Shared application state for the hierarchical experiment manager.

The GUI should default to a clean, balanced experiment workflow:
- enough visibility to understand what is running
- enough controls to tune the hierarchical pipeline
- minimal vestigial toggles from older GUI iterations

Current baseline:
- experiment rounds: 10
- leaf solve rounds: 2
- workers: 0 (auto)
- plateau threshold: 2
- compose spacing: 6 mm

The state in this module is intentionally conservative:
- defaults should be stable and broadly useful
- disabled controls should represent future work, not clutter
- page-level cleanup flags live here so the GUI can progressively remove
  old or redundant panels without scattering that logic across pages
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .db import Database

if TYPE_CHECKING:
    from .experiment_runner import ExperimentRunner


HIERARCHICAL_CONTROLS = [
    {
        "key": "leaf_rounds",
        "label": "Leaf Solve Rounds",
        "default": 2,
        "min": 1,
        "max": None,
        "step": 1,
        "enabled": True,
        "group": "Leaf Solving",
        "description": "How many local solve attempts each leaf gets. Higher is slower but more likely to find a legal placement.",
    },
    {
        "key": "top_level_rounds",
        "label": "Top-Level Assembly Rounds",
        "default": 1,
        "min": 1,
        "max": 3,
        "step": 1,
        "enabled": True,
        "group": "Top-Level Assembly",
        "description": "Keep this narrow until top-level routing fidelity is more trustworthy.",
    },
    {
        "key": "compose_spacing_mm",
        "label": "Parent Composition Spacing (mm)",
        "default": 6.0,
        "min": 4.0,
        "max": 20.0,
        "step": 1.0,
        "enabled": True,
        "group": "Top-Level Assembly",
        "description": "Balanced spacing range for routine runs. Widen only for debugging or special cases.",
    },
]

def _detect_project_files(root: Path) -> tuple[str, str, str]:
    """Auto-detect KiCad project files from the project root.

    Looks for ``*.kicad_pro`` files and derives the PCB and schematic
    filenames from the project name.

    Returns:
        (project_name, pcb_filename, schematic_filename)
    """
    pro_files = sorted(root.glob("*.kicad_pro"))
    if pro_files:
        name = pro_files[0].stem
        return name, f"{name}.kicad_pcb", f"{name}.kicad_sch"
    return "project", "project.kicad_pcb", "project.kicad_sch"


def _project_root() -> Path:
    """Find the KiCad project root.

    Walk upward from the current working directory looking for any
    ``*.kicad_pro`` file.  Falls back to cwd.
    """
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if list(p.glob("*.kicad_pro")):
            return p
    return cwd


_PROJECT_ROOT = _project_root()
_PROJECT_NAME, _DEFAULT_PCB, _DEFAULT_SCH = _detect_project_files(_PROJECT_ROOT)


DEFAULT_STRATEGY = {
    "rounds": 10,
    "workers": 0,
    "plateau_threshold": 2,
    "seed": 0,
    "pcb_file": _DEFAULT_PCB,
    "schematic_file": _DEFAULT_SCH,
    "parent_selector": "/",
    "only_selectors": [],
}

DEFAULT_SCORE_WEIGHTS = {
    "leaf_acceptance": 0.55,
    "leaf_routing_quality": 0.20,
    "parent_composition": 0.10,
    "parent_routed": 0.15,
}

DEFAULT_TOGGLES = {
    "show_leaf_artifacts": True,
    "track_composition_outputs": True,
    "render_png": True,
    "save_round_details": True,
    "show_top_level_progress": True,
    "import_best_as_preset": True,
    "freerouting_hide_window": True,
}



DEFAULT_GUI_CLEANUP = {
    "show_analysis_tab": True,
    "show_legacy_presets": False,
}


def _default_mutation_bounds() -> dict[str, list[float | int]]:
    from kicraft.autoplacer.config import CONFIG_SEARCH_SPACE

    bounds: dict[str, list[float | int]] = {}
    for key, spec in CONFIG_SEARCH_SPACE.items():
        bounds[key] = [spec["min"], spec["max"]]
    return bounds


# ---------------------------------------------------------------------------
# Placement & Routing parameter definitions exposed to the GUI.
# Each entry defines the parameter key, display label, default value,
# min/max bounds, step, and logical group for collapsible UI sections.
# Only values that differ from their default are written to the config overlay.
# ---------------------------------------------------------------------------

PLACEMENT_PARAMS: list[dict[str, Any]] = [
    # -- Placement Physics --
    {"key": "orderedness", "label": "Orderedness", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "group": "Placement Physics", "description": "How strongly passives snap into rows/columns (0 = organic, 1 = full grid)"},
    {"key": "force_attract_k", "label": "Attraction strength", "default": 0.02, "min": 0.001, "max": 0.2, "step": 0.005, "group": "Placement Physics", "description": "Force pulling connected components together"},
    {"key": "force_repel_k", "label": "Repulsion strength", "default": 200.0, "min": 50.0, "max": 1000.0, "step": 10.0, "group": "Placement Physics", "description": "Force pushing overlapping components apart"},
    {"key": "cooling_factor", "label": "Cooling factor", "default": 0.97, "min": 0.80, "max": 0.999, "step": 0.005, "group": "Placement Physics", "description": "How fast the force-directed solver settles (lower = faster)"},
    {"key": "reheat_strength", "label": "Reheat strength", "default": 0.1, "min": 0.0, "max": 0.4, "step": 0.02, "group": "Placement Physics", "description": "Random perturbation kick at 50% iterations to escape local minima"},
    {"key": "max_placement_iterations", "label": "Max iterations", "default": 300, "min": 100, "max": 5000, "step": 50, "group": "Placement Physics", "description": "Total force-directed placement iterations"},
    {"key": "intra_cluster_iters", "label": "Intra-cluster iterations", "default": 80, "min": 10, "max": 500, "step": 10, "group": "Placement Physics", "description": "Iterations for arranging components within a functional group"},
    {"key": "placement_convergence_threshold", "label": "Convergence threshold", "default": 0.5, "min": 0.01, "max": 2.0, "step": 0.05, "group": "Placement Physics", "description": "Stop early if movement drops below this threshold"},
    # -- Board Geometry --
    {"key": "board_width_mm", "label": "Board width (mm)", "default": 90.0, "min": 30.0, "max": 200.0, "step": 1.0, "group": "Board Geometry", "description": "PCB width in millimeters"},
    {"key": "board_height_mm", "label": "Board height (mm)", "default": 58.0, "min": 20.0, "max": 150.0, "step": 1.0, "group": "Board Geometry", "description": "PCB height in millimeters"},
    {"key": "edge_margin_mm", "label": "Edge margin (mm)", "default": 6.0, "min": 0.5, "max": 15.0, "step": 0.5, "group": "Board Geometry", "description": "Keep-out distance from board edges"},
    {"key": "subcircuit_margin_mm", "label": "Subcircuit margin (mm)", "default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5, "group": "Board Geometry", "description": "Extra space around leaf subcircuit bounding box"},
    {"key": "parent_spacing_mm", "label": "Parent spacing (mm)", "default": 6.0, "min": 0.5, "max": 20.0, "step": 0.5, "group": "Board Geometry", "description": "Spacing between subcircuits when composing into the parent board"},
    {"key": "placement_clearance_mm", "label": "Placement clearance (mm)", "default": 2.5, "min": 0.5, "max": 8.0, "step": 0.25, "group": "Board Geometry", "description": "Minimum gap between component bounding boxes"},
    {"key": "placement_grid_mm", "label": "Placement grid (mm)", "default": 1.0, "min": 0.1, "max": 2.54, "step": 0.1, "group": "Board Geometry", "description": "Snap grid resolution for component placement"},
    # -- Edge & Connectors --
    {"key": "edge_jitter_mm", "label": "Edge jitter (mm)", "default": 5.0, "min": 0.0, "max": 15.0, "step": 0.5, "group": "Edge & Connectors", "description": "Random displacement along edge for edge-pinned parts"},
    {"key": "connector_gap_mm", "label": "Connector gap (mm)", "default": 2.0, "min": 0.0, "max": 8.0, "step": 0.5, "group": "Edge & Connectors", "description": "Spacing between connectors on the same edge"},
    {"key": "connector_edge_inset_mm", "label": "Connector inset (mm)", "default": 1.0, "min": -2.0, "max": 5.0, "step": 0.25, "group": "Edge & Connectors", "description": "Inset from board edge (0 = flush, negative = overhang)"},
    {"key": "connector_pad_margin_mm", "label": "Connector pad margin (mm)", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.25, "group": "Edge & Connectors", "description": "Extra pad margin for DRC clearance on connectors"},
    {"key": "courtyard_padding_mm", "label": "Courtyard padding (mm)", "default": 0.5, "min": 0.0, "max": 3.0, "step": 0.1, "group": "Edge & Connectors", "description": "Extra margin when scoring courtyard overlaps"},
    {"key": "pad_inset_margin_mm", "label": "Pad inset margin (mm)", "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.1, "group": "Edge & Connectors", "description": "Minimum distance pads must be inside Edge.Cuts boundary"},
    # -- Component Behavior --
    {"key": "tht_backside_min_area_mm2", "label": "THT backside threshold (mm2)", "default": 50.0, "min": 10.0, "max": 200.0, "step": 5.0, "group": "Component Behavior", "description": "THT parts above this area placed on back copper layer"},
    {"key": "smt_opposite_tht", "label": "SMT opposite THT", "default": True, "min": None, "max": None, "step": None, "group": "Component Behavior", "type": "bool", "description": "Attract SMT to regions backed by THT on opposite side"},
    {"key": "align_large_pairs", "label": "Align large pairs", "default": True, "min": None, "max": None, "step": None, "group": "Component Behavior", "type": "bool", "description": "Force large similar-footprint component pairs side-by-side"},
    {"key": "hierarchical_placement", "label": "Hierarchical placement", "default": True, "min": None, "max": None, "step": None, "group": "Component Behavior", "type": "bool", "description": "Use group-based hierarchical placement (vs flat global)"},
    {"key": "unlock_all_footprints", "label": "Unlock all footprints", "default": True, "min": None, "max": None, "step": None, "group": "Component Behavior", "type": "bool", "description": "Unlock batteries/connectors/mounting holes for placement"},
    {"key": "enable_board_size_search", "label": "Board size search", "default": True, "min": None, "max": None, "step": None, "group": "Component Behavior", "type": "bool", "description": "Allow autoexperiment to vary board dimensions"},
    # -- SA Refinement --
    {"key": "sa_refine_enabled", "label": "SA refinement enabled", "default": True, "min": None, "max": None, "step": None, "group": "SA Refinement", "type": "bool", "description": "Run simulated annealing refinement after force-directed pass"},
    {"key": "sa_refine_iterations", "label": "SA iterations", "default": 1000, "min": 100, "max": 10000, "step": 100, "group": "SA Refinement", "description": "Number of simulated annealing steps"},
    {"key": "sa_refine_initial_temp", "label": "SA initial temperature", "default": 5.0, "min": 0.5, "max": 30.0, "step": 0.5, "group": "SA Refinement", "description": "Starting temperature for SA"},
    {"key": "sa_refine_cooling_rate", "label": "SA cooling rate", "default": 0.995, "min": 0.9, "max": 0.9999, "step": 0.001, "group": "SA Refinement", "description": "Temperature decay per SA step (higher = slower cooling)"},
    {"key": "sa_refine_move_radius_mm", "label": "SA move radius (mm)", "default": 2.0, "min": 0.2, "max": 8.0, "step": 0.2, "group": "SA Refinement", "description": "Max random displacement per SA step"},
    {"key": "sa_refine_swap_probability", "label": "SA swap probability", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "group": "SA Refinement", "description": "Chance of swapping two components vs moving one"},
    {"key": "sa_refine_rotation_probability", "label": "SA rotation probability", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05, "group": "SA Refinement", "description": "Chance of rotating a component during SA"},
    # -- Routing --
    {"key": "signal_width_mm", "label": "Signal trace width (mm)", "default": 0.127, "min": 0.05, "max": 2.0, "step": 0.01, "group": "Routing", "description": "Width of signal traces (0.127 = 5 mil)"},
    {"key": "power_width_mm", "label": "Power trace width (mm)", "default": 0.127, "min": 0.05, "max": 5.0, "step": 0.01, "group": "Routing", "description": "Width of power traces (increase for high-current paths)"},
    {"key": "via_drill_mm", "label": "Via drill (mm)", "default": 0.3, "min": 0.15, "max": 1.0, "step": 0.05, "group": "Routing", "description": "Via drill hole diameter (0.2 typical fab min)"},
    {"key": "via_size_mm", "label": "Via size (mm)", "default": 0.6, "min": 0.3, "max": 1.5, "step": 0.05, "group": "Routing", "description": "Via annular ring outer diameter"},
    {"key": "freerouting_timeout_s", "label": "FreeRouting timeout (s)", "default": 60, "min": 10, "max": 600, "step": 10, "group": "Routing", "description": "Max seconds FreeRouting is allowed to run"},
    {"key": "freerouting_max_passes", "label": "FreeRouting max passes", "default": 40, "min": 5, "max": 200, "step": 5, "group": "Routing", "description": "Max routing passes for FreeRouting"},
    {"key": "gnd_zone_margin_mm", "label": "GND zone margin (mm)", "default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1, "group": "Routing", "description": "Clearance margin for the automatic GND copper zone pour"},
    {"key": "gnd_zone_net", "label": "GND zone net", "default": "GND", "min": None, "max": None, "step": None, "group": "Routing", "type": "text", "description": "Net name for automatic ground zone pour (empty to disable)"},
    {"key": "gnd_zone_layer", "label": "GND zone layer", "default": "B.Cu", "min": None, "max": None, "step": None, "group": "Routing", "type": "text", "description": "Copper layer for GND zone (F.Cu or B.Cu)"},
    # -- Zone Pour --
    {"key": "zone_clearance_mm", "label": "Zone clearance (mm)", "default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05, "group": "Zone Pour", "description": "Copper zone clearance from pads and traces"},
    {"key": "zone_min_thickness_mm", "label": "Zone min thickness (mm)", "default": 0.25, "min": 0.1, "max": 0.8, "step": 0.05, "group": "Zone Pour", "description": "Minimum copper fill thickness for zone pour"},
    {"key": "zone_thermal_gap_mm", "label": "Thermal relief gap (mm)", "default": 0.5, "min": 0.2, "max": 1.5, "step": 0.1, "group": "Zone Pour", "description": "Gap between pad and zone fill in thermal relief connections"},
    {"key": "zone_thermal_spoke_mm", "label": "Thermal spoke width (mm)", "default": 0.5, "min": 0.2, "max": 1.5, "step": 0.1, "group": "Zone Pour", "description": "Width of thermal relief spoke connections to pads"},
    # -- Thermal --
    {"key": "thermal_radius_mm", "label": "Thermal keepout radius (mm)", "default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5, "group": "Thermal", "description": "Keep-away radius around thermal components for heat dissipation"},
    {"key": "thermal_refs", "label": "Thermal components", "default": "", "min": None, "max": None, "step": None, "group": "Thermal", "type": "list", "description": "Component references with thermal keepout (comma-separated)"},
    # -- Net Classification --
    {"key": "power_nets", "label": "Power nets", "default": "", "min": None, "max": None, "step": None, "group": "Net Classification", "type": "list", "description": "Net names classified as power (comma-separated, get wider traces)"},
    {"key": "ignorable_drc_patterns", "label": "Ignorable DRC patterns", "default": "", "min": None, "max": None, "step": None, "group": "Net Classification", "type": "list", "description": "Regex patterns for DRC violations to ignore (comma-separated)"},
]


@dataclass
class AppState:
    """Mutable singleton holding current GUI state."""

    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    project_name: str = field(default_factory=lambda: _PROJECT_NAME)
    db: Database | None = field(default=None)

    hierarchical_controls: list[dict[str, Any]] = field(
        default_factory=lambda: [{**d} for d in HIERARCHICAL_CONTROLS]
    )
    strategy: dict[str, Any] = field(default_factory=lambda: {**DEFAULT_STRATEGY})
    score_weights: dict[str, float] = field(
        default_factory=lambda: {**DEFAULT_SCORE_WEIGHTS}
    )
    toggles: dict[str, Any] = field(default_factory=lambda: {**DEFAULT_TOGGLES})
    gui_cleanup: dict[str, Any] = field(default_factory=lambda: {**DEFAULT_GUI_CLEANUP})
    placement_config: dict[str, Any] = field(default_factory=dict)
    mutation_bounds: dict[str, list[float | int]] = field(
        default_factory=_default_mutation_bounds
    )

    active_experiment_id: int | None = None
    runner_pid: int | None = None
    _runner: ExperimentRunner | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.db is None:
            self.db = Database()
        # Seed strategy with hierarchical control defaults so leaf_rounds,
        # top_level_rounds, compose_spacing_mm all have a value before a
        # preset is loaded. Start and Setup then read the same source.
        for control in self.hierarchical_controls:
            key = control["key"]
            self.strategy.setdefault(key, control["default"])

    @property
    def runner(self) -> ExperimentRunner:
        """Lazy-init singleton experiment runner."""
        if self._runner is None:
            from .experiment_runner import ExperimentRunner

            self._runner = ExperimentRunner(
                self.project_root,
                self.scripts_dir,
                self.experiments_dir,
            )
        return self._runner

    @property
    def experiments_dir(self) -> Path:
        return self.project_root / ".experiments"

    @property
    def scripts_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent / "cli"

    def to_config_dict(self) -> dict[str, Any]:
        """Build the full hierarchical config dict for the GUI and persistence."""
        config: dict[str, Any] = {}

        for control in self.hierarchical_controls:
            key = control["key"]
            config[key] = control["default"]
            if control.get("enabled", False):
                config[f"_control_{key}"] = {
                    "min": control["min"],
                    "max": control["max"],
                    "step": control["step"],
                    "group": control.get("group", "General"),
                    "description": control.get("description", ""),
                }

        config["_strategy"] = {**self.strategy}
        config["_score_weights"] = {**self.score_weights}
        config["_gui_cleanup"] = {**self.gui_cleanup}
        config["_placement_config"] = {**self.placement_config}
        config["_mutation_bounds"] = {k: list(v) for k, v in self.mutation_bounds.items()}
        config.update(self.toggles)
        config["pipeline"] = "hierarchical_subcircuits"
        return config

    def load_from_config(self, config: dict[str, Any]) -> None:
        """Restore state from a saved config dict."""
        for control in self.hierarchical_controls:
            key = control["key"]
            if key in config:
                control["default"] = config[key]
                # Mirror the value into strategy so Start sees it. Without
                # this, leaf_rounds (and similar) get stranded in
                # control["default"] while the runner reads strategy.
                self.strategy[key] = config[key]
            control_key = f"_control_{key}"
            if control_key in config and isinstance(config[control_key], dict):
                meta = config[control_key]
                control["min"] = meta.get("min", control["min"])
                control["max"] = meta.get("max", control["max"])
                control["step"] = meta.get("step", control["step"])
                control["enabled"] = True

        if "_strategy" in config and isinstance(config["_strategy"], dict):
            self.strategy.update(config["_strategy"])

        if "_score_weights" in config and isinstance(config["_score_weights"], dict):
            self.score_weights.update(config["_score_weights"])

        if "_gui_cleanup" in config and isinstance(config["_gui_cleanup"], dict):
            self.gui_cleanup.update(config["_gui_cleanup"])

        if "_placement_config" in config and isinstance(config["_placement_config"], dict):
            self.placement_config.update(config["_placement_config"])

        if "_mutation_bounds" in config and isinstance(config["_mutation_bounds"], dict):
            from kicraft.autoplacer.config import CONFIG_SEARCH_SPACE, normalize_bounds

            self.mutation_bounds = _default_mutation_bounds()
            for key, bounds in config["_mutation_bounds"].items():
                if key not in CONFIG_SEARCH_SPACE:
                    continue
                if not isinstance(bounds, list) or len(bounds) < 2:
                    continue
                try:
                    lo, hi = float(bounds[0]), float(bounds[1])
                except (TypeError, ValueError):
                    continue
                result = normalize_bounds(key, lo, hi)
                if result is None:
                    continue
                self.mutation_bounds[key] = [result[0], result[1]]
        else:
            self.mutation_bounds = _default_mutation_bounds()

        for key in DEFAULT_TOGGLES:
            if key in config:
                self.toggles[key] = config[key]

    def get_control_ranges(self) -> dict[str, list[float | int]]:
        """Return user-specified mutation bounds for searchable parameters."""
        return {k: list(v) for k, v in self.mutation_bounds.items()}

    def save_session_state(self) -> None:
        """Persist current mutation bounds to disk for cross-session restore."""
        state_path = self.experiments_dir / "gui_session_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"_mutation_bounds": {k: list(v) for k, v in self.mutation_bounds.items()}}
        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
        except OSError:
            pass

    def restore_session_state(self) -> None:
        """Restore mutation bounds from the last session if available."""
        state_path = self.experiments_dir / "gui_session_state.json"
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        if "_mutation_bounds" in data:
            self.load_from_config({"_mutation_bounds": data["_mutation_bounds"]})

    def get_enabled_controls(self) -> list[dict[str, Any]]:
        return [c for c in self.hierarchical_controls if c.get("enabled", False)]

    def get_disabled_controls(self) -> list[dict[str, Any]]:
        return [c for c in self.hierarchical_controls if not c.get("enabled", False)]

    def get_only_selectors_text(self) -> str:
        selectors = self.strategy.get("only_selectors", [])
        if not selectors:
            return ""
        return "\n".join(str(s) for s in selectors)

    def set_only_selectors_from_text(self, text: str) -> None:
        selectors = [line.strip() for line in text.splitlines() if line.strip()]
        self.strategy["only_selectors"] = selectors


_state: AppState | None = None


def get_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state
