"""Shared application state for the hierarchical experiment manager.

The GUI should default to a clean, balanced experiment workflow:
- enough visibility to understand what is running
- enough controls to tune the hierarchical pipeline
- minimal vestigial toggles from older GUI iterations

Current baseline:
- experiment rounds: 10
- leaf solve rounds: 2
- workers: 2
- plateau threshold: 2
- compose spacing: 6 mm

The state in this module is intentionally conservative:
- defaults should be stable and broadly useful
- disabled controls should represent future work, not clutter
- page-level cleanup flags live here so the GUI can progressively remove
  old or redundant panels without scattering that logic across pages
"""

from __future__ import annotations

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
        "max": 6,
        "step": 1,
        "enabled": True,
        "group": "Leaf Solving",
        "description": "Balanced default: 2. Increase only when extra leaf quality clearly justifies the runtime.",
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
    "workers": 2,
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
    "render_parent_png": True,
    "preserve_existing_subcircuit_artifacts": True,
    "show_only_accepted_frames": False,
    "show_status_json": True,
    "show_leaf_artifacts": True,
    "show_parent_progress": True,
    "track_composition_outputs": True,
    "enable_progression_viewer": True,
    "prefer_kept_frames": False,
}



DEFAULT_GUI_CLEANUP = {
    "show_analysis_tab": True,
    "show_legacy_imports": False,
    "show_legacy_presets": False,
    "show_raw_status_json": True,
    "show_visuals_panel": False,
}


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

    active_experiment_id: int | None = None
    runner_pid: int | None = None
    _runner: ExperimentRunner | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.db is None:
            self.db = Database()

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
        config.update(self.toggles)
        config["pipeline"] = "hierarchical_subcircuits"
        return config

    def load_from_config(self, config: dict[str, Any]) -> None:
        """Restore state from a saved config dict."""
        for control in self.hierarchical_controls:
            key = control["key"]
            if key in config:
                control["default"] = config[key]
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

        for key in DEFAULT_TOGGLES:
            if key in config:
                self.toggles[key] = config[key]

    def get_control_ranges(self) -> dict[str, list[float | int]]:
        """Return enabled hierarchical control ranges."""
        ranges: dict[str, list[float | int]] = {}
        for control in self.hierarchical_controls:
            if control.get("enabled", False):
                ranges[control["key"]] = [control["min"], control["max"]]
        return ranges

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
