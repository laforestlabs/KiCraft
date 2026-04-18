"""Tests for kicraft.autoplacer.config — defaults, loading, and discovery."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from kicraft.autoplacer.config import (
    DEFAULT_CONFIG,
    discover_project_config,
    load_project_config,
)

# ---------------------------------------------------------------------------
# Locate the example config shipped with the repo
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent  # KiCraft/
_EXAMPLE_CONFIG = _REPO_ROOT / "examples" / "llups_autoplacer.json"


# ---- DEFAULT_CONFIG structure ---------------------------------------------

class TestDefaultConfig:
    """Verify DEFAULT_CONFIG contains all expected top-level keys."""

    REQUIRED_KEYS = [
        "placement_clearance_mm",
        "board_width_mm",
        "board_height_mm",
        "signal_width_mm",
        "power_width_mm",
        "via_drill_mm",
        "via_size_mm",
        "placement_grid_mm",
        "edge_margin_mm",
        "force_attract_k",
        "force_repel_k",
        "cooling_factor",
        "max_placement_iterations",
        "freerouting_timeout_s",
        "freerouting_max_passes",
        "component_zones",
        "ic_groups",
        "power_nets",
        "gnd_zone_net",
        "subcircuit_margin_mm",
    ]

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_has_key(self, key: str):
        assert key in DEFAULT_CONFIG, f"DEFAULT_CONFIG missing expected key '{key}'"

    def test_is_dict(self):
        assert isinstance(DEFAULT_CONFIG, dict)

    def test_board_dimensions_are_positive(self):
        assert DEFAULT_CONFIG["board_width_mm"] > 0
        assert DEFAULT_CONFIG["board_height_mm"] > 0

    def test_placement_clearance_positive(self):
        assert DEFAULT_CONFIG["placement_clearance_mm"] > 0

    def test_power_nets_is_set(self):
        assert isinstance(DEFAULT_CONFIG["power_nets"], set)

    def test_component_zones_is_dict(self):
        assert isinstance(DEFAULT_CONFIG["component_zones"], dict)

    def test_ic_groups_is_dict(self):
        assert isinstance(DEFAULT_CONFIG["ic_groups"], dict)


# ---- load_project_config --------------------------------------------------

class TestLoadProjectConfig:
    """Test loading project config files."""

    def test_load_example_config(self):
        """Loading the shipped example config returns a non-empty dict."""
        if not _EXAMPLE_CONFIG.exists():
            pytest.skip("Example config not found (expected in examples/)")
        cfg = load_project_config(str(_EXAMPLE_CONFIG))
        assert isinstance(cfg, dict)
        assert len(cfg) > 0

    def test_example_config_has_ic_groups(self):
        """Example config defines ic_groups."""
        if not _EXAMPLE_CONFIG.exists():
            pytest.skip("Example config not found")
        cfg = load_project_config(str(_EXAMPLE_CONFIG))
        assert "ic_groups" in cfg
        assert isinstance(cfg["ic_groups"], dict)
        assert len(cfg["ic_groups"]) > 0

    def test_example_config_power_nets_converted_to_set(self):
        """power_nets list in JSON is converted to a Python set."""
        if not _EXAMPLE_CONFIG.exists():
            pytest.skip("Example config not found")
        cfg = load_project_config(str(_EXAMPLE_CONFIG))
        assert "power_nets" in cfg
        assert isinstance(cfg["power_nets"], set)

    def test_example_config_merges_over_defaults(self):
        """Project config values should override DEFAULT_CONFIG values."""
        if not _EXAMPLE_CONFIG.exists():
            pytest.skip("Example config not found")
        cfg = load_project_config(str(_EXAMPLE_CONFIG))
        merged = {**DEFAULT_CONFIG, **cfg}
        # The merged dict should have all default keys ...
        for key in DEFAULT_CONFIG:
            assert key in merged
        # ... plus any project-specific keys
        for key in cfg:
            assert key in merged
            assert merged[key] == cfg[key]

    def test_nonexistent_path_raises(self):
        """A non-existent explicit path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_project_config("/tmp/surely_does_not_exist_config_xyz.json")

    def test_none_path_returns_dict(self):
        """Passing None falls back to auto-discovery (returns dict, possibly empty)."""
        cfg = load_project_config(None)
        assert isinstance(cfg, dict)

    def test_load_minimal_json(self, tmp_path: Path):
        """A minimal JSON file loads successfully."""
        cfg_file = tmp_path / "test_config.json"
        cfg_file.write_text(json.dumps({"board_width_mm": 42.0}))
        cfg = load_project_config(str(cfg_file))
        assert cfg["board_width_mm"] == 42.0

    def test_load_power_nets_list_to_set(self, tmp_path: Path):
        """A JSON power_nets list is converted to a set."""
        cfg_file = tmp_path / "test_config.json"
        cfg_file.write_text(json.dumps({"power_nets": ["VCC", "GND"]}))
        cfg = load_project_config(str(cfg_file))
        assert cfg["power_nets"] == {"VCC", "GND"}


# ---- discover_project_config ---------------------------------------------

class TestDiscoverProjectConfig:
    """Test automatic config file discovery."""

    def test_returns_none_for_empty_dir(self, tmp_path: Path):
        """An empty directory has no config to discover."""
        result = discover_project_config(tmp_path)
        assert result is None

    def test_returns_none_for_dir_without_config(self, tmp_path: Path):
        """A directory with unrelated files has no config to discover."""
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "schematic.kicad_sch").write_text("")
        result = discover_project_config(tmp_path)
        assert result is None

    def test_finds_generic_autoplacer_json(self, tmp_path: Path):
        """Discovers 'autoplacer.json' in the project directory."""
        cfg_file = tmp_path / "autoplacer.json"
        cfg_file.write_text(json.dumps({"board_width_mm": 50}))
        result = discover_project_config(tmp_path)
        assert result is not None
        assert result == cfg_file

    def test_finds_stem_autoplacer_json(self, tmp_path: Path):
        """Discovers '<dirname>_autoplacer.json' in the project directory."""
        # Create a directory named "MyProject"
        proj_dir = tmp_path / "MyProject"
        proj_dir.mkdir()
        cfg_file = proj_dir / "MyProject_autoplacer.json"
        cfg_file.write_text(json.dumps({"board_height_mm": 60}))
        result = discover_project_config(proj_dir)
        assert result is not None
        assert result == cfg_file

    def test_generic_takes_priority_over_stem(self, tmp_path: Path):
        """'autoplacer.json' is preferred over '<stem>_autoplacer.json'."""
        proj_dir = tmp_path / "Proj"
        proj_dir.mkdir()
        generic = proj_dir / "autoplacer.json"
        stem = proj_dir / "Proj_autoplacer.json"
        generic.write_text(json.dumps({"source": "generic"}))
        stem.write_text(json.dumps({"source": "stem"}))
        result = discover_project_config(proj_dir)
        assert result == generic
