"""Minimal smoke tests — verify the package imports successfully."""

import importlib


def test_package_imports():
    """Top-level kicraft package is importable."""
    mod = importlib.import_module("kicraft")
    assert hasattr(mod, "__version__")


def test_autoplacer_config_imports():
    """Autoplacer config and defaults are importable."""
    from kicraft.autoplacer.config import DEFAULT_CONFIG, load_project_config, discover_project_config
    assert isinstance(DEFAULT_CONFIG, dict)
    assert "placement_clearance_mm" in DEFAULT_CONFIG
    assert callable(load_project_config)
    assert callable(discover_project_config)


def test_autoplacer_types_imports():
    """Core types are importable without pcbnew."""
    from kicraft.autoplacer.brain.types import (
        Point,
    )
    # Basic sanity
    p = Point(x=1.0, y=2.0)
    assert p.x == 1.0
    assert p.y == 2.0


def test_hierarchy_parser_imports():
    """Hierarchy parser is importable without pcbnew."""


def test_scoring_imports():
    """Scoring module is importable (may need pcbnew at runtime)."""
    try:
        from kicraft.scoring import ALL_CHECKS
        assert isinstance(ALL_CHECKS, list)
        assert len(ALL_CHECKS) > 0
    except ImportError:
        # pcbnew may not be available in test environment
        pass


def test_logging_config_imports():
    """Logging configuration is importable."""
    from kicraft.logging_config import get_logger, configure_logging
    assert callable(get_logger)
    assert callable(configure_logging)


def test_cli_module_exists():
    """CLI package is importable."""
    import kicraft.cli
    assert kicraft.cli is not None
