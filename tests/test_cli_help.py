"""Tests that CLI entry points respond to --help without crashing.

Commands that fail because pcbnew (KiCad Python bindings) is unavailable
at import time are automatically skipped rather than marked as failures.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_cli_help(command: str) -> subprocess.CompletedProcess:
    """Run a CLI command with --help and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", command, "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _skip_if_pcbnew_import_error(result: subprocess.CompletedProcess, name: str):
    """Skip the test if the command failed due to missing pcbnew."""
    if result.returncode != 0:
        stderr = result.stderr
        if "pcbnew" in stderr.lower() or "ModuleNotFoundError" in stderr:
            pytest.skip(
                f"{name} requires pcbnew (KiCad Python bindings) which is not available"
            )


# ---------------------------------------------------------------------------
# Commands that do NOT import pcbnew at module level (should always work)
# ---------------------------------------------------------------------------

class TestCLIHelpNoPcbnew:
    """CLI commands that work without pcbnew."""

    def test_clean_experiments_help(self):
        result = _run_cli_help("kicraft.cli.clean_experiments")
        _skip_if_pcbnew_import_error(result, "clean-experiments")
        assert result.returncode == 0
        assert "clean" in result.stdout.lower() or "experiment" in result.stdout.lower()

    def test_inspect_subcircuits_help(self):
        result = _run_cli_help("kicraft.cli.inspect_subcircuits")
        _skip_if_pcbnew_import_error(result, "inspect-subcircuits")
        assert result.returncode == 0
        assert "schematic" in result.stdout.lower() or "subcircuit" in result.stdout.lower()

    def test_render_pcb_help(self):
        result = _run_cli_help("kicraft.cli.render_pcb")
        _skip_if_pcbnew_import_error(result, "render-pcb")
        assert result.returncode == 0
        assert "pcb" in result.stdout.lower() or "render" in result.stdout.lower()

    def test_solve_subcircuits_help(self):
        result = _run_cli_help("kicraft.cli.solve_subcircuits")
        _skip_if_pcbnew_import_error(result, "solve-subcircuits")
        assert result.returncode == 0
        assert "schematic" in result.stdout.lower() or "subcircuit" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Commands that MAY import pcbnew at module level — always guarded
# ---------------------------------------------------------------------------

class TestCLIHelpMayNeedPcbnew:
    """CLI commands that may need pcbnew; skipped gracefully if unavailable."""

    def test_score_layout_help(self):
        result = _run_cli_help("kicraft.cli.score_layout")
        _skip_if_pcbnew_import_error(result, "score-layout")
        assert result.returncode == 0

    def test_list_footprints_help(self):
        result = _run_cli_help("kicraft.cli.list_footprints")
        _skip_if_pcbnew_import_error(result, "list-footprints")
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Additional CLI entry points
# ---------------------------------------------------------------------------

class TestCLIHelpAdditional:
    """Additional CLI commands."""

    def test_compose_subcircuits_help(self):
        result = _run_cli_help("kicraft.cli.compose_subcircuits")
        _skip_if_pcbnew_import_error(result, "compose-subcircuits")
        assert result.returncode == 0

    def test_parse_schematic_help(self):
        result = _run_cli_help("kicraft.cli.parse_schematic")
        _skip_if_pcbnew_import_error(result, "parse-schematic")
        assert result.returncode == 0

    def test_inspect_solved_subcircuits_help(self):
        result = _run_cli_help("kicraft.cli.inspect_solved_subcircuits")
        _skip_if_pcbnew_import_error(result, "inspect-solved-subcircuits")
        assert result.returncode == 0

    def test_export_subcircuit_artifacts_help(self):
        result = _run_cli_help("kicraft.cli.export_subcircuit_artifacts")
        _skip_if_pcbnew_import_error(result, "export-subcircuit-artifacts")
        assert result.returncode == 0
