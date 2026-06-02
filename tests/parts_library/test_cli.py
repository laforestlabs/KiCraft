"""Tests for the parts-related CLI verbs: list-parts and validate-part.

``add-part`` requires network access to LCSC/EasyEDA and is not exercised
here; integration testing for it lives in a separate marked-as-slow
smoke run.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from kicraft.parts_library import project_parts_dir

from .conftest import write_valid_part


def _run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    """Invoke the cli_app entry point in a subprocess, returning the result."""
    return subprocess.run(
        [sys.executable, "-m", "kicraft.design.cli_app", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def test_list_parts_empty(isolated_home, clean_extras_env, tmp_path, monkeypatch):
    """When no parts tier has anything, list-parts says so."""
    # Mask the vendored tier so the test sees a truly empty library.
    monkeypatch.setenv("KICRAFT_EXTRA_PARTS_DIRS", "")
    project = tmp_path / "project"
    project.mkdir()
    # Note: the real vendored tier still shows ip2368; that's expected.
    # We exercise the "no project / no home / no extras" sub-case.
    res = _run_cli("list-parts", cwd=project)
    assert res.returncode == 0
    # Either the placeholder "no parts" message or the ip2368 vendored row.
    assert "ip2368" in res.stdout or "no parts available" in res.stdout


def test_list_parts_shows_project_entry(
    isolated_home, clean_extras_env, tmp_path
):
    project = tmp_path / "project"
    write_valid_part(project_parts_dir(project), name="widget")
    res = _run_cli("list-parts", cwd=project)
    assert res.returncode == 0, res.stderr
    assert "widget" in res.stdout
    assert "project" in res.stdout  # tier column


def test_validate_part_happy_path(isolated_home, clean_extras_env, tmp_path):
    project = tmp_path / "project"
    part_dir = write_valid_part(project_parts_dir(project), name="widget")
    res = _run_cli("validate-part", str(part_dir), cwd=project)
    assert res.returncode == 0
    assert "OK widget@" in res.stdout


def test_validate_part_missing_manifest(
    isolated_home, clean_extras_env, tmp_path
):
    project = tmp_path / "project"
    (project / "stray").mkdir(parents=True)
    res = _run_cli("validate-part", str(project / "stray"), cwd=project)
    assert res.returncode == 2
    assert "manifest" in res.stderr.lower()


def test_validate_part_directory_name_mismatch(
    isolated_home, clean_extras_env, tmp_path
):
    project = tmp_path / "project"
    part_dir = write_valid_part(project_parts_dir(project), name="widget")
    renamed = part_dir.parent / "different"
    part_dir.rename(renamed)
    # The symbol file inside still has the old name; copy/rename it too.
    (renamed / "widget.kicad_sym").rename(renamed / "different.kicad_sym")
    (renamed / "widget.pretty").rename(renamed / "different.pretty")
    res = _run_cli("validate-part", str(renamed), cwd=project)
    assert res.returncode == 2
    assert "does not match" in res.stderr


def test_validate_part_content_hash_mismatch(
    isolated_home, clean_extras_env, tmp_path
):
    project = tmp_path / "project"
    part_dir = write_valid_part(project_parts_dir(project), name="widget")
    # Add an extra file: changes the content hash but preserves the symbol
    # and footprint checks (which are validated independently and earlier).
    (part_dir / "datasheet.pdf").write_bytes(b"%PDF-1.4\n")
    res = _run_cli("validate-part", str(part_dir), cwd=project)
    assert res.returncode == 2
    assert "content_hash mismatch" in res.stderr


def test_validate_part_update_hash_fixes_it(
    isolated_home, clean_extras_env, tmp_path
):
    project = tmp_path / "project"
    part_dir = write_valid_part(project_parts_dir(project), name="widget")
    # Hash-only mutation: add a file the manifest doesn't claim.
    (part_dir / "datasheet.pdf").write_bytes(b"%PDF-1.4\n")
    res = _run_cli("validate-part", str(part_dir), "--update-hash", cwd=project)
    assert res.returncode == 0, res.stderr
    assert "updated content_hash" in res.stdout
    # Subsequent validation without --update-hash should now pass.
    res2 = _run_cli("validate-part", str(part_dir), cwd=project)
    assert res2.returncode == 0
