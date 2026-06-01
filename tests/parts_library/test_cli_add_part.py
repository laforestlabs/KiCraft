"""Tests for the ``add-part --symbol/--footprint`` file-import path.

The ``--from-lcsc`` path requires live network access to LCSC/EasyEDA and
is exercised separately by an opt-in smoke test. These tests stay
hermetic by feeding the CLI a tmp-file symbol + footprint pair.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kicraft.parts_library import (
    load_manifest,
    project_parts_dir,
    verify_content_hash,
)


# ---------- helpers ----------


def _run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "kicraft.design.cli_app", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


_MIN_SYMBOL = (
    "(kicad_symbol_lib\n"
    '\t(version 20231120)\n'
    '\t(generator "external_tool")\n'
    '\t(symbol "Widget"\n'
    '\t\t(property "Reference" "U" (at 0 0 0))\n'
    '\t\t(symbol "Widget_1_1"\n'
    '\t\t\t(pin input line (at 0 0 0) (length 2.54)\n'
    '\t\t\t\t(name "VCC") (number "1")\n'
    '\t\t\t)\n'
    '\t\t)\n'
    '\t)\n'
    ")\n"
)


_PREFIXED_SYMBOL = (
    "(kicad_symbol_lib\n"
    '\t(version 20231120)\n'
    '\t(generator "external_tool")\n'
    '\t(symbol "OldLib:Widget"\n'
    '\t\t(property "Reference" "U" (at 0 0 0))\n'
    '\t\t(symbol "OldLib:Widget_1_1"\n'
    '\t\t\t(pin input line (at 0 0 0) (length 2.54)\n'
    '\t\t\t\t(name "VCC") (number "1")\n'
    '\t\t\t)\n'
    '\t\t)\n'
    '\t)\n'
    ")\n"
)


_MIN_FOOTPRINT = (
    '(footprint "WidgetFP-0603"\n'
    "\t(version 20231120)\n"
    '\t(generator "external_tool")\n'
    '\t(layer "F.Cu")\n'
    ")\n"
)


@pytest.fixture
def symbol_path(tmp_path: Path) -> Path:
    p = tmp_path / "widget.kicad_sym"
    p.write_text(_MIN_SYMBOL)
    return p


@pytest.fixture
def footprint_path(tmp_path: Path) -> Path:
    p = tmp_path / "widget.kicad_mod"
    p.write_text(_MIN_FOOTPRINT)
    return p


# ---------- happy path ----------


def test_file_import_happy_path(
    isolated_home, clean_extras_env, tmp_path, symbol_path, footprint_path
):
    project = tmp_path / "project"
    project.mkdir()
    res = _run_cli(
        "add-part",
        "--symbol", str(symbol_path),
        "--footprint", str(footprint_path),
        "--mpn", "WIDGET-1",
        "--name", "widget-one",
        "--sourcing", "lcsc=C12345",
        "--sourcing", "digikey=ND-WIDGET-1-ND",
        "--tag", "passive",
        "--description", "A test widget.",
        cwd=project,
    )
    assert res.returncode == 0, res.stderr
    assert "OK added widget-one@0.1.0" in res.stdout

    part_dir = project_parts_dir(project) / "widget-one"
    assert part_dir.is_dir()
    assert (part_dir / "widget-one.kicad_sym").is_file()
    assert (part_dir / "widget-one.pretty" / "WidgetFP-0603.kicad_mod").is_file()

    manifest = load_manifest(part_dir)
    assert manifest.name == "widget-one"
    assert manifest.mpn == "WIDGET-1"
    assert manifest.sourcing == {"lcsc": "C12345", "digikey": "nd-widget-1-nd"} or \
           manifest.sourcing == {"lcsc": "C12345", "digikey": "ND-WIDGET-1-ND"}
    # The vendor key is normalized to lowercase; the part-number value is preserved.
    assert manifest.sourcing["digikey"] == "ND-WIDGET-1-ND"
    assert manifest.tags == ["passive"]
    assert manifest.symbol_name == "Widget"
    assert manifest.footprint_name == "WidgetFP-0603"
    assert manifest.provenance.source == "file-import"
    # Content hash matches what's on disk.
    assert verify_content_hash(part_dir, manifest)


# ---------- normalization ----------


def test_file_import_strips_library_prefix(
    isolated_home, clean_extras_env, tmp_path, footprint_path
):
    """When the source .kicad_sym uses 'OldLib:Widget', the bundle keeps
    only 'Widget' so the bundle's library prefix (= directory name) is the
    only place the prefix lives."""
    sym = tmp_path / "prefixed.kicad_sym"
    sym.write_text(_PREFIXED_SYMBOL)
    project = tmp_path / "project"
    project.mkdir()

    res = _run_cli(
        "add-part",
        "--symbol", str(sym),
        "--footprint", str(footprint_path),
        "--mpn", "WIDGET-2",
        "--name", "widget-two",
        cwd=project,
    )
    assert res.returncode == 0, res.stderr

    part_dir = project_parts_dir(project) / "widget-two"
    manifest = load_manifest(part_dir)
    assert manifest.symbol_name == "Widget"  # prefix stripped
    sym_text = (part_dir / "widget-two.kicad_sym").read_text()
    assert '(symbol "Widget"' in sym_text
    assert '(symbol "OldLib:Widget"' not in sym_text
    # Sub-symbol unit references should also be renamed.
    assert '(symbol "Widget_1_1"' in sym_text
    assert '(symbol "OldLib:Widget_1_1"' not in sym_text


# ---------- input validation ----------


def test_file_import_requires_mpn(
    isolated_home, clean_extras_env, tmp_path, symbol_path, footprint_path
):
    project = tmp_path / "project"
    project.mkdir()
    res = _run_cli(
        "add-part",
        "--symbol", str(symbol_path),
        "--footprint", str(footprint_path),
        "--name", "no-mpn",
        cwd=project,
    )
    assert res.returncode == 2
    assert "--mpn is required" in res.stderr


def test_file_import_requires_both_paths(
    isolated_home, clean_extras_env, tmp_path, symbol_path
):
    """--symbol without --footprint is rejected."""
    project = tmp_path / "project"
    project.mkdir()
    res = _run_cli(
        "add-part",
        "--symbol", str(symbol_path),
        "--mpn", "FOO",
        cwd=project,
    )
    assert res.returncode == 2
    assert "must be supplied together" in res.stderr


def test_file_import_mutually_exclusive_with_lcsc(
    isolated_home, clean_extras_env, tmp_path, symbol_path, footprint_path
):
    project = tmp_path / "project"
    project.mkdir()
    res = _run_cli(
        "add-part",
        "--from-lcsc", "C123",
        "--symbol", str(symbol_path),
        "--footprint", str(footprint_path),
        "--mpn", "X",
        cwd=project,
    )
    assert res.returncode == 2
    assert "mutually exclusive" in res.stderr


def test_file_import_missing_symbol_file(
    isolated_home, clean_extras_env, tmp_path, footprint_path
):
    project = tmp_path / "project"
    project.mkdir()
    res = _run_cli(
        "add-part",
        "--symbol", str(tmp_path / "does-not-exist.kicad_sym"),
        "--footprint", str(footprint_path),
        "--mpn", "X",
        "--name", "missing",
        cwd=project,
    )
    assert res.returncode == 2
    assert "symbol file not found" in res.stderr


def test_file_import_malformed_symbol(
    isolated_home, clean_extras_env, tmp_path, footprint_path
):
    """A .kicad_sym file with no (symbol "...") block should be rejected."""
    sym = tmp_path / "empty.kicad_sym"
    sym.write_text("(kicad_symbol_lib (version 20231120))")
    project = tmp_path / "project"
    project.mkdir()
    res = _run_cli(
        "add-part",
        "--symbol", str(sym),
        "--footprint", str(footprint_path),
        "--mpn", "X",
        "--name", "empty",
        cwd=project,
    )
    assert res.returncode == 2
    assert "no top-level" in res.stderr or "no.*symbol" in res.stderr.lower()


def test_file_import_bad_sourcing_format(
    isolated_home, clean_extras_env, tmp_path, symbol_path, footprint_path
):
    project = tmp_path / "project"
    project.mkdir()
    res = _run_cli(
        "add-part",
        "--symbol", str(symbol_path),
        "--footprint", str(footprint_path),
        "--mpn", "X",
        "--name", "bad-sourcing",
        "--sourcing", "no-equals-sign",
        cwd=project,
    )
    assert res.returncode == 2
    assert "missing '='" in res.stderr


def test_file_import_overwrite_rejected_without_flag(
    isolated_home, clean_extras_env, tmp_path, symbol_path, footprint_path
):
    """Second add-part with the same slug fails unless --overwrite is set."""
    project = tmp_path / "project"
    project.mkdir()
    args = [
        "add-part",
        "--symbol", str(symbol_path),
        "--footprint", str(footprint_path),
        "--mpn", "X",
        "--name", "ovr-test",
    ]
    res1 = _run_cli(*args, cwd=project)
    assert res1.returncode == 0
    res2 = _run_cli(*args, cwd=project)
    assert res2.returncode == 2
    assert "already exists" in res2.stderr
    res3 = _run_cli(*args, "--overwrite", cwd=project)
    assert res3.returncode == 0, res3.stderr
