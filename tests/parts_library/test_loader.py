"""Tests for kicraft.parts_library.loader — the 4-tier search."""
from __future__ import annotations

from pathlib import Path

import pytest

from kicraft.parts_library import (
    Tier,
    find_part,
    load_all_with_overrides,
    project_parts_dir,
    resolve_tier_dirs,
)

from .conftest import write_valid_part


def test_resolve_tier_dirs_order(isolated_home, clean_extras_env, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    tiers = resolve_tier_dirs(project)
    # Project first, then home, then vendored
    names = [t.value for t, _ in tiers]
    assert names[0] == "project"
    assert names[1] == "home"
    assert names[2] == "vendored"


def test_resolve_tier_dirs_includes_extras(isolated_home, monkeypatch, tmp_path):
    extra1 = tmp_path / "extra1"
    extra2 = tmp_path / "extra2"
    extra1.mkdir()
    extra2.mkdir()
    monkeypatch.setenv("KICRAFT_EXTRA_PARTS_DIRS", f"{extra1}:{extra2}")
    project = tmp_path / "p"
    project.mkdir()
    tiers = resolve_tier_dirs(project)
    paths = [str(p) for _, p in tiers]
    # Last two entries are the extras, in declared order.
    assert paths[-2:] == [str(extra1), str(extra2)]
    assert tiers[-2][0] == Tier.EXTRA
    assert tiers[-1][0] == Tier.EXTRA


def test_resolve_tier_dirs_without_project_root(isolated_home, clean_extras_env):
    tiers = resolve_tier_dirs(None)
    names = [t.value for t, _ in tiers]
    assert names[0] == "home"  # no project tier
    assert "project" not in names


def test_find_part_project_wins(isolated_home, clean_extras_env, tmp_path):
    """Same-named part in both project and home: project wins."""
    project = tmp_path / "project"
    write_valid_part(project_parts_dir(project), name="widget")
    write_valid_part(isolated_home / ".kicraft" / "parts", name="widget")

    found = find_part("widget", project)
    assert found is not None
    assert found.tier == Tier.PROJECT


def test_find_part_falls_through_to_home(isolated_home, clean_extras_env, tmp_path):
    """No project copy: home wins."""
    project = tmp_path / "project"
    write_valid_part(isolated_home / ".kicraft" / "parts", name="widget")
    found = find_part("widget", project)
    assert found is not None
    assert found.tier == Tier.HOME


def test_find_part_returns_none_when_absent(isolated_home, clean_extras_env, tmp_path):
    found = find_part("never-installed-anywhere", tmp_path / "project")
    assert found is None


def test_load_all_separates_active_and_shadowed(
    isolated_home, clean_extras_env, tmp_path
):
    project = tmp_path / "project"
    write_valid_part(project_parts_dir(project), name="alpha")
    write_valid_part(isolated_home / ".kicraft" / "parts", name="alpha")  # shadowed
    write_valid_part(isolated_home / ".kicraft" / "parts", name="beta")  # home-only

    active, shadowed, broken = load_all_with_overrides(project)
    active_names = sorted(p.manifest.name for p in active)
    assert active_names == ["alpha", "beta"]
    # alpha appears at both project and home; the home copy is shadowed.
    active_alpha = next(p for p in active if p.manifest.name == "alpha")
    assert active_alpha.tier == Tier.PROJECT
    assert len(shadowed) == 1
    assert shadowed[0].manifest.name == "alpha"
    assert shadowed[0].tier == Tier.HOME
    assert broken == []


def test_broken_part_reported_not_raised(isolated_home, clean_extras_env, tmp_path):
    """Bad manifest must surface as a BrokenPart, not a crash."""
    project = tmp_path / "project"
    bad = project_parts_dir(project) / "broken"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text("{ not valid json")
    active, _shadowed, broken = load_all_with_overrides(project)
    assert active == []
    assert len(broken) == 1
    assert broken[0].dir == bad
    assert "manifest" in broken[0].reason


def test_name_mismatch_directory_is_broken(
    isolated_home, clean_extras_env, tmp_path
):
    """A directory whose name doesn't match manifest.name fails validation."""
    project = tmp_path / "project"
    part_dir = write_valid_part(project_parts_dir(project), name="widget")
    # Rename the directory but leave the manifest's name field unchanged.
    renamed = part_dir.parent / "different-name"
    part_dir.rename(renamed)
    active, _shadowed, broken = load_all_with_overrides(project)
    assert active == []
    assert any("directory name" in b.reason for b in broken)


def test_edited_files_break_content_hash(isolated_home, clean_extras_env, tmp_path):
    project = tmp_path / "project"
    part_dir = write_valid_part(project_parts_dir(project), name="widget")
    # Edit the symbol file after the manifest was finalized.
    (part_dir / "widget.kicad_sym").write_text("(kicad_symbol_lib (version 999))")
    active, _shadowed, broken = load_all_with_overrides(project)
    assert active == []
    assert any("content_hash" in b.reason for b in broken)
