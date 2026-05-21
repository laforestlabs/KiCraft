"""Tests for the unified symbol+footprint resolver."""
from __future__ import annotations

from pathlib import Path

import pytest

from kicraft.circuitchat.synthesis.parts_lookup import (
    LibraryNotFoundError,
    resolve_footprint_library_path,
    resolve_symbol_library_path,
)
from kicraft.circuitchat.synthesis.symbol_library import (
    SymbolNotFoundError,
    extract_symbol_block,
)
from kicraft.parts_library import project_parts_dir

from .conftest import write_valid_part


def test_resolver_finds_project_local_symbol(
    isolated_home, clean_extras_env, tmp_path
):
    project = tmp_path / "project"
    write_valid_part(project_parts_dir(project), name="widget")
    path = resolve_symbol_library_path(
        "widget",
        project_root=project,
        stock_dir=tmp_path / "nonexistent-stock",
    )
    assert path.name == "widget.kicad_sym"
    assert "widget" in str(path)


def test_resolver_finds_project_local_footprint(
    isolated_home, clean_extras_env, tmp_path
):
    project = tmp_path / "project"
    write_valid_part(project_parts_dir(project), name="widget")
    path = resolve_footprint_library_path(
        "widget",
        project_root=project,
        stock_dir=tmp_path / "nonexistent-stock",
    )
    assert path.name == "widget.pretty"
    assert path.is_dir()


def test_resolver_falls_through_to_stock(
    isolated_home, clean_extras_env, tmp_path
):
    """If no parts tier has the library, the stock fallback is used."""
    fake_stock = tmp_path / "stock"
    fake_stock.mkdir()
    (fake_stock / "FakeStock.kicad_sym").write_text("(kicad_symbol_lib)")
    path = resolve_symbol_library_path(
        "FakeStock",
        project_root=tmp_path / "no-such-project",
        stock_dir=fake_stock,
    )
    assert path.parent == fake_stock


def test_resolver_project_shadows_stock(
    isolated_home, clean_extras_env, tmp_path
):
    """Same library name in project tier and stock: project wins."""
    project = tmp_path / "project"
    write_valid_part(project_parts_dir(project), name="widget")

    fake_stock = tmp_path / "stock"
    fake_stock.mkdir()
    (fake_stock / "widget.kicad_sym").write_text("(stock version)")

    path = resolve_symbol_library_path(
        "widget", project_root=project, stock_dir=fake_stock
    )
    assert "parts" in str(path)  # came from project tier
    assert "stock" not in str(path)


def test_resolver_raises_when_nothing_matches(
    isolated_home, clean_extras_env, tmp_path
):
    with pytest.raises(LibraryNotFoundError):
        resolve_symbol_library_path(
            "ghost-lib",
            project_root=tmp_path / "no-project",
            stock_dir=tmp_path / "no-stock",
        )


def test_extract_symbol_block_via_project_tier(
    isolated_home, clean_extras_env, tmp_path, monkeypatch
):
    """End-to-end: extract a symbol whose library lives in the project tier."""
    project = tmp_path / "project"
    write_valid_part(project_parts_dir(project), name="widget")
    # Make project_root the CWD so the default resolver finds it.
    monkeypatch.chdir(project)
    block = extract_symbol_block(
        "widget", "Widget", stock_dir=tmp_path / "no-stock"
    )
    assert block.startswith('(symbol "widget:Widget"')


def test_symbol_not_found_in_resolved_library(
    isolated_home, clean_extras_env, tmp_path
):
    """Library file resolves, but the named symbol isn't in it."""
    project = tmp_path / "project"
    write_valid_part(project_parts_dir(project), name="widget")
    with pytest.raises(SymbolNotFoundError):
        extract_symbol_block(
            "widget",
            "NotInLibrary",
            project_root=project,
            stock_dir=tmp_path / "no-stock",
        )
