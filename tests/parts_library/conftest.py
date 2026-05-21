"""Shared fixtures for parts_library tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.parts_library import (
    PartManifest,
    Provenance,
    compute_content_hash,
    dump_manifest,
)

VALID_SYMBOL = (
    "(kicad_symbol_lib\n"
    '\t(version 20231120)\n'
    '\t(generator "kicraft_parts_library")\n'
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

VALID_FOOTPRINT = (
    "(footprint \"WidgetFP\"\n"
    "\t(version 20231120)\n"
    "\t(generator \"kicraft_parts_library\")\n"
    "\t(layer \"F.Cu\")\n"
    ")\n"
)


def make_valid_manifest(name: str = "widget", footprint_name: str = "WidgetFP") -> PartManifest:
    """Return a syntactically valid manifest with a placeholder content_hash."""
    return PartManifest(
        schema_version="1",
        name=name,
        version="0.1.0",
        content_hash="sha256:" + "0" * 64,
        description=f"Test widget {name}.",
        mpn=f"{name.upper()}-MPN",
        sourcing={"lcsc": "C999999"},
        datasheet_url=None,
        tags=["test"],
        watch_out_for=None,
        symbol_name="Widget",
        footprint_name=footprint_name,
        kicad_version_min="9.0.0",
        provenance=Provenance(
            source="vendored",
            source_project_stem=None,
            added_at="2026-05-21T00:00:00Z",
            kicad_version="9.0.0",
        ),
    )


def write_valid_part(
    base: Path,
    name: str = "widget",
    *,
    symbol: str = VALID_SYMBOL,
    footprint: str = VALID_FOOTPRINT,
    footprint_name: str = "WidgetFP",
) -> Path:
    """Write a complete, validated part bundle into ``base/<name>/``.

    Returns the part directory. The manifest's ``content_hash`` is filled
    in after the bundle files are written so it matches.
    """
    part_dir = base / name
    part_dir.mkdir(parents=True, exist_ok=True)
    (part_dir / f"{name}.kicad_sym").write_text(symbol)
    fp_dir = part_dir / f"{name}.pretty"
    fp_dir.mkdir(exist_ok=True)
    (fp_dir / f"{footprint_name}.kicad_mod").write_text(footprint)
    manifest = make_valid_manifest(name=name, footprint_name=footprint_name)
    dump_manifest(manifest, part_dir)
    actual = compute_content_hash(part_dir)
    dump_manifest(manifest.model_copy(update={"content_hash": actual}), part_dir)
    return part_dir


@pytest.fixture
def isolated_home(monkeypatch, tmp_path: Path) -> Path:
    """Redirect ``Path.home()`` to a tmp dir so home-tier lookups don't
    accidentally touch the developer's real ``~/.kicraft/``.
    """
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: fake_home))
    return fake_home


@pytest.fixture
def clean_extras_env(monkeypatch) -> None:
    """Clear the extras env var so tests see only the tiers they set up."""
    monkeypatch.delenv("KICRAFT_EXTRA_PARTS_DIRS", raising=False)


@pytest.fixture(autouse=True)
def mask_vendored_tier(monkeypatch, tmp_path: Path) -> Path:
    """Redirect the vendored parts dir to an empty tmp dir for every test.

    Without this fixture the installed ``kicraft/parts_library/ip2368/``
    bundle leaks into every test that calls ``load_all_with_overrides``
    or ``find_part``, polluting assertions about empty/active sets.
    Tests that want to assert vendored behavior can override this fixture.
    """
    empty_vendored = tmp_path / "_vendored_empty"
    empty_vendored.mkdir()
    monkeypatch.setattr(
        "kicraft.parts_library.loader.vendored_parts_dir",
        lambda: empty_vendored,
    )
    return empty_vendored
