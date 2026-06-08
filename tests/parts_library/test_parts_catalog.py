"""Tests for the web part-catalog view-model and SVG preview generation.

Pure helpers (catalog listing, lcsc_url, usage_markdown) run everywhere; the SVG
exports go through ``kicad-cli`` and skip when it is not installed (mirroring the
other kicad-cli-gated tests in this repo).

Note on tiers: this package's ``conftest.py`` has an autouse fixture that empties
the vendored tier, so the listing/doc tests build a part in an isolated *home*
tier, while the SVG tests load a real vendored bundle directly (the renderer only
needs a ``LoadedPart`` pointing at the bundle on disk, independent of the loader's
tier search).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import kicraft.parts_library.loader as loader_mod
from kicraft.parts_library import LoadedPart
from kicraft.parts_library.loader import Tier, _load_one
from kicraft.server import parts_catalog as pc

from .conftest import make_valid_manifest, write_valid_part

_HAS_KICAD_CLI = shutil.which("kicad-cli") is not None


def _real_part(name: str) -> LoadedPart:
    """Load a real shipped bundle directly, bypassing the vendored-tier mask."""
    vendored = Path(loader_mod.__file__).resolve().parent
    part = _load_one(vendored / name, Tier.VENDORED)
    assert isinstance(part, LoadedPart), f"{name} did not load: {part}"
    return part


# ---------- pure helpers (no kicad-cli) ----------


def test_lcsc_url_builds_from_code():
    m = make_valid_manifest(name="widget")  # conftest sets sourcing lcsc = C999999
    assert pc.lcsc_url(m) == "https://www.lcsc.com/product-detail/C999999.html"


def test_lcsc_url_none_without_valid_code():
    no_code = make_valid_manifest(name="widget").model_copy(update={"sourcing": {}})
    assert pc.lcsc_url(no_code) is None
    junk = make_valid_manifest(name="widget").model_copy(
        update={"sourcing": {"lcsc": "not-a-code"}}
    )
    assert pc.lcsc_url(junk) is None


def test_usage_markdown_has_essentials(isolated_home, clean_extras_env):
    write_valid_part(isolated_home / ".kicraft" / "parts", name="widget")
    part = pc.get_part("widget")
    assert part is not None
    doc = pc.usage_markdown(part)
    assert "# WIDGET-MPN" in doc            # heading is the part number
    assert "Test widget widget." in doc     # the description
    assert "`widget:Widget`" in doc          # BOM symbol id
    assert "`widget:WidgetFP`" in doc        # BOM footprint id
    assert "Tags: test" in doc
    assert "Library tier: Yours" in doc      # home tier -> "Yours"


def test_catalog_lists_user_part(isolated_home, clean_extras_env):
    write_valid_part(isolated_home / ".kicraft" / "parts", name="widget")
    parts = pc.catalog()
    by_name = {p.manifest.name: p for p in parts}
    assert "widget" in by_name
    assert pc.tier_label(by_name["widget"].tier) == "Yours"


# ---------- SVG previews (need kicad-cli) ----------


@pytest.mark.skipif(not _HAS_KICAD_CLI, reason="kicad-cli not installed")
def test_symbol_and_footprint_svgs(tmp_path, monkeypatch):
    # Land the on-disk cache under tmp so the test is hermetic.
    monkeypatch.setattr(pc.tempfile, "gettempdir", lambda: str(tmp_path))
    part = _real_part("ams1117-3v3")

    syms = pc.symbol_svgs(part)
    assert syms, "expected at least one symbol-unit svg"
    for s in syms:
        assert s.is_file() and s.stat().st_size > 0
        head = s.read_text(encoding="utf-8", errors="ignore")[:200].lower()
        assert "<svg" in head or "<?xml" in head

    fp = pc.footprint_svg(part)
    assert fp is not None and fp.is_file() and fp.stat().st_size > 0
    head = fp.read_text(encoding="utf-8", errors="ignore")[:200].lower()
    assert "<svg" in head or "<?xml" in head


@pytest.mark.skipif(not _HAS_KICAD_CLI, reason="kicad-cli not installed")
def test_preview_cache_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(pc.tempfile, "gettempdir", lambda: str(tmp_path))
    part = _real_part("ams1117-3v3")

    first = pc.symbol_svgs(part)
    assert first
    stamps = {p: p.stat().st_mtime_ns for p in first}

    second = pc.symbol_svgs(part)
    assert second == first  # identical paths
    for p in second:
        assert p.stat().st_mtime_ns == stamps[p], "cache was regenerated, not reused"
    assert (first[0].parent / ".ok").exists()  # completeness sentinel
