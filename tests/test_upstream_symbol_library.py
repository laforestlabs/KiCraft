"""Tests for upstream.synthesis.symbol_library.

These touch the real KiCad stock libraries at /usr/share/kicad/symbols.
Skip the file if that directory isn't present (e.g. on machines without KiCad).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kicraft.upstream.synthesis.symbol_library import (
    DEFAULT_KICAD_SYMBOL_DIR,
    SymbolNotFoundError,
    build_lib_symbols_block,
    extract_symbol_block,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_KICAD_SYMBOL_DIR.is_dir(),
    reason="KiCad symbol library not installed at the default path",
)


def test_extract_device_c_qualified() -> None:
    text = extract_symbol_block("Device", "C")
    assert text.startswith('(symbol "Device:C"')


def test_extract_resolves_extends_chain() -> None:
    # Device:C_Small extends Device:C — output must not contain (extends ...).
    text = extract_symbol_block("Device", "C_Small")
    assert text.startswith('(symbol "Device:C_Small"')
    assert "(extends " not in text


def test_extract_does_not_prefix_match() -> None:
    # Device:R exists; Device:R_Small also exists. Asking for R must not pull R_Small.
    r_block = extract_symbol_block("Device", "R")
    assert r_block.startswith('(symbol "Device:R"')
    # R is short; previous regression matched longer names by substring.
    assert 'R_Small' not in r_block.split("\n", 1)[0]


def test_missing_library_raises() -> None:
    with pytest.raises(SymbolNotFoundError):
        extract_symbol_block("DefinitelyNotALibrary", "Anything")


def test_missing_symbol_raises() -> None:
    with pytest.raises(SymbolNotFoundError):
        extract_symbol_block("Device", "GhostSymbol_xyz_123")


def test_build_lib_symbols_deduplicates() -> None:
    block = build_lib_symbols_block(
        [("Device", "C"), ("Device", "R"), ("Device", "C"), ("Device", "R")]
    )
    assert block.count('(symbol "Device:C"') == 1
    assert block.count('(symbol "Device:R"') == 1


def test_build_lib_symbols_all_or_nothing() -> None:
    # Mix one valid and one bogus; nothing should write before the failure.
    with pytest.raises(SymbolNotFoundError):
        build_lib_symbols_block([("Device", "C"), ("Device", "GhostSymbol_xyz")])


def test_build_lib_symbols_empty_returns_short_form() -> None:
    block = build_lib_symbols_block([])
    assert "(lib_symbols)" in block


def test_custom_symbol_dir(tmp_path: Path) -> None:
    fake_lib = tmp_path / "Fake.kicad_sym"
    fake_lib.write_text(
        '(kicad_symbol_lib (version 20231120)\n'
        '\t(symbol "Widget"\n'
        '\t\t(property "Reference" "U" (at 0 0 0))\n'
        '\t)\n'
        ')\n'
    )
    block = extract_symbol_block("Fake", "Widget", symbol_dir=tmp_path)
    assert block.startswith('(symbol "Fake:Widget"')
