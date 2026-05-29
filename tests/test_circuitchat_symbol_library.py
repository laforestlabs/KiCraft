"""Tests for circuitchat.synthesis.symbol_library.

These touch the real KiCad stock libraries at /usr/share/kicad/symbols.
Skip the file if that directory isn't present (e.g. on machines without KiCad).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from kicraft.circuitchat.synthesis.symbol_library import (
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


def test_extract_renames_inherited_subunits() -> None:
    # A derived (extends) symbol inherits the base's unit/body-style
    # sub-symbols, which KiCad names `<parent>_<unit>_<body>`. They MUST be
    # re-prefixed with the DERIVED name, or KiCad rejects the embedded symbol
    # and the sheet that uses it loads empty (no components, dangling
    # inter-sheet labels). Regression: Device:C_Small extends Device:C, so its
    # bodies must be `C_Small_*`, never the base `C_0_*`/`C_1_*`.
    text = extract_symbol_block("Device", "C_Small")
    subunits = re.findall(r'\(symbol "([A-Za-z0-9_.+\-]+_\d+_\d+)"', text)
    assert subunits, "expected unit/body sub-symbols in a multi-unit symbol"
    assert all(su.startswith("C_Small_") for su in subunits), subunits
    # the base-named bodies must not survive the rename
    assert '(symbol "C_0_' not in text
    assert '(symbol "C_1_' not in text


def test_extract_renames_subunits_when_base_differs() -> None:
    # USBLC6-2SC6 extends USBLC6-2P6 in the stock Power_Protection lib — the
    # case that originally broke synthesis. The derived name does NOT begin
    # with the base name, so this guards the non-prefix path.
    if not (DEFAULT_KICAD_SYMBOL_DIR / "Power_Protection.kicad_sym").is_file():
        pytest.skip("Power_Protection library not installed")
    text = extract_symbol_block("Power_Protection", "USBLC6-2SC6")
    subunits = re.findall(r'\(symbol "([A-Za-z0-9_.+\-]+_\d+_\d+)"', text)
    assert subunits
    assert all(su.startswith("USBLC6-2SC6_") for su in subunits), subunits
    assert '(symbol "USBLC6-2P6' not in text


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
    block = extract_symbol_block("Fake", "Widget", stock_dir=tmp_path)
    assert block.startswith('(symbol "Fake:Widget"')
