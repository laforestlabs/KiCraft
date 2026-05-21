"""Tests for circuitchat.synthesis.symbol_pinout.

Touches the real KiCad stock libraries at /usr/share/kicad/symbols.
Skip the file if that directory isn't present (matches the pattern in
test_circuitchat_symbol_library.py).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kicraft.circuitchat.synthesis.symbol_library import (
    DEFAULT_KICAD_SYMBOL_DIR,
    SymbolNotFoundError,
)
from kicraft.circuitchat.synthesis.symbol_pinout import lookup_pins

pytestmark = pytest.mark.skipif(
    not DEFAULT_KICAD_SYMBOL_DIR.is_dir(),
    reason="KiCad symbol library not installed at the default path",
)


def test_device_r_two_passive_pins() -> None:
    info = lookup_pins("Device:R")
    assert info["symbol"] == "Device:R"
    assert len(info["pins"]) == 2
    nums = {p["number"] for p in info["pins"]}
    assert nums == {"1", "2"}
    assert all(p["electrical_type"] == "passive" for p in info["pins"])


def test_device_c_two_passive_pins() -> None:
    info = lookup_pins("Device:C")
    assert len(info["pins"]) == 2
    nums = {p["number"] for p in info["pins"]}
    assert nums == {"1", "2"}


def test_extends_resolves_to_base_pins() -> None:
    # Device:C_Small extends Device:C — the pin list must come from the
    # base after extends resolution.
    base = lookup_pins("Device:C")
    derived = lookup_pins("Device:C_Small")
    base_nums = sorted(p["number"] for p in base["pins"])
    derived_nums = sorted(p["number"] for p in derived["pins"])
    assert base_nums == derived_nums


def test_pin_record_shape() -> None:
    info = lookup_pins("Device:R")
    p = info["pins"][0]
    assert set(p.keys()) >= {
        "number", "name", "electrical_type",
        "position", "orientation", "length", "unit",
    }
    assert isinstance(p["position"], dict)
    assert {"x", "y"} <= set(p["position"].keys())
    assert isinstance(p["orientation"], int)


def test_missing_library_raises() -> None:
    with pytest.raises(SymbolNotFoundError):
        lookup_pins("DefinitelyNotALibrary:Anything")


def test_missing_symbol_raises() -> None:
    with pytest.raises(SymbolNotFoundError):
        lookup_pins("Device:GhostSymbol_xyz_123")


def test_bad_lib_id_raises() -> None:
    with pytest.raises(SymbolNotFoundError):
        lookup_pins("NoColonHere")


def test_lru_cache_returns_consistent_data() -> None:
    info1 = lookup_pins("Device:R")
    info2 = lookup_pins("Device:R")
    # Same content (caching is transparent; just check the contract).
    assert info1 == info2


def test_custom_symbol_dir(tmp_path: Path) -> None:
    fake_lib = tmp_path / "Fake.kicad_sym"
    fake_lib.write_text(
        '(kicad_symbol_lib (version 20231120)\n'
        '\t(symbol "Widget"\n'
        '\t\t(property "Reference" "U" (at 0 0 0))\n'
        '\t\t(symbol "Widget_1_1"\n'
        '\t\t\t(pin input line\n'
        '\t\t\t\t(at -2.54 0 0)\n'
        '\t\t\t\t(length 2.54)\n'
        '\t\t\t\t(name "IN" (effects (font (size 1 1))))\n'
        '\t\t\t\t(number "1" (effects (font (size 1 1))))\n'
        '\t\t\t)\n'
        '\t\t\t(pin output line\n'
        '\t\t\t\t(at 2.54 0 180)\n'
        '\t\t\t\t(length 2.54)\n'
        '\t\t\t\t(name "OUT" (effects (font (size 1 1))))\n'
        '\t\t\t\t(number "2" (effects (font (size 1 1))))\n'
        '\t\t\t)\n'
        '\t\t)\n'
        '\t)\n'
        ')\n'
    )
    info = lookup_pins("Fake:Widget", stock_dir=tmp_path)
    assert {p["number"] for p in info["pins"]} == {"1", "2"}
    assert {p["electrical_type"] for p in info["pins"]} == {"input", "output"}
