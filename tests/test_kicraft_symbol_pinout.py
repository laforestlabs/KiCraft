"""Tests for kicraft.synthesis.symbol_pinout.

Touches the real KiCad stock libraries at /usr/share/kicad/symbols.
Skip the file if that directory isn't present (matches the pattern in
test_kicraft_symbol_library.py).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kicraft.design.synthesis.symbol_library import (
    DEFAULT_KICAD_SYMBOL_DIR,
    SymbolNotFoundError,
)
from kicraft.design.synthesis.symbol_pinout import lookup_pins

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


def test_ne555_includes_shared_power_pins() -> None:
    """The NE555's VCC (8) / GND (1) live in the NE555D_0_0 sub-symbol -- unit 0,
    body style 0 -- shared across units/styles. They MUST be returned: dropping
    them left the router unable to wire the power pins, so KiCad ERC failed with
    'Pin not connected' on VCC/GND for every stock-symbol IC like the 555."""
    info = lookup_pins("Timer:NE555D")
    by_num = {p["number"]: p for p in info["pins"]}
    assert set(by_num) == {"1", "2", "3", "4", "5", "6", "7", "8"}
    assert by_num["8"]["name"] == "VCC" and by_num["8"]["electrical_type"] == "power_in"
    assert by_num["1"]["name"] == "GND" and by_num["1"]["electrical_type"] == "power_in"


def test_demorgan_body_style_pins_not_duplicated() -> None:
    """A logic gate carries a DeMorgan body style (_<unit>_2) that repeats the
    same pin numbers; the extractor must dedupe by number, not double them."""
    nums = [p["number"] for p in lookup_pins("74xx:74LS00")["pins"]]
    assert len(nums) == len(set(nums)), f"duplicate pins: {nums}"


def test_all_units_returns_every_section() -> None:
    """`all_units=True` exposes every functional unit of a multi-unit symbol so
    callers can reason about a quad op-amp's four amplifiers; the default stays
    unit-1-only (the emitter instantiates one unit per part)."""
    default = lookup_pins("74xx:74LS00")
    full = lookup_pins("74xx:74LS00", all_units=True)
    # A 74LS00 is a quad NAND -> 4 functional units.
    assert default["unit_count"] == full["unit_count"] >= 2
    # all_units returns strictly more pins, each tagged with its unit.
    assert len(full["pins"]) > len(default["pins"])
    assert {p["unit"] for p in full["pins"]} == set(range(1, full["unit_count"] + 1))
    # default path is unchanged: every returned pin is unit 1.
    assert {p["unit"] for p in default["pins"]} == {1}
    # no duplicate pin numbers across units (global numbering + DeMorgan dedupe).
    nums = [p["number"] for p in full["pins"]]
    assert len(nums) == len(set(nums))
