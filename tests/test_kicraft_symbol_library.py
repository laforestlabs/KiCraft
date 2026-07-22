"""Tests for kicraft.synthesis.symbol_library.

These touch the real KiCad stock libraries at /usr/share/kicad/symbols.
Skip the file if that directory isn't present (e.g. on machines without KiCad).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from kicraft.design.synthesis.symbol_library import (
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


def test_passive_device_input_pins_retyped_passive(tmp_path: Path) -> None:
    # easyeda2kicad imports switch contacts typed `input`, which trips KiCad ERC
    # pin_not_driven ("Input pin not driven by any Output pins") on every
    # non-power net. extract_symbol_block must retype a *passive* device's pins
    # `passive` (how KiCad's own Switch:/Device: symbols model a contact), keyed
    # off the symbol's intrinsic Reference prefix. (Connectors -> bidirectional;
    # see test_connector_input_pins_retyped_bidirectional.)
    lib = tmp_path / "Sw.kicad_sym"
    lib.write_text(
        '(kicad_symbol_lib (version 20211014)\n'
        '\t(symbol "DSW"\n'
        '\t\t(property "Reference" "SW" (at 0 0 0))\n'
        '\t\t(symbol "DSW_0_1"\n'
        '\t\t\t(pin input line (at -10 2.54 0) (length 5)'
        ' (name "A1") (number "1"))\n'
        '\t\t\t(pin input line (at 10 2.54 180) (length 5)'
        ' (name "B1") (number "6"))\n'
        '\t\t)\n'
        '\t)\n'
        ')\n'
    )
    block = extract_symbol_block("Sw", "DSW", stock_dir=tmp_path)
    assert "(pin input" not in block
    assert block.count("(pin passive") == 2


def test_relay_input_pins_retyped_passive(tmp_path: Path) -> None:
    # easyeda2kicad types a bare relay symbol's Reference "RLY" (instances are
    # placed K1..Kn) and its coil/contact pins arrive `input`, tripping ERC
    # pin_not_driven on the coil low-side net (the driver, e.g. a ULN2003
    # collector, is typed Unspecified, not Output). A relay coil is a passive
    # load, not a logic input, so the normalizer must retype it `passive`
    # (matching KiCad's stock Relay:* library).
    lib = tmp_path / "Rly.kicad_sym"
    lib.write_text(
        '(kicad_symbol_lib (version 20211014)\n'
        '\t(symbol "SRD"\n'
        '\t\t(property "Reference" "RLY" (at 0 0 0))\n'
        '\t\t(symbol "SRD_0_1"\n'
        '\t\t\t(pin input line (at -10 2.54 0) (length 5)'
        ' (name "coil+") (number "1"))\n'
        '\t\t\t(pin input line (at -10 0 0) (length 5)'
        ' (name "coil-") (number "2"))\n'
        '\t\t\t(pin input line (at 10 0 180) (length 5)'
        ' (name "COM") (number "3"))\n'
        '\t\t)\n'
        '\t)\n'
        ')\n'
    )
    block = extract_symbol_block("Rly", "SRD", stock_dir=tmp_path)
    assert "(pin input" not in block
    assert block.count("(pin passive") == 3


def test_battery_input_pins_retyped_passive(tmp_path: Path) -> None:
    # An easyeda2kicad battery holder (Reference "BT") arrives with `input`
    # terminals, tripping ERC pin_not_driven. A battery cell is a passive source
    # in KiCad's stock model (Device:Battery_Cell pins are passive), so the
    # normalizer must retype its contacts `passive`.
    lib = tmp_path / "Bat.kicad_sym"
    lib.write_text(
        '(kicad_symbol_lib (version 20211014)\n'
        '\t(symbol "HOLDER"\n'
        '\t\t(property "Reference" "BT" (at 0 0 0))\n'
        '\t\t(symbol "HOLDER_0_1"\n'
        '\t\t\t(pin input line (at -10 2.54 0) (length 5)'
        ' (name "+") (number "1"))\n'
        '\t\t\t(pin input line (at -10 0 0) (length 5)'
        ' (name "-") (number "2"))\n'
        '\t\t)\n'
        '\t)\n'
        ')\n'
    )
    block = extract_symbol_block("Bat", "HOLDER", stock_dir=tmp_path)
    assert "(pin input" not in block
    assert block.count("(pin passive") == 2


def test_connector_input_pins_retyped_bidirectional(tmp_path: Path) -> None:
    # A connector is a boundary to a possibly-active off-board device (a socketed
    # microSD card, a sensor module). Its signal contacts carry a live bus whose
    # driver may be off-schematic, so the normalizer retypes connector `input`
    # pins `bidirectional` -- NOT `passive`, which would declare the bus inert
    # and mask a genuinely floating host-driven line.
    lib = tmp_path / "Conn.kicad_sym"
    lib.write_text(
        '(kicad_symbol_lib (version 20211014)\n'
        '\t(symbol "SOCKET"\n'
        '\t\t(property "Reference" "J" (at 0 0 0))\n'
        '\t\t(symbol "SOCKET_0_1"\n'
        '\t\t\t(pin input line (at -10 2.54 0) (length 5)'
        ' (name "CMD") (number "3"))\n'
        '\t\t\t(pin input line (at -10 0 0) (length 5)'
        ' (name "CLK") (number "5"))\n'
        '\t\t)\n'
        '\t)\n'
        ')\n'
    )
    block = extract_symbol_block("Conn", "SOCKET", stock_dir=tmp_path)
    assert "(pin input" not in block
    assert block.count("(pin bidirectional") == 2
    assert "(pin passive" not in block


def test_assigned_refdes_overrides_bogus_intrinsic_reference(tmp_path: Path) -> None:
    # KC-8DXUS6: easyeda fills a part's intrinsic Reference with an arbitrary
    # string ("Card" for a microSD socket) whose alpha prefix matches no device
    # class, so the intrinsic path falls to the safe default (bidirectional --
    # never a surviving `input`). KiCraft's assigned refdes is authoritative and
    # must steer the part into the RIGHT bucket: a tactile switch assigned SW1 is
    # a passive contact, so its `input` pins become `passive`, not the default
    # bidirectional.
    lib = tmp_path / "Card.kicad_sym"
    lib.write_text(
        '(kicad_symbol_lib (version 20211014)\n'
        '\t(symbol "TF"\n'
        '\t\t(property "Reference" "Card" (at 0 0 0))\n'
        '\t\t(symbol "TF_0_1"\n'
        '\t\t\t(pin input line (at -10 2.54 0) (length 5)'
        ' (name "A") (number "1"))\n'
        '\t\t)\n'
        '\t)\n'
        ')\n'
    )
    # Unrecognized intrinsic "Card" -> safe default: bidirectional, never `input`.
    default = extract_symbol_block("Card", "TF", stock_dir=tmp_path)
    assert "(pin input" not in default
    assert default.count("(pin bidirectional") == 1
    # Assigned refdes SW1 -> switch -> passive, overriding the bogus intrinsic.
    fixed = extract_symbol_block("Card", "TF", stock_dir=tmp_path, ref_prefix="SW1")
    assert "(pin input" not in fixed
    assert fixed.count("(pin passive") == 1
    assert "(pin bidirectional" not in fixed


def test_assigned_refdes_retypes_ic_pins_bidirectional(tmp_path: Path) -> None:
    # KC-UFHJ42: an IC input pin (refdes U3) in KiCraft is driven by pins typed
    # Unspecified (the curated-library convention), which KiCad does not count as
    # an Output driver -- so a legitimately-wired input trips ERC pin_not_driven.
    # The assigned-refdes path retypes it `bidirectional` (needs no driver, yet
    # satisfies a connected input); a truly floating pin is caught by the
    # net-coverage gate, not this check.
    lib = tmp_path / "Mcu.kicad_sym"
    lib.write_text(
        '(kicad_symbol_lib (version 20211014)\n'
        '\t(symbol "MCU"\n'
        '\t\t(property "Reference" "U" (at 0 0 0))\n'
        '\t\t(symbol "MCU_0_1"\n'
        '\t\t\t(pin input line (at -10 0 0) (length 5)'
        ' (name "IN") (number "1"))\n'
        '\t\t)\n'
        '\t)\n'
        ')\n'
    )
    block = extract_symbol_block("Mcu", "MCU", stock_dir=tmp_path, ref_prefix="U3")
    assert "(pin input" not in block
    assert block.count("(pin bidirectional") == 1


def test_active_device_input_pins_retyped_bidirectional(tmp_path: Path) -> None:
    # KC-UFHJ42: active devices (Reference "U", "Q", ...) get their `input` pins
    # retyped `bidirectional`. In KiCraft's Unspecified/passive driver convention
    # KiCad never sees an Output driver, so an MCU RUN pin (fed by a reset button)
    # or a MOSFET gate (fed by a GPIO) would otherwise fail ERC pin_not_driven.
    # bidirectional needs no driver yet satisfies a connected input; a genuinely
    # floating pin is caught by the §9.11 net-coverage gate, not by pin type.
    lib = tmp_path / "Ic.kicad_sym"
    lib.write_text(
        '(kicad_symbol_lib (version 20211014)\n'
        '\t(symbol "GATE"\n'
        '\t\t(property "Reference" "U" (at 0 0 0))\n'
        '\t\t(symbol "GATE_0_1"\n'
        '\t\t\t(pin input line (at -10 0 0) (length 5)'
        ' (name "IN") (number "1"))\n'
        '\t\t)\n'
        '\t)\n'
        ')\n'
    )
    block = extract_symbol_block("Ic", "GATE", stock_dir=tmp_path)
    assert "(pin input" not in block
    assert block.count("(pin bidirectional") == 1


def test_stock_transistor_gate_retyped_bidirectional() -> None:
    # KC-UFHJ42, real-symbol guard: a discrete MOSFET's gate is typed `input` in
    # KiCad's stock Device library; driven by a GPIO typed Unspecified it fails
    # ERC pin_not_driven (self-eval run 488, 8x Q_NMOS LED-channel gates). The
    # embed choke point must retype the gate bidirectional while leaving the
    # source/drain (passive) alone.
    block = extract_symbol_block("Device", "Q_NMOS", ref_prefix="Q1")
    assert "(pin input" not in block
    assert "(pin bidirectional" in block


@pytest.mark.parametrize("node", ["PH", "SW", "LX", "PHASE", "SWITCH"])
def test_switch_node_pin_retyped_power_out(tmp_path: Path, node: str) -> None:
    # A switching regulator's switch/phase node drives the inductor — it is an
    # output. Vendored/easyeda symbols mistype it `power_in`, tripping ERC
    # power_pin_not_driven on the wired switch net. extract_symbol_block must
    # retype it `power_out` while leaving genuine power inputs (VIN/GND) alone.
    lib = tmp_path / "Reg.kicad_sym"
    lib.write_text(
        '(kicad_symbol_lib (version 20211014)\n'
        '\t(symbol "BUCK"\n'
        '\t\t(property "Reference" "U" (at 0 0 0))\n'
        '\t\t(symbol "BUCK_0_1"\n'
        '\t\t\t(pin power_in line (at -10 5 0) (length 5)'
        ' (name "VIN") (number "1"))\n'
        '\t\t\t(pin power_in line (at -10 -5 0) (length 5)'
        ' (name "GND") (number "2"))\n'
        f'\t\t\t(pin power_in line (at 10 0 180) (length 5)'
        f' (name "{node}") (number "3"))\n'
        '\t\t)\n'
        '\t)\n'
        ')\n'
    )
    block = extract_symbol_block("Reg", "BUCK", stock_dir=tmp_path)
    assert f'power_out line (at 10 0 180) (length 5) (name "{node}")' in block
    # VIN/GND power inputs are untouched.
    assert block.count("(pin power_in") == 2


def test_tps54331_vendored_symbol_ph_is_power_out() -> None:
    # The vendored TPS54331 data fix (+ the generalized normalizer): pin 8 "PH"
    # must resolve to power_out, not power_in.
    block = extract_symbol_block("tps54331", "TPS54331DDAR")
    m = re.search(r'\(pin\s+(\w+)\b[^()]*(?:\([^()]*\)[^()]*)*?\(name\s+"PH"', block)
    assert m is not None and m.group(1) == "power_out"


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
