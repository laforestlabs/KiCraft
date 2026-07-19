"""Electromechanical symbols must carry PASSIVE pins (self-eval 2026-07-19 run_27).

easyeda2kicad exports connector/switch/relay/battery symbols with pins typed
'input' (sometimes 'unspecified'); any net strapped only through such pins then
fails ERC "Input pin not driven by any Output pins" even though the schematic
is electrically fine. Three layers are pinned here:

1. `_normalize_emech_pin_types` retypes input/unspecified -> passive, gated on
   the symbol's reference prefix (J/P/X/CN/K/S/SW/BT/TB), and never touches
   power_* / no_connect / free pins or IC (U) symbols;
2. `add-part` applies it at fetch time (covered via the helper — the fetch
   itself needs network);
3. the vendored bundles that shipped the defect are fixed on disk.
"""
from __future__ import annotations

from pathlib import Path

from kicraft.design.cli_app import _normalize_emech_pin_types

SWITCH_SYM = """(kicad_symbol_lib
  (symbol "DSHP03TSGER"
    (property "Reference" "S" (at 0 0 0))
    (symbol "DSHP03TSGER_0_1"
      (pin input (at -7.62 2.54 0) (length 2.54))
      (pin unspecified (at -7.62 0 0) (length 2.54))
      (pin passive (at -7.62 -2.54 0) (length 2.54))
      (pin no_connect (at 7.62 0 180) (length 2.54))
    )
  )
)
"""

IC_SYM = SWITCH_SYM.replace('"Reference" "S"', '"Reference" "U"')

BATTERY_POWER_SYM = """(kicad_symbol_lib
  (symbol "CR2032H"
    (property "Reference" "BT" (at 0 0 0))
    (symbol "CR2032H_0_1"
      (pin power_out (at 0 2.54 270) (length 2.54))
      (pin input (at 0 -2.54 90) (length 2.54))
    )
  )
)
"""


def test_retypes_input_and_unspecified_on_switch():
    fixed, n = _normalize_emech_pin_types(SWITCH_SYM)
    assert n == 2
    assert "(pin input" not in fixed
    assert "(pin unspecified" not in fixed
    assert fixed.count("(pin passive") == 3
    # deliberate ERC semantics untouched
    assert "(pin no_connect" in fixed


def test_ic_symbols_are_never_touched():
    fixed, n = _normalize_emech_pin_types(IC_SYM)
    assert n == 0
    assert fixed == IC_SYM


def test_power_pins_survive_on_battery_class():
    fixed, n = _normalize_emech_pin_types(BATTERY_POWER_SYM)
    assert n == 1  # only the mistyped 'input' pin
    assert "(pin power_out" in fixed


def test_vendored_defect_bundles_are_fixed_on_disk():
    for name in (
        "dip-switch-3pos",
        "screw-terminal-5mm-3p",
        "usb-micro-b-receptacle-5p",
    ):
        text = Path(f"kicraft/parts_library/{name}/{name}.kicad_sym").read_text()
        assert "(pin input" not in text, name
        _, n = _normalize_emech_pin_types(text)
        assert n == 0, name
