"""§9.27 — _symbol_footprint_pin_mismatches: every wireable symbol pin number
must exist as a pad on the part's footprint.

Regression source (KC-V8YWN8 / KC-B8NQEE): KiCad's generic transistor symbols
use LETTERS as their literal pin numbers (Device:Q_NPN -> B/C/E, Device:Q_NMOS
-> G/D/S). Paired with a numbered footprint (SOT-23 pads 1/2/3) the schematic
stays self-consistent (ERC clean) but no pad can ever bind a net — the seed
PCB leaves the part's pads netless, netless pads produce no ratsnest, and the
routed board ships with electrically dead copper that no DRC gate can see.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import kicraft.design.cli_app as cli_app
from kicraft.design.cli_app import (
    _footprint_pad_numbers,
    _symbol_footprint_pin_mismatches,
)
from kicraft.design.models import BOM, BomPart

_STOCK_SYMBOLS = Path("/usr/share/kicad/symbols/Device.kicad_sym")
needs_stock_kicad = pytest.mark.skipif(
    not _STOCK_SYMBOLS.is_file(), reason="stock KiCad libraries not installed"
)


def _part(ref, symbol, footprint):
    return BomPart(ref=ref, value="x", symbol=symbol, footprint=footprint,
                   sheet="A")


def _bom(*parts):
    return BOM(parts=list(parts), connections=[])


@needs_stock_kicad
def test_letter_pinned_generic_on_numbered_footprint_is_rejected(tmp_path):
    # The exact KC-V8YWN8 pairing.
    bad = _symbol_footprint_pin_mismatches(
        _bom(_part("Q1", "Device:Q_NPN", "Package_TO_SOT_SMD:SOT-23")),
        tmp_path,
    )
    assert len(bad) == 1
    assert "Q1" in bad[0]
    for pin in ("B", "C", "E"):
        assert pin in bad[0]
    assert "dead copper" in bad[0]


@needs_stock_kicad
def test_numbered_symbol_on_matching_footprint_passes(tmp_path):
    bad = _symbol_footprint_pin_mismatches(
        _bom(_part("R1", "Device:R", "Resistor_SMD:R_0603_1608Metric")),
        tmp_path,
    )
    assert bad == []


@needs_stock_kicad
def test_extra_footprint_pads_are_fine(tmp_path, monkeypatch):
    # Thermal/shield/mounting pads beyond the symbol's pins must not flag.
    monkeypatch.setattr(cli_app, "_footprint_pad_numbers",
                        lambda fp, root: {"1", "2", "3", "EP"})
    bad = _symbol_footprint_pin_mismatches(
        _bom(_part("U1", "Device:R", "whatever:with-thermal-pad")),
        tmp_path,
    )
    assert bad == []


def test_no_connect_pins_are_exempt(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_app, "lookup_pins", lambda sym, project_root=None: {
        "pins": [
            {"number": "1", "name": "A", "electrical_type": "passive"},
            {"number": "NC", "name": "NC", "electrical_type": "no_connect"},
        ]
    })
    monkeypatch.setattr(cli_app, "_footprint_pad_numbers",
                        lambda fp, root: {"1"})
    bad = _symbol_footprint_pin_mismatches(
        _bom(_part("U1", "fake:sym", "fake:fp")), tmp_path)
    assert bad == []


def test_wireable_pin_without_pad_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_app, "lookup_pins", lambda sym, project_root=None: {
        "pins": [
            {"number": "1", "name": "IN", "electrical_type": "input"},
            {"number": "G", "name": "G", "electrical_type": "input"},
        ]
    })
    monkeypatch.setattr(cli_app, "_footprint_pad_numbers",
                        lambda fp, root: {"1", "2"})
    bad = _symbol_footprint_pin_mismatches(
        _bom(_part("Q9", "fake:sym", "fake:fp")), tmp_path)
    assert len(bad) == 1 and "Q9" in bad[0] and "G" in bad[0]


def test_unresolvable_symbol_or_footprint_is_someone_elses_problem(tmp_path):
    # Owned by _unresolved_symbols/_unresolved_footprints — no double report.
    bad = _symbol_footprint_pin_mismatches(
        _bom(_part("U9", "NoSuchLib:NoSuchSym", "NoSuchLib:NoSuchFp")),
        tmp_path,
    )
    assert bad == []


@needs_stock_kicad
def test_footprint_pad_numbers_reads_stock_kicad_mod(tmp_path):
    pads = _footprint_pad_numbers("Package_TO_SOT_SMD:SOT-23", tmp_path)
    assert pads == {"1", "2", "3"}


def test_zero_numbered_pad_footprint_rejects_wireable_pins(tmp_path, monkeypatch):
    # 2026-07-19 review §4.2 (live board 627): a footprint that RESOLVES with
    # zero NUMBERED pads (plain NPTH MountingHole -- pad number "") must
    # reject a symbol with wireable pins (Mechanical:MountingHole_Pad), not
    # skip the check like an unresolvable footprint.
    monkeypatch.setattr(cli_app, "lookup_pins", lambda sym, project_root=None: {
        "pins": [{"number": "1", "electrical_type": "passive"}],
        "unit_count": 1,
    })
    monkeypatch.setattr(
        cli_app, "_footprint_pad_numbers", lambda fp, root: set()
    )
    bad = _symbol_footprint_pin_mismatches(
        _bom(_part("H1", "Mechanical:MountingHole_Pad",
                   "MountingHole:MountingHole_3.2mm_M3")),
        tmp_path,
    )
    assert len(bad) == 1
    assert "H1" in bad[0]
    assert "none numbered" in bad[0]


def test_unresolvable_footprint_still_skips(tmp_path, monkeypatch):
    # None (unresolvable) stays someone else's problem (_unresolved_footprints).
    monkeypatch.setattr(cli_app, "lookup_pins", lambda sym, project_root=None: {
        "pins": [{"number": "1", "electrical_type": "passive"}],
        "unit_count": 1,
    })
    monkeypatch.setattr(
        cli_app, "_footprint_pad_numbers", lambda fp, root: None
    )
    bad = _symbol_footprint_pin_mismatches(
        _bom(_part("H1", "Mechanical:MountingHole_Pad", "Nope:Missing")),
        tmp_path,
    )
    assert bad == []
