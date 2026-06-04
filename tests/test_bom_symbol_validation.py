"""The BOM-commit symbol-resolution check: catch hallucinated symbols early.

Mirrors the existing footprint check, run at BOM commit so a non-existent symbol
name is rejected where the model can still fix it with the lookup tools, instead
of cascading into an unrecoverable wiring stage-prep failure. Needs the KiCad
stock symbol libraries (same as the other symbol tests in this suite).
"""
from __future__ import annotations

from kicraft.design.cli_app import _unresolved_symbols
from kicraft.design.models import BOM, BomPart


def _part(ref: str, symbol: str) -> BomPart:
    return BomPart(ref=ref, value="x", symbol=symbol,
                   footprint="Resistor_SMD:R_0603_1608Metric", sheet="MAIN")


def test_hallucinated_symbol_is_flagged():
    # 'Conn_02x08' (the real run-2 offender) is not a KiCad symbol; the genuine
    # one is Conn_02x08_Odd_Even etc.
    bom = BOM(parts=[_part("J1", "Connector_Generic:Conn_02x08")])
    bad = _unresolved_symbols(bom)
    assert len(bad) == 1 and "Conn_02x08" in bad[0]


def test_resolvable_stock_symbol_is_not_flagged():
    assert _unresolved_symbols(BOM(parts=[_part("R1", "Device:R")])) == []


def test_only_distinct_symbols_are_checked():
    bom = BOM(parts=[_part("R1", "Device:R"), _part("R2", "Device:R")])
    assert _unresolved_symbols(bom) == []
