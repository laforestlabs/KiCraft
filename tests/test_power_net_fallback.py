"""Power nets without a stock KiCad power symbol must not crash synthesis.

Regression for the 555 / coin-cell web failure: the wiring named the supply
rail ``VBAT``; ``_POWER_SYMBOL_MAP`` mapped it to ``power:VBAT`` which does NOT
exist in stock KiCad, so the emitter died with ``SymbolNotFoundError`` inside
``_emit_leaf`` before ERC ever ran (and the web app's exit-5 ERC-recovery never
engaged). The fix renders any power-classified net that lacks a stock power
symbol as a global label (hierarchy-wide by name) plus a single PWR_FLAG.
"""
from __future__ import annotations

import shutil

import pytest

from kicraft.design.models import (
    BOM,
    Architecture,
    BomPart,
    ConversationState,
    FunctionalBlock,
    FunctionalSpec,
    IntentSlot,
    NetConnection,
    PinEndpoint,
    Sheet,
)
from kicraft.design.synthesis.placement import place_sheet
from kicraft.design.synthesis.router import _POWER_SYMBOL_MAP, route_sheet
from kicraft.design.synthesis.symbol_library import (
    DEFAULT_KICAD_SYMBOL_DIR,
    SymbolNotFoundError,
)
from kicraft.design.synthesis.symbol_pinout import lookup_pins
from kicraft.design.synthesize import run

pytestmark = pytest.mark.skipif(
    not DEFAULT_KICAD_SYMBOL_DIR.is_dir(),
    reason="KiCad symbol library not installed",
)


def test_power_symbol_map_targets_exist() -> None:
    """Every mapped target must resolve to a real stock symbol. A phantom
    (the old power:VBAT / power:VSYS / power:PGND) crashes synthesis at emit
    time, so the map must only ever promise symbols that exist."""
    missing = []
    for _name, lib_id in _POWER_SYMBOL_MAP:
        try:
            lookup_pins(lib_id)
        except SymbolNotFoundError:
            missing.append(lib_id)
    assert not missing, f"_POWER_SYMBOL_MAP points at non-existent symbols: {missing}"


def test_unmapped_power_rail_emits_global_label_not_symbol() -> None:
    """A power-classified net with no stock symbol (VBAT) renders as a global
    label, never a ``power:<name>`` symbol that may not exist."""
    sheet = Sheet(name="RAIL", stem="RAIL", function="rail")
    arch = Architecture(sheets=[sheet], power_nets=["VBAT", "GND"], inter_sheet_nets=[])
    parts = [
        BomPart(ref="R1", value="1k", symbol="Device:R",
                footprint="Resistor_SMD:R_0805_2012Metric", sheet="RAIL"),
        BomPart(ref="R2", value="1k", symbol="Device:R",
                footprint="Resistor_SMD:R_0805_2012Metric", sheet="RAIL"),
    ]
    bom = BOM(parts=parts, connections=[
        NetConnection(net_name="VBAT", sheet="RAIL", endpoints=[
            PinEndpoint(ref="R1", pin="1"), PinEndpoint(ref="R2", pin="1")])])
    routed = route_sheet("RAIL", "RAIL", place_sheet(sheet, parts, bom), bom, arch)

    assert [g.text for g in routed.global_labels] == ["VBAT", "VBAT"]
    # Not a stock power symbol (would be power:VBAT -> missing) ...
    assert all("VBAT" not in ps.lib_id for ps in routed.power_symbols)
    # ... and not a plain sheet-local signal label either.
    assert routed.labels == []


def _minimal_vbat_state() -> ConversationState:
    sheet = Sheet(name="MAIN", stem="MAIN", function="coin-cell rail")
    arch = Architecture(sheets=[sheet], power_nets=["VBAT", "GND"], inter_sheet_nets=[])
    parts = [
        BomPart(ref="BT1", value="CR2032", symbol="Device:Battery_Cell",
                footprint="Battery:BatteryHolder_Keystone_3034_1x20mm", sheet="MAIN"),
        BomPart(ref="R1", value="1k", symbol="Device:R",
                footprint="Resistor_SMD:R_0805_2012Metric", sheet="MAIN"),
    ]
    conns = [
        NetConnection(net_name="VBAT", sheet="MAIN", endpoints=[
            PinEndpoint(ref="BT1", pin="1"), PinEndpoint(ref="R1", pin="1")]),
        NetConnection(net_name="GND", sheet="MAIN", endpoints=[
            PinEndpoint(ref="BT1", pin="2"), PinEndpoint(ref="R1", pin="2")]),
    ]
    return ConversationState(
        project_stem="VBAT_RAIL",
        intent=IntentSlot(goal="coin cell rail"),
        functional_spec=FunctionalSpec(blocks=[
            FunctionalBlock(name="POWER", category="power", purpose="CR2032 supply")]),
        architecture=arch,
        bom=BOM(parts=parts, connections=conns),
    )


@pytest.mark.skipif(shutil.which("kicad-cli") is None, reason="kicad-cli not installed")
def test_vbat_rail_synthesizes_without_crash_and_erc_clean(tmp_path) -> None:
    """The exact failure class: a VBAT rail must synthesize (no crash) and be
    ERC-clean, rendered as a global label rather than a missing power symbol."""
    _artifacts, results = run(_minimal_vbat_state(), tmp_path)  # must not raise

    erc = next((r for r in results if r.name.startswith("9.12")), None)
    assert erc is not None and erc.ok, f"ERC: {erc.message if erc else 'missing'}"

    sch = (tmp_path / "MAIN.kicad_sch").read_text()
    assert 'global_label "VBAT"' in sch
    assert "power:VBAT" not in sch
