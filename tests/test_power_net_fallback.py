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
from kicraft.design.synthesis.emitter import _power_nets_with_driver
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


def _regulated_rail_state() -> ConversationState:
    """An AMS1117 LDO: +5V in (connector-fed, undriven), +3V3 out (driven by the
    regulator's power-output VO pin), GND. The +3V3 rail is the regression case —
    it already has a real driver and must NOT also get a PWR_FLAG."""
    sheet = Sheet(name="REG", stem="REG", function="3v3 ldo")
    arch = Architecture(sheets=[sheet], power_nets=["+5V", "+3V3", "GND"],
                        inter_sheet_nets=[])
    parts = [
        BomPart(ref="U1", value="AMS1117-3.3", symbol="Regulator_Linear:AMS1117-3.3",
                footprint="Package_TO_SOT_SMD:SOT-223-3_TabPin2", sheet="REG"),
        BomPart(ref="C1", value="10uF", symbol="Device:C",
                footprint="Capacitor_SMD:C_0805_2012Metric", sheet="REG"),
        BomPart(ref="C2", value="22uF", symbol="Device:C",
                footprint="Capacitor_SMD:C_0805_2012Metric", sheet="REG"),
    ]
    conns = [
        NetConnection(net_name="+5V", sheet="REG", endpoints=[      # power_in only
            PinEndpoint(ref="U1", pin="3"), PinEndpoint(ref="C1", pin="1")]),
        NetConnection(net_name="+3V3", sheet="REG", endpoints=[     # U1.2 VO = power_out
            PinEndpoint(ref="U1", pin="2"), PinEndpoint(ref="C2", pin="1")]),
        NetConnection(net_name="GND", sheet="REG", endpoints=[
            PinEndpoint(ref="U1", pin="1"), PinEndpoint(ref="C1", pin="2"),
            PinEndpoint(ref="C2", pin="2")]),
    ]
    return ConversationState(
        project_stem="LDO_3V3",
        intent=IntentSlot(goal="3v3 supply"),
        functional_spec=FunctionalSpec(blocks=[
            FunctionalBlock(name="POWER", category="power", purpose="AMS1117 3v3")]),
        architecture=arch,
        bom=BOM(parts=parts, connections=conns),
    )


def test_driven_power_rail_excluded_from_pwr_flag() -> None:
    """Only the rail with a real power-output driver (+3V3, fed by the LDO's VO
    pin) is reported as driven; connector/passive-fed rails (+5V, GND) are not.
    A PWR_FLAG on a driven net would short ERC ('Power output and Power output
    are connected') — the exact failure seen on a charger's V_BAT output rail."""
    bom = _regulated_rail_state().bom
    assert _power_nets_with_driver(bom) == {"+3V3"}


@pytest.mark.skipif(shutil.which("kicad-cli") is None, reason="kicad-cli not installed")
def test_driven_rail_synthesizes_erc_clean_no_power_output_short(tmp_path) -> None:
    """End-to-end: a regulator-driven rail must be ERC-clean. With the old
    'flag every power net' logic the LDO's VO (power_out) plus the rail's
    PWR_FLAG (also power_out) tripped a power-output short; the driver-aware
    flag assignment fixes it without leaving +5V/GND undriven."""
    _artifacts, results = run(_regulated_rail_state(), tmp_path)

    erc = next((r for r in results if r.name.startswith("9.12")), None)
    assert erc is not None and erc.ok, f"ERC: {erc.message if erc else 'missing'}"


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
