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
    InterSheetNet,
    NetConnection,
    PinEndpoint,
    Sheet,
    SheetPin,
)
from kicraft.design.synthesis.emitter import _power_nets_by_pins, _power_nets_with_driver
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


def _opamp_vout_state() -> ConversationState:
    """An MCP6001 op-amp whose VOUT (pin 1, electrical_type ``output``) feeds an
    output header. ``VOUT`` matches the ``^V…OUT`` power-name pattern, so it is
    name-classified as a power net — but its only driver is a *signal* output,
    not a ``power_out``. The regression (run_02 r2r-dac): VOUT slipped the
    power_out-only driver check, got a PWR_FLAG, and ERC reported 'Output and
    Power output are connected'."""
    sheet = Sheet(name="DAC", stem="DAC", function="r2r dac output buffer")
    arch = Architecture(sheets=[sheet], power_nets=["VCC", "VOUT", "GND"],
                        inter_sheet_nets=[])
    # Unity-gain buffer so every op-amp pin connects (VIN- ← VOUT feedback,
    # VIN+ ← AIN input) — keeps the test focused on the VOUT power-output short,
    # not on floating-input ERC noise.
    parts = [
        BomPart(ref="U1", value="MCP6001", symbol="mcp6001:MCP6001T-I_OT",
                footprint="Package_TO_SOT_SMD:SOT-23-5", sheet="DAC"),
        BomPart(ref="J1", value="OUT", symbol="Connector:Conn_01x04_Pin",
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical",
                sheet="DAC"),
    ]
    conns = [
        NetConnection(net_name="VCC", sheet="DAC", endpoints=[      # U1.5 VDD power_in
            PinEndpoint(ref="U1", pin="5"), PinEndpoint(ref="J1", pin="2")]),
        NetConnection(net_name="GND", sheet="DAC", endpoints=[      # U1.2 VSS power_in
            PinEndpoint(ref="U1", pin="2"), PinEndpoint(ref="J1", pin="3")]),
        NetConnection(net_name="VOUT", sheet="DAC", endpoints=[     # U1.1 VOUT output
            PinEndpoint(ref="U1", pin="1"),                         # → J1.1 + VIN- fb
            PinEndpoint(ref="U1", pin="4"), PinEndpoint(ref="J1", pin="1")]),
        NetConnection(net_name="AIN", sheet="DAC", endpoints=[      # U1.3 VIN+ input
            PinEndpoint(ref="U1", pin="3"), PinEndpoint(ref="J1", pin="4")]),
    ]
    return ConversationState(
        project_stem="R2R_DAC",
        intent=IntentSlot(goal="r2r dac"),
        functional_spec=FunctionalSpec(blocks=[
            FunctionalBlock(name="DAC", category="process", purpose="r2r dac buffer")]),
        architecture=arch,
        bom=BOM(parts=parts, connections=conns),
    )


def test_signal_output_driven_power_net_excluded_from_pwr_flag() -> None:
    """A power-named net driven by a plain ``output`` pin (an op-amp VOUT) is
    reported as driven and must NOT get a PWR_FLAG. Regression run_02 r2r-dac:
    the driver check recognized only ``power_out``, so VOUT got a redundant flag
    and ERC shorted 'Output and Power output are connected'. VCC (undriven
    power_in) is still flagged."""
    bom = _opamp_vout_state().bom
    driven = _power_nets_with_driver(bom)
    assert "VOUT" in driven
    assert "VCC" not in driven


@pytest.mark.skipif(shutil.which("kicad-cli") is None, reason="kicad-cli not installed")
def test_opamp_vout_synthesizes_erc_clean_no_output_power_short(tmp_path) -> None:
    """End-to-end run_02 regression: an op-amp VOUT (name-classified power,
    driven by a signal output) must synthesize ERC-clean — no 'Output and Power
    output are connected' from a redundant PWR_FLAG."""
    _artifacts, results = run(_opamp_vout_state(), tmp_path)

    erc = next((r for r in results if r.name.startswith("9.12")), None)
    assert erc is not None and erc.ok, f"ERC: {erc.message if erc else 'missing'}"


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



def _usb_pd_buck_state() -> ConversationState:
    """KC-W93GXR end-to-end: a 2-sheet USB-PD supply where an intermediate input
    bus (V20_BUS) crosses from a SOURCE connector sheet to a BUCK sheet and
    feeds the regulator's VI (power_in). V20_BUS misses the power-NAME patterns
    and has no power_out driver (the connector pins are passive), so the build
    used to die at the ERC gate with VI 'not driven'."""
    source = Sheet(name="SOURCE", stem="SOURCE", function="usb-pd input")
    buck = Sheet(name="BUCK", stem="BUCK", function="3v3 buck")
    arch = Architecture(
        sheets=[source, buck],
        power_nets=["V20_BUS", "+3V3", "GND"],
        inter_sheet_nets=[InterSheetNet(name="V20_BUS", endpoints=[
            SheetPin(sheet="SOURCE", direction="output"),
            SheetPin(sheet="BUCK", direction="input")])],
    )
    parts = [
        BomPart(ref="J1", value="USB-PD", symbol="Connector:Conn_01x02_Pin",
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
                sheet="SOURCE"),
        BomPart(ref="U1", value="AMS1117-3.3", symbol="Regulator_Linear:AMS1117-3.3",
                footprint="Package_TO_SOT_SMD:SOT-223-3_TabPin2", sheet="BUCK"),
        BomPart(ref="C1", value="10uF", symbol="Device:C",
                footprint="Capacitor_SMD:C_0805_2012Metric", sheet="BUCK"),
        BomPart(ref="C2", value="22uF", symbol="Device:C",
                footprint="Capacitor_SMD:C_0805_2012Metric", sheet="BUCK"),
    ]
    conns = [
        NetConnection(net_name="V20_BUS", sheet="SOURCE", endpoints=[
            PinEndpoint(ref="J1", pin="1")]),
        NetConnection(net_name="GND", sheet="SOURCE", endpoints=[
            PinEndpoint(ref="J1", pin="2")]),
        NetConnection(net_name="V20_BUS", sheet="BUCK", endpoints=[      # VI power_in
            PinEndpoint(ref="U1", pin="3"), PinEndpoint(ref="C1", pin="1")]),
        NetConnection(net_name="+3V3", sheet="BUCK", endpoints=[         # VO power_out
            PinEndpoint(ref="U1", pin="2"), PinEndpoint(ref="C2", pin="1")]),
        NetConnection(net_name="GND", sheet="BUCK", endpoints=[
            PinEndpoint(ref="U1", pin="1"), PinEndpoint(ref="C1", pin="2"),
            PinEndpoint(ref="C2", pin="2")]),
    ]
    return ConversationState(
        project_stem="USB_PD_BUCK",
        intent=IntentSlot(goal="usb-pd buck supply"),
        functional_spec=FunctionalSpec(blocks=[
            FunctionalBlock(name="POWER", category="power", purpose="usb-pd to 3v3")]),
        architecture=arch,
        bom=BOM(parts=parts, connections=conns),
    )


@pytest.mark.skipif(shutil.which("kicad-cli") is None, reason="kicad-cli not installed")
def test_intermediate_input_bus_synthesizes_erc_clean(tmp_path) -> None:
    """KC-W93GXR regression, the actual build gate: an undriven intermediate
    input bus feeding a regulator's VI (power_in) must synthesize ERC-clean.
    The emitter assigns V20_BUS a flag (pin-aware); the fix makes route_sheet
    place it even though the bus is rendered as a signal/hier net."""
    _artifacts, results = run(_usb_pd_buck_state(), tmp_path)

    erc = next((r for r in results if r.name.startswith("9.12")), None)
    assert erc is not None and erc.ok, f"ERC: {erc.message if erc else 'missing'}"


def test_pin_detected_power_net_gets_pwr_flag() -> None:
    """A net not matching POWER_NET_PATTERNS (e.g. PD_20V) but carrying
    power_in pins must still be detected as a power net and receive a
    PWR_FLAG when undriven.  Regression: KC-HFW3R6 — PD_20V feeds three
    buck regulator VIN pins (all power_in) but the source PFET has only
    unspecified pins, so the net went undetected and ERC flagged every
    regulator's VIN as undriven."""
    from kicraft.design.synthesis.emitter import _power_nets_by_pins, _power_nets_with_driver

    sheet = Sheet(name="RAIL", stem="RAIL", function="buck")
    # U1: TPS54331 buck — VIN (pin 2) = power_in, PH (pin 8) = power_out
    # Q1: AO3401A PFET — all pins unspecified (simulates the real case)
    parts = [
        BomPart(ref="U1", value="TPS54331DDAR", symbol="tps54331:TPS54331DDAR",
                footprint="tps54331:SOIC-8_L4.9-W3.9-P1.27-LS6.0-BL-EP",
                sheet="RAIL"),
        BomPart(ref="Q1", value="AO3401A", symbol="ao3401a:AO3401A",
                footprint="ao3401a:SOT-23_L2.9-W1.3-P1.90-LS2.4-BR",
                sheet="RAIL"),
    ]
    conns = [
        # PD_20V: Q1.3 (source, unspecified) → U1.2 (VIN, power_in)
        NetConnection(net_name="PD_20V", sheet="RAIL", endpoints=[
            PinEndpoint(ref="Q1", pin="3"),
            PinEndpoint(ref="U1", pin="2"),
        ]),
        # +12V: U1.8 (PH, power_out) → output
        NetConnection(net_name="+12V", sheet="RAIL", endpoints=[
            PinEndpoint(ref="U1", pin="8"),
        ]),
        NetConnection(net_name="GND", sheet="RAIL", endpoints=[
            PinEndpoint(ref="U1", pin="7"),
        ]),
    ]
    bom = BOM(parts=parts, connections=conns)

    pin_power = _power_nets_by_pins(bom)
    assert "PD_20V" in pin_power, (
        "PD_20V must be detected as a power net — U1.2 is power_in"
    )

    driven = _power_nets_with_driver(bom)
    assert "PD_20V" not in driven, (
        "PD_20V must NOT be driven — Q1.3 is unspecified, not power_out"
    )
    assert "+12V" in driven, (
        "+12V must be driven — U1.8 is power_out"
    )