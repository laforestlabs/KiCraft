"""End-to-end ERC golden for the cluster schematic layout.

Builds representative leaves (a clustered LDO and a two-anchor USB-UART
sheet with decoupling caps, a crystal, series resistors and a pull-up),
emits the real .kicad_sch set, and runs the §9.12 ERC check. A regression
in placement/router (a wire that doesn't land on its pin, two power symbols
shorting, an off-grid endpoint) shows up here as an ERC error.
"""
from __future__ import annotations

import shutil

import pytest

from kicraft.design.models import (
    BOM, Architecture, BomPart, InterSheetNet, NetConnection, PinEndpoint,
    Sheet, SheetPin)
from kicraft.design.synthesis.emitter import emit_schematic
from kicraft.design.synthesis.symbol_library import DEFAULT_KICAD_SYMBOL_DIR
from kicraft.design.synthesis.validation import (
    check_erc,
    check_netlist_faithfulness,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_KICAD_SYMBOL_DIR.is_dir() or shutil.which("kicad-cli") is None,
    reason="needs KiCad symbols + kicad-cli",
)


def _P(r, p):
    return PinEndpoint(ref=r, pin=p)


def _ldo() -> tuple[Architecture, BOM]:
    arch = Architecture(
        sheets=[Sheet(name="LDO 3V3", stem="LDO_3V3", function="ldo")],
        power_nets=["VBUS", "+3V3", "GND"], inter_sheet_nets=[])
    parts = [
        BomPart(ref="U1", value="AP2112K-3.3",
                symbol="Regulator_Linear:AP2112K-3.3",
                footprint="Package_TO_SOT_SMD:SOT-23-5", sheet="LDO 3V3"),
        BomPart(ref="C1", value="1uF", symbol="Device:C",
                footprint="Capacitor_SMD:C_0402_1005Metric", sheet="LDO 3V3"),
        BomPart(ref="C2", value="1uF", symbol="Device:C",
                footprint="Capacitor_SMD:C_0402_1005Metric", sheet="LDO 3V3"),
        BomPart(ref="R1", value="100k", symbol="Device:R",
                footprint="Resistor_SMD:R_0402_1005Metric", sheet="LDO 3V3"),
    ]
    bom = BOM(
        parts=parts, ic_groups={"U1": ["C1", "C2", "R1"]},
        connections=[
            NetConnection(net_name="VBUS", sheet="LDO 3V3", endpoints=[
                _P("U1", "1"), _P("C1", "1"), _P("R1", "1")]),
            NetConnection(net_name="+3V3", sheet="LDO 3V3", endpoints=[
                _P("U1", "5"), _P("C2", "1")]),
            NetConnection(net_name="GND", sheet="LDO 3V3", endpoints=[
                _P("U1", "2"), _P("C1", "2"), _P("C2", "2")]),
            NetConnection(net_name="EN_PU", sheet="LDO 3V3", endpoints=[
                _P("U1", "3"), _P("R1", "2")]),
        ],
        no_connect_pins=[_P("U1", "4")])
    return arch, bom


def _usb_uart() -> tuple[Architecture, BOM]:
    arch = Architecture(
        sheets=[Sheet(name="USB UART", stem="USB_UART", function="ch340g")],
        power_nets=["VCC", "V3", "GND"], inter_sheet_nets=[])
    parts = [
        BomPart(ref="U1", value="CH340G", symbol="Interface_USB:CH340G",
                footprint="Package_SO:SOIC-16_3.9x9.9mm_P1.27mm", sheet="USB UART"),
        BomPart(ref="J1", value="UART", symbol="Connector_Generic:Conn_01x06",
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Vertical",
                sheet="USB UART"),
        BomPart(ref="C1", value="100nF", symbol="Device:C",
                footprint="Capacitor_SMD:C_0402_1005Metric", sheet="USB UART"),
        BomPart(ref="C2", value="100nF", symbol="Device:C",
                footprint="Capacitor_SMD:C_0402_1005Metric", sheet="USB UART"),
        BomPart(ref="C3", value="10uF", symbol="Device:C",
                footprint="Capacitor_SMD:C_0805_2012Metric", sheet="USB UART"),
        BomPart(ref="R1", value="33", symbol="Device:R",
                footprint="Resistor_SMD:R_0402_1005Metric", sheet="USB UART"),
        BomPart(ref="R2", value="10k", symbol="Device:R",
                footprint="Resistor_SMD:R_0402_1005Metric", sheet="USB UART"),
        BomPart(ref="R3", value="33", symbol="Device:R",
                footprint="Resistor_SMD:R_0402_1005Metric", sheet="USB UART"),
        BomPart(ref="Y1", value="12MHz", symbol="Device:Crystal",
                footprint="Crystal:Crystal_SMD_3225-4Pin_3.2x2.5mm", sheet="USB UART"),
    ]
    conns = [
        NetConnection(net_name="VCC", sheet="USB UART", endpoints=[
            _P("U1", "16"), _P("C1", "1"), _P("C3", "1"), _P("J1", "1"), _P("R2", "1")]),
        NetConnection(net_name="V3", sheet="USB UART", endpoints=[
            _P("U1", "4"), _P("C2", "1")]),
        NetConnection(net_name="GND", sheet="USB UART", endpoints=[
            _P("U1", "1"), _P("C1", "2"), _P("C2", "2"), _P("C3", "2"), _P("J1", "6")]),
        NetConnection(net_name="XTAL_XI", sheet="USB UART", endpoints=[
            _P("U1", "7"), _P("Y1", "1")]),
        NetConnection(net_name="XTAL_XO", sheet="USB UART", endpoints=[
            _P("U1", "8"), _P("Y1", "2")]),
        NetConnection(net_name="TXD_S", sheet="USB UART", endpoints=[
            _P("U1", "2"), _P("R1", "1")]),
        NetConnection(net_name="UART_TX", sheet="USB UART", endpoints=[
            _P("R1", "2"), _P("J1", "3")]),
        NetConnection(net_name="RXD_S", sheet="USB UART", endpoints=[
            _P("U1", "3"), _P("R3", "1")]),
        NetConnection(net_name="UART_RX", sheet="USB UART", endpoints=[
            _P("R3", "2"), _P("J1", "4")]),
        NetConnection(net_name="DTR_PU", sheet="USB UART", endpoints=[
            _P("U1", "13"), _P("R2", "2")]),
    ]
    nc = [_P("U1", n) for n in ("5", "6", "9", "10", "11", "12", "14", "15")]
    nc += [_P("J1", "2"), _P("J1", "5")]
    bom = BOM(
        parts=parts,
        ic_groups={"U1": ["C1", "C2", "C3", "R1", "R2", "R3", "Y1"]},
        signal_flow_order=["U1", "J1"], connections=conns, no_connect_pins=nc)
    return arch, bom


def _usb_pd_buck() -> tuple[Architecture, BOM]:
    """KC-W93GXR shape: a SOURCE sheet whose connector feeds an intermediate
    input bus V20_BUS (undriven -- connector pins are passive) that crosses to a
    BUCK sheet feeding the regulator's VI (power_in). V20_BUS misses the
    power-NAME patterns, so it is rendered as a hierarchical (signal) inter-sheet
    net; its PWR_FLAG must be placed by route_sheet for ERC to pass."""
    source = Sheet(name="SOURCE", stem="SOURCE", function="usb-pd input")
    buck = Sheet(name="BUCK", stem="BUCK", function="buck")
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
    bom = BOM(parts=parts, connections=[
        NetConnection(net_name="V20_BUS", sheet="SOURCE", endpoints=[_P("J1", "1")]),
        NetConnection(net_name="GND", sheet="SOURCE", endpoints=[_P("J1", "2")]),
        NetConnection(net_name="V20_BUS", sheet="BUCK", endpoints=[
            _P("U1", "3"), _P("C1", "1")]),      # U1.3 = VI (power_in)
        NetConnection(net_name="+3V3", sheet="BUCK", endpoints=[
            _P("U1", "2"), _P("C2", "1")]),      # U1.2 = VO (power_out) -> driven
        NetConnection(net_name="GND", sheet="BUCK", endpoints=[
            _P("U1", "1"), _P("C1", "2"), _P("C2", "2")]),
    ])
    return arch, bom


@pytest.mark.parametrize("builder,stem", [(_ldo, "LDODEMO"), (_usb_uart, "UARTDEMO")])
def test_clustered_sheet_is_erc_clean(tmp_path, builder, stem) -> None:
    arch, bom = builder()
    emit_schematic(tmp_path, stem, arch, bom, title=stem)
    result = check_erc(tmp_path, stem)
    assert result.ok, f"{stem} ERC: {result.message}\n" + "\n".join(result.offenders)


def _dual_opamp() -> tuple[Architecture, BOM]:
    """run_28 audio-jack-buffer shape: a TL072 dual op-amp as two unity-gain
    buffers. Unit B's pins (5/6/7) and the power unit's (4/8) only reach the
    netlist if the emitter instantiates EVERY unit, not just unit A."""
    arch = Architecture(
        sheets=[Sheet(name="BUFFER", stem="BUFFER", function="opamp buffer")],
        power_nets=["VCC", "GND"], inter_sheet_nets=[])
    parts = [
        BomPart(ref="U1", value="TL072", symbol="Amplifier_Operational:TL072",
                footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm", sheet="BUFFER"),
        BomPart(ref="J1", value="IN", symbol="Connector_Generic:Conn_01x03",
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical",
                sheet="BUFFER"),
        BomPart(ref="J2", value="OUT", symbol="Connector_Generic:Conn_01x03",
                footprint="Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical",
                sheet="BUFFER"),
        BomPart(ref="C1", value="100nF", symbol="Device:C",
                footprint="Capacitor_SMD:C_0402_1005Metric", sheet="BUFFER"),
    ]
    bom = BOM(
        parts=parts, ic_groups={"U1": ["C1"]},
        signal_flow_order=["J1", "U1", "J2"],
        connections=[
            NetConnection(net_name="IN1", sheet="BUFFER", endpoints=[
                _P("J1", "1"), _P("U1", "3")]),
            NetConnection(net_name="IN2", sheet="BUFFER", endpoints=[
                _P("J1", "2"), _P("U1", "5")]),
            NetConnection(net_name="OUT1", sheet="BUFFER", endpoints=[
                _P("U1", "1"), _P("U1", "2"), _P("J2", "1")]),
            NetConnection(net_name="OUT2", sheet="BUFFER", endpoints=[
                _P("U1", "7"), _P("U1", "6"), _P("J2", "2")]),
            NetConnection(net_name="VCC", sheet="BUFFER", endpoints=[
                _P("U1", "8"), _P("C1", "1")]),
            NetConnection(net_name="GND", sheet="BUFFER", endpoints=[
                _P("U1", "4"), _P("C1", "2"), _P("J1", "3"), _P("J2", "3")]),
        ])
    return arch, bom


def test_multi_unit_opamp_draws_every_unit(tmp_path) -> None:
    """The emitter must instantiate ALL units of a multi-unit symbol; unit-B
    pins otherwise never reach the netlist ('pin missing from netlist:
    U1.5/6/7', self-eval run_28 both 2026-07 batches) and KiCad ERC reports
    unplaced units + dangling labels."""
    arch, bom = _dual_opamp()
    emit_schematic(tmp_path, "BUFDEMO", arch, bom, title="BUFDEMO")

    sch_text = (tmp_path / "BUFFER.kicad_sch").read_text()
    # Stock TL072 = unit A + unit B + the shared power unit C.
    for unit in (1, 2, 3):
        assert f"(unit {unit})" in sch_text, f"unit {unit} not instantiated"
    assert sch_text.count('(reference "U1")') == 3

    result = check_erc(tmp_path, "BUFDEMO")
    assert result.ok, f"BUFDEMO ERC: {result.message}\n" + "\n".join(result.offenders)

    faith = check_netlist_faithfulness(tmp_path, "BUFDEMO", bom)
    assert faith.ok, (
        f"BUFDEMO §9.13: {faith.message}\n" + "\n".join(faith.offenders)
    )


def test_intermediate_input_bus_is_erc_clean(tmp_path) -> None:
    """A pin-detected, signal-rendered inter-sheet power bus must be ERC-clean:
    route_sheet places the PWR_FLAG the emitter assigned, so the regulator's
    VI pin is no longer reported undriven (the KC-W93GXR failure)."""
    arch, bom = _usb_pd_buck()
    emit_schematic(tmp_path, "PDBUCK", arch, bom, title="PDBUCK")
    result = check_erc(tmp_path, "PDBUCK")
    assert result.ok, f"PDBUCK ERC: {result.message}\n" + "\n".join(result.offenders)
