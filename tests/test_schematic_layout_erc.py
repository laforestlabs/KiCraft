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
    BOM, Architecture, BomPart, NetConnection, PinEndpoint, Sheet)
from kicraft.design.synthesis.emitter import emit_schematic
from kicraft.design.synthesis.symbol_library import DEFAULT_KICAD_SYMBOL_DIR
from kicraft.design.synthesis.validation import check_erc

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


@pytest.mark.parametrize("builder,stem", [(_ldo, "LDODEMO"), (_usb_uart, "UARTDEMO")])
def test_clustered_sheet_is_erc_clean(tmp_path, builder, stem) -> None:
    arch, bom = builder()
    emit_schematic(tmp_path, stem, arch, bom, title=stem)
    result = check_erc(tmp_path, stem)
    assert result.ok, f"{stem} ERC: {result.message}\n" + "\n".join(result.offenders)
