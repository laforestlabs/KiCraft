"""isolate_array_sheets: an array sheet must hold only array members + companions.

KC-WXN3SN lumped a 4x8 LED array AND its 3-pin power/data header onto one "LED
ARRAY" sheet. The leaf solver locks the LED grid compactly but strands the header
~60mm away (board 76% empty). The fix moves every non-member, non-companion part
off the array sheet onto its own sheet -> its own leaf, re-splitting connections
per sheet and declaring the now cross-sheet SIGNAL nets inter-sheet (power joins
via global power symbols). These tests pin that, including that the result passes
the §9.14/§9.15 schematic-coverage gates.
"""
from __future__ import annotations

from kicraft.design.models import (
    BOM,
    Architecture,
    ArraySpec,
    BomPart,
    NetConnection,
    PinEndpoint,
    Sheet,
)
from kicraft.design.synthesis.array_decaps import isolate_array_sheets
from kicraft.design.synthesis.validation import (
    check_inter_sheet_nets_realized,
    check_no_dangling_signal_nets,
)

ARRAY_SHEET = "LED ARRAY"


def _led(ref: str) -> BomPart:
    return BomPart(ref=ref, value="WS2812", sheet=ARRAY_SHEET,
                   symbol="Device:LED", footprint="LED_SMD:LED_1515")


def _bom_array_with_header() -> BOM:
    leds = [_led(f"D{i}") for i in range(1, 5)]  # 2x2 array
    cap = BomPart(ref="C1", value="100nF", sheet=ARRAY_SHEET,
                  symbol="Device:C", footprint="Capacitor_SMD:C_0402_1005Metric")
    j1 = BomPart(ref="J1", value="Conn_01x03", sheet=ARRAY_SHEET,
                 symbol="Connector_Generic:Conn_01x03",
                 footprint="Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical")
    conns = [
        NetConnection(net_name="5V", sheet=ARRAY_SHEET, endpoints=[
            PinEndpoint(ref="J1", pin="1"), PinEndpoint(ref="C1", pin="1"),
            *[PinEndpoint(ref=f"D{i}", pin="2") for i in range(1, 5)]]),
        NetConnection(net_name="GND", sheet=ARRAY_SHEET, endpoints=[
            PinEndpoint(ref="J1", pin="2"), PinEndpoint(ref="C1", pin="2"),
            *[PinEndpoint(ref=f"D{i}", pin="4") for i in range(1, 5)]]),
        NetConnection(net_name="DATA", sheet=ARRAY_SHEET, endpoints=[
            PinEndpoint(ref="J1", pin="3"), PinEndpoint(ref="D1", pin="1")]),
        # Internal daisy-chain tie (sheet-local, both pins on array members).
        NetConnection(net_name="DCHAIN_1", sheet=ARRAY_SHEET, endpoints=[
            PinEndpoint(ref="D1", pin="3"), PinEndpoint(ref="D2", pin="1")]),
    ]
    return BOM(
        parts=[*leds, cap, j1],
        connections=conns,
        arrays=[ArraySpec(refs=[f"D{i}" for i in range(1, 5)], rows=2, cols=2)],
    )


def _arch() -> Architecture:
    return Architecture(
        sheets=[Sheet(name=ARRAY_SHEET, stem="LED_ARRAY", function="leds")],
        power_nets=["5V", "GND"], inter_sheet_nets=[],
    )


def _conns_by_net_sheet(bom):
    return {(c.net_name, c.sheet): {f"{e.ref}.{e.pin}" for e in c.endpoints}
            for c in bom.connections}


def test_header_moved_to_own_sheet_array_stays_pure():
    bom, arch = _bom_array_with_header(), _arch()
    moved = isolate_array_sheets(bom, arch)
    assert moved == ["J1"]

    by_ref = {p.ref: p.sheet for p in bom.parts}
    assert by_ref["J1"] != ARRAY_SHEET, "header left the array sheet"
    new_sheet = by_ref["J1"]
    assert new_sheet == "HEADER", "connector-only displacement names the sheet HEADER"
    # Members + companion decap stay put.
    for ref in ("D1", "D2", "D3", "D4", "C1"):
        assert by_ref[ref] == ARRAY_SHEET
    # The new sheet is registered in the architecture.
    assert {s.name for s in arch.sheets} == {ARRAY_SHEET, "HEADER"}


def test_connections_resplit_and_cross_sheet_signal_declared():
    bom, arch = _bom_array_with_header(), _arch()
    isolate_array_sheets(bom, arch)
    cbs = _conns_by_net_sheet(bom)
    # 5V/GND/DATA each split into a HEADER side and an array side.
    assert cbs[("5V", "HEADER")] == {"J1.1"}
    assert cbs[("DATA", "HEADER")] == {"J1.3"}
    assert cbs[("DATA", ARRAY_SHEET)] == {"D1.1"}
    assert "C1.1" in cbs[("5V", ARRAY_SHEET)]
    # Internal daisy-chain tie untouched (both pins stayed on the array sheet).
    assert cbs[("DCHAIN_1", ARRAY_SHEET)] == {"D1.3", "D2.1"}

    isn = {n.name: n for n in arch.inter_sheet_nets}
    assert "DATA" in isn, "cross-sheet signal net must be declared inter-sheet"
    assert {e.sheet for e in isn["DATA"].endpoints} == {"HEADER", ARRAY_SHEET}
    assert all(e.direction == "bidirectional" for e in isn["DATA"].endpoints)
    # Power/ground join via global power symbols, NOT inter-sheet declarations.
    assert "5V" not in isn and "GND" not in isn


def test_result_passes_schematic_coverage_gates():
    bom, arch = _bom_array_with_header(), _arch()
    isolate_array_sheets(bom, arch)
    assert check_inter_sheet_nets_realized(arch, bom).ok, "§9.14 must hold"
    assert check_no_dangling_signal_nets(arch, bom).ok, "§9.15 must hold"


def test_moved_part_perimeter_zone_is_dropped():
    # An internal header lumped on the array sheet often carries a spurious
    # edge/corner zone; moving it must drop that zone so the composer places its
    # leaf next to the array, not pinned flush to a far board edge (where its
    # power trace then hugs the edge -> copper_edge_clearance).
    bom, arch = _bom_array_with_header(), _arch()
    bom.component_zones["J1"] = {"edge": "bottom"}
    isolate_array_sheets(bom, arch)
    assert "J1" not in bom.component_zones, "perimeter edge zone must be dropped"


def test_pure_array_sheet_is_a_noop():
    # Same array + companion but NO stray part -> nothing to move.
    bom = BOM(
        parts=[_led("D1"), _led("D2"), _led("D3"), _led("D4"),
               BomPart(ref="C1", value="100nF", sheet=ARRAY_SHEET,
                       symbol="Device:C", footprint="Capacitor_SMD:C_0402_1005Metric")],
        connections=[
            NetConnection(net_name="5V", sheet=ARRAY_SHEET, endpoints=[
                PinEndpoint(ref="C1", pin="1"),
                *[PinEndpoint(ref=f"D{i}", pin="2") for i in range(1, 5)]]),
            # C1 must wire BOTH pins to power/ground to read as an array companion.
            NetConnection(net_name="GND", sheet=ARRAY_SHEET, endpoints=[
                PinEndpoint(ref="C1", pin="2"),
                *[PinEndpoint(ref=f"D{i}", pin="4") for i in range(1, 5)]]),
        ],
        arrays=[ArraySpec(refs=[f"D{i}" for i in range(1, 5)], rows=2, cols=2)],
    )
    arch = _arch()
    assert isolate_array_sheets(bom, arch) == []
    assert {s.name for s in arch.sheets} == {ARRAY_SHEET}
    assert arch.inter_sheet_nets == []


def test_no_arrays_is_a_noop():
    bom = BOM(parts=[BomPart(ref="U1", value="x", sheet="MCU",
                             symbol="Device:R", footprint="Resistor_SMD:R_0402_1005Metric")])
    arch = Architecture(sheets=[Sheet(name="MCU", stem="MCU", function="mcu")],
                        power_nets=[], inter_sheet_nets=[])
    assert isolate_array_sheets(bom, arch) == []
