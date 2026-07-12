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
    InterSheetNet,
    NetConnection,
    PinEndpoint,
    Sheet,
    SheetPin,
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


def test_ring_member_series_resistors_stay_with_their_leds():
    # star-ornament (run_33, self-eval 20260710T211406Z): five LED sheets, each
    # holding a ring member D plus its series current-limit R; the MCU drives
    # Rx.1 over a CTRL net the wiring stage already declared inter-sheet. The R
    # is a member companion (2-pin, shares the LED's signal net) -- relocating
    # it to a SUPPORT sheet split CTRL/ANODE across sheets and left the CTRL
    # declaration stale (30 ERC errors). Nothing must move.
    sheets = [Sheet(name="POWER AND MCU", stem="POWER_AND_MCU", function="mcu")]
    parts = [BomPart(ref="U1", value="ATtiny402", sheet="POWER AND MCU",
                     symbol="MCU_Microchip_ATtiny:ATtiny402-SSN",
                     footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")]
    conns: list[NetConnection] = []
    isn: list[InterSheetNet] = []
    for i in (1, 2, 3):
        sh = f"LED {i}"
        sheets.append(Sheet(name=sh, stem=f"LED_{i}", function="led"))
        parts += [
            BomPart(ref=f"D{i}", value="LED", sheet=sh, symbol="Device:LED",
                    footprint="LED_SMD:LED_0603_1608Metric"),
            BomPart(ref=f"R{i}", value="220R", sheet=sh, symbol="Device:R",
                    footprint="Resistor_SMD:R_0402_1005Metric"),
        ]
        conns += [
            NetConnection(net_name=f"LED_{i}_CTRL", sheet="POWER AND MCU",
                          endpoints=[PinEndpoint(ref="U1", pin=str(i))]),
            NetConnection(net_name=f"LED_{i}_CTRL", sheet=sh,
                          endpoints=[PinEndpoint(ref=f"R{i}", pin="1")]),
            NetConnection(net_name=f"LED_{i}_ANODE", sheet=sh, endpoints=[
                PinEndpoint(ref=f"R{i}", pin="2"),
                PinEndpoint(ref=f"D{i}", pin="1")]),
            NetConnection(net_name="GND", sheet=sh,
                          endpoints=[PinEndpoint(ref=f"D{i}", pin="2")]),
        ]
        isn.append(InterSheetNet(name=f"LED_{i}_CTRL", endpoints=[
            SheetPin(sheet="POWER AND MCU", direction="output"),
            SheetPin(sheet=sh, direction="input")]))
    bom = BOM(parts=parts, connections=conns,
              arrays=[ArraySpec(refs=["D1", "D2", "D3"], pattern="ring")])
    arch = Architecture(sheets=sheets, power_nets=["GND"], inter_sheet_nets=isn)
    before = [n.model_dump() for n in arch.inter_sheet_nets]

    assert isolate_array_sheets(bom, arch) == []
    assert {p.sheet for p in bom.parts if p.ref.startswith("R")} == \
        {"LED 1", "LED 2", "LED 3"}, "series resistors stay beside their LEDs"
    assert len(arch.sheets) == 4, "no SUPPORT sheet appears"
    assert [n.model_dump() for n in arch.inter_sheet_nets] == before
    assert check_inter_sheet_nets_realized(arch, bom).ok, "§9.14 must hold"
    assert check_no_dangling_signal_nets(arch, bom).ok, "§9.15 must hold"


def _bom_arch_with_declared_nets():
    # The KC-WXN3SN fixture plus an MCU sheet and two nets the wiring stage
    # already declared inter-sheet through the array sheet: CTRL (MCU -> J1
    # only) and DATA (MCU -> J1.3 + D1.1, i.e. still touching a member).
    bom, arch = _bom_array_with_header(), _arch()
    arch.sheets.append(Sheet(name="MCU", stem="MCU", function="mcu"))
    bom.parts.append(BomPart(ref="U1", value="MCU", sheet="MCU",
                             symbol="Device:R",
                             footprint="Resistor_SMD:R_0402_1005Metric"))
    bom.connections += [
        NetConnection(net_name="CTRL", sheet="MCU",
                      endpoints=[PinEndpoint(ref="U1", pin="1")]),
        NetConnection(net_name="CTRL", sheet=ARRAY_SHEET,
                      endpoints=[PinEndpoint(ref="J1", pin="4")]),
        NetConnection(net_name="DATA", sheet="MCU",
                      endpoints=[PinEndpoint(ref="U1", pin="2")]),
    ]
    arch.inter_sheet_nets += [
        InterSheetNet(name="CTRL", endpoints=[
            SheetPin(sheet="MCU", direction="output"),
            SheetPin(sheet=ARRAY_SHEET, direction="input")]),
        InterSheetNet(name="DATA", endpoints=[
            SheetPin(sheet="MCU", direction="output"),
            SheetPin(sheet=ARRAY_SHEET, direction="input")]),
    ]
    return bom, arch


def test_stale_declarations_reconciled_when_part_moves():
    # Moving J1 re-homes the array-side endpoints of nets that were ALREADY
    # declared inter-sheet at wiring time. The declarations must follow the
    # part, else the emitter draws a sheet pin on the array sheet with nothing
    # inside it (hier_label_mismatch) and a dangling label on the new sheet.
    bom, arch = _bom_arch_with_declared_nets()
    assert isolate_array_sheets(bom, arch) == ["J1"]
    isn = {n.name: {e.sheet: e.direction for e in n.endpoints}
           for n in arch.inter_sheet_nets}
    # CTRL's array-side endpoint was ONLY J1 -> 1-for-1 sheet swap, the moved
    # endpoint inherits the removed sheet's direction.
    assert isn["CTRL"] == {"MCU": "output", "HEADER": "input"}
    # DATA still touches the array (D1.1) AND now the header -> grows to three
    # sheets; kept sheets keep their directions, the new one joins bidirectional.
    assert isn["DATA"] == {"MCU": "output", ARRAY_SHEET: "input",
                           "HEADER": "bidirectional"}
    assert check_inter_sheet_nets_realized(arch, bom).ok, "§9.14 must hold"
    assert check_no_dangling_signal_nets(arch, bom).ok, "§9.15 must hold"


def test_no_arrays_is_a_noop():
    bom = BOM(parts=[BomPart(ref="U1", value="x", sheet="MCU",
                             symbol="Device:R", footprint="Resistor_SMD:R_0402_1005Metric")])
    arch = Architecture(sheets=[Sheet(name="MCU", stem="MCU", function="mcu")],
                        power_nets=[], inter_sheet_nets=[])
    assert isolate_array_sheets(bom, arch) == []
