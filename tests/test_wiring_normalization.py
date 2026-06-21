"""Tests for the two deterministic wiring-commit normalizers that fix the
KC-WFFXZ3 terminal wiring failure:

  * ``bridge_duplicate_pins`` — put internally-shorted duplicate pads (KiCad
    ``N'``) on their terminal's net, so §9.11 net coverage stops flagging a pad
    the package already ties together (the EVQP7A01P 1/1', 2/2' tactile switch).
  * ``reconcile_inter_sheet_nets`` — derive the signal inter-sheet contract from
    the crossings wiring actually realized, so the wiring stage is never handed
    a cross-sheet contract it cannot edit (DTR/RTS declared into the ESP32 sheet
    with no consumer there; EN/IO0 wired across sheets but never declared).
"""
from __future__ import annotations

from kicraft.design.models import (
    BOM,
    Architecture,
    BomPart,
    InterSheetNet,
    NetConnection,
    PinEndpoint,
    Sheet,
    SheetPin,
)
from kicraft.design.synthesis.validation import (
    _pin_base,
    bridge_duplicate_pins,
    check_inter_sheet_nets_realized,
    check_net_coverage,
    check_no_dangling_signal_nets,
    reconcile_inter_sheet_nets,
)

SWITCH_SYM = "evq-p7a01p:EVQP7A01P"  # real 4-pad symbol: pins 1, 2, 1', 2'
SWITCH_FP = "evq-p7a01p:SW-SMD_EVQP7A01P"


def _r(ref: str, sheet: str) -> BomPart:
    return BomPart(
        ref=ref, value="x", symbol="Device:R",
        footprint="Resistor_SMD:R_0402_1005Metric", sheet=sheet,
    )


def _sw(ref: str, sheet: str) -> BomPart:
    return BomPart(ref=ref, value="btn", symbol=SWITCH_SYM,
                   footprint=SWITCH_FP, sheet=sheet)


# ---------- bridge_duplicate_pins (§9.11 prime-pin trap) ----------


def test_pin_base_strips_trailing_primes() -> None:
    assert _pin_base("1") == "1"
    assert _pin_base("1'") == "1"
    assert _pin_base("2''") == "2"
    assert _pin_base("A1") == "A1"


def test_bridge_covers_unwired_duplicate_pads() -> None:
    """SW1 wired on pin 1 (RESET) and pin 2 (GND); the 1'/2' pads were left
    uncovered (the exact §9.11 trip on KC-WFFXZ3). Bridge puts each prime on its
    sibling's net, and §9.11 then passes."""
    bom = BOM(
        parts=[_sw("SW1", "MCU"), _r("R1", "MCU")],
        connections=[
            NetConnection(net_name="RESET", sheet="MCU",
                          endpoints=[PinEndpoint(ref="SW1", pin="1"),
                                     PinEndpoint(ref="R1", pin="1")]),
            NetConnection(net_name="GND", sheet="MCU",
                          endpoints=[PinEndpoint(ref="SW1", pin="2"),
                                     PinEndpoint(ref="R1", pin="2")]),
        ],
    )
    assert not check_net_coverage(bom).ok  # 1' and 2' uncovered before

    bridged = bridge_duplicate_pins(bom)
    assert set(bridged) == {"SW1.1' -> RESET", "SW1.2' -> GND"}

    reset = next(c for c in bom.connections if c.net_name == "RESET")
    assert PinEndpoint(ref="SW1", pin="1'") in reset.endpoints
    assert check_net_coverage(bom).ok  # every pad now accounted for


def test_bridge_is_noop_when_pads_already_covered() -> None:
    bom = BOM(
        parts=[_sw("SW1", "MCU")],
        connections=[
            NetConnection(net_name="RESET", sheet="MCU",
                          endpoints=[PinEndpoint(ref="SW1", pin="1"),
                                     PinEndpoint(ref="SW1", pin="1'")]),
            NetConnection(net_name="GND", sheet="MCU",
                          endpoints=[PinEndpoint(ref="SW1", pin="2"),
                                     PinEndpoint(ref="SW1", pin="2'")]),
        ],
    )
    assert bridge_duplicate_pins(bom) == []


def test_bridge_leaves_fully_unwired_terminal_for_coverage() -> None:
    """A terminal with no pad wired is not ours to invent a net for — §9.11
    must still flag it."""
    bom = BOM(
        parts=[_sw("SW1", "MCU")],
        connections=[
            NetConnection(net_name="RESET", sheet="MCU",
                          endpoints=[PinEndpoint(ref="SW1", pin="1"),
                                     PinEndpoint(ref="SW1", pin="1'")]),
        ],
    )
    assert bridge_duplicate_pins(bom) == []  # terminal 2 untouched
    cov = check_net_coverage(bom)
    assert not cov.ok and any("SW1.2" in o for o in cov.offenders)


def test_bridge_respects_explicit_no_connect_on_a_duplicate_pad() -> None:
    """If the model deliberately marked the second pad no_connect, leave it —
    don't create a pin that is both NC'd and netted."""
    bom = BOM(
        parts=[_sw("SW1", "MCU")],
        connections=[
            NetConnection(net_name="RESET", sheet="MCU",
                          endpoints=[PinEndpoint(ref="SW1", pin="1")]),
            NetConnection(net_name="GND", sheet="MCU",
                          endpoints=[PinEndpoint(ref="SW1", pin="2"),
                                     PinEndpoint(ref="SW1", pin="2'")]),
        ],
        no_connect_pins=[PinEndpoint(ref="SW1", pin="1'")],
    )
    assert bridge_duplicate_pins(bom) == []  # 1' stays NC, not wired to RESET
    reset = next(c for c in bom.connections if c.net_name == "RESET")
    assert [e.pin for e in reset.endpoints] == ["1"]


def test_bridge_skips_terminal_whose_pads_are_on_different_nets() -> None:
    """Two pads of one internally-shorted terminal on different nets is a real
    short — bridge must not silently merge them."""
    bom = BOM(
        parts=[_sw("SW1", "MCU")],
        connections=[
            NetConnection(net_name="A", sheet="MCU",
                          endpoints=[PinEndpoint(ref="SW1", pin="1")]),
            NetConnection(net_name="B", sheet="MCU",
                          endpoints=[PinEndpoint(ref="SW1", pin="1'")]),
            NetConnection(net_name="GND", sheet="MCU",
                          endpoints=[PinEndpoint(ref="SW1", pin="2"),
                                     PinEndpoint(ref="SW1", pin="2'")]),
        ],
    )
    assert bridge_duplicate_pins(bom) == []
    a = next(c for c in bom.connections if c.net_name == "A")
    assert [e.pin for e in a.endpoints] == ["1"]  # not merged with B


# ---------- reconcile_inter_sheet_nets (§9.14/§9.15 unwinnable contract) ----------


def _two_sheets(inter: list[InterSheetNet], conns: list[NetConnection]):
    arch = Architecture(
        sheets=[Sheet(name="USB", stem="USB", function="bridge"),
                Sheet(name="MCU", stem="MCU", function="mcu")],
        power_nets=["GND"],
        inter_sheet_nets=inter,
    )
    bom = BOM(parts=[_r("U1", "USB"), _r("U2", "MCU")], connections=conns)
    return arch, bom


def _gnd() -> InterSheetNet:
    return InterSheetNet(name="GND", endpoints=[
        SheetPin(sheet="USB", direction="bidirectional"),
        SheetPin(sheet="MCU", direction="bidirectional")])


def test_reconcile_adds_realized_undeclared_crossing() -> None:
    """EN/IO0 case: wiring wires a signal net across two sheets that architecture
    never declared. §9.15 flags it dangling on both sides; reconcile promotes it
    to inter-sheet, after which §9.14 and §9.15 both pass."""
    arch, bom = _two_sheets(
        inter=[_gnd()],
        conns=[
            NetConnection(net_name="EN", sheet="USB",
                          endpoints=[PinEndpoint(ref="U1", pin="1")]),
            NetConnection(net_name="EN", sheet="MCU",
                          endpoints=[PinEndpoint(ref="U2", pin="1")]),
        ],
    )
    assert not check_no_dangling_signal_nets(arch, bom).ok  # EN dangles each side

    changes = reconcile_inter_sheet_nets(arch, bom)
    assert any(c.startswith("+EN") for c in changes)
    assert "EN" in {n.name for n in arch.inter_sheet_nets}
    assert check_no_dangling_signal_nets(arch, bom).ok
    assert check_inter_sheet_nets_realized(arch, bom).ok


def test_reconcile_drops_declared_but_locally_consumed_crossing() -> None:
    """DTR/RTS case: architecture declared DTR crossing to MCU, but wiring only
    ever wires it on USB (its consumers — the auto-reset transistors — live
    there). §9.14 can never pass as declared; reconcile drops the phantom
    crossing and §9.14 passes."""
    arch, bom = _two_sheets(
        inter=[_gnd(), InterSheetNet(name="DTR", endpoints=[
            SheetPin(sheet="USB", direction="output"),
            SheetPin(sheet="MCU", direction="input")])],
        conns=[
            NetConnection(net_name="DTR", sheet="USB",
                          endpoints=[PinEndpoint(ref="U1", pin="1"),
                                     PinEndpoint(ref="U1", pin="2")]),
        ],
    )
    assert not check_inter_sheet_nets_realized(arch, bom).ok  # MCU side has no label

    changes = reconcile_inter_sheet_nets(arch, bom)
    assert any(c.startswith("-DTR") for c in changes)
    assert "DTR" not in {n.name for n in arch.inter_sheet_nets}
    assert check_inter_sheet_nets_realized(arch, bom).ok
    assert check_no_dangling_signal_nets(arch, bom).ok  # DTR now a valid 2-pin local net


def test_reconcile_is_noop_on_correct_design() -> None:
    arch, bom = _two_sheets(
        inter=[_gnd(), InterSheetNet(name="SIG", endpoints=[
            SheetPin(sheet="USB", direction="output"),
            SheetPin(sheet="MCU", direction="input")])],
        conns=[
            NetConnection(net_name="SIG", sheet="USB",
                          endpoints=[PinEndpoint(ref="U1", pin="1")]),
            NetConnection(net_name="SIG", sheet="MCU",
                          endpoints=[PinEndpoint(ref="U2", pin="1")]),
        ],
    )
    before = [n.model_dump() for n in arch.inter_sheet_nets]
    assert reconcile_inter_sheet_nets(arch, bom) == []
    assert [n.model_dump() for n in arch.inter_sheet_nets] == before


def test_reconcile_preserves_power_nets_verbatim() -> None:
    """GND joins globally via power symbols, not per-pin connections, so it is
    preserved even though no connection realizes it on two sheets."""
    arch, bom = _two_sheets(inter=[_gnd()], conns=[
        NetConnection(net_name="GND", sheet="USB",
                      endpoints=[PinEndpoint(ref="U1", pin="2")]),
    ])
    changes = reconcile_inter_sheet_nets(arch, bom)
    assert all("GND" not in c for c in changes)
    assert "GND" in {n.name for n in arch.inter_sheet_nets}


def test_reconcile_does_not_merge_inconsistently_named_dangles() -> None:
    """The SOIL_MOISTURE_BLE bug: D+/D- split into differently-named single-sheet
    nets. Each name is on one sheet, so reconcile promotes nothing and §9.15
    still catches them."""
    arch, bom = _two_sheets(inter=[_gnd()], conns=[
        NetConnection(net_name="USB_DP_POWER", sheet="USB",
                      endpoints=[PinEndpoint(ref="U1", pin="1")]),
        NetConnection(net_name="USB_DP_ESP32", sheet="MCU",
                      endpoints=[PinEndpoint(ref="U2", pin="1")]),
    ])
    changes = reconcile_inter_sheet_nets(arch, bom)
    assert not any(c.startswith("+") for c in changes)
    assert not check_no_dangling_signal_nets(arch, bom).ok


def test_reconcile_kc_wffxzu_autoreset_end_to_end() -> None:
    """The KC-WFFXZ3 deadlock in one design: DTR/RTS declared crossing but wired
    only on USB, while EN/IO0 cross to the MCU but were never declared. Before
    reconcile §9.14 and §9.15 both fail and the wiring stage cannot fix either
    (it cannot edit inter_sheet_nets). After reconcile both pass."""
    arch, bom = _two_sheets(
        inter=[_gnd(),
               InterSheetNet(name="DTR", endpoints=[
                   SheetPin(sheet="USB", direction="output"),
                   SheetPin(sheet="MCU", direction="input")]),
               InterSheetNet(name="RTS", endpoints=[
                   SheetPin(sheet="USB", direction="output"),
                   SheetPin(sheet="MCU", direction="input")])],
        conns=[
            # DTR/RTS consumed locally on USB by the auto-reset transistors.
            NetConnection(net_name="DTR", sheet="USB",
                          endpoints=[PinEndpoint(ref="U1", pin="1"),
                                     PinEndpoint(ref="U2", pin="2")]),
            NetConnection(net_name="RTS", sheet="USB",
                          endpoints=[PinEndpoint(ref="U1", pin="2"),
                                     PinEndpoint(ref="U2", pin="1")]),
            # EN/IO0: transistor collector (USB) -> MCU reset/boot pin.
            NetConnection(net_name="EN", sheet="USB",
                          endpoints=[PinEndpoint(ref="U1", pin="1")]),
            NetConnection(net_name="EN", sheet="MCU",
                          endpoints=[PinEndpoint(ref="U2", pin="1")]),
            NetConnection(net_name="IO0", sheet="USB",
                          endpoints=[PinEndpoint(ref="U1", pin="2")]),
            NetConnection(net_name="IO0", sheet="MCU",
                          endpoints=[PinEndpoint(ref="U2", pin="2")]),
        ],
    )
    assert not check_inter_sheet_nets_realized(arch, bom).ok
    assert not check_no_dangling_signal_nets(arch, bom).ok

    reconcile_inter_sheet_nets(arch, bom)

    names = {n.name for n in arch.inter_sheet_nets}
    assert {"EN", "IO0", "GND"} <= names and "DTR" not in names and "RTS" not in names
    assert check_inter_sheet_nets_realized(arch, bom).ok
    assert check_no_dangling_signal_nets(arch, bom).ok
