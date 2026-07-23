"""Guard test for the dense-SoC sheet partition (dense-soc plan P3).

The contract that matters is ELECTRICAL, not cosmetic: a detachable subfunction
(debug header, button, battery) may move to its own sheet; a decoupling cap or a
crystal load cap may NEVER leave the IC whose pins it must hug -- moving it is
exactly the split the v1 plan proposed and the rediagnosis rejected.

Synthetic BOMs shaped like the nRF52840 beacon that motivated the plan; no LLM.
"""
from __future__ import annotations

from kicraft.design.models import (
    BOM,
    Architecture,
    BomPart,
    NetConnection,
    PinEndpoint,
    Sheet,
)
from kicraft.design.synthesis.sheet_partition import (
    split_dense_sheets_enabled,
    split_dense_soc_sheets,
)

SHEET = "MCU"


def _part(ref: str, sheet: str = SHEET) -> BomPart:
    return BomPart(
        ref=ref, value="x", symbol="Device:R",
        footprint="Resistor_SMD:R_0603_1608Metric", sheet=sheet,
    )


def _conn(net: str, *refs: str) -> NetConnection:
    return NetConnection(
        net_name=net, sheet=SHEET,
        endpoints=[PinEndpoint(ref=r, pin="1") for r in refs],
    )


def _soc_bom() -> tuple[BOM, Architecture]:
    """One sheet: nRF52840-shaped. U1 + 10 decaps + 2 crystals with load caps +
    an SWD header + a button + a battery holder = 20 routable parts."""
    refs = ["U1"] + [f"C{i}" for i in range(1, 11)] + [
        "X1", "C11", "C12", "X2", "C13", "C14", "J1", "SW1", "BT1", "R1",
    ]
    parts = [_part(r) for r in refs]
    conns = [
        # U1 has many pins -> it is the hub.
        NetConnection(
            net_name="VDD", sheet=SHEET,
            endpoints=[PinEndpoint(ref="U1", pin=str(i)) for i in range(1, 9)]
            + [PinEndpoint(ref=f"C{i}", pin="1") for i in range(1, 11)]
            + [PinEndpoint(ref="J1", pin="1"), PinEndpoint(ref="BT1", pin="1")],
        ),
        NetConnection(
            net_name="GND", sheet=SHEET,
            endpoints=[PinEndpoint(ref="U1", pin="9")]
            + [PinEndpoint(ref=f"C{i}", pin="2") for i in range(1, 15)]
            + [PinEndpoint(ref=r, pin="2") for r in ("X1", "X2", "J1", "SW1", "BT1")],
        ),
        # crystals + their load caps
        _conn("X1_OSC1", "U1", "X1", "C11"),
        _conn("X1_OSC2", "U1", "X1", "C12"),
        _conn("X2_OSC1", "U1", "X2", "C13"),
        _conn("X2_OSC2", "U1", "X2", "C14"),
        # detachable subfunctions
        _conn("SWDIO", "U1", "J1"),
        _conn("SWDCLK", "U1", "J1"),
        _conn("RESET", "U1", "J1", "R1"),
        _conn("BUTTON", "U1", "SW1"),
    ]
    arch = Architecture(sheets=[Sheet(name=SHEET, stem="MCU", function="mcu")],
                        power_nets=["VDD", "GND"], inter_sheet_nets=[])
    return BOM(parts=parts, connections=conns), arch


def test_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("KICRAFT_SPLIT_DENSE_SHEETS", raising=False)
    assert split_dense_sheets_enabled() is False
    monkeypatch.setenv("KICRAFT_SPLIT_DENSE_SHEETS", "1")
    assert split_dense_sheets_enabled() is True


def test_decaps_and_crystals_never_leave_their_ic() -> None:
    bom, arch = _soc_bom()
    moved = split_dense_soc_sheets(bom, arch)
    sheet_by_ref = {p.ref: p.sheet for p in bom.parts}

    assert moved, "a 20-part SoC sheet must shed something"
    # The electrical invariant: every decap and every crystal (with its load
    # caps) still shares U1's sheet.
    for ref in ["U1"] + [f"C{i}" for i in range(1, 15)] + ["X1", "X2"]:
        assert sheet_by_ref[ref] == SHEET, f"{ref} must stay with its IC"
    # ... and what moved is a real subfunction (the debug header / the button).
    assert {"J1", "SW1"} & set(moved)


def test_sheet_lands_at_or_below_the_threshold() -> None:
    bom, arch = _soc_bom()
    split_dense_soc_sheets(bom, arch, max_routable=15)
    counts: dict[str, int] = {}
    for p in bom.parts:
        counts[p.sheet] = counts.get(p.sheet, 0) + 1
    # The SoC sheet cannot always reach the threshold (its decaps are pinned),
    # but it must have shed every detachable group before giving up.
    assert counts[SHEET] < 20
    assert len(arch.sheets) >= 2


def test_cross_sheet_signals_are_declared() -> None:
    bom, arch = _soc_bom()
    moved = split_dense_soc_sheets(bom, arch)
    declared = {n.name for n in arch.inter_sheet_nets}
    sheet_by_ref = {p.ref: p.sheet for p in bom.parts}
    for conn in bom.connections:
        sheets = {sheet_by_ref[ep.ref] for ep in conn.endpoints}
        if len(sheets) > 1:
            assert conn.net_name in declared or conn.net_name in ("VDD", "GND")
    # every connection is confined to one sheet after the re-split
    for conn in bom.connections:
        assert {sheet_by_ref[ep.ref] for ep in conn.endpoints} == {conn.sheet}
    assert any("detachable subfunction" in a for a in bom.assumptions)
    assert moved == sorted(moved)


def test_small_sheet_untouched() -> None:
    bom, arch = _soc_bom()
    for p in bom.parts:  # pretend the sheet is small
        pass
    assert split_dense_soc_sheets(bom, arch, max_routable=99) == []
    assert len(arch.sheets) == 1


def test_no_hub_no_split() -> None:
    # A sheet of 20 loose passives has no SoC to detach FROM.
    parts = [_part(f"R{i}") for i in range(1, 21)]
    conns = [_conn(f"N{i}", f"R{i}", f"R{i + 1}") for i in range(1, 20)]
    arch = Architecture(sheets=[Sheet(name=SHEET, stem="MCU", function="mcu")],
                        power_nets=["VDD", "GND"], inter_sheet_nets=[])
    bom = BOM(parts=parts, connections=conns)
    assert split_dense_soc_sheets(bom, arch) == []
