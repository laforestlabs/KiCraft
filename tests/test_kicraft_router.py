"""Tests for kicraft.synthesis.router.

Verifies the label-based router's structural properties:
- signal nets emit one short stub + one label per pin (no trunks/junctions)
- inter-sheet signal nets use hierarchical labels; power nets use power
  symbols (never a hierarchical label)
- power nets produce one PowerSymbol per endpoint on a straight stub
- no_connect_pins land at pin positions

Touches real KiCad stock libraries via lookup_pins. Skip when symbols
aren't installed.
"""
from __future__ import annotations

import pytest

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
from kicraft.design.synthesis.placement import place_sheet
from kicraft.design.synthesis.router import (
    power_symbol_for,
    route_sheet,
)
from kicraft.design.synthesis.symbol_library import DEFAULT_KICAD_SYMBOL_DIR

pytestmark = pytest.mark.skipif(
    not DEFAULT_KICAD_SYMBOL_DIR.is_dir(),
    reason="KiCad symbol library not installed at the default path",
)


def _ldo_arch() -> Architecture:
    return Architecture(
        sheets=[Sheet(name="LDO 3V3", stem="LDO_3V3", function="ldo")],
        power_nets=["VBUS", "+3V3", "GND"],
        inter_sheet_nets=[
            InterSheetNet(
                name="VBUS",
                endpoints=[
                    SheetPin(sheet="LDO 3V3", direction="input"),
                    SheetPin(sheet="LDO 3V3", direction="output"),
                ],
            )
        ],
    )


def _ldo_parts() -> list[BomPart]:
    return [
        BomPart(
            ref="U1", value="AP2112K-3.3",
            symbol="Regulator_Linear:AP2112K-3.3",
            footprint="Package_TO_SOT_SMD:SOT-23-5",
            sheet="LDO 3V3",
        ),
        BomPart(
            ref="C1", value="1uF",
            symbol="Device:C",
            footprint="Capacitor_SMD:C_0402_1005Metric",
            sheet="LDO 3V3",
        ),
        BomPart(
            ref="C2", value="1uF",
            symbol="Device:C",
            footprint="Capacitor_SMD:C_0402_1005Metric",
            sheet="LDO 3V3",
        ),
    ]


def _do_route(connections: list[NetConnection], no_connect=()):
    parts = _ldo_parts()
    bom = BOM(
        parts=parts,
        ic_groups={"U1": ["C1", "C2"]},
        connections=connections,
        no_connect_pins=list(no_connect),
    )
    arch = _ldo_arch()
    placed = place_sheet(arch.sheets[0], parts, bom)
    return route_sheet("LDO_3V3", "LDO 3V3", placed, bom, arch)


def test_power_symbol_for_known_rails() -> None:
    assert power_symbol_for("GND") == "power:GND"
    assert power_symbol_for("+3V3") == "power:+3V3"
    assert power_symbol_for("VBUS") == "power:VBUS"
    assert power_symbol_for("/GND") == "power:GND"
    assert power_symbol_for("NOT_A_RAIL") is None


def test_ground_net_emits_power_symbols_no_trunk() -> None:
    routed = _do_route(
        connections=[
            NetConnection(
                net_name="GND",
                endpoints=[
                    PinEndpoint(ref="U1", pin="2"),
                    PinEndpoint(ref="C1", pin="2"),
                    PinEndpoint(ref="C2", pin="2"),
                ],
                sheet="LDO 3V3",
            )
        ]
    )
    # One power symbol per endpoint.
    assert len(routed.power_symbols) == 3
    assert {p.lib_id for p in routed.power_symbols} == {"power:GND"}
    # No long horizontal trunk wires (only short stubs to the symbol).
    # We don't assert exact wire count because right/left-exit pins emit
    # an L-stub (2 segments). But there must be no junctions.
    assert routed.junctions == []
    # No net labels for power.
    assert routed.labels == []


def test_two_pin_local_net_draws_wire_and_one_label() -> None:
    # A local 2-pin signal net is now a REAL wire between the pins, carrying a
    # single net label (not two floating stub labels). "EN_LOCAL" is a signal
    # name (not a power rail), so it routes as a wire, not power symbols.
    routed = _do_route(
        connections=[
            NetConnection(
                net_name="EN_LOCAL",
                endpoints=[
                    PinEndpoint(ref="U1", pin="3"),  # EN
                    PinEndpoint(ref="C2", pin="1"),
                ],
                sheet="LDO 3V3",
            )
        ]
    )
    assert routed.junctions == []
    # Exactly one net label names the wire (keeps kicad-cli ERC happy).
    assert [lab.text for lab in routed.labels] == ["EN_LOCAL"]
    # A real connecting wire exists (1-2 segments for a straight run or an L).
    assert 1 <= len(routed.wires) <= 2


def test_local_net_emits_one_label_per_pin_at_distinct_positions() -> None:
    routed = _do_route(
        connections=[
            NetConnection(
                net_name="FB_SENSE",
                endpoints=[
                    PinEndpoint(ref="U1", pin="1"),
                    PinEndpoint(ref="C1", pin="1"),
                    PinEndpoint(ref="C2", pin="1"),
                ],
                sheet="LDO 3V3",
            )
        ]
    )
    labels = [lab for lab in routed.labels if lab.text == "FB_SENSE"]
    assert len(labels) == 3
    # No two pins share a node, so the three labels sit at distinct points.
    coords = {(lab.x_mm, lab.y_mm) for lab in labels}
    assert len(coords) == 3
    assert routed.junctions == []


def test_power_inter_sheet_net_uses_power_symbol_not_hier_label() -> None:
    # VBUS is both a power net and an inter_sheet net. Power nets connect
    # globally via power symbols, so the router emits a power symbol and NOT
    # a hierarchical label (and the root emitter likewise omits a sheet pin
    # for power nets — see emitter RC3a).
    routed = _do_route(
        connections=[
            NetConnection(
                net_name="VBUS",
                endpoints=[PinEndpoint(ref="U1", pin="1")],
                sheet="LDO 3V3",
            )
        ]
    )
    assert routed.hier_labels == []
    assert [p.lib_id for p in routed.power_symbols] == ["power:VBUS"]


def test_no_connect_pins_emit_markers() -> None:
    routed = _do_route(
        connections=[
            NetConnection(
                net_name="GND",
                endpoints=[PinEndpoint(ref="U1", pin="2")],
                sheet="LDO 3V3",
            )
        ],
        no_connect=[PinEndpoint(ref="U1", pin="4")],  # NC pin on AP2112K
    )
    assert len(routed.no_connects) == 1
