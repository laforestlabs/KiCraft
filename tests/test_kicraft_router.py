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


def test_power_stub_retreats_off_a_foreign_pin() -> None:
    # run_05 regression: U1's GND pin stub stepped one full grid right, landing
    # exactly on R4's pin 2 (a VBUS pin one grid step away) -- in KiCad a pin
    # touching a wire end connects, so the two GLOBAL rails merged and ERC
    # reported the two rails' PWR_FLAGs as conflicting power outputs. The stub
    # must retreat to half a grid step instead of stamping onto the pin.
    from kicraft.design.synthesis.router import (
        GRID_MM,
        RoutedSheet,
        _Endpoint,
        _route_power,
    )

    routed = RoutedSheet()
    e = _Endpoint(x=153.67, y=95.25, exit="right", ref="U1", pin="3")
    all_pins = [(153.67, 95.25, "U1", "3"), (153.67 + GRID_MM, 95.25, "R4", "2")]
    _route_power(routed, "GND", [e], frozenset(), all_pins, [])
    assert len(routed.power_symbols) == 1
    sym = routed.power_symbols[0]
    assert sym.x_mm == pytest.approx(153.67 + GRID_MM / 2)
    assert routed.wires[0].x2_mm == pytest.approx(153.67 + GRID_MM / 2)


def test_power_stub_boxed_in_falls_back_to_global_label() -> None:
    # A foreign pin at the half-grid point too: no safe stub exists in the exit
    # direction, so the pin gets a global label at its own position -- the net
    # stays named, and nothing is stamped onto foreign copper.
    from kicraft.design.synthesis.router import (
        GRID_MM,
        RoutedSheet,
        _Endpoint,
        _route_power,
    )

    routed = RoutedSheet()
    e = _Endpoint(x=10.0, y=10.0, exit="right", ref="U1", pin="3")
    all_pins = [
        (10.0, 10.0, "U1", "3"),
        (10.0 + GRID_MM / 2, 10.0, "R4", "2"),
        (10.0 + GRID_MM, 10.0, "R4", "1"),
    ]
    _route_power(routed, "GND", [e], frozenset(), all_pins, [])
    assert routed.wires == []
    assert routed.power_symbols == []
    assert len(routed.global_labels) == 1
    lbl = routed.global_labels[0]
    assert lbl.text == "GND" and (lbl.x_mm, lbl.y_mm) == (10.0, 10.0)


def test_opposing_power_stubs_of_two_rails_never_meet() -> None:
    # Two pins of DIFFERENT rails facing each other, two grid steps apart:
    # full-length stubs would meet head-on at the midpoint and short the rails.
    # The second stub must retreat to half a grid step.
    from kicraft.design.synthesis.router import (
        GRID_MM,
        RoutedSheet,
        _Endpoint,
        _route_power,
    )

    from kicraft.design.synthesis.router import _pt_on_axis_seg

    routed = RoutedSheet()
    gnd = _Endpoint(x=10.0, y=10.0, exit="right", ref="U1", pin="1")
    vbus = _Endpoint(x=10.0 + 2 * GRID_MM, y=10.0, exit="left", ref="J1", pin="1")
    all_pins = [(10.0, 10.0, "U1", "1"), (10.0 + 2 * GRID_MM, 10.0, "J1", "1")]
    power_stubs: list = []
    _route_power(routed, "GND", [gnd], frozenset(), all_pins, power_stubs)
    _route_power(routed, "VBUS", [vbus], frozenset(), all_pins, power_stubs)
    # Horizontal exits now elbow so the symbols stand upright; segment
    # counts are an implementation detail. The safety property is what
    # matters: no segment of one rail touches any segment of the other.
    gnd_segs = [s for (net, *s) in power_stubs if net == "GND"]
    vbus_segs = [s for (net, *s) in power_stubs if net == "VBUS"]
    assert gnd_segs and vbus_segs
    for (ax1, ay1, ax2, ay2) in gnd_segs:
        for (bx1, by1, bx2, by2) in vbus_segs:
            for (px, py) in ((ax1, ay1), (ax2, ay2)):
                assert not _pt_on_axis_seg(px, py, bx1, by1, bx2, by2)
            for (px, py) in ((bx1, by1), (bx2, by2)):
                assert not _pt_on_axis_seg(px, py, ax1, ay1, ax2, ay2)


def test_label_decollision_never_hops_to_foreign_stub() -> None:
    # run_09's exact failure geometry: two parallel one-grid pin stubs one grid
    # apart (ATTINY85 pins 5/6), each ending in its own net label. A body rect
    # makes ISP_MISO's label collide at BOTH reading directions and at every
    # point of its own stub, while the foreign stub's end is clear. The slide
    # used to accept any wire -> the label hopped onto ISP_MOSI's stub: its own
    # stub went dangling (pin_not_connected) and MISO merged with MOSI (two
    # labels, one wire -- silent). The slide must stay on the label's own
    # connected component; when nowhere on it is clear, leave the label alone.
    from kicraft.design.synthesis.router import (
        NetLabel,
        RoutedSheet,
        WireSegment,
        _pt_on_axis_seg,
        _resolve_label_collisions,
    )

    miso_stub = WireSegment(218.44, 156.21, 220.98, 156.21)
    mosi_stub = WireSegment(218.44, 158.75, 220.98, 158.75)
    routed = RoutedSheet(
        wires=[miso_stub, mosi_stub],
        labels=[
            NetLabel(text="ISP_MISO", x_mm=220.98, y_mm=156.21, angle_deg=0),
            NetLabel(text="ISP_MOSI", x_mm=220.98, y_mm=158.75, angle_deg=0),
        ],
    )
    # Collides with any MISO-label rect at y=156.21 (both directions, every
    # slide position along its own stub, and the outward extension) but clears
    # rects anchored at y=158.75 (the foreign stub).
    body_rects = [(205.0, 155.2, 245.0, 157.2)]
    _resolve_label_collisions(routed, body_rects, all_pins=[])

    miso = next(l for l in routed.labels if l.text == "ISP_MISO")
    on_own = _pt_on_axis_seg(
        miso.x_mm, miso.y_mm,
        miso_stub.x1_mm, miso_stub.y1_mm, miso_stub.x2_mm, miso_stub.y2_mm,
    )
    on_foreign = _pt_on_axis_seg(
        miso.x_mm, miso.y_mm,
        mosi_stub.x1_mm, mosi_stub.y1_mm, mosi_stub.x2_mm, mosi_stub.y2_mm,
    )
    assert on_own and not on_foreign, (
        f"ISP_MISO label left its net: anchor=({miso.x_mm},{miso.y_mm}) "
        f"on_own={on_own} on_foreign={on_foreign}"
    )
    # And the two labels never share an anchor (the silent-merge signature).
    anchors = [(l.x_mm, l.y_mm) for l in routed.labels]
    assert len(set(anchors)) == len(anchors)


def test_label_decollision_slides_along_own_wire() -> None:
    # The legitimate use of the slide is preserved: a label colliding at its
    # anchor but with clear space farther along ITS OWN wire moves there.
    from kicraft.design.synthesis.router import (
        GRID_MM,
        NetLabel,
        RoutedSheet,
        WireSegment,
        _pt_on_axis_seg,
        _resolve_label_collisions,
    )

    run = WireSegment(10.0, 10.0, 10.0 + 4 * GRID_MM, 10.0)
    routed = RoutedSheet(
        wires=[run],
        labels=[NetLabel(text="EN", x_mm=10.0, y_mm=10.0, angle_deg=0)],
    )
    # A small body sitting right of the anchor: collides at the anchor in both
    # reading directions, clear two grids along the wire.
    body_rects = [(9.0, 9.0, 12.0, 11.0)]
    _resolve_label_collisions(routed, body_rects, all_pins=[])

    lab = routed.labels[0]
    assert (lab.x_mm, lab.y_mm) != (10.0, 10.0), "label did not move"
    assert _pt_on_axis_seg(
        lab.x_mm, lab.y_mm, run.x1_mm, run.y1_mm, run.x2_mm, run.y2_mm
    ), "label slid off its own wire"
