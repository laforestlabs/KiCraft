"""Tests for kicraft.synthesis.placement (bbox grid layout).

Touches the real KiCad stock libraries via lookup_pins to derive pin
counts and extents. Skip the file if KiCad's symbols aren't installed at
the default path (matches the pattern in test_kicraft_symbol_library).
"""
from __future__ import annotations

import pytest

from kicraft.design.models import BOM, BomPart, Sheet
from kicraft.design.synthesis.placement import place_sheet
from kicraft.design.synthesis.symbol_library import DEFAULT_KICAD_SYMBOL_DIR
from kicraft.design.synthesis.symbol_pinout import lookup_pins

pytestmark = pytest.mark.skipif(
    not DEFAULT_KICAD_SYMBOL_DIR.is_dir(),
    reason="KiCad symbol library not installed at the default path",
)


def _ldo_sheet() -> Sheet:
    return Sheet(name="LDO 3V3", stem="LDO_3V3", function="3.3V LDO")


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


def _abs_pin_coords(placed, parts) -> list[tuple[str, float, float]]:
    """Absolute (ref, x, y) of every pin connection point, as the router
    computes them (origin + local, y flipped)."""
    symbol_by_ref = {p.ref: p.symbol for p in parts}
    out: list[tuple[str, float, float]] = []
    for pp in placed:
        for pin in lookup_pins(symbol_by_ref[pp.ref])["pins"]:
            out.append(
                (
                    pp.ref,
                    round(pp.x_mm + pin["position"]["x"], 3),
                    round(pp.y_mm - pin["position"]["y"], 3),
                )
            )
    return out


def test_multipin_part_is_anchor() -> None:
    parts = _ldo_parts()
    # No connections: roles can't be inferred, but ic_groups still clusters the
    # caps onto U1 (as generic "other" members).
    bom = BOM(parts=parts, ic_groups={"U1": ["C1", "C2"]})
    placed = place_sheet(_ldo_sheet(), parts, bom)
    by_ref = {p.ref: p for p in placed}
    assert by_ref["U1"].role == "anchor"  # 5-pin IC anchors the cluster
    assert by_ref["C1"].role != "anchor"
    assert by_ref["C2"].role != "anchor"


def test_no_two_parts_share_a_pin_coordinate() -> None:
    # The core placement contract under label-based routing: distinct parts
    # never put a pin on the same node (which would short their nets).
    parts = _ldo_parts()
    bom = BOM(parts=parts, ic_groups={"U1": ["C1", "C2"]})
    placed = place_sheet(_ldo_sheet(), parts, bom)
    coords = [(x, y) for _, x, y in _abs_pin_coords(placed, parts)]
    assert len(coords) == len(set(coords)), "two pins coincide"


def test_part_origins_distinct() -> None:
    parts = _ldo_parts()
    bom = BOM(parts=parts)  # no ic_groups
    placed = place_sheet(_ldo_sheet(), parts, bom)
    origins = [(p.x_mm, p.y_mm) for p in placed]
    assert len(origins) == len(set(origins))


def test_placement_is_deterministic() -> None:
    parts = _ldo_parts()
    bom = BOM(parts=parts, ic_groups={"U1": ["C1", "C2"]})
    assert place_sheet(_ldo_sheet(), parts, bom) == place_sheet(
        _ldo_sheet(), parts, bom
    )


def test_returns_same_order_as_input() -> None:
    parts = _ldo_parts()
    bom = BOM(parts=parts)
    placed = place_sheet(_ldo_sheet(), parts, bom)
    assert [p.ref for p in placed] == [p.ref for p in parts]


def test_empty_sheet_returns_empty() -> None:
    bom = BOM(parts=[])
    placed = place_sheet(_ldo_sheet(), [], bom)
    assert placed == []


# ---------- cluster placement (with connections, roles inferred) ----------

from kicraft.design.models import NetConnection, PinEndpoint  # noqa: E402
from kicraft.design.synthesis.sch_geometry import (  # noqa: E402
    pin_abs_position,
    pin_exit_direction,
)


def _ldo_clustered_bom() -> BOM:
    """LDO with a pull-up so roles (decoupling / pull-up) can be inferred."""
    parts = _ldo_parts() + [
        BomPart(ref="R1", value="100k", symbol="Device:R",
                footprint="Resistor_SMD:R_0402_1005Metric", sheet="LDO 3V3"),
    ]

    def P(r, p):
        return PinEndpoint(ref=r, pin=p)

    return BOM(
        parts=parts,
        ic_groups={"U1": ["C1", "C2", "R1"]},
        connections=[
            NetConnection(net_name="VBUS", sheet="LDO 3V3",
                          endpoints=[P("U1", "1"), P("C1", "1"), P("R1", "1")]),
            NetConnection(net_name="+3V3", sheet="LDO 3V3",
                          endpoints=[P("U1", "5"), P("C2", "1")]),
            NetConnection(net_name="GND", sheet="LDO 3V3",
                          endpoints=[P("U1", "2"), P("C1", "2"), P("C2", "2")]),
            NetConnection(net_name="EN_PU", sheet="LDO 3V3",
                          endpoints=[P("U1", "3"), P("R1", "2")]),
        ],
    )


def _pin_geo(placed, parts, ref):
    pp = {p.ref: p for p in placed}[ref]
    sym = {p.ref: p.symbol for p in parts}[ref]
    out = {}
    for pin in lookup_pins(sym)["pins"]:
        out[pin["number"]] = (
            pin_abs_position(pp.x_mm, pp.y_mm, pp.rotation_deg, pin),
            pin_exit_direction(pp.rotation_deg, pin),
        )
    return pp, out


def test_decoupling_cap_sits_above_ic_rail_up() -> None:
    bom = _ldo_clustered_bom()
    placed = place_sheet(_ldo_sheet(), bom.parts, bom)
    u1 = {p.ref: p for p in placed}["U1"]
    c1, c1pins = _pin_geo(placed, bom.parts, "C1")
    assert c1.role == "decoupling"
    # Cap is above the IC (smaller y = higher on the sheet).
    assert c1.y_mm < u1.y_mm
    # Its VBUS (rail) pin 1 exits up; its GND pin 2 exits down.
    assert c1pins["1"][1] == "up"
    assert c1pins["2"][1] == "down"


def test_pullup_far_pin_points_away_from_ic() -> None:
    bom = _ldo_clustered_bom()
    placed = place_sheet(_ldo_sheet(), bom.parts, bom)
    _, u1pins = _pin_geo(placed, bom.parts, "U1")
    r1, r1pins = _pin_geo(placed, bom.parts, "R1")
    assert r1.role == "pullup"
    en_xy = u1pins["3"][0]              # EN pin (served)
    near_xy = r1pins["2"][0]           # R1 signal pin (taps EN net)
    far_xy, far_exit = r1pins["1"]      # R1 rail pin (VBUS)
    # The near pin is closer to EN than the far pin: the part is rotated so its
    # far pin points away from the IC, never back onto the served pin.
    import math
    d_near = math.dist(en_xy, near_xy)
    d_far = math.dist(en_xy, far_xy)
    assert d_far > d_near


def _pin_bbox(placed_part, parts):
    sym = {p.ref: p.symbol for p in parts}[placed_part.ref]
    xs, ys = [], []
    for pin in lookup_pins(sym)["pins"]:
        x, y = pin_abs_position(
            placed_part.x_mm, placed_part.y_mm, placed_part.rotation_deg, pin)
        xs.append(x)
        ys.append(y)
    if not xs:
        return (placed_part.x_mm, placed_part.y_mm, placed_part.x_mm, placed_part.y_mm)
    return (min(xs), min(ys), max(xs), max(ys))


def test_cluster_parts_do_not_overlap() -> None:
    bom = _ldo_clustered_bom()
    placed = place_sheet(_ldo_sheet(), bom.parts, bom)
    boxes = [(p.ref, _pin_bbox(p, bom.parts)) for p in placed]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (ra, (ax0, ay0, ax1, ay1)) = boxes[i]
            (rb, (bx0, by0, bx1, by1)) = boxes[j]
            # Pin bounding boxes must not interpenetrate (a 0.5 mm slack only).
            overlap = (ax0 < bx1 - 0.5 and bx0 < ax1 - 0.5
                       and ay0 < by1 - 0.5 and by0 < ay1 - 0.5)
            assert not overlap, f"{ra} overlaps {rb}"
