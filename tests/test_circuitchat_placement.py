"""Tests for circuitchat.synthesis.placement (bbox grid layout).

Touches the real KiCad stock libraries via lookup_pins to derive pin
counts and extents. Skip the file if KiCad's symbols aren't installed at
the default path (matches the pattern in test_circuitchat_symbol_library).
"""
from __future__ import annotations

import pytest

from kicraft.circuitchat.models import BOM, BomPart, Sheet
from kicraft.circuitchat.synthesis.placement import place_sheet
from kicraft.circuitchat.synthesis.symbol_library import DEFAULT_KICAD_SYMBOL_DIR
from kicraft.circuitchat.synthesis.symbol_pinout import lookup_pins

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


def test_anchor_is_highest_pin_part() -> None:
    parts = _ldo_parts()
    bom = BOM(parts=parts, ic_groups={"U1": ["C1", "C2"]})
    placed = place_sheet(_ldo_sheet(), parts, bom)
    by_ref = {p.ref: p for p in placed}
    # U1 (5 pins) anchors; the 2-pin caps are ordinary grid cells.
    assert by_ref["U1"].role == "anchor"
    assert by_ref["C1"].role == "grid"
    assert by_ref["C2"].role == "grid"


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
