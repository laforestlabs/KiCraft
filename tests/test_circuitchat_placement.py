"""Tests for circuitchat.synthesis.placement.

Touches the real KiCad stock libraries via lookup_pins to derive
pin counts. Skip the file if KiCad's symbols aren't installed at the
default path (matches the pattern in test_circuitchat_symbol_library).
"""
from __future__ import annotations

import pytest

from kicraft.circuitchat.models import BOM, BomPart, Sheet
from kicraft.circuitchat.synthesis.placement import (
    ANCHOR_X_MM,
    ANCHOR_Y_MM,
    PERIPHERAL_X_MM,
    place_sheet,
)
from kicraft.circuitchat.synthesis.symbol_library import DEFAULT_KICAD_SYMBOL_DIR

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


def test_anchor_is_highest_pin_part() -> None:
    parts = _ldo_parts()
    bom = BOM(parts=parts, ic_groups={"U1": ["C1", "C2"]})
    placed = place_sheet(_ldo_sheet(), parts, bom)
    by_ref = {p.ref: p for p in placed}
    assert by_ref["U1"].role == "anchor"
    assert by_ref["U1"].x_mm == ANCHOR_X_MM
    assert by_ref["U1"].y_mm == ANCHOR_Y_MM


def test_ic_group_members_placed_in_ring_1() -> None:
    parts = _ldo_parts()
    bom = BOM(parts=parts, ic_groups={"U1": ["C1", "C2"]})
    placed = place_sheet(_ldo_sheet(), parts, bom)
    by_ref = {p.ref: p for p in placed}
    assert by_ref["C1"].role == "ring1"
    assert by_ref["C2"].role == "ring1"
    # Distance from anchor is ≤ 12 mm (first-ring radius bound).
    for ref in ("C1", "C2"):
        dx = abs(by_ref[ref].x_mm - ANCHOR_X_MM)
        dy = abs(by_ref[ref].y_mm - ANCHOR_Y_MM)
        assert dx <= 12.0 and dy <= 12.0


def test_peripherals_go_to_right_column() -> None:
    parts = _ldo_parts()
    bom = BOM(parts=parts)  # no ic_groups → C1 and C2 are peripherals
    placed = place_sheet(_ldo_sheet(), parts, bom)
    by_ref = {p.ref: p for p in placed}
    assert by_ref["C1"].role == "peripheral"
    assert by_ref["C1"].x_mm == PERIPHERAL_X_MM
    assert by_ref["C2"].x_mm == PERIPHERAL_X_MM
    # Distinct y positions.
    assert by_ref["C1"].y_mm != by_ref["C2"].y_mm


def test_placement_is_deterministic() -> None:
    parts = _ldo_parts()
    bom = BOM(parts=parts, ic_groups={"U1": ["C1", "C2"]})
    p1 = place_sheet(_ldo_sheet(), parts, bom)
    p2 = place_sheet(_ldo_sheet(), parts, bom)
    assert p1 == p2


def test_empty_sheet_returns_empty() -> None:
    bom = BOM(parts=[])
    placed = place_sheet(_ldo_sheet(), [], bom)
    assert placed == []
