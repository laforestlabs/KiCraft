"""Tests for the standard form-factor registry (PR1).

Locks the Arduino Uno R3 shield template geometry (golden) and verifies brief
detection wires a matched standard into ``FormFactor.standard``. No pcbnew,
no build -- pure data + string matching.
"""

from __future__ import annotations

import pytest

from kicraft.design.synthesis.form_factor import extract_form_factor
from kicraft.form_factors import (
    FixedConnector,
    all_templates,
    get_template,
    match_standard,
)


# ---------------------------------------------------------------------------
# Registry + matching
# ---------------------------------------------------------------------------


class TestMatchStandard:
    @pytest.mark.parametrize(
        "brief",
        [
            "An Arduino-Uno-format prototyping shield with stacking headers.",
            "arduino uno shield with an SMT regulator",
            "A plain Arduino shield for sensors",
            "uno shield, 3.3V rail",
            "Arduino Uno R3 shield",
        ],
    )
    def test_matches_arduino_shield(self, brief):
        t = match_standard(brief)
        assert t is not None
        assert t.key == "arduino_uno_shield"

    @pytest.mark.parametrize(
        "brief",
        [
            "A CAN bus node with an STM32 and a DB9 connector.",
            "A guardian shield circuit that protects the input",  # 'shield' but not arduino
            "round 50 mm coaster board",
            "",
            "an arduinolike board",  # no word boundary
        ],
    )
    def test_no_false_match(self, brief):
        assert match_standard(brief) is None

    def test_longest_alias_wins(self):
        # "arduino uno shield" should resolve the same key as the generic
        # "arduino shield"; both map to arduino_uno_shield here, but the
        # ordering guarantees the specific alias is tried first.
        assert match_standard("arduino uno shield").key == "arduino_uno_shield"

    def test_get_template_roundtrip(self):
        assert get_template("arduino_uno_shield") is not None
        assert get_template("nope") is None
        assert get_template(None) is None

    def test_all_templates_nonempty(self):
        keys = {t.key for t in all_templates()}
        assert "arduino_uno_shield" in keys


# ---------------------------------------------------------------------------
# Golden geometry -- Arduino Uno R3 shield
# ---------------------------------------------------------------------------


class TestArduinoShieldGolden:
    def setup_method(self):
        self.t = get_template("arduino_uno_shield")

    def test_board_dimensions(self):
        assert self.t.board_width_mm == pytest.approx(68.58)
        assert self.t.board_height_mm == pytest.approx(53.34)

    def test_four_headers_with_expected_pin_counts(self):
        by_role = {c.role: c for c in self.t.fixed_connectors}
        assert set(by_role) == {"digital_high", "digital_low", "power", "analog"}
        assert by_role["digital_high"].pins == 10
        assert by_role["digital_low"].pins == 8
        assert by_role["power"].pins == 8
        assert by_role["analog"].pins == 6
        assert sum(c.pins for c in self.t.fixed_connectors) == 32

    def test_famous_0p16_inch_digital_offset(self):
        by_role = {c.role: c for c in self.t.fixed_connectors}
        hi = by_role["digital_high"]  # SCL..D8 (D8 is the last pin)
        lo = by_role["digital_low"]   # D7..D0 (D7 is the first pin)
        d8_x = hi.pin_positions()[-1][1]
        d7_x = lo.pin_positions()[0][1]
        # The D7<->D8 gap is deliberately 0.16" (4.064 mm), not the 0.1" pitch.
        assert d7_x - d8_x == pytest.approx(4.064, abs=1e-3)

    def test_pin_semantics_cover_the_arduino_io_contract(self):
        nets = self.t.canonical_nets()
        for d in range(14):
            assert f"D{d}" in nets
        for a in range(6):
            assert f"A{a}" in nets
        assert {"+5V", "+3V3", "GND", "VIN", "IOREF", "RESET", "AREF", "SDA", "SCL"} <= nets

    def test_pin_positions_advance_on_pitch(self):
        hi = {c.role: c for c in self.t.fixed_connectors}["digital_high"]
        pos = hi.pin_positions()
        assert pos[0][0] == "SCL"
        # consecutive pins are one pitch apart on the X axis
        assert pos[1][1] - pos[0][1] == pytest.approx(2.54)
        assert all(p[2] == pytest.approx(hi.y_mm) for p in pos)  # same row (Y)

    def test_mounting_holes_present(self):
        assert len(self.t.mounting_holes) == 4

    def test_datum_validated(self):
        # Datum transcribed from the Alarm-Siren KiCad Arduino library, so the
        # placement path may lay a board out on it.
        assert self.t.validated is True

    def test_pin1_coordinates_match_authoritative_datum(self):
        by_role = {c.role: c for c in self.t.fixed_connectors}
        # Pin-1 centres, top-left frame (y = footprint_y + 53.34).
        assert (by_role["digital_high"].x_mm, by_role["digital_high"].y_mm) == pytest.approx((18.796, 2.54))
        assert (by_role["digital_low"].x_mm, by_role["digital_low"].y_mm) == pytest.approx((45.72, 2.54))
        assert (by_role["power"].x_mm, by_role["power"].y_mm) == pytest.approx((27.94, 50.8))
        assert (by_role["analog"].x_mm, by_role["analog"].y_mm) == pytest.approx((50.8, 50.8))
        # The authoritative D8 pad sits at x=41.656 (last pin of digital_high).
        assert by_role["digital_high"].pin_positions()[-1][1] == pytest.approx(41.656)

    def test_connector_pin_count_consistency_enforced(self):
        with pytest.raises(ValueError):
            FixedConnector(
                role="bad", pins=3, footprint="x", x_mm=0, y_mm=0, axis="x",
                net_by_pin=("A", "B"),  # 2 != 3
            )


# ---------------------------------------------------------------------------
# Intent-stage wiring
# ---------------------------------------------------------------------------


class TestExtractFormFactorStandard:
    def test_shield_brief_sets_standard(self):
        ff = extract_form_factor(
            "An Arduino-Uno-format prototyping shield with stacking through-hole "
            "headers and an onboard SMT 3.3 V regulator."
        )
        assert ff is not None
        assert ff.standard == "arduino_uno_shield"
        assert ff.shape == "rect"  # a shield is a rectangular board
        assert ff.size_mm == pytest.approx(68.58)

    def test_standard_takes_precedence_over_shape(self):
        # Even if a shape word appears, a named standard wins.
        ff = extract_form_factor("A round-cornered Arduino Uno shield")
        assert ff is not None and ff.standard == "arduino_uno_shield"

    def test_shape_only_brief_unaffected(self):
        ff = extract_form_factor("A circular 50 mm coaster PCB")
        assert ff is not None
        assert ff.standard is None
        assert ff.shape == "circle"

    def test_plain_brief_returns_none(self):
        assert extract_form_factor("A CAN bus node with an STM32 MCU") is None
