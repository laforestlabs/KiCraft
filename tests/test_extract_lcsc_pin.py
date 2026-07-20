"""extract_lcsc_pin: package-size codes vs real C-numbers (2026-07-19 §4.7).

The denylist that stops "package C0603"-style prose reading as an LCSC pin
collided with C1812 -- a real, in-stock LCSC part whose digits are also a chip
size code. An explicit "LCSC" immediately before the token now disambiguates
in favor of a pin; bare prose keeps the exclusion.
"""
from __future__ import annotations

from kicraft.design.synthesis.fab_export import extract_lcsc_pin


def test_plain_c_number_extracts():
    assert extract_lcsc_pin("use LCSC C2837270 for this") == "C2837270"


def test_package_prose_is_not_a_pin():
    assert extract_lcsc_pin("100nF X7R, package C1206") is None
    assert extract_lcsc_pin("0.1uF C1812 ceramic") is None


def test_zero_led_size_codes_are_not_pins():
    assert extract_lcsc_pin("package C0603") is None


def test_lcsc_prefixed_ambiguous_code_is_a_pin():
    assert extract_lcsc_pin("sourcing: LCSC C1812 (3.6pF C0G 0805)") == "C1812"
    assert extract_lcsc_pin("lcsc: C2512") == "C2512"


def test_first_real_pin_wins_over_earlier_package_token():
    assert (
        extract_lcsc_pin("package C1206, sourced as LCSC C482911")
        == "C482911"
    )
