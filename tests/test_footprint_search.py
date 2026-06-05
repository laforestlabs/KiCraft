"""search_footprints / lookup_footprint: keyword discovery + verification of stock
KiCad footprint ids.

The footprint analog of test_symbol_search. The BOM stage's weak server model guessed
plausible-but-nonexistent stock footprints (e.g. Connector_BarrelJack:BarrelJack_2.1mm_
P5.5mm, which has no .kicad_mod) and, with no footprint-discovery tool, could not find
the real names. These tests pin the discovery + verification behavior. Needs the KiCad
stock footprint libraries, like the symbol tests need the stock symbol libraries.
"""
from __future__ import annotations

import pytest

from kicraft.design.synthesis.footprint_library import (
    FootprintNotFoundError,
    lookup_footprint,
    search_footprints,
)


def test_finds_known_pinheader():
    res = search_footprints("pinheader 2x08")
    assert res, "expected 2x08 pin-header footprints"
    assert all("2x08" in s.lower() for s in res)
    assert any("PinHeader_2x08" in s for s in res)


def test_finds_the_exact_p254_vertical_header():
    # The 2x8 2.54mm header the cat-feeder needed; the real id carries the pitch + orientation.
    res = search_footprints("pinheader 2x08 p2.54mm vertical", limit=100)
    assert "Connector_PinHeader_2.54mm:PinHeader_2x08_P2.54mm_Vertical" in res


def test_footprint_stopword_is_dropped():
    # The model habitually appends "footprint"; it must not zero an otherwise-good query.
    assert search_footprints("pinheader 2x08 footprint") == search_footprints("pinheader 2x08")


def test_finds_barreljack_for_the_offender():
    # J1 died on a guessed BarrelJack_2.1mm_P5.5mm; a broad keyword surfaces real ids.
    res = search_footprints("barreljack")
    assert res, "expected barrel-jack footprints"
    assert all("barreljack" in s.lower() for s in res)
    assert any(s.startswith("Connector_BarrelJack:") for s in res)


def test_results_are_library_name_ids():
    res = search_footprints("connector")
    assert res
    assert all(":" in s for s in res)


def test_respects_the_limit():
    assert len(search_footprints("connector", limit=3)) <= 3


def test_empty_for_no_match():
    assert search_footprints("zzqq_nope_xyzzy_nothing") == []


def test_blank_query_returns_empty():
    assert search_footprints("   ") == []


def test_lookup_resolves_pad_count():
    info = lookup_footprint("Resistor_SMD:R_0603_1608Metric")
    assert info["footprint"] == "Resistor_SMD:R_0603_1608Metric"
    assert info["pad_count"] == 2


def test_lookup_rejects_non_library_name_form():
    with pytest.raises(ValueError):
        lookup_footprint("not_a_colon_form")


def test_lookup_missing_footprint_raises():
    # The exact hallucinated id from the failed cat-feeder run.
    with pytest.raises(FootprintNotFoundError):
        lookup_footprint("Connector_BarrelJack:BarrelJack_2.1mm_P5.5mm")
