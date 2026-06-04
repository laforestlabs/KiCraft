"""search_symbols: keyword discovery of stock KiCad symbol ids.

Lets the BOM stage find the correct 'Library:Name' instead of guessing it (the
dominant failure mode the Flash-vs-Pro A/B exposed). Needs the KiCad stock symbol
libraries, like the other symbol tests in this suite.
"""
from __future__ import annotations

import re

from kicraft.design.synthesis.symbol_library import search_symbols


def test_finds_stock_passive():
    # a specific keyword surfaces the symbol (broad terms can exceed the limit)
    assert "Device:R" in search_symbols("device:r", limit=100)


def test_finds_real_connector_variants_not_the_guessed_name():
    # The model hallucinated 'Connector_Generic:Conn_02x08' (no such symbol); the
    # real ones carry a suffix (Conn_02x08_Odd_Even, ...). search_symbols surfaces
    # those so the model can pick a real id by keyword.
    res = search_symbols("conn 02x08")
    assert res, "expected at least one 2x8 connector match"
    assert all("02x08" in s.lower() for s in res)
    assert any("Conn_02x08" in s for s in res)


def test_results_are_top_level_ids_not_unit_subsymbols():
    res = search_symbols("device")
    assert res
    assert not any(re.search(r"_\d+_\d+$", s) for s in res)  # no 'R_0_1' sub-symbols
    assert all(":" in s for s in res)                        # all are 'Library:Name'


def test_respects_the_limit():
    assert len(search_symbols("device", limit=3)) <= 3


def test_empty_for_no_match():
    assert search_symbols("zzqq_nope_xyzzy_nothing") == []


def test_blank_query_returns_empty():
    assert search_symbols("   ") == []
