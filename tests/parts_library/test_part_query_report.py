"""Tests for the part-query-report aggregation."""
from __future__ import annotations

from kicraft.cli.part_query_report import _is_bundle_lib, format_report, summarize

# A representative slice of the query log: two designs that both used the
# curated ams1117 bundle, fetched an uncatalogued part twice, resolved another
# via JLCPCB, failed to resolve a bogus MPN, and ran a fruitless stock search.
_EVENTS = [
    {"tool": "list_parts", "outcome": "listed", "n_active": 22},
    {"tool": "lookup_lcsc_id", "outcome": "hit", "query": "AMS1117-3.3",
     "lcsc": "C6186", "source": "parts-library", "library_name": "ams1117-3v3"},
    {"tool": "lookup_lcsc_id", "outcome": "hit", "query": "AMS1117-3.3",
     "lcsc": "C6186", "source": "parts-library", "library_name": "ams1117-3v3"},
    {"tool": "lookup_footprint", "outcome": "hit", "query": "ams1117-3v3:SOT-223",
     "lib": "ams1117-3v3"},
    {"tool": "lookup_footprint", "outcome": "hit", "query": "Device:R", "lib": "Device"},
    {"tool": "add_part_from_lcsc", "outcome": "fetched", "lcsc": "C2913201",
     "library_name": "esp32-s3-wroom-1b", "into": "home", "maturity": "prototype"},
    {"tool": "add_part_from_lcsc", "outcome": "fetched", "lcsc": "C2913201",
     "library_name": "esp32-s3-wroom-1b", "into": "home", "maturity": "prototype"},
    {"tool": "lookup_lcsc_id", "outcome": "resolved", "query": "XPT2046",
     "lcsc": "C13298", "source": "jlcpcb"},
    {"tool": "lookup_lcsc_id", "outcome": "miss", "query": "totally-bogus", "n_candidates": 0},
    {"tool": "search_footprints", "outcome": "miss", "query": "flux capacitor", "n_matches": 0},
]


def test_is_bundle_lib():
    assert _is_bundle_lib("ams1117-3v3")          # curated bundle slug
    assert _is_bundle_lib("esp32-wroom-32e-n4")
    assert not _is_bundle_lib("Device")           # stock KiCad lib (CamelCase)
    assert not _is_bundle_lib("Connector_PinHeader_2.54mm")
    assert not _is_bundle_lib(None)
    assert not _is_bundle_lib("")


def test_summarize_buckets():
    s = summarize(_EVENTS)
    assert s["n_events"] == len(_EVENTS)
    # Library popularity: 2 lcsc hits + 1 footprint-by-bundle ref = 3; the
    # stock "Device" footprint lookup is NOT counted as a bundle hit.
    assert s["lib_hits"]["ams1117-3v3"] == 3
    assert "Device" not in s["lib_hits"]
    # Misses that we cached (fetched) and the bundle slug they became.
    assert s["fetches"]["C2913201"] == 2
    assert s["fetch_name"]["C2913201"] == "esp32-s3-wroom-1b"
    # JLCPCB-resolved (miss, not yet bundled) and unresolved MPNs.
    assert s["jlcpcb"]["C13298"] == 1
    assert s["unresolved"]["totally-bogus"] == 1
    # Empty stock searches keyed by tool:query.
    assert s["search_miss"]["search_footprints:flux capacitor"] == 1


def test_format_report_smoke():
    text = format_report(summarize(_EVENTS), top=5)
    assert "LIBRARY HITS" in text
    assert "ams1117-3v3" in text
    assert "ADD-TO-LIBRARY" in text
    assert "C2913201" in text
    assert "UNRESOLVED" in text
