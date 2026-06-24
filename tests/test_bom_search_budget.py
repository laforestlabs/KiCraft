"""BOM-stage cost controls: the per-MPN search budget and the MPN->LCSC cache.

These directly target the dominant BOM cost identified in the part-query log:
lookup_lcsc_id is 56% of all BOM tool calls, and a single part was looked up
47x across a window because resolved parts were never cached and a weak model
re-spells the same MPN for round after round.
"""
from __future__ import annotations

import json

import types

from kicraft.parts_library import mpn_cache
from kicraft.server import stage_driver
from kicraft.server.stage_driver import _bom_executor, _normalize_mpn


def test_key_for_folds_whitespace_case_and_lcsc_url(monkeypatch):
    monkeypatch.setenv(mpn_cache.ENV_PATH, "/tmp/_kicraft_mpn_key_test.json")
    assert mpn_cache.key_for("  bmp280  ") == "BMP280"
    assert mpn_cache.key_for("BMP280") == "BMP280"            # case-folded
    # a pasted product URL collapses to the bare LCSC C-number
    assert mpn_cache.key_for(
        "https://www.lcsc.com/product-detail/C7386355.html?s_z=n") == "C7386355"


def test_mpncache_put_get_roundtrip_persists_across_loads(monkeypatch, tmp_path):
    monkeypatch.setenv(mpn_cache.ENV_PATH, str(tmp_path / "cache.json"))
    assert mpn_cache.get("BMP280") is None            # empty -> None, no crash
    mpn_cache.put("BMP280", "C83291", "parts-library")
    got = mpn_cache.get("bmp280")                     # case-insensitive lookup
    assert got == {"lcsc": "C83291", "source": "parts-library",
                   "ts": got["ts"]}
    # the file on disk is plain JSON a human can audit
    data = json.loads((tmp_path / "cache.json").read_text())
    assert data["BMP280"]["lcsc"] == "C83291"


def test_mpncache_survives_corrupt_file(monkeypatch, tmp_path):
    monkeypatch.setenv(mpn_cache.ENV_PATH, str(tmp_path / "cache.json"))
    (tmp_path / "cache.json").write_text("{not json")
    assert mpn_cache.get("anything") is None          # corrupt -> empty, no raise


def test_cacheable_only_freezes_precise_identifiers():
    # precise: a bare C-number, or a whitespace-free token that carries a digit
    assert mpn_cache.cacheable("C190004")
    assert mpn_cache.cacheable("BMP280")
    assert mpn_cache.cacheable("VL53L1CXV0FY/1")
    assert mpn_cache.cacheable("SK-12D07VG4")
    assert mpn_cache.cacheable("https://lcsc.com/p/C7386355.html")
    # fuzzy: a descriptive phrase (whitespace) or a bare word (no digit) is NOT
    # frozen -- a heuristic 'best match' must stay free to re-resolve.
    assert not mpn_cache.cacheable("BME280 Bosch")
    assert not mpn_cache.cacheable("SPDT slide switch SMD")
    assert not mpn_cache.cacheable("diode")
    assert not mpn_cache.cacheable("")


def test_mpncache_put_noops_for_fuzzy_keyword(monkeypatch, tmp_path):
    """A keyword search result must never be cached: freezing one heuristic
    match per machine would make a wrong first hit permanent."""
    monkeypatch.setenv(mpn_cache.ENV_PATH, str(tmp_path / "cache.json"))
    mpn_cache.put("SPDT slide switch SMD", "C431540", "easyeda")
    assert mpn_cache.get("SPDT slide switch SMD") is None
    # a precise MPN resolved by the same network tier still caches
    mpn_cache.put("CH224K", "C970725", "easyeda")
    assert mpn_cache.get("CH224K") == {"lcsc": "C970725", "source": "easyeda",
                                       "ts": mpn_cache.get("CH224K")["ts"]}


def _executor(monkeypatch, tmp_path):
    # keep the cache + query log off the real user machine while spawning the CLI
    monkeypatch.setenv(mpn_cache.ENV_PATH, str(tmp_path / "cache.json"))
    monkeypatch.setenv("KICRAFT_QUERY_LOG", str(tmp_path / "q.jsonl"))
    return _bom_executor(tmp_path)


def test_bom_executor_caps_repeated_lookup_per_mpn(monkeypatch, tmp_path):
    """A bare LCSC id resolves offline, so this exercises only the cap, not the
    network. The same MPN may be queried at most _BOM_MPN_QUERY_CAP times; after
    that the executor returns a terminal 'stop retrying' result instead of
    spawning the lookup again."""
    ex = _executor(monkeypatch, tmp_path)
    first = ex("lookup_lcsc_id", {"mpn": "C190004"})
    assert json.loads(first)["ok"] is True and json.loads(first)["lcsc"] == "C190004"
    second = ex("lookup_lcsc_id", {"mpn": "c190004"})   # case-folded = same part
    assert json.loads(second)["ok"] is True
    third = ex("lookup_lcsc_id", {"mpn": "C190004"})
    assert "already been attempted" in third and "STOP retrying" in third
    fourth = ex("lookup_lcsc_id", {"mpn": "C190004"})
    assert "already been attempted" in fourth


def test_bom_executor_budget_is_per_spelling(monkeypatch, tmp_path):
    """Distinct normalized MPNs each get their own budget (we fold case and
    pasted LCSC URLs, but do NOT merge a part's different genuine spellings)."""
    ex = _executor(monkeypatch, tmp_path)
    first = ex("lookup_lcsc_id", {"mpn": "C190004"})
    assert json.loads(first)["ok"] is True
    ex("lookup_lcsc_id", {"mpn": "C190004"})            # second C190004
    capped = ex("lookup_lcsc_id", {"mpn": "C190004"})  # third -> capped
    assert "already been attempted" in capped
    # a DIFFERENT part still resolves normally on its first call
    diff = ex("lookup_lcsc_id", {"mpn": "C83291"})
    assert json.loads(diff)["ok"] is True


def test_normalize_mpn_helper():
    assert _normalize_mpn("  bmp280 ") == "BMP280"
    assert _normalize_mpn("https://lcsc.com/p/C7386355.html") == "C7386355"
    assert _normalize_mpn("") == ""


def test_read_only_lookups_memoized_within_stage(monkeypatch, tmp_path):
    """An identical symbol/footprint lookup or search is answered once per stage
    and reused, so a re-issuing model does not re-spawn the CLI subprocess. The
    library-mutating tools (list_parts) are NOT memoized."""
    calls: list[list[str]] = []

    def fake_run(cmd, cwd=None):
        calls.append(cmd)
        return types.SimpleNamespace(stdout=f"out:{cmd[-1]}", stderr="", returncode=0)

    monkeypatch.setattr(stage_driver, "_run", fake_run)
    ex = _bom_executor(tmp_path)

    a = ex("lookup_symbol", {"symbol": "Device:R"})
    b = ex("lookup_symbol", {"symbol": "Device:R"})      # exact repeat -> memo
    assert a == b
    assert sum("lookup-symbol" in c for c in calls) == 1   # only one subprocess

    ex("lookup_symbol", {"symbol": "Device:C"})          # different arg -> runs
    assert sum("lookup-symbol" in c for c in calls) == 2

    ex("search_footprints", {"query": "0603"})
    ex("search_footprints", {"query": "0603"})           # memoized
    assert sum("search-footprints" in c for c in calls) == 1

    # list_parts mutates conceptually (a fetch can add bundles) -> never memoized
    ex("list_parts", {})
    ex("list_parts", {})
    assert sum("list-parts" in c for c in calls) == 2