"""search_library_parts + the curated tier of the search-symbols /
search-footprints / list-parts CLI tools.

Regression source: KC-9EZE3S. The BOM stage searched "BNC connector" /
"trimpot 3296" seven ways and saw only stock KiCad ids — the vendored
``bnc-pcb-jack`` bundle (whose footprint, LCSC and 3D model agree) was
unreachable: the prompt's parts table is filtered to core_blocks and the
search tools were stock-only. The model paired a stock Amphenol *vertical*
footprint with the Kinghelm *elbow* BNC's C# and a "fab-ready" board shipped
unbuildable. These tests pin the curated tier: keyword search must surface
non-core bundles, and the CLI must present a bundle's symbol+footprint as a
pair.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from kicraft.design.cli_app import (
    _cmd_list_parts,
    _cmd_search_footprints,
    _cmd_search_symbols,
)
from kicraft.design.library import search_library_parts

REPO = Path(__file__).resolve().parents[1]


# ---------- search_library_parts ----------


def test_finds_bnc_bundle_for_the_kc9eze3s_query():
    # "connector" appears nowhere in the bnc-pcb-jack manifest; the partial
    # fallback must still surface it for the model's actual first query —
    # and rank it FIRST (name hit beats the description-only "connector"
    # hits on unrelated bundles).
    parts = search_library_parts("BNC connector", REPO)
    assert parts and parts[0].manifest.name == "bnc-pcb-jack"


def test_collapsed_matching_finds_the_trimpot():
    # "trimpot" only matches "trim-pot-…" with separators collapsed.
    parts = search_library_parts("trimpot 3296", REPO)
    assert any(p.manifest.name == "trim-pot-3296w-10k" for p in parts)


def test_all_term_matches_rank_before_partial():
    parts = search_library_parts("bnc jack", REPO)
    assert parts, "expected at least the bnc bundle"
    m = parts[0].manifest
    hay = " ".join([m.name, m.mpn or "", m.description or ""]).lower()
    assert "bnc" in hay and "jack" in hay


def test_kind_stopwords_are_dropped():
    with_stop = [p.manifest.name for p in search_library_parts("bnc footprint", REPO)]
    without = [p.manifest.name for p in search_library_parts("bnc", REPO)]
    assert with_stop == without
    assert "bnc-pcb-jack" in with_stop


def test_respects_the_limit():
    assert len(search_library_parts("connector", REPO, limit=3)) <= 3


def test_blank_and_stopword_only_queries_return_empty():
    assert search_library_parts("", REPO) == []
    assert search_library_parts("footprint symbol", REPO) == []


def test_no_match_returns_empty():
    assert search_library_parts("zzqq_nope_xyzzy_nothing", REPO) == []


# ---------- CLI commands (what the BOM-stage tools actually return) ----------


def _ns(**kw):
    return argparse.Namespace(**kw)


def test_search_symbols_cli_surfaces_curated_pair_first(capsys, monkeypatch):
    monkeypatch.chdir(REPO)
    assert _cmd_search_symbols(_ns(query="BNC connector", limit=10)) == 0
    out = capsys.readouterr().out
    assert "curated bundles" in out
    # The bundle's symbol and footprint ids ride the same line, as a pair.
    pair_line = next(line for line in out.splitlines() if "bnc-pcb-jack:" in line)
    assert "symbol bnc-pcb-jack:" in pair_line
    assert "footprint bnc-pcb-jack:" in pair_line


def test_search_footprints_cli_surfaces_curated_and_stock(capsys, monkeypatch):
    monkeypatch.chdir(REPO)
    assert _cmd_search_footprints(_ns(query="BNC", limit=10)) == 0
    out = capsys.readouterr().out
    assert "bnc-pcb-jack:" in out
    # The stock Amphenol id the KC-9EZE3S run picked is still discoverable…
    assert "Connector_Coaxial:BNC_Amphenol_031-5539_Vertical" in out
    # …but only after the curated section.
    assert out.index("bnc-pcb-jack:") < out.index("Connector_Coaxial:")


def test_search_cli_miss_message_names_both_tiers(capsys, monkeypatch):
    monkeypatch.chdir(REPO)
    assert _cmd_search_symbols(_ns(query="zzqq_nope_xyzzy", limit=10)) == 0
    err = capsys.readouterr().err
    assert "curated bundles" in err and "stock KiCad" in err


def test_list_parts_query_filters_the_table(capsys, monkeypatch):
    monkeypatch.chdir(REPO)
    assert _cmd_list_parts(_ns(query="bnc")) == 0
    out = capsys.readouterr().out
    assert "bnc-pcb-jack" in out
    # The filter must actually filter: a full table carries hundreds of rows.
    rows = [line for line in out.splitlines() if line.startswith("| `")]
    assert 0 < len(rows) <= 64


def test_list_parts_without_query_is_the_full_table(capsys, monkeypatch):
    monkeypatch.chdir(REPO)
    assert _cmd_list_parts(_ns(query=None)) == 0
    out = capsys.readouterr().out
    rows = [line for line in out.splitlines() if line.startswith("| `")]
    assert len(rows) > 100  # the whole library, not a slice


def test_list_parts_query_miss_says_so(capsys, monkeypatch):
    monkeypatch.chdir(REPO)
    assert _cmd_list_parts(_ns(query="zzqq_nope_xyzzy")) == 0
    out = capsys.readouterr().out
    assert "no library parts match" in out


# ---------- stock column (below-floor bundle visibility) ----------


def _synthetic_part(name, cid):
    from types import SimpleNamespace
    m = SimpleNamespace(name=name, mpn=name.upper(), sourcing={"lcsc": cid},
                        tags=[], symbol_name="SYM", footprint_name="FP",
                        maturity="prototype", watch_out_for=None)
    return SimpleNamespace(manifest=m, tier=SimpleNamespace(value="home"))


def test_parts_block_adds_stock_column_and_flags_below_floor(monkeypatch):
    # The model picks bundles from this table; a below-floor bundle it can't
    # SEE is adopted then bounced by §9.26 a commit later (the dominant BOM
    # retry). With a floor + the offline catalog, the stock column flags it.
    from kicraft.design import library
    from kicraft.parts_library import jlcparts
    parts = [_synthetic_part("lo-part", "C11"),
             _synthetic_part("hi-part", "C22"),
             _synthetic_part("unknown-part", "C33")]
    stock = {"C11": 12, "C22": 500_000}  # C33 absent -> not in the catalog
    monkeypatch.setattr(jlcparts, "available", lambda: True)
    monkeypatch.setattr(jlcparts, "lookup",
                        lambda cid: ({"stock": stock[cid]} if cid in stock else None))

    block = library._format_available_parts_block(parts, stock_floor=100)
    assert "| name | mpn | sourcing | stock | tags |" in block  # column added
    assert "12 ⚠<100" in block          # below floor -> flagged
    assert "500,000" in block and "500,000 ⚠" not in block  # above floor -> plain
    assert "unknown-part" in block      # not-in-catalog row still rendered (stock —)


def test_parts_block_omits_stock_column_without_a_floor(monkeypatch):
    # Back-compat: no floor supplied -> the original 8-column table, no catalog
    # lookups (callers that don't care about stock pay nothing).
    from kicraft.design import library
    from kicraft.parts_library import jlcparts
    monkeypatch.setattr(jlcparts, "available",
                        lambda: (_ for _ in ()).throw(AssertionError("no lookup expected")))
    block = library._format_available_parts_block(
        [_synthetic_part("p", "C11")], stock_floor=None)
    assert "| name | mpn | sourcing | tags |" in block and "| stock |" not in block
