"""Web app: live BOM part pricing (the cost column + total in the Parts table).

The lookup is split so the decision logic is testable without the network:
``_resolve_part``/``_price_key`` (how a part maps to a vendor query), ``_pick_price``
(which JLCPCB search result + price to use), and the ``_inspector_spec`` BOM branch
(cost cells, the total footer, and the pending state). All pure functions, tested
directly like the other ``kicraft.server`` tests."""
from __future__ import annotations

from kicraft.server import web


# ------------------------------------------------------ _resolve_part / _price_key

def test_resolve_part_embedded_id_manifest_mpn_kw_and_none():
    # 1. an embedded LCSC id in the symbol name wins outright
    assert web._resolve_part(
        {"symbol": "u:USBLC6-2SC6_C2687116", "footprint": "u:SOT-23-6"}) == ("id", "C2687116")
    # 2. a curated-bundle part ("<lib>:<name>") resolves to its manifest LCSC id --
    #    the exact part -- ahead of an MPN keyword search (which is also blocked).
    assert web._resolve_part(
        {"symbol": "tp4056:TP4056", "footprint": "tp4056:ESOP-8",
         "mpn": "TP4056-42-ESOP8"}) == ("id", "C16581")
    # 3. a non-bundle part with an MPN falls back to an MPN search
    assert web._resolve_part(
        {"symbol": "x:CHIP", "footprint": "x:QFN", "mpn": "SOMEMPN-123"}) == ("mpn", "SOMEMPN-123")
    # 4. a generic passive -> value + package-size keyword
    assert web._resolve_part(
        {"symbol": "Device:R", "footprint": "Resistor_SMD:R_0402_1005Metric",
         "value": "5.1k"}) == ("kw", "5.1k 0402")
    # 5. nothing to go on
    assert web._resolve_part(
        {"value": "", "symbol": "Device:X", "footprint": "Foo:BAR"}) is None


def test_price_key_mirrors_resolution():
    assert web._price_key({"symbol": "u:X_C2687116", "footprint": "u:y"}) == "id:C2687116"
    assert web._price_key(
        {"symbol": "Device:C", "footprint": "Capacitor_SMD:C_0805_2012Metric",
         "value": "100nF"}) == "kw:100nF 0805"
    assert web._price_key({"value": "", "symbol": "Device:X", "footprint": "Foo:BAR"}) is None


# --------------------------------------------------------------------- _pick_price

# Models the GCT "USB1046" case: the first in-stock result is an expensive false
# positive (a TI TUSB1046 mux), the actual part (a GCT connector) is the cheapest
# in stock, and an even-cheaper row is out of stock so it must be skipped.
_RESULTS = [
    {"lcsc": "C2151061", "price": 4.0149, "stock": 53},   # first in stock, wrong + dear
    {"lcsc": "C6307429", "price": 0.8426, "stock": 95},    # cheapest in stock -> want this
    {"lcsc": "C5815145", "price": 0.50, "stock": 0},       # cheaper but OUT of stock
]


def test_pick_price_takes_cheapest_in_stock_not_first():
    # Both keyword and MPN must skip the $4 false positive and the OOS $0.50 row.
    for kind in ("kw", "mpn"):
        r = web._pick_price(kind, "USB1046", _RESULTS)
        assert r["lcsc"] == "C6307429" and r["unit_price"] == 0.8426, kind


def test_pick_price_id_prefers_exact_match_over_cheapest():
    # An embedded LCSC id names a specific part: take it even though it's not cheapest.
    assert web._pick_price("id", "C2151061", _RESULTS)["lcsc"] == "C2151061"


def test_pick_price_falls_back_to_cheapest_when_all_out_of_stock():
    oos = [{"lcsc": "A", "price": 2.0, "stock": 0}, {"lcsc": "B", "price": 1.0, "stock": 0}]
    assert web._pick_price("kw", "x", oos)["lcsc"] == "B"


def test_pick_price_none_when_nothing_priced():
    assert web._pick_price("kw", "x", [{"price": 0, "stock": 1}, {"price": None}]) is None


def test_vendor_cell_links_to_priced_product_when_available():
    # Once priced, the vendor link points to the exact product we priced (so the
    # link and the cost agree), even for an MPN that would otherwise be a search.
    p = {"symbol": "Connector:USB_C", "footprint": "C:USB_C", "mpn": "USB1046"}
    prices = {"mpn:USB1046": {"unit_price": 0.8426, "lcsc": "C6307429"}}
    assert web._vendor_cell(p, prices) == {
        "text": "C6307429", "href": "https://www.lcsc.com/product-detail/C6307429.html"}
    # without a price -> unchanged search fallback
    assert web._vendor_cell(p)["href"] == "https://www.lcsc.com/search?q=USB1046"


# ----------------------------------------------------- bom _inspector_spec (cost)

_SJ = {"bom": {"parts": [
    {"ref": "R1", "value": "5.1k", "symbol": "Device:R",
     "footprint": "Resistor_SMD:R_0402_1005Metric", "sheet": "X"},
    {"ref": "U1", "value": "USBLC6", "symbol": "u:USBLC6_C2687116",
     "footprint": "u:SOT-23-6", "sheet": "X"},
]}}


def test_bom_cost_column_pending_when_unpriced():
    secs = web._inspector_spec("bom", _SJ, {}, None, [], prices={})
    parts = next(s for s in secs if s["title"] == "Parts")
    assert parts["columns"] == ["ref", "value", "cost", "vendor",
                                "footprint", "sheet", "symbol"]
    assert [r[2] for r in parts["rows"]] == ["...", "..."]      # cost cells
    assert parts["foot"][0][2] == "pricing..."
    assert "fetching" in parts["note"]


def test_bom_cost_column_and_total_when_priced():
    seed = {"kw:5.1k 0402": {"unit_price": 0.0009, "lcsc": "C25905"},
            "id:C2687116": {"unit_price": 0.18, "lcsc": "C2687116"}}
    secs = web._inspector_spec("bom", _SJ, {}, None, [], prices=seed)
    parts = next(s for s in secs if s["title"] == "Parts")
    assert [r[2] for r in parts["rows"]] == ["$0.0009", "$0.1800"]
    # total row sits under the cost column (index 2), label under value (index 1)
    assert parts["foot"][0][1] == "TOTAL (est.)"
    assert parts["foot"][0][2] == "$0.18"          # 0.0009 + 0.18, money-rounded
    assert "(2/2 priced)" in parts["note"]


def test_bom_cost_na_for_unmatched_and_unresolvable():
    # A part that resolves but the lookup returned no match (cached None) -> "n/a";
    # a part that doesn't resolve at all -> "n/a" too (no key to price on).
    sj = {"bom": {"parts": [
        {"ref": "R1", "value": "5.1k", "symbol": "Device:R",
         "footprint": "Resistor_SMD:R_0402_1005Metric", "sheet": "X"},
        {"ref": "X1", "value": "", "symbol": "Device:X", "footprint": "Foo:BAR", "sheet": "X"},
    ]}}
    secs = web._inspector_spec("bom", sj, {}, None, [], prices={"kw:5.1k 0402": None})
    parts = next(s for s in secs if s["title"] == "Parts")
    assert [r[2] for r in parts["rows"]] == ["n/a", "n/a"]
    assert parts["foot"][0][2] == "$0.0000"        # nothing priced -> zero total
    assert "(0/2 priced)" in parts["note"]


def test_fmt_total_switches_to_cents_above_a_dime():
    assert web._fmt_total(2.409) == "$2.41"
    assert web._fmt_total(0.0009) == "$0.0009"


def test_price_cache_ignores_stale_schema_and_roundtrips(tmp_path):
    import json
    k = "kw:__pricing_schema_probe__"
    web._PRICE_CACHE.pop(k, None)
    kdir = tmp_path / ".kicraft"
    kdir.mkdir(parents=True)
    # legacy flat file (no _schema) and an old-schema file must both be ignored,
    # so a _pick_price change re-fetches rather than serving the stale price.
    (kdir / web._PRICE_FILE).write_text(json.dumps({k: {"unit_price": 9.99, "lcsc": "CX"}}))
    web._load_price_cache(tmp_path)
    assert k not in web._PRICE_CACHE
    (kdir / web._PRICE_FILE).write_text(json.dumps({"_schema": 1, "prices": {k: {"unit_price": 9.0}}}))
    web._load_price_cache(tmp_path)
    assert k not in web._PRICE_CACHE
    # a current-schema file loads
    (kdir / web._PRICE_FILE).write_text(
        json.dumps({"_schema": web._PRICE_SCHEMA, "prices": {k: {"unit_price": 1.0, "lcsc": "CY"}}}))
    web._load_price_cache(tmp_path)
    assert web._PRICE_CACHE.get(k) == {"unit_price": 1.0, "lcsc": "CY"}
    # save writes the current schema (so it round-trips)
    web._save_price_cache(tmp_path, {k})
    written = json.loads((kdir / web._PRICE_FILE).read_text())
    assert written["_schema"] == web._PRICE_SCHEMA and k in written["prices"]
    web._PRICE_CACHE.pop(k, None)
