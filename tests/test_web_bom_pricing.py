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
        {"symbol": "tp4056:TP4056_C725790", "footprint": "tp4056:ESOP-8",
         "mpn": "TP4056"}) == ("id", "C725790")
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


def test_pick_price_id_prices_an_oos_exact_match_honestly():
    # The id names the exact part the BOM ships: price it even out of stock
    # (the stock rides along so the UI can flag it) — never substitute.
    r = web._pick_price("id", "C5815145", _RESULTS)
    assert r["lcsc"] == "C5815145" and r["stock"] == 0


def test_pick_price_id_returns_none_when_exact_id_absent():
    # Pricing a DIFFERENT part under an id key was a lie; now it's "no price".
    assert web._pick_price("id", "C0000001", _RESULTS) is None


def test_pick_price_returns_none_when_all_out_of_stock():
    # KC-4AZ7PE hardening: an out-of-stock row must never win a kw/mpn pick —
    # an honest "no price" beats presenting a dead part as the priced source.
    oos = [{"lcsc": "A", "price": 2.0, "stock": 0}, {"lcsc": "B", "price": 1.0, "stock": 0}]
    assert web._pick_price("kw", "x", oos) is None
    assert web._pick_price("mpn", "x", oos) is None


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
    assert parts["columns"] == ["ref", "value", "cost", "stock (JLC/retail)",
                                "vendor", "footprint", "sheet", "symbol"]
    assert [r[2] for r in parts["rows"]] == ["...", "..."]      # cost cells
    assert parts["foot"][0][2] == "pricing..."
    assert "fetching" in parts["note"]


def test_bom_cost_column_and_total_when_priced():
    seed = {"kw:5.1k 0402": {"unit_price": 0.0009, "lcsc": "C25905",
                             "stock": 8_912_345, "retail_stock": 16_614},
            "id:C2687116": {"unit_price": 0.18, "lcsc": "C2687116",
                            "stock": 500, "retail_stock": None}}
    secs = web._inspector_spec("bom", _SJ, {}, None, [], prices=seed)
    parts = next(s for s in secs if s["title"] == "Parts")
    assert [r[2] for r in parts["rows"]] == ["$0.0009", "$0.1800"]
    # both inventories render compactly; None retail = unverified dash
    assert [r[3] for r in parts["rows"]] == ["8.9M / 16.6k", "500 / —"]
    # total row sits under the cost column (index 2), label under value (index 1)
    assert parts["foot"][0][1] == "TOTAL (est.)"
    assert parts["foot"][0][2] == "$0.18"          # 0.0009 + 0.18, money-rounded
    assert "(2/2 priced)" in parts["note"]
    assert "JLCPCB assembly" in parts["note"]      # the stock-column legend


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
    # a current-schema entry whose retail reading was unverified (None) is
    # left unmerged so the reopen re-fetches and self-heals
    (kdir / web._PRICE_FILE).write_text(
        json.dumps({"_schema": web._PRICE_SCHEMA,
                    "prices": {k: {"unit_price": 1.0, "lcsc": "CY",
                                   "retail_stock": None}}}))
    web._load_price_cache(tmp_path)
    assert k not in web._PRICE_CACHE
    # a current-schema file with a verified retail reading loads
    (kdir / web._PRICE_FILE).write_text(
        json.dumps({"_schema": web._PRICE_SCHEMA,
                    "prices": {k: {"unit_price": 1.0, "lcsc": "CY",
                                   "retail_stock": 5}}}))
    web._load_price_cache(tmp_path)
    assert web._PRICE_CACHE.get(k) == {"unit_price": 1.0, "lcsc": "CY",
                                       "retail_stock": 5}
    # save writes the current schema (so it round-trips)
    web._save_price_cache(tmp_path, {k})
    written = json.loads((kdir / web._PRICE_FILE).read_text())
    assert written["_schema"] == web._PRICE_SCHEMA and k in written["prices"]
    web._PRICE_CACHE.pop(k, None)


# ----------------------------------------------- offline JLC catalog pricing

def _mk_catalog(tmp_path, monkeypatch, rows):
    import sqlite3
    db = tmp_path / "jlc.sqlite3"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE jlc_components (
        lcsc INTEGER PRIMARY KEY, mfr TEXT, package TEXT, manufacturer TEXT,
        library_type TEXT, stock INTEGER, price TEXT, description TEXT,
        joints INTEGER)""")
    con.executemany("INSERT INTO jlc_components (lcsc, mfr, package, manufacturer, library_type, stock, price, description) VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    monkeypatch.setenv("KICRAFT_JLCPARTS_DB", str(db))


def test_fetch_price_id_uses_offline_catalog_with_breaks(tmp_path, monkeypatch):
    _mk_catalog(tmp_path, monkeypatch, [
        (190004, "VL53L1CXV0FY/1", "LGA-12", "ST", "expand", 5640,
         "1-9:4.817,10-29:4.2745,30-99:3.9983,100-499:3.3492,1000-:3.1586", "ToF"),
    ])

    def _no_net(cid):
        raise AssertionError("easyeda must not be hit when the catalog prices it")
    monkeypatch.setattr(web, "_easyeda_lcsc_price", _no_net)

    r = web._fetch_price("id:C190004")
    assert r["unit_price"] == 4.817 and r["stock"] == 5640
    assert r["price_10"] == 4.2745 and r["price_100"] == 3.3492


def test_fetch_price_id_falls_back_to_easyeda_when_catalog_lacks_part(
        tmp_path, monkeypatch):
    _mk_catalog(tmp_path, monkeypatch, [])
    monkeypatch.setattr(
        web, "_easyeda_lcsc_price",
        lambda cid: {"unit_price": 1.23, "lcsc": cid, "stock": 7})
    r = web._fetch_price("id:C42")
    assert r["unit_price"] == 1.23 and "price_10" not in r


def test_fetch_price_keyword_via_offline_catalog(tmp_path, monkeypatch):
    _mk_catalog(tmp_path, monkeypatch, [
        (1525, "CL21B104KBCNNNC", "0805", "Samsung", "base", 900000,
         "1-:0.0046", "50V 100nF X7R +-10% 0805 MLCC ROHS"),
        (9999, "EXPENSIVE-FALSE-POSITIVE", "0805", "x", "expand", 5,
         "1-:4.00", "100nF 0805 something"),
    ])
    r = web._fetch_price("kw:100nF 0805")
    assert r["lcsc"] == "C1525" and r["unit_price"] == 0.0046  # cheapest in stock


def test_fetch_price_keyword_unavailable_without_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_JLCPARTS_DB", str(tmp_path / "absent.sqlite3"))
    import pytest
    with pytest.raises(web._SourceUnavailable):
        web._fetch_price("kw:100nF 0805")


# --------------------------------------------- retail stock rides every pick

class _FakeRetail:
    """Stands in for web.lcsc_retail: enabled + stock, no network."""

    from kicraft.parts_library.lcsc_retail import RetailUnavailable

    def __init__(self, stock_by_lcsc=None, up=True):
        self.by = stock_by_lcsc or {}
        self.up = up

    def enabled(self):
        return True

    def stock(self, cid):
        if not self.up:
            raise self.RetailUnavailable("down")
        return {"lcsc": cid, "stock": self.by.get(cid, 5000), "min_buy": 1,
                "checked_at": "t"}


def test_fetch_price_attaches_live_retail_stock(tmp_path, monkeypatch):
    _mk_catalog(tmp_path, monkeypatch, [
        (190004, "VL53L1CXV0FY/1", "LGA-12", "ST", "expand", 5640,
         "1-9:4.817,1000-:3.1586", "ToF"),
    ])
    monkeypatch.setattr(web, "lcsc_retail", _FakeRetail({"C190004": 321}))
    r = web._fetch_price("id:C190004")
    assert r["retail_stock"] == 321 and r["retail_min_buy"] == 1


def test_fetch_price_retail_outage_marks_unverified(tmp_path, monkeypatch):
    _mk_catalog(tmp_path, monkeypatch, [
        (190004, "VL53L1CXV0FY/1", "LGA-12", "ST", "expand", 5640,
         "1-:4.817", "ToF"),
    ])
    monkeypatch.setattr(web, "lcsc_retail", _FakeRetail(up=False))
    r = web._fetch_price("id:C190004")
    assert r["retail_stock"] is None  # unverified — not persisted, self-heals


def test_fetch_price_retail_disabled_marks_unverified(tmp_path, monkeypatch):
    # The suite-wide conftest sets KICRAFT_LCSC_RETAIL=0: the real module is
    # in place but disabled, so picks carry retail_stock None with no fetch.
    _mk_catalog(tmp_path, monkeypatch, [
        (190004, "VL53L1CXV0FY/1", "LGA-12", "ST", "expand", 5640,
         "1-:4.817", "ToF"),
    ])
    r = web._fetch_price("id:C190004")
    assert r["retail_stock"] is None and r["retail_min_buy"] is None


# ----------------------------------------------- _pick_price anti-churn ranking

# KC-V8YWN8: cheapest-only keyword picks landed on $0.0008 Extended long-tail
# rows that delisted within weeks of the offline dump (R2's C22356624 404'd on
# live LCSC). Basic parts and floor-clearing stock now outrank bare price.

def test_pick_price_prefers_basic_over_cheaper_extended():
    rows = [
        {"lcsc": "EXT", "price": 0.0008, "stock": 2_000_000, "type": "Extended"},
        {"lcsc": "BAS", "price": 0.0030, "stock": 800_000, "type": "Basic"},
    ]
    r = web._pick_price("kw", "1k 0603", rows)
    assert r["lcsc"] == "BAS" and r["type"] == "Basic"


def test_pick_price_prefers_floor_clearing_stock_over_cheaper_trickle():
    rows = [
        {"lcsc": "DRY", "price": 0.001, "stock": 49, "type": "Extended"},
        {"lcsc": "WET", "price": 0.002, "stock": 100_000, "type": "Extended"},
    ]
    assert web._pick_price("kw", "10k 0603", rows)["lcsc"] == "WET"


def test_pick_price_cheapest_still_breaks_ties_within_a_tier():
    rows = [
        {"lcsc": "DEAR", "price": 0.02, "stock": 90_000, "type": "Basic"},
        {"lcsc": "CHEAP", "price": 0.01, "stock": 80_000, "type": "Basic"},
    ]
    assert web._pick_price("kw", "100 0603", rows)["lcsc"] == "CHEAP"


def test_pick_price_stock_floor_env_override(monkeypatch):
    rows = [
        {"lcsc": "LOW", "price": 0.001, "stock": 60, "type": "Extended"},
        {"lcsc": "HIGH", "price": 0.002, "stock": 100_000, "type": "Extended"},
    ]
    monkeypatch.setenv("KICRAFT_BOM_STOCK_FLOOR", "50")
    # Floor lowered to 50 -> both clear it -> cheapest wins again.
    assert web._pick_price("kw", "x", rows)["lcsc"] == "LOW"


def test_pick_price_id_exact_match_ignores_ranking():
    rows = [
        {"lcsc": "C1", "price": 4.0, "stock": 10, "type": "Extended"},
        {"lcsc": "C2", "price": 0.5, "stock": 1_000_000, "type": "Basic"},
    ]
    assert web._pick_price("id", "C1", rows)["lcsc"] == "C1"
