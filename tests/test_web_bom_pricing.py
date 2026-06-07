"""Web app: live BOM part pricing (the cost column + total in the Parts table).

The lookup is split so the decision logic is testable without the network:
``_resolve_part``/``_price_key`` (how a part maps to a vendor query), ``_pick_price``
(which JLCPCB search result + price to use), and the ``_inspector_spec`` BOM branch
(cost cells, the total footer, and the pending state). All pure functions, tested
directly like the other ``kicraft.server`` tests."""
from __future__ import annotations

from kicraft.server import web


# ------------------------------------------------------ _resolve_part / _price_key

def test_resolve_part_three_tiers_and_none():
    assert web._resolve_part(
        {"symbol": "u:USBLC6-2SC6_C2687116", "footprint": "u:SOT-23-6"}) == ("id", "C2687116")
    assert web._resolve_part(
        {"symbol": "tp4056:TP4056", "footprint": "tp4056:ESOP-8",
         "mpn": "TP4056-42-ESOP8"}) == ("mpn", "TP4056-42-ESOP8")
    assert web._resolve_part(
        {"symbol": "Device:R", "footprint": "Resistor_SMD:R_0402_1005Metric",
         "value": "5.1k"}) == ("kw", "5.1k 0402")
    assert web._resolve_part(
        {"value": "", "symbol": "Device:X", "footprint": "Foo:BAR"}) is None


def test_price_key_mirrors_resolution():
    assert web._price_key({"symbol": "u:X_C2687116", "footprint": "u:y"}) == "id:C2687116"
    assert web._price_key(
        {"symbol": "Device:C", "footprint": "Capacitor_SMD:C_0805_2012Metric",
         "value": "100nF"}) == "kw:100nF 0805"
    assert web._price_key({"value": "", "symbol": "Device:X", "footprint": "Foo:BAR"}) is None


# --------------------------------------------------------------------- _pick_price

_RESULTS = [
    {"lcsc": "C111", "price": 0.02, "stock": 0},      # out of stock
    {"lcsc": "C222", "price": 0.01, "stock": 500},    # cheapest in stock
    {"lcsc": "C333", "price": 0.05, "stock": 9},
]


def test_pick_price_keyword_takes_cheapest_in_stock():
    r = web._pick_price("kw", "5.1k 0402", _RESULTS)
    assert r["lcsc"] == "C222" and r["unit_price"] == 0.01


def test_pick_price_id_prefers_exact_match():
    assert web._pick_price("id", "C333", _RESULTS)["lcsc"] == "C333"


def test_pick_price_mpn_takes_first_in_stock():
    assert web._pick_price("mpn", "whatever", _RESULTS)["lcsc"] == "C222"


def test_pick_price_none_when_nothing_priced():
    assert web._pick_price("kw", "x", [{"price": 0, "stock": 1}, {"price": None}]) is None


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
