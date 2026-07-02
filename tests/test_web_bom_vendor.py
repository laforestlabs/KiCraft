"""Web app: per-part vendor links in the BOM "Parts" inspector table.

``_vendor_cell`` turns one BOM part into a clickable LCSC lookup, best-effort by
what the part carries; ``_inspector_spec("bom", ...)`` wires a "vendor" column of
those cells. Both are pure functions, tested directly like the other
``kicraft.server`` tests rather than through a browser."""
from __future__ import annotations

from kicraft.server import web


# --------------------------------------------------------------- _vendor_cell

def test_vendor_cell_embedded_lcsc_id_links_to_product_page():
    # Vendored easyeda parts bake the catalogue id into the symbol name.
    p = {"value": "USBLC6-2SC6", "symbol": "usblc6-2sc6:USBLC6-2SC6_C2687116",
         "footprint": "usblc6-2sc6:SOT-23-6_L2.9-W1.6-P0.95"}
    cell = web._vendor_cell(p)
    assert cell == {"text": "C2687116",
                    "href": "https://www.lcsc.com/product-detail/C2687116.html"}


def test_vendor_cell_mpn_falls_back_to_search():
    # A part NOT drawn from a curated bundle, but with an MPN, links to an MPN search.
    p = {"value": "Foo", "symbol": "x:CHIP", "footprint": "x:QFN", "mpn": "SOMEMPN-123"}
    cell = web._vendor_cell(p)
    assert cell["text"] == "SOMEMPN-123"
    assert cell["href"] == "https://www.lcsc.com/search?q=SOMEMPN-123"


def test_vendor_cell_library_part_links_to_manifest_lcsc():
    # A part from a curated bundle ("tp4056:…") resolves to that bundle's exact
    # LCSC id from its manifest, so the link points at the precise product even
    # with no live price -- and ahead of the (less precise) MPN it also carries.
    p = {"value": "TP4056", "symbol": "tp4056:TP4056_C725790",
         "footprint": "tp4056:ESOP-8", "mpn": "TP4056"}
    cell = web._vendor_cell(p)
    assert cell == {"text": "C725790",
                    "href": "https://www.lcsc.com/product-detail/C725790.html"}


def test_vendor_cell_passive_searches_by_value_and_size():
    p = {"value": "5.1k", "symbol": "Device:R",
         "footprint": "Resistor_SMD:R_0402_1005Metric"}
    cell = web._vendor_cell(p)
    assert cell["text"] == "search"
    assert cell["href"] == "https://www.lcsc.com/search?q=5.1k%200402"


def test_vendor_cell_does_not_misread_package_class_as_lcsc_id():
    # "C_0805" is a capacitor package-class prefix, NOT a C<digits> catalogue id.
    p = {"value": "100nF", "symbol": "Device:C",
         "footprint": "Capacitor_SMD:C_0805_2012Metric"}
    cell = web._vendor_cell(p)
    assert cell["text"] == "search"
    assert "product-detail" not in cell["href"]
    assert cell["href"] == "https://www.lcsc.com/search?q=100nF%200805"


def test_vendor_cell_nothing_to_search_returns_empty():
    p = {"value": "", "symbol": "Device:X", "footprint": "Foo:BAR"}
    assert web._vendor_cell(p) == ""


# ------------------------------------------------------- bom _inspector_spec

def test_bom_inspector_spec_adds_vendor_column_with_links():
    sj = {"bom": {"parts": [
        {"ref": "U4", "value": "USBLC6-2SC6",
         "symbol": "usblc6-2sc6:USBLC6-2SC6_C2687116",
         "footprint": "usblc6-2sc6:SOT-23-6", "sheet": "USB"},
        {"ref": "R1", "value": "5.1k", "symbol": "Device:R",
         "footprint": "Resistor_SMD:R_0402_1005Metric", "sheet": "USB"},
    ]}}
    # prices={} -> unpriced, so vendor cells use the static resolution (deterministic).
    secs = web._inspector_spec("bom", sj, {}, None, [], prices={})
    parts = next(s for s in secs if s.get("title") == "Parts")
    assert parts["columns"] == ["ref", "value", "cost", "stock (JLC/retail)",
                                "vendor", "footprint", "sheet", "symbol"]
    vendor_idx = parts["columns"].index("vendor")
    # every row's vendor cell is a {"text","href"} link dict here
    u4 = parts["rows"][0]
    assert u4[0] == "U4"
    assert u4[vendor_idx]["href"].endswith("/C2687116.html")
    assert "search?q=" in parts["rows"][1][vendor_idx]["href"]
