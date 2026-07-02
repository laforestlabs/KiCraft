"""BOM part resolution + price selection/formatting — the *pure* pricing helpers.

Extracted from web.py (refactor roadmap Phase 3). This module is I/O-free: how a
BOM part maps to a vendor query/link (`_resolve_part`, `_vendor_cell`, `_price_key`),
how one search result is chosen (`_pick_price`), and price string formatting. The
live network fetch + the process/disk price cache stay in web.py for now (they are
web-session-coupled via `_ensure_bom_prices(state)` and are the seam tests monkeypatch).
"""
from __future__ import annotations

import re
from urllib.parse import quote

from kicraft.parts_library.jlcparts import (
    bom_keyword, chip_value_matches, is_multi_element_array,
)

from .parts_catalog import get_part

# LCSC part id baked into a vendored symbol/footprint name (e.g.
# "USBLC6-2SC6_C2687116"); the negative lookbehind keeps it off footprint tokens
# like "C_0805" where the C is a package-class prefix, not a catalogue id.
_LCSC_ID_RE = re.compile(r"(?<![A-Za-z0-9])C\d{4,}")
# A bare LCSC catalogue id (full string), e.g. a manifest's "C16581".
_LCSC_CODE_RE = re.compile(r"C\d{4,}$")
# Curated parts-library bundle name -> its LCSC code (or None), memoized. A
# bundle's symbol/footprint id is "<name>:<...>" with no embedded catalogue id,
# but its manifest records the exact LCSC part it was built from.
_LIB_LCSC_CACHE: dict[str, str | None] = {}


def _lib_lcsc(lib: str) -> str | None:
    """The LCSC C-number for a curated parts-library bundle named ``lib``, else
    None. Cached by name (one catalog probe per distinct library, memoized), so
    it is cheap to call from the per-row resolution path."""
    if not lib or ":" in lib:
        return None
    if lib in _LIB_LCSC_CACHE:
        return _LIB_LCSC_CACHE[lib]
    code = None
    try:
        part = get_part(lib)
        if part is not None:
            c = (part.manifest.sourcing or {}).get("lcsc", "").strip().upper()
            if _LCSC_CODE_RE.match(c):
                code = c
    except Exception:  # pragma: no cover - a catalog probe must never break pricing
        code = None
    _LIB_LCSC_CACHE[lib] = code
    return code


def _resolve_part(p: dict) -> tuple[str, str] | None:
    """How to find this part at a vendor, as ``(kind, query)``: an LCSC id baked
    into the symbol/footprint name ("id", vendored easyeda parts); else the C#
    the §9.26 BOM gate pinned into ``sourcing_note`` ("id" — keeps the UI cost
    and vendor link on the exact part the fab BOM ships); else the exact LCSC
    id from a curated-bundle manifest ("id"); else the manufacturer part
    number ("mpn"); else a keyword from value + package ("kw", generic
    passives — see ``jlcparts.bom_keyword``). None when there is nothing to go
    on. Shared by the vendor link and the price lookup so both point at the
    same part."""
    sym = p.get("symbol") or ""
    fp = p.get("footprint") or ""
    m = _LCSC_ID_RE.search(sym) or _LCSC_ID_RE.search(fp)
    if m:
        return ("id", m.group(0))
    m = _LCSC_ID_RE.search(p.get("sourcing_note") or "")
    if m:
        return ("id", m.group(0))
    # A part drawn from a curated parts-library bundle ("<lib>:<name>"): price by
    # the bundle's exact LCSC id from its manifest. More precise than an MPN
    # keyword search and, crucially, it resolves through the still-working
    # easyeda.com endpoint instead of the WAF-blocked JLCPCB keyword search.
    for ref in (sym, fp):
        code = _lib_lcsc(ref.split(":", 1)[0])
        if code:
            return ("id", code)
    mpn = (p.get("mpn") or "").strip()
    if mpn:
        return ("mpn", mpn)
    terms = bom_keyword(p.get("value") or "", fp)
    return ("kw", terms) if terms else None


def _vendor_cell(p: dict, prices: dict | None = None) -> dict | str:
    """A clickable LCSC link for one BOM part. When the part has been priced, link
    to the exact product we priced (its LCSC id) so the link and the cost column
    always agree and the price is verifiable; otherwise an LCSC id -> the product
    page, an MPN or generic passive -> an LCSC search. "" when nothing resolves."""
    r = _resolve_part(p)
    if not r:
        return ""
    kind, q = r
    if prices is not None:
        res = prices.get(f"{kind}:{q}")
        if isinstance(res, dict) and res.get("lcsc"):
            cid = res["lcsc"]
            return {"text": cid, "href": f"https://www.lcsc.com/product-detail/{cid}.html"}
    if kind == "id":
        return {"text": q, "href": f"https://www.lcsc.com/product-detail/{q}.html"}
    return {"text": q if kind == "mpn" else "search",
            "href": "https://www.lcsc.com/search?q=" + quote(q)}


def _price_key(p: dict) -> str | None:
    r = _resolve_part(p)
    return f"{r[0]}:{r[1]}" if r else None


# Rank keyword/MPN picks away from the churn-prone bottom of the catalog: a
# sub-floor Extended row is routinely delisted or drained weeks after the dump
# is built (KC-V8YWN8's R2 pick 404'd on live LCSC three weeks in). Same env
# knob as the §9.26 BOM gate so both floors move together.
_KW_STOCK_FLOOR = 500


def _stock_floor() -> int:
    import os
    try:
        return int(os.environ.get("KICRAFT_BOM_STOCK_FLOOR", "")
                   or _KW_STOCK_FLOOR)
    except ValueError:
        return _KW_STOCK_FLOOR


# A "<value> <chip size>" keyword (bom_keyword's generic-passive shape,
# e.g. "10k 0603"; the size token mirrors jlcparts._FP_CHIP_SIZE_RE): the
# pick below must be a single 2-pad part of that value, so array rows and
# wrong-value substring matches are ineligible.
_CHIP_KW_RE = re.compile(r"^(\S+)\s+(\d{3,4})$")


def _pick_price(kind: str, query: str, results: list[dict]) -> dict | None:
    """Choose one JLCPCB search result and pull its unit price. For an LCSC id
    the exact id wins (it names a specific part; a different row must never be
    priced in its place — None when the exact id is absent). For an
    MPN/keyword, only in-stock rows are eligible (an out-of-stock part must
    never be selected — None beats a dead pick), preferring JLC **Basic**
    parts, then rows clearing the stock floor, then cheapest. Cheapest-only
    (the old rule) systematically landed on $0.0008 Extended long-tail
    listings that delist within weeks of the offline dump; Basic parts are
    JLC's stable no-setup-fee tier and are the anti-churn signal. Cheapest
    still breaks ties (a vague MPN/keyword pulls in false positives: e.g.
    "USB1046" returns both $4+ TI TUSB1046 muxes and the $0.84 GCT USB
    connector, and the connector is the one we want). Returns ``{"unit_price",
    "lcsc","stock","type"}`` or None when nothing usable came back. Pure: no
    network."""
    def price_of(r):
        try:
            return float(r.get("price"))
        except (TypeError, ValueError):
            return None
    priced = [r for r in results if (price_of(r) or 0) > 0]
    if not priced:
        return None
    if kind == "id":
        # The id names the exact part the BOM ships: price it even when out
        # of stock (its stock rides along so the UI can flag it) and never
        # substitute another row.
        r = next((x for x in priced
                  if str(x.get("lcsc", "")).upper() == query.upper()), None)
        if r is None:
            return None
    else:
        pool = [x for x in priced if (x.get("stock") or 0) > 0]
        if kind == "kw":
            m = _CHIP_KW_RE.match(query.strip())
            if m:
                # Generic chip passive: never price an array or a
                # wrong-value substring match (10k vs 510kΩ) — the same
                # ineligibility the §9.26 BOM walk applies when pinning.
                pool = [x for x in pool
                        if not is_multi_element_array(x)
                        and chip_value_matches(m.group(1), x)]
        if not pool:
            return None  # honest "no price" beats pricing a dead pick
        floor = _stock_floor()
        r = min(pool, key=lambda x: (
            0 if x.get("type") == "Basic" else 1,
            0 if (x.get("stock") or 0) >= floor else 1,
            price_of(x),
        ))
    return {"unit_price": price_of(r), "lcsc": r.get("lcsc"),
            "stock": r.get("stock"), "type": r.get("type")}


def _fmt_price(x: float) -> str:
    return f"${x:.4f}"


def _fmt_stock(n) -> str:
    """Compact stock count for the BOM table: 8912345 -> '8.9M', 16614 ->
    '16.6k', 0 -> '0', None -> '—' (unverified/unknown)."""
    if n is None:
        return "—"
    try:
        n = int(n)
    except (TypeError, ValueError):
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _fmt_total(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.10 else f"${x:.4f}"
