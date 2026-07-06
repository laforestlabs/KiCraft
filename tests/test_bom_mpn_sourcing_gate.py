"""§9.26 — _resolve_bom_mpn_sourcing: every BOM part must be a real part in
stock BOTH for JLCPCB assembly (offline jlcparts dump) AND at the lcsc.com
retail storefront (live lcsc_retail check).

Regression sources: KC-T6ERHM (J1/J2 shipped an out-of-stock terminal block;
MPN strings were unchecked LLM prose and the fab BOM's LCSC column shipped
blank) and KC-4AZ7PE (auto-picked 0603 passives had millions in the JLC dump
but 0 at the retail storefront — the two inventories are separate pools).
The gate verifies against the offline catalog + live retail and auto-pins
confident matches into sourcing_note (where fab_export reads the C#).
"""
from __future__ import annotations

import kicraft.design.cli_app as cli_app
from kicraft.design.cli_app import (
    _check_passive_array_mismatch, _resolve_bom_mpn_sourcing,
)
from kicraft.design.models import BOM, BomPart
from kicraft.parts_library.lcsc_retail import RetailUnavailable


def _part(ref="D1", mpn="SS34", note=None, symbol="Device:D_Schottky",
          footprint="Diode_SMD:D_SMA"):
    return BomPart(ref=ref, value="x", symbol=symbol, footprint=footprint,
                   sheet="A", mpn=mpn, sourcing_note=note)


def _bom(*parts):
    return BOM(parts=list(parts), connections=[])


class _FakeCatalog:
    """Stands in for the jlcparts module: available + search + lookup. The
    pure keyword helpers delegate to the real module — they touch no I/O and
    the gate's tier-4 behavior depends on their real normalization."""

    def __init__(self, by_mpn=None, by_lcsc=None, up=True, age=None):
        self.by_mpn = by_mpn or {}
        self.by_lcsc = by_lcsc or {}
        self.up = up
        self.age = age

    def available(self):
        return self.up

    def dump_age_days(self):
        return self.age

    def search(self, query, limit=10):
        return self.by_mpn.get(query.upper(), [])

    def lookup(self, lcsc_id):
        return self.by_lcsc.get(str(lcsc_id).upper())

    @staticmethod
    def bom_keyword(value, footprint):
        from kicraft.parts_library.jlcparts import bom_keyword
        return bom_keyword(value, footprint)

    @staticmethod
    def relax_keyword(kw):
        from kicraft.parts_library.jlcparts import relax_keyword
        return relax_keyword(kw)

    @staticmethod
    def is_unsourceable_hardware(footprint):
        from kicraft.parts_library.jlcparts import is_unsourceable_hardware
        return is_unsourceable_hardware(footprint)


class _FakeRetail:
    """Stands in for the lcsc_retail module. Default: everything plentiful at
    retail, so tests that aren't about retail behave as before the check."""

    RetailUnavailable = RetailUnavailable

    def __init__(self, by_lcsc=None, up=True, on=True, default_stock=1_000_000):
        self.by_lcsc = by_lcsc or {}
        self.up = up
        self.on = on
        self.default_stock = default_stock
        self.calls: list[str] = []

    def enabled(self):
        return self.on

    def retail_floor(self):
        import os
        try:
            return int(os.environ.get("KICRAFT_BOM_RETAIL_STOCK_FLOOR", "")
                       or 100)
        except ValueError:
            return 100

    def stock(self, cid):
        cid = str(cid).upper()
        self.calls.append(cid)
        if not self.up:
            raise RetailUnavailable("storefront down")
        e = self.by_lcsc.get(cid, {"stock": self.default_stock, "min_buy": 1})
        return {"lcsc": cid, "stock": e["stock"],
                "min_buy": e.get("min_buy", 1), "checked_at": "t"}

    def in_stock(self, cid, *, picky):
        info = self.stock(cid)
        need = max(info["min_buy"], self.retail_floor() if picky else 1)
        return info["stock"] >= need, info


def _install(monkeypatch, catalog, retail=None):
    monkeypatch.setattr(cli_app, "jlcparts", catalog)
    monkeypatch.setattr(cli_app, "lcsc_retail", retail or _FakeRetail())
    # No library bundles in play: parts resolve purely by MPN in these tests.
    monkeypatch.setattr(cli_app, "_load_library_parts", lambda root: ([], []))


def test_real_mpn_is_auto_pinned_into_sourcing_note(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog(by_mpn={
        "SS34": [{"lcsc": "C8678", "model": "SS34", "stock": 3941831}]}))
    p = _part()
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note == "LCSC C8678"


def test_hallucinated_mpn_is_an_offender(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog())
    p = _part(mpn="TOTALLYFAKE-99")
    bad, warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert len(bad) == 1 and "TOTALLYFAKE-99" in bad[0] and "not found" in bad[0]
    assert warns == []
    assert p.sourcing_note is None  # never pin what we couldn't verify


def test_low_stock_is_an_offender(tmp_path, monkeypatch):
    # The KC-T6ERHM L1: real Bourns part, 49 units in the snapshot.
    _install(monkeypatch, _FakeCatalog(by_mpn={
        "SRR1280-100M": [{"lcsc": "C2041557", "model": "SRR1280-100M", "stock": 49}]}))
    p = _part(ref="L1", mpn="SRR1280-100M", symbol="Device:L",
              footprint="Inductor_SMD:L_12x12mm_H4.5mm")
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert len(bad) == 1 and "JLC stock 49" in bad[0]
    assert p.sourcing_note is None


def test_stock_floor_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_BOM_STOCK_FLOOR", "10")
    _install(monkeypatch, _FakeCatalog(by_mpn={
        "SRR1280-100M": [{"lcsc": "C2041557", "model": "SRR1280-100M", "stock": 49}]}))
    p = _part(ref="L1", mpn="SRR1280-100M")
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note == "LCSC C2041557"


def test_family_prefix_match_pins_the_orderable_variant(tmp_path, monkeypatch):
    # Stock-ordered results: the first model extending the family name wins.
    _install(monkeypatch, _FakeCatalog(by_mpn={
        "SS34": [{"lcsc": "C407539", "model": "SS34F", "stock": 900000},
                 {"lcsc": "C8678", "model": "SS34", "stock": 100}]}))
    p = _part()
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note == "LCSC C407539"


def test_explicit_sourcing_note_lcsc_is_verified_not_searched(tmp_path, monkeypatch):
    cat = _FakeCatalog(by_lcsc={"C9864": {"lcsc": "C9864", "model": "TPS5430DDAR",
                                          "stock": 148617}})
    _install(monkeypatch, cat)
    p = _part(ref="U1", mpn="TPS5430DDAR", note="LCSC C9864 (buck IC)")
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note == "LCSC C9864 (buck IC)"  # untouched


def test_fabricated_sourcing_note_lcsc_is_an_offender(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog())
    p = _part(ref="U1", mpn="TPS5430DDAR", note="LCSC C99999999")
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert len(bad) == 1 and "C99999999" in bad[0]


def test_mpnless_passive_with_no_match_stays_unpinned_not_blocked(
    tmp_path, monkeypatch
):
    # Tier 4 (KC-V8YWN8): a generic with a keyword but no catalog match must
    # NOT bounce the model (the search misses legitimate parts); it stays
    # visibly unpriced instead.
    _install(monkeypatch, _FakeCatalog())
    p = _part(ref="C1", mpn=None, symbol="Device:C",
              footprint="Capacitor_SMD:C_0603_1608Metric")
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note is None


# ------------------------------------------------- tier 4: keyword sourcing

def _kw_catalog(term, rows):
    # search() keys are upper-cased queries.
    return _FakeCatalog(by_mpn={term.upper(): rows})


def test_mpnless_passive_is_keyword_pinned(tmp_path, monkeypatch):
    _install(monkeypatch, _kw_catalog("100nF 0603", [
        {"lcsc": "C1590", "model": "CL10B104KB8NNNC", "stock": 5_000_000,
         "type": "Basic"}]))
    p = _part(ref="C1", mpn=None, symbol="Device:C",
              footprint="Capacitor_SMD:C_0603_1608Metric")
    p.value = "100nF"
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note == "LCSC C1590"


def test_keyword_pick_prefers_basic_over_stocked_extended(tmp_path, monkeypatch):
    _install(monkeypatch, _kw_catalog("1k 0603", [
        {"lcsc": "CEXT", "model": "CHURN", "stock": 2_000_000, "type": "Extended"},
        {"lcsc": "CBAS", "model": "0603WAF1001T5E", "stock": 900_000,
         "type": "Basic"}]))
    p = _part(ref="R2", mpn=None, symbol="Device:R",
              footprint="Resistor_SMD:R_0603_1608Metric")
    p.value = "1k"
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note == "LCSC CBAS"


def test_keyword_qualifiers_relax_on_miss(tmp_path, monkeypatch):
    # "0.1uF 25V X7R 0603" matches nothing; the relaxed "0.1uF 0603" must hit.
    _install(monkeypatch, _kw_catalog("0.1uF 0603", [
        {"lcsc": "C14663", "model": "CC0603KRX7R8BB104", "stock": 3_000_000,
         "type": "Basic"}]))
    p = _part(ref="C2", mpn=None, symbol="Device:C",
              footprint="Capacitor_SMD:C_0603_1608Metric")
    p.value = "0.1uF 25V X7R"
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note == "LCSC C14663"


def test_pin_header_kicadism_is_normalized_and_pinned(tmp_path, monkeypatch):
    # The J2 case: value "PinHeader_1x02" matches nothing; the footprint
    # normalizes to a term the catalog answers.
    _install(monkeypatch, _kw_catalog("pin header 2.54mm 1x2P", [
        {"lcsc": "C492401", "model": "PZ254V-11-02P", "stock": 1_486_041,
         "type": "Extended"}]))
    p = _part(ref="J2", mpn=None, symbol="Connector_Generic:Conn_01x02",
              footprint="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical")
    p.value = "PinHeader_1x02"
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note == "LCSC C492401"


def test_test_points_are_not_sourcing_offenders(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog())
    p = _part(ref="TP1", mpn=None, symbol="Connector:TestPoint",
              footprint="TestPoint:TestPoint_Pad_D1.5mm")
    p.value = "TestPoint"
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note is None


def test_part_with_nothing_searchable_is_an_offender(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog())
    p = _part(ref="X1", mpn=None, symbol="foo:bar", footprint="foo:bar")
    p.value = ""
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert len(bad) == 1 and "X1" in bad[0] and "unsourceable" in bad[0]


def test_catalog_unavailable_fails_open(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog(up=False))
    p = _part(mpn="TOTALLYFAKE-99")
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note is None


# ------------------------------------------------- library bundles (stock)

class _Man:
    name = "tps5430"
    sourcing = {"lcsc": "C9864"}


class _Loaded:
    manifest = _Man()


def _install_bundle(monkeypatch, catalog, retail=None):
    monkeypatch.setattr(cli_app, "jlcparts", catalog)
    monkeypatch.setattr(cli_app, "lcsc_retail", retail or _FakeRetail())
    monkeypatch.setattr(cli_app, "_load_library_parts",
                        lambda root: ([_Loaded()], []))


def _bundle_part():
    return _part(ref="U1", mpn="TPS5430DDAR", symbol="tps5430:TPS5430DDAR",
                 footprint="tps5430:ESOP-8")


def test_library_bundle_in_stock_everywhere_passes_unpinned(tmp_path, monkeypatch):
    """Existence stays the _unresolved_lcsc gate's territory — no search, no
    pin here — but the manifest C# is now stock-checked in both inventories."""
    _install_bundle(monkeypatch, _FakeCatalog(by_lcsc={
        "C9864": {"lcsc": "C9864", "model": "TPS5430DDAR", "stock": 148617}}))
    p = _bundle_part()
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note is None


def test_library_bundle_low_jlc_stock_is_an_offender(tmp_path, monkeypatch):
    # Bundles were previously exempt from stock checks entirely.
    _install_bundle(monkeypatch, _FakeCatalog(by_lcsc={
        "C9864": {"lcsc": "C9864", "model": "TPS5430DDAR", "stock": 12}}))
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(_bundle_part()), tmp_path)
    assert len(bad) == 1 and "tps5430" in bad[0] and "only 12" in bad[0]
    assert "add_part_from_lcsc" in bad[0]


def test_library_bundle_retail_dry_is_an_offender(tmp_path, monkeypatch):
    _install_bundle(
        monkeypatch,
        _FakeCatalog(by_lcsc={
            "C9864": {"lcsc": "C9864", "model": "TPS5430DDAR", "stock": 148617}}),
        _FakeRetail(by_lcsc={"C9864": {"stock": 0, "min_buy": 1}}))
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(_bundle_part()), tmp_path)
    assert len(bad) == 1 and "retail storefront" in bad[0]
    assert "add_part_from_lcsc" in bad[0]


def test_shared_bundle_lcsc_is_live_checked_once(tmp_path, monkeypatch):
    retail = _FakeRetail()
    _install_bundle(monkeypatch, _FakeCatalog(by_lcsc={
        "C9864": {"lcsc": "C9864", "model": "TPS5430DDAR", "stock": 148617}}),
        retail)
    parts = [_bundle_part() for _ in range(5)]
    for i, p in enumerate(parts):
        p.ref = f"U{i + 1}"
    assert _resolve_bom_mpn_sourcing(_bom(*parts), tmp_path) == ([], [])
    assert retail.calls == ["C9864"]  # memoized per gate pass


# ------------------------------------------------- retail-dry picks (KC-4AZ7PE)

def test_explicit_pin_retail_dry_is_an_offender_naming_both(tmp_path, monkeypatch):
    # The KC-4AZ7PE shape: millions in the JLC dump, 0 at the storefront.
    _install(monkeypatch,
             _FakeCatalog(by_lcsc={
                 "C25804": {"lcsc": "C25804", "model": "0603WAF1002T5E",
                            "stock": 7_612_043}}),
             _FakeRetail(by_lcsc={"C25804": {"stock": 0, "min_buy": 100}}))
    p = _part(ref="R1", mpn=None, note="LCSC C25804")
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert len(bad) == 1
    assert "JLCPCB assembly" in bad[0] and "retail storefront" in bad[0]
    assert "BOTH" in bad[0]


def test_explicit_pin_low_retail_but_orderable_passes(tmp_path, monkeypatch):
    # Veto threshold is the listing's own min buy, not the picky floor: a
    # deliberately chosen niche part with 40 sellable units is orderable.
    _install(monkeypatch,
             _FakeCatalog(by_lcsc={
                 "C77": {"lcsc": "C77", "model": "NICHE-IC", "stock": 9000}}),
             _FakeRetail(by_lcsc={"C77": {"stock": 40, "min_buy": 1}}))
    p = _part(ref="U3", mpn=None, note="LCSC C77")
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])


def test_mpn_walk_skips_retail_dry_variant_to_next_in_stock(tmp_path, monkeypatch):
    _install(monkeypatch,
             _FakeCatalog(by_mpn={"SS34": [
                 {"lcsc": "C407539", "model": "SS34F", "stock": 900_000},
                 {"lcsc": "C8678", "model": "SS34", "stock": 800_000}]}),
             _FakeRetail(by_lcsc={"C407539": {"stock": 0, "min_buy": 1}}))
    p = _part()
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note == "LCSC C8678"  # dry SS34F skipped, not bounced


def test_mpn_walk_exhausted_is_an_offender_enumerating_tries(tmp_path, monkeypatch):
    _install(monkeypatch,
             _FakeCatalog(by_mpn={"SS34": [
                 {"lcsc": "C407539", "model": "SS34F", "stock": 900_000},
                 {"lcsc": "C8678", "model": "SS34", "stock": 100}]}),
             _FakeRetail(by_lcsc={"C407539": {"stock": 0, "min_buy": 1}}))
    p = _part()
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert len(bad) == 1 and "no orderable variant" in bad[0]
    assert "SS34F" in bad[0] and "retail stock 0" in bad[0]
    assert "SS34 (C8678) JLC stock 100" in bad[0]
    assert p.sourcing_note is None


def test_kw_walk_picks_next_candidate_when_basic_is_retail_dry(tmp_path, monkeypatch):
    _install(monkeypatch,
             _kw_catalog("1k 0603", [
                 {"lcsc": "CEXT", "model": "CHURN", "stock": 2_000_000,
                  "type": "Extended"},
                 {"lcsc": "CBAS", "model": "0603WAF1001T5E", "stock": 900_000,
                  "type": "Basic"}]),
             _FakeRetail(by_lcsc={"CBAS": {"stock": 0, "min_buy": 100}}))
    p = _part(ref="R2", mpn=None, symbol="Device:R",
              footprint="Resistor_SMD:R_0603_1608Metric")
    p.value = "1k"
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note == "LCSC CEXT"  # Basic preferred but dry → next


def test_kw_walk_exhausted_stays_unpinned_with_warning_not_offender(
        tmp_path, monkeypatch):
    retail = _FakeRetail(default_stock=0)  # everything dry at retail
    _install(monkeypatch,
             _kw_catalog("1k 0603", [
                 {"lcsc": "CBAS", "model": "0603WAF1001T5E", "stock": 900_000,
                  "type": "Basic"}]),
             retail)
    p = _part(ref="R2", mpn=None, symbol="Device:R",
              footprint="Resistor_SMD:R_0603_1608Metric")
    p.value = "1k"
    bad, warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert bad == []
    assert len(warns) == 1 and "R2" in warns[0] and "retail" in warns[0]
    assert p.sourcing_note is None


def test_kw_walk_respects_the_live_check_cap(tmp_path, monkeypatch):
    rows = [{"lcsc": f"C{i}", "model": f"R{i}", "stock": 1_000_000,
             "type": "Extended"} for i in range(10)]
    retail = _FakeRetail(default_stock=0)  # all dry → full walk
    _install(monkeypatch, _kw_catalog("1k 0603", rows), retail)
    p = _part(ref="R2", mpn=None, symbol="Device:R",
              footprint="Resistor_SMD:R_0603_1608Metric")
    p.value = "1k"
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert bad == [] and p.sourcing_note is None
    assert len(retail.calls) == cli_app._RETAIL_WALK_CAP_KW


def test_retail_outage_fails_open_with_a_warning(tmp_path, monkeypatch):
    _install(monkeypatch,
             _FakeCatalog(by_mpn={
                 "SS34": [{"lcsc": "C8678", "model": "SS34", "stock": 3_941_831}]}),
             _FakeRetail(up=False))
    p = _part()
    bad, warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert bad == []
    assert p.sourcing_note == "LCSC C8678"  # accepted, not bounced
    assert len(warns) == 1 and "unverified" in warns[0] and "C8678" in warns[0]


def test_retail_disabled_makes_zero_live_calls_and_no_warnings(
        tmp_path, monkeypatch):
    retail = _FakeRetail(on=False, default_stock=0)  # would be dry if consulted
    _install(monkeypatch,
             _FakeCatalog(by_mpn={
                 "SS34": [{"lcsc": "C8678", "model": "SS34", "stock": 3_941_831}]}),
             retail)
    p = _part()
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])
    assert p.sourcing_note == "LCSC C8678"
    assert retail.calls == []


def test_stale_dump_age_emits_a_warning(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog(by_mpn={
        "SS34": [{"lcsc": "C8678", "model": "SS34", "stock": 3_941_831}]},
        age=21.0))
    bad, warns = _resolve_bom_mpn_sourcing(_bom(_part()), tmp_path)
    assert bad == []
    assert any("21 days old" in w and "jlcparts-update" in w for w in warns)


# ------------------------------------------------- §9.28 array-on-passive

def test_array_lcsc_on_single_resistor_is_an_offender(tmp_path, monkeypatch):
    # §9.28: an 8-pin 0603x4 resistor array (C29718) on a 2-pad R_0603
    # footprint can never land — fewer pads than pins.
    _install(monkeypatch, _FakeCatalog(by_lcsc={
        "C29718": {"lcsc": "C29718", "package": "0603x4", "joints": 8,
                   "stock": 826216, "description": "10kΩ 4 RES ARRAY 0603x4"}}))
    p = _part(ref="R1", mpn=None, note="LCSC C29718", symbol="Device:R",
              footprint="Resistor_SMD:R_0603_1608Metric")
    bad = _check_passive_array_mismatch(_bom(p), tmp_path)
    assert len(bad) == 1
    assert "C29718" in bad[0] and "array" in bad[0]


def test_single_resistor_lcsc_passes(tmp_path, monkeypatch):
    # A genuine 2-joint 0603 resistor on R_0603 is not an array — no offender.
    _install(monkeypatch, _FakeCatalog(by_lcsc={
        "C25804": {"lcsc": "C25804", "package": "0603", "joints": 2,
                   "stock": 1000}}))
    p = _part(ref="R1", mpn=None, note="LCSC C25804", symbol="Device:R",
              footprint="Resistor_SMD:R_0603_1608Metric")
    assert _check_passive_array_mismatch(_bom(p), tmp_path) == []


# --------------------------------------- §9.26 walk vs arrays / wrong values
# KC-8XZS9Q: the tier-4 keyword walk auto-pinned the Basic 0603x4 resistor
# array C29718 for a generic "10k 0603" — §9.28 then rejected the pipeline's
# own pin, an unwinnable retry loop. The walk (kw AND MPN-family tiers) must
# never offer a multi-element array, or a wrong-value substring match
# ("10k" inside "510kΩ"), to a single 2-pad passive.

def _kw_row(lcsc, model, typ, stock, desc, joints=2, package="0603"):
    return {"lcsc": lcsc, "model": model, "brand": None, "package": package,
            "stock": stock, "joints": joints, "type": typ, "price": 0.001,
            "description": desc}


# The real "10k 0603" catalog top rows (stock-ordered), abridged.
_KW_10K_0603 = [
    _kw_row("C2930027", "FRC0603J103", "Extended", 2_042_275,
            "100mW 10kΩ 75V Thick Film Resistor ±5% 0603 Chip Resistor"),
    _kw_row("C2907178", "FRC0603J514", "Extended", 1_373_394,
            "100mW 510kΩ 75V Thick Film Resistor ±5% 0603 Chip Resistor"),
    _kw_row("C5126214", "FRH0603B1002TS", "Extended", 979_259,
            "100mW 10kΩ 75V Thick Film Resistor ±0.1% 0603 Chip Resistor"),
    _kw_row("C29718", "4D03WGJ0103T5E", "Basic", 826_216,
            "10kΩ 4 62.5mW 8 ±5% 0603x4 Resistor Networks, Arrays",
            joints=8, package="0603x4"),
    _kw_row("C23192", "0603WAF5103T5E", "Basic", 419_842,
            "100mW 510kΩ 75V Thick Film Resistor ±1% 0603 Chip Resistor"),
]


def _generic_10k(ref="R1"):
    return BomPart(ref=ref, value="10k", symbol="Device:R",
                   footprint="Resistor_SMD:R_0603_1608Metric", sheet="A")


def test_kw_walk_pins_a_true_single_never_the_array_or_a_wrong_value(
        tmp_path, monkeypatch):
    # Basic-first ranking would try C29718 (array) then C23192 (510kΩ) —
    # both ineligible. First true single C2930027 is retail-dry, so the
    # walk must land on C5126214, and §9.28 must agree with the pin.
    retail = _FakeRetail(by_lcsc={"C2930027": {"stock": 0}})
    _install(monkeypatch, _FakeCatalog(
        by_mpn={"10K 0603": _KW_10K_0603},
        by_lcsc={"C5126214": {"lcsc": "C5126214", "package": "0603",
                              "joints": 2, "stock": 979_259}}), retail)
    p = _generic_10k()
    bad, warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert (bad, warns) == ([], [])
    assert p.sourcing_note == "LCSC C5126214"
    # ineligible rows never even reached a retail check
    assert retail.calls == ["C2930027", "C5126214"]
    assert _check_passive_array_mismatch(_bom(p), tmp_path) == []


def test_kw_walk_all_matches_ineligible_stays_unpinned_not_bounced(
        tmp_path, monkeypatch):
    # Only an array and a wrong-value row match: nothing is pinnable, and
    # nothing bounces to the model — the part stays visibly unpriced.
    _install(monkeypatch, _FakeCatalog(
        by_mpn={"10K 0603": [_KW_10K_0603[3], _KW_10K_0603[4]]}))
    p = _generic_10k()
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert bad == []
    assert p.sourcing_note is None


def test_kw_walk_array_footprint_may_pin_an_array_part(tmp_path, monkeypatch):
    # A genuine resistor-array footprint is NOT a single passive: the array
    # part stays eligible (no over-filtering).
    _install(monkeypatch, _FakeCatalog(by_mpn={"10K": [_KW_10K_0603[3]]}))
    p = BomPart(ref="RN1", value="10k", symbol="Device:R_Network04",
                footprint="Resistor_SMD:R_Array_Convex_4x0603", sheet="A")
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert bad == []
    assert p.sourcing_note == "LCSC C29718"


def test_mpn_family_walk_skips_array_sibling_on_single_passive(
        tmp_path, monkeypatch):
    # Family-prefix broadening must not swap a single passive for its
    # better-stocked array sibling.
    _install(monkeypatch, _FakeCatalog(by_mpn={"YC164": [
        _kw_row("C110924", "YC164-FR-0710KL", "Extended", 5_000_000,
                "10kΩ x4 RES ARRAY 0603x4", joints=8, package="0603x4"),
        _kw_row("C98220", "YC164S", "Extended", 2_000_000,
                "100mW 10kΩ 0603 Chip Resistor"),
    ]}))
    p = BomPart(ref="R1", value="10k", symbol="Device:R", mpn="YC164",
                footprint="Resistor_SMD:R_0603_1608Metric", sheet="A")
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert bad == []
    assert p.sourcing_note == "LCSC C98220"


def test_mpn_only_array_variants_is_an_offender_that_says_why(
        tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog(by_mpn={"YC164": [
        _kw_row("C110924", "YC164-FR-0710KL", "Extended", 5_000_000,
                "10kΩ x4 RES ARRAY 0603x4", joints=8, package="0603x4"),
    ]}))
    p = BomPart(ref="R1", value="10k", symbol="Device:R", mpn="YC164",
                footprint="Resistor_SMD:R_0603_1608Metric", sheet="A")
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert len(bad) == 1 and "multi-element array" in bad[0]
    assert p.sourcing_note is None


# ------------------------------------- §9.26 identity cross-check (KC-9EZE3S)
# A pinned C# that is real and in stock can still be the WRONG PART: KC-9EZE3S
# shipped "fab-ready" with RV1 (trimmer symbol, 3296W THT footprint) pinned to
# C852472 — a 100k 0402 chip resistor (the value matched, the category didn't)
# — and vertical Amphenol BNC footprints pinned to an elbow BNC. The gate must
# compare what the catalog says the part IS against what the BOM claims.

_RV1_CHIP_RESISTOR = _kw_row(
    "C852472", "RT0402BRD07100KL", "Extended", 470_011,
    "-55℃~+155℃ 100kΩ 50V 62.5mW Thin Film Resistor ±0.1% 0402 "
    "Chip Resistor - Surface Mount ROHS", package="0402")

_ELBOW_BNC = _kw_row(
    "C2837587", "KH-BNC50-3511", "Extended", 8_954,
    "-55℃~+155℃ 1 3GHz 50Ω 9.5mm BNC Board Side Elbow Inner Bore 插件 "
    "Coaxial Connectors (RF) ROHS", package="插件")

_REAL_3296W_TRIMMER = _kw_row(
    "C5501675", "3296W-1-104", "Extended", 60_000,
    "100kΩ ±10% 3296W Trimmer Potentiometer Through Hole ROHS",
    package="插件", joints=3)


def test_chip_resistor_pinned_to_trimmer_footprint_is_an_offender(
        tmp_path, monkeypatch):
    # The KC-9EZE3S RV1: identity conflict must bounce even though the part
    # is real, floor-clearing, and retail-stocked.
    _install(monkeypatch, _FakeCatalog(by_lcsc={"C852472": _RV1_CHIP_RESISTOR}))
    p = _part(ref="RV1", mpn="", note="LCSC C852472",
              symbol="Device:R_Potentiometer_Trim",
              footprint="Potentiometer_THT:Potentiometer_Bourns_3296W_Vertical")
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert len(bad) == 1
    assert "C852472" in bad[0]
    assert "fixed resistor" in bad[0] or "chip package" in bad[0]


def test_elbow_bnc_pinned_to_vertical_footprint_is_an_offender(
        tmp_path, monkeypatch):
    # The KC-9EZE3S J1/J2: a right-angle part cannot mount on the vertical
    # Amphenol footprint the BOM chose.
    _install(monkeypatch, _FakeCatalog(by_lcsc={"C2837587": _ELBOW_BNC}))
    p = _part(ref="J1", mpn="", note="LCSC C2837587",
              symbol="Connector:Conn_Coaxial",
              footprint="Connector_Coaxial:BNC_Amphenol_031-5539_Vertical")
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert len(bad) == 1
    assert "C2837587" in bad[0] and "elbow" in bad[0].lower()


def test_matching_trimmer_pin_passes_identity(tmp_path, monkeypatch):
    # The pick the model was actually offered (Bourns 3296W-1-104): same
    # footprint, matching part — no offender.
    _install(monkeypatch,
             _FakeCatalog(by_lcsc={"C5501675": _REAL_3296W_TRIMMER}))
    p = _part(ref="RV1", mpn="", note="LCSC C5501675",
              symbol="Device:R_Potentiometer_Trim",
              footprint="Potentiometer_THT:Potentiometer_Bourns_3296W_Vertical")
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])


def test_tht_pinheader_with_tht_package_passes_identity(tmp_path, monkeypatch):
    # "Vertical" in a footprint name must only conflict with an explicit
    # elbow/right-angle catalog description — a plain THT part is fine.
    row = _kw_row("C124375", "PZ254V-11-40P", "Extended", 500_000,
                  "2.54mm 1x40P male pin header Through Hole ROHS",
                  package="插件", joints=40)
    _install(monkeypatch, _FakeCatalog(by_lcsc={"C124375": row}))
    p = _part(ref="J3", mpn="", note="LCSC C124375",
              symbol="Connector_Generic:Conn_01x40",
              footprint=("Connector_PinHeader_2.54mm:"
                         "PinHeader_1x40_P2.54mm_Vertical"))
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])


def test_bundle_footprint_is_exempt_from_identity_heuristics(
        tmp_path, monkeypatch):
    # A curated bundle's footprint is drawn for its own part; bundle naming
    # (e.g. 'ANT-TH_KH-BNC50-3511') doesn't follow stock conventions, so the
    # heuristics must not read it. Same elbow part, bundle footprint: passes.
    class _Man:
        name = "bnc-pcb-jack"
        sourcing = {"lcsc": "C2837587"}

    class _Loaded:
        manifest = _Man()

    _install(monkeypatch, _FakeCatalog(by_lcsc={"C2837587": _ELBOW_BNC}))
    monkeypatch.setattr(cli_app, "_load_library_parts",
                        lambda root: ([_Loaded()], []))
    p = _part(ref="J1", mpn="", note="LCSC C2837587",
              symbol="bnc-pcb-jack:KH-BNC50-3511",
              footprint="bnc-pcb-jack:ANT-TH_KH-BNC50-3511")
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == ([], [])


def test_identity_conflict_outranks_stock_verdicts(tmp_path, monkeypatch):
    # A wrong part must bounce AS a wrong part, not as a stock problem —
    # here the wrong part is also below the JLC floor, and the identity
    # message must win (the stock message would send the model hunting for
    # a better-stocked wrong part).
    row = dict(_RV1_CHIP_RESISTOR, stock=3)
    _install(monkeypatch, _FakeCatalog(by_lcsc={"C852472": row}))
    p = _part(ref="RV1", mpn="", note="LCSC C852472",
              symbol="Device:R_Potentiometer_Trim",
              footprint="Potentiometer_THT:Potentiometer_Bourns_3296W_Vertical")
    bad, _warns = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert len(bad) == 1 and "drawn for" in bad[0]
