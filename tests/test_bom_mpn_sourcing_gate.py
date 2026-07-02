"""§9.26 — _resolve_bom_mpn_sourcing: every BOM part that names an MPN must be
a real, orderable LCSC part.

Regression source (KC-T6ERHM): J1/J2 shipped an out-of-stock terminal block and
D1/L1 carried MPNs no pipeline stage ever verified — MPN strings on
stock-symbol parts were unchecked LLM prose, and the fab BOM's LCSC column
shipped blank. The gate verifies against the offline jlcparts catalog and
auto-pins confident matches into sourcing_note (where fab_export reads the C#).
"""
from __future__ import annotations

import kicraft.design.cli_app as cli_app
from kicraft.design.cli_app import _resolve_bom_mpn_sourcing
from kicraft.design.models import BOM, BomPart


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

    def __init__(self, by_mpn=None, by_lcsc=None, up=True):
        self.by_mpn = by_mpn or {}
        self.by_lcsc = by_lcsc or {}
        self.up = up

    def available(self):
        return self.up

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


def _install(monkeypatch, catalog):
    monkeypatch.setattr(cli_app, "jlcparts", catalog)
    # No library bundles in play: parts resolve purely by MPN in these tests.
    monkeypatch.setattr(cli_app, "_load_library_parts", lambda root: ([], []))


def test_real_mpn_is_auto_pinned_into_sourcing_note(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog(by_mpn={
        "SS34": [{"lcsc": "C8678", "model": "SS34", "stock": 3941831}]}))
    p = _part()
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == []
    assert p.sourcing_note == "LCSC C8678"


def test_hallucinated_mpn_is_an_offender(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog())
    p = _part(mpn="TOTALLYFAKE-99")
    bad = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert len(bad) == 1 and "TOTALLYFAKE-99" in bad[0] and "not found" in bad[0]
    assert p.sourcing_note is None  # never pin what we couldn't verify


def test_low_stock_is_an_offender(tmp_path, monkeypatch):
    # The KC-T6ERHM L1: real Bourns part, 49 units in the snapshot.
    _install(monkeypatch, _FakeCatalog(by_mpn={
        "SRR1280-100M": [{"lcsc": "C2041557", "model": "SRR1280-100M", "stock": 49}]}))
    p = _part(ref="L1", mpn="SRR1280-100M", symbol="Device:L",
              footprint="Inductor_SMD:L_12x12mm_H4.5mm")
    bad = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert len(bad) == 1 and "49 in stock" in bad[0]
    assert p.sourcing_note is None


def test_stock_floor_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_BOM_STOCK_FLOOR", "10")
    _install(monkeypatch, _FakeCatalog(by_mpn={
        "SRR1280-100M": [{"lcsc": "C2041557", "model": "SRR1280-100M", "stock": 49}]}))
    p = _part(ref="L1", mpn="SRR1280-100M")
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == []
    assert p.sourcing_note == "LCSC C2041557"


def test_family_prefix_match_pins_the_orderable_variant(tmp_path, monkeypatch):
    # Stock-ordered results: the first model extending the family name wins.
    _install(monkeypatch, _FakeCatalog(by_mpn={
        "SS34": [{"lcsc": "C407539", "model": "SS34F", "stock": 900000},
                 {"lcsc": "C8678", "model": "SS34", "stock": 100}]}))
    p = _part()
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == []
    assert p.sourcing_note == "LCSC C407539"


def test_explicit_sourcing_note_lcsc_is_verified_not_searched(tmp_path, monkeypatch):
    cat = _FakeCatalog(by_lcsc={"C9864": {"lcsc": "C9864", "model": "TPS5430DDAR",
                                          "stock": 148617}})
    _install(monkeypatch, cat)
    p = _part(ref="U1", mpn="TPS5430DDAR", note="LCSC C9864 (buck IC)")
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == []
    assert p.sourcing_note == "LCSC C9864 (buck IC)"  # untouched


def test_fabricated_sourcing_note_lcsc_is_an_offender(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog())
    p = _part(ref="U1", mpn="TPS5430DDAR", note="LCSC C99999999")
    bad = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
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
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == []
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
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == []
    assert p.sourcing_note == "LCSC C1590"


def test_keyword_pick_prefers_basic_over_stocked_extended(tmp_path, monkeypatch):
    _install(monkeypatch, _kw_catalog("1k 0603", [
        {"lcsc": "CEXT", "model": "CHURN", "stock": 2_000_000, "type": "Extended"},
        {"lcsc": "CBAS", "model": "0603WAF1001T5E", "stock": 900_000,
         "type": "Basic"}]))
    p = _part(ref="R2", mpn=None, symbol="Device:R",
              footprint="Resistor_SMD:R_0603_1608Metric")
    p.value = "1k"
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == []
    assert p.sourcing_note == "LCSC CBAS"


def test_keyword_qualifiers_relax_on_miss(tmp_path, monkeypatch):
    # "0.1uF 25V X7R 0603" matches nothing; the relaxed "0.1uF 0603" must hit.
    _install(monkeypatch, _kw_catalog("0.1uF 0603", [
        {"lcsc": "C14663", "model": "CC0603KRX7R8BB104", "stock": 3_000_000,
         "type": "Basic"}]))
    p = _part(ref="C2", mpn=None, symbol="Device:C",
              footprint="Capacitor_SMD:C_0603_1608Metric")
    p.value = "0.1uF 25V X7R"
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == []
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
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == []
    assert p.sourcing_note == "LCSC C492401"


def test_test_points_are_not_sourcing_offenders(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog())
    p = _part(ref="TP1", mpn=None, symbol="Connector:TestPoint",
              footprint="TestPoint:TestPoint_Pad_D1.5mm")
    p.value = "TestPoint"
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == []
    assert p.sourcing_note is None


def test_part_with_nothing_searchable_is_an_offender(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog())
    p = _part(ref="X1", mpn=None, symbol="foo:bar", footprint="foo:bar")
    p.value = ""
    bad = _resolve_bom_mpn_sourcing(_bom(p), tmp_path)
    assert len(bad) == 1 and "X1" in bad[0] and "unsourceable" in bad[0]


def test_catalog_unavailable_fails_open(tmp_path, monkeypatch):
    _install(monkeypatch, _FakeCatalog(up=False))
    p = _part(mpn="TOTALLYFAKE-99")
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == []
    assert p.sourcing_note is None


def test_library_bundle_parts_are_skipped(tmp_path, monkeypatch):
    """Parts whose symbol resolves to a library bundle with a manifest LCSC are
    the _unresolved_lcsc gate's territory — no search, no pin here."""

    class _Man:
        name = "tps5430"
        sourcing = {"lcsc": "C9864"}

    class _Loaded:
        manifest = _Man()

    monkeypatch.setattr(cli_app, "jlcparts", _FakeCatalog())  # search would MISS
    monkeypatch.setattr(cli_app, "_load_library_parts",
                        lambda root: ([_Loaded()], []))
    p = _part(ref="U1", mpn="TPS5430DDAR", symbol="tps5430:TPS5430DDAR",
              footprint="tps5430:ESOP-8")
    assert _resolve_bom_mpn_sourcing(_bom(p), tmp_path) == []
    assert p.sourcing_note is None
