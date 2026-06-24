"""Tests for `kicraft lookup-lcsc-id` (MPN -> LCSC resolution).

The explicit-id and library-hit paths are offline and deterministic. The
easyeda-search path is exercised with a monkeypatched search so the tests
never touch the network.
"""
from __future__ import annotations

import json

import pytest

import kicraft.design.cli_app as cli_app
from kicraft.design.cli_app import _parse_easyeda_search, _pick_lcsc, main


def _run(capsys: pytest.CaptureFixture, *argv: str) -> tuple[int, dict]:
    rc = main(list(argv))
    out = capsys.readouterr().out
    return rc, (json.loads(out) if out.strip() else {})


# ---------- _pick_lcsc selection logic (pure) ----------


def test_pick_lcsc_prefers_exact_model():
    results = [
        {"lcsc": "C1", "model": "BMP280_3.3", "stock": 99, "type": "Extended"},
        {"lcsc": "C2", "model": "BMP280", "stock": 1, "type": "Extended"},
    ]
    assert _pick_lcsc("bmp280", results)["lcsc"] == "C2"  # exact wins over stock


def test_pick_lcsc_exact_tie_breaks_on_stock_then_basic():
    results = [
        {"lcsc": "C1", "model": "X", "stock": 5, "type": "Extended"},
        {"lcsc": "C2", "model": "X", "stock": 50, "type": "Extended"},
    ]
    assert _pick_lcsc("X", results)["lcsc"] == "C2"


def test_pick_lcsc_single_result_taken():
    assert _pick_lcsc("whatever", [{"lcsc": "C9", "model": "OTHER"}])["lcsc"] == "C9"


def test_pick_lcsc_ambiguous_returns_none():
    results = [{"lcsc": "C1", "model": "A"}, {"lcsc": "C2", "model": "B"}]
    assert _pick_lcsc("Z", results) is None


# ---------- CLI ----------


@pytest.fixture(autouse=True)
def _no_local_jlc_catalog(tmp_path, monkeypatch):
    """Point the offline JLC catalog at a missing file so these tests
    exercise the explicit-id / parts-library / easyeda paths regardless of
    whether the host has the real 5 GB catalog installed. Also isolate the
    persistent MPN->LCSC cache to a per-test temp file: otherwise the host's
    real ~/.kicraft/mpn_cache.json short-circuits the tier under test (a
    previously-resolved MPN returns 'mpn-cache' instead of exercising
    parts-library/easyeda), and a test resolution would pollute the real cache."""
    monkeypatch.setenv("KICRAFT_JLCPARTS_DB", str(tmp_path / "absent.sqlite3"))
    from kicraft.parts_library import mpn_cache
    monkeypatch.setenv(mpn_cache.ENV_PATH, str(tmp_path / "mpn_cache.json"))


def test_lookup_library_hit(capsys):
    # EVQP7A01P is vendored with LCSC C79167 — resolved offline, no network.
    rc, payload = _run(capsys, "lookup-lcsc-id", "EVQP7A01P")
    assert rc == 0
    assert payload["ok"] is True
    assert payload["lcsc"] == "C79167"
    assert payload["source"] == "parts-library"


def test_lookup_search_hit_monkeypatched(capsys, monkeypatch):
    # A synthetic MPN that is NOT in the vendored parts-library, so resolution
    # actually falls through to the (monkeypatched) easyeda search under test.
    canned = [
        {"lcsc": "C83291", "model": "ICTESTPRESSURE1", "brand": "Bosch",
         "package": "LGA-8", "description": "pressure sensor"},
        {"lcsc": "C999", "model": "ICTESTPRESSURE1_3.3", "brand": "x",
         "package": "y", "description": None},
    ]
    monkeypatch.setattr(cli_app, "_search_easyeda_components", lambda kw, **_: canned)
    rc, payload = _run(capsys, "lookup-lcsc-id", "ICTESTPRESSURE1")
    assert rc == 0
    assert payload["ok"] is True
    assert payload["lcsc"] == "C83291"
    assert payload["source"] == "easyeda"


def test_lookup_miss_monkeypatched(capsys, monkeypatch):
    monkeypatch.setattr(cli_app, "_search_easyeda_components", lambda kw, **_: [])
    rc, payload = _run(capsys, "lookup-lcsc-id", "DEFINITELY_NOT_A_PART_X1")
    assert rc == 4
    assert payload["ok"] is False
    assert payload["candidates"] == []
    # A genuine no-match caps the retry burn instead of inviting more variants.
    assert "at most ONCE" in payload["hint"]


def test_lookup_backend_unreachable_tells_model_to_stop(capsys, monkeypatch):
    monkeypatch.setattr(cli_app, "_search_easyeda_components", lambda kw, **_: None)
    rc, payload = _run(capsys, "lookup-lcsc-id", "VL53L1X")
    assert rc == 4
    assert payload["ok"] is False
    assert "unreachable" in payload["error"]
    assert "Do NOT retry" in payload["hint"]


def test_lookup_explicit_id_short_circuits_offline(capsys, monkeypatch):
    # Any network call would blow up — explicit ids must resolve without one.
    def _boom(kw, **_):
        raise AssertionError("network search must not run for an explicit id")
    monkeypatch.setattr(cli_app, "_search_easyeda_components", _boom)

    rc, payload = _run(capsys, "lookup-lcsc-id", "C7386355")
    assert rc == 0 and payload["lcsc"] == "C7386355"
    assert payload["source"] == "explicit-id"

    rc, payload = _run(
        capsys, "lookup-lcsc-id",
        "https://www.lcsc.com/product-detail/C7386355.html?s_z=n_q_t_VL53L")
    assert rc == 0 and payload["lcsc"] == "C7386355"


def test_lookup_mpn_embedding_c_digits_is_not_an_id(capsys, monkeypatch):
    # 'C8051F320' is a Silicon Labs MCU, not LCSC id C8051.
    monkeypatch.setattr(cli_app, "_search_easyeda_components", lambda kw, **_: [])
    rc, payload = _run(capsys, "lookup-lcsc-id", "C8051F320")
    assert rc == 4
    assert payload["ok"] is False


# ---------- offline JLC catalog path ----------


def _mk_catalog(tmp_path, monkeypatch, rows):
    import sqlite3
    db = tmp_path / "jlc.sqlite3"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE jlc_components (
        lcsc INTEGER PRIMARY KEY, mfr TEXT, package TEXT, manufacturer TEXT,
        library_type TEXT, stock INTEGER, price TEXT, description TEXT)""")
    con.executemany("INSERT INTO jlc_components VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    monkeypatch.setenv("KICRAFT_JLCPARTS_DB", str(db))


def test_lookup_jlcparts_exact_hit_no_network(capsys, monkeypatch, tmp_path):
    _mk_catalog(tmp_path, monkeypatch, [
        (190004, "VL53L1CXV0FY/1", "LGA-12(2.5x4.9)", "STMicroelectronics",
         "expand", 5640, "1-9:4.817,1000-:3.1586", "4m I2C ToF sensor ROHS"),
    ])

    def _boom(kw, **_):
        raise AssertionError("network search must not run when the catalog hits")
    monkeypatch.setattr(cli_app, "_search_easyeda_components", _boom)

    rc, payload = _run(capsys, "lookup-lcsc-id", "VL53L1CXV0FY/1")
    assert rc == 0
    assert payload["lcsc"] == "C190004" and payload["source"] == "jlcparts"
    assert payload["match"]["stock"] == 5640
    assert payload["match"]["type"] == "Extended"
    assert payload["match"]["price"] == 4.817


def test_lookup_jlcparts_ambiguous_lists_candidates_with_stock(capsys, monkeypatch,
                                                               tmp_path):
    _mk_catalog(tmp_path, monkeypatch, [
        (190004, "VL53L1CXV0FY/1", "LGA-12", "ST", "expand", 5640, "1-:4.8", "ToF"),
        (2970716, "VL53L1CBV0FY/1", "SMD-12P", "ST", "expand", 2963, "1-:5.0", "ToF"),
    ])
    rc, payload = _run(capsys, "lookup-lcsc-id", "VL53L1C")
    assert rc == 4
    # In-stock-first candidate list; the model is told to pick one.
    assert [c["lcsc"] for c in payload["candidates"]] == ["C190004", "C2970716"]
    assert "add_part_from_lcsc" in payload["hint"]


def test_lookup_jlcparts_miss_falls_through_to_easyeda(capsys, monkeypatch, tmp_path):
    _mk_catalog(tmp_path, monkeypatch, [
        (25744, "RC0805FR-07100KL", "0805", "YAGEO", "base", 1, "1-:0.0041", "res"),
    ])
    canned = [{"lcsc": "C83291", "model": "ICTESTPRESSURE1", "brand": "Bosch",
               "package": "LGA-8", "description": None}]
    monkeypatch.setattr(cli_app, "_search_easyeda_components", lambda kw, **_: canned)
    # Synthetic, unvendored MPN: misses the catalog and the parts-library, so it
    # exercises the jlcparts -> easyeda fall-through this test is named for.
    rc, payload = _run(capsys, "lookup-lcsc-id", "ICTESTPRESSURE1")
    assert rc == 0
    assert payload["lcsc"] == "C83291" and payload["source"] == "easyeda"


def test_parse_easyeda_search_flattens_and_dedupes():
    payload = {
        "success": True,
        "result": {"lists": {
            "lcsc": [{
                "title": "VL53L1X",
                "description": "ToF sensor",
                "lcsc": {"id": 1, "number": "C2924337"},
                "szlcsc": {"id": 1, "number": "C2924337"},
                "dataStr": {"head": {"c_para": {
                    "name": "VL53L1X", "package": "LGA-12_L4.8-W2.4-P0.8-RB"}}},
            }],
            "SMT": [
                {"title": "VL53L1X", "lcsc": {"number": "C2924337"}},  # dupe
                {"title": "VL53L0X", "szlcsc": {"number": "C94274"}, "dataStr": "raw"},
            ],
        }},
    }
    rows = _parse_easyeda_search(payload)
    assert [r["lcsc"] for r in rows] == ["C2924337", "C94274"]
    assert rows[0]["package"] == "LGA-12_L4.8-W2.4-P0.8-RB"
    assert rows[0]["description"] == "ToF sensor"
    assert rows[1]["package"] is None  # non-dict dataStr tolerated


def test_pick_lcsc_zero_stock_exact_defers_to_in_stock_candidates():
    # "VL53L1X" exact-matches a zero-stock placeholder row; the orderable
    # real MPN (in stock) must surface as a candidate instead of losing.
    results = [
        {"lcsc": "C2924337", "model": "VL53L1X", "stock": 0, "type": "Extended"},
        {"lcsc": "C190004", "model": "VL53L1CXV0FY/1", "stock": 5640, "type": "Extended"},
    ]
    assert _pick_lcsc("VL53L1X", results) is None
    # ...but with nothing else in stock, the exact match still wins.
    results[1]["stock"] = 0
    assert _pick_lcsc("VL53L1X", results)["lcsc"] == "C2924337"
