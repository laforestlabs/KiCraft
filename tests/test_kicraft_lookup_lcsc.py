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


def test_lookup_library_hit(capsys):
    # EVQP7A01P is vendored with LCSC C79167 — resolved offline, no network.
    rc, payload = _run(capsys, "lookup-lcsc-id", "EVQP7A01P")
    assert rc == 0
    assert payload["ok"] is True
    assert payload["lcsc"] == "C79167"
    assert payload["source"] == "parts-library"


def test_lookup_search_hit_monkeypatched(capsys, monkeypatch):
    canned = [
        {"lcsc": "C83291", "model": "BMP280", "brand": "Bosch",
         "package": "LGA-8", "description": "pressure sensor"},
        {"lcsc": "C999", "model": "BMP280_3.3", "brand": "x",
         "package": "y", "description": None},
    ]
    monkeypatch.setattr(cli_app, "_search_easyeda_components", lambda kw, **_: canned)
    rc, payload = _run(capsys, "lookup-lcsc-id", "BMP280")
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
