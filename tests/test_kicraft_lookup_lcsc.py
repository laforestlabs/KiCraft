"""Tests for `kicraft lookup-lcsc-id` (MPN -> LCSC resolution).

The library-hit path is offline and deterministic. The JLCPCB-search path is
exercised with a monkeypatched search so the tests never touch the network.
"""
from __future__ import annotations

import json

import pytest

from kicraft.design.cli_app import _pick_lcsc, main


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


def test_lookup_jlcpcb_hit_monkeypatched(capsys, monkeypatch):
    import easyeda2kicad.easyeda.easyeda_api as api_mod

    canned = {
        "results": [
            {"lcsc": "C83291", "model": "BMP280", "brand": "Bosch",
             "package": "LGA-8", "stock": 5000, "type": "Extended"},
            {"lcsc": "C999", "model": "BMP280_3.3", "brand": "x",
             "package": "y", "stock": 0, "type": "Extended"},
        ]
    }
    monkeypatch.setattr(
        api_mod.EasyedaApi, "search_jlcpcb_components",
        lambda self, **kw: canned,
    )
    rc, payload = _run(capsys, "lookup-lcsc-id", "BMP280")
    assert rc == 0
    assert payload["ok"] is True
    assert payload["lcsc"] == "C83291"
    assert payload["source"] == "jlcpcb"


def test_lookup_miss_monkeypatched(capsys, monkeypatch):
    import easyeda2kicad.easyeda.easyeda_api as api_mod

    monkeypatch.setattr(
        api_mod.EasyedaApi, "search_jlcpcb_components",
        lambda self, **kw: {"results": []},
    )
    rc, payload = _run(capsys, "lookup-lcsc-id", "DEFINITELY_NOT_A_PART_X1")
    assert rc == 4
    assert payload["ok"] is False
    assert payload["candidates"] == []
