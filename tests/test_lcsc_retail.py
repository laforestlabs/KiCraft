"""Live lcsc.com retail-storefront stock client (parts_library.lcsc_retail).

The wmsc endpoint is monkeypatched at the urllib layer — no network in CI.
Regression source (KC-4AZ7PE): the offline JLC dump said 5-15M stock for 0603
passives the retail storefront had 0 of, so the BOM gate now needs this
second, live inventory reading.
"""
from __future__ import annotations

import io
import json
import urllib.error

import pytest

from kicraft.parts_library import lcsc_retail


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _serve(monkeypatch, payloads, calls):
    """Monkeypatch urlopen: payloads maps C# -> dict (JSON body), bytes (raw
    body), or Exception (raised). Appends each fetched C# to ``calls``."""
    def fake_urlopen(req, timeout=0, context=None):
        cid = req.full_url.rsplit("=", 1)[-1]
        calls.append(cid)
        body = payloads[cid]
        if isinstance(body, Exception):
            raise body
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        return _Resp(body)
    monkeypatch.setattr(lcsc_retail.urllib.request, "urlopen", fake_urlopen)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Fresh disk cache + in-process state per test; retail enabled (the
    suite-wide conftest disables it, which is irrelevant here — enabled() is
    only consulted by callers, not by stock() itself)."""
    monkeypatch.setenv(lcsc_retail.ENV_PATH, str(tmp_path / "retail.json"))
    monkeypatch.setattr(lcsc_retail, "_GAP_S", 0.0)  # no politeness sleeps
    lcsc_retail.clear_cache()
    yield
    lcsc_retail.clear_cache()


def _ok(stock, min_buy=1):
    return {"code": 200, "msg": None,
            "result": {"stockNumber": stock, "minBuyNumber": min_buy}}


# ------------------------------------------------------------ parsing


def test_stock_parses_stock_and_min_buy(monkeypatch):
    calls = []
    _serve(monkeypatch, {"C25804": _ok(0, 100)}, calls)
    info = lcsc_retail.stock("C25804")
    assert info["stock"] == 0 and info["min_buy"] == 100
    assert info["lcsc"] == "C25804" and info["checked_at"]


def test_null_result_means_not_sold_at_retail_not_an_error(monkeypatch):
    # code==200 with result:null is a real answer (part not on the retail
    # storefront), not an outage — existence is the offline dump's job.
    calls = []
    _serve(monkeypatch, {"C42": {"code": 200, "msg": None, "result": None}}, calls)
    info = lcsc_retail.stock("C42")
    assert info["stock"] == 0 and info["min_buy"] == 1


def test_non_200_app_code_is_unavailable_not_dry(monkeypatch):
    # A JSON-wrapped throttle is indistinguishable from a miss; treating it
    # as stock 0 would wrongly bounce a fine part, so it fails OPEN.
    calls = []
    _serve(monkeypatch, {"C42": {"code": 429, "msg": "slow down"}}, calls)
    with pytest.raises(lcsc_retail.RetailUnavailable):
        lcsc_retail.stock("C42")


def test_transport_error_and_non_json_are_unavailable(monkeypatch):
    calls = []
    _serve(monkeypatch, {"C1": urllib.error.URLError("boom")}, calls)
    with pytest.raises(lcsc_retail.RetailUnavailable):
        lcsc_retail.stock("C1")
    lcsc_retail.clear_cache()  # reset the breaker the failure tripped
    _serve(monkeypatch, {"C2": b"<html>WAF says hi</html>"}, calls)
    with pytest.raises(lcsc_retail.RetailUnavailable):
        lcsc_retail.stock("C2")


# ------------------------------------------------------------ caching


def test_disk_ttl_cache_survives_a_fresh_process(monkeypatch):
    # Each stage-commit attempt is a fresh subprocess: simulate one with
    # clear_cache() (drops all in-process state) and require the second read
    # to come from disk, not the network.
    calls = []
    _serve(monkeypatch, {"C7": _ok(5000, 10)}, calls)
    assert lcsc_retail.stock("C7")["stock"] == 5000
    lcsc_retail.clear_cache()
    assert lcsc_retail.stock("C7")["stock"] == 5000
    assert calls == ["C7"]  # exactly one network hit


def test_expired_disk_entry_refetches(monkeypatch, tmp_path):
    calls = []
    _serve(monkeypatch, {"C7": _ok(123)}, calls)
    cache = tmp_path / "retail.json"
    cache.write_text(json.dumps({"C7": {"stock": 999, "min_buy": 1, "ts": 1}}))
    assert lcsc_retail.stock("C7")["stock"] == 123  # stale ts=1 → refetch
    assert calls == ["C7"]


def test_circuit_breaker_fails_fast_after_a_failure(monkeypatch):
    calls = []
    _serve(monkeypatch, {"C1": urllib.error.URLError("waf"), "C2": _ok(1)}, calls)
    with pytest.raises(lcsc_retail.RetailUnavailable):
        lcsc_retail.stock("C1")
    # Breaker open: the next part is refused without a network attempt.
    with pytest.raises(lcsc_retail.RetailUnavailable):
        lcsc_retail.stock("C2")
    assert calls == ["C1"]


def test_corrupt_disk_cache_is_treated_as_empty(monkeypatch, tmp_path):
    (tmp_path / "retail.json").write_text("{not json")
    calls = []
    _serve(monkeypatch, {"C9": _ok(77)}, calls)
    assert lcsc_retail.stock("C9")["stock"] == 77


# ------------------------------------------------------------ kill switch


def test_enabled_defaults_on_and_env_disables(monkeypatch):
    monkeypatch.delenv("KICRAFT_LCSC_RETAIL", raising=False)
    monkeypatch.delenv("KICRAFT_LLM_MODE", raising=False)
    assert lcsc_retail.enabled() is True
    for v in ("0", "off", "no", "false", "OFF"):
        monkeypatch.setenv("KICRAFT_LCSC_RETAIL", v)
        assert lcsc_retail.enabled() is False


def test_enabled_off_under_mock_and_replay(monkeypatch):
    monkeypatch.delenv("KICRAFT_LCSC_RETAIL", raising=False)
    for mode in ("mock", "replay"):
        monkeypatch.setenv("KICRAFT_LLM_MODE", mode)
        assert lcsc_retail.enabled() is False
    monkeypatch.setenv("KICRAFT_LLM_MODE", "live")
    assert lcsc_retail.enabled() is True


# ------------------------------------------------------------ thresholds


def test_in_stock_veto_needs_only_min_buy(monkeypatch):
    calls = []
    _serve(monkeypatch, {"C1": _ok(40, 1)}, calls)
    ok, info = lcsc_retail.in_stock("C1", picky=False)
    assert ok is True and info["stock"] == 40  # 40 >= max(1, 1)


def test_in_stock_picky_requires_the_retail_floor(monkeypatch):
    calls = []
    _serve(monkeypatch, {"C1": _ok(40, 1)}, calls)
    ok, _ = lcsc_retail.in_stock("C1", picky=True)
    assert ok is False  # 40 < max(1, floor 100)


def test_in_stock_min_buy_beats_the_floor_when_larger(monkeypatch):
    calls = []
    # 150 in stock, sold in reels of 200: veto fails (can't buy one reel).
    _serve(monkeypatch, {"C1": _ok(150, 200)}, calls)
    ok, _ = lcsc_retail.in_stock("C1", picky=False)
    assert ok is False


def test_retail_floor_env_override(monkeypatch):
    calls = []
    _serve(monkeypatch, {"C1": _ok(40, 1)}, calls)
    monkeypatch.setenv("KICRAFT_BOM_RETAIL_STOCK_FLOOR", "30")
    ok, _ = lcsc_retail.in_stock("C1", picky=True)
    assert ok is True  # floor lowered to 30
    monkeypatch.setenv("KICRAFT_BOM_RETAIL_STOCK_FLOOR", "junk")
    assert lcsc_retail.retail_floor() == 100  # bad value → default


def test_bare_number_is_normalized_to_c_number(monkeypatch):
    calls = []
    _serve(monkeypatch, {"C25804": _ok(1)}, calls)
    assert lcsc_retail.stock("25804")["lcsc"] == "C25804"
    assert calls == ["C25804"]
