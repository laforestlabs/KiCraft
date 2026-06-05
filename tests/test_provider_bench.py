"""Offline tests for the OpenRouter provider benchmark (kicraft.cli.provider_bench).

Network is faked (a streaming SSE stand-in + a fake /endpoints GET) and the clock
is monkeypatched, so nothing here spends tokens or depends on wall time. Mirrors the
fakes in tests/test_client_provider.py.

Covers:
- the request is pinned to exactly one backend (order + allow_fallbacks False) and
  carries the production cache breakpoint, with no internal keys leaking,
- timed_stream parses TTFT / throughput / cost / cache from a fake stream,
- bench_provider aggregates cold+warm and applies the advertised-price fallback,
- recommend ranks by measured warm cost (switch to a cheaper caching backend, but
  stay when only the current pin caches) and emits paste-ready env config,
- discover_endpoints parses, filters, and sorts cheapest-first,
- the plot renders to a PNG headlessly.
"""
from __future__ import annotations

import json

import pytest

import kicraft.cli.provider_bench as pb
from kicraft.server.config import Settings


# ---- fakes ----------------------------------------------------------------

class _Clock:
    """Deterministic perf_counter: +0.5s per call, so every timed call sees
    ttft=0.5s and total=1.0s regardless of how many calls precede it."""
    def __init__(self, step=0.5):
        self.t = 0.0
        self.step = step

    def __call__(self):
        v = self.t
        self.t += self.step
        return v


class _FakeStream:
    """Minimal stand-in for requests' streaming Response (context manager)."""
    def __init__(self, chunks):
        self._lines = [f"data: {json.dumps(c)}" for c in chunks] + ["data: [DONE]"]

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeGet:
    def __init__(self, endpoints):
        self._endpoints = endpoints

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": {"endpoints": self._endpoints}}


class _FakeHTTP:
    """Programmable http: pops a chunk batch per post(); returns endpoints on get()."""
    def __init__(self, post_batches=None, endpoints=None):
        self._posts = list(post_batches or [])
        self._endpoints = endpoints
        self.post_payloads = []

    def post(self, url, headers=None, json=None, timeout=None, stream=None):
        self.post_payloads.append(json)
        return _FakeStream(self._posts.pop(0))

    def get(self, url, headers=None, timeout=None):
        return _FakeGet(self._endpoints)


def _usage(cached=0, cost=0.001, intok=3500, outtok=40, provider="DeepSeek"):
    return {"provider": provider,
            "usage": {"prompt_tokens": intok, "completion_tokens": outtok, "cost": cost,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


def _completion(cached=0, cost=0.001, intok=3500, outtok=40, provider="DeepSeek"):
    """A full fake stream: one content token, a finish, then the usage chunk."""
    return [{"choices": [{"delta": {"content": "{\"a\":1}"}}]},
            {"choices": [{"finish_reason": "stop", "delta": {}}]},
            _usage(cached=cached, cost=cost, intok=intok, outtok=outtok, provider=provider)]


# ---- request shaping ------------------------------------------------------

def test_build_request_pins_single_provider():
    s = Settings(api_key="k", provider_order=["deepseek"])
    p = pb.build_request(s, "deepinfra/fp4", pb.build_messages(300), 128)
    assert p["provider"] == {"order": ["deepinfra/fp4"], "allow_fallbacks": False}
    assert p["model"] == s.model and p["stream"] is True and p["max_tokens"] == 128
    # production cache breakpoint applied to the system message
    blk = p["messages"][0]["content"]
    assert isinstance(blk, list) and blk[0]["cache_control"] == {"type": "ephemeral"}
    assert not any(k.startswith("_") for k in p)


def test_build_messages_is_sized_and_identical():
    a = pb.build_messages(2000)
    b = pb.build_messages(2000)
    assert a == b                                    # deterministic
    assert len(a[0]["content"]) >= 2000 * 4          # padded to ~tokens*4 chars
    assert a[1]["role"] == "user"


# ---- timed_stream ---------------------------------------------------------

def test_timed_stream_parses_metrics(monkeypatch):
    monkeypatch.setattr(pb.time, "perf_counter", _Clock())
    http = _FakeHTTP(post_batches=[_completion(cached=3000, cost=0.0004,
                                               intok=3500, outtok=40)])
    s = Settings(api_key="k", provider_order=["deepseek"])
    m = pb.timed_stream(s, pb.build_request(s, "deepseek", pb.build_messages(500), 200),
                        http=http)
    assert m.ok and m.ttft_s == 0.5 and m.total_s == 1.0
    assert m.prompt_tokens == 3500 and m.completion_tokens == 40
    assert m.cached_tokens == 3000 and m.cost_usd == 0.0004
    assert m.provider_echo == "DeepSeek" and m.finish_reason == "stop"
    assert m.cache_hit_pct() == pytest.approx(3000 / 3500 * 100)
    assert m.gen_tps == pytest.approx(40 / 0.5)      # out / (total - ttft)


def test_timed_stream_records_error_not_raise():
    class _Boom:
        def post(self, *a, **k):
            raise RuntimeError("backend down")
    s = Settings(api_key="k", provider_order=["deepseek"])
    m = pb.timed_stream(s, pb.build_request(s, "x", pb.build_messages(200), 50),
                        http=_Boom())
    assert m.ok is False and "RuntimeError" in m.error


# ---- bench_provider -------------------------------------------------------

def test_bench_provider_cold_warm_and_price_fallback(monkeypatch):
    monkeypatch.setattr(pb.time, "perf_counter", _Clock())
    monkeypatch.setattr(pb.time, "sleep", lambda *a: None)
    ep = pb.Endpoint("DeepSeek", "deepseek", price_prompt=0.14, price_completion=0.28)
    # both calls omit cost (0.0) -> advertised-price fallback; warm is cache-heavy
    http = _FakeHTTP(post_batches=[_completion(cached=0, cost=0.0),
                                   _completion(cached=3200, cost=0.0)])
    s = Settings(api_key="k", provider_order=["deepseek"])
    pr = pb.bench_provider(s, ep, pb.build_messages(500), 200, repeat=2, http=http)
    assert len(pr.calls) == 2 and pr.ok and pr.pinned_ok
    assert pr.warm_cache_hit_pct == pytest.approx(3200 / 3500 * 100)
    assert pr.mean_ttft_s == 0.5
    expected = (3500 * 0.14 + 40 * 0.28) / 1_000_000.0   # advertised-price fallback
    assert pr.warm.cost_usd == pytest.approx(expected)
    assert pr.total_cost_usd == pytest.approx(2 * expected)


def test_warm_picks_most_cached_call_not_last():
    ep = pb.Endpoint("X", "x", 0.14, 0.28)
    cold = pb.Measurement(ok=True, cost_usd=0.0005, prompt_tokens=3500, cached_tokens=0)
    hot = pb.Measurement(ok=True, cost_usd=0.0001, prompt_tokens=3500, cached_tokens=3300)
    cool = pb.Measurement(ok=True, cost_usd=0.0005, prompt_tokens=3500, cached_tokens=0)
    pr = pb.ProviderResult(endpoint=ep, calls=[cold, hot, cool])
    assert pr.warm is hot                        # steady-state = best-cached warm, not last
    assert pr.warm_cost() == 0.0001


# ---- recommend ------------------------------------------------------------

def _mk_result(name, tag, pp, pc, warm_cost, cache_pct, ttft=1.0, tps=40.0,
               pinned=True, quant=None):
    intok = 3500
    m = pb.Measurement(ok=True, ttft_s=ttft, total_s=ttft + 1.0, prompt_tokens=intok,
                       completion_tokens=40, cached_tokens=int(cache_pct / 100 * intok),
                       cost_usd=warm_cost, provider_echo=name if pinned else "Other",
                       finish_reason="stop", gen_tps=tps)
    return pb.ProviderResult(endpoint=pb.Endpoint(name, tag, pp, pc, quant), calls=[m],
                             pinned_ok=pinned, warm_cache_hit_pct=cache_pct,
                             mean_ttft_s=ttft, mean_gen_tps=tps, total_cost_usd=warm_cost)


def _mk_failed(name, tag, pp, pc, err="HTTPError: 404 Client Error: Not Found"):
    m = pb.Measurement(ok=False, error=err)
    return pb.ProviderResult(endpoint=pb.Endpoint(name, tag, pp, pc), calls=[m],
                             pinned_ok=False)


def test_recommend_switches_to_cheaper_caching_backend():
    s = Settings(api_key="k", provider_order=["deepseek"])                                  # provider_order ["deepseek"]
    cur = _mk_result("DeepSeek", "deepseek", 0.14, 0.28, warm_cost=0.0005, cache_pct=70)
    cheap = _mk_result("Baidu", "baidu/fp8", 0.098, 0.197, warm_cost=0.0003, cache_pct=65)
    rec = pb.recommend([cur, cheap], s)
    assert rec["pick"].endpoint.tag == "baidu/fp8" and rec["stay"] is False
    assert rec["saving_pct"] == pytest.approx(100 * (0.0005 - 0.0003) / 0.0005)
    assert rec["env_lines"][0] == "KICRAFT_PROVIDER_ORDER=baidu/fp8"
    assert any(l.startswith("KICRAFT_MAX_PRICE_PROMPT=") for l in rec["env_lines"])


def test_recommend_stays_when_only_pin_caches():
    s = Settings(api_key="k", provider_order=["deepseek"])
    # cheaper sticker but no caching -> higher *measured* warm cost than the pin
    cur = _mk_result("DeepSeek", "deepseek", 0.14, 0.28, warm_cost=0.0003, cache_pct=70)
    nocache = _mk_result("Baidu", "baidu/fp8", 0.098, 0.197, warm_cost=0.0009, cache_pct=0)
    rec = pb.recommend([cur, nocache], s)
    assert rec["pick"].endpoint.tag == "deepseek" and rec["stay"] is True


def test_recommend_ignores_unpinned_results():
    s = Settings(api_key="k", provider_order=["deepseek"])
    cur = _mk_result("DeepSeek", "deepseek", 0.14, 0.28, 0.0005, 70)
    # cheapest + caching but the pin never actually served it -> not eligible
    ghost = _mk_result("Ghost", "ghost", 0.05, 0.10, 0.0001, 80, pinned=False)
    rec = pb.recommend([cur, ghost], s)
    assert rec["pick"].endpoint.tag == "deepseek"


def test_recommend_downranks_low_quant_unless_allowed():
    s = Settings(api_key="k", provider_order=["deepseek"])
    pin = _mk_result("DeepSeek", "deepseek", 0.14, 0.28, 0.0005, 70)
    fp8 = _mk_result("Novita", "novita/fp8", 0.14, 0.28, 0.00018, 92, quant="fp8")
    fp4 = _mk_result("DeepInfra", "deepinfra/fp4", 0.10, 0.20, 0.00010, 92, quant="fp4")
    # default: fp4 is cheapest warm but down-ranked -> pick the fp8 backend,
    # and fp4 is surfaced as the cheaper alternative.
    rec = pb.recommend([pin, fp8, fp4], s)
    assert rec["pick"].endpoint.tag == "novita/fp8"
    assert rec["low_quant_alt"].endpoint.tag == "deepinfra/fp4"
    # opt in -> fp4 becomes the primary pick.
    rec2 = pb.recommend([pin, fp8, fp4], s, allow_low_quant=True)
    assert rec2["pick"].endpoint.tag == "deepinfra/fp4"


def test_report_warns_when_current_pin_failed_to_resolve():
    s = Settings(api_key="k", provider_order=["deepseek"])                                   # provider_order ["deepseek"]
    dead = _mk_failed("DeepSeek", "deepseek", 0.14, 0.28)       # 404 when pinned
    alt = _mk_result("Novita", "novita/fp8", 0.14, 0.28, 0.00018, 92, quant="fp8")
    rec = pb.recommend([dead, alt], s)
    assert rec["pick"].endpoint.tag == "novita/fp8"
    report = pb.format_report([dead, alt], rec, s, 0.01,
                              {"prompt_tokens": 3500, "max_tokens": 200, "repeat": 2})
    assert "did NOT resolve" in report and "silently routing" in report


# ---- discover_endpoints ---------------------------------------------------

def _endpoints_json():
    return [
        {"provider_name": "DeepSeek", "tag": "deepseek",
         "pricing": {"prompt": "0.00000014", "completion": "0.00000028"},
         "quantization": "unknown", "context_length": 1000},
        {"provider_name": "Baidu", "tag": "baidu/fp8",
         "pricing": {"prompt": "0.0000000983", "completion": "0.0000001966"},
         "quantization": "fp8", "context_length": 2000},
    ]


def test_discover_endpoints_parses_sorts_filters():
    s = Settings(api_key="k", provider_order=["deepseek"])
    http = _FakeHTTP(endpoints=_endpoints_json())
    eps = pb.discover_endpoints(s, http=http)
    assert [e.provider_name for e in eps] == ["Baidu", "DeepSeek"]   # cheapest first
    assert eps[0].price_prompt == pytest.approx(0.0983)              # per-token -> $/Mtok
    only = pb.discover_endpoints(s, providers=["deepseek"], http=http)
    assert [e.tag for e in only] == ["deepseek"]
    top1 = pb.discover_endpoints(s, top_n=1, http=http)
    assert len(top1) == 1 and top1[0].provider_name == "Baidu"


# ---- plot -----------------------------------------------------------------

def test_render_plot_writes_png(tmp_path):
    s = Settings(api_key="k", provider_order=["deepseek"])
    cur = _mk_result("DeepSeek", "deepseek", 0.14, 0.28, 0.0005, 70)
    cheap = _mk_result("Baidu", "baidu/fp8", 0.098, 0.197, 0.0003, 65)
    rec = pb.recommend([cur, cheap], s)
    out = tmp_path / "bench.png"
    pb.render_plot([cur, cheap], rec, s, str(out))
    assert out.exists() and out.stat().st_size > 0
