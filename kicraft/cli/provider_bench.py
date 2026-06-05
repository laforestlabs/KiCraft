#!/usr/bin/env python3
"""Benchmark the OpenRouter backends that serve KiCraft's default model.

A single OpenRouter model id (e.g. ``deepseek/deepseek-v4-flash``) is served by
many backends at different prices, latencies, throughputs, and caching behaviour.
The web server pins a ``provider_order`` of verified caching backends
(``kicraft/server/config.py``); this tool is how that set is measured and chosen
head to head, and how a re-pin is justified.

For every backend that serves the resolved model it sends the *same* sized prompt
twice (a cold call to prime the cache, then a warm call) pinned to that one
backend, and records:

  * cost        - OpenRouter's real billed ``usage.cost`` (cache-discounted on the
                  warm call), with the backend's advertised price as a fallback
  * TTFT        - wall-clock time to the first streamed token (latency)
  * throughput  - generated tokens / second over the streaming body
  * cache hit   - ``cached_tokens / prompt_tokens`` on the warm call

It then prints a comparison table + a recommendation (stay on the current pin vs
switch, ranked by *measured warm cost* so a backend that caches well beats a
cheaper one that does not), writes the raw results to JSON, and renders a
multi-panel comparison plot.

This spends real money, but very little: the default sized prompt + 200 output
tokens over ~14 backends x 2 calls is ~2-3 US cents. A ``--max-spend`` ceiling
aborts the sweep before it can exceed the budget, and nothing is written to the
production spend ledger (so ``web-cost-report`` stays clean).

    provider-bench                              # all backends of the default model
    provider-bench --providers deepseek,baidu   # just these two
    provider-bench --top-n 5 --max-tokens 120    # 5 cheapest, smaller output
    provider-bench --no-plot --json              # machine-readable, no PNG
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

from kicraft.server.client import CappedOpenRouterClient, estimate_cost
from kicraft.server.config import Settings

# Recommendation tuning.
_CACHE_HIT_THRESHOLD = 20.0   # warm hit% a backend must clear to count as "caching"
_PRICE_HEADROOM = 1.30        # max_price ceiling = chosen backend price x this
_TTFT_TOLERANCE_S = 4.0       # a pick must not be slower than this on mean TTFT
# Aggressive quantizations that can degrade reasoning quality. Excluded from the
# *primary* pick for a design tool unless --allow-low-quant is passed; still shown
# as the cheapest-warm alternative so the saving is visible.
_LOW_QUANT = {"fp4", "int4", "nf4"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Endpoint:
    """One backend serving the model, from /models/{id}/endpoints."""
    provider_name: str            # human name, e.g. "DeepSeek"
    tag: str                      # routing slug, e.g. "deepseek" or "deepinfra/fp4"
    price_prompt: float           # advertised $/Mtok input
    price_completion: float       # advertised $/Mtok output
    quantization: str | None = None
    context_length: int | None = None

    def label(self) -> str:
        q = f" ({self.quantization})" if self.quantization and self.quantization != "unknown" else ""
        return f"{self.provider_name}{q}"


@dataclass
class Measurement:
    """One completion's timing + usage."""
    ok: bool
    error: str | None = None
    ttft_s: float | None = None
    total_s: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    provider_echo: str | None = None   # provider OpenRouter says served the call
    finish_reason: str | None = None
    gen_tps: float | None = None       # completion tokens / streaming seconds

    def cache_hit_pct(self) -> float:
        return (100.0 * self.cached_tokens / self.prompt_tokens) if self.prompt_tokens else 0.0


@dataclass
class ProviderResult:
    """Aggregated cold+warm measurements for one backend."""
    endpoint: Endpoint
    calls: list[Measurement] = field(default_factory=list)
    pinned_ok: bool = False            # did OpenRouter actually serve from the pinned backend?
    warm_cache_hit_pct: float = 0.0
    mean_ttft_s: float | None = None
    mean_gen_tps: float | None = None
    total_cost_usd: float = 0.0

    @property
    def cold(self) -> Measurement | None:
        return self.calls[0] if self.calls else None

    @property
    def warm(self) -> Measurement | None:
        """The steady-state (cache-hot) call: the most-cached successful call after
        the cold one, tie-broken by lowest cost. Prompt caching is sticky once it
        engages but can take more than one warm call to populate, so the *best*
        warm call characterizes steady state more robustly than the last one.
        Falls back to any successful call when repeat==1."""
        warms = [m for m in self.calls[1:] if m.ok]
        if warms:
            return max(warms, key=lambda m: (m.cache_hit_pct(), -m.cost_usd))
        for m in reversed(self.calls):
            if m.ok:
                return m
        return None

    @property
    def ok(self) -> bool:
        return any(c.ok for c in self.calls)

    def warm_cost(self) -> float | None:
        """Real $/call after caching: the truest cost signal for ranking."""
        w = self.warm
        return w.cost_usd if w else None


# ---------------------------------------------------------------------------
# OpenRouter helpers
# ---------------------------------------------------------------------------

def _auth(settings: Settings) -> dict:
    return {"Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json", "X-Title": "KiCraft-bench"}


def _to_mtok(per_token) -> float:
    """OpenRouter prices are per-token strings; convert to $/Mtok."""
    try:
        return float(per_token or 0.0) * 1_000_000.0
    except (TypeError, ValueError):
        return 0.0


def discover_endpoints(settings: Settings, providers=None, top_n=None,
                       http=None) -> list[Endpoint]:
    """List the backends serving ``settings.model`` (free GET), cheapest first.

    ``providers`` (iterable of names/tags) filters to a subset; ``top_n`` keeps
    only the N cheapest by input price.
    """
    http = http or requests
    url = f"{settings.base_url}/models/{settings.model}/endpoints"
    r = http.get(url, headers=_auth(settings), timeout=30)
    r.raise_for_status()
    data = r.json().get("data", {}) or {}
    eps: list[Endpoint] = []
    for e in data.get("endpoints", []) or []:
        pr = e.get("pricing", {}) or {}
        eps.append(Endpoint(
            provider_name=e.get("provider_name") or "?",
            tag=e.get("tag") or "",
            price_prompt=_to_mtok(pr.get("prompt")),
            price_completion=_to_mtok(pr.get("completion")),
            quantization=e.get("quantization"),
            context_length=e.get("context_length"),
        ))
    if providers:
        want = {p.strip().lower() for p in providers if p.strip()}
        eps = [e for e in eps if e.provider_name.lower() in want
               or e.tag.lower() in want or e.tag.split("/")[0].lower() in want]
    eps.sort(key=lambda e: (e.price_prompt, e.price_completion))
    if top_n:
        eps = eps[:top_n]
    return eps


# ---------------------------------------------------------------------------
# Workload: one deterministic, cacheable, self-contained prompt
# ---------------------------------------------------------------------------

_SYSTEM_HEADER = (
    "You are KiCraft, an expert assistant that designs KiCad printed-circuit-board "
    "projects. You reason about functional blocks, part selection, net topology, and "
    "manufacturability, and you answer strictly in the requested format.\n\n"
    "DESIGN BRIEF (reference spec, treat as stable context):\n"
)

# A stable block repeated to pad the prompt to the requested size. Its content is
# irrelevant; it only has to be long, identical across providers, and worth caching.
_SPEC_BLOCK = (
    "- The board is a small USB-C powered sensor hub. Power: USB-C 5V in, a 3.3V LDO "
    "rail for logic, reverse-polarity and ESD protection on the input. MCU: an "
    "ESP32-class module with decoupling and a boot/enable button pair. Sensors: an "
    "I2C temperature/humidity sensor and an SPI barometric sensor, each with pull-ups "
    "and local bypass caps. Connectivity: a 4-pin Qwiic header and a 6-pin programming "
    "header. Indicators: a power LED and a user-controllable status LED with current "
    "limiting. Layout: two-layer, ground pour on the bottom, fine-pitch parts on top, "
    "antenna keep-out respected for the radio module. Manufacturing: 0.153mm minimum "
    "clearance, JLCPCB-compatible stackup, all parts in the in-house bundle library."
)


def build_messages(prompt_tokens: int = 3500) -> list[dict]:
    """A sized system prefix (padded to ~prompt_tokens) + a fixed small-JSON task.

    Identical bytes for every backend, so cost / TTFT / throughput / cache are
    measured apples-to-apples. Token count is approximated as chars/4.
    """
    target_chars = max(prompt_tokens, 200) * 4
    sys_text = _SYSTEM_HEADER
    while len(sys_text) < target_chars:
        sys_text += "\n" + _SPEC_BLOCK
    sys_text = sys_text[:target_chars]
    user = ('From the design brief above, output ONLY a compact single-line JSON '
            'object with exactly these keys: "summary" (string, at most 15 words), '
            '"part_count" (integer estimate), "nets" (array of at most 5 short '
            'net-name strings). No prose, no markdown, no code fence.')
    return [{"role": "system", "content": sys_text},
            {"role": "user", "content": user}]


def build_request(settings: Settings, tag: str, messages: list[dict],
                  max_tokens: int) -> dict:
    """A streaming completion payload pinned to exactly one backend.

    ``allow_fallbacks=False`` means OpenRouter must use ``tag`` or fail (so we
    never silently measure a different backend). Reuses the production
    cache-control breakpoint so caching behaves exactly as it does live.
    """
    payload = {
        "model": settings.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "usage": {"include": True},
        "provider": {"order": [tag], "allow_fallbacks": False},
    }
    if settings.enable_prompt_cache:
        CappedOpenRouterClient._apply_cache_control(payload["messages"])
    return payload


def timed_stream(settings: Settings, payload: dict, http=None) -> Measurement:
    """Run one pinned completion, timing TTFT + total and parsing usage.

    Records cost as OpenRouter's billed ``usage.cost`` (0.0 if the provider omits
    it; the caller fills an advertised-price fallback). Any transport/stream error
    is captured as a failed Measurement rather than raised, so one bad backend
    does not abort the sweep.
    """
    http = http or requests
    body = {k: v for k, v in payload.items() if not k.startswith("_")}
    t0 = time.perf_counter()
    ttft = None
    content_len = 0
    usage: dict = {}
    provider_echo = None
    finish = None
    try:
        with http.post(f"{settings.base_url}/chat/completions",
                       headers=_auth(settings), json=body,
                       timeout=settings.request_timeout_s, stream=True) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("provider"):
                    provider_echo = chunk["provider"]
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    delta = ch.get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        content_len += len(piece)
    except Exception as e:  # transport, HTTP status, or stream error
        return Measurement(ok=False, error=f"{type(e).__name__}: {e}",
                           total_s=time.perf_counter() - t0,
                           provider_echo=provider_echo, finish_reason=finish)
    total = time.perf_counter() - t0
    in_tok = int(usage.get("prompt_tokens") or 0)
    out_tok = int(usage.get("completion_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    cost = float(usage.get("cost") or 0.0)
    gen_tps = None
    if out_tok and ttft is not None and total > ttft:
        gen_tps = out_tok / (total - ttft)
    return Measurement(ok=True, ttft_s=ttft, total_s=total, prompt_tokens=in_tok,
                       completion_tokens=out_tok, cached_tokens=cached, cost_usd=cost,
                       provider_echo=provider_echo, finish_reason=finish, gen_tps=gen_tps)


def _matches(echo: str | None, ep: Endpoint) -> bool:
    if not echo:
        return False
    e = echo.lower()
    return e == ep.provider_name.lower() or e in ep.tag.lower() or ep.tag.split("/")[0].lower() == e


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def bench_provider(settings: Settings, ep: Endpoint, messages: list[dict],
                   max_tokens: int, repeat: int = 2, http=None,
                   sleep_s: float = 0.5) -> ProviderResult:
    """Run ``repeat`` pinned calls (cold then warm) and aggregate the metrics."""
    pr = ProviderResult(endpoint=ep)
    for i in range(repeat):
        payload = build_request(settings, ep.tag, messages, max_tokens)
        m = timed_stream(settings, payload, http=http)
        # Advertised-price fallback when OpenRouter omitted the real cost.
        if m.ok and m.cost_usd <= 0.0 and (m.prompt_tokens or m.completion_tokens):
            m.cost_usd = (m.prompt_tokens * ep.price_prompt
                          + m.completion_tokens * ep.price_completion) / 1_000_000.0
            if m.cost_usd <= 0.0:  # price also missing -> conservative estimate
                m.cost_usd = estimate_cost(settings.model, m.prompt_tokens, m.completion_tokens)
        pr.calls.append(m)
        if sleep_s and i < repeat - 1:
            time.sleep(sleep_s)
    ok_calls = [c for c in pr.calls if c.ok]
    pr.pinned_ok = any(_matches(c.provider_echo, ep) for c in ok_calls)
    hit_src = pr.warm
    pr.warm_cache_hit_pct = hit_src.cache_hit_pct() if hit_src else 0.0
    pr.mean_ttft_s = _mean([c.ttft_s for c in ok_calls])
    pr.mean_gen_tps = _mean([c.gen_tps for c in ok_calls])
    pr.total_cost_usd = sum(c.cost_usd for c in pr.calls)
    return pr


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def _is_low_quant(ep: Endpoint) -> bool:
    return (ep.quantization or "").lower() in _LOW_QUANT


def recommend(results: list[ProviderResult], settings: Settings,
              allow_low_quant: bool = False) -> dict:
    """Rank backends and choose a recommended pin.

    Ranking is by *measured warm cost per call* (real $, cache-discounted) among
    backends that actually served the pinned call, so a backend that caches well
    beats a cheaper one that does not. Latency is a tie-breaker / gate, and an
    aggressively quantized backend (fp4/int4) is kept out of the primary pick
    unless ``allow_low_quant`` is set (its cost is still surfaced as an alternative).
    """
    usable = [r for r in results if r.ok and r.pinned_ok and r.warm_cost() is not None]
    current_tag = settings.provider_order[0] if settings.provider_order else None
    current = next((r for r in results if r.endpoint.tag == current_tag
                    or r.endpoint.tag.split("/")[0] == current_tag), None)

    def superlative(pool, key, reverse=False):
        pool = [r for r in pool if key(r) is not None]
        if not pool:
            return None
        return (max if reverse else min)(pool, key=key)

    cheapest_sticker = superlative(usable, lambda r: r.endpoint.price_prompt)
    cheapest_warm = superlative(usable, lambda r: r.warm_cost())
    fastest = superlative(usable, lambda r: r.mean_ttft_s)
    highest_tps = superlative(usable, lambda r: r.mean_gen_tps, reverse=True)
    best_cache = superlative(usable, lambda r: r.warm_cache_hit_pct, reverse=True)

    # Pick: lowest measured warm cost among caching, not-too-slow, full-enough
    # precision backends; fall back to lowest warm cost overall, then cheapest sticker.
    caching = [r for r in usable if r.warm_cache_hit_pct >= _CACHE_HIT_THRESHOLD
               and (r.mean_ttft_s is None or r.mean_ttft_s <= _TTFT_TOLERANCE_S)
               and (allow_low_quant or not _is_low_quant(r.endpoint))]
    if caching:
        pick = min(caching, key=lambda r: (r.warm_cost(), r.mean_ttft_s or 9e9))
    elif cheapest_warm is not None:
        pick = cheapest_warm
    else:
        pick = cheapest_sticker

    # The cheapest low-quant backend, surfaced only when it actually undercuts the
    # quality-safe pick (so "you could save more with fp4" is an informed choice).
    low_quant_alt = superlative([r for r in usable if _is_low_quant(r.endpoint)],
                                lambda r: r.warm_cost())
    if not (low_quant_alt and pick and low_quant_alt.endpoint.tag != pick.endpoint.tag
            and (low_quant_alt.warm_cost() or 9e9) < (pick.warm_cost() or 9e9)):
        low_quant_alt = None

    saving_pct = None
    if pick and current and current.warm_cost():
        cw, pw = current.warm_cost(), pick.warm_cost()
        if cw and cw > 0:
            saving_pct = 100.0 * (cw - pw) / cw

    env_lines = []
    if pick:
        cap_p = round(pick.endpoint.price_prompt * _PRICE_HEADROOM + 0.005, 2)
        cap_c = round(pick.endpoint.price_completion * _PRICE_HEADROOM + 0.005, 2)
        env_lines = [
            f"KICRAFT_PROVIDER_ORDER={pick.endpoint.tag}",
            f"KICRAFT_MAX_PRICE_PROMPT={cap_p}",
            f"KICRAFT_MAX_PRICE_COMPLETION={cap_c}",
        ]

    return {
        "current": current, "pick": pick, "low_quant_alt": low_quant_alt,
        "stay": bool(pick and current and pick.endpoint.tag == current.endpoint.tag),
        "saving_pct": saving_pct, "env_lines": env_lines,
        "cheapest_sticker": cheapest_sticker, "cheapest_warm": cheapest_warm,
        "fastest_ttft": fastest, "highest_tps": highest_tps, "best_cache": best_cache,
    }


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _fmt_s(x):
    return f"{x:.2f}s" if x is not None else "   -  "


def _fmt_tps(x):
    return f"{x:5.1f}" if x is not None else "   - "


def format_report(results, rec, settings, spent_usd, meta) -> str:
    out = []
    out.append("=" * 84)
    out.append(f"  Provider benchmark: {settings.model}")
    out.append(f"  {len(results)} backend(s) | sized prompt ~{meta['prompt_tokens']} tok "
               f"| max_tokens {meta['max_tokens']} | repeat {meta['repeat']} "
               f"| cache {'on' if settings.enable_prompt_cache else 'off'}")
    out.append("=" * 84)
    out.append("  {:<22} {:>13}  {:>9}  {:>7}  {:>6}  {:>6}  {:>3}".format(
        "backend", "$/Mtok in/out", "warm $/call", "TTFT", "tok/s", "cache", "pin"))
    out.append("  " + "-" * 80)
    # sort the table by measured warm cost (cheapest real cost first)
    def sort_key(r):
        wc = r.warm_cost()
        return (wc if wc is not None else 9e9, r.endpoint.price_prompt)
    cur_tag = rec["current"].endpoint.tag if rec["current"] else None
    pick_tag = rec["pick"].endpoint.tag if rec["pick"] else None
    for r in sorted(results, key=sort_key):
        ep = r.endpoint
        marker = ""
        if ep.tag == pick_tag:
            marker = " *"   # recommended
        elif ep.tag == cur_tag:
            marker = " @"   # current pin
        if not r.ok:
            err = (r.calls[-1].error if r.calls else "no call") or "error"
            out.append("  {:<22} {:>13}  {:>9}  {}".format(
                ep.label()[:22], f"{ep.price_prompt:.3f}/{ep.price_completion:.3f}",
                "FAILED", err[:34]))
            continue
        wc = r.warm_cost()
        out.append("  {:<22} {:>13}  {:>9}  {:>7}  {:>6}  {:>5.0f}% {:>3}{}".format(
            ep.label()[:22], f"{ep.price_prompt:.3f}/{ep.price_completion:.3f}",
            f"${wc:.6f}" if wc is not None else "-",
            _fmt_s(r.mean_ttft_s), _fmt_tps(r.mean_gen_tps),
            r.warm_cache_hit_pct, "ok" if r.pinned_ok else "?", marker))
    out.append("  " + "-" * 80)
    out.append("  legend:  @ = current pin    * = recommended    "
               "pin: did the pinned backend actually serve?")
    out.append("")
    out += _format_recommendation(rec, settings)
    out.append("")
    out.append(f"  Total spent this run: ${spent_usd:.4f}")
    out.append("=" * 84)
    return "\n".join(out)


def _one(r):
    return f"{r.endpoint.label()} [{r.endpoint.tag}]" if r else "n/a"


def _format_recommendation(rec, settings) -> list:
    out = ["  " + "=" * 80, "  RECOMMENDATION", "  " + "-" * 80]
    cur, pick = rec["current"], rec["pick"]
    out.append(f"  Current pin : {_one(cur)}"
               + (f"  warm ${cur.warm_cost():.6f}/call, cache {cur.warm_cache_hit_pct:.0f}%"
                  if cur and cur.warm_cost() is not None else ""))
    if cur is not None and not cur.ok:
        # The pinned backend never served a call (e.g. 404 / data-policy filter).
        # With allow_fallbacks=True in production this is silent: requests drift to
        # OpenRouter's default backend, so the "pin" is not actually pinning.
        err = (cur.calls[-1].error if cur.calls else "no response") or "error"
        out.append(f"  !! WARNING : the current pin '{cur.endpoint.tag}' did NOT resolve "
                   f"({err[:48]}).")
        out.append("              With allow_fallbacks=True, production is silently "
                   "routing to OpenRouter's")
        out.append("              default backend, not this pin. An explicit, verified "
                   "re-pin is advised.")
    if not pick:
        out.append("  No backend produced a usable measurement; cannot recommend.")
        return out
    if rec["stay"]:
        out.append(f"  Verdict     : STAY on {pick.endpoint.tag} -- it is already the "
                   f"lowest measured warm cost.")
    else:
        sv = rec["saving_pct"]
        sv_txt = f" (~{sv:.0f}% cheaper per call)" if sv is not None else ""
        out.append(f"  Verdict     : SWITCH to {_one(pick)}{sv_txt}")
        out.append(f"                warm ${pick.warm_cost():.6f}/call, "
                   f"cache {pick.warm_cache_hit_pct:.0f}%, "
                   f"TTFT {_fmt_s(pick.mean_ttft_s)}, {_fmt_tps(pick.mean_gen_tps)} tok/s")
        if _is_low_quant(pick.endpoint):
            out.append(f"                CAUTION: {pick.endpoint.quantization} "
                       "quantization may reduce answer quality; validate on a real design.")
    alt = rec.get("low_quant_alt")
    if alt:
        out.append(f"  Cheaper alt : {_one(alt)} at ${alt.warm_cost():.6f}/call "
                   f"({alt.endpoint.quantization}) -- pass --allow-low-quant to select it, "
                   "but verify quality first.")
    out.append("  " + "-" * 80)
    out.append("  superlatives:")
    out.append(f"    cheapest sticker : {_one(rec['cheapest_sticker'])}")
    out.append(f"    cheapest warm $  : {_one(rec['cheapest_warm'])}")
    out.append(f"    fastest TTFT     : {_one(rec['fastest_ttft'])}")
    out.append(f"    highest tok/s    : {_one(rec['highest_tps'])}")
    out.append(f"    best cache hit%  : {_one(rec['best_cache'])}")
    if rec["env_lines"]:
        out.append("  " + "-" * 80)
        out.append("  Paste into .env to adopt the recommendation (bounded fallback):")
        for line in rec["env_lines"]:
            out.append(f"      {line}")
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def render_plot(results, rec, settings, output_path: str) -> None:
    """Multi-panel comparison plot. Uses the Agg backend (headless)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [r for r in results if r.ok]
    if not usable:
        return
    cur_tag = rec["current"].endpoint.tag if rec["current"] else None
    pick_tag = rec["pick"].endpoint.tag if rec["pick"] else None

    def colors(rows):
        cs = []
        for r in rows:
            if r.endpoint.tag == pick_tag:
                cs.append("#2ecc71")      # recommended = green
            elif r.endpoint.tag == cur_tag:
                cs.append("#3498db")      # current pin = blue
            else:
                cs.append("#95a5a6")      # other = grey
        return cs

    def panel(ax, key, title, xlabel, reverse=False, pct=False, money=False):
        rows = [r for r in usable if key(r) is not None]
        rows.sort(key=key, reverse=reverse)
        labels = [r.endpoint.label() for r in rows]
        vals = [key(r) for r in rows]
        y = range(len(rows))
        ax.barh(list(y), vals, color=colors(rows), alpha=0.9)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(True, alpha=0.3, axis="x")
        for yi, v in zip(y, vals):
            txt = (f"{v:.0f}%" if pct else (f"${v:.5f}" if money else f"{v:.2f}"))
            ax.text(v, yi, " " + txt, va="center", fontsize=6)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"OpenRouter backend benchmark: {settings.model}",
                 fontsize=14, fontweight="bold")

    panel(axes[0][0], lambda r: r.warm_cost(), "Measured warm cost / call (real $)",
          "USD per call (lower better)", money=True)
    panel(axes[0][1], lambda r: r.endpoint.price_prompt, "Sticker price (input)",
          "$/Mtok prompt (lower better)")
    panel(axes[0][2], lambda r: r.warm_cache_hit_pct, "Warm cache hit rate",
          "% prompt tokens cached (higher better)", reverse=True, pct=True)
    panel(axes[1][0], lambda r: r.mean_ttft_s, "Latency (time to first token)",
          "seconds (lower better)")
    panel(axes[1][1], lambda r: r.mean_gen_tps, "Generation throughput",
          "tokens/sec (higher better)", reverse=True)

    # Scatter: cost vs latency, the core trade-off.
    ax = axes[1][2]
    sc = [r for r in usable if r.warm_cost() is not None and r.mean_ttft_s is not None]
    for r in sc:
        c = ("#2ecc71" if r.endpoint.tag == pick_tag
             else "#3498db" if r.endpoint.tag == cur_tag else "#95a5a6")
        ax.scatter(r.mean_ttft_s, r.warm_cost(), c=c, s=60, zorder=5,
                   edgecolors="black", linewidths=0.5)
        ax.annotate(r.endpoint.provider_name, (r.mean_ttft_s, r.warm_cost()),
                    textcoords="offset points", xytext=(4, 3), fontsize=6)
    ax.set_title("Cost vs latency trade-off", fontsize=10, fontweight="bold")
    ax.set_xlabel("mean TTFT (s)", fontsize=8)
    ax.set_ylabel("warm $/call", fontsize=8)
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#2ecc71", label="Recommended"),
                        Patch(color="#3498db", label="Current pin"),
                        Patch(color="#95a5a6", label="Other backend")],
               loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def _measurement_dict(m: Measurement) -> dict:
    return {"ok": m.ok, "error": m.error, "ttft_s": m.ttft_s, "total_s": m.total_s,
            "prompt_tokens": m.prompt_tokens, "completion_tokens": m.completion_tokens,
            "cached_tokens": m.cached_tokens, "cost_usd": m.cost_usd,
            "provider_echo": m.provider_echo, "finish_reason": m.finish_reason,
            "gen_tps": m.gen_tps, "cache_hit_pct": m.cache_hit_pct()}


def _result_dict(r: ProviderResult) -> dict:
    ep = r.endpoint
    return {
        "provider_name": ep.provider_name, "tag": ep.tag,
        "price_prompt_mtok": ep.price_prompt, "price_completion_mtok": ep.price_completion,
        "quantization": ep.quantization, "context_length": ep.context_length,
        "pinned_ok": r.pinned_ok, "warm_cache_hit_pct": r.warm_cache_hit_pct,
        "mean_ttft_s": r.mean_ttft_s, "mean_gen_tps": r.mean_gen_tps,
        "warm_cost_usd": r.warm_cost(), "total_cost_usd": r.total_cost_usd,
        "calls": [_measurement_dict(m) for m in r.calls],
    }


def results_to_json(results, rec, settings, spent_usd, meta) -> dict:
    return {
        "model": settings.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta": meta, "total_spent_usd": spent_usd,
        "current_pin": settings.provider_order,
        "recommendation": {
            "stay": rec["stay"],
            "pick_tag": rec["pick"].endpoint.tag if rec["pick"] else None,
            "current_pin_resolved": bool(rec["current"] and rec["current"].ok),
            "low_quant_alt_tag": rec["low_quant_alt"].endpoint.tag if rec["low_quant_alt"] else None,
            "saving_pct": rec["saving_pct"], "env_lines": rec["env_lines"],
        },
        "results": [_result_dict(r) for r in results],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="provider-bench",
        description="Benchmark OpenRouter backends serving KiCraft's model for cost, "
                    "latency, throughput, and cache hit-rate, then recommend a pin.")
    ap.add_argument("--model", help="override the model id (default: resolved from env)")
    ap.add_argument("--providers", help="comma-separated subset of provider names/tags")
    ap.add_argument("--top-n", type=int, help="benchmark only the N cheapest backends")
    ap.add_argument("--prompt-tokens", type=int, default=3500,
                    help="approx size of the sized system prompt (default 3500)")
    ap.add_argument("--max-tokens", type=int, default=200,
                    help="output token cap per call (default 200)")
    ap.add_argument("--repeat", type=int, default=3,
                    help="calls per backend; the 1st primes the cache, the rest measure "
                         "the cache-hot steady state. >=3 is more robust to cache noise "
                         "(default 3)")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="seconds between cold and warm calls (default 0.5)")
    ap.add_argument("--max-spend", type=float, default=0.25,
                    help="abort the sweep before exceeding this USD budget (default 0.25)")
    ap.add_argument("--timeout", type=int,
                    help="per-request timeout in seconds (default: server setting); a "
                         "low value keeps one slow/hung backend from stalling the sweep")
    ap.add_argument("--out-json", help="results JSON path (default provider_bench_<ts>.json)")
    ap.add_argument("--out-plot", help="plot PNG path (default provider_bench_<ts>.png)")
    ap.add_argument("--no-plot", action="store_true", help="skip rendering the PNG")
    ap.add_argument("--allow-low-quant", action="store_true",
                    help="allow fp4/int4 backends as the primary recommendation "
                         "(default: down-ranked for design-quality reasons)")
    ap.add_argument("--json", action="store_true", help="also print the results JSON to stdout")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    if args.model:
        settings.model = args.model.strip()
    if args.timeout:
        settings.request_timeout_s = args.timeout

    http = requests.Session()
    try:
        eps = discover_endpoints(
            settings,
            providers=args.providers.split(",") if args.providers else None,
            top_n=args.top_n, http=http)
    except Exception as e:
        print(f"error: could not list endpoints for {settings.model}: {e}", file=sys.stderr)
        return 2
    if not eps:
        print(f"error: no backends found for {settings.model} "
              f"(check --providers / --model)", file=sys.stderr)
        return 2

    messages = build_messages(args.prompt_tokens)
    meta = {"prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens,
            "repeat": args.repeat}

    print(f"Benchmarking {len(eps)} backend(s) of {settings.model} "
          f"(repeat={args.repeat}, max_tokens={args.max_tokens}, budget=${args.max_spend})")
    results: list[ProviderResult] = []
    spent = 0.0
    for ep in eps:
        # Pre-flight worst-case estimate for the next backend; stop before overspend.
        est = args.repeat * (args.prompt_tokens * ep.price_prompt
                             + args.max_tokens * ep.price_completion) / 1_000_000.0
        if spent + est > args.max_spend:
            print(f"  ! budget guard: stopping before {ep.provider_name} "
                  f"(spent ${spent:.4f}, next ~${est:.4f}, cap ${args.max_spend})")
            break
        pr = bench_provider(settings, ep, messages, args.max_tokens,
                            repeat=args.repeat, http=http, sleep_s=args.sleep)
        spent += pr.total_cost_usd
        results.append(pr)
        wc = pr.warm_cost()
        status = (f"warm ${wc:.6f}/call  TTFT {_fmt_s(pr.mean_ttft_s)}  "
                  f"{_fmt_tps(pr.mean_gen_tps)} tok/s  cache {pr.warm_cache_hit_pct:.0f}%"
                  if pr.ok else f"FAILED ({pr.calls[-1].error if pr.calls else '?'})")
        print(f"  - {ep.label():<24} {status}")

    if not results:
        print("error: no backends were benchmarked", file=sys.stderr)
        return 2

    rec = recommend(results, settings, allow_low_quant=args.allow_low_quant)
    print()
    print(format_report(results, rec, settings, spent, meta))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = args.out_json or f"provider_bench_{stamp}.json"
    payload = results_to_json(results, rec, settings, spent, meta)
    Path(out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved results: {out_json}")

    if not args.no_plot:
        out_plot = args.out_plot or f"provider_bench_{stamp}.png"
        try:
            render_plot(results, rec, settings, out_plot)
        except Exception as e:
            print(f"warning: plot failed: {e}", file=sys.stderr)

    if args.json:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
