"""Live lcsc.com retail-storefront stock for LCSC C-numbers.

The offline jlcparts dump (``jlcparts.py``) tracks JLCPCB *assembly*
inventory. The lcsc.com retail storefront is a separate stock pool: a part
with millions in JLC assembly stock can be sold out at retail (KC-4AZ7PE's
0603 passives: dump said 5-15M, storefront said 0). The BOM sourcing gate
requires a pick to be in stock at BOTH, so this module is the one shared
live retail reading for the gate (§9.26), pricing, and the lookup tool.

Storefront endpoint: ``wmsc.lcsc.com/ftps/wm/product/detail`` — the JSON API
behind the lcsc.com product page (``result.stockNumber`` is the number the
page shows). Unauthenticated and, unlike jlcpcb.com's keyword API, not
WAF-blocked from server IPs today; treat it as fragile anyway: every failure
raises ``RetailUnavailable`` so callers fail open ("can't verify — don't
block"), a circuit breaker keeps an outage cheap, and ``KICRAFT_LCSC_RETAIL=0``
turns the whole thing off.

Readings are cached on disk (``~/.kicraft/lcsc_retail_cache.json``, override
``$KICRAFT_LCSC_RETAIL_CACHE``) because each stage-commit attempt runs in a
fresh subprocess — an in-process cache would re-fetch the whole BOM on every
one of the stage's up-to-5 attempts. Entries carry only part identifiers and
stock counts, no design content; writes are best-effort and atomic, a missing
or corrupt file is treated as empty.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ENV_PATH = "KICRAFT_LCSC_RETAIL_CACHE"

RETAIL_URL = "https://wmsc.lcsc.com/ftps/wm/product/detail?productCode={cid}"

# Same browser UA the easyeda price fallback uses (web.py): the endpoint
# serves browsers; a bare urllib UA invites the WAF.
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.lcsc.com/",
}
_SSL_CTX = ssl.create_default_context()

_TTL_S = 900        # one reading spans a full 5-attempt commit cycle + pricing
_BREAKER_S = 120    # after a failure, fail fast instead of 10s-timing-out per part
_GAP_S = 0.15       # politeness gap between real network hits
_TIMEOUT_S = 10

# Picky floor: the walk-time threshold for parts *we* are choosing (generics
# with dozens of equivalents). 100 covers the min-buy-100 passive reels and
# gives churn headroom. Deliberately chosen parts (explicit pins, bundles)
# are vetoed only when genuinely unorderable (>= min_buy), not by this floor.
_RETAIL_FLOOR = 100

_LOCK = threading.Lock()
_MEM: dict[str, dict] = {}
_breaker_until = 0.0
_last_hit = 0.0


class RetailUnavailable(Exception):
    """Transport/HTTP/WAF trouble — a transient outage, not a stock answer.

    Callers fail open (treat the part as retail-unverified), never closed."""


def enabled() -> bool:
    """Kill switch. Off when KICRAFT_LCSC_RETAIL is falsy or the run is a
    mock/replay one (loadtest, self-eval replay: deterministic and $0 — a live
    network gate would make them flaky for nothing)."""
    if os.environ.get("KICRAFT_LCSC_RETAIL", "").strip().lower() in (
            "0", "off", "no", "false"):
        return False
    if os.environ.get("KICRAFT_LLM_MODE", "").strip().lower() in (
            "mock", "replay"):
        return False
    return True


def retail_floor() -> int:
    try:
        return int(os.environ.get("KICRAFT_BOM_RETAIL_STOCK_FLOOR", "")
                   or _RETAIL_FLOOR)
    except ValueError:
        return _RETAIL_FLOOR


def cache_path() -> Path:
    override = os.environ.get(ENV_PATH)
    if override:
        return Path(override)
    return Path.home() / ".kicraft" / "lcsc_retail_cache.json"


def _norm(cid: str) -> str:
    s = str(cid or "").strip().upper()
    return s if s.startswith("C") else f"C{s}"


def _load_disk() -> dict:
    try:
        data = json.loads(cache_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_disk(cid: str, entry: dict) -> None:
    try:
        data = _load_disk()
        data[cid] = entry
        p = cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass


def _fresh(entry: dict | None) -> dict | None:
    if not isinstance(entry, dict):
        return None
    try:
        if time.time() - float(entry.get("ts", 0)) < _TTL_S:
            return entry
    except (TypeError, ValueError):
        pass
    return None


def _fetch(cid: str) -> dict:
    """One real storefront hit. Serialized + rate-gapped across threads."""
    global _breaker_until, _last_hit
    if time.time() < _breaker_until:
        raise RetailUnavailable(f"lcsc retail: circuit open after a recent "
                                f"failure; skipping {cid}")
    wait = _GAP_S - (time.time() - _last_hit)
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(RETAIL_URL.format(cid=cid), headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S,
                                    context=_SSL_CTX) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        _breaker_until = time.time() + _BREAKER_S
        raise RetailUnavailable(f"lcsc retail {cid}: {e}") from e
    finally:
        _last_hit = time.time()
    if not isinstance(data, dict) or data.get("code") != 200:
        # A JSON envelope with a non-200 application code could be a throttle
        # as easily as a lookup miss — can't tell them apart, so treat it as
        # an outage (fail open), never as "stock 0" (which would bounce a
        # possibly-fine part).
        _breaker_until = time.time() + _BREAKER_S
        raise RetailUnavailable(f"lcsc retail {cid}: unexpected response "
                                f"{str(data)[:120]}")
    result = data.get("result")
    if not isinstance(result, dict):
        # code==200 with a null result is a real answer: the C# is not sold
        # on the retail storefront (existence is separately guaranteed by the
        # offline dump).
        return {"stock": 0, "min_buy": 1}
    try:
        stock_n = int(result.get("stockNumber") or 0)
    except (TypeError, ValueError):
        stock_n = 0
    try:
        min_buy = max(1, int(result.get("minBuyNumber") or 1))
    except (TypeError, ValueError):
        min_buy = 1
    return {"stock": stock_n, "min_buy": min_buy}


def stock(cid: str) -> dict:
    """Live retail reading for one C#: ``{"lcsc","stock","min_buy",
    "checked_at"}``. TTL-cached (memory, then disk); raises
    ``RetailUnavailable`` on any transport/WAF trouble."""
    cid = _norm(cid)
    with _LOCK:
        entry = _fresh(_MEM.get(cid)) or _fresh(_load_disk().get(cid))
        if entry is None:
            got = _fetch(cid)
            entry = {"stock": got["stock"], "min_buy": got["min_buy"],
                     "ts": time.time()}
            _save_disk(cid, entry)
        _MEM[cid] = entry
    return {"lcsc": cid, "stock": int(entry["stock"]),
            "min_buy": int(entry.get("min_buy") or 1),
            "checked_at": _dt.datetime.fromtimestamp(
                float(entry["ts"]), _dt.timezone.utc).isoformat()}


def attach_stock(payload: dict, cid: str | None, *, nullable: bool) -> dict:
    """Pin ``retail_stock``/``retail_min_buy`` from the live storefront onto
    ``payload`` -- the ONE wrap of enabled()/stock()/RetailUnavailable shared
    by the BOM lookup tool (cli_app) and the web pricing cache, which used to
    carry drifting copies with different outage encodings.

    nullable=True (web pricing): both keys are always present and None means
    unverified -- the cost UI shows it as such and the price cache refuses to
    merge None, so a reopen re-checks. nullable=False (BOM lookup): keys
    appear only on success and an outage sets ``payload["retail"] =
    "unverified"`` for the model to read; disabled is a silent no-op."""
    if nullable:
        # Reset, not setdefault: the keys mean THIS call's reading, so a
        # stale value from an earlier merge must not survive an outage.
        payload["retail_stock"] = None
        payload["retail_min_buy"] = None
    if not cid or not enabled():
        return payload
    try:
        info = stock(str(cid))
        payload["retail_stock"] = info["stock"]
        payload["retail_min_buy"] = info["min_buy"]
    except RetailUnavailable:
        if not nullable:
            payload["retail"] = "unverified"
    return payload


def in_stock(cid: str, *, picky: bool) -> tuple[bool, dict]:
    """Whether ``cid`` is orderable at retail, plus the reading.

    picky=False (veto): a deliberately chosen part (explicit pin, library
    bundle) only fails when it genuinely cannot be ordered — stock below the
    listing's own minimum buy. picky=True: the candidate-walk threshold for
    parts we are choosing among many equivalents — also requires the retail
    floor, so a near-dry listing isn't picked over a plentiful one."""
    info = stock(cid)
    need = max(info["min_buy"], retail_floor() if picky else 1)
    return info["stock"] >= need, info


def clear_cache() -> None:
    """Test isolation: drop in-process state (the disk cache is redirected
    per-test via $KICRAFT_LCSC_RETAIL_CACHE)."""
    global _breaker_until, _last_hit
    with _LOCK:
        _MEM.clear()
        _breaker_until = 0.0
        _last_hit = 0.0


__all__ = ["ENV_PATH", "RETAIL_URL", "RetailUnavailable", "attach_stock",
           "cache_path", "clear_cache", "enabled", "in_stock", "retail_floor",
           "stock"]
