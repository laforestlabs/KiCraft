"""Persistent MPN -> LCSC resolution cache.

The BOM stage's dominant cost is repeated ``lookup-lcsc-id`` calls for the
same MPN across design runs: a part resolved once (e.g. BMP280, 47 lookups in
one window) but never vendored as a bundle gets re-resolved from scratch on
every later run. The resolution itself is deterministic for the same MPN, so
caching a single successful hit makes every repeat resolve instant, offline,
and free of the network round-trip (and the model-attention it costs).

The cache is a per-machine JSON file at ``~/.kicraft/mpn_cache.json`` (override
with ``$KICRAFT_MPN_CACHE``), next to the spend ledger and the part-query log.
It stores only ``{normalized_mpn: {lcsc, source, ts}}`` (part identifiers, no
design content). Writes are best-effort and atomic; a missing or corrupt file
is treated as empty, so telemetry/storage trouble never breaks a tool call.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path

ENV_PATH = "KICRAFT_MPN_CACHE"

# A bare LCSC C-number anywhere in the query is the canonical key for it: a
# pasted lcsc.com/jlcpcb.com URL and the bare "C190004" all mean one part.
_LCSC_RE = re.compile(r"(?<![A-Za-z0-9])C\d{4,8}(?![A-Za-z0-9])", re.IGNORECASE)


def cache_path() -> Path:
    override = os.environ.get(ENV_PATH)
    if override:
        return Path(override)
    return Path.home() / ".kicraft" / "mpn_cache.json"


def key_for(mpn: str) -> str:
    """Canonical cache key for an MPN/keyword-ish query: the bare LCSC C-number
    if one is present, else the uppercased, whitespace-stripped string."""
    s = (mpn or "").strip()
    m = _LCSC_RE.search(s)
    if m:
        return m.group(0).upper()
    return s.upper()


def cacheable(mpn: str) -> bool:
    """Whether a query is a *precise* part identifier safe to freeze, vs a fuzzy
    free-text search that should re-resolve every time.

    Caching a keyword search ('SPDT slide switch SMD', 'BME280 Bosch') would
    freeze one heuristic 'best match' per machine forever — a wrong first hit
    can never be corrected, and a re-phrasing should be free to find a better
    part. So we cache ONLY: a bare LCSC C-number, or a single whitespace-free
    token that carries at least one digit (real MPNs effectively always do:
    BMP280, VL53L1CXV0FY/1, SK-12D07VG4). A descriptive phrase (has whitespace)
    or a bare word ('diode', no digit) is never cached."""
    s = (mpn or "").strip()
    if not s:
        return False
    if _LCSC_RE.search(s):
        return True
    return (not any(c.isspace() for c in s)) and any(c.isdigit() for c in s)


def _load() -> dict:
    try:
        return json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def get(mpn: str) -> dict | None:
    """Return the cached {lcsc, source, ts} for ``mpn`` or None when absent."""
    entry = _load().get(key_for(mpn))
    return entry if isinstance(entry, dict) else None


def put(mpn: str, lcsc: str, source: str) -> None:
    """Record a successful single-LCSC resolution. Best-effort: never raises.

    No-ops for fuzzy free-text queries (see ``cacheable``): only a precise part
    identifier is frozen, so a heuristic keyword match is never cached."""
    if not lcsc or not cacheable(mpn):
        return
    try:
        data = _load()
        data[key_for(mpn)] = {"lcsc": lcsc, "source": source,
                              "ts": _dt.datetime.now(_dt.timezone.utc).isoformat()}
        p = cache_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass


__all__ = ["ENV_PATH", "cache_path", "key_for", "cacheable", "get", "put"]