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
    """Record a successful single-LCSC resolution. Best-effort: never raises."""
    if not lcsc:
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


__all__ = ["ENV_PATH", "cache_path", "key_for", "get", "put"]