#!/usr/bin/env python3
"""Summarize the parts-library query log: catalog popularity and misses.

The BOM stage records every part query (see ``kicraft.parts_library.query_log``).
This rolls those events up to answer the two questions that drive catalog work:

  * which curated bundles are actually used? (popularity -> worth polishing,
    e.g. adding a 3D model and promoting prototype -> reviewed -> production)
  * which parts keep MISSING the library and fall back to an LCSC fetch or a
    stock-KiCad search? (add-to-library candidates -> cheaper, more consistent
    BOMs, since every miss is extra tool rounds on the hosted model)

This is the parts-library counterpart to ``web-cost-report`` (spend) and
``token-report`` (offline transcript cost).

    part-query-report                          # ~/.kicraft/part_queries.jsonl
    part-query-report --since 2026-06-01 --top 30
    part-query-report /path/to/part_queries.jsonl --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# query_log is pure-stdlib (no pcbnew/pydantic), so importing it is cheap.
from kicraft.parts_library.query_log import log_path, read_events

# A curated-bundle name (lowercase kebab) vs a stock KiCad lib prefix (CamelCase
# or underscores, e.g. "Device", "Connector_PinHeader_2.54mm"). Mirrors
# parts_library.manifest.PART_NAME_RE so a symbol/footprint lookup can be
# attributed to a bundle without importing the heavy module.
_BUNDLE_NAME_RE = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$")


def _is_bundle_lib(lib) -> bool:
    return bool(lib) and bool(_BUNDLE_NAME_RE.match(lib))


def summarize(events) -> dict:
    tools: Counter = Counter()
    callers: Counter = Counter()
    lib_hits: Counter = Counter()      # bundle -> times referenced (popularity)
    fetches: Counter = Counter()       # lcsc -> times fetched into the library
    fetch_name: dict = {}              # lcsc -> bundle slug it was saved as
    jlcpcb: Counter = Counter()        # lcsc -> times resolved via JLCPCB (a miss)
    unresolved: Counter = Counter()    # mpn -> times no LCSC id was found at all
    search_miss: Counter = Counter()   # (tool, keyword) -> empty stock searches
    n = 0
    for e in events:
        n += 1
        tool = e.get("tool", "?")
        outcome = e.get("outcome", "?")
        tools[tool] += 1
        if e.get("caller"):
            callers[e["caller"]] += 1
        ln = e.get("library_name")
        if tool == "lookup_lcsc_id" and outcome == "hit" and ln:
            lib_hits[ln] += 1
        elif tool in ("lookup_symbol", "lookup_footprint") and outcome == "hit" \
                and _is_bundle_lib(e.get("lib")):
            lib_hits[e["lib"]] += 1
        if tool == "add_part_from_lcsc" and outcome == "fetched":
            lcsc = e.get("lcsc") or e.get("query") or "?"
            fetches[lcsc] += 1
            fetch_name.setdefault(lcsc, e.get("library_name") or "")
        if tool == "lookup_lcsc_id" and outcome == "resolved":
            jlcpcb[e.get("lcsc") or "?"] += 1
        if tool == "lookup_lcsc_id" and outcome == "miss":
            unresolved[e.get("query") or "?"] += 1
        if tool in ("search_symbols", "search_footprints") and outcome == "miss":
            search_miss[f"{tool}:{e.get('query') or '?'}"] += 1
    return {
        "n_events": n,
        "tools": dict(tools),
        "callers": dict(callers),
        "lib_hits": dict(lib_hits),
        "fetches": dict(fetches),
        "fetch_name": fetch_name,
        "jlcpcb": dict(jlcpcb),
        "unresolved": dict(unresolved),
        "search_miss": dict(search_miss),
    }


def _current_maturities() -> dict:
    """Best-effort {bundle_name: maturity} for annotating popularity.

    Reads the home + vendored tiers via the loader; returns {} if anything
    goes wrong so the report still works without the kicraft library present."""
    try:
        from kicraft.parts_library import load_all_with_overrides
        active, _shadowed, _broken = load_all_with_overrides(None)
        return {p.manifest.name: p.manifest.maturity for p in active}
    except Exception:  # noqa: BLE001
        return {}


def format_report(s: dict, top: int = 20) -> str:
    out: list[str] = []
    out.append("=" * 72)
    out.append(f"  KiCraft part-query log  ({s['n_events']} events)")
    out.append("=" * 72)
    if s["tools"]:
        tools = "  ".join(f"{k}={v}" for k, v in sorted(s["tools"].items(), key=lambda kv: -kv[1]))
        out.append("  by tool:   " + tools)
    if s["callers"]:
        callers = "  ".join(f"{k}={v}" for k, v in sorted(s["callers"].items(), key=lambda kv: -kv[1]))
        out.append("  by caller: " + callers)

    maturities = _current_maturities()
    out.append("  " + "-" * 68)
    out.append("  LIBRARY HITS  (used bundles -> polish / 3D / promote candidates):")
    if s["lib_hits"]:
        for name, c in sorted(s["lib_hits"].items(), key=lambda kv: -kv[1])[:top]:
            mat = maturities.get(name)
            flag = ""
            if mat == "prototype":
                flag = "   <- prototype, REVIEW candidate"
            elif mat:
                flag = f"   [{mat}]"
            out.append(f"    {c:>4}x  {name}{flag}")
    else:
        out.append("    (none recorded yet)")

    # Misses that became (or could become) library parts.
    out.append("  " + "-" * 68)
    out.append("  MISSES -> ADD-TO-LIBRARY candidates (LCSC fetched or JLCPCB-resolved):")
    miss = Counter()
    for lcsc, c in s["fetches"].items():
        miss[lcsc] += c
    for lcsc, c in s["jlcpcb"].items():
        miss[lcsc] += c
    if miss:
        for lcsc, c in sorted(miss.items(), key=lambda kv: -kv[1])[:top]:
            saved = s["fetch_name"].get(lcsc)
            tail = f"  (fetched as {saved})" if saved else "  (resolved, not bundled)"
            out.append(f"    {c:>4}x  {lcsc}{tail}")
    else:
        out.append("    (none recorded yet)")

    if s["unresolved"]:
        out.append("  " + "-" * 68)
        out.append("  UNRESOLVED MPNs  (no LCSC id found: ambiguous or missing):")
        for mpn, c in sorted(s["unresolved"].items(), key=lambda kv: -kv[1])[:top]:
            out.append(f"    {c:>4}x  {mpn}")

    if s["search_miss"]:
        out.append("  " + "-" * 68)
        out.append("  STOCK-SEARCH MISSES  (keywords that matched nothing):")
        for kw, c in sorted(s["search_miss"].items(), key=lambda kv: -kv[1])[:top]:
            out.append(f"    {c:>4}x  {kw}")

    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Summarize the parts-library query log: which curated bundles "
                    "are popular (polish/promote candidates) and which queries miss "
                    "the library (add-to-library candidates).")
    ap.add_argument("log", nargs="?", default=None,
                    help=f"path to part_queries.jsonl (default: {log_path()})")
    ap.add_argument("--since", metavar="ISO_TS",
                    help="only count events with ts >= this ISO timestamp (e.g. 2026-06-01)")
    ap.add_argument("--caller", metavar="TAG",
                    help="only count events from this caller (e.g. web / cli)")
    ap.add_argument("--top", type=int, default=20, help="rows per section (default 20)")
    ap.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = ap.parse_args(argv)

    path = Path(args.log) if args.log else log_path()
    events = [
        e for e in read_events(path)
        if (not args.since or str(e.get("ts", "")) >= args.since)
        and (not args.caller or e.get("caller") == args.caller)
    ]
    if not events:
        where = f" in {path}" if path.is_file() else f" ({path} does not exist yet)"
        print("no part-query events found" + where, file=sys.stderr)
        return 0
    s = summarize(events)
    if args.json:
        print(json.dumps(s, indent=2, default=str))
    else:
        print(format_report(s, top=args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
