"""Whitelisted BOM lookup tools and per-invocation executor state."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

BOM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_parts",
            "description": "List curated parts-library bundles available to this project "
            "(vendored + any fetched). Returns a table with the exact symbol and "
            "footprint strings to use verbatim in the BOM. The full table is "
            "large — pass 'query' keywords to filter it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "optional keywords to filter the "
                        "table, e.g. 'bnc' or 'trimmer 3296'",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_symbol",
            "description": "Verify a KiCad symbol in 'Library:Name' form exists and list its pins.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "e.g. 'Device:R' or 'usb-c-16p:TYPE-C-31-M-12'",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_symbols",
            "description": "Find the correct symbol id by keyword when you do not know the exact "
            "'Library:Name'. Searches curated parts-library bundles first (returned "
            "as symbol+footprint pairs — adopt the pair), then stock KiCad symbols. "
            "Returns ids to use verbatim. Use this instead of guessing a symbol name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "keywords, e.g. 'conn 02x08', 'crystal', 'n-channel mosfet'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_footprints",
            "description": "Find the correct footprint id by keyword when you do not know the exact "
            "'Library:Name'. Searches curated parts-library bundles first (returned "
            "as symbol+footprint pairs — adopt the pair), then stock KiCad footprints. "
            "Returns ids to use verbatim. Use this instead of guessing a footprint name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "keywords, e.g. 'pinheader 2x08', 'barreljack', 'sot-23'",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_footprint",
            "description": "Verify a KiCad footprint in 'Library:Name' form exists and report its "
            "pad count.",
            "parameters": {
                "type": "object",
                "properties": {
                    "footprint": {
                        "type": "string",
                        "description": "e.g. 'Resistor_SMD:R_0603_1608Metric'",
                    }
                },
                "required": ["footprint"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_lcsc_id",
            "description": "Resolve a manufacturer part number, search keyword, or pasted LCSC "
            "id / product URL to an LCSC part number (C#####). Returns {ok, lcsc} "
            "or a candidates list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mpn": {
                        "type": "string",
                        "description": "MPN or keyword, e.g. 'SK-12D07VG4' or "
                        "'SPDT slide switch SMD'",
                    }
                },
                "required": ["mpn"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_part_from_lcsc",
            "description": "Fetch a real symbol+footprint bundle from LCSC into the project parts "
            "library. Afterwards call list_parts to get the exact '<name>:<symbol>' "
            "and '<name>:<footprint>' strings. Not needed for core-default rows "
            "that name a bundle: those are already in the library.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lcsc_id": {"type": "string", "description": "LCSC part number like C2837270"},
                    "name": {"type": "string", "description": "optional library slug"},
                },
                "required": ["lcsc_id"],
            },
        },
    },
]

# LCSC ids users paste into briefs/answers — bare (C7386355) or inside an
# lcsc.com / jlcpcb.com product URL. Mirrors cli_app._LCSC_ID_RE.
_LCSC_ID_RE = re.compile(r"(?<![A-Za-z0-9])C\d{4,8}(?![A-Za-z0-9])", re.IGNORECASE)

# A part's MPN spelling variants all mean the same lookup: normalize before the
# per-MPN budget key and the resolution cache. Strip whitespace, uppercase, and
# collapse any pasted lcsc.com/jlcpcb.com product URL to its bare C-number.
def _normalize_mpn(raw: str) -> str:
    s = (raw or "").strip().upper()
    m = _LCSC_ID_RE.search(s)
    return m.group(0).upper() if m else s


# The BOM SEARCH BUDGET tells the model "1 lookup_lcsc_id query + at most 1
# retry per part." A weak model (deepseek-v4-flash) ignores that and re-spells
# the same MPN for round after round (e.g. VL53L1CXV0FY/1, VL53L1C, VL53L1X:
# 56 lookups for one part). Enforce it server-side instead: cap calls per
# normalized MPN within one stage attempt, then return a terminal result that
# forbids further retries for that part.
_BOM_MPN_QUERY_CAP = 2


# Read-only BOM tools that are pure for the life of a stage: the stock KiCad
# symbol/footprint libraries don't change mid-stage, so an identical call always
# returns the same answer. A weak model re-issues these verifications round after
# round (symbol/footprint lookups+searches are ~70% of BOM tool calls in the
# part-query log), so memoize them per stage -- an exact repeat then skips the
# CLI subprocess entirely. list_parts and add_part_from_lcsc are excluded (the
# library mutates when a part is fetched), and lookup_lcsc_id is excluded (it
# owns the per-MPN cap + the network resolution cache).
_MEMOIZED_BOM_TOOLS = frozenset(
    {"lookup_symbol", "lookup_footprint", "search_symbols", "search_footprints"}
)


def _new_bundle_rows(list_parts_stdout: str, lcsc_id: str, name: str | None) -> str:
    """After add_part_from_lcsc, the model needs the exact '<name>:<symbol>' /
    '<name>:<footprint>' strings for the ONE bundle it just fetched -- not the
    whole ~42 KB parts table re-dumped (which then rides the conversation in
    every later tool round, the dominant BOM token cost). Return the markdown
    table header plus only the row(s) for the new bundle, matched by its LCSC
    C-number (rendered 'lcsc:C#####' in the sourcing column) or its slug. Falls
    back to a bounded slice if the row can't be located (e.g. add-part failed),
    so the model always gets something actionable."""
    lines = (list_parts_stdout or "").splitlines()
    sep = next((i for i, ln in enumerate(lines) if re.match(r"\s*\|\s*-{2,}", ln)), None)
    if sep is None:  # unexpected format -> don't silently hide it, just bound it
        return (list_parts_stdout or "")[:8000]
    m = _LCSC_ID_RE.search(lcsc_id or "")
    cnum = m.group(0).upper() if m else ""
    slug = (name or "").strip().lower()
    rows = [
        ln
        for ln in lines[sep + 1 :]
        if ln.lstrip().startswith("|")
        and ((cnum and cnum in ln.upper()) or (slug and f"`{slug}`" in ln.lower()))
    ]
    header = "\n".join(lines[: sep + 1])
    if not rows:  # couldn't pin the new row; give the model the header + a hint
        return header + "\n(new row not located; call list_parts for the full table)"
    return (header + "\n" + "\n".join(rows))[:8000]

def build_bom_executor(workspace: Path, runner: Callable, command_prefix: list[str]):
    """Return an executor(name, args) -> str backed by the kicraft CLI (cwd=workspace)."""
    lcsc_calls: dict[str, int] = {}  # normalized MPN -> attempts this stage (search budget)
    memo: dict[tuple[str, str], str] = {}  # read-only lookups, deduped per stage
    resolution_ledger: dict[str, dict[str, str]] = {}

    def execute(name: str, args: dict) -> str:
        ckey: tuple[str, str] | None = None
        if name in _MEMOIZED_BOM_TOOLS:
            arg = str(
                args.get("symbol") or args.get("footprint") or args.get("query") or ""
            ).strip()
            ckey = (name, arg)
            if ckey in memo:  # identical lookup already answered this stage
                return memo[ckey]
        if name == "list_parts":
            # Generous cap: the vendored library alone renders ~600 chars per
            # bundle, and the core-defaults adoption rule sends the model HERE
            # for exact ids. The full table has outgrown the cap (~55KB for
            # 260+ bundles), so a truncated tail must SAY so and point at the
            # query filter — a silent cut reads as "that part doesn't exist".
            cmd = ["list-parts"]
            query = str(args.get("query") or "").strip()
            if query:
                cmd.append(query)
            r = runner(command_prefix + cmd, workspace)
            out = r.stdout or r.stderr
            if len(out) > 40000:
                cut = out.rfind("\n", 0, 40000)
                out = out[: cut if cut > 0 else 40000] + (
                    "\n… TABLE TRUNCATED — call list_parts again with a "
                    '\'query\' (e.g. {"query": "bnc"}) to see the rest.'
                )
            return out
        if name == "lookup_symbol":
            r = runner(command_prefix + ["lookup-symbol", str(args.get("symbol", ""))], workspace)
            return memo.setdefault(ckey, (r.stdout or r.stderr)[:3000])
        if name == "search_symbols":
            r = runner(command_prefix + ["search-symbols", str(args.get("query", ""))], workspace)
            return memo.setdefault(ckey, (r.stdout or r.stderr)[:3000])
        if name == "search_footprints":
            r = runner(command_prefix + ["search-footprints", str(args.get("query", ""))], workspace)
            return memo.setdefault(ckey, (r.stdout or r.stderr)[:3000])
        if name == "lookup_footprint":
            r = runner(command_prefix + ["lookup-footprint", str(args.get("footprint", ""))], workspace)
            return memo.setdefault(ckey, (r.stdout or r.stderr)[:3000])
        if name == "lookup_lcsc_id":
            mpn = str(args.get("mpn", ""))
            key = _normalize_mpn(mpn)
            lcsc_calls[key] = lcsc_calls.get(key, 0) + 1
            if lcsc_calls[key] > _BOM_MPN_QUERY_CAP:
                # SEARCH BUDGET exceeded for this MPN: the result cannot change
                # by retrying, so stop the wasted subprocess + the LLM round it
                # triggers. Tell the model to stop retrying this part.
                return (
                    f"Resolution for '{mpn}' has already been attempted "
                    f"{_BOM_MPN_QUERY_CAP} times; repeating it cannot change "
                    "the answer. STOP retrying this part: use the result from "
                    "the first lookup above, ask the user for an exact LCSC "
                    "C-number (C#####), or use the closest stock KiCad "
                    "symbol/footprint and record the substitution in "
                    "assumptions. Do NOT call lookup_lcsc_id for this MPN "
                    "again this stage."
                )
            r = runner(command_prefix + ["lookup-lcsc-id", mpn], workspace)
            out = (r.stdout or r.stderr)[:3000]
            try:
                resolved = json.loads(out)
            except (TypeError, json.JSONDecodeError):
                resolved = {}
            if resolved.get("ok") and resolved.get("lcsc"):
                resolution_ledger[key] = {
                    "requested_part": mpn,
                    "accepted_lcsc_id": str(resolved["lcsc"]),
                    "exact_symbol": "",
                    "exact_footprint": "",
                    "source_tool": "lookup_lcsc_id",
                }
            return out
        if name == "add_part_from_lcsc":
            # Persist fetched parts to the shared HOME tier (not project): a part
            # the model needs once is then reused by every later design as a
            # `prototype`-badged bundle, so the catalog self-grows and repeated
            # LCSC fetches (the dominant BOM cost) amortize away.
            lcsc_id = str(args.get("lcsc_id", ""))
            name = str(args["name"]) if args.get("name") else None
            cmd = ["add-part", "--from-lcsc", lcsc_id, "--into", "home"]
            if name:
                cmd += ["--name", name]
            r = runner(command_prefix + cmd, workspace)
            lp = runner(command_prefix + ["list-parts"], workspace)
            new_rows = _new_bundle_rows(lp.stdout, lcsc_id, name)
            for line in new_rows.splitlines():
                columns = [column.strip().strip("`") for column in line.split("|")[1:-1]]
                if len(columns) >= 7 and lcsc_id.upper() in columns[2].upper():
                    ledger_key = next(
                        (
                            key
                            for key, row in resolution_ledger.items()
                            if row["accepted_lcsc_id"].upper() == lcsc_id.upper()
                        ),
                        _normalize_mpn(lcsc_id),
                    )
                    resolution_ledger[ledger_key] = {
                        "requested_part": resolution_ledger.get(ledger_key, {}).get(
                            "requested_part", lcsc_id
                        ),
                        "accepted_lcsc_id": lcsc_id.upper(),
                        "exact_symbol": columns[5],
                        "exact_footprint": columns[6],
                        "source_tool": "add_part_from_lcsc",
                    }
                    break
            return (
                f"add-part exit={r.returncode}\n{(r.stdout + chr(10) + r.stderr).strip()[:1500]}"
                f"\n\nNEWLY ADDED BUNDLE (use these strings verbatim; call "
                f"list_parts for the full library):\n{new_rows}"
            )
        return f"unknown tool: {name}"

    execute.resolution_ledger = resolution_ledger

    return execute
