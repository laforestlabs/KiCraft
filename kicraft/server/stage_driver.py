"""Generic stage driver: drive KiCraft design stages through the capped client.

This is the agent loop ported out of Claude Code. For each stage it loads the
real stage spec + the slot's Pydantic JSON schema, asks the model (via the
capped gateway) to draft the slot JSON, and commits it with the existing
deterministic CLI. If stage-commit rejects the slot, the validation error is
fed back to the model to self-correct (the same loop the Claude Code skill
runs). The BOM stage additionally gets tools (list_parts / lookup_symbol /
lookup_lcsc_id / add_part_from_lcsc) so the model resolves real symbols and
footprints (or fetches them from LCSC) instead of guessing.

    python -m kicraft.server.stage_driver \\
        --workspace /tmp/lamp --stages intent,functional_spec,architecture,bom \\
        --brief "a USB-C powered LED night light, no microcontroller"
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from kicraft.design import models
from kicraft.fsutil import atomic_write_text
from kicraft.parts_library import jlcparts, lcsc_retail

from .client import CappedOpenRouterClient, make_client
from .config import Settings
from .pricing import _stock_floor
from .spend_guard import BudgetExceeded, SpendGuard


def _bundle_sourcing_lcsc(bundle: str) -> str:
    """The vendored bundle's pinned sourcing C#, or '' (best-effort, offline)."""
    try:
        manifest = (
            _REPO / "kicraft" / "parts_library" / bundle / "manifest.json"
        )
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return str((data.get("sourcing") or {}).get("lcsc") or "").strip()
    except Exception:
        return ""

# The repo venv has no `kicraft` console script; cli_app.py has a __main__ guard.
KICRAFT = [sys.executable, "-m", "kicraft.design.cli_app"]


def _child_cpu_s() -> float:
    """User+system CPU seconds consumed by this process's child subprocesses
    (the stage-prep/commit calls and BOM tool lookups). RUSAGE_CHILDREN accumulates
    over the whole process, so the driver snapshots a before/after delta per stage.
    On non-POSIX this reports 0 (resource.RUSAGE_CHILDREN is unavailable); the
    ledger column then stays null.

    CAVEAT — reliable only single-flight: RUSAGE_CHILDREN is per-PROCESS, not
    per-thread. The web app runs designs in concurrent _run_design threads in one
    process, so when two designs are in flight the stage windows overlap and each
    one's cpu_s delta absorbs the other's subprocess CPU. wall_s (a monotonic
    delta) stays correct under concurrency; cpu_s does not. Trust cpu_s only for
    serial measurement (one design at a time, e.g. a single self-eval), and read
    the aggregate cpu/wall ratio as a rough latency-vs-CPU signal, not an exact
    per-stage figure. A future fix could tag each stage_runs row as
    cpu-contended when other stages overlapped its window."""
    try:
        u = resource.getrusage(resource.RUSAGE_CHILDREN)
    except (AttributeError, ValueError):
        return 0.0
    return float(u.ru_utime + u.ru_stime)


def _record_stage_ledger(client, *, run_id, stage, **kw) -> None:
    """Best-effort write to the spend ledger's ``stage_runs`` table. Real clients
    carry a ``guard`` (SpendGuard) that owns ``record_stage``; the mock/replay
    client's guard does not, so this is a silent no-op there."""
    guard = getattr(client, "guard", None)
    if guard is None or not hasattr(guard, "record_stage"):
        return
    try:
        guard.record_stage(run_id=run_id, stage=stage, **kw)
    except Exception:  # ledger trouble must never fail a design run
        pass

_REPO = Path(__file__).resolve().parents[2]
_SPEC_DIRS = [
    _REPO / "kicraft" / "skill_assets" / "skill" / "stages",   # packaged (preferred)
    _REPO / ".claude" / "skills" / "kicraft" / "stages",       # current dev location
]

# Canonical stage -> slot model, mirroring cli_app._apply_slot's owned-field map.
SLOT_MODEL = {
    "intent": models.IntentSlot,
    "functional_spec": models.FunctionalSpec,
    "architecture": models.Architecture,
    "bom": models.BOM,
}
# wiring is not a standalone slot model: it sets bom.connections + bom.no_connect_pins.
SUPPORTED_STAGES = (*SLOT_MODEL.keys(), "wiring")
# Full design order from a brief to a synthesizable state.
DESIGN_STAGES = ("intent", "functional_spec", "architecture", "bom", "wiring")

# Tools exposed to the model during the BOM stage (OpenAI tool-spec form).
BOM_TOOLS = [
    {"type": "function", "function": {
        "name": "list_parts",
        "description": "List curated parts-library bundles available to this project "
                       "(vendored + any fetched). Returns a table with the exact symbol and "
                       "footprint strings to use verbatim in the BOM. The full table is "
                       "large — pass 'query' keywords to filter it.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "optional keywords to filter the "
                      "table, e.g. 'bnc' or 'trimmer 3296'"}}}}},
    {"type": "function", "function": {
        "name": "lookup_symbol",
        "description": "Verify a KiCad symbol in 'Library:Name' form exists and list its pins.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "e.g. 'Device:R' or "
                       "'usb-c-16p:TYPE-C-31-M-12'"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "search_symbols",
        "description": "Find the correct symbol id by keyword when you do not know the exact "
                       "'Library:Name'. Searches curated parts-library bundles first (returned "
                       "as symbol+footprint pairs — adopt the pair), then stock KiCad symbols. "
                       "Returns ids to use verbatim. Use this instead of guessing a symbol name.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "keywords, e.g. 'conn 02x08', 'crystal', "
                      "'n-channel mosfet'"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_footprints",
        "description": "Find the correct footprint id by keyword when you do not know the exact "
                       "'Library:Name'. Searches curated parts-library bundles first (returned "
                       "as symbol+footprint pairs — adopt the pair), then stock KiCad footprints. "
                       "Returns ids to use verbatim. Use this instead of guessing a footprint name.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "keywords, e.g. 'pinheader 2x08', "
                      "'barreljack', 'sot-23'"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "lookup_footprint",
        "description": "Verify a KiCad footprint in 'Library:Name' form exists and report its "
                       "pad count.",
        "parameters": {"type": "object", "properties": {
            "footprint": {"type": "string", "description": "e.g. "
                          "'Resistor_SMD:R_0603_1608Metric'"}}, "required": ["footprint"]}}},
    {"type": "function", "function": {
        "name": "lookup_lcsc_id",
        "description": "Resolve a manufacturer part number, search keyword, or pasted LCSC "
                       "id / product URL to an LCSC part number (C#####). Returns {ok, lcsc} "
                       "or a candidates list.",
        "parameters": {"type": "object", "properties": {
            "mpn": {"type": "string", "description": "MPN or keyword, e.g. 'SK-12D07VG4' or "
                    "'SPDT slide switch SMD'"}}, "required": ["mpn"]}}},
    {"type": "function", "function": {
        "name": "add_part_from_lcsc",
        "description": "Fetch a real symbol+footprint bundle from LCSC into the project parts "
                       "library. Afterwards call list_parts to get the exact '<name>:<symbol>' "
                       "and '<name>:<footprint>' strings. Not needed for core-default rows "
                       "that name a bundle: those are already in the library.",
        "parameters": {"type": "object", "properties": {
            "lcsc_id": {"type": "string", "description": "LCSC part number like C2837270"},
            "name": {"type": "string", "description": "optional library slug"}},
            "required": ["lcsc_id"]}}},
]


def _spec_text(stage: str) -> str:
    for d in _SPEC_DIRS:
        p = d / f"{stage}.md"
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return ""


def _stage_extra(stage: str) -> str:
    if stage == "intent":
        # The flat-output sentence is load-bearing: the CURRENT DESIGN STATE
        # block shows '"intent": null' as a top-level state key, and 4 of 9
        # recent boards wrapped the slot under an "intent" key to match it --
        # a guaranteed "goal: Field required" bounce (2026-07-19 review §5.1).
        return ('\n- Output the slot\'s fields directly at the JSON top level -- '
                'do NOT wrap them under an "intent" (or any other) key. '
                'Also include a top-level "project_stem" string (2-3 significant words, '
                'UPPER_SNAKE_CASE, <=32 chars) as a SIBLING key in that same flat '
                'object, e.g. {"project_stem": "LED_RING", "goal": "...", '
                '"constraints": [...], ...}. It is stripped from the slot and passed '
                'separately, per the spec.')
    if stage == "bom":
        return (
            "\n- TOOLS available this stage: list_parts (curated bundles + exact symbol/"
            "footprint strings); search_symbols / search_footprints (find a stock KiCad symbol / "
            "footprint id by keyword); lookup_symbol (verify a 'Library:Name' symbol exists + "
            "pins); lookup_footprint (verify a footprint exists + pad count); lookup_lcsc_id "
            "(MPN/keyword -> LCSC C-number); add_part_from_lcsc (fetch a real symbol+footprint "
            "bundle into the project).\n"
            "- STOCK IS A HARD GATE: never specify a low-stock part. A pick must clear at "
            "least 100 units of JLCPCB-assembly stock ('stock' in lookup_lcsc_id output / the "
            "`stock` column in list_parts) AND be orderable at the lcsc.com retail storefront "
            "('retail_stock'); the commit gate bounces a pick that fails either, forcing a "
            "costly re-draft. The selection tools already enforce this: lookup_lcsc_id vetoes a "
            "below-floor match and hands back a stock-sorted candidate list, and a list_parts "
            "bundle flagged `⚠<100` is below the floor — do NOT adopt it. Generic passives "
            "always have in-stock equivalents; never fight for a specific dry C#.\n"
            "- A SOURCING C# MUST MATCH THE FOOTPRINT: an 'LCSC C#' in sourcing_note must be "
            "the exact part the symbol/footprint were drawn for — never a merely-equivalent "
            "part (the commit gate bounces e.g. a chip resistor pinned to a trimmer footprint, "
            "or an elbow/right-angle connector pinned to a _Vertical footprint). If the right "
            "orderable part exists only as a C#, add_part_from_lcsc it and use that bundle's "
            "own symbol/footprint ids instead of a stock footprint.\n"
            "- NEVER guess a stock 'Library:Name'. If unsure of the exact symbol OR footprint id, "
            "call search_symbols / search_footprints by keyword (e.g. 'conn 02x08', "
            "'pinheader 2x08', 'barreljack') to find it; a symbol or footprint that does not "
            "resolve is rejected at commit.\n"
            "- Use a library bundle VERBATIM when one matches (e.g. usb-c-16p for a USB-C "
            "receptacle): symbol '<name>:<sym>', footprint '<name>:<fp>'.\n"
            "- Trivial / generic parts from STOCK KiCad: discrete passives (R, C, L, LED, diode) "
            "AND generic mechanical/connectors (pin headers, barrel jacks, battery holders, basic "
            "switches). Use Device:R / Device:C / Device:L / Device:LED for passives. For their "
            "footprints use these DEFAULTS VERBATIM (already verified -- do NOT call a tool to "
            "check them): R -> Resistor_SMD:R_0603_1608Metric, C -> Capacitor_SMD:C_0603_1608Metric, "
            "L -> Inductor_SMD:L_0805_2012Metric, LED -> LED_SMD:LED_0805_2012Metric, "
            "diode -> Diode_SMD:D_SOD-123. The house default for R and C is 0603; prefer 0603 or "
            "SMALLER and go to a LARGER package (0805/1206/...) ONLY when the required value or "
            "power/voltage rating is not available in 0603 or smaller. For DECOUPLING/bypass caps "
            "on an LED array, size them to sit beside the LEDs: 0603 by default, or 0402 when the "
            "LED package is smaller than 2.5 mm (e.g. WS2812 2020/1515/1313). Only call "
            "search_footprints for a passive if the board needs a DIFFERENT package (e.g. 0402, or "
            "through-hole 'LED_THT:LED_D5.0mm...'). For connectors/mechanical, call "
            "search_footprints for the exact id (e.g. "
            "'Connector_PinHeader_2.54mm:PinHeader_2x08_P2.54mm_Vertical'); when the results "
            "include a curated bundle, the bundle WINS over a stock footprint. Connectors "
            "with a mating or orientation constraint (RF/coax: BNC, SMA, U.FL; USB; card "
            "sockets) are NOT generic: use a curated bundle, or resolve the real part with "
            "lookup_lcsc_id + add_part_from_lcsc — never a stock footprint drawn for a "
            "different manufacturer's connector.\n"
            "- ICs, sensors, MCUs, regulators, or ANY part where a specific MPN matters: do NOT "
            "pick a stock symbol/footprint. Resolve the real part: lookup_lcsc_id then "
            "add_part_from_lcsc, then list_parts to read the exact '<name>:<sym>' / '<name>:<fp>' "
            "strings. Substituting a generic stock part for a specific IC is wrong.\n"
            "- SEARCH BUDGET: lookup_lcsc_id is one query + at most one retry per part (retry "
            "with the bare part family, no descriptive words). If it still misses — or reports "
            "the backend unreachable — STOP searching for that part: either ask the user for "
            "the LCSC C-number (one clarifying question can cover several parts) or use the "
            "closest stock KiCad symbol/footprint and record the substitution in assumptions.\n"
            "- NEVER SILENTLY SWAP A BRIEF-NAMED PART CLASS. When the brief itself names a "
            "specific part class or technology (e.g. 'air-core inductor', 'film capacitors', "
            "'1 A power LED', 'SMT I2C OLED module') and neither a curated bundle nor a "
            "faithful in-stock part matches that class, do NOT quietly substitute a different "
            "class (ferrite-core, ceramic, an 0805 indicator LED, a pin header). Either ask "
            "ONE clarifying question offering the concrete substitute, or — if proceeding — "
            "add an assumptions entry naming BOTH the asked-for class and the substitute "
            "('brief asked for X; substituted Y because Z'). This applies ONLY to classes the "
            "brief names explicitly, not to ordinary generic passives.\n"
            "- POLARIZED caps: an electrolytic/tantalum bulk or reservoir cap uses symbol Device:CP with a polarized footprint (a CP_* or Capacitor_Tantalum_* footprint) -- NEVER Device:C / C_* (non-polarized ceramic/film only); the symbol/footprint polarity mismatch is rejected at commit (9.25).\n"
            "- EFFICIENCY: you have a HARD budget of 6 tool-call rounds this stage. Batch every independent lookup (e.g. several "
            "search_footprints, search_symbols, or lookup_lcsc_id for different parts), "
            "request them TOGETHER in a single turn (emit multiple tool calls at once) "
            "instead of one per turn -- running out of rounds forces an immediate final "
            "answer with whatever resolved so far.\n"
            "- COMPACT OUTPUT: when the BOM has many parts (e.g. a 200-LED array + decoupling "
            "caps = 400+ parts), use COMPACT single-line JSON per part — no pretty-printing, "
            "no indentation. OMIT null fields (datasheet, mpn, sourcing_note, side, source_leaf "
            "when null). For array members that are identical except ref, emit each on one line "
            "with only the fields that differ (ref) plus value/symbol/footprint/sheet. The "
            "output token budget is finite; verbose JSON truncates and fails.\n"
            "- Every symbol AND footprint MUST resolve to a real file. When finished, output "
            "ONLY the BOM slot JSON.")
    if stage == "wiring":
        return (
            "\n- NET COVERAGE IS ENFORCED: every (ref, pin) of every part listed in "
            "extras.symbol_pinouts MUST appear either in a connections[].endpoints entry or "
            "in no_connect_pins. Omitting any pin fails the commit.\n"
            "- Use exact pin NUMBERS from extras.symbol_pinouts (never pin names).\n"
            "- Put genuinely-unused pins (USB-C SBU1/SBU2, shield, spare CC) in no_connect_pins.\n"
            "- net_name should match an architecture power_net or inter_sheet_net verbatim "
            "where applicable; connection.sheet must equal a bom part's sheet.\n"
            "- PULL-UP / PULL-DOWN: a pull resistor has TWO pins. Wire its signal-side pin "
            "on a signal net alongside the IC pin it serves, and its rail-side pin on the "
            "power/ground net (use +3V3/GND/... as connection.net_name). A power/ground "
            "connection MAY hold a single endpoint -- that lone rail pin is NOT \"dangling\". "
            "Never write a rail name into an endpoint.ref (refs are part refs like \"R1\").\n"
            "- BOM SHORTFALL = SELF-REPAIR, NOT A USER QUESTION: if the ONLY thing preventing "
            "full net coverage is that the BOM lacks a supporting passive an IC requires (a "
            "decoupling/bypass cap for a dedicated DEC/VDD/AVDD/bypass pin, a mandatory pull-up, "
            "a crystal load cap), do NOT no-connect that pin and do NOT ask the user to choose. "
            "Emit ONE blocking question tagged for automatic BOM repair — the pipeline re-runs "
            "the BOM stage to add the parts, then re-runs wiring; the user is never asked:\n"
            '    {"questions": [{"text": "<exactly what to add: how many parts, what value, '
            'which IC pins each serves>", "blocking": true, "reconcile_target": "bom"}]}\n'
            "Make the text a precise BOM instruction, not a choice. Reserve untagged questions "
            "(no reconcile_target) for genuine design-intent ambiguity the user alone can settle.\n"
            "- COMPACT OUTPUT: for a board with many repeated parts (an LED array, a channel "
            "bank), use COMPACT single-line JSON per connection -- no pretty-printing, no "
            "indentation. The output token budget is finite; verbose JSON truncates and the "
            "whole draft fails as 'no JSON in reply'.")
    return ""


def _format_core_defaults_block(rows) -> str | None:
    """Compact prompt rendering of the core-components registry: the curated
    default part per common functional block (repo catalog core_blocks.json,
    synced into /admin/core-components). The notes and price/stock snapshots
    are deliberately dropped; this table rides the user prompt through every
    BOM tool round, so it must stay small (~6.5KB for 49 rows). Rows the admin
    disabled are skipped, and rows whose default C# is missing or below the
    JLC stock floor in the current offline catalog are omitted (with a caveat
    line) so the model never adopts a dry default — no live retail lookup
    here; the lookup tool + §9.26 gate own that. None when nothing remains."""
    live = [r for r in (rows or []) if r.get("enabled", True)]
    dropped = []
    if live and jlcparts.available():
        floor = _stock_floor()
        kept = []
        for r in live:
            cid = (r.get("default_lcsc") or "").strip()
            if not cid and r.get("bundle"):
                # Bundle rows carry their C# in the vendored manifest, not in
                # default_lcsc -- they were invisible to this dry filter, and
                # the prompt tells the model NOT to re-verify bundle rows, so
                # a dry bundle default was a guaranteed §9.26 bounce on every
                # adoption (live board 631, drv8833: 3,299 assembly but 0
                # lcsc.com retail -- 2026-07-19 review §5.2).
                cid = _bundle_sourcing_lcsc(str(r.get("bundle")))
            if cid:
                hit = jlcparts.lookup(cid)
                # None = pruned out of the catalog (curated C#s are real
                # parts, so absence means effectively dry) — same fate as a
                # sub-floor row.
                if hit is None or (hit.get("stock") or 0) < floor:
                    dropped.append(
                        f"{r.get('function_key')} ({r.get('default_mpn')})")
                    continue
                # Retail: act only on a FRESH cached storefront reading (the
                # BOM tools populate the cache during normal runs); no
                # network from the prompt-assembly path.
                retail = lcsc_retail.cached_stock(cid)
                if (
                    lcsc_retail.enabled()
                    and retail is not None
                    and retail < lcsc_retail.retail_floor()
                ):
                    dropped.append(
                        f"{r.get('function_key')} ({r.get('default_mpn')}; "
                        "retail-dry)")
                    continue
            kept.append(r)
        live = kept
    if not live:
        return None
    lines = [
        "## Core component defaults",
        "Curated default part per common functional block. Precedence: a matching "
        "curated bundle in the available-parts table > the core default below > "
        "research tools. When a needed function matches a row and no stated "
        "constraint disqualifies it, adopt the default directly:",
        "- Rows with a `bundle` are ALREADY in the parts library: take the exact "
        "'<bundle>:<symbol>' / '<bundle>:<footprint>' strings from the "
        "available-parts table (extras.parts_block) or list_parts. Do NOT call "
        "add_part_from_lcsc or lookup_lcsc_id for these.",
        "- Passive series rows (no C-number) name the package to use with stock "
        "Device:R / Device:C symbols.",
        "",
        "| function_key | block | qualifier | default part | LCSC | package | bundle |",
        "|---|---|---|---|---|---|---|",
    ]
    caveats = []
    for r in live:
        cells = (r.get("function_key"), r.get("display_name"), r.get("qualifier"),
                 r.get("default_mpn"), r.get("default_lcsc"), r.get("package"),
                 r.get("bundle"))
        lines.append("| " + " | ".join(str(c) if c else "-" for c in cells) + " |")
        if re.search(r"\b(WLP|CSP|BGA)\b", r.get("package") or ""):
            caveats.append(
                f"- {r.get('function_key')} ({r.get('default_mpn')}): "
                f"{r.get('package')} is machine-assembly-only; if hand assembly "
                "is required, research an alternative instead of adopting it.")
    if caveats:
        lines += ["", "### Package caveats", *caveats]
    if dropped:
        lines += ["", f"({len(dropped)} core default(s) omitted — below the "
                      f"JLC stock floor in the current catalog: "
                      f"{', '.join(dropped)})"]
    return "\n".join(lines)


def _schema_for(stage: str) -> str:
    if stage == "wiring":
        return json.dumps({
            "type": "object",
            "properties": {
                "connections": {"type": "array",
                                "items": models.NetConnection.model_json_schema()},
                "no_connect_pins": {"type": "array",
                                    "items": models.PinEndpoint.model_json_schema()},
            },
            "required": ["connections", "no_connect_pins"],
        })
    return json.dumps(SLOT_MODEL[stage].model_json_schema())


# Hand-written compact example instance per stage. A mid-tier model pattern-
# matches one worked example far more reliably than it infers nested-optional
# shape from a raw $defs/anyOf schema dump (2026-07-19 review §7.1); the BOM
# and wiring stages carry them (the costly, retry-prone stages). Each example
# is validated against the real Pydantic models in
# tests/test_stage_driver_prompt_examples.py, so a schema change that breaks
# an example fails the suite instead of teaching the model a bounce.
_WORKED_EXAMPLES = {
    "bom": (
        '{"parts": ['
        '{"ref": "U1", "value": "AMS1117-3.3", "symbol": "ams1117-3v3:AMS1117-3.3", '
        '"footprint": "ams1117-3v3:SOT-223-3_TabPin2", "sheet": "POWER", '
        '"mpn": "AMS1117-3.3", "sourcing_note": "LCSC C6186"}, '
        '{"ref": "C1", "value": "10uF", "symbol": "Device:C", '
        '"footprint": "Capacitor_SMD:C_0603_1608Metric", "sheet": "POWER"}, '
        '{"ref": "C2", "value": "100nF", "symbol": "Device:C", '
        '"footprint": "Capacitor_SMD:C_0603_1608Metric", "sheet": "POWER"}, '
        '{"ref": "R1", "value": "10k", "symbol": "Device:R", '
        '"footprint": "Resistor_SMD:R_0603_1608Metric", "sheet": "POWER"}, '
        '{"ref": "R2", "value": "10k", "symbol": "Device:R", '
        '"footprint": "Resistor_SMD:R_0603_1608Metric", "sheet": "POWER"}, '
        '{"ref": "J1", "value": "DC barrel jack", '
        '"symbol": "Connector:Barrel_Jack_Switch", '
        '"footprint": "Connector_BarrelJack:BarrelJack_Horizontal", '
        '"sheet": "POWER"}], '
        '"ic_groups": {"U1": ["C1", "C2"]}, '
        '"thermal_refs": ["U1"], '
        '"signal_flow_order": ["U1"], '
        '"component_zones": {"J1": {"edge": "left"}}, '
        '"assumptions": ["Input jack on the left edge (defaulted)"], '
        '"substitutions": [{"wanted": "LD1117-3.3", "got": "AMS1117-3.3", '
        '"reason": "spec-named LDO out of retail stock; same pinout/dropout"}]}'
    ),
    "wiring": (
        '{"connections": ['
        '{"net_name": "VIN", "sheet": "POWER", "endpoints": '
        '[{"ref": "J1", "pin": "1"}, {"ref": "U1", "pin": "3"}, '
        '{"ref": "C1", "pin": "1"}]}, '
        '{"net_name": "+3V3", "sheet": "POWER", "endpoints": '
        '[{"ref": "U1", "pin": "2"}, {"ref": "C2", "pin": "1"}, '
        '{"ref": "R1", "pin": "1"}]}, '
        '{"net_name": "GND", "sheet": "POWER", "endpoints": '
        '[{"ref": "J1", "pin": "2"}, {"ref": "U1", "pin": "1"}, '
        '{"ref": "C1", "pin": "2"}, {"ref": "C2", "pin": "2"}, '
        '{"ref": "R2", "pin": "1"}]}, '
        '{"net_name": "NRST", "sheet": "POWER", "endpoints": '
        '[{"ref": "R1", "pin": "2"}, {"ref": "U1", "pin": "4"}]}, '
        '{"net_name": "BOOT0", "sheet": "POWER", "endpoints": '
        '[{"ref": "R2", "pin": "2"}, {"ref": "U1", "pin": "5"}]}], '
        '"no_connect_pins": [{"ref": "J1", "pin": "3"}]}'
    ),
}


def _worked_example(stage: str) -> str:
    example = _WORKED_EXAMPLES.get(stage)
    if not example:
        return ""
    return (
        "\nWorked example of a VALID slot (a tiny 3.3V-regulator board -- "
        "match its SHAPE and compact one-line-per-item style, not its "
        "content):\n" + example + "\n"
    )


def build_system(stage: str) -> str:
    spec = _spec_text(stage)
    schema = _schema_for(stage)
    return (
        f"You are the '{stage}' stage of KiCraft, a PCB design assistant running as a server "
        f"(not Claude Code). Draft the '{stage}' slot of the design state.\n\n"
        "Output ONLY a single JSON object: no prose, no markdown fences.\n\n"
        "Follow this stage specification (ignore references to SKILL.md or sub-agents; you "
        "produce the slot JSON and may use the listed tools):\n"
        f"=== SPEC ===\n{spec}\n=== END SPEC ===\n\n"
        "The JSON MUST validate against this Pydantic JSON schema (enums, required fields, and "
        f"string patterns are strict):\n{schema}\n"
        f"{_worked_example(stage)}\n"
        "Rules:\n"
        "- Output only the slot JSON object.\n"
        "- Use only allowed enum values; honor every naming pattern and uniqueness/reference "
        "constraint.\n"
        '- Every "assumptions" entry must end with "(defaulted)".\n'
        "- CLARIFYING QUESTIONS: if the brief is too ambiguous to make a sound choice that "
        "materially changes the board, you MAY ask the user instead of guessing. To ask, "
        "output ONLY this shape (no slot this turn):\n"
        '  {"questions": [{"text": "...", "options": ["a suggested answer", "..."], '
        '"blocking": true}]}\n'
        "Ask at most 3 genuinely blocking questions, and only when a wrong guess would waste "
        "a real board; otherwise choose a sensible default, record it in \"assumptions\", and "
        "output the slot."
        f"{_stage_extra(stage)}"
    )


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            text = text[a:b + 1]
    return json.loads(text)


def _fallback_stem(brief: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", brief.upper())[:3]
    return ("_".join(words)[:32]) or "PROJECT"


# LCSC ids users paste into briefs/answers — bare (C7386355) or inside an
# lcsc.com / jlcpcb.com product URL. Mirrors cli_app._LCSC_ID_RE.
_LCSC_ID_RE = re.compile(r"(?<![A-Za-z0-9])C\d{4,8}(?![A-Za-z0-9])", re.IGNORECASE)


def _bom_part_hints(*texts: str) -> str:
    """Prompt block naming any LCSC ids the user pasted ('' when none).

    A user-supplied C-number is an explicit part choice: surfacing it up front
    lets the BOM model fetch it directly instead of keyword-searching for the
    part (the dominant source of wasted tool calls and clarifying questions).
    """
    ids = sorted({m.group(0).upper()
                  for t in texts if t for m in _LCSC_ID_RE.finditer(t)})
    if not ids:
        return ""
    return ("\n\nUSER-SUPPLIED LCSC PART NUMBERS (found in the brief/answers): "
            + ", ".join(ids)
            + "\nThese are explicit part choices: call add_part_from_lcsc for each "
            "FIRST and use the fetched bundle for the matching BOM line instead "
            "of searching for alternatives.")


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    # Tag part-query telemetry from the web path so part-query-report can split
    # hosted vs offline usage (query_log reads $KICRAFT_CALLER). Honors an
    # explicit override if the environment already set one.
    env = {**os.environ, "KICRAFT_CALLER": os.environ.get("KICRAFT_CALLER", "web")}
    return subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=(str(cwd) if cwd else None))


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
    {"lookup_symbol", "lookup_footprint", "search_symbols", "search_footprints"})


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
    sep = next((i for i, ln in enumerate(lines)
                if re.match(r"\s*\|\s*-{2,}", ln)), None)
    if sep is None:  # unexpected format -> don't silently hide it, just bound it
        return (list_parts_stdout or "")[:8000]
    m = _LCSC_ID_RE.search(lcsc_id or "")
    cnum = m.group(0).upper() if m else ""
    slug = (name or "").strip().lower()
    rows = [ln for ln in lines[sep + 1:]
            if ln.lstrip().startswith("|")
            and ((cnum and cnum in ln.upper())
                 or (slug and f"`{slug}`" in ln.lower()))]
    header = "\n".join(lines[:sep + 1])
    if not rows:  # couldn't pin the new row; give the model the header + a hint
        return header + "\n(new row not located; call list_parts for the full table)"
    return (header + "\n" + "\n".join(rows))[:8000]


def _bom_executor(workspace: Path):
    """Return an executor(name, args) -> str backed by the kicraft CLI (cwd=workspace)."""
    lcsc_calls: dict[str, int] = {}  # normalized MPN -> attempts this stage (search budget)
    memo: dict[tuple[str, str], str] = {}  # read-only lookups, deduped per stage
    def execute(name: str, args: dict) -> str:
        ckey: tuple[str, str] | None = None
        if name in _MEMOIZED_BOM_TOOLS:
            arg = str(args.get("symbol") or args.get("footprint")
                      or args.get("query") or "").strip()
            ckey = (name, arg)
            if ckey in memo:  # identical lookup already answered this stage
                return memo[ckey]
        if name == "list_parts":
            # Generous cap: the vendored library alone renders ~600 chars per
            # bundle, and the core-defaults adoption rule sends the model HERE
            # for exact ids. The full table has outgrown the cap (~55KB for
            # 260+ bundles), so a truncated tail must SAY so and point at the
            # query filter — a silent cut reads as "that part doesn't exist".
            cmd = KICRAFT + ["list-parts"]
            query = str(args.get("query") or "").strip()
            if query:
                cmd.append(query)
            r = _run(cmd, workspace)
            out = r.stdout or r.stderr
            if len(out) > 40000:
                cut = out.rfind("\n", 0, 40000)
                out = out[: cut if cut > 0 else 40000] + (
                    "\n… TABLE TRUNCATED — call list_parts again with a "
                    "'query' (e.g. {\"query\": \"bnc\"}) to see the rest."
                )
            return out
        if name == "lookup_symbol":
            r = _run(KICRAFT + ["lookup-symbol", str(args.get("symbol", ""))], workspace)
            return memo.setdefault(ckey, (r.stdout or r.stderr)[:3000])
        if name == "search_symbols":
            r = _run(KICRAFT + ["search-symbols", str(args.get("query", ""))], workspace)
            return memo.setdefault(ckey, (r.stdout or r.stderr)[:3000])
        if name == "search_footprints":
            r = _run(KICRAFT + ["search-footprints", str(args.get("query", ""))], workspace)
            return memo.setdefault(ckey, (r.stdout or r.stderr)[:3000])
        if name == "lookup_footprint":
            r = _run(KICRAFT + ["lookup-footprint", str(args.get("footprint", ""))], workspace)
            return memo.setdefault(ckey, (r.stdout or r.stderr)[:3000])
        if name == "lookup_lcsc_id":
            mpn = str(args.get("mpn", ""))
            key = _normalize_mpn(mpn)
            lcsc_calls[key] = lcsc_calls.get(key, 0) + 1
            if lcsc_calls[key] > _BOM_MPN_QUERY_CAP:
                # SEARCH BUDGET exceeded for this MPN: the result cannot change
                # by retrying, so stop the wasted subprocess + the LLM round it
                # triggers. Tell the model to stop retrying this part.
                return (f"Resolution for '{mpn}' has already been attempted "
                        f"{_BOM_MPN_QUERY_CAP} times; repeating it cannot change "
                        "the answer. STOP retrying this part: use the result from "
                        "the first lookup above, ask the user for an exact LCSC "
                        "C-number (C#####), or use the closest stock KiCad "
                        "symbol/footprint and record the substitution in "
                        "assumptions. Do NOT call lookup_lcsc_id for this MPN "
                        "again this stage.")
            r = _run(KICRAFT + ["lookup-lcsc-id", mpn], workspace)
            return (r.stdout or r.stderr)[:3000]
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
            r = _run(KICRAFT + cmd, workspace)
            lp = _run(KICRAFT + ["list-parts"], workspace)
            # Return ONLY the freshly-fetched bundle's row(s), not the whole
            # ~42 KB table: the dump would otherwise persist in context for every
            # later round. The model can still call list_parts for the full table.
            return (f"add-part exit={r.returncode}\n{(r.stdout + chr(10) + r.stderr).strip()[:1500]}"
                    f"\n\nNEWLY ADDED BUNDLE (use these strings verbatim; call "
                    f"list_parts for the full library):\n"
                    f"{_new_bundle_rows(lp.stdout, lcsc_id, name)}")
        return f"unknown tool: {name}"
    return execute


def _commit(stage, slot, state_path, brief, project_stem=None, workspace=None) -> tuple[bool, dict]:
    sf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(slot, sf)
    sf.close()
    # Positionals (stage, state) BEFORE options: Python 3.12's argparse won't bind a
    # trailing optional positional that follows an option (e.g. --slot-file).
    cmd = KICRAFT + ["stage-commit", stage, str(state_path), "--slot-file", sf.name, "--no-archive"]
    if stage == "intent":
        cmd += ["--project-stem", project_stem or _fallback_stem(brief)]
    proc = _run(cmd, workspace)
    Path(sf.name).unlink(missing_ok=True)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out = {"ok": False, "errors": [proc.stdout.strip() or proc.stderr.strip()]}
    return (proc.returncode == 0 and bool(out.get("ok"))), out


def _stamp_stage_status(state_path, stage: str, ok: bool, *,
                        cost_usd=None, attempts=None, rounds=None,
                        tool_calls=None, wall_s=None, cpu_s=None, error=None) -> None:
    """Record a stage's durable outcome in state.json's stage_status block (a real
    ConversationState field, so the CLI's load/validate/dump round-trip preserves
    it). This is what lets a reopened project restore its pipeline progress
    without the ephemeral event stream. wall_s/cpu_s/rounds/tool_calls fill the
    prior measurement gap: how long a stage took, how much child CPU it burned,
    and how many tool rounds it cost (the written ledger records the same for the
    cross-run report). Tolerates a missing state.json (a first-stage failure
    before any commit). Atomic write: the web render timer reads this file
    concurrently."""
    p = Path(state_path)
    try:
        sj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        sj = {}
    entry: dict = {"ok": bool(ok),
                   "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    if cost_usd is not None:
        entry["cost_usd"] = round(float(cost_usd), 6)
    if attempts is not None:
        entry["attempts"] = int(attempts)
    if wall_s is not None:
        entry["wall_s"] = round(float(wall_s), 3)
    if cpu_s is not None:
        entry["cpu_s"] = round(float(cpu_s), 3)
    if rounds is not None:
        entry["rounds"] = int(rounds)
    if tool_calls is not None:
        entry["tool_calls"] = int(tool_calls)
    if error is not None:
        entry["error"] = str(error)
    block = sj.get("stage_status") or {}
    block[stage] = entry
    sj["stage_status"] = block
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(p, json.dumps(sj, indent=2) + "\n")


# Per-stage self-correction budget. Wiring must satisfy whole-board net coverage
# (§9.11) in a single slot; on a complex board the model needs more correction
# passes than the simpler, smaller-slot stages, so they floor higher (BOM must
# also resolve every symbol/footprint to a real library entry within its budget).
_STAGE_MIN_RETRIES = {"wiring": 4, "bom": 4}

# In-stream reasoning-loop breakout budget: when the client aborts a completion
# (finish_reason="reasoning_loop"), retry once with reasoning disabled + higher
# temperature to escape the deterministic cycle. A second loop in a row means the
# model cannot serialize even without reasoning -- fail with an explicit
# "reasoning_loop" label rather than "no JSON in reply".
_MAX_LOOP_RETRIES = 1


def _stage_max_retries(stage: str, default: int) -> int:
    return max(default, _STAGE_MIN_RETRIES.get(stage, 0))


# Tool-loop round budget for the BOM stage. The default (12) lets a weak model
# burn a dozen round-trips re-verifying a trivial 9-part BOM; 6 is plenty to
# resolve real parts, and client.chat_with_tools converges earlier when the
# model thrashes (identical-call cache + forced-final). Each stage attempt gets
# its own loop, so this is per-attempt.
_BOM_MAX_ROUNDS = 6


# Per-stage output token budget. Wiring emits the whole-board netlist in one
# slot; BOM for a large array (200 LEDs + 200 decoupling caps = 401 parts)
# emits every part in one JSON object. Both overflow the default cap and
# truncate into invalid JSON ("no JSON in reply"), so they floor higher.
_STAGE_MIN_TOKENS = {"wiring": 8192, "bom": 16384}


def _stage_max_tokens(stage: str, default: int) -> int:
    return max(default, _STAGE_MIN_TOKENS.get(stage, 0))


def _committed_bom_refs(state_path) -> list[str]:
    """Refs the committed BOM already contains -- the only refs wiring may use."""
    try:
        sj = json.loads(Path(state_path).read_text(encoding="utf-8"))
        parts = (sj.get("bom") or {}).get("parts") or []
        return sorted(str(p.get("ref")) for p in parts if isinstance(p, dict) and p.get("ref"))
    except (OSError, json.JSONDecodeError, AttributeError):
        return []


def _retry_feedback(out: dict, *, stage: str | None = None,
                    valid_refs: list[str] | None = None) -> str:
    """Self-correction message fed back to the model after a rejected commit.

    Names the exact errors and offending pins, then instructs a *preserving patch*
    (keep every already-valid entry, change only the flagged ones) rather than a
    full redraft. On large list-shaped slots like wiring, re-emitting the whole slot
    each attempt tends to regress already-correct connections (whack-a-mole), so the
    preservation instruction is the lever that helps a weaker model converge.
    """
    msg = f"stage-commit rejected that with errors: {json.dumps(out.get('errors'))}"
    if out.get("offenders"):
        msg += f"  offenders: {json.dumps(out.get('offenders'))}"
        shown = len(out.get("offenders") or [])
        total = int(out.get("offenders_total") or 0)
        if total > shown:
            # Without the total, the model fixed the visible slice, got
            # bounced with a DIFFERENT slice, and burned the retry budget
            # chasing a moving target (2026-07-19 review §5.5).
            msg += (f"  NOTE: only {shown} of {total} offenders are shown -- "
                    "fix ALL instances of this defect class across the whole "
                    "slot, not just the ones listed.")
    msg += (". Return the COMPLETE corrected slot JSON, preserving every entry that was "
            "already valid and changing ONLY the items listed above. When an offender lists "
            "'real options: ...', replace the bad id with ONE of those exact ids verbatim "
            "(do not invent or abbreviate); otherwise call search_symbols / search_footprints "
            "to find a real id. Do not drop or alter parts of the slot that were not flagged. "
            "Use COMPACT single-line JSON per part (omit null fields) so the output fits the "
            "token budget — verbose pretty-printed JSON truncates and fails. "
            "Output ONLY the slot JSON.")
    # Unknown-ref in wiring means the model tried to wire a part the BOM lacks --
    # it cannot add parts, so retrying with an invented ref just re-fails. Point
    # it at the real refs and the reconcile escape hatch so it stops thrashing and
    # escalates the deficit instead of burning the retry budget (WS6).
    if stage == "wiring" and "unknown ref" in json.dumps(out.get("errors") or ""):
        msg += (" NOTE: the wiring stage can ONLY connect refs the BOM already contains -- it "
                "CANNOT add parts. Do not invent a ref. If a part you need is genuinely missing "
                "from the BOM, do NOT wire a made-up ref: instead PARK with a single blocking "
                "question whose \"reconcile_target\" is \"bom\", naming the missing part and the "
                "IC pins it serves; the pipeline will add it and re-run wiring.")
        if valid_refs:
            msg += f" The only refs you may reference are: {valid_refs}."
    # A power/ground NAME written as an endpoint ref (the model wiring '+3V3'
    # / 'GND' as a pin's ref instead of a connection.net_name) fails the REF_RE
    # pattern. The raw Pydantic text names a regex, not the fix: rails are
    # net_name values, and a power/ground connection MAY hold a single endpoint
    # (a pull resistor's rail-side pin). Point it at the correct shape instead
    # of letting it thrash (KC-6DCV66 wired +3V3/GND as refs and died).
    if stage == "wiring":
        errs = json.dumps(out.get("errors") or "")
        rails = {
            m.group(1)
            for m in re.finditer(r"PinEndpoint\.ref '([^']+)' must match", errs)
            if models.is_power_or_ground_name(m.group(1))
        }
        if rails:
            msg += (" NOTE: " + ", ".join(sorted(rails))
                    + " is a power/ground NET NAME, not a component ref. Wire a "
                    "rail by setting connection.net_name to it and listing the part "
                    "pins as endpoints -- never put a rail name in endpoint.ref (refs "
                    "are part refs like \"R1\"). A power/ground connection MAY hold a "
                    "single endpoint (a pull-up/pull-down resistor's rail-side pin); "
                    "it is not dangling.")
    return msg


def _normalize_questions(raw_list, stage: str) -> list[dict]:
    """Coerce a model-emitted questions payload into Question-shaped dicts (so the
    state.json open_questions list stays schema-valid). Caps count and lengths."""
    out = []
    for q in raw_list:
        if isinstance(q, dict) and str(q.get("text", "")).strip():
            # reconcile_target marks a deficit the pipeline repairs itself (re-drive
            # the named stage) rather than a question for the user. Whitelisted so
            # the model can't route a park to an arbitrary/looping target.
            target = q.get("reconcile_target")
            out.append({
                "text": str(q["text"]).strip()[:500],
                "stage": stage,
                "blocking": bool(q.get("blocking", True)),
                "material": bool(q.get("material", True)),
                "options": [str(o)[:200] for o in (q.get("options") or [])][:6],
                "answer": None,
                "reconcile_target": (target if target in ("bom",) else None),
            })
    return out[:5]


def _attach_questions(state_path, stage: str, questions: list[dict]) -> None:
    """Write the stage's clarifying questions into state.json's open_questions
    (replacing any prior ones for this stage), so a reopened/parked project shows
    them. Tolerates a not-yet-created state.json (a first-stage question)."""
    try:
        sj = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        sj = {}
    kept = [q for q in (sj.get("open_questions") or []) if q.get("stage") != stage]
    sj["open_questions"] = kept + list(questions)
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(state_path, json.dumps(sj, indent=2) + "\n")


def _client_model(client) -> str | None:
    """Best-effort display name of the model a client will call (shown in the UI)."""
    return getattr(getattr(client, "s", None), "model", None)


def _design_temperature(client) -> float:
    """Sampling temperature for the design stages, from settings (default 0.2 when
    a client carries no settings, e.g. the mock). Lowering it toward 0 cuts the
    run-to-run variance that makes self-eval regressions hard to read."""
    return float(getattr(getattr(client, "s", None), "design_temperature", 0.2))

def _design_reasoning(client, stage: str) -> dict | None:
    """OpenRouter reasoning control for a design stage, from the client settings.
    A mock (or a settings object without the policy method) yields None = no
    reasoning control, which is also safe."""
    fn = getattr(getattr(client, "s", None), "design_reasoning", None)
    return fn(stage) if callable(fn) else None


# A reasoning model can burn its whole output budget re-deriving one decision and
# emit NO content (finish_reason="length" with empty text). That is not a truncated
# JSON answer; it is a stuck reasoning loop. Doubling max_tokens only feeds the loop,
# and greedy decoding (design_temperature=0.0) reproduces it identically next attempt.
# Detect the signature and break it instead: keep the budget, raise temperature to
# escape the deterministic cycle, tell the model to commit. (KC-B7MB7P: architecture
# looped for thousands of tokens on the GND-sheet question.)
_REASONING_LOOP_RETRY_MSG = (
    "You spent your entire output budget reconsidering the same decision and "
    "produced no JSON at all. Stop re-deriving it: commit to your first choice, "
    "record any default in 'assumptions' ending '(defaulted)', and output ONLY the "
    "slot JSON now."
)


def _json_failure_recovery(finish, had_content, cur_max_tokens, temperature):
    """Return (user_message, new_max_tokens, new_temperature) after a failed JSON parse.

    ``had_content`` True means the model emitted answer text (a truncated JSON answer);
    False with finish=length means it looped in reasoning and wrote nothing.
    """
    if finish == "length" and not had_content:
        return (_REASONING_LOOP_RETRY_MSG, cur_max_tokens,
                max(temperature + 0.4, 0.4))
    if finish == "length":
        return ("Your reply was cut off at the output token limit, so the JSON was "
                "truncated and invalid. The limit has been raised; output ONLY the "
                "slot JSON and keep it compact.",
                min(cur_max_tokens * 2, 32768), temperature)
    return ("That was not a single valid JSON object. Output ONLY the slot JSON.",
            cur_max_tokens, temperature)


def drive_stage(client, stage, brief, state_path, workspace, max_tokens=4096, max_retries=2,
                progress=None, answers=None, instruction=None, meta_ctx=None,
                core_defaults=None) -> dict:
    run_id = (meta_ctx or {}).get("run_id")
    t0 = time.monotonic()
    cpu0 = _child_cpu_s()
    if progress:
        progress({"kind": "stage_start", "stage": stage, "model": _client_model(client)})
    prep = _run(KICRAFT + ["stage-prep", stage, str(state_path)], workspace)
    if prep.returncode != 0:
        err = (prep.stderr.strip() or prep.stdout.strip())[:600]
        _wall = round(time.monotonic() - t0, 3)
        _cpu = round(_child_cpu_s() - cpu0, 3)
        _stamp_stage_status(state_path, stage, False, wall_s=_wall, cpu_s=_cpu)
        _record_stage_ledger(client, run_id=run_id, stage=stage, ok=False,
                             attempts=None, rounds=None, tool_calls=None,
                             wall_s=_wall, cpu_s=_cpu, cost_usd=0.0)
        if progress:
            progress({"kind": "stage_done", "stage": stage, "ok": False})
        return {"stage": stage, "commit_ok": False, "cost_usd": 0.0,
                "wall_s": _wall, "cpu_s": _cpu,
                "error": f"stage-prep failed: {err}"}
    prep_json = json.loads(prep.stdout)
    extras = prep_json.get("extras") or {}

    # Core-components registry (admin-curated default parts): rendered fresh from
    # the rows the caller fetched on this run, never persisted into state.json,
    # so admin edits land on every resume/re-drive.
    if stage in ("architecture", "bom") and core_defaults:
        block = _format_core_defaults_block(core_defaults)
        if block:
            extras["core_defaults_block"] = block

    # Bookkeeping the model has no use for stays out of its prompt.
    prompt_state = dict(prep_json["state"])
    prompt_state.pop("stage_status", None)
    # R5: For the wiring stage, project the BOM to a compact digest
    # (ref, sheet, symbol, value) instead of the full BOM slot. Pin data
    # already arrives via the symbol_pinouts extras; the full BOM's
    # sourcing/footprint/datasheet fields are noise for wiring. This is
    # PROMPT-ONLY — committed state is untouched (nothing persists
    # prompt_state).
    if stage == "wiring" and isinstance(prompt_state.get("bom"), dict):
        full_bom = prompt_state["bom"]
        prompt_state["bom"] = {
            "parts": [
                {"ref": p.get("ref"), "sheet": p.get("sheet"),
                 "symbol": p.get("symbol"), "value": p.get("value")}
                for p in full_bom.get("parts", [])
            ],
            "connections": full_bom.get("connections", []),
            "no_connect_pins": full_bom.get("no_connect_pins", []),
        }
    user = (f"PROJECT BRIEF:\n{brief}\n\n"
            f"CURRENT DESIGN STATE (JSON):\n{json.dumps(prompt_state)}")
    if extras:
        # bom carries the full parts table + core defaults (the adoption rule
        # depends on both being complete), wiring carries symbol_pinouts.
        budget = {"wiring": 40000, "bom": 20000}.get(stage, 24000)
        user += f"\n\nSTAGE EXTRAS (reference data from stage-prep):\n{json.dumps(extras)[:budget]}"
    if answers:
        qa = "\n".join(f"Q: {a.get('text', '')}\nA: {a.get('answer', '')}" for a in answers)
        user += f"\n\nThe user answered your earlier clarifying question(s):\n{qa}"
    if instruction:
        user += (f"\n\nThe user requests this change to the {stage}: {instruction}\n"
                 "Re-draft the slot to honor it, keeping everything else consistent.")
    if stage == "bom":
        user += _bom_part_hints(brief, instruction or "",
                                *(str(a.get("answer", "")) for a in (answers or [])))
    user += f"\n\nProduce the {stage} slot JSON now."

    messages = [{"role": "system", "content": build_system(stage)},
                {"role": "user", "content": user}]
    tools = BOM_TOOLS if stage == "bom" else None
    executor = _bom_executor(workspace) if stage == "bom" else None

    # Retries rebuild the conversation from this pristine base instead of
    # appending to it. chat_with_tools mutates the list it's handed (it appends
    # every tool-call turn + tool result), so a naive append-feedback-and-loop
    # re-sends the WHOLE accumulated transcript on every later attempt — BOM
    # snowballed to ~830K input tokens for ~28K output (30:1) this way. A retry
    # only needs the task, the model's last slot, and the correction: resolved
    # parts persist in the mpn cache + parts library and the executor memo
    # dedupes any re-issued lookup, so the dropped transcript is free to rebuild.
    base_messages = list(messages)

    def _lean_retry(assistant_text: str | None, user_msg: str) -> list[dict]:
        msgs = list(base_messages)
        if assistant_text:
            msgs.append({"role": "assistant", "content": assistant_text})
        msgs.append({"role": "user", "content": user_msg})
        return msgs

    total_cost = 0.0
    last: dict = {}
    cur_max_tokens = max_tokens
    temperature = _design_temperature(client)
    reasoning = _design_reasoning(client, stage)
    loop_retries = 0
    for attempt in range(max_retries + 1):
        ctx = {**(meta_ctx or {}), "stage": stage, "attempt": attempt}
        tool_calls_ct = None
        if tools:
            r = client.chat_with_tools(messages, tools, executor, max_tokens=cur_max_tokens,
                                       temperature=temperature,
                                       max_rounds=_BOM_MAX_ROUNDS, progress=progress,
                                       meta_ctx=ctx, reasoning=reasoning)
            raw, rounds = r["text"], r.get("rounds")
            tool_calls_ct = r.get("tool_calls")
            finish = r.get("finish_reason")
            total_cost += r["cost_usd"]
            had_content = bool(r["text"])
            loop_detected = bool(r.get("loop_detected"))
        else:
            res = client.chat(messages, max_tokens=cur_max_tokens, temperature=temperature,
                              progress=progress, meta_ctx=ctx, reasoning=reasoning)
            content_text = res.get("text") or ""
            raw = content_text or res.get("reasoning") or ""
            finish = res.get("finish_reason")
            rounds = None
            total_cost += res["cost_usd"]
            had_content = bool(content_text)
            loop_detected = bool(res.get("loop_detected"))

        # In-stream reasoning-loop abort: retry once with reasoning disabled and a
        # higher temperature to escape the deterministic cycle, then fail honestly.
        if loop_detected:
            last = {"error": "reasoning_loop", "reply_head": (raw or "")[:200]}
            if progress:
                progress({"kind": "retry", "stage": stage,
                          "errors": ["reasoning loop detected — retrying with reasoning disabled"]})
            if loop_retries >= _MAX_LOOP_RETRIES:
                break
            loop_retries += 1
            reasoning = {"enabled": False}
            temperature = max(temperature + 0.4, 0.4)
            messages = _lean_retry(None, _REASONING_LOOP_RETRY_MSG)
            continue

        try:
            obj = _extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            last = {"error": "no JSON in reply", "reply_head": (raw or "")[:200],
                    "rounds": rounds, "tool_calls": tool_calls_ct}
            retry_msg, cur_max_tokens, temperature = _json_failure_recovery(
                finish, had_content, cur_max_tokens, temperature)
            messages = _lean_retry(None, retry_msg)
            continue

        # A clarifying-question payload parks the stage (no slot this turn). No slot
        # model has a top-level "questions" key, so the shape is unambiguous. Never
        # re-park right after an answer (caps the back-and-forth at one round/stage).
        qpayload = obj.get("questions") if isinstance(obj, dict) else None
        if isinstance(qpayload, list) and qpayload:
            qs = _normalize_questions(qpayload, stage)
            # A reconcile_target park is the pipeline's ESCALATION (a BOM shortfall
            # wiring can't fix), not a user question. Surface it even after answers
            # were applied, so the shared bom-reconcile re-drive can add the parts
            # -- otherwise the "do not ask more questions" retry below burns the
            # stage's whole budget on a park it can never satisfy (WS6).
            is_reconcile_park = any(q.get("reconcile_target") for q in qs)
            if any(q["blocking"] for q in qs) and (not answers or is_reconcile_park):
                _attach_questions(state_path, stage, qs)
                if progress:
                    progress({"kind": "question", "stage": stage, "questions": qs})
                return {"stage": stage, "commit_ok": False, "needs_input": True,
                        "questions": qs, "cost_usd": total_cost, "attempts": attempt + 1}
            messages = _lean_retry(None,
                                   "Do not ask more questions. Apply sensible defaults (record each "
                                   "in assumptions, ending '(defaulted)') and output ONLY the slot "
                                   "JSON now.")
            continue

        project_stem = obj.pop("project_stem", None)
        ok, out = _commit(stage, dict(obj), state_path, brief, project_stem, workspace)
        if ok:
            _wall = round(time.monotonic() - t0, 3)
            _cpu = round(_child_cpu_s() - cpu0, 3)
            _stamp_stage_status(state_path, stage, True,
                                cost_usd=total_cost, attempts=attempt + 1,
                                rounds=rounds, tool_calls=tool_calls_ct,
                                wall_s=_wall, cpu_s=_cpu)
            _record_stage_ledger(client, run_id=run_id, stage=stage, ok=True,
                                 attempts=attempt + 1, rounds=rounds,
                                 tool_calls=tool_calls_ct, wall_s=_wall,
                                 cpu_s=_cpu, cost_usd=total_cost)
            if progress:
                progress({"kind": "stage_done", "stage": stage, "ok": True,
                          "cost": total_cost, "attempts": attempt + 1})
            return {"stage": stage, "commit_ok": True, "cost_usd": total_cost,
                    "attempts": attempt + 1, "rounds": rounds, "tool_calls": tool_calls_ct,
                    "wall_s": _wall, "cpu_s": _cpu, "commit": out, "slot": obj}
        last = {"commit": out}
        if progress:
            progress({"kind": "retry", "stage": stage, "errors": out.get("errors"),
                      "offenders": out.get("offenders")})
        # Echo the FULL slot the model just emitted (raw) so the preserving-patch
        # instruction in _retry_feedback can change only the flagged parts; the
        # slot is bounded by max_tokens and is far smaller than the tool transcript
        # this replaces. For wiring, pass the committed BOM refs so an unknown-ref
        # rejection can name the real refs + the reconcile escape hatch (WS6).
        _valid_refs = _committed_bom_refs(state_path) if stage == "wiring" else None
        messages = _lean_retry(raw, _retry_feedback(out, stage=stage, valid_refs=_valid_refs))

    _wall = round(time.monotonic() - t0, 3)
    _cpu = round(_child_cpu_s() - cpu0, 3)
    _stamp_stage_status(state_path, stage, False,
                        cost_usd=total_cost, attempts=max_retries + 1,
                        rounds=rounds, tool_calls=tool_calls_ct,
                        wall_s=_wall, cpu_s=_cpu, error=last.get("error"))
    _record_stage_ledger(client, run_id=run_id, stage=stage, ok=False,
                         attempts=max_retries + 1, rounds=rounds,
                         tool_calls=tool_calls_ct, wall_s=_wall, cpu_s=_cpu,
                         cost_usd=total_cost)
    if progress:
        progress({"kind": "stage_done", "stage": stage, "ok": False, "cost": total_cost})
    return {"stage": stage, "commit_ok": False, "cost_usd": total_cost,
            "attempts": max_retries + 1, "rounds": rounds, "tool_calls": tool_calls_ct,
            "wall_s": _wall, "cpu_s": _cpu, **last}


def drive_chain(stages, brief, workspace, max_tokens=4096, max_retries=2, on_stage=None,
                progress=None, client=None, answers=None, instruction=None, run_id=None,
                core_defaults=None):
    ws = Path(workspace)
    (ws / ".kicraft").mkdir(parents=True, exist_ok=True)
    state_path = ws / ".kicraft" / "state.json"
    if client is None:
        # make_client() builds Settings.from_env() itself for the live client, and
        # skips it entirely for the mock (which needs no OPENROUTER_API_KEY).
        client = make_client()
    base_ctx = {"run_id": run_id} if run_id else {}
    results = []
    for i, stage in enumerate(stages):
        # answers/instruction belong to the stage being resumed or edited, which
        # is the first stage of this chain; downstream stages re-draft cleanly.
        r = drive_stage(client, stage, brief, state_path, ws,
                        _stage_max_tokens(stage, max_tokens),
                        _stage_max_retries(stage, max_retries), progress=progress,
                        answers=(answers if i == 0 else None),
                        instruction=(instruction if i == 0 else None),
                        meta_ctx=base_ctx, core_defaults=core_defaults)
        results.append(r)
        if on_stage:
            on_stage(r)
        cost = r.get("cost_usd")
        cstr = f"${cost:.6f}" if isinstance(cost, (int, float)) else "n/a"
        tag = "ok  " if r.get("commit_ok") else "FAIL"
        extra = f" rounds={r['rounds']}" if r.get("rounds") else ""
        if r.get("tool_calls") is not None:
            extra += f" tools={r['tool_calls']}"
        line = f"  [{tag}] {stage:<16} cost={cstr}  attempts={r.get('attempts', '-')}{extra}"
        if not r.get("commit_ok"):
            line += f"\n         -> {r.get('error') or r.get('commit')}"
            if r.get("reply_head"):
                line += f"\n         reply_head: {r['reply_head']!r}"
        if r.get("needs_input"):
            line += "\n         -> parked: awaiting a clarifying answer from the user"
        print(line)
        if not r.get("commit_ok") or r.get("needs_input"):
            break
    return results, client.guard.status(), str(state_path)


class _BudgetGuard:
    """Wrap a SpendGuard with a per-run USD ceiling on top of the global ones.

    ``preflight()`` (called before every model completion) refuses once this
    run's delta past the snapshot reaches ``budget_usd``. Granularity is one
    completion, so a run may overshoot by at most a single call. Everything
    else (record / record_stage / status / spent_*) delegates to the base.
    """

    def __init__(self, base: SpendGuard, budget_usd: float):
        self._base = base
        self._budget = float(budget_usd)
        self._start = base.spent_total()

    def _delta(self) -> float:
        return self._base.spent_total() - self._start

    def preflight(self) -> None:
        self._base.preflight()
        if self._delta() >= self._budget:
            raise BudgetExceeded(
                f"run budget ${self._budget:.2f} exhausted "
                f"(spent ${self._delta():.4f})"
            )

    def __getattr__(self, name):
        return getattr(self._base, name)


def make_budget_client(budget_usd: float = 0.25):
    """A client whose guard additionally refuses once THIS run spends
    ``budget_usd`` (on top of the global daily/total ceilings). Mock/replay
    mode spends $0 and returns the plain mock client (no budget needed)."""
    if os.environ.get("KICRAFT_LLM_MODE", "live").strip().lower() in ("mock", "replay"):
        return make_client()
    settings = Settings.from_env()
    guard = SpendGuard(settings)
    if budget_usd and budget_usd > 0:
        guard = _BudgetGuard(guard, budget_usd)
    return CappedOpenRouterClient(settings, guard=guard)


def run_pipeline(brief, workspace, stages=DESIGN_STAGES, budget_usd=0.25,
                 max_tokens=4096, max_retries=2, build=True, quality="good",
                 progress=None, core_defaults=None, client=None) -> dict:
    """Full end-to-end run: drive the LLM design stages (budget-capped), then —
    if every stage committed — run the deterministic build. This is the harness
    for testing LLM-prompt / guardrail changes against a real board."""
    client = client or make_budget_client(budget_usd)
    results, guard, state_path = drive_chain(
        list(stages), brief, workspace, max_tokens=max_tokens,
        max_retries=max_retries, client=client, progress=progress,
        core_defaults=core_defaults)
    all_committed = (
        len(results) == len(stages)
        and all(r.get("commit_ok") for r in results)
    )
    build_rc = None
    if build and all_committed:
        build_rc = _run(
            KICRAFT + ["build", ".kicraft/state.json", "generated",
                       "--no-archive", "--quality", quality],
            cwd=Path(workspace),
        ).returncode
    return {
        "stages": results,
        "all_committed": all_committed,
        "guard": guard,
        "state_path": str(state_path),
        "build_rc": build_rc,
    }


def drive_replay(state_path, stage, budget_usd=0.25, max_retries=2,
                 progress=None, core_defaults=None, client=None) -> dict:
    """Re-run ONE design stage from a frozen, already-committed state.json — the
    LLM-side repro harness for prompt/guardrail changes (mirrors ``cli_app
    replay`` for the deterministic place/route stages). Copies the state into a
    temp workspace (the source is never mutated), reads the brief from it, and
    drives ``stage`` with a budget-capped client."""
    src = Path(state_path).expanduser().resolve()
    if not src.is_file():
        return {"error": f"state.json not found: {src}"}
    try:
        state = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"error": f"could not read {src}: {e}"}
    brief = ((state.get("intent") or {}).get("goal") or "").strip()
    if not brief:
        brief_txt = src.parent.parent / "brief.txt"
        if brief_txt.is_file():
            brief = brief_txt.read_text(encoding="utf-8").strip()
    if not brief:
        return {"error": f"no brief recoverable from {src} (intent.goal or brief.txt)"}
    if stage not in SUPPORTED_STAGES:
        return {"error": f"unsupported stage {stage!r}; supported: {list(SUPPORTED_STAGES)}"}

    tmp = Path(tempfile.mkdtemp(prefix="kc-replay-"))
    (tmp / ".kicraft").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, tmp / ".kicraft" / "state.json")

    client = client or make_budget_client(budget_usd)
    results, guard, spath = drive_chain(
        [stage], brief, tmp, max_retries=max_retries,
        client=client, progress=progress, core_defaults=core_defaults)
    return {
        "brief": brief,
        "workspace": str(tmp),
        "state_path": str(spath),
        "stage": results[0] if results else None,
        "guard": guard,
        "all_committed": bool(results) and results[0].get("commit_ok"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="kicraft.stage_driver",
        description="Drive KiCraft design stages through the capped gateway.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run", help="drive the LLM design stages (optionally + build) from a brief")
    p_run.add_argument("--brief", required=True, help="the user's project description")
    p_run.add_argument("--workspace", required=True,
                       help="project dir (holds .kicraft/state.json)")
    p_run.add_argument("--stages", default=",".join(DESIGN_STAGES),
                       help="comma-separated stages in order")
    p_run.add_argument("--max-tokens", type=int, default=4096)
    p_run.add_argument("--max-retries", type=int, default=2,
                       help="self-correction attempts per stage after a rejected commit")
    p_run.add_argument("--budget", type=float, default=0.25,
                       help="per-run USD cap on LLM spend (default $0.25)")
    p_run.add_argument("--no-build", action="store_true",
                       help="stop after the LLM stages (skip the deterministic build)")
    p_run.add_argument("--quality", choices=["fast", "draft", "good", "best"],
                       default="good")
    p_run.set_defaults(func=_cmd_run)

    p_replay = sub.add_parser(
        "replay", help="re-run ONE LLM stage from a frozen, committed state.json")
    p_replay.add_argument("--state", required=True,
                          help="path to a committed state.json")
    p_replay.add_argument("--stage", required=True,
                          help=f"stage to re-drive; one of {list(SUPPORTED_STAGES)}")
    p_replay.add_argument("--max-retries", type=int, default=2)
    p_replay.add_argument("--budget", type=float, default=0.25)
    p_replay.set_defaults(func=_cmd_replay)

    args = ap.parse_args(argv)
    return args.func(args)


def _cmd_run(args) -> int:
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in SUPPORTED_STAGES]
    if bad:
        print(f"unsupported stage(s): {bad}; supported: {list(SUPPORTED_STAGES)}",
              file=sys.stderr)
        return 2
    print(f"driving {stages} (LLM budget ${args.budget:.2f}) for: {args.brief!r}\n")
    out = run_pipeline(
        args.brief, Path(args.workspace), stages=stages,
        budget_usd=args.budget, max_tokens=args.max_tokens,
        max_retries=args.max_retries, build=not args.no_build,
        quality=args.quality)
    guard = out["guard"]
    print(f"\ncommitted stages: {'all' if out['all_committed'] else 'partial/failed'}")
    print(f"build rc: {out['build_rc'] if out['build_rc'] is not None else 'skipped'}")
    print(f"total spent: ${guard['spent_total_usd']:.6f}  "
          f"(today remaining ${guard['daily_remaining_usd']:.4f})")
    print(f"state: {out['state_path']}")
    return 0 if out["all_committed"] else 1


def _cmd_replay(args) -> int:
    print(f"replaying stage {args.stage!r} from {args.state!r} "
          f"(LLM budget ${args.budget:.2f})\n")
    out = drive_replay(args.state, args.stage,
                       budget_usd=args.budget, max_retries=args.max_retries)
    if "error" in out:
        print(f"replay failed: {out['error']}", file=sys.stderr)
        return 2
    # drive_chain already printed the per-stage [ok/FAIL] line; only add the
    # replay-specific footer here.
    print(f"\nworkspace: {out['workspace']}  (source state untouched)")
    print(f"state: {out['state_path']}")
    return 0 if out["all_committed"] else 1


if __name__ == "__main__":
    sys.exit(main())
