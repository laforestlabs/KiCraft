"""Stage specification loading and system-prompt construction."""
from __future__ import annotations

import json
import re
from pathlib import Path

from kicraft.parts_library import jlcparts, lcsc_retail

from .config import STAGE_COLLECTION_BOUNDS, CollectionBound
from .pricing import _stock_floor
from .stage_contracts import StageResponseContract

def _bundle_sourcing_lcsc(bundle: str) -> str:
    """The vendored bundle's pinned sourcing C#, or '' (best-effort, offline)."""
    try:
        manifest = _REPO / "kicraft" / "parts_library" / bundle / "manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return str((data.get("sourcing") or {}).get("lcsc") or "").strip()
    except Exception:
        return ""

_REPO = Path(__file__).resolve().parents[2]
_SPEC_DIRS = [
    _REPO / "kicraft" / "skill_assets" / "skill" / "stages",  # packaged (preferred)
    _REPO / ".claude" / "skills" / "kicraft" / "stages",  # current dev location
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
        return (
            "\n- Output the slot's fields directly at the JSON top level -- "
            'do NOT wrap them under an "intent" (or any other) key. '
            'Also include a top-level "project_stem" string (2-3 significant words, '
            "UPPER_SNAKE_CASE, <=32 chars) as a SIBLING key in that same flat "
            'object, e.g. {"project_stem": "LED_RING", "goal": "...", '
            '"constraints": [...], ...}. It is stripped from the slot and passed '
            "separately, per the spec."
        )
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
            "- COMPACT OUTPUT: keep heterogeneous components in `parts`. Put repeated "
            "identical components in `part_runs`: either `refs` with an explicit ordered "
            "reference list, or `ref_prefix` + integer `start`/`end`, plus one shared "
            "value/symbol/footprint/sheet payload. Omit null fields. Canonical expansion "
            "is deterministic and still enforces 500 total and 450 per sheet before commit.\n"
            "- SHEET NAMES ARE CLOSED: every parts[].sheet and part_runs[].sheet must copy one "
            "architecture.sheets[].name verbatim; never abbreviate, correct, or invent a sheet name.\n"
            "- Every symbol AND footprint MUST resolve to a real file. When finished, output "
            "ONLY the BOM slot JSON."
        )
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
            'connection MAY hold a single endpoint -- that lone rail pin is NOT "dangling". '
            'Never write a rail name into an endpoint.ref (refs are part refs like "R1").\n'
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
            "whole draft fails as 'no JSON in reply'."
        )
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
                    dropped.append(f"{r.get('function_key')} ({r.get('default_mpn')})")
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
                    dropped.append(f"{r.get('function_key')} ({r.get('default_mpn')}; retail-dry)")
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
        cells = (
            r.get("function_key"),
            r.get("display_name"),
            r.get("qualifier"),
            r.get("default_mpn"),
            r.get("default_lcsc"),
            r.get("package"),
            r.get("bundle"),
        )
        lines.append("| " + " | ".join(str(c) if c else "-" for c in cells) + " |")
        if re.search(r"\b(WLP|CSP|BGA)\b", r.get("package") or ""):
            caveats.append(
                f"- {r.get('function_key')} ({r.get('default_mpn')}): "
                f"{r.get('package')} is machine-assembly-only; if hand assembly "
                "is required, research an alternative instead of adopting it."
            )
    if caveats:
        lines += ["", "### Package caveats", *caveats]
    if dropped:
        lines += [
            "",
            f"({len(dropped)} core default(s) omitted — below the "
            f"JLC stock floor in the current catalog: "
            f"{', '.join(dropped)})",
        ]
    return "\n".join(lines)

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


def _collection_bounds_sentence(bounds: tuple[CollectionBound, ...]) -> str:
    """Render collection policies once for both normal and recovery prompts."""
    clauses: list[str] = []
    for bound in bounds:
        clause = f"The `{bound.field}` collection must contain at most {bound.total} items total"
        if bound.per_group is not None and bound.group_key is not None:
            clause += f" and at most {bound.per_group} items per `{bound.group_key}`"
        clauses.append(clause + ".")
    return " ".join(clauses)


def _bounded_output_contract(stage: str, bounds: tuple[CollectionBound, ...] | None = None) -> str:
    configured = STAGE_COLLECTION_BOUNDS.get(stage, ()) if bounds is None else bounds
    sentence = _collection_bounds_sentence(configured)
    if not sentence:
        return ""
    return (
        "\n\n=== BOUNDED OUTPUT POLICY ===\n"
        f"{sentence} Emit only components required by the architecture and intent; "
        "do not pad the collection with speculative or repeated parts."
        "\n=== END BOUNDED OUTPUT POLICY ==="
    )


def build_system(contract: StageResponseContract, collection_bounds: tuple[CollectionBound, ...] | None = None) -> str:
    stage = contract.stage
    spec = _spec_text(stage)
    schema = json.dumps(contract.schema)
    return (
        f"You are the '{stage}' stage of KiCraft, a PCB design assistant running as a server "
        f"(not Claude Code). Draft the '{stage}' slot of the design state.\n\n"
        "Output ONLY a single JSON object: no prose, no markdown fences.\n\n"
        "Follow this stage specification (ignore references to SKILL.md or sub-agents; you "
        "produce the slot JSON and may use the listed tools):\n"
        f"=== SPEC ===\n{spec}\n=== END SPEC ==="
        f"{_bounded_output_contract(stage, collection_bounds)}\n\n"
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
        'a real board; otherwise choose a sensible default, record it in "assumptions", and '
        "output the slot."
        f"{_stage_extra(stage)}"
    )

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
    ids = sorted({m.group(0).upper() for t in texts if t for m in _LCSC_ID_RE.finditer(t)})
    if not ids:
        return ""
    return (
        "\n\nUSER-SUPPLIED LCSC PART NUMBERS (found in the brief/answers): "
        + ", ".join(ids)
        + "\nThese are explicit part choices: call add_part_from_lcsc for each "
        "FIRST and use the fetched bundle for the matching BOM line instead "
        "of searching for alternatives."
    )
