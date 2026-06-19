"""`kicraft` — non-interactive helpers used by the Claude Code skill.

The skill at ``.claude/skills/kicraft/`` drives the LLM conversation;
this CLI handles the deterministic side: validating a state file,
listing reusable leaves the skill can show to the model, and emitting
the KiCad project once the four slots are populated.

Subcommands:

- ``validate STATE.json`` -- load through ``ConversationState`` and, if
  the architecture slot is present and the leaf library is non-empty,
  also run the library interface check.
- ``list-leaves`` -- print the "Available leaves" block the architecture
  stage used to inject. The skill pastes this into its context before
  asking the model to fill the architecture slot.
- ``synthesize STATE.json OUT_DIR [--smoke]`` -- run the mechanical
  synthesizer. Wraps ``kicraft.design.synthesize.run``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from kicraft.parts_library import jlcparts

from .library import (
    ArchitectureLibraryError,
    _format_available_leaves_block,
    _format_available_parts_block,
    _load_library_leaves,
    _load_library_parts,
    _validate_library_picks,
)
from .models import (
    Architecture,
    BOM,
    ChatMsg,
    ConversationState,
    FunctionalSpec,
    IntentSlot,
    Question,
)
from .synthesize import SynthesisInputError, run as run_synth
from .synthesis.symbol_library import search_symbols
from .synthesis.footprint_library import (
    FootprintNotFoundError,
    lookup_footprint,
    search_footprints,
)
from .synthesis.symbol_pinout import SymbolNotFoundError, lookup_pins
from .synthesis.parts_lookup import (
    LibraryNotFoundError,
    resolve_footprint_library_path,
)
from .synthesis.validation import (
    CheckResult,
    SynthesisValidationError,
    check_family_wiring_contracts,
    check_inter_sheet_nets_realized,
    check_net_coverage,
    check_no_dangling_signal_nets,
    check_pin_existence,
    check_power_pin_polarity,
    check_rf_feed_isolation,
    check_sheets_have_parts,
    check_single_net_per_pin,
    check_two_terminal_self_short,
)
from kicraft.parts_library import Maturity
from kicraft.parts_library.query_log import record as _log_query

# `placement` is deterministic (user placement rules, no LLM); committing it
# invalidates nothing upstream and merely requires a rebuild to take effect.
KNOWN_STAGES = ("intent", "functional_spec", "architecture", "bom", "wiring",
                "placement")


_SAFE_STEM_RE = re.compile(r"[^A-Z0-9_]")


def _default_archive_root() -> Path:
    return Path.home() / ".kicraft" / "sessions"


def _utc_compact_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resemble_candidates(ident: str, search_fn, limit: int = 6) -> list[str]:
    """Real ids resembling an unresolved ``Library:Name``, kept in-category.

    Surfaced in commit-rejection feedback so a weak model picks a real id in one
    guided round instead of repeatedly guessing -- the dominant BOM-stage retry
    cost. Queries most-specific first (the full name, catching a truncation like
    ``SW_SPST_PTS645`` -> ``...PTS645Sx43SMTR92`` or a pin-count like
    ``Conn_02x08``), then broadens to the library family token (``Inductor_THT``
    -> ``Inductor``) so a hallucinated variant (``L_Radial_D50.0mm``) returns
    real *inductors*, not whatever shares a generic word.
    """
    library, _, name = ident.partition(":")
    lib_fam = re.split(r"[^A-Za-z]+", library)[0] if library else ""
    name_toks = [t for t in re.split(r"[^A-Za-z0-9]+", name or ident) if len(t) > 1]
    alpha = [t for t in name_toks if not any(c.isdigit() for c in t)]
    queries: list[str] = []
    if name_toks:
        queries.append(" ".join(name_toks))               # exact-ish: truncation / pin-count
    if lib_fam and alpha:
        queries.append(" ".join([lib_fam] + alpha[:2]))   # category + family word
    if lib_fam:
        queries.append(lib_fam)                           # category only
    if alpha:
        queries.append(" ".join(alpha[:2]))
    for q in queries:
        hits = search_fn(q, limit=limit)
        if hits:
            return hits[:limit]
    return []


def _footprint_candidates(fp: str, limit: int = 6) -> list[str]:
    return _resemble_candidates(fp, search_footprints, limit)


def _symbol_candidates(sym: str, limit: int = 6) -> list[str]:
    return _resemble_candidates(sym, search_symbols, limit)


def _candidate_hint(cands: list[str]) -> str:
    return f" -- real options: {', '.join(cands)}" if cands else ""


def _unresolved_footprints(bom, project_root: Path) -> list[str]:
    """Return a human-readable list of BOM parts whose ``footprint`` does
    not resolve to a real ``.kicad_mod`` on disk (across the four parts-
    library tiers + stock KiCad). An empty list means every footprint
    resolves. Catches LLM footprint-name hallucination (e.g. a plausible
    truncation like ``SW_SPST_PTS645`` for ``SW_SPST_PTS645Sx43SMTR92``).

    Each offender carries up to a few real look-alike footprint ids so the
    commit-rejection feedback can steer the model to a valid pick in one round.
    """
    bad: list[str] = []
    for part in bom.parts:
        fp = part.footprint or ""
        library, _, name = fp.partition(":")
        if not library or not name:
            bad.append(f"{part.ref}: footprint {fp!r} is not 'Library:Name'"
                       + _candidate_hint(_footprint_candidates(fp)))
            continue
        try:
            pretty = resolve_footprint_library_path(library, project_root=project_root)
        except LibraryNotFoundError:
            bad.append(f"{part.ref}: footprint library {library!r} not found (footprint {fp!r})"
                       + _candidate_hint(_footprint_candidates(fp)))
            continue
        if not (pretty / f"{name}.kicad_mod").is_file():
            bad.append(f"{part.ref}: no '{name}.kicad_mod' in {pretty} (footprint {fp!r})"
                       + _candidate_hint(_footprint_candidates(fp)))
    return bad


def _unresolved_symbols(bom) -> list[str]:
    """Return BOM parts whose ``symbol`` does not resolve to a real pin inventory.

    Runs the SAME check the wiring stage-prep runs (``lookup_pins`` over each
    distinct symbol), but at BOM-commit time, so a hallucinated symbol name is
    rejected while the model still has the BOM lookup tools to fix it, instead of
    cascading into an unrecoverable wiring stage-prep failure. Empty list means
    every symbol resolves.
    """
    bad: list[str] = []
    seen: set[str] = set()
    for part in bom.parts:
        sym = part.symbol
        if sym in seen:
            continue
        seen.add(sym)
        try:
            info = lookup_pins(sym)
        except (SymbolNotFoundError, ValueError) as e:
            bad.append(f"{part.ref}: symbol {sym!r} did not resolve ({e})"
                       + _candidate_hint(_symbol_candidates(sym)))
            continue
        if not info.get("pins"):
            bad.append(f"{part.ref}: symbol {sym!r} resolved but exposes no pins")
    return bad


def _read_or_create_session_id(state_dir: Path, state: ConversationState) -> str:
    sid_path = state_dir / "session_id"
    if sid_path.exists():
        existing = sid_path.read_text().strip()
        if existing:
            return existing
    stem = (state.project_stem or "UNNAMED").upper()
    stem = _SAFE_STEM_RE.sub("_", stem)[:32] or "UNNAMED"
    sid = f"{_utc_compact_now()}_{stem}"
    state_dir.mkdir(parents=True, exist_ok=True)
    sid_path.write_text(sid + "\n")
    return sid


def _write_manifest(
    archive_dir: Path,
    session_id: str,
    state: ConversationState,
    synth_results: list[CheckResult] | None,
    artifacts_subdir: str | None,
) -> None:
    slots_filled = [
        name
        for name in ("intent", "functional_spec", "architecture", "bom")
        if getattr(state, name) is not None
    ]
    blocking_qs = [q for q in state.open_questions if q.blocking]
    history_ts = [m.timestamp for m in state.history if getattr(m, "timestamp", None)]
    manifest = {
        "session_id": session_id,
        "project_stem": state.project_stem,
        "started_at": history_ts[0].isoformat() if history_ts else None,
        "ended_at": history_ts[-1].isoformat() if history_ts else None,
        "slots_filled": slots_filled,
        "open_questions": len(state.open_questions),
        "blocking_questions": len(blocking_qs),
        "synth_ok": (
            None if synth_results is None else all(r.ok for r in synth_results)
        ),
        "synth_results": (
            None
            if synth_results is None
            else [
                {"name": r.name, "ok": r.ok, "message": r.message}
                for r in synth_results
            ]
        ),
        "artifacts_subdir": artifacts_subdir,
        "archived_at": _utc_compact_now(),
    }
    (archive_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )


def _archive_session(
    state_path: Path,
    state: ConversationState,
    archive_root: Path,
    *,
    synth_dir: Path | None = None,
    synth_results: list[CheckResult] | None = None,
) -> Path:
    """Snapshot .kicraft/ into <archive_root>/<session_id>/.

    Always copies state.json, session_id, and log.jsonl (if present).
    When `synth_dir` is supplied, the synthesized project tree is copied
    under `<archive>/generated/<synth_dir.name>/`. `feedback.md` in the
    destination is never touched so user notes survive re-archives.
    """
    state_dir = state_path.parent
    session_id = _read_or_create_session_id(state_dir, state)
    dest = archive_root / session_id
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy2(state_path, dest / "state.json")
    sid_src = state_dir / "session_id"
    if sid_src.exists():
        shutil.copy2(sid_src, dest / "session_id")
    log_src = state_dir / "log.jsonl"
    if log_src.exists():
        shutil.copy2(log_src, dest / "log.jsonl")

    artifacts_subdir: str | None = None
    if synth_dir is not None and synth_dir.exists():
        gen_dest = dest / "generated" / synth_dir.name
        gen_dest.parent.mkdir(parents=True, exist_ok=True)
        if gen_dest.exists():
            shutil.rmtree(gen_dest)
        shutil.copytree(synth_dir, gen_dest)
        artifacts_subdir = f"generated/{synth_dir.name}"

    _write_manifest(dest, session_id, state, synth_results, artifacts_subdir)
    return dest


def _load_state(path: Path) -> ConversationState:
    data = json.loads(path.read_text())
    return ConversationState.model_validate(data)


def _cmd_validate(args: argparse.Namespace) -> int:
    state_path = Path(args.state)
    try:
        state = _load_state(state_path)
    except ValidationError as e:
        print(f"schema validation failed:\n{e}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError) as e:
        print(f"could not read {state_path}: {e}", file=sys.stderr)
        return 2

    if state.architecture is not None:
        leaves = _load_library_leaves()
        if leaves:
            try:
                _validate_library_picks(state.architecture, leaves)
            except ArchitectureLibraryError as e:
                print(f"library validation failed: {e}", file=sys.stderr)
                return 3

    if state.architecture is not None and state.bom is not None:
        sp = check_sheets_have_parts(state.architecture, state.bom)
        if not sp.ok:
            print(f"{sp.name}: {sp.message}", file=sys.stderr)
            for o in sp.offenders[:20]:
                print(f"  - {o}", file=sys.stderr)
            return 3

    if state.bom is not None and state.bom.connections:
        checks = [
            check_pin_existence(state.bom),
            check_net_coverage(state.bom),
            check_power_pin_polarity(state.bom),
            check_two_terminal_self_short(state.bom),
            check_rf_feed_isolation(state.bom),
            check_single_net_per_pin(state.bom),
            check_family_wiring_contracts(state.bom),
        ]
        if state.architecture is not None:
            checks.append(
                check_inter_sheet_nets_realized(state.architecture, state.bom)
            )
            # §9.15 inverse: a signal net wired to a single pin that was never
            # declared inter-sheet connects to nothing (the SOIL_MOISTURE_BLE
            # USB D+/D- dangle).
            checks.append(
                check_no_dangling_signal_nets(state.architecture, state.bom)
            )
        for check in checks:
            if not check.ok:
                print(f"{check.name}: {check.message}", file=sys.stderr)
                for o in check.offenders[:20]:
                    print(f"  - {o}", file=sys.stderr)
                return 3

    filled = [
        name
        for name in ("intent", "functional_spec", "architecture", "bom")
        if getattr(state, name) is not None
    ]
    blocking_qs = [q for q in state.open_questions if q.blocking]
    print(
        json.dumps(
            {
                "ok": True,
                "project_stem": state.project_stem,
                "slots_filled": filled,
                "open_questions": len(state.open_questions),
                "blocking_questions": len(blocking_qs),
            },
            indent=2,
        )
    )
    return 0


def _lib_prefix(ref: str) -> str | None:
    """The 'Library' part of a 'Library:Name' id, or None if unprefixed.

    Lets the query log attribute a symbol/footprint lookup to a specific
    library: a curated bundle name, or a stock KiCad lib like 'Device'."""
    return ref.split(":", 1)[0] if ref and ":" in ref else None


def _cmd_lookup_symbol(args: argparse.Namespace) -> int:
    try:
        info = lookup_pins(args.symbol)
    except SymbolNotFoundError as e:
        _log_query("lookup_symbol", outcome="miss", query=args.symbol,
                   lib=_lib_prefix(args.symbol))
        print(str(e), file=sys.stderr)
        return 2
    except ValueError as e:
        _log_query("lookup_symbol", outcome="error", query=args.symbol)
        print(str(e), file=sys.stderr)
        return 2
    _log_query("lookup_symbol", outcome="hit", query=args.symbol,
               lib=_lib_prefix(args.symbol))
    print(json.dumps(info, indent=2))
    return 0


def _cmd_search_symbols(args: argparse.Namespace) -> int:
    matches = search_symbols(args.query, limit=args.limit)
    _log_query("search_symbols", outcome=("hit" if matches else "miss"),
               query=args.query, n_matches=len(matches))
    if not matches:
        print(f"no stock KiCad symbols match {args.query!r}; try fewer or broader terms",
              file=sys.stderr)
        return 0
    for sym in matches:
        print(sym)
    return 0


def _cmd_lookup_footprint(args: argparse.Namespace) -> int:
    try:
        info = lookup_footprint(args.footprint)
    except FootprintNotFoundError as e:
        _log_query("lookup_footprint", outcome="miss", query=args.footprint,
                   lib=_lib_prefix(args.footprint))
        print(str(e), file=sys.stderr)
        return 2
    except ValueError as e:
        _log_query("lookup_footprint", outcome="error", query=args.footprint)
        print(str(e), file=sys.stderr)
        return 2
    _log_query("lookup_footprint", outcome="hit", query=args.footprint,
               lib=_lib_prefix(args.footprint))
    print(json.dumps(info, indent=2))
    return 0


def _cmd_search_footprints(args: argparse.Namespace) -> int:
    matches = search_footprints(args.query, limit=args.limit)
    _log_query("search_footprints", outcome=("hit" if matches else "miss"),
               query=args.query, n_matches=len(matches))
    if not matches:
        print(f"no stock KiCad footprints match {args.query!r}; try fewer or broader terms",
              file=sys.stderr)
        return 0
    for fp in matches:
        print(fp)
    return 0


def _cmd_electrical_review(args: argparse.Namespace) -> int:
    """Layer 3: independent LLM electrical review of a committed design.

    Reviews the BOM + function-named netlist for topology/value/completeness
    defects the deterministic §9 gates cannot judge. Prints a JSON report.
    Exits 3 when --fail-on-blocker is set and a blocker is found.
    """
    from .synthesis.electrical_review import (
        build_design_digest,
        has_blocker,
        review_design,
    )

    state_path = Path(args.state)
    try:
        state = _load_state(state_path)
    except ValidationError as e:
        print(f"schema validation failed:\n{e}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError) as e:
        print(f"could not read {state_path}: {e}", file=sys.stderr)
        return 2

    if state.bom is None or not state.bom.connections:
        print(json.dumps({"ok": False, "error": "no wired BOM to review",
                          "findings": []}, indent=2))
        return 0

    project_root = state_path.resolve().parent.parent
    digest = build_design_digest(state, project_root=project_root)
    if args.digest_only:
        print(digest)
        return 0

    from kicraft.server.client import make_client
    from kicraft.server.config import Settings

    settings = Settings.from_env()
    client = make_client(settings.for_review())
    # Default to the design model (deepseek-v4-flash -- cheap) but give it a
    # higher thinking budget; the review is a one-shot reasoning task.
    model = args.model or settings.review_model or settings.model
    reasoning = settings.review_reasoning()
    result = review_design(client, digest, model=model, reasoning=reasoning,
                           max_tokens=settings.review_max_tokens)
    print(json.dumps({
        "ok": result["ok"],
        "error": result["error"],
        "cost_usd": round(result["cost_usd"], 6),
        "findings": result["findings"],
    }, indent=2))
    if args.fail_on_blocker and result["ok"] and has_blocker(result["findings"]):
        return 3
    return 0


def _cmd_list_leaves(_: argparse.Namespace) -> int:
    leaves = _load_library_leaves()
    block = _format_available_leaves_block(leaves)
    if block is None:
        print("(no leaves available in the library)")
        return 0
    print(block)
    return 0


def _cmd_list_parts(_: argparse.Namespace) -> int:
    active, _broken = _load_library_parts(Path.cwd())
    _log_query("list_parts", outcome="listed", n_active=len(active))
    block = _format_available_parts_block(active)
    if block is None:
        print("(no parts available in the library)")
        return 0
    print(block)
    return 0


# LCSC part numbers as users paste them — bare (C7386355) or inside an
# lcsc.com / jlcpcb.com product URL (where they follow '_' or '/'). The
# lookarounds reject MPNs that merely embed a C+digits run (e.g. C8051F320).
# stage_driver._LCSC_ID_RE mirrors this pattern for brief/answer scanning.
_LCSC_ID_RE = re.compile(r"(?<![A-Za-z0-9])C\d{4,8}(?![A-Za-z0-9])", re.IGNORECASE)

_EASYEDA_SEARCH_URL = "https://easyeda.com/api/components/search"
# easyeda.com serves this search to browsers only: bare library User-Agents
# get a WAF 403, a normal browser string does not. Same host the by-C# symbol/
# footprint/price fetches already rely on (jlcpcb.com's own search API is
# hard-blocked from datacenter IPs).
_BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def _parse_easyeda_search(payload: dict) -> list[dict]:
    """Flatten an easyeda.com components/search payload into candidate rows
    ({lcsc, model, brand, package, description}) so selection and printing
    stay source-agnostic."""
    lists = ((payload or {}).get("result") or {}).get("lists") or {}
    rows: list[dict] = []
    seen: set[str] = set()
    for source in ("lcsc", "SMT"):
        for r in lists.get(source) or []:
            number = ((r.get("lcsc") or {}).get("number")
                      or (r.get("szlcsc") or {}).get("number"))
            if not number or number in seen:
                continue
            seen.add(number)
            data_str = r.get("dataStr")
            c_para = ((data_str.get("head") or {}).get("c_para") or {}
                      if isinstance(data_str, dict) else {})
            rows.append({
                "lcsc": number,
                "model": r.get("title") or c_para.get("name"),
                "brand": c_para.get("Manufacturer"),
                "package": c_para.get("package"),
                "description": (r.get("description") or "")[:120] or None,
            })
    return rows


def _search_easyeda_components(keyword: str, page_size: int = 10) -> list[dict] | None:
    """Keyword-search the LCSC catalog via easyeda.com.

    Returns parsed candidate rows, or None when the backend is unreachable —
    callers must distinguish that from [] (a genuine no-match) so the BOM
    agent stops burning retries on keyword variants when the network is down.
    """
    data = urllib.parse.urlencode({
        "wd": keyword, "type": "3", "doctype[]": "2",
        "page": "1", "pageSize": str(page_size),
    }).encode()
    req = urllib.request.Request(
        _EASYEDA_SEARCH_URL, data=data,
        headers={"User-Agent": _BROWSER_UA, "Accept": "application/json",
                 "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or not payload.get("success"):
        return None
    return _parse_easyeda_search(payload)


def _pick_lcsc(mpn: str, results: list[dict]) -> dict | None:
    """Choose the single best search result for `mpn`, or None if
    ambiguous. Prefer an exact (case-insensitive) match on the part's
    model/MPN, breaking ties by stock (desc) then Basic-over-Extended. With
    no exact match but exactly one result, take it; otherwise return None so
    the caller surfaces the candidate list rather than guessing wrong."""
    target = (mpn or "").strip().upper()
    exact = [r for r in results if (r.get("model") or "").strip().upper() == target]
    if exact:
        exact.sort(key=lambda r: (-(r.get("stock") or 0), r.get("type") != "Basic"))
        best = exact[0]
        # A known-zero-stock exact hit while other candidates ARE in stock is
        # usually a placeholder catalog row (e.g. a bare family name like
        # "VL53L1X"); surface the candidate list so the orderable real MPN
        # wins. stock=None (source doesn't report it) never triggers this.
        if best.get("stock") == 0 and any(
            (r.get("stock") or 0) > 0 for r in results if r is not best
        ):
            return None
        return best
    if len(results) == 1:
        return results[0]
    return None


def _cmd_lookup_lcsc_id(args: argparse.Namespace) -> int:
    """Resolve an MPN / keyword / pasted LCSC id-or-URL to an LCSC part number.

    Order: explicit C-number in the query (offline), parts-library manifests
    (offline, authoritative), the offline JLC catalog (jlcparts dump: stock,
    Basic/Extended, qty-1 price), then an easyeda.com keyword search (network).
    Prints JSON; exits 0 when a single LCSC id is resolved, 4 otherwise (with
    a candidate list to choose from). Lets the BOM sub-agent own MPN->LCSC
    resolution without the main thread reaching for WebSearch.
    """
    mpn = args.mpn
    target = mpn.strip().upper()

    # 0. The query already contains an LCSC id (bare, or inside a pasted
    #    lcsc.com/jlcpcb.com product URL): nothing to search.
    m = _LCSC_ID_RE.search(mpn)
    if m:
        lcsc = m.group(0).upper()
        _log_query("lookup_lcsc_id", outcome="hit", query=mpn, lcsc=lcsc,
                   source="explicit-id")
        print(json.dumps(
            {"ok": True, "mpn": mpn, "lcsc": lcsc, "source": "explicit-id"},
            indent=2,
        ))
        return 0

    # 1. Parts-library manifests — authoritative and offline.
    active, _broken = _load_library_parts(Path.cwd())
    for part in active:
        man = part.manifest
        if (man.mpn or "").strip().upper() == target:
            lcsc = (man.sourcing or {}).get("lcsc")
            if lcsc:
                _log_query("lookup_lcsc_id", outcome="hit", query=mpn, lcsc=lcsc,
                           source="parts-library", library_name=man.name)
                print(json.dumps(
                    {"ok": True, "mpn": mpn, "lcsc": lcsc,
                     "source": "parts-library", "name": man.name},
                    indent=2,
                ))
                return 0

    # 2. Offline JLC catalog (jlcparts dump) — richer than the network search
    #    (live stock, Basic/Extended, qty-1 price) and answers without network.
    #    Falls through only when the catalog is absent or has nothing.
    if jlcparts.available():
        results = jlcparts.search(mpn)
        if results:
            fields = ("lcsc", "model", "brand", "package", "stock", "type",
                      "price", "description")
            best = _pick_lcsc(mpn, results)
            if best and best.get("lcsc"):
                _log_query("lookup_lcsc_id", outcome="resolved", query=mpn,
                           lcsc=best["lcsc"], source="jlcparts")
                print(json.dumps(
                    {"ok": True, "mpn": mpn, "lcsc": best["lcsc"],
                     "source": "jlcparts",
                     "match": {k: best.get(k) for k in fields}},
                    indent=2,
                ))
                return 0
            _log_query("lookup_lcsc_id", outcome="miss", query=mpn,
                       n_candidates=len(results))
            print(json.dumps(
                {"ok": False, "mpn": mpn,
                 "candidates": [{k: r.get(k) for k in fields} for r in results],
                 "hint": "no single exact match; pick the candidate that fits "
                         "(prefer in-stock) and pass it to add_part_from_lcsc"},
                indent=2,
            ))
            return 4

    # 3. easyeda.com keyword search — network, best-effort.
    results = _search_easyeda_components(mpn)
    if results is None:
        _log_query("lookup_lcsc_id", outcome="error", query=mpn,
                   error="search-backend-unreachable")
        print(json.dumps(
            {"ok": False, "mpn": mpn, "candidates": [],
             "error": "part search backend unreachable",
             "hint": "Do NOT retry other keywords — searches will keep failing "
                     "this session. Ask the user for an LCSC C-number (C#####) "
                     "or use the closest stock KiCad part and record the "
                     "substitution in assumptions."},
            indent=2,
        ))
        return 4

    fields = ("lcsc", "model", "brand", "package", "description")
    best = _pick_lcsc(mpn, results)
    if best and best.get("lcsc"):
        _log_query("lookup_lcsc_id", outcome="resolved", query=mpn,
                   lcsc=best["lcsc"], source="easyeda")
        print(json.dumps(
            {"ok": True, "mpn": mpn, "lcsc": best["lcsc"], "source": "easyeda",
             "match": {k: best.get(k) for k in fields}},
            indent=2,
        ))
        return 0

    _log_query("lookup_lcsc_id", outcome="miss", query=mpn,
               n_candidates=len(results))
    print(json.dumps(
        {"ok": False, "mpn": mpn,
         "candidates": [{k: r.get(k) for k in fields} for r in results[:10]],
         "hint": ("pick a candidate and pass it to add-part --from-lcsc C<NNNNN>"
                  if results else
                  "no LCSC match; retry at most ONCE with the bare part family "
                  "(strip suffixes and descriptive words). If that misses too, "
                  "ask the user for a C-number or use the closest stock KiCad "
                  "part and record the substitution in assumptions.")},
        indent=2,
    ))
    return 4


def _cmd_jlcparts_update(args: argparse.Namespace) -> int:
    """Download/refresh the offline JLC parts catalog."""
    try:
        stats = jlcparts.update(
            dest=Path(args.dest) if args.dest else None,
            base_url=args.base_url,
            min_stock=args.min_stock,
            progress=lambda msg: print(msg, file=sys.stderr),
        )
    except Exception as e:
        print(f"jlcparts-update failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, **stats}, indent=2))
    return 0


def _slugify_libname(s: str) -> str:
    """Lowercase + collapse non-alphanumeric runs into single dashes.

    The result must satisfy ``parts_library.PART_NAME_RE`` (at least two
    chars, starts with a letter, ends with letter or digit). If the input
    doesn't yield a valid slug, returns the empty string so the caller
    can prompt for an explicit ``--name``.
    """
    import re as _re

    out = _re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    from kicraft.parts_library import PART_NAME_RE
    return out if PART_NAME_RE.match(out) else ""


def _resolve_dest_dir(into: str) -> Path:
    if into == "home":
        return Path.home() / ".kicraft" / "parts"
    if into == "vendored":
        # The repo-shipped library itself: used when vendoring core-default
        # bundles (the slug/sanitize/3D logic is identical to the other tiers).
        from kicraft.parts_library.loader import vendored_parts_dir

        return vendored_parts_dir()
    return Path.cwd() / ".kicraft" / "parts"


def _scan_symbol_name(text: str) -> str | None:
    """Return the raw name of the first top-level (symbol "...") in a .kicad_sym.

    The returned string is the verbatim contents of the quotes — either an
    unprefixed name (``IP2368``) or a library-prefixed one (``OldLib:IP2368``).
    The caller strips the library prefix when deciding the bundle's
    ``symbol_name`` and passes both the raw and stripped forms to
    :func:`_normalize_symbol_text` so the embedded prefix can be rewritten.
    """
    import re as _re

    for m in _re.finditer(r'\(symbol\s+"([^"]+)"', text):
        raw = m.group(1)
        # Top-level symbols are followed by other top-level forms, not by an
        # immediate sub-symbol "<name>_<unit>_<style>" form. Skip sub-symbol
        # entries by detecting their numeric "<name>_<unit>_<style>" suffix.
        if _re.match(r'^.+_\d+_\d+$', raw):
            continue
        return raw
    return None


def _scan_footprint_name(text: str) -> str | None:
    """Return the name of the (footprint "...") block in a .kicad_mod file."""
    import re as _re

    m = _re.search(r'\(footprint\s+"([^"]+)"', text)
    return m.group(1) if m else None


def _normalize_symbol_text(text: str, original_name: str, target_name: str) -> str:
    """Rewrite (symbol "Old" ...) → (symbol "New" ...) once if names differ.

    Preserves library-wrapper boilerplate and sub-symbol references that
    embed the original name (``<name>_<unit>_1``) so KiCad still parses
    the resulting file as a complete unit.
    """
    if original_name == target_name:
        return text
    text = text.replace(
        f'(symbol "{original_name}"', f'(symbol "{target_name}"', 1
    )
    # Rename sub-symbol entries (`(symbol "Old_1_1" ...)`) so units stay
    # tied to their parent. Limit to forms that look like real units.
    import re as _re

    return _re.sub(
        r'\(symbol\s+"' + _re.escape(original_name) + r'_(\d+)_(\d+)"',
        lambda m: f'(symbol "{target_name}_{m.group(1)}_{m.group(2)}"',
        text,
    )


_KICAD_NAME_ILLEGAL_RE = re.compile(r"[^A-Za-z0-9_.+-]")


def _sanitize_kicad_name(name: str) -> str:
    """Strip characters illegal in a KiCad 'Library:Name' segment.

    A fetched part's symbol/footprint name comes from EasyEDA/LCSC (or a user
    file) and can contain characters outside the ``[A-Za-z0-9_.+-]`` set that
    ``BomPart``'s SYMBOL_RE / FOOTPRINT_RE allow — most commonly the ``#`` in
    EasyEDA symbol names like ``DS3231SN#_C722469`` — which makes the resulting
    ``Library:Name`` reference unusable in a BOM. Remove them so the reference
    (and the on-disk name it must match) is always legal.
    """
    return _KICAD_NAME_ILLEGAL_RE.sub("", name or "")


def _sanitize_footprint_text(fp_text: str, raw_name: str) -> tuple[str, str]:
    """Return (sanitized_name, fp_text) with the in-file ``(footprint "...")``
    header rewritten to match, so the on-disk name stays consistent with the
    ``Library:Name`` the manifest declares."""
    name = _sanitize_kicad_name(raw_name)
    if name != raw_name:
        fp_text = fp_text.replace(f'(footprint "{raw_name}"', f'(footprint "{name}"', 1)
    return name, fp_text


_MODEL_STANZA_RE = re.compile(r'(\(model\s+")([^"]*)(")')

# Stock KiCad 3D models (passives etc.) resolve through the system install;
# bundle-local models resolve through the project copy synthesis stages.
_STOCK_3D_PREFIX = "${KICAD9_3DMODEL_DIR}/"

_DEFAULT_MODEL_STANZA = (
    '\t(model "{path}"\n'
    "\t\t(offset (xyz 0 0 0))\n"
    "\t\t(scale (xyz 1 1 1))\n"
    "\t\t(rotate (xyz 0 0 0))\n"
    "\t)\n"
)


def _model_stanza_paths(fp_text: str) -> list[str]:
    """All 3D model paths referenced by ``(model \"...\")`` stanzas, in order."""
    return [m.group(2) for m in _MODEL_STANZA_RE.finditer(fp_text)]


def _rewrite_model_stanza(fp_text: str, new_path: str) -> tuple[str, int]:
    """Point the footprint's ``(model ...)`` stanza(s) at *new_path*.

    Only the quoted path is replaced; offset/scale/rotate are preserved
    (they came from the original easyeda2kicad export and stay correct for
    the same model). When no stanza exists, a zero-transform stanza is
    appended before the closing paren so a freshly fetched model still gets
    referenced. Returns ``(new_text, stanza_count)``.
    """
    rewritten, n = _MODEL_STANZA_RE.subn(
        lambda m: m.group(1) + new_path + m.group(3), fp_text
    )
    if n:
        return rewritten, n
    trimmed = fp_text.rstrip()
    if not trimmed.endswith(")"):
        return fp_text, 0
    stanza = _DEFAULT_MODEL_STANZA.format(path=new_path)
    return trimmed[:-1] + stanza + ")\n", 1


def _check_3d_model_paths(
    part_dir: Path, part_name: str, fp_text: str
) -> list[str]:
    """Problems with the footprint's 3D model references; empty if clean.

    Accepted forms: a stock ``${KICAD9_3DMODEL_DIR}/...`` reference
    (resolved by the system KiCad install) or
    ``${KIPRJMOD}/3dmodels/<part_name>/<file>`` backed by a real file in the
    bundle's ``3d/`` dir (synthesis copies it into the generated project).
    Anything else (notably the bare ``/NAME.wrl`` paths easyeda2kicad leaves
    when exported without a model path) cannot resolve anywhere.
    """
    problems: list[str] = []
    expected_prefix = f"${{KIPRJMOD}}/3dmodels/{part_name}/"
    for path in _model_stanza_paths(fp_text):
        if path.startswith(_STOCK_3D_PREFIX):
            continue
        if path.startswith(expected_prefix):
            basename = path[len(expected_prefix):]
            if not basename or "/" in basename:
                problems.append(
                    f"3D model path {path!r} must be a flat file directly "
                    f"under {expected_prefix}"
                )
            elif not (part_dir / "3d" / basename).is_file():
                problems.append(
                    f"3D model path {path!r} has no backing file "
                    f"{part_dir / '3d' / basename}"
                )
            continue
        problems.append(
            f"3D model path {path!r} resolves nowhere: expected "
            f"{_STOCK_3D_PREFIX}... or {expected_prefix}<file>"
        )
    return problems


def _parse_sourcing_args(entries: list[str]) -> dict[str, str]:
    """Parse repeated ``--sourcing vendor=part_number`` into a dict.

    Raises ``ValueError`` with a useful message on malformed input.
    """
    from kicraft.parts_library import SOURCING_KEY_RE

    out: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"--sourcing entry {entry!r} missing '=' (expected vendor=part_number)"
            )
        key, _, value = entry.partition("=")
        key = key.strip().lower()
        value = value.strip()
        if not value:
            raise ValueError(f"--sourcing entry {entry!r} has empty value")
        if not SOURCING_KEY_RE.match(key):
            raise ValueError(
                f"--sourcing vendor key {key!r} must be lowercase alphanumeric/dashes"
            )
        out[key] = value
    return out


def _finalize_part_bundle(
    part_dir: Path,
    manifest: "PartManifest",  # noqa: F821 — forward ref to avoid top-level import
) -> None:
    """Write the manifest, recompute the content hash, write it again.

    Two writes is intentional: the first establishes a deterministic
    set of non-manifest files on disk, the second stores the hash of
    those files so subsequent verification passes.
    """
    from kicraft.parts_library import compute_content_hash, dump_manifest

    dump_manifest(manifest, part_dir)
    actual = compute_content_hash(part_dir)
    dump_manifest(manifest.model_copy(update={"content_hash": actual}), part_dir)


def _ensure_vendored_courtyard_clearance(
    pretty_dir: Path,
    footprint_name: str,
    *,
    min_clearance_mm: float = 0.2,
) -> None:
    """Footprint-hygiene check run when vendoring a part: grow the footprint's
    courtyard so it clears every pad by ``min_clearance_mm``.

    A courtyard that sits at (or inside) its own pad copper makes the part read
    as physically smaller than its copper, which downstream board-outline /
    placement geometry treats as the part's extent. Keeping the courtyard a
    clearance outboard of the pads keeps that geometry honest. Best-effort: only
    re-saves (round-trips through pcbnew) when a grow is actually needed, and
    skips silently if pcbnew is unavailable, so it never blocks a part fetch.
    """
    try:
        import pcbnew

        from kicraft.parts_library.footprint_courtyard import (
            ensure_courtyard_clears_pads,
        )
    except ImportError:
        return
    try:
        fp = pcbnew.FootprintLoad(str(pretty_dir), footprint_name)
        if fp is None:
            return
        if ensure_courtyard_clears_pads(fp, min_clearance_mm=min_clearance_mm):
            pcbnew.PCB_IO_KICAD_SEXPR().FootprintSave(str(pretty_dir), fp)
            print(
                f"add-part: grew {footprint_name} courtyard to clear pads by "
                f">= {min_clearance_mm} mm",
                file=sys.stderr,
            )
    except Exception as exc:  # noqa: BLE001 - hygiene check, never fatal
        print(
            f"add-part: courtyard clearance check skipped "
            f"({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )


def _add_part_from_files(args: argparse.Namespace) -> int:
    """Bundle a part from user-supplied .kicad_sym + .kicad_mod files.

    Use this path when the part isn't on LCSC: the user obtains KiCad
    files from anywhere (SnapEDA web UI, Ultra Librarian, silicon-vendor
    sites, hand-edited) and points add-part at them. The bundle layout
    and manifest format are identical to the LCSC path.
    """
    from kicraft.parts_library import (
        PartManifest,
        Provenance,
    )

    sym_src = Path(args.symbol).expanduser().resolve()
    fp_src = Path(args.footprint).expanduser().resolve()

    if not sym_src.is_file():
        print(f"symbol file not found: {sym_src}", file=sys.stderr)
        return 2
    if not fp_src.is_file():
        print(f"footprint file not found: {fp_src}", file=sys.stderr)
        return 2
    if not args.mpn:
        print("--mpn is required when using --symbol/--footprint", file=sys.stderr)
        return 2

    sym_text = sym_src.read_text()
    fp_text = fp_src.read_text()

    raw_symbol_name = _scan_symbol_name(sym_text)
    if raw_symbol_name is None:
        print(
            f"no top-level (symbol \"...\") found in {sym_src}",
            file=sys.stderr,
        )
        return 2
    # The bundle's symbol_name is unprefixed — the bundle's library prefix is
    # the directory name, not whatever the source file used.
    stripped_symbol_name = raw_symbol_name.split(":", 1)[-1]
    symbol_name = _sanitize_kicad_name(args.symbol_name or stripped_symbol_name)

    footprint_name = _scan_footprint_name(fp_text)
    if footprint_name is None:
        print(
            f"no (footprint \"...\") found in {fp_src}",
            file=sys.stderr,
        )
        return 2
    footprint_name, fp_text = _sanitize_footprint_text(fp_text, footprint_name)

    try:
        sourcing = _parse_sourcing_args(args.sourcing or [])
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    libname = (
        (args.name and _slugify_libname(args.name))
        or _slugify_libname(args.mpn)
        or _slugify_libname(symbol_name)
    )
    if not libname:
        print(
            f"could not derive a valid library name from --mpn {args.mpn!r}; "
            f"rerun with --name <slug>",
            file=sys.stderr,
        )
        return 2

    dest_base = _resolve_dest_dir(args.into)
    part_dir = dest_base / libname
    if part_dir.exists() and not args.overwrite:
        print(
            f"part directory already exists: {part_dir}\n"
            f"  rerun with --overwrite to replace, or pass --name to use a "
            f"different slug",
            file=sys.stderr,
        )
        return 2
    if part_dir.exists() and args.overwrite:
        import shutil as _shutil

        _shutil.rmtree(part_dir)

    pretty_dir = part_dir / f"{libname}.pretty"
    pretty_dir.mkdir(parents=True, exist_ok=True)

    # Normalize the symbol so its bare name matches what the manifest declares
    # (no library prefix; matches the bundle's library-prefix convention).
    normalized_sym = _normalize_symbol_text(sym_text, raw_symbol_name, symbol_name)
    (part_dir / f"{libname}.kicad_sym").write_text(normalized_sym)
    (pretty_dir / f"{footprint_name}.kicad_mod").write_text(fp_text)
    _ensure_vendored_courtyard_clearance(pretty_dir, footprint_name)

    import datetime as _dt

    manifest = PartManifest(
        schema_version="1",
        name=libname,
        version="0.1.0",
        content_hash="sha256:" + "0" * 64,
        description=args.description or f"{args.mpn} (imported)",
        mpn=args.mpn,
        sourcing=sourcing,
        datasheet_url=args.datasheet_url,
        tags=list(args.tag or []),
        watch_out_for=args.watch_out_for,
        maturity=args.maturity,
        symbol_name=symbol_name,
        footprint_name=footprint_name,
        kicad_version_min="9.0.0",
        provenance=Provenance(
            source="file-import",
            source_project_stem=None,
            added_at=_dt.datetime.now(_dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            kicad_version="9.0.0",
        ),
    )
    _finalize_part_bundle(part_dir, manifest)

    print(
        f"OK added {libname}@0.1.0 -> {part_dir}\n"
        f"  symbol:    {libname}:{symbol_name}\n"
        f"  footprint: {libname}:{footprint_name}\n"
        f"  mpn:       {manifest.mpn}\n"
        f"  sourcing:  {', '.join(f'{k}:{v}' for k, v in sourcing.items()) or '—'}\n"
        f"  tier:      {args.into}\n"
        f"  maturity:  {manifest.maturity}"
    )
    return 0


def _cmd_add_part(args: argparse.Namespace) -> int:
    """Bundle a part for the parts library, from LCSC or from supplied files.

    Two dispatch modes:

    - ``--from-lcsc C<NNNNN>`` — fetch from EasyEDA via the easyeda2kicad
      library. Self-contained; no other args needed in the simple case.
    - ``--symbol PATH --footprint PATH --mpn MPN`` — bundle user-supplied
      KiCad files (from SnapEDA web UI, Ultra Librarian, vendor sites,
      hand-edited, etc.). Vendor-agnostic; works without credentials.

    Both paths write the canonical layout::

        <dest>/<libname>/manifest.json
        <dest>/<libname>/<libname>.kicad_sym
        <dest>/<libname>/<libname>.pretty/<footprint_name>.kicad_mod

    and finalize with the content_hash. After this runs, ``list-parts``
    shows the new entry and the resolver picks it up the next time a
    BOM references ``<libname>:<sym>``.
    """
    using_lcsc = bool(args.from_lcsc)
    using_files = bool(args.symbol or args.footprint)
    if using_lcsc and using_files:
        print(
            "add-part: --from-lcsc and --symbol/--footprint are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    if using_files:
        if not (args.symbol and args.footprint):
            print(
                "add-part: --symbol and --footprint must be supplied together",
                file=sys.stderr,
            )
            return 2
        return _add_part_from_files(args)
    if not using_lcsc:
        print(
            "add-part requires either --from-lcsc <ID> OR "
            "--symbol PATH --footprint PATH --mpn MPN",
            file=sys.stderr,
        )
        return 2

    try:
        from easyeda2kicad.easyeda.easyeda_api import EasyedaApi
        from easyeda2kicad.easyeda.easyeda_importer import (
            Easyeda3dModelImporter,
            EasyedaFootprintImporter,
            EasyedaSymbolImporter,
        )
        from easyeda2kicad.kicad.export_kicad_3d_model import Exporter3dModelKicad
        from easyeda2kicad.kicad.export_kicad_footprint import ExporterFootprintKicad
        from easyeda2kicad.kicad.export_kicad_symbol import ExporterSymbolKicad
    except ImportError as exc:
        print(
            f"easyeda2kicad not installed in this Python environment: {exc}\n"
            f"install with: pip install easyeda2kicad",
            file=sys.stderr,
        )
        return 2

    from kicraft.parts_library import PartManifest, Provenance

    lcsc_id = args.from_lcsc

    print(f"fetching {lcsc_id} from EasyEDA/LCSC...", file=sys.stderr)
    api = EasyedaApi(use_cache=False)
    cad_data = api.get_cad_data_of_component(lcsc_id=lcsc_id)
    if not cad_data:
        print(f"EasyEDA returned no data for {lcsc_id}", file=sys.stderr)
        return 2

    ee_symbol = EasyedaSymbolImporter(easyeda_cp_cad_data=cad_data).get_symbol()
    ee_footprint = EasyedaFootprintImporter(
        easyeda_cp_cad_data=cad_data
    ).get_footprint()

    # Derive the library name from --name, then MPN, then symbol info name.
    libname = (
        (args.name and _slugify_libname(args.name))
        or _slugify_libname(ee_symbol.info.mpn or "")
        or _slugify_libname(ee_symbol.info.name or "")
    )
    if not libname:
        print(
            f"could not derive a valid library name from MPN "
            f"{ee_symbol.info.mpn!r} or symbol name {ee_symbol.info.name!r}; "
            f"rerun with --name <slug> (lowercase, alphanumeric, dashes)",
            file=sys.stderr,
        )
        return 2

    part_dir = _resolve_dest_dir(args.into) / libname
    if part_dir.exists() and not args.overwrite:
        print(
            f"part directory already exists: {part_dir}\n"
            f"  rerun with --overwrite to replace, or pass --name to use a "
            f"different slug",
            file=sys.stderr,
        )
        return 2
    if part_dir.exists() and args.overwrite:
        import shutil as _shutil

        _shutil.rmtree(part_dir)

    pretty_dir = part_dir / f"{libname}.pretty"
    pretty_dir.mkdir(parents=True, exist_ok=True)

    # Symbol: write to <libname>.kicad_sym (fresh file).
    sym_path = part_dir / f"{libname}.kicad_sym"
    sym_exporter = ExporterSymbolKicad(
        symbol=ee_symbol, lib_path=str(sym_path), custom_fields={}
    )
    if not sym_exporter.save_to_lib(
        lib_path=str(sym_path), footprint_lib_name=libname, overwrite=True
    ):
        print(f"failed to write symbol for {lcsc_id}", file=sys.stderr)
        return 2
    # EasyEDA symbol names can carry characters illegal in a KiCad 'Library:Name'
    # (e.g. the '#' in 'DS3231SN#_C722469'), which fails BomPart's SYMBOL_RE. Sanitize
    # the reference and rewrite the on-disk (symbol "...") header to match.
    # Trust the FILE for the starting name, not ee_symbol.info.name: the
    # exporter itself mangles some characters when writing (a '/' in the MPN
    # lands as '_', e.g. 'PD15-22C/TR8' -> 'PD15-22C_TR8'), and a manifest
    # symbol_name derived from info.name would then never match the file.
    sym_text = sym_path.read_text()
    raw_symbol_name = _scan_symbol_name(sym_text) or ee_symbol.info.name
    symbol_name = _sanitize_kicad_name(raw_symbol_name)
    if symbol_name != raw_symbol_name:
        sym_path.write_text(
            _normalize_symbol_text(sym_text, raw_symbol_name, symbol_name)
        )

    # 3D model: fetched by default so the bundle renders with a body in the
    # board's 3D view. A missing or failed download must never fail the part
    # fetch; the footprint is then written without any (model ...) stanza.
    model_basename: str | None = None
    if not args.no_3d:
        try:
            ee_model = Easyeda3dModelImporter(
                easyeda_cp_cad_data=cad_data, download_raw_3d_model=True
            ).output
            exporter_3d = Exporter3dModelKicad(model_3d=ee_model)
            if exporter_3d.output is not None and exporter_3d.export(
                output_dir=str(part_dir / "3d")
            ):
                raw_model = exporter_3d.output.name
                safe_model = _sanitize_kicad_name(raw_model)
                if safe_model != raw_model:
                    for ext in (".wrl", ".step"):
                        src = part_dir / "3d" / f"{raw_model}{ext}"
                        if src.is_file():
                            src.replace(part_dir / "3d" / f"{safe_model}{ext}")
                if (part_dir / "3d" / f"{safe_model}.wrl").is_file():
                    model_basename = f"{safe_model}.wrl"
        except Exception as exc:  # noqa: BLE001 - network/parse errors
            print(
                f"3D model fetch failed for {lcsc_id}: {exc}; continuing "
                f"without (re-run `fetch-3d` on the bundle later)",
                file=sys.stderr,
            )
        if model_basename is None:
            print(
                f"no 3D model for {lcsc_id}; the bundle will render without "
                f"a body (re-run `fetch-3d` or drop files into 3d/ later)",
                file=sys.stderr,
            )

    # Footprint: write into the .pretty dir, its model stanza pointing at the
    # project-relative path synthesis stages into generated projects.
    raw_footprint_name = ee_footprint.info.name
    footprint_name = _sanitize_kicad_name(raw_footprint_name)
    fp_path = pretty_dir / f"{footprint_name}.kicad_mod"
    fp_exporter = ExporterFootprintKicad(footprint=ee_footprint)
    if model_basename is None:
        # Suppress the stanza entirely: exporting with an empty model path
        # would emit the old broken bare "/NAME.wrl" reference.
        fp_exporter.output.model_3d = None
    fp_exporter.export(
        footprint_full_path=str(fp_path),
        model_3d_path=f"${{KIPRJMOD}}/3dmodels/{libname}",
    )
    if footprint_name != raw_footprint_name:
        _, fp_fixed = _sanitize_footprint_text(fp_path.read_text(), raw_footprint_name)
        fp_path.write_text(fp_fixed)
    if model_basename is not None:
        # The stanza was written with EasyEDA's raw model name; repoint it at
        # the (possibly sanitized) file we actually have on disk.
        fp_text = fp_path.read_text()
        fp_fixed, _count = _rewrite_model_stanza(
            fp_text, f"${{KIPRJMOD}}/3dmodels/{libname}/{model_basename}"
        )
        if fp_fixed != fp_text:
            fp_path.write_text(fp_fixed)

    _ensure_vendored_courtyard_clearance(pretty_dir, footprint_name)

    # Compose the manifest, then compute content_hash and rewrite once.
    sourcing: dict[str, str] = {"lcsc": lcsc_id}
    try:
        sourcing.update(_parse_sourcing_args(args.sourcing or []))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    description = args.description or (
        ee_symbol.info.description
        or f"{ee_symbol.info.manufacturer or ''} {ee_symbol.info.mpn or ''}".strip()
        or f"part {symbol_name}"
    )
    datasheet = args.datasheet_url or ee_symbol.info.datasheet or None
    if datasheet and not (
        datasheet.startswith("http://") or datasheet.startswith("https://")
    ):
        datasheet = None

    import datetime as _dt

    manifest = PartManifest(
        schema_version="1",
        name=libname,
        version="0.1.0",
        content_hash="sha256:" + "0" * 64,
        description=description,
        mpn=ee_symbol.info.mpn or symbol_name,
        sourcing=sourcing,
        datasheet_url=datasheet,
        tags=list(args.tag or []),
        watch_out_for=args.watch_out_for,
        maturity=args.maturity,
        symbol_name=symbol_name,
        footprint_name=footprint_name,
        kicad_version_min="9.0.0",
        provenance=Provenance(
            source="easyeda2kicad",
            source_project_stem=None,
            added_at=_dt.datetime.now(_dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            kicad_version="9.0.0",
        ),
    )
    _finalize_part_bundle(part_dir, manifest)
    _log_query("add_part_from_lcsc", outcome="fetched", query=lcsc_id, lcsc=lcsc_id,
               library_name=libname, into=args.into, maturity=manifest.maturity)

    print(
        f"OK added {libname}@0.1.0 -> {part_dir}\n"
        f"  symbol:    {libname}:{symbol_name}\n"
        f"  footprint: {libname}:{footprint_name}\n"
        f"  mpn:       {manifest.mpn}\n"
        f"  sourcing:  {', '.join(f'{k}:{v}' for k, v in sourcing.items())}\n"
        f"  tier:      {args.into}\n"
        f"  maturity:  {manifest.maturity}"
    )
    return 0


def _cmd_promote_part(args: argparse.Namespace) -> int:
    """Raise a bundle's maturity badge.

    Recomputes ``content_hash`` from the current files while it writes the new
    badge, so promoting to ``production`` right after dropping in a 3D model
    re-blesses the bundle in one step (a stale hash would otherwise mark it
    broken). With files unchanged the hash is identical, so a plain
    prototype -> reviewed bump is effectively metadata-only. Promoting to
    ``production`` requires a real 3D model (``3d/*.step|stp|wrl``).
    """
    from kicraft.parts_library import compute_content_hash, dump_manifest, load_manifest
    from kicraft.parts_library.loader import (
        home_parts_dir,
        project_parts_dir,
        vendored_parts_dir,
    )

    base = {
        "project": lambda: project_parts_dir(Path.cwd()),
        "home": home_parts_dir,
        "vendored": vendored_parts_dir,
    }[args.tier]()
    part_dir = base / args.name
    if not (part_dir / "manifest.json").is_file():
        print(f"no bundle {args.name!r} in {args.tier} tier ({part_dir})", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(part_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"could not read manifest for {args.name!r}: {exc}", file=sys.stderr)
        return 2

    if args.to == "production":
        model_dir = part_dir / "3d"
        models = (
            [p for p in model_dir.glob("*") if p.suffix.lower() in {".step", ".stp", ".wrl"}]
            if model_dir.is_dir()
            else []
        )
        if not models:
            print(
                f"refusing to promote {args.name!r} to production: no 3D model found "
                f"(expected {part_dir}/3d/*.step|*.stp|*.wrl). Add one first.",
                file=sys.stderr,
            )
            return 2

    old = manifest.maturity
    new_hash = compute_content_hash(part_dir)
    if old == args.to and manifest.content_hash == new_hash:
        print(f"{args.name} is already {args.to} ({part_dir})")
        return 0
    dump_manifest(
        manifest.model_copy(update={"maturity": args.to, "content_hash": new_hash}),
        part_dir,
    )
    print(f"promoted {args.name}: {old} -> {args.to}  ({part_dir})")
    return 0


def _cmd_validate_part(args: argparse.Namespace) -> int:
    """Validate a parts-library directory: schema, files, content_hash.

    Four checks, in order: (1) the manifest parses and the directory
    name matches ``name``; (2) the symbol + footprint files declared in
    the manifest exist and parse; (3) every footprint ``(model ...)`` path
    resolves (stock ``${KICAD9_3DMODEL_DIR}`` reference, or a
    ``${KIPRJMOD}/3dmodels/<name>/`` path backed by a file in the bundle's
    ``3d/`` dir; footprints with no stanza pass); (4) the recomputed
    content_hash matches the value stored in the manifest. With
    ``--update-hash``, recompute and rewrite the manifest's
    ``content_hash`` instead of failing (used when authoring a new part
    or after deliberate edits).
    """
    from pydantic import ValidationError as _PydValidationError

    from kicraft.parts_library import (
        compute_content_hash,
        dump_manifest,
        footprint_file_path,
        load_manifest,
        manifest_path,
        symbol_file_path,
    )

    part_dir = Path(args.path).resolve()
    if not part_dir.is_dir():
        print(f"not a directory: {part_dir}", file=sys.stderr)
        return 2

    if not manifest_path(part_dir).is_file():
        print(f"missing manifest.json in {part_dir}", file=sys.stderr)
        return 2

    try:
        manifest = load_manifest(part_dir)
    except _PydValidationError as e:
        print(f"manifest schema error:\n{e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — JSON decode etc.
        print(f"manifest read error: {e}", file=sys.stderr)
        return 2

    if part_dir.name != manifest.name:
        print(
            f"directory name {part_dir.name!r} does not match manifest "
            f"name {manifest.name!r}",
            file=sys.stderr,
        )
        return 2

    sym = symbol_file_path(part_dir)
    if not sym.is_file():
        print(f"missing symbol file {sym}", file=sys.stderr)
        return 2
    sym_text = sym.read_text()
    needle = f'(symbol "{manifest.symbol_name}"'
    if needle not in sym_text:
        print(
            f"symbol {manifest.symbol_name!r} not found in {sym.name}",
            file=sys.stderr,
        )
        return 2

    fp = footprint_file_path(part_dir, manifest.footprint_name)
    if not fp.is_file():
        print(f"missing footprint file {fp}", file=sys.stderr)
        return 2
    fp_text = fp.read_text()
    if "(footprint " not in fp_text and "(module " not in fp_text:
        print(
            f"footprint file {fp} does not contain a (footprint ...) or "
            f"(module ...) block",
            file=sys.stderr,
        )
        return 2

    problems = _check_3d_model_paths(part_dir, manifest.name, fp_text)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(
            "  fix with `fetch-3d <part-dir>` (or hand-edit the stanza "
            "to a stock ${KICAD9_3DMODEL_DIR} model) and rerun",
            file=sys.stderr,
        )
        return 2

    actual = compute_content_hash(part_dir)
    if actual != manifest.content_hash:
        if args.update_hash:
            updated = manifest.model_copy(update={"content_hash": actual})
            dump_manifest(updated, part_dir)
            print(f"updated content_hash for {manifest.name}@{manifest.version}: {actual}")
            return 0
        print(
            f"content_hash mismatch:\n  manifest: {manifest.content_hash}\n  actual:   {actual}\n"
            f"  rerun with --update-hash to accept the current files",
            file=sys.stderr,
        )
        return 2

    print(f"OK {manifest.name}@{manifest.version} ({part_dir})")
    return 0


def _cmd_fetch_3d(args: argparse.Namespace) -> int:
    """Fetch EasyEDA 3D models into part bundles and fix their stanzas.

    For each bundle: download the component's 3D model (WRL with baked
    colors + raw STEP) into ``<part>/3d/``, rewrite the footprint's
    ``(model ...)`` path to the project-relative
    ``${KIPRJMOD}/3dmodels/<name>/<model>.wrl`` scheme that synthesis
    stages into generated projects, then re-bless the manifest's
    content_hash. Bundles already on a stock ``${KICAD9_3DMODEL_DIR}``
    reference are left alone (the system KiCad install resolves those).
    """
    from kicraft.parts_library import (
        compute_content_hash,
        dump_manifest,
        footprint_file_path,
        load_manifest,
    )

    if args.all_vendored:
        from kicraft.parts_library.loader import vendored_parts_dir

        base = vendored_parts_dir()
        part_dirs = sorted(
            d for d in base.iterdir()
            if d.is_dir() and (d / "manifest.json").is_file()
        )
    else:
        part_dirs = [Path(p).resolve() for p in args.paths]
    if not part_dirs:
        print(
            "fetch-3d: no part directories (pass paths or --all-vendored)",
            file=sys.stderr,
        )
        return 2

    if not args.report:
        try:
            from easyeda2kicad.easyeda.easyeda_api import EasyedaApi
            from easyeda2kicad.easyeda.easyeda_importer import (
                Easyeda3dModelImporter,
            )
            from easyeda2kicad.kicad.export_kicad_3d_model import (
                Exporter3dModelKicad,
            )
        except ImportError as exc:
            print(
                f"easyeda2kicad not installed in this Python environment: {exc}\n"
                f"install with: pip install easyeda2kicad",
                file=sys.stderr,
            )
            return 2

    _MODEL_EXTS = {".step", ".stp", ".wrl"}
    buckets: dict[str, list[str]] = {
        "fetched": [], "already": [], "stock": [], "no-3d": [], "failed": [],
    }

    for part_dir in part_dirs:
        label = part_dir.name
        try:
            manifest = load_manifest(part_dir)
        except Exception as exc:  # noqa: BLE001
            buckets["failed"].append(f"{label}: manifest unreadable ({exc})")
            continue
        fp = footprint_file_path(part_dir, manifest.footprint_name)
        if not fp.is_file():
            buckets["failed"].append(f"{label}: missing footprint {fp.name}")
            continue
        fp_text = fp.read_text()
        stanza_paths = _model_stanza_paths(fp_text)

        if any(p.startswith(_STOCK_3D_PREFIX) for p in stanza_paths):
            buckets["stock"].append(label)
            continue

        prefix = f"${{KIPRJMOD}}/3dmodels/{manifest.name}/"
        model_dir = part_dir / "3d"
        has_models = model_dir.is_dir() and any(
            p.suffix.lower() in _MODEL_EXTS for p in model_dir.iterdir()
        )
        if (
            not args.overwrite
            and has_models
            and stanza_paths
            and all(p.startswith(prefix) for p in stanza_paths)
        ):
            buckets["already"].append(label)
            continue

        if args.report:
            lcsc = (manifest.sourcing or {}).get("lcsc")
            why = "needs fetch" if lcsc else "no lcsc id in sourcing"
            buckets["no-3d" if not lcsc else "fetched"].append(
                f"{label}: {why}"
            )
            continue

        lcsc = (manifest.sourcing or {}).get("lcsc")
        if not lcsc:
            buckets["no-3d"].append(f"{label}: no lcsc id in sourcing")
            continue

        print(f"fetching 3D model for {label} ({lcsc})...", file=sys.stderr)
        try:
            cad_data = EasyedaApi(use_cache=False).get_cad_data_of_component(
                lcsc_id=lcsc
            )
            if not cad_data:
                buckets["failed"].append(f"{label}: EasyEDA returned no data")
                continue
            ee_model = Easyeda3dModelImporter(
                easyeda_cp_cad_data=cad_data, download_raw_3d_model=True
            ).output
            exporter = Exporter3dModelKicad(model_3d=ee_model)
        except Exception as exc:  # noqa: BLE001 - network/parse errors
            buckets["failed"].append(f"{label}: fetch error ({exc})")
            continue
        if ee_model is None or exporter.output is None:
            buckets["no-3d"].append(f"{label}: no 3D model on EasyEDA")
            continue

        if args.overwrite and model_dir.is_dir():
            import shutil as _shutil

            _shutil.rmtree(model_dir)
        if not exporter.export(output_dir=str(model_dir)):
            buckets["failed"].append(f"{label}: 3D export wrote nothing")
            continue

        # EasyEDA model names can carry characters we keep out of paths;
        # rename on disk so the stanza and the file always agree.
        raw_name = exporter.output.name
        safe_name = _sanitize_kicad_name(raw_name)
        if safe_name != raw_name:
            for ext in (".wrl", ".step"):
                src = model_dir / f"{raw_name}{ext}"
                if src.is_file():
                    src.replace(model_dir / f"{safe_name}{ext}")
        wrl = model_dir / f"{safe_name}.wrl"
        if not wrl.is_file():
            buckets["failed"].append(f"{label}: export produced no .wrl")
            continue

        old_basenames = {Path(p).name for p in stanza_paths}
        fp_new, _count = _rewrite_model_stanza(fp_text, f"{prefix}{wrl.name}")
        fp.write_text(fp_new)
        if old_basenames and wrl.name not in old_basenames:
            print(
                f"  WARNING {label}: model name changed "
                f"({', '.join(sorted(old_basenames))} -> {wrl.name}); the kept "
                f"offset/rotate may not fit the new model, review the render",
                file=sys.stderr,
            )

        dump_manifest(
            manifest.model_copy(
                update={"content_hash": compute_content_hash(part_dir)}
            ),
            part_dir,
        )
        buckets["fetched"].append(label)

    headline = {
        "fetched": "needs fetch" if args.report else "fetched",
        "already": "already has 3D",
        "stock": "stock KiCad model (skipped)",
        "no-3d": "no 3D available",
        "failed": "FAILED",
    }
    for key, names in buckets.items():
        if names:
            print(f"{headline[key]} ({len(names)}):")
            for name in names:
                print(f"  {name}")
    return 1 if buckets["failed"] else 0


def _cmd_archive(args: argparse.Namespace) -> int:
    state_path = Path(args.state)
    try:
        state = _load_state(state_path)
    except ValidationError as e:
        print(f"schema validation failed:\n{e}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError) as e:
        print(f"could not read {state_path}: {e}", file=sys.stderr)
        return 2

    archive_root = Path(args.root).expanduser().resolve()
    dest = _archive_session(state_path.resolve(), state, archive_root)
    print(f"archived to {dest}")
    return 0


def _apply_slot(
    state: ConversationState,
    stage: str,
    slot_data: dict,
    *,
    project_stem: str | None,
) -> None:
    """Mutate ``state`` so the named stage's owned fields reflect ``slot_data``.

    Owned-field map:
      intent          -> state.intent (and top-level project_stem if given)
      functional_spec -> state.functional_spec
      architecture    -> state.architecture
      bom             -> state.bom, preserving existing wiring fields
                         (connections / no_connect_pins are owned by wiring)
      wiring          -> state.bom.connections + state.bom.no_connect_pins
    """
    if stage == "intent":
        state.intent = IntentSlot.model_validate(slot_data)
        if project_stem is not None:
            state.project_stem = project_stem
    elif stage == "functional_spec":
        state.functional_spec = FunctionalSpec.model_validate(slot_data)
    elif stage == "architecture":
        state.architecture = Architecture.model_validate(slot_data)
    elif stage == "bom":
        merged = dict(slot_data)
        if state.bom is not None:
            merged.setdefault(
                "connections",
                [c.model_dump() for c in state.bom.connections],
            )
            merged.setdefault(
                "no_connect_pins",
                [n.model_dump() for n in state.bom.no_connect_pins],
            )
        state.bom = BOM.model_validate(merged)
    elif stage == "wiring":
        if state.bom is None:
            raise ValueError("wiring stage requires bom slot to be populated")
        existing = state.bom.model_dump()
        if "connections" in slot_data:
            existing["connections"] = slot_data["connections"]
        if "no_connect_pins" in slot_data:
            existing["no_connect_pins"] = slot_data["no_connect_pins"]
        state.bom = BOM.model_validate(existing)
    elif stage == "placement":
        from .models import PlacementSection

        state.placement = PlacementSection.model_validate(slot_data)
    else:
        raise ValueError(f"unknown stage {stage!r}; expected one of {KNOWN_STAGES}")


def _cmd_stage_prep(args: argparse.Namespace) -> int:
    """Single-shot collector for a stage. Side-effect free.

    Returns JSON on stdout containing the current ``ConversationState``
    plus stage-specific extras the LLM stage needs to draft its slot:
      - architecture: ``leaves_block`` (rendered "Available leaves" markdown)
      - bom:          ``parts_block`` (rendered "Available parts" markdown)
      - wiring:       ``symbol_pinouts`` mapping every distinct BomPart.symbol
                      to its pin inventory (one batched lookup instead of N)
    """
    stage = args.stage
    if stage not in KNOWN_STAGES:
        print(f"unknown stage {stage!r}; expected one of {KNOWN_STAGES}", file=sys.stderr)
        return 2

    state_path = Path(args.state)
    if state_path.exists():
        try:
            state = _load_state(state_path)
        except ValidationError as e:
            print(f"schema validation failed:\n{e}", file=sys.stderr)
            return 2
        except (OSError, json.JSONDecodeError) as e:
            print(f"could not read {state_path}: {e}", file=sys.stderr)
            return 2
    else:
        state = ConversationState()

    extras: dict = {}

    if stage == "architecture":
        leaves = _load_library_leaves()
        extras["leaves_block"] = _format_available_leaves_block(leaves)

    elif stage == "bom":
        parts, _broken = _load_library_parts(state_path.parent.parent.resolve())
        extras["parts_block"] = _format_available_parts_block(parts)

    elif stage == "wiring":
        if state.bom is None:
            print(
                "stage-prep wiring requires the bom slot to be populated",
                file=sys.stderr,
            )
            return 4
        pinouts: dict[str, dict] = {}
        unresolved: list[str] = []
        seen: set[str] = set()
        for part in state.bom.parts:
            sym = part.symbol
            if sym in seen:
                continue
            seen.add(sym)
            try:
                info = lookup_pins(sym)
            except (SymbolNotFoundError, ValueError) as e:
                unresolved.append(f"{sym}: {e}")
                continue
            if not info.get("pins"):
                unresolved.append(f"{sym}: resolved but exposes no pins")
                continue
            pinouts[sym] = info
        # Fail loudly rather than emitting a partial dict with {"error": ...}
        # entries: a silent gap is what drove the wiring sub-agent to read
        # /usr/share/kicad/symbols directly (a hard-rule violation). The
        # sub-agent must fix the BOM (re-fetch / correct the symbol) and
        # retry, or surface a question — never fall back to Read.
        if unresolved:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "errors": [
                            "stage-prep wiring could not resolve a pinout for "
                            "every BOM symbol; correct the BOM and retry"
                        ],
                        "offenders": unresolved,
                    },
                    indent=2,
                )
            )
            return 4
        extras["symbol_pinouts"] = pinouts

    output = {
        "stage": stage,
        "state": json.loads(state.model_dump_json()),
        "extras": extras,
    }
    print(json.dumps(output, indent=2, default=str))
    return 0


def _cmd_stage_commit(args: argparse.Namespace) -> int:
    """Atomic stage commit: validate the proposed slot, merge into state, archive.

    Replaces the old ``draft -> write -> validate -> rewrite -> validate ->
    archive`` chain with one command. Returns JSON on stdout describing
    success or the specific validation failure so the LLM can self-correct.
    """
    stage = args.stage
    if stage not in KNOWN_STAGES:
        print(
            json.dumps({"ok": False, "errors": [f"unknown stage {stage!r}"]}, indent=2)
        )
        return 2

    state_path = Path(args.state)

    if state_path.exists():
        try:
            state = _load_state(state_path)
        except ValidationError as e:
            print(
                json.dumps(
                    {"ok": False, "errors": [f"existing state invalid: {e}"]},
                    indent=2,
                )
            )
            return 2
        except (OSError, json.JSONDecodeError) as e:
            print(
                json.dumps(
                    {"ok": False, "errors": [f"could not read state: {e}"]},
                    indent=2,
                )
            )
            return 2
    else:
        state = ConversationState()

    slot_path = Path(args.slot_file)
    try:
        slot_data = json.loads(slot_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(
            json.dumps(
                {"ok": False, "errors": [f"could not read slot-file {slot_path}: {e}"]},
                indent=2,
            )
        )
        return 2

    try:
        _apply_slot(state, stage, slot_data, project_stem=args.project_stem)
    except (ValueError, ValidationError) as e:
        print(
            json.dumps(
                {"ok": False, "errors": [f"slot validation failed: {e}"]},
                indent=2,
                default=str,
            )
        )
        return 3

    new_questions: list[Question] = []
    if args.questions_file:
        try:
            q_data = json.loads(Path(args.questions_file).read_text())
            new_questions = [Question.model_validate(q) for q in q_data]
        except (OSError, json.JSONDecodeError, ValidationError) as e:
            print(
                json.dumps(
                    {"ok": False, "errors": [f"could not read questions-file: {e}"]},
                    indent=2,
                    default=str,
                )
            )
            return 2

    state.replace_open_questions_for_stage(stage, new_questions)

    if args.history_message:
        state.history.append(ChatMsg(role="assistant", content=args.history_message))

    try:
        state = ConversationState.model_validate(state.model_dump())
    except ValidationError as e:
        print(
            json.dumps(
                {"ok": False, "errors": [f"post-merge state invalid: {e}"]},
                indent=2,
                default=str,
            )
        )
        return 3

    if state.architecture is not None:
        leaves = _load_library_leaves()
        if leaves:
            try:
                _validate_library_picks(state.architecture, leaves)
            except ArchitectureLibraryError as e:
                print(
                    json.dumps(
                        {"ok": False, "errors": [f"library validation failed: {e}"]},
                        indent=2,
                    )
                )
                return 3

    if state.bom is not None and state.bom.connections:
        checks = [
            check_pin_existence(state.bom),
            check_net_coverage(state.bom),
            check_power_pin_polarity(state.bom),
            check_two_terminal_self_short(state.bom),
            check_rf_feed_isolation(state.bom),
            check_single_net_per_pin(state.bom),
            check_family_wiring_contracts(state.bom),
        ]
        if state.architecture is not None:
            # Architecture declared these inter-sheet nets; the wiring stage
            # must realize each signal endpoint, or the emitter leaves a sheet
            # pin with no hierarchical label (caught only by §9.12 ERC at
            # synthesis time otherwise).
            checks.append(
                check_inter_sheet_nets_realized(state.architecture, state.bom)
            )
            # The inverse failure: a signal net wired to a single pin that was
            # never declared inter-sheet dangles ("Label not connected to
            # anything") -- the SOIL_MOISTURE_BLE USB D+/D- build failure.
            checks.append(
                check_no_dangling_signal_nets(state.architecture, state.bom)
            )
        for check in checks:
            if not check.ok:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "errors": [f"{check.name}: {check.message}"],
                            "offenders": check.offenders[:20],
                        },
                        indent=2,
                    )
                )
                return 3

    # Every footprint must resolve to a real .kicad_mod before the BOM is
    # committed — otherwise the bad name only surfaces at synthesis/PCB time.
    # Architecture sheets the BOM left empty produce blank leaves and, where an
    # inter-sheet net crosses them, orphan sheet pins. Catch it at BOM commit,
    # where the model can still add the missing parts.
    if stage == "bom" and state.bom is not None and state.architecture is not None:
        sp = check_sheets_have_parts(state.architecture, state.bom)
        if not sp.ok:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "errors": [f"{sp.name}: {sp.message}"],
                        "offenders": sp.offenders[:20],
                    },
                    indent=2,
                )
            )
            return 3

    if stage == "bom" and state.bom is not None:
        bad_fps = _unresolved_footprints(
            state.bom, state_path.resolve().parent.parent
        )
        if bad_fps:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "errors": ["footprint(s) do not resolve to a real .kicad_mod"],
                        "offenders": bad_fps[:20],
                    },
                    indent=2,
                )
            )
            return 3

    # Every symbol must likewise resolve to a real pin inventory before commit,
    # mirroring the footprint check. A hallucinated symbol name otherwise only
    # surfaces at wiring stage-prep, where the model can no longer fix it with the
    # BOM lookup tools.
    if stage == "bom" and state.bom is not None:
        bad_syms = _unresolved_symbols(state.bom)
        if bad_syms:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "errors": [
                            "symbol(s) do not resolve to a pin inventory; "
                            "use the lookup tools to fix the BOM"
                        ],
                        "offenders": bad_syms[:20],
                    },
                    indent=2,
                )
            )
            return 3

    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(state.model_dump_json(indent=2) + "\n")

    archive_warning: str | None = None
    if not args.no_archive:
        archive_root = (
            Path(args.archive_root).expanduser().resolve()
            if args.archive_root
            else _default_archive_root().resolve()
        )
        try:
            _archive_session(state_path.resolve(), state, archive_root)
        except OSError as e:
            archive_warning = f"archive failed: {e}"

    summary: dict = {
        "ok": True,
        "stage": stage,
        "project_stem": state.project_stem,
        "slots_filled": [
            name
            for name in ("intent", "functional_spec", "architecture", "bom")
            if getattr(state, name) is not None
        ],
        "open_questions": len(state.open_questions),
        "blocking_questions": sum(1 for q in state.open_questions if q.blocking),
    }
    if archive_warning:
        summary["archive_warning"] = archive_warning
    # Placement rules referencing refs the BOM no longer carries are
    # tolerated (parts churn across BOM re-runs); synthesis drops them
    # with a warning. Surface them at commit time too so the UI can show
    # which rules are currently inert.
    if stage == "placement" and state.placement is not None and state.bom is not None:
        known = {p.ref for p in state.bom.parts}
        stale = sorted(
            (set(state.placement.component_zones) | set(state.placement.thermal_refs))
            - known
        )
        if stale:
            summary["warnings"] = [
                f"placement rule(s) for ref(s) not in the BOM (ignored at "
                f"synthesis): {', '.join(stale)}"
            ]
    print(json.dumps(summary, indent=2))
    return 0


def _persist_artifacts(state, state_path: Path, artifacts) -> None:
    """Record `artifacts` on the state and persist state.json, so downstream
    tooling sees the produced paths + status even when checks failed."""
    state.artifacts = artifacts
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(state.model_dump_json(indent=2) + "\n")


def _write_synthesis_check(
    state_path: Path, project_stem: str | None, results, *, ok: bool
) -> None:
    """Write `<.kicraft>/synthesis_check.json` summarizing every check run."""
    summary = {
        "project_stem": project_stem,
        "status": "ok" if ok else "failed",
        "checked_at": _utc_compact_now(),
        "failed_checks": [r.name for r in results if not r.ok],
        "checks": [
            {
                "name": r.name,
                "ok": r.ok,
                "message": r.message,
                "offenders": r.offenders[:20],
            }
            for r in results
        ],
    }
    out = state_path.parent / "synthesis_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")


def _cmd_synthesize(args: argparse.Namespace) -> int:
    state_path = Path(args.state)
    out_dir = Path(args.out_dir)
    try:
        state = _load_state(state_path)
    except ValidationError as e:
        print(f"schema validation failed:\n{e}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError) as e:
        print(f"could not read {state_path}: {e}", file=sys.stderr)
        return 2

    if state.project_stem and out_dir.name != state.project_stem:
        out_dir = out_dir / state.project_stem

    try:
        artifacts, results = run_synth(
            state, out_dir, smoke=args.smoke, smoke_timeout_s=args.smoke_timeout
        )
    except SynthesisInputError as e:
        print(f"synthesis input error: {e}", file=sys.stderr)
        return 4
    except SynthesisValidationError as e:
        # Files were written before the checks ran; persist the artifact
        # record (status="failed") and the check summary so the failure is
        # inspectable, then surface it and exit non-zero.
        if e.artifacts is not None:
            _persist_artifacts(state, state_path, e.artifacts)
        _write_synthesis_check(state_path, state.project_stem, e.results, ok=False)
        print(str(e), file=sys.stderr)
        return 5

    _persist_artifacts(state, state_path, artifacts)
    _write_synthesis_check(state_path, state.project_stem, results, ok=True)

    print(f"wrote {artifacts.project_dir}")
    for r in results:
        print(f"  [{r.name}] {'ok' if r.ok else 'FAIL'}: {r.message}")

    if not args.no_archive:
        archive_root = (
            Path(args.archive_root).expanduser().resolve()
            if args.archive_root
            else _default_archive_root().resolve()
        )
        try:
            dest = _archive_session(
                state_path.resolve(),
                state,
                archive_root,
                synth_dir=Path(artifacts.project_dir),
                synth_results=results,
            )
            print(f"archived to {dest}")
        except OSError as e:
            print(f"warning: archive failed: {e}", file=sys.stderr)
    return 0


# --- build orchestrator: synthesize -> optimize+route -> promote -> gate -> fab -

_QUALITY_PRESETS = {
    "fast": {"engine": "solve-hierarchy"},
    # "draft" validated by the 2026-06-12 sweep (logs/draft_sweep/
    # 20260612T025445Z): ~1.8x faster than "good" at an equal-or-better
    # fab-ready rate; KICRAFT_QUALITY_PRESETS can override per-process.
    "draft": {"engine": "autoexperiment", "leaf_rounds": 2, "leaf_attempts": 2,
              "parent_rounds": 2},
    "good": {"engine": "autoexperiment", "leaf_rounds": 3, "leaf_attempts": 3,
             "parent_rounds": 3},
    "best": {"engine": "autoexperiment", "leaf_rounds": 6, "leaf_attempts": 3,
             "parent_rounds": 6},
}


def _quality_presets() -> dict:
    """_QUALITY_PRESETS with KICRAFT_QUALITY_PRESETS (a JSON object of
    per-preset overrides, e.g. '{"draft": {"leaf_rounds": 1}}') merged on top.
    Lets experiment harnesses sweep preset parameters without code edits."""
    presets = {name: dict(cfg) for name, cfg in _QUALITY_PRESETS.items()}
    raw = os.environ.get("KICRAFT_QUALITY_PRESETS", "").strip()
    if raw:
        try:
            overrides = json.loads(raw)
            for name, cfg in overrides.items():
                presets.setdefault(name, {}).update(cfg)
        except (ValueError, AttributeError, TypeError):
            print(
                "warning: ignoring malformed KICRAFT_QUALITY_PRESETS",
                file=sys.stderr,
            )
    return presets


def _run_layout(quality: str, root_sch: Path, pcb: Path,
                *, seed: int | None = None, route: bool = True) -> int:
    """Run the placement+routing engine in-process (inherits this env's pcbnew).

    ``seed`` pins the placement RNG so two runs of the same workspace produce the
    same placement (the determinism guarantee `replay` relies on). When ``None``
    (the `build` default) NO seed is forwarded, preserving each engine's existing
    behavior -- autoexperiment then draws a *random* master seed per run, so its
    optimization search keeps exploring. An explicit seed is threaded to BOTH
    engines: solve-hierarchy to the leaf solver, autoexperiment as its master
    seed. `replay` always passes one, which is why it is reproducible.

    ``route`` toggles FreeRouting. ``route=False`` (placement only) is honored by
    the solve-hierarchy (``fast``) engine -- the fast, deterministic path used to
    validate placement stability. The autoexperiment engines always route (their
    search scores rounds by routed DRC), so they ignore ``route=False``.

    autoexperiment qualities run in TWO phases instead of one combined loop:

      1. LEAF phase (``--leaves-only``): solve every leaf with a parameter-
         mutation search -- ``leaf_rounds`` mutated configs x ``leaf_attempts``
         seeds = N designs per leaf -- then **auto-pin each leaf's best**.
      2. PARENT phase (``--parents-only``): compose + route the parent for
         ``parent_rounds``, using the pinned leaves.

    The old single combined loop kept the best *parent round* and never
    independently pinned a leaf's best, so a sparse leaf (e.g. a USB-C+LDO power
    sheet) inherited whatever sprawled placement the winning parent round had.
    Decoupling lets each leaf find AND keep its tight cluster before the parent
    runs -- which is the manual ``solve-leaves -> pin -> compose`` flow that
    reliably placed these boards.
    """
    presets = _quality_presets()
    preset = presets.get(quality, presets["good"])
    seed_args = ["--seed", str(seed)] if seed is not None else []
    if preset["engine"] == "solve-hierarchy":
        from kicraft.cli.solve_hierarchy import main as _solve_hierarchy_main

        argv = [str(root_sch), "--pcb", str(pcb), *seed_args]
        if route:
            argv.append("--route")
        return _solve_hierarchy_main(argv)
    from kicraft.cli.autoexperiment import main as _autoexperiment_main

    if not route:
        print("[build]   note: --no-route is honored only by the fast "
              "(solve-hierarchy) engine; this quality always routes.",
              file=sys.stderr)
    common = [str(pcb), "--schematic", str(root_sch), *seed_args]
    print(f"[build]   leaf phase: {preset['leaf_rounds']}x{preset['leaf_attempts']} "
          f"designs/leaf + auto-pin best ...")
    leaf_rc = _autoexperiment_main(common + [
        "--leaves-only", "--rounds", str(preset["leaf_rounds"]),
        "--leaf-rounds", str(preset["leaf_attempts"])])
    if leaf_rc != 0:
        return leaf_rc
    print(f"[build]   parent phase: {preset['parent_rounds']} round(s) from pinned leaves ...")
    return _autoexperiment_main(common + [
        "--parents-only", "--rounds", str(preset["parent_rounds"])])


def _find_routed_parent(project_dir: Path) -> Path | None:
    """Locate the best routed parent board produced by the layout engine."""
    try:
        from kicraft.cli.solve_hierarchy import _find_parent_artifact

        art = _find_parent_artifact(project_dir)
        if art is not None:
            routed = Path(art) / "parent_routed.kicad_pcb"
            if routed.exists():
                return routed
    except Exception:
        pass
    hits = sorted(project_dir.glob("**/parent_routed.kicad_pcb"))
    return hits[-1] if hits else None


def _find_placed_parent(project_dir: Path) -> Path | None:
    """Best *placed* parent board: the routed one if present, else the
    pre-freerouting compose (what ``--no-route`` produces). Used by `replay
    --no-route`, the fast placement-only iteration mode whose footprint
    positions are the determinism guarantee we assert."""
    routed = _find_routed_parent(project_dir)
    if routed is not None:
        return routed
    try:
        from kicraft.cli.solve_hierarchy import _find_parent_artifact

        art = _find_parent_artifact(project_dir)
        if art is not None:
            pre = Path(art) / "parent_pre_freerouting.kicad_pcb"
            if pre.exists():
                return pre
    except Exception:
        pass
    hits = sorted(project_dir.glob("**/parent_pre_freerouting.kicad_pcb"))
    return hits[-1] if hits else None


def _find_best_leaf_board(project_dir: Path) -> Path | None:
    """Richest single-leaf board, as a last resort when the parent compose
    produced no parent board at all: a routed leaf if any, else a placed
    (pre-freerouting) leaf, else even a placement the legality gate REJECTED
    (``leaf_illegal_pre_stamp`` -- e.g. an array decap stranded outside the leaf
    outline, the KC-93X3X3 rc6 shape).

    A single leaf is only part of a multi-leaf board, and a rejected placement is
    not fab-ready, but each is a REAL placed mini-PCB that shows where the build
    actually got -- far better to show (so the failure is visible and the user can
    diagnose it) than the raw, uncomposed scatter board. The caller still returns
    rc6 and exports no fab package, so this is a PREVIEW only, never a fab-ready
    promotion.

    The autoexperiment writes per-round snapshots (``round_NNNN_leaf_*``); the
    single-pass solver writes the bare canonical names; a rejected round leaves
    ONLY ``leaf_illegal_pre_stamp`` + the ``round_NNNN_`` pre-freerouting snapshot
    (never the bare names). Match all of them, richest tier first, and within a
    tier pick the most recently written."""
    sub = project_dir / ".experiments" / "subcircuits"
    if not sub.is_dir():
        return None
    for pattern in (
        "*/leaf_routed.kicad_pcb",
        "*/round_*_leaf_routed.kicad_pcb",
        "*/leaf_pre_freerouting.kicad_pcb",
        "*/round_*_leaf_pre_freerouting.kicad_pcb",
        "*/leaf_illegal_pre_stamp.kicad_pcb",
    ):
        hits = [p for p in sub.glob(pattern) if p.is_file()]
        if hits:
            return max(hits, key=lambda p: p.stat().st_mtime)
    return None


def _count_leaf_subcircuits(root_sch: Path) -> int:
    """Number of non-root leaf subcircuits the hierarchical layout engine would
    place. 0 means a degenerate hierarchy -- the root schematic references no child
    sheets (a flat or stale synth), so there is nothing to compose/route. Mirrors the
    engine's own selection (solve_subcircuits: non-root nodes that are leaves)."""
    from kicraft.autoplacer.brain.hierarchy_parser import parse_hierarchy

    graph = parse_hierarchy(project_dir=root_sch.parent, top_schematic=root_sch)
    return len(graph.leaf_nodes()) - (1 if graph.root.is_leaf else 0)


def _degenerate_hierarchy_error(root_sch: Path) -> str | None:
    """An actionable error string if the hierarchy has 0 leaf subcircuits, else None.

    `build` checks this BEFORE the minutes-long layout run, so a degenerate design
    fails fast with the real cause instead of the late, misleading "board not
    routable as placed" (which the engine reaches when the parent compose, given no
    leaves, produces no routed board)."""
    if _count_leaf_subcircuits(root_sch) == 0:
        return (
            "the synthesized design has no leaf subcircuits to place/route -- the root "
            "schematic references no child sheets (a degenerate or stale hierarchy). "
            "Re-run synthesis so the components are organized into sheets."
        )
    return None


def _connector_stranded_refs(pcb: Path) -> list[str]:
    """Edge-zoned connectors stranded inboard of their board edge.

    A fully-routed board can be electrically clean (no shorts/unconnected) yet
    unusable because an edge-mount connector (USB-C, screw terminal, ...) sits
    inboard of the board edge it is zoned to -- the plug cannot physically mate.
    DRC cannot see this, so the fab-readiness gate re-checks it here, independent
    of compose (which now routes + promotes such boards rather than failing the
    build). Returns [] when the project carries no edge zones or on any error, so
    this never invents a failure for boards the gate cannot evaluate.
    """
    try:
        import glob
        import json

        from kicraft.autoplacer.brain.connector_edge_gap import connector_edge_gaps
        from kicraft.autoplacer.config import DEFAULT_CONFIG

        zone_files = sorted(glob.glob(str(pcb.parent / "*_autoplacer.json")))
        zones: dict = {}
        if zone_files:
            payload = json.loads(Path(zone_files[0]).read_text(encoding="utf-8"))
            zones = payload.get("component_zones", payload.get("zones", {})) or {}
        if not zones:
            return []
        tol = float(DEFAULT_CONFIG.get("connector_edge_inboard_tol_mm", 1.0))
        gaps = connector_edge_gaps(str(pcb), zones, inboard_tol_mm=tol)
        return [
            f"connector_stranded:{g.ref}@{g.gap_mm:.2f}mm({g.edge})"
            for g in gaps
            if g.gap_mm < -tol
        ]
    except Exception:
        return []


def _verify_routed_board(pcb: Path) -> dict:
    """Acceptance gate: no shorts, no unconnected (connector-shield items waived),
    and no edge-zoned connector stranded inboard of its board edge."""
    from kicraft.autoplacer.config import DEFAULT_CONFIG
    from kicraft.autoplacer.freerouting_runner import validate_routed_board

    v = validate_routed_board(str(pcb), cfg=dict(DEFAULT_CONFIG))
    drc = v.get("drc", {}) or {}
    shorts = int(drc.get("shorts", 0) or 0)
    unconnected = int(drc.get("unconnected", 0) or 0)
    accepted = bool(v.get("accepted", False))
    reasons = list(v.get("rejection_reasons", []))
    strand = _connector_stranded_refs(pcb)
    if strand:
        accepted = False
        for reason in strand:
            if reason not in reasons:
                reasons.append(reason)
    return {
        "ok": accepted and shorts == 0 and unconnected == 0,
        "shorts": shorts,
        "unconnected": unconnected,
        "reasons": reasons,
        "tracks": v.get("track_summary", {}) or {},
    }


def _missing_component_refs(expected_refs, board_refs) -> list[str]:
    """Expected component refs absent from the routed board (silent drops).

    A board can verify "clean" (no shorts, no unconnected) while having silently
    dropped whole components -- if the dropped parts took their nets with them
    there is no ratsnest left to flag. Comparing the expected reference set
    (from the BOM) against the footprints actually on the board catches that.
    Verified empirically: a healthy routed board carries exactly one footprint
    per non-power schematic ref, so a missing ref is a genuine drop.

    Returns [] when ``board_refs`` is empty/unknown (a count failure, or a truly
    empty board) so this never double-fires with the ``empty_board`` gate.
    """
    board = set(board_refs or [])
    if not board:
        return []
    return sorted(r for r in expected_refs if r not in board)


def _lower_project_netclass_clearance(pro_path: Path, clearance_mm: float) -> bool:
    """Cap every netclass clearance in a ``.kicad_pro`` at ``clearance_mm``.

    kicad-cli DRC enforces netclass clearances from the project file (not the
    board), so this is the store that must match the clearance the board was
    routed to. Only lowers, never widens. Returns True if anything changed."""
    try:
        data = json.loads(pro_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    classes = (data.get("net_settings") or {}).get("classes") or []
    changed = False
    for c in classes:
        if isinstance(c, dict) and float(c.get("clearance", 0) or 0) > clearance_mm:
            c["clearance"] = clearance_mm
            changed = True
    if changed:
        try:
            pro_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            return False
    return changed


def _align_project_clearance_to_routing(project_dir: Path, stem: str, pcb: Path) -> None:
    """Bring the project's netclass clearances down to the fine-pitch clearance
    the board was routed to, so the verify gate's DRC validates against the same
    rule FreeRouting used. No-op when the board did not need a fine-pitch lower
    (``_resolve_fine_pitch_rule`` returns None). See freerouting_runner."""
    try:
        from kicraft.autoplacer.config import DEFAULT_CONFIG
        from kicraft.autoplacer.freerouting_runner import _resolve_fine_pitch_rule

        clearance_um, _ = _resolve_fine_pitch_rule(str(pcb), dict(DEFAULT_CONFIG))
    except Exception:  # noqa: BLE001 -- best-effort; keep the original rule on error
        return
    if clearance_um is None:
        return
    clearance_mm = round(clearance_um / 1000.0, 4)
    if _lower_project_netclass_clearance(project_dir / f"{stem}.kicad_pro", clearance_mm):
        print(f"[build]     lowered project netclass clearance(s) to {clearance_mm} mm "
              "to match the fine-pitch routing")


def _maybe_electrical_review(state, project_dir: Path) -> dict:
    """Layer 4: optional LLM electrical-review fab gate.

    ON by default; set ``KICRAFT_ELECTRICAL_REVIEW=0`` (or false/no/off) to
    disable. Reviews the committed design for topology/value/completeness defects
    the deterministic gates and DRC cannot see, and reports whether a
    BLOCKER-severity finding means a structurally-sound board should not be
    declared fab-ready.

    Fail-soft: the disable decision reads the env var directly (so honoring an
    opt-out never needs an API key), and ANY error (no API key ->
    ``Settings.from_env`` SystemExit, network, malformed model output) skips the
    gate rather than blocking a sound board. Only a definite blocker from a
    successful review blocks.
    """
    if state is None or state.bom is None or not state.bom.connections:
        return {"ran": False, "findings": [], "blocked": False, "cost_usd": 0.0}
    if os.environ.get("KICRAFT_ELECTRICAL_REVIEW", "").strip().lower() in (
        "0", "false", "no", "off"
    ):
        return {"ran": False, "findings": [], "blocked": False, "cost_usd": 0.0}
    try:
        from kicraft.server.client import make_client
        from kicraft.server.config import Settings

        from .synthesis.electrical_review import (
            build_design_digest,
            has_blocker,
            review_design,
        )

        s = Settings.from_env()
        client = make_client(s.for_review())
        digest = build_design_digest(state, project_root=project_dir)
        model = s.review_model or s.model
        reasoning = s.review_reasoning()
        res = review_design(client, digest, model=model, reasoning=reasoning,
                            max_tokens=s.review_max_tokens)
        if not res["ok"]:
            print(f"[build] electrical review skipped (model error: {res['error']})",
                  file=sys.stderr)
            return {"ran": False, "findings": [], "blocked": False, "cost_usd": res["cost_usd"]}
        return {"ran": True, "findings": res["findings"],
                "blocked": has_blocker(res["findings"]), "cost_usd": res["cost_usd"]}
    except (Exception, SystemExit) as e:  # Settings.from_env raises SystemExit w/o a key
        print(f"[build] electrical review skipped ({type(e).__name__}: {e})", file=sys.stderr)
        return {"ran": False, "findings": [], "blocked": False, "cost_usd": 0.0}


def _promote_verify_fab(state, state_path: Path, artifacts, stem: str,
                        project_dir: Path, pcb: Path,
                        *, done_label: str = "BUILD COMPLETE",
                        do_fab: bool = True) -> int:
    """Steps 3-5 of the build tail: promote the routed parent to the
    project's main PCB, gate it (no shorts, no unconnected), and export
    the fab package. Shared by `build` and `manual-route`.

    Promotion is unconditional: the candidate stays at ``<stem>.kicad_pcb``
    whether or not it passes the verify gate, so the project always shows
    the board this build actually produced and a failure can be inspected
    directly (no restore-the-last-good-board fallback -- a failed gate
    fails loudly with rc 7 and exports no fab package; the UI marks any
    previously exported package invalid). The candidate must sit at the
    real path during verification because kicad-cli DRC reads netclass
    clearances from the sibling ``<stem>.kicad_pro``.
    """
    # 3. Promote the best board the layout engine produced to the project's
    #    main PCB. A routed parent is ideal; if the parent never routed (rc6),
    #    promote the richest artifact we DID reach -- the composed, placed
    #    parent (which carries the leaf-level routing stamped in), else a single
    #    placed/routed leaf -- so the project ALWAYS shows the real board this
    #    build produced. Never leave the raw, uncomposed scatter board as the
    #    preview: it misrepresents a build that actually placed (and usually
    #    routed) the design as one that never started.
    routed = _find_routed_parent(project_dir)
    if routed is None:
        partial = _find_placed_parent(project_dir) or _find_best_leaf_board(project_dir)
        if partial is not None:
            shutil.copy2(partial, pcb)
            print(
                f"[build] 3/5 no routed parent; promoted best partial board "
                f"for inspection -> {pcb.name} ({partial.name})"
            )
        else:
            print(
                f"[build] 3/5 no parent or leaf board produced; "
                f"leaving {pcb.name} as-is",
                file=sys.stderr,
            )
        print(
            "error: the layout engine produced no routed parent board -- the "
            "parent compose/route failed (board not routable as placed). "
            "Inspect .experiments/.../_search for rejected candidates.",
            file=sys.stderr,
        )
        return 6
    shutil.copy2(routed, pcb)
    print(f"[build] 3/5 promoted routed parent -> {pcb.name}")

    # Align the project's netclass clearances with the (possibly fine-pitch
    # lowered) clearance the board was routed to, so the verify gate validates
    # against the rule FreeRouting actually used, not a wider declared one.
    _align_project_clearance_to_routing(project_dir, stem, pcb)

    # 4. Verification gate: no shorts, no unconnected, and no silently-dropped
    #    components (a board that dropped parts can still verify "clean").
    gate = _verify_routed_board(pcb)
    expected_refs = {p.ref for p in (state.bom.parts if state and state.bom else [])}
    missing_refs = _missing_component_refs(expected_refs, gate["tracks"].get("footprint_refs"))
    print(
        f"[build] 4/5 verify: shorts={gate['shorts']} unconnected={gate['unconnected']} "
        f"traces={gate['tracks'].get('traces', '?')} "
        f"components={gate['tracks'].get('footprints', '?')}/{len(expected_refs) or '?'}"
    )
    if not gate["ok"] or missing_refs:
        print(
            f"[build]     kept failed board {pcb.name} for inspection "
            "(no fab package exported; any earlier package is now stale)",
            file=sys.stderr,
        )
        if missing_refs:
            print(
                f"error: routed board is INCOMPLETE -- "
                f"{len(missing_refs)} expected component(s) missing from the board: "
                f"{', '.join(missing_refs)}",
                file=sys.stderr,
            )
        if not gate["ok"]:
            print(
                f"error: routed board is NOT fab-ready -- shorts={gate['shorts']}, "
                f"unconnected={gate['unconnected']}, reasons={gate['reasons']}",
                file=sys.stderr,
            )
        return 7

    # 4b. Layer-4 electrical-review gate (cost-gated, off by default): catches
    #     designs that are structurally fab-ready (DRC-clean, all parts present)
    #     but electrically wrong -- the class deterministic checks cannot judge.
    review = _maybe_electrical_review(state, project_dir)
    if review["ran"]:
        for f in review["findings"]:
            sev = f.get("severity", "note").upper()
            print(f"[build]     review {sev}: [{f.get('area', '')}] {f.get('issue', '')}")
        if review["blocked"]:
            print(
                f"[build]     kept board {pcb.name} for inspection (no fab package; "
                f"electrical review found a blocker)",
                file=sys.stderr,
            )
            blockers = "; ".join(f["issue"] for f in review["findings"]
                                 if f.get("severity") == "blocker")
            print(
                f"error: routed board is NOT fab-ready -- electrical review blocker(s): {blockers}",
                file=sys.stderr,
            )
            return 7
        print(f"[build] 4/5 electrical review: "
              f"{len(review['findings'])} non-blocking finding(s), "
              f"cost ${review['cost_usd']:.4f}")

    if not do_fab:
        print(f"[build] 5/5 skipped fab export (--no-fab); verified board at {pcb.name}")
        print()
        print(f"{done_label}: {stem}")
        print(f"  routed PCB : {pcb}")
        print(
            f"  DRC        : 0 shorts, 0 unconnected "
            f"({gate['tracks'].get('traces', '?')} traces, "
            f"{gate['tracks'].get('vias', '?')} vias)"
        )
        return 0

    # 5. Export the fab package (Gerbers + drill + CPL + BOM, zipped).
    print("[build] 5/5 export fab package (Gerbers + drill + CPL + BOM + STEP + 3D render) ...")
    from kicraft.design.synthesis.fab_export import export_fab

    bom_parts = [p.model_dump() for p in state.bom.parts]
    fab = export_fab(str(pcb), str(project_dir), stem, bom_parts=bom_parts)

    artifacts.routed_pcb = pcb
    artifacts.fab_zip = Path(fab["zip"])
    artifacts.step_file = Path(fab["step"]) if fab.get("step") else None
    artifacts.board_3d_png = (
        Path(fab["board_3d_png"]) if fab.get("board_3d_png") else None
    )
    _persist_artifacts(state, state_path, artifacts)

    print()
    print(f"{done_label}: {stem}")
    print(f"  routed PCB : {pcb}")
    print(
        f"  DRC        : 0 shorts, 0 unconnected "
        f"({gate['tracks'].get('traces', '?')} traces, {gate['tracks'].get('vias', '?')} vias)"
    )
    print(f"  fab package: {fab['zip']}")
    print(f"  contents   : {', '.join(fab['files'])}")
    if fab.get("step"):
        print(f"  STEP model : {fab['step']}")
    if fab.get("board_3d_png"):
        print(f"  3D render  : {fab['board_3d_png']}")
    return 0


def _cmd_manual_route(args: argparse.Namespace) -> int:
    """Route + promote a saved manual layout, end to end.

    Expects a workspace that already carries a synthesized project
    (generated/<STEM> with the seed PCB and routed leaf artifacts) and
    a ``.experiments/manual/manual_layout.json`` written by the layout
    editor. Runs compose --manual-layout --route under a host-wide
    build slot, then the same promote/verify/fab tail as `build`.
    """
    import subprocess

    from kicraft.build_slots import build_slot

    from .models import ArtifactPaths

    state_path = Path(args.state)
    out_dir = Path(args.out_dir)
    try:
        state = _load_state(state_path)
    except ValidationError as e:
        print(f"schema validation failed:\n{e}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError) as e:
        print(f"could not read {state_path}: {e}", file=sys.stderr)
        return 2

    if state.bom is None:
        print("error: manual-route needs a staged state with a BOM.",
              file=sys.stderr)
        return 3
    if state.project_stem and out_dir.name != state.project_stem:
        out_dir = out_dir / state.project_stem
    stem = state.project_stem
    project_dir = out_dir
    pcb = project_dir / f"{stem}.kicad_pcb"
    manual_layout = project_dir / ".experiments" / "manual" / "manual_layout.json"
    if not manual_layout.is_file():
        print(f"error: no saved manual layout at {manual_layout}; save a "
              "layout in the editor first.", file=sys.stderr)
        return 3
    if not pcb.is_file():
        print(f"error: no synthesized board at {pcb}; run a build first.",
              file=sys.stderr)
        return 3

    # Persisted artifact paths come from the ORIGINAL build's workspace;
    # in a rehydrated workspace they're stale, so rebuild from disk.
    artifacts = state.artifacts
    if artifacts is None or Path(artifacts.project_dir) != project_dir:
        artifacts = ArtifactPaths(
            project_dir=project_dir,
            project_stem=stem,
            root_sch=project_dir / f"{stem}.kicad_sch",
            leaf_schs=sorted(
                p for p in project_dir.glob("*.kicad_sch") if p.stem != stem
            ),
            kicad_pro=project_dir / f"{stem}.kicad_pro",
            autoplacer_json=project_dir / f"{stem}_autoplacer.json",
        )

    with build_slot(echo=lambda line: print(line, flush=True)):
        print("[build] 2/5 route the saved manual layout (FreeRouting) -- "
              "may take minutes ...")
        cmd = [
            sys.executable, "-m", "kicraft.cli.compose_subcircuits",
            "--project", str(project_dir),
            "--parent", "/",
            "--pcb", str(pcb),
            "--manual-layout", str(manual_layout),
            "--output", str(project_dir / ".experiments" / "manual"
                            / "manual_routed.json"),
            "--route",
        ]
        rc = subprocess.run(cmd, cwd=str(project_dir)).returncode
        if rc != 0:
            print(f"error: manual compose/route exited {rc}", file=sys.stderr)
            return 6
        return _promote_verify_fab(state, state_path, artifacts, stem,
                                   project_dir, pcb,
                                   done_label="MANUAL ROUTE COMPLETE")


def _cmd_build(args: argparse.Namespace) -> int:
    state_path = Path(args.state)
    out_dir = Path(args.out_dir)
    try:
        state = _load_state(state_path)
    except ValidationError as e:
        print(f"schema validation failed:\n{e}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError) as e:
        print(f"could not read {state_path}: {e}", file=sys.stderr)
        return 2

    if state.bom is None or not state.bom.connections:
        print(
            "error: build needs a fully-staged state with wiring "
            "(bom.connections). Run the KiCraft stages first.",
            file=sys.stderr,
        )
        return 3
    if state.project_stem and out_dir.name != state.project_stem:
        out_dir = out_dir / state.project_stem

    # 1. Synthesize: schematic + seed PCB + ERC gate.
    print("[build] 1/5 synthesize (schematic + seed PCB + ERC) ...")
    try:
        artifacts, results = run_synth(state, out_dir, smoke=False)
    except SynthesisInputError as e:
        print(f"synthesis input error: {e}", file=sys.stderr)
        return 4
    except SynthesisValidationError as e:
        if e.artifacts is not None:
            _persist_artifacts(state, state_path, e.artifacts)
        _write_synthesis_check(state_path, state.project_stem, e.results, ok=False)
        # The schematics ARE written (synthesis emits them before the check
        # gate), so they remain viewable in the Synthesize tab. List each failed
        # check and its offenders so the exact problem (which pin/wire/net) is
        # visible in the live build log, not just a pass/fail count.
        print("synthesis checks failed (schematics were written and are viewable "
              "in the Synthesize tab):", file=sys.stderr)
        for r in (e.results or []):
            if not r.ok:
                print(f"  [{r.name}] {r.message}", file=sys.stderr)
                for o in (r.offenders or [])[:20]:
                    print(f"      - {o}", file=sys.stderr)
        return 5
    _persist_artifacts(state, state_path, artifacts)
    _write_synthesis_check(state_path, state.project_stem, results, ok=True)

    stem = state.project_stem
    project_dir = Path(artifacts.project_dir)
    root_sch = Path(artifacts.root_sch)
    pcb = project_dir / f"{stem}.kicad_pcb"
    print(f"[build]     synthesized {project_dir} (ERC clean)")

    # Fail fast on a degenerate (0-leaf) hierarchy with an actionable message, before
    # the minutes-long layout run reaches the misleading "board not routable as placed".
    degenerate = _degenerate_hierarchy_error(root_sch)
    if degenerate:
        print(f"error: {degenerate}", file=sys.stderr)
        return 6

    # Preflight the routing toolchain (Java + FreeRouting jar) so a misconfigured
    # host fails immediately with an actionable message instead of after the
    # minutes-long placement that then can't route ("board not routable as placed").
    from kicraft.autoplacer.freerouting_runner import (
        FreeroutingUnavailableError,
        preflight_routing_toolchain,
    )

    try:
        preflight_routing_toolchain()
    except FreeroutingUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 6

    # 2..5 saturate the host (parallel leaf solvers + FreeRouting JVMs), so they
    # run under a host-wide build slot; the wait line below is the queue signal
    # callers (build worker, web log tail) key their timeouts and UI off.
    from kicraft.build_slots import build_slot

    with build_slot(echo=lambda line: print(line, flush=True)):
        return _layout_route_fab(args, state, state_path, artifacts, results,
                                 stem, project_dir, root_sch, pcb)


def _layout_route_fab(args, state, state_path, artifacts, results,
                      stem: str, project_dir: Path, root_sch: Path, pcb: Path) -> int:
    """The heavy tail of `build` (place+route, verify gate, fab export, archive),
    split out so _cmd_build can hold a host-wide build slot across exactly this.

    Also the seam `replay` reuses to re-run place+route on a fixed workspace.
    The replay-only knobs are read off ``args`` with build-preserving defaults:
    ``seed`` (placement RNG, default None = engine's own/random), ``route``
    (default True), and ``no_fab`` (default False) -- `_cmd_build`'s namespace
    carries none of them, so build is byte-for-byte unchanged."""
    seed = getattr(args, "seed", None)
    route = getattr(args, "route", True)
    do_fab = not getattr(args, "no_fab", False)
    done_label = getattr(args, "done_label", "BUILD COMPLETE")

    # 2. Optimize placement (+ route) via the layout engine.
    action = "place + route" if route else "place (no route)"
    seed_label = seed if seed is not None else "auto"
    print(f"[build] 2/5 {action} (quality={args.quality}, seed={seed_label}) "
          "-- may take minutes ...")
    rc = _run_layout(args.quality, root_sch, pcb, seed=seed, route=route)
    if rc != 0:
        print(f"error: layout/route engine exited {rc}", file=sys.stderr)
        return 6

    if not route:
        # Placement-only: promote the placed (pre-freerouting) parent so the
        # positions are inspectable, but skip the routed-board verify/fab tail.
        placed = _find_placed_parent(project_dir)
        if placed is None:
            print("error: the layout engine produced no placed parent board "
                  "(parent compose failed).", file=sys.stderr)
            return 6
        shutil.copy2(placed, pcb)
        print(f"[build] 3/5 promoted placed parent -> {pcb.name} "
              "(placement only; no verify/fab)")
        print()
        print(f"{done_label}: {stem}")
        print(f"  placed PCB : {pcb}")
    else:
        rc = _promote_verify_fab(state, state_path, artifacts, stem, project_dir,
                                 pcb, done_label=done_label, do_fab=do_fab)
        if rc != 0:
            return rc

    if not getattr(args, "no_archive", True):
        archive_root = (
            Path(args.archive_root).expanduser().resolve()
            if getattr(args, "archive_root", None)
            else _default_archive_root().resolve()
        )
        try:
            dest = _archive_session(
                state_path.resolve(),
                state,
                archive_root,
                synth_dir=project_dir,
                synth_results=results,
            )
            print(f"  archived   : {dest}")
        except OSError as e:
            print(f"warning: archive failed: {e}", file=sys.stderr)
    return 0


def _pin_deterministic_placement_env() -> None:
    """Pin the process env so the placement subprocesses `replay` spawns are
    reproducible. ``PYTHONHASHSEED`` is the dominant lever: the placement solver
    iterates ``set``/``dict`` of string refs and dedups force-states by ``hash``,
    all salted per-process by the hash seed -- left unpinned, two runs of the same
    seeded workspace diverge at mm scale (empirically verified). The thread caps
    remove residual floating-point reduction jitter from multi-threaded numpy in
    the force solver. ``setdefault`` so an explicitly-set value (e.g. to probe
    salt-robustness) is honored. The placement runs in child processes that read
    these at startup, so setting them on the parent here is sufficient; routing
    (FreeRouting) remains best-effort-stable regardless."""
    os.environ.setdefault("PYTHONHASHSEED", "0")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


class _ReplayInputError(Exception):
    """A workspace cannot be replayed (missing/ambiguous synthesized artifacts)."""


def _discover_stem(project_dir: Path) -> str | None:
    """The project stem of a synthesized workspace = the single basename that
    owns the full root artifact set (``<stem>.kicad_pro`` + ``.kicad_pcb`` +
    ``.kicad_sch``). Returns None when zero or more than one candidate matches."""
    candidates = [
        p.stem
        for p in sorted(project_dir.glob("*.kicad_pro"))
        if (project_dir / f"{p.stem}.kicad_pcb").exists()
        and (project_dir / f"{p.stem}.kicad_sch").exists()
    ]
    return candidates[0] if len(candidates) == 1 else None


def _find_state_json(project_dir: Path, stem: str) -> Path | None:
    """Locate the workspace's ``state.json`` by walking up from the project dir
    (a synthesized workspace lives at ``<work>/generated/<STEM>/`` with the state
    at ``<work>/.kicraft/state.json``). Only returns a state whose
    ``project_stem`` matches ``stem`` so an unrelated ancestor state is ignored."""
    base = project_dir
    for _ in range(6):
        cand = base / ".kicraft" / "state.json"
        if cand.is_file():
            try:
                if _load_state(cand).project_stem == stem:
                    return cand
            except (OSError, json.JSONDecodeError, ValidationError):
                pass
        if base.parent == base:
            break
        base = base.parent
    return None


def _resolve_synthesized_workspace(args: argparse.Namespace):
    """Resolve an already-synthesized workspace for `replay` and validate that the
    artifacts place+route consume exist. Supports both input modes and returns
    ``(state, state_path, artifacts, stem, project_dir, root_sch, pcb)``.

    Raises ``_ReplayInputError`` (caller maps to rc 3) on a missing/ambiguous
    workspace -- never calls synthesis."""
    from .models import ArtifactPaths

    project = getattr(args, "project", None)
    if project:
        project_dir = Path(project).expanduser().resolve()
        if not project_dir.is_dir():
            raise _ReplayInputError(f"--project {project_dir} is not a directory")
        stem = _discover_stem(project_dir)
        if stem is None:
            raise _ReplayInputError(
                f"could not identify a single synthesized project in {project_dir} "
                "(need exactly one <stem>.kicad_pro with sibling .kicad_pcb + "
                ".kicad_sch). Use the `replay STATE.json OUT_DIR` form instead.")
        state_path = _find_state_json(project_dir, stem)
        state = _load_state(state_path) if state_path is not None else None
    else:
        if not getattr(args, "state", None) or not getattr(args, "out_dir", None):
            raise _ReplayInputError(
                "provide either `--project DIR` or positional `STATE.json OUT_DIR`")
        state_path = Path(args.state)
        try:
            state = _load_state(state_path)
        except ValidationError as e:
            raise _ReplayInputError(f"schema validation failed:\n{e}") from e
        except (OSError, json.JSONDecodeError) as e:
            raise _ReplayInputError(f"could not read {state_path}: {e}") from e
        out_dir = Path(args.out_dir)
        if state.project_stem and out_dir.name != state.project_stem:
            out_dir = out_dir / state.project_stem
        project_dir = out_dir.resolve()
        stem = state.project_stem
        if not stem:
            raise _ReplayInputError(
                "state has no project_stem; cannot locate artifacts")

    root_sch = project_dir / f"{stem}.kicad_sch"
    pcb = project_dir / f"{stem}.kicad_pcb"
    kicad_pro = project_dir / f"{stem}.kicad_pro"
    autoplacer_json = project_dir / f"{stem}_autoplacer.json"

    # Validate the artifacts the placement engine actually consumes. The
    # autoplacer.json is part of a complete synthesized set but is read only by
    # the UI panels, not the placer -- a missing one is a warning, not a failure.
    missing = [p for p in (root_sch, pcb, kicad_pro) if not p.is_file()]
    if missing:
        raise _ReplayInputError(
            "missing required synthesized artifact(s):\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\n(run synthesis / `kicraft build` first to produce them)")
    if not autoplacer_json.is_file():
        print(f"warning: {autoplacer_json.name} absent (UI seed file; not needed "
              "for placement) -- continuing", file=sys.stderr)

    if state is None:
        state = ConversationState(project_stem=stem)

    # Persisted ArtifactPaths from the original build are stale in a rehydrated or
    # copied workspace, so rebuild them from disk when they don't match.
    artifacts = state.artifacts
    if artifacts is None or Path(artifacts.project_dir).resolve() != project_dir:
        artifacts = ArtifactPaths(
            project_dir=project_dir,
            project_stem=stem,
            root_sch=root_sch,
            leaf_schs=sorted(
                p for p in project_dir.glob("*.kicad_sch") if p.stem != stem
            ),
            kicad_pro=kicad_pro,
            autoplacer_json=autoplacer_json,
        )
    return state, state_path, artifacts, stem, project_dir, root_sch, pcb


def _cmd_replay(args: argparse.Namespace) -> int:
    """Re-run ONLY place + route on an already-synthesized workspace -- `build`
    minus its synthesize step. No LLM / synthesis stage runs, so a placement code
    change can be tested against a FIXED input and produce a reproducible board.

    Two input modes:
      replay STATE.json OUT_DIR  -- resolve artifacts via state.artifacts
      replay --project DIR       -- discover artifacts on disk (no state.json
                                    needed; preferred for iteration/testing)
    """
    from kicraft.build_slots import build_slot

    # Pin the placement RNG/threading env BEFORE spawning any layout subprocess,
    # so the determinism guarantee holds (see helper).
    _pin_deterministic_placement_env()

    try:
        (state, state_path, artifacts, stem,
         project_dir, root_sch, pcb) = _resolve_synthesized_workspace(args)
    except _ReplayInputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    # Fail fast on a degenerate (0-leaf) hierarchy, as `build` does.
    degenerate = _degenerate_hierarchy_error(root_sch)
    if degenerate:
        print(f"error: {degenerate}", file=sys.stderr)
        return 6

    # A state.json-less project-mode replay can't build a BOM-backed fab package
    # or a session archive; downgrade loudly instead of crashing deep in the tail.
    if state_path is None:
        if not getattr(args, "no_fab", False):
            print("note: no state.json near --project; skipping fab export (needs "
                  "the BOM). Use `replay STATE.json OUT_DIR` to export a package.",
                  file=sys.stderr)
            args.no_fab = True
        args.no_archive = True

    # Determinism guard: synthesis must NOT run, so the root schematic the placer
    # reads must be byte-identical before and after the replay.
    sch_before = root_sch.read_bytes()

    if args.route:
        from kicraft.autoplacer.freerouting_runner import (
            FreeroutingUnavailableError,
            preflight_routing_toolchain,
        )

        try:
            preflight_routing_toolchain()
        except FreeroutingUnavailableError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 6

    args.done_label = "REPLAY COMPLETE"
    print(f"[replay] re-running place{'+route' if args.route else ''} on "
          f"{project_dir} (quality={args.quality}, seed={args.seed}) "
          "-- no synthesis")
    with build_slot(echo=lambda line: print(line, flush=True)):
        rc = _layout_route_fab(args, state, state_path, artifacts, [],
                               stem, project_dir, root_sch, pcb)

    if root_sch.read_bytes() != sch_before:
        print("error: replay mutated the root schematic -- synthesis must not run "
              "during replay (determinism violated).", file=sys.stderr)
        return 8
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="kicraft",
        description=(
            "Deterministic helpers for the KiCraft skill. The LLM-driven "
            "conversation lives in the Claude Code skill at "
            ".claude/skills/kicraft/; this CLI handles state validation, "
            "leaf-library listing, and file synthesis."
        ),
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="validate a state.json file")
    p_val.add_argument("state", help="path to state.json")
    p_val.set_defaults(func=_cmd_validate)

    p_erev = sub.add_parser(
        "electrical-review",
        help="LLM electrical review (Layer 3) of a committed design: topology, "
             "values, decoupling, programming path, protection",
    )
    p_erev.add_argument("state", help="path to state.json")
    p_erev.add_argument("--model", default=None, help="review model override")
    p_erev.add_argument("--digest-only", action="store_true",
                        help="print the structured review digest and exit (no LLM call)")
    p_erev.add_argument("--fail-on-blocker", action="store_true",
                        help="exit 3 if the review reports a blocker-severity finding")
    p_erev.set_defaults(func=_cmd_electrical_review)

    p_list = sub.add_parser(
        "list-leaves",
        help="print the markdown block listing every reusable leaf",
    )
    p_list.set_defaults(func=_cmd_list_leaves)

    p_list_parts = sub.add_parser(
        "list-parts",
        help=(
            "print the markdown block listing every parts-library bundle "
            "(project / home / vendored / extras tiers)"
        ),
    )
    p_list_parts.set_defaults(func=_cmd_list_parts)

    p_lcsc = sub.add_parser(
        "lookup-lcsc-id",
        help=(
            "resolve a manufacturer part number (MPN) to an LCSC part number "
            "via the parts library, the offline JLC catalog, then an "
            "easyeda.com keyword search"
        ),
    )
    p_lcsc.add_argument("mpn", help="manufacturer part number to resolve")
    p_lcsc.set_defaults(func=_cmd_lookup_lcsc_id)

    p_jlc = sub.add_parser(
        "jlcparts-update",
        help=(
            "download/refresh the offline JLC parts catalog (yaqwsx/jlcparts "
            "nightly dump: ~650 MB download; in-stock-pruned by default)"
        ),
    )
    p_jlc.add_argument("--base-url", default=jlcparts.DATA_URL,
                       help="override the dataset URL (testing)")
    p_jlc.add_argument("--dest", default=None,
                       help="override the catalog path (default: "
                            "~/.kicraft/jlcparts/cache.sqlite3 or $KICRAFT_JLCPARTS_DB)")
    p_jlc.add_argument("--min-stock", type=int, default=5,
                       help="prune rows below this stock to shrink the catalog "
                            "(default 5; 0 keeps everything, ~5.6 GB)")
    p_jlc.set_defaults(func=_cmd_jlcparts_update)

    p_add_part = sub.add_parser(
        "add-part",
        help=(
            "bundle a part for the parts library, from LCSC (via easyeda2kicad) "
            "or from user-supplied .kicad_sym + .kicad_mod files"
        ),
    )
    p_add_part.add_argument(
        "--from-lcsc",
        metavar="LCSC_ID",
        help="LCSC part number (e.g. C2837135). Mutually exclusive with --symbol/--footprint.",
    )
    p_add_part.add_argument(
        "--symbol",
        metavar="PATH",
        help=(
            "path to a .kicad_sym file (typically downloaded from SnapEDA, "
            "Ultra Librarian, or a silicon-vendor library). Pair with --footprint."
        ),
    )
    p_add_part.add_argument(
        "--footprint",
        metavar="PATH",
        help="path to a .kicad_mod file. Pair with --symbol.",
    )
    p_add_part.add_argument(
        "--symbol-name",
        default=None,
        metavar="NAME",
        help=(
            "override the symbol name written into the bundle "
            "(default: parsed from the .kicad_sym file)"
        ),
    )
    p_add_part.add_argument(
        "--mpn",
        default=None,
        help=(
            "manufacturer part number. Required with --symbol/--footprint; "
            "auto-detected from EasyEDA when using --from-lcsc."
        ),
    )
    p_add_part.add_argument(
        "--sourcing",
        action="append",
        metavar="VENDOR=ID",
        help=(
            "extra sourcing entry (repeatable). E.g. --sourcing digikey=ND-12-34 "
            "--sourcing mouser=581-XYZ. Vendor key must be lowercase alphanumeric."
        ),
    )
    p_add_part.add_argument(
        "--description",
        default=None,
        help="short part description for the manifest (default: derived from source)",
    )
    p_add_part.add_argument(
        "--datasheet-url",
        default=None,
        help="datasheet URL for the manifest",
    )
    p_add_part.add_argument(
        "--tag",
        action="append",
        default=[],
        help="add a tag (repeatable). E.g. --tag power --tag buck-boost",
    )
    p_add_part.add_argument(
        "--watch-out-for",
        default=None,
        help=(
            "free-form note recorded in the manifest, surfaced as the "
            "'Watch out for' block when the BOM stage lists this part"
        ),
    )
    p_add_part.add_argument(
        "--into",
        choices=["project", "home", "vendored"],
        default="project",
        help=(
            "destination tier (default: project, i.e. <cwd>/.kicraft/parts/); "
            "'vendored' writes into the repo parts library itself (for "
            "vendoring core-default bundles)"
        ),
    )
    p_add_part.add_argument(
        "--name",
        default=None,
        help=(
            "explicit library slug (lowercase, alphanumeric, dashes); "
            "auto-derived from MPN if omitted"
        ),
    )
    p_add_part.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing part directory with the same slug",
    )
    p_add_part.add_argument(
        "--no-3d",
        action="store_true",
        help=(
            "skip downloading the EasyEDA 3D model (--from-lcsc only); the "
            "footprint is then written without a (model ...) stanza"
        ),
    )
    p_add_part.add_argument(
        "--maturity",
        choices=list(get_args(Maturity)),
        default="prototype",
        help=(
            "quality badge for the new bundle (default: prototype). "
            "prototype=auto-fetched/unreviewed; reviewed=human-checked; "
            "production=polished + verified"
        ),
    )
    p_add_part.set_defaults(func=_cmd_add_part)

    p_val_part = sub.add_parser(
        "validate-part",
        help=(
            "validate a parts-library directory: manifest schema, required "
            "files, recomputed content_hash"
        ),
    )
    p_val_part.add_argument("path", help="path to the part directory")
    p_val_part.add_argument(
        "--update-hash",
        action="store_true",
        help="recompute content_hash and rewrite the manifest instead of failing",
    )
    p_val_part.set_defaults(func=_cmd_validate_part)

    p_promote = sub.add_parser(
        "promote-part",
        help="raise a bundle's maturity badge (prototype -> reviewed -> production)",
    )
    p_promote.add_argument("name", help="bundle name (library slug)")
    p_promote.add_argument(
        "--to",
        required=True,
        choices=["reviewed", "production"],
        help="target maturity; production requires a 3D model present in the bundle",
    )
    p_promote.add_argument(
        "--tier",
        choices=["project", "home", "vendored"],
        default="home",
        help="which tier's copy to promote (default: home)",
    )
    p_promote.set_defaults(func=_cmd_promote_part)

    p_fetch3d = sub.add_parser(
        "fetch-3d",
        help=(
            "download EasyEDA 3D models (WRL + STEP) into part bundles and "
            "point their footprints at ${KIPRJMOD}/3dmodels/<name>/..."
        ),
    )
    p_fetch3d.add_argument(
        "paths",
        nargs="*",
        help="part bundle directories (each containing a manifest.json)",
    )
    p_fetch3d.add_argument(
        "--all-vendored",
        action="store_true",
        help="process every bundle in the vendored parts library",
    )
    p_fetch3d.add_argument(
        "--overwrite",
        action="store_true",
        help="re-download even when the bundle already has 3d/ models",
    )
    p_fetch3d.add_argument(
        "--report",
        action="store_true",
        help=(
            "classify bundles (already / stock / needs fetch) without "
            "touching the network or any files"
        ),
    )
    p_fetch3d.set_defaults(func=_cmd_fetch_3d)

    p_look = sub.add_parser(
        "lookup-symbol",
        help="print pin inventory for a stock KiCad symbol (Library:Name)",
    )
    p_look.add_argument("symbol", help="KiCad symbol id, e.g. 'Device:R'")
    p_look.set_defaults(func=_cmd_lookup_symbol)

    p_search = sub.add_parser(
        "search-symbols",
        help="list stock KiCad symbols whose Library:Name matches keywords",
    )
    p_search.add_argument("query", help="keywords, e.g. 'conn 02x08' or 'crystal'")
    p_search.add_argument("--limit", type=int, default=40)
    p_search.set_defaults(func=_cmd_search_symbols)

    p_look_fp = sub.add_parser(
        "lookup-footprint",
        help="verify a stock KiCad footprint (Library:Name) exists and report its pad count",
    )
    p_look_fp.add_argument(
        "footprint", help="KiCad footprint id, e.g. 'Resistor_SMD:R_0603_1608Metric'"
    )
    p_look_fp.set_defaults(func=_cmd_lookup_footprint)

    p_search_fp = sub.add_parser(
        "search-footprints",
        help="list stock KiCad footprints whose Library:Name matches keywords",
    )
    p_search_fp.add_argument("query", help="keywords, e.g. 'pinheader 2x08' or 'barreljack'")
    p_search_fp.add_argument("--limit", type=int, default=40)
    p_search_fp.set_defaults(func=_cmd_search_footprints)

    p_syn = sub.add_parser(
        "synthesize",
        help="emit the KiCad project from a complete state.json",
    )
    p_syn.add_argument("state", help="path to state.json")
    p_syn.add_argument("out_dir", help="output directory (project_stem appended if absent)")
    p_syn.add_argument(
        "--smoke",
        action="store_true",
        help="run the solve-subcircuits smoke check (slow; needs PCB)",
    )
    p_syn.add_argument(
        "--smoke-timeout",
        type=float,
        default=60.0,
        help="timeout in seconds for the smoke check (default 60s)",
    )
    p_syn.add_argument(
        "--archive-root",
        default=None,
        help=f"archive root for the session snapshot (default {_default_archive_root()})",
    )
    p_syn.add_argument(
        "--no-archive",
        action="store_true",
        help="skip the post-synthesis session archive",
    )
    p_syn.set_defaults(func=_cmd_synthesize)

    p_build = sub.add_parser(
        "build",
        help="one shot: synthesize + place + route + verify + export fab package",
    )
    p_build.add_argument("state", help="path to state.json")
    p_build.add_argument(
        "out_dir", help="output directory (project_stem appended if absent)"
    )
    p_build.add_argument(
        "--quality",
        choices=["fast", "draft", "good", "best"],
        default="good",
        help=(
            "fast=single-pass solve-hierarchy; draft=reduced autoexperiment "
            "(quickest optimized result); good/best=autoexperiment optimization"
        ),
    )
    p_build.add_argument(
        "--archive-root",
        default=None,
        help=f"archive root for the session snapshot (default {_default_archive_root()})",
    )
    p_build.add_argument(
        "--no-archive",
        action="store_true",
        help="skip the post-build session archive",
    )
    p_build.set_defaults(func=_cmd_build)

    p_replay = sub.add_parser(
        "replay",
        help=(
            "re-run ONLY place + route on an already-synthesized workspace "
            "(deterministic; no LLM/synthesis) -- the repro harness for "
            "placement/compose code changes"
        ),
        description=(
            "Re-runs placement + routing (+ promote/verify/fab) on a fixed, "
            "already-synthesized workspace WITHOUT re-running synthesis, so a "
            "code change produces a reproducible board against a frozen input. "
            "Determinism is guaranteed for placement (pinned --seed); routing "
            "(FreeRouting) is best-effort-stable. Use --no-route for a fast, "
            "fully deterministic placement-only check."
        ),
    )
    p_replay.add_argument(
        "state", nargs="?",
        help="path to state.json (omit when using --project)",
    )
    p_replay.add_argument(
        "out_dir", nargs="?",
        help="output directory (project_stem appended if absent)",
    )
    p_replay.add_argument(
        "--project", default=None,
        help="discover artifacts on disk under this synthesized project dir "
             "(no state.json needed; preferred for testing)",
    )
    p_replay.add_argument(
        "--quality",
        choices=["fast", "draft", "good", "best"],
        default="fast",
        help="fast=deterministic single-pass solve-hierarchy (default; honors "
             "--no-route); draft/good/best=autoexperiment search (always routes)",
    )
    p_replay.add_argument(
        "--seed", type=int, default=0,
        help="placement RNG seed (default 0). Same seed + same workspace => "
             "same placement",
    )
    p_replay.add_argument(
        "--route", dest="route", action="store_true", default=True,
        help="route after placement (default)",
    )
    p_replay.add_argument(
        "--no-route", dest="route", action="store_false",
        help="placement only -- promote the placed board and skip "
             "routing/verify/fab (fast determinism check; fast engine only)",
    )
    p_replay.add_argument(
        "--no-fab", action="store_true",
        help="skip the fab-package export (stop after the verify gate)",
    )
    p_replay.add_argument(
        "--archive", dest="no_archive", action="store_false", default=True,
        help="archive a session snapshot (replay skips this by default)",
    )
    p_replay.set_defaults(func=_cmd_replay)

    p_mroute = sub.add_parser(
        "manual-route",
        help=(
            "route + promote a saved manual layout "
            "(.experiments/manual/manual_layout.json) and export the fab "
            "package; needs a previously synthesized workspace"
        ),
    )
    p_mroute.add_argument("state", help="path to state.json")
    p_mroute.add_argument(
        "out_dir", help="output directory (project_stem appended if absent)"
    )
    p_mroute.set_defaults(func=_cmd_manual_route)

    p_prep = sub.add_parser(
        "stage-prep",
        help=(
            "single-shot collector: print state + stage-specific extras as JSON "
            "(leaf library for architecture; batched symbol pinouts for wiring)"
        ),
    )
    p_prep.add_argument("stage", choices=KNOWN_STAGES, help="stage name")
    p_prep.add_argument(
        "state",
        nargs="?",
        default=".kicraft/state.json",
        help="path to state.json (default .kicraft/state.json)",
    )
    p_prep.set_defaults(func=_cmd_stage_prep)

    p_commit = sub.add_parser(
        "stage-commit",
        help=(
            "atomic stage commit: validate the proposed slot, merge into "
            "state.json, append history, archive"
        ),
    )
    p_commit.add_argument("stage", choices=KNOWN_STAGES, help="stage name")
    p_commit.add_argument(
        "--slot-file",
        required=True,
        help="path to a JSON file containing the proposed slot value",
    )
    p_commit.add_argument(
        "--questions-file",
        default=None,
        help="optional path to a JSON file with a list of Question dicts to attach to the stage",
    )
    p_commit.add_argument(
        "--history-message",
        default=None,
        help="optional assistant summary text to append to state.history",
    )
    p_commit.add_argument(
        "--project-stem",
        default=None,
        help="for the intent stage, the top-level project_stem to set (e.g. ESP32_WEATHER_STATION)",
    )
    p_commit.add_argument(
        "state",
        nargs="?",
        default=".kicraft/state.json",
        help="path to state.json (default .kicraft/state.json)",
    )
    p_commit.add_argument(
        "--no-archive",
        action="store_true",
        help="skip the post-commit session archive",
    )
    p_commit.add_argument(
        "--archive-root",
        default=None,
        help=f"archive root directory (default {_default_archive_root()})",
    )
    p_commit.set_defaults(func=_cmd_stage_commit)

    p_arch = sub.add_parser(
        "archive",
        help="snapshot the current .kicraft/ into the central session archive",
    )
    p_arch.add_argument(
        "state",
        nargs="?",
        default=".kicraft/state.json",
        help="path to state.json (default .kicraft/state.json)",
    )
    p_arch.add_argument(
        "--root",
        default=str(_default_archive_root()),
        help=f"archive root directory (default {_default_archive_root()})",
    )
    p_arch.set_defaults(func=_cmd_archive)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
