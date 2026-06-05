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
import re
import shutil
import sys
from pathlib import Path

from pydantic import ValidationError

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
    check_inter_sheet_nets_realized,
    check_net_coverage,
    check_pin_existence,
    check_sheets_have_parts,
)

KNOWN_STAGES = ("intent", "functional_spec", "architecture", "bom", "wiring")


_SAFE_STEM_RE = re.compile(r"[^A-Z0-9_]")


def _default_archive_root() -> Path:
    return Path.home() / ".kicraft" / "sessions"


def _utc_compact_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _unresolved_footprints(bom, project_root: Path) -> list[str]:
    """Return a human-readable list of BOM parts whose ``footprint`` does
    not resolve to a real ``.kicad_mod`` on disk (across the four parts-
    library tiers + stock KiCad). An empty list means every footprint
    resolves. Catches LLM footprint-name hallucination (e.g. a plausible
    truncation like ``SW_SPST_PTS645`` for ``SW_SPST_PTS645Sx43SMTR92``).
    """
    bad: list[str] = []
    for part in bom.parts:
        fp = part.footprint or ""
        library, _, name = fp.partition(":")
        if not library or not name:
            bad.append(f"{part.ref}: footprint {fp!r} is not 'Library:Name'")
            continue
        try:
            pretty = resolve_footprint_library_path(library, project_root=project_root)
        except LibraryNotFoundError:
            bad.append(
                f"{part.ref}: footprint library {library!r} not found (footprint {fp!r})"
            )
            continue
        if not (pretty / f"{name}.kicad_mod").is_file():
            bad.append(
                f"{part.ref}: no '{name}.kicad_mod' in {pretty} (footprint {fp!r})"
            )
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
            bad.append(f"{part.ref}: symbol {sym!r} did not resolve ({e})")
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
        checks = [check_pin_existence(state.bom), check_net_coverage(state.bom)]
        if state.architecture is not None:
            checks.append(
                check_inter_sheet_nets_realized(state.architecture, state.bom)
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


def _cmd_lookup_symbol(args: argparse.Namespace) -> int:
    try:
        info = lookup_pins(args.symbol)
    except SymbolNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(json.dumps(info, indent=2))
    return 0


def _cmd_search_symbols(args: argparse.Namespace) -> int:
    matches = search_symbols(args.query, limit=args.limit)
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
        print(str(e), file=sys.stderr)
        return 2
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(json.dumps(info, indent=2))
    return 0


def _cmd_search_footprints(args: argparse.Namespace) -> int:
    matches = search_footprints(args.query, limit=args.limit)
    if not matches:
        print(f"no stock KiCad footprints match {args.query!r}; try fewer or broader terms",
              file=sys.stderr)
        return 0
    for fp in matches:
        print(fp)
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
    block = _format_available_parts_block(active)
    if block is None:
        print("(no parts available in the library)")
        return 0
    print(block)
    return 0


def _pick_lcsc(mpn: str, results: list[dict]) -> dict | None:
    """Choose the single best JLCPCB search result for `mpn`, or None if
    ambiguous. Prefer an exact (case-insensitive) match on the part's
    model/MPN, breaking ties by stock (desc) then Basic-over-Extended. With
    no exact match but exactly one result, take it; otherwise return None so
    the caller surfaces the candidate list rather than guessing wrong."""
    target = (mpn or "").strip().upper()
    exact = [r for r in results if (r.get("model") or "").strip().upper() == target]
    if exact:
        exact.sort(key=lambda r: (-(r.get("stock") or 0), r.get("type") != "Basic"))
        return exact[0]
    if len(results) == 1:
        return results[0]
    return None


def _cmd_lookup_lcsc_id(args: argparse.Namespace) -> int:
    """Resolve a manufacturer part number to an LCSC part number.

    Checks the parts-library manifests first (offline, authoritative), then
    falls back to a JLCPCB keyword search. Prints JSON; exits 0 when a single
    LCSC id is resolved, 4 otherwise (with a candidate list to choose from).
    Lets the BOM sub-agent own MPN->LCSC resolution without the main thread
    reaching for WebSearch.
    """
    mpn = args.mpn
    target = mpn.strip().upper()

    # 1. Parts-library manifests — authoritative and offline.
    active, _broken = _load_library_parts(Path.cwd())
    for part in active:
        m = part.manifest
        if (m.mpn or "").strip().upper() == target:
            lcsc = (m.sourcing or {}).get("lcsc")
            if lcsc:
                print(json.dumps(
                    {"ok": True, "mpn": mpn, "lcsc": lcsc,
                     "source": "parts-library", "name": m.name},
                    indent=2,
                ))
                return 0

    # 2. JLCPCB keyword search — network, best-effort. search_jlcpcb_components
    #    already degrades to an empty result list on any network/parse error.
    try:
        from easyeda2kicad.easyeda.easyeda_api import EasyedaApi
    except ImportError as e:
        print(json.dumps(
            {"ok": False, "mpn": mpn, "candidates": [],
             "error": f"easyeda2kicad not installed: {e}"},
            indent=2,
        ))
        return 4

    results = (
        EasyedaApi().search_jlcpcb_components(keyword=mpn, page_size=10).get("results", [])
    )
    fields = ("lcsc", "model", "brand", "package", "stock", "type")
    best = _pick_lcsc(mpn, results)
    if best and best.get("lcsc"):
        print(json.dumps(
            {"ok": True, "mpn": mpn, "lcsc": best["lcsc"], "source": "jlcpcb",
             "match": {k: best.get(k) for k in fields}},
            indent=2,
        ))
        return 0

    print(json.dumps(
        {"ok": False, "mpn": mpn,
         "candidates": [{k: r.get(k) for k in fields} for r in results[:10]],
         "hint": "no unambiguous LCSC id; pick a candidate and pass it to "
                 "add-part --from-lcsc C<NNNNN>"},
        indent=2,
    ))
    return 4


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
        f"  tier:      {args.into}"
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
            EasyedaFootprintImporter,
            EasyedaSymbolImporter,
        )
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
    raw_symbol_name = ee_symbol.info.name
    symbol_name = _sanitize_kicad_name(raw_symbol_name)
    if symbol_name != raw_symbol_name:
        sym_path.write_text(
            _normalize_symbol_text(sym_path.read_text(), raw_symbol_name, symbol_name)
        )

    # Footprint: write into the .pretty dir. 3D model path is left empty
    # (no .step yet); user can re-fetch with a 3D flag in a follow-up.
    raw_footprint_name = ee_footprint.info.name
    footprint_name = _sanitize_kicad_name(raw_footprint_name)
    fp_path = pretty_dir / f"{footprint_name}.kicad_mod"
    ExporterFootprintKicad(footprint=ee_footprint).export(
        footprint_full_path=str(fp_path), model_3d_path=""
    )
    if footprint_name != raw_footprint_name:
        _, fp_fixed = _sanitize_footprint_text(fp_path.read_text(), raw_footprint_name)
        fp_path.write_text(fp_fixed)

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

    print(
        f"OK added {libname}@0.1.0 -> {part_dir}\n"
        f"  symbol:    {libname}:{symbol_name}\n"
        f"  footprint: {libname}:{footprint_name}\n"
        f"  mpn:       {manifest.mpn}\n"
        f"  sourcing:  {', '.join(f'{k}:{v}' for k, v in sourcing.items())}\n"
        f"  tier:      {args.into}"
    )
    return 0


def _cmd_validate_part(args: argparse.Namespace) -> int:
    """Validate a parts-library directory: schema, files, content_hash.

    Three checks, in order: (1) the manifest parses and the directory
    name matches ``name``; (2) the symbol + footprint files declared in
    the manifest exist and parse; (3) the recomputed content_hash matches
    the value stored in the manifest. With ``--update-hash``, recompute
    and rewrite the manifest's ``content_hash`` instead of failing — used
    when authoring a new part or after deliberate edits.
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
        checks = [check_pin_existence(state.bom), check_net_coverage(state.bom)]
        if state.architecture is not None:
            # Architecture declared these inter-sheet nets; the wiring stage
            # must realize each signal endpoint, or the emitter leaves a sheet
            # pin with no hierarchical label (caught only by §9.12 ERC at
            # synthesis time otherwise).
            checks.append(
                check_inter_sheet_nets_realized(state.architecture, state.bom)
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
    "fast": {"engine": "solve-hierarchy", "rounds": 1},
    "good": {"engine": "autoexperiment", "rounds": 3},
    "best": {"engine": "autoexperiment", "rounds": 6},
}


def _run_layout(quality: str, root_sch: Path, pcb: Path) -> int:
    """Run the placement+routing engine in-process (inherits this env's pcbnew)."""
    preset = _QUALITY_PRESETS.get(quality, _QUALITY_PRESETS["good"])
    if preset["engine"] == "solve-hierarchy":
        from kicraft.cli.solve_hierarchy import main as _solve_hierarchy_main

        return _solve_hierarchy_main([str(root_sch), "--pcb", str(pcb), "--route"])
    from kicraft.cli.autoexperiment import main as _autoexperiment_main

    return _autoexperiment_main(
        [str(pcb), "--schematic", str(root_sch), "--rounds", str(preset["rounds"])]
    )


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


def _verify_routed_board(pcb: Path) -> dict:
    """Acceptance gate: no shorts, no unconnected (connector-shield items waived)."""
    from kicraft.autoplacer.config import DEFAULT_CONFIG
    from kicraft.autoplacer.freerouting_runner import validate_routed_board

    v = validate_routed_board(str(pcb), cfg=dict(DEFAULT_CONFIG))
    drc = v.get("drc", {}) or {}
    shorts = int(drc.get("shorts", 0) or 0)
    unconnected = int(drc.get("unconnected", 0) or 0)
    return {
        "ok": bool(v.get("accepted", False)) and shorts == 0 and unconnected == 0,
        "shorts": shorts,
        "unconnected": unconnected,
        "reasons": v.get("rejection_reasons", []),
        "tracks": v.get("track_summary", {}) or {},
    }


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
        print(f"synthesis checks failed (not ERC-clean):\n{e}", file=sys.stderr)
        return 5
    _persist_artifacts(state, state_path, artifacts)
    _write_synthesis_check(state_path, state.project_stem, results, ok=True)

    stem = state.project_stem
    project_dir = Path(artifacts.project_dir)
    root_sch = Path(artifacts.root_sch)
    pcb = project_dir / f"{stem}.kicad_pcb"
    print(f"[build]     synthesized {project_dir} (ERC clean)")

    # 2. Optimize placement + route (leaves then parent) via the layout engine.
    print(f"[build] 2/5 place + route (quality={args.quality}) -- may take minutes ...")
    rc = _run_layout(args.quality, root_sch, pcb)
    if rc != 0:
        print(f"error: layout/route engine exited {rc}", file=sys.stderr)
        return 6

    # 3. Promote the routed parent to the project's main PCB.
    routed = _find_routed_parent(project_dir)
    if routed is None:
        print(
            "error: the layout engine produced no routed parent board -- the "
            "parent compose/route failed (board not routable as placed). "
            "Inspect .experiments/.../_search for rejected candidates.",
            file=sys.stderr,
        )
        return 6
    shutil.copy2(routed, pcb)
    print(f"[build] 3/5 promoted routed parent -> {pcb.name}")

    # 4. Verification gate: no shorts, no unconnected.
    gate = _verify_routed_board(pcb)
    print(
        f"[build] 4/5 verify: shorts={gate['shorts']} unconnected={gate['unconnected']} "
        f"traces={gate['tracks'].get('traces', '?')}"
    )
    if not gate["ok"]:
        print(
            f"error: routed board is NOT fab-ready -- shorts={gate['shorts']}, "
            f"unconnected={gate['unconnected']}, reasons={gate['reasons']}",
            file=sys.stderr,
        )
        return 7

    # 5. Export the fab package (Gerbers + drill + CPL + BOM, zipped).
    print("[build] 5/5 export fab package (Gerbers + drill + CPL + BOM) ...")
    from kicraft.design.synthesis.fab_export import export_fab

    bom_parts = [p.model_dump() for p in state.bom.parts]
    fab = export_fab(str(pcb), str(project_dir), stem, bom_parts=bom_parts)

    artifacts.routed_pcb = pcb
    artifacts.fab_zip = Path(fab["zip"])
    _persist_artifacts(state, state_path, artifacts)

    print()
    print(f"BUILD COMPLETE: {stem}")
    print(f"  routed PCB : {pcb}")
    print(
        f"  DRC        : 0 shorts, 0 unconnected "
        f"({gate['tracks'].get('traces', '?')} traces, {gate['tracks'].get('vias', '?')} vias)"
    )
    print(f"  fab package: {fab['zip']}")
    print(f"  contents   : {', '.join(fab['files'])}")

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
                synth_dir=project_dir,
                synth_results=results,
            )
            print(f"  archived   : {dest}")
        except OSError as e:
            print(f"warning: archive failed: {e}", file=sys.stderr)
    return 0


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
            "via the parts library, then a JLCPCB keyword search"
        ),
    )
    p_lcsc.add_argument("mpn", help="manufacturer part number to resolve")
    p_lcsc.set_defaults(func=_cmd_lookup_lcsc_id)

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
        choices=["project", "home"],
        default="project",
        help="destination tier (default: project, i.e. <cwd>/.kicraft/parts/)",
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
        choices=["fast", "good", "best"],
        default="good",
        help="fast=single-pass solve-hierarchy; good/best=autoexperiment optimization",
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
