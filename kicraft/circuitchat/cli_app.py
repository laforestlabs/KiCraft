"""`kicraft-circuitchat` — non-interactive helpers used by the Claude Code skill.

The skill at ``.claude/skills/circuitchat/`` drives the LLM conversation;
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
  synthesizer. Wraps ``kicraft.circuitchat.synthesize.run``.
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
from .synthesis.symbol_pinout import SymbolNotFoundError, lookup_pins
from .synthesis.validation import (
    CheckResult,
    SynthesisValidationError,
    check_net_coverage,
    check_pin_existence,
)

KNOWN_STAGES = ("intent", "functional_spec", "architecture", "bom", "wiring")


_SAFE_STEM_RE = re.compile(r"[^A-Z0-9_]")


def _default_archive_root() -> Path:
    return Path.home() / ".kicraft" / "sessions"


def _utc_compact_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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

    if state.bom is not None and state.bom.connections:
        for check in (check_pin_existence(state.bom), check_net_coverage(state.bom)):
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


def _cmd_add_part(args: argparse.Namespace) -> int:
    """Fetch a part from LCSC via easyeda2kicad and bundle it for the parts library.

    Writes the canonical layout::

        <dest>/<libname>/manifest.json
        <dest>/<libname>/<libname>.kicad_sym
        <dest>/<libname>/<libname>.pretty/<footprint_name>.kicad_mod

    Then computes the content_hash and rewrites the manifest. After this
    runs, ``list-parts`` shows the new entry and the resolver picks it
    up automatically the next time a BOM references ``<libname>:<sym>``.
    """
    if not args.from_lcsc:
        print("add-part requires --from-lcsc <LCSC_ID>", file=sys.stderr)
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

    from kicraft.parts_library import (
        PartManifest,
        Provenance,
        compute_content_hash,
        dump_manifest,
    )

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

    # Pick destination tier.
    if args.into == "home":
        dest_base = Path.home() / ".kicraft" / "parts"
    else:  # project (default)
        dest_base = Path.cwd() / ".kicraft" / "parts"
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
    symbol_name = ee_symbol.info.name

    # Footprint: write into the .pretty dir. 3D model path is left empty
    # (no .step yet); user can re-fetch with a 3D flag in a follow-up.
    footprint_name = ee_footprint.info.name
    fp_path = pretty_dir / f"{footprint_name}.kicad_mod"
    ExporterFootprintKicad(footprint=ee_footprint).export(
        footprint_full_path=str(fp_path), model_3d_path=""
    )

    # Compose the manifest, then compute content_hash and rewrite once.
    sourcing: dict[str, str] = {"lcsc": lcsc_id}
    if ee_symbol.info.mpn and ee_symbol.info.manufacturer:
        # MPN goes in its own field; manufacturer is informational only.
        pass

    description = (
        ee_symbol.info.description
        or f"{ee_symbol.info.manufacturer or ''} {ee_symbol.info.mpn or ''}".strip()
        or f"part {symbol_name}"
    )
    datasheet = ee_symbol.info.datasheet or None
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
        tags=[],
        watch_out_for=None,
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
    dump_manifest(manifest, part_dir)
    actual_hash = compute_content_hash(part_dir)
    dump_manifest(manifest.model_copy(update={"content_hash": actual_hash}), part_dir)

    print(
        f"OK added {libname}@0.1.0 -> {part_dir}\n"
        f"  symbol:    {libname}:{symbol_name}\n"
        f"  footprint: {libname}:{footprint_name}\n"
        f"  mpn:       {manifest.mpn}\n"
        f"  sourcing:  lcsc:{lcsc_id}\n"
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
    if "(footprint " not in fp.read_text():
        print(f"footprint file {fp} does not contain a (footprint ...) block", file=sys.stderr)
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
        for part in state.bom.parts:
            sym = part.symbol
            if sym in pinouts:
                continue
            try:
                pinouts[sym] = lookup_pins(sym)
            except (SymbolNotFoundError, ValueError) as e:
                pinouts[sym] = {"error": str(e)}
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
        for check in (check_pin_existence(state.bom), check_net_coverage(state.bom)):
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
        print(str(e), file=sys.stderr)
        return 5

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="kicraft-circuitchat",
        description=(
            "Deterministic helpers for the CircuitChat skill. The LLM-driven "
            "conversation lives in the Claude Code skill at "
            ".claude/skills/circuitchat/; this CLI handles state validation, "
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

    p_add_part = sub.add_parser(
        "add-part",
        help=(
            "fetch a part from LCSC via easyeda2kicad and bundle it for the "
            "parts library (writes symbol + footprint + stub manifest)"
        ),
    )
    p_add_part.add_argument(
        "--from-lcsc",
        metavar="LCSC_ID",
        help="LCSC part number (e.g. C2837135)",
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
