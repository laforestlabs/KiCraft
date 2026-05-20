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
    _load_library_leaves,
    _validate_library_picks,
)
from .models import ConversationState
from .synthesize import SynthesisInputError, run as run_synth
from .synthesis.symbol_pinout import SymbolNotFoundError, lookup_pins
from .synthesis.validation import (
    CheckResult,
    SynthesisValidationError,
    check_net_coverage,
    check_pin_existence,
)


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
