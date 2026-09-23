"""Run current manufacturing without reserializing a native design state.

The native design pipeline owns its canonical ``state.json`` schema.  Current
manufacturing can add fields while loading that state, so it runs against the
sibling ``build_state.json`` snapshot.  The only cross-schema handoff back to
native state is the shared ``artifacts`` JSON object.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from kicraft.fsutil import atomic_write_text


_SUPPORTED_COMMANDS = frozenset({"build", "manual-route", "replay"})
_SNAPSHOT_NAME = "build_state.json"


def _wrapper_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _publication_error(rc: int, message: str) -> int:
    """Report a failed handoff without hiding a manufacturing failure code."""
    print(f"error: {message}", file=sys.stderr)
    return rc if rc else 1


def _load_canonical(state_path: Path) -> tuple[bytes, dict] | None:
    try:
        raw = state_path.read_bytes()
    except OSError as exc:
        _wrapper_error(f"could not read canonical state {state_path}: {exc}")
        return None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _wrapper_error(f"canonical state {state_path} is not valid JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        _wrapper_error(f"canonical state {state_path} must be a JSON object")
        return None
    return raw, payload


def _publish_snapshot_artifacts(
    state_path: Path,
    original_bytes: bytes,
    canonical: dict,
    snapshot_path: Path,
    rc: int,
) -> int:
    """Merge the snapshot's shared artifact payload without model validation."""
    try:
        snapshot_payload = json.loads(snapshot_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _publication_error(rc, f"could not read build snapshot {snapshot_path}: {exc}")
    if not isinstance(snapshot_payload, dict) or "artifacts" not in snapshot_payload:
        return _publication_error(
            rc,
            f"build snapshot {snapshot_path} has no artifacts field; canonical state was not changed",
        )

    try:
        unchanged = state_path.read_bytes() == original_bytes
    except OSError as exc:
        return _publication_error(rc, f"could not re-read canonical state {state_path}: {exc}")
    if not unchanged:
        return _publication_error(
            rc,
            "canonical state changed while manufacturing ran; artifacts remain in "
            f"{snapshot_path} and were not published",
        )

    canonical["artifacts"] = snapshot_payload["artifacts"]
    try:
        atomic_write_text(
            state_path,
            json.dumps(canonical, indent=2, ensure_ascii=False) + "\n",
        )
    except (OSError, TypeError, ValueError) as exc:
        return _publication_error(rc, f"could not publish build artifacts to {state_path}: {exc}")
    return rc


def _synthesize_native(snapshot_path: Path, out_dir: Path) -> int:
    """Compile the snapshot into a schematic with CURRENT manufacturing.

    One design's manufacture must validate with one library, and manufacturing is
    this tree's job for every backend (see pipeline.build_command). Compiling the
    snapshot with the PINNED tree instead ran its older check set on a design the
    current tree had just accepted: bc6a2f8's §9.9 has no pinless-part exemption,
    so a correct mounting-hole sheet failed synthesis outright and no legacy
    design could ever reach placement (2026-09-23, KC-CG58R4: "9.9 connectivity:
    PCB_MECHANICAL.kicad_sch: 4 components, 0 wires, 0 power symbols" -- the same
    snapshot synthesizes clean here, ERC included). The snapshot still isolates
    the canonical state schema; only the interpreter and the check set change."""
    from kicraft.cli.artifact_paths import provenance_path

    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "kicraft.design.cli_app",
            "synthesize",
            str(snapshot_path.resolve()),
            str(out_dir.resolve()),
            "--no-archive",
        ],
        cwd=str(repo_root),
        env={**os.environ, "PYTHONPATH": str(repo_root)},
    )
    if completed.returncode:
        return completed.returncode
    artifacts = json.loads(snapshot_path.read_text(encoding="utf-8"))["artifacts"]
    pcb = Path(artifacts["project_dir"]) / f"{artifacts['project_stem']}.kicad_pcb"
    # Fresh synthesis replaced the board. Old partial-board provenance must not
    # make replay restore a seed from a previous design.
    provenance_path(pcb).unlink(missing_ok=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``build|manual-route|replay STATE OUT [options]`` through a snapshot."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 3 or args[0] not in _SUPPORTED_COMMANDS:
        return _wrapper_error("expected build, manual-route, or replay followed by STATE OUT_DIR")

    state_path = Path(args[1])
    snapshot_path = state_path.with_name(_SNAPSHOT_NAME)
    if snapshot_path == state_path:
        return _wrapper_error(f"canonical state cannot itself be named {_SNAPSHOT_NAME}")

    loaded = _load_canonical(state_path)
    if loaded is None:
        return 2
    original_bytes, canonical = loaded
    try:
        atomic_write_text(snapshot_path, original_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        return _wrapper_error(f"could not create build snapshot {snapshot_path}: {exc}")

    forwarded = list(args)
    forwarded[1] = str(snapshot_path)
    if args[0] == "build":
        rc = _synthesize_native(snapshot_path, Path(args[2]))
        if rc:
            return _publish_snapshot_artifacts(
                state_path, original_bytes, canonical, snapshot_path, rc
            )
        # Keep build's defaults; explicit caller options follow and override them.
        forwarded = [
            "replay",
            str(snapshot_path),
            args[2],
            "--quality",
            "good",
            "--archive",
            *args[3:],
        ]

    # Import only after the snapshot exists: this wrapper itself never loads the
    # canonical state through the current model.
    from kicraft.design import cli_app

    rc = cli_app.main(forwarded)
    return _publish_snapshot_artifacts(state_path, original_bytes, canonical, snapshot_path, rc)


if __name__ == "__main__":
    raise SystemExit(main())
