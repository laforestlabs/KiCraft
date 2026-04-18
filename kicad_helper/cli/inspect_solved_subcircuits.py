#!/usr/bin/env python3
"""Inspect and transform solved subcircuit artifacts.

This CLI is a composition-side debug tool for the subcircuits redesign.

It loads canonical solved subcircuit artifacts from `.experiments/subcircuits`,
prints summaries, optionally emits JSON, and can apply rigid transforms
(translation + rotation) to inspect how a solved child artifact would behave
when instantiated inside a parent composition.

Supported workflows:
- inspect one artifact directory
- inspect all artifact directories under a project
- print canonical artifact summaries
- print transformed rigid-instance summaries
- emit JSON for downstream tooling/debugging

Usage:
    python3 inspect_solved_subcircuits.py --project .
    python3 inspect_solved_subcircuits.py --artifact .experiments/subcircuits/<slug>
    python3 inspect_solved_subcircuits.py --project . --json
    python3 inspect_solved_subcircuits.py --project . --origin-x 50 --origin-y 20 --rotation 90
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


from kicad_helper.autoplacer.brain.subcircuit_instances import (
    artifact_debug_dict,
    artifact_summary,
    load_solved_artifacts,
    transform_loaded_artifact,
    transformed_debug_dict,
    transformed_summary,
)
from kicad_helper.autoplacer.brain.types import Point


def _discover_artifact_dirs(project_dir: Path) -> list[Path]:
    """Find solved subcircuit artifact directories under a project."""
    root = project_dir / ".experiments" / "subcircuits"
    if not root.exists():
        return []

    artifact_dirs: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        metadata = child / "metadata.json"
        debug = child / "debug.json"
        if metadata.exists() and debug.exists():
            artifact_dirs.append(child)
    return artifact_dirs


def _iter_artifact_dirs(
    project_dir: Path | None,
    artifact_dirs: list[str],
) -> list[str | Path]:
    """Resolve artifact directories from CLI inputs."""
    resolved: list[str | Path] = []

    for artifact in artifact_dirs:
        path = Path(artifact).resolve()
        if path not in resolved:
            resolved.append(path)

    if project_dir is not None:
        for path in _discover_artifact_dirs(project_dir.resolve()):
            if path not in resolved:
                resolved.append(path)

    return resolved


def _print_human_summary(
    loaded_artifacts,
    origin: Point | None,
    rotation: float,
) -> None:
    print("=== Solved Subcircuit Artifacts ===")
    print(f"artifacts : {len(loaded_artifacts)}")
    print()

    for artifact in loaded_artifacts:
        print(f"- {artifact_summary(artifact)}")
        print(f"  artifact_dir : {artifact.artifact_dir}")
        print(f"  metadata_json: {artifact.source_files.get('metadata_json', '')}")
        print(f"  debug_json   : {artifact.source_files.get('debug_json', '')}")
        print(f"  mini_pcb     : {artifact.source_files.get('mini_pcb', '')}")

        if origin is not None:
            transformed = transform_loaded_artifact(
                artifact,
                origin=origin,
                rotation=rotation,
            )
            print(f"  transformed  : {transformed_summary(transformed)}")
        print()


def _json_payload(
    loaded_artifacts,
    origin: Point | None,
    rotation: float,
) -> dict:
    payload = {
        "artifact_count": len(loaded_artifacts),
        "artifacts": [],
    }

    for artifact in loaded_artifacts:
        item = {
            "artifact": artifact_debug_dict(artifact),
        }
        if origin is not None:
            transformed = transform_loaded_artifact(
                artifact,
                origin=origin,
                rotation=rotation,
            )
            item["transformed"] = transformed_debug_dict(transformed)
        payload["artifacts"].append(item)

    return payload


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and transform solved subcircuit artifacts"
    )
    parser.add_argument(
        "--project",
        help="Project directory containing .experiments/subcircuits",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Specific solved artifact directory to inspect (repeatable)",
    )
    parser.add_argument(
        "--origin-x",
        type=float,
        help="Optional rigid transform origin X in mm",
    )
    parser.add_argument(
        "--origin-y",
        type=float,
        help="Optional rigid transform origin Y in mm",
    )
    parser.add_argument(
        "--rotation",
        type=float,
        default=0.0,
        help="Optional rigid transform rotation in degrees (default: 0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    project_dir = Path(args.project).resolve() if args.project else None
    artifact_dirs = _iter_artifact_dirs(project_dir, args.artifact)

    if not artifact_dirs:
        print(
            "error: no solved subcircuit artifacts found; provide --artifact or --project",
            file=sys.stderr,
        )
        return 2

    try:
        loaded_artifacts = load_solved_artifacts(list(artifact_dirs))
    except Exception as exc:
        print(f"error: failed to load solved artifacts: {exc}", file=sys.stderr)
        return 1

    origin = None
    if args.origin_x is not None or args.origin_y is not None:
        if args.origin_x is None or args.origin_y is None:
            print(
                "error: both --origin-x and --origin-y are required for transforms",
                file=sys.stderr,
            )
            return 2
        origin = Point(args.origin_x, args.origin_y)

    if args.json:
        print(
            json.dumps(_json_payload(loaded_artifacts, origin, args.rotation), indent=2)
        )
        return 0

    _print_human_summary(loaded_artifacts, origin, args.rotation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
