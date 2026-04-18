#!/usr/bin/env python3
"""Visible hierarchical pipeline runner for KiCad projects.

This script is intended to be the user-facing entrypoint for the subcircuits
redesign branch. It orchestrates the current bottom-up hierarchical flow:

1. Solve lowest-level leaf subcircuits with real placement + FreeRouting
2. Persist accepted routed leaf artifacts under `.experiments/subcircuits/`
3. Compose a selected parent from those routed child artifacts
4. Optionally stamp and route a visible parent board for inspection

The long-term goal is a single command that walks the full hierarchy layer by
layer like legos until the top-level parent is assembled. This first version
focuses on making the current leaf-to-parent flow reproducible and visible.

Notes:
- This runner intentionally prefers the real subcircuit pipeline over
  autoexperiment-style board-only optimization.
- Leaf routing is expected to go through FreeRouting, not the lightweight
  Manhattan fallback path.
- Parent routing is currently delegated to the existing hierarchical demo
  scaffold until the production parent pipeline fully absorbs that behavior.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[4]


def _detect_project_files(project_dir):
    """Auto-detect KiCad project files from directory."""
    pro_files = list(project_dir.glob("*.kicad_pro"))
    if pro_files:
        stem = pro_files[0].stem
        return project_dir / f"{stem}.kicad_sch", project_dir / f"{stem}.kicad_pcb"
    return None, None


DEFAULT_SCHEMATIC, DEFAULT_PCB = _detect_project_files(PROJECT_DIR)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / ".experiments" / "hierarchical_pipeline"


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    label: str,
    capture_json: bool = False,
) -> tuple[int, str, dict[str, Any] | None]:
    print()
    print("=" * 78)
    print(label)
    print("=" * 78)
    print("Command:")
    print("  " + " ".join(cmd))
    print()

    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    elapsed = time.monotonic() - start

    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(
            proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr
        )

    print(f"[{label}] exit_code={proc.returncode} elapsed_s={elapsed:.2f}")

    payload = None
    if capture_json and proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = None

    return proc.returncode, proc.stdout, payload


def _artifact_root(project_dir: Path) -> Path:
    return project_dir / ".experiments" / "subcircuits"


def _discover_artifact_dirs(project_dir: Path) -> list[Path]:
    root = _artifact_root(project_dir)
    if not root.exists():
        return []
    artifact_dirs: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "metadata.json").exists() and (
            child / "solved_layout.json"
        ).exists():
            artifact_dirs.append(child)
    return artifact_dirs


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _accepted_artifacts(project_dir: Path) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for artifact_dir in _discover_artifact_dirs(project_dir):
        solved_layout_path = artifact_dir / "solved_layout.json"
        metadata_path = artifact_dir / "metadata.json"
        try:
            solved_layout = _load_json(solved_layout_path)
            metadata = _load_json(metadata_path)
        except Exception:
            continue

        validation = solved_layout.get("validation", {})
        if validation.get("accepted") is True:
            accepted.append(
                {
                    "artifact_dir": str(artifact_dir),
                    "sheet_name": solved_layout.get("sheet_name", ""),
                    "instance_path": solved_layout.get("instance_path", ""),
                    "trace_count": len(solved_layout.get("traces", [])),
                    "via_count": len(solved_layout.get("vias", [])),
                    "anchor_validation": solved_layout.get("anchor_validation", {}),
                    "validation": validation,
                    "metadata": metadata,
                    "solved_layout": solved_layout,
                }
            )
    return accepted


def _summarize_leaf_results(project_dir: Path) -> dict[str, Any]:
    artifacts = _accepted_artifacts(project_dir)
    return {
        "artifact_root": str(_artifact_root(project_dir)),
        "accepted_artifact_count": len(artifacts),
        "accepted_artifacts": artifacts,
    }


def _print_leaf_summary(summary: dict[str, Any]) -> None:
    print()
    print("Accepted routed leaf artifacts")
    print("-" * 78)
    print(f"artifact_root           : {summary['artifact_root']}")
    print(f"accepted_artifact_count : {summary['accepted_artifact_count']}")
    print()

    for artifact in summary["accepted_artifacts"]:
        anchor_validation = artifact.get("anchor_validation", {})
        print(f"- {artifact['sheet_name']} [{artifact['instance_path']}]")
        print(f"  artifact_dir          : {artifact['artifact_dir']}")
        print(f"  traces                : {artifact['trace_count']}")
        print(f"  vias                  : {artifact['via_count']}")
        print(
            "  all_required_anchors  : "
            f"{anchor_validation.get('all_required_ports_anchored', False)}"
        )
        print()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the visible hierarchical subcircuit pipeline"
    )
    parser.add_argument(
        "--project",
        default=str(PROJECT_DIR),
        help="Project directory (default: repository root)",
    )
    parser.add_argument(
        "--schematic",
        default=str(DEFAULT_SCHEMATIC) if DEFAULT_SCHEMATIC else None,
        help="Top-level schematic path (auto-detected from *.kicad_pro)",
    )
    parser.add_argument(
        "--pcb",
        default=str(DEFAULT_PCB) if DEFAULT_PCB else None,
        help="Top-level PCB path (auto-detected from *.kicad_pro)",
    )
    parser.add_argument(
        "--config",
        help="Optional config file passed through to child commands",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Leaf solve rounds per subcircuit (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for leaf solving (default: 0)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Restrict to specific leaf/instance selectors (repeatable)",
    )
    parser.add_argument(
        "--parent",
        default="/",
        help="Parent selector for composition/visible assembly (default: /)",
    )
    parser.add_argument(
        "--compose-only",
        action="store_true",
        help="Stop after parent composition JSON snapshot",
    )
    parser.add_argument(
        "--skip-parent-routing",
        action="store_true",
        help="Skip visible parent board stamping/FreeRouting stage",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for pipeline artifacts",
    )
    parser.add_argument(
        "--jar",
        help="Override FreeRouting jar path for parent routing stage",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        help="Override FreeRouting timeout for parent routing stage",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        help="Override FreeRouting max passes for parent routing stage",
    )
    parser.add_argument(
        "--render-png",
        action="store_true",
        help="Render PNG snapshots during visible parent stage",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open generated board/snapshot files after the visible parent stage",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit final machine-readable summary as JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    project_dir = Path(args.project).resolve()
    schematic = Path(args.schematic).resolve()
    pcb = Path(args.pcb).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not project_dir.exists():
        print(f"error: project directory not found: {project_dir}", file=sys.stderr)
        return 2
    if not schematic.exists():
        print(f"error: schematic not found: {schematic}", file=sys.stderr)
        return 2
    if not pcb.exists():
        print(f"error: pcb not found: {pcb}", file=sys.stderr)
        return 2

    final_summary: dict[str, Any] = {
        "project_dir": str(project_dir),
        "schematic": str(schematic),
        "pcb": str(pcb),
        "output_dir": str(output_dir),
        "leaf_solve": {},
        "leaf_artifacts": {},
        "composition": {},
        "parent_pipeline": {},
        "status": "started",
    }

    solve_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "solve_subcircuits.py"),
        str(schematic),
        "--pcb",
        str(pcb),
        "--rounds",
        str(max(1, args.rounds)),
        "--seed",
        str(args.seed),
        "--route",
        "--json",
    ]
    if args.config:
        solve_cmd.extend(["--config", args.config])
    for selector in args.only:
        solve_cmd.extend(["--only", selector])

    rc, _, solve_payload = _run_command(
        solve_cmd,
        cwd=project_dir,
        label="Phase 1: Solve routed leaf subcircuits",
        capture_json=True,
    )
    final_summary["leaf_solve"] = {
        "exit_code": rc,
        "result": solve_payload,
    }
    if rc != 0:
        final_summary["status"] = "leaf_solve_failed"
        if args.json:
            print(json.dumps(final_summary, indent=2, sort_keys=True))
        return rc

    leaf_summary = _summarize_leaf_results(project_dir)
    final_summary["leaf_artifacts"] = leaf_summary
    _print_leaf_summary(leaf_summary)

    if leaf_summary["accepted_artifact_count"] == 0:
        print(
            "error: no accepted routed leaf artifacts were produced",
            file=sys.stderr,
        )
        final_summary["status"] = "no_accepted_leaf_artifacts"
        if args.json:
            print(json.dumps(final_summary, indent=2, sort_keys=True))
        return 1

    composition_json = output_dir / "parent_composition.json"
    compose_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "compose_subcircuits.py"),
        "--project",
        str(project_dir),
        "--parent",
        args.parent,
        "--mode",
        "packed",
        "--spacing-mm",
        "6",
        "--output",
        str(composition_json),
        "--json",
    ]
    for selector in args.only:
        compose_cmd.extend(["--only", selector])

    rc, _, compose_payload = _run_command(
        compose_cmd,
        cwd=project_dir,
        label="Phase 2: Compose parent from accepted routed leaves",
        capture_json=True,
    )
    final_summary["composition"] = {
        "exit_code": rc,
        "result": compose_payload,
        "output_json": str(composition_json),
    }
    if rc != 0:
        final_summary["status"] = "composition_failed"
        if args.json:
            print(json.dumps(final_summary, indent=2, sort_keys=True))
        return rc

    if args.compose_only:
        final_summary["status"] = "compose_only_complete"
        if args.json:
            print(json.dumps(final_summary, indent=2, sort_keys=True))
        else:
            print()
            print("Compose-only run complete.")
            print(f"composition_json : {composition_json}")
        return 0

    if args.skip_parent_routing:
        final_summary["status"] = "leaf_and_composition_complete"
        if args.json:
            print(json.dumps(final_summary, indent=2, sort_keys=True))
        else:
            print()
            print("Leaf solve + parent composition complete.")
            print(f"composition_json : {composition_json}")
        return 0

    parent_output_json = output_dir / "parent_pipeline.json"
    parent_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "compose_subcircuits.py"),
        "--project",
        str(project_dir),
        "--parent",
        args.parent,
        "--mode",
        "packed",
        "--spacing-mm",
        "6",
        "--pcb",
        str(pcb),
        "--route",
        "--output",
        str(parent_output_json),
    ]
    if args.jar:
        parent_cmd.extend(["--jar", args.jar])
    if args.config:
        parent_cmd.extend(["--config", args.config])
    for selector in args.only:
        parent_cmd.extend(["--only", selector])

    rc, parent_stdout, _ = _run_command(
        parent_cmd,
        cwd=project_dir,
        label="Phase 3: Parent assembly and routing",
        capture_json=False,
    )
    final_summary["parent_pipeline"] = {
        "exit_code": rc,
        "output_json": str(parent_output_json),
        "stdout_captured": parent_stdout,
    }
    if rc != 0:
        final_summary["status"] = "parent_pipeline_failed"
        if args.json:
            print(json.dumps(final_summary, indent=2, sort_keys=True))
        return rc

    final_summary["status"] = "complete"

    if args.json:
        print(json.dumps(final_summary, indent=2, sort_keys=True))
        return 0

    print()
    print("=" * 78)
    print("Hierarchical pipeline complete")
    print("=" * 78)
    print(f"accepted_leaf_artifacts : {leaf_summary['accepted_artifact_count']}")
    print(f"composition_json        : {composition_json}")
    print(f"parent_pipeline_json    : {parent_output_json}")
    print("Goal direction          : leaf-first -> routed artifacts -> parent assembly")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
