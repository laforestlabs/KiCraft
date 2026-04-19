#!/usr/bin/env python3
"""Unified hierarchical subcircuit solver.

Orchestrates the full bottom-up subcircuit pipeline:
1. Parse the schematic hierarchy
2. Compute bottom-up levels (leaves -> parents -> grandparents -> root)
3. Solve all leaf subcircuits (placement + FreeRouting)
4. For each subsequent level, compose children into parent(s)
5. Stamp parent board with preserved child copper
6. Route parent interconnect nets via FreeRouting
7. Persist the final routed parent artifact

Supports arbitrary N-level hierarchies, not just 2-level (leaves -> root).

Usage:
    python3 solve_hierarchy.py LLUPS.kicad_sch \
        --pcb LLUPS.kicad_pcb \
        --rounds 1 \
        --route
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kicraft.autoplacer.brain.hierarchy_parser import (
        HierarchyGraph,
        HierarchyNode,
    )

SCRIPTS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(project_dir: Path) -> dict:
    """Load project configuration, merging defaults with project-specific."""
    from kicraft.autoplacer.config import (
        DEFAULT_CONFIG,
        discover_project_config,
        load_project_config,
    )

    cfg = dict(DEFAULT_CONFIG)
    proj_cfg_path = discover_project_config(str(project_dir))
    if proj_cfg_path:
        cfg.update(load_project_config(str(proj_cfg_path)))
    return cfg


def _compute_levels(hierarchy: HierarchyGraph) -> list[list[HierarchyNode]]:
    """Return bottom-up levels: index 0 = leaves, 1 = their parents, etc.

    Each node's level is defined as:
      * 0 for leaf nodes (no children)
      * max(child levels) + 1 for non-leaf nodes
    """
    level_cache: dict[int, int] = {}

    def _level(node: HierarchyNode) -> int:
        nid = id(node)
        if nid not in level_cache:
            if node.is_leaf:
                level_cache[nid] = 0
            else:
                level_cache[nid] = max(_level(c) for c in node.children) + 1
        return level_cache[nid]

    groups: dict[int, list[HierarchyNode]] = defaultdict(list)
    for node in hierarchy.iter_nodes():
        groups[_level(node)].append(node)

    if not groups:
        return []
    return [groups.get(i, []) for i in range(max(groups) + 1)]


# ---------------------------------------------------------------------------
# Phase runners (unchanged)
# ---------------------------------------------------------------------------


def _solve_leaves(
    schematic: str,
    pcb: str,
    rounds: int,
    route: bool,
    only: list[str] | None = None,
) -> str:
    """Solve all leaf subcircuits by delegating to solve_subcircuits.py."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "solve_subcircuits.py"),
        schematic,
        "--pcb",
        pcb,
        "--rounds",
        str(rounds),
    ]
    if route:
        cmd.append("--route")
    for pattern in only or []:
        cmd.extend(["--only", pattern])

    print(f"  command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=False, text=True, timeout=900
    )
    if result.returncode != 0:
        raise RuntimeError(f"Leaf solving failed with exit code {result.returncode}")
    return ""


def _compose_and_route_parent(
    project_dir: Path,
    pcb: str,
    parent: str,
    cfg: dict,
    spacing_mm: float,
    mode: str,
    route: bool,
) -> str:
    """Compose leaves into parent and optionally route via FreeRouting."""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "compose_subcircuits.py"),
        "--project",
        str(project_dir),
        "--parent",
        parent,
        "--pcb",
        pcb,
        "--mode",
        mode,
        "--spacing-mm",
        str(spacing_mm),
        "--stamp",
    ]
    if route:
        cmd.append("--route")
    jar = cfg.get("freerouting_jar")
    if jar:
        cmd.extend(["--jar", jar])

    print(f"  command: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=False, text=True, timeout=600
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Parent composition/routing failed with exit code {result.returncode}"
        )
    return ""


# ---------------------------------------------------------------------------
# Artifact finders
# ---------------------------------------------------------------------------


def _count_leaf_artifacts(project_dir: Path) -> int:
    """Count the number of solved leaf artifacts."""
    artifacts_dir = project_dir / ".experiments" / "subcircuits"
    if not artifacts_dir.exists():
        return 0
    count = 0
    for d in artifacts_dir.iterdir():
        if d.is_dir() and (d / "metadata.json").exists():
            count += 1
    return count


def _find_parent_artifact(project_dir: Path) -> Path | None:
    """Find the parent artifact directory (the one with parent_routed.kicad_pcb)."""
    artifacts_dir = project_dir / ".experiments" / "subcircuits"
    if not artifacts_dir.exists():
        return None
    for d in artifacts_dir.iterdir():
        if d.is_dir() and (d / "parent_routed.kicad_pcb").exists():
            return d
    # Fall back to parent_pre_freerouting
    for d in artifacts_dir.iterdir():
        if d.is_dir() and (d / "parent_pre_freerouting.kicad_pcb").exists():
            return d
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified hierarchical subcircuit solver"
    )
    parser.add_argument(
        "schematic",
        help="Path to root .kicad_sch file",
    )
    parser.add_argument(
        "--pcb",
        required=True,
        help="Path to .kicad_pcb board file",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Placement rounds per leaf (default: 1)",
    )
    parser.add_argument(
        "--route",
        action="store_true",
        help="Enable FreeRouting for leaves and parent",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Restrict to specific subcircuits (repeatable)",
    )
    parser.add_argument(
        "--leaf-only",
        action="store_true",
        help="Stop after leaf solving (skip parent composition)",
    )
    parser.add_argument(
        "--skip-leaves",
        action="store_true",
        help="Skip leaf solving, use existing artifacts",
    )
    parser.add_argument(
        "--spacing-mm",
        type=float,
        default=5.0,
        help="Parent composition spacing in mm (default: 5.0)",
    )
    parser.add_argument(
        "--mode",
        choices=("row", "column", "grid"),
        default="grid",
        help="Parent composition layout mode (default: grid)",
    )
    parser.add_argument(
        "--jar",
        help="FreeRouting JAR path override",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    project_dir = Path(args.schematic).resolve().parent
    cfg = _load_config(project_dir)
    if args.jar:
        cfg["freerouting_jar"] = args.jar

    # --- Parse hierarchy and compute levels ---
    from kicraft.autoplacer.brain.hierarchy_parser import parse_hierarchy

    hierarchy = parse_hierarchy(str(project_dir))
    levels = _compute_levels(hierarchy)

    print("=" * 60)
    print("HIERARCHICAL SUBCIRCUIT SOLVER")
    print("=" * 60)
    print(f"project     : {project_dir}")
    print(f"schematic   : {args.schematic}")
    print(f"pcb         : {args.pcb}")
    print(f"rounds      : {args.rounds}")
    print(f"route       : {args.route}")
    print(f"mode        : {args.mode}")
    print(f"spacing_mm  : {args.spacing_mm}")
    print(f"  hierarchy levels: {len(levels)}")
    for i, level_nodes in enumerate(levels):
        names = [n.definition.id.sheet_name for n in level_nodes]
        print(f"    level {i}: {names}")
    print()

    t0 = time.time()

    # --- Level-by-level bottom-up solve ---
    for level_idx, level_nodes in enumerate(levels):
        if level_idx == 0:
            # Level 0: solve leaf subcircuits
            if not args.skip_leaves:
                print(
                    f"--- Level {level_idx}: Solving "
                    f"{len(level_nodes)} leaf subcircuit(s) ---"
                )
                print()
                try:
                    _solve_leaves(
                        args.schematic,
                        args.pcb,
                        args.rounds,
                        args.route,
                        args.only,
                    )
                except Exception as exc:
                    print(
                        f"\nerror: leaf solving failed: {exc}",
                        file=sys.stderr,
                    )
                    return 1
                print()
            else:
                print(
                    f"--- Level {level_idx}: Skipping leaf solving "
                    f"(using existing artifacts) ---"
                )
                print()

            leaf_count = _count_leaf_artifacts(project_dir)
            print(f"  leaf artifacts found: {leaf_count}")
            print()

            if args.leaf_only:
                elapsed = time.time() - t0
                print(f"--- Done (--leaf-only) in {elapsed:.1f}s ---")
                return 0
        else:
            # Level 1+: compose each parent from its already-solved children
            for node in level_nodes:
                parent_name = node.definition.id.sheet_name
                child_names = [
                    c.definition.id.sheet_name for c in node.children
                ]
                print(
                    f"--- Level {level_idx}: Composing parent "
                    f"'{parent_name}' from {child_names} ---"
                )
                print()
                try:
                    _compose_and_route_parent(
                        project_dir,
                        args.pcb,
                        parent_name,
                        cfg,
                        args.spacing_mm,
                        args.mode,
                        args.route,
                    )
                except Exception as exc:
                    print(
                        f"\nerror: composition/routing of '{parent_name}' "
                        f"failed: {exc}",
                        file=sys.stderr,
                    )
                    return 1
                print()

    # --- Summary ---
    elapsed = time.time() - t0
    leaf_count = _count_leaf_artifacts(project_dir)
    parent_artifact = _find_parent_artifact(project_dir)

    print("=" * 60)
    print("HIERARCHICAL SOLVE COMPLETE")
    print("=" * 60)
    print(f"  elapsed       : {elapsed:.1f}s")
    print(f"  levels        : {len(levels)}")
    print(f"  leaf_artifacts: {leaf_count}")
    if parent_artifact:
        routed_pcb = parent_artifact / "parent_routed.kicad_pcb"
        pre_pcb = parent_artifact / "parent_pre_freerouting.kicad_pcb"
        solved = parent_artifact / "solved_layout.json"
        if routed_pcb.exists():
            print(f"  parent_routed : {routed_pcb}")
        elif pre_pcb.exists():
            print(f"  parent_stamped: {pre_pcb}")
        if solved.exists():
            print(f"  solved_layout : {solved}")
    else:
        print("  parent_artifact: (not found)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
