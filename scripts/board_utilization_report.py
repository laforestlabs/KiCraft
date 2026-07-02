#!/usr/bin/env python3
"""Fleet PCB area-utilization report (PCB area-compaction plan, Phase 0).

Scans promoted project boards and reports, per board and as fleet medians:

    area_utilization   Σ footprint courtyard-bbox areas / board area
    aspect_ratio       max/min of the Edge.Cuts bbox
    bbox_utilization   Σ areas / bbox around all courtyards

This is the baseline instrument behind the 2026-07-02 investigation numbers
(median utilization 14.7%, 15/30 boards below 15%, six boards >= 197mm wide)
so post-change comparisons are one command.

Run with the repo venv (pcbnew required):
    .venv/bin/python scripts/board_utilization_report.py                # newest 30
    .venv/bin/python scripts/board_utilization_report.py --newest 0    # all
    .venv/bin/python scripts/board_utilization_report.py --json out.json
    .venv/bin/python scripts/board_utilization_report.py --pcb a.kicad_pcb b.kicad_pcb
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kicraft.cli.inspect_parent import board_utilization  # noqa: E402


def _discover_promoted_boards(projects_root: Path) -> list[Path]:
    """Promoted project boards: <root>/<uid>/<pid>/generated/<stem>/<stem>.kicad_pcb."""
    boards: list[Path] = []
    for candidate in sorted(projects_root.glob("*/*/generated/*/")):
        stem = candidate.name
        pcb = candidate / f"{stem}.kicad_pcb"
        if pcb.is_file():
            boards.append(pcb)
    return boards


def _brief_hint(pcb: Path) -> str:
    """First line of the project's brief.txt, if resolvable."""
    try:
        brief = pcb.parents[2] / "brief.txt"
        if brief.is_file():
            return brief.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0][:60]
    except (OSError, IndexError):
        pass
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--projects-root",
        default=str(Path.home() / ".kicraft" / "projects"),
        help="Projects root to scan (default: ~/.kicraft/projects)",
    )
    parser.add_argument(
        "--newest",
        type=int,
        default=30,
        help="Only the N newest boards by mtime (0 = all; default: 30)",
    )
    parser.add_argument(
        "--pcb",
        nargs="*",
        default=[],
        help="Explicit .kicad_pcb paths instead of scanning the projects root",
    )
    parser.add_argument("--json", help="Also write the full report as JSON to this path")
    args = parser.parse_args(argv)

    if args.pcb:
        boards = [Path(p) for p in args.pcb]
    else:
        projects_root = Path(args.projects_root).expanduser()
        if not projects_root.is_dir():
            print(f"error: projects root not found: {projects_root}", file=sys.stderr)
            return 2
        boards = _discover_promoted_boards(projects_root)
        boards.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if args.newest > 0:
            boards = boards[: args.newest]

    if not boards:
        print("error: no promoted boards found", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for pcb in boards:
        try:
            metrics = board_utilization(pcb)
        except Exception as exc:  # unparsable/corrupt board: report, don't die
            print(f"warning: skipped {pcb}: {exc}", file=sys.stderr)
            continue
        # <root>/<uid>/<pid>/generated/<stem>/<stem>.kicad_pcb -> "uid/pid"
        try:
            project = f"{pcb.parents[3].name}/{pcb.parents[2].name}"
        except IndexError:
            project = pcb.parent.name
        rows.append(
            {
                "project": project,
                "pcb": str(pcb),
                "brief": _brief_hint(pcb),
                **metrics,
            }
        )

    if not rows:
        print("error: no boards could be measured", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["area_utilization"])
    print(
        f"{'project':<10} {'WxH mm':>15} {'util%':>7} {'aspect':>7} "
        f"{'bbox_util%':>10} {'parts':>5}  brief"
    )
    for r in rows:
        print(
            f"{r['project']:<10} "
            f"{r['board_width_mm']:>7.1f}x{r['board_height_mm']:<7.1f} "
            f"{r['area_utilization'] * 100:>7.1f} "
            f"{r['aspect_ratio']:>7.2f} "
            f"{r['bbox_utilization'] * 100:>10.1f} "
            f"{int(r['footprint_count']):>5}  {r['brief']}"
        )

    utils = [r["area_utilization"] for r in rows]
    aspects = [r["aspect_ratio"] for r in rows]
    n = len(rows)
    summary = {
        "boards": n,
        "median_area_utilization": round(statistics.median(utils), 4),
        "median_aspect_ratio": round(statistics.median(aspects), 3),
        "median_bbox_utilization": round(
            statistics.median([r["bbox_utilization"] for r in rows]), 4
        ),
        "below_15pct_util": sum(1 for u in utils if u < 0.15),
        "aspect_over_2": sum(1 for a in aspects if a > 2.0),
        "width_over_197mm": sum(
            1
            for r in rows
            if max(r["board_width_mm"], r["board_height_mm"]) >= 197.0
        ),
    }
    print()
    print(
        f"fleet: n={n} "
        f"median_util={summary['median_area_utilization'] * 100:.1f}% "
        f"median_aspect={summary['median_aspect_ratio']:.2f} "
        f"median_bbox_util={summary['median_bbox_utilization'] * 100:.1f}% "
        f"| <15% util: {summary['below_15pct_util']}/{n} "
        f"| aspect>2: {summary['aspect_over_2']}/{n} "
        f"| >=197mm wide: {summary['width_over_197mm']}/{n}"
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": summary, "boards": rows}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"json report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
