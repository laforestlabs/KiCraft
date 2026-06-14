#!/usr/bin/env python3
"""Replay a fixed corpus of synthesized workspaces and diff placement vs golden.

The deterministic-placement harness that the placement/compose simplification
(Part 2 of docs/plans/place-route-replay-and-codebase-simplification.md) is
validated against: before a refactor, snapshot the corpus's placement geometry
(``--update``); after, re-run and assert NOTHING moved. A "no-op refactor" that
changes any board is then a visible, located diff -- not a silent regression.

What it does, per workspace under the corpus root:
  1. copy it to a scratch dir (the workspace is never mutated in place),
  2. ``kicraft replay --project ... --quality fast --no-route --no-fab --seed 0``
     -- placement only, the part `replay` guarantees reproducible (the composed
     parent is NOT compared: it consumes routed leaves and so inherits
     FreeRouting's best-effort nondeterminism),
  3. read each leaf ``leaf_pre_freerouting.kicad_pcb`` into a geometry map
     (ref -> x_mm, y_mm, rotation_deg),
  4. compare to ``<root>/<name>.golden.json`` (or write it with ``--update``).

A workspace is any subdirectory holding exactly one ``<stem>.kicad_pcb`` with a
sibling ``<stem>.kicad_sch`` + ``<stem>.kicad_pro`` (the same shape `replay
--project` discovers). Drop more workspaces in to grow the corpus -- a
stranded-connector case, a flat board, etc. (the plan's intended ~6).

Usage:
    python scripts/replay_corpus.py                 # check vs goldens
    python scripts/replay_corpus.py --update        # (re)write goldens
    python scripts/replay_corpus.py --root DIR      # alternate corpus root

Exit code: 0 = all match (or updated), 1 = drift detected, 2 = infra error.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "tests" / "fixtures" / "replay_workspace"
ROUND = 4  # mm precision for position; degrees rounded to 3


def _discover_workspaces(root: Path) -> list[Path]:
    """Subdirs shaped like a synthesized workspace (one identifiable stem)."""
    out: list[Path] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        stems = [
            pro.stem
            for pro in d.glob("*.kicad_pro")
            if (d / f"{pro.stem}.kicad_pcb").exists()
            and (d / f"{pro.stem}.kicad_sch").exists()
        ]
        if len(stems) == 1:
            out.append(d)
    return out


def _leaf_geometry(project_dir: Path) -> dict[str, dict[str, list]]:
    """Per-leaf footprint geometry from each leaf_pre_freerouting.kicad_pcb."""
    import pcbnew  # imported lazily so --help works without KiCad

    geo: dict[str, dict[str, list]] = {}
    for p in sorted(
        glob.glob(
            str(project_dir / ".experiments" / "subcircuits"
                / "*" / "leaf_pre_freerouting.kicad_pcb")
        )
    ):
        board = pcbnew.LoadBoard(p)
        geo[Path(p).parent.name] = {
            fp.GetReference(): [
                round(pcbnew.ToMM(fp.GetPosition().x), ROUND),
                round(pcbnew.ToMM(fp.GetPosition().y), ROUND),
                round(fp.GetOrientationDegrees(), 3),
            ]
            for fp in board.GetFootprints()
        }
    return geo


def _replay(workspace: Path, scratch: Path) -> Path:
    """Copy the workspace into scratch, replay placement, return the copy dir."""
    dest = scratch / workspace.name
    shutil.copytree(workspace, dest)
    # Start from a clean slate so placement regenerates deterministically.
    shutil.rmtree(dest / ".experiments", ignore_errors=True)
    rc = subprocess.run(
        [
            sys.executable, "-m", "kicraft.design.cli_app", "replay",
            "--project", str(dest), "--quality", "fast",
            "--no-route", "--no-fab", "--seed", "0",
        ],
        cwd=str(REPO_ROOT),
    ).returncode
    if rc != 0:
        raise RuntimeError(f"replay exited {rc} for {workspace.name}")
    return dest


def _diff(golden: dict, actual: dict) -> list[str]:
    """Human-readable lines describing every divergence (empty == match)."""
    lines: list[str] = []
    for leaf in sorted(set(golden) | set(actual)):
        if leaf not in golden:
            lines.append(f"  + new leaf {leaf}")
            continue
        if leaf not in actual:
            lines.append(f"  - missing leaf {leaf}")
            continue
        g, a = golden[leaf], actual[leaf]
        for ref in sorted(set(g) | set(a)):
            if g.get(ref) != a.get(ref):
                lines.append(f"  ~ {leaf}/{ref}: golden={g.get(ref)} now={a.get(ref)}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help=f"corpus root (default {DEFAULT_ROOT})")
    ap.add_argument("--update", action="store_true",
                    help="write/refresh the golden snapshots instead of checking")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    workspaces = _discover_workspaces(root)
    if not workspaces:
        print(f"error: no synthesized workspaces under {root}", file=sys.stderr)
        return 2

    print(f"replay corpus: {len(workspaces)} workspace(s) under {root}")
    drift = False
    with tempfile.TemporaryDirectory(prefix="replay_corpus_") as tmp:
        scratch = Path(tmp)
        for ws in workspaces:
            print(f"\n=== {ws.name} ===")
            try:
                placed = _replay(ws, scratch)
                actual = _leaf_geometry(placed)
            except Exception as exc:  # noqa: BLE001 -- report + continue the corpus
                print(f"  ERROR: {exc}", file=sys.stderr)
                drift = True
                continue
            golden_path = root / f"{ws.name}.golden.json"
            if args.update:
                golden_path.write_text(json.dumps(actual, indent=2, sort_keys=True),
                                       encoding="utf-8")
                n = sum(len(v) for v in actual.values())
                print(f"  wrote golden ({len(actual)} leaves, {n} footprints)")
                continue
            if not golden_path.exists():
                print(f"  NO GOLDEN ({golden_path.name}); run with --update first",
                      file=sys.stderr)
                drift = True
                continue
            golden = json.loads(golden_path.read_text(encoding="utf-8"))
            lines = _diff(golden, actual)
            if lines:
                drift = True
                print(f"  DRIFT ({len(lines)} change(s)):")
                print("\n".join(lines))
            else:
                n = sum(len(v) for v in actual.values())
                print(f"  OK -- {len(actual)} leaves, {n} footprints match golden")

    if args.update:
        print("\ngoldens updated.")
        return 0
    if drift:
        print("\nFAIL: placement drifted from golden (or a workspace errored).")
        return 1
    print("\nPASS: all workspaces match their golden placement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
