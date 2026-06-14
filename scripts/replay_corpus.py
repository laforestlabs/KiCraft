#!/usr/bin/env python3
"""Replay a fixed corpus of synthesized workspaces and diff placement vs golden.

The deterministic-placement harness the placement/compose simplification
(Part 2 of docs/plans/place-route-replay-and-codebase-simplification.md) is
validated against: before a refactor, snapshot the corpus geometry
(``--update``); after, re-run and assert NOTHING moved. A "no-op refactor" that
changes any board is then a visible, located diff -- not a silent regression.

Two modes (run both by default):

* ``leaf``   -- ``replay --no-route`` on each workspace; diff the per-leaf
  placement (``leaf_pre_freerouting.kicad_pcb``), which `replay` guarantees
  reproducible (pinned seed + hash seed). Snapshot: ``<name>.leaf.golden.json``.

* ``parent`` -- compose-only on the workspace's COMMITTED, FROZEN leaf artifacts
  (``.experiments/subcircuits/<leaf>/`` with paths tokenized as
  ``__KICRAFT_PROJECT_DIR__``), run with hash+thread pinning. This is the
  validation gate for parent-frame refactors (Levers 2.1/2.3, the convention
  bug). Parent placement is deterministic GIVEN frozen leaf inputs + pinned
  threads (empirically verified); a full replay's parent is NOT reproducible
  because leaf stamping (vias/pour -> size_reduction -> block size) is
  nondeterministic, so the parent gate freezes the leaves instead of
  regenerating them. Snapshot: ``<name>.parent.golden.json``.

Usage:
    python scripts/replay_corpus.py                  # check both modes
    python scripts/replay_corpus.py --mode parent    # one mode
    python scripts/replay_corpus.py --update         # (re)write goldens

Exit: 0 = all match (or updated), 1 = drift, 2 = infra error.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "tests" / "fixtures" / "replay_workspace"
PATH_TOKEN = "__KICRAFT_PROJECT_DIR__"
ROUND = 4  # mm precision for position; degrees rounded to 3

# Env that makes placement reproducible: hash seed pins set/dict + force-state
# dedup order; single-thread numpy removes FP-reduction jitter that otherwise
# flips discrete solver branches in the parent placement.
PINNED_ENV = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


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


def _footprints(pcb_path: str) -> dict[str, list]:
    import pcbnew  # lazy so --help works without KiCad

    board = pcbnew.LoadBoard(pcb_path)
    return {
        fp.GetReference(): [
            round(pcbnew.ToMM(fp.GetPosition().x), ROUND),
            round(pcbnew.ToMM(fp.GetPosition().y), ROUND),
            round(fp.GetOrientationDegrees(), 3),
        ]
        for fp in board.GetFootprints()
    }


# ---- leaf mode ---------------------------------------------------------------


def _leaf_geometry(project_dir: Path) -> dict[str, dict[str, list]]:
    geo: dict[str, dict[str, list]] = {}
    for p in sorted(
        glob.glob(
            str(project_dir / ".experiments" / "subcircuits"
                / "*" / "leaf_pre_freerouting.kicad_pcb")
        )
    ):
        geo[Path(p).parent.name] = _footprints(p)
    return geo


def _run_leaf(workspace: Path, scratch: Path) -> dict:
    dest = scratch / workspace.name
    shutil.copytree(workspace, dest)
    shutil.rmtree(dest / ".experiments", ignore_errors=True)  # regenerate leaves
    rc = subprocess.run(
        [sys.executable, "-m", "kicraft.design.cli_app", "replay",
         "--project", str(dest), "--quality", "fast", "--no-route",
         "--no-fab", "--seed", "0"],
        cwd=str(REPO_ROOT),
    ).returncode
    if rc != 0:
        raise RuntimeError(f"replay exited {rc}")
    return _leaf_geometry(dest)


# ---- parent mode -------------------------------------------------------------


def _detokenize(project_dir: Path) -> None:
    """Point the committed frozen-leaf artifacts at this project copy."""
    real = str(project_dir)
    for jf in (project_dir / ".experiments").rglob("*.json"):
        text = jf.read_text(encoding="utf-8")
        if PATH_TOKEN in text:
            jf.write_text(text.replace(PATH_TOKEN, real), encoding="utf-8")


def _parent_spacing(workspace: Path, stem: str) -> str:
    """Per-fixture parent-compose clearance (mm). Most boards compose at the
    2.0 default; a fixture that packs tighter (e.g. an extra edge connector)
    records its own ``parent_compose_spacing_mm`` in its autoplacer config so
    the gate composes it the same way it was frozen."""
    cfg = workspace / f"{stem}_autoplacer.json"
    if cfg.is_file():
        try:
            return str(json.loads(cfg.read_text(encoding="utf-8"))
                       .get("parent_compose_spacing_mm", 2.0))
        except Exception:  # noqa: BLE001 -- malformed config -> default
            pass
    return "2.0"


def _run_parent(workspace: Path, stem: str, scratch: Path) -> dict:
    dest = scratch / workspace.name
    shutil.copytree(workspace, dest)
    _detokenize(dest)
    # Drop any stale parent artifact so compose regenerates it from the frozen
    # leaves; keep the leaf artifacts (the whole point of the parent gate).
    for p in glob.glob(str(dest / ".experiments" / "subcircuits"
                            / "subcircuit__*" / "parent_pre_freerouting.kicad_pcb")):
        os.remove(p)
    env = {**os.environ, **PINNED_ENV}
    rc = subprocess.run(
        [sys.executable, "-m", "kicraft.cli.compose_subcircuits",
         "--project", str(dest), "--parent", stem,
         "--pcb", str(dest / f"{stem}.kicad_pcb"),
         "--spacing-mm", _parent_spacing(workspace, stem), "--stamp", "--seed", "0"],
        cwd=str(REPO_ROOT), env=env,
    ).returncode
    if rc != 0:
        raise RuntimeError(f"compose exited {rc}")
    hits = sorted(glob.glob(str(dest / ".experiments" / "subcircuits"
                                / "subcircuit__*" / "parent_pre_freerouting.kicad_pcb")))
    if not hits:
        raise RuntimeError("compose produced no parent_pre_freerouting board")
    return {"__parent__": _footprints(hits[-1])}


# ---- shared driver -----------------------------------------------------------


def _diff(golden: dict, actual: dict) -> list[str]:
    lines: list[str] = []
    for grp in sorted(set(golden) | set(actual)):
        if grp not in golden:
            lines.append(f"  + new group {grp}")
            continue
        if grp not in actual:
            lines.append(f"  - missing group {grp}")
            continue
        g, a = golden[grp], actual[grp]
        for ref in sorted(set(g) | set(a)):
            if g.get(ref) != a.get(ref):
                lines.append(f"  ~ {grp}/{ref}: golden={g.get(ref)} now={a.get(ref)}")
    return lines


def _check_mode(mode: str, ws: Path, stem: str, scratch: Path,
                root: Path, update: bool) -> bool:
    """Run one mode for one workspace. Returns True on drift/error.

    A fixture opts into a mode by the presence of its golden: in check mode a
    missing golden is a SKIP (the fixture isn't registered for this mode), not
    a failure -- so a parent-only fixture (no leaf golden) doesn't break a
    ``--mode both`` run. Use ``--mode <m> --update`` to mint a new golden."""
    golden_path = root / f"{ws.name}.{mode}.golden.json"
    if not update and not golden_path.exists():
        print(f"  [{mode}] SKIP (no {golden_path.name})")
        return False
    try:
        actual = _run_leaf(ws, scratch) if mode == "leaf" else _run_parent(ws, stem, scratch)
    except Exception as exc:  # noqa: BLE001 -- report + keep going
        print(f"  [{mode}] ERROR: {exc}", file=sys.stderr)
        return True
    if update:
        golden_path.write_text(json.dumps(actual, indent=2, sort_keys=True),
                               encoding="utf-8")
        n = sum(len(v) for v in actual.values())
        print(f"  [{mode}] wrote golden ({len(actual)} group(s), {n} footprints)")
        return False
    if not golden_path.exists():
        print(f"  [{mode}] NO GOLDEN ({golden_path.name}); run --update first",
              file=sys.stderr)
        return True
    lines = _diff(json.loads(golden_path.read_text(encoding="utf-8")), actual)
    if lines:
        print(f"  [{mode}] DRIFT ({len(lines)} change(s)):")
        print("\n".join(lines))
        return True
    n = sum(len(v) for v in actual.values())
    print(f"  [{mode}] OK -- {n} footprints match golden")
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help=f"corpus root (default {DEFAULT_ROOT})")
    ap.add_argument("--mode", choices=["leaf", "parent", "both"], default="both",
                    help="which validation to run (default both)")
    ap.add_argument("--update", action="store_true",
                    help="write/refresh the golden snapshots instead of checking")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    workspaces = _discover_workspaces(root)
    if not workspaces:
        print(f"error: no synthesized workspaces under {root}", file=sys.stderr)
        return 2
    modes = ["leaf", "parent"] if args.mode == "both" else [args.mode]

    print(f"replay corpus: {len(workspaces)} workspace(s), modes={modes}")
    drift = False
    with tempfile.TemporaryDirectory(prefix="replay_corpus_") as tmp:
        for ws in workspaces:
            print(f"\n=== {ws.name} ===")
            stem = next(p.stem for p in ws.glob("*.kicad_pro"))
            for mode in modes:
                # Fresh scratch subdir per (workspace, mode) so copies don't clash.
                scratch = Path(tmp) / f"{ws.name}_{mode}"
                scratch.mkdir(parents=True, exist_ok=True)
                if _check_mode(mode, ws, stem, scratch, root, args.update):
                    drift = True

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
