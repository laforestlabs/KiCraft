#!/usr/bin/env python3
"""Deterministic A/B: run PARENT compose on each self-eval workspace's FROZEN
routed leaves and report rc + shorts + board size. Isolates parent-compose
changes (RC1/RC2/RC3) from synthesis + routing noise. Usage:

    python scripts/ab_compose.py <self_eval_OUT_dir>

Run it once on each git arm (my branch, main) and diff the printed table."""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PINNED = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
          "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def _workspaces(out: Path) -> list[tuple[str, Path, str]]:
    found = []
    for run in sorted(out.glob("run_*")):
        gens = list((run / "generated").glob("*")) if (run / "generated").is_dir() else []
        for g in gens:
            pros = [p for p in g.glob("*.kicad_pro")
                    if (g / f"{p.stem}.kicad_pcb").exists()
                    and (g / f"{p.stem}.kicad_sch").exists()]
            if len(pros) == 1:
                found.append((run.name, g, pros[0].stem))
    return found


def _shorts(board: str) -> int:
    import pcbnew
    b = pcbnew.LoadBoard(board)
    tr = [t for t in b.GetTracks() if t.Type() == pcbnew.PCB_TRACE_T]
    n = 0
    for i in range(len(tr)):
        bi = tr[i].GetBoundingBox()
        for j in range(i + 1, len(tr)):
            if tr[i].GetLayer() != tr[j].GetLayer():
                continue
            if tr[i].GetNetCode() == tr[j].GetNetCode():
                continue
            if bi.Intersects(tr[j].GetBoundingBox()):
                n += 1
    return n


def _board_size(board: str):
    import pcbnew
    b = pcbnew.LoadBoard(board)
    e = b.GetBoardEdgesBoundingBox()
    return round(pcbnew.ToMM(e.GetWidth()), 1), round(pcbnew.ToMM(e.GetHeight()), 1)


def _run_one(src: Path, stem: str, scratch: Path) -> dict:
    dest = scratch / src.name
    shutil.copytree(src, dest)
    src_abs, dest_abs = str(src.resolve()), str(dest.resolve())
    for jf in (dest / ".experiments").rglob("*.json"):
        try:
            t = jf.read_text(encoding="utf-8")
        except Exception:
            continue
        if src_abs in t:
            jf.write_text(t.replace(src_abs, dest_abs), encoding="utf-8")
    for p in glob.glob(str(dest / ".experiments" / "subcircuits"
                           / "subcircuit__*" / "parent_placed.kicad_pcb")):
        os.remove(p)
    proc = subprocess.run(
        [sys.executable, "-m", "kicraft.cli.compose_subcircuits",
         "--project", str(dest), "--parent", stem,
         "--pcb", str(dest / f"{stem}.kicad_pcb"),
         "--spacing-mm", "2.0", "--stamp", "--seed", "0"],
        cwd=str(REPO), env={**os.environ, **PINNED},
        capture_output=True, text=True,
    )
    out = {"rc": proc.returncode}
    # Resolve the freshly-composed placed board via the central resolver, not a
    # sorted(glob)[-1] (alphabetical), so we never measure a stale board.
    from kicraft.cli.artifact_paths import resolve_parent_board
    board = resolve_parent_board(dest, kind="placed")
    if board is not None:
        out["shorts"] = _shorts(str(board))
        out["size"] = _board_size(str(board))
    else:
        m = re.search(r"per-candidate:[^\n]*", proc.stdout + proc.stderr)
        out["abort"] = m.group(0)[:120] if m else (proc.stderr.strip().splitlines() or [""])[-1][:120]
    return out


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    filters = sys.argv[2:]
    wss = _workspaces(out)
    if filters:
        wss = [w for w in wss if any(f in w[0] for f in filters)]
    print(f"# {len(wss)} workspace(s) from {out.name}")
    with tempfile.TemporaryDirectory(prefix="ab_compose_") as tmp:
        for run_name, src, stem in wss:
            scratch = Path(tmp) / run_name
            scratch.mkdir(parents=True, exist_ok=True)
            try:
                r = _run_one(src, stem, scratch)
            except Exception as exc:  # noqa: BLE001
                r = {"error": str(exc)[:120]}
            extra = (f"shorts={r.get('shorts')} size={r.get('size')}" if r.get("rc") == 0
                     else r.get("abort") or r.get("error") or "")
            print(f"{run_name:32s} rc={r.get('rc')} {extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
