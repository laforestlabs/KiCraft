#!/usr/bin/env python3
"""Add an F.Silkscreen rectangle at each leaf board's Edge.Cuts boundary.

Run this once on existing leaves so they pick up the visible boundary
that the manual layout canvas (and parent renders) expects. Future
stamps add the silk automatically via _apply_board_outline; this CLI
exists for projects that have leaves already on disk from earlier
runs.

Usage:
    python -m kicraft.cli.add_leaf_silk_outline [path]

``path`` may be a single .kicad_pcb file or a directory; when omitted,
defaults to ``.experiments/subcircuits`` under the current working
directory and processes every ``leaf_routed.kicad_pcb`` it finds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


_RETROFIT_SCRIPT = r"""
import json, pcbnew

with open("__JSON_PATH__") as _f:
    _paths = json.load(_f)

_OUTLINE_WIDTH_MM = 0.15

for _entry in _paths:
    _src = _entry["src"]
    board = pcbnew.LoadBoard(_src)

    # Bbox of existing Edge.Cuts segments.
    _xs, _ys = [], []
    for _d in board.GetDrawings():
        if _d.GetLayer() != pcbnew.Edge_Cuts:
            continue
        if _d.GetShape() == pcbnew.SHAPE_T_SEGMENT:
            for _pt in (_d.GetStart(), _d.GetEnd()):
                _xs.append(pcbnew.ToMM(_pt.x))
                _ys.append(pcbnew.ToMM(_pt.y))
        elif _d.GetShape() == pcbnew.SHAPE_T_RECT:
            _xs += [pcbnew.ToMM(_d.GetStart().x), pcbnew.ToMM(_d.GetEnd().x)]
            _ys += [pcbnew.ToMM(_d.GetStart().y), pcbnew.ToMM(_d.GetEnd().y)]
    if not _xs or not _ys:
        print("[skip] " + _src + " (no Edge.Cuts found)")
        continue
    _l, _t = min(_xs), min(_ys)
    _r, _b = max(_xs), max(_ys)

    # Drop any prior leaf-outline silk at the 0.15 mm tag width so
    # re-running the retrofit is idempotent.
    _to_remove = [
        _d for _d in board.GetDrawings()
        if _d.GetLayer() == pcbnew.F_SilkS
        and _d.GetShape() == pcbnew.SHAPE_T_SEGMENT
        and abs(pcbnew.ToMM(_d.GetWidth()) - _OUTLINE_WIDTH_MM) < 1e-3
    ]
    for _d in _to_remove:
        board.Remove(_d)

    _corners = [
        (pcbnew.FromMM(_l), pcbnew.FromMM(_t)),
        (pcbnew.FromMM(_r), pcbnew.FromMM(_t)),
        (pcbnew.FromMM(_r), pcbnew.FromMM(_b)),
        (pcbnew.FromMM(_l), pcbnew.FromMM(_b)),
    ]
    for _i in range(4):
        _x1, _y1 = _corners[_i]
        _x2, _y2 = _corners[(_i + 1) % 4]
        _silk = pcbnew.PCB_SHAPE(board)
        _silk.SetShape(pcbnew.SHAPE_T_SEGMENT)
        _silk.SetLayer(pcbnew.F_SilkS)
        _silk.SetWidth(pcbnew.FromMM(_OUTLINE_WIDTH_MM))
        _silk.SetStart(pcbnew.VECTOR2I(_x1, _y1))
        _silk.SetEnd(pcbnew.VECTOR2I(_x2, _y2))
        board.Add(_silk)

    pcbnew.SaveBoard(_src, board)
    print("[ok]   " + _src + (
        " bbox=(%.2f,%.2f)-(%.2f,%.2f) mm" % (_l, _t, _r, _b)
    ))
"""


def _discover_pcbs(target: Path) -> list[Path]:
    if target.is_file() and target.suffix == ".kicad_pcb":
        return [target]
    if target.is_dir():
        # Prefer canonical leaf_routed.kicad_pcb; fall back to any other
        # leaf board in the dir tree.
        canonical = sorted(target.rglob("leaf_routed.kicad_pcb"))
        if canonical:
            return canonical
        return sorted(target.rglob("*.kicad_pcb"))
    return []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "path",
        nargs="?",
        default=".experiments/subcircuits",
        help="Leaf PCB or directory to scan (default: .experiments/subcircuits)",
    )
    args = p.parse_args(argv)

    target = Path(args.path).resolve()
    pcbs = _discover_pcbs(target)
    if not pcbs:
        print(f"error: no .kicad_pcb files found under {target}", file=sys.stderr)
        return 2

    print(f"Adding F.Silkscreen leaf outline to {len(pcbs)} board(s)…")

    payload = [{"src": str(pcb)} for pcb in pcbs]
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(payload, f)
        json_path = f.name

    script = _RETROFIT_SCRIPT.replace("__JSON_PATH__", json_path)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    finally:
        try:
            Path(json_path).unlink()
        except OSError:
            pass

    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
