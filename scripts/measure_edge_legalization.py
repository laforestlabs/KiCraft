#!/usr/bin/env python3
"""Read-only $0 instrument for the copper-to-edge work.

For every edge-zoned connector in a self-eval corpus, report the gap from the
board edge to (a) the connector's OUTERMOST edge-facing feature ("mouth" =
courtyard ∪ pads) and (b) its edge-facing SOLDER PADS only. The difference
(pad_gap − mouth_gap) is how far the pads sit BEHIND the mouth: the headroom a
"snap the edge to the mouth" placement fix has before pads reach the edge.

    mouth_gap  >0 = mouth past edge (overhang/flush-proud), <0 = mouth inboard
    pad_gap    distance from board edge to nearest edge-facing pad copper
               (must be ≥ clearance for the copper_edge_clearance rule)

Run with the repo venv (pcbnew):
    .venv/bin/python scripts/measure_edge_legalization.py [CORPUS_DIR]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pcbnew

CORPUS = sys.argv[1] if len(sys.argv) > 1 else (
    "logs/self_eval/20260618T142304Z"
)
_SIDES = ("left", "right", "top", "bottom")


def _mm(v: float) -> float:
    return pcbnew.ToMM(v)


def _edge_centerline(board):
    e = board.GetBoardEdgesBoundingBox()
    widths = [
        _mm(d.GetWidth())
        for d in board.GetDrawings()
        if d.GetLayer() == pcbnew.Edge_Cuts
    ]
    h = (max(widths) if widths else 0.05) / 2.0
    return (_mm(e.GetLeft()) + h, _mm(e.GetTop()) + h,
            _mm(e.GetRight()) - h, _mm(e.GetBottom()) - h)


def _pad_bbox(fp):
    bb = None
    for p in fp.Pads():
        pb = p.GetBoundingBox()
        if bb is None:
            bb = pcbnew.BOX2I(pb.GetOrigin(), pb.GetSize())
        else:
            bb.Merge(pb)
    return bb


def _court_bbox(fp):
    for layer in (pcbnew.F_CrtYd, pcbnew.B_CrtYd):
        try:
            poly = fp.GetCourtyard(layer)
        except Exception:
            poly = None
        if poly is not None and poly.OutlineCount() > 0:
            return poly.BBox()
    return fp.GetBoundingBox(False, False)


def _gap(side, edge, bb):
    """Outward-positive gap from board edge `side` to bbox `bb` (mm)."""
    L, T, R, B = edge
    x0, y0, x1, y1 = _mm(bb.GetLeft()), _mm(bb.GetTop()), _mm(bb.GetRight()), _mm(bb.GetBottom())
    if side == "left":
        return L - x0
    if side == "right":
        return x1 - R
    if side == "top":
        return T - y0
    if side == "bottom":
        return y1 - B
    raise ValueError(side)


def _find_board_and_zones(stem_dir):
    """Return (board_path, component_zones) for a generated/<STEM> dir."""
    stem = os.path.basename(stem_dir.rstrip("/"))
    zones = {}
    zf = os.path.join(stem_dir, f"{stem}_autoplacer.json")
    if os.path.exists(zf):
        payload = json.load(open(zf))
        zones = payload.get("component_zones", payload.get("zones", {})) or {}
    # Prefer the promoted board (what the gate judged); fall back to best parent.
    cands = [
        os.path.join(stem_dir, f"{stem}.kicad_pcb"),
        *sorted(glob.glob(os.path.join(stem_dir, ".experiments", "best", "parent_routed.kicad_pcb"))),
        *sorted(glob.glob(os.path.join(stem_dir, ".experiments", "**", "parent_routed.kicad_pcb"), recursive=True)),
    ]
    board = next((c for c in cands if os.path.exists(c)), None)
    return board, zones


def main():
    rows = []
    for run in sorted(glob.glob(os.path.join(CORPUS, "run_*"))):
        for stem_dir in sorted(glob.glob(os.path.join(run, "generated", "*"))):
            if not os.path.isdir(stem_dir):
                continue
            board_path, zones = _find_board_and_zones(stem_dir)
            edge_conns = {r: z for r, z in zones.items()
                          if isinstance(z, dict) and z.get("edge") in _SIDES}
            if not board_path or not edge_conns:
                continue
            board = pcbnew.LoadBoard(board_path)
            edge = _edge_centerline(board)
            fps = {fp.GetReference(): fp for fp in board.GetFootprints()}
            for ref, z in sorted(edge_conns.items()):
                fp = fps.get(ref)
                if fp is None:
                    continue
                side = z["edge"]
                pads = _pad_bbox(fp)
                if pads is None:
                    continue
                court = _court_bbox(fp)
                mouth = pcbnew.BOX2I(court.GetOrigin(), court.GetSize())
                mouth.Merge(pads)
                pad_gap = _gap(side, edge, pads)
                mouth_gap = _gap(side, edge, mouth)
                rows.append((os.path.basename(run), ref, side,
                             round(mouth_gap, 4), round(pad_gap, 4),
                             round(pad_gap - mouth_gap, 4)))
    w = max((len(r[0]) for r in rows), default=8)
    print(f"{'run':<{w}}  {'ref':<5} {'side':<6} {'mouth_gap':>10} {'pad_gap':>9} {'pad_behind_mouth':>16}")
    for run, ref, side, mg, pg, behind in rows:
        flag = "  <-- pad < 0.2" if pg < 0.2 else ""
        print(f"{run:<{w}}  {ref:<5} {side:<6} {mg:>10.4f} {pg:>9.4f} {behind:>16.4f}{flag}")


if __name__ == "__main__":
    main()
