"""Connector edge-attachment acceptance metric (plan Part 3).

For each edge-zoned component, the signed OUTWARD gap between its "mouth" (the
OUTERMOST of its courtyard and pad copper on the assigned side -- see
``_mouth_bbox``) and the board edge it is assigned to:

    gap > 0  -> mouth is PAST the board edge (overhang; e.g. a USB-C body)
    gap ~ 0  -> flush
    gap < 0  -> mouth is INBOARD of the edge (the stranding bug -- a connector
                pulled several mm in from the edge it was zoned to)

Acceptance (the regression gate): ``gap >= -inboard_tol_mm`` (not buried) AND
``gap <= max_overhang_mm`` (not absurdly far out). The stranding bug this plan
hunts shows up as a large NEGATIVE gap.

Edge assignments come from the project's ``component_zones`` (the
``<stem>_autoplacer.json`` ``{ref: {"edge": "left|right|top|bottom"}}`` map).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_SIDES = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class EdgeGap:
    ref: str
    edge: str
    gap_mm: float  # outward-positive: + past board edge, - inboard (stranded)
    ok: bool


def edge_gap_mm(
    edge: str,
    board: tuple[float, float, float, float],
    court: tuple[float, float, float, float],
) -> float:
    """Signed OUTWARD gap (mm) for one component's courtyard ``court`` against
    the board ``board``, both ``(x0, y0, x1, y1)`` (left, top, right, bottom in
    KiCad Y-down). Positive = mouth past the board edge (overhang), 0 = flush,
    negative = inboard (stranded)."""
    bx0, by0, bx1, by1 = board
    cx0, cy0, cx1, cy1 = court
    if edge == "left":
        return bx0 - cx0
    if edge == "right":
        return cx1 - bx1
    if edge == "top":
        return by0 - cy0
    if edge == "bottom":
        return cy1 - by1
    raise ValueError(f"unsupported edge: {edge}")


def _mouth_bbox(fp, pcbnew):
    """The part's OUTERMOST physical extent toward a board edge, in board
    coords: the union of its courtyard and its pad copper.

    A part's "mouth" -- the feature that must reach the board edge -- is its
    most-outward feature on the zoned side. For a USB-C the courtyard (shell)
    overhangs the pads; for a switch/header the pads can sit PROUD of an inset
    courtyard. Using courtyard alone then falsely reads such a part as inboard
    even when its pads are flush at the edge. The union is the right "mouth":
    pads are copper that must be inside the board, courtyard is the keep-out,
    and the edge should meet whichever reaches furthest out."""
    bb = None
    for layer in (pcbnew.F_CrtYd, pcbnew.B_CrtYd):
        try:
            poly = fp.GetCourtyard(layer)
        except Exception:  # noqa: BLE001 -- API shape varies across KiCad
            poly = None
        if poly is not None and poly.OutlineCount() > 0:
            bb = poly.BBox()
            break
    if bb is None:
        bb = fp.GetBoundingBox(False, False)  # no courtyard drawn: no text/invisible
    for pad in fp.Pads():
        bb.Merge(pad.GetBoundingBox())
    return bb


def connector_edge_gaps(
    board_path: str,
    component_zones: dict[str, Any] | None,
    *,
    inboard_tol_mm: float = 0.5,
    max_overhang_mm: float = 6.0,
) -> list[EdgeGap]:
    """Compute the outward mouth-to-edge gap for every edge-zoned component on
    ``board_path``. Returns one :class:`EdgeGap` per resolvable zoned ref."""
    import pcbnew

    board = pcbnew.LoadBoard(str(board_path))
    edges = board.GetBoardEdgesBoundingBox()
    bx0 = pcbnew.ToMM(edges.GetLeft())
    by0 = pcbnew.ToMM(edges.GetTop())
    bx1 = pcbnew.ToMM(edges.GetRight())
    by1 = pcbnew.ToMM(edges.GetBottom())

    fps = {fp.GetReference(): fp for fp in board.GetFootprints()}
    out: list[EdgeGap] = []
    for ref, zone in (component_zones or {}).items():
        edge = (zone or {}).get("edge")
        if edge not in _SIDES:
            continue
        fp = fps.get(ref)
        if fp is None:
            continue
        bb = _mouth_bbox(fp, pcbnew)
        court = (
            pcbnew.ToMM(bb.GetLeft()), pcbnew.ToMM(bb.GetTop()),
            pcbnew.ToMM(bb.GetRight()), pcbnew.ToMM(bb.GetBottom()),
        )
        gap = edge_gap_mm(edge, (bx0, by0, bx1, by1), court)
        ok = (gap >= -inboard_tol_mm) and (gap <= max_overhang_mm)
        out.append(EdgeGap(ref, edge, round(gap, 4), ok))
    return out


def stranded(gaps: list[EdgeGap]) -> list[EdgeGap]:
    """The subset that fails the acceptance gate (stranded or absurd overhang)."""
    return [g for g in gaps if not g.ok]
