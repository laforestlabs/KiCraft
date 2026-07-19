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

# Ref-designator classes that get zoned to an edge for USER ACCESS, not for an
# off-board mating mouth. A coin-cell holder (BT*) wants to sit NEAR its edge
# for cell swaps, but nothing plugs into it from off the board: neither the
# flush (stranded) nor the mouth-facing gate describes a real fab defect for
# it, and its cell-insertion opening legitimately points anywhere (self-eval
# 2026-07-19 run_31: a fab-clean badge was rejected on BT1 "misoriented").
# The edge zone itself stays -- placement still biases the part edgeward.
_ACCESS_ONLY_REF_PREFIXES = frozenset({"BT"})


def _access_only_ref(ref: str) -> bool:
    """True for refs whose edge zone is an accessibility hint, not a mating
    contract (see ``_ACCESS_ONLY_REF_PREFIXES``)."""
    prefix = ref.rstrip("0123456789")
    return prefix.upper() in _ACCESS_ONLY_REF_PREFIXES


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
        if _access_only_ref(ref):
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


@dataclass(frozen=True)
class FacingVerdict:
    ref: str
    edge: str
    status: str  # "ok" | "misoriented" | "unknown_mouth"
    opening_board_deg: float | None  # board-space mouth angle; None = undetectable
    outward_deg: float  # board-space outward angle of the assigned edge


def connector_facings(
    board_path: str,
    component_zones: dict[str, Any] | None,
    *,
    tol_deg: float = 5.0,
    min_directional_depth_mm: float = 4.0,
) -> list[FacingVerdict]:
    """Does each edge-zoned connector's wire-entry mouth face OFF-board?

    The positional gate above is rotation-invariant: a 90-degree screw terminal
    can sit bbox-flush against its zoned edge with the wire mouth pointing
    parallel to (or into) the board -- electrically clean, physically unusable
    (KC-YJ7Q69). This companion metric compares the mouth's board-space angle
    (``detect_opening_direction`` expressed in board coords) against the zoned
    edge's outward normal:

        ok            -> mouth points off-board (within ``tol_deg``)
        misoriented   -> mouth detectable and NOT pointing off-board
        unknown_mouth -> TH connector with a deep (directional) body but no
                         detectable opening: nothing to verify against. Fix the
                         footprint (add a "PCB Edge" Dwgs.User marker) -- these
                         parts are exactly one silent inversion away from
                         shipping misoriented.

    Shallow-bodied undetectable parts (bare pin-header strips, vertical
    receptacles) are omitted: they have no meaningful mouth to verify. The
    depth cut runs on ``_mouth_bbox`` (courtyard + pad copper), which measures
    a bare 2.54mm strip at 3.63mm and a 2P screw terminal at 7.89mm -- the
    default sits between them (a body-calibrated 3.0 read every vertical
    strip as directional, spamming 17 unverifiable warnings per servo board).
    """
    import pcbnew

    from kicraft.autoplacer.hardware.adapter import detect_opening_direction

    from .types import Layer, angles_close, edge_outward_angle, opening_board_angle

    board = pcbnew.LoadBoard(str(board_path))
    fps = {fp.GetReference(): fp for fp in board.GetFootprints()}
    out: list[FacingVerdict] = []
    for ref, zone in (component_zones or {}).items():
        edge = (zone or {}).get("edge")
        if edge not in _SIDES:
            continue
        if _access_only_ref(ref):
            continue
        fp = fps.get(ref)
        if fp is None:
            continue
        layer = Layer.BACK if fp.GetLayer() == pcbnew.B_Cu else Layer.FRONT
        outward = edge_outward_angle(layer, edge)
        opening_local = detect_opening_direction(fp)
        if opening_local is None:
            has_hole = any(p.HasHole() for p in fp.Pads())
            bb = _mouth_bbox(fp, pcbnew)
            depth = min(
                pcbnew.ToMM(bb.GetWidth()), pcbnew.ToMM(bb.GetHeight())
            )
            if has_hole and depth > min_directional_depth_mm:
                out.append(FacingVerdict(ref, edge, "unknown_mouth", None, outward))
            continue
        opening_board = opening_board_angle(
            opening_local, fp.GetOrientationDegrees()
        )
        status = "ok" if angles_close(opening_board, outward, tol_deg) else "misoriented"
        out.append(FacingVerdict(ref, edge, status, opening_board, outward))
    return out
