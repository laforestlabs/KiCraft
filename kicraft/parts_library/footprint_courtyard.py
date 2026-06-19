"""Footprint courtyard hygiene: keep the courtyard clear of pad copper.

A footprint's courtyard is the keep-out rectangle that downstream placement and
board-outline code treats as the part's physical extent. When a part's solder
pads sit AT or only marginally inside the courtyard boundary (some edge-mount
connectors/switches put their pads right at the body front), the composed board
edge -- which is placed relative to that courtyard -- ends up cutting through
the pad's copper-to-edge clearance band, tripping ``copper_edge_clearance`` DRC.

The fix is at the footprint, not the gate: ensure the courtyard encloses every
pad by at least ``min_clearance_mm``. Applied when vendoring a new part
(``add-part``) so every library footprint is well-formed at the source, and
available to the composer for parts that were embedded before the check existed.

This grows ONLY when a pad is closer than the margin (never shrinks a courtyard,
never touches a footprint whose pads already clear). A grown courtyard becomes a
rectangle that is the union of the old courtyard and every pad expanded by the
margin -- a slightly larger rectangular keep-out, which is always fab-safe.
"""
from __future__ import annotations

from typing import Any

# Pads must clear the courtyard boundary by at least this much. 0.2 mm matches
# the JLCPCB routed board-edge copper clearance the board outline is sized to.
DEFAULT_COURTYARD_PAD_CLEARANCE_MM = 0.2

# Tolerance so a pad sitting at exactly the clearance doesn't trigger a no-op
# "grow" from floating-point noise (and so an int-nm round-trip stays clear).
_TOL_MM = 1e-4


def _courtyard_layers(pcbnew: Any, footprint: Any) -> list[int]:
    present = []
    for layer in (pcbnew.F_CrtYd, pcbnew.B_CrtYd):
        try:
            poly = footprint.GetCourtyard(layer)
        except Exception:  # noqa: BLE001 -- API shape varies across KiCad
            poly = None
        if poly is not None and poly.OutlineCount() > 0:
            present.append(layer)
    return present


def _pads_union_box(footprint: Any):
    """Union of all pad bounding boxes as ``(left, top, right, bottom)`` in
    integer nm, or ``None`` if the footprint has no pads. Plain ints (no
    ``BOX2I``) to avoid swig object churn across repeated calls."""
    l = t = r = b = None
    for pad in list(footprint.Pads()):
        pb = pad.GetBoundingBox()
        pl, pt, pr, pb_ = pb.GetLeft(), pb.GetTop(), pb.GetRight(), pb.GetBottom()
        l = pl if l is None else min(l, pl)
        t = pt if t is None else min(t, pt)
        r = pr if r is None else max(r, pr)
        b = pb_ if b is None else max(b, pb_)
    return None if l is None else (l, t, r, b)


def ensure_courtyard_clears_pads(
    footprint: Any,
    *,
    min_clearance_mm: float = DEFAULT_COURTYARD_PAD_CLEARANCE_MM,
) -> bool:
    """Grow ``footprint``'s courtyard so it clears every pad by ``min_clearance_mm``.

    Operates on each courtyard layer present (F.CrtYd / B.CrtYd). Returns True if
    any courtyard was grown. Requires ``pcbnew``; the footprint is mutated in
    place (caller saves). Safe no-op for a footprint with no pads or no courtyard.
    """
    import pcbnew

    pads = _pads_union_box(footprint)
    if pads is None:
        return False
    pl, pt, pr, pb = pads
    clr = pcbnew.FromMM(float(min_clearance_mm))
    tol = pcbnew.FromMM(_TOL_MM)
    req_l, req_t = pl - clr, pt - clr
    req_r, req_b = pr + clr, pb + clr

    grew = False
    for layer in _courtyard_layers(pcbnew, footprint):
        cb = footprint.GetCourtyard(layer).BBox()
        new_l, new_t = min(cb.GetLeft(), req_l), min(cb.GetTop(), req_t)
        new_r, new_b = max(cb.GetRight(), req_r), max(cb.GetBottom(), req_b)
        if (cb.GetLeft() - new_l <= tol and new_r - cb.GetRight() <= tol
                and cb.GetTop() - new_t <= tol and new_b - cb.GetBottom() <= tol):
            continue  # already clears every pad by the margin on this layer
        # Replace this layer's courtyard graphics with the grown rectangle.
        width = next(
            (it.GetWidth() for it in footprint.GraphicalItems()
             if it.GetLayer() == layer),
            pcbnew.FromMM(0.05),
        )
        for it in [g for g in footprint.GraphicalItems() if g.GetLayer() == layer]:
            footprint.Remove(it)
        corners = [(new_l, new_t), (new_r, new_t), (new_r, new_b), (new_l, new_b)]
        for i in range(4):
            seg = pcbnew.PCB_SHAPE(footprint)
            seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
            seg.SetLayer(layer)
            seg.SetWidth(width)
            seg.SetStart(pcbnew.VECTOR2I(int(corners[i][0]), int(corners[i][1])))
            seg.SetEnd(pcbnew.VECTOR2I(int(corners[(i + 1) % 4][0]),
                                       int(corners[(i + 1) % 4][1])))
            footprint.Add(seg)
        grew = True
    return grew


def courtyard_pad_clearance_mm(footprint: Any) -> float | None:
    """Smallest gap (mm) from any pad edge to the courtyard boundary, over all
    courtyard layers. Negative = a pad pokes outside the courtyard. None if the
    footprint has no courtyard or no pads. Diagnostic / test helper."""
    import pcbnew

    pads = _pads_union_box(footprint)
    layers = _courtyard_layers(pcbnew, footprint)
    if pads is None or not layers:
        return None
    pl, pt, pr, pb = pads
    worst = None
    for layer in layers:
        cb = footprint.GetCourtyard(layer).BBox()
        gaps = (
            pcbnew.ToMM(pl - cb.GetLeft()),
            pcbnew.ToMM(cb.GetRight() - pr),
            pcbnew.ToMM(pt - cb.GetTop()),
            pcbnew.ToMM(cb.GetBottom() - pb),
        )
        m = min(gaps)
        worst = m if worst is None else min(worst, m)
    return worst
