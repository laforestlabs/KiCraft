"""Geometric silkscreen reference-designator legalization.

No stage manages per-footprint refdes, so on a dense array a default ~1 mm
"D147" label is bigger than its 1.5-2 mm part and overlaps neighbours. This
pass runs on a loaded pcbnew board (at the end of parent/leaf stamping) and,
per footprint reference:

  * shrink-to-fit its courtyard (down to a readable floor) and centre it; if it
    fits and clears neighbouring courtyards, keep it on silk;
  * otherwise move it to F.Fab/B.Fab -- the designator survives for assembly /
    pick-and-place but no longer clutters the silkscreen.

The resulting invariant is checkable: every *visible silk* refdes fits within
its own courtyard (minus margin) and overlaps no other footprint's courtyard.
Pure geometry, deterministic, no synthesis hints (so it captures dense array
members without plumbing an array_member flag).
"""
from __future__ import annotations

import pcbnew

# Discrete text heights (mm) tried largest-first when fitting a refdes to its
# courtyard. The smallest is the readability floor: a refdes that needs to go
# below it to fit is hidden to Fab instead of shrunk into illegibility.
_TEXT_HEIGHTS_MM = (1.0, 0.9, 0.8, 0.7)
_THICKNESS_RATIO = 0.15  # KiCad's nominal stroke:height ratio


def _courtyard_bbox(fp):
    layer = pcbnew.F_CrtYd if fp.GetLayer() == pcbnew.F_Cu else pcbnew.B_CrtYd
    try:
        bb = fp.GetCourtyard(layer).BBox()
    except Exception:
        return None
    if bb.GetWidth() <= 0 or bb.GetHeight() <= 0:
        return None
    return bb


def _fits(item_bbox, court, margin_iu) -> bool:
    return (
        item_bbox.GetWidth() <= court.GetWidth() - 2 * margin_iu
        and item_bbox.GetHeight() <= court.GetHeight() - 2 * margin_iu
    )


def _overlaps(a, b) -> bool:
    ox = min(a.GetRight(), b.GetRight()) - max(a.GetLeft(), b.GetLeft())
    oy = min(a.GetBottom(), b.GetBottom()) - max(a.GetTop(), b.GetTop())
    return ox > 0 and oy > 0


def _move_to_fab(fp, item) -> None:
    item.SetLayer(pcbnew.B_Fab if fp.GetLayer() == pcbnew.B_Cu else pcbnew.F_Fab)
    item.SetVisible(True)  # keep for assembly / pick-and-place


def legalize_refdes(board, cfg: dict | None = None) -> dict[str, list[str]]:
    """Legalize every footprint reference designator on ``board`` in place.

    Returns ``{"kept": [...], "moved_to_fab": [...]}`` (refs) for logging/tests.
    """
    cfg = cfg or {}
    margin = pcbnew.FromMM(float(cfg.get("refdes_courtyard_margin_mm", 0.1)))

    courtyards: dict[str, object] = {}
    for fp in board.Footprints():
        cb = _courtyard_bbox(fp)
        if cb is not None:
            courtyards[fp.GetReferenceAsString()] = cb

    kept: list[str] = []
    moved: list[str] = []
    for fp in board.Footprints():
        ref = fp.GetReferenceAsString()
        item = fp.Reference()
        if not item.IsVisible():
            continue
        court = courtyards.get(ref)
        if court is None:
            continue  # no courtyard to reason about -> leave as authored

        orig_size = item.GetTextSize()
        orig_thick = item.GetTextThickness()

        # 1. Largest readable size that fits the courtyard.
        fitted = False
        for h in _TEXT_HEIGHTS_MM:
            hi = pcbnew.FromMM(h)
            item.SetTextSize(pcbnew.VECTOR2I(hi, hi))
            item.SetTextThickness(pcbnew.FromMM(h * _THICKNESS_RATIO))
            if _fits(item.GetBoundingBox(), court, margin):
                fitted = True
                break
        if not fitted:
            item.SetTextSize(orig_size)
            item.SetTextThickness(orig_thick)
            _move_to_fab(fp, item)
            moved.append(ref)
            continue

        # 2. Centre it in the courtyard.
        item.SetPosition(court.GetCenter())

        # 3. If the centred label still crosses a neighbour courtyard, hide it.
        rb = item.GetBoundingBox()
        if any(
            other_ref != ref and _overlaps(rb, other_court)
            for other_ref, other_court in courtyards.items()
        ):
            item.SetTextSize(orig_size)
            item.SetTextThickness(orig_thick)
            _move_to_fab(fp, item)
            moved.append(ref)
            continue

        kept.append(ref)

    return {"kept": kept, "moved_to_fab": moved}
