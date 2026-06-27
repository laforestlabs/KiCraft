"""Measure courtyard-overlap *magnitude* on a routed board (pcbnew geometry).

The kicad-cli DRC only reports a COUNT of ``courtyards_overlap`` violations, not
how deeply two courtyards intersect. But a fraction-of-a-millimetre courtyard
clip on an otherwise electrically-perfect board (no shorts, no unconnected) is
not a fab blocker -- courtyards carry an assembly-clearance margin, so a shallow
clip usually means the actual part bodies still clear. This module computes the
real intersection (area + penetration depth) per overlapping same-side pair so
the verify gate can treat a *minor* overlap as a warning (board still exported +
3D-rendered) while a *gross* overlap (parts physically colliding) still fails.

The placement solver's final courtyard-separation pass (Step 16) is the primary
fix -- it stops gross overlaps being produced. This measurement is the severity
backstop for the residue the solver cannot move (e.g. two pinned parts).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class CourtyardOverlap:
    ref_a: str
    ref_b: str
    layer: str  # "F" or "B"
    area_mm2: float
    penetration_mm: float  # min side of the intersection bbox

    def is_minor(self, max_penetration_mm: float, max_area_mm2: float) -> bool:
        return (
            self.penetration_mm <= max_penetration_mm
            and self.area_mm2 <= max_area_mm2
        )

    def to_dict(self) -> dict:
        return {
            "ref_a": self.ref_a,
            "ref_b": self.ref_b,
            "layer": self.layer,
            "area_mm2": round(self.area_mm2, 4),
            "penetration_mm": round(self.penetration_mm, 4),
        }


def measure_courtyard_overlaps(pcb_path: str) -> list[CourtyardOverlap]:
    """Return every same-side footprint-courtyard intersection on the board.

    Best-effort: returns ``[]`` if pcbnew is unavailable or the board cannot be
    loaded -- callers treat an empty result from a board the DRC flagged as
    "could not measure" and keep the conservative (hard-fail) verdict.
    """
    try:
        import pcbnew
    except Exception:
        return []

    try:
        board = pcbnew.LoadBoard(pcb_path)
    except Exception:
        return []
    if board is None:
        return []

    try:
        return _measure(board)
    except Exception:
        # Any pcbnew/geometry hiccup -> "could not measure"; the caller keeps the
        # conservative hard-fail rather than mis-grading an overlap as minor.
        return []


def _measure(board) -> list["CourtyardOverlap"]:
    import pcbnew

    overlaps: list[CourtyardOverlap] = []
    for layer, tag in ((pcbnew.F_CrtYd, "F"), (pcbnew.B_CrtYd, "B")):
        ents: list[tuple[str, object, tuple[float, float, float, float]]] = []
        for fp in board.GetFootprints():
            try:
                poly = fp.GetCourtyard(layer)
            except Exception:
                poly = None
            if not poly or poly.OutlineCount() == 0:
                continue
            bb = poly.BBox()
            ents.append(
                (
                    fp.GetReference(),
                    poly,
                    (
                        bb.GetLeft() / 1e6,
                        bb.GetTop() / 1e6,
                        bb.GetRight() / 1e6,
                        bb.GetBottom() / 1e6,
                    ),
                )
            )

        for (ra, pa, ba), (rb, pb, bb_) in itertools.combinations(ents, 2):
            # Cheap axis-aligned reject before the polygon boolean.
            if ba[2] < bb_[0] or bb_[2] < ba[0] or ba[3] < bb_[1] or bb_[3] < ba[1]:
                continue
            inter = pcbnew.SHAPE_POLY_SET(pa)
            try:
                inter.BooleanIntersection(pb)
            except Exception:
                continue
            if inter.OutlineCount() == 0:
                continue
            area = inter.Area() / 1e12  # nm^2 -> mm^2
            if area <= 1e-6:
                continue
            ibb = inter.BBox()
            dx = (ibb.GetRight() - ibb.GetLeft()) / 1e6
            dy = (ibb.GetBottom() - ibb.GetTop()) / 1e6
            overlaps.append(
                CourtyardOverlap(
                    ref_a=ra,
                    ref_b=rb,
                    layer=tag,
                    area_mm2=area,
                    penetration_mm=min(dx, dy),
                )
            )
    return overlaps


def classify_courtyard_overlaps(
    overlaps: list[CourtyardOverlap],
    *,
    max_penetration_mm: float,
    max_area_mm2: float,
) -> tuple[list[CourtyardOverlap], list[CourtyardOverlap]]:
    """Split measured overlaps into (minor, gross) by the magnitude thresholds."""
    minor = [
        o for o in overlaps if o.is_minor(max_penetration_mm, max_area_mm2)
    ]
    gross = [
        o for o in overlaps if not o.is_minor(max_penetration_mm, max_area_mm2)
    ]
    return minor, gross
