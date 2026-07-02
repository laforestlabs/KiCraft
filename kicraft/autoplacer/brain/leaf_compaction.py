"""Deterministic post-SA placement compaction (PCB area-compaction plan, Phase 3).

Even on a right-sized canvas, force equilibrium + SA leave slack between
parts: attraction/repulsion balance well above the placement clearance, and
SA's small move radius cannot close multi-mm gaps (RC3 of the plan). This
pass closes that slack deterministically: a per-axis sweep that slides each
unlocked component toward the placed-bbox centroid as far as legality
allows -- courtyard clearance to every neighbour, antenna keep-outs,
mounting-hole keep-ins, pad-inside-board -- and nothing further.

It is deliberately NOT a force/SA budget increase (which would destabilize
the converged behavior the whole tuned config depends on); it is a bounded,
seedless geometry step in the same move-primitive family as the solver's
Step-16 courtyard separation and the composer's extremal slide.

Pure geometry, no pcbnew, no RNG. Mutates ``comps`` in place and returns a
summary dict for debug.json.
"""

from __future__ import annotations

from typing import Any

from .placement_utils import (
    _back_courtyard,
    _blocker_pair_compatible,
    _effective_bbox,
    _update_pad_positions,
)
from .types import Component, Point

# A slide smaller than this is noise -- skip it to keep the pass idempotent.
_MIN_SLIDE_MM = 0.05
# Extra breathing room kept from keep-out / keep-in rects (they are hard DRC
# rule areas, not courtyards, so the full placement clearance is not owed).
_RECT_MARGIN_MM = 0.1


def _pair_blocks(a: Component, b: Component) -> bool:
    """Whether *b* constrains *a*'s slide (same predicate as Step 16).

    Only a genuine opposite-side dual-layer stack is exempt; leaf-path
    components (no block metadata) always block each other.
    """
    if getattr(a, "array_member", False) and getattr(b, "array_member", False):
        return False  # grid is self-legal by construction
    if _blocker_pair_compatible(a, b) and _back_courtyard(a) != _back_courtyard(b):
        return False
    return True


def _resolved_keepout_rects(
    keepout_rects: list[Any],
    comps: dict[str, Component],
) -> list[tuple[Point, Point, str]]:
    """Owner-tracked keep-out rects at the owners' CURRENT positions.

    Mirrors PlacementSolver._keepout_rect_now: the rect was sampled at
    extraction time and rides rigidly on its owner footprint.
    """
    out: list[tuple[Point, Point, str]] = []
    for kr in keepout_rects or []:
        tl, br = kr.tl, kr.br
        origin = getattr(kr, "owner_origin", None)
        owner_ref = str(getattr(kr, "owner_ref", "") or "")
        if origin is not None:
            owner = comps.get(owner_ref)
            if owner is not None:
                dx = owner.pos.x - origin.x
                dy = owner.pos.y - origin.y
                if dx or dy:
                    tl = Point(tl.x + dx, tl.y + dy)
                    br = Point(br.x + dx, br.y + dy)
        out.append((tl, br, owner_ref))
    return out


def _keep_in_rects(
    keep_in_specs: list[dict[str, Any]],
    comps: dict[str, Component],
) -> list[tuple[Point, Point, str]]:
    """Protected-component keep-in rects (mounting holes etc.), current pos."""
    out: list[tuple[Point, Point, str]] = []
    for entry in keep_in_specs or []:
        ref = entry.get("ref")
        protected = comps.get(ref)
        if protected is None:
            continue
        margin = float(entry.get("margin_mm", 0.0))
        tl, br = protected.bbox(margin)
        out.append((tl, br, str(ref)))
    return out


def _interval(tl: Point, br: Point, axis: str) -> tuple[float, float]:
    return (tl.x, br.x) if axis == "x" else (tl.y, br.y)


def _max_slide_along_axis(
    comp: Component,
    comps: dict[str, Component],
    *,
    axis: str,
    direction: float,
    want: float,
    clearance: float,
    board_outline: tuple[Point, Point],
    pad_inset: float,
    rects: list[tuple[Point, Point, str]],
) -> float:
    """Largest legal slide of *comp* (<= want) toward *direction* on *axis*.

    Legality model (matches the solver's clearance semantics): a pair is fine
    when its bboxes are separated by >= the gap on AT LEAST one axis. An
    obstacle therefore constrains an x-slide only when its PERPENDICULAR
    separation is already below the gap ("in the slide lane"); the slide must
    then stop ``gap`` short of the obstacle's facing edge. An obstacle that
    already overlaps the slide axis while in-lane means the current state is
    tighter than the gap -- freeze (never push through or worsen it).
    """
    perp_axis = "y" if axis == "x" else "x"
    c_tl, c_br = _effective_bbox(comp, 0.0)
    c_lo, c_hi = _interval(c_tl, c_br, axis)
    c_perp_lo, c_perp_hi = _interval(c_tl, c_br, perp_axis)
    allowed = want

    def _constrain(o_lo: float, o_hi: float, o_perp_lo: float, o_perp_hi: float,
                   gap: float, freeze_on_overlap: bool) -> bool:
        """Apply one obstacle interval. Returns False when frozen (slide 0)."""
        nonlocal allowed
        # In-lane iff perpendicular separation < gap (overlap counts).
        if min(c_perp_hi + gap, o_perp_hi) - max(c_perp_lo - gap, o_perp_lo) <= 0:
            return True
        if direction > 0:
            if o_hi <= c_lo:
                return True  # entirely behind the move
            if o_lo < c_hi:
                # overlapping on the slide axis while in-lane
                if freeze_on_overlap:
                    allowed = 0.0
                    return False
                return True
            allowed = min(allowed, max(0.0, o_lo - gap - c_hi))
        else:
            if o_lo >= c_hi:
                return True
            if o_hi > c_lo:
                if freeze_on_overlap:
                    allowed = 0.0
                    return False
                return True
            allowed = min(allowed, max(0.0, c_lo - (o_hi + gap)))
        return allowed > 0.0

    for other_ref, other in comps.items():
        if other_ref == comp.ref:
            continue
        if not _pair_blocks(comp, other):
            continue
        o_tl, o_br = _effective_bbox(other, 0.0)
        o_lo, o_hi = _interval(o_tl, o_br, axis)
        o_perp_lo, o_perp_hi = _interval(o_tl, o_br, perp_axis)
        if not _constrain(o_lo, o_hi, o_perp_lo, o_perp_hi, clearance,
                          freeze_on_overlap=True):
            return 0.0

    # Keep-out / keep-in rule areas (owner is exempt from its own keep-out).
    # The squeeze never enters one; a comp already overlapping a rect (should
    # not happen after the Step 9.1/9.2 passes) is frozen, not pushed deeper.
    for r_tl, r_br, owner_ref in rects:
        if owner_ref == comp.ref:
            continue
        r_lo, r_hi = _interval(r_tl, r_br, axis)
        r_perp_lo, r_perp_hi = _interval(r_tl, r_br, perp_axis)
        if not _constrain(r_lo, r_hi, r_perp_lo, r_perp_hi, _RECT_MARGIN_MM,
                          freeze_on_overlap=True):
            return 0.0

    # Board bounds: the comp's physical extent stays >= pad_inset inside the
    # outline. Slides run toward the placed centroid so this rarely binds,
    # but a centroid near an edge-pinned cluster can pull parts edge-ward.
    b_tl, b_br = board_outline
    b_lo, b_hi = _interval(b_tl, b_br, axis)
    if direction > 0:
        allowed = min(allowed, (b_hi - pad_inset) - c_hi)
    else:
        allowed = min(allowed, c_lo - (b_lo + pad_inset))

    return max(0.0, allowed)


def compact_toward_centroid(
    comps: dict[str, Component],
    *,
    board_outline: tuple[Point, Point],
    clearance_mm: float,
    keepout_rects: list[Any] | None = None,
    keep_in_specs: list[dict[str, Any]] | None = None,
    pad_inset_mm: float = 0.3,
    passes: int = 6,
) -> dict[str, Any]:
    """Per-axis squeeze of all unlocked components toward the placed centroid.

    Components are processed nearest-to-centroid first so inner parts pack
    first and outer parts can then close the remaining distance. Locked /
    pinned parts (edge connectors, mounting holes) and array members never
    move but still block. Deterministic: sorted iteration, no RNG.

    Convergence is geometric: each slide is capped at the distance to the
    (recomputed) centroid, so a free cluster anchored by a locked part
    closes roughly half its remaining gap per pass. The pass loop breaks
    early once a whole pass moves less than the minimum slide.
    """
    summary: dict[str, Any] = {
        "enabled": True,
        "passes": 0,
        "moved_components": 0,
        "total_slide_mm": 0.0,
        "per_pass_slide_mm": [],
    }
    if not comps:
        return summary

    movable = [
        ref
        for ref, c in sorted(comps.items())
        if not c.locked and not getattr(c, "array_member", False)
    ]
    if not movable:
        return summary

    for _pass in range(max(1, int(passes))):
        pass_slide = 0.0
        for axis in ("x", "y"):
            # Placed-bbox centroid from CURRENT physical extents.
            bboxes = [c.physical_bbox() for c in comps.values()]
            centroid = Point(
                (min(b[0].x for b in bboxes) + max(b[1].x for b in bboxes)) / 2.0,
                (min(b[0].y for b in bboxes) + max(b[1].y for b in bboxes)) / 2.0,
            )
            rects = _resolved_keepout_rects(keepout_rects or [], comps) + (
                _keep_in_rects(keep_in_specs or [], comps)
            )

            def _axis_dist(ref: str) -> float:
                c = comps[ref]
                center = c.body_center if c.body_center is not None else c.pos
                return abs(
                    (center.x - centroid.x) if axis == "x" else (center.y - centroid.y)
                )

            for ref in sorted(movable, key=lambda r: (_axis_dist(r), r)):
                comp = comps[ref]
                center = comp.body_center if comp.body_center is not None else comp.pos
                gap_to_centroid = (
                    centroid.x - center.x if axis == "x" else centroid.y - center.y
                )
                if abs(gap_to_centroid) < _MIN_SLIDE_MM:
                    continue
                direction = 1.0 if gap_to_centroid > 0 else -1.0
                slide = _max_slide_along_axis(
                    comp,
                    comps,
                    axis=axis,
                    direction=direction,
                    want=abs(gap_to_centroid),
                    clearance=max(0.0, float(clearance_mm)),
                    board_outline=board_outline,
                    pad_inset=max(0.0, float(pad_inset_mm)),
                    rects=rects,
                )
                if slide < _MIN_SLIDE_MM:
                    continue
                old_pos = Point(comp.pos.x, comp.pos.y)
                if axis == "x":
                    comp.pos.x += direction * slide
                else:
                    comp.pos.y += direction * slide
                _update_pad_positions(comp, old_pos, comp.rotation)
                pass_slide += slide
                summary["moved_components"] += 1

        summary["passes"] += 1
        summary["per_pass_slide_mm"].append(round(pass_slide, 3))
        summary["total_slide_mm"] += pass_slide
        if pass_slide < _MIN_SLIDE_MM:
            break

    summary["total_slide_mm"] = round(summary["total_slide_mm"], 3)
    return summary


__all__ = ["compact_toward_centroid"]
