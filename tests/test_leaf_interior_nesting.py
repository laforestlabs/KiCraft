"""PR-N1 of docs/plans/shaped-compose-leaf-nesting.md: interior-hole
representation + the containment allowance in ``can_overlap_sparse``.

The seam-short regression the same-side veto exists for (RC2) is pinned
explicitly: any PARTIAL overlap between same-side leaves stays forbidden;
only full containment inside a genuinely enclosed interior hole is allowed.
No pcbnew needed -- blocker sets are constructed directly.
"""

from __future__ import annotations

from kicraft.autoplacer.brain.subcircuit_composer import (
    LeafBlockerSet,
    can_overlap_sparse,
    compute_interior_free_rects,
)
from kicraft.autoplacer.brain.types import Point


def _rect(x0, y0, x1, y1):
    return (Point(float(x0), float(y0)), Point(float(x1), float(y1)))


def _annulus_blocker_set(*, with_interior_decap=False, holes=()):
    """A 57x57 ring leaf: 12 pad rects around r~24, empty centre. Optionally
    an interior decap at r~18 (the ring companions really sit INSIDE)."""
    import math

    pads = []
    for k in range(12):
        ang = 2.0 * math.pi * k / 12.0
        cx, cy = 28.5 + 24.0 * math.cos(ang), 28.5 + 24.0 * math.sin(ang)
        pads.append(_rect(cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5))
    comp_rects = {f"D{k+1}": pads[k] for k in range(12)}
    if with_interior_decap:
        comp_rects["C3"] = _rect(28.5 - 1.0, 9.0, 28.5 + 1.0, 11.5)  # r~18 top
        pads.append(comp_rects["C3"])
    return LeafBlockerSet(
        front_pads=tuple(pads),
        back_pads=(),
        tht_drills=(),
        leaf_outline=_rect(0, 0, 57, 57),
        component_rects=comp_rects,
        interior_free_rects=tuple(holes),
    )


def _small_front_leaf(w=16.0, h=14.0):
    """A guest leaf (MCU-like): front SMT pads spanning w x h."""
    return LeafBlockerSet(
        front_pads=(_rect(1, 1, w - 1, h - 1),),
        back_pads=(),
        tht_drills=(),
        leaf_outline=_rect(0, 0, w, h),
        component_rects={"U1": _rect(1, 1, w - 1, h - 1)},
    )


# --------------------------------------------------------------------------- #
# compute_interior_free_rects
# --------------------------------------------------------------------------- #

def test_annulus_yields_a_centered_interior_hole():
    bs = _annulus_blocker_set()
    holes = compute_interior_free_rects(bs, min_side_mm=8.0)
    assert holes, "the annulus interior must be found"
    (hmin, hmax) = holes[0]
    # Roughly centered and usefully large (ring inner clearance ~ r=21).
    cx, cy = (hmin.x + hmax.x) / 2.0, (hmin.y + hmax.y) / 2.0
    assert abs(cx - 28.5) < 3.0 and abs(cy - 28.5) < 3.0
    assert (hmax.x - hmin.x) >= 20.0 and (hmax.y - hmin.y) >= 20.0
    # The hole is INTERIOR: it must not touch the outline boundary.
    assert hmin.x > 2.0 and hmin.y > 2.0 and hmax.x < 55.0 and hmax.y < 55.0


def test_interior_decap_shrinks_or_shifts_the_hole():
    plain = compute_interior_free_rects(_annulus_blocker_set(), min_side_mm=8.0)
    with_c3 = compute_interior_free_rects(
        _annulus_blocker_set(with_interior_decap=True), min_side_mm=8.0
    )
    assert with_c3, "hole must survive an interior companion"
    area = lambda r: (r[1].x - r[0].x) * (r[1].y - r[0].y)  # noqa: E731
    assert area(with_c3[0]) < area(plain[0])
    # And the decap rect is not inside the reported hole.
    c3 = _rect(27.5, 9.0, 29.5, 11.5)
    (hmin, hmax) = with_c3[0]
    assert not (
        c3[0].x >= hmin.x and c3[1].x <= hmax.x
        and c3[0].y >= hmin.y and c3[1].y <= hmax.y
    )


def test_open_bay_is_not_a_hole():
    # A U-shape: empty space reachable from the boundary is OUTSIDE, not a
    # hole -- nesting there would abut open board edge, not enclosed FR4.
    bs = LeafBlockerSet(
        front_pads=(
            _rect(0, 0, 5, 40),    # left wall
            _rect(35, 0, 40, 40),  # right wall
            _rect(0, 35, 40, 40),  # bottom wall -- top edge (y=0) stays open
        ),
        back_pads=(),
        tht_drills=(),
        leaf_outline=_rect(0, 0, 40, 40),
    )
    assert compute_interior_free_rects(bs, min_side_mm=8.0) == ()


def test_min_side_filters_small_holes():
    bs = _annulus_blocker_set()
    assert compute_interior_free_rects(bs, min_side_mm=60.0) == ()


def test_hole_computation_is_deterministic():
    a = compute_interior_free_rects(_annulus_blocker_set(), min_side_mm=8.0)
    b = compute_interior_free_rects(_annulus_blocker_set(), min_side_mm=8.0)
    assert a == b


# --------------------------------------------------------------------------- #
# can_overlap_sparse containment allowance
# --------------------------------------------------------------------------- #

def _holes_for(bs):
    return compute_interior_free_rects(bs, min_side_mm=8.0)


def test_nested_guest_inside_hole_is_allowed():
    host = _annulus_blocker_set()
    host = _annulus_blocker_set(holes=_holes_for(host))
    guest = _small_front_leaf()
    # Host at world origin (20, 20); guest centered in the hole:
    # hole centre ~ (48.5, 48.5) world; guest is 16x14.
    assert can_overlap_sparse(
        host, Point(20.0, 20.0), 0.0,
        guest, Point(20.0 + 28.5 - 8.0, 20.0 + 28.5 - 7.0), 0.0,
    ) is True


def test_partial_overlap_stays_forbidden_seam_regression():
    host = _annulus_blocker_set(holes=_holes_for(_annulus_blocker_set()))
    guest = _small_front_leaf()
    # Guest straddling the ring copper (half in, half out) -- the exact
    # seam-adjacency RC2 exists to forbid.
    assert can_overlap_sparse(
        host, Point(20.0, 20.0), 0.0,
        guest, Point(20.0 + 50.0, 20.0 + 22.0), 0.0,
    ) is False
    # And fully outside/side-by-side same-side leaves remain incompatible
    # (bbox overlap semantics unchanged for non-nested pairs).
    assert can_overlap_sparse(
        host, Point(20.0, 20.0), 0.0,
        guest, Point(20.0 + 20.0, 20.0 + 45.0), 0.0,
    ) is False


def test_no_holes_means_veto_unchanged():
    host = _annulus_blocker_set()  # interior_free_rects deliberately empty
    guest = _small_front_leaf()
    assert can_overlap_sparse(
        host, Point(20.0, 20.0), 0.0,
        guest, Point(20.0 + 28.5 - 8.0, 20.0 + 28.5 - 7.0), 0.0,
    ) is False


def test_non_cardinal_host_rotation_is_conservative():
    host = _annulus_blocker_set(holes=_holes_for(_annulus_blocker_set()))
    guest = _small_front_leaf()
    assert can_overlap_sparse(
        host, Point(20.0, 20.0), 45.0,
        guest, Point(20.0 + 28.5 - 8.0, 20.0 + 28.5 - 7.0), 0.0,
    ) is False


def test_guest_bigger_than_hole_is_forbidden():
    host = _annulus_blocker_set(holes=_holes_for(_annulus_blocker_set()))
    guest = _small_front_leaf(w=40.0, h=40.0)  # cannot fit the ~26mm hole
    assert can_overlap_sparse(
        host, Point(20.0, 20.0), 0.0,
        guest, Point(20.0 + 28.5 - 20.0, 20.0 + 28.5 - 20.0), 0.0,
    ) is False
