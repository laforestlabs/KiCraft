"""Final courtyard-separation legalization (PlacementSolver Step 16).

The earlier overlap passes run BEFORE pinned-restore / clamp / keep-out-clear,
so a same-side pair the solver separated can drift back into overlap and survive
to the routed board as a ``courtyards_overlap`` DRC error. ``_resolve_courtyard_
overlaps`` runs LAST and guarantees no same-side courtyard overlap remains,
moving only the unlocked partner so pinned parts keep their positions.
"""

from __future__ import annotations

from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.placement_utils import _bbox_overlap_xy, _effective_bbox
from kicraft.autoplacer.brain.types import BoardState, Component, Layer, Point


def _comp(ref: str, x: float, y: float, *, w: float = 2.0, h: float = 2.0,
          locked: bool = False, layer: Layer = Layer.FRONT) -> Component:
    return Component(
        ref=ref, value="", pos=Point(x, y), rotation=0.0, layer=layer,
        width_mm=w, height_mm=h, kind="passive", locked=locked,
        body_center=Point(x, y),
    )


def _solver(comps: dict[str, Component]) -> PlacementSolver:
    state = BoardState(
        components=comps, board_outline=(Point(0.0, 0.0), Point(60.0, 60.0))
    )
    return PlacementSolver(state, {"courtyard_overlap_min_gap_mm": 0.15}, seed=0)


def _courtyards_overlap(a: Component, b: Component) -> bool:
    ox, oy = _bbox_overlap_xy(*_effective_bbox(a, 0.0), *_effective_bbox(b, 0.0))
    return ox > 0 and oy > 0


def test_two_free_parts_get_separated():
    # 2x2 courtyards centered 0.5mm apart -> heavy overlap.
    comps = {"R1": _comp("R1", 10.0, 10.0), "R2": _comp("R2", 10.5, 10.0)}
    s = _solver(comps)
    assert _courtyards_overlap(comps["R1"], comps["R2"])  # precondition

    unresolved = s._resolve_courtyard_overlaps(comps)

    assert unresolved == 0
    assert not _courtyards_overlap(comps["R1"], comps["R2"])


def test_locked_partner_never_moves():
    # A pinned (locked) connector and a passive whose courtyards overlap: only
    # the passive may move; the locked part keeps its exact position.
    comps = {
        "J1": _comp("J1", 10.0, 10.0, w=4.0, h=4.0, locked=True),
        "R1": _comp("R1", 11.5, 10.0),
    }
    s = _solver(comps)
    assert _courtyards_overlap(comps["J1"], comps["R1"])

    s._resolve_courtyard_overlaps(comps)

    assert comps["J1"].pos.x == 10.0 and comps["J1"].pos.y == 10.0  # untouched
    assert not _courtyards_overlap(comps["J1"], comps["R1"])


def test_both_locked_same_edge_slides_apart_along_edge():
    # Two edge-pinned (locked) connectors on the same edge whose courtyards
    # overlap (run_26: servo headers at 3.40mm pitch vs 3.63mm courtyards).
    # The pin fixes only the perpendicular coordinate, so the pass slides one
    # ALONG the edge: same y (flushness preserved), separated in x.
    comps = {
        "J1": _comp("J1", 10.0, 10.0, locked=True),
        "J2": _comp("J2", 10.5, 10.0, locked=True),
    }
    s = _solver(comps)
    assert _courtyards_overlap(comps["J1"], comps["J2"])  # precondition

    unresolved = s._resolve_courtyard_overlaps(comps)

    assert unresolved == 0
    assert not _courtyards_overlap(comps["J1"], comps["J2"])
    assert comps["J1"].pos.y == 10.0 and comps["J2"].pos.y == 10.0  # edge kept


def test_both_locked_mounting_holes_stay_untouched():
    # Mounting holes are the user's spec, not an edge pin: never slid.
    comps = {
        "H1": _comp("H1", 10.0, 10.0, locked=True),
        "H2": _comp("H2", 10.5, 10.0, locked=True),
    }
    for c in comps.values():
        c.kind = "mounting_hole"
    s = _solver(comps)

    unresolved = s._resolve_courtyard_overlaps(comps)

    assert unresolved == 1
    assert comps["H1"].pos.x == 10.0 and comps["H2"].pos.x == 10.5  # both untouched


def test_non_overlapping_parts_untouched():
    comps = {"R1": _comp("R1", 10.0, 10.0), "R2": _comp("R2", 30.0, 30.0)}
    s = _solver(comps)

    unresolved = s._resolve_courtyard_overlaps(comps)

    assert unresolved == 0
    assert comps["R1"].pos.x == 10.0 and comps["R2"].pos.x == 30.0


def test_push_falls_back_to_free_axis_when_pinned_axis_blocked():
    # A free part flush against the RIGHT board edge whose smaller overlap with a
    # locked neighbour is on the X (board-edge) axis: pushing it further right is
    # clamped, so the pass must fall back to the Y axis. Without the fallback the
    # pair never separates (the run_06 USB-C breakout courtyard survivor).
    # Board is 60x60; clamp pins x at 60 - hw - 1 = 57.
    comps = {
        "J1": _comp("J1", 54.0, 14.0, w=4.0, h=4.0, locked=True),
        "R1": _comp("R1", 57.0, 15.0, w=4.0, h=4.0),  # already at the x-clamp
    }
    s = _solver(comps)
    assert _courtyards_overlap(comps["J1"], comps["R1"])  # precondition
    # smaller overlap is on X (the blocked, board-edge axis)
    ox, oy = _bbox_overlap_xy(
        *_effective_bbox(comps["J1"], 0.0), *_effective_bbox(comps["R1"], 0.0)
    )
    assert ox < oy

    s._resolve_courtyard_overlaps(comps)

    assert comps["J1"].pos.x == 54.0 and comps["J1"].pos.y == 14.0  # locked
    assert comps["R1"].pos.x == 57.0  # pinned x axis preserved (only y moved)
    assert not _courtyards_overlap(comps["J1"], comps["R1"])  # separated via Y


# --- parent-side exemption: only OPPOSITE-side stacks are courtyard-exempt ---

from kicraft.autoplacer.brain.placement_utils import _back_courtyard  # noqa: E402
from kicraft.autoplacer.brain.subcircuit_composer import LeafBlockerSet  # noqa: E402


def _empty_blocker() -> LeafBlockerSet:
    # No real copper -> can_overlap_sparse() returns True for any pair, so the
    # pair is "copper compatible". Courtyard exemption must then turn on the
    # SIDE, not on copper compatibility.
    return LeafBlockerSet(
        front_pads=(), back_pads=(), tht_drills=(),
        leaf_outline=(Point(0.0, 0.0), Point(2.0, 2.0)),
    )


def _block(ref: str, x: float, y: float, side: str, *, w: float = 4.0,
           h: float = 4.0) -> Component:
    return Component(
        ref=ref, value="", pos=Point(x, y), rotation=0.0, layer=Layer.FRONT,
        width_mm=w, height_mm=h, kind="subcircuit", locked=False,
        body_center=Point(x, y), block_blocker_set=_empty_blocker(),
        block_side=side,
    )


def test_back_courtyard_predicate():
    assert _back_courtyard(_block("A", 0, 0, "back")) is True
    assert _back_courtyard(_block("A", 0, 0, "front")) is False
    assert _back_courtyard(_block("A", 0, 0, "none")) is False
    forced = _block("A", 0, 0, "front")
    forced.block_force_back_only = True
    assert _back_courtyard(forced) is True


def test_same_side_compatible_blocks_are_separated():
    # Two copper-compatible SAME-side blocks (e.g. two THT pin-headers whose
    # annular rings don't touch) still share one courtyard layer -> a real
    # courtyards_overlap DRC. Copper compatibility must NOT exempt them.
    comps = {
        "A": _block("A", 10.0, 10.0, "none"),
        "B": _block("B", 11.0, 10.0, "none"),
    }
    s = _solver(comps)
    assert _courtyards_overlap(comps["A"], comps["B"])  # precondition

    s._resolve_courtyard_overlaps(comps)

    assert not _courtyards_overlap(comps["A"], comps["B"])  # separated


def test_opposite_side_stack_is_exempt():
    # A genuine opposite-side stack (front block over a back block) has its
    # courtyards on different copper layers (F.CrtYd vs B.CrtYd) and never
    # DRC-overlaps -- the pass must leave the deliberate stack intact.
    comps = {
        "A": _block("A", 10.0, 10.0, "front"),
        "B": _block("B", 11.0, 10.0, "back"),
    }
    s = _solver(comps)
    assert _courtyards_overlap(comps["A"], comps["B"])  # precondition

    s._resolve_courtyard_overlaps(comps)

    # Untouched: the stack is preserved.
    assert comps["A"].pos.x == 10.0 and comps["B"].pos.x == 11.0
    assert _courtyards_overlap(comps["A"], comps["B"])


# --- magnitude classification (verify-gate severity) ---

from kicraft.autoplacer.courtyard_overlap import (  # noqa: E402
    CourtyardOverlap,
    classify_courtyard_overlaps,
)


def test_classify_minor_vs_gross():
    minor = CourtyardOverlap("R7", "SW2", "F", area_mm2=0.23, penetration_mm=0.31)
    gross_deep = CourtyardOverlap("U1", "U2", "F", area_mm2=0.4, penetration_mm=1.2)
    gross_big = CourtyardOverlap("J1", "J2", "F", area_mm2=3.0, penetration_mm=0.4)
    m, g = classify_courtyard_overlaps(
        [minor, gross_deep, gross_big], max_penetration_mm=0.5, max_area_mm2=0.5
    )
    assert m == [minor]
    assert set(g) == {gross_deep, gross_big}


def test_minor_requires_both_thresholds():
    # Shallow but large-area -> gross; small-area but deep -> gross.
    shallow_big = CourtyardOverlap("A", "B", "F", area_mm2=2.0, penetration_mm=0.1)
    assert not shallow_big.is_minor(0.5, 0.5)
    small_deep = CourtyardOverlap("A", "B", "F", area_mm2=0.05, penetration_mm=0.9)
    assert not small_deep.is_minor(0.5, 0.5)
    tiny = CourtyardOverlap("A", "B", "F", area_mm2=0.05, penetration_mm=0.2)
    assert tiny.is_minor(0.5, 0.5)
