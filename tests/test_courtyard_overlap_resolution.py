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


def test_both_locked_is_reported_unresolved():
    # Two pinned parts overlapping -- the pass cannot move either; it reports
    # the residual so the (minor) gate tolerance can take over.
    comps = {
        "J1": _comp("J1", 10.0, 10.0, locked=True),
        "J2": _comp("J2", 10.5, 10.0, locked=True),
    }
    s = _solver(comps)

    unresolved = s._resolve_courtyard_overlaps(comps)

    assert unresolved == 1
    assert comps["J1"].pos.x == 10.0 and comps["J2"].pos.x == 10.5  # both untouched


def test_non_overlapping_parts_untouched():
    comps = {"R1": _comp("R1", 10.0, 10.0), "R2": _comp("R2", 30.0, 30.0)}
    s = _solver(comps)

    unresolved = s._resolve_courtyard_overlaps(comps)

    assert unresolved == 0
    assert comps["R1"].pos.x == 10.0 and comps["R2"].pos.x == 30.0


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
