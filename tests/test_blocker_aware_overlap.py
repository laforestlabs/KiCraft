"""Blocker-aware overlap gates inside PlacementSolver and PlacementScorer.

These tests cover the new contract that lets the unified solver treat the
parent-side path differently from the leaf-side path:

  - When two components both carry ``block_blocker_set`` and their sparse
    blocker sets do not conflict, the solver permits bbox overlap and the
    scorer waives the courtyard penalty.
  - When at least one component lacks a blocker set, both pathways fall
    back to today's bbox-only behavior (the leaf contract).
  - The world-frame artifact origin computation is rotation-aware.

The leaf-equivalence guarantee (no behavior change with all blocker sets
left as None) is also verified end-to-end against the full pytest suite.
"""

from __future__ import annotations

import math

from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.placement_utils import (
    _blocker_pair_compatible,
    _world_artifact_origin,
)
from kicraft.autoplacer.brain.subcircuit_composer import LeafBlockerSet
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Point,
)


def _make_blocker_set(
    *,
    front_pads=(),
    back_pads=(),
    tht_drills=(),
    leaf_outline=(Point(0.0, 0.0), Point(10.0, 10.0)),
) -> LeafBlockerSet:
    return LeafBlockerSet(
        front_pads=tuple(front_pads),
        back_pads=tuple(back_pads),
        tht_drills=tuple(tht_drills),
        leaf_outline=leaf_outline,
    )


def _make_block(
    ref: str,
    *,
    pos: Point,
    rotation: float = 0.0,
    width: float = 10.0,
    height: float = 10.0,
    blocker_set: LeafBlockerSet | None = None,
    layer: Layer = Layer.FRONT,
) -> Component:
    comp = Component(
        ref=ref,
        value="",
        pos=Point(pos.x, pos.y),
        rotation=rotation,
        layer=layer,
        width_mm=width,
        height_mm=height,
        kind="subcircuit",
    )
    if blocker_set is not None:
        comp.block_blocker_set = blocker_set
        # Body center sits at local (width/2, height/2); placing the
        # block at pos means the artifact origin is pos - rotated(offset).
        comp.block_artifact_origin_offset = Point(width / 2.0, height / 2.0)
    return comp


def _board_state(comps: dict[str, Component], outline_size: float = 200.0) -> BoardState:
    return BoardState(
        components=comps,
        nets={},
        traces=[],
        vias=[],
        silkscreen=[],
        board_outline=(
            Point(-outline_size / 2.0, -outline_size / 2.0),
            Point(outline_size / 2.0, outline_size / 2.0),
        ),
    )


# ---------------------------------------------------------------------------
# _world_artifact_origin: the rotation math is the trickiest part.


def test_world_origin_zero_rotation():
    blockers = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(2.0, 2.0))])
    comp = _make_block("A", pos=Point(50.0, 50.0), rotation=0.0, blocker_set=blockers)
    origin = _world_artifact_origin(comp)
    assert math.isclose(origin.x, 45.0, abs_tol=1e-9)
    assert math.isclose(origin.y, 45.0, abs_tol=1e-9)


def test_world_origin_90_rotation():
    blockers = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(2.0, 2.0))])
    comp = _make_block("A", pos=Point(50.0, 50.0), rotation=90.0, blocker_set=blockers)
    origin = _world_artifact_origin(comp)
    # (5, 5) rotated by +90 -> (-5, 5); world_origin = pos - rotated = (55, 45)
    assert math.isclose(origin.x, 55.0, abs_tol=1e-9)
    assert math.isclose(origin.y, 45.0, abs_tol=1e-9)


def test_world_origin_180_rotation():
    blockers = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(2.0, 2.0))])
    comp = _make_block("A", pos=Point(50.0, 50.0), rotation=180.0, blocker_set=blockers)
    origin = _world_artifact_origin(comp)
    # (5, 5) rotated by 180 -> (-5, -5); world_origin = pos - rotated = (55, 55)
    assert math.isclose(origin.x, 55.0, abs_tol=1e-9)
    assert math.isclose(origin.y, 55.0, abs_tol=1e-9)


def test_world_origin_270_rotation():
    blockers = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(2.0, 2.0))])
    comp = _make_block("A", pos=Point(50.0, 50.0), rotation=270.0, blocker_set=blockers)
    origin = _world_artifact_origin(comp)
    # (5, 5) rotated by 270 -> (5, -5); world_origin = pos - rotated = (45, 55)
    assert math.isclose(origin.x, 45.0, abs_tol=1e-9)
    assert math.isclose(origin.y, 55.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# _blocker_pair_compatible: front+front, front+back, mixed.


def test_compatible_front_only_vs_back_only():
    front_only = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    back_only = _make_blocker_set(back_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    a = _make_block("A", pos=Point(0.0, 0.0), blocker_set=front_only)
    b = _make_block("B", pos=Point(0.0, 0.0), blocker_set=back_only)
    assert _blocker_pair_compatible(a, b) is True


def test_incompatible_front_front():
    front_a = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    front_b = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    a = _make_block("A", pos=Point(0.0, 0.0), blocker_set=front_a)
    b = _make_block("B", pos=Point(0.0, 0.0), blocker_set=front_b)
    assert _blocker_pair_compatible(a, b) is False


def test_mixed_pair_one_without_blocker_set_returns_false():
    front_only = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    a = _make_block("A", pos=Point(0.0, 0.0), blocker_set=front_only)
    # B has no blocker_set: leaf contract, falls through to bbox-only.
    b = _make_block("B", pos=Point(0.0, 0.0), blocker_set=None)
    assert _blocker_pair_compatible(a, b) is False


def test_compatible_under_each_rotation():
    front_only = _make_blocker_set(
        front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))]
    )
    back_only = _make_blocker_set(
        back_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))]
    )
    for rot_a in (0.0, 90.0, 180.0, 270.0):
        for rot_b in (0.0, 90.0, 180.0, 270.0):
            a = _make_block(
                "A", pos=Point(50.0, 50.0), rotation=rot_a, blocker_set=front_only
            )
            b = _make_block(
                "B", pos=Point(50.0, 50.0), rotation=rot_b, blocker_set=back_only
            )
            assert _blocker_pair_compatible(a, b) is True, (
                f"front-only x back-only must overlap at rotation ({rot_a}, {rot_b})"
            )


def test_incompatible_under_each_rotation():
    front_a = _make_blocker_set(
        front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))]
    )
    front_b = _make_blocker_set(
        front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))]
    )
    for rot_a in (0.0, 90.0, 180.0, 270.0):
        for rot_b in (0.0, 90.0, 180.0, 270.0):
            a = _make_block(
                "A", pos=Point(50.0, 50.0), rotation=rot_a, blocker_set=front_a
            )
            b = _make_block(
                "B", pos=Point(50.0, 50.0), rotation=rot_b, blocker_set=front_b
            )
            assert _blocker_pair_compatible(a, b) is False, (
                f"front-only x front-only must NOT overlap at rotation ({rot_a}, {rot_b})"
            )


# ---------------------------------------------------------------------------
# _resolve_overlaps: same-side pair separates; opposite-side pair stays put.


def _bboxes_overlap(a: Component, b: Component) -> bool:
    a_tl, a_br = a.bbox(0.0)
    b_tl, b_br = b.bbox(0.0)
    return (
        a_tl.x < b_br.x
        and a_br.x > b_tl.x
        and a_tl.y < b_br.y
        and a_br.y > b_tl.y
    )


def test_resolve_overlaps_separates_same_side_pair():
    front_a = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    front_b = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    a = _make_block("A", pos=Point(50.0, 50.0), blocker_set=front_a)
    b = _make_block("B", pos=Point(52.0, 50.0), blocker_set=front_b)
    state = _board_state({"A": a, "B": b})
    solver = PlacementSolver(state, config={}, seed=0)
    solver._resolve_overlaps(state.components)
    assert not _bboxes_overlap(a, b), "same-side blockers must be pushed apart"


def test_resolve_overlaps_keeps_opposite_side_pair():
    front_only = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    back_only = _make_blocker_set(back_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    a = _make_block("A", pos=Point(50.0, 50.0), blocker_set=front_only)
    b = _make_block("B", pos=Point(52.0, 50.0), blocker_set=back_only)
    state = _board_state({"A": a, "B": b})
    solver = PlacementSolver(state, config={}, seed=0)
    solver._resolve_overlaps(state.components)
    # Positions should be untouched (or close to it) because the pair is
    # blocker-compatible: bbox overlap is legal.
    assert _bboxes_overlap(a, b), (
        "opposite-side blockers should keep their bbox overlap intact"
    )
    assert math.isclose(a.pos.x, 50.0, abs_tol=0.01)
    assert math.isclose(b.pos.x, 52.0, abs_tol=0.01)


# ---------------------------------------------------------------------------
# _score_courtyard_overlap: opposite-side pair contributes no penalty.


def test_courtyard_score_skips_compatible_overlap():
    front_only = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    back_only = _make_blocker_set(back_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    a = _make_block("A", pos=Point(50.0, 50.0), blocker_set=front_only)
    b = _make_block("B", pos=Point(50.0, 50.0), blocker_set=back_only)
    state = _board_state({"A": a, "B": b})
    scorer = PlacementScorer(state, config={})
    assert scorer._score_courtyard_overlap() == 100.0


def test_courtyard_score_penalizes_incompatible_overlap():
    front_a = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    front_b = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    a = _make_block("A", pos=Point(50.0, 50.0), blocker_set=front_a)
    b = _make_block("B", pos=Point(50.0, 50.0), blocker_set=front_b)
    state = _board_state({"A": a, "B": b})
    scorer = PlacementScorer(state, config={})
    assert scorer._score_courtyard_overlap() < 100.0


def test_courtyard_score_mixed_pair_uses_bbox_only():
    front_only = _make_blocker_set(front_pads=[(Point(0.0, 0.0), Point(10.0, 10.0))])
    a = _make_block("A", pos=Point(50.0, 50.0), blocker_set=front_only)
    # B has no blocker_set (leaf path).
    b = _make_block("B", pos=Point(50.0, 50.0), blocker_set=None)
    state = _board_state({"A": a, "B": b})
    scorer = PlacementScorer(state, config={})
    assert scorer._score_courtyard_overlap() < 100.0
