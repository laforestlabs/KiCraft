"""Guard for the parent-compose compactness fix (docs/plans/parent-compose-
compactness-plan.md).

Fix 1 (RC-P1): ``_seed_outline_dimensions`` / ``_edge_bank_geometry`` must
account for edge-pinned children PER EDGE -- opposing banks (left vs right,
top vs bottom) add their *depths* across the board while same-edge members
stack along the edge. The old floor summed every edge child's width into one
horizontal row, inflating a ~100 mm board's seed to 218 mm when 9 of 10 leaves
were edge-pinned (KC-AXHQTP).

Fix 2: ``_refit_seed_from_placement`` derives a right-sized seed from a
completed placement so the search can re-solve tighter, returning ``None`` when
pass 1 was already tight.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import kicraft.cli.compose_subcircuits as cs
from kicraft.autoplacer.brain.types import Point


def _spec(idx, target, value, *, strict=True):
    """A minimal PlacementSpec stand-in: one constraint pinning child ``idx``."""
    c = SimpleNamespace(target=target, value=value, strict=strict, child_index=idx)
    return SimpleNamespace(child_index=idx, constraints=[c])


def _derived(specs):
    return SimpleNamespace(child_specs={s.child_index: s for s in specs})


def _fake_artifacts(monkeypatch, dims):
    """Stub N leaf artifacts with fixed content bboxes (w, h)."""
    arts = [object() for _ in dims]
    sizes = {id(a): wh for a, wh in zip(arts, dims)}

    def _fake_transform(art, origin, rotation):
        w, h = sizes[id(art)]
        return SimpleNamespace(bounding_box=(Point(0.0, 0.0), Point(w, h)))

    monkeypatch.setattr(cs, "transform_loaded_artifact", _fake_transform)
    return arts


def _box(x0, y0, x1, y1):
    return (Point(x0, y0), Point(x1, y1))


# --- _edge_bank_geometry: the exact per-edge arithmetic ---------------------


def test_edge_bank_geometry_opposing_lr_banks_add_depth_not_width():
    # 5 left + 4 right leaves (the KC-AXHQTP shape). Widths increase per bank.
    widths = {0: 10, 1: 12, 2: 14, 3: 16, 4: 18, 5: 20, 6: 25, 7: 30, 8: 35}
    heights = {0: 15, 1: 15, 2: 15, 3: 15, 4: 15, 5: 18, 6: 18, 7: 18, 8: 18}
    derived = _derived(
        [_spec(i, "edge", "left") for i in range(5)]
        + [_spec(i, "edge", "right") for i in range(5, 9)]
    )
    banks = cs._edge_bank_geometry(derived, widths, heights, spacing_mm=2.0)

    assert banks["lr_present"] is True
    assert banks["tb_present"] is False
    # Depth into the board = widest LEFT member (18) + widest RIGHT member (35),
    # NOT the 180 mm sum of all nine edge widths that RC-P1 used to compute.
    assert banks["lr_depth_w"] == pytest.approx(18.0 + 35.0)
    # Height = the taller bank's vertical stack:
    #   left  = 5*15 + 6*2 = 87 ; right = 4*18 + 5*2 = 82 -> 87.
    assert banks["lr_stack_h"] == pytest.approx(87.0)


def test_edge_bank_geometry_top_bottom_is_the_transpose():
    widths = {0: 10, 1: 20, 2: 30, 3: 40}
    heights = {0: 5, 1: 6, 2: 7, 3: 8}
    derived = _derived(
        [_spec(0, "edge", "top"), _spec(1, "edge", "top"),
         _spec(2, "edge", "bottom"), _spec(3, "edge", "bottom")]
    )
    banks = cs._edge_bank_geometry(derived, widths, heights, spacing_mm=2.0)

    assert banks["tb_present"] is True
    assert banks["lr_present"] is False
    # Width = the wider bank's horizontal stack:
    #   top = 10+20 + 3*2 = 36 ; bottom = 30+40 + 3*2 = 76 -> 76.
    assert banks["tb_stack_w"] == pytest.approx(76.0)
    # Depth = tallest TOP (6) + tallest BOTTOM (8).
    assert banks["tb_depth_h"] == pytest.approx(6.0 + 8.0)


def test_edge_bank_geometry_corners_contribute_max_not_sum():
    widths = {0: 20, 1: 10}
    heights = {0: 14, 1: 8}
    derived = _derived(
        [_spec(0, "corner", "top-left"), _spec(1, "corner", "bottom-right")]
    )
    banks = cs._edge_bank_geometry(derived, widths, heights, spacing_mm=2.0)

    assert banks["lr_present"] is False and banks["tb_present"] is False
    assert banks["corner_w"] == pytest.approx(20.0)  # max, not 30
    assert banks["corner_h"] == pytest.approx(14.0)  # max, not 22


# --- _seed_outline_dimensions: end-to-end (the shipped floor) ---------------


def test_seed_not_inflated_by_opposing_edge_width_sum(monkeypatch):
    # 5 left + 4 right + 1 unconstrained interior leaf.
    dims = [
        (10, 15), (12, 15), (14, 15), (16, 15), (18, 15),   # left bank
        (20, 18), (25, 18), (30, 18), (35, 18),             # right bank
        (8, 8),                                             # interior
    ]
    arts = _fake_artifacts(monkeypatch, dims)
    derived = _derived(
        [_spec(i, "edge", "left") for i in range(5)]
        + [_spec(i, "edge", "right") for i in range(5, 9)]
    )
    # area_overhead=1.0 keeps the area basis from masking the edge floor.
    w, h = cs._seed_outline_dimensions(arts, derived, 2.0, area_overhead=1.0)

    # The OLD single-row floor summed all nine edge widths (+gaps) -> ~200 mm.
    # Per-edge banks cap the width contribution at maxw(left)+maxw(right) ~ 59.
    assert w < 100.0
    # The taller edge stack (87 mm) drives the seed height.
    assert h >= 86.0


def test_seed_still_covers_biggest_single_child(monkeypatch):
    # No constraints at all: the max-single-child solvability floor still holds
    # (the sum*0.6 fallback was removed, not the max-child floor).
    arts = _fake_artifacts(monkeypatch, [(50.0, 40.0), (10.0, 8.0)])
    derived = _derived([])
    w, h = cs._seed_outline_dimensions(arts, derived, 2.0, area_overhead=0.5)
    assert w >= 50.0 + 2.0 * 4
    assert h >= 40.0 + 2.0 * 4


# --- _refit_seed_from_placement (Fix 2) -------------------------------------


def test_refit_shrinks_seed_when_interior_floated_in_a_big_canvas():
    # A 213x101 seed (KC-AXHQTP): two left + two right banks pinned flush to the
    # oversized edges, one small interior block floated mid-board.
    placed = {
        0: _box(0, 0, 18, 40),       # left bank
        1: _box(0, 42, 18, 82),      # left bank
        2: _box(180, 0, 215, 45),    # right bank
        3: _box(180, 47, 215, 92),   # right bank
        4: _box(95, 40, 110, 60),    # interior: 15 wide x 20 tall
    }
    derived = _derived(
        [_spec(0, "edge", "left"), _spec(1, "edge", "left"),
         _spec(2, "edge", "right"), _spec(3, "edge", "right")]
    )
    seed = (213.0, 101.0)
    refit = cs._refit_seed_from_placement(placed, derived, 2.0, seed)

    assert refit is not None
    rw, rh = refit
    # width = left depth (18) | interior (15) | right depth (35) + gaps (3*2).
    assert rw == pytest.approx(18.0 + 15.0 + 35.0 + 2.0 * 3)
    assert rw < 100.0                      # collapsed from 213 mm
    assert rw <= seed[0] and rh <= seed[1]  # a re-fit only ever tightens


def test_refit_returns_none_when_already_tight():
    # Two interior blocks already snug in a 66x34 seed -> no meaningful slack.
    placed = {0: _box(0, 0, 30, 30), 1: _box(32, 0, 62, 30)}
    derived = _derived([])
    assert cs._refit_seed_from_placement(placed, derived, 2.0, (66.0, 34.0)) is None
