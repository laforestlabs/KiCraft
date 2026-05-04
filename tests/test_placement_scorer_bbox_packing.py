"""Tests for PlacementScorer.bbox_packing -- the sub-metric that
penalises isolated drift inside the seed frame.

Drives the metric directly via ``_score_bbox_packing`` rather than the
full ``score()`` pipeline so a regression in any other sub-metric does
not mask a regression here.
"""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Point,
)


def _comp(ref: str, x: float, y: float, w: float = 2.0, h: float = 2.0) -> Component:
    """Minimal Component centered at (x, y) with no pads.

    ``physical_bbox()`` falls back to the courtyard bbox when no pads are
    present, so the placed bbox is exactly w * h centred on (x, y).
    """
    return Component(
        ref=ref,
        value="",
        pos=Point(x, y),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=w,
        height_mm=h,
        kind="passive",
    )


def _state(comps: list[Component]) -> BoardState:
    return BoardState(
        components={c.ref: c for c in comps},
        board_outline=(Point(0.0, 0.0), Point(100.0, 100.0)),
    )


def test_tight_cluster_scores_high():
    # Four 2x2 components touching in a 4x4 footprint:
    # bbox spans (0,0) to (4,4) = 16 mm²; total component area = 16 mm².
    # fill = 1.0 -> score = min(100, 1.0 * 150) = 100.
    comps = [
        _comp("R1", 1.0, 1.0),
        _comp("R2", 3.0, 1.0),
        _comp("R3", 1.0, 3.0),
        _comp("R4", 3.0, 3.0),
    ]
    score = PlacementScorer(_state(comps))._score_bbox_packing()
    assert score >= 90.0, f"tight cluster scored {score:.2f}, expected >= 90"


def test_spread_cluster_scores_low():
    # Same four 2x2 components at the corners of a ~50x50 region.
    # bbox spans (0,0) to (50,50) = 2500 mm²; total area = 16 mm².
    # fill ≈ 0.0064 -> score ≈ 0.96.
    comps = [
        _comp("R1", 1.0, 1.0),
        _comp("R2", 49.0, 1.0),
        _comp("R3", 1.0, 49.0),
        _comp("R4", 49.0, 49.0),
    ]
    score = PlacementScorer(_state(comps))._score_bbox_packing()
    assert score < 50.0, f"spread cluster scored {score:.2f}, expected < 50"


def test_single_component_returns_100():
    # len(comps) < 2 -> no spread to score.
    score = PlacementScorer(_state([_comp("R1", 5.0, 5.0)]))._score_bbox_packing()
    assert score == pytest.approx(100.0)


def test_zero_components_returns_100():
    # Empty state -> no opinion. Differs from _score_parent_composition,
    # which has its own fallback path returning 0; that asymmetry is
    # explicit at each call site, not in the helper.
    score = PlacementScorer(_state([]))._score_bbox_packing()
    assert score == pytest.approx(100.0)


def test_zero_area_returns_100():
    # Pathological zero-width components: total_area <= 0 early-out.
    # Defends SA acceptance against NaN propagation: math.exp(NaN) is NaN,
    # `rng.random() < NaN` is always False, so a NaN here would silently
    # reject every SA move.
    c1 = _comp("R1", 1.0, 1.0, w=0.0, h=2.0)
    c2 = _comp("R2", 2.0, 1.0, w=0.0, h=2.0)
    score = PlacementScorer(_state([c1, c2]))._score_bbox_packing()
    assert score == pytest.approx(100.0)
