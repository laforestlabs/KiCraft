"""Anti-sprawl candidate-scorer helpers: net_dist de-saturation + sprawl penalty.

Regression for KC-8AG6FU: a 215.8x222.7mm board (9% packing) for a small ESP32 +
3 steppers. The candidate scorer let it win because (a) net_dist saturated to 0
on any large board and (b) the sprawl penalty divided by the *span* of the
components (~= the whole outline -> ratio ~1, never tripped).
"""
from __future__ import annotations

import pytest

from kicraft.cli.compose_subcircuits import _net_dist_score, _sprawl_penalty


# --- net_dist: bounded inverse, never flatlines -------------------------------


def test_net_dist_full_score_at_zero_ratsnest():
    assert _net_dist_score(0.0) == 100.0


def test_net_dist_monotonic_decreasing():
    vals = [_net_dist_score(r) for r in (0, 200, 1000, 1500, 2000, 4000)]
    assert all(a > b for a, b in zip(vals, vals[1:])), vals


def test_net_dist_never_saturates_to_zero_on_large_boards():
    # The old `100 - 0.1*ratsnest` hit 0 at 1000mm and stayed there. The fix
    # must keep a positive, discriminating gradient well past that.
    assert _net_dist_score(1000.0) > 0.0
    assert _net_dist_score(4000.0) > 0.0
    # The key property the saturation broke: a tighter board outranks a looser
    # one even when both are "large".
    assert _net_dist_score(1200.0) > _net_dist_score(1500.0)


def test_net_dist_tracks_old_slope_near_origin():
    # Within a few points of the old linear 100 - 0.1*r for small ratsnest, so
    # well-packed boards are scored ~as before (minimal golden churn).
    assert _net_dist_score(200.0) == pytest.approx(100.0 * 1000.0 / 1200.0)
    # old linear at 200mm was 80; the inverse is ~83 -- close.
    assert abs(_net_dist_score(200.0) - 80.0) < 5.0


# --- sprawl penalty: summed-courtyard denominator -----------------------------


def test_no_penalty_when_reasonably_packed():
    # 40% courtyard packing -> sprawl 2.5 -> under the 3.0 threshold.
    sprawl, penalty = _sprawl_penalty(outline_area_mm2=1000.0, summed_courtyard_area_mm2=400.0)
    assert sprawl == 2.5
    assert penalty == 0.0


def test_penalty_ramps_then_caps():
    # sprawl 5 (20% packing) -> 5*(5-3) = 10
    _, p5 = _sprawl_penalty(5000.0, 1000.0)
    assert p5 == 10.0
    # sprawl 8 (12.5% packing) -> capped at 25
    _, p8 = _sprawl_penalty(8000.0, 1000.0)
    assert p8 == 25.0
    # far past the cap stays capped
    _, p40 = _sprawl_penalty(40000.0, 1000.0)
    assert p40 == 25.0


def test_kc_8ag6fu_spread_clusters_now_penalized():
    # The real board: 215.8 x 222.7mm outline, ~55 small parts. Summed courtyard
    # area is a small fraction of the outline -> high sprawl -> the penalty must
    # fire. (The OLD span-based denominator used the ~205x215mm spread of the
    # parts, giving sprawl ~1.09 and ZERO penalty -- the bug.)
    outline_area = 215.8 * 222.7
    summed_courtyard = 3500.0  # generous estimate of 55 small-part courtyards
    sprawl, penalty = _sprawl_penalty(outline_area, summed_courtyard)
    assert sprawl > 8.0, f"expected heavy sprawl, got {sprawl:.1f}"
    assert penalty == 25.0
    # And the OLD span-based metric would NOT have fired:
    span_area = 205.0 * 215.0
    old_sprawl = outline_area / span_area
    assert old_sprawl < 1.2 and old_sprawl <= 3.0  # would have been 0 penalty


def test_compact_candidate_beats_sprawled_one():
    # Same parts (same summed area), two outlines: the compact one must incur a
    # strictly smaller penalty so it wins the candidate search.
    summed = 1500.0
    _, p_compact = _sprawl_penalty(60.0 * 50.0, summed)   # 3000mm^2 outline
    _, p_sprawled = _sprawl_penalty(216.0 * 222.0, summed)  # 47952mm^2 outline
    assert p_compact < p_sprawled
    assert p_compact == 0.0  # 3000/1500 = sprawl 2.0 -> no penalty
