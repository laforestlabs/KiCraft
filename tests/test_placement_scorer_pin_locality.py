"""Tests for PlacementScorer._score_pin_locality -- the connectivity-first
objective that pulls each 2-pad passive against the anchor pin pair it bridges.

Drives the sub-metric directly (and via ``score()`` for the byte-identical
early-out) so a regression here is not masked by another sub-metric. Shares the
single-source ``pin_locality_for_passive`` kernel with the metric module, so
these also guard that the scorer's view agrees with ``leaf_pin_locality``.
"""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Pad,
    Point,
)


def _pad(owner: str, pad_id: str, x: float, y: float, net: str) -> Pad:
    return Pad(ref=owner, pad_id=pad_id, pos=Point(x, y), net=net, layer=Layer.FRONT)


def _ic(ref: str, x: float, y: float, pads: list[Pad]) -> Component:
    return Component(
        ref=ref, value="", pos=Point(x, y), rotation=0.0, layer=Layer.FRONT,
        width_mm=6.0, height_mm=6.0, kind="ic", pads=pads, body_center=Point(x, y),
    )


def _cap(ref: str, x: float, y: float, pads: list[Pad], rot: float = 0.0) -> Component:
    return Component(
        ref=ref, value="", pos=Point(x, y), rotation=rot, layer=Layer.FRONT,
        width_mm=2.0, height_mm=1.0, kind="passive", pads=pads, body_center=Point(x, y),
    )


def _state(comps: list[Component]) -> BoardState:
    return BoardState(
        components={c.ref: c for c in comps},
        board_outline=(Point(0.0, 0.0), Point(100.0, 100.0)),
    )


# U1 power pins: +3V3 at (0,0), GND at (0,2).
def _u1() -> Component:
    return _ic("U1", -3, 1, [_pad("U1", "1", 0.0, 0.0, "+3V3"), _pad("U1", "2", 0.0, 2.0, "GND")])


def _decap(ref: str, x: float) -> Component:
    """A decap straddling +3V3/GND, offset ``x`` mm to the right of U1's pins."""
    return _cap(ref, x, 1, [_pad(ref, "1", x, 0.0, "+3V3"), _pad(ref, "2", x, 2.0, "GND")])


def test_early_out_neutral_when_unweighted():
    # No psw_pin_locality -> the term is a no-op (100), scoring byte-identical.
    scorer = PlacementScorer(_state([_u1(), _decap("C1", 8.0)]))
    assert scorer._score_pin_locality() == pytest.approx(100.0)


def test_closer_cap_scores_higher():
    cfg = {"psw_pin_locality": 0.2}
    near = PlacementScorer(_state([_u1(), _decap("C1", 0.5)]), cfg)._score_pin_locality()
    far = PlacementScorer(_state([_u1(), _decap("C2", 10.0)]), cfg)._score_pin_locality()
    assert near > far
    assert far < 100.0  # a 10 mm decap is not "local"


def test_neutral_when_no_anchors():
    # Two passives, no IC/connector anchor -> nothing to be local to.
    cfg = {"psw_pin_locality": 0.2}
    comps = [_decap("C1", 1.0), _decap("C2", 3.0)]
    assert PlacementScorer(_state(comps), cfg)._score_pin_locality() == pytest.approx(100.0)


def test_total_penalizes_a_far_decap_when_weighted():
    # With the term weighted, the full weighted total must drop for a far decap.
    cfg = {"psw_pin_locality": 0.3}
    near = PlacementScorer(_state([_u1(), _decap("C1", 0.5)]), cfg).score().total
    far = PlacementScorer(_state([_u1(), _decap("C2", 12.0)]), cfg).score().total
    assert near > far


def test_weight_toggles_total_on_same_state():
    # On a FIXED placement, turning the term ON must move the total (a far decap
    # is penalized); OFF it is neutral. Isolates the term from geometry changes
    # (comparing two different placements would also move net_distance etc.).
    state = _state([_u1(), _decap("C1", 12.0)])
    off = PlacementScorer(state).score().total
    on = PlacementScorer(state, {"psw_pin_locality": 0.3}).score().total
    assert on < off
