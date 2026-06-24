"""C2: the parent FreeRouting budget scales with the cross-leaf interconnect count."""
from __future__ import annotations

from kicraft.cli._compose_route import _scale_parent_route_budget
from kicraft.autoplacer.config import DEFAULT_CONFIG


def _cfg(**over):
    c = {"freerouting_max_passes": 20, "freerouting_timeout_s": 60}
    c.update(over)
    return c


def test_sparse_parent_keeps_base_budget():
    # Below the threshold the flat defaults are untouched.
    assert _scale_parent_route_budget(3, _cfg()) == (20, 60)
    assert _scale_parent_route_budget(9, _cfg()) == (20, 60)


def test_dense_parent_raises_passes_and_timeout():
    mp, to = _scale_parent_route_budget(15, _cfg())
    assert mp == 40 and to == 180          # bumped at/over the threshold


def test_only_raises_never_lowers_a_hand_tuned_config():
    # A config already tuned higher than the dense defaults is preserved.
    mp, to = _scale_parent_route_budget(50, _cfg(freerouting_max_passes=60,
                                                 freerouting_timeout_s=600))
    assert mp == 60 and to == 600


def test_thresholds_are_configurable():
    cfg = _cfg(parent_dense_interconnect_threshold=2,
               parent_dense_max_passes=30, parent_dense_timeout_s=90)
    assert _scale_parent_route_budget(2, cfg) == (30, 90)
    assert _scale_parent_route_budget(1, cfg) == (20, 60)


def test_default_config_carries_the_knobs():
    for k in ("parent_dense_interconnect_threshold", "parent_dense_max_passes",
              "parent_dense_timeout_s"):
        assert k in DEFAULT_CONFIG
