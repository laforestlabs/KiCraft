"""Parent FreeRouting budget scales with board complexity.

Two independent scalers (both only raise, never lower):
- C2: dense-interconnect passes/timeout (cross-leaf net count)
- Component-count timeout floor (FreeRouting processes the whole board)
"""
from __future__ import annotations

from kicraft.cli._compose_route import _scale_parent_route_budget
from kicraft.autoplacer.config import DEFAULT_CONFIG


def _cfg(**over):
    c = {"freerouting_max_passes": 20, "freerouting_timeout_s": 60}
    c.update(over)
    return c


# --- Dense-interconnect scaler (C2) -----------------------------------------

def test_sparse_parent_keeps_base_budget():
    # Below the threshold, with few components, the flat defaults are untouched.
    assert _scale_parent_route_budget(3, 5, _cfg()) == (20, 60)
    assert _scale_parent_route_budget(9, 5, _cfg()) == (20, 60)


def test_dense_parent_raises_passes_and_timeout():
    mp, to = _scale_parent_route_budget(15, 5, _cfg())
    assert mp == 40 and to == 180          # bumped at/over the threshold


def test_very_dense_parent_timeout_scales_per_interconnect():
    # run_10's shape: 39 cross-leaf nets. 39 * 8 s = 312 s beats the flat
    # 180 s dense floor; a 22-net parent stays on the floor.
    mp, to = _scale_parent_route_budget(39, 29, _cfg())
    assert to == 312
    _mp, to22 = _scale_parent_route_budget(22, 29, _cfg())
    assert to22 == 180


def test_only_raises_never_lowers_a_hand_tuned_config():
    # A config already tuned higher than the dense defaults is preserved.
    mp, to = _scale_parent_route_budget(50, 5, _cfg(freerouting_max_passes=60,
                                                     freerouting_timeout_s=600))
    assert mp == 60 and to == 600


def test_thresholds_are_configurable():
    cfg = _cfg(parent_dense_interconnect_threshold=2,
               parent_dense_max_passes=30, parent_dense_timeout_s=90)
    assert _scale_parent_route_budget(2, 5, cfg) == (30, 90)
    assert _scale_parent_route_budget(1, 5, cfg) == (20, 60)


# --- Component-count timeout scaler -----------------------------------------

def test_component_count_raises_timeout():
    # 200 components * 1.0 s = 200 s, well above the 60 s base.
    mp, to = _scale_parent_route_budget(3, 200, _cfg())
    assert mp == 20                        # passes unaffected by component count
    assert to == 200


def test_component_count_does_not_lower_base():
    # Few components: the per-component floor is below the base, so base wins.
    mp, to = _scale_parent_route_budget(3, 10, _cfg())
    assert to == 60


def test_component_count_cap():
    # The cap clamps the per-component timeout.
    cfg = _cfg(parent_freerouting_s_per_component=10.0,
               parent_freerouting_timeout_cap_s=300)
    mp, to = _scale_parent_route_budget(3, 200, cfg)
    assert to == 300                       # 200 * 10 = 2000, capped to 300


def test_component_count_and_dense_interconnect_stack():
    # Both scalers apply; the timeout is the max of the two floors.
    mp, to = _scale_parent_route_budget(15, 200, _cfg())
    assert mp == 40                        # dense-interconnect raises passes
    assert to == max(200, 180)             # max(comp 200, dense 180) = 200


# --- GND-plane probe cap (attempt 1) -----------------------------------------

from kicraft.cli._compose_route import _probe_timeout_s


def test_probe_cap_holds_for_sparse_parents():
    # Few interconnects: the flat 120 s hang-detection cap applies.
    assert _probe_timeout_s(4, 180, {}) == 120


def test_probe_cap_scales_with_interconnect_count():
    # run_10's shape: 39 cross-leaf nets. 39 * 5 s = 195 s beats the flat cap;
    # the scaled parent budget still bounds it from above.
    assert _probe_timeout_s(39, 180, {}) == 180
    assert _probe_timeout_s(39, 300, {}) == 195


def test_probe_cap_never_exceeds_the_route_budget():
    assert _probe_timeout_s(100, 90, {}) == 90


def test_probe_cap_is_configurable():
    cfg = {"parent_gnd_plane_probe_timeout_s": 60,
           "parent_probe_s_per_interconnect": 2.0}
    assert _probe_timeout_s(10, 600, cfg) == 60      # 20 < 60 -> flat cap
    assert _probe_timeout_s(50, 600, cfg) == 100     # 100 > 60 -> scaled


# --- Config defaults ---------------------------------------------------------

def test_default_config_carries_the_knobs():
    for k in ("parent_dense_interconnect_threshold", "parent_dense_max_passes",
              "parent_dense_timeout_s", "parent_freerouting_s_per_component",
              "parent_freerouting_timeout_cap_s",
              "parent_gnd_plane_probe_timeout_s",
              "parent_probe_s_per_interconnect",
              "parent_s_per_interconnect"):
        assert k in DEFAULT_CONFIG
