"""Tests for _mutate_config and CONFIG_SEARCH_SPACE in autoexperiment.

All tests use synthetic data only; no pcbnew dependency.
"""

from __future__ import annotations

import random


from kicraft.autoplacer.config import CONFIG_SEARCH_SPACE, DEFAULT_CONFIG
from kicraft.cli.autoexperiment import (
    _config_overlay_from_defaults,
    _jsonable_config_value,
    _mutate_config,
)


# ---------------------------------------------------------------------------
# CONFIG_SEARCH_SPACE validation
# ---------------------------------------------------------------------------


class TestConfigSearchSpaceStructure:
    """Validate CONFIG_SEARCH_SPACE entries have correct schema."""

    def test_all_keys_exist_in_default_config(self):
        """Every search space key must exist in DEFAULT_CONFIG."""
        for key in CONFIG_SEARCH_SPACE:
            assert key in DEFAULT_CONFIG, f"{key} in search space but not in DEFAULT_CONFIG"

    def test_required_fields_present(self):
        """Each entry must have min, max, sigma, type."""
        for key, spec in CONFIG_SEARCH_SPACE.items():
            assert "min" in spec, f"{key} missing 'min'"
            assert "max" in spec, f"{key} missing 'max'"
            assert "sigma" in spec, f"{key} missing 'sigma'"
            assert "type" in spec, f"{key} missing 'type'"

    def test_min_less_than_max(self):
        """min must be strictly less than max."""
        for key, spec in CONFIG_SEARCH_SPACE.items():
            assert float(spec["min"]) < float(spec["max"]), (
                f"{key}: min ({spec['min']}) >= max ({spec['max']})"
            )

    def test_sigma_positive(self):
        """sigma must be positive."""
        for key, spec in CONFIG_SEARCH_SPACE.items():
            assert float(spec["sigma"]) > 0, f"{key}: sigma must be positive"

    def test_type_is_valid(self):
        """type must be 'float' or 'int'."""
        for key, spec in CONFIG_SEARCH_SPACE.items():
            assert spec["type"] in ("float", "int"), (
                f"{key}: type must be 'float' or 'int', got '{spec['type']}'"
            )

    def test_default_within_bounds(self):
        """DEFAULT_CONFIG value must be within [min, max] for each search space key."""
        for key, spec in CONFIG_SEARCH_SPACE.items():
            default_val = DEFAULT_CONFIG[key]
            if isinstance(default_val, (int, float)):
                assert float(spec["min"]) <= float(default_val) <= float(spec["max"]), (
                    f"{key}: default {default_val} outside [{spec['min']}, {spec['max']}]"
                )

    def test_int_params_have_integer_bounds(self):
        """Integer-typed params should have integer min/max."""
        for key, spec in CONFIG_SEARCH_SPACE.items():
            if spec["type"] == "int":
                assert float(spec["min"]) == int(spec["min"]), (
                    f"{key}: int type but non-integer min"
                )
                assert float(spec["max"]) == int(spec["max"]), (
                    f"{key}: int type but non-integer max"
                )


# ---------------------------------------------------------------------------
# _mutate_config tests
# ---------------------------------------------------------------------------


class TestMutateConfigBasic:
    """Basic behavior of _mutate_config."""

    def test_returns_dict(self):
        rng = random.Random(42)
        result = _mutate_config(DEFAULT_CONFIG, CONFIG_SEARCH_SPACE, rng)
        assert isinstance(result, dict)

    def test_returns_subset_of_search_space_keys(self):
        """Mutated keys must be a subset of search space keys."""
        rng = random.Random(42)
        result = _mutate_config(
            DEFAULT_CONFIG, CONFIG_SEARCH_SPACE, rng, mutation_rate=1.0
        )
        for key in result:
            assert key in CONFIG_SEARCH_SPACE, f"mutated key {key} not in search space"

    def test_mutation_rate_zero_returns_empty(self):
        """mutation_rate=0 means nothing gets mutated."""
        rng = random.Random(42)
        result = _mutate_config(
            DEFAULT_CONFIG, CONFIG_SEARCH_SPACE, rng, mutation_rate=0.0
        )
        assert result == {}

    def test_mutation_rate_one_mutates_all(self):
        """mutation_rate=1.0 mutates every eligible key."""
        rng = random.Random(42)
        result = _mutate_config(
            DEFAULT_CONFIG, CONFIG_SEARCH_SPACE, rng, mutation_rate=1.0,
            enable_board_size=True,
        )
        # Should have all keys mutated
        assert len(result) == len(CONFIG_SEARCH_SPACE)


class TestMutateConfigClamping:
    """Values are clamped to [min, max]."""

    def test_values_within_bounds(self):
        """All mutated values must be within [min, max]."""
        rng = random.Random(123)
        for _ in range(50):
            result = _mutate_config(
                DEFAULT_CONFIG, CONFIG_SEARCH_SPACE, rng, mutation_rate=1.0,
                enable_board_size=True,
            )
            for key, val in result.items():
                spec = CONFIG_SEARCH_SPACE[key]
                assert float(spec["min"]) <= float(val) <= float(spec["max"]), (
                    f"{key}: value {val} outside [{spec['min']}, {spec['max']}]"
                )

    def test_int_params_are_integers(self):
        """Integer-typed params produce integer values."""
        rng = random.Random(99)
        for _ in range(20):
            result = _mutate_config(
                DEFAULT_CONFIG, CONFIG_SEARCH_SPACE, rng, mutation_rate=1.0,
                enable_board_size=True,
            )
            for key, val in result.items():
                spec = CONFIG_SEARCH_SPACE[key]
                if spec["type"] == "int":
                    assert isinstance(val, int), f"{key}: expected int, got {type(val)}"


class TestSearchSpaceContents:
    """Regression guard: only true heuristics belong in the search space.

    Board dimensions, fab/circuit constraints (signal/power widths, vias,
    zone fab limits, pad inset, thermal radius) and FreeRouting operational
    budgets are NOT optimization knobs. They must stay out of
    CONFIG_SEARCH_SPACE so a parameter sweep can't game the score by
    enlarging boards or fattening traces.
    """

    FORBIDDEN_KEYS = {
        "board_width_mm",
        "board_height_mm",
        "signal_width_mm",
        "power_width_mm",
        "via_drill_mm",
        "via_size_mm",
        "freerouting_timeout_s",
        "freerouting_max_passes",
        "zone_clearance_mm",
        "zone_min_thickness_mm",
        "zone_thermal_gap_mm",
        "zone_thermal_spoke_mm",
        "pad_inset_margin_mm",
        "thermal_radius_mm",
    }

    def test_constraints_not_searchable(self):
        present = self.FORBIDDEN_KEYS & set(CONFIG_SEARCH_SPACE.keys())
        assert not present, (
            f"fab/derived constraints leaked into CONFIG_SEARCH_SPACE: "
            f"{sorted(present)}"
        )

    def test_board_size_excluded_even_when_enabled(self):
        """enable_board_size=True is now a no-op since board dims are not
        in the search space; mutate must not produce them either way."""
        rng = random.Random(42)
        result = _mutate_config(
            DEFAULT_CONFIG, CONFIG_SEARCH_SPACE, rng,
            mutation_rate=1.0, enable_board_size=True,
        )
        assert "board_width_mm" not in result
        assert "board_height_mm" not in result


class TestMutateConfigReproducibility:
    """Same seed produces same results."""

    def test_deterministic_with_same_seed(self):
        rng1 = random.Random(77)
        result1 = _mutate_config(
            DEFAULT_CONFIG, CONFIG_SEARCH_SPACE, rng1, mutation_rate=0.5
        )
        rng2 = random.Random(77)
        result2 = _mutate_config(
            DEFAULT_CONFIG, CONFIG_SEARCH_SPACE, rng2, mutation_rate=0.5
        )
        assert result1 == result2

    def test_different_seeds_produce_different_results(self):
        rng1 = random.Random(1)
        result1 = _mutate_config(
            DEFAULT_CONFIG, CONFIG_SEARCH_SPACE, rng1, mutation_rate=1.0,
            enable_board_size=True,
        )
        rng2 = random.Random(9999)
        result2 = _mutate_config(
            DEFAULT_CONFIG, CONFIG_SEARCH_SPACE, rng2, mutation_rate=1.0,
            enable_board_size=True,
        )
        assert result1 != result2


class TestMutateConfigCustomBase:
    """Works with a custom base config (non-default values)."""

    def test_mutates_from_custom_base(self):
        custom = dict(DEFAULT_CONFIG)
        custom["orderedness"] = 0.9  # near max
        rng = random.Random(42)
        result = _mutate_config(
            custom, CONFIG_SEARCH_SPACE, rng, mutation_rate=1.0,
            enable_board_size=True,
        )
        # orderedness should still be within bounds
        if "orderedness" in result:
            assert 0.0 <= result["orderedness"] <= 1.0

    def test_empty_search_space_returns_empty(self):
        """Empty search space means nothing to mutate."""
        rng = random.Random(42)
        result = _mutate_config(DEFAULT_CONFIG, {}, rng, mutation_rate=1.0)
        assert result == {}


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestJsonableConfigValue:
    """_jsonable_config_value handles various types."""

    def test_set_to_sorted_list(self):
        assert _jsonable_config_value({"b", "a", "c"}) == ["a", "b", "c"]

    def test_dict_recursive(self):
        result = _jsonable_config_value({"x": {1, 2}})
        assert result == {"x": [1, 2]}

    def test_list_recursive(self):
        result = _jsonable_config_value([{3, 1, 2}])
        assert result == [[1, 2, 3]]

    def test_scalar_passthrough(self):
        assert _jsonable_config_value(42) == 42
        assert _jsonable_config_value(3.14) == 3.14
        assert _jsonable_config_value("hello") == "hello"


class TestConfigOverlayFromDefaults:
    """_config_overlay_from_defaults extracts non-default values."""

    def test_all_defaults_returns_empty(self):
        overlay = _config_overlay_from_defaults(DEFAULT_CONFIG)
        assert overlay == {}

    def test_changed_value_included(self):
        cfg = dict(DEFAULT_CONFIG)
        cfg["orderedness"] = 0.99
        overlay = _config_overlay_from_defaults(cfg)
        assert "orderedness" in overlay
        assert overlay["orderedness"] == 0.99

    def test_new_key_included(self):
        cfg = dict(DEFAULT_CONFIG)
        cfg["new_key_xyz"] = 123
        overlay = _config_overlay_from_defaults(cfg)
        assert "new_key_xyz" in overlay
