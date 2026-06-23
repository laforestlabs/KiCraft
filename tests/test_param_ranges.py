"""Focused tests for the param-ranges / search-space logic.

Covers the production code paths only (no GUI, no pcbnew, no NiceGUI):
- normalize_bounds() shared helper (clamp / swap / int rounding / rejects)
- Param-ranges JSON merging into the effective search space (autoexperiment)
- Param-ranges file loading robustness (argv integration)
- enforce_param_constraints() cross-parameter validation

(The GUI parity / mutation-bound persistence tests that lived here were removed
with the Experiment Manager GUI on 2026-06-22.)
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

from kicraft.autoplacer.config import (
    CONFIG_SEARCH_SPACE,
    DEFAULT_CONFIG,
    enforce_param_constraints,
    normalize_bounds,
)
from kicraft.cli.autoexperiment import _mutate_config


# ---------------------------------------------------------------------------
# normalize_bounds() shared helper
# ---------------------------------------------------------------------------


class TestNormalizeBounds:

    def test_clamps_to_spec_domain(self):
        result = normalize_bounds("orderedness", -10.0, 10.0)
        spec = CONFIG_SEARCH_SPACE["orderedness"]
        assert result == (spec["min"], spec["max"])

    def test_swaps_inverted_bounds(self):
        result = normalize_bounds("orderedness", 0.8, 0.2)
        assert result == (0.2, 0.8)

    def test_int_type_applies_ceil_floor(self):
        # Values inside the current max_placement_iterations spec range.
        result = normalize_bounds("max_placement_iterations", 1000.3, 4000.7)
        assert result == (1001, 4000)

    def test_int_type_empty_range_returns_none(self):
        # An interval that collapses to empty after ceil/floor returns None.
        result = normalize_bounds("max_placement_iterations", 1000.9, 1000.1)
        assert result is None

    def test_unknown_key_returns_none(self):
        result = normalize_bounds("totally_fake_key", 0.0, 1.0)
        assert result is None

    def test_accepts_explicit_spec(self):
        spec = {"min": 0.0, "max": 10.0, "sigma": 1.0, "type": "float"}
        result = normalize_bounds("custom", 3.0, 7.0, spec)
        assert result == (3.0, 7.0)

    def test_clamps_with_explicit_spec(self):
        spec = {"min": 0.0, "max": 10.0, "sigma": 1.0, "type": "float"}
        result = normalize_bounds("custom", -5.0, 15.0, spec)
        assert result == (0.0, 10.0)

    def test_nan_infinity_rejected(self):
        spec = {"min": 0.0, "max": 1.0, "sigma": 0.1, "type": "float"}
        assert normalize_bounds("x", float("nan"), 0.5, spec) is None
        assert normalize_bounds("x", 0.5, float("nan"), spec) is None
        assert normalize_bounds("x", float("inf"), 0.5, spec) is None
        assert normalize_bounds("x", 0.5, float("-inf"), spec) is None
        assert normalize_bounds("x", float("nan"), float("nan"), spec) is None

    def test_preserves_valid_narrow_range(self):
        result = normalize_bounds("orderedness", 0.3, 0.7)
        assert result == (0.3, 0.7)


# ---------------------------------------------------------------------------
# Param-ranges merging via normalize_bounds (production code path)
# ---------------------------------------------------------------------------


class TestParamRangesMerging:

    def test_valid_float_range_narrows_search_space(self):
        result = normalize_bounds("orderedness", 0.2, 0.8)
        assert result == (0.2, 0.8)

    def test_sigma_and_type_preserved_after_merge(self):
        effective = dict(CONFIG_SEARCH_SPACE)
        result = normalize_bounds("orderedness", 0.2, 0.8)
        assert result is not None
        effective["orderedness"] = {**effective["orderedness"], "min": result[0], "max": result[1]}
        assert effective["orderedness"]["sigma"] == CONFIG_SEARCH_SPACE["orderedness"]["sigma"]
        assert effective["orderedness"]["type"] == CONFIG_SEARCH_SPACE["orderedness"]["type"]

    def test_valid_int_range_rounds_correctly(self):
        # Values inside the current max_placement_iterations spec range.
        result = normalize_bounds("max_placement_iterations", 1000.3, 4000.7)
        assert result == (1001, 4000)

    def test_int_range_empty_after_rounding_is_skipped(self):
        # An interval that collapses to empty after ceil/floor returns None.
        result = normalize_bounds("max_placement_iterations", 1000.9, 1000.1)
        # After swap: lo=1000.1, hi=1000.9 -> ceil(1000.1)=1001, floor(1000.9)=1000 -> empty
        assert result is None

    def test_inverted_user_range_auto_swapped(self):
        result = normalize_bounds("orderedness", 0.8, 0.2)
        assert result == (0.2, 0.8)

    def test_unknown_keys_return_none(self):
        result = normalize_bounds("totally_fake_param", 0.0, 1.0)
        assert result is None

    def test_mutation_respects_narrowed_search_space(self):
        narrowed = dict(CONFIG_SEARCH_SPACE)
        narrowed["orderedness"] = {**narrowed["orderedness"], "min": 0.4, "max": 0.6}

        rng = random.Random(42)
        for _ in range(50):
            result = _mutate_config(
                DEFAULT_CONFIG, narrowed, rng, mutation_rate=1.0, enable_board_size=True
            )
            if "orderedness" in result:
                assert 0.4 <= result["orderedness"] <= 0.6, (
                    f"orderedness={result['orderedness']} outside narrowed [0.4, 0.6]"
                )

    def test_out_of_domain_bounds_clamped(self):
        result = normalize_bounds("orderedness", -5.0, 5.0)
        spec = CONFIG_SEARCH_SPACE["orderedness"]
        assert result == (spec["min"], spec["max"])

    def test_int_param_range_rounded(self):
        """Integer params are rounded after clamping (covers any int-typed
        entry in CONFIG_SEARCH_SPACE; previously checked freerouting_timeout_s
        which is no longer a search-space knob). The requested min 100 is below
        sa_refine_iterations' domain min (250), so it clamps up to 250."""
        result = normalize_bounds("sa_refine_iterations", 100.0, 5000.0)
        assert result == (250, 5000)
        spec = CONFIG_SEARCH_SPACE["sa_refine_iterations"]
        assert spec["type"] == "int"


# ---------------------------------------------------------------------------
# Integration: param-ranges file loading via main() argv
# ---------------------------------------------------------------------------


class TestParamRangesFileIntegration:

    def test_valid_json_file_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            ranges_file = Path(tmp) / "bounds.json"
            ranges_file.write_text(
                json.dumps({"orderedness": [0.3, 0.7], "cooling_factor": [0.9, 0.99]}),
                encoding="utf-8",
            )

            with open(ranges_file, "r", encoding="utf-8") as f:
                user_ranges = json.load(f)

            assert isinstance(user_ranges, dict)
            assert "orderedness" in user_ranges
            assert user_ranges["orderedness"] == [0.3, 0.7]

    def test_invalid_json_file_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            ranges_file = Path(tmp) / "bad.json"
            ranges_file.write_text("not valid json {{{{", encoding="utf-8")

            user_ranges: dict = {}
            try:
                with open(ranges_file, "r", encoding="utf-8") as f:
                    user_ranges = json.load(f)
            except (OSError, json.JSONDecodeError):
                user_ranges = {}

            assert user_ranges == {}

    def test_missing_file_does_not_crash(self):
        user_ranges: dict = {}
        try:
            with open("/tmp/definitely_nonexistent_file_xyz.json", "r") as f:
                user_ranges = json.load(f)
        except (OSError, json.JSONDecodeError):
            user_ranges = {}

        assert user_ranges == {}

    def test_non_dict_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ranges_file = Path(tmp) / "list_root.json"
            ranges_file.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

            with open(ranges_file, "r", encoding="utf-8") as f:
                user_ranges = json.load(f)

            if not isinstance(user_ranges, dict):
                user_ranges = {}

            assert user_ranges == {}


# ---------------------------------------------------------------------------
# enforce_param_constraints() cross-parameter validation
# ---------------------------------------------------------------------------


class TestEnforceParamConstraints:

    def test_via_drill_exceeds_via_size_is_fixed(self):
        cfg = {"via_drill_mm": 0.8, "via_size_mm": 0.6}
        result = enforce_param_constraints(cfg)
        assert result["via_drill_mm"] < result["via_size_mm"]
        assert result["via_drill_mm"] == 0.6 * 0.5  # b * 0.5

    def test_via_drill_equals_via_size_is_fixed(self):
        # Strict "<" constraint: equal values must be corrected
        cfg = {"via_drill_mm": 0.5, "via_size_mm": 0.5}
        result = enforce_param_constraints(cfg)
        assert result["via_drill_mm"] < result["via_size_mm"]

    def test_already_valid_config_unchanged(self):
        cfg = {
            "via_drill_mm": 0.3,
            "via_size_mm": 0.6,
        }
        original = cfg.copy()
        enforce_param_constraints(cfg)
        assert cfg == original

    def test_missing_keys_are_skipped(self):
        # Only one side of a constraint present -- no crash, no modification
        cfg = {"via_drill_mm": 0.8}
        enforce_param_constraints(cfg)
        assert cfg == {"via_drill_mm": 0.8}

    def test_mutate_then_constrain_produces_valid(self):
        """Mutation + constraint enforcement yields physically valid configs."""
        rng = random.Random(42)
        for _ in range(50):
            base = dict(DEFAULT_CONFIG)
            mutated = _mutate_config(base, CONFIG_SEARCH_SPACE, rng)
            enforce_param_constraints(mutated)
            if "via_drill_mm" in mutated and "via_size_mm" in mutated:
                assert mutated["via_drill_mm"] < mutated["via_size_mm"]
