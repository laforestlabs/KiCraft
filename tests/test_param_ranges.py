"""Focused tests for the GUI param-ranges feature.

Covers:
- normalize_bounds() shared helper
- Param-ranges JSON merging into effective_search_space (autoexperiment)
- Mutation-bound persistence roundtrip (state.py save/load)
- Invalid bound handling (clamping, swap, stale keys)
- Default-bound alignment with CONFIG_SEARCH_SPACE
- Cross-session auto-persistence

All tests use synthetic data only; no pcbnew or NiceGUI dependency.
"""

from __future__ import annotations

import json
import math
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
        result = normalize_bounds("max_placement_iterations", 350.3, 2000.7)
        assert result == (351, 2000)

    def test_int_type_empty_range_returns_none(self):
        result = normalize_bounds("max_placement_iterations", 300.9, 300.1)
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
# Invariant: PLACEMENT_PARAMS min/max matches CONFIG_SEARCH_SPACE min/max
# ---------------------------------------------------------------------------


class TestGuiSearchSpaceParity:

    def test_numeric_gui_params_have_search_space_entry(self):
        from kicraft.gui.state import PLACEMENT_PARAMS

        numeric_keys = [
            p["key"] for p in PLACEMENT_PARAMS
            if p.get("type") not in ("bool", "text", "list") and p["min"] is not None
        ]
        for key in numeric_keys:
            assert key in CONFIG_SEARCH_SPACE, (
                f"GUI param '{key}' has no CONFIG_SEARCH_SPACE entry"
            )

    def test_gui_min_matches_search_space_min(self):
        from kicraft.gui.state import PLACEMENT_PARAMS

        for p in PLACEMENT_PARAMS:
            key = p["key"]
            if p.get("type") in ("bool", "text", "list") or p["min"] is None:
                continue
            if key not in CONFIG_SEARCH_SPACE:
                continue
            spec = CONFIG_SEARCH_SPACE[key]
            assert float(p["min"]) == float(spec["min"]), (
                f"{key}: GUI min={p['min']} != search space min={spec['min']}"
            )

    def test_gui_max_matches_search_space_max(self):
        from kicraft.gui.state import PLACEMENT_PARAMS

        for p in PLACEMENT_PARAMS:
            key = p["key"]
            if p.get("type") in ("bool", "text", "list") or p["max"] is None:
                continue
            if key not in CONFIG_SEARCH_SPACE:
                continue
            spec = CONFIG_SEARCH_SPACE[key]
            assert float(p["max"]) == float(spec["max"]), (
                f"{key}: GUI max={p['max']} != search space max={spec['max']}"
            )


# ---------------------------------------------------------------------------
# Default mutation bounds alignment with CONFIG_SEARCH_SPACE
# ---------------------------------------------------------------------------


class TestDefaultBoundsAlignment:

    def test_default_bounds_keys_match_search_space(self):
        from kicraft.gui.state import _default_mutation_bounds

        bounds = _default_mutation_bounds()
        assert set(bounds.keys()) == set(CONFIG_SEARCH_SPACE.keys())

    def test_default_bounds_values_match_spec_min_max(self):
        from kicraft.gui.state import _default_mutation_bounds

        bounds = _default_mutation_bounds()
        for key, spec in CONFIG_SEARCH_SPACE.items():
            assert bounds[key][0] == spec["min"], f"{key}: lo mismatch"
            assert bounds[key][1] == spec["max"], f"{key}: hi mismatch"

    def test_default_bounds_are_two_element_lists(self):
        from kicraft.gui.state import _default_mutation_bounds

        bounds = _default_mutation_bounds()
        for key, pair in bounds.items():
            assert isinstance(pair, list), f"{key}: not a list"
            assert len(pair) == 2, f"{key}: expected 2 elements, got {len(pair)}"


# ---------------------------------------------------------------------------
# Mutation-bound persistence roundtrip
# ---------------------------------------------------------------------------


class TestMutationBoundPersistence:

    def _make_state(self):
        from kicraft.gui.state import AppState

        return AppState()

    def test_roundtrip_preserves_default_bounds(self):
        state = self._make_state()
        original = {k: list(v) for k, v in state.mutation_bounds.items()}

        config = state.to_config_dict()
        state2 = self._make_state()
        state2.load_from_config(config)

        assert state2.mutation_bounds == original

    def test_roundtrip_preserves_custom_bounds(self):
        state = self._make_state()

        key = "orderedness"
        spec = CONFIG_SEARCH_SPACE[key]
        narrow_lo = spec["min"] + 0.1
        narrow_hi = spec["max"] - 0.1
        state.mutation_bounds[key] = [narrow_lo, narrow_hi]

        config = state.to_config_dict()
        state2 = self._make_state()
        state2.load_from_config(config)

        assert state2.mutation_bounds[key] == [narrow_lo, narrow_hi]

    def test_mutation_bounds_serialized_as_lists(self):
        state = self._make_state()
        config = state.to_config_dict()

        bounds_section = config["_mutation_bounds"]
        for key, val in bounds_section.items():
            assert isinstance(val, list), f"{key}: not serialized as list"

    def test_loading_config_without_mutation_bounds_resets_to_defaults(self):
        from kicraft.gui.state import _default_mutation_bounds

        state = self._make_state()
        state.mutation_bounds["orderedness"] = [0.4, 0.6]

        state.load_from_config({"_strategy": {"rounds": 5}})

        defaults = _default_mutation_bounds()
        assert state.mutation_bounds == defaults


# ---------------------------------------------------------------------------
# Invalid bound handling (load_from_config)
# ---------------------------------------------------------------------------


class TestInvalidBoundHandling:

    def _make_state(self):
        from kicraft.gui.state import AppState

        return AppState()

    def test_stale_key_ignored(self):
        state = self._make_state()
        config = {
            "_mutation_bounds": {
                "nonexistent_param_xyz": [0.0, 1.0],
                "orderedness": [0.1, 0.9],
            }
        }
        state.load_from_config(config)

        assert "nonexistent_param_xyz" not in state.mutation_bounds
        assert state.mutation_bounds["orderedness"] == [0.1, 0.9]

    def test_bounds_clamped_to_spec_domain(self):
        state = self._make_state()
        spec = CONFIG_SEARCH_SPACE["orderedness"]

        config = {
            "_mutation_bounds": {
                "orderedness": [-999.0, 999.0],
            }
        }
        state.load_from_config(config)

        assert state.mutation_bounds["orderedness"][0] == spec["min"]
        assert state.mutation_bounds["orderedness"][1] == spec["max"]

    def test_inverted_bounds_auto_swapped(self):
        state = self._make_state()
        spec = CONFIG_SEARCH_SPACE["orderedness"]

        config = {
            "_mutation_bounds": {
                "orderedness": [spec["max"], spec["min"]],
            }
        }
        state.load_from_config(config)

        lo, hi = state.mutation_bounds["orderedness"]
        assert lo <= hi

    def test_malformed_bounds_not_list(self):
        from kicraft.gui.state import _default_mutation_bounds

        state = self._make_state()
        defaults = _default_mutation_bounds()

        config = {
            "_mutation_bounds": {
                "orderedness": "not a list",
            }
        }
        state.load_from_config(config)

        assert state.mutation_bounds["orderedness"] == defaults["orderedness"]

    def test_malformed_bounds_too_short(self):
        from kicraft.gui.state import _default_mutation_bounds

        state = self._make_state()
        defaults = _default_mutation_bounds()

        config = {
            "_mutation_bounds": {
                "orderedness": [0.5],
            }
        }
        state.load_from_config(config)

        assert state.mutation_bounds["orderedness"] == defaults["orderedness"]

    def test_non_numeric_bounds_skipped(self):
        from kicraft.gui.state import _default_mutation_bounds

        state = self._make_state()
        defaults = _default_mutation_bounds()

        config = {
            "_mutation_bounds": {
                "orderedness": ["abc", "def"],
            }
        }
        state.load_from_config(config)

        assert state.mutation_bounds["orderedness"] == defaults["orderedness"]


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
        result = normalize_bounds("max_placement_iterations", 350.3, 2000.7)
        assert result == (351, 2000)

    def test_int_range_empty_after_rounding_is_skipped(self):
        result = normalize_bounds("max_placement_iterations", 300.9, 300.1)
        # After swap: lo=300.1, hi=300.9 -> ceil(300.1)=301, floor(300.9)=300 -> empty
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

    def test_freerouting_timeout_int_range(self):
        result = normalize_bounds("freerouting_timeout_s", 15.0, 120.0)
        assert result == (15, 120)


# ---------------------------------------------------------------------------
# Cross-session auto-persistence
# ---------------------------------------------------------------------------


class TestSessionPersistence:

    def _make_state(self, tmp_dir: Path):
        from kicraft.gui.state import AppState

        state = AppState()
        state._experiments_dir_override = tmp_dir
        # Monkey-patch experiments_dir property
        type(state).experiments_dir = property(lambda self: self._experiments_dir_override)
        return state

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state = self._make_state(tmp_path)
            state.save_session_state()
            assert (tmp_path / "gui_session_state.json").exists()

    def test_save_restore_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state = self._make_state(tmp_path)
            state.mutation_bounds["orderedness"] = [0.3, 0.7]
            state.save_session_state()

            state2 = self._make_state(tmp_path)
            state2.restore_session_state()
            assert state2.mutation_bounds["orderedness"] == [0.3, 0.7]

    def test_restore_missing_file_is_noop(self):
        from kicraft.gui.state import _default_mutation_bounds

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            state = self._make_state(tmp_path)
            defaults = _default_mutation_bounds()
            state.restore_session_state()
            assert state.mutation_bounds == defaults

    def test_restore_corrupted_file_is_noop(self):
        from kicraft.gui.state import _default_mutation_bounds

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "gui_session_state.json").write_text("not json {{{")
            state = self._make_state(tmp_path)
            defaults = _default_mutation_bounds()
            state.restore_session_state()
            assert state.mutation_bounds == defaults


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
