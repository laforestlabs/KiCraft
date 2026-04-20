"""Tests for autoexperiment scoring helpers -- area compactness and board dims.

All tests use synthetic/mock data only; no pcbnew dependency.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from kicraft.cli.autoexperiment import (
    _extract_parent_board_dimensions,
    _score_round,
)


class TestExtractParentBoardDimensions:
    """Tests for _extract_parent_board_dimensions() robustness."""

    def test_missing_file_returns_zero(self, tmp_path: Path):
        result = _extract_parent_board_dimensions(tmp_path / "nonexistent.json")
        assert result == (0.0, 0.0)

    def test_empty_json_returns_zero(self, tmp_path: Path):
        f = tmp_path / "empty.json"
        f.write_text("{}")
        assert _extract_parent_board_dimensions(f) == (0.0, 0.0)

    def test_missing_state_returns_zero(self, tmp_path: Path):
        f = tmp_path / "no_state.json"
        f.write_text(json.dumps({"other": "data"}))
        assert _extract_parent_board_dimensions(f) == (0.0, 0.0)

    def test_state_not_dict_returns_zero(self, tmp_path: Path):
        f = tmp_path / "state_list.json"
        f.write_text(json.dumps({"state": [1, 2, 3]}))
        assert _extract_parent_board_dimensions(f) == (0.0, 0.0)

    def test_missing_bounding_box_returns_zero(self, tmp_path: Path):
        f = tmp_path / "no_bbox.json"
        f.write_text(json.dumps({"state": {"foo": "bar"}}))
        assert _extract_parent_board_dimensions(f) == (0.0, 0.0)

    def test_bounding_box_not_dict_returns_zero(self, tmp_path: Path):
        f = tmp_path / "bbox_list.json"
        f.write_text(json.dumps({"state": {"bounding_box": "invalid"}}))
        assert _extract_parent_board_dimensions(f) == (0.0, 0.0)

    def test_non_numeric_width_returns_zero(self, tmp_path: Path):
        f = tmp_path / "bad_width.json"
        f.write_text(json.dumps({
            "state": {"bounding_box": {"width_mm": "bad", "height_mm": 10.0}}
        }))
        assert _extract_parent_board_dimensions(f) == (0.0, 0.0)

    def test_non_numeric_height_returns_zero(self, tmp_path: Path):
        f = tmp_path / "bad_height.json"
        f.write_text(json.dumps({
            "state": {"bounding_box": {"width_mm": 50.0, "height_mm": [1, 2]}}
        }))
        assert _extract_parent_board_dimensions(f) == (0.0, 0.0)

    def test_negative_values_clamped_to_zero(self, tmp_path: Path):
        f = tmp_path / "negative.json"
        f.write_text(json.dumps({
            "state": {"bounding_box": {"width_mm": -5.0, "height_mm": -3.0}}
        }))
        assert _extract_parent_board_dimensions(f) == (0.0, 0.0)

    def test_valid_dimensions_returned(self, tmp_path: Path):
        f = tmp_path / "valid.json"
        f.write_text(json.dumps({
            "state": {"bounding_box": {"width_mm": 80.0, "height_mm": 55.0}}
        }))
        assert _extract_parent_board_dimensions(f) == (80.0, 55.0)

    def test_null_values_fallback_to_zero(self, tmp_path: Path):
        f = tmp_path / "null_vals.json"
        f.write_text(json.dumps({
            "state": {"bounding_box": {"width_mm": None, "height_mm": None}}
        }))
        assert _extract_parent_board_dimensions(f) == (0.0, 0.0)

    def test_malformed_json_returns_zero(self, tmp_path: Path):
        f = tmp_path / "malformed.json"
        f.write_text("not json at all {{{")
        assert _extract_parent_board_dimensions(f) == (0.0, 0.0)


class TestScoreRoundAreaCompactness:
    """Tests for area_compactness_score integration in _score_round()."""

    def _make_leaf(self, w: float, h: float) -> dict:
        return {
            "trace_count": 5,
            "via_count": 2,
            "solved_layout": {
                "bounding_box": {"width_mm": w, "height_mm": h},
            },
        }

    def _call_score_round(
        self,
        parent_board_area_mm2: float = 0.0,
        child_total_area_mm2: float = 0.0,
    ) -> tuple:
        leafs = [self._make_leaf(10.0, 10.0)]
        return _score_round(
            accepted_leafs=leafs,
            all_leafs=leafs,
            composition_ok=True,
            parent_routed=True,
            parent_copper_accounting=None,
            baseline_score=None,
            recent_scores=[],
            plateau_count=0,
            parent_board_area_mm2=parent_board_area_mm2,
            child_total_area_mm2=child_total_area_mm2,
        )

    def test_zero_areas_yield_zero_compactness(self):
        score, breakdown, notes, context = self._call_score_round(0.0, 0.0)
        assert breakdown["absolute_area_compactness"] == 0.0
        assert breakdown["area_utilization_ratio"] == 0.0

    def test_zero_parent_area_yields_zero(self):
        score, breakdown, notes, context = self._call_score_round(0.0, 100.0)
        assert breakdown["absolute_area_compactness"] == 0.0

    def test_zero_child_area_yields_zero(self):
        score, breakdown, notes, context = self._call_score_round(200.0, 0.0)
        assert breakdown["absolute_area_compactness"] == 0.0

    def test_half_utilization(self):
        score, breakdown, notes, context = self._call_score_round(200.0, 100.0)
        assert breakdown["area_utilization_ratio"] == pytest.approx(0.5)
        assert breakdown["absolute_area_compactness"] == pytest.approx(4.5)

    def test_full_utilization_caps_at_nine(self):
        score, breakdown, notes, context = self._call_score_round(100.0, 100.0)
        assert breakdown["area_utilization_ratio"] == pytest.approx(1.0)
        assert breakdown["absolute_area_compactness"] == pytest.approx(9.0)

    def test_over_utilization_capped(self):
        score, breakdown, notes, context = self._call_score_round(50.0, 100.0)
        assert breakdown["area_utilization_ratio"] == pytest.approx(1.0)
        assert breakdown["absolute_area_compactness"] == pytest.approx(9.0)

    def test_area_compactness_contributes_to_total(self):
        score_no_area, _, _, _ = self._call_score_round(0.0, 0.0)
        score_with_area, _, _, _ = self._call_score_round(200.0, 100.0)
        assert score_with_area > score_no_area

    def test_area_in_score_context(self):
        _, _, _, context = self._call_score_round(200.0, 100.0)
        assert "area_compactness_score" in context
        assert context["area_compactness_score"] == pytest.approx(4.5)
        assert "area_utilization_ratio" in context
        assert context["area_utilization_ratio"] == pytest.approx(0.5)

    def test_area_in_score_notes(self):
        _, _, notes, _ = self._call_score_round(200.0, 100.0)
        note_text = "\n".join(notes)
        assert "area_utilization_ratio=0.500" in note_text
        assert "parent_board_area_mm2=200.0" in note_text
        assert "child_total_area_mm2=100.0" in note_text
        assert "area_compactness_score=4.500" in note_text


class TestSmallerParentBoardKept:
    """Integration: experiment keep/discard favors smaller parent boards."""

    def _make_leaf(self, w: float, h: float) -> dict:
        return {
            "trace_count": 5,
            "via_count": 2,
            "solved_layout": {
                "bounding_box": {"width_mm": w, "height_mm": h},
            },
        }

    def test_smaller_parent_scores_higher_than_larger(self):
        leafs = [self._make_leaf(10.0, 10.0), self._make_leaf(12.0, 8.0)]
        child_area = 10.0 * 10.0 + 12.0 * 8.0  # 196 mm2

        small_parent_area = 300.0  # 65% utilization
        large_parent_area = 900.0  # 22% utilization

        score_small, bd_small, _, _ = _score_round(
            accepted_leafs=leafs,
            all_leafs=leafs,
            composition_ok=True,
            parent_routed=True,
            parent_copper_accounting=None,
            baseline_score=None,
            recent_scores=[],
            plateau_count=0,
            parent_board_area_mm2=small_parent_area,
            child_total_area_mm2=child_area,
        )
        score_large, bd_large, _, _ = _score_round(
            accepted_leafs=leafs,
            all_leafs=leafs,
            composition_ok=True,
            parent_routed=True,
            parent_copper_accounting=None,
            baseline_score=None,
            recent_scores=[],
            plateau_count=0,
            parent_board_area_mm2=large_parent_area,
            child_total_area_mm2=child_area,
        )

        assert score_small > score_large, (
            f"Smaller parent ({small_parent_area} mm2) should score higher "
            f"than larger ({large_parent_area} mm2): {score_small} vs {score_large}"
        )
        assert bd_small["absolute_area_compactness"] > bd_large["absolute_area_compactness"]

    def test_spacing_affects_board_size_in_config(self):
        from kicraft.autoplacer.config import CONFIG_SEARCH_SPACE, DEFAULT_CONFIG

        assert "parent_spacing_mm" in CONFIG_SEARCH_SPACE
        assert "parent_spacing_mm" in DEFAULT_CONFIG
        spec = CONFIG_SEARCH_SPACE["parent_spacing_mm"]
        assert spec["min"] == 0.5
        assert spec["max"] == 6.0
        assert DEFAULT_CONFIG["parent_spacing_mm"] == 2.0

    def test_build_compose_cmd_uses_spacing(self):
        from kicraft.cli.autoexperiment import _build_compose_cmd

        cmd = _build_compose_cmd(
            project_dir=Path("/tmp/proj"),
            parent="top",
            output_json=Path("/tmp/out.json"),
            only=[],
            spacing_mm=3.5,
        )
        idx = cmd.index("--spacing-mm")
        assert cmd[idx + 1] == "3.5"

    def test_build_compose_cmd_default_spacing(self):
        from kicraft.cli.autoexperiment import _build_compose_cmd

        cmd = _build_compose_cmd(
            project_dir=Path("/tmp/proj"),
            parent="top",
            output_json=Path("/tmp/out.json"),
            only=[],
        )
        idx = cmd.index("--spacing-mm")
        assert cmd[idx + 1] == "2.0"

    def test_keep_discard_gate_favors_compact_parent(self):
        """Simulate the actual keep/discard decision from autoexperiment.

        Two rounds with identical leafs but different parent board areas.
        The round with the smaller parent must produce a higher score that
        crosses the keep_threshold relative to the larger-parent baseline.
        """
        leafs = [self._make_leaf(10.0, 10.0), self._make_leaf(12.0, 8.0)]
        child_area = 10.0 * 10.0 + 12.0 * 8.0

        large_parent_area = 900.0
        small_parent_area = 300.0

        score_large, _, _, _ = _score_round(
            accepted_leafs=leafs,
            all_leafs=leafs,
            composition_ok=True,
            parent_routed=True,
            parent_copper_accounting=None,
            baseline_score=None,
            recent_scores=[],
            plateau_count=0,
            parent_board_area_mm2=large_parent_area,
            child_total_area_mm2=child_area,
        )

        score_small, _, _, _ = _score_round(
            accepted_leafs=leafs,
            all_leafs=leafs,
            composition_ok=True,
            parent_routed=True,
            parent_copper_accounting=None,
            baseline_score=score_large,
            recent_scores=[score_large],
            plateau_count=0,
            parent_board_area_mm2=small_parent_area,
            child_total_area_mm2=child_area,
        )

        keep_threshold = 0.5
        improvement_vs_best = score_small - score_large
        is_meaningful_improvement = improvement_vs_best >= keep_threshold

        assert is_meaningful_improvement, (
            f"Compact parent (area={small_parent_area}) should be kept over "
            f"large parent (area={large_parent_area}): "
            f"improvement={improvement_vs_best:.3f}, threshold={keep_threshold}"
        )
        assert improvement_vs_best > 0, (
            f"Score with small parent ({score_small:.3f}) must exceed "
            f"score with large parent ({score_large:.3f})"
        )
