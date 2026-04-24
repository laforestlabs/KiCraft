"""Tests for autoexperiment scoring helpers.

All tests use synthetic/mock data only; no pcbnew dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.cli.autoexperiment import (
    _extract_parent_board_dimensions,
    _read_composer_quality_score,
    _score_round,
)


def _write_parent_pipeline(
    tmp_path: Path,
    score_total: float,
    breakdown: dict[str, float] | None = None,
) -> Path:
    payload = {
        "state": {
            "score_total": score_total,
            "score_breakdown": dict(breakdown or {}),
            "bounding_box": {"width_mm": 80.0, "height_mm": 55.0},
        }
    }
    path = tmp_path / "parent_pipeline.json"
    path.write_text(json.dumps(payload))
    return path


class TestReadComposerQualityScore:
    def test_reads_total_and_breakdown(self, tmp_path: Path):
        path = _write_parent_pipeline(
            tmp_path,
            72.5,
            {"anchor_coverage": 100.0, "area_utilization": 55.5},
        )
        total, breakdown = _read_composer_quality_score(path)
        assert total == pytest.approx(72.5)
        assert breakdown["anchor_coverage"] == pytest.approx(100.0)
        assert breakdown["area_utilization"] == pytest.approx(55.5)

    def test_missing_file_returns_zero(self, tmp_path: Path):
        total, breakdown = _read_composer_quality_score(tmp_path / "missing.json")
        assert total == 0.0
        assert breakdown == {}

    def test_malformed_returns_zero(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("garbage {{")
        total, breakdown = _read_composer_quality_score(path)
        assert total == 0.0
        assert breakdown == {}

    def test_clamps_to_0_100(self, tmp_path: Path):
        path = _write_parent_pipeline(tmp_path, 145.0)
        total, _ = _read_composer_quality_score(path)
        assert total == 100.0
        path = _write_parent_pipeline(tmp_path, -22.0)
        total, _ = _read_composer_quality_score(path)
        assert total == 0.0


class TestScoreRoundTiers:
    """Three-tier gating: partial_leaves, not_routed, functional."""

    def test_functional_tier_uses_composer_score(self, tmp_path: Path):
        path = _write_parent_pipeline(
            tmp_path, 68.5, {"area_utilization": 44.0, "child_layout_quality": 81.0}
        )
        score, breakdown, notes, tier = _score_round(
            leaf_accepted=6,
            leaf_total=6,
            parent_routed=True,
            parent_output_json=path,
        )
        assert tier == "functional"
        assert score == pytest.approx(68.5)
        assert breakdown["composer_score_total"] == pytest.approx(68.5)
        assert breakdown["area_utilization"] == pytest.approx(44.0)
        assert "tier=functional" in notes

    def test_not_routed_tier_fixed_20(self, tmp_path: Path):
        # Even if composer reports a high score, a failed route is 20.
        path = _write_parent_pipeline(tmp_path, 99.0)
        score, breakdown, _, tier = _score_round(
            leaf_accepted=6,
            leaf_total=6,
            parent_routed=False,
            parent_output_json=path,
        )
        assert tier == "not_routed"
        assert score == 20.0
        assert "not_routed_penalty" in breakdown

    def test_partial_leaves_tier_proportional(self, tmp_path: Path):
        path = _write_parent_pipeline(tmp_path, 75.0)
        score, breakdown, _, tier = _score_round(
            leaf_accepted=4,
            leaf_total=6,
            parent_routed=False,
            parent_output_json=path,
        )
        assert tier == "partial_leaves"
        # 4/6 * 15 = 10
        assert score == pytest.approx(10.0)
        assert breakdown["leaf_partial_credit"] == pytest.approx(10.0)

    def test_partial_leaves_dominates_route_status(self, tmp_path: Path):
        # If leaves are partial, it doesn't matter if routing succeeded.
        path = _write_parent_pipeline(tmp_path, 80.0)
        score_routed, _, _, tier_routed = _score_round(
            leaf_accepted=3, leaf_total=6, parent_routed=True, parent_output_json=path
        )
        score_unrouted, _, _, tier_unrouted = _score_round(
            leaf_accepted=3, leaf_total=6, parent_routed=False, parent_output_json=path
        )
        assert tier_routed == tier_unrouted == "partial_leaves"
        assert score_routed == score_unrouted == pytest.approx(7.5)

    def test_functional_has_real_headroom(self, tmp_path: Path):
        # Observed composer scores across 6 rounds were 66-75. A mediocre
        # layout should score in that range, NOT near 100 -- the user should
        # see room to improve.
        path = _write_parent_pipeline(tmp_path, 67.0)
        score_mediocre, _, _, _ = _score_round(
            leaf_accepted=6, leaf_total=6, parent_routed=True, parent_output_json=path
        )
        assert 50.0 <= score_mediocre <= 80.0, (
            f"mediocre functional board scored {score_mediocre}; "
            f"should sit mid-range to show improvement is possible"
        )

        path = _write_parent_pipeline(tmp_path, 95.0)
        score_great, _, _, _ = _score_round(
            leaf_accepted=6, leaf_total=6, parent_routed=True, parent_output_json=path
        )
        assert score_great > score_mediocre + 20

    def test_gap_between_routed_and_unrouted(self, tmp_path: Path):
        path = _write_parent_pipeline(tmp_path, 70.0)
        score_routed, _, _, _ = _score_round(
            leaf_accepted=6, leaf_total=6, parent_routed=True, parent_output_json=path
        )
        score_unrouted, _, _, _ = _score_round(
            leaf_accepted=6, leaf_total=6, parent_routed=False, parent_output_json=path
        )
        # A non-functional board must score much lower than a functional one.
        assert score_routed - score_unrouted >= 40

    def test_no_leaves_scores_zero(self, tmp_path: Path):
        path = _write_parent_pipeline(tmp_path, 80.0)
        score, _, _, tier = _score_round(
            leaf_accepted=0, leaf_total=0, parent_routed=False, parent_output_json=path
        )
        assert score == 0.0
        assert tier == "partial_leaves"

    def test_score_does_not_depend_on_recent_history(self, tmp_path: Path):
        # Purely absolute -- no rolling/recent/baseline inflation anymore.
        # Calling twice must always produce the same value for the same inputs.
        path = _write_parent_pipeline(tmp_path, 70.0)
        a = _score_round(
            leaf_accepted=6, leaf_total=6, parent_routed=True, parent_output_json=path
        )[0]
        b = _score_round(
            leaf_accepted=6, leaf_total=6, parent_routed=True, parent_output_json=path
        )[0]
        assert a == b


class TestExtractParentBoardDimensions:
    """Tests for _extract_parent_board_dimensions() robustness.

    Retained because the helper is still used for diagnostics in the live
    status writer and the GUI expects bounding_box parity on the pipeline
    JSON.
    """

    def test_missing_file_returns_zero(self, tmp_path: Path):
        result = _extract_parent_board_dimensions(tmp_path / "nonexistent.json")
        assert result == (0.0, 0.0)

    def test_valid_dimensions_returned(self, tmp_path: Path):
        f = tmp_path / "valid.json"
        f.write_text(json.dumps({
            "state": {"bounding_box": {"width_mm": 80.0, "height_mm": 55.0}}
        }))
        assert _extract_parent_board_dimensions(f) == (80.0, 55.0)

    def test_malformed_json_returns_zero(self, tmp_path: Path):
        f = tmp_path / "malformed.json"
        f.write_text("not json at all {{{")
        assert _extract_parent_board_dimensions(f) == (0.0, 0.0)


class TestKeepDiscardUsesComposerQuality:
    """Smaller (more-compact) parents should score higher through the composer."""

    def test_higher_composer_score_drives_kept_decision(self, tmp_path: Path):
        (tmp_path / "a").mkdir(exist_ok=True)
        (tmp_path / "b").mkdir(exist_ok=True)
        path_low = tmp_path / "a" / "parent_pipeline.json"
        path_low.write_text(json.dumps({
            "state": {"score_total": 60.0, "score_breakdown": {}}
        }))
        path_high = tmp_path / "b" / "parent_pipeline.json"
        path_high.write_text(json.dumps({
            "state": {"score_total": 72.0, "score_breakdown": {}}
        }))

        score_low, _, _, _ = _score_round(
            leaf_accepted=6, leaf_total=6, parent_routed=True, parent_output_json=path_low
        )
        score_high, _, _, _ = _score_round(
            leaf_accepted=6, leaf_total=6, parent_routed=True, parent_output_json=path_high
        )
        assert score_high - score_low >= 0.5  # crosses keep threshold
