"""Tests for autoexperiment scoring helpers.

All tests use synthetic/mock data only; no pcbnew dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.cli.autoexperiment import (
    HierarchyRound,
    _accepted_leaf_artifacts,
    _all_leaf_artifacts,
    _leaf_progress_counts,
    _planned_leaf_paths,
    _write_live_status,
    _write_round_detail,
    _extract_parent_board_dimensions,
    _read_composer_quality_score,
    _score_round,
)


def _write_parent_pipeline(
    tmp_path: Path,
    score_total: float,
    breakdown: dict[str, float] | None = None,
    stamp_drc: dict[str, int] | None = None,
) -> Path:
    state: dict = {
        "score_total": score_total,
        "score_breakdown": dict(breakdown or {}),
        "bounding_box": {"width_mm": 80.0, "height_mm": 55.0},
    }
    if stamp_drc is not None:
        state["stamp_drc"] = dict(stamp_drc)
    payload = {"state": state}
    path = tmp_path / "parent_pipeline.json"
    path.write_text(json.dumps(payload))
    return path

@pytest.mark.parametrize("failed_artifact", [False, True])
def test_planned_leaf_progress_includes_missing_or_failed_instance(
    tmp_path: Path, failed_artifact: bool
):
    root = tmp_path / "root.kicad_sch"
    sheets = "".join(
        f' (sheet (uuid "{i}") (property "Sheetname" "Leaf{i}")'
        ' (property "Sheetfile" "child.kicad_sch"))'
        for i in range(5)
    )
    root.write_text(f'(kicad_sch (uuid "root"){sheets})')
    (tmp_path / "child.kicad_sch").write_text('(kicad_sch (uuid "child"))')
    paths = _planned_leaf_paths(root)
    assert paths == {f"/{i}" for i in range(5)}

    artifacts = tmp_path / ".experiments" / "subcircuits"
    for i in range(5 if failed_artifact else 4):
        directory = artifacts / f"leaf{i}"
        directory.mkdir(parents=True)
        directory.joinpath("metadata.json").write_text("{}")
        directory.joinpath("solved_layout.json").write_text(json.dumps({
            "sheet_name": f"Leaf{i}",
            "instance_path": f"/{i}",
            "validation": {"leaf_acceptance_result": {"accepted": i < 4}},
        }))
    all_leafs = _all_leaf_artifacts(tmp_path)
    accepted = _accepted_leaf_artifacts(tmp_path)
    total, count = _leaf_progress_counts(paths, accepted)
    assert (total, count) == (5, 4)

    score, _, _, tier = _score_round(
        leaf_accepted=count, leaf_total=total, parent_routed=False,
        parent_output_json=tmp_path / "missing.json",
    )
    assert tier == "partial_leaves"
    assert score == pytest.approx(12.0)
    status = tmp_path / "status.json"
    _write_live_status(
        status, tmp_path / "status.txt", phase="complete", rounds_total=1,
        round_num=1, best_score=0, kept_count=0, latest_score=score,
        latest_marker=None, start_ts=0, current_stage="done",
        leaf_total=total, leaf_accepted=count, parent_routed=False,
    )
    assert json.loads(status.read_text())["hierarchy"]["leaf_total"] == 5
    assert json.loads(status.read_text())["hierarchy"]["leaf_accepted"] == 4

    result = HierarchyRound(
        round_num=1, seed=1, mode="hierarchical", score=score, kept=False,
        duration_s=0, leaf_total=total, leaf_accepted=count, tier=tier,
    )
    _write_round_detail(
        tmp_path, result, accepted_leafs=accepted, all_leafs=all_leafs,
        solve_exit_code=6, parent_route_exit_code=None, solve_stdout="",
        solve_stderr="failed leaf", parent_route_stdout="", parent_route_stderr="",
    )
    detail = json.loads((tmp_path / "round_0001.json").read_text())
    assert detail["hierarchy"]["leaf_total"] == 5
    assert detail["hierarchy"]["leaf_accepted"] == 4
    assert len(detail["all_leaf_artifacts"]) == (5 if failed_artifact else 4)
    assert len(detail["accepted_leaf_artifacts"]) == 4


def test_planned_leaf_progress_scopes_selected_parent_descendants(tmp_path: Path):
    root = tmp_path / "root.kicad_sch"
    root.write_text(
        '(kicad_sch (uuid "root")'
        ' (sheet (uuid "a") (property "Sheetname" "A")'
        ' (property "Sheetfile" "branch.kicad_sch"))'
        ' (sheet (uuid "b") (property "Sheetname" "B")'
        ' (property "Sheetfile" "branch.kicad_sch")))'
    )
    (tmp_path / "branch.kicad_sch").write_text(
        '(kicad_sch (uuid "branch")'
        ' (sheet (uuid "one") (property "Sheetname" "One")'
        ' (property "Sheetfile" "leaf.kicad_sch"))'
        ' (sheet (uuid "two") (property "Sheetname" "Two")'
        ' (property "Sheetfile" "leaf.kicad_sch")))'
    )
    (tmp_path / "leaf.kicad_sch").write_text('(kicad_sch (uuid "leaf"))')
    selected = _planned_leaf_paths(root, "a")
    assert selected == {"/a/one", "/a/two"}
    assert _planned_leaf_paths(root, "/") == {
        "/a/one", "/a/two", "/b/one", "/b/two",
    }
    accepted = [{"instance_path": "/a/one"}, {"instance_path": "/b/one"}]
    assert _leaf_progress_counts(selected, accepted) == (2, 1)


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

    def test_not_routed_tier_clean_stamp_is_20(self, tmp_path: Path):
        # No stamp-time violations: best-case not_routed score = 20.
        path = _write_parent_pipeline(tmp_path, 99.0)
        score, breakdown, _, tier = _score_round(
            leaf_accepted=6,
            leaf_total=6,
            parent_routed=False,
            parent_output_json=path,
        )
        assert tier == "not_routed"
        assert score == 20.0
        assert breakdown["not_routed_base"] == 20.0
        assert breakdown["stamp_short_penalty"] == 0.0
        assert breakdown["stamp_clearance_penalty"] == 0.0

    def test_not_routed_tier_grades_by_stamp_drc(self, tmp_path: Path):
        # Pre-route violations drop the score below 20 so the search has
        # a gradient to follow toward fewer overlapping leaves.
        path = _write_parent_pipeline(
            tmp_path,
            99.0,
            stamp_drc={"shorts": 4, "clearance": 10},
        )
        score, breakdown, notes, tier = _score_round(
            leaf_accepted=6,
            leaf_total=6,
            parent_routed=False,
            parent_output_json=path,
        )
        assert tier == "not_routed"
        # 20 - (4 * 0.5) - (10 * 0.1) = 20 - 2 - 1 = 17
        assert score == pytest.approx(17.0)
        assert breakdown["stamp_short_penalty"] == pytest.approx(-2.0)
        assert breakdown["stamp_clearance_penalty"] == pytest.approx(-1.0)
        assert any("stamp_shorts=4" in n for n in notes)

    def test_not_routed_tier_penalty_caps(self, tmp_path: Path):
        # Penalties are capped so the score floors at 0, not negative.
        path = _write_parent_pipeline(
            tmp_path,
            99.0,
            stamp_drc={"shorts": 1000, "clearance": 1000},
        )
        score, breakdown, _, tier = _score_round(
            leaf_accepted=6,
            leaf_total=6,
            parent_routed=False,
            parent_output_json=path,
        )
        assert tier == "not_routed"
        # short_penalty capped at 15, clearance_penalty capped at 5
        # → 20 - 15 - 5 = 0
        assert score == 0.0
        assert breakdown["stamp_short_penalty"] == -15.0
        assert breakdown["stamp_clearance_penalty"] == -5.0

    def test_routed_dirty_tier_scores_above_not_routed(self, tmp_path: Path):
        # Electrically complete (traces, 0 shorts, 0 unconnected) but rejected on
        # soft DRC -> routed_dirty tier, scored well above the not_routed cap (20)
        # so the optimizer keeps a 99%-routed parent instead of discarding it.
        path = _write_parent_pipeline(tmp_path, 80.0)
        rv = {
            "track_summary": {"traces": 286},
            "drc": {"shorts": 0, "unconnected": 0, "total": 50},
        }
        score, breakdown, notes, tier = _score_round(
            leaf_accepted=6,
            leaf_total=6,
            parent_routed=False,
            parent_output_json=path,
            parent_routed_validation=rv,
        )
        assert tier == "routed_dirty"
        assert score > 20.0
        # 40 + 0.20*80 - min(15, 50*0.2=10) = 40 + 16 - 10 = 46
        assert score == pytest.approx(46.0)
        assert any("tier=routed_dirty" in n for n in notes)

    def test_routed_dirty_grades_by_remaining_drc(self, tmp_path: Path):
        # Fewer post-route DRC violations -> higher score: a gradient to descend
        # toward a clean board (the whole point -- no flat cliff).
        path = _write_parent_pipeline(tmp_path, 80.0)

        def rv(total):
            return {
                "track_summary": {"traces": 286},
                "drc": {"shorts": 0, "unconnected": 0, "total": total},
            }

        dirty, *_ = _score_round(
            leaf_accepted=6,
            leaf_total=6,
            parent_routed=False,
            parent_output_json=path,
            parent_routed_validation=rv(60),
        )
        cleaner, *_ = _score_round(
            leaf_accepted=6,
            leaf_total=6,
            parent_routed=False,
            parent_output_json=path,
            parent_routed_validation=rv(5),
        )
        assert cleaner > dirty

    def test_routed_with_unconnected_is_not_routed_dirty(self, tmp_path: Path):
        # Not electrically complete (unconnected nets) stays not_routed --
        # routed_dirty is reserved for a fully-connected parent.
        path = _write_parent_pipeline(tmp_path, 80.0)
        rv = {
            "track_summary": {"traces": 200},
            "drc": {"shorts": 0, "unconnected": 3, "total": 10},
        }
        _, _, _, tier = _score_round(
            leaf_accepted=6,
            leaf_total=6,
            parent_routed=False,
            parent_output_json=path,
            parent_routed_validation=rv,
        )
        assert tier == "not_routed"

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


class TestPromotedButRejectedParent:
    """A compose that exits 0 with validation accepted=False (promoted
    strand/courtyard board) must NOT score as functional (2026-07-19 §2.2)."""

    def _rejected_validation(self, reasons: list[str]) -> dict:
        return {
            "accepted": False,
            "rejection_reasons": reasons,
            "drc": {"shorts": 0, "unconnected": 0, "total": 3},
            "track_summary": {"traces": 120},
        }

    def test_rejected_parent_scores_routed_dirty(self, tmp_path: Path):
        pipeline = _write_parent_pipeline(tmp_path, score_total=85.0)
        score, breakdown, notes, tier = _score_round(
            leaf_accepted=3,
            leaf_total=3,
            parent_routed=True,
            parent_output_json=pipeline,
            parent_routed_validation=self._rejected_validation(
                ["courtyards_overlap"]
            ),
        )
        assert tier == "routed_dirty"
        assert score <= 70.0
        assert "rejected=courtyards_overlap" in notes

    def test_rejected_parent_loses_to_accepted_round(self, tmp_path: Path):
        pipeline = _write_parent_pipeline(tmp_path, score_total=85.0)
        rejected_score, _, _, _ = _score_round(
            leaf_accepted=3,
            leaf_total=3,
            parent_routed=True,
            parent_output_json=pipeline,
            parent_routed_validation=self._rejected_validation(
                ["connector_stranded:J2@-1.37mm(right)"]
            ),
        )
        accepted_score, _, _, tier = _score_round(
            leaf_accepted=3,
            leaf_total=3,
            parent_routed=True,
            parent_output_json=pipeline,
            parent_routed_validation={"accepted": True, "drc": {}},
        )
        assert tier == "functional"
        assert accepted_score > rejected_score

    def test_accepted_parent_still_functional(self, tmp_path: Path):
        pipeline = _write_parent_pipeline(tmp_path, score_total=72.0)
        score, _, notes, tier = _score_round(
            leaf_accepted=2,
            leaf_total=2,
            parent_routed=True,
            parent_output_json=pipeline,
            parent_routed_validation={"accepted": True, "drc": {}},
        )
        assert tier == "functional"
        assert score == 72.0

    def test_no_validation_stays_functional(self, tmp_path: Path):
        # Back-compat: callers that never pass validation keep old behavior.
        pipeline = _write_parent_pipeline(tmp_path, score_total=60.0)
        _, _, _, tier = _score_round(
            leaf_accepted=2,
            leaf_total=2,
            parent_routed=True,
            parent_output_json=pipeline,
        )
        assert tier == "functional"


class TestParentComposedReporting:
    """The round record must not alias ``parent_composed`` to ``parent_routed``.

    Reading routing success back as composition is what made the 2026-10-01
    handoff conclude "no leaf was accepted, no parent was composed" for two
    briefs whose parents were composed and routed on disk.
    """

    def test_missing_snapshot_is_not_composed(self, tmp_path: Path):
        from kicraft.cli.autoexperiment import _parent_was_composed

        assert _parent_was_composed(tmp_path / "absent.json") is False

    def test_empty_composition_is_not_composed(self, tmp_path: Path):
        from kicraft.cli.autoexperiment import _parent_was_composed

        path = tmp_path / "parent_pipeline.json"
        path.write_text(json.dumps({"state": {"entry_count": 0}, "composition": {}}))
        assert _parent_was_composed(path) is False

    def test_composed_but_unrouted_is_composed(self, tmp_path: Path):
        from kicraft.cli.autoexperiment import _parent_was_composed

        path = tmp_path / "parent_pipeline.json"
        path.write_text(
            json.dumps(
                {
                    "state": {"entry_count": 6, "routed_validation": {"accepted": False}},
                    "composition": {"summary": "BOARD [/] children=6 components=73"},
                }
            )
        )
        assert _parent_was_composed(path) is True


def _leaf_validation(*, unconnected: int, nets: list[str], accepted: bool = True) -> dict:
    """A validation block shaped like a real ``solved_layout.json`` (routing verdict)."""
    return {
        "accepted": accepted,
        "anchor_summary": {"all_required_present": True, "missing_required": []},
        "board_exists": True,
        "board_path": "leaf_routed.kicad_pcb",
        "copper_outside_outline": {"ok": True, "outside_tracks": 0, "outside_vias": 0},
        "drc": {
            "unconnected": unconnected,
            "unconnected_nets": list(nets),
            "shorts": 0,
            "clearance": 0,
            "items_not_allowed": 0,
            "ran": True,
            "returncode": 0,
            "violations": 0,
        },
        "interface_port_names": ["MOTOR_IN1"],
        "malformed_board_geometry": False,
        "obviously_illegal_routed_geometry": False,
        "pre_route_validation": {},
        "python_exception": False,
        "rejection_reasons": [],
        "render_diagnostics": {},
        "round_board_pre_route": "",
        "round_board_routed": "",
        "signal_unconnected_repair": {},
        "track_summary": {},
    }


class TestLeafGateVerdict:
    """A pinned snapshot must never read raw routing validation as the gate.

    New round snapshots persist their gate verdict with the board; retained
    snapshots lacking it are re-derived here under strict defaults. Project
    1125's RP2040 CONTROL leaf once read accepted while its own DRC said
    "Found 13 unconnected pads".
    """

    def test_unconnected_round_snapshot_is_rejected(self):
        from kicraft.cli.autoexperiment import _leaf_gate_accepted

        validation = _leaf_validation(
            unconnected=13, nets=["QSPI_SD0", "gpio7"], accepted=True
        )
        assert validation["accepted"] is True  # routing verdict says fine
        assert _leaf_gate_accepted(validation, {}) is False

    def test_clean_round_snapshot_is_accepted(self):
        from kicraft.cli.autoexperiment import _leaf_gate_accepted

        assert _leaf_gate_accepted(_leaf_validation(unconnected=0, nets=[]), {}) is True

    def test_persisted_gate_verdict_is_authoritative(self):
        from kicraft.cli.autoexperiment import _leaf_gate_accepted

        rejected = _leaf_validation(unconnected=0, nets=[])
        rejected["leaf_acceptance_result"] = {"accepted": False}
        assert _leaf_gate_accepted(rejected) is False

        accepted = _leaf_validation(unconnected=0, nets=[])
        accepted["accepted"] = False
        accepted["leaf_acceptance_result"] = {"accepted": True}
        assert _leaf_gate_accepted(accepted) is True

    def test_missing_config_hash_keeps_the_strict_unconnected_gate(self):
        from kicraft.cli.autoexperiment import _leaf_acceptance_config_for

        config = _leaf_acceptance_config_for({})
        assert config.max_unconnected == 0
