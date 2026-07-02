"""Warning-level area observations in leaf acceptance (area-compaction Phase 4).

The area_utilization gate NEVER rejects -- it attaches structured warnings
(and notes) when a leaf ships wasteful, so waste is part of the acceptance
record instead of invisible until a fleet scan.
"""

from __future__ import annotations

from kicraft.autoplacer.brain.leaf_acceptance import (
    LeafAcceptanceConfig,
    acceptance_config_from_dict,
    evaluate_leaf_acceptance,
)


def _validation(util: float, aspect: float, parts: int) -> dict:
    return {
        "board_exists": True,
        "drc": {"shorts": 0, "unconnected": 0, "clearance": 0, "courtyard": 0},
        "track_summary": {"traces": 5, "vias": 1},
        "board_metrics": {
            "area_utilization": util,
            "aspect_ratio": aspect,
            "component_count": parts,
        },
    }


class TestAreaObservationGate:
    def test_wasteful_leaf_warns_but_passes(self):
        result = evaluate_leaf_acceptance(_validation(0.074, 7.3, 11), {})
        assert result.accepted is True
        gate = result.gate_results["area_utilization"]
        assert gate["passed"] is True
        assert gate["warning"] is True
        assert len(gate["warnings"]) == 2  # low util AND high aspect
        assert any("AREA WARNING" in n for n in result.notes)

    def test_healthy_leaf_no_warning(self):
        result = evaluate_leaf_acceptance(_validation(0.30, 1.2, 11), {})
        gate = result.gate_results["area_utilization"]
        assert gate["passed"] is True
        assert gate["warning"] is False
        assert not any("AREA WARNING" in n for n in result.notes)

    def test_small_part_count_exempt_from_util_warning(self):
        # A 2-part connector breakout at 10% util is normal, not waste
        result = evaluate_leaf_acceptance(_validation(0.10, 1.2, 2), {})
        gate = result.gate_results["area_utilization"]
        assert gate["warning"] is False

    def test_aspect_warns_regardless_of_part_count(self):
        result = evaluate_leaf_acceptance(_validation(0.40, 5.0, 2), {})
        gate = result.gate_results["area_utilization"]
        assert gate["warning"] is True
        assert "aspect ratio" in gate["warnings"][0]

    def test_missing_metrics_skips_quietly(self):
        validation = _validation(0.074, 7.3, 11)
        del validation["board_metrics"]
        result = evaluate_leaf_acceptance(validation, {})
        gate = result.gate_results["area_utilization"]
        assert gate["passed"] is True
        assert gate.get("skipped") is True

    def test_never_causes_rejection(self):
        result = evaluate_leaf_acceptance(_validation(0.01, 12.0, 40), {})
        assert "area_utilization" not in result.rejection_reasons
        assert result.accepted is True

    def test_config_thresholds_threaded(self):
        cfg = acceptance_config_from_dict(
            {
                "leaf_area_warn_utilization": 0.40,
                "leaf_area_warn_aspect": 1.5,
                "leaf_area_warn_min_parts": 2,
            }
        )
        assert cfg.area_warn_utilization == 0.40
        assert cfg.area_warn_aspect == 1.5
        assert cfg.area_warn_min_parts == 2
        result = evaluate_leaf_acceptance(_validation(0.30, 2.0, 3), {}, cfg)
        gate = result.gate_results["area_utilization"]
        assert gate["warning"] is True
        assert len(gate["warnings"]) == 2

    def test_defaults(self):
        cfg = LeafAcceptanceConfig()
        assert cfg.area_warn_utilization == 0.15
        assert cfg.area_warn_aspect == 4.0
        assert cfg.area_warn_min_parts == 5
