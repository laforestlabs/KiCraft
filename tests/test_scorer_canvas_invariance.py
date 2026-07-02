"""Scorer canvas-invariance (area-compaction Phase 2).

- _score_net_distance in "content" mode (the LEAF-solve default via
  local_solver_config): identical layout on two canvas sizes scores
  identically; None/unset (the parent-path default) keeps the legacy
  canvas-dependent board_diag normalization.
- _score_compactness strict curve (leaf default) vs legacy (parent default).
- psw defaults stay in sync between types.DEFAULT_PLACEMENT_WEIGHTS and
  config.DEFAULT_CONFIG; the aspect raise is deferred to the tuner re-run
  (a hand-raise regressed parent compose on multi-leaf boards).
"""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
from kicraft.autoplacer.brain.types import (
    DEFAULT_PLACEMENT_WEIGHTS,
    BoardState,
    Component,
    Layer,
    Net,
    Pad,
    Point,
)
from kicraft.autoplacer.config import DEFAULT_CONFIG


def _comp(ref: str, x: float, y: float, w: float = 6.0, h: float = 6.0) -> Component:
    pad = Pad(ref=ref, pad_id="1", pos=Point(x, y), net="N1", layer=Layer.FRONT)
    return Component(
        ref=ref, value="x", pos=Point(x, y), rotation=0.0, layer=Layer.FRONT,
        width_mm=w, height_mm=h, pads=[pad],
    )


def _state(board_w: float, board_h: float) -> BoardState:
    comps = {
        "R1": _comp("R1", 10.0, 10.0),
        "R2": _comp("R2", 18.0, 10.0),
    }
    nets = {"N1": Net(name="N1", pad_refs=[("R1", "1"), ("R2", "1")])}
    return BoardState(
        components=comps,
        nets=nets,
        board_outline=(Point(0.0, 0.0), Point(board_w, board_h)),
    )


class TestNetDistanceCanvasInvariance:
    def test_content_mode_invariant_to_canvas(self):
        """Same layout, two canvas sizes -> same net_distance score."""
        cfg = {"placement_score_net_scale": "content"}
        small = PlacementScorer(_state(50.0, 50.0), cfg)._score_net_distance()
        large = PlacementScorer(_state(200.0, 200.0), cfg)._score_net_distance()
        assert small == pytest.approx(large)

    def test_unset_keeps_legacy_board_diag(self):
        """None/unset (the parent-path default) is canvas-DEPENDENT legacy."""
        small = PlacementScorer(_state(50.0, 50.0), {})._score_net_distance()
        large = PlacementScorer(_state(200.0, 200.0), {})._score_net_distance()
        # Legacy normalization: bigger canvas -> bigger denominator -> the
        # same absolute sprawl scores BETTER (this is RC4, kept only as the
        # replay-comparison fallback).
        assert large > small

    def test_content_mode_still_rewards_shorter_nets(self):
        near = _state(100.0, 100.0)
        far = _state(100.0, 100.0)
        far.components["R2"].pos = Point(35.0, 25.0)
        far.components["R2"].pads[0].pos = Point(35.0, 25.0)
        cfg = {"placement_score_net_scale": "content"}
        s_near = PlacementScorer(near, cfg)._score_net_distance()
        s_far = PlacementScorer(far, cfg)._score_net_distance()
        assert s_near > s_far


class TestCompactnessCurve:
    def test_strict_curve_zeroes_sprawl(self):
        # 72 mm^2 of parts on a 50x50 canvas: fill 2.9% -> 0 strict score
        # (the legacy curve gave the same board a ~29-point floor)
        scorer = PlacementScorer(
            _state(50.0, 50.0), {"placement_compactness_curve": "strict"}
        )
        assert scorer._score_compactness() == pytest.approx(0.0, abs=1.0)

    def test_unset_keeps_legacy_curve(self):
        scorer = PlacementScorer(_state(50.0, 50.0), {})
        fill = 72.0 / 2500.0
        assert scorer._score_compactness() == pytest.approx(
            min(100.0, fill * 150.0 + 25.0)
        )

    def test_strict_curve_saturates_at_dense_fill(self):
        state = _state(16.0, 16.0)  # 72 mm^2 on 256 mm^2 -> fill 0.28
        scorer = PlacementScorer(state, {"placement_compactness_curve": "strict"})
        score = scorer._score_compactness()
        assert 50.0 <= score <= 100.0


class TestAspectWeightDefaults:
    def test_weight_in_sync(self):
        # The 0.02 -> 0.08 raise regressed parent compose (535/530 A/B) and
        # is deferred to the CMA-ES tuner re-run; the invariant here is SYNC.
        assert DEFAULT_CONFIG["psw_aspect_ratio"] == pytest.approx(
            DEFAULT_PLACEMENT_WEIGHTS["aspect_ratio"]
        )

    def test_all_psw_defaults_match_weights(self):
        for key, value in DEFAULT_CONFIG.items():
            if key.startswith("psw_"):
                weight_key = key[len("psw_"):]
                assert DEFAULT_PLACEMENT_WEIGHTS[weight_key] == pytest.approx(value), (
                    f"{key} out of sync with DEFAULT_PLACEMENT_WEIGHTS"
                )


class TestScoringDefaultsStayLegacy:
    def test_local_solver_config_does_not_flip_scoring(self):
        """Content scoring is OPT-IN pending the CMA-ES retune: a leaf-scoped
        auto-flip regressed 535's J1 leaf routing (replay A/B 2026-07-02)."""
        from kicraft.autoplacer.brain.leaf_size_reduction import local_solver_config
        from kicraft.autoplacer.brain.subcircuit_extractor import (
            extract_leaf_board_state,
        )
        from kicraft.autoplacer.brain.types import (
            SubCircuitDefinition,
            SubCircuitId,
        )

        state = _state(100.0, 100.0)
        leaf = SubCircuitDefinition(
            id=SubCircuitId(
                sheet_name="L", sheet_file="l.kicad_sch", instance_path="/l"
            ),
            schematic_path="/nonexistent/l.kicad_sch",
            component_refs=["R1", "R2"],
            ports=[],
            child_ids=[],
            parent_id=None,
            is_leaf=True,
        )
        extraction = extract_leaf_board_state(leaf, state, margin_mm=5.0)
        local = local_solver_config({}, extraction)
        assert local.get("placement_score_net_scale") is None
        assert local.get("placement_compactness_curve") is None

        explicit = local_solver_config(
            {"placement_score_net_scale": "content",
             "placement_compactness_curve": "strict"},
            extraction,
        )
        assert explicit["placement_score_net_scale"] == "content"
        assert explicit["placement_compactness_curve"] == "strict"
