"""Content-derived leaf canvas + grow-on-failure ladder (area-compaction Phase 1).

Covers:
- derive_content_canvas: fill-target sizing, aspect from edge zones,
  largest-part / clearance floors
- set_extraction_canvas: outline/envelope replacement without touching
  component positions or translation
- board_utilization_metrics (Phase 0 helper)
- the _solve_leaf_subcircuit canvas ladder: content-first, grow on failure,
  seed-bbox terminal fallback, array-leaf exemption, seed-bbox mode parity

All synthetic data; no pcbnew, no FreeRouting.
"""

from __future__ import annotations


import pytest

from kicraft.autoplacer.brain.placement_utils import board_utilization_metrics
from kicraft.autoplacer.brain.subcircuit_extractor import (
    derive_content_canvas,
    extract_leaf_board_state,
    set_extraction_canvas,
)
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Net,
    Pad,
    PlacementScore,
    Point,
    SolveRoundResult,
    SubCircuitDefinition,
    SubCircuitId,
)


def _comp(ref: str, x: float, y: float, w: float = 2.0, h: float = 1.0) -> Component:
    return Component(
        ref=ref,
        value="x",
        pos=Point(x, y),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=w,
        height_mm=h,
    )


# ---------------------------------------------------------------------------
# derive_content_canvas
# ---------------------------------------------------------------------------


class TestDeriveContentCanvas:
    def test_area_matches_fill_target(self):
        """Canvas area ~= component area / fill target when clearance floor
        does not dominate (big parts, small clearance)."""
        comps = {
            f"Q{i}": _comp(f"Q{i}", 20.0 * i, 20.0, w=10.0, h=6.0) for i in range(6)
        }
        total = 6 * 10.0 * 6.0  # 360 mm^2
        w, h = derive_content_canvas(
            comps, fill_target=0.28, placement_clearance_mm=0.5
        )
        assert w * h == pytest.approx(total / 0.28, rel=0.05)
        # near-square by default
        assert max(w, h) / min(w, h) == pytest.approx(1.0, abs=0.05)

    def test_kc4w7knw_scale(self):
        """The plan's worked example: ~361 mm^2 of parts at fill 0.28 lands
        near a 41x31 mm hand-layout canvas, nowhere near 195 mm wide."""
        comps = {
            f"P{i}": _comp(f"P{i}", 20.0 + 20.0 * i, 20.0, w=6.7, h=4.9)
            for i in range(11)
        }
        w, h = derive_content_canvas(
            comps, fill_target=0.28, placement_clearance_mm=2.84
        )
        assert max(w, h) < 60.0
        assert min(w, h) > 15.0

    def test_edge_zones_widen_flow_axis(self):
        comps = {
            "J1": _comp("J1", 0.0, 0.0, w=8.0, h=8.0),
            "J2": _comp("J2", 50.0, 0.0, w=8.0, h=8.0),
            "U1": _comp("U1", 25.0, 0.0, w=8.0, h=8.0),
        }
        zones = {"J1": {"edge": "left"}, "J2": {"edge": "right"}}
        w_wide, h_wide = derive_content_canvas(
            comps, fill_target=0.28, placement_clearance_mm=0.5, component_zones=zones
        )
        assert w_wide > h_wide

        zones_v = {"J1": {"edge": "top"}, "J2": {"edge": "bottom"}}
        w_tall, h_tall = derive_content_canvas(
            comps, fill_target=0.28, placement_clearance_mm=0.5, component_zones=zones_v
        )
        assert h_tall > w_tall

    def test_zone_for_foreign_ref_ignored(self):
        """Zones for refs NOT in this leaf must not skew the aspect."""
        comps = {"R1": _comp("R1", 0.0, 0.0, w=4.0, h=4.0)}
        zones = {"J9": {"edge": "left"}, "J8": {"edge": "right"}}
        w, h = derive_content_canvas(
            comps, fill_target=0.28, placement_clearance_mm=0.5, component_zones=zones
        )
        assert w == pytest.approx(h)

    def test_largest_part_floor(self):
        """A single 30x4 part forces both sides >= 30 + margins even though
        area/fill would allow a narrower canvas."""
        comps = {"J1": _comp("J1", 0.0, 0.0, w=30.0, h=4.0)}
        w, h = derive_content_canvas(
            comps, fill_target=0.28, placement_clearance_mm=2.0
        )
        assert w >= 34.0  # 30 + 2*max(2.0, clearance)
        assert h >= 34.0  # rotation-safe: the part may solve at 90 degrees

    def test_clearance_floor_dominates_for_many_small_parts(self):
        """20 tiny passives: clearance-padded packing, not raw area, sets the
        canvas so the legalizer has a satisfiable box."""
        comps = {f"C{i}": _comp(f"C{i}", i, 0.0, w=1.0, h=0.5) for i in range(20)}
        w, h = derive_content_canvas(
            comps, fill_target=0.28, placement_clearance_mm=2.84
        )
        padded = 20 * (1.0 + 2.84) * (0.5 + 2.84) * 1.15
        assert w * h >= padded * 0.99
        # raw area / fill would have been far too small
        assert w * h > (20 * 0.5) / 0.28

    def test_empty_components_min_side(self):
        w, h = derive_content_canvas({}, fill_target=0.28)
        assert w >= 5.0 and h >= 5.0


# ---------------------------------------------------------------------------
# set_extraction_canvas
# ---------------------------------------------------------------------------


def _leaf_and_state(n_parts: int = 4, pitch: float = 20.0):
    comps = {}
    nets = {}
    for i in range(n_parts):
        ref = f"R{i + 1}"
        pad = Pad(
            ref=ref, pad_id="1", pos=Point(20.0 + pitch * i, 20.0),
            net="N1", layer=Layer.FRONT,
        )
        comps[ref] = Component(
            ref=ref, value="x", pos=Point(20.0 + pitch * i, 20.0),
            rotation=0.0, layer=Layer.FRONT, width_mm=4.0, height_mm=3.0,
            pads=[pad],
        )
    nets["N1"] = Net(
        name="N1", pad_refs=[(f"R{i + 1}", "1") for i in range(n_parts)]
    )
    state = BoardState(
        components=comps,
        nets=nets,
        board_outline=(Point(0.0, 0.0), Point(200.0, 100.0)),
    )
    leaf = SubCircuitDefinition(
        id=SubCircuitId(
            sheet_name="LEAF", sheet_file="leaf.kicad_sch", instance_path="/leaf"
        ),
        schematic_path="/nonexistent/leaf.kicad_sch",
        component_refs=list(comps.keys()),
        ports=[],
        child_ids=[],
        parent_id=None,
        is_leaf=True,
    )
    return leaf, state


class TestSetExtractionCanvas:
    def test_outline_envelope_updated_positions_untouched(self):
        leaf, state = _leaf_and_state()
        extraction = extract_leaf_board_state(leaf, state, margin_mm=10.0)
        before_positions = {
            ref: (c.pos.x, c.pos.y)
            for ref, c in extraction.local_state.components.items()
        }
        before_translation = (extraction.translation.x, extraction.translation.y)

        set_extraction_canvas(extraction, 33.0, 21.0, note="content_canvas test")

        tl, br = extraction.local_state.board_outline
        assert (tl.x, tl.y) == (0.0, 0.0)
        assert (br.x, br.y) == (33.0, 21.0)
        assert extraction.envelope.width_mm == 33.0
        assert extraction.envelope.height_mm == 21.0
        assert extraction.local_state.board_width == pytest.approx(33.0)
        for ref, c in extraction.local_state.components.items():
            assert (c.pos.x, c.pos.y) == before_positions[ref]
        assert (extraction.translation.x, extraction.translation.y) == (
            before_translation
        )
        assert any("content_canvas test" in n for n in extraction.notes)


# ---------------------------------------------------------------------------
# board_utilization_metrics (Phase 0)
# ---------------------------------------------------------------------------


class TestBoardUtilizationMetrics:
    def test_known_values(self):
        comps = {
            "A": _comp("A", 5.0, 5.0, w=10.0, h=10.0),
            "B": _comp("B", 25.0, 5.0, w=10.0, h=10.0),
        }
        m = board_utilization_metrics(comps, 40.0, 20.0)
        assert m["component_area_mm2"] == pytest.approx(200.0)
        assert m["area_utilization"] == pytest.approx(200.0 / 800.0)
        assert m["aspect_ratio"] == pytest.approx(2.0)
        # placed bbox: x [0,30], y [0,10] -> 300
        assert m["placed_bbox_area_mm2"] == pytest.approx(300.0)
        assert m["bbox_utilization"] == pytest.approx(200.0 / 300.0, abs=1e-3)

    def test_empty_is_zero_not_perfect(self):
        m = board_utilization_metrics({}, 40.0, 20.0)
        assert m["area_utilization"] == 0.0
        assert m["bbox_utilization"] == 0.0

    def test_degenerate_board(self):
        m = board_utilization_metrics(
            {"A": _comp("A", 0, 0)}, 0.0, 0.0
        )
        assert m["area_utilization"] == 0.0
        assert m["aspect_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Canvas ladder in _solve_leaf_subcircuit
# ---------------------------------------------------------------------------


class _FakeNode:
    def __init__(self, definition):
        self.definition = definition
        self.id = definition.id


def _run_ladder(monkeypatch, cfg, accept_plan, rounds=1):
    """Drive _solve_leaf_subcircuit with a fake per-round solver.

    ``accept_plan(call_index, extraction)`` -> True for an accepted
    (trivial-pass) round, False for a routing failure. Returns
    (solved, calls) where calls is a list of (round_index, board_w, board_h).
    """
    from kicraft.cli import solve_subcircuits as ss

    leaf, state = _leaf_and_state(n_parts=6)
    node = _FakeNode(leaf)
    calls: list[tuple[int, float, float]] = []

    def fake_solve_one_round(extraction, round_cfg, seed, round_index, route):
        idx = len(calls)
        calls.append(
            (
                round_index,
                extraction.local_state.board_width,
                extraction.local_state.board_height,
            )
        )
        accepted = accept_plan(idx, extraction)
        if accepted:
            routing = {
                "enabled": True,
                "skipped": True,
                "reason": "no_internal_nets",
                "failed": False,
                "validation": {"accepted": True},
                "_trace_segments": [],
                "_via_objects": [],
            }
        else:
            routing = {
                "enabled": True,
                "failed": True,
                "reason": "routing_failed",
                "validation": {"accepted": False},
                "_trace_segments": [],
                "_via_objects": [],
            }
        return SolveRoundResult(
            round_index=round_index,
            seed=seed,
            score=50.0 if accepted else float("-inf"),
            placement=PlacementScore(),
            components=dict(extraction.local_state.components),
            routing=routing,
            routed=accepted,
        )

    def fake_size_reduction(extraction, best_round, _cfg):
        return extraction, best_round, {"attempted": False}

    monkeypatch.setattr(ss, "_solve_one_round", fake_solve_one_round)
    monkeypatch.setattr(ss, "_attempt_leaf_size_reduction", fake_size_reduction)

    solved = ss._solve_leaf_subcircuit(
        node=node,
        full_state=state,
        cfg=cfg,
        rounds=rounds,
        base_seed=7,
        route=True,
        experiment_round=1,
    )
    return solved, calls


class TestCanvasLadder:
    def test_content_canvas_used_when_first_round_accepts(self, monkeypatch):
        cfg = {"leaf_canvas_mode": "content", "leaf_canvas_fill_target": 0.28}
        solved, calls = _run_ladder(monkeypatch, cfg, lambda i, e: True)
        assert len(calls) == 1
        _, w, h = calls[0]
        # 6 parts of 4x3 -> content canvas far below the 200mm seed board
        assert w < 60.0 and h < 60.0
        assert solved.scheduling_metadata["canvas_mode"] == "content"
        assert solved.scheduling_metadata["canvas_attempts"] == ["0.28"]

    def test_ladder_grows_to_seed_bbox_on_failure(self, monkeypatch):
        cfg = {
            "leaf_canvas_mode": "content",
            "leaf_canvas_fill_target": 0.28,
            "leaf_canvas_fill_ladder": [0.22, 0.17],
            "subcircuit_margin_mm": 10.0,
        }
        # Accept only the final (seed-bbox) attempt
        solved, calls = _run_ladder(monkeypatch, cfg, lambda i, e: i == 3)
        assert len(calls) == 4
        # Round indices stay monotonic across attempts
        assert [c[0] for c in calls] == [0, 1, 2, 3]
        # Canvas areas grow along the ladder; last is the seed envelope
        areas = [w * h for _, w, h in calls]
        assert areas[0] < areas[1] < areas[2] < areas[3]
        seed_w = calls[-1][1]
        assert seed_w > 100.0  # seed scatter spans ~100mm + margins
        assert solved.scheduling_metadata["canvas_attempts"] == [
            "0.28",
            "0.22",
            "0.17",
            "seed-bbox",
        ]
        # The persisted extraction is the one the winning round solved on
        assert solved.extraction.local_state.board_width == pytest.approx(seed_w)

    def test_seed_bbox_mode_single_attempt(self, monkeypatch):
        cfg = {"leaf_canvas_mode": "seed-bbox", "subcircuit_margin_mm": 10.0}
        solved, calls = _run_ladder(monkeypatch, cfg, lambda i, e: True)
        assert len(calls) == 1
        assert calls[0][1] > 100.0
        assert solved.scheduling_metadata["canvas_mode"] == "seed-bbox"
        assert solved.scheduling_metadata["canvas_attempts"] == ["seed-bbox"]

    def test_array_leaf_exempt(self, monkeypatch):
        cfg = {
            "leaf_canvas_mode": "content",
            "subcircuit_margin_mm": 10.0,
            "arrays": [{"refs": ["R1", "R2", "R3"]}],
        }
        solved, calls = _run_ladder(monkeypatch, cfg, lambda i, e: True)
        assert len(calls) == 1
        assert calls[0][1] > 100.0  # seed-bbox canvas
        assert solved.scheduling_metadata["canvas_mode"] == "seed-bbox"
        assert solved.scheduling_metadata["canvas_array_leaf_exempt"] is True

    def test_best_effort_uses_matching_extraction(self, monkeypatch):
        """No attempt accepted anywhere, but attempt 0 produced a routed
        board -> best-effort compose must use attempt 0's canvas."""
        from kicraft.cli import solve_subcircuits as ss

        cfg = {
            "leaf_canvas_mode": "content",
            "leaf_canvas_fill_target": 0.28,
            "leaf_canvas_fill_ladder": [],
            "subcircuit_margin_mm": 10.0,
        }
        leaf, state = _leaf_and_state(n_parts=6)
        node = _FakeNode(leaf)
        calls = []

        def fake_solve_one_round(extraction, round_cfg, seed, round_index, route):
            idx = len(calls)
            calls.append(extraction)
            routing = {
                "enabled": True,
                "failed": True,
                "reason": "routing_failed",
                "validation": {"accepted": False},
                "_trace_segments": [],
                "_via_objects": [],
            }
            if idx == 0:
                # freerouting "failed" but stamped a board -> best_routed
                routing["routed_board_path"] = __file__  # any existing file
            return SolveRoundResult(
                round_index=round_index,
                seed=seed,
                score=10.0 - idx,
                placement=PlacementScore(),
                components=dict(extraction.local_state.components),
                routing=routing,
                routed=False,
            )

        monkeypatch.setattr(ss, "_solve_one_round", fake_solve_one_round)
        monkeypatch.setattr(
            ss,
            "_attempt_leaf_size_reduction",
            lambda e, b, c: (e, b, {"attempted": False}),
        )

        solved = ss._solve_leaf_subcircuit(
            node=node,
            full_state=state,
            cfg=cfg,
            rounds=1,
            base_seed=7,
            route=True,
            experiment_round=1,
        )
        # attempt 0 (content) produced the only routed board; the persisted
        # extraction must be attempt 0's small canvas, not the seed-bbox
        assert solved.extraction.local_state.board_width == pytest.approx(
            calls[0].local_state.board_width
        )
        assert solved.extraction.local_state.board_width < 60.0

    def test_all_attempts_fail_raises_with_attempt_trail(self, monkeypatch):

        cfg = {
            "leaf_canvas_mode": "content",
            "leaf_canvas_fill_target": 0.28,
            "leaf_canvas_fill_ladder": [0.22],
            "subcircuit_margin_mm": 10.0,
        }
        with pytest.raises(RuntimeError, match="canvas attempt"):
            _run_ladder(monkeypatch, cfg, lambda i, e: False)
