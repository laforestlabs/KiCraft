"""Tests for PlacementSolver._sa_refine -- simulated annealing refinement.

All tests use synthetic data only; no pcbnew dependency.
"""

from __future__ import annotations

import copy


from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Net,
    Pad,
    Point,
)
from kicraft.autoplacer.brain.placement_solver import PlacementSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_component(
    ref: str,
    x: float,
    y: float,
    width: float = 5.0,
    height: float = 3.0,
    locked: bool = False,
) -> Component:
    """Build a minimal component with one pad at center."""
    return Component(
        ref=ref,
        value="R1k",
        pos=Point(x, y),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=width,
        height_mm=height,
        pads=[
            Pad(ref=ref, pad_id="1", pos=Point(x - 1.0, y), net="N1", layer=Layer.FRONT),
            Pad(ref=ref, pad_id="2", pos=Point(x + 1.0, y), net="N2", layer=Layer.FRONT),
        ],
        locked=locked,
        kind="passive",
    )


def _make_board_state(components: dict[str, Component]) -> BoardState:
    """Build a minimal BoardState with components and basic nets."""
    nets = {
        "N1": Net(name="N1", pad_refs=[(r, "1") for r in components]),
        "N2": Net(name="N2", pad_refs=[(r, "2") for r in components]),
    }
    return BoardState(
        components=components,
        nets=nets,
        traces=[],
        vias=[],
        board_outline=(Point(0, 0), Point(50, 40)),
    )


def _make_solver(state: BoardState, seed: int = 42) -> PlacementSolver:
    """Create a PlacementSolver with default config."""
    cfg = {
        "force_attract_k": 0.02,
        "force_repel_k": 200.0,
        "cooling_factor": 0.97,
        "max_placement_iterations": 50,
        "placement_clearance_mm": 2.5,
        "sa_refine_enabled": True,
        "sa_refine_iterations": 100,
        "sa_refine_initial_temp": 5.0,
        "sa_refine_cooling_rate": 0.995,
        "sa_refine_move_radius_mm": 2.0,
        "sa_refine_swap_probability": 0.3,
        "sa_refine_rotation_probability": 0.2,
    }
    return PlacementSolver(state, config=cfg, seed=seed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSARefineBasic:
    """Basic _sa_refine behavior."""

    def test_returns_dict_of_components(self):
        """SA refine returns a dict mapping ref -> Component."""
        comps = {
            "R1": _make_component("R1", 10, 10),
            "R2": _make_component("R2", 30, 30),
        }
        state = _make_board_state(comps)
        solver = _make_solver(state)

        from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
        scorer = PlacementScorer(state, solver.cfg)

        result = solver._sa_refine(
            comps, state, scorer,
            max_iters=50, init_temp=5.0, cooling_rate=0.995,
            move_radius=2.0, swap_prob=0.3, rotation_prob=0.2,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"R1", "R2"}
        for ref, comp in result.items():
            assert isinstance(comp, Component)

    def test_preserves_all_component_refs(self):
        """Result has exactly the same component refs as input."""
        comps = {
            f"R{i}": _make_component(f"R{i}", 10 + i * 8, 15 + i * 5)
            for i in range(5)
        }
        state = _make_board_state(comps)
        solver = _make_solver(state)

        from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
        scorer = PlacementScorer(state, solver.cfg)

        result = solver._sa_refine(
            comps, state, scorer, max_iters=30,
        )
        assert set(result.keys()) == set(comps.keys())

    def test_empty_components_returns_empty(self):
        """No components => empty dict returned."""
        state = _make_board_state({})
        solver = _make_solver(state)

        from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
        scorer = PlacementScorer(state, solver.cfg)

        result = solver._sa_refine({}, state, scorer, max_iters=10)
        assert result == {}


class TestSARefineLockedComponents:
    """Locked components are not moved."""

    def test_locked_components_stay_put(self):
        """Locked component positions remain unchanged."""
        locked_comp = _make_component("J1", 25.0, 20.0, locked=True)
        unlocked_comp = _make_component("R1", 10.0, 10.0, locked=False)
        comps = {"J1": locked_comp, "R1": unlocked_comp}
        state = _make_board_state(comps)
        solver = _make_solver(state)

        from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
        scorer = PlacementScorer(state, solver.cfg)

        original_j1_pos = Point(locked_comp.pos.x, locked_comp.pos.y)
        result = solver._sa_refine(
            comps, state, scorer, max_iters=200,
        )
        # J1 must not have moved
        assert result["J1"].pos.x == original_j1_pos.x
        assert result["J1"].pos.y == original_j1_pos.y

    def test_all_locked_returns_original(self):
        """When all components are locked, SA returns them unchanged."""
        comps = {
            "R1": _make_component("R1", 10, 10, locked=True),
            "R2": _make_component("R2", 30, 30, locked=True),
        }
        state = _make_board_state(comps)
        solver = _make_solver(state)

        from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
        scorer = PlacementScorer(state, solver.cfg)

        result = solver._sa_refine(comps, state, scorer, max_iters=50)
        for ref in comps:
            assert result[ref].pos.x == comps[ref].pos.x
            assert result[ref].pos.y == comps[ref].pos.y


class TestSARefineBoundsClamping:
    """Components stay within board bounds."""

    def test_components_within_board_outline(self):
        """After SA, all component positions are within board bounds."""
        # Place components near edges to stress clamping
        comps = {
            "R1": _make_component("R1", 2, 2),   # near top-left
            "R2": _make_component("R2", 48, 38),  # near bottom-right
            "R3": _make_component("R3", 25, 20),  # center
        }
        state = _make_board_state(comps)
        solver = _make_solver(state, seed=99)

        from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
        scorer = PlacementScorer(state, solver.cfg)

        result = solver._sa_refine(
            comps, state, scorer,
            max_iters=200, move_radius=10.0,  # large radius to test clamping
        )
        tl = state.board_outline[0]
        br = state.board_outline[1]
        for ref, comp in result.items():
            assert tl.x <= comp.pos.x <= br.x, (
                f"{ref}: x={comp.pos.x} outside [{tl.x}, {br.x}]"
            )
            assert tl.y <= comp.pos.y <= br.y, (
                f"{ref}: y={comp.pos.y} outside [{tl.y}, {br.y}]"
            )


class TestSARefineReproducibility:
    """Same seed produces same results."""

    def test_deterministic_with_same_seed(self):
        comps = {
            "R1": _make_component("R1", 10, 10),
            "R2": _make_component("R2", 30, 30),
            "R3": _make_component("R3", 20, 20),
        }
        state1 = _make_board_state(copy.deepcopy(comps))
        solver1 = _make_solver(state1, seed=42)
        from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
        scorer1 = PlacementScorer(state1, solver1.cfg)
        result1 = solver1._sa_refine(
            state1.components, state1, scorer1, max_iters=50,
        )

        state2 = _make_board_state(copy.deepcopy(comps))
        solver2 = _make_solver(state2, seed=42)
        scorer2 = PlacementScorer(state2, solver2.cfg)
        result2 = solver2._sa_refine(
            state2.components, state2, scorer2, max_iters=50,
        )

        for ref in comps:
            assert abs(result1[ref].pos.x - result2[ref].pos.x) < 0.001
            assert abs(result1[ref].pos.y - result2[ref].pos.y) < 0.001


class TestSARefineScoreImprovement:
    """SA should not make the score worse (returns best seen)."""

    def test_best_score_at_least_initial(self):
        """SA returns a state with score >= initial score."""
        # Put components in overlapping positions (bad) - SA should improve
        comps = {
            "R1": _make_component("R1", 20, 20),
            "R2": _make_component("R2", 21, 20),  # nearly overlapping R1
            "R3": _make_component("R3", 20, 21),  # nearly overlapping R1
        }
        state = _make_board_state(comps)
        solver = _make_solver(state, seed=7)

        from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
        scorer = PlacementScorer(state, solver.cfg)

        initial_score = scorer.score().total
        result = solver._sa_refine(
            comps, state, scorer, max_iters=500,
        )

        # Score the result
        state.components = result
        final_score = scorer.score().total

        # SA should return best-seen, which is >= initial
        assert final_score >= initial_score
