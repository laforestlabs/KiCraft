"""Aggregate raw EvalResults into the 3-axis objective + scalarization.

Objective axes (the user's "Balanced / Pareto" choice):
  fab_ready_rate  (MAXIMIZE)  fraction of boards that route DRC-clean
  mean_drc        (MINIMIZE)  mean shorts+unconnected across boards
  mean_wall_s     (MINIMIZE)  mean place+route wall-time across boards

Routing (FreeRouting) is only best-effort deterministic, so each (config, board)
is replicated over K seeds and averaged here; placement is byte-deterministic, so
seed variance is purely routing noise. Use common random numbers (the same seed
set across all configs in a generation) for low-variance paired ranking.

CMA-ES needs a scalar; ``scalarize`` collapses the axes under a weight vector.
We run several weightings (correctness/balanced/speed) so the union of their
results sweeps the Pareto front. ``dominates``/``pareto_front`` operate on the
true 3-axis objective and drive promotion, independent of any scalarization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from kicraft.tuning.evaluate import EvalResult

# Reference scales for normalizing the minimize-axes into ~[0, 1] in scalarize.
# drc: a board with ~5 net violations is "quite bad"; time: ~120 s is a typical
# place+route. These only set the trade-off scale, not the Pareto comparison.
REF_DRC = 5.0
REF_WALL_S = 120.0


@dataclass(frozen=True)
class BoardAggregate:
    board: str
    n_seeds: int
    fab_ready_rate: float  # fraction of seeds that were fab-ready
    mean_drc: float        # mean shorts+unconnected over seeds
    mean_wall_s: float


@dataclass(frozen=True)
class CorpusObjectives:
    fab_ready_rate: float   # mean over boards (MAXIMIZE)
    mean_drc: float         # mean over boards (MINIMIZE)
    mean_wall_s: float      # mean over boards (MINIMIZE)
    worst_board_fab: float  # min board fab_ready_rate (robustness)
    n_boards: int

    def axes(self) -> tuple[float, float, float]:
        """The 3 comparison axes, all oriented so SMALLER is better."""
        # negate fab so the whole tuple is "minimize" for a uniform dominance test
        return (-self.fab_ready_rate, self.mean_drc, self.mean_wall_s)


def aggregate_board(results: Sequence[EvalResult]) -> BoardAggregate:
    if not results:
        raise ValueError("aggregate_board: no results")
    board = results[0].board
    n = len(results)
    fab = sum(1 for r in results if r.fab_ready) / n
    drc = sum(r.drc_total for r in results) / n
    wall = sum(r.wall_s for r in results) / n
    return BoardAggregate(board, n, fab, drc, wall)


def aggregate_corpus(board_aggs: Sequence[BoardAggregate]) -> CorpusObjectives:
    if not board_aggs:
        raise ValueError("aggregate_corpus: no boards")
    n = len(board_aggs)
    fab = sum(b.fab_ready_rate for b in board_aggs) / n
    drc = sum(b.mean_drc for b in board_aggs) / n
    wall = sum(b.mean_wall_s for b in board_aggs) / n
    worst = min(b.fab_ready_rate for b in board_aggs)
    return CorpusObjectives(fab, drc, wall, worst, n)


def aggregate_results(results: Iterable[EvalResult]) -> CorpusObjectives:
    """One-shot: group EvalResults by board, aggregate seeds, then the corpus."""
    by_board: dict[str, list[EvalResult]] = {}
    for r in results:
        by_board.setdefault(r.board, []).append(r)
    return aggregate_corpus([aggregate_board(rs) for rs in by_board.values()])


# --- Pareto ---------------------------------------------------------------

def dominates(a: CorpusObjectives, b: CorpusObjectives, *, eps: float = 1e-9) -> bool:
    """True iff ``a`` Pareto-dominates ``b``: no worse on every axis, strictly
    better on at least one. Axes oriented smaller-is-better via ``.axes()``."""
    aa, ba = a.axes(), b.axes()
    no_worse = all(x <= y + eps for x, y in zip(aa, ba))
    strictly = any(x < y - eps for x, y in zip(aa, ba))
    return no_worse and strictly


def pareto_front(objs: Sequence[CorpusObjectives]) -> list[int]:
    """Indices of the non-dominated objectives."""
    keep: list[int] = []
    for i, oi in enumerate(objs):
        if not any(j != i and dominates(oj, oi) for j, oj in enumerate(objs)):
            keep.append(i)
    return keep


# --- scalarization (CMA steering) -----------------------------------------

# weights: how much each axis matters. fab is a reward (+), drc/time are costs (-),
# robustness penalizes the gap between mean and worst board (overfit guard).
SCALARIZATIONS: dict[str, dict[str, float]] = {
    "correctness": {"fab": 1.0, "drc": 0.30, "time": 0.05, "robust": 0.30},
    "balanced":    {"fab": 1.0, "drc": 0.25, "time": 0.20, "robust": 0.20},
    "speed":       {"fab": 1.0, "drc": 0.20, "time": 0.50, "robust": 0.15},
}


def scalarize(
    obj: CorpusObjectives,
    weights: dict[str, float],
    *,
    ref_drc: float = REF_DRC,
    ref_wall_s: float = REF_WALL_S,
) -> float:
    """Collapse the 3 axes (+ robustness) into a scalar CMA MAXIMIZES."""
    robust_gap = obj.fab_ready_rate - obj.worst_board_fab
    return (
        weights.get("fab", 1.0) * obj.fab_ready_rate
        - weights.get("drc", 0.0) * (obj.mean_drc / ref_drc)
        - weights.get("time", 0.0) * (obj.mean_wall_s / ref_wall_s)
        - weights.get("robust", 0.0) * robust_gap
    )
