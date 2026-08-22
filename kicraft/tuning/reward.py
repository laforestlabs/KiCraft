"""Aggregate raw EvalResults into the multi-axis objective + scalarization.

Objective axes (the user's "Balanced / Pareto" choice):
  fab_ready_rate   (MAXIMIZE)  fraction of boards that route DRC-clean
  mean_drc         (MINIMIZE)  mean shorts+unconnected across boards
  mean_wall_s      (MINIMIZE)  mean place+route wall-time across boards
  mean_area_mm2    (MINIMIZE)  mean Edge.Cuts bbox area across boards (board size)
  mean_orderedness (MAXIMIZE)  mean layout-quality sub-score 0-100 across boards

Routing (KiCad Routing Tools) is only best-effort deterministic, so each (config, board)
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
# time: ~120 s is a typical place+route. These only set the trade-off scale, not
# the Pareto comparison.
#
# drc: fab-ready ALREADY requires zero shorts+unconnected, so fab_ready_rate is
# the real correctness signal; the drc axis only separates the partially-failed
# boards and must stay a tie-breaker, not a co-equal of fab. Real boards carry
# ~5-15 residual DRC even when good, so the original ref of 5 made an ordinary
# board's penalty (0.25·12/5 = 0.6) dwarf a 50-point fab swing — the optimizer
# was rewarded for NOT routing (empty board => drc 0).
#
# Calibration vs the empty-board sentinel (evaluate.MISSING_BOARD_PENALTY,
# currently 100 — dropped from the original 999 to stop optimizer thrashing):
# a typical real board costs 0.25·12/40 ≈ 0.075 (gentle tie-breaker) while a
# missing/empty board costs 0.25·100/40 ≈ 0.63 — decisively worse than any
# real board (~8x) on this axis, on top of the full fab-axis miss, without
# the old 999-era ~6-per-board crush that caused thrashing. If
# MISSING_BOARD_PENALTY changes again, re-derive both numbers here
# (tests/test_tuning.py pins the relationship).
REF_DRC = 40.0
REF_WALL_S = 120.0
# Area scale (mm^2): the corpus baseline mean board area, so the size axis
# normalizes to ~1.0 at baseline and trades off on the SAME footing as fab
# (max 1.0) rather than dwarfing it. The first i10 baseline measured mean area
# ~12621 mm^2 across the 16 train boards; an earlier guess of 3000 made the area
# cost term ~1.26 -- larger than the entire fab term -- so the optimizer chased
# small boards over routable ones, inverting the intent that fab is the gate.
# Recalibrate here if the corpus's typical board size changes materially.
REF_AREA = 12000.0


@dataclass(frozen=True)
class BoardAggregate:
    board: str
    n_seeds: int
    fab_ready_rate: float  # fraction of seeds that were fab-ready
    mean_drc: float        # mean shorts+unconnected over seeds
    mean_wall_s: float
    mean_area_mm2: float = 0.0   # mean effective board area over seeds (MINIMIZE)
    mean_orderedness: float = 0.0  # mean layout-quality 0-100 over seeds (MAXIMIZE)


@dataclass(frozen=True)
class CorpusObjectives:
    fab_ready_rate: float   # mean over boards (MAXIMIZE)
    mean_drc: float         # mean over boards (MINIMIZE)
    mean_wall_s: float      # mean over boards (MINIMIZE)
    worst_board_fab: float  # min board fab_ready_rate (robustness)
    n_boards: int
    mean_area_mm2: float = 0.0    # mean over boards (MINIMIZE)
    mean_orderedness: float = 0.0  # mean over boards 0-100 (MAXIMIZE)

    def axes(self) -> tuple[float, float, float, float, float]:
        """The comparison axes, all oriented so SMALLER is better."""
        # negate fab and orderedness so the whole tuple is "minimize" for a
        # uniform dominance test.
        return (
            -self.fab_ready_rate,
            self.mean_drc,
            self.mean_wall_s,
            self.mean_area_mm2,
            -self.mean_orderedness,
        )


def aggregate_board(results: Sequence[EvalResult]) -> BoardAggregate:
    if not results:
        raise ValueError("aggregate_board: no results")
    board = results[0].board
    n = len(results)
    fab = sum(1 for r in results if r.fab_ready) / n
    drc = sum(r.drc_total for r in results) / n
    wall = sum(r.wall_s for r in results) / n
    area = sum(r.board_area_mm2 for r in results) / n
    order = sum(r.orderedness for r in results) / n
    return BoardAggregate(board, n, fab, drc, wall, area, order)


def aggregate_corpus(board_aggs: Sequence[BoardAggregate]) -> CorpusObjectives:
    if not board_aggs:
        raise ValueError("aggregate_corpus: no boards")
    n = len(board_aggs)
    fab = sum(b.fab_ready_rate for b in board_aggs) / n
    drc = sum(b.mean_drc for b in board_aggs) / n
    wall = sum(b.mean_wall_s for b in board_aggs) / n
    area = sum(b.mean_area_mm2 for b in board_aggs) / n
    order = sum(b.mean_orderedness for b in board_aggs) / n
    worst = min(b.fab_ready_rate for b in board_aggs)
    return CorpusObjectives(fab, drc, wall, worst, n, area, order)


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

# weights: how much each axis matters. fab/order are rewards (+), drc/time/area are
# costs (-), robustness penalizes the gap between mean and worst board (overfit
# guard). The legacy presets omit "area"/"order" keys, so .get() leaves those axes
# at weight 0 and their behavior is byte-unchanged. "all_four" is the preset that
# pursues every objective the user asked for: routability, size, speed, orderedness.
SCALARIZATIONS: dict[str, dict[str, float]] = {
    "correctness": {"fab": 1.0, "drc": 0.30, "time": 0.05, "robust": 0.30},
    "balanced":    {"fab": 1.0, "drc": 0.25, "time": 0.20, "robust": 0.20},
    "speed":       {"fab": 1.0, "drc": 0.20, "time": 0.50, "robust": 0.15},
    "all_four":    {"fab": 1.0, "drc": 0.25, "time": 0.20, "robust": 0.20,
                    "area": 0.30, "order": 0.30},
}


def scalarize(
    obj: CorpusObjectives,
    weights: dict[str, float],
    *,
    ref_drc: float = REF_DRC,
    ref_wall_s: float = REF_WALL_S,
    ref_area: float = REF_AREA,
) -> float:
    """Collapse the objective axes (+ robustness) into a scalar CMA MAXIMIZES."""
    robust_gap = obj.fab_ready_rate - obj.worst_board_fab
    return (
        weights.get("fab", 1.0) * obj.fab_ready_rate
        - weights.get("drc", 0.0) * (obj.mean_drc / ref_drc)
        - weights.get("time", 0.0) * (obj.mean_wall_s / ref_wall_s)
        - weights.get("area", 0.0) * (obj.mean_area_mm2 / ref_area)
        + weights.get("order", 0.0) * (obj.mean_orderedness / 100.0)
        - weights.get("robust", 0.0) * robust_gap
    )
