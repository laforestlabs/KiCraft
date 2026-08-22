"""Scheduling policy for the hierarchical autoexperiment round loop.

Every decision about WHETHER another round runs -- and with what restrictions
-- lives here in one unit-testable object, instead of as `if`s and loop-scoped
mutables threaded through ``autoexperiment.main()`` (see
docs/plans/autoexperiment-round-scheduler.md). The loop body stays mechanism
(subprocesses, artifacts, status writes) and consults the scheduler at the
same points the policies used to sit inline:

    plan_next(...)        loop top: stop? rounds left? wall budget? rescue?
    observe_solve(...)    after the leaf solve: streak aborts
    observe_parent(...)   after the parent route: generic backoff updates
    observe_round_end(..) after scoring: keep/best verdict
    observe_duration(...) loop bottom: wall-duration EMA the budget gate reads

Decisions come back as :class:`RoundPlan` (run a round) or :class:`Finalize`
(stop; ``reason`` carries the exact log line the inline code used to print, so
build-log grep tooling keeps working). The scheduler does no I/O of its own --
the one disk-derived input (which leaves still have no accepted artifact, for
the wall-budget rescue round) is injected as a callable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Exit code for an early-abort on a structurally unroutable leaf. Any non-zero
# leaf-phase rc is forwarded by cli_app._run_layout and mapped to a route
# failure (rc6) by _layout_route_fab; distinct from argparse's 2.
_RC_LEAF_UNROUTABLE = 3

# Below this remaining wall budget there is no point in a rescue round -- the
# leaf solve needs a meaningful slice (>= the ~60s seed-bbox floor) plus the
# parent compose/route tail.
_WALL_RESCUE_MIN_S = 120.0


@dataclass(frozen=True)
class RoundPlan:
    """One round the scheduler wants run. ``only``/``leaf_deadline_s`` are
    set on a wall-budget rescue round (solve restricted to unpinned leaves,
    deadline clamped to the budget slice); empty/None means the caller's own
    ``--only`` and configured deadline apply. ``parent_refit`` False forces
    ``candidate_search.parent_refit=false`` for this round's compose (the
    re-fit backoff below); None leaves the configured default in place."""

    round_num: int
    only: tuple[str, ...] = ()
    leaf_deadline_s: float | None = None
    note: str = ""  # non-empty -> print verbatim (the rescue announcement)
    parent_refit: bool | None = None
    # >1.0 scales parent_seed_area_overhead for this round's compose: the
    # congestion-growth valve below trades compactness back for routability
    # after rounds whose ROUTED parent was rejected for unconnected nets.
    seed_overhead_scale: float = 1.0


@dataclass(frozen=True)
class Finalize:
    """Stop the search. ``reason`` is the log line to print (verbatim parity
    with the old inline policies); ``announce=False`` marks the historically
    silent stops (stop request, round-count exhaustion). ``rc_hint`` set means
    return that exit code immediately instead of finalizing best-so-far."""

    reason: str
    rc_hint: int | None = None
    announce: bool = True


def _update_unroutable_streak(
    streak: dict[str, int],
    struct_fail: dict[str, list[str]],
    abort_rounds: int,
) -> str | None:
    """Fold this round's structural leaf failures into the per-leaf streak and
    return the leaf that has now failed ``>= abort_rounds`` consecutive rounds
    (or ``None``). A leaf that did NOT fail structurally this round resets to 0
    (it recovered). ``abort_rounds <= 0`` disables the early-abort.
    """
    for leaf in list(streak):
        if leaf not in struct_fail:
            streak[leaf] = 0
    for leaf in struct_fail:
        streak[leaf] = streak.get(leaf, 0) + 1
    if abort_rounds <= 0:
        return None
    blown = sorted(p for p, n in streak.items() if n >= abort_rounds)
    return blown[0] if blown else None


def _update_quality_streak(
    streak: dict[str, dict[str, Any]],
    quality_fail: dict[str, list[str]],
    abort_rounds: int,
) -> str | None:
    """Fold this round's quality rejections into a per-leaf streak keyed by the
    rejection SIGNATURE. A leaf whose SAME rejection persists ``>= abort_rounds``
    consecutive rounds is stuck -- placement-param mutation is not helping it, so
    it is returned so the search can finalize best-so-far instead of re-solving it
    into the watchdog (WS2). A leaf that recovers or changes signature resets.
    ``abort_rounds <= 0`` disables.
    """
    for leaf in list(streak):
        if leaf not in quality_fail:
            del streak[leaf]
    for leaf, reasons in quality_fail.items():
        sig = ",".join(reasons)
        prev = streak.get(leaf)
        if prev is not None and prev.get("sig") == sig:
            prev["n"] += 1
        else:
            streak[leaf] = {"sig": sig, "n": 1}
    if abort_rounds <= 0:
        return None
    blown = sorted(p for p, s in streak.items() if s.get("n", 0) >= abort_rounds)
    return blown[0] if blown else None


@dataclass
class RoundScheduler:
    """Owns every termination/steering policy of the round loop.

    State the policies need (streaks, EMA, best score, rescue latch) lives
    here; ``main()`` holds none of it. All methods are plain arithmetic --
    no subprocesses, no filesystem -- so the policies unit-test in
    milliseconds (tests/test_round_scheduler.py).
    """

    rounds: int
    max_wall_s: float = 0.0
    unroutable_abort_rounds: int = 0
    quality_abort_rounds: int = 0
    keep_threshold: float = 0.5
    # Wall-budget rescue round (N2a): spend an otherwise-wasted remaining
    # budget on ONE solve restricted to leaves with zero accepted artifacts.
    # Disabled when the caller already restricts leaves (--only) or skips the
    # leaf solve entirely (--parents-only).
    rescue_enabled: bool = False
    rescue_min_s: float = _WALL_RESCUE_MIN_S
    # round_num -> instance paths of leaves with NO accepted artifact on disk.
    unpinned_leaves: Callable[[int], list[str]] | None = None

    best_score: float = field(default=-1.0, init=False)
    kept_count: int = field(default=0, init=False)
    ema_round_s: float | None = field(default=None, init=False)
    _next_round: int = field(default=1, init=False)
    _unroutable_streak: dict[str, int] = field(default_factory=dict, init=False)
    _quality_streak: dict[str, dict[str, Any]] = field(
        default_factory=dict, init=False
    )
    _rescue_attempted: bool = field(default=False, init=False)
    # Re-fit backoff (self-eval 2026-07-17 T3): once a round's ROUTED parent is
    # rejected with a re-fit candidate as the winner, the re-fit traded fab
    # success for compactness on this design -- later rounds force
    # candidate_search.parent_refit=false so the pass-1 (roomier) candidates
    # compete unopposed. Latches for the rest of the search: the re-fit is
    # deterministic per seed family, so re-enabling it just repeats the loss.
    _refit_backoff: bool = field(default=False, init=False)
    _refit_backoff_announced: bool = field(default=False, init=False)
    # Congestion-growth valve (self-eval 2026-07-17 T3 follow-up): each round
    # whose routed parent is rejected for UNCONNECTED nets grows the next
    # round's seed-overhead by 30% (capped at 2x) -- an unroutable-because-
    # cramped placement needs room, which is exactly what rounds are for.
    # Healthy runs never increment, so compactness is untouched there.
    _congestion_rounds: int = field(default=0, init=False)

    # -- loop top -----------------------------------------------------------

    def plan_next(
        self, *, elapsed_s: float, stop_requested: bool
    ) -> RoundPlan | Finalize:
        if stop_requested:
            return Finalize("stop requested", announce=False)
        if self._next_round > self.rounds:
            return Finalize(f"all {self.rounds} round(s) run", announce=False)
        # Wall budget: don't start a round the budget can't absorb -- finalize
        # the best-so-far board instead of marching into a SIGKILL with zero
        # artifacts (WS2). No EMA yet (round 1) -> always allowed to run.
        if self.max_wall_s > 0 and self.ema_round_s is not None:
            if elapsed_s + self.ema_round_s > self.max_wall_s:
                remaining = self.max_wall_s - elapsed_s
                rescue = self._rescue_plan(elapsed_s, remaining)
                if rescue is not None:
                    return rescue
                return Finalize(
                    f"[wall-budget] elapsed {elapsed_s:.0f}s + est. next round "
                    f"{self.ema_round_s:.0f}s > budget {self.max_wall_s:.0f}s; "
                    f"finalizing best-so-far after {self._next_round - 1} "
                    f"round(s)"
                )
        note = ""
        if self._refit_backoff and not self._refit_backoff_announced:
            self._refit_backoff_announced = True
            note = (
                f"[refit-backoff] round {self._next_round} onward runs with "
                "candidate_search.parent_refit=false: an earlier round's "
                "routed parent was rejected with the re-fit candidate as "
                "winner (compactness traded away fab success on this design)"
            )
        plan = RoundPlan(
            round_num=self._next_round,
            parent_refit=False if self._refit_backoff else None,
            seed_overhead_scale=self.seed_overhead_scale(),
            note=note,
        )
        self._next_round += 1
        return plan

    def seed_overhead_scale(self) -> float:
        return min(2.0, 1.0 + 0.3 * self._congestion_rounds)

    def _rescue_plan(
        self, elapsed_s: float, remaining_s: float
    ) -> RoundPlan | None:
        if (
            self._rescue_attempted
            or not self.rescue_enabled
            or remaining_s < self.rescue_min_s
            or self.unpinned_leaves is None
        ):
            return None
        unpinned = self.unpinned_leaves(self._next_round)
        if not unpinned:
            return None
        self._rescue_attempted = True
        # Leaf slice of the rescue: leave headroom for the parent
        # compose/route tail; the solve-side seed-bbox reserve (N1)
        # guarantees the fallback rung inside this slice.
        leaf_s = max(60.0, remaining_s * 0.6)
        plan = RoundPlan(
            round_num=self._next_round,
            only=tuple(unpinned),
            leaf_deadline_s=leaf_s,
            # The re-fit backoff and congestion growth apply to EVERY later
            # round, the rescue round included -- it is the run's last shot at
            # a routable parent, exactly where replaying a known-losing
            # configuration hurts most.
            parent_refit=False if self._refit_backoff else None,
            seed_overhead_scale=self.seed_overhead_scale(),
            note=(
                f"[wall-budget] elapsed {elapsed_s:.0f}s + est. next round "
                f"{self.ema_round_s:.0f}s > budget {self.max_wall_s:.0f}s; "
                f"{len(unpinned)} leaf(s) still have no accepted artifact -- "
                f"spending the remaining {remaining_s:.0f}s on a rescue round "
                f"for {', '.join(unpinned)} (leaf slice {leaf_s:.0f}s) "
                f"instead of re-solving pinned leaves"
            ),
        )
        self._next_round += 1
        return plan

    # -- after the leaf solve -------------------------------------------------

    def observe_solve(
        self,
        *,
        struct_fail: dict[str, list[str]],
        quality_fail: dict[str, list[str]],
    ) -> Finalize | None:
        """Streak policies over the solve outcome. Structural unroutability
        aborts with an exit code (placement mutation cannot fix a router throw);
        a repeated identical quality rejection finalizes best-so-far (WS2)."""
        blown = _update_unroutable_streak(
            self._unroutable_streak, struct_fail, self.unroutable_abort_rounds
        )
        if blown is not None:
            return Finalize(
                f"[abort] leaf {blown} is structurally unroutable "
                f"({','.join(struct_fail[blown])}) after "
                f"{self._unroutable_streak[blown]} round(s) -- stopping the "
                f"search instead of retrying to the build watchdog wall. "
                f"Reported as a route failure with the evidence above.",
                rc_hint=_RC_LEAF_UNROUTABLE,
            )
        stuck = _update_quality_streak(
            self._quality_streak, quality_fail, self.quality_abort_rounds
        )
        if stuck is not None:
            entry = self._quality_streak[stuck]
            return Finalize(
                f"[quality-stop] leaf {stuck} produced the same rejection "
                f"({entry['sig']}) for {entry['n']} consecutive round(s) -- "
                f"placement mutation is not improving it; finalizing "
                f"best-so-far instead of re-solving into the watchdog."
            )
        return None

    # -- after the parent route -----------------------------------------------

    def observe_parent(
        self,
        *,
        rejected_refit_winner: bool = False,
        rejected_unconnected: bool = False,
    ) -> None:
        """Update generic re-fit and congestion backoff state."""
        if rejected_refit_winner:
            self._refit_backoff = True
        if rejected_unconnected:
            self._congestion_rounds += 1

    # -- after scoring ----------------------------------------------------------

    def improvement_vs_best(self, score: float) -> float:
        return score if self.best_score < 0.0 else round(score - self.best_score, 3)

    def observe_round_end(self, *, score: float, subprocesses_ok: bool) -> bool:
        """Keep/best verdict: a round is promoted to best only when its
        subprocesses succeeded AND it improves by >= keep_threshold (or is the
        first scored round). Returns the verdict and folds it into best/kept."""
        kept = subprocesses_ok and (
            self.best_score < 0.0
            or self.improvement_vs_best(score) >= self.keep_threshold
        )
        if kept:
            self.best_score = score
            self.kept_count += 1
        return kept

    # -- loop bottom -----------------------------------------------------------

    def observe_duration(self, duration_s: float) -> None:
        """Fold the observed round duration into the wall-budget EMA."""
        duration_s = max(0.0, duration_s)
        self.ema_round_s = (
            duration_s
            if self.ema_round_s is None
            else 0.5 * self.ema_round_s + 0.5 * duration_s
        )
