"""Unit tests for the autoexperiment RoundScheduler.

These pin the round loop's termination/steering policies -- stop, round count,
wall-budget EMA, the rescue round, streak aborts, cap-out stop, keep/best --
with synthetic outcomes and no subprocesses, which is the payoff of extracting
them from main() (docs/plans/autoexperiment-round-scheduler.md).
"""
from __future__ import annotations

from kicraft.cli._round_scheduler import (
    Finalize,
    RoundPlan,
    RoundScheduler,
    _RC_LEAF_UNROUTABLE,
)


def _drain_round(s: RoundScheduler, *, score=10.0, ok=True, dur=100.0) -> bool:
    kept = s.observe_round_end(score=score, subprocesses_ok=ok)
    s.observe_duration(dur)
    return kept


def test_round_count_exhaustion_and_numbering():
    s = RoundScheduler(rounds=2)
    p1 = s.plan_next(elapsed_s=0, stop_requested=False)
    assert isinstance(p1, RoundPlan) and p1.round_num == 1 and p1.only == ()
    p2 = s.plan_next(elapsed_s=0, stop_requested=False)
    assert isinstance(p2, RoundPlan) and p2.round_num == 2
    done = s.plan_next(elapsed_s=0, stop_requested=False)
    assert isinstance(done, Finalize) and not done.announce and done.rc_hint is None


def test_stop_request_wins_and_is_silent():
    s = RoundScheduler(rounds=5)
    d = s.plan_next(elapsed_s=0, stop_requested=True)
    assert isinstance(d, Finalize) and not d.announce


def test_wall_budget_gate_needs_an_ema_and_uses_it():
    s = RoundScheduler(rounds=10, max_wall_s=100.0)
    # No EMA yet -> round 1 always runs, even over budget.
    assert isinstance(s.plan_next(elapsed_s=99, stop_requested=False), RoundPlan)
    s.observe_duration(60.0)
    # 50 elapsed + 60 EMA > 100 -> finalize with the wall-budget log line.
    d = s.plan_next(elapsed_s=50, stop_requested=False)
    assert isinstance(d, Finalize) and d.announce
    assert "[wall-budget]" in d.reason and "after 1 round(s)" in d.reason
    # 30 elapsed + 60 EMA <= 100 -> runs.
    assert isinstance(s.plan_next(elapsed_s=30, stop_requested=False), RoundPlan)


def test_ema_is_exponential_half_half():
    s = RoundScheduler(rounds=3)
    s.observe_duration(100.0)
    assert s.ema_round_s == 100.0
    s.observe_duration(50.0)
    assert s.ema_round_s == 75.0


def test_crashed_route_time_is_half_discounted_in_ema():
    # self-eval 2026-07-27 run_29 (fix-plan P3.2): one 398s round whose parent
    # route died in FreeRouting crash retries priced the estimator out of ALL
    # remaining rounds on a 648s budget. Half the crashed route time is
    # discounted, so the search gets exactly one more budget-bounded attempt.
    s = RoundScheduler(rounds=3, max_wall_s=648.0)
    s.observe_duration(398.0, crashed_route_s=397.7)
    assert s.ema_round_s is not None and abs(s.ema_round_s - 199.15) < 0.01
    # 398 elapsed + ~199 est <= 648 -> the next round RUNS.
    assert isinstance(s.plan_next(elapsed_s=398, stop_requested=False), RoundPlan)
    # A second crashed round exhausts the budget honestly.
    s.observe_duration(398.0, crashed_route_s=397.7)
    d = s.plan_next(elapsed_s=796, stop_requested=False)
    assert isinstance(d, Finalize) and "[wall-budget]" in d.reason


def test_crashed_route_discount_never_goes_negative():
    s = RoundScheduler(rounds=3)
    s.observe_duration(10.0, crashed_route_s=100.0)
    assert s.ema_round_s == 0.0


def test_wall_budget_rescue_round_fires_once_with_clamped_deadline():
    calls: list[int] = []

    def unpinned(round_num: int) -> list[str]:
        calls.append(round_num)
        return ["/mcu-leaf"]

    s = RoundScheduler(
        rounds=10, max_wall_s=1000.0, rescue_enabled=True, unpinned_leaves=unpinned
    )
    assert isinstance(s.plan_next(elapsed_s=0, stop_requested=False), RoundPlan)
    s.observe_duration(600.0)
    # 500 elapsed + 600 EMA > 1000, remaining 500 >= 120 -> rescue, not finalize.
    d = s.plan_next(elapsed_s=500, stop_requested=False)
    assert isinstance(d, RoundPlan)
    assert d.round_num == 2 and calls == [2]
    assert d.only == ("/mcu-leaf",)
    assert d.leaf_deadline_s == 300.0  # max(60, 500 * 0.6)
    assert "rescue round for /mcu-leaf" in d.note
    # One rescue only: the same starved state now finalizes.
    d2 = s.plan_next(elapsed_s=500, stop_requested=False)
    assert isinstance(d2, Finalize) and "[wall-budget]" in d2.reason


def test_wall_budget_rescue_skipped_when_disabled_starved_or_all_pinned():
    def none_unpinned(_rn: int) -> list[str]:
        return []

    # rescue_enabled=False (--only / --parents-only callers).
    s = RoundScheduler(rounds=10, max_wall_s=100.0, unpinned_leaves=lambda rn: ["/a"])
    s.observe_duration(90.0)
    assert isinstance(s.plan_next(elapsed_s=50, stop_requested=False), Finalize)
    # Remaining below the floor: 100-90=10 < 120.
    s2 = RoundScheduler(
        rounds=10, max_wall_s=100.0, rescue_enabled=True,
        unpinned_leaves=lambda rn: ["/a"],
    )
    s2.observe_duration(90.0)
    assert isinstance(s2.plan_next(elapsed_s=90, stop_requested=False), Finalize)
    # Every leaf already pinned.
    s3 = RoundScheduler(
        rounds=10, max_wall_s=1000.0, rescue_enabled=True,
        unpinned_leaves=none_unpinned,
    )
    s3.observe_duration(600.0)
    assert isinstance(s3.plan_next(elapsed_s=500, stop_requested=False), Finalize)


def test_unroutable_streak_aborts_with_exit_code_and_resets_on_recovery():
    s = RoundScheduler(rounds=10, unroutable_abort_rounds=2)
    fail = {"/mcu": ["router_throw"]}
    assert s.observe_solve(struct_fail=fail, quality_fail={}) is None
    # Recovery resets the streak...
    assert s.observe_solve(struct_fail={}, quality_fail={}) is None
    assert s.observe_solve(struct_fail=fail, quality_fail={}) is None
    # ...two consecutive failures abort with the route-failure exit code.
    d = s.observe_solve(struct_fail=fail, quality_fail={})
    assert isinstance(d, Finalize)
    assert d.rc_hint == _RC_LEAF_UNROUTABLE
    assert "[abort] leaf /mcu" in d.reason and "router_throw" in d.reason


def test_quality_streak_finalizes_on_same_signature_only():
    s = RoundScheduler(rounds=10, quality_abort_rounds=2)
    assert s.observe_solve(
        struct_fail={}, quality_fail={"/led": ["unconnected"]}
    ) is None
    # A DIFFERENT rejection signature resets the streak.
    assert s.observe_solve(
        struct_fail={}, quality_fail={"/led": ["illegal_routed_geometry"]}
    ) is None
    d = s.observe_solve(
        struct_fail={}, quality_fail={"/led": ["illegal_routed_geometry"]}
    )
    assert isinstance(d, Finalize) and d.rc_hint is None
    assert "[quality-stop] leaf /led" in d.reason
    assert "illegal_routed_geometry" in d.reason


def test_parent_capout_needs_consecutive_capped_rounds():
    s = RoundScheduler(rounds=10, parent_capout_rounds=2)
    # Capped once, then a completed route resets the streak.
    assert s.observe_parent(routed=False, elapsed_s=590, cap_s=600) is None
    assert s.observe_parent(routed=True, elapsed_s=590, cap_s=600) is None
    assert s.observe_parent(routed=False, elapsed_s=580, cap_s=600) is None
    d = s.observe_parent(routed=False, elapsed_s=599, cap_s=600)
    assert isinstance(d, Finalize) and "[parent-capout]" in d.reason
    # A fast failure (not near the cap) is not a cap-out.
    s2 = RoundScheduler(rounds=10)
    assert s2.observe_parent(routed=False, elapsed_s=30, cap_s=600) is None
    assert s2.observe_parent(routed=False, elapsed_s=30, cap_s=600) is None


def test_keep_best_threshold_and_subprocess_gate():
    s = RoundScheduler(rounds=10)
    # First scored round is kept regardless of threshold (best_score < 0)...
    assert s.improvement_vs_best(10.0) == 10.0
    assert s.observe_round_end(score=10.0, subprocesses_ok=True) is True
    assert s.best_score == 10.0 and s.kept_count == 1
    # ...but never with a failed subprocess (the stale-cache promotion trap).
    s_fail = RoundScheduler(rounds=10)
    assert s_fail.observe_round_end(score=50.0, subprocesses_ok=False) is False
    assert s_fail.best_score == -1.0 and s_fail.kept_count == 0
    # Sub-threshold improvement is discarded; >= 0.5 is kept.
    assert s.observe_round_end(score=10.3, subprocesses_ok=True) is False
    assert s.best_score == 10.0
    assert s.improvement_vs_best(10.6) == 0.6
    assert s.observe_round_end(score=10.6, subprocesses_ok=True) is True
    assert s.best_score == 10.6 and s.kept_count == 2


def test_drain_round_helper_matches_loop_order():
    # Sanity: verdict then duration, as main() calls them.
    s = RoundScheduler(rounds=2, max_wall_s=50.0)
    p = s.plan_next(elapsed_s=0, stop_requested=False)
    assert isinstance(p, RoundPlan)
    assert _drain_round(s, score=5.0, ok=True, dur=40.0) is True
    d = s.plan_next(elapsed_s=20, stop_requested=False)
    assert isinstance(d, Finalize) and "[wall-budget]" in d.reason


def test_refit_backoff_latches_flags_rounds_and_announces_once():
    # self-eval 2026-07-17 T3: a routed parent rejected with the re-fit
    # candidate as winner latches parent_refit=False for every later round.
    s = RoundScheduler(rounds=4)
    p1 = s.plan_next(elapsed_s=0, stop_requested=False)
    assert isinstance(p1, RoundPlan) and p1.parent_refit is None and p1.note == ""
    # Round 1's parent: routed but validation-rejected with a refit winner.
    assert s.observe_parent(
        routed=True, elapsed_s=10.0, cap_s=600.0, rejected_refit_winner=True
    ) is None
    p2 = s.plan_next(elapsed_s=0, stop_requested=False)
    assert isinstance(p2, RoundPlan) and p2.parent_refit is False
    assert "[refit-backoff]" in p2.note
    # The announcement prints once; the restriction persists.
    p3 = s.plan_next(elapsed_s=0, stop_requested=False)
    assert isinstance(p3, RoundPlan) and p3.parent_refit is False and p3.note == ""


def test_refit_backoff_not_latched_by_ordinary_failures():
    s = RoundScheduler(rounds=3)
    s.plan_next(elapsed_s=0, stop_requested=False)
    # Rejected parent WITHOUT a refit winner (pass-1 lost on its own merits),
    # and a clean round: neither may trigger the backoff.
    assert s.observe_parent(
        routed=False, elapsed_s=10.0, cap_s=600.0, rejected_refit_winner=False
    ) is None
    p = s.plan_next(elapsed_s=0, stop_requested=False)
    assert isinstance(p, RoundPlan) and p.parent_refit is None and p.note == ""


def test_congestion_growth_scales_seed_overhead_capped():
    s = RoundScheduler(rounds=8)
    p = s.plan_next(elapsed_s=0, stop_requested=False)
    assert p.seed_overhead_scale == 1.0
    s.observe_parent(routed=True, elapsed_s=10, cap_s=600,
                     rejected_unconnected=True)
    assert s.plan_next(elapsed_s=0, stop_requested=False).seed_overhead_scale == 1.3
    for _ in range(5):
        s.observe_parent(routed=True, elapsed_s=10, cap_s=600,
                         rejected_unconnected=True)
    # 6 congested rounds -> 1 + 1.8 capped at 2.0.
    assert s.plan_next(elapsed_s=0, stop_requested=False).seed_overhead_scale == 2.0
    # A cap-out or clean round does not increment.
    s2 = RoundScheduler(rounds=3)
    s2.observe_parent(routed=False, elapsed_s=590, cap_s=600)
    assert s2.plan_next(elapsed_s=0, stop_requested=False).seed_overhead_scale == 1.0
