"""FIX 2b — turn a structurally-unroutable leaf from a 2400s watchdog kill
(rc=-9, no board, no log) into a fast, diagnosable route failure with evidence:

  * ``_structural_unroutable_leaves`` classifies solve_subcircuits' terminal
    "No accepted routed leaf artifact produced ... : <reasons>" lines, keeping
    only the STRUCTURAL ones (a router throw / an unrepairable illegal
    placement) that re-running the outer search cannot fix.
  * ``_tee_build_log`` mirrors the build's stdout/stderr into ``build.log``,
    flushed per line, so a SIGKILL still leaves partial evidence on disk. The
    web build worker opts out (``KICRAFT_BUILD_LOG=external``) since it writes
    the same file itself.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from kicraft.cli.autoexperiment import (
    _RC_LEAF_UNROUTABLE,
    _structural_unroutable_leaves,
    _update_unroutable_streak,
)
from kicraft.design.cli_app import _tee_build_log

# The exact terminal lines solve_subcircuits emits (run_27 / run_17 signatures).
_ROUTING_EXCEPTION = (
    "error: leaf solve failures (successful leaves persisted): "
    "/87d30157:No accepted routed leaf artifact produced for /87d30157 after "
    "12 round(s) across 4 canvas attempt(s) (0.28, 0.22, 0.17, seed-bbox): "
    "routing_exception"
)
_LEGALITY_REPAIR = (
    "warning: leaf x: No accepted routed leaf artifact produced for /abc after "
    "12 round(s) across 4 canvas attempt(s) (0.28, seed-bbox): "
    "leaf_pre_stamp_legality_repair,routing_exception"
)
_QUALITY_MISS = (
    "No accepted routed leaf artifact produced for /q after 12 round(s) across "
    "1 canvas attempt(s) (None): leaf_routed_artifact_validation"
)


def test_structural_parser_flags_router_throw_and_illegal_placement():
    assert _structural_unroutable_leaves(_ROUTING_EXCEPTION) == {
        "/87d30157": ["routing_exception"]
    }
    assert _structural_unroutable_leaves(_LEGALITY_REPAIR) == {
        "/abc": ["leaf_pre_stamp_legality_repair", "routing_exception"]
    }


def test_structural_parser_ignores_recoverable_quality_miss():
    # A routed board that failed a DRC/opens gate CAN improve across rounds, so
    # it must NOT trip the early-abort -- only the structural reasons do.
    assert _structural_unroutable_leaves(_QUALITY_MISS) == {}
    assert _structural_unroutable_leaves("[build] 2/5 place + route ...") == {}


def test_structural_parser_multi_leaf():
    combined = _ROUTING_EXCEPTION + "\n" + _LEGALITY_REPAIR + "\n" + _QUALITY_MISS
    got = _structural_unroutable_leaves(combined)
    assert set(got) == {"/87d30157", "/abc"}  # the quality-miss leaf excluded


def test_streak_default_aborts_on_first_structural_round():
    # abort_rounds=1 (the default): a leaf that reports the terminal structural
    # failure once aborts immediately -- it already exhausted 12 internal rounds,
    # so run_27's ~25-min round 1 aborts before the 2400s wall instead of
    # starting a doomed round 2.
    streak: dict[str, int] = {}
    assert _update_unroutable_streak(streak, {"/leaf": ["routing_exception"]}, 1) == "/leaf"


def test_streak_threshold_two_needs_two_consecutive():
    streak: dict[str, int] = {}
    fail = {"/leaf": ["leaf_pre_stamp_legality_repair"]}
    assert _update_unroutable_streak(streak, fail, 2) is None      # round 1
    assert _update_unroutable_streak(streak, fail, 2) == "/leaf"   # round 2


def test_streak_resets_when_leaf_recovers():
    streak: dict[str, int] = {}
    fail = {"/leaf": ["routing_exception"]}
    assert _update_unroutable_streak(streak, fail, 2) is None      # fail once
    assert _update_unroutable_streak(streak, {}, 2) is None        # recovers -> reset
    assert streak["/leaf"] == 0
    assert _update_unroutable_streak(streak, fail, 2) is None      # only 1 again


def test_streak_disabled_never_aborts():
    streak: dict[str, int] = {}
    fail = {"/leaf": ["routing_exception"]}
    for _ in range(10):
        assert _update_unroutable_streak(streak, fail, 0) is None


def test_abort_rc_is_nonzero_and_distinct_from_argparse():
    # Non-zero so cli_app._run_layout forwards it and _layout_route_fab maps it
    # to a route failure (rc6); != 2 (argparse bad-args).
    assert _RC_LEAF_UNROUTABLE not in (0, 2)


def test_tee_build_log_writes_flushed_lines(tmp_path):
    log = tmp_path / ".kicraft" / "build.log"
    with _tee_build_log(log):
        print("[build] 1/5 synthesize ...")
        print("[abort] leaf /x structurally unroutable", file=sys.stderr)
    assert log.read_text().splitlines() == [
        "[build] 1/5 synthesize ...",
        "[abort] leaf /x structurally unroutable",
    ]


def test_tee_build_log_opt_out_when_external(tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_BUILD_LOG", "external")
    log = tmp_path / ".kicraft" / "build.log"
    with _tee_build_log(log):
        print("captured by the worker itself, not here")
    assert not log.exists()  # no second writer on the worker's file


def test_tee_restores_streams_even_on_exception(tmp_path):
    log = tmp_path / "build.log"
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        with _tee_build_log(log):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert sys.stdout is saved_out and sys.stderr is saved_err
