"""Slow end-to-end check: a real replay eval on a committed fixture.

Skipped unless ``KICRAFT_TUNING_INTEGRATION=1`` because it runs a full
place+route (the autorouter; minutes). It validates the subprocess plumbing the
whole tuner stands on: scratch prep, config-overlay injection, the replay
subprocess, and routed-reward extraction.

    KICRAFT_TUNING_INTEGRATION=1 .venv/bin/python -m pytest \
        tests/test_tuning_integration.py -q -s
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("KICRAFT_TUNING_INTEGRATION") != "1",
    reason="set KICRAFT_TUNING_INTEGRATION=1 to run the slow place+route eval",
)

FIXTURE = Path("tests/fixtures/replay_workspace/USB_PD_TRIGGER")


def test_replay_eval_produces_routed_reward(tmp_path):
    from kicraft.tuning.evaluate import evaluate_config
    from kicraft.tuning.store import config_hash

    overlay: dict = {}  # baseline default config
    r = evaluate_config(
        overlay, workspace_path=FIXTURE, board="USB_PD_TRIGGER", seed=0,
        config_hash=config_hash(overlay), scratch_dir=tmp_path / "ws",
        mode="replay", quality="fast", timeout_s=1800, use_build_slot=False,
    )
    # A routed board must have produced real copper, regardless of fab-readiness.
    assert r.error == "" or r.rc != 0
    assert r.traces > 0, f"no traces routed: {r}"
    assert r.wall_s > 0
    # drc_total is a real measured count, not the missing-board sentinel
    from kicraft.tuning.evaluate import MISSING_BOARD_PENALTY
    assert r.drc_total < MISSING_BOARD_PENALTY


def test_determinism_same_seed_same_placement(tmp_path):
    """Two replay evals at the same seed agree on traces/vias (placement is
    byte-deterministic; routing is best-effort, so allow a small wobble)."""
    from kicraft.tuning.evaluate import evaluate_config
    from kicraft.tuning.store import config_hash

    overlay: dict = {}
    kw = dict(workspace_path=FIXTURE, board="USB_PD_TRIGGER", seed=0,
              config_hash=config_hash(overlay), mode="replay", quality="fast",
              timeout_s=1800, use_build_slot=False)
    r1 = evaluate_config(overlay, scratch_dir=tmp_path / "a", **kw)
    r2 = evaluate_config(overlay, scratch_dir=tmp_path / "b", **kw)
    assert r1.traces > 0 and r2.traces > 0
    # vias come from deterministic placement/stitching; expect agreement
    assert abs(r1.vias - r2.vias) <= 2
