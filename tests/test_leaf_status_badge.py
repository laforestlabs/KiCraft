"""Regression tests for the leaf-card status badge.

A fresh autoexperiment run starts by deleting per-leaf debug.json
and round_NNNN_* snapshots while keeping the canonical leaf_routed.
kicad_pcb / solved_layout.json from the previous run intact (so
pins.json keeps applying). The Monitor's leaf cards used to flip to
red FAILED immediately because:

  * ``_determine_leaf_status`` read stale ``solved_layout.json`` and
    returned "accepted" / "routing"
  * the ``selected_round`` filter then forced "failed" since
    ``debug.json`` was gone and no current-run rounds existed yet

Both branches ignored that the run was actually in progress and
hadn't reached the leaf yet. These tests lock the new behaviour:
during a run, missing ``debug.json`` means "queued, waiting its
turn" -- not "failed."
"""

from __future__ import annotations

import json
from pathlib import Path

from kicraft.gui.components.pipeline_graph import _determine_leaf_status


def _seed_canonical(leaf_dir: Path, *, accepted: bool = True) -> None:
    """Seed a leaf dir with canonical files but no debug.json -- this is
    the post-cleanup state at the start of a fresh phased run."""
    leaf_dir.mkdir(parents=True, exist_ok=True)
    (leaf_dir / "metadata.json").write_text('{"sheet_name": "FOO"}')
    (leaf_dir / "leaf_routed.kicad_pcb").write_text("(canonical)")
    (leaf_dir / "leaf_pre_freerouting.kicad_pcb").write_text("(canonical)")
    payload = {"validation": {"accepted": accepted}}
    (leaf_dir / "solved_layout.json").write_text(json.dumps(payload))


def test_run_in_progress_no_debug_returns_queued(tmp_path: Path):
    """The user-visible bug: post-cleanup canonical files exist, but
    debug.json is gone because the run just started. The badge must
    say "queued / WAITING", not the alarming "FAILED" or stale
    "ACCEPTED"."""
    leaf = tmp_path / "leaf"
    _seed_canonical(leaf, accepted=True)
    assert _determine_leaf_status(leaf, run_in_progress=True) == "queued"


def test_idle_no_debug_with_canonical_returns_accepted(tmp_path: Path):
    """When the runner is idle (no run active), the canonical files
    represent the LAST successful run -- showing them as accepted is
    the correct UX."""
    leaf = tmp_path / "leaf"
    _seed_canonical(leaf, accepted=True)
    assert _determine_leaf_status(leaf, run_in_progress=False) == "accepted"


def test_run_in_progress_with_debug_falls_through_to_canonical(tmp_path: Path):
    """Once debug.json appears for a leaf during a run, the leaf has
    been processed -- trust the canonical files again."""
    leaf = tmp_path / "leaf"
    _seed_canonical(leaf, accepted=True)
    (leaf / "debug.json").write_text('{"extra": {"all_rounds": []}}')
    assert _determine_leaf_status(leaf, run_in_progress=True) == "accepted"


def test_run_in_progress_with_debug_failure_returns_failed(tmp_path: Path):
    """A leaf whose debug.json reports an error during a run gets
    flagged failed even before the canonical files update."""
    leaf = tmp_path / "leaf"
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "debug.json").write_text('{"failed": true}')
    assert _determine_leaf_status(leaf, run_in_progress=True) == "failed"


def test_empty_dir_returns_pending(tmp_path: Path):
    """A leaf that has never been touched returns pending."""
    leaf = tmp_path / "leaf"
    leaf.mkdir(parents=True, exist_ok=True)
    assert _determine_leaf_status(leaf, run_in_progress=False) == "pending"


def test_idle_with_rejection_returns_failed(tmp_path: Path):
    leaf = tmp_path / "leaf"
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "metadata.json").write_text('{"sheet_name": "FOO"}')
    (leaf / "solved_layout.json").write_text(
        json.dumps({"validation": {"rejected": True}})
    )
    assert _determine_leaf_status(leaf, run_in_progress=False) == "failed"


def test_running_with_partial_artifacts_returns_queued(tmp_path: Path):
    """Even if leaf_routed.kicad_pcb exists from a prior run, an active
    run with no debug.json means the leaf is queued. We don't want to
    show "routing" (orange) for a leaf that hasn't even been visited
    yet this run."""
    leaf = tmp_path / "leaf"
    leaf.mkdir(parents=True, exist_ok=True)
    (leaf / "leaf_routed.kicad_pcb").write_text("(stale canonical)")
    assert _determine_leaf_status(leaf, run_in_progress=True) == "queued"


def test_default_run_in_progress_false_preserves_old_behaviour(tmp_path: Path):
    """``run_in_progress`` defaults to False so callers that haven't
    been updated still get the legacy "trust canonical" semantics.
    Important for any integration test or old code path."""
    leaf = tmp_path / "leaf"
    _seed_canonical(leaf, accepted=True)
    assert _determine_leaf_status(leaf) == "accepted"
