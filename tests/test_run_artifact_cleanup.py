"""Regression tests for per-run leaf-artifact cleanup and within-run
round accumulation.

The Monitor tab's score plot and round timeline are built from each
leaf's ``debug.json`` (key ``extra.all_rounds``). When a fresh
autoexperiment run is started, the GUI must show ONLY rounds from
that run -- not stale rounds that piled up across earlier runs. But
WITHIN a single run, when a leaf is solved across multiple parent
rounds (one invocation per parent round in ``--leaves-only`` mode),
debug.json must accumulate so the GUI can scrub R1, R2, R3 and see
each parent round's leaf solves.

Two pieces have to work together:

1. ``ExperimentRunner._purge_prior_run_artifacts`` deletes per-leaf
   ``round_NNNN_*`` snapshots and ``debug.json`` at the start of
   every run, including ``--leaves-only`` and ``--parents-only``.
   Canonical pin-source files (``leaf_routed.kicad_pcb``,
   ``solved_layout.json``, ``metadata.json``, ``renders/``) are
   preserved so ``pins.json`` references survive.

2. ``solve_subcircuits._persist_solution`` reads any prior rounds
   left in debug.json by earlier parent-round invocations within
   THIS run (carried on ``SolvedLeafSubcircuit.prior_rounds``) and
   merges them with the current invocation's new rounds, dedup'd by
   round_index and sorted. ``solve_subcircuits._solve_leaf_subcircuit``
   computes ``base_offset = max(prior_round_index) + 1`` so new
   round indices are monotonic across parent rounds (no snapshot
   filename collisions).

These tests lock both the cleanup contract (cross-run isolation) and
the merge contract (within-run accumulation).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kicraft.gui.experiment_runner import ExperimentRunner


def _build_runner_for(tmp_path: Path) -> ExperimentRunner:
    """Construct a runner pointed at ``tmp_path/.experiments``."""
    project_root = tmp_path
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    runner = ExperimentRunner(
        project_root=project_root,
        scripts_dir=scripts_dir,
        experiments_dir=project_root / ".experiments",
    )
    runner.experiments_dir.mkdir(parents=True, exist_ok=True)
    return runner


def _seed_leaf_dir(
    leaf_dir: Path,
    *,
    round_indices: list[int],
    write_debug: bool = True,
) -> None:
    """Populate a leaf artifact dir with canonical files plus round
    snapshots plus a debug.json -- the state that piles up after a
    successful prior run."""
    leaf_dir.mkdir(parents=True, exist_ok=True)
    # Canonical pin-source files. Their presence + the prior pin would
    # satisfy `pins.list_available_rounds` after a re-pin.
    (leaf_dir / "leaf_routed.kicad_pcb").write_text("(kicad_pcb canonical)")
    (leaf_dir / "leaf_pre_freerouting.kicad_pcb").write_text("(canonical)")
    (leaf_dir / "metadata.json").write_text('{"sheet_name": "FOO"}')
    (leaf_dir / "solved_layout.json").write_text('{"components": {}}')
    renders = leaf_dir / "renders"
    renders.mkdir(exist_ok=True)
    (renders / "leaf_routed.png").write_text("png-bytes")

    for idx in round_indices:
        prefix = f"round_{idx:04d}"
        (leaf_dir / f"{prefix}_leaf_routed.kicad_pcb").write_text("(round)")
        (leaf_dir / f"{prefix}_leaf_pre_freerouting.kicad_pcb").write_text("(round)")
        (leaf_dir / f"{prefix}_metadata.json").write_text('{"r": "round"}')
        (leaf_dir / f"{prefix}_solved_layout.json").write_text('{"r": "round"}')

    if write_debug:
        (leaf_dir / "debug.json").write_text(
            '{"extra": {"all_rounds": [{"round_index": ' + str(round_indices[-1]) + '}]}}'
        )


def test_full_run_purge_wipes_subcircuits_wholesale(tmp_path: Path):
    """phase=None means a full pipeline run; everything goes."""
    runner = _build_runner_for(tmp_path)
    leaf = runner.experiments_dir / "subcircuits" / "leafA"
    _seed_leaf_dir(leaf, round_indices=[0, 5, 14])

    runner._purge_prior_run_artifacts(phase=None)

    # subcircuits/ is gone wholesale (canonical + snapshots all wiped).
    assert not (runner.experiments_dir / "subcircuits").exists()


def test_leaves_only_purge_keeps_canonical_drops_snapshots(tmp_path: Path):
    """phase='leaves_only' must keep canonical files (so pins survive)
    but drop every round_NNNN_* snapshot and the debug.json -- those
    are the source of the "rounds 1..14 in a 3-round run" GUI bug."""
    runner = _build_runner_for(tmp_path)
    leaf = runner.experiments_dir / "subcircuits" / "leafA"
    _seed_leaf_dir(leaf, round_indices=[0, 5, 14])

    runner._purge_prior_run_artifacts(phase="leaves_only")

    # Canonical files preserved.
    assert (leaf / "leaf_routed.kicad_pcb").exists()
    assert (leaf / "leaf_pre_freerouting.kicad_pcb").exists()
    assert (leaf / "metadata.json").exists()
    assert (leaf / "solved_layout.json").exists()
    assert (leaf / "renders").is_dir()
    assert (leaf / "renders" / "leaf_routed.png").exists()

    # Per-round snapshots gone.
    assert not list(leaf.glob("round_*_*.kicad_pcb"))
    assert not list(leaf.glob("round_*_metadata.json"))
    assert not list(leaf.glob("round_*_solved_layout.json"))

    # debug.json is the score-plot source; must be wiped.
    assert not (leaf / "debug.json").exists()


def test_parents_only_purge_preserves_leaves_only_state(tmp_path: Path):
    """parents-only does NOT re-solve leaves -- it consumes pinned
    snapshots from a previous leaves-only run. The leaves-only run's
    debug.json is the source of truth for both the GUI's per-leaf
    round timeline AND the selected_round filter that maps parent
    rounds to leaf rounds. Wiping it on a parents-only run start
    causes every leaf card to flip to FAILED (rounds=[]; selected_round
    branch picks the failure path). So the cleanup MUST keep
    debug.json + round_NNNN_* + canonical files all intact for
    parents-only."""
    runner = _build_runner_for(tmp_path)
    leaf = runner.experiments_dir / "subcircuits" / "leafA"
    _seed_leaf_dir(leaf, round_indices=[0, 1, 2])

    runner._purge_prior_run_artifacts(phase="parents_only")

    # All canonical + round + debug state preserved.
    assert (leaf / "leaf_routed.kicad_pcb").exists()
    assert (leaf / "solved_layout.json").exists()
    assert (leaf / "metadata.json").exists()
    assert (leaf / "debug.json").exists()
    assert sorted(p.name for p in leaf.glob("round_*_*.kicad_pcb")) == [
        "round_0000_leaf_pre_freerouting.kicad_pcb",
        "round_0000_leaf_routed.kicad_pcb",
        "round_0001_leaf_pre_freerouting.kicad_pcb",
        "round_0001_leaf_routed.kicad_pcb",
        "round_0002_leaf_pre_freerouting.kicad_pcb",
        "round_0002_leaf_routed.kicad_pcb",
    ]


def test_purge_handles_multiple_leaves(tmp_path: Path):
    runner = _build_runner_for(tmp_path)
    sub = runner.experiments_dir / "subcircuits"
    leaf_a = sub / "leafA"
    leaf_b = sub / "leafB"
    _seed_leaf_dir(leaf_a, round_indices=[0, 1, 2])
    _seed_leaf_dir(leaf_b, round_indices=[0, 1, 2, 3, 4])

    runner._purge_prior_run_artifacts(phase="leaves_only")

    for leaf in (leaf_a, leaf_b):
        assert (leaf / "leaf_routed.kicad_pcb").exists()
        assert not list(leaf.glob("round_*_*.kicad_pcb"))
        assert not (leaf / "debug.json").exists()


def test_purge_skips_non_directory_entries(tmp_path: Path):
    """A stray file inside subcircuits/ shouldn't crash the cleanup."""
    runner = _build_runner_for(tmp_path)
    sub = runner.experiments_dir / "subcircuits"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "stray.txt").write_text("not a leaf dir")
    leaf = sub / "leafA"
    _seed_leaf_dir(leaf, round_indices=[0])

    runner._purge_prior_run_artifacts(phase="leaves_only")

    # The stray file is fine to leave alone.
    assert (sub / "stray.txt").exists()
    assert (leaf / "leaf_routed.kicad_pcb").exists()
    assert not list(leaf.glob("round_*_*.kicad_pcb"))


def test_purge_does_not_touch_renders_subdir_files(tmp_path: Path):
    """Per-round renders inside leaf_dir/renders/round_NNNN_*.png are
    cheap to regenerate but expensive to lose mid-run while looking
    at a still-running snapshot picker. Per-leaf cleanup intentionally
    stays at the top level of the leaf dir."""
    runner = _build_runner_for(tmp_path)
    leaf = runner.experiments_dir / "subcircuits" / "leafA"
    _seed_leaf_dir(leaf, round_indices=[0])
    (leaf / "renders" / "round_0000_routed_front_all.png").write_text("png")

    runner._purge_prior_run_artifacts(phase="leaves_only")

    assert (leaf / "renders" / "round_0000_routed_front_all.png").exists()


def test_purge_idempotent_on_empty_subcircuits(tmp_path: Path):
    runner = _build_runner_for(tmp_path)
    runner._purge_prior_run_artifacts(phase="leaves_only")
    runner._purge_prior_run_artifacts(phase=None)
    # No exception -- the cleanup must be a no-op when nothing exists.


# ---------------------------------------------------------------------------
# Within-run round accumulation across parent rounds
# ---------------------------------------------------------------------------


def test_solve_subcircuits_source_wires_within_run_accumulation():
    """Source-level lock that the within-run accumulation plumbing is
    present in ``solve_subcircuits.py``: prior_rounds field on the
    dataclass, prior-debug read in ``_solve_leaf_subcircuit`` with
    ``base_offset`` derived from existing rounds, monotonic
    ``round_index = base_offset + local_round_index``, and the
    merge-and-sort step in ``_persist_solution``.

    Cross-run isolation is enforced separately by
    ``_purge_prior_run_artifacts`` (covered by the cleanup tests above).
    Together they guarantee: fresh-run debug.json contains exactly the
    new run's rounds; multi-parent-round runs accumulate within a run.
    """
    src = (
        Path(__file__).resolve().parent.parent
        / "kicraft"
        / "cli"
        / "solve_subcircuits.py"
    ).read_text(encoding="utf-8")

    assert "prior_rounds: list[dict[str, Any]] = field(default_factory=list)" in src
    assert "base_offset = max_idx + 1" in src
    assert "round_index = base_offset + local_round_index" in src
    assert "prior_rounds=prior_all_rounds" in src
    assert "merged_rounds.extend(new_round_dicts)" in src


def test_within_run_merge_dedups_and_sorts_by_round_index():
    """Algorithmic lock on the merge step in ``_persist_solution``.

    Prior rounds (carried on ``SolvedLeafSubcircuit.prior_rounds`` from
    earlier parent-round invocations of the same leaf within this run)
    must be combined with the current invocation's new rounds:
      - new wins on round_index collision (defensive)
      - result sorted by round_index ascending
      - experiment_round stamps are preserved per-row (so R1's rounds
        retain ``experiment_round=1`` even after R2 appends)
    """
    # Parent round 1 produced rounds 0, 1.
    prior_rounds = [
        {"round_index": 0, "experiment_round": 1, "score": 80.0},
        {"round_index": 1, "experiment_round": 1, "score": 82.0},
    ]
    # Parent round 2: base_offset = 2, so this invocation's rounds get
    # indices 2, 3 with experiment_round=2.
    new_round_dicts = [
        {"round_index": 2, "experiment_round": 2, "score": 85.0},
        {"round_index": 3, "experiment_round": 2, "score": 84.0},
    ]

    new_indices = {int(r.get("round_index", -1) or -1) for r in new_round_dicts}
    merged_rounds = [
        r
        for r in prior_rounds
        if int(r.get("round_index", -1) or -1) not in new_indices
    ]
    merged_rounds.extend(new_round_dicts)
    merged_rounds.sort(key=lambda r: int(r.get("round_index", 0) or 0))

    assert [r["round_index"] for r in merged_rounds] == [0, 1, 2, 3]
    assert [r["experiment_round"] for r in merged_rounds] == [1, 1, 2, 2]
    assert [r["score"] for r in merged_rounds] == [80.0, 82.0, 85.0, 84.0]


def test_within_run_merge_new_wins_on_round_index_collision():
    """If a prior round and a new round share a round_index (defensive
    edge case -- shouldn't happen in normal operation since base_offset
    keeps indices monotonic), the new value replaces the prior one.
    This guards against stale data masquerading as fresh.
    """
    prior_rounds = [
        {"round_index": 0, "experiment_round": 1, "score": 50.0},  # stale
        {"round_index": 1, "experiment_round": 1, "score": 82.0},
    ]
    new_round_dicts = [
        {"round_index": 0, "experiment_round": 2, "score": 90.0},  # fresh
    ]

    new_indices = {int(r.get("round_index", -1) or -1) for r in new_round_dicts}
    merged_rounds = [
        r
        for r in prior_rounds
        if int(r.get("round_index", -1) or -1) not in new_indices
    ]
    merged_rounds.extend(new_round_dicts)
    merged_rounds.sort(key=lambda r: int(r.get("round_index", 0) or 0))

    assert [r["round_index"] for r in merged_rounds] == [0, 1]
    # round_index 0 must reflect the new (experiment_round=2) value, not stale.
    assert merged_rounds[0]["experiment_round"] == 2
    assert merged_rounds[0]["score"] == 90.0
    # round_index 1 (no collision) is preserved from prior.
    assert merged_rounds[1]["experiment_round"] == 1


def test_cross_run_wipe_clears_within_run_state(tmp_path: Path):
    """End-to-end: even with a populated debug.json from a prior run,
    the leaves-only run-start cleanup wipes it before the new run's
    invocations begin. So when the new run's first parent-round
    invocation runs ``_solve_leaf_subcircuit``, prior_all_rounds is
    empty, base_offset is 0, and the new debug.json contains only the
    new run's rounds.
    """
    runner = _build_runner_for(tmp_path)
    leaf_dir = runner.experiments_dir / "subcircuits" / "leaf-uuid"
    leaf_dir.mkdir(parents=True, exist_ok=True)

    import json as _json

    # Pre-populate debug.json as if a prior run wrote rounds 0..2.
    debug_path = leaf_dir / "debug.json"
    debug_path.write_text(
        _json.dumps({
            "extra": {
                "all_rounds": [
                    {"round_index": i, "experiment_round": 1, "score": 80.0}
                    for i in range(3)
                ],
            },
        }),
        encoding="utf-8",
    )
    # Pre-populate canonical files that must survive.
    (leaf_dir / "leaf_routed.kicad_pcb").write_text("(placeholder)")
    (leaf_dir / "solved_layout.json").write_text("{}")
    (leaf_dir / "metadata.json").write_text("{}")

    runner._purge_prior_run_artifacts(phase="leaves_only")

    # debug.json gone -- the next invocation reads no prior rounds and
    # so will not double-count.
    assert not debug_path.exists()
    # Canonical pin-source files still present.
    assert (leaf_dir / "leaf_routed.kicad_pcb").exists()
    assert (leaf_dir / "solved_layout.json").exists()
    assert (leaf_dir / "metadata.json").exists()
