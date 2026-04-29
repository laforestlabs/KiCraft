"""Regression tests for per-run leaf-artifact cleanup.

The Monitor tab's score plot and round timeline are built from each
leaf's ``debug.json`` (key ``extra.all_rounds``). When a fresh
autoexperiment run is started, the GUI must show ONLY rounds from
that run -- not stale rounds that piled up across earlier runs.

Two pieces have to work together for the user-visible "rounds 1..N
in an N-round run" property to hold:

1. ``ExperimentRunner._purge_prior_run_artifacts`` deletes per-leaf
   ``round_NNNN_*`` snapshots and ``debug.json`` at the start of
   every run, including ``--leaves-only`` and ``--parents-only``.
   Canonical pin-source files (``leaf_routed.kicad_pcb``,
   ``solved_layout.json``, ``metadata.json``, ``renders/``) are
   preserved so ``pins.json`` references survive.

2. The leaf solver's round_index resets to 0 on each invocation
   (no ``base_offset`` continuation across runs). Together with (1)
   this means a fresh 3-round run produces round_NNNN files
   round_0000 / round_0001 / round_0002 -- not round_0014 / 0015 /
   0016 stacked on top of a prior run's leftovers.

These tests lock the cleanup contract.
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


def test_parents_only_purge_also_drops_per_leaf_snapshots(tmp_path: Path):
    """parents-only typically doesn't re-solve leaves, so the score
    plot for each leaf shouldn't keep showing prior leaves-only data
    once a new run starts. Same cleanup applies."""
    runner = _build_runner_for(tmp_path)
    leaf = runner.experiments_dir / "subcircuits" / "leafA"
    _seed_leaf_dir(leaf, round_indices=[0, 5, 14])

    runner._purge_prior_run_artifacts(phase="parents_only")

    assert (leaf / "leaf_routed.kicad_pcb").exists()
    assert not (leaf / "debug.json").exists()
    assert not list(leaf.glob("round_*_*.kicad_pcb"))


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
# Cumulative round_index regression
# ---------------------------------------------------------------------------


def test_solve_leaf_resets_round_index_each_run(monkeypatch: pytest.MonkeyPatch):
    """Smoke check that the cumulative ``base_offset`` logic is gone:
    even with a populated debug.json declaring prior rounds, the leaf
    solver no longer reads it for offset purposes. We exercise this by
    grepping the source for the removed symbols rather than running a
    full leaf solve (which requires pcbnew + a real schematic).
    """
    src = (
        Path(__file__).resolve().parent.parent
        / "kicraft"
        / "cli"
        / "solve_subcircuits.py"
    ).read_text(encoding="utf-8")

    # No reference to the cumulative-round-index plumbing should remain
    # outside of the explanatory comment that documents why it's gone.
    assert "base_offset = max_idx + 1" not in src
    assert "prior_all_rounds: list" not in src
    assert "prior_rounds: list" not in src

    # And the loop must increment from 0, not from a base_offset.
    assert "round_index = local_round_index" in src
