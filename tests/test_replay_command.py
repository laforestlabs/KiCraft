"""kicraft replay: deterministic place+route re-run on a fixed synthesized
workspace (NO synthesis / LLM).

Two layers:

* Fast unit tests (no pcbnew, no router): workspace resolution in both input
  modes, missing-artifact gating, the synthesis-skip guarantee, and bounded
  two-phase search scheduling -- all driven with monkeypatched seams.
* An opt-in end-to-end determinism test (``KICRAFT_REPLAY_E2E=1``): runs
  ``replay --project ... --no-route`` twice on the committed
  ``tests/fixtures/replay_workspace`` and asserts the per-leaf PLACEMENT
  (``leaf_placed.kicad_pcb``) is identical across runs. The composed
  parent is intentionally NOT asserted byte-stable -- it consumes the routed
  leaf boards and so inherits the autorouter's best-effort nondeterminism.
"""

from __future__ import annotations

import contextlib
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import kicraft.design.cli_app as cli_app

FIXTURE = Path(__file__).parent / "fixtures" / "replay_workspace" / "USB_PD_TRIGGER"


def _stub_workspace(tmp_path: Path, stem: str = "DEMO") -> Path:
    """A minimal synthesized-workspace shape: root + one leaf sch, seed pcb,
    project file, autoplacer json. Contents are placeholders -- the unit tests
    monkeypatch everything that would actually parse them."""
    d = tmp_path / stem
    d.mkdir()
    (d / f"{stem}.kicad_sch").write_text("(kicad_sch root)\n", encoding="utf-8")
    (d / "LEAF_A.kicad_sch").write_text("(kicad_sch leaf)\n", encoding="utf-8")
    (d / f"{stem}.kicad_pcb").write_text("(kicad_pcb)\n", encoding="utf-8")
    (d / f"{stem}.kicad_pro").write_text("{}\n", encoding="utf-8")
    (d / f"{stem}_autoplacer.json").write_text("{}\n", encoding="utf-8")
    return d


# ---- workspace resolution ----------------------------------------------------


def test_resolve_project_mode_discovers_artifacts(tmp_path):
    d = _stub_workspace(tmp_path)
    args = SimpleNamespace(project=str(d), state=None, out_dir=None)
    state, sp, arts, stem, pdir, root, pcb = cli_app._resolve_synthesized_workspace(args)
    assert stem == "DEMO"
    assert pdir == d.resolve()
    assert root == d / "DEMO.kicad_sch"
    assert pcb == d / "DEMO.kicad_pcb"
    assert sp is None  # no sibling state.json -> project-only mode
    assert [p.name for p in arts.leaf_schs] == ["LEAF_A.kicad_sch"]


def test_persist_artifacts_without_state_path_updates_memory_only(tmp_path):
    project = _stub_workspace(tmp_path)
    state, state_path, artifacts, *_ = cli_app._resolve_synthesized_workspace(
        SimpleNamespace(project=str(project), state=None, out_dir=None)
    )

    cli_app._persist_artifacts(state, state_path, artifacts)

    assert state_path is None
    assert state.artifacts is artifacts
    assert not (project / "state.json").exists()


def test_resolve_project_mode_missing_pcb_cannot_identify(tmp_path):
    """In project mode, the stem is identified from the full triad; drop the
    pcb and there is no identifiable project (clear, actionable error)."""
    d = _stub_workspace(tmp_path)
    (d / "DEMO.kicad_pcb").unlink()
    args = SimpleNamespace(project=str(d), state=None, out_dir=None)
    with pytest.raises(cli_app._ReplayInputError) as e:
        cli_app._resolve_synthesized_workspace(args)
    assert "could not identify" in str(e.value)


def test_resolve_state_mode_missing_artifact_raises(tmp_path):
    """In state.json mode the stem comes from the state, so a genuinely missing
    artifact is reported as such (lists the missing file, rc 3)."""
    from kicraft.design.models import ConversationState

    st = ConversationState(project_stem="DEMO")
    state_path = tmp_path / "state.json"
    state_path.write_text(st.model_dump_json(), encoding="utf-8")
    proj = tmp_path / "DEMO"
    proj.mkdir()
    (proj / "DEMO.kicad_sch").write_text("(kicad_sch)\n", encoding="utf-8")
    (proj / "DEMO.kicad_pro").write_text("{}\n", encoding="utf-8")
    # DEMO.kicad_pcb intentionally absent

    args = SimpleNamespace(project=None, state=str(state_path), out_dir=str(tmp_path))
    with pytest.raises(cli_app._ReplayInputError) as e:
        cli_app._resolve_synthesized_workspace(args)
    assert "DEMO.kicad_pcb" in str(e.value)


def test_resolve_ambiguous_project_raises(tmp_path):
    d = _stub_workspace(tmp_path)
    for ext in ("kicad_pro", "kicad_pcb", "kicad_sch"):
        (d / f"OTHER.{ext}").write_text("x", encoding="utf-8")
    args = SimpleNamespace(project=str(d), state=None, out_dir=None)
    with pytest.raises(cli_app._ReplayInputError):
        cli_app._resolve_synthesized_workspace(args)


def test_resolve_missing_autoplacer_is_warning_not_error(tmp_path, capsys):
    d = _stub_workspace(tmp_path)
    (d / "DEMO_autoplacer.json").unlink()
    args = SimpleNamespace(project=str(d), state=None, out_dir=None)
    res = cli_app._resolve_synthesized_workspace(args)
    assert res[3] == "DEMO"  # still resolves
    assert "autoplacer.json absent" in capsys.readouterr().err


def test_resolve_neither_mode_raises(tmp_path):
    args = SimpleNamespace(project=None, state=None, out_dir=None)
    with pytest.raises(cli_app._ReplayInputError):
        cli_app._resolve_synthesized_workspace(args)


# ---- argparse wiring ---------------------------------------------------------





def test_replay_flag_parsing(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli_app, "_cmd_replay", lambda args: seen.update(vars(args)) or 0)
    cli_app.main(["replay", "--project", "/tmp/x", "--seed", "9", "--no-route", "--no-fab"])
    assert seen["seed"] == 9
    assert seen["route"] is False
    assert seen["no_fab"] is True


# ---- the synthesis-skip guarantee + seed/route threading ---------------------


def _patch_replay_seams(monkeypatch, recorder):
    """Make `_cmd_replay` callable without a real layout run: no degenerate
    check, a no-op build slot, and a `_layout_route_fab` that records the
    knobs it received and returns rc 0."""
    monkeypatch.setattr(cli_app, "_degenerate_hierarchy_error", lambda root: None)
    monkeypatch.setattr(
        "kicraft.build_slots.build_slot",
        lambda **k: contextlib.nullcontext(),
    )

    def fake_lrf(args, state, sp, arts, results, stem, pdir, root, pcb):
        recorder.update(
            seed=args.seed,
            route=args.route,
            quality=args.quality,
            no_fab=getattr(args, "no_fab", None),
            root=root,
        )
        return 0

    monkeypatch.setattr(cli_app, "_layout_route_fab", fake_lrf)


def test_replay_never_calls_synth_and_threads_seed(tmp_path, monkeypatch):
    d = _stub_workspace(tmp_path)

    def boom(*a, **k):
        raise AssertionError("run_synth must NOT run during replay")

    monkeypatch.setattr(cli_app, "run_synth", boom)
    rec = {}
    _patch_replay_seams(monkeypatch, rec)

    args = SimpleNamespace(
        project=str(d),
        state=None,
        out_dir=None,
        quality="fast",
        seed=7,
        route=False,
        no_fab=False,
        no_archive=True,
    )
    rc = cli_app._cmd_replay(args)
    assert rc == 0
    assert rec["seed"] == 7
    assert rec["route"] is False
    assert rec["quality"] == "fast"
    # project mode has no state.json -> fab is force-skipped (needs the BOM)
    assert rec["no_fab"] is True


def test_routed_replay_preflights_krt_once(tmp_path, monkeypatch):
    d = _stub_workspace(tmp_path)
    rec = {}
    _patch_replay_seams(monkeypatch, rec)
    calls = []
    monkeypatch.setattr(
        cli_app,
        "_preflight_project_router",
        lambda project_dir: calls.append(Path(project_dir)) or {},
    )
    args = SimpleNamespace(
        project=str(d),
        state=None,
        out_dir=None,
        quality="fast",
        seed=0,
        route=True,
        no_fab=True,
        no_archive=True,
    )
    assert cli_app._cmd_replay(args) == 0
    assert calls == [d]
    assert rec["route"] is True


def test_replay_pins_deterministic_env(tmp_path, monkeypatch):
    d = _stub_workspace(tmp_path)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    _patch_replay_seams(monkeypatch, {})
    args = SimpleNamespace(
        project=str(d),
        state=None,
        out_dir=None,
        quality="fast",
        seed=0,
        route=False,
        no_fab=True,
        no_archive=True,
    )
    assert cli_app._cmd_replay(args) == 0
    assert os.environ["PYTHONHASHSEED"] == "0"


def test_replay_detects_synthesis_mutation(tmp_path, monkeypatch):
    """If anything rewrites the root schematic during a replay, the run fails
    loudly (rc 8) -- the no-synthesis invariant must hold."""
    from kicraft import build_slots

    d = _stub_workspace(tmp_path)
    monkeypatch.setattr(cli_app, "_degenerate_hierarchy_error", lambda root: None)
    monkeypatch.setattr(build_slots, "build_slot", lambda **k: contextlib.nullcontext())

    def mutating(args, state, sp, arts, results, stem, pdir, root, pcb):
        root.write_text("(kicad_sch root) MUTATED\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(cli_app, "_layout_route_fab", mutating)
    args = SimpleNamespace(
        project=str(d),
        state=None,
        out_dir=None,
        quality="fast",
        seed=0,
        route=False,
        no_fab=True,
        no_archive=True,
    )
    assert cli_app._cmd_replay(args) == 8


# ---- shared wall-clock ownership across layout phases ------------------------


@pytest.mark.parametrize(
    ("leaf_seconds", "parent_round_seconds", "expected_rc"),
    [(768.0, 362.0, 0), (617.0, 593.0, 6), (1053.0, 362.0, 6)],
)
def test_run_layout_search_uses_only_remaining_combined_budget(
    tmp_path, monkeypatch, leaf_seconds, parent_round_seconds, expected_rc,
):
    import kicraft.cli.autoexperiment as experiment
    from kicraft.cli._round_scheduler import Finalize, RoundScheduler

    project = _stub_workspace(tmp_path)
    clock = [0.0]
    monkeypatch.setenv("KICRAFT_BUILD_MAX_WALL_S", "1620")
    monkeypatch.setenv("KICRAFT_LEAF_SOLVE_MAX_WALL_S", "351")
    monkeypatch.setattr(cli_app.time, "monotonic", lambda: clock[0])

    def bounded_search(argv):
        args = experiment.parse_args(argv)
        if args.leaves_only:
            clock[0] += leaf_seconds
            return 0
        started = clock[0]
        scheduler = RoundScheduler(rounds=args.rounds, max_wall_s=args.max_wall_s)
        for _ in range(2):
            plan = scheduler.plan_next(
                elapsed_s=clock[0] - started, stop_requested=False,
            )
            if isinstance(plan, Finalize):
                return 6
            clock[0] += parent_round_seconds
            scheduler.observe_round_end(score=20.0, subprocesses_ok=plan.round_num == 2)
            scheduler.observe_duration(parent_round_seconds)
            if plan.round_num == 2:
                return 0
        pytest.fail("the permitted second search round was never reached")

    monkeypatch.setattr(experiment, "main", bounded_search)
    rc = cli_app._run_layout(
        "good", project / "DEMO.kicad_sch", project / "DEMO.kicad_pcb",
    )
    assert rc == expected_rc
    assert clock[0] <= 1620 * 0.95


def test_run_layout_does_not_start_parent_after_combined_deadline(tmp_path, monkeypatch):
    import kicraft.cli.autoexperiment as experiment

    project = _stub_workspace(tmp_path)
    clock = [0.0]
    monkeypatch.setenv("KICRAFT_BUILD_MAX_WALL_S", "1620")
    monkeypatch.setenv("KICRAFT_LEAF_SOLVE_MAX_WALL_S", "351")
    monkeypatch.setattr(cli_app.time, "monotonic", lambda: clock[0])

    def exhausted_leaf_search(argv):
        args = experiment.parse_args(argv)
        if not args.leaves_only:
            pytest.fail("a parent search started after the shared deadline")
        clock[0] += 1620 * 0.95
        return 0

    monkeypatch.setattr(experiment, "main", exhausted_leaf_search)
    assert cli_app._run_layout(
        "good", project / "DEMO.kicad_sch", project / "DEMO.kicad_pcb",
    ) != 0


# ---- end-to-end determinism (opt-in; slow; spawns layout subprocesses) -------


def _leaf_placements(project_dir: Path) -> dict[str, dict]:
    """Per-leaf footprint geometry from each ``leaf_placed.kicad_pcb``
    (the deterministic placement output), keyed by the leaf artifact dir name."""
    import pcbnew

    out: dict[str, dict] = {}
    for p in glob.glob(
        str(project_dir / ".experiments" / "subcircuits" / "*" / "leaf_placed.kicad_pcb")
    ):
        board = pcbnew.LoadBoard(p)
        out[Path(p).parent.name] = {
            fp.GetReference(): (
                round(pcbnew.ToMM(fp.GetPosition().x), 4),
                round(pcbnew.ToMM(fp.GetPosition().y), 4),
                round(fp.GetOrientationDegrees(), 3),
            )
            for fp in board.GetFootprints()
        }
    return out


def _replay_once(dest: Path) -> None:
    shutil.copytree(FIXTURE, dest)
    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "kicraft.design.cli_app",
            "replay",
            "--project",
            str(dest),
            "--quality",
            "fast",
            "--no-route",
            "--no-fab",
            "--seed",
            "0",
        ],
        cwd=str(FIXTURE.parent.parent.parent),  # repo root
    ).returncode
    assert rc == 0, f"replay exited {rc}"


@pytest.mark.skipif(
    not os.environ.get("KICRAFT_REPLAY_E2E"),
    reason="set KICRAFT_REPLAY_E2E=1 to run (slow; spawns compose)",
)
@pytest.mark.skipif(
    not (FIXTURE / ".experiments").is_dir(),
    reason="frozen-leaf parent-corpus fixture missing",
)
def test_parent_corpus_matches_golden():
    """The parent-placement gate (Levers 2.1/2.3): compose-only on the committed
    frozen leaf artifacts must reproduce the golden parent placement."""
    pytest.importorskip("pcbnew")
    repo_root = Path(__file__).resolve().parent.parent
    rc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "replay_corpus.py"), "--mode", "parent"],
        cwd=str(repo_root),
    ).returncode
    assert rc == 0, "parent corpus drifted from golden (or errored)"


@pytest.mark.skipif(
    not os.environ.get("KICRAFT_REPLAY_E2E"),
    reason="set KICRAFT_REPLAY_E2E=1 to run (slow; spawns the layout engine)",
)
@pytest.mark.skipif(not FIXTURE.is_dir(), reason="replay fixture missing")
def test_replay_placement_is_deterministic(tmp_path):
    pytest.importorskip("pcbnew")
    a = tmp_path / "run_a" / "USB_PD_TRIGGER"
    b = tmp_path / "run_b" / "USB_PD_TRIGGER"
    _replay_once(a)
    _replay_once(b)

    pa = _leaf_placements(a)
    pb = _leaf_placements(b)
    assert pa, "no leaf placements were produced"
    assert set(pa) == set(pb), "leaf set differs across runs"
    for leaf in pa:
        assert pa[leaf] == pb[leaf], (
            f"leaf {leaf} placement diverged across two replays:\n"
            f"  run a: {pa[leaf]}\n  run b: {pb[leaf]}"
        )
