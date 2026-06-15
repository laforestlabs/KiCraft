"""Unit tests for the kicraft.tuning framework (fast; no place+route).

The expensive end-to-end check (a real replay eval on a fixture) lives in
``test_tuning_integration`` and is skipped by default — these tests cover the
pure logic: param mapping, optimizer convergence, reward/Pareto, the sqlite
cache, and the brief-level corpus split.
"""
from __future__ import annotations

import pytest

from kicraft.tuning import corpus, reward, space, store
from kicraft.tuning.evaluate import EvalResult
from kicraft.tuning.optimizer import load_optimizer, make_optimizer
from kicraft.tuning.screen import _pearson


# --- space ----------------------------------------------------------------

def test_space_encode_decode_roundtrip():
    active = space.all_param_names()[:6]
    v0 = space.initial_vector(active)
    overlay = space.decode(v0, active)
    v1 = space.encode(overlay, active)
    assert v0 == pytest.approx(v1, abs=1e-6)
    # decode produces only the active params, valid types
    assert set(overlay) == set(active)


def test_space_never_tune_excluded():
    names = space.all_param_names()
    assert "board_width_mm" not in names
    assert "board_height_mm" not in names


def test_space_int_param_denormalizes_to_int():
    # sa_refine_iterations is an int param
    val = space.denormalize("sa_refine_iterations", 0.5)
    assert isinstance(val, int)


# --- optimizer ------------------------------------------------------------

def _optimize_to_target(opt, target, iters):
    for _ in range(iters):
        if opt.stop():
            break
        X = opt.ask()
        F = [-sum((xi - ti) ** 2 for xi, ti in zip(x, target)) for x in X]
        opt.tell(X, F)
    return opt.best()


def test_cmaes_converges_and_checkpoints():
    target = [0.3, 0.7, 0.2, 0.9]
    opt = make_optimizer(4, x0=[0.5] * 4, popsize=10, seed=1)
    xb, fb = _optimize_to_target(opt, target, 60)
    assert xb == pytest.approx(target, abs=0.06)
    # checkpoint roundtrip continues without error
    opt2 = load_optimizer(opt.state_dict())
    assert opt2.dim == 4
    opt2.ask()


# --- reward / Pareto ------------------------------------------------------

def _res(board, seed, fab, drc, wall):
    return EvalResult("h", board, seed, "replay", 0 if fab else 7, fab,
                      drc, 0, drc, 10, 2, 100.0, wall)


def test_aggregate_seed_and_corpus():
    rs = [_res("b1", s, True, 0, 100) for s in range(3)]
    rs += [_res("b1", 9, False, 4, 100)]  # one failing seed
    ba = reward.aggregate_board(rs)
    assert ba.n_seeds == 4
    assert ba.fab_ready_rate == pytest.approx(0.75)
    assert ba.mean_drc == pytest.approx(1.0)


def test_dominance_and_front():
    # A: perfect & clean but slow; B: fast but a board fails
    A = reward.aggregate_results([_res("b1", s, True, 0, 120) for s in range(2)]
                                 + [_res("b2", s, True, 0, 120) for s in range(2)])
    B = reward.aggregate_results([_res("b1", s, True, 0, 80) for s in range(2)]
                                 + [_res("b2", s, False, 6, 80) for s in range(2)])
    # neither dominates (A better fab+drc, B better wall) -> both on front
    assert not reward.dominates(A, B)
    assert not reward.dominates(B, A)
    assert sorted(reward.pareto_front([A, B])) == [0, 1]
    # a strictly-better config dominates
    C = reward.aggregate_results([_res("b1", s, True, 0, 100) for s in range(2)]
                                 + [_res("b2", s, True, 0, 100) for s in range(2)])
    D = reward.aggregate_results([_res("b1", s, True, 1, 110) for s in range(2)]
                                 + [_res("b2", s, True, 1, 110) for s in range(2)])
    assert reward.dominates(C, D)
    assert reward.pareto_front([C, D]) == [0]


def test_scalarize_prefers_clean_board():
    good = reward.aggregate_results([_res("b1", s, True, 0, 100) for s in range(2)])
    bad = reward.aggregate_results([_res("b1", s, False, 5, 100) for s in range(2)])
    w = reward.SCALARIZATIONS["balanced"]
    assert reward.scalarize(good, w) > reward.scalarize(bad, w)


# --- store ----------------------------------------------------------------

def test_config_hash_stable_under_rounding_and_order():
    h1 = store.config_hash({"edge_margin_mm": 6.0, "force_attract_k": 0.02})
    h2 = store.config_hash({"force_attract_k": 0.02000004, "edge_margin_mm": 6.0})
    assert h1 == h2


def test_store_cache_roundtrip(tmp_path):
    s = store.Store(tmp_path / "t.db")
    h = store.config_hash({"edge_margin_mm": 6.0})
    r = EvalResult(h, "b1", 0, "replay", 0, True, 0, 0, 0, 10, 2, 100.0, 88.0)
    assert s.lookup(h, "b1", 0, "replay") is None
    s.record(r)
    got = s.lookup(h, "b1", 0, "replay")
    assert got == r
    s.upsert_config(h, {"edge_margin_mm": 6.0}, "test")
    assert s.get_overlay(h) == {"edge_margin_mm": 6.0}
    s.close()


# --- corpus ---------------------------------------------------------------

def test_split_by_brief_keeps_briefs_whole():
    ws = [
        corpus.Workspace(path=None, name=f"w{i}", stem="s", brief=b)
        for i, b in enumerate(["alpha", "alpha", "beta", "gamma", "delta", "eps"])
    ]
    corpus.split_by_brief(ws, holdout_frac=0.5, seed=0)
    by_brief: dict[str, set] = {}
    for w in ws:
        by_brief.setdefault(w.brief, set()).add(w.split)
    # the two 'alpha' workspaces must land in the SAME split
    assert by_brief["alpha"] == {next(iter(by_brief["alpha"]))}
    assert len(by_brief["alpha"]) == 1


def test_discover_corpus_on_fixtures():
    ws = corpus.discover_corpus(["tests/fixtures/replay_workspace"])
    names = {w.name for w in ws}
    assert "USB_PD_TRIGGER" in names


# --- screen helper --------------------------------------------------------

def test_pearson():
    assert _pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert _pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)
    assert _pearson([1, 1, 1], [1, 2, 3]) == 0.0  # constant -> 0


# --- benchmark brief set + freeze -----------------------------------------

def test_benchmark_set_well_formed():
    from kicraft.tuning import benchmark as B

    slugs = [e["slug"] for e in B.BENCHMARK_PROMPTS]
    assert len(slugs) == len(set(slugs)), "slugs must be unique"
    assert all(e["brief"].strip() for e in B.BENCHMARK_PROMPTS)
    assert len(B.briefs()) == len(B.BENCHMARK_PROMPTS)
    # every declared archetype trait is represented by >=1 brief
    assert set(B.coverage()) == set(B.ARCHETYPE_TRAITS)
    assert all(v >= 1 for v in B.coverage().values())
    assert len(B.BENCHMARK_PROMPTS) >= 24


def test_report_data_load_run(tmp_path):
    """report_data turns a run dir (sqlite + checkpoint) into chart series."""
    import json

    from kicraft.tuning import report_data
    from kicraft.tuning.store import Store, config_hash

    run = tmp_path / "run"
    run.mkdir()
    s = Store(run / "tuning.db")
    active = ["edge_margin_mm", "placement_clearance_mm"]
    ovA = {"edge_margin_mm": 4.0, "placement_clearance_mm": 2.0}
    ovB = {"edge_margin_mm": 8.0, "placement_clearance_mm": 3.0}
    hb, ha, hb2 = config_hash({}), config_hash(ovA), config_hash(ovB)
    s.upsert_config(ha, ovA); s.upsert_config(hb2, ovB)
    # gen 0: A (best, j=0.7) and B (j=0.3); holdout monitors A
    s.record_generation("t", 0, ha, scalarization="balanced", j=0.7, is_train=True,
                        fab_ready_rate=0.8, mean_drc=1.0, mean_wall_s=110.0)
    s.record_generation("t", 0, hb2, scalarization="balanced", j=0.3, is_train=True,
                        fab_ready_rate=0.5, mean_drc=3.0, mean_wall_s=120.0)
    s.record_generation("t", 0, ha, scalarization="balanced", j=0.65, is_train=False,
                        fab_ready_rate=0.75, mean_drc=1.0, mean_wall_s=110.0)
    # gen 1: A improves
    s.record_generation("t", 1, ha, scalarization="balanced", j=0.75, is_train=True,
                        fab_ready_rate=0.85, mean_drc=0.5, mean_wall_s=108.0)
    s.close()
    (run / "checkpoint.json").write_text(json.dumps({
        "run_id": "t", "gen": 2, "active": active, "scalarization": "balanced",
        "archive": [
            {"hash": hb, "overlay": {}, "fab": 0.6, "drc": 2.0, "wall": 100.0,
             "worst": 0.6, "baseline": True},
            {"hash": ha, "overlay": ovA, "fab": 0.85, "drc": 0.5, "wall": 108.0,
             "worst": 0.85},
            {"hash": hb2, "overlay": ovB, "fab": 0.5, "drc": 3.0, "wall": 120.0,
             "worst": 0.5},
        ],
    }))

    d = report_data.load_run(run)
    assert d["active_params"] == active
    assert d["n_gens"] == 2
    # time series: gen-best train + holdout
    g0 = next(g for g in d["gens"] if g["gen"] == 0)
    assert g0["train"]["hash"] == ha and g0["train"]["fab"] == 0.8  # best of gen 0
    assert g0["holdout"]["fab"] == 0.75
    # parameter convergence trace follows the gen-best overlay
    em = d["param_traces"]["edge_margin_mm"]
    assert [p["gen"] for p in em] == [0, 1]
    assert em[0]["value"] == 4.0 and 0.0 <= em[0]["norm"] <= 1.0
    # baseline + Pareto: B is dominated by the baseline, A and baseline are not
    assert d["baseline"]["fab"] == 0.6
    front = {p["hash"] for p in d["points"] if p["front"]}
    assert ha in front and hb in front and hb2 not in front
    # discovery finds the run
    assert run in report_data.discover_runs([tmp_path])


def test_freeze_corpus_roundtrip(tmp_path):
    import shutil
    from pathlib import Path

    from kicraft.tuning import corpus

    runs = tmp_path / "runs"
    runs.mkdir()
    shutil.copytree(Path("tests/fixtures/replay_workspace/USB_PD_TRIGGER"),
                    runs / "USB_PD_TRIGGER")
    dest = tmp_path / "corpus"
    frozen = corpus.freeze_corpus([runs], dest, holdout_frac=0.0)
    assert len(frozen) == 1
    assert (dest / "manifest.json").exists()
    # discover reads the frozen corpus + manifest split back
    ws = corpus.discover_corpus([dest])
    assert {w.name for w in ws} == {"USB_PD_TRIGGER"}
    assert ws[0].split == "train"


# --- orchestrator loop (mocked evaluator: fast + deterministic) -----------

def test_orchestrator_loop_checkpoint_report_resume(tmp_path, monkeypatch):
    """Full ask->tell->archive->checkpoint->report->resume with a synthetic,
    deterministic objective so the loop is exercised without real routing."""
    from kicraft.tuning import orchestrator as orch
    from kicraft.tuning.corpus import Workspace
    from kicraft.tuning.reward import CorpusObjectives
    from kicraft.tuning.store import config_hash

    boards = [Workspace(path=tmp_path, name=f"b{i}", stem="s", brief=f"brief{i}")
              for i in range(3)]
    monkeypatch.setattr(orch, "discover_corpus", lambda roots: list(boards))

    # Objective peaks at edge_margin_mm=4, placement_clearance_mm=2.
    def fake_eval(overlay, workspaces, seeds, **kw):
        edge = float(overlay.get("edge_margin_mm", 6.0))
        clr = float(overlay.get("placement_clearance_mm", 2.84))
        fab = max(0.0, 1.0 - (abs(edge - 4.0) / 10 + abs(clr - 2.0) / 4))
        obj = CorpusObjectives(fab, 10 * (1 - fab), 100.0, fab, len(workspaces))
        return obj, [], config_hash(overlay)

    monkeypatch.setattr(orch, "evaluate_overlay", fake_eval)

    out = tmp_path / "run"
    out.mkdir()
    (out / "screen.json").write_text(
        '{"active": ["edge_margin_mm", "placement_clearance_mm"], "frozen": [],'
        ' "correlations": {}, "n_samples": 0, "scalarization": "balanced",'
        ' "samples": []}'
    )
    settings = orch.TuneSettings(
        corpus_roots=[str(tmp_path)], out_dir=str(out), seeds=(0,),
        max_gens=3, popsize=6, holdout_frac=0.0,
    )
    report = orch.run_tuning(settings, run_id="t", log=lambda m: None)

    assert (out / "checkpoint.json").exists()
    assert (out / "report.json").exists()
    assert report["pareto_front"], "front should be non-empty"
    assert report["baseline"] is not None
    assert report["n_configs_evaluated"] > 1

    # best front config should improve fab over the baseline default
    base_fab = report["baseline"]["fab"]
    best_fab = max(a["fab"] for a in report["pareto_front"])
    assert best_fab >= base_fab

    # resume continues from the checkpoint without error and advances the gen
    settings2 = orch.TuneSettings(
        corpus_roots=[str(tmp_path)], out_dir=str(out), seeds=(0,),
        max_gens=5, popsize=6, holdout_frac=0.0,
    )
    orch.run_tuning(settings2, run_id="t", log=lambda m: None, resume=True)
    import json as _json
    gen = _json.loads((out / "checkpoint.json").read_text())["gen"]
    assert gen == 5
