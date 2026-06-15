"""CLI for the KiCraft tuning framework.

    python -m kicraft.tuning.cli corpus-stats --corpus DIR [DIR ...]
    python -m kicraft.tuning.cli screen       --corpus DIR --out RUNDIR [...]
    python -m kicraft.tuning.cli run          --corpus DIR --out RUNDIR [...]
    python -m kicraft.tuning.cli resume       --out RUNDIR
    python -m kicraft.tuning.cli report       --out RUNDIR
    python -m kicraft.tuning.cli promote      --out RUNDIR [--pick HASH|IDX]

Everything is CPU-bound and $0 LLM: it re-runs place+route on already-synthesized
workspaces. Long runs should go in the background (``run_in_background``) and at
low OS priority (evals self-nice).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--corpus", nargs="+", required=True,
                   help="corpus root dir(s) of synthesized workspaces")
    p.add_argument("--out", required=True, help="run output dir")
    p.add_argument("--mode", choices=["replay", "compose"], default="replay")
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated routing seeds for replication (default 0,1,2)")
    p.add_argument("--scalarization", choices=["correctness", "balanced", "speed"],
                   default="balanced")
    p.add_argument("--workers", type=int, default=None,
                   help="max concurrent evals (default: build slots - 1)")
    p.add_argument("--quality", default="fast",
                   choices=["fast", "draft", "good", "best"])
    p.add_argument("--timeout", type=int, default=1200, help="per-eval timeout (s)")
    p.add_argument("--top-k", type=int, default=10, help="active params after screening")
    p.add_argument("--screen-samples", type=int, default=40)
    p.add_argument("--holdout-frac", type=float, default=0.3)
    p.add_argument("--split-seed", type=int, default=0)


def _seeds(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip() != "")


def _settings_from_args(a: argparse.Namespace):
    from kicraft.tuning.orchestrator import TuneSettings

    return TuneSettings(
        corpus_roots=list(a.corpus), out_dir=a.out, mode=a.mode,
        seeds=_seeds(a.seeds), scalarization=a.scalarization,
        popsize=getattr(a, "popsize", None), max_gens=getattr(a, "gens", 30),
        max_workers=a.workers, quality=a.quality, timeout_s=a.timeout,
        holdout_frac=a.holdout_frac, split_seed=a.split_seed, top_k=a.top_k,
        n_screen_samples=a.screen_samples, cma_seed=getattr(a, "cma_seed", 0),
    )


def _cmd_corpus_stats(a) -> int:
    from kicraft.tuning.corpus import (corpus_stats, discover_corpus,
                                       split_by_brief)

    ws = discover_corpus(a.corpus)
    if not ws:
        print(f"no synthesized workspaces found under {a.corpus}", file=sys.stderr)
        return 1
    split_by_brief(ws, holdout_frac=a.holdout_frac, seed=a.split_seed)
    print(json.dumps(corpus_stats(ws), indent=2))
    for w in ws:
        brief = (w.brief[:60] + "...") if len(w.brief) > 63 else w.brief
        print(f"  {w.split or '-':8s} {w.name:28s} {brief}")
    return 0


def _cmd_screen(a) -> int:
    from kicraft.tuning.corpus import discover_corpus, split_by_brief, train
    from kicraft.tuning.screen import screen
    from kicraft.tuning.store import Store

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ws = discover_corpus(a.corpus)
    split_by_brief(ws, holdout_frac=a.holdout_frac, seed=a.split_seed)
    tr = train(ws)
    store = Store(str(out / "tuning.db"))
    sr = screen(
        tr, store=store, scratch_root=str(out / "scratch"),
        n_samples=a.screen_samples, seeds=_seeds(a.seeds), mode=a.mode,
        scalarization=a.scalarization, top_k=a.top_k, max_workers=a.workers,
        quality=a.quality, timeout_s=a.timeout,
        progress=lambda i, n, j: print(f"  screen {i}/{n} J={j:.3f}", flush=True),
    )
    sr.to_json(out / "screen.json")
    store.close()
    print("\nparam sensitivity (|corr| with routed J):")
    for p in sorted(sr.correlations, key=lambda k: abs(sr.correlations[k]),
                    reverse=True):
        mark = "*" if p in sr.active else " "
        print(f"  {mark} {p:34s} {sr.correlations[p]:+.3f}")
    print(f"\nactive ({len(sr.active)}): {sr.active}")
    return 0


def _cmd_run(a) -> int:
    from kicraft.tuning.orchestrator import SCREEN_NAME, run_tuning

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.active:
        from kicraft.tuning.space import all_param_names
        valid = set(all_param_names())
        active = [p.strip() for p in a.active.split(",") if p.strip()]
        bad = [p for p in active if p not in valid]
        if bad:
            print(f"unknown tunable param(s): {bad}\nvalid: {sorted(valid)}",
                  file=sys.stderr)
            return 2
        # Pre-seed screen.json so the orchestrator skips the screening pass.
        (out / SCREEN_NAME).write_text(json.dumps({
            "active": active, "frozen": [], "correlations": {},
            "n_samples": 0, "scalarization": a.scalarization, "samples": []}))
        print(f"[run] using {len(active)} given active params (screening skipped)")
    run_id = a.run_id or time.strftime("tune-%Y%m%dT%H%M%SZ", time.gmtime())
    run_tuning(_settings_from_args(a), run_id=run_id,
               log=lambda m: print(m, flush=True), resume=False)
    return 0


def _cmd_resume(a) -> int:
    from kicraft.tuning.orchestrator import CHECKPOINT_NAME, TuneSettings, run_tuning

    ck = Path(a.out) / CHECKPOINT_NAME
    if not ck.exists():
        print(f"no {CHECKPOINT_NAME} in {a.out}", file=sys.stderr)
        return 1
    saved = json.loads(ck.read_text(encoding="utf-8")).get("settings", {})
    saved["out_dir"] = a.out
    if a.gens is not None:
        saved["max_gens"] = a.gens
    settings = TuneSettings(**{k: v for k, v in saved.items()
                               if k in TuneSettings.__dataclass_fields__})
    settings.seeds = tuple(settings.seeds)
    run_id = json.loads(ck.read_text(encoding="utf-8")).get("run_id", "resumed")
    run_tuning(settings, run_id=run_id, log=lambda m: print(m, flush=True),
               resume=True)
    return 0


def _cmd_report(a) -> int:
    rep = Path(a.out) / "report.json"
    if not rep.exists():
        print(f"no report.json in {a.out}", file=sys.stderr)
        return 1
    d = json.loads(rep.read_text(encoding="utf-8"))
    base = d.get("baseline") or {}
    print(f"run {d['run_id']} | scalarization={d['scalarization']} | "
          f"{d['n_configs_evaluated']} configs | {d['n_train']} train boards")
    if base:
        print(f"baseline: fab={base['fab']:.2f} drc={base['drc']:.2f} "
              f"wall={base['wall']:.0f}s")
    print(f"\nPareto front ({len(d['pareto_front'])}):")
    print(f"  {'fab':>5} {'drc':>6} {'wall':>7}  config")
    for a_ in d["pareto_front"]:
        tag = " (baseline)" if a_.get("baseline") else ""
        n_keys = len(a_.get("overlay", {}))
        print(f"  {a_['fab']:5.2f} {a_['drc']:6.2f} {a_['wall']:7.0f}  "
              f"{a_['hash']} ({n_keys} keys){tag}")
    return 0


def _binom_two_sided_p(k: int, n: int) -> float:
    from math import comb
    if n == 0:
        return 1.0
    probs = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
    pk = probs[k]
    return min(1.0, sum(p for p in probs if p <= pk + 1e-12))


def _cmd_promote(a) -> int:
    """Validate a front config vs baseline on HOLDOUT with fresh seeds."""
    from kicraft.tuning import reward as R
    from kicraft.tuning.corpus import discover_corpus, holdout, split_by_brief
    from kicraft.tuning.runner import evaluate_overlay
    from kicraft.tuning.store import Store

    out = Path(a.out)
    rep = out / "report.json"
    if not rep.exists():
        print(f"no report.json in {a.out}; run first", file=sys.stderr)
        return 1
    d = json.loads(rep.read_text(encoding="utf-8"))
    front = [x for x in d["pareto_front"] if not x.get("baseline")]
    if not front:
        print("empty Pareto front", file=sys.stderr)
        return 1
    pick = front[0]
    if a.pick:
        match = [x for x in front if x["hash"] == a.pick]
        if match:
            pick = match[0]
        elif a.pick.isdigit() and int(a.pick) < len(front):
            pick = front[int(a.pick)]
    overlay = pick["overlay"]

    ws = discover_corpus(a.corpus)
    split_by_brief(ws, holdout_frac=a.holdout_frac, seed=a.split_seed)
    ho = holdout(ws)
    if not ho:
        print("no holdout boards (corpus too small / all train); cannot validate",
              file=sys.stderr)
        return 2
    seeds = _seeds(a.seeds)
    store = Store(str(out / "tuning.db"))
    scratch = str(out / "scratch")
    print(f"validating {pick['hash']} ({len(overlay)} keys) vs baseline on "
          f"{len(ho)} holdout board(s), fresh seeds {list(seeds)} ...")
    b_obj, b_res, _ = evaluate_overlay({}, ho, seeds, scratch_root=scratch,
                                       mode=a.mode, store=store,
                                       max_workers=a.workers, quality=a.quality,
                                       timeout_s=a.timeout, source="promote-baseline")
    c_obj, c_res, _ = evaluate_overlay(overlay, ho, seeds, scratch_root=scratch,
                                       mode=a.mode, store=store,
                                       max_workers=a.workers, quality=a.quality,
                                       timeout_s=a.timeout, source="promote-candidate")
    store.close()

    # paired fab-ready flips per (board, seed)
    b_map = {(r.board, r.seed): r for r in b_res}
    wins = losses = 0
    for r in c_res:
        b = b_map.get((r.board, r.seed))
        if b is None:
            continue
        if r.fab_ready and not b.fab_ready:
            wins += 1
        elif b.fab_ready and not r.fab_ready:
            losses += 1
    p = _binom_two_sided_p(wins, wins + losses)
    dom = R.dominates(c_obj, b_obj)
    no_fab_regress = c_obj.fab_ready_rate >= b_obj.fab_ready_rate - 1e-9

    print("\n  axis        baseline   candidate")
    print(f"  fab_ready   {b_obj.fab_ready_rate:8.3f}   {c_obj.fab_ready_rate:8.3f}")
    print(f"  mean_drc    {b_obj.mean_drc:8.3f}   {c_obj.mean_drc:8.3f}")
    print(f"  mean_wall_s {b_obj.mean_wall_s:8.1f}   {c_obj.mean_wall_s:8.1f}")
    print(f"\n  fab-ready flips: +{wins}/-{losses}  (sign-test p={p:.3f})")
    print(f"  candidate Pareto-dominates baseline: {dom}")
    verdict = ("PROMOTE" if no_fab_regress and (dom or (wins > losses and p < 0.05))
               else "HOLD")
    print(f"\n  VERDICT: {verdict}")
    if verdict == "PROMOTE":
        print("\n  Winning overlay (apply to DEFAULT_CONFIG in "
              "kicraft/autoplacer/config.py):")
        print("  " + json.dumps(overlay, indent=2, sort_keys=True).replace("\n", "\n  "))
        ov_path = out / "promoted_overlay.json"
        ov_path.write_text(json.dumps(overlay, indent=2, sort_keys=True))
        print(f"\n  (also written to {ov_path})")
    return 0


def _cmd_benchmark(a) -> int:
    """Print the diverse Phase-0 benchmark brief set."""
    from kicraft.tuning.benchmark import (ARCHETYPE_TRAITS, BENCHMARK_PROMPTS,
                                          briefs, coverage)

    if a.briefs_only:
        for b in briefs():
            print(b)
        return 0
    cov = coverage()
    print(f"{len(BENCHMARK_PROMPTS)} benchmark briefs across "
          f"{len(cov)} archetypes:\n")
    for arch, trait in ARCHETYPE_TRAITS.items():
        print(f"  {arch} ({cov.get(arch, 0)}) — {trait}")
        for e in BENCHMARK_PROMPTS:
            if e["archetype"] == arch:
                print(f"      [{e['slug']}] {e['brief']}")
        print()
    return 0


def _cmd_corpus_freeze(a) -> int:
    """Snapshot already-synthesized run dirs into a relocatable $0 corpus."""
    from kicraft.tuning.corpus import corpus_stats, freeze_corpus

    frozen = freeze_corpus(a.runs, a.dest, holdout_frac=a.holdout_frac,
                           split_seed=a.split_seed)
    print(f"froze {len(frozen)} workspace(s) -> {a.dest}")
    print(json.dumps(corpus_stats(frozen), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kicraft.tuning.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("benchmark", help="print the Phase-0 benchmark brief set")
    p.add_argument("--briefs-only", action="store_true",
                   help="print one brief per line (pipe into synthesis)")
    p.set_defaults(func=_cmd_benchmark)

    p = sub.add_parser("corpus-freeze",
                       help="freeze synthesized run dirs into a $0 corpus")
    p.add_argument("--runs", nargs="+", required=True,
                   help="synthesized run root(s) (e.g. a self-eval batch dir)")
    p.add_argument("--dest", required=True, help="corpus output dir")
    p.add_argument("--holdout-frac", type=float, default=0.3)
    p.add_argument("--split-seed", type=int, default=0)
    p.set_defaults(func=_cmd_corpus_freeze)

    p = sub.add_parser("corpus-stats")
    p.add_argument("--corpus", nargs="+", required=True)
    p.add_argument("--holdout-frac", type=float, default=0.3)
    p.add_argument("--split-seed", type=int, default=0)
    p.set_defaults(func=_cmd_corpus_stats)

    p = sub.add_parser("screen")
    _add_common(p)
    p.set_defaults(func=_cmd_screen)

    p = sub.add_parser("run")
    _add_common(p)
    p.add_argument("--gens", type=int, default=30)
    p.add_argument("--popsize", type=int, default=None)
    p.add_argument("--cma-seed", type=int, default=0)
    p.add_argument("--active", default=None,
                   help="comma-separated param names to tune; pre-seeds "
                        "screen.json and SKIPS the screening pass")
    p.add_argument("--run-id", default=None)
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("resume")
    p.add_argument("--out", required=True)
    p.add_argument("--gens", type=int, default=None)
    p.set_defaults(func=_cmd_resume)

    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("promote")
    p.add_argument("--corpus", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pick", default=None, help="front config hash or index")
    p.add_argument("--mode", choices=["replay", "compose"], default="replay")
    p.add_argument("--seeds", default="100,101,102,103,104")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--quality", default="fast", choices=["fast", "draft", "good", "best"])
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--holdout-frac", type=float, default=0.3)
    p.add_argument("--split-seed", type=int, default=0)
    p.set_defaults(func=_cmd_promote)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
