#!/usr/bin/env python3
"""Synthesize the diverse benchmark brief set into a frozen, committable tuning corpus.

Drives the SAME headless pipeline self-eval uses (``evaluate_one`` -> the five LLM
design stages + the deterministic build) for each brief in
``kicraft.tuning.benchmark.BENCHMARK_PROMPTS``, then lean-freezes the synthesized
``generated/<stem>/`` triple into ``<repo>/tuning_corpus/<slug>/`` (dropping the
bulky ``.experiments`` tree — replay regenerates it) and commits + pushes that one
board before moving on. So an interrupted run leaves a clean, partial, pushed
corpus; re-running skips boards already present.

LLM-costed (design stage, ~$0.03/brief; ~$1-3 for the full set); the build is
free CPU. Set SYNTH_BUDGET_USD to cap spend (default 8). Run in the background:

    nohup .venv/bin/python scripts/synth_benchmark_corpus.py > synth.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "tuning_corpus"
WORK = Path.home() / ".kicraft" / "tuning" / "benchmark_synth"
BUDGET_USD = float(os.environ.get("SYNTH_BUDGET_USD", "8"))
HOLDOUT_FRAC = 0.3  # must match the tuner's run/promote default so splits agree
SPLIT_SEED = 0


def _log(msg: str) -> None:
    print(msg, flush=True)


def _has_triple(d: Path) -> bool:
    from kicraft.tuning.workspace import discover_stem
    try:
        discover_stem(d)
        return True
    except Exception:  # noqa: BLE001
        return False


def _find_generated(rundir: Path) -> Path | None:
    """The synthesized ``generated/<stem>/`` dir with a full kicad triple."""
    gen = rundir / "generated"
    if not gen.is_dir():
        return None
    for d in sorted(p for p in gen.iterdir() if p.is_dir()):
        if _has_triple(d):
            return d
    return None


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(REPO),
                          capture_output=True, text=True)


def _rebuild_manifest(prompts: list[dict]) -> list:
    """Author the corpus manifest from the known slug->brief map (authoritative
    briefs), with a deterministic brief-level train/holdout split."""
    from kicraft.tuning.corpus import Workspace, split_by_brief, write_manifest
    from kicraft.tuning.workspace import discover_stem

    ws = []
    for e in prompts:
        d = CORPUS / e["slug"]
        if not d.is_dir():
            continue
        try:
            stem = discover_stem(d)
        except Exception:  # noqa: BLE001
            continue
        ws.append(Workspace(path=d, name=e["slug"], stem=stem, brief=e["brief"]))
    split_by_brief(ws, holdout_frac=HOLDOUT_FRAC, seed=SPLIT_SEED)
    write_manifest(CORPUS, ws)
    return ws


def _commit_push(slug: str, n: int, total: int, brief: str) -> None:
    _git("add", str(CORPUS))
    msg = (f"data(tuning): benchmark corpus board {slug} ({n}/{total})\n\n"
           f"Brief: {brief}\n\n"
           "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>")
    r = _git("commit", "-m", msg)
    if r.returncode != 0 and "nothing to commit" not in (r.stdout + r.stderr):
        _log(f"  git commit warn: {(r.stderr or r.stdout).strip()[:160]}")
        return
    p = _git("push", "origin", "HEAD")
    if p.returncode != 0:
        _log(f"  git push warn: {(p.stderr or p.stdout).strip()[:160]}")


def main() -> int:
    from kicraft.eval.self_eval import evaluate_one
    from kicraft.server.client import CappedOpenRouterClient
    from kicraft.server.config import Settings
    from kicraft.tuning import corpus as C
    from kicraft.tuning.benchmark import BENCHMARK_PROMPTS

    CORPUS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    client = CappedOpenRouterClient(Settings.from_env())

    total = len(BENCHMARK_PROMPTS)
    spent = done = skipped = failed = 0
    spent = 0.0
    _log(f"synthesizing {total} benchmark briefs -> {CORPUS} (budget ${BUDGET_USD})")

    for i, e in enumerate(BENCHMARK_PROMPTS):
        slug, brief = e["slug"], e["brief"]
        dest = CORPUS / slug
        if dest.is_dir() and _has_triple(dest):
            _log(f"[{i + 1}/{total}] {slug}: already present — skip")
            skipped += 1
            continue
        if spent >= BUDGET_USD:
            _log(f"budget ${BUDGET_USD:.2f} reached (spent ${spent:.2f}); stopping")
            break

        _log(f"[{i + 1}/{total}] {slug}: synthesizing ...")
        t0 = time.time()
        rec = evaluate_one(client, i, e, WORK, judge_model=None,
                           skip_judge=True, build_timeout_s=2400)
        cost = float(rec.get("design_cost_usd") or 0.0)
        spent += cost
        gen = _find_generated(Path(rec["rundir"]))
        if gen is None:
            _log(f"  FAILED: design_status={rec.get('design_status')} "
                 f"err={str(rec.get('design_error'))[:120]} "
                 f"(${cost:.3f}, cum ${spent:.2f})")
            failed += 1
            continue

        C.freeze_workspace(gen, dest, brief=brief, lean=True)
        _rebuild_manifest(BENCHMARK_PROMPTS)
        _commit_push(slug, i + 1, total, brief)
        done += 1
        _log(f"  OK build_rc={rec.get('build_rc')} ({rec.get('build_label')}) "
             f"${cost:.3f}  cum ${spent:.2f}  {time.time() - t0:.0f}s")

    _log(f"\nDONE: {done} added, {skipped} skipped, {failed} failed; "
         f"spend ~${spent:.2f}")
    ws = C.discover_corpus([CORPUS])
    C.split_by_brief(ws, holdout_frac=HOLDOUT_FRAC, seed=SPLIT_SEED)
    _log(json.dumps(C.corpus_stats(ws), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
