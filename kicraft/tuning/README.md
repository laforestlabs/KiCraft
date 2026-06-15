# kicraft.tuning — cross-corpus default-config auto-tuner

Tunes KiCraft's **global** `DEFAULT_CONFIG` (placement + routing-effort knobs) to
maximize *routed* outcomes across a corpus of already-synthesized boards, using
spare CPU. Every evaluation re-runs place+route on a frozen workspace — **$0 LLM
cost**, placement byte-deterministic, only FreeRouting stochastic.

This is the missing **outer** loop. `autoexperiment.py` already searches params
*per build* (keeping the best by `PlacementScore`); this package optimizes the
*default* against the true objective nothing else targets: fab-ready rate, DRC
cleanliness, and build wall-time, as a **Pareto** trade-off.

## Pipeline

```
corpus-stats   discover synthesized workspaces; brief-level train/holdout split
screen         random-sample configs, correlate each param with routed J,
               keep the top-k sensitive params as CMA's active dims
run            CMA-ES (cma pkg) over the active dims on the TRAIN corpus;
               K-seed replication (CRN) averages out routing noise; monitors
               the gen-best on HOLDOUT; checkpoints every generation
resume         continue an interrupted run from checkpoint.json (cached evals free)
report         print the Pareto front + baseline
promote        re-validate a front config vs the current default on HOLDOUT with
               fresh seeds; paired sign-test + dominance -> PROMOTE / HOLD
```

## Phase 0 — build the corpus (one-time, LLM-costed; run on the cloud box)

The tuner is only as trustworthy as its corpus. `benchmark.py` holds ~28
archetype-diverse briefs (single-passive, USB-C/connector, fine-pitch, RF,
power/thermal, mixed THT+SMT, hierarchical, dense-I/O). Synthesize them **once**,
then freeze the results so every later tuning run re-routes them at $0.

```bash
PY=.venv/bin/python
$PY -m kicraft.tuning.cli benchmark                 # review the brief set
$PY -m kicraft.tuning.cli benchmark --briefs-only   # one per line, to drive synthesis

# Synthesize each brief end-to-end (design + build) on the CLOUD box — reuse the
# existing self-eval / build pipeline; this is the only step that spends LLM $.
# Then freeze the produced run dirs into a relocatable corpus + manifest:
$PY -m kicraft.tuning.cli corpus-freeze --runs <self_eval_batch_dir> \
    --dest path/to/tuning_corpus --holdout-frac 0.3
```

## Quick start (after the corpus exists)

```bash
PY=.venv/bin/python
CORPUS=path/to/tuning_corpus           # frozen synthesized workspaces
OUT=~/.kicraft/tuning/run1

# 0. sanity-check the corpus + split
$PY -m kicraft.tuning.cli corpus-stats --corpus $CORPUS

# 1. screen down to the sensitive params (writes $OUT/screen.json)
$PY -m kicraft.tuning.cli screen --corpus $CORPUS --out $OUT \
    --screen-samples 120 --seeds 0,1,2 --top-k 10

# 2. run the optimizer (long; background it). One scalarization per run; run
#    'correctness' / 'balanced' / 'speed' for full Pareto coverage.
$PY -m kicraft.tuning.cli run --corpus $CORPUS --out $OUT \
    --gens 30 --seeds 0,1,2 --scalarization balanced

# 3. inspect + validate a winner, then apply its overlay to DEFAULT_CONFIG
$PY -m kicraft.tuning.cli report  --out $OUT
$PY -m kicraft.tuning.cli promote --corpus $CORPUS --out $OUT
```

## Design notes

- **Eval modes.** `replay` (default) re-runs leaf placement + parent compose +
  route, so leaf-placement params are observable — the right mode for tuning
  placement. `compose` re-runs only parent compose+route on frozen leaves
  (cheaper, but blind to leaf params); useful for parent-only A/B.
- **Determinism.** Evals run as subprocesses with `PYTHONHASHSEED=0` +
  single-thread BLAS pinned (`PINNED_ENV`); this must be set before interpreter
  start, which is why eval is a subprocess, not in-process.
- **Reward, not proxy.** The objective is the routed DRC outcome
  (`_verify_routed_board`), never `PlacementScore` (the inner loop already
  maximizes that — optimizing it here would be circular).
- **Never starve user builds.** Each eval self-gates on a `build_slot` and runs
  under `nice`/`ionice`; concurrency defaults to `slot_count() - 1`.
- **Overfitting guard.** Split is **by brief**; CMA optimizes on train only;
  holdout is monitored but never fed back; promotion requires no holdout
  regression. The reward also penalizes the worst board (`robust` weight).
- **Fab-constrained knobs are off-limits.** Only layout heuristics + routing
  *effort* (passes/auto-clearance) are tuned — never widths/drills/clearances
  dictated by the fab. See the comment block above `CONFIG_SEARCH_SPACE`.

## Caching / resume

`store.py` keys every result on `(config_hash, board, seed, mode)`. A re-run, a
`resume`, or a larger K only evaluates cache misses. `config_hash` canonicalizes
the overlay (rounded floats, sorted keys) so numerically-equal configs collide.
