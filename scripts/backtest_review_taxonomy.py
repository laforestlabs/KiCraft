#!/usr/bin/env python3
"""Offline backtest of the electrical-review severity clamp + lazy corroboration.

Replays the frozen bakeoff (`logs/bakeoff/<ts>/results.jsonl` -- every model x effort
arm x design, with each pass's RAW findings) through the PRODUCTION
`clamp_findings` + corroboration-merge logic, deterministically and with NO API
calls. It answers the two questions the gate change must satisfy:

  1. RECALL -- do the confirmed natural blockers still hard-block after clamp +
     2-pass corroboration, for the default model (minimax-m3)?
  2. OVER-BLOCK -- does the block rate on DRC-sound boards drop (the whole point)?

It also prints the documented recall risk (run_02 R-2R, which the default model
catches only intermittently) rather than hiding it.

Usage:  .venv/bin/python scripts/backtest_review_taxonomy.py [BAKEOFF_DIR]
Exit code 1 if a must-block invariant regresses; 0 otherwise (or skip if no corpus).
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

from kicraft.design.synthesis.electrical_review import (
    _findings_agree,
    clamp_findings,
)

DEFAULT_MODEL = "minimax"
# The natural, confirmed blockers that MUST still block (synthetics are excluded:
# section 9.16-9.20 catch them deterministically pre-LLM). run_02 is reported but
# NOT asserted -- the default model catches R-2R only intermittently, an accepted
# fail-toward-shipping trade (it still surfaces as a warning; the glm fallback
# catches it 3/3).
MUST_BLOCK = {"run_27_stepper-a4988", "run_15_buck-3a", "run_08_rs485-terminal",
              "run_19_relay-quad", "run_22_esp32-dual-motor"}
REPORT_ONLY = {"run_02_r2r-dac"}


def _bakeoff_dir(argv) -> Path:
    if len(argv) > 1:
        return Path(argv[1])
    root = Path(__file__).resolve().parents[1] / "logs" / "bakeoff"
    cands = sorted(p for p in root.glob("*") if (p / "results.jsonl").exists())
    if not cands:
        print(f"no bakeoff corpus under {root} -- nothing to backtest (skip).")
        sys.exit(0)
    return cands[-1]


def _raw_blocked(passf) -> bool:
    """Pre-fix gate: any model-severity blocker hard-blocks."""
    return any(f.get("severity") == "blocker" for f in passf)


def _clamp_blocked(passf) -> bool:
    """Clamp-only gate: a blocker-eligible blocker survives the severity ceiling."""
    return any(f["severity"] == "blocker" for f in clamp_findings(passf))


def _corroborated(pass_a, pass_b) -> bool:
    """The production 2-pass merge: a clamped blocker in pass A sticks iff a clamped
    blocker in pass B agrees (same category + overlapping/empty refdes)."""
    a = clamp_findings(pass_a)
    b = clamp_findings(pass_b)
    a_blk = [f for f in a if f["severity"] == "blocker"]
    if not a_blk:
        return False
    b_blk = [f for f in b if f["severity"] == "blocker"]
    return any(any(_findings_agree(c, o) for o in b_blk) for c in a_blk)


def main() -> int:
    bdir = _bakeoff_dir(sys.argv)
    labels = json.loads((bdir / "labels.json").read_text())
    rows = [json.loads(l) for l in (bdir / "results.jsonl").read_text().splitlines() if l.strip()]

    role = {d["design_id"]: d for d in labels["designs"] + labels["synthetics"]}
    label_area = {did: (d["true_blockers"][0]["area"] if d.get("true_blockers") else "")
                  for did, d in role.items()}
    natural_blockers = [d["design_id"] for d in labels["designs"] if d.get("true_blockers")]
    sound = [d["design_id"] for d in labels["designs"]
             if d.get("fab_sound") and not d.get("true_blockers")]

    # passes[(model, design)] -> [raw_findings, ...]  (parsed/ok passes only)
    passes: dict = defaultdict(list)
    models = set()
    for r in rows:
        parts = r.get("cell", "").split("|")
        if len(parts) < 3 or not r.get("ok"):
            continue
        model, did = parts[0], parts[1]
        models.add(model)
        passes[(model, did)].append(r.get("findings", []) or [])

    def rate(fn_pair, model, did):
        """Mean over ORDERED same-model pass pairs (each pass screens as pass 1)."""
        ps = passes.get((model, did), [])
        pairs = [(i, j) for i, j in itertools.permutations(range(len(ps)), 2)]
        if not pairs:
            return None
        return sum(fn_pair(ps[i], ps[j]) for i, j in pairs) / len(pairs)

    def single_rate(fn, model, did):
        ps = passes.get((model, did), [])
        return (sum(fn(p) for p in ps) / len(ps)) if ps else None

    other_models = sorted(models - {DEFAULT_MODEL})
    print("=" * 72)
    print(f"electrical-review taxonomy + corroboration backtest")
    print(f"corpus: {bdir}")
    print(f"models: {sorted(models)}  |  default = {DEFAULT_MODEL}")
    print("=" * 72)

    print("\nRECALL -- corroborated-block rate on confirmed blockers "
          "(ordered same-model pass pairs):")
    print(f"  {'design':28s} {'area':20s} {DEFAULT_MODEL:>8s}  {'others(mean)':>12s}")
    failures = []
    for did in natural_blockers:
        mm = rate(_corroborated, DEFAULT_MODEL, did)
        others = [rate(_corroborated, m, did) for m in other_models]
        others = [x for x in others if x is not None]
        omean = sum(others) / len(others) if others else float("nan")
        tag = ""
        if did in MUST_BLOCK and (mm is None or mm <= 0.5):
            tag = "   <-- REGRESSION"
            failures.append(did)
        if did in REPORT_ONLY:
            tag = "   (documented recall risk; not asserted)"
        mms = f"{mm:.2f}" if mm is not None else "  - "
        print(f"  {did:28s} {label_area.get(did,''):20s} {mms:>8s}  {omean:>12.2f}{tag}")

    print("\nOVER-BLOCK -- block rate on DRC-sound boards (lower is better), "
          "mean over all models:")
    def mean_over_models(fn_or_pair, did, paired):
        vals = []
        for m in models:
            v = rate(fn_or_pair, m, did) if paired else single_rate(fn_or_pair, m, did)
            if v is not None:
                vals.append(v)
        return sum(vals) / len(vals) if vals else None

    agg = {"raw": [], "clamp": [], "corrob": []}
    residual = []
    for did in sound:
        raw = mean_over_models(_raw_blocked, did, paired=False)
        clamp = mean_over_models(_clamp_blocked, did, paired=False)
        corr = mean_over_models(_corroborated, did, paired=True)
        if raw is not None:
            agg["raw"].append(raw); agg["clamp"].append(clamp); agg["corrob"].append(corr)
        if corr and corr > 0:   # a sound board that STILL corroborate-blocks somewhere
            residual.append((did, corr))
    def m(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")
    print(f"  {'':28s} {'raw':>8s} {'clamp-only':>12s} {'clamp+corrob':>14s}")
    print(f"  {'mean over sound designs':28s} {m(agg['raw']):>8.2f} "
          f"{m(agg['clamp']):>12.2f} {m(agg['corrob']):>14.2f}")
    if residual:
        print("  sound boards that still corroborate-block for some model "
              "(hallucination residual, e.g. run_17):")
        for did, c in sorted(residual, key=lambda x: -x[1]):
            print(f"    {did:28s} corrob-block={c:.2f}")

    print("\n" + "-" * 72)
    if failures:
        print(f"FAIL: {len(failures)} must-block design(s) regressed for "
              f"{DEFAULT_MODEL}: {failures}")
        return 1
    print(f"PASS: all {len(MUST_BLOCK)} must-block designs still corroborate-block "
          f"for {DEFAULT_MODEL}; over-block dropped raw->corrob "
          f"{m(agg['raw']):.2f}->{m(agg['corrob']):.2f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
