#!/usr/bin/env python3
"""Phase 1 routing-parity sweep — does the Stage-3 packer regress leaf routing?

N-of-3 medians, packer ON vs OFF, per design. $0 (no LLM). Leaf-level
solve+route only (the packer touches leaf placement; parent compose is
untouched). For each design we stage a fresh copy per variant, run
``solve_subcircuits.py --route`` at 3 seeds, and compare the MEDIAN total
unconnected pads (and routed-leaf count) on vs off. KiCad Routing Tools is only
best-effort-stable, so the median across seeds is the honest signal — a single
run is noise.

    SCRATCH=/tmp/parity python scripts/phase1_routing_parity.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import statistics
import subprocess
import sys

BATCH = "logs/self_eval/20260707T193651Z"
DESIGNS = [
    ("run_10_rp2040-min", "MINIMAL_RP2040_BOARD"),
]
SEEDS = [0, 1, 2]
SCRATCH = os.environ.get("SCRATCH", "/tmp/parity")
START = "===SOLVE_SUBCIRCUITS_JSON_START==="
END = "===SOLVE_SUBCIRCUITS_JSON_END==="


def _iter_strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _iter_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _iter_strings(v)


def _routing_stats(payload) -> tuple[int, int]:
    """(total unconnected pads across routed leaves, routed-leaf count)."""
    total = 0
    routed = 0
    for s in _iter_strings(payload):
        if "Drc report for leaf_routed" in s:
            routed += 1
            m = re.search(r"Found (\d+) unconnected pads", s)
            total += int(m.group(1)) if m else 0
    return total, routed


def _stage(run_dir: str, stem: str, variant: str) -> str:
    src = os.path.join(BATCH, run_dir, "generated", stem)
    dst = os.path.join(SCRATCH, f"{stem}_{variant}")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    exp = os.path.join(dst, ".experiments")
    if os.path.isdir(exp):
        shutil.rmtree(exp)
    if variant == "off":
        cfgs = [f for f in os.listdir(dst) if f.endswith("_autoplacer.json")]
        if cfgs:
            p = os.path.join(dst, cfgs[0])
            cfg = json.load(open(p))
            # Full classic baseline: no group-rigid, no packer, no soft tidiness.
            cfg["leaf_group_rigid"] = False
            cfg["leaf_structured_local_layout"] = False
            cfg["leaf_psw_tidiness"] = 0.0
            json.dump(cfg, open(p, "w"), indent=2)
    return dst


def _solve(proj: str, stem: str, seed: int) -> tuple[int, int]:
    env = dict(os.environ, PYTHONHASHSEED="0")
    cmd = [
        ".venv/bin/python", "kicraft/cli/solve_subcircuits.py",
        os.path.join(proj, f"{stem}.kicad_sch"),
        "--pcb", os.path.join(proj, f"{stem}.kicad_pcb"),
        "--rounds", "1", "--seed", str(seed), "--route", "--json",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    if START not in out or END not in out:
        return (-1, -1)
    payload = json.loads(out[out.index(START) + len(START):out.index(END)])
    return _routing_stats(payload)


def main() -> int:
    os.makedirs(SCRATCH, exist_ok=True)
    results = {}
    for run_dir, stem in DESIGNS:
        results[stem] = {}
        for variant in ("off", "on"):
            proj = _stage(run_dir, stem, variant)
            uncs, routs = [], []
            for seed in SEEDS:
                u, r = _solve(proj, stem, seed)
                uncs.append(u)
                routs.append(r)
                print(f"  {stem:24} {variant:3} seed={seed} unconnected={u} routed_leaves={r}",
                      flush=True)
            results[stem][variant] = {
                "unconnected_seeds": uncs,
                "unconnected_median": statistics.median(uncs),
                "routed_median": statistics.median(routs),
            }
        # Incremental dump so partial results survive an interrupted sweep.
        json.dump(results, open(os.path.join(SCRATCH, "parity_results.json"), "w"),
                  indent=2)
        print(flush=True)

    print("\n=== ROUTING PARITY (median over 3 seeds) ===")
    hdr = f"{'design':<26}{'off_unc':>9}{'on_unc':>9}{'off_rt':>8}{'on_rt':>8}{'verdict':>10}"
    print(hdr)
    print("-" * len(hdr))
    regressions = 0
    for stem, r in results.items():
        ou, nu = r["off"]["unconnected_median"], r["on"]["unconnected_median"]
        orr, nr = r["off"]["routed_median"], r["on"]["routed_median"]
        verdict = "ok" if nu <= ou else "REGRESS"
        if nu > ou:
            regressions += 1
        print(f"{stem:<26}{ou:>9}{nu:>9}{orr:>8}{nr:>8}{verdict:>10}")
    print("-" * len(hdr))
    print(f"regressions (on > off unconnected): {regressions}/{len(results)}")
    json.dump(results, open(os.path.join(SCRATCH, "parity_results.json"), "w"), indent=2)
    print(f"wrote {os.path.join(SCRATCH, 'parity_results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
