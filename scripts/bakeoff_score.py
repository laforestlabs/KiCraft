#!/usr/bin/env python3
"""Score the electrical-review bakeoff.

Two layers:
  1. MECHANICAL (this script, no LLM): per (model, arm) across the corpus --
     FBR (false-block rate on sound designs), blocker-flag rate on blocker
     designs (the recall CEILING, before semantic matching), synthetic-floor
     hit rate, JSON-ok rate, $/review, latency p50/p90, finding counts.
  2. SEMANTIC (separate, Claude-Code graders): for each blocker/synthetic
     design this script dumps an ANONYMISED grader packet (the frozen label +
     digest + every (model,arm)'s findings under opaque ids) so a grader can
     mark each labelled blocker HIT/MISS and flag unmatched blocker findings
     for user triage. The anon->real map is written separately (never shown to
     the grader).

Usage: python scripts/bakeoff_score.py [--bakeoff-dir DIR]
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def pctl(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
    return round(xs[k], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bakeoff-dir", default="logs/bakeoff/20260618T200126Z")
    args = ap.parse_args()
    bdir = Path(args.bakeoff_dir)

    labels = json.loads((bdir / "labels.json").read_text())
    role = {}
    label_of = {}
    for d in labels["designs"]:
        role[d["design_id"]] = d["role"]
        label_of[d["design_id"]] = d
    for s in labels["synthetics"]:
        role[s["design_id"]] = "synthetic"
        label_of[s["design_id"]] = s

    rows = [json.loads(l) for l in (bdir / "results.jsonl").read_text().splitlines() if l.strip()]
    # index cells by (model_label, arm)
    def parse(cell):
        m, did, arm, rep = cell.split("|")
        return m, did, arm
    cells = collections.defaultdict(list)  # (model,arm) -> [row...]
    for r in rows:
        m, did, arm = parse(r["cell"])
        r["_design"] = did
        cells[(m, arm)].append(r)

    blocker_ids = [d for d, ro in role.items() if ro == "blocker"]
    sound_ids = [d for d, ro in role.items() if ro == "sound"]
    synth_ids = [d for d, ro in role.items() if ro == "synthetic"]
    # split sound into truly-clean (no expected warnings -> any block is a pure
    # over-block/hallucination) vs warn (block = escalating a warning to a gate-fail)
    sound_clean = [d for d in sound_ids if not label_of[d].get("expected_warnings")]
    sound_warn = [d for d in sound_ids if label_of[d].get("expected_warnings")]

    print(f"corpus: {len(blocker_ids)} blockers, {len(sound_ids)} sound, {len(synth_ids)} synthetic")
    print(f"cells in results: {len(rows)}\n")
    hdr = f"{'model':9s} {'arm':7s} {'ok%':>4s} {'flag/blk':>8s} {'FBRc':>5s} {'FBRw':>5s} {'synth':>5s} {'$/rev':>7s} {'p50':>5s} {'p90':>6s}"
    print(hdr)
    print("-" * len(hdr))
    table = []
    for (m, arm) in sorted(cells):
        rs = cells[(m, arm)]
        bymap = {r["_design"]: r for r in rs}
        n = len(rs)
        okrate = sum(r.get("ok", False) for r in rs) / n if n else 0
        # blocker-flag rate (recall ceiling): has_blocker on blocker designs
        bcells = [bymap[d] for d in blocker_ids if d in bymap]
        flag = sum(r.get("has_blocker", False) for r in bcells) / len(bcells) if bcells else 0
        # FBR: has_blocker on sound designs (fully mechanical), split clean vs warn
        cl = [bymap[d] for d in sound_clean if d in bymap]
        wn = [bymap[d] for d in sound_warn if d in bymap]
        fbrc = sum(r.get("has_blocker", False) for r in cl) / len(cl) if cl else 0
        fbrw = sum(r.get("has_blocker", False) for r in wn) / len(wn) if wn else 0
        # synthetic floor
        sycells = [bymap[d] for d in synth_ids if d in bymap]
        syn = sum(r.get("has_blocker", False) for r in sycells) / len(sycells) if sycells else 0
        costs = [r.get("cost_usd", 0) or 0 for r in rs]
        lats = [r.get("latency_s") for r in rs if r.get("latency_s")]
        row = dict(model=m, arm=arm, n=n, ok=round(okrate, 2),
                   flag_blk=round(flag, 2), fbr_clean=round(fbrc, 2), fbr_warn=round(fbrw, 2),
                   synth=round(syn, 2),
                   cost=round(statistics.mean(costs), 4) if costs else 0,
                   p50=pctl(lats, 50), p90=pctl(lats, 90),
                   n_blk=len(bcells), n_clean=len(cl), n_warn=len(wn), n_syn=len(sycells))
        table.append(row)
        print(f"{m:9s} {arm:7s} {okrate*100:3.0f}% {flag*100:6.0f}% {fbrc*100:4.0f}% {fbrw*100:4.0f}% "
              f"{syn*100:4.0f}% ${row['cost']:.4f} {str(row['p50']):>5s} {str(row['p90']):>6s}")

    (bdir / "mechanical_scorecard.json").write_text(json.dumps(table, indent=2))

    print("\nover-block by sound design (cells flagging a blocker / total):")
    for d in sound_clean + sound_warn:
        cs = [r for rs in cells.values() for r in rs if r["_design"] == d]
        nb = sum(r.get("has_blocker", False) for r in cs)
        tag = "warn " if d in sound_warn else "CLEAN"
        bar = "#" * round(20 * nb / max(1, len(cs)))
        print(f"  {tag} {d:30s} {nb:2d}/{len(cs):2d} {bar}")

    # ---- assemble anonymised grader packets for blocker + synthetic designs ----
    pkt_dir = bdir / "grader_packets"
    pkt_dir.mkdir(exist_ok=True)
    anon_map = {}  # design -> {anon_id: "model|arm"}
    n_pkt = 0
    for did in blocker_ids + synth_ids:
        lab = label_of[did]
        digest = (bdir / "corpus" / did / "digest.txt").read_text()
        finding_sets = []
        amap = {}
        idx = 0
        for (m, arm), rs in sorted(cells.items()):
            r = next((x for x in rs if x["_design"] == did), None)
            if not r or not r.get("ok"):
                continue
            aid = f"R{idx:02d}"
            amap[aid] = f"{m}|{arm}"
            finding_sets.append({"id": aid, "findings": r.get("findings", [])})
            idx += 1
        anon_map[did] = amap
        packet = {
            "design_id": did,
            "labeled_true_blockers": lab.get("true_blockers", []),
            "expected_warnings": lab.get("expected_warnings", []),
            "digest": digest,
            "anonymized_model_findings": finding_sets,
            "task": ("For each anonymized id, mark each labeled true-blocker HIT "
                     "(the id emitted a severity='blocker' finding that semantically "
                     "matches that defect -- same part/net/failure) or MISS. A defect "
                     "found but graded warning/note = MISS (operationally the board "
                     "ships). List any severity='blocker' finding that matches NO "
                     "labeled defect as UNMATCHED (for user triage: real miss in our "
                     "labels, or hallucination). Do NOT re-derive correctness; match "
                     "against the labels only."),
        }
        (pkt_dir / f"{did}.json").write_text(json.dumps(packet, indent=2))
        n_pkt += 1
    (bdir / "anon_map.json").write_text(json.dumps(anon_map, indent=2))
    print(f"\nwrote {n_pkt} grader packets -> {pkt_dir}/ (+ anon_map.json, mechanical_scorecard.json)")


if __name__ == "__main__":
    main()
