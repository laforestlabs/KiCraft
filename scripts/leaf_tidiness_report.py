#!/usr/bin/env python3
"""Read-only $0 instrument for the placement-streamline work (Phase 0).

Walk a self-eval corpus (or any tree of ``solved_layout.json`` files) and report
per-leaf and corpus-level *tidiness* metrics — how orderly vs random the placed
passives look — using ``kicraft.autoplacer.brain.leaf_tidiness``:

    orientation_consensus_grouped_pct  passives in a group pointing the group's
                                       dominant way (100 = all agree)
    orientation_consensus_leaf_pct     same, across every passive in the leaf
    alignment_residual_mm              off-axis scatter within a row (0 = straight)
    packing_fill_pct                   component area / placement bbox (higher=tight)

No pcbnew, no placement change — reads the frozen JSON only.

    python scripts/leaf_tidiness_report.py [CORPUS_DIR] [--json OUT.json]

Default corpus: yesterday's self-eval batch.
"""
from __future__ import annotations

import json
import os
import sys

from kicraft.autoplacer.brain.leaf_tidiness import (
    aggregate,
    leaf_tidiness,
    parts_from_layout,
)

DEFAULT_CORPUS = "logs/self_eval/20260707T193651Z"


def _design_of(path: str) -> str:
    # .../generated/<STEM>/.experiments/subcircuits/<id>/solved_layout.json
    parts = path.split(os.sep)
    if "generated" in parts:
        i = parts.index("generated")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "?"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    corpus = args[0] if args else DEFAULT_CORPUS
    json_out = None
    for a in sys.argv[1:]:
        if a.startswith("--json="):
            json_out = a.split("=", 1)[1]

    # os.walk (not glob) so we descend into the dotted ``.experiments/`` tree
    # where the leaves live — glob's ``**`` skips dot-directories.
    layout_paths = sorted(
        os.path.join(root, "solved_layout.json")
        for root, _dirs, files in os.walk(corpus)
        if "solved_layout.json" in files
    )
    if not layout_paths:
        print(f"no solved_layout.json under {corpus!r}", file=sys.stderr)
        return 1

    per_leaf = []
    for path in layout_paths:
        try:
            layout = json.load(open(path))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skip {path}: {exc}", file=sys.stderr)
            continue
        design = _design_of(path)
        sheet = layout.get("sheet_name") or layout.get("subcircuit_id", "?")
        m = leaf_tidiness(parts_from_layout(layout), label=f"{design}/{sheet}")
        per_leaf.append((design, m))

    # Per-design rollup (mean over that design's leaves).
    designs: dict[str, list] = {}
    for design, m in per_leaf:
        designs.setdefault(design, []).append(m)

    print(f"corpus: {corpus}")
    print(f"leaves: {len(per_leaf)}  designs: {len(designs)}\n")
    hdr = f"{'design':<26} {'lf':>3} {'grp%':>6} {'leaf%':>6} {'resid':>6} {'fill%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for design in sorted(designs):
        agg = aggregate(designs[design])
        print(
            f"{design[:26]:<26} {agg['n_leaves']:>3} "
            f"{_fmt(agg['orientation_consensus_grouped_pct']):>6} "
            f"{_fmt(agg['orientation_consensus_leaf_pct']):>6} "
            f"{_fmt(agg['alignment_residual_mm']):>6} "
            f"{_fmt(agg['packing_fill_pct']):>6}"
        )

    corpus_agg = aggregate([m for _, m in per_leaf])
    print("-" * len(hdr))
    print(
        f"{'CORPUS':<26} {corpus_agg['n_leaves']:>3} "
        f"{_fmt(corpus_agg['orientation_consensus_grouped_pct']):>6} "
        f"{_fmt(corpus_agg['orientation_consensus_leaf_pct']):>6} "
        f"{_fmt(corpus_agg['alignment_residual_mm']):>6} "
        f"{_fmt(corpus_agg['packing_fill_pct']):>6}"
    )
    print(
        f"\nleaves with >=1 passive group: {corpus_agg['n_leaves_with_groups']}"
        f" / {corpus_agg['n_leaves']}"
    )

    if json_out:
        payload = {
            "corpus": corpus,
            "aggregate": corpus_agg,
            "per_design": {d: aggregate(ms) for d, ms in designs.items()},
            "per_leaf": [m.to_dict() for _, m in per_leaf],
        }
        json.dump(payload, open(json_out, "w"), indent=2)
        print(f"\nwrote {json_out}")
    return 0


def _fmt(v) -> str:
    return "-" if v is None else f"{v:g}"


if __name__ == "__main__":
    raise SystemExit(main())
