#!/usr/bin/env python3
"""Phase 1 A/B: apply the Stage-3 structured-layout packer to frozen leaves and
measure the tidiness delta + a DRC-safety check — no solver, no routing, $0.

The packer (``apply_structured_local_layout``) is a pure post-placement
transform, so applying it to a frozen ``solved_layout.json`` faithfully
simulates running it as the late pass Phase 1 wires into ``solve()``. For each
leaf we report tidiness before/after and confirm the pass introduces no new
same-layer courtyard overlaps and pushes nothing off the (component-extent)
board bound.

    python scripts/phase1_packer_ab.py [CORPUS_DIR]
"""
from __future__ import annotations

import copy
import json
import os
import sys

from kicraft.autoplacer.brain.leaf_structured_layout import (
    apply_structured_local_layout,
)
from kicraft.autoplacer.brain.leaf_tidiness import (
    aggregate,
    leaf_tidiness,
    parts_from_components,
)
from kicraft.autoplacer.brain.subcircuit_instances import _component_from_dict
from kicraft.autoplacer.brain.types import Point

DEFAULT_CORPUS = "logs/self_eval/20260707T193651Z"


def _components(layout):
    raw = layout.get("components", []) or []
    if isinstance(raw, dict):
        raw = list(raw.values())
    return {c["ref"]: _component_from_dict(c) for c in raw if c.get("ref")}


def _board_bound(comps, margin=2.0):
    bxs, bys = [], []
    for c in comps.values():
        tl, br = c.physical_bbox()
        bxs += [tl.x, br.x]
        bys += [tl.y, br.y]
    return (Point(min(bxs) - margin, min(bys) - margin),
            Point(max(bxs) + margin, max(bys) + margin))


def _courtyard_overlaps(comps):
    items = list(comps.values())
    n = 0
    for i in range(len(items)):
        a = items[i]
        a_tl, a_br = a.bbox(0.0)
        for j in range(i + 1, len(items)):
            b = items[j]
            if a.layer != b.layer:
                continue
            b_tl, b_br = b.bbox(0.0)
            ox = min(a_br.x, b_br.x) - max(a_tl.x, b_tl.x)
            oy = min(a_br.y, b_br.y) - max(a_tl.y, b_tl.y)
            if ox > 0.05 and oy > 0.05:
                n += 1
    return n


def _offboard(comps, bound):
    tl, br = bound
    n = 0
    for c in comps.values():
        p_tl, p_br = c.physical_bbox()
        if p_tl.x < tl.x or p_br.x > br.x or p_tl.y < tl.y or p_br.y > br.y:
            n += 1
    return n


def main() -> int:
    corpus = next((a for a in sys.argv[1:] if not a.startswith("--")), DEFAULT_CORPUS)
    paths = sorted(
        os.path.join(root, "solved_layout.json")
        for root, _d, files in os.walk(corpus)
        if "solved_layout.json" in files
    )
    if not paths:
        print(f"no solved_layout.json under {corpus!r}", file=sys.stderr)
        return 1

    before_all, after_all = [], []
    new_overlaps = 0
    new_offboard = 0
    total_aligned = total_rotated = 0
    groups_placed = groups_skipped = 0
    leaves_touched = 0

    for path in paths:
        try:
            layout = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        comps = _components(layout)
        if not comps:
            continue
        bound = _board_bound(comps)

        before = leaf_tidiness(parts_from_components(comps))
        ov_before = _courtyard_overlaps(comps)
        off_before = _offboard(comps, bound)

        after_comps = copy.deepcopy(comps)
        summary = apply_structured_local_layout(after_comps, board_outline=bound)
        after = leaf_tidiness(parts_from_components(after_comps))

        before_all.append(before)
        after_all.append(after)
        total_aligned += summary["members_aligned"]
        total_rotated += summary["members_rotated"]
        groups_placed += summary["groups_placed"]
        groups_skipped += summary["groups_skipped"]
        if summary["members_aligned"]:
            leaves_touched += 1

        ov_after = _courtyard_overlaps(after_comps)
        off_after = _offboard(after_comps, bound)
        new_overlaps += max(0, ov_after - ov_before)
        new_offboard += max(0, off_after - off_before)

    b = aggregate(before_all)
    a = aggregate(after_all)
    print(f"corpus: {corpus}   leaves: {len(before_all)}   touched: {leaves_touched}\n")
    hdr = f"{'metric':<34}{'before':>10}{'after':>10}{'delta':>10}"
    print(hdr)
    print("-" * len(hdr))
    for key, name in [
        ("orientation_consensus_grouped_pct", "orientation consensus (grouped) %"),
        ("orientation_consensus_leaf_pct", "orientation consensus (leaf) %"),
        ("alignment_residual_mm", "alignment residual (mm)"),
        ("packing_fill_pct", "packing fill %"),
    ]:
        bv, av = b[key], a[key]
        d = (av - bv) if (bv is not None and av is not None) else None
        print(f"{name:<34}{_f(bv):>10}{_f(av):>10}{_f(d):>10}")
    print("-" * len(hdr))
    print(
        f"groups: placed={groups_placed} skipped={groups_skipped}   "
        f"members: aligned={total_aligned} rotated={total_rotated}"
    )
    print(f"SAFETY  new courtyard overlaps: {new_overlaps}   "
          f"new off-board parts: {new_offboard}")
    return 0


def _f(v) -> str:
    return "-" if v is None else f"{v:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
