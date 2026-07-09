#!/usr/bin/env python3
"""Visual analysis framework — render placed leaves as annotated diagnostic SVGs
and a combined HTML gallery. Pairs with ``leaf_tidiness_report.py`` (numbers);
this is the picture. $0, no pcbnew.

    python scripts/leaf_layout_viz.py [CORPUS_DIR] --designs A,B --out DIR

Reads ``solved_layout.json`` files, reconstructs components, and writes one SVG
per leaf plus ``index.html`` embedding them all with their metrics. Default
corpus: yesterday's self-eval batch; default designs: a dense + a sparse one.
"""
from __future__ import annotations

import json
import os
import sys

from kicraft.autoplacer.brain.leaf_group_rigid import (
    build_rigid_groups,
    sync_rigid_groups,
)
from kicraft.autoplacer.brain.leaf_layout_svg import render_leaf_svg
from kicraft.autoplacer.brain.leaf_tidiness import leaf_tidiness, parts_from_components
from kicraft.autoplacer.brain.subcircuit_instances import _component_from_dict
from kicraft.autoplacer.brain.types import Point

DEFAULT_CORPUS = "logs/self_eval/20260707T193651Z"
DEFAULT_DESIGNS = {"MINIMAL_RP2040_BOARD", "HIGH_SIDE_LOAD_SWITCH", "TPS5430_BUCK_CONVERTER"}


def _components(layout):
    raw = layout.get("components", []) or []
    if isinstance(raw, dict):
        raw = list(raw.values())
    return {c["ref"]: _component_from_dict(c) for c in raw if c.get("ref")}


def _board_bound(comps, margin=2.0):
    xs, ys = [], []
    for c in comps.values():
        tl, br = c.physical_bbox()
        xs += [tl.x, br.x]
        ys += [tl.y, br.y]
    return (Point(min(xs) - margin, min(ys) - margin),
            Point(max(xs) + margin, max(ys) + margin))


def _design_of(path):
    parts = path.split(os.sep)
    return parts[parts.index("generated") + 1] if "generated" in parts else "?"


def main() -> int:
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    corpus = pos[0] if pos else DEFAULT_CORPUS
    designs = DEFAULT_DESIGNS
    out_dir = os.path.join(
        os.environ.get("SCRATCH", "/tmp"), "leaf_viz"
    )
    rigid = False
    for a in sys.argv[1:]:
        if a.startswith("--designs="):
            designs = set(a.split("=", 1)[1].split(","))
        elif a.startswith("--out="):
            out_dir = a.split("=", 1)[1]
        elif a == "--rigid":
            rigid = True
    os.makedirs(out_dir, exist_ok=True)

    layouts = []
    for root, _d, files in os.walk(corpus):
        if "solved_layout.json" in files:
            p = os.path.join(root, "solved_layout.json")
            if _design_of(p) in designs:
                layouts.append(p)

    cards = []
    for path in sorted(layouts):
        try:
            layout = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        comps = _components(layout)
        if len(comps) < 1:
            continue
        design = _design_of(path)
        sheet = layout.get("sheet_name") or "?"
        bound = _board_bound(comps)
        if rigid:
            # Apply the rigid-group representation in place (tidy by construction)
            # so we can see it vs the raw SA baseline. No re-optimization here —
            # this shows the *tidiness* of the primitive, not routability.
            sync_rigid_groups(comps, build_rigid_groups(comps))
        title = f"{design} / {sheet}" + (" [rigid]" if rigid else "")
        svg = render_leaf_svg(comps, bound, title=title)
        m = leaf_tidiness(parts_from_components(comps), label=title)
        fname = f"{design}__{sheet}".replace("/", "_").replace(" ", "_") + ".svg"
        open(os.path.join(out_dir, fname), "w").write(svg)
        cards.append((title, m, svg))
        print(f"  rendered {fname}  "
              f"orient={m.orientation_consensus_grouped_pct} "
              f"resid={m.alignment_residual_mm} fill={m.packing_fill_pct}")

    # combined gallery
    body = [
        "<h1>Leaf layout diagnostics</h1>",
        f"<p class='sub'>{len(cards)} leaves from {os.path.basename(corpus)}</p>",
    ]
    for title, m, svg in cards:
        body.append(
            f"<section><h2>{title}</h2><div class='svg'>{svg}</div></section>"
        )
    html = (
        "<!doctype html><meta charset='utf-8'><title>Leaf diagnostics</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#f9f9f7;"
        "color:#0b0b0b;margin:24px;max-width:900px}h1{font-size:20px}"
        ".sub{color:#898781}section{margin:24px 0;padding:16px;background:#fff;"
        "border:1px solid rgba(11,11,11,.1);border-radius:8px}"
        "h2{font-size:14px;color:#52514e}.svg{overflow-x:auto}</style>"
        + "".join(body)
    )
    idx = os.path.join(out_dir, "index.html")
    open(idx, "w").write(html)
    print(f"\nwrote {len(cards)} SVGs + {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
