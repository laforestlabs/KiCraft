#!/usr/bin/env python3
"""Corpus-wide A/B for the soft-tidiness placement term — classic SA vs
soft-tidiness across a set of designs — measuring tidiness (orientation / row
residual / fill) and routing (unconnected), and emitting a self-contained HTML
results page with side-by-side classic|soft diagnostic renders per leaf. $0, no
LLM.

    python scripts/soft_tidiness_ab.py [--out DIR] [--designs A,B] [--seeds 0,1,2]

Each design is staged twice: ``classic`` (leaf_psw_tidiness=0, both hard-tidiness
flags off) and ``soft`` (defaults — psw_tidiness=0.15). Both are solved+routed at
each seed; placement (hence tidiness + renders) is deterministic per seed, while
routing unconnected is reported as a per-seed list (FreeRouting is only
best-effort-stable — the rigorous routing verdict is the N-of-3 median).

With no ``--out`` the gallery is written to ``logs/tidiness_ab/run-<UTC>/`` — the
root the web app's **/admin/tidiness-ab** page discovers — so a bare run shows up
in the admin section on a headless box (view it there, or open ``<out>/index.html``
directly).
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from kicraft.autoplacer.brain.leaf_layout_svg import render_leaf_svg
from kicraft.autoplacer.brain.leaf_tidiness import (
    leaf_pin_locality,
    leaf_tidiness,
    parts_from_components,
)
from kicraft.autoplacer.brain.subcircuit_instances import _component_from_dict
from kicraft.autoplacer.brain.types import Point

BATCH = "logs/self_eval/20260707T193651Z"
# A dense+sparse mix; extend for the full corpus run.
DESIGNS = [
    ("run_17_led-cc-driver", "1A_LED_DRIVER"),        # sparse, messy baseline
    ("run_16_highside-switch-10a", "HIGH_SIDE_LOAD_SWITCH"),  # sparse, clean
    ("run_08_rs485-terminal", "ISOLATED_RS485"),      # medium
    ("run_02_r2r-dac", "AN_R_2R"),                    # passive-heavy, messy
    ("run_10_rp2040-min", "MINIMAL_RP2040_BOARD"),    # dense (routing-safety canary)
]
START = "===SOLVE_SUBCIRCUITS_JSON_START==="
END = "===SOLVE_SUBCIRCUITS_JSON_END==="

# Each variant is a cfg-patch written into the staged project's *_autoplacer.json
# before solving. Run compares exactly two (``--variants=baseline,candidate``):
#   classic  hard-off legacy SA           soft     the shipped soft-tidiness term
#   pinloc   pin-locality term on cont.SA  grid    discrete grid + SA-as-assignment
VARIANTS = {
    "classic": {"leaf_psw_tidiness": 0.0, "leaf_group_rigid": False,
                "leaf_structured_local_layout": False, "leaf_grid_assignment": False},
    "soft": {},  # pipeline defaults (psw_tidiness=0.15)
    "pinloc": {"leaf_psw_pin_locality": 0.25},
    "grid": {"leaf_grid_assignment": True},
}


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


def _stage(run_dir, stem, variant, scratch):
    src = os.path.join(BATCH, run_dir, "generated", stem)
    dst = os.path.join(scratch, f"{stem}_{variant}")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    shutil.rmtree(os.path.join(dst, ".experiments"), ignore_errors=True)
    patch = VARIANTS.get(variant, {})
    if patch:
        cfgs = [f for f in os.listdir(dst) if f.endswith("_autoplacer.json")]
        if cfgs:
            p = os.path.join(dst, cfgs[0])
            cfg = json.load(open(p))
            cfg.update(patch)
            json.dump(cfg, open(p, "w"), indent=2)
    return dst


def _solve(proj, stem, seed):
    env = dict(os.environ, PYTHONHASHSEED="0")
    cmd = [
        ".venv/bin/python", "kicraft/cli/solve_subcircuits.py",
        os.path.join(proj, f"{stem}.kicad_sch"),
        "--pcb", os.path.join(proj, f"{stem}.kicad_pcb"),
        "--rounds", "1", "--seed", str(seed), "--route", "--json",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    unc = -1
    if START in out and END in out:
        payload = json.loads(out[out.index(START) + len(START):out.index(END)])
        unc = 0
        for s in _iter_strings(payload):
            if "Drc report for leaf_routed" in s:
                m = re.search(r"Found (\d+) unconnected pads", s)
                unc += int(m.group(1)) if m else 0
    return unc


def _iter_strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _iter_strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from _iter_strings(v)


def _leaf_layouts(proj):
    """{sheet_name: components} from canonical solved_layout, else latest round."""
    out = {}
    for sub in glob.glob(os.path.join(proj, ".experiments", "subcircuits", "*")):
        canon = os.path.join(sub, "solved_layout.json")
        path = canon if os.path.exists(canon) else None
        if path is None:
            rounds = sorted(glob.glob(os.path.join(sub, "round_*_solved_layout.json")))
            if rounds:
                path = rounds[-1]
        if path is None:
            continue
        try:
            d = json.load(open(path))
        except (OSError, json.JSONDecodeError):
            continue
        comps = _components(d)
        if comps:
            out[d.get("sheet_name", os.path.basename(sub))] = comps
    return out


def _default_out_dir() -> str:
    """Default gallery dir: a timestamped run under the repo's ``logs/tidiness_ab``,
    which the web app's /admin/tidiness-ab page discovers -- so a bare ``python
    scripts/soft_tidiness_ab.py`` shows up in admin without any flags."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stamp = time.strftime("run-%Y%m%dT%H%M%SZ", time.gmtime())
    return os.path.join(repo, "logs", "tidiness_ab", stamp)


def main() -> int:
    out_dir = _default_out_dir()
    designs = DESIGNS
    seeds = [0]
    variants = ("classic", "soft")  # baseline, candidate
    for a in sys.argv[1:]:
        if a.startswith("--out="):
            out_dir = a.split("=", 1)[1]
        elif a.startswith("--designs="):
            want = set(a.split("=", 1)[1].split(","))
            designs = [d for d in DESIGNS if d[1] in want]
        elif a.startswith("--seeds="):
            seeds = [int(x) for x in a.split("=", 1)[1].split(",")]
        elif a.startswith("--variants="):
            names = a.split("=", 1)[1].split(",")
            assert len(names) == 2 and all(n in VARIANTS for n in names), (
                f"--variants needs exactly two of {sorted(VARIANTS)}")
            variants = tuple(names)
    base, cand = variants
    # Scratch (heavy: staged project copies + .experiments) lives in a tempdir, NOT
    # under out_dir, so the discoverable gallery dir stays just index.html/summary.json.
    scratch = tempfile.mkdtemp(prefix="soft_ab_work_")

    results = []
    for run_dir, stem in designs:
        print(f"[{stem}] staging + solving {base}/{cand} ...", flush=True)
        unc = {v: [] for v in variants}
        projs = {}
        for variant in variants:
            projs[variant] = _stage(run_dir, stem, variant, scratch)
            for seed in seeds:
                if seed != seeds[0]:
                    shutil.rmtree(os.path.join(projs[variant], ".experiments"),
                                  ignore_errors=True)
                unc[variant].append(_solve(projs[variant], stem, seed))
        lay = {v: _leaf_layouts(projs[v]) for v in variants}
        leaves = []
        for sheet in sorted(set(lay[base]) & set(lay[cand])):
            row = {"sheet": sheet}
            for v in variants:
                comps = lay[v][sheet]
                parts = parts_from_components(comps)
                m = leaf_tidiness(parts)
                pl = leaf_pin_locality(parts)
                row[v] = {
                    "orient": m.orientation_consensus_grouped_pct,
                    "resid": m.alignment_residual_mm,
                    "fill": m.packing_fill_pct,
                    "pinloc": pl.pin_locality_pct,
                    "pin_mm": pl.mean_worst_pad_dist_mm,
                    "n_groups": m.n_groups,
                    "svg": render_leaf_svg(comps, _board_bound(comps),
                                           title=f"{stem} / {sheet} [{v}]"),
                }
            leaves.append(row)
        results.append({
            "design": stem,
            "variants": list(variants),  # [baseline, candidate] -- for the admin summary
            "unconnected": {v: _med(unc[v]) for v in variants},
            "unconnected_seeds": unc,
            "leaves": leaves,
        })
        print(f"[{stem}] {len(leaves)} comparable leaves; "
              f"unc {base}={_med(unc[base])} {cand}={_med(unc[cand])}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    _write_html(results, os.path.join(out_dir, "index.html"), variants, seeds)
    json.dump(
        [{k: v for k, v in r.items() if k != "leaves"} for r in results],
        open(os.path.join(out_dir, "summary.json"), "w"), indent=2,
    )
    print(f"\nwrote {out_dir}/index.html")
    return 0


def _med(xs):
    xs = [x for x in xs if x is not None and x >= 0]
    return statistics.median(xs) if xs else None


def _fmt(v, s=""):
    return "—" if v is None else f"{v:.1f}{s}"


def _delta_cell(base, cand, higher_better=True, suffix=""):
    if base is None or cand is None:
        return f"<td>{_fmt(base, suffix)}</td><td>{_fmt(cand, suffix)}</td><td>—</td>"
    d = cand - base
    good = (d >= 0) if higher_better else (d <= 0)
    color = "#0ca30c" if (abs(d) < 0.05 or good) else "#d03b3b"
    sign = "+" if d >= 0 else ""
    return (f"<td>{_fmt(base, suffix)}</td><td>{_fmt(cand, suffix)}</td>"
            f"<td style='color:{color};font-weight:600'>{sign}{d:.1f}</td>")


def _write_html(results, path, variants, seeds):
    base, cand = variants
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Placement A/B</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,sans-serif;background:#f9f9f7;",
        "color:#0b0b0b;margin:24px;max-width:1200px}h1{font-size:22px}",
        "h2{font-size:16px;color:#52514e;margin-top:32px}",
        ".sub{color:#898781}table{border-collapse:collapse;margin:12px 0;font-size:13px}",
        "th,td{padding:5px 10px;text-align:right;border-bottom:1px solid #e1e0d9}",
        "th{color:#898781;font-weight:600;text-align:right}td:first-child,th:first-child{text-align:left}",
        ".pair{display:flex;gap:12px;overflow-x:auto;padding:8px 0}",
        ".pair figure{margin:0}.pair figcaption{font-size:11px;color:#898781;margin-bottom:4px}",
        "section{margin:16px 0;padding:16px;background:#fff;border:1px solid rgba(11,11,11,.1);border-radius:8px}",
        "</style>",
        f"<h1>Placement A/B — {base} vs {cand}</h1>",
        f"<p class='sub'>{len(results)} designs, seeds {seeds}. "
        "Tidiness, pin-locality &amp; renders are deterministic per seed; unconnected "
        "is indicative (rigorous routing verdict = N-of-3 median sweep). "
        "<b>pin mm</b> = mean distance from each passive's worst pad to its nearest "
        "same-net IC pin (lower is better; the real objective).</p>",
    ]

    # Summary table.
    parts.append("<h2>Summary (per design)</h2><table><tr>"
                 f"<th>design</th><th>leaves</th>"
                 f"<th>orient {base}</th><th>orient {cand}</th><th>Δ</th>"
                 f"<th>resid {base}</th><th>resid {cand}</th><th>Δ</th>"
                 f"<th>pin mm {base}</th><th>pin mm {cand}</th><th>Δ</th>"
                 f"<th>unc {base}</th><th>unc {cand}</th></tr>")
    for r in results:
        oc = _mean([lf[base]["orient"] for lf in r["leaves"]])
        ox = _mean([lf[cand]["orient"] for lf in r["leaves"]])
        rc = _mean([lf[base]["resid"] for lf in r["leaves"]])
        rx = _mean([lf[cand]["resid"] for lf in r["leaves"]])
        pc = _mean([lf[base]["pin_mm"] for lf in r["leaves"]])
        px = _mean([lf[cand]["pin_mm"] for lf in r["leaves"]])
        parts.append(
            f"<tr><td>{r['design']}</td><td>{len(r['leaves'])}</td>"
            + _delta_cell(oc, ox, higher_better=True, suffix="%")
            + _delta_cell(rc, rx, higher_better=False)
            + _delta_cell(pc, px, higher_better=False)
            + f"<td>{_fmt(r['unconnected'][base])}</td>"
            f"<td>{_fmt(r['unconnected'][cand])}</td></tr>"
        )
    parts.append("</table>")

    # Per-leaf renders.
    for r in results:
        if not r["leaves"]:
            continue
        parts.append(f"<h2>{r['design']}</h2>")
        for lf in r["leaves"]:
            c, s = lf[base], lf[cand]
            parts.append(
                f"<section><table><tr><th>{lf['sheet']}</th>"
                "<th>orient</th><th>resid mm</th><th>fill %</th>"
                "<th>pin mm</th><th>pin loc %</th></tr>"
                f"<tr><td>{base}</td><td>{_fmt(c['orient'])}</td>"
                f"<td>{_fmt(c['resid'])}</td><td>{_fmt(c['fill'])}</td>"
                f"<td>{_fmt(c['pin_mm'])}</td><td>{_fmt(c['pinloc'])}</td></tr>"
                f"<tr><td>{cand}</td><td>{_fmt(s['orient'])}</td>"
                f"<td>{_fmt(s['resid'])}</td><td>{_fmt(s['fill'])}</td>"
                f"<td>{_fmt(s['pin_mm'])}</td><td>{_fmt(s['pinloc'])}</td></tr></table>"
                "<div class='pair'>"
                f"<figure><figcaption>{base}</figcaption>{c['svg']}</figure>"
                f"<figure><figcaption>{cand}</figcaption>{s['svg']}</figure>"
                "</div></section>"
            )
    open(path, "w").write("".join(parts))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


if __name__ == "__main__":
    raise SystemExit(main())
