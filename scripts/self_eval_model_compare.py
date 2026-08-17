#!/usr/bin/env python3
"""A/B comparison of two self-eval batches (design-model change).

Usage: python scripts/self_eval_model_compare.py <BASE_DIR> <NEW_DIR>

Base first. Both dirs must contain a ``summary.json`` produced by
``kicraft.eval.self_eval`` (schema: flat top-level meta fields -- design_model,
judge_model, rubric_version, n, graded_n, fab_ready, mean_final, median_final,
grade_counts, gate_counts, n_errored, total_cost_usd, wall_s -- plus ``runs[]``
of per-brief records keyed by ``slug``/``repeat``).

The comparison is only valid when judge and rubric match between batches (the
score is meaningless otherwise), so the script exits 2 on any mismatch before
printing anything else.

Flags per brief (any of):
  - |d final| >= 15  (the documented same-model run-to-run noise floor is ~12;
                     15 is the conservative flag line)
  - build_label changed (fab-ready <-> not; a build flip is the strongest
    per-brief signal since place/route is deterministic given the design)
  - grade letter changed
  - gates non-empty on either side
  - error non-empty on either side (design_error / judge_ok false)
  - final < 60 on either side

Output goes to stdout and to ``$NEW_DIR/model_compare.md``. Exits 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FLAG_DELTA = 15.0
LOW_FINAL = 60.0

_HEADLINE_KEYS = [
    "design_model", "judge_model", "rubric_version", "n", "graded_n",
    "fab_ready", "mean_final", "median_final", "grade_counts", "gate_counts",
    "n_errored", "total_cost_usd", "wall_s",
]


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def runs_by_slug(d: dict) -> dict:
    return {r["slug"]: r for r in d.get("runs", []) if r.get("repeat") is None}


def _fmt(v, nd=1) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _row_flags(base: dict, new: dict) -> list[str]:
    flags = []
    bf, nf = base.get("final"), new.get("final")
    if bf is not None and nf is not None and abs(nf - bf) >= FLAG_DELTA:
        flags.append("|d|>=15")
    if (base.get("build_label") or "") != (new.get("build_label") or ""):
        flags.append("build flip")
    if (base.get("grade") or "") != (new.get("grade") or ""):
        flags.append("grade flip")
    if (base.get("gates") or []) or (new.get("gates") or []):
        flags.append("gates")
    if base.get("design_error") or new.get("design_error") or not (
            base.get("judge_ok", True) and new.get("judge_ok", True)):
        flags.append("error")
    if (bf is not None and bf < LOW_FINAL) or (nf is not None and nf < LOW_FINAL):
        flags.append("low score")
    return flags


def build_report(base: dict, new: dict, base_dir: str, new_dir: str) -> tuple[str, dict]:
    """Return (markdown, stats)."""
    b_runs, n_runs = runs_by_slug(base), runs_by_slug(new)
    slugs = sorted(set(b_runs) & set(n_runs))
    only_base = sorted(set(b_runs) - set(n_runs))
    only_new = sorted(set(n_runs) - set(b_runs))

    rows = []
    for slug in slugs:
        br, nr = b_runs[slug], n_runs[slug]
        bf, nf = br.get("final"), nr.get("final")
        d = (nf - bf) if (bf is not None and nf is not None) else None
        rows.append({
            "slug": slug, "base": br, "new": nr, "delta": d,
            "flags": _row_flags(br, nr),
        })
    rows.sort(key=lambda r: (abs(r["delta"]) if r["delta"] is not None else -1.0),
              reverse=True)

    flag_counts: dict[str, int] = {}
    for r in rows:
        for f in r["flags"]:
            flag_counts[f] = flag_counts.get(f, 0) + 1
    build_flips = [r for r in rows if "build flip" in r["flags"]]
    fab_ready_d = (new.get("fab_ready") or 0) - (base.get("fab_ready") or 0)
    mean_d = ((new.get("mean_final") or 0.0) - (base.get("mean_final") or 0.0))
    median_d = ((new.get("median_final") or 0.0) - (base.get("median_final") or 0.0))
    top3 = [(r["slug"], r["delta"]) for r in rows[:3] if r["delta"] is not None]

    lines: list[str] = []
    add = lines.append

    add(f"# Self-eval model A/B: {base_dir} vs {new_dir}")
    add("")
    add("Compare script: `scripts/self_eval_model_compare.py` — base first.")
    add("")
    add(f"Matched briefs (repeat=null intersection): **{len(rows)}/{len(slugs)}** "
        f"(base-only: {only_base or 'none'}, new-only: {only_new or 'none'})")
    add("")
    add("## Headlines")
    add("")
    add("| field | base | new |")
    add("|---|---|---|")
    for k in _HEADLINE_KEYS:
        add(f"| {k} | {_fmt(base.get(k))} | {_fmt(new.get(k))} |")
    add("")
    add(f"fab_ready d: **{fab_ready_d:+d}** | mean d: **{mean_d:+.1f}** | "
        f"median d: **{median_d:+.1f}**")
    add("")
    add("## Per-brief (sorted by |d final| desc)")
    add("")
    add("| slug | base final | new final | d | base build -> new build | "
        "base grade -> new grade | flags |")
    add("|---|---|---|---|---|---|---|")
    for r in rows:
        br, nr = r["base"], r["new"]
        dcell = _fmt(r["delta"], 1) if r["delta"] is not None else "—"
        if r["delta"] is not None and r["delta"] > 0:
            dcell = f"+{dcell}"
        add(f"| {r['slug']} | {_fmt(br.get('final'))} | {_fmt(nr.get('final'))} "
            f"| {dcell} | {_fmt(br.get('build_label'))} -> {_fmt(nr.get('build_label'))} "
            f"| {_fmt(br.get('grade'))} -> {_fmt(nr.get('grade'))} "
            f"| {', '.join(r['flags']) or ''} |")
    add("")
    add("## Flag summary")
    add("")
    if flag_counts:
        add("| flag | count |")
        add("|---|---|")
        for f in sorted(flag_counts):
            add(f"| {f} | {flag_counts[f]} |")
        add("")
    else:
        add("No flagged briefs.")
        add("")
    if build_flips:
        add("Build flips (fab-ready <-> not):")
        add("")
        for r in build_flips:
            add(f"- {r['slug']}: {_fmt(r['base'].get('build_label'))} -> "
                f"{_fmt(r['new'].get('build_label'))}")
        add("")
    if top3:
        add("Top-3 largest |d|:")
        add("")
        for slug, d in top3:
            sign = "+" if d > 0 else ""
            add(f"- {slug}: {sign}{d:.1f}")
        add("")
    add(f"Total spend: base ${base.get('total_cost_usd', 0.0):.2f} in "
        f"{base.get('wall_s', 0.0) / 3600:.1f} h vs new "
        f"${new.get('total_cost_usd', 0.0):.2f} in {new.get('wall_s', 0.0) / 3600:.1f} h.")

    stats = {
        "matched": len(rows),
        "flag_counts": flag_counts,
        "build_flips": [r["slug"] for r in build_flips],
        "fab_ready_d": fab_ready_d,
        "mean_d": mean_d,
        "median_d": median_d,
        "top3": top3,
    }
    return "\n".join(lines) + "\n", stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base_dir", help="baseline self-eval output dir (contains summary.json)")
    ap.add_argument("new_dir", help="candidate self-eval output dir (contains summary.json)")
    args = ap.parse_args()

    base_dir, new_dir = Path(args.base_dir), Path(args.new_dir)
    try:
        base = load(base_dir / "summary.json")
        new = load(new_dir / "summary.json")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"error: cannot load summary.json from one of the dirs: {e}", file=sys.stderr)
        return 1

    if base.get("judge_model") != new.get("judge_model") or \
            base.get("rubric_version") != new.get("rubric_version"):
        print("comparison invalid: judge/rubric differ", file=sys.stderr)
        print(f"  base: judge={base.get('judge_model')!r} rubric={base.get('rubric_version')!r}",
              file=sys.stderr)
        print(f"  new:  judge={new.get('judge_model')!r} rubric={new.get('rubric_version')!r}",
              file=sys.stderr)
        return 2

    report, stats = build_report(base, new, str(base_dir), str(new_dir))
    print(report)
    out = new_dir / "model_compare.md"
    out.write_text(report)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
