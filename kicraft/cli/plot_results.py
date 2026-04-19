#!/usr/bin/env python3
"""Unified plotting CLI for KiCraft experiment and scoring dashboards.

Auto-detects the input format:
  - *.jsonl file  -> experiment-loop results  (was: plot-experiments)
  - directory     -> score_*.json results     (was: plot-scores)

Usage:
    plot-results experiments.jsonl              # experiment log
    plot-results results/                       # scoring directory
    plot-results input -o out.png               # explicit output
    plot-results input --format experiments     # force format
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_figure(n_panels: int):
    """Create a multi-panel figure with consistent sizing."""
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, 3.5 * n_panels),
        gridspec_kw={"hspace": 0.35},
    )
    if n_panels == 1:
        axes = [axes]
    return fig, axes


def _plot_drc_bars(ax, x, shorts, unconnected, third, fourth,
                   third_label: str, fourth_label: str, bar_w: float = 0.6):
    """Stacked bar chart for DRC violation counts."""
    ax.bar(x, shorts, bar_w, label="Shorts", color="#e74c3c", alpha=0.85)
    ax.bar(x, unconnected, bar_w, bottom=shorts,
           label="Unconnected", color="#e67e22", alpha=0.85)
    bot2 = [s + u for s, u in zip(shorts, unconnected)]
    ax.bar(x, third, bar_w, bottom=bot2, label=third_label,
           color="#f1c40f", alpha=0.85)
    bot3 = [b + c for b, c in zip(bot2, third)]
    ax.bar(x, fourth, bar_w, bottom=bot3, label=fourth_label,
           color="#95a5a6", alpha=0.85)

    ax.set_ylabel("DRC Violations", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3, axis="y")


def _save_figure(fig, output_path: str):
    """Save figure with project-standard settings."""
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Format A: experiments.jsonl
# ---------------------------------------------------------------------------

def _load_experiments(path: str):
    experiments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                experiments.append(json.loads(line))
    return experiments


def _plot_experiments(experiments, output_path: str):
    """Full dashboard for experiment-loop results."""
    from matplotlib.lines import Line2D

    rounds = list(range(1, len(experiments) + 1))
    scores = [e["score"] for e in experiments]
    modes = [e["mode"] for e in experiments]
    kept = [e["kept"] for e in experiments]

    has_breakdown = "placement_score" in experiments[0]
    has_drc = "drc_total" in experiments[0]
    has_timing = "placement_ms" in experiments[0]

    n_panels = 2 + int(has_breakdown) + int(has_drc) + int(has_timing)
    fig, axes = _make_figure(n_panels)
    ax_idx = 0

    # --- Panel 1: Score per round with best-so-far line ---
    ax = axes[ax_idx]; ax_idx += 1

    best, cur_best = [], 0
    for s, k in zip(scores, kept):
        if k:
            cur_best = s
        best.append(cur_best)

    for r, s, m, k in zip(rounds, scores, modes, kept):
        color = "#2ecc71" if k else ("#e74c3c" if m == "major" else "#95a5a6")
        marker = "D" if m == "major" else "o"
        ax.scatter(r, s, c=color, marker=marker, s=40, zorder=5,
                   edgecolors="black" if k else "none",
                   linewidths=1.2 if k else 0)

    first_kept = next((i for i, b in enumerate(best) if b > 0), 0)
    ax.plot(rounds[first_kept:], best[first_kept:], "k-", linewidth=2, alpha=0.7)

    nonzero = [s for s in scores if s > 0]
    if nonzero:
        ax.set_ylim(min(nonzero) - 1, max(nonzero) + 1)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Autoexperiment: PCB Layout Optimization",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
               markeredgecolor="black", markersize=7, label="Kept (minor)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#2ecc71",
               markeredgecolor="black", markersize=7, label="Kept (major)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6",
               markersize=7, label="Discarded (minor)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#e74c3c",
               markersize=7, label="Discarded (major)"),
        Line2D([0], [0], color="black", linewidth=2, label="Best so far"),
    ], loc="lower right", fontsize=7, ncol=3)

    # --- Panel 2 (optional): Category breakdown ---
    if has_breakdown:
        ax = axes[ax_idx]; ax_idx += 1
        categories = {
            "Placement":    [e.get("placement_score", 0) for e in experiments],
            "Route Compl.": [e.get("route_completion", 0) for e in experiments],
            "Trace Eff.":   [e.get("trace_efficiency", 0) for e in experiments],
            "Via Score":    [e.get("via_score", 0) for e in experiments],
            "Courtyard":    [e.get("courtyard_overlap", 0) for e in experiments],
            "Containment":  [e.get("board_containment", 0) for e in experiments],
        }
        colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c", "#1abc9c"]
        for (label, vals), c in zip(categories.items(), colors):
            ax.plot(rounds, vals, "-", color=c, alpha=0.7, linewidth=1.2, label=label)
        ax.set_ylabel("Category Score (0-100)", fontsize=10)
        ax.set_title("Scoring Breakdown by Category", fontsize=11)
        ax.set_ylim(-2, 105)
        ax.legend(loc="lower right", fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)

    # --- Panel 3 (optional): DRC violations ---
    if has_drc:
        ax = axes[ax_idx]; ax_idx += 1
        _plot_drc_bars(
            ax, rounds,
            shorts=[e.get("drc_shorts", 0) for e in experiments],
            unconnected=[e.get("drc_unconnected", 0) for e in experiments],
            third=[e.get("drc_clearance", 0) for e in experiments],
            fourth=[e.get("drc_courtyard", 0) for e in experiments],
            third_label="Clearance",
            fourth_label="Courtyard",
            bar_w=0.8,
        )
        ax.set_title("DRC Violations per Experiment", fontsize=11)

    # --- Panel 4 (optional): Phase timing ---
    if has_timing:
        ax = axes[ax_idx]; ax_idx += 1
        p_ms = [e.get("placement_ms", 0) for e in experiments]
        r_ms = [e.get("routing_ms", 0) for e in experiments]
        ax.bar(rounds, [v / 1000 for v in p_ms], 0.8,
               label="Placement", color="#3498db")
        ax.bar(rounds, [v / 1000 for v in r_ms], 0.8,
               bottom=[v / 1000 for v in p_ms],
               label="Routing", color="#2ecc71")
        ax.set_ylabel("Time (seconds)", fontsize=10)
        ax.set_title("Phase Timing per Round", fontsize=11)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3, axis="y")

    # --- Last panel: Config delta heatmap ---
    ax = axes[ax_idx]
    kept_exps = [e for e in experiments if e["kept"]]
    if kept_exps:
        all_keys = set()
        for e in kept_exps:
            all_keys.update(e.get("config_delta", {}).keys())
        all_keys = sorted(k for k in all_keys if any(
            isinstance(e.get("config_delta", {}).get(k), (int, float))
            for e in kept_exps
        ))
        if all_keys:
            data, labels = [], []
            for e in kept_exps:
                row = [
                    float(e["config_delta"].get(k, 0))
                    if isinstance(e["config_delta"].get(k, 0), (int, float))
                    else 0.0
                    for k in all_keys
                ]
                data.append(row)
                idx = next(i for i, exp in enumerate(experiments) if exp is e) + 1
                labels.append(f"#{idx}")

            ax.set_title(
                "Config Values (Kept Experiments) \u2014 per-param normalized",
                fontsize=11,
            )
            if data and data[0]:
                arr = np.array(data).T
                for ri in range(arr.shape[0]):
                    lo, hi = arr[ri].min(), arr[ri].max()
                    if hi - lo > 1e-9:
                        arr[ri] = (arr[ri] - lo) / (hi - lo)
                    else:
                        arr[ri] = 0.5
                im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_yticks(range(len(all_keys)))
                raw = np.array(data).T
                ylabels = []
                for ki, k in enumerate(all_keys):
                    lo, hi = raw[ki].min(), raw[ki].max()
                    ylabels.append(f"{k.replace('_', ' ')}\n[{lo:.3g}-{hi:.3g}]")
                ax.set_yticklabels(ylabels, fontsize=7)
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label("Normalized (0=min, 1=max)", fontsize=8)
            else:
                ax.text(0.5, 0.5, "No param changes in kept experiments",
                        ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No param changes in kept experiments",
                    ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "No experiments kept (baseline was best)",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Experiment Round", fontsize=11)

    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Format B: score_*.json  (+optional session.json)
# ---------------------------------------------------------------------------

def _load_score_results(results_dir: str):
    files = sorted(glob.glob(os.path.join(results_dir, "score_*.json")))
    runs = []
    for fpath in files:
        with open(fpath) as fh:
            data = json.load(fh)
        ts = datetime.fromisoformat(data["timestamp"])
        entry = {
            "timestamp": ts,
            "overall": data["overall_score"],
            "file": os.path.basename(fpath),
            "tokens": data.get("token_usage", {}).get("total_tokens", 0),
        }
        for cat_name, cat_data in data["categories"].items():
            entry[cat_name] = cat_data["score"]
            entry[f"{cat_name}_weight"] = cat_data["weight"]
            if cat_name == "drc_markers":
                metrics = cat_data.get("metrics", {})
                entry["shorts"] = metrics.get("shorts", 0)
                entry["unconnected"] = metrics.get("unconnected", 0)
                entry["crossings"] = metrics.get("crossings", 0)
                entry["clearance"] = metrics.get("major", 0)
        runs.append(entry)
    return runs


def _load_session(results_dir: str):
    path = os.path.join(results_dir, "session.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def _plot_scores(runs, session, output_path: str):
    """Full dashboard for scoring-run results."""
    if len(runs) < 2:
        print("Need at least 2 scoring runs to plot.")
        return

    indices = list(range(len(runs)))
    has_drc = any("shorts" in r for r in runs)
    has_tokens = any(r.get("tokens", 0) > 0 for r in runs)
    has_session = session and len(session.get("iterations", [])) > 0

    nrows = 2 + int(has_drc) + int(has_tokens)
    fig, axes = _make_figure(nrows)
    ax_idx = 0

    # --- Panel 1: Overall score (with change-classification background) ---
    ax = axes[ax_idx]; ax_idx += 1
    overall = [r["overall"] for r in runs]

    if has_session:
        iters = session["iterations"]
        cls_colors = {
            "no_change": "#f0f0f0",
            "minor_tweak": "#e8f5e9",
            "moderate_rework": "#fff3e0",
            "major_redesign": "#ffebee",
            "baseline": "#e3f2fd",
        }
        for i, it in enumerate(iters):
            if i >= len(runs):
                break
            cls = (it.get("changes", {}).get("classification", "baseline")
                   if it.get("changes") else "baseline")
            ax.axvspan(i - 0.4, i + 0.4,
                       color=cls_colors.get(cls, "#f0f0f0"), alpha=0.6)

    ax.plot(indices, overall, "k-o", linewidth=2.5, markersize=8, zorder=5)
    ax.fill_between(indices, overall, alpha=0.08, color="black")
    for i, v in enumerate(overall):
        ax.annotate(f"{v:.0f}", (i, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score (0-100)", fontsize=11)
    ax.set_title("PCB Layout Score", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    if has_session:
        ax.legend(handles=[
            mpatches.Patch(color="#e3f2fd", label="Baseline"),
            mpatches.Patch(color="#e8f5e9", label="Minor Tweak"),
            mpatches.Patch(color="#fff3e0", label="Moderate Rework"),
            mpatches.Patch(color="#ffebee", label="Major Redesign"),
        ], loc="lower left", fontsize=7, ncol=4)

    # --- Panel 2: Per-category breakdown ---
    ax = axes[ax_idx]; ax_idx += 1
    scored_cats = [
        k for k in runs[0]
        if not k.endswith("_weight")
        and k not in ("timestamp", "overall", "file", "tokens",
                      "shorts", "unconnected", "crossings", "clearance")
        and runs[0].get(f"{k}_weight", 0) > 0
    ]
    cat_colors = {
        "drc_markers": "#e74c3c", "trace_widths": "#e67e22",
        "connectivity": "#2ecc71", "placement": "#3498db",
        "vias": "#9b59b6", "geometry": "#1abc9c",
        "compactness": "#f39c12", "orientation": "#d35400",
    }
    for cat in scored_cats:
        vals = [r.get(cat, 0) for r in runs]
        w = runs[-1].get(f"{cat}_weight", 0)
        label = f"{cat.replace('_', ' ').title()} ({w:.0%})"
        ax.plot(indices, vals, "-o", color=cat_colors.get(cat, "#666"),
                linewidth=1.5, markersize=4, label=label)
    ax.set_ylabel("Category Score", fontsize=11)
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # --- Panel 3 (optional): DRC violations ---
    if has_drc:
        ax = axes[ax_idx]; ax_idx += 1
        _plot_drc_bars(
            ax, np.array(indices),
            shorts=[r.get("shorts", 0) for r in runs],
            unconnected=[r.get("unconnected", 0) for r in runs],
            third=[r.get("crossings", 0) for r in runs],
            fourth=[r.get("clearance", 0) for r in runs],
            third_label="Crossings",
            fourth_label="Clearance",
        )
        # Total annotations
        for i in indices:
            t = sum(runs[i].get(k, 0)
                    for k in ("shorts", "unconnected", "crossings", "clearance"))
            if t > 0:
                ax.annotate(f"{t}", (i, t), textcoords="offset points",
                            xytext=(0, 4), ha="center", fontsize=8)

    # --- Panel 4 (optional): Token usage ---
    if has_tokens:
        ax = axes[ax_idx]; ax_idx += 1
        tokens = [r.get("tokens", 0) for r in runs]
        ax.bar(indices, tokens, 0.6, color="#3498db", alpha=0.7)
        cumulative = list(np.cumsum(tokens))
        ax2 = ax.twinx()
        ax2.plot(indices, cumulative, "k--o", markersize=4, linewidth=1.5)
        ax2.set_ylabel("Cumulative Tokens", fontsize=10)
        for i, v in enumerate(tokens):
            if v > 0:
                ax.annotate(f"{v:,}", (i, v), textcoords="offset points",
                            xytext=(0, 4), ha="center", fontsize=7)
        ax.set_ylabel("Tokens / Run", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    # X-axis labels on last panel
    last_ax = axes[-1]
    last_ax.set_xticks(indices)
    last_ax.set_xticklabels(
        [f"#{i+1}\n{r['timestamp'].strftime('%H:%M')}" for i, r in enumerate(runs)],
        fontsize=7,
    )
    last_ax.set_xlabel("Iteration", fontsize=11)

    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _detect_format(path: str) -> str:
    """Return 'experiments' or 'scores' based on the input path."""
    if path.endswith(".jsonl"):
        return "experiments"
    if os.path.isdir(path):
        return "scores"
    # Fallback: peek at file contents
    if os.path.isfile(path):
        with open(path) as f:
            first = f.readline().strip()
        if first.startswith("{"):
            return "experiments"  # jsonl is one JSON object per line
    sys.exit(f"Cannot auto-detect format for '{path}'. "
             "Use --format experiments|scores.")


def main():
    parser = argparse.ArgumentParser(
        prog="plot-results",
        description="Plot experiment or scoring dashboards for KiCraft.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to experiments.jsonl file or scoring results directory.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PNG path (default: auto-derived from input).",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["experiments", "scores"],
        default=None,
        help="Force input format instead of auto-detecting.",
    )
    args = parser.parse_args()

    # Resolve input path
    input_path = args.input
    if input_path is None:
        # Try common defaults
        for candidate in ("experiments.jsonl", "results"):
            if os.path.exists(candidate):
                input_path = candidate
                break
        if input_path is None:
            parser.error("No input path given and no default found "
                         "(experiments.jsonl or results/).")

    fmt = args.format or _detect_format(input_path)

    if fmt == "experiments":
        output = args.output or input_path.replace(".jsonl", ".png")
        experiments = _load_experiments(input_path)
        print(f"Loaded {len(experiments)} experiments from {input_path}")
        _plot_experiments(experiments, output)
    else:
        output = args.output or os.path.join(input_path, "dashboard.png")
        runs = _load_score_results(input_path)
        session = _load_session(input_path)
        desc = f"Loaded {len(runs)} scoring runs from {input_path}"
        if session:
            desc += f", {len(session.get('iterations', []))} session iterations"
        print(desc)
        _plot_scores(runs, session, output)


if __name__ == "__main__":
    main()
