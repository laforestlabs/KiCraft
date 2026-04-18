#!/usr/bin/env python3
"""Plot scoring and session history as a multi-panel dashboard.

Reads both score JSON files and session.json to produce a comprehensive
view of layout progress: overall score, per-category breakdown, DRC issue
counts, token usage, and change classification.

Usage:
    python3 plot_scores.py [results_dir] [output.png]
"""
import glob
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_score_results(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "score_*.json")))
    runs = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        ts = datetime.fromisoformat(data["timestamp"])
        entry = {
            "timestamp": ts,
            "overall": data["overall_score"],
            "file": os.path.basename(f),
            "tokens": data.get("token_usage", {}).get("total_tokens", 0),
        }
        for cat_name, cat_data in data["categories"].items():
            entry[cat_name] = cat_data["score"]
            entry[f"{cat_name}_weight"] = cat_data["weight"]
            # DRC details
            if cat_name == "drc_markers":
                metrics = cat_data.get("metrics", {})
                entry["shorts"] = metrics.get("shorts", 0)
                entry["unconnected"] = metrics.get("unconnected", 0)
                entry["crossings"] = metrics.get("crossings", 0)
                entry["clearance"] = metrics.get("major", 0)
        runs.append(entry)
    return runs


def load_session(results_dir):
    path = os.path.join(results_dir, "session.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def plot_dashboard(runs, session, output_path):
    if len(runs) < 2:
        print("Need at least 2 runs to plot.")
        return

    indices = list(range(len(runs)))
    has_drc = any("shorts" in r for r in runs)
    has_tokens = any(r.get("tokens", 0) > 0 for r in runs)
    has_session = session and len(session.get("iterations", [])) > 0

    nrows = 2 + int(has_drc) + int(has_tokens)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 3.5 * nrows),
                              gridspec_kw={"hspace": 0.35})
    ax_idx = 0

    # --- Panel 1: Overall score with change classification background ---
    ax = axes[ax_idx]; ax_idx += 1
    overall = [r["overall"] for r in runs]

    # Color background by change classification from session
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
            cls = it.get("changes", {}).get("classification", "baseline") if it.get("changes") else "baseline"
            ax.axvspan(i - 0.4, i + 0.4, color=cls_colors.get(cls, "#f0f0f0"), alpha=0.6)

    ax.plot(indices, overall, "k-o", linewidth=2.5, markersize=8, zorder=5)
    ax.fill_between(indices, overall, alpha=0.08, color="black")
    for i, v in enumerate(overall):
        ax.annotate(f"{v:.0f}", (i, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score (0-100)", fontsize=11)
    ax.set_title("PCB Layout Score", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Legend for change classification
    if has_session:
        patches = [
            mpatches.Patch(color="#e3f2fd", label="Baseline"),
            mpatches.Patch(color="#e8f5e9", label="Minor Tweak"),
            mpatches.Patch(color="#fff3e0", label="Moderate Rework"),
            mpatches.Patch(color="#ffebee", label="Major Redesign"),
        ]
        ax.legend(handles=patches, loc="lower left", fontsize=7, ncol=4)

    # --- Panel 2: Per-category breakdown ---
    ax = axes[ax_idx]; ax_idx += 1
    scored_cats = [k for k in runs[0]
                   if k.endswith("_weight") is False
                   and k not in ("timestamp", "overall", "file", "tokens",
                                 "shorts", "unconnected", "crossings", "clearance")
                   and not k.endswith("_weight")
                   and runs[0].get(f"{k}_weight", 0) > 0]

    colors = {
        "drc_markers": "#e74c3c", "trace_widths": "#e67e22",
        "connectivity": "#2ecc71", "placement": "#3498db",
        "vias": "#9b59b6", "geometry": "#1abc9c",
        "compactness": "#f39c12", "orientation": "#d35400",
    }
    for cat in scored_cats:
        vals = [r.get(cat, 0) for r in runs]
        w = runs[-1].get(f"{cat}_weight", 0)
        label = f"{cat.replace('_', ' ').title()} ({w:.0%})"
        ax.plot(indices, vals, "-o", color=colors.get(cat, "#666"),
                linewidth=1.5, markersize=4, label=label)

    ax.set_ylabel("Category Score", fontsize=11)
    ax.legend(loc="lower right", ncol=2, fontsize=7)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: DRC issue counts (stacked bar) ---
    if has_drc:
        ax = axes[ax_idx]; ax_idx += 1
        shorts = [r.get("shorts", 0) for r in runs]
        unconnected = [r.get("unconnected", 0) for r in runs]
        crossings = [r.get("crossings", 0) for r in runs]
        clearance = [r.get("clearance", 0) for r in runs]

        x = np.array(indices)
        w = 0.6
        ax.bar(x, shorts, w, label="Shorts", color="#e74c3c", alpha=0.85)
        ax.bar(x, unconnected, w, bottom=shorts, label="Unconnected", color="#e67e22", alpha=0.85)
        bot2 = [s + u for s, u in zip(shorts, unconnected)]
        ax.bar(x, crossings, w, bottom=bot2, label="Crossings", color="#f1c40f", alpha=0.85)
        bot3 = [b + c for b, c in zip(bot2, crossings)]
        ax.bar(x, clearance, w, bottom=bot3, label="Clearance", color="#95a5a6", alpha=0.85)

        # Total label on top
        totals = [s + u + c + cl for s, u, c, cl in zip(shorts, unconnected, crossings, clearance)]
        for i, t in enumerate(totals):
            if t > 0:
                ax.annotate(f"{t}", (i, t), textcoords="offset points",
                            xytext=(0, 4), ha="center", fontsize=8)

        ax.set_ylabel("DRC Violations", fontsize=11)
        ax.legend(loc="upper right", fontsize=7, ncol=4)
        ax.grid(True, alpha=0.3, axis="y")

    # --- Panel 4: Token usage ---
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

    # X-axis labels
    last_ax = axes[-1]
    last_ax.set_xticks(indices)
    labels = []
    for i, r in enumerate(runs):
        t = r["timestamp"].strftime("%H:%M")
        labels.append(f"#{i+1}\n{t}")
    last_ax.set_xticklabels(labels, fontsize=7)
    last_ax.set_xlabel("Iteration", fontsize=11)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Dashboard saved to: {output_path}")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else RESULTS_DIR
    output = sys.argv[2] if len(sys.argv) > 2 else os.path.join(results_dir, "dashboard.png")
    runs = load_score_results(results_dir)
    session = load_session(results_dir)
    print(f"Loaded {len(runs)} scoring runs" +
          (f", {len(session['iterations'])} session iterations" if session else ""))
    plot_dashboard(runs, session, output)


if __name__ == "__main__":
    main()
