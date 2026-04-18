#!/usr/bin/env python3
"""Plot experiment loop results from experiments.jsonl.

Usage:
    python3 plot_experiments.py [experiments.jsonl] [output.png]
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_experiments(path):
    experiments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                experiments.append(json.loads(line))
    return experiments


def plot_experiments(experiments, output_path):
    rounds = list(range(1, len(experiments) + 1))
    scores = [e["score"] for e in experiments]
    modes = [e["mode"] for e in experiments]
    kept = [e["kept"] for e in experiments]

    # Check if new fields are present
    has_breakdown = "placement_score" in experiments[0]
    has_drc = "drc_total" in experiments[0]
    has_timing = "placement_ms" in experiments[0]

    n_panels = 2 + int(has_breakdown) + int(has_drc) + int(has_timing)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.5 * n_panels),
                              gridspec_kw={"hspace": 0.35})
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    # --- Panel 1: Score per round with best line ---
    ax1 = axes[ax_idx]; ax_idx += 1

    # Running best
    best = []
    cur_best = 0
    for s, k in zip(scores, kept):
        if k:
            cur_best = s
        best.append(cur_best)

    for i, (r, s, m, k) in enumerate(zip(rounds, scores, modes, kept)):
        color = "#2ecc71" if k else ("#e74c3c" if m == "major" else "#95a5a6")
        marker = "D" if m == "major" else "o"
        ax1.scatter(r, s, c=color, marker=marker, s=40, zorder=5,
                    edgecolors="black" if k else "none", linewidths=1.2 if k else 0)

    first_kept_idx = next((i for i, b in enumerate(best) if b > 0), 0)
    ax1.plot(rounds[first_kept_idx:], best[first_kept_idx:],
             "k-", linewidth=2, alpha=0.7)

    all_scores_nonzero = [s for s in scores if s > 0]
    if all_scores_nonzero:
        ax1.set_ylim(min(all_scores_nonzero) - 1, max(all_scores_nonzero) + 1)

    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_title("Autoexperiment: PCB Layout Optimization", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    ax1.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
               markeredgecolor='black', markersize=7, label='Kept (minor)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#2ecc71',
               markeredgecolor='black', markersize=7, label='Kept (major)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#95a5a6',
               markersize=7, label='Discarded (minor)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#e74c3c',
               markersize=7, label='Discarded (major)'),
        Line2D([0], [0], color='black', linewidth=2, label='Best so far'),
    ], loc="lower right", fontsize=7, ncol=3)

    # --- Panel 2: Category score breakdown ---
    if has_breakdown:
        ax2 = axes[ax_idx]; ax_idx += 1

        categories = {
            "Placement":    [e.get("placement_score", 0) for e in experiments],
            "Route Compl.": [e.get("route_completion", 0) for e in experiments],
            "Trace Eff.":   [e.get("trace_efficiency", 0) for e in experiments],
            "Via Score":    [e.get("via_score", 0) for e in experiments],
            "Courtyard":    [e.get("courtyard_overlap", 0) for e in experiments],
            "Containment":  [e.get("board_containment", 0) for e in experiments],
        }
        colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c", "#1abc9c"]

        for (label, vals), color in zip(categories.items(), colors):
            ax2.plot(rounds, vals, "-", color=color, alpha=0.7, linewidth=1.2, label=label)

        ax2.set_ylabel("Category Score (0-100)", fontsize=10)
        ax2.set_title("Scoring Breakdown by Category", fontsize=11)
        ax2.set_ylim(-2, 105)
        ax2.legend(loc="lower right", fontsize=7, ncol=3)
        ax2.grid(True, alpha=0.3)

    # --- Panel 3: DRC violations ---
    if has_drc:
        ax3 = axes[ax_idx]; ax_idx += 1

        shorts =      [e.get("drc_shorts", 0) for e in experiments]
        unconnected = [e.get("drc_unconnected", 0) for e in experiments]
        clearance =   [e.get("drc_clearance", 0) for e in experiments]
        courtyard =   [e.get("drc_courtyard", 0) for e in experiments]

        bar_w = 0.8
        ax3.bar(rounds, shorts, bar_w, label="Shorts", color="#e74c3c")
        ax3.bar(rounds, unconnected, bar_w, bottom=shorts, label="Unconnected", color="#e67e22")
        bottoms2 = [s + u for s, u in zip(shorts, unconnected)]
        ax3.bar(rounds, clearance, bar_w, bottom=bottoms2, label="Clearance", color="#f1c40f")
        bottoms3 = [b + c for b, c in zip(bottoms2, clearance)]
        ax3.bar(rounds, courtyard, bar_w, bottom=bottoms3, label="Courtyard", color="#95a5a6")

        ax3.set_ylabel("DRC Violations", fontsize=10)
        ax3.set_title("DRC Violations per Experiment", fontsize=11)
        ax3.legend(loc="upper right", fontsize=7, ncol=4)
        ax3.grid(True, alpha=0.3, axis="y")

    # --- Panel: Phase Timing Breakdown ---
    if has_timing:
        ax_t = axes[ax_idx]; ax_idx += 1

        p_ms = [e.get("placement_ms", 0) for e in experiments]
        r_ms = [e.get("routing_ms", 0) for e in experiments]

        bar_w = 0.8
        ax_t.bar(rounds, [v / 1000 for v in p_ms], bar_w,
                 label="Placement", color="#3498db")
        ax_t.bar(rounds, [v / 1000 for v in r_ms], bar_w,
                 bottom=[v / 1000 for v in p_ms],
                 label="Routing", color="#2ecc71")

        ax_t.set_ylabel("Time (seconds)", fontsize=10)
        ax_t.set_title("Phase Timing per Round", fontsize=11)
        ax_t.legend(loc="upper right", fontsize=7, ncol=2)
        ax_t.grid(True, alpha=0.3, axis="y")

    # --- Last Panel: Config delta heatmap ---
    ax_last = axes[ax_idx]

    kept_exps = [e for e in experiments if e["kept"]]
    if kept_exps:
        all_keys = set()
        for e in kept_exps:
            all_keys.update(e.get("config_delta", {}).keys())
        # Filter to numeric-only keys (skip lists, dicts, strings, bools)
        all_keys = sorted(k for k in all_keys if any(
            isinstance(e.get("config_delta", {}).get(k), (int, float))
            for e in kept_exps
        ))

        if all_keys:
            data = []
            labels = []
            for e in kept_exps:
                row = [float(e["config_delta"].get(k, 0))
                       if isinstance(e["config_delta"].get(k, 0), (int, float))
                       else 0.0
                       for k in all_keys]
                data.append(row)
                kept_idx = next(i for i, exp in enumerate(experiments) if exp is e) + 1
                labels.append(f"#{kept_idx}")

            ax_last.set_title("Config Values (Kept Experiments) — per-param normalized", fontsize=11)
            if len(data) > 0 and len(data[0]) > 0:
                arr = np.array(data).T
                for row_i in range(arr.shape[0]):
                    row_min, row_max = arr[row_i].min(), arr[row_i].max()
                    if row_max - row_min > 1e-9:
                        arr[row_i] = (arr[row_i] - row_min) / (row_max - row_min)
                    else:
                        arr[row_i] = 0.5
                im = ax_last.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
                ax_last.set_xticks(range(len(labels)))
                ax_last.set_xticklabels(labels, fontsize=8)
                ax_last.set_yticks(range(len(all_keys)))
                raw = np.array(data).T
                ylabels = []
                for ki, k in enumerate(all_keys):
                    lo, hi = raw[ki].min(), raw[ki].max()
                    ylabels.append(f"{k.replace('_', ' ')}\n[{lo:.3g}–{hi:.3g}]")
                ax_last.set_yticklabels(ylabels, fontsize=7)
                cbar = plt.colorbar(im, ax=ax_last, shrink=0.8)
                cbar.set_label("Normalized (0=min, 1=max)", fontsize=8)
            else:
                ax_last.text(0.5, 0.5, "No param changes in kept experiments",
                             ha="center", va="center", transform=ax_last.transAxes)
        else:
            ax_last.text(0.5, 0.5, "No param changes in kept experiments",
                         ha="center", va="center", transform=ax_last.transAxes)
    else:
        ax_last.text(0.5, 0.5, "No experiments kept (baseline was best)",
                     ha="center", va="center", transform=ax_last.transAxes)

    ax_last.set_xlabel("Experiment Round", fontsize=11)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")


def main():
    default_log = os.path.join(os.path.dirname(__file__),
                               ".experiments", "experiments.jsonl")
    log_path = sys.argv[1] if len(sys.argv) > 1 else default_log
    output = sys.argv[2] if len(sys.argv) > 2 else log_path.replace(".jsonl", ".png")

    experiments = load_experiments(log_path)
    print(f"Loaded {len(experiments)} experiments")
    plot_experiments(experiments, output)


if __name__ == "__main__":
    main()
