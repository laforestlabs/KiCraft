#!/usr/bin/env python3
"""Render a failure heatmap from round detail JSONs.

Aggregates per-net routing failure frequency across all rounds and
renders a board-space heatmap showing hot zones where routing
consistently fails. Reads pad positions from the PCB to map net
failures to board coordinates.

Usage:
    python3 render_failure_heatmap.py <experiments_dir> <pcb_path> [--output heatmap.png]
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_round_details(rounds_dir: str) -> list[dict]:
    """Load all round detail JSONs."""
    details = []
    for path in sorted(glob.glob(os.path.join(rounds_dir, "round_*.json"))):
        with open(path) as f:
            details.append(json.load(f))
    return details


def aggregate_failures(details: list[dict]) -> dict[str, dict]:
    """Build per-net failure stats from round details.

    Returns: {net_name: {total_rounds, failures, success_rate, reasons}}
    """
    stats: dict[str, dict] = {}
    for d in details:
        per_net = d.get("per_net", [])
        for nr in per_net:
            name = nr.get("net", "")
            if not name:
                continue
            if name not in stats:
                stats[name] = {"total": 0, "failures": 0, "reasons": {}}
            stats[name]["total"] += 1
            if not nr.get("success", True):
                stats[name]["failures"] += 1
                reason = nr.get("failure_reason", "unknown") or "unknown"
                stats[name]["reasons"][reason] = (
                    stats[name]["reasons"].get(reason, 0) + 1
                )

    # Compute success rate
    for name, s in stats.items():
        s["success_rate"] = (
            (s["total"] - s["failures"]) / s["total"]
            if s["total"] > 0
            else 1.0
        )
    return stats


def parse_pad_positions(pcb_path: str) -> dict[str, list[tuple[float, float]]]:
    """Parse net → pad positions from .kicad_pcb file.

    Simple regex-based parser — extracts pad positions and net assignments.
    Returns: {net_name: [(x_mm, y_mm), ...]}
    """
    import re
    net_pads: dict[str, list[tuple[float, float]]] = {}
    try:
        with open(pcb_path) as f:
            text = f.read()
    except OSError:
        return net_pads

    # Find net definitions: (net N "name")
    net_map = {}
    for m in re.finditer(r'\(net\s+(\d+)\s+"([^"]+)"\)', text):
        net_map[m.group(1)] = m.group(2)

    # Find footprints and their pads with positions
    # This is a simplified parser — matches at(x y) and net assignment within pad blocks
    re.compile(
        r'\(footprint\b[^)]*\(at\s+([\d.\-]+)\s+([\d.\-]+)(?:\s+[\d.\-]+)?\)'
    )
    re.compile(
        r'\(pad\b[^)]*\(at\s+([\d.\-]+)\s+([\d.\-]+)[^)]*\)'
        r'.*?\(net\s+(\d+)\s+"[^"]*"\)',
        re.DOTALL,
    )

    # Simpler approach: find all pads with net assignments
    for m in re.finditer(
        r'\(pad\s+\S+\s+\S+\s+\S+[^)]*\(at\s+([\d.\-]+)\s+([\d.\-]+)[^)]*\)'
        r'[^)]*(?:\([^)]*\))*[^)]*\(net\s+(\d+)\s+"([^"]+)"\)',
        text,
    ):
        x, y = float(m.group(1)), float(m.group(2))
        net_name = m.group(4)
        if net_name not in net_pads:
            net_pads[net_name] = []
        net_pads[net_name].append((x, y))

    return net_pads


def parse_board_outline(pcb_path: str) -> tuple[float, float, float, float]:
    """Parse board outline rect: (x0, y0, x1, y1)."""
    import re
    try:
        with open(pcb_path) as f:
            text = f.read()
        m = re.search(
            r'\(gr_rect\s+\(start\s+([\d.\-]+)\s+([\d.\-]+)\)\s+'
            r'\(end\s+([\d.\-]+)\s+([\d.\-]+)\)',
            text,
        )
        if m:
            return tuple(float(v) for v in m.groups())
    except OSError:
        pass
    return (0, 0, 90, 58)


def render_heatmap(
    net_stats: dict[str, dict],
    net_pads: dict[str, list[tuple[float, float]]],
    board_outline: tuple[float, float, float, float],
    output_path: str,
    grid_resolution: float = 1.0,
) -> None:
    """Render failure heatmap to PNG.

    Args:
        net_stats: Per-net failure aggregation from aggregate_failures()
        net_pads: Net → pad positions from parse_pad_positions()
        board_outline: (x0, y0, x1, y1) in mm
        output_path: Output PNG path
        grid_resolution: Grid cell size in mm (default 1mm)
    """
    x0, y0, x1, y1 = board_outline
    bw = abs(x1 - x0)
    bh = abs(y1 - y0)
    nx = int(bw / grid_resolution) + 1
    ny = int(bh / grid_resolution) + 1
    heat = np.zeros((ny, nx), dtype=float)

    # For each net with failures, distribute failure weight to pad locations
    for net_name, stats in net_stats.items():
        if stats["failures"] == 0:
            continue
        pads = net_pads.get(net_name, [])
        if not pads:
            continue
        # Weight = failure rate
        weight = stats["failures"] / max(stats["total"], 1)
        for px, py in pads:
            gx = int((px - min(x0, x1)) / grid_resolution)
            gy = int((py - min(y0, y1)) / grid_resolution)
            if 0 <= gx < nx and 0 <= gy < ny:
                heat[gy, gx] += weight

    # Apply gaussian blur for smoother visualization
    try:
        from scipy.ndimage import gaussian_filter
        heat = gaussian_filter(heat, sigma=2.0)
    except ImportError:
        pass  # No scipy — use raw data

    fig, ax = plt.subplots(1, 1, figsize=(10, 10 * bh / bw))

    # Draw heatmap
    extent = [min(x0, x1), max(x0, x1), max(y0, y1), min(y0, y1)]
    if heat.max() > 0:
        im = ax.imshow(
            heat,
            extent=extent,
            cmap="YlOrRd",
            interpolation="bilinear",
            alpha=0.7,
            aspect="equal",
        )
        plt.colorbar(im, ax=ax, label="Failure intensity", shrink=0.8)
    else:
        ax.text(
            (x0 + x1) / 2, (y0 + y1) / 2,
            "No routing failures detected",
            ha="center", va="center", fontsize=14,
        )

    # Draw board outline
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (min(x0, x1), min(y0, y1)), bw, bh,
        fill=False, edgecolor="black", linewidth=2,
    ))

    # Annotate top-failing nets
    failing_nets = sorted(
        ((name, s) for name, s in net_stats.items() if s["failures"] > 0),
        key=lambda x: x[1]["failures"],
        reverse=True,
    )[:10]

    for net_name, stats in failing_nets:
        pads = net_pads.get(net_name, [])
        if not pads:
            continue
        cx = sum(p[0] for p in pads) / len(pads)
        cy = sum(p[1] for p in pads) / len(pads)
        rate = stats["failures"] / stats["total"] * 100
        ax.annotate(
            f"{net_name}\n{rate:.0f}% fail",
            (cx, cy),
            fontsize=6,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

    ax.set_title("Routing Failure Heatmap", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_xlim(min(x0, x1) - 2, max(x0, x1) + 2)
    ax.set_ylim(max(y0, y1) + 2, min(y0, y1) - 2)  # Y inverted for PCB coords

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Failure heatmap saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate routing failure heatmap from experiment rounds")
    parser.add_argument("experiments_dir",
                        help="Path to .experiments directory")
    parser.add_argument("pcb", help="Path to .kicad_pcb file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PNG path (default: <experiments_dir>/failure_heatmap.png)")
    parser.add_argument("--resolution", "-r", type=float, default=1.0,
                        help="Grid cell size in mm (default: 1.0)")
    args = parser.parse_args()

    rounds_dir = os.path.join(args.experiments_dir, "rounds")
    if not os.path.isdir(rounds_dir):
        print(f"No rounds/ directory found in {args.experiments_dir}")
        sys.exit(1)

    details = load_round_details(rounds_dir)
    if not details:
        print("No round detail JSONs found.")
        sys.exit(1)

    print(f"Loaded {len(details)} round details")

    net_stats = aggregate_failures(details)
    failing = {n: s for n, s in net_stats.items() if s["failures"] > 0}
    print(f"Found {len(failing)} nets with failures out of {len(net_stats)} total")

    net_pads = parse_pad_positions(args.pcb)
    board_outline = parse_board_outline(args.pcb)

    output = args.output or os.path.join(args.experiments_dir, "failure_heatmap.png")
    render_heatmap(net_stats, net_pads, board_outline, output,
                   grid_resolution=args.resolution)


if __name__ == "__main__":
    main()
