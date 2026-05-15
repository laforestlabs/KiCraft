#!/usr/bin/env python3
"""Render DRC violations as visual overlays on a PCB snapshot.

Takes a PCB path + DRC violation list (from quick_drc) and produces
a PNG with violations highlighted:
  - Shorts: red X markers with connecting dashed lines
  - Unconnected: orange circles
  - Clearance: yellow halos
  - Courtyard: magenta rectangles

The base PCB image is produced by the unified ``kicraft.render``
pipeline (same renderer the monitor, manual layout canvas, and CLI
previews use) so the DRC overlay's pixels match what the user sees
elsewhere. This module only adds the violation markers on top.

Usage:
    python3 render_drc_overlay.py <pcb_path> <round_json> [--output overlay.png]

Or use as a library:
    from kicraft.cli.render_drc_overlay import render_overlay
    render_overlay(pcb_path, violations, output_png, board_mm=(140, 90))
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from kicraft.render import render_pcb
from kicraft.render.edge_cuts import parse_edge_cuts_aabb


# Use the same layer set the monitor's "front_all" view uses for the
# base PCB image so the overlay reads on top of the picture the user
# would otherwise be looking at. Rendered with style=None (transparent
# background) so the white canvas underneath shows through outside
# Edge.Cuts.
_BASE_LAYERS = "F.Cu,B.Cu,F.SilkS,Edge.Cuts"


def render_overlay(
    pcb_path: str,
    violations: list[dict],
    output_png: str,
    board_mm: tuple[float, float] = (140.0, 90.0),
    canvas_px: int = 1200,
) -> bool:
    """Render PCB with DRC violation overlays.

    Args:
        pcb_path: Path to .kicad_pcb file
        violations: List of dicts with keys: type, x_mm, y_mm, net1, net2
        output_png: Output PNG path
        board_mm: Board dimensions (width, height) in mm -- kept for
            backwards-compatible API; the real board extent is parsed
            from the PCB's Edge.Cuts.
        canvas_px: Output canvas size in pixels

    Returns:
        True if successful, False otherwise
    """
    if not violations:
        return False

    located = [v for v in violations if v.get("x_mm") is not None]
    if not located:
        return False

    # Read the true Edge.Cuts AABB; fall back to the caller's board_mm
    # only if the file has no Edge.Cuts geometry.
    ec = parse_edge_cuts_aabb(Path(pcb_path))
    if ec is not None:
        board_x0, board_y0, board_x1, board_y1 = ec
        bw, bh = board_x1 - board_x0, board_y1 - board_y0
    else:
        bw, bh = board_mm
        board_x0, board_y0 = 0.0, 0.0

    scale = (canvas_px * 0.80) / max(bw, bh)
    target_w = int(round(bw * scale))
    target_h = int(round(bh * scale))
    ox = (canvas_px - target_w) / 2
    oy = (canvas_px - target_h) / 2

    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            base_png = tmp.name
        extent = render_pcb(
            Path(pcb_path),
            Path(base_png),
            layers=_BASE_LAYERS,
            style=None,
        )
        if extent is None:
            return False

        # Composite the Edge.Cuts-clipped base PNG onto a white
        # canvas_px square, board scaled and centered, then draw the
        # violation markers on top.
        cmd = [
            "magick",
            "-size", f"{canvas_px}x{canvas_px}",
            "xc:white",
            "(",
            base_png,
            "-resize", f"{target_w}x{target_h}!",
            ")",
            "-gravity", "center",
            "-compose", "Over", "-composite",
            "-gravity", "NorthWest",
        ]

        for v in located:
            px = ox + (v["x_mm"] - board_x0) * scale
            py = oy + (v["y_mm"] - board_y0) * scale
            vtype = v.get("type", "")
            r = max(8, int(scale * 1.5))

            if vtype == "shorting_items":
                cmd.extend([
                    "-fill", "none", "-stroke", "red", "-strokewidth", "3",
                    "-draw", f"line {px-r},{py-r} {px+r},{py+r}",
                    "-draw", f"line {px-r},{py+r} {px+r},{py-r}",
                    "-draw", f"circle {px},{py} {px+r+4},{py}",
                ])
            elif vtype == "unconnected_items":
                cmd.extend([
                    "-fill", "none", "-stroke", "orange", "-strokewidth", "2",
                    "-draw", f"circle {px},{py} {px+r},{py}",
                ])
            elif vtype in ("clearance", "hole_clearance", "copper_edge_clearance"):
                cmd.extend([
                    "-fill", "rgba(255,255,0,0.3)", "-stroke", "yellow",
                    "-strokewidth", "2",
                    "-draw", f"circle {px},{py} {px+r+2},{py}",
                ])
            elif vtype == "courtyards_overlap":
                cmd.extend([
                    "-fill", "none", "-stroke", "magenta", "-strokewidth", "2",
                    "-draw", f"rectangle {px-r},{py-r} {px+r},{py+r}",
                ])

        font_size = max(14, canvas_px // 60)
        legend_y = 20
        for label, color in [("SHORT", "red"), ("UNCONNECTED", "orange"),
                              ("CLEARANCE", "yellow"), ("COURTYARD", "magenta")]:
            count = sum(1 for v in located if _vtype_matches(v.get("type", ""), label))
            if count > 0:
                cmd.extend([
                    "-fill", color, "-stroke", "none",
                    "-gravity", "NorthEast",
                    "-pointsize", str(font_size),
                    "-annotate", f"+10+{legend_y}",
                    f"{label}: {count}",
                ])
                legend_y += font_size + 4

        cmd.append(output_png)
        subprocess.run(cmd, capture_output=True, check=True)
        return True

    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False
    finally:
        try:
            os.remove(base_png)
        except (OSError, NameError):
            pass


def _vtype_matches(vtype: str, label: str) -> bool:
    mapping = {
        "SHORT": "shorting_items",
        "UNCONNECTED": "unconnected_items",
        "CLEARANCE": ("clearance", "hole_clearance", "copper_edge_clearance"),
        "COURTYARD": "courtyards_overlap",
    }
    expected = mapping.get(label, "")
    if isinstance(expected, tuple):
        return vtype in expected
    return vtype == expected


def main():
    parser = argparse.ArgumentParser(
        description="Render DRC violation overlay on PCB snapshot")
    parser.add_argument("pcb", help="Path to .kicad_pcb file")
    parser.add_argument("round_json", help="Path to round detail JSON")
    parser.add_argument("--output", "-o", default="drc_overlay.png",
                        help="Output PNG path")
    parser.add_argument("--canvas", type=int, default=1200,
                        help="Canvas size in pixels (default: 1200)")
    args = parser.parse_args()

    with open(args.round_json) as f:
        detail = json.load(f)

    violations = detail.get("drc", {}).get("violations", [])
    if not violations:
        print("No DRC violations with coordinates found.")
        return

    ok = render_overlay(args.pcb, violations, args.output, canvas_px=args.canvas)
    if ok:
        print(f"DRC overlay saved: {args.output}")
    else:
        print("Failed to render DRC overlay.")


if __name__ == "__main__":
    main()
