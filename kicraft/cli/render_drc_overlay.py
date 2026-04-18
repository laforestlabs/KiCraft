#!/usr/bin/env python3
"""Render DRC violations as visual overlays on a PCB snapshot.

Takes a PCB path + DRC violation list (from quick_drc) and produces
a PNG with violations highlighted:
  - Shorts: red X markers with connecting dashed lines
  - Unconnected: orange circles
  - Clearance: yellow halos
  - Courtyard: magenta rectangles

Usage:
    python3 render_drc_overlay.py <pcb_path> <round_json> [--output overlay.png]

Or use as a library:
    from render_drc_overlay import render_overlay
    render_overlay(pcb_path, violations, output_png, board_mm=(140, 90))
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import tempfile


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
        board_mm: Board dimensions (width, height) in mm
        canvas_px: Output canvas size in pixels

    Returns:
        True if successful, False otherwise
    """
    if not violations:
        return False

    # Filter violations that have coordinates
    located = [v for v in violations if v.get("x_mm") is not None]
    if not located:
        return False

    try:
        # Export base SVG
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            svg_path = tmp.name
        subprocess.run([
            "kicad-cli", "pcb", "export", "svg",
            "--layers", "F.Cu,B.Cu,F.SilkS,Edge.Cuts",
            "--mode-single", "--fit-page-to-board",
            "--exclude-drawing-sheet", "--drill-shape-opt", "2",
            "-o", svg_path, pcb_path,
        ], capture_output=True, check=True)

        # Compute board → pixel transform
        bw, bh = board_mm
        scale = (canvas_px * 0.80) / max(bw, bh)
        target_w = int(round(bw * scale))
        target_h = int(round(bh * scale))
        # Board origin offset (centered in canvas)
        ox = (canvas_px - target_w) / 2
        oy = (canvas_px - target_h) / 2

        # Parse board origin from PCB to get coordinate offset
        board_x0, board_y0 = _parse_board_origin(pcb_path)

        # Build ImageMagick command
        cmd = [
            "magick",
            "-density", "300",
            "-background", "white",
            svg_path,
            "-flatten",
            "-resize", f"{target_w}x{target_h}!",
            "-gravity", "center",
            "-extent", f"{canvas_px}x{canvas_px}",
        ]

        # Draw violation markers
        for v in located:
            # Convert mm to pixel coordinates
            px = ox + (v["x_mm"] - board_x0) * scale
            py = oy + (v["y_mm"] - board_y0) * scale
            vtype = v.get("type", "")
            r = max(8, int(scale * 1.5))  # marker radius

            if vtype == "shorting_items":
                # Red X marker
                cmd.extend([
                    "-fill", "none", "-stroke", "red", "-strokewidth", "3",
                    "-draw", f"line {px-r},{py-r} {px+r},{py+r}",
                    "-draw", f"line {px-r},{py+r} {px+r},{py-r}",
                    "-draw", f"circle {px},{py} {px+r+4},{py}",
                ])
            elif vtype == "unconnected_items":
                # Orange circle
                cmd.extend([
                    "-fill", "none", "-stroke", "orange", "-strokewidth", "2",
                    "-draw", f"circle {px},{py} {px+r},{py}",
                ])
            elif vtype in ("clearance", "hole_clearance", "copper_edge_clearance"):
                # Yellow halo
                cmd.extend([
                    "-fill", "rgba(255,255,0,0.3)", "-stroke", "yellow",
                    "-strokewidth", "2",
                    "-draw", f"circle {px},{py} {px+r+2},{py}",
                ])
            elif vtype == "courtyards_overlap":
                # Magenta rectangle
                cmd.extend([
                    "-fill", "none", "-stroke", "magenta", "-strokewidth", "2",
                    "-draw", f"rectangle {px-r},{py-r} {px+r},{py+r}",
                ])

        # Add legend
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
        os.remove(svg_path)
        return True

    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


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


def _parse_board_origin(pcb_path: str) -> tuple[float, float]:
    """Parse board outline origin from .kicad_pcb file."""
    import re
    try:
        with open(pcb_path) as f:
            text = f.read()
        m = re.search(
            r'\(gr_rect\s+\(start\s+([\d.\-]+)\s+([\d.\-]+)\)',
            text,
        )
        if m:
            return float(m.group(1)), float(m.group(2))
    except OSError:
        pass
    return 0.0, 0.0


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

    # Try to get board dimensions from the round JSON or parse from PCB
    import re
    board_mm = (140.0, 90.0)
    try:
        with open(args.pcb) as f:
            pcb_text = f.read()
        m = re.search(r'\(gr_rect\s+\(start\s+([\d.\-]+)\s+([\d.\-]+)\)\s+'
                      r'\(end\s+([\d.\-]+)\s+([\d.\-]+)\)', pcb_text)
        if m:
            x0, y0, x1, y1 = (float(v) for v in m.groups())
            board_mm = (abs(x1 - x0), abs(y1 - y0))
    except (OSError, ValueError):
        pass

    ok = render_overlay(args.pcb, violations, args.output,
                        board_mm=board_mm, canvas_px=args.canvas)
    if ok:
        print(f"DRC overlay saved: {args.output}")
    else:
        print("Failed to render DRC overlay.")


if __name__ == "__main__":
    main()
