#!/usr/bin/env python3
"""Render PCB layers to PNG images for visual analysis.

CLI on top of the unified ``kicraft.render.render_pcb`` pipeline. The
``VIEWS`` registry below names each preset (layer set + chrome style),
and each view delegates the actual rasterization to the unified
renderer so the monitor / pipeline-graph PNGs and the manual layout
canvas PNG come out of the same code path.

Why the styled previews look as they do:
- tightly clipped to the board outline (Edge.Cuts) so off-board silk
  text never leaks into the preview
- dark surround so the board edge is visible against a navy chrome
- cyan accent border so the rendered tile is easy to spot
- gentle contrast/saturation boost for readability
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from kicraft.render import MonitorStyle, render_pcb

# Layer sets + chrome configs for each preset view. Each entry's
# ``post`` keys map 1:1 onto ``MonitorStyle`` fields.
VIEWS = {
    "front_all": {
        # Top-down PCBnew-like view. F.Cu+B.Cu together triggers the
        # composite path in the renderer (B.Cu rendered at reduced
        # opacity so it doesn't obscure front detail). F.Mask is
        # intentionally OMITTED: KiCad renders it as an opaque
        # solder-mask-colored fill over the whole board, which would
        # obscure B.Cu and produce a "blank blue PCB" when the router
        # put traces on the back layer.
        #
        # Post settings deliberately gentle (contrast 1.15, sat 1.05):
        # the heavier boost used elsewhere washed out B.Cu against
        # the saturated background.
        "layers": "B.Cu,F.Cu,F.SilkS,Edge.Cuts",
        "desc": "Top-down view: both copper layers + silkscreen + outline (PCBnew-like)",
        "post": {
            "contrast": 1.15,
            "saturation": 1.05,
            "brightness": 1.00,
            "background": "#020617",
            "border_color": "#67e8f9",
            "border_width": 6,
            "padding": 52,
        },
    },
    "back_all": {
        "layers": "B.Cu,B.SilkS,B.Mask,Edge.Cuts",
        "desc": "Back copper + silkscreen + mask + outline",
        "mirror": True,
        "post": {
            "contrast": 1.38,
            "saturation": 1.24,
            "brightness": 0.90,
            "background": "#020617",
            "border_color": "#67e8f9",
            "border_width": 6,
            "padding": 52,
        },
    },
    "copper_both": {
        "layers": "F.Cu,B.Cu,Edge.Cuts",
        "desc": "Both copper layers + outline",
        "post": {
            "contrast": 1.34,
            "saturation": 1.18,
            "brightness": 0.90,
            "background": "#020617",
            "border_color": "#22d3ee",
            "border_width": 6,
            "padding": 52,
        },
    },
    "front_copper": {
        "layers": "F.Cu,Edge.Cuts",
        "desc": "Front copper traces and pads only",
        "post": {
            "contrast": 1.30,
            "saturation": 1.12,
            "brightness": 0.90,
            "background": "#020617",
            "border_color": "#22d3ee",
            "border_width": 6,
            "padding": 52,
        },
    },
    "back_copper": {
        "layers": "B.Cu,Edge.Cuts",
        "desc": "Back copper (ground plane, traces)",
        "mirror": True,
        "post": {
            "contrast": 1.30,
            "saturation": 1.12,
            "brightness": 0.90,
            "background": "#020617",
            "border_color": "#22d3ee",
            "border_width": 6,
            "padding": 52,
        },
    },
    "courtyard": {
        "layers": "F.CrtYd,B.CrtYd,Edge.Cuts",
        "desc": "Component courtyards for overlap review",
        "post": {
            "contrast": 1.34,
            "saturation": 1.02,
            "brightness": 0.90,
            "background": "#030712",
            "border_color": "#c4b5fd",
            "border_width": 6,
            "padding": 52,
        },
    },
}

DEFAULT_DPI = 420
DEFAULT_MAX_PX = 3200


def _which_or_warn(name: str) -> str | None:
    path = shutil.which(name)
    if path is None:
        print(f"error: required executable not found on PATH: {name}", file=sys.stderr)
    return path


def render_view(
    pcb_path: str,
    view_name: str,
    view_cfg: dict,
    output_dir: str,
    dpi: int = DEFAULT_DPI,
    max_px: int = DEFAULT_MAX_PX,
) -> str | None:
    """Render a single named view to PNG. Returns the output path on
    success or None on failure. Edge.Cuts clipping, F.Cu+B.Cu compositing,
    and chrome are all handled by the unified renderer."""
    png_path = os.path.join(output_dir, f"{view_name}.png")
    post = dict(view_cfg.get("post") or {})
    style = MonitorStyle(
        background=post.get("background", "#020617"),
        border_color=post.get("border_color", "#67e8f9"),
        border_width=int(post.get("border_width", 6)),
        padding=int(post.get("padding", 52)),
        contrast=float(post.get("contrast", 1.12)),
        saturation=float(post.get("saturation", 1.08)),
        brightness=float(post.get("brightness", 1.00)),
        max_px=max_px,
    )
    extent = render_pcb(
        Path(pcb_path),
        Path(png_path),
        layers=view_cfg["layers"],
        mirror=bool(view_cfg.get("mirror")),
        dpi=dpi,
        style=style,
    )
    return png_path if extent is not None else None


def render_all(pcb_path, output_dir, views=None):
    """Render all (or selected) views. Returns dict of view_name -> png_path."""
    os.makedirs(output_dir, exist_ok=True)
    selected = views or list(VIEWS.keys())
    results = {}

    if _which_or_warn("kicad-cli") is None or _which_or_warn("magick") is None:
        return results

    for name in selected:
        if name not in VIEWS:
            print(f"  Unknown view: {name}", file=sys.stderr)
            continue
        path = render_view(pcb_path, name, VIEWS[name], output_dir)
        if path:
            results[name] = path
            size_kb = os.path.getsize(path) / 1024
            print(f"  {name}: {path} ({size_kb:.0f} KB)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Render PCB layers to PNG")
    parser.add_argument("pcb", help="Path to .kicad_pcb file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: renders/ next to PCB)",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        choices=list(VIEWS.keys()),
        help="Specific views to render (default: all)",
    )
    parser.add_argument("--list", action="store_true", help="List available views")
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Rasterization DPI (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--max-px",
        type=int,
        default=DEFAULT_MAX_PX,
        help=f"Maximum output width/height in pixels after crop (default: {DEFAULT_MAX_PX})",
    )
    args = parser.parse_args()

    if args.list:
        for name, cfg in VIEWS.items():
            print(f"  {name:<20} {cfg['desc']}")
        return

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(args.pcb) or ".",
        "renders",
    )
    print(f"Rendering {args.pcb}:")
    results = render_all(args.pcb, out_dir, args.views)
    print(f"\n{len(results)} views rendered to {out_dir}")


if __name__ == "__main__":
    main()
