#!/usr/bin/env python3
"""Render PCB layers to PNG images for visual analysis.

Thin argparse wrapper on top of ``kicraft.render.render_views``. The
preset registry lives in ``kicraft.render.views.VIEWS`` and is shared
with every other consumer (the GUI monitor, the score-time visual
check, the subcircuit diagnostics bundle, the parent compose
stamper) so the previews can never drift between code paths.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from kicraft.render import VIEWS, render_views
from kicraft.render.pcb_renderer import (
    DEFAULT_DPI, DEFAULT_MAX_PX, _rasterizer_available,
)


def _which_or_warn(name: str) -> str | None:
    path = shutil.which(name)
    if path is None:
        print(f"error: required executable not found on PATH: {name}", file=sys.stderr)
    return path


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
    if _which_or_warn("kicad-cli") is None:
        sys.exit(1)
    if not _rasterizer_available():
        print(
            "error: no rasterizer available on PATH (need ImageMagick 6/7 "
            "`magick`/`convert`, or the cairosvg package)",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Rendering {args.pcb}:")
    results = render_views(
        Path(args.pcb),
        Path(out_dir),
        views=args.views,
        dpi=args.dpi,
        max_px=args.max_px,
    )
    for name, path in results.items():
        size_kb = path.stat().st_size / 1024
        print(f"  {name}: {path} ({size_kb:.0f} KB)")
    print(f"\n{len(results)} views rendered to {out_dir}")


if __name__ == "__main__":
    main()
