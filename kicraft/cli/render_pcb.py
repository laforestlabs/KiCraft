#!/usr/bin/env python3
"""Render PCB layers to PNG images for visual analysis.

This version improves preview quality for the experiment manager by:
- tightly cropping to the actual rendered board content
- adding a contrasting dark surround so the board edge is visible
- drawing a visible border around the rendered board image
- boosting contrast/saturation slightly for readability
- making silkscreen-inclusive views easier to inspect

The script still uses `kicad-cli` for SVG export and ImageMagick (`magick`)
for rasterization/post-processing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

# Layer sets for different views
VIEWS = {
    "front_all": {
        "layers": "F.Cu,F.SilkS,F.Mask,Edge.Cuts",
        "desc": "Front copper + silkscreen + mask + outline",
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
DEFAULT_BACKGROUND = "#020617"
DEFAULT_BORDER = "#67e8f9"
DEFAULT_BORDER_WIDTH = 6
DEFAULT_PADDING = 52


def _which_or_warn(name: str) -> str | None:
    path = shutil.which(name)
    if path is None:
        print(f"error: required executable not found on PATH: {name}", file=sys.stderr)
    return path


def _run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _safe_remove(path: str | os.PathLike[str]) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _svg_export(
    pcb_path: str,
    svg_path: str,
    layers: str,
    *,
    mirror: bool = False,
) -> bool:
    cmd = [
        "kicad-cli",
        "pcb",
        "export",
        "svg",
        "--layers",
        layers,
        "--mode-single",
        "--fit-page-to-board",
        "--exclude-drawing-sheet",
        "--drill-shape-opt",
        "2",
        "-o",
        svg_path,
    ]
    if mirror:
        cmd.append("--mirror")
    cmd.append(pcb_path)

    result = _run(cmd)
    if result.returncode != 0:
        print(
            f"  SVG export failed for layers '{layers}': {result.stderr.strip()}",
            file=sys.stderr,
        )
        return False
    return True


def _postprocess_png(
    input_png: str,
    output_png: str,
    *,
    max_px: int,
    background: str = DEFAULT_BACKGROUND,
    border_color: str = DEFAULT_BORDER,
    border_width: int = DEFAULT_BORDER_WIDTH,
    padding: int = DEFAULT_PADDING,
    contrast: float = 1.12,
    saturation: float = 1.08,
    brightness: float = 1.00,
) -> bool:
    """Crop tightly, improve contrast, and add a visible surround/border."""
    cmd = [
        "magick",
        input_png,
        "-alpha",
        "remove",
        "-alpha",
        "off",
        "-fuzz",
        "2%",
        "-trim",
        "+repage",
        "-bordercolor",
        background,
        "-border",
        str(padding),
        "-resize",
        f"{max_px}x{max_px}>",
        "-brightness-contrast",
        _brightness_contrast_arg(brightness, contrast),
        "-modulate",
        _modulate_arg(brightness, saturation),
        "-bordercolor",
        border_color,
        "-border",
        str(border_width),
        output_png,
    ]
    result = _run(cmd)
    if result.returncode != 0:
        print(
            f"  PNG post-process failed: {result.stderr.strip()}",
            file=sys.stderr,
        )
        return False
    return True


def _brightness_contrast_arg(brightness: float, contrast: float) -> str:
    # ImageMagick expects percentages. Keep brightness subtle and use contrast
    # as the main readability control.
    brightness_pct = int(round((brightness - 1.0) * 100.0))
    contrast_pct = int(round((contrast - 1.0) * 100.0))
    return f"{brightness_pct}x{contrast_pct}"


def _modulate_arg(brightness: float, saturation: float) -> str:
    brightness_pct = int(round(brightness * 100.0))
    saturation_pct = int(round(saturation * 100.0))
    return f"{brightness_pct},{saturation_pct},100"


def _svg_to_png(
    svg_path: str,
    png_path: str,
    *,
    dpi: int,
    max_px: int,
    post_cfg: dict | None = None,
) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        raw_png = tmp.name

    try:
        result = _run(
            [
                "magick",
                "-background",
                "#e5e7eb",
                "-density",
                str(dpi),
                svg_path,
                raw_png,
            ]
        )
        if result.returncode != 0:
            print(
                f"  PNG conversion failed: {result.stderr.strip()}",
                file=sys.stderr,
            )
            return False

        cfg = dict(post_cfg or {})
        return _postprocess_png(
            raw_png,
            png_path,
            max_px=max_px,
            background=cfg.get("background", DEFAULT_BACKGROUND),
            border_color=cfg.get("border_color", DEFAULT_BORDER),
            border_width=int(cfg.get("border_width", DEFAULT_BORDER_WIDTH)),
            padding=int(cfg.get("padding", DEFAULT_PADDING)),
            contrast=float(cfg.get("contrast", 1.12)),
            saturation=float(cfg.get("saturation", 1.08)),
            brightness=float(cfg.get("brightness", 1.00)),
        )
    finally:
        _safe_remove(raw_png)


def render_view(
    pcb_path, view_name, view_cfg, output_dir, dpi=DEFAULT_DPI, max_px=DEFAULT_MAX_PX
):
    """Render a single view to PNG. Returns output path or None on failure."""
    png_path = os.path.join(output_dir, f"{view_name}.png")
    layers = view_cfg["layers"]

    # For views with both copper layers, render them separately and composite
    # B.Cu at reduced opacity so it doesn't obscure F.Cu detail.
    if "F.Cu" in layers and "B.Cu" in layers:
        front_layers = (
            layers.replace("B.Cu,", "").replace(",B.Cu", "").replace("B.Cu", "")
        )
        front_layers = ",".join(
            layer_name for layer_name in front_layers.split(",") if layer_name
        )
        back_layers = "B.Cu,Edge.Cuts"
        return _render_composite(
            pcb_path,
            front_layers,
            back_layers,
            png_path,
            view_cfg,
            dpi,
            max_px,
        )

    svg_path = os.path.join(output_dir, f"{view_name}.svg")
    ok = _svg_export(
        pcb_path,
        svg_path,
        layers,
        mirror=bool(view_cfg.get("mirror")),
    )
    if not ok:
        return None

    try:
        ok = _svg_to_png(
            svg_path,
            png_path,
            dpi=dpi,
            max_px=max_px,
            post_cfg=view_cfg.get("post"),
        )
        if not ok:
            return None
        return png_path
    finally:
        _safe_remove(svg_path)


def _render_composite(
    pcb_path,
    front_layers,
    back_layers,
    png_path,
    view_cfg,
    dpi,
    max_px,
    back_opacity=0.52,
):
    """Render front and back layers separately, composite with alpha."""
    svg_front = None
    svg_back = None
    raw_png = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_front = f.name
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            svg_back = f.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            raw_png = f.name

        mirror = bool(view_cfg.get("mirror"))

        if not _svg_export(pcb_path, svg_front, front_layers, mirror=mirror):
            return None
        if not _svg_export(pcb_path, svg_back, back_layers, mirror=mirror):
            return None

        result = _run(
            [
                "magick",
                "-background",
                "none",
                "-density",
                str(dpi),
                "(",
                svg_back,
                "-channel",
                "A",
                "-evaluate",
                "multiply",
                str(back_opacity),
                "+channel",
                ")",
                "(",
                svg_front,
                ")",
                "-background",
                "#e5e7eb",
                "-layers",
                "merge",
                raw_png,
            ]
        )
        if result.returncode != 0:
            print(f"  Composite failed: {result.stderr.strip()}", file=sys.stderr)
            return None

        ok = _postprocess_png(
            raw_png,
            png_path,
            max_px=max_px,
            background=view_cfg.get("post", {}).get("background", DEFAULT_BACKGROUND),
            border_color=view_cfg.get("post", {}).get("border_color", DEFAULT_BORDER),
            border_width=int(
                view_cfg.get("post", {}).get("border_width", DEFAULT_BORDER_WIDTH)
            ),
            padding=int(view_cfg.get("post", {}).get("padding", DEFAULT_PADDING)),
            contrast=float(view_cfg.get("post", {}).get("contrast", 1.12)),
            saturation=float(view_cfg.get("post", {}).get("saturation", 1.05)),
            brightness=float(view_cfg.get("post", {}).get("brightness", 1.00)),
        )
        return png_path if ok else None
    except FileNotFoundError as e:
        print(f"  Composite render failed: {e}", file=sys.stderr)
        return None
    finally:
        if svg_front:
            _safe_remove(svg_front)
        if svg_back:
            _safe_remove(svg_back)
        if raw_png:
            _safe_remove(raw_png)


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
