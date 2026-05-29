#!/usr/bin/env python3
"""Compare GUI's PCB render against KiCad's default-visibility render.

Given a ``.kicad_pcb``, this renders the same file two ways:

* ``gui.png`` -- via ``kicraft.render.render_views`` with the ``front_all``
  preset. This is exactly what the Experiment Manager shows in the parent /
  leaf preview cards.

* ``kicad.png`` -- via ``kicad-cli pcb export svg`` with the KiCad PCB
  editor's default-visible layer set (copper + silkscreen + mask + fab +
  courtyard + paste + edge cuts + user comment/drawing). This is the
  closest CLI proxy for "what you see when you double-click the file in
  the KiCad project manager."

Outputs side-by-side and difference-overlay PNGs plus a ``report.json``
carrying pixel-diff statistics and the extracted footprint placement
table. Non-zero exit when the fraction of differing pixels exceeds
``--max-frac`` (default disabled) -- the script is informational by
default and only fails CI when the caller asks it to.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Make the project package importable when the tool is run as a script
# from a working directory other than the repo root. Idempotent: a
# duplicate sys.path entry is harmless.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# SVG default namespace -- registered so ET.write() emits the SVG without
# the "ns0:" prefix it would otherwise stamp on every element.
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


# Layer set the KiCad PCB editor shows by default for a fresh board.
# F.Fab / B.Fab carry the footprint VALUE text (e.g. "ESP32-WROOM-32E-N4");
# F.CrtYd / B.CrtYd carry courtyard outlines; both are visible in the
# editor and so must appear in the ground-truth render too.
KICAD_DEFAULT_LAYERS = (
    "F.Cu,B.Cu,"
    "F.SilkS,B.SilkS,"
    "F.Mask,B.Mask,"
    "F.Fab,B.Fab,"
    "F.CrtYd,B.CrtYd,"
    "F.Paste,B.Paste,"
    "Edge.Cuts,Cmts.User,Dwgs.User"
)

# What the GUI's ``front_all`` preset asks kicad-cli for.
GUI_PRESET_LAYERS = "B.Cu,F.Cu,F.SilkS,Edge.Cuts"


def _label_font(size: int = 16) -> ImageFont.ImageFont:
    """Return a TrueType font for crosshair labels and side-by-side
    captions. PIL's default bitmap font is ~6px tall and unreadable at
    200 DPI; we try a few common TTFs and fall back to the bitmap font
    only when no TrueType is on the system."""
    for candidate in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "Arial.ttf",
        "Helvetica.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_via_kicad_cli(
    pcb: Path, out_png: Path, layers: str, *, dpi: int, viewbox: tuple[float, float, float, float] | None = None,
) -> None:
    """Render with kicad-cli using all requested layers. If ``viewbox`` is
    provided, rewrite the SVG viewBox to that AABB (matching the GUI's
    Edge.Cuts-AABB clipping) so renders can be compared apples-to-apples
    at the same framing; otherwise use --fit-page-to-board's natural
    page sizing (closer to what users see in the editor)."""
    with tempfile.TemporaryDirectory() as td:
        svg = Path(td) / "out.svg"
        cmd = [
            "kicad-cli", "pcb", "export", "svg",
            "--layers", layers,
            "--mode-single",
            "--exclude-drawing-sheet",
            "--drill-shape-opt", "2",
            "-o", str(svg),
        ]
        if viewbox is None:
            cmd.append("--fit-page-to-board")
        cmd.append(str(pcb))
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)

        if viewbox is not None:
            x0, y0, x1, y1 = viewbox
            w, h = x1 - x0, y1 - y0
            # Set attributes on the parsed SVG root rather than substituting
            # over a regex that assumed an attribute order kicad-cli is
            # free to change between releases.
            tree = ET.parse(svg)
            root = tree.getroot()
            root.set("width", f"{w:.4f}mm")
            root.set("height", f"{h:.4f}mm")
            root.set("viewBox", f"{x0:.4f} {y0:.4f} {w:.4f} {h:.4f}")
            tree.write(svg, encoding="utf-8", xml_declaration=True)

        subprocess.run(
            [
                "magick",
                "-background", "white",
                "-density", str(dpi),
                str(svg),
                "PNG32:" + str(out_png),
            ],
            check=True, capture_output=True, timeout=60,
        )


def annotate_footprints(
    png: Path,
    out_png: Path,
    footprints: list[dict],
    viewbox: tuple[float, float, float, float],
) -> None:
    """Overlay a small crosshair + ref text at each footprint's expected
    image position, computed from the .kicad_pcb (x,y) and the render's
    viewBox AABB. If the rendered geometry is faithful, every label lands
    on its component; misaligned crosshairs are bugs."""
    img = Image.open(png).convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = _label_font(16)
    x0, y0, x1, y1 = viewbox
    w_mm, h_mm = x1 - x0, y1 - y0
    iw, ih = img.size
    for fp in footprints:
        x, y = fp.get("x_mm"), fp.get("y_mm")
        if x is None or y is None:
            continue
        # Skip out-of-viewbox refs to avoid clutter
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        px = int(round((x - x0) / w_mm * iw))
        py = int(round((y - y0) / h_mm * ih))
        # Crosshair + ref text
        r = 8
        draw.line([(px - r, py), (px + r, py)], fill=(0, 255, 255, 255), width=2)
        draw.line([(px, py - r), (px, py + r)], fill=(0, 255, 255, 255), width=2)
        draw.text((px + r + 2, py - 8), fp["ref"], fill=(0, 255, 255, 255), font=font)
    img.save(out_png)


def render_via_gui_path(pcb: Path, out_png: Path) -> None:
    """Invoke the same render path the Experiment Manager uses for parent
    / leaf previews. Equivalent to what is written to
    ``round_NNNN/parent_stamped.png`` / ``parent_routed.png``."""
    from kicraft.render import render_views

    with tempfile.TemporaryDirectory() as td:
        results = render_views(
            pcb,
            Path(td),
            views=["front_all"],
        )
        src = results.get("front_all") if isinstance(results, dict) else None
        if src is None or not Path(src).is_file():
            raise RuntimeError(
                f"render_views did not produce a 'front_all' PNG for {pcb}"
            )
        shutil.copy(str(src), str(out_png))


def composite_side_by_side(
    a: Path, b: Path, out: Path, *, label_a: str, label_b: str
) -> None:
    img_a = Image.open(a).convert("RGBA")
    img_b = Image.open(b).convert("RGBA")
    h = max(img_a.height, img_b.height)

    def _fit_height(im: Image.Image, target: int) -> Image.Image:
        if im.height == target:
            return im
        w = max(1, int(round(im.width * (target / im.height))))
        return im.resize((w, target), Image.LANCZOS)

    img_a = _fit_height(img_a, h)
    img_b = _fit_height(img_b, h)
    pad = 30
    combined = Image.new(
        "RGBA",
        (img_a.width + img_b.width + 3 * pad, h + 2 * pad + 24),
        (32, 32, 32, 255),
    )
    combined.paste(img_a, (pad, pad + 24))
    combined.paste(img_b, (img_a.width + 2 * pad, pad + 24))
    draw = ImageDraw.Draw(combined)
    font = _label_font(18)
    draw.text((pad, 8), label_a, fill=(255, 255, 255, 255), font=font)
    draw.text((img_a.width + 2 * pad, 8), label_b, fill=(255, 255, 255, 255), font=font)
    combined.save(out)


def pixel_diff(
    a: Path, b: Path, out_overlay: Path, *, threshold: int
) -> dict:
    """Per-pixel diff with a threshold. Returns a stats dict and writes a
    diff overlay PNG: the GUI render desaturated under a red mask
    highlighting every pixel whose per-channel max diff exceeds the
    threshold."""
    img_a = Image.open(a).convert("RGB")
    img_b = Image.open(b).convert("RGB")
    if img_a.size != img_b.size:
        img_b_r = img_b.resize(img_a.size, Image.LANCZOS)
    else:
        img_b_r = img_b

    arr_a = np.asarray(img_a, dtype=np.int16)
    arr_b = np.asarray(img_b_r, dtype=np.int16)
    diff_per_channel = np.abs(arr_a - arr_b)
    diff = diff_per_channel.max(axis=2)
    differs = diff > threshold

    # Desaturated base + red mask
    gray = arr_a.mean(axis=2, keepdims=True).astype(np.uint8)
    base = np.repeat(gray, 3, axis=2)
    base = (base.astype(np.int16) // 2 + 64).clip(0, 255).astype(np.uint8)
    mask = differs[..., None]
    red = np.array([255, 0, 0], dtype=np.uint8)
    overlay = np.where(mask, red, base).astype(np.uint8)
    Image.fromarray(overlay).save(out_overlay)

    total = int(diff.size)
    n_diff = int(differs.sum())
    return {
        "image_size": [int(img_a.width), int(img_a.height)],
        "pixels_differing_above_threshold": n_diff,
        "total_pixels": total,
        "fraction_differing": round(n_diff / total, 4) if total else 0.0,
        "mean_pixel_diff": round(float(diff.mean()), 3),
        "max_pixel_diff": int(diff.max()),
        "threshold": threshold,
    }


def extract_footprints(pcb: Path) -> list[dict]:
    """Pull every footprint's ref, position, rotation, and side via
    ``pcbnew.LoadBoard`` -- the same parser KiCad uses to read the board
    on disk. Avoids the regex fragility of an inline s-expression walker
    (parens inside quoted property values, long property tables pushing
    the placement past a fixed byte window, locale-dependent file
    reads)."""
    import pcbnew

    board = pcbnew.LoadBoard(str(pcb))
    out: list[dict] = []
    for fp in board.GetFootprints():
        pos = fp.GetPosition()
        # GetLayerName returns the canonical KiCad layer string
        # ("F.Cu" / "B.Cu"); the .kicad_pcb's (layer "...") record uses
        # the same name, so this matches the previous regex output.
        layer = board.GetLayerName(fp.GetLayer())
        try:
            rot_deg = float(fp.GetOrientation().AsDegrees())
        except AttributeError:
            # Older pcbnew bindings: GetOrientationDegrees() returns float
            rot_deg = float(fp.GetOrientationDegrees())
        out.append({
            "ref": fp.GetReference(),
            "x_mm": pcbnew.ToMM(pos.x),
            "y_mm": pcbnew.ToMM(pos.y),
            "rot_deg": rot_deg,
            "layer": layer,
        })
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare GUI render vs KiCad default render of a .kicad_pcb"
    )
    parser.add_argument("pcb", type=Path, help=".kicad_pcb file to compare")
    parser.add_argument(
        "-o", "--out-dir",
        type=Path,
        default=Path("./render_diff_out"),
        help="Output directory (default: ./render_diff_out)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--threshold",
        type=int,
        default=24,
        help="Per-channel pixel diff threshold (0..255). Default 24.",
    )
    parser.add_argument(
        "--max-frac",
        type=float,
        default=None,
        help="If set, exit non-zero when fraction of differing pixels exceeds this.",
    )
    args = parser.parse_args(argv)

    if not args.pcb.is_file():
        print(f"error: not a file: {args.pcb}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gui_png = args.out_dir / "gui.png"
    kicad_png = args.out_dir / "kicad.png"
    sxs_png = args.out_dir / "side_by_side.png"
    diff_png = args.out_dir / "diff_overlay.png"
    report_path = args.out_dir / "report.json"

    print(f"rendering via GUI path  -> {gui_png}")
    render_via_gui_path(args.pcb, gui_png)

    # Render kicad-cli twice: once at its natural fit-to-board framing (what
    # the user sees in the editor), and once clipped to the SAME Edge.Cuts
    # viewBox as the GUI so we can compare component positions directly.
    # The two calls are independent; run them in parallel since each
    # spawns kicad-cli + magick and the SVG-export step dominates wall time.
    from kicraft.render.edge_cuts import parse_edge_cuts_aabb
    ec = parse_edge_cuts_aabb(args.pcb)
    print(f"Edge.Cuts AABB: {ec}")

    kicad_clipped_png = args.out_dir / "kicad_edgecuts_clip.png"
    print(f"rendering via kicad-cli (natural framing) -> {kicad_png}")
    if ec is not None:
        print(f"rendering via kicad-cli (Edge.Cuts viewBox) -> {kicad_clipped_png}")
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = [
            ex.submit(
                render_via_kicad_cli,
                args.pcb, kicad_png, KICAD_DEFAULT_LAYERS, dpi=args.dpi,
            ),
        ]
        if ec is not None:
            futures.append(ex.submit(
                render_via_kicad_cli,
                args.pcb, kicad_clipped_png, KICAD_DEFAULT_LAYERS,
                dpi=args.dpi, viewbox=ec,
            ))
        for f in futures:
            f.result()  # surface any exception from the worker

    composite_side_by_side(
        gui_png, kicad_png, sxs_png,
        label_a="GUI (front_all preset)",
        label_b="kicad-cli (default layers, natural framing)",
    )
    stats = pixel_diff(gui_png, kicad_png, diff_png, threshold=args.threshold)
    fps = extract_footprints(args.pcb)

    # Annotated overlays: crosshair every footprint at its file-level
    # position, projected through each render's viewBox. Misaligned
    # crosshairs reveal coordinate-handling bugs in that render path.
    if ec is not None:
        annotate_footprints(
            gui_png, args.out_dir / "gui_annotated.png", fps, ec,
        )
        if kicad_clipped_png.is_file():
            annotate_footprints(
                kicad_clipped_png,
                args.out_dir / "kicad_clipped_annotated.png",
                fps, ec,
            )
        composite_side_by_side(
            args.out_dir / "gui_annotated.png",
            args.out_dir / "kicad_clipped_annotated.png",
            args.out_dir / "annotated_side_by_side.png",
            label_a="GUI render + footprint crosshairs",
            label_b="kicad-cli (same viewBox) + footprint crosshairs",
        )

    report = {
        "pcb": str(args.pcb),
        "gui_png": str(gui_png),
        "kicad_png": str(kicad_png),
        "side_by_side": str(sxs_png),
        "diff_overlay": str(diff_png),
        "gui_layers": GUI_PRESET_LAYERS,
        "kicad_layers": KICAD_DEFAULT_LAYERS,
        "diff_stats": stats,
        "footprints": fps,
    }
    report_path.write_text(json.dumps(report, indent=2))

    print(f"report          -> {report_path}")
    print(f"side-by-side    -> {sxs_png}")
    print(f"diff overlay    -> {diff_png}")
    print(
        f"pixel diff: {stats['pixels_differing_above_threshold']}"
        f"/{stats['total_pixels']} = {stats['fraction_differing'] * 100:.1f}% "
        f"differing (threshold={args.threshold}, mean={stats['mean_pixel_diff']})"
    )

    if args.max_frac is not None and stats["fraction_differing"] > args.max_frac:
        print(
            f"FAIL: fraction_differing {stats['fraction_differing']:.4f} > "
            f"max_frac {args.max_frac:.4f}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
