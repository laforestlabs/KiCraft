"""Canvas-only PNG render for the Manual Layout tab.

The manual layout canvas draws each leaf as an ``<image>`` element at an
explicit mm extent. For the image to land 1:1 with the post-route board
the canvas needs a PNG whose pixel aspect matches its declared mm box
exactly. The styled ``routed_front_all.png`` pipeline (cyan border, navy
padding, contrast boost, ``-trim``, alpha-key crop) drifts that aspect
because it crops to saturated content rather than Edge.Cuts, so the
canvas would non-uniformly stretch the result.

``render_leaf_canvas`` rasterizes ``leaf_routed.kicad_pcb`` straight from
``kicad-cli``'s SVG export with no trimming or chrome. The SVG's
``viewBox`` is the leaf-local mm extent of every pixel in the resulting
PNG, and we return that tuple so the canvas can place the image at the
correct mm coordinates. A sidecar JSON next to the PNG caches the extent
across page loads; ``mtime`` against the source ``.kicad_pcb`` invalidates
both the PNG and the sidecar.

Transparent background: leaves stack on the canvas and we want the
parent PCB fill (and overlapping leaf content) to show through where a
leaf has empty margin. ``magick`` is configured with ``-background none``
and we keep alpha through the whole pipeline.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# Bumping this forces every cached canvas PNG + sidecar to be regenerated
# on next page load. Use when the renderer's output changes shape (DPI,
# layers, background, viewBox semantics).
RENDERER_VERSION = 1

# 420 DPI matches render_pcb.py's existing leaf rasterization density.
# At 420 DPI, 1 mm = ~16.5 px, plenty for the canvas's typical zoom.
DEFAULT_DPI = 420

# Layer set mirrors render_pcb.VIEWS["front_all"] minus B.Cu (the canvas
# view is single-sided top-down). Edge.Cuts is included for leaves that
# ship a board outline; leaves without one still render via the bbox of
# F.SilkS + F.Cu.
_LAYERS = "F.Cu,F.SilkS,Edge.Cuts"

# kicad-cli emits a single root <svg ... viewBox="X Y W H" width="Wmm"
# height="Hmm"> with all coordinates in mm. The viewBox is the leaf-local
# mm extent of the rendered content; that's the box the canvas needs.
_VIEWBOX_RE = re.compile(
    r'viewBox="\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*"'
)
_WIDTH_MM_RE = re.compile(r'\bwidth="\s*([-\d.eE+]+)mm\s*"')
_HEIGHT_MM_RE = re.compile(r'\bheight="\s*([-\d.eE+]+)mm\s*"')


def _parse_svg_extent(svg_path: Path) -> tuple[float, float, float, float] | None:
    """Return (x_mm, y_mm, width_mm, height_mm) from kicad-cli's SVG.

    Prefers ``viewBox`` (which carries the leaf-local origin); falls back
    to root ``width``/``height`` attributes with origin ``(0, 0)`` if the
    viewBox is missing or malformed. The whole root element lives in the
    first few KB of the file so reading the head is enough.
    """
    try:
        head = svg_path.read_bytes()[:8192].decode("utf-8", errors="ignore")
    except OSError:
        return None
    m = _VIEWBOX_RE.search(head)
    if m:
        try:
            x, y, w, h = (float(g) for g in m.groups())
        except ValueError:
            x = y = w = h = 0.0
        if w > 0 and h > 0:
            return (x, y, w, h)
    wm = _WIDTH_MM_RE.search(head)
    hm = _HEIGHT_MM_RE.search(head)
    if wm and hm:
        try:
            return (0.0, 0.0, float(wm.group(1)), float(hm.group(1)))
        except ValueError:
            return None
    return None


def _sidecar_path(out_png: Path) -> Path:
    return out_png.with_suffix(out_png.suffix + ".extent.json")


def _read_sidecar(
    out_png: Path, leaf_pcb: Path
) -> tuple[float, float, float, float] | None:
    """Cache hit only when sidecar AND PNG are both newer than the source,
    and the sidecar declares the current ``RENDERER_VERSION``.
    """
    sidecar = _sidecar_path(out_png)
    try:
        src_mtime = leaf_pcb.stat().st_mtime
        if out_png.stat().st_mtime < src_mtime:
            return None
        if sidecar.stat().st_mtime < src_mtime:
            return None
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if int(data.get("renderer_version", 0)) != RENDERER_VERSION:
        return None
    try:
        return (
            float(data["x_mm"]),
            float(data["y_mm"]),
            float(data["width_mm"]),
            float(data["height_mm"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _write_sidecar(out_png: Path, extent: tuple[float, float, float, float]) -> None:
    x, y, w, h = extent
    payload = {
        "renderer_version": RENDERER_VERSION,
        "x_mm": x,
        "y_mm": y,
        "width_mm": w,
        "height_mm": h,
    }
    try:
        _sidecar_path(out_png).write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def render_leaf_canvas(
    leaf_pcb: Path,
    out_png: Path,
    *,
    dpi: int = DEFAULT_DPI,
) -> tuple[float, float, float, float] | None:
    """Render the manual-layout-canvas PNG for a single leaf.

    Returns ``(x_mm, y_mm, width_mm, height_mm)`` -- the leaf-local mm
    extent of the resulting image -- or ``None`` if any tool is missing
    or the export fails. The PNG sits at ``out_png`` and a sidecar JSON
    next to it caches the extent for subsequent calls.

    The cache is keyed on ``leaf_pcb``'s mtime plus ``RENDERER_VERSION``;
    a re-route updates the source file's mtime, which busts the cache on
    the next call.
    """
    if not leaf_pcb.is_file():
        return None

    cached = _read_sidecar(out_png, leaf_pcb)
    if cached is not None:
        return cached

    if shutil.which("kicad-cli") is None or shutil.which("magick") is None:
        return None

    out_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
        svg_path = Path(tmp.name)

    try:
        svg_cmd = [
            "kicad-cli",
            "pcb",
            "export",
            "svg",
            "--layers",
            _LAYERS,
            "--mode-single",
            "--fit-page-to-board",
            "--exclude-drawing-sheet",
            "--drill-shape-opt",
            "2",
            "-o",
            str(svg_path),
            str(leaf_pcb),
        ]
        try:
            subprocess.run(
                svg_cmd, check=True, capture_output=True, timeout=20
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None

        extent = _parse_svg_extent(svg_path)
        if extent is None:
            return None

        # -background none + PNG output keeps the alpha channel, so empty
        # leaf margin is transparent and stacked leaves don't paint over
        # each other with an opaque rectangle. No -trim / -border /
        # -resize: the PNG's pixel aspect must match the viewBox's mm
        # aspect by construction, which is the whole point of this path.
        png_cmd = [
            "magick",
            "-background",
            "none",
            "-density",
            str(dpi),
            str(svg_path),
            "PNG32:" + str(out_png),
        ]
        try:
            subprocess.run(
                png_cmd, check=True, capture_output=True, timeout=20
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
    finally:
        try:
            svg_path.unlink()
        except OSError:
            pass

    if not out_png.is_file():
        return None

    _write_sidecar(out_png, extent)
    return extent
