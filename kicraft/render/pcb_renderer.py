"""Single PCB-to-PNG pipeline used by both the manual layout canvas
and the monitor / pipeline-graph views.

The core function ``render_pcb`` always produces a PNG whose content
extent equals the board's Edge.Cuts AABB by construction: the SVG
viewBox emitted by kicad-cli (which is expanded by ``--fit-page-to-board``
to include footprint reference-designator silk text that hangs past
Edge.Cuts) is rewritten to the Edge.Cuts AABB before magick rasterizes,
so anything outside the physical board is clipped. The PNG's pixel
aspect equals the Edge.Cuts mm aspect exactly.

The optional ``MonitorStyle`` adapter composites the raw transparent
PNG onto the PCB substrate color and applies a contrast/saturation
boost -- it is the only place the two consumers differ. The page
itself supplies any framing around the resulting tile.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kicraft.render.edge_cuts import parse_edge_cuts_aabb


DEFAULT_DPI = 420
DEFAULT_MAX_PX = 3200


@dataclass(frozen=True)
class EdgeCutsExtent:
    """Leaf-local mm rectangle that the rendered PNG's content area
    covers. Callers can use these to place the PNG on a parent canvas
    (the manual layout) or to derive a px-to-mm scale (anyone who
    needs to overlay geometry on top)."""

    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float

    @property
    def aabb(self) -> tuple[float, float, float, float]:
        return (
            self.x_mm,
            self.y_mm,
            self.x_mm + self.width_mm,
            self.y_mm + self.height_mm,
        )


@dataclass(frozen=True)
class MonitorStyle:
    """Styling for the monitor tab / pipeline graph previews. Composites
    the raw transparent-background board image onto a PCB substrate
    fill and applies a contrast/saturation boost. No padding or border
    -- the surrounding page already supplies framing. With ``style=None``,
    ``render_pcb`` skips this layer entirely and writes the raw
    transparent-background PNG the manual layout canvas consumes
    directly.

    ``board_background`` is composited onto the transparent pixels
    INSIDE Edge.Cuts (where there's no copper or silk) -- the PCB
    substrate color. Picked dark so copper and silk pop."""

    board_background: str = "#1e1e1e"
    contrast: float = 1.12
    saturation: float = 1.08
    brightness: float = 1.00
    max_px: int = DEFAULT_MAX_PX


# kicad-cli's SVG root element. Replacing the matched attribute string
# with one carrying the Edge.Cuts AABB clips the rasterized output to
# the physical board outline.
_SVG_ROOT_RE = re.compile(
    r'width="[^"]*"\s+height="[^"]*"\s+viewBox="[^"]*"'
)


def _rewrite_svg_viewbox(
    svg_path: Path, extent: tuple[float, float, float, float]
) -> bool:
    """Overwrite ``svg_path`` so the root element's viewBox and width/
    height attributes equal ``extent`` (mm). magick uses the viewBox to
    decide what's visible, so any content outside ``extent`` gets
    clipped during rasterization."""
    try:
        text = svg_path.read_text(encoding="utf-8")
    except OSError:
        return False
    x0, y0, x1, y1 = extent
    w, h = x1 - x0, y1 - y0
    replacement = (
        f'width="{w:.4f}mm" height="{h:.4f}mm" '
        f'viewBox="{x0:.4f} {y0:.4f} {w:.4f} {h:.4f}"'
    )
    new_text, n = _SVG_ROOT_RE.subn(replacement, text, count=1)
    if n != 1:
        return False
    try:
        svg_path.write_text(new_text, encoding="utf-8")
    except OSError:
        return False
    return True


def _svg_export(
    pcb_path: Path,
    svg_path: Path,
    layers: str,
    *,
    mirror: bool,
) -> bool:
    cmd = [
        "kicad-cli", "pcb", "export", "svg",
        "--layers", layers,
        "--mode-single",
        "--fit-page-to-board",
        "--exclude-drawing-sheet",
        "--drill-shape-opt", "2",
        "-o", str(svg_path),
    ]
    if mirror:
        cmd.append("--mirror")
    cmd.append(str(pcb_path))
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return True


def _rasterize_single(
    svg_path: Path,
    out_png: Path,
    *,
    dpi: int,
) -> bool:
    cmd = [
        "magick",
        "-background", "none",
        "-density", str(dpi),
        str(svg_path),
        "PNG32:" + str(out_png),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return out_png.is_file()


def _rasterize_composite(
    svg_front: Path,
    svg_back: Path,
    out_png: Path,
    *,
    dpi: int,
    back_opacity: float,
) -> bool:
    """F.Cu+B.Cu views render the two copper layers separately and
    composite with reduced opacity on the back layer so it doesn't
    obscure front detail. Same magick incantation as the legacy
    render_pcb.py pipeline -- the input SVGs are now Edge.Cuts-clipped
    so the resulting composite is clipped too."""
    cmd = [
        "magick",
        "-background", "none",
        "-density", str(dpi),
        "(", str(svg_back),
        "-channel", "A", "-evaluate", "multiply", str(back_opacity),
        "+channel", ")",
        "(", str(svg_front), ")",
        "-background", "none",
        "-layers", "merge",
        "PNG32:" + str(out_png),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return out_png.is_file()


def _brightness_contrast_arg(brightness: float, contrast: float) -> str:
    return f"{int(round((brightness - 1.0) * 100.0))}x{int(round((contrast - 1.0) * 100.0))}"


def _modulate_arg(brightness: float, saturation: float) -> str:
    return f"{int(round(brightness * 100.0))},{int(round(saturation * 100.0))},100"


def _apply_monitor_style(
    raw_png: Path, out_png: Path, style: MonitorStyle
) -> bool:
    """Convert the transparent-background Edge.Cuts PNG into the styled
    monitor preview: PCB-color substrate inside Edge.Cuts, contrast +
    saturation boost. No padding or border -- the surrounding page
    supplies framing.

    ``-trim`` is intentionally absent -- the input PNG is already
    clipped to Edge.Cuts.
    """
    cmd = [
        "magick", str(raw_png),
        # Fill transparent areas INSIDE Edge.Cuts with the PCB substrate
        # color. After this the image is opaque, exactly Edge.Cuts-sized,
        # and the board area reads as a panel against whatever the page
        # background is.
        "-background", style.board_background,
        "-alpha", "remove",
        "-alpha", "off",
        "-resize", f"{style.max_px}x{style.max_px}>",
        "-brightness-contrast", _brightness_contrast_arg(style.brightness, style.contrast),
        "-modulate", _modulate_arg(style.brightness, style.saturation),
        str(out_png),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return out_png.is_file()


def _has_both_copper(layers: str) -> bool:
    parts = {p.strip() for p in layers.split(",")}
    return "F.Cu" in parts and "B.Cu" in parts


def render_pcb(
    pcb_path: Path,
    out_png: Path,
    *,
    layers: str,
    mirror: bool = False,
    dpi: int = DEFAULT_DPI,
    style: MonitorStyle | None = None,
) -> EdgeCutsExtent | None:
    """Render ``pcb_path`` to ``out_png`` clipped to its Edge.Cuts AABB.

    The returned ``EdgeCutsExtent`` is the leaf-local mm rectangle the
    PNG's content covers. With ``style=None`` the PNG is transparent
    outside that rectangle (or, equivalently, the PNG IS that rectangle
    -- no margin); with ``style`` set, ``MonitorStyle`` fills the
    transparent pixels inside Edge.Cuts with the PCB substrate color
    and bumps contrast/saturation -- the pixel rectangle is still the
    Edge.Cuts AABB and the extent still describes it in mm.

    Returns ``None`` when ``kicad-cli`` or ``magick`` are missing, the
    PCB has no Edge.Cuts geometry, or any subprocess fails.
    """
    if not pcb_path.is_file():
        return None
    if shutil.which("kicad-cli") is None or shutil.which("magick") is None:
        return None

    ec = parse_edge_cuts_aabb(pcb_path)
    if ec is None:
        return None
    x0, y0, x1, y1 = ec
    extent = EdgeCutsExtent(
        x_mm=x0, y_mm=y0, width_mm=x1 - x0, height_mm=y1 - y0,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)

    composite = _has_both_copper(layers)
    tmp_svgs: list[Path] = []
    raw_png_path: Path | None = None
    try:
        if composite:
            front_layers_list = [
                ly for ly in (s.strip() for s in layers.split(","))
                if ly and ly != "B.Cu"
            ]
            front_layers = ",".join(front_layers_list)
            back_layers = "B.Cu,Edge.Cuts"
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
                svg_front = Path(f.name)
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
                svg_back = Path(f.name)
            tmp_svgs.extend([svg_front, svg_back])
            if not _svg_export(pcb_path, svg_front, front_layers, mirror=mirror):
                return None
            if not _svg_export(pcb_path, svg_back, back_layers, mirror=mirror):
                return None
            if not _rewrite_svg_viewbox(svg_front, ec):
                return None
            if not _rewrite_svg_viewbox(svg_back, ec):
                return None
            if style is None:
                if not _rasterize_composite(
                    svg_front, svg_back, out_png, dpi=dpi, back_opacity=0.52,
                ):
                    return None
            else:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    raw_png_path = Path(f.name)
                if not _rasterize_composite(
                    svg_front, svg_back, raw_png_path, dpi=dpi, back_opacity=0.52,
                ):
                    return None
                if not _apply_monitor_style(raw_png_path, out_png, style):
                    return None
        else:
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
                svg = Path(f.name)
            tmp_svgs.append(svg)
            if not _svg_export(pcb_path, svg, layers, mirror=mirror):
                return None
            if not _rewrite_svg_viewbox(svg, ec):
                return None
            if style is None:
                if not _rasterize_single(svg, out_png, dpi=dpi):
                    return None
            else:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    raw_png_path = Path(f.name)
                if not _rasterize_single(svg, raw_png_path, dpi=dpi):
                    return None
                if not _apply_monitor_style(raw_png_path, out_png, style):
                    return None
    finally:
        for p in tmp_svgs:
            try:
                p.unlink()
            except OSError:
                pass
        if raw_png_path is not None:
            try:
                raw_png_path.unlink()
            except OSError:
                pass

    return extent
