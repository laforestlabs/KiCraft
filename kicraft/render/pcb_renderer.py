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

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kicraft.render.edge_cuts import parse_edge_cuts_aabb
from kicraft.render.views import VIEWS


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

    ``board_background`` is composited UNDER the transparent pixels
    INSIDE Edge.Cuts (where there's no copper or silk) -- the PCB
    substrate color. Accepts any magick color spec including RGBA
    (e.g. ``rgba(255,182,193,0.5)``) so the output PNG can carry partial
    alpha in the substrate area and the page background bleeds through;
    this makes the leaf edge visually unambiguous against the GUI's
    dark surface."""

    board_background: str = "rgba(255,182,193,0.5)"
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
    """Export ``pcb_path`` to SVG. We DELIBERATELY OMIT
    ``--fit-page-to-board`` -- that flag asks kicad-cli to inflate the
    page just enough to contain the board plus a margin, AND to
    translate the path coordinates into that inflated page (asymmetric
    in x and y; not a simple centering). When the renderer then
    rewrites the viewBox to the Edge.Cuts AABB (0..w, 0..h), it lands
    on the top-left corner of the inflated page rather than on the
    actual board content -- visible as a half-cropped leaf for tall
    narrow boards.

    Without the flag, kicad-cli emits the SVG with the full page size
    in the viewBox but path coordinates equal the raw KiCad mm of the
    board. The subsequent viewBox rewrite to the Edge.Cuts AABB then
    aligns by construction with the actual content.
    """
    cmd = [
        "kicad-cli", "pcb", "export", "svg",
        "--layers", layers,
        "--mode-single",
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


_RGBA_RE = re.compile(
    r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d.]+))?\s*\)"
)


def _parse_substrate_color(spec: str) -> tuple[int, int, int, int]:
    """Parse a magick-style color spec into an RGBA tuple. Accepts
    ``rgba(r,g,b,a)`` with a 0..1 alpha float, ``rgb(r,g,b)``, or any
    other spec PIL can resolve. Falls back to opaque pink on parse
    failure so the substrate fill never silently disappears."""
    m = _RGBA_RE.match(spec.strip())
    if m:
        r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        a_token = m.group(4)
        a = int(round(float(a_token) * 255)) if a_token is not None else 255
        return (r, g, b, max(0, min(255, a)))
    try:
        from PIL import ImageColor
        rgba = ImageColor.getrgb(spec)
        return rgba if len(rgba) == 4 else (*rgba, 255)
    except Exception:
        return (255, 182, 193, 128)


def _build_substrate_masked_png(raw_png: Path, out_png: Path, color: str) -> bool:
    """Paint the substrate fill UNDER ``raw_png``'s transparent pixels,
    masked to the actual Edge.Cuts polygon.

    The naive "paint every pixel of the clone with substrate, then put
    raw on top" approach fills the full Edge.Cuts AABB rectangle. For
    leaves with a ROUNDED Edge.Cuts polygon, the AABB corner regions
    fall outside the polygon -- they should stay transparent so the
    page background bleeds through, but they were getting filled with
    substrate, producing a visible pink halo past the silk frame.

    Fix: flood-fill the alpha channel from a transparent corner pixel
    with a sentinel value, marking the OUTSIDE-of-polygon transparent
    region. The Edge.Cuts stroke (opaque) blocks the flood; INSIDE-of-
    polygon transparent pixels remain at alpha=0. Substrate is then
    painted only where alpha is still 0. Sharp-rectangle Edge.Cuts
    (no corner gap to start the flood) falls through to fill the whole
    AABB, matching the legacy behavior for parent boards.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False
    try:
        raw = Image.open(raw_png).convert("RGBA")
    except OSError:
        return False
    w, h = raw.size

    alpha = raw.getchannel("A")
    outside_seed: tuple[int, int] | None = None
    for cx, cy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if alpha.getpixel((cx, cy)) == 0:
            outside_seed = (cx, cy)
            break
    if outside_seed is not None:
        # Sentinel value 1 distinguishes outside-of-polygon pixels from
        # inside-of-polygon transparent pixels (both started at 0).
        ImageDraw.floodfill(alpha, outside_seed, 1)
    mask = alpha.point(lambda v: 255 if v == 0 else 0)

    substrate = Image.new("RGBA", (w, h), _parse_substrate_color(color))
    base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    base.paste(substrate, (0, 0), mask=mask)
    result = Image.alpha_composite(base, raw)
    try:
        result.save(out_png, "PNG")
    except OSError:
        return False
    return out_png.is_file()


def _apply_monitor_style(
    raw_png: Path, out_png: Path, style: MonitorStyle
) -> bool:
    """Convert the transparent-background Edge.Cuts PNG into the styled
    monitor preview: PCB-color substrate under the actual Edge.Cuts
    polygon (not just the AABB), plus a contrast/saturation boost. No
    padding or border -- the surrounding page supplies framing.

    The substrate fill is built in PIL because flood-filling the alpha
    channel from a corner is what keeps the rounded-Edge.Cuts corner
    regions transparent; magick then handles resize + brightness /
    contrast / saturation post.

    Output keeps the alpha channel so an RGBA ``board_background``
    (e.g. translucent pink) lets the page show through the substrate;
    copper/silk pixels remain fully opaque because raw composites on
    top using its own alpha.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        masked_png = Path(f.name)
    try:
        if not _build_substrate_masked_png(raw_png, masked_png, style.board_background):
            return False
        cmd = [
            "magick", str(masked_png),
            "-resize", f"{style.max_px}x{style.max_px}>",
            "-brightness-contrast",
            _brightness_contrast_arg(style.brightness, style.contrast),
            "-modulate", _modulate_arg(style.brightness, style.saturation),
            "PNG32:" + str(out_png),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return False
        return out_png.is_file()
    finally:
        try:
            masked_png.unlink()
        except OSError:
            pass


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

    # All magick writes target ``stage_png`` (a sibling of out_png so
    # os.replace stays on the same filesystem). After every step
    # succeeds, we ``os.replace(stage_png, out_png)`` to swap the file
    # in atomically. The atomic swap creates a NEW inode for out_png;
    # any hardlinks an earlier round made to the previous out_png keep
    # pointing at the old, now-frozen inode. Without this, magick's
    # truncate-and-write on out_png would clobber the bytes seen
    # through any hardlinked round_NNNN snapshot of a prior render.
    with tempfile.NamedTemporaryFile(
        suffix=".png", delete=False, dir=str(out_png.parent),
        prefix=f".{out_png.stem}.",
    ) as f:
        stage_png = Path(f.name)

    composite = _has_both_copper(layers)
    tmp_svgs: list[Path] = []
    raw_png_path: Path | None = None
    success = False
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
                    svg_front, svg_back, stage_png, dpi=dpi, back_opacity=0.52,
                ):
                    return None
            else:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    raw_png_path = Path(f.name)
                if not _rasterize_composite(
                    svg_front, svg_back, raw_png_path, dpi=dpi, back_opacity=0.52,
                ):
                    return None
                if not _apply_monitor_style(raw_png_path, stage_png, style):
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
                if not _rasterize_single(svg, stage_png, dpi=dpi):
                    return None
            else:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    raw_png_path = Path(f.name)
                if not _rasterize_single(svg, raw_png_path, dpi=dpi):
                    return None
                if not _apply_monitor_style(raw_png_path, stage_png, style):
                    return None
        os.replace(stage_png, out_png)
        success = True
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
        if not success:
            try:
                stage_png.unlink()
            except OSError:
                pass

    return extent


def render_views(
    pcb_path: Path,
    output_dir: Path,
    *,
    views: list[str] | None = None,
    name_template: str = "{view}.png",
    dpi: int = DEFAULT_DPI,
    max_px: int = DEFAULT_MAX_PX,
    template_fields: dict[str, str | int] | None = None,
) -> dict[str, Path]:
    """Render one or more named views to PNGs in ``output_dir``.

    ``name_template`` is a Python str.format template controlling the
    output filename per view. ``{view}`` is the view name; any extra
    fields the caller wants (e.g. ``{round:04d}``, ``{stage}``) are
    supplied via ``template_fields``. Default ``"{view}.png"`` gives
    the historical ``front_all.png`` / ``back_all.png`` layout.

    Returns ``{view_name: output_path}`` for the views that rendered
    successfully. Views unknown to the registry are silently skipped --
    the caller's responsibility to validate names against ``VIEWS``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = views or list(VIEWS.keys())
    fields = dict(template_fields or {})
    results: dict[str, Path] = {}

    for view_name in selected:
        cfg = VIEWS.get(view_name)
        if cfg is None:
            continue
        out_path = output_dir / name_template.format(view=view_name, **fields)
        post = dict(cfg.get("post") or {})
        style = MonitorStyle(
            contrast=float(post.get("contrast", 1.12)),
            saturation=float(post.get("saturation", 1.08)),
            brightness=float(post.get("brightness", 1.00)),
            max_px=max_px,
        )
        extent = render_pcb(
            pcb_path,
            out_path,
            layers=cfg["layers"],
            mirror=bool(cfg.get("mirror")),
            dpi=dpi,
            style=style,
        )
        if extent is not None:
            results[view_name] = out_path
    return results
