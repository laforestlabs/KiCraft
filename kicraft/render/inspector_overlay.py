"""PIL-based diagnostic overlays for the inspect_parent tool.

These two renderers (``annotated_top.png``, ``stacking_heatmap.png``)
draw from analysis data (Bbox + footprint courtyards + DRC findings)
rather than rasterizing the PCB through kicad-cli + magick. They live
separately from ``kicraft.render.pcb_renderer`` so an agent searching
"where do PCB renders come from?" finds the kicad-cli pipeline and is
not distracted by these inspector diagrams; conversely an agent
working on the inspector report has one module to read.

The functions take a ``Report`` from ``kicraft.cli.inspect_parent``;
the type is imported lazily inside the functions so this module does
not pull pcbnew into module-load (it does not need it).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from kicraft.cli.inspect_parent import Bbox, Report


def _world_to_image(
    point: tuple[float, float],
    outline: "Bbox",
    img_size: tuple[int, int],
    padding_px: int,
) -> tuple[int, int]:
    """World mm -> image pixel, with ``padding_px`` around the board."""
    width_px, height_px = img_size
    inner_w = max(1, width_px - 2 * padding_px)
    inner_h = max(1, height_px - 2 * padding_px)
    sx = inner_w / max(0.001, outline.width)
    sy = inner_h / max(0.001, outline.height)
    s = min(sx, sy)
    px = padding_px + (point[0] - outline.min_x) * s
    py = padding_px + (point[1] - outline.min_y) * s
    return int(px), int(py)


def _world_scale(outline: "Bbox", img_size: tuple[int, int], padding_px: int) -> float:
    width_px, height_px = img_size
    inner_w = max(1, width_px - 2 * padding_px)
    inner_h = max(1, height_px - 2 * padding_px)
    sx = inner_w / max(0.001, outline.width)
    sy = inner_h / max(0.001, outline.height)
    return min(sx, sy)


def _load_font(size: int):
    for path in (
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_annotated_top(report: "Report", output: Path) -> Path:
    """Render an annotated top-view PNG.

    Drawn from scratch (does not require pcbnew render). Shows:
    - board outline (cyan)
    - front-side courtyards (green)
    - back-side courtyards (red, dashed via cross-hatch)
    - edge-marker arrows pointing from marker to expected board edge
    - constraint anchor labels
    - footprint refs

    The colors and labels are chosen for AI-agent legibility, not
    pretty-print. An agent reading this PNG should be able to see
    instantly: "is the connector marker at the board edge?", "how
    much front-side area is unused?".
    """
    bo = report.board_outline
    pad = 60
    aspect = bo.height / max(0.01, bo.width)
    width = 1200
    height = int(width * aspect) + 2 * pad
    width += 2 * pad
    img = Image.new("RGB", (width, height), "#0b1220")
    draw = ImageDraw.Draw(img, "RGBA")
    font_label = _load_font(16)
    font_small = _load_font(11)
    font_title = _load_font(20)
    scale = _world_scale(bo, (width, height), pad)

    # Title.
    draw.text(
        (pad, 10),
        f"{Path(report.pcb_path).name}  board {bo.width:.1f} x {bo.height:.1f} mm",
        fill="#e5e7eb",
        font=font_title,
    )

    # Board outline.
    tl = _world_to_image((bo.min_x, bo.min_y), bo, (width, height), pad)
    br = _world_to_image((bo.max_x, bo.max_y), bo, (width, height), pad)
    draw.rectangle([tl, br], outline="#22d3ee", width=3)

    # Back courtyards first (so front overlays them).
    for fp in report.footprints:
        if fp.layer != "back":
            continue
        c = fp.courtyard
        a = _world_to_image((c.min_x, c.min_y), bo, (width, height), pad)
        b = _world_to_image((c.max_x, c.max_y), bo, (width, height), pad)
        draw.rectangle([a, b], outline="#f87171", width=2, fill=(248, 113, 113, 60))
        # Label at top-left of bbox.
        draw.text((a[0] + 3, a[1] + 3), fp.ref, fill="#fecaca", font=font_small)

    # Front courtyards on top.
    for fp in report.footprints:
        if fp.layer != "front":
            continue
        c = fp.courtyard
        a = _world_to_image((c.min_x, c.min_y), bo, (width, height), pad)
        b = _world_to_image((c.max_x, c.max_y), bo, (width, height), pad)
        draw.rectangle([a, b], outline="#34d399", width=1)
        # Show ref only for largish footprints to avoid clutter.
        if c.area > 5:
            draw.text((a[0] + 3, a[1] + 3), fp.ref, fill="#a7f3d0", font=font_small)

    # Edge findings: draw arrow from marker to nearest board edge.
    for f in report.edge_findings:
        marker_world = f.marker_world
        if f.edge == "left":
            target_world = (bo.min_x, marker_world[1])
        elif f.edge == "right":
            target_world = (bo.max_x, marker_world[1])
        elif f.edge == "top":
            target_world = (marker_world[0], bo.min_y)
        else:
            target_world = (marker_world[0], bo.max_y)
        m = _world_to_image(marker_world, bo, (width, height), pad)
        t = _world_to_image(target_world, bo, (width, height), pad)
        color = (
            "#fb923c" if f.interpretation.startswith("BUG")
            else "#facc15" if f.interpretation.startswith("WARN")
            else "#22d3ee"
        )
        draw.line([m, t], fill=color, width=3)
        draw.ellipse([m[0] - 4, m[1] - 4, m[0] + 4, m[1] + 4], fill=color)
        label = f"{f.ref} {f.marker_distance_from_edge_mm:+.2f} mm"
        # Place label slightly offset toward the board interior.
        ox, oy = (8, -6)
        if f.edge == "right":
            ox = -8 - len(label) * 7
        if f.edge == "bottom":
            oy = 6
        draw.text((m[0] + ox, m[1] + oy), label, fill=color, font=font_label)

    # DRC violations -- only show errors prominently; cluster nearby
    # violations of the same (type, ref) so an AI agent doesn't get a
    # cloud of overlapping markers on a single cluster of pins.
    drc_pin_clusters: dict[tuple[str, str | None], tuple[float, float, int]] = {}
    for v in report.drc.violations:
        if v.pos is None:
            continue
        ref = v.refs[0] if v.refs else None
        key = (v.type, ref)
        if key in drc_pin_clusters:
            cx, cy, count = drc_pin_clusters[key]
            drc_pin_clusters[key] = (
                (cx * count + v.pos[0]) / (count + 1),
                (cy * count + v.pos[1]) / (count + 1),
                count + 1,
            )
        else:
            drc_pin_clusters[key] = (v.pos[0], v.pos[1], 1)

    # Draw errors with bigger emphasis than warnings.
    sev_by_key = {
        (v.type, v.refs[0] if v.refs else None): v.severity
        for v in report.drc.violations
    }
    for (vtype, ref), (cx, cy, count) in drc_pin_clusters.items():
        sev = sev_by_key.get((vtype, ref), "warning")
        color = "#f87171" if sev == "error" else "#facc15"
        radius = 9 if sev == "error" else 6
        p = _world_to_image((cx, cy), bo, (width, height), pad)
        draw.ellipse(
            [p[0] - radius, p[1] - radius, p[0] + radius, p[1] + radius],
            outline=color,
            width=2 if sev == "error" else 1,
        )
        draw.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=color)
        label = f"{vtype}" + (f" x{count}" if count > 1 else "")
        draw.text(
            (p[0] + radius + 3, p[1] - 7),
            label,
            fill=color,
            font=font_small,
        )

    # Legend.
    legend_y = height - pad + 8
    draw.text((pad, legend_y), "front", fill="#34d399", font=font_label)
    draw.text((pad + 70, legend_y), "back", fill="#f87171", font=font_label)
    draw.text(
        (pad + 140, legend_y),
        "marker->edge (cyan=ok, yellow=warn, orange=bug)  DRC: red=err  yellow=warn",
        fill="#e5e7eb",
        font=font_label,
    )

    img.save(output)
    return output


def render_stacking_heatmap(report: "Report", output: Path) -> Path:
    """Render a 5 mm-grid heatmap of layer occupancy.

    Cells are color-coded:
      - empty  : black
      - front  : green
      - back   : red
      - stacked: yellow (front + back; the goal for dual-layer parents)
    """
    from kicraft.cli.inspect_parent import Bbox  # local import; pcbnew lazy

    bo = report.board_outline
    pad = 60
    aspect = bo.height / max(0.01, bo.width)
    width = 1200
    height = int(width * aspect) + 2 * pad
    width += 2 * pad
    img = Image.new("RGB", (width, height), "#0b1220")
    draw = ImageDraw.Draw(img, "RGBA")
    font_title = _load_font(20)
    font_legend = _load_font(14)

    front_courts = [fp.courtyard for fp in report.footprints if fp.layer == "front"]
    back_courts = [fp.courtyard for fp in report.footprints if fp.layer == "back"]
    nx = max(1, int(round(bo.width / report.grid_mm)))
    ny = max(1, int(round(bo.height / report.grid_mm)))
    for ix in range(nx):
        for iy in range(ny):
            cx = bo.min_x + (ix + 0.5) * report.grid_mm
            cy = bo.min_y + (iy + 0.5) * report.grid_mm
            cell = Bbox(
                cx - report.grid_mm / 2,
                cy - report.grid_mm / 2,
                cx + report.grid_mm / 2,
                cy + report.grid_mm / 2,
            )
            on_front = any(cell.overlaps(c) for c in front_courts)
            on_back = any(cell.overlaps(c) for c in back_courts)
            if on_front and on_back:
                color = (250, 204, 21, 200)  # stacked - yellow
            elif on_front:
                color = (52, 211, 153, 130)  # front - green
            elif on_back:
                color = (248, 113, 113, 130)  # back - red
            else:
                color = (24, 24, 27, 0)  # empty - leave board bg
            a = _world_to_image((cell.min_x, cell.min_y), bo, (width, height), pad)
            b = _world_to_image((cell.max_x, cell.max_y), bo, (width, height), pad)
            draw.rectangle([a, b], fill=color, outline=(255, 255, 255, 30))

    # Board outline.
    tl = _world_to_image((bo.min_x, bo.min_y), bo, (width, height), pad)
    br = _world_to_image((bo.max_x, bo.max_y), bo, (width, height), pad)
    draw.rectangle([tl, br], outline="#22d3ee", width=3)

    draw.text(
        (pad, 10),
        f"Stacking heatmap  stacked {report.stacked_fraction * 100:.1f}% / "
        f"wasted {report.wasted_fraction * 100:.1f}%",
        fill="#e5e7eb",
        font=font_title,
    )
    legend_y = height - pad + 8
    draw.rectangle([pad, legend_y, pad + 18, legend_y + 14], fill="#facc15")
    draw.text((pad + 24, legend_y - 1), "stacked", fill="#e5e7eb", font=font_legend)
    draw.rectangle([pad + 110, legend_y, pad + 128, legend_y + 14], fill="#34d399")
    draw.text((pad + 134, legend_y - 1), "front", fill="#e5e7eb", font=font_legend)
    draw.rectangle([pad + 200, legend_y, pad + 218, legend_y + 14], fill="#f87171")
    draw.text((pad + 224, legend_y - 1), "back", fill="#e5e7eb", font=font_legend)
    draw.rectangle([pad + 290, legend_y, pad + 308, legend_y + 14], fill="#18181b", outline="#fff")
    draw.text((pad + 314, legend_y - 1), "empty", fill="#e5e7eb", font=font_legend)

    img.save(output)
    return output
