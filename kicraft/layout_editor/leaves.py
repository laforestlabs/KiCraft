"""Leaf discovery for the manual layout editor.

Scans ``.experiments/subcircuits/`` for solved leaves, renders (or
cache-hits) their canvas PNGs, and packages everything the canvas
needs into ``LeafInfo`` records. The host app supplies how a rendered
PNG on disk maps to a browser URL via the ``url_for`` callable: the
offline GUI mounts ``.experiments`` at ``/experiments`` while the web
app serves renders through per-project HMAC token routes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from kicraft.layout_editor.render import render_leaf_canvas
from kicraft.render.edge_cuts import parse_edge_cuts_aabb

# Maps a rendered PNG (resolved on disk) to the URL the browser loads
# it from, or None when the file can't be exposed. Implementations
# should cache-bust (the default appends ``?v=<mtime>``) so a re-routed
# leaf's fresh render isn't shadowed by the browser cache.
LeafUrlFor = Callable[[Path], str | None]


_LEAF_COLORS = [
    "#60a5fa",  # blue
    "#34d399",  # emerald
    "#fbbf24",  # amber
    "#f87171",  # red
    "#c084fc",  # purple
    "#22d3ee",  # cyan
    "#a3e635",  # lime
    "#fb923c",  # orange
    "#f472b6",  # pink
    "#94a3b8",  # slate
]


@dataclass(slots=True)
class LeafInfo:
    """One discovered leaf available for manual placement.

    ``silk_min_*`` / ``silk_max_*`` is the axis-aligned bbox of the
    leaf solver's rounded-rect silk polygon (computed post-route from
    courtyards + pad copper + traces + vias). The bbox drives the
    leaf-image rect, the hit target, and the overflow check; the
    visible silkscreen outline is baked into the rendered PNG itself,
    so no separate polygon overlay is drawn on the canvas. Falls
    back to the leaf's Edge.Cuts dimensions when no silk poly is on
    disk (rare, only when the leaf solver had no ``group_labels``
    match).
    """

    instance_path: str
    sheet_name: str
    width_mm: float
    height_mm: float
    artifact_dir: Path
    render_url: str | None = None
    color: str = "#60a5fa"
    silk_min_x: float = 0.0
    silk_min_y: float = 0.0
    silk_max_x: float = 0.0
    silk_max_y: float = 0.0
    # Leaf-local mm extent of ``render_url``'s PNG. With the unified
    # renderer (v2 sidecar) this equals the Edge.Cuts AABB by
    # construction -- the renderer rewrites kicad-cli's SVG viewBox to
    # Edge.Cuts before rasterizing, so the PNG content lands exactly
    # inside the physical board outline. Kept as a distinct field
    # (rather than aliasing ``edge_*``) so the canvas code reads
    # self-documenting at each call site (PNG placement vs hit-test).
    image_x_mm: float = 0.0
    image_y_mm: float = 0.0
    image_width_mm: float = 0.0
    image_height_mm: float = 0.0
    # Leaf-local Edge.Cuts AABB -- the leaf's PHYSICAL extent, parsed
    # from the canonical leaf_routed.kicad_pcb's ``gr_line`` items on
    # the Edge.Cuts layer. This is the single source of truth for the
    # leaf's footprint on the parent board: hit testing, snapping,
    # overflow against the outline, AND inter-leaf overlap detection
    # all run against this rectangle. Falls back to (0, 0, width_mm,
    # height_mm) from metadata when no Edge.Cuts is present.
    edge_min_x: float = 0.0
    edge_min_y: float = 0.0
    edge_max_x: float = 0.0
    edge_max_y: float = 0.0


def experiments_mount_url_for(experiments_dir: Path) -> LeafUrlFor:
    """URL builder for hosts that mount ``experiments_dir`` at
    ``/experiments`` (the offline GUI). Appends an mtime cache-buster
    so the browser picks up a freshly-rendered PNG without a hard
    reload."""

    def url_for(png_path: Path) -> str | None:
        try:
            rel = png_path.relative_to(experiments_dir)
        except ValueError:
            return None
        return f"/experiments/{rel.as_posix()}?v={int(png_path.stat().st_mtime)}"

    return url_for


def _silk_bbox_from_solved_layout(
    leaf_dir: Path,
) -> tuple[float, float, float, float] | None:
    """Axis-aligned bbox of the leaf solver's silkscreen poly.

    The leaf solver writes a single closed rounded-rect poly per leaf
    via ``_build_leaf_silkscreen``. We only need its bbox -- the poly
    itself is rendered into the leaf PNG via F.Silkscreen and the
    canvas re-uses that PNG instead of drawing the polygon a second
    time. Returns None when no poly silk is on disk.
    """
    sl_path = leaf_dir / "solved_layout.json"
    try:
        sl = json.loads(sl_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for elem in sl.get("silkscreen", []) or []:
        if elem.get("kind") != "poly":
            continue
        xs: list[float] = []
        ys: list[float] = []
        for pt in elem.get("points", []) or []:
            try:
                xs.append(float(pt["x"]))
                ys.append(float(pt["y"]))
            except (KeyError, TypeError, ValueError):
                continue
        # The leaf solver writes one poly per leaf (the rounded outline).
        # Subsequent polys would be component-level body silk that we do
        # not want to use as the overall leaf bbox.
        if xs and ys:
            return (min(xs), min(ys), max(xs), max(ys))
        return None
    return None


def _canvas_render_for(
    leaf_dir: Path, url_for: LeafUrlFor
) -> tuple[str | None, tuple[float, float, float, float] | None]:
    """Return ``(render_url, (x_mm, y_mm, w_mm, h_mm))`` for the manual
    layout canvas.

    Renders ``leaf_routed.kicad_pcb`` to a transparent-background PNG with
    its leaf-local mm extent recorded in a sidecar JSON, so the canvas
    can place the image at the exact mm coordinates it represents (no
    aspect drift, no stale cache once the leaf is re-routed).
    """
    pcb = leaf_dir / "leaf_routed.kicad_pcb"
    if not pcb.is_file():
        return (None, None)
    out_png = leaf_dir / "renders" / "leaf_canvas.png"
    extent = render_leaf_canvas(pcb, out_png)
    if extent is None or not out_png.is_file():
        return (None, None)
    return (url_for(out_png), extent)


def discover_leaves(
    experiments_dir: Path, *, url_for: LeafUrlFor | None = None
) -> list[LeafInfo]:
    """Scan .experiments/subcircuits/ for solved leaves.

    Each subdir with a ``metadata.json`` and a ``leaf_routed.kicad_pcb``
    is considered a placeable leaf. The order is stable (sorted by
    sheet_name) so leaf colours stay consistent across renders.
    ``url_for`` maps each rendered PNG to a browser URL; defaults to
    the offline GUI's ``/experiments`` mount scheme.
    """
    if url_for is None:
        url_for = experiments_mount_url_for(experiments_dir)
    sub_root = experiments_dir / "subcircuits"
    if not sub_root.is_dir():
        return []

    leaves: list[LeafInfo] = []
    for leaf_dir in sorted(sub_root.iterdir()):
        if not leaf_dir.is_dir():
            continue
        meta_path = leaf_dir / "metadata.json"
        routed = leaf_dir / "leaf_routed.kicad_pcb"
        if not meta_path.exists() or not routed.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        outline = meta.get("local_board_outline") or {}
        try:
            w = float(outline.get("width_mm", 0.0))
            h = float(outline.get("height_mm", 0.0))
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        render_url, extent = _canvas_render_for(leaf_dir, url_for)
        if extent is None:
            image_x, image_y, image_w, image_h = 0.0, 0.0, w, h
        else:
            image_x, image_y, image_w, image_h = extent

        # silk_min/max = the leaf solver's silk-poly bbox. Cosmetic only.
        poly_bbox = _silk_bbox_from_solved_layout(leaf_dir)
        if poly_bbox is None:
            silk_min_x, silk_min_y, silk_max_x, silk_max_y = 0.0, 0.0, w, h
        else:
            silk_min_x, silk_min_y, silk_max_x, silk_max_y = poly_bbox

        # edge_min/max = the leaf's Edge.Cuts AABB from the canonical
        # leaf_routed.kicad_pcb. This is the single source of truth for
        # the leaf's physical extent: it's what gets stamped on the
        # parent board, what defines whether two leaves physically
        # collide, and what the canvas snaps / overflow-checks against.
        # Falls back to the metadata's local_board_outline (which is
        # what the leaf solver was told to fit into) when Edge.Cuts
        # is missing from the PCB file -- conservative default.
        edge_bbox = parse_edge_cuts_aabb(leaf_dir / "leaf_routed.kicad_pcb")
        if edge_bbox is None:
            edge_min_x, edge_min_y, edge_max_x, edge_max_y = 0.0, 0.0, w, h
        else:
            edge_min_x, edge_min_y, edge_max_x, edge_max_y = edge_bbox

        leaves.append(
            LeafInfo(
                instance_path=str(meta.get("instance_path", "")),
                sheet_name=str(meta.get("sheet_name", leaf_dir.name)),
                width_mm=w,
                height_mm=h,
                artifact_dir=leaf_dir,
                render_url=render_url,
                silk_min_x=silk_min_x,
                silk_min_y=silk_min_y,
                silk_max_x=silk_max_x,
                silk_max_y=silk_max_y,
                image_x_mm=image_x,
                image_y_mm=image_y,
                image_width_mm=image_w,
                image_height_mm=image_h,
                edge_min_x=edge_min_x,
                edge_min_y=edge_min_y,
                edge_max_x=edge_max_x,
                edge_max_y=edge_max_y,
            )
        )

    leaves.sort(key=lambda lf: lf.sheet_name)
    for i, lf in enumerate(leaves):
        lf.color = _LEAF_COLORS[i % len(_LEAF_COLORS)]
    return leaves
