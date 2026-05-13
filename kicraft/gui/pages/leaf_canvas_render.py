"""Canvas-only PNG render for the Manual Layout tab.

Thin wrapper around ``kicraft.render.render_pcb`` (the unified
PCB-to-PNG pipeline shared with the monitor / pipeline-graph views).
This module owns the *cache* -- the canvas sidecar JSON next to the
PNG and the mtime-based invalidation against the source PCB and the
``pins.json`` file -- while the actual rasterization, viewBox-to-
Edge.Cuts override, and pixel aspect handling live in the unified
renderer so the canvas and the monitor cannot drift.

Bumping ``RENDERER_VERSION`` forces every cached canvas PNG + sidecar
to be regenerated on next page load; use when the renderer's output
shape changes (DPI, layers, background, viewBox semantics).
"""

from __future__ import annotations

import json
from pathlib import Path

from kicraft.render import EdgeCutsExtent, render_pcb

# Bump this to invalidate every cached canvas PNG + sidecar. v2 = the
# PNG content extent is now Edge.Cuts (was kicad-cli's fit-page-to-
# board viewBox, which included silk text hanging past the board).
RENDERER_VERSION = 2

DEFAULT_DPI = 420

# Front-side, top-down. Edge.Cuts is included because the unified
# renderer reads it from the PCB anyway, but we still ask kicad-cli to
# include it so the SVG carries the board outline as a visible stroke
# inside the rasterized image. B.Cu is intentionally omitted -- canvas
# is single-sided.
_LAYERS = "F.Cu,F.SilkS,Edge.Cuts"


def _sidecar_path(out_png: Path) -> Path:
    return out_png.with_suffix(out_png.suffix + ".extent.json")


def _experiments_dir_for(leaf_pcb: Path) -> Path | None:
    """``.experiments`` root inferred from a leaf PCB path. Leaf PCBs
    live at ``<experiments>/subcircuits/<leaf_key>/leaf_routed.kicad_pcb``;
    walk up two parents. Returns None when the layout doesn't match so
    callers fall back to the PCB-only mtime check."""
    try:
        sub_dir = leaf_pcb.parent.parent
        if sub_dir.name != "subcircuits":
            return None
        return sub_dir.parent
    except (AttributeError, OSError):
        return None


def _invalidating_mtime(leaf_pcb: Path) -> float:
    """Latest mtime among all signals that should bust the canvas
    cache. ``pins.json`` is checked alongside the leaf PCB because
    pin/unpin can swap the canonical file's content without advancing
    its mtime."""
    mt = leaf_pcb.stat().st_mtime
    exp_dir = _experiments_dir_for(leaf_pcb)
    if exp_dir is not None:
        try:
            pins_mt = (exp_dir / "pins.json").stat().st_mtime
            if pins_mt > mt:
                mt = pins_mt
        except OSError:
            pass
    return mt


def _read_sidecar(
    out_png: Path, leaf_pcb: Path
) -> tuple[float, float, float, float] | None:
    """Cache hit only when sidecar AND PNG are both newer than every
    invalidation signal, and the sidecar declares the current
    ``RENDERER_VERSION``."""
    sidecar = _sidecar_path(out_png)
    try:
        src_mtime = _invalidating_mtime(leaf_pcb)
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


def _write_sidecar(out_png: Path, extent: EdgeCutsExtent) -> None:
    payload = {
        "renderer_version": RENDERER_VERSION,
        "x_mm": extent.x_mm,
        "y_mm": extent.y_mm,
        "width_mm": extent.width_mm,
        "height_mm": extent.height_mm,
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
    """Render the manual-layout-canvas PNG for a single leaf. Returns
    ``(x_mm, y_mm, width_mm, height_mm)`` -- the leaf's Edge.Cuts AABB,
    which is also the rendered PNG's content extent (the unified
    renderer clips to that rectangle so both bounds equal by
    construction). ``None`` on render failure.

    The sidecar JSON next to ``out_png`` caches the extent; cache is
    keyed on the PCB's mtime, ``pins.json``'s mtime, and
    ``RENDERER_VERSION``.
    """
    if not leaf_pcb.is_file():
        return None

    cached = _read_sidecar(out_png, leaf_pcb)
    if cached is not None:
        return cached

    extent = render_pcb(
        leaf_pcb,
        out_png,
        layers=_LAYERS,
        dpi=dpi,
        style=None,
    )
    if extent is None:
        return None

    _write_sidecar(out_png, extent)
    return (extent.x_mm, extent.y_mm, extent.width_mm, extent.height_mm)
