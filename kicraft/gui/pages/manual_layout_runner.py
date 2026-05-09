"""Backend helpers for the Manual Layout page.

Discovers solved leaves, loads the most-recent auto layout (or a seeded
grid) as the canvas starting state, persists user-supplied placements
to ``manual_layout.json``, and spawns ``compose_subcircuits`` as a
background subprocess for the stamp / route phases.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kicraft.autoplacer.brain.manual_layout import (
    MOUNTING_HOLE_CORNERS,
    ManualLayout,
    ManualLeafPlacement,
    ManualMountingHole,
    ManualParentLocalPlacement,
    save_manual_layout,
)
from kicraft.autoplacer.brain.types import Point


DEFAULT_OUTLINE_W_MM = 80.0
DEFAULT_OUTLINE_H_MM = 60.0
DEFAULT_SPACING_MM = 2.0


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

    Two bboxes per leaf:

    * ``silk_min_*`` / ``silk_max_*`` -- the TIGHT content bbox
      (pads + traces + vias) the canvas uses as the leaf hit /
      preview / overflow rectangle. NOT the leaf solver's silk poly
      bbox: that's always larger because it hugs component bodies +
      0.5 mm margin (BATT's body is 76 mm long while the pads only
      cover 32 mm). Using the silk-poly bbox made the dashed red
      "extent" indicator visibly oversized on most leaves.

    * ``silk_polygon_points`` -- the actual rounded-rect silkscreen
      outline the leaf solver wrote into solved_layout.json
      (32 arc-sampled points). The canvas overlays this as a thin
      yellow polygon so the user sees the silk shape that will land
      on the stamped board, alongside the tight content bbox.
    """

    instance_path: str
    sheet_name: str
    width_mm: float
    height_mm: float
    artifact_dir: Path
    render_url: str | None = None
    color: str = "#60a5fa"
    # Tight content bbox in leaf-local coords; defaults to
    # (0,0)-(w,h) when no usable geometry is found.
    silk_min_x: float = 0.0
    silk_min_y: float = 0.0
    silk_max_x: float = 0.0
    silk_max_y: float = 0.0
    silk_corner_radius_mm: float = 1.0
    # Silkscreen poly points in leaf-local coords -- closed polygon
    # of (x, y) tuples. Empty if the leaf has no poly silk.
    silk_polygon_points: list[tuple[float, float]] = field(default_factory=list)


def _silk_polygon_from_solved_layout(leaf_dir: Path) -> list[tuple[float, float]]:
    """Return the leaf solver's silkscreen poly points in leaf-local coords.

    The leaf solver writes a single closed rounded-rect poly per leaf
    (32 arc-sampled points) via ``_silkscreen_for_label``. The canvas
    overlays this polygon as a thin yellow outline so the user sees
    the same silkscreen shape that lands on the stamped board, on
    top of the routed PNG and the dashed content bbox.

    Returns an empty list when no poly silk is on disk.
    """
    sl_path = leaf_dir / "solved_layout.json"
    try:
        sl = json.loads(sl_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    points: list[tuple[float, float]] = []
    for elem in sl.get("silkscreen", []) or []:
        if elem.get("kind") != "poly":
            continue
        for pt in elem.get("points", []) or []:
            try:
                points.append((float(pt["x"]), float(pt["y"])))
            except (KeyError, TypeError, ValueError):
                continue
        # The leaf solver writes one poly per leaf (the rounded outline).
        # Subsequent polys would be component-level body silk that we do
        # not want to draw as the overall leaf outline.
        break
    return points


def _silk_bbox_from_solved_layout(leaf_dir: Path) -> tuple[float, float, float, float] | None:
    """Compute a tight bbox of the leaf's visible copper + silk content.

    NOT the leaf solver's _silkscreen_for_label rounded poly: that
    poly's bbox is (component_BODIES ± 0.5 mm margin), which on
    leaves like BATT (where the battery holder body is 76 mm long
    but pads only appear at the ends) overstates the visible extent
    by 50%+. The user expectation is "the rectangle around what's
    actually drawn" -- pads, traces, vias.

    Compute from:
      * each pad bbox = pos ± size/2 (already on the canonical
        components dict)
      * each trace endpoint ± width/2
      * each via pos ± drill/2

    Plus a small visual margin so the rect doesn't crop the
    outermost copper. Returns None when the layout has no usable
    geometry.
    """
    sl_path = leaf_dir / "solved_layout.json"
    try:
        sl = json.loads(sl_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    xs: list[float] = []
    ys: list[float] = []

    def _accumulate(x: float, y: float, half: float = 0.0) -> None:
        xs.append(x - half)
        ys.append(y - half)
        xs.append(x + half)
        ys.append(y + half)

    for ref, comp in (sl.get("components") or {}).items():
        for pad in comp.get("pads", []) or []:
            try:
                px = float(pad["pos"]["x"])
                py = float(pad["pos"]["y"])
                sz = pad.get("size_mm") or {}
                hw = float(sz.get("x", 0.0)) / 2.0
                hh = float(sz.get("y", 0.0)) / 2.0
            except (KeyError, TypeError, ValueError):
                continue
            xs += [px - hw, px + hw]
            ys += [py - hh, py + hh]

    for trace in sl.get("traces") or []:
        try:
            x1 = float(trace["start"]["x"])
            y1 = float(trace["start"]["y"])
            x2 = float(trace["end"]["x"])
            y2 = float(trace["end"]["y"])
            half = float(trace.get("width_mm", 0.2)) / 2.0
        except (KeyError, TypeError, ValueError):
            continue
        for x, y in ((x1, y1), (x2, y2)):
            _accumulate(x, y, half)

    for via in sl.get("vias") or []:
        try:
            vx = float(via["pos"]["x"])
            vy = float(via["pos"]["y"])
            r = float(via.get("size_mm", 0.6)) / 2.0
        except (KeyError, TypeError, ValueError):
            continue
        _accumulate(vx, vy, r)

    if not xs or not ys:
        return None
    margin = 0.5
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)


def _render_url_for(experiments_dir: Path, leaf_dir: Path) -> str | None:
    """Map a leaf's routed render to its /experiments URL, after stripping
    the post-process cyan border + dark padding.

    Two-pass ImageMagick ``-trim`` removes the cyan border first (corner
    pixels are cyan) and then the dark navy padding now exposed at the
    corners. The cropped result is cached at ``*_tight.png`` next to the
    source so successive page loads avoid the magick subprocess.

    Falls back to pre_route_front_all.png if routing hasn't completed
    yet for this leaf. Returns None when neither file exists or magick
    is unavailable.
    """
    candidates = (
        leaf_dir / "renders" / "routed_front_all.png",
        leaf_dir / "renders" / "pre_route_front_all.png",
    )
    for src in candidates:
        if not src.is_file():
            continue
        tight = _make_tight_render(src) or src
        try:
            rel = tight.relative_to(experiments_dir)
        except ValueError:
            return None
        # Cache-bust on mtime so the canvas picks up new renders
        # without forcing a hard browser reload.
        return f"/experiments/{rel.as_posix()}?v={int(tight.stat().st_mtime)}"
    return None


def _make_tight_render(src: Path) -> Path | None:
    """Strip the post-process border + padding from a leaf render.

    Cached: the trimmed copy is regenerated only when the source PNG is
    newer than the cached file. Returns the cached path on success,
    None if magick isn't available (caller falls back to the source).
    """
    import shutil
    import subprocess

    out = src.with_name(src.stem + "_tight.png")
    try:
        if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
            return out
    except OSError:
        pass

    if shutil.which("magick") is None:
        return None

    # The render_pcb post-process bakes in a #020617 (DEFAULT_BACKGROUND)
    # navy fill, so trimming exposes that as the now-uniform corner. Two
    # trim passes peel cyan-border + navy-padding, then a -transparent
    # pass keys out the remaining navy fill that fills the leaf interior
    # so neighboring leaves don't occlude each other when their bboxes
    # overlap on the manual canvas.
    try:
        subprocess.run(
            [
                "magick",
                str(src),
                "-fuzz",
                "8%",
                "-trim",
                "+repage",
                "-fuzz",
                "8%",
                "-trim",
                "+repage",
                "-alpha",
                "set",
                "-fuzz",
                "12%",
                "-transparent",
                "#020617",
                "-transparent",
                "#0b192f",
                str(out),
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    return out if out.is_file() else None


def discover_leaves(experiments_dir: Path) -> list[LeafInfo]:
    """Scan .experiments/subcircuits/ for solved leaves.

    Each subdir with a ``metadata.json`` and a ``leaf_routed.kicad_pcb``
    is considered a placeable leaf. The order is stable (sorted by
    sheet_name) so leaf colours stay consistent across renders.
    """
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
        silk_bbox = _silk_bbox_from_solved_layout(leaf_dir)
        if silk_bbox is None:
            silk_min_x, silk_min_y, silk_max_x, silk_max_y = 0.0, 0.0, w, h
        else:
            silk_min_x, silk_min_y, silk_max_x, silk_max_y = silk_bbox

        silk_poly = _silk_polygon_from_solved_layout(leaf_dir)

        leaves.append(
            LeafInfo(
                instance_path=str(meta.get("instance_path", "")),
                sheet_name=str(meta.get("sheet_name", leaf_dir.name)),
                width_mm=w,
                height_mm=h,
                artifact_dir=leaf_dir,
                render_url=_render_url_for(experiments_dir, leaf_dir),
                silk_min_x=silk_min_x,
                silk_min_y=silk_min_y,
                silk_max_x=silk_max_x,
                silk_max_y=silk_max_y,
                silk_polygon_points=silk_poly,
            )
        )

    leaves.sort(key=lambda lf: lf.sheet_name)
    for i, lf in enumerate(leaves):
        lf.color = _LEAF_COLORS[i % len(_LEAF_COLORS)]
    return leaves


def load_initial_layout(
    experiments_dir: Path, leaves: list[LeafInfo]
) -> dict[str, Any]:
    """Return canvas-ready initial state.

    Tries the most-recent auto compose snapshot first
    (``parent_composition_routed.json``), then any saved
    ``manual_layout.json``, and finally a seeded-grid fallback.
    """
    saved = experiments_dir / "manual" / "manual_layout.json"
    if saved.is_file():
        try:
            return _layout_to_canvas(json.loads(saved.read_text(encoding="utf-8")), leaves)
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    auto = _find_latest_auto_layout(experiments_dir)
    if auto is not None:
        try:
            return _auto_layout_to_canvas(auto, leaves)
        except (KeyError, TypeError, ValueError):
            pass

    return _seeded_grid(leaves)


def save_manual_layout_json(
    experiments_dir: Path,
    payload: dict[str, Any],
    leaves: list[LeafInfo],
) -> Path:
    """Persist the canvas state to manual_layout.json (schema v1)."""
    placements = []
    by_path = {lf.instance_path: lf for lf in leaves}
    for entry in payload.get("placements", []) or []:
        ip = str(entry.get("instance_path", ""))
        if ip not in by_path:
            continue
        placements.append(
            ManualLeafPlacement(
                instance_path=ip,
                origin=Point(
                    float(entry["origin"]["x"]),
                    float(entry["origin"]["y"]),
                ),
                rotation=float(entry.get("rotation", 0.0)),
            )
        )

    outline_raw = payload.get("board_outline") or {}
    min_pt = Point(
        float(outline_raw["min"]["x"]), float(outline_raw["min"]["y"])
    )
    max_pt = Point(
        float(outline_raw["max"]["x"]), float(outline_raw["max"]["y"])
    )

    parent_local = []
    for entry in payload.get("parent_local", []) or []:
        parent_local.append(
            ManualParentLocalPlacement(
                ref=str(entry["ref"]),
                pos=Point(
                    float(entry["pos"]["x"]), float(entry["pos"]["y"])
                ),
            )
        )

    mounting_holes = []
    for entry in payload.get("mounting_holes", []) or []:
        try:
            corner = entry.get("corner")
            if corner is not None and corner not in MOUNTING_HOLE_CORNERS:
                corner = None
            mounting_holes.append(
                ManualMountingHole(
                    index=int(entry.get("index", len(mounting_holes))),
                    corner=corner,
                    inset_mm=float(entry.get("inset_mm", 5.0)),
                    pos=Point(
                        float(entry["pos"]["x"]), float(entry["pos"]["y"])
                    ),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    layout = ManualLayout(
        placements=placements,
        board_outline=(min_pt, max_pt),
        parent_local=parent_local,
        mounting_holes=mounting_holes,
    )
    out_path = experiments_dir / "manual" / "manual_layout.json"
    return save_manual_layout(layout, out_path)


def find_latest_parent_pcb(experiments_dir: Path) -> Path | None:
    """Find the most recent parent_routed/parent_pre_freerouting PCB.

    Prefers ``parent_routed.kicad_pcb`` when it exists and is newer
    than the stamped board; otherwise returns the stamped board so the
    user can inspect a saved layout that hasn't been routed yet.
    Returns None when neither file exists.
    """
    sub_root = experiments_dir / "subcircuits"
    if not sub_root.is_dir():
        return None
    candidates: list[Path] = []
    for d in sub_root.iterdir():
        if not d.is_dir():
            continue
        for name in ("parent_routed.kicad_pcb", "parent_pre_freerouting.kicad_pcb"):
            f = d / name
            if f.is_file():
                candidates.append(f)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def open_in_pcbnew(pcb_path: Path) -> None:
    """Launch pcbnew on the given board, detached from the GUI process.

    pcbnew is the right binary for opening a .kicad_pcb directly --
    `kicad <file>` invokes the project manager which only handles
    .kicad_pro. We Popen with start_new_session so killing the GUI
    doesn't take pcbnew with it.
    """
    import subprocess

    subprocess.Popen(
        ["pcbnew", str(pcb_path)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def run_manual_compose(
    *,
    project_root: Path,
    experiments_dir: Path,
    manual_layout_path: Path,
    pcb_file: str,
    parent: str = "/",
    route: bool = False,
) -> dict[str, Any]:
    """Spawn compose_subcircuits with --manual-layout and await completion.

    Returns a dict with keys: ``rc``, ``elapsed_s``, ``stamp_drc``,
    ``log_tail``, ``output_json``.
    """
    import time

    manual_dir = experiments_dir / "manual"
    manual_dir.mkdir(parents=True, exist_ok=True)
    output_json = manual_dir / (
        "manual_routed.json" if route else "manual_stamped.json"
    )
    log_path = manual_dir / ("route.log" if route else "stamp.log")

    pcb_path = project_root / pcb_file
    cmd = [
        sys.executable,
        "-m",
        "kicraft.cli.compose_subcircuits",
        "--project",
        str(project_root),
        "--parent",
        parent,
        "--pcb",
        str(pcb_path),
        "--manual-layout",
        str(manual_layout_path),
        "--output",
        str(output_json),
    ]
    if route:
        cmd.append("--route")
    else:
        cmd.append("--stamp")

    t0 = time.perf_counter()
    log_fh = open(log_path, "w", buffering=1, encoding="utf-8")
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(project_root),
            stdout=log_fh,
            stderr=log_fh.fileno(),
        )
        rc = await proc.wait()
    finally:
        log_fh.close()
    elapsed = time.perf_counter() - t0

    stamp_drc: dict[str, Any] = {}
    if output_json.exists():
        try:
            data = json.loads(output_json.read_text(encoding="utf-8"))
            stamp_drc = (data.get("state") or {}).get("stamp_drc", {}) or {}
        except (OSError, json.JSONDecodeError):
            pass

    log_tail = ""
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()[-25:]
            log_tail = "".join(lines)
        except OSError:
            pass

    return {
        "rc": rc,
        "elapsed_s": elapsed,
        "stamp_drc": stamp_drc,
        "log_tail": log_tail,
        "output_json": str(output_json),
    }


# --- Internals -------------------------------------------------------------


def _seeded_grid(leaves: list[LeafInfo]) -> dict[str, Any]:
    """Lay leaves on a tidy grid sized to roughly square aspect."""
    if not leaves:
        return {
            "placements": [],
            "board_outline": {
                "min": {"x": 0.0, "y": 0.0},
                "max": {"x": DEFAULT_OUTLINE_W_MM, "y": DEFAULT_OUTLINE_H_MM},
            },
            "mounting_holes": [],
        }
    cols = max(1, int(math.ceil(math.sqrt(len(leaves)))))
    cell_w = max(lf.width_mm for lf in leaves) + DEFAULT_SPACING_MM
    cell_h = max(lf.height_mm for lf in leaves) + DEFAULT_SPACING_MM
    placements = []
    for i, lf in enumerate(leaves):
        col = i % cols
        row = i // cols
        x = DEFAULT_SPACING_MM + col * cell_w
        y = DEFAULT_SPACING_MM + row * cell_h
        placements.append(
            {
                "instance_path": lf.instance_path,
                "origin": {"x": x, "y": y},
                "rotation": 0.0,
            }
        )
    rows = (len(leaves) + cols - 1) // cols
    w = DEFAULT_SPACING_MM + cols * cell_w
    h = DEFAULT_SPACING_MM + rows * cell_h
    return {
        "placements": placements,
        "board_outline": {
            "min": {"x": 0.0, "y": 0.0},
            "max": {"x": w, "y": h},
        },
        "mounting_holes": [],
    }


def _find_latest_auto_layout(experiments_dir: Path) -> dict[str, Any] | None:
    """Find the freshest parent_pipeline.json under .experiments/.

    Recent runs land in ``.experiments/hierarchical_autoexperiment/
    round_NNNN/parent_pipeline.json``; older `iter*` archives live
    under sibling directories with the same per-round shape. Pick the
    file with the latest mtime.
    """
    candidates = list(experiments_dir.glob("**/round_*/parent_pipeline.json"))
    candidates += list(experiments_dir.glob("**/parent_composition_routed.json"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _auto_layout_to_canvas(
    auto: dict[str, Any], leaves: list[LeafInfo]
) -> dict[str, Any]:
    """Map a parent_composition_routed.json snapshot onto canvas state.

    Only entries whose ``instance_path`` matches a current leaf are
    used; missing leaves fall back to grid positions so the canvas
    never starts with a blank rect.
    """
    state = auto.get("state") or auto
    entries = state.get("entries") or []
    bbox = state.get("bounding_box") or {}

    placements_by_path: dict[str, dict[str, Any]] = {}
    for e in entries:
        ip = str(e.get("instance_path", ""))
        origin = e.get("origin") or {}
        placements_by_path[ip] = {
            "instance_path": ip,
            "origin": {
                "x": float(origin.get("x", 0.0)),
                "y": float(origin.get("y", 0.0)),
            },
            "rotation": float(e.get("rotation", 0.0)),
        }

    fallback = _seeded_grid(leaves)
    fallback_by_path = {p["instance_path"]: p for p in fallback["placements"]}

    placements = []
    for lf in leaves:
        if lf.instance_path in placements_by_path:
            placements.append(placements_by_path[lf.instance_path])
        else:
            placements.append(fallback_by_path[lf.instance_path])

    try:
        tl = bbox["top_left"]
        br = bbox["bottom_right"]
        outline = {
            "min": {"x": float(tl["x"]), "y": float(tl["y"])},
            "max": {"x": float(br["x"]), "y": float(br["y"])},
        }
    except (KeyError, TypeError, ValueError):
        outline = fallback["board_outline"]

    return {"placements": placements, "board_outline": outline, "mounting_holes": []}


def _layout_to_canvas(
    payload: dict[str, Any], leaves: list[LeafInfo]
) -> dict[str, Any]:
    """Map a saved manual_layout.json onto canvas state."""
    placements_in = payload.get("placements") or []
    by_path: dict[str, dict[str, Any]] = {}
    for e in placements_in:
        ip = str(e.get("instance_path", ""))
        origin = e.get("origin") or {}
        by_path[ip] = {
            "instance_path": ip,
            "origin": {
                "x": float(origin.get("x", 0.0)),
                "y": float(origin.get("y", 0.0)),
            },
            "rotation": float(e.get("rotation", 0.0)),
        }

    fallback = _seeded_grid(leaves)
    fallback_by_path = {p["instance_path"]: p for p in fallback["placements"]}

    placements = []
    for lf in leaves:
        placements.append(by_path.get(lf.instance_path, fallback_by_path[lf.instance_path]))

    outline_in = payload.get("board_outline") or {}
    try:
        outline = {
            "min": {
                "x": float(outline_in["min"]["x"]),
                "y": float(outline_in["min"]["y"]),
            },
            "max": {
                "x": float(outline_in["max"]["x"]),
                "y": float(outline_in["max"]["y"]),
            },
        }
    except (KeyError, TypeError, ValueError):
        outline = fallback["board_outline"]

    holes_in = payload.get("mounting_holes") or []
    holes_out: list[dict[str, Any]] = []
    for h in holes_in:
        try:
            corner = h.get("corner")
            if corner is not None and corner not in MOUNTING_HOLE_CORNERS:
                corner = None
            holes_out.append(
                {
                    "index": int(h.get("index", len(holes_out))),
                    "corner": corner,
                    "inset_mm": float(h.get("inset_mm", 5.0)),
                    "pos": {
                        "x": float(h.get("pos", {}).get("x", 0.0)),
                        "y": float(h.get("pos", {}).get("y", 0.0)),
                    },
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    return {
        "placements": placements,
        "board_outline": outline,
        "mounting_holes": holes_out,
    }
