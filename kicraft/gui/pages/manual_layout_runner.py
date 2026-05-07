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
    ManualLayout,
    ManualLeafPlacement,
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
    """One discovered leaf available for manual placement."""

    instance_path: str
    sheet_name: str
    width_mm: float
    height_mm: float
    artifact_dir: Path
    render_url: str | None = None
    color: str = "#60a5fa"


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
        leaves.append(
            LeafInfo(
                instance_path=str(meta.get("instance_path", "")),
                sheet_name=str(meta.get("sheet_name", leaf_dir.name)),
                width_mm=w,
                height_mm=h,
                artifact_dir=leaf_dir,
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

    layout = ManualLayout(
        placements=placements,
        board_outline=(min_pt, max_pt),
        parent_local=parent_local,
    )
    out_path = experiments_dir / "manual" / "manual_layout.json"
    return save_manual_layout(layout, out_path)


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

    return {"placements": placements, "board_outline": outline}


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

    return {"placements": placements, "board_outline": outline}
