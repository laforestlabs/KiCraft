"""Backend helpers for the manual layout editor.

Loads the most-recent auto layout (or a seeded grid) as the canvas
starting state, persists user-supplied placements to
``manual_layout.json``, and spawns ``compose_subcircuits`` as a
background subprocess for the stamp / route phases. Host-agnostic:
the offline GUI and the web app both drive these helpers.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import sys
from pathlib import Path
from typing import Any

from kicraft.autoplacer.brain.types import Point
from kicraft.layout_editor.leaves import LeafInfo
from kicraft.layout_editor.model import (
    DEFAULT_MOUNTING_HOLE_SCREW,
    MOUNTING_HOLE_CORNERS,
    ManualLayout,
    ManualLeafPlacement,
    ManualMountingHole,
    ManualParentLocalPlacement,
    save_manual_layout,
)
from kicraft.layout_editor.outline import OutlineSpec

_RECT_SHAPE = {"shape": "rect", "corner_radius_mm": 0.0, "chamfer_mm": 0.0}


DEFAULT_OUTLINE_W_MM = 80.0
DEFAULT_OUTLINE_H_MM = 60.0
DEFAULT_SPACING_MM = 2.0


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
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
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

    outline_raw = payload.get("outline")
    if outline_raw:
        outline = OutlineSpec.from_dict(outline_raw)
    else:
        # Legacy canvas payload: rectangular board_outline only.
        legacy = payload.get("board_outline") or {}
        outline = OutlineSpec.rect(
            Point(float(legacy["min"]["x"]), float(legacy["min"]["y"])),
            Point(float(legacy["max"]["x"]), float(legacy["max"]["y"])),
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
                    screw=str(entry.get("screw", DEFAULT_MOUNTING_HOLE_SCREW)),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    layout = ManualLayout(
        placements=placements,
        outline=outline,
        parent_local=parent_local,
        mounting_holes=mounting_holes,
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
            start_new_session=True,
        )
        try:
            rc = await proc.wait()
        except asyncio.CancelledError:
            # The awaiting task was cancelled (layout_panel's wait_for stamp
            # timeout): kill the whole compose process group -- pcbnew /
            # kicad-cli children included -- or the orphan keeps writing
            # parent_pre_freerouting.kicad_pcb / manual_stamped.json into the
            # workspace, racing the next save/stamp or queued build.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.kill()
                except (ProcessLookupError, OSError):
                    pass
            raise
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
            "outline_shape": dict(_RECT_SHAPE),
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
        "outline_shape": dict(_RECT_SHAPE),
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

    return {
        "placements": placements,
        "board_outline": outline,
        "outline_shape": dict(_RECT_SHAPE),
        "mounting_holes": [],
    }


def _layout_to_canvas(
    payload: dict[str, Any], leaves: list[LeafInfo]
) -> dict[str, Any]:
    """Map a saved manual_layout.json (v1 or v2) onto canvas state.

    Raises ``ValueError`` on a malformed payload; the caller falls
    back to the auto layout / seeded grid.
    """
    layout = ManualLayout.from_dict(payload)
    by_path = {
        p.instance_path: {
            "instance_path": p.instance_path,
            "origin": {"x": p.origin.x, "y": p.origin.y},
            "rotation": float(p.rotation),
        }
        for p in layout.placements
    }

    fallback = _seeded_grid(leaves)
    fallback_by_path = {p["instance_path"]: p for p in fallback["placements"]}

    placements = []
    for lf in leaves:
        placements.append(by_path.get(lf.instance_path, fallback_by_path[lf.instance_path]))

    spec = layout.outline
    return {
        "placements": placements,
        "board_outline": {
            "min": {"x": spec.min_pt.x, "y": spec.min_pt.y},
            "max": {"x": spec.max_pt.x, "y": spec.max_pt.y},
        },
        "outline_shape": {
            "shape": spec.shape,
            "corner_radius_mm": spec.corner_radius_mm,
            "chamfer_mm": spec.chamfer_mm,
        },
        "mounting_holes": [
            {
                "index": h.index,
                "corner": h.corner,
                "inset_mm": h.inset_mm,
                "pos": {"x": h.pos.x, "y": h.pos.y},
                "screw": h.screw,
            }
            for h in layout.mounting_holes
        ],
        # Opaque canvas passthrough (the canvas echoes it via getState):
        # dropping it here made every web-panel save wipe the layout's
        # parent_local overrides while the composer still honors them.
        "parent_local": [
            {"ref": p.ref, "pos": {"x": p.pos.x, "y": p.pos.y}}
            for p in layout.parent_local
        ],
    }
