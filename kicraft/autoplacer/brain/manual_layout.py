"""Manual layout schema: user-specified leaf placements + board outline.

Captures the output of the GUI's manual layout mode so
``compose_subcircuits`` can bypass the solver and stamp/route the
user's placements directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import Point


SCHEMA_VERSION = "manual_layout.v1"


@dataclass(slots=True)
class ManualLeafPlacement:
    """One user-placed leaf, identified by its instance_path."""

    instance_path: str
    origin: Point
    rotation: float = 0.0


@dataclass(slots=True)
class ManualParentLocalPlacement:
    """One user-placed parent-local component (e.g., a mounting hole)."""

    ref: str
    pos: Point


# Allowed corners. ``None`` means "declared but not pinned" so the
# composer can leave the hole at its natural position rather than
# snapping it.
MOUNTING_HOLE_CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")


@dataclass(slots=True)
class ManualMountingHole:
    """One user-configured mounting hole.

    ``corner``: one of MOUNTING_HOLE_CORNERS or None (unpinned).
    ``inset_mm``: distance from the named corner along both axes.
    ``pos``: the resolved (x, y) the GUI computes from outline +
    corner + inset, persisted so the composer doesn't have to
    re-derive it.
    """

    index: int
    corner: str | None
    inset_mm: float
    pos: Point


@dataclass(slots=True)
class ManualLayout:
    """User-specified parent layout: placements + outline.

    ``board_outline`` is (min_pt, max_pt) in mm and is treated as
    authoritative -- the auto outline-fit pass is skipped when a manual
    layout is provided. ``parent_local`` is optional; entries override
    constraint-snapped positions for mounting holes / edge connectors
    that the user dragged in the GUI. ``mounting_holes`` carries the
    GUI's per-hole corner pin choices.
    """

    placements: list[ManualLeafPlacement]
    board_outline: tuple[Point, Point]
    parent_local: list[ManualParentLocalPlacement] = field(default_factory=list)
    mounting_holes: list[ManualMountingHole] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def placement_by_path(self) -> dict[str, ManualLeafPlacement]:
        return {p.instance_path: p for p in self.placements}

    def parent_local_by_ref(self) -> dict[str, ManualParentLocalPlacement]:
        return {p.ref: p for p in self.parent_local}

    def to_dict(self) -> dict[str, Any]:
        min_pt, max_pt = self.board_outline
        return {
            "schema_version": self.schema_version,
            "board_outline": {
                "min": {"x": min_pt.x, "y": min_pt.y},
                "max": {"x": max_pt.x, "y": max_pt.y},
            },
            "placements": [
                {
                    "instance_path": p.instance_path,
                    "origin": {"x": p.origin.x, "y": p.origin.y},
                    "rotation": float(p.rotation),
                }
                for p in self.placements
            ],
            "parent_local": [
                {"ref": p.ref, "pos": {"x": p.pos.x, "y": p.pos.y}}
                for p in self.parent_local
            ],
            "mounting_holes": [
                {
                    "index": h.index,
                    "corner": h.corner,
                    "inset_mm": h.inset_mm,
                    "pos": {"x": h.pos.x, "y": h.pos.y},
                }
                for h in self.mounting_holes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManualLayout":
        if not isinstance(data, dict):
            raise ValueError("manual layout must be a JSON object")
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manual layout schema_version: {version!r} "
                f"(expected {SCHEMA_VERSION!r})"
            )
        outline_raw = data.get("board_outline") or {}
        try:
            min_pt = Point(
                float(outline_raw["min"]["x"]), float(outline_raw["min"]["y"])
            )
            max_pt = Point(
                float(outline_raw["max"]["x"]), float(outline_raw["max"]["y"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid board_outline: {exc}") from exc
        if max_pt.x <= min_pt.x or max_pt.y <= min_pt.y:
            raise ValueError(
                f"board_outline must satisfy max>min, got min={min_pt}, max={max_pt}"
            )

        placements: list[ManualLeafPlacement] = []
        for entry in data.get("placements", []) or []:
            try:
                ip = str(entry["instance_path"])
                origin = Point(
                    float(entry["origin"]["x"]), float(entry["origin"]["y"])
                )
                rot = float(entry.get("rotation", 0.0))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid placement entry: {exc}") from exc
            placements.append(
                ManualLeafPlacement(instance_path=ip, origin=origin, rotation=rot)
            )

        parent_local: list[ManualParentLocalPlacement] = []
        for entry in data.get("parent_local", []) or []:
            try:
                ref = str(entry["ref"])
                pos = Point(float(entry["pos"]["x"]), float(entry["pos"]["y"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid parent_local entry: {exc}") from exc
            parent_local.append(ManualParentLocalPlacement(ref=ref, pos=pos))

        mounting_holes: list[ManualMountingHole] = []
        for entry in data.get("mounting_holes", []) or []:
            try:
                idx = int(entry["index"])
                corner = entry.get("corner")
                if corner is not None and corner not in MOUNTING_HOLE_CORNERS:
                    raise ValueError(f"invalid corner: {corner!r}")
                inset = float(entry.get("inset_mm", 5.0))
                pos = Point(float(entry["pos"]["x"]), float(entry["pos"]["y"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid mounting_holes entry: {exc}") from exc
            mounting_holes.append(
                ManualMountingHole(index=idx, corner=corner, inset_mm=inset, pos=pos)
            )

        return cls(
            placements=placements,
            board_outline=(min_pt, max_pt),
            parent_local=parent_local,
            mounting_holes=mounting_holes,
            schema_version=version,
        )


def load_manual_layout(path: str | Path) -> ManualLayout:
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        return ManualLayout.from_dict(json.load(f))


def save_manual_layout(layout: ManualLayout, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(layout.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)
    return p
