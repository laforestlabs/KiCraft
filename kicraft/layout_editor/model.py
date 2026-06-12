"""Manual layout schema: user-specified leaf placements + board outline.

Captures the output of the manual layout editor (offline GUI or web)
so ``compose_subcircuits`` can bypass the solver and stamp/route the
user's placements directly.

Schema history:

- ``manual_layout.v1``: rectangular ``board_outline`` (min/max only),
  mounting holes without a screw size. Still loadable; migrated to v2
  on read (shape ``rect``, screw ``M3``).
- ``manual_layout.v2``: first-class ``outline`` (``OutlineSpec``:
  rect / rounded_rect / circle / chamfered_rect + parameters);
  mounting holes carry ``screw``. Always written on save.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kicraft.autoplacer.brain.types import Point
from kicraft.layout_editor.outline import OutlineSpec


SCHEMA_VERSION = "manual_layout.v2"
SCHEMA_VERSION_V1 = "manual_layout.v1"

DEFAULT_MOUNTING_HOLE_SCREW = "M3"


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
    ``screw``: fastener size key ("M2" / "M2.5" / "M3" / "M4"); drives
    the drill/pad geometry when the composer synthesizes a footprint
    for the hole.
    """

    index: int
    corner: str | None
    inset_mm: float
    pos: Point
    screw: str = DEFAULT_MOUNTING_HOLE_SCREW


@dataclass(slots=True)
class ManualLayout:
    """User-specified parent layout: placements + outline.

    ``outline`` is authoritative -- the auto outline-fit pass is
    skipped and the outline-repair grow is disabled when a manual
    layout is provided. ``parent_local`` is optional; entries override
    constraint-snapped positions for mounting holes / edge connectors
    that the user dragged in the GUI. ``mounting_holes`` carries the
    GUI's per-hole corner pin choices.
    """

    placements: list[ManualLeafPlacement]
    outline: OutlineSpec
    parent_local: list[ManualParentLocalPlacement] = field(default_factory=list)
    mounting_holes: list[ManualMountingHole] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    @property
    def board_outline(self) -> tuple[Point, Point]:
        """AABB view, for consumers that predate outline shapes."""
        return self.outline.aabb()

    def placement_by_path(self) -> dict[str, ManualLeafPlacement]:
        return {p.instance_path: p for p in self.placements}

    def parent_local_by_ref(self) -> dict[str, ManualParentLocalPlacement]:
        return {p.ref: p for p in self.parent_local}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "outline": self.outline.to_dict(),
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
                    "screw": h.screw,
                }
                for h in self.mounting_holes
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManualLayout":
        if not isinstance(data, dict):
            raise ValueError("manual layout must be a JSON object")
        version = data.get("schema_version")
        if version == SCHEMA_VERSION:
            outline = OutlineSpec.from_dict(data.get("outline") or {})
        elif version == SCHEMA_VERSION_V1:
            # v1: rectangular board_outline only.
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
            outline = OutlineSpec.rect(min_pt, max_pt)
        else:
            raise ValueError(
                f"unsupported manual layout schema_version: {version!r} "
                f"(expected {SCHEMA_VERSION!r} or {SCHEMA_VERSION_V1!r})"
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
                screw = str(entry.get("screw", DEFAULT_MOUNTING_HOLE_SCREW))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid mounting_holes entry: {exc}") from exc
            mounting_holes.append(
                ManualMountingHole(
                    index=idx, corner=corner, inset_mm=inset, pos=pos, screw=screw
                )
            )

        return cls(
            placements=placements,
            outline=outline,
            parent_local=parent_local,
            mounting_holes=mounting_holes,
            schema_version=SCHEMA_VERSION,
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
