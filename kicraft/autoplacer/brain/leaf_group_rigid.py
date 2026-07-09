"""Rigid tidy groups — the representation change at the heart of the streamline.

Instead of placing every passive independently and then re-tidying (which fights
the router — see the RP2040 routing regression), a functional group (anchor +
its passives) becomes a **rigid unit that is tidy by construction**. The group's
passives are laid out once as a straight row/column at a fixed pitch with uniform
orientation, stored as offsets from the anchor; thereafter the group moves and
rotates as one. The optimizer only ever places anchors and free parts, so every
state it visits is already tidy — there is nothing to clean up, and routability
is optimized *within* the tidy space rather than against a post-pass.

Two operations:

* :func:`build_rigid_groups` — freeze each group's tidy internal layout as
  board-frame offsets from its anchor (computed on the side the passives already
  gravitate to, so it respects the solver's routability intuition), plus the
  anchor's build-time rotation so a later anchor rotation carries the group.
* :func:`sync_rigid_groups` — re-place every member from its anchor's *current*
  pose. Idempotent; call it whenever anchors have moved and before scoring.

Pure geometry, KiCad-CW rotation via :mod:`geometry`. No pcbnew, no RNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import rotate_component_in_place, rotate_vector
from .leaf_tidiness import (
    assign_passive_groups,
    orientation_axis,
    parts_from_components,
)
from .placement_utils import _update_pad_positions
from .types import Component, Point


@dataclass(slots=True)
class RigidGroup:
    """A group frozen as rigid offsets from its anchor."""

    anchor_ref: str
    member_refs: tuple[str, ...]
    anchor_rot_build: float
    offset_board: dict[str, Point] = field(default_factory=dict)  # member_bc - anchor_bc
    member_rot_build: dict[str, float] = field(default_factory=dict)

    def child_refs(self) -> set[str]:
        return set(self.member_refs)


def _bc(c: Component) -> Point:
    return c.body_center if c.body_center is not None else c.pos


def _dominant_axis(members: list[Component]) -> str:
    h = sum(1 for m in members if orientation_axis(m.rotation) == "H")
    return "H" if h >= (len(members) - h) else "V"


def _snap(v: float, grid: float) -> float:
    return round(v / grid) * grid if grid > 0 else v


def build_rigid_groups(
    components: dict[str, Component],
    *,
    pitch_gap_mm: float = 0.6,
    grid_mm: float = 0.5,
) -> list[RigidGroup]:
    """Freeze each functional group's tidy internal layout as anchor offsets.

    The row is laid on the side the members currently occupy relative to the
    anchor (preserving routability intent), straightened onto a shared axis at
    uniform pitch, with every member on the group's dominant cardinal axis.
    """
    rigid: list[RigidGroup] = []
    for g in assign_passive_groups(parts_from_components(components)):
        anchor = components.get(g.anchor_ref)
        members = [
            components[r]
            for r in g.passive_refs
            if r in components
            and not components[r].locked
            and not getattr(components[r], "array_member", False)
        ]
        if anchor is None or len(members) < 2:
            continue

        a_bc = _bc(anchor)
        rel = {m.ref: Point(_bc(m).x - a_bc.x, _bc(m).y - a_bc.y) for m in members}
        xs = [p.x for p in rel.values()]
        ys = [p.y for p in rel.values()]
        horizontal = (max(xs) - min(xs)) >= (max(ys) - min(ys))
        dominant = _dominant_axis(members)
        target_rot = 0.0 if dominant == "H" else 90.0

        # Uniform pitch = widest member extent along the row axis + gap. Extent
        # uses the member's dimension AT its target orientation.
        def _extent(m: Component) -> float:
            rot90 = orientation_axis(m.rotation) != dominant
            w, h = (m.height_mm, m.width_mm) if rot90 else (m.width_mm, m.height_mm)
            return w if horizontal else h

        pitch = max(_extent(m) for m in members) + max(0.0, pitch_gap_mm)

        order = sorted(members, key=lambda m: rel[m.ref].x if horizontal else rel[m.ref].y)
        if horizontal:
            par_c = sum(xs) / len(xs)
            perp = _snap(sum(ys) / len(ys), grid_mm)
        else:
            par_c = sum(ys) / len(ys)
            perp = _snap(sum(xs) / len(xs), grid_mm)
        start = par_c - (len(order) - 1) * pitch / 2.0

        offsets: dict[str, Point] = {}
        rots: dict[str, float] = {}
        for i, m in enumerate(order):
            par = _snap(start + i * pitch, grid_mm)
            offsets[m.ref] = Point(par, perp) if horizontal else Point(perp, par)
            rots[m.ref] = target_rot

        rigid.append(
            RigidGroup(
                anchor_ref=g.anchor_ref,
                member_refs=tuple(m.ref for m in order),
                anchor_rot_build=anchor.rotation,
                offset_board=offsets,
                member_rot_build=rots,
            )
        )
    return rigid


def _place(comp: Component, target_bc: Point, target_rot: float) -> None:
    """Rotate ``comp`` to ``target_rot`` then translate its body_center to
    ``target_bc``. Mutates in place."""
    drot = (target_rot - comp.rotation) % 360.0
    if abs(drot) > 1e-6:
        rotate_component_in_place(comp, drot)
    bc = _bc(comp)
    dx, dy = target_bc.x - bc.x, target_bc.y - bc.y
    old = Point(comp.pos.x, comp.pos.y)
    comp.pos = Point(comp.pos.x + dx, comp.pos.y + dy)
    _update_pad_positions(comp, old, comp.rotation)


def sync_rigid_groups(
    components: dict[str, Component], rigid: list[RigidGroup]
) -> None:
    """Re-place every member from its anchor's current pose (rigid follow)."""
    for rg in rigid:
        anchor = components.get(rg.anchor_ref)
        if anchor is None:
            continue
        a_bc = _bc(anchor)
        drot = (anchor.rotation - rg.anchor_rot_build) % 360.0
        for ref in rg.member_refs:
            m = components.get(ref)
            if m is None:
                continue
            off = rotate_vector(rg.offset_board[ref], drot)
            target_bc = Point(a_bc.x + off.x, a_bc.y + off.y)
            target_rot = (rg.member_rot_build[ref] + drot) % 360.0
            _place(m, target_bc, target_rot)


def group_child_refs(rigid: list[RigidGroup]) -> set[str]:
    """All member refs that move only via their anchor (SA must skip these)."""
    out: set[str] = set()
    for rg in rigid:
        out |= rg.child_refs()
    return out


__all__ = [
    "RigidGroup",
    "build_rigid_groups",
    "sync_rigid_groups",
    "group_child_refs",
]
