"""Alignment-group detection and post-placement repair.

Some components are conceptually a unit and should land in a clean,
predictable arrangement: parallel batteries (BT1/BT2), arrays of
matching pin headers, an LED row, etc. Pure force-directed + SA
placement gets close but rarely produces the exact alignment a human
expects -- enough cosmetic drift to make case design hard and trigger
"this layout looks off" instincts.

This module detects those groups from configuration alone (no schema
changes required for existing projects) and applies a deterministic
repair pass after the placement solver finishes:

* **Detection**: walk ``cfg["ic_groups"]``. A group is an alignment
  candidate when its leader plus every supporting component shares the
  same KiCad ``value`` field (e.g. all "18650", all "LED_5mm"). The
  axis -- which direction the members are distributed along, and which
  axis they're aligned on -- is derived from
  ``cfg["component_zones"]`` for the leader: ``zone:bottom``/``top``
  arranges in a row (members distributed along X, aligned on Y);
  ``edge:left``/``right`` arranges in a column (distributed along Y,
  aligned on X).

* **Repair**: snap every member's perpendicular-axis coordinate to the
  group's current mean perpendicular value, then redistribute on the
  parallel axis at a body-width-derived pitch, ordered by each
  member's current parallel position. The center of the group's
  parallel-axis span is preserved so the alignment doesn't yank the
  group across the board, just tightens its internal arrangement.

The detection works for any N >= 2 members, so two batteries, a row
of three pin headers, or a strip of eight LEDs are handled by the
same code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kicraft.autoplacer.brain.types import Component, Point


@dataclass(frozen=True)
class AlignmentGroup:
    """A set of components that should share one axis and a fixed pitch."""

    leader: str
    members: tuple[str, ...]  # leader-first, then peers in declaration order
    axis: str  # "row" (members along X, share Y) or "column" (along Y, share X)
    pitch_mm: float

    @property
    def parallel_axis(self) -> str:
        """The axis members are distributed along."""
        return "x" if self.axis == "row" else "y"

    @property
    def perpendicular_axis(self) -> str:
        """The axis members are aligned on."""
        return "y" if self.axis == "row" else "x"


def detect_alignment_groups(
    cfg: dict[str, Any],
    components: dict[str, Component],
    *,
    pitch_clearance_mm: float = 1.0,
    axis_inference_tolerance_mm: float = 1.5,
) -> list[AlignmentGroup]:
    """Find ic_groups whose members are placement-alignment candidates.

    A group is a candidate when:

    * Every member listed in ``ic_groups[leader]`` (plus the leader
      itself) is present in ``components``.
    * Every member has the same non-empty ``value``.
    * The members' current positions, OR the leader's
      ``component_zones`` entry, imply an alignment axis.

    Axis derivation, in order of preference:

    1. **Position-based**: if the members' current positions vary much
       more on one axis than the other (max-delta on one axis exceeds
       ``axis_inference_tolerance_mm`` while the other doesn't), the
       smaller-spread axis is the aligned axis. This catches the
       common "user already laid them out roughly correctly in the
       schematic" case and works for batteries stacked vertically,
       headers in a row, LEDs on a strip, etc., regardless of zone
       config.
    2. **Zone-based fallback**: ``component_zones[leader]`` of
       ``zone:top``/``zone:bottom`` => row; ``edge:left``/``edge:right``
       => column. Used when positions are ambiguous (both axes have
       small or comparably large deltas), or when the placer hasn't
       run yet and components are stacked at the origin.

    Pitch defaults to the largest physical extent across the group on
    the parallel axis plus ``pitch_clearance_mm``; override via
    ``cfg["alignment_pitch_mm"]`` for explicit control.
    """
    ic_groups = cfg.get("ic_groups", {}) or {}
    component_zones = cfg.get("component_zones", {}) or {}
    explicit_pitch = cfg.get("alignment_pitch_mm")

    groups: list[AlignmentGroup] = []
    for leader_ref, peer_refs in ic_groups.items():
        if not isinstance(peer_refs, (list, tuple)):
            continue
        member_refs = [leader_ref] + [str(r) for r in peer_refs if r != leader_ref]
        if len(member_refs) < 2:
            continue
        members = []
        values = set()
        for ref in member_refs:
            comp = components.get(ref)
            if comp is None:
                break
            members.append(comp)
            values.add(str(comp.value or "").strip())
        if len(members) != len(member_refs):
            continue
        if len(values) != 1 or "" in values:
            continue  # mixed values -> not an alignment group

        # Corner-pinned leaders are never alignment groups. Two mounting
        # holes at top-left + bottom-right corners have the same value,
        # may incidentally share one axis, and would be force-aligned by
        # position-based inference -- yanking them off their corners.
        # The corner zone is an explicit signal that each member lives
        # at its own corner.
        leader_zone = component_zones.get(leader_ref)
        if isinstance(leader_zone, dict) and "corner" in leader_zone:
            continue

        axis = _infer_axis_from_positions(
            members, tolerance_mm=axis_inference_tolerance_mm
        )
        if axis is None:
            axis = _zone_to_axis(leader_zone)
        if axis is None:
            continue

        if explicit_pitch is not None:
            pitch = float(explicit_pitch)
        else:
            pitch = _derive_pitch(members, axis, pitch_clearance_mm)

        groups.append(
            AlignmentGroup(
                leader=str(leader_ref),
                members=tuple(member_refs),
                axis=axis,
                pitch_mm=pitch,
            )
        )
    return groups


def _infer_axis_from_positions(
    members: list[Component],
    *,
    tolerance_mm: float,
) -> str | None:
    """Look at members' current positions; return the axis they're aligned on.

    Returns "row" if the members' Y-spread is small relative to their
    X-spread (i.e. they sit on a roughly horizontal line, distributed
    along X), "column" if vice versa, or None when both spreads are
    comparable (no clear linear arrangement).

    The threshold ``tolerance_mm`` is the maximum across-axis spread
    that still counts as "aligned"; if BOTH axes are within tolerance,
    we don't have enough signal and return None (caller falls back to
    zone hints). The aligned axis must also be at least 2x tighter
    than the distributed axis to count as inferred -- otherwise a
    group sitting in a small cluster would be ambiguous.
    """
    if len(members) < 2:
        return None
    xs = [c.pos.x for c in members]
    ys = [c.pos.y for c in members]
    x_spread = max(xs) - min(xs)
    y_spread = max(ys) - min(ys)
    if max(x_spread, y_spread) <= tolerance_mm:
        return None  # all stacked at one point; no signal
    if y_spread <= tolerance_mm and x_spread > 2.0 * y_spread:
        return "row"  # aligned on Y, distributed on X
    if x_spread <= tolerance_mm and y_spread > 2.0 * x_spread:
        return "column"  # aligned on X, distributed on Y
    return None


def apply_alignment_repair(
    components: dict[str, Component],
    groups: list[AlignmentGroup],
) -> dict[str, Component]:
    """Snap each group's members onto a shared axis at fixed pitch.

    Returns ``components`` (mutated in place) for caller convenience.
    Members not currently in the dict are silently skipped -- the
    caller is responsible for keeping ``groups`` and ``components``
    consistent.

    Each member's perpendicular-axis coordinate is snapped to the
    group mean. Members are sorted by their current parallel-axis
    coordinate and redistributed at fixed pitch around the group's
    parallel-axis center, preserving where the group sits on the
    board.
    """
    for group in groups:
        present = [components[ref] for ref in group.members if ref in components]
        if len(present) < 2:
            continue

        if group.parallel_axis == "x":
            mean_perp = sum(c.pos.y for c in present) / len(present)
            present.sort(key=lambda c: c.pos.x)
            center_par = sum(c.pos.x for c in present) / len(present)
            for i, comp in enumerate(present):
                offset = (i - (len(present) - 1) / 2.0) * group.pitch_mm
                new_x = center_par + offset
                _move_component(comp, Point(new_x, mean_perp))
        else:
            mean_perp = sum(c.pos.x for c in present) / len(present)
            present.sort(key=lambda c: c.pos.y)
            center_par = sum(c.pos.y for c in present) / len(present)
            for i, comp in enumerate(present):
                offset = (i - (len(present) - 1) / 2.0) * group.pitch_mm
                new_y = center_par + offset
                _move_component(comp, Point(mean_perp, new_y))

    return components


def _zone_to_axis(zone_spec: Any) -> str | None:
    """Map a ``component_zones`` entry to an alignment axis.

    Returns "row" (members in a horizontal row, distributed along X
    and sharing a Y), "column" (vertical column, distributed along Y,
    sharing an X), or ``None`` if the zone doesn't imply alignment
    (e.g. corner pins or unrecognised values).
    """
    if not isinstance(zone_spec, dict):
        return None
    if zone_spec.get("zone") in ("bottom", "top"):
        return "row"
    if zone_spec.get("edge") in ("left", "right"):
        return "column"
    return None


def _derive_pitch(
    members: list[Component],
    axis: str,
    clearance_mm: float,
) -> float:
    """Pitch = max body extent on the parallel axis + clearance.

    Uses ``physical_bbox`` so pad copper that extends past the
    courtyard contributes -- otherwise a connector with outboard pads
    could be packed close enough to short with its neighbour.
    """
    max_extent = 0.0
    for comp in members:
        tl, br = comp.physical_bbox()
        if axis == "row":
            extent = br.x - tl.x
        else:
            extent = br.y - tl.y
        max_extent = max(max_extent, extent)
    return max_extent + max(0.0, clearance_mm)


def _move_component(comp: Component, new_pos: Point) -> None:
    """Translate a component (pos, body_center, every pad) to ``new_pos``."""
    delta = Point(new_pos.x - comp.pos.x, new_pos.y - comp.pos.y)
    comp.pos = new_pos
    if comp.body_center is not None:
        comp.body_center = Point(
            comp.body_center.x + delta.x,
            comp.body_center.y + delta.y,
        )
    for pad in comp.pads:
        pad.pos = Point(pad.pos.x + delta.x, pad.pos.y + delta.y)
