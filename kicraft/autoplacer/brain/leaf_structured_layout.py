"""Stage 3 of the placement streamline — structured local layout for passives.

A single deterministic pass that lays each anchor's passive group (see
``leaf_tidiness.assign_passive_groups`` — "functional rows") as a straight
row/column at a fixed, courtyard-legal pitch with **uniform orientation** (every
member on the group's dominant cardinal axis). This is the tidiness the force+SA
optimizer cannot produce: SA picks each passive's angle independently for its own
routing, so neighbours disagree 0° vs 90° — the "random" look.

Design constraints that make it safe to drop into the existing pipeline:

* **Runs late, packs legal.** It is meant to run as the last placement step
  before the final courtyard-separation pass — *after* the overlap/clearance
  passes that would otherwise blow a tight row apart. Its pitch keeps courtyards
  a real gap apart, so the final courtyard pass finds nothing to undo.
* **Best-effort, never regresses DRC.** Each member is moved only if its
  candidate placement stays inside the board, clear of every non-member's
  courtyard, and out of every keep-out. A member that can't be placed cleanly is
  left exactly where the solver put it — the pass can tidy less, never break more.
* **Passives only.** Anchors, connectors, mounting holes, locked and array parts
  never move; they only constrain.
* **Centralized geometry.** Rotation goes through
  ``geometry.rotate_component_in_place`` (KiCad CW, AABB-synced); translation
  through ``placement_utils._update_pad_positions``. No local rotation math.

Pure geometry, no pcbnew, no RNG. Mutates ``components`` in place, returns a
summary dict for debug.json.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

from .geometry import rotate_component_in_place
from .leaf_tidiness import (
    PassiveGroup,
    assign_passive_groups,
    orientation_axis,
    parts_from_components,
)
from .placement_utils import _update_pad_positions
from .types import Component, Point


def _snap(v: float, grid: float) -> float:
    return round(v / grid) * grid if grid > 0 else v


def _target_dims(comp: Component, rotate90: bool) -> tuple[float, float]:
    """Courtyard (w, h) a member would have after its target rotation."""
    return (comp.height_mm, comp.width_mm) if rotate90 else (comp.width_mm, comp.height_mm)


def _dominant_axis(comps: list[Component]) -> str:
    h = sum(1 for c in comps if orientation_axis(c.rotation) == "H")
    return "H" if h >= (len(comps) - h) else "V"


# Nets with more pads than this across the leaf are treated as power/global
# (GND, VCC, rails) — poured/planed, not point-to-point routed — so they are
# excluded from the routability (HPWL) guard, whose signal would otherwise be
# swamped by a rail's board-spanning bbox that a passive move barely changes.
_POWER_NET_PAD_THRESHOLD = 6


def _signal_nets(components: dict[str, Component], member_refs: set[str]) -> set[str]:
    """Nets a group's members connect, minus the high-fanout power/global nets."""
    pad_count: dict[str, int] = {}
    member_nets: set[str] = set()
    for ref, comp in components.items():
        for pad in comp.pads:
            if not pad.net:
                continue
            pad_count[pad.net] = pad_count.get(pad.net, 0) + 1
            if ref in member_refs:
                member_nets.add(pad.net)
    return {
        net for net in member_nets
        if pad_count.get(net, 0) <= _POWER_NET_PAD_THRESHOLD
    }


def _hpwl(
    components: dict[str, Component],
    nets: set[str],
    override: dict[str, Component],
) -> float:
    """Total half-perimeter wirelength over ``nets``, using ``override[ref]``
    geometry where present (the candidate placement) else the live component.

    HPWL is the standard cheap routability proxy: a placement that stretches a
    net's bounding box is harder to route. Summed over the group's signal nets,
    it tells us whether a tidy row would fight the router before we commit it."""
    if not nets:
        return 0.0
    pts: dict[str, list[Point]] = {}
    for ref, comp in components.items():
        c = override.get(ref, comp)
        for pad in c.pads:
            if pad.net in nets:
                pts.setdefault(pad.net, []).append(pad.pos)
    total = 0.0
    for net_pts in pts.values():
        if len(net_pts) < 2:
            continue
        xs = [p.x for p in net_pts]
        ys = [p.y for p in net_pts]
        total += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return total


def _rects_overlap(
    a_tl: Point, a_br: Point, b_tl: Point, b_br: Point, margin: float
) -> bool:
    return not (
        a_br.x <= b_tl.x - margin
        or a_tl.x >= b_br.x + margin
        or a_br.y <= b_tl.y - margin
        or a_tl.y >= b_br.y + margin
    )


def _candidate_is_legal(
    candidate: Component,
    non_members: list[Component],
    *,
    board_outline: tuple[Point, Point],
    pad_inset: float,
    courtyard_margin: float,
    keepout_rects: list[tuple[Point, Point]],
) -> bool:
    """A moved passive is legal iff it stays on-board, clear of every non-member
    courtyard (same layer), and out of every keep-out rect."""
    p_tl, p_br = candidate.physical_bbox()
    b_tl, b_br = board_outline
    if (
        p_tl.x < b_tl.x + pad_inset
        or p_br.x > b_br.x - pad_inset
        or p_tl.y < b_tl.y + pad_inset
        or p_br.y > b_br.y - pad_inset
    ):
        return False

    c_tl, c_br = candidate.bbox(courtyard_margin)
    for other in non_members:
        if other.layer != candidate.layer:
            continue
        o_tl, o_br = other.bbox(0.0)
        if _rects_overlap(c_tl, c_br, o_tl, o_br, 0.0):
            return False

    for k_tl, k_br in keepout_rects:
        if _rects_overlap(p_tl, p_br, k_tl, k_br, 0.0):
            return False
    return True


def _place_member(comp: Component, target: Point, rotate90: bool) -> None:
    """Rotate ``comp`` to its target axis (if needed), then translate so its
    body_center lands on ``target``. Mutates ``comp`` in place."""
    if rotate90:
        rotate_component_in_place(comp, 90.0)
    bc = comp.body_center if comp.body_center is not None else comp.pos
    move_x = target.x - bc.x
    move_y = target.y - bc.y
    old = Point(comp.pos.x, comp.pos.y)
    comp.pos = Point(comp.pos.x + move_x, comp.pos.y + move_y)
    _update_pad_positions(comp, old, comp.rotation)


def apply_structured_local_layout(
    components: dict[str, Component],
    *,
    board_outline: tuple[Point, Point],
    pitch_gap_mm: float = 0.6,
    grid_mm: float = 0.5,
    pad_inset_mm: float = 0.3,
    courtyard_margin_mm: float = 0.15,
    max_hpwl_increase: float = 0.15,
    keepout_rects: Optional[list[tuple[Point, Point]]] = None,
) -> dict[str, Any]:
    """Lay each functional passive group as a tidy row/column. See module docs.

    ``keepout_rects`` are pre-resolved ``(tl, br)`` rectangles in board coords
    (RF near-field, keep-ins). ``max_hpwl_increase`` is the routability guard: a
    tidy row is committed only if it grows the group's signal-net wirelength by
    no more than this fraction (0.15 = 15%) — so tidiness never fights the router
    on dense leaves. Returns a summary for debug.json.
    """
    summary: dict[str, Any] = {
        "enabled": True,
        "groups": 0,
        "groups_placed": 0,
        "groups_skipped": 0,
        "groups_skipped_routability": 0,
        "members_aligned": 0,
        "members_rotated": 0,
    }
    keepout_rects = keepout_rects or []

    groups: list[PassiveGroup] = assign_passive_groups(parts_from_components(components))
    if not groups:
        return summary

    for group in groups:
        members = [
            components[r]
            for r in group.passive_refs
            if r in components
            and not components[r].locked
            and not getattr(components[r], "array_member", False)
        ]
        if len(members) < 2:
            continue
        summary["groups"] += 1

        # Row (distributed along X, sharing a Y) vs column, from current spread.
        centers = [
            (m.body_center if m.body_center is not None else m.pos) for m in members
        ]
        xs = [p.x for p in centers]
        ys = [p.y for p in centers]
        horizontal = (max(xs) - min(xs)) >= (max(ys) - min(ys))

        dominant = _dominant_axis(members)
        rot_needed = {m.ref: (orientation_axis(m.rotation) != dominant) for m in members}

        # Uniform pitch = widest member extent along the row axis + gap.
        extents = []
        for m in members:
            w, h = _target_dims(m, rot_needed[m.ref])
            extents.append(w if horizontal else h)
        pitch = max(extents) + max(0.0, pitch_gap_mm)

        # Preserve where the group already sits: center on the parallel axis,
        # shared mean on the perpendicular axis.
        member_set = set(group.passive_refs)
        non_members = [c for r, c in components.items() if r not in member_set]

        order = sorted(members, key=lambda m: (
            (m.body_center or m.pos).x if horizontal else (m.body_center or m.pos).y
        ))
        if horizontal:
            base_par = sum(xs) / len(xs)
            base_perp = sum(ys) / len(ys)
        else:
            base_par = sum(ys) / len(ys)
            base_perp = sum(xs) / len(xs)

        # Atomic per group with a bounded relocation search. A straight row at
        # the group's current center often lands on the anchor IC's courtyard;
        # rather than abandon the group, try shifting the whole row (mostly
        # perpendicular — beside the IC — plus small parallel nudges) to the
        # nearest fully-legal position. We commit the WHOLE row or nothing:
        # uniform pitch makes intra-group collisions impossible, so a fully
        # placed row never overlaps itself, and we never place partially. The
        # pass therefore introduces zero new overlaps.
        candidates = _search_legal_row(
            order,
            base_par=base_par,
            base_perp=base_perp,
            pitch=pitch,
            horizontal=horizontal,
            rot_needed=rot_needed,
            grid_mm=grid_mm,
            non_members=non_members,
            board_outline=board_outline,
            pad_inset_mm=pad_inset_mm,
            courtyard_margin_mm=courtyard_margin_mm,
            keepout_rects=keepout_rects,
        )
        if candidates is None:
            summary["groups_skipped"] += 1
            continue

        # Routability guard: a legal tidy row can still fight the router by
        # stretching the group's signal nets (the RP2040-dense-leaf regression).
        # Commit only if the group's signal-net HPWL doesn't grow beyond the
        # tolerance; otherwise keep the solver's routability-driven placement.
        signal_nets = _signal_nets(components, member_set)
        if signal_nets:
            override = {comp.ref: cand for comp, cand, _ in candidates}
            before = _hpwl(components, signal_nets, {})
            after = _hpwl(components, signal_nets, override)
            if after > before * (1.0 + max(0.0, max_hpwl_increase)):
                summary["groups_skipped_routability"] += 1
                continue

        summary["groups_placed"] += 1
        for comp, candidate, r90 in candidates:
            comp.pos = candidate.pos
            comp.rotation = candidate.rotation
            comp.body_center = candidate.body_center
            comp.width_mm = candidate.width_mm
            comp.height_mm = candidate.height_mm
            comp.pads = candidate.pads
            summary["members_aligned"] += 1
            if r90:
                summary["members_rotated"] += 1

    return summary


def _offset_sequence(step: float, n: int) -> list[float]:
    """[0, +step, -step, +2step, -2step, ...] out to ``n`` rings."""
    seq = [0.0]
    for k in range(1, n + 1):
        seq.extend((k * step, -k * step))
    return seq


def _search_legal_row(
    order: list[Component],
    *,
    base_par: float,
    base_perp: float,
    pitch: float,
    horizontal: bool,
    rot_needed: dict[str, bool],
    grid_mm: float,
    non_members: list[Component],
    board_outline: tuple[Point, Point],
    pad_inset_mm: float,
    courtyard_margin_mm: float,
    keepout_rects: list[tuple[Point, Point]],
) -> Optional[list[tuple[Component, Component, bool]]]:
    """Return committable (comp, candidate, r90) triples for the first fully-legal
    row placement found by a bounded shift search, or ``None`` if none fits.

    Perpendicular is the primary escape axis (slide the row off the anchor);
    parallel gets a couple of small nudges. Deterministic ring order.
    """
    start0 = base_par - (len(order) - 1) * pitch / 2.0
    perp_offs = _offset_sequence(max(0.5, grid_mm), 12)   # up to ~6mm each way
    par_offs = _offset_sequence(max(0.5, pitch / 2.0), 2)

    for dperp in perp_offs:
        perp = _snap(base_perp + dperp, grid_mm)
        for dpar in par_offs:
            start = start0 + dpar
            row: list[tuple[Component, Component, bool]] = []
            ok = True
            for i, comp in enumerate(order):
                par = _snap(start + i * pitch, grid_mm)
                target = Point(par, perp) if horizontal else Point(perp, par)
                r90 = rot_needed[comp.ref]
                candidate = copy.deepcopy(comp)
                _place_member(candidate, target, r90)
                if not _candidate_is_legal(
                    candidate,
                    non_members,
                    board_outline=board_outline,
                    pad_inset=pad_inset_mm,
                    courtyard_margin=courtyard_margin_mm,
                    keepout_rects=keepout_rects,
                ):
                    ok = False
                    break
                row.append((comp, candidate, r90))
            if ok:
                return row
    return None


__all__ = ["apply_structured_local_layout"]
