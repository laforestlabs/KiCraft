"""Discrete anchor-relative placement grid + SA-as-assignment.

The connectivity-first answer to "tidy vs routable": instead of placing passives
on *continuous* positions and then re-tidying (which drifts a decap 6-20 mm from
the IC pins it bridges), restrict each passive to a discrete set of **pin-adjacent
slots** generated from the placed anchors' pad geometry. SA stops nudging x,y and
instead *assigns* which passive occupies which slot (and at what admitted
rotation). Every candidate state is grid-aligned and pre-spaced, so:

* **tidiness is structural** — rows/uniform spacing are guaranteed, nothing to
  score;
* **legality is by construction** — slots are culled up front against anchor
  courtyards, keep-outs and the board edge, so overlaps cannot occur;
* **the routable choice is the tidy choice** — a decap is assigned to the slot
  next to its IC power/ground pins, so pin-locality (the real objective) is what
  the assignment optimizes.

Two operations, mirroring :mod:`leaf_group_rigid`:

* :func:`build_anchor_grid` — over-provision pin-adjacent slots around every
  placed anchor (plus a straight lane per anchor-less passive array), pre-spaced
  at a courtyard-legal pitch, illegal slots culled.
* :func:`grid_assignment_sa` — Metropolis search over slot occupancy (swap two
  occupants, move one to a free slot, re-rotate within a slot's admitted set),
  scored by the same :class:`PlacementScorer` (pin-locality + routing terms; no
  tidiness term needed).

Pure geometry + the shared PlacementScorer; KiCad-CW rotation via :mod:`geometry`.
No pcbnew. Deterministic (sorted iteration + the solver's RNG).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Optional

from .geometry import rotate_component_in_place
from .leaf_tidiness import (
    _ANCHOR_KINDS,
    assign_passive_groups,
    orientation_axis,
    parts_from_components,
)
from .placement_utils import _update_pad_positions
from .types import Component, Point


@dataclass(slots=True)
class Slot:
    """One allowed placement location for a passive body center."""

    sid: int
    pos: Point  # target body-center, world mm (grid-snapped)
    admitted_rotations: tuple[float, ...]  # rotations a passive may take here
    anchor_ref: str  # the anchor this slot hangs off ("array:<ref>" for a lane)
    near_pins: tuple[tuple[str, str], ...]  # (anchor_ref, pad_id) pins adjacent
    nets: frozenset[str]  # nets of those pins -> fast candidate matching
    side: str  # "N"/"E"/"S"/"W" (anchor edge) or "lane" (anchor-less array)
    occupant: Optional[str] = None


@dataclass
class Grid:
    """The slot set plus its live occupancy bookkeeping."""

    slots: list[Slot] = field(default_factory=list)
    by_net: dict[str, list[int]] = field(default_factory=dict)
    free: set[int] = field(default_factory=set)
    occupied_by_ref: dict[str, int] = field(default_factory=dict)
    rotation_by_ref: dict[str, float] = field(default_factory=dict)

    def child_refs(self) -> set[str]:
        """Passives that move only via the grid (SA-assignment owns them)."""
        return set(self.occupied_by_ref)


# --------------------------------------------------------------------------- #
# Small geometry helpers (self-contained; mirror leaf_group_rigid).
# --------------------------------------------------------------------------- #


def _bc(c: Component) -> Point:
    return c.body_center if c.body_center is not None else c.pos


def _snap(v: float, grid: float) -> float:
    return round(v / grid) * grid if grid > 0 else v


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _place_component(comp: Component, target_bc: Point, target_rot: float) -> None:
    """Rotate ``comp`` to ``target_rot`` then translate its body_center to
    ``target_bc`` (pads kept in sync). Mutates in place -- same primitive as
    ``leaf_group_rigid._place``."""
    drot = (target_rot - comp.rotation) % 360.0
    if 1e-6 < drot < 360.0 - 1e-6:
        rotate_component_in_place(comp, drot)
    bc = _bc(comp)
    dx, dy = target_bc.x - bc.x, target_bc.y - bc.y
    old = Point(comp.pos.x, comp.pos.y)
    comp.pos = Point(comp.pos.x + dx, comp.pos.y + dy)
    _update_pad_positions(comp, old, comp.rotation)


def _admitted(side: str, policy: str) -> tuple[float, ...]:
    """Rotations a slot admits, per the orientation policy (decision (c)).

    ``auto`` keeps a passive's long axis parallel to the anchor edge it sits on
    (H on N/S lanes, V on E/W lanes) -- the tidy, route-natural choice.
    """
    if policy == "force_h":
        return (0.0, 180.0)
    if policy == "force_v":
        return (90.0, 270.0)
    if policy == "both":
        return (0.0, 90.0, 180.0, 270.0)
    # auto
    return (0.0, 180.0) if side in ("N", "S", "lane") else (90.0, 270.0)


def _pick_rotation(admitted: tuple[float, ...], current: float) -> float:
    """The admitted rotation whose axis matches ``current`` (keeps the part's
    current orientation when the slot allows it), else the first admitted."""
    cur = orientation_axis(current)
    for r in admitted:
        if orientation_axis(r) == cur:
            return r
    return admitted[0]


def _inside(pos: Point, half: float, tl: Point, br: Point, inset: float) -> bool:
    return (
        tl.x + inset + half <= pos.x <= br.x - inset - half
        and tl.y + inset + half <= pos.y <= br.y - inset - half
    )


def _overlaps_rect(pos: Point, half: float, rtl: Point, rbr: Point) -> bool:
    return (
        pos.x - half < rbr.x and pos.x + half > rtl.x
        and pos.y - half < rbr.y and pos.y + half > rtl.y
    )


# --------------------------------------------------------------------------- #
# Slot generation.
# --------------------------------------------------------------------------- #


def _gridable_passives(comps: dict[str, Component]) -> list[Component]:
    return [
        c for c in comps.values()
        if (c.kind or "") == "passive"
        and not c.locked
        and not getattr(c, "array_member", False)
        and c.pads
    ]


def build_anchor_grid(
    placed_comps: dict[str, Component],
    *,
    board_outline: tuple[Point, Point],
    pitch_gap_mm: float,
    rings: int = 2,
    lateral: int = 1,
    overprovision: float = 10.0,
    max_slots: int = 400,
    orientation_policy: str = "auto",
    grid_snap: float = 0.5,
    keepout_rects: Optional[list[tuple[Point, Point]]] = None,
    pad_inset_mm: float = 0.2,
) -> Grid:
    """Build the anchor-relative slot grid from the *placed* anchors.

    Pin-adjacent slots step outward (``rings``) from each anchor pad along the
    outward cardinal normal, with a lateral spread (``lateral``) so same-side
    pads share a lane. Pitch = widest passive extent + ``pitch_gap_mm`` so slots
    are courtyard-legal by construction. Slots overlapping an anchor courtyard, a
    keep-out, or off the board are culled. An anchor-less passive array (no IC in
    the leaf) instead gets one straight lane per chain-ordered cluster.
    """
    keepout_rects = keepout_rects or []
    tl, br = board_outline
    passives = _gridable_passives(placed_comps)
    if not passives:
        return Grid()

    long_extent = max(max(c.width_mm, c.height_mm) for c in passives)
    half = long_extent / 2.0
    pitch = long_extent + max(0.0, pitch_gap_mm)  # ring STEP + min inter-slot sep
    # ring-1 sits TIGHT to the pins (a decap belongs ~1 mm away, not a full
    # clearance out); the up-front cull drops any ring-1 that would overlap the
    # anchor courtyard, so the closest surviving ring is as pin-adjacent as
    # legality allows. Decoupling from the ring step is what gets pin_mm to ~1-2.
    base = half + min(max(0.0, pitch_gap_mm), 0.6)

    anchors = [c for c in placed_comps.values() if (c.kind or "") in _ANCHOR_KINDS]
    anchor_rects = [
        (Point(_bc(a).x - a.width_mm / 2, _bc(a).y - a.height_mm / 2),
         Point(_bc(a).x + a.width_mm / 2, _bc(a).y + a.height_mm / 2))
        for a in anchors
    ]

    def _legal(pos: Point) -> bool:
        if not _inside(pos, half, tl, br, pad_inset_mm):
            return False
        for rtl, rbr in anchor_rects:
            if _overlaps_rect(pos, half, rtl, rbr):
                return False
        for rtl, rbr in keepout_rects:
            if _overlaps_rect(pos, half, rtl, rbr):
                return False
        return True

    # Accumulate candidate slots keyed by snapped position so slots between two
    # pins merge -- carrying BOTH nets, so a decap straddling them matches
    # strongly (and gets both admitted-rotation sets unioned).
    cand: dict[tuple[float, float], dict] = {}

    def _add(pos: Point, admitted: tuple[float, ...], anchor_ref: str,
             pins: list[tuple[str, str]], nets: frozenset[str], side: str) -> None:
        if not _legal(pos):
            return
        key = (round(pos.x, 3), round(pos.y, 3))
        e = cand.get(key)
        if e is None:
            cand[key] = {
                "pos": pos, "admitted": set(admitted), "anchor": anchor_ref,
                "pins": set(pins), "nets": set(nets), "side": side,
            }
        else:
            e["admitted"].update(admitted)
            e["nets"].update(nets)
            e["pins"].update(pins)

    # A decap's *body* sits at the slot; the pins it can bridge are those within
    # roughly one pitch of the slot. So a slot's nets/pins are ALL nearby anchor
    # pads, not just the pad it was generated outward from -- this makes a slot
    # next to a power/ground pin PAIR carry both nets, so a decap that shares
    # both is preferred there and lands straddling the pair (pin-locality), not
    # next to a lone GND pad with its power pad left dangling.
    reach = pitch
    for a in anchors:
        a_bc = _bc(a)
        a_pads = [(pad.pos, pad.net, pad.pad_id) for pad in a.pads if pad.net]
        for pad in a.pads:
            if not pad.net:
                continue
            vx, vy = pad.pos.x - a_bc.x, pad.pos.y - a_bc.y
            if abs(vx) >= abs(vy):
                nx, ny, side = (1.0 if vx >= 0 else -1.0), 0.0, ("E" if vx >= 0 else "W")
                lat = (0.0, 1.0)  # spread vertically along the E/W edge
            else:
                nx, ny, side = 0.0, (1.0 if vy >= 0 else -1.0), ("N" if vy >= 0 else "S")
                lat = (1.0, 0.0)  # spread horizontally along the N/S edge
            admitted = _admitted(side, orientation_policy)
            for k in range(rings):
                out = base + k * pitch
                for j in range(-lateral, lateral + 1):
                    spos = Point(
                        _snap(pad.pos.x + nx * out + lat[0] * j * pitch, grid_snap),
                        _snap(pad.pos.y + ny * out + lat[1] * j * pitch, grid_snap),
                    )
                    near = [(pn, pid) for (pd, pn, pid) in a_pads
                            if _dist(pd, spos) <= reach] or [(pad.net, pad.pad_id)]
                    _add(spos, admitted, a.ref,
                         [(a.ref, pid) for (_pn, pid) in near],
                         frozenset(pn for (pn, _pid) in near), side)

    # Anchor-less arrays (no IC in the leaf): a straight lane per chain-ordered
    # cluster. Order emerges from the score (net_distance), not an imposed row.
    if not anchors:
        admitted = _admitted("lane", orientation_policy)
        for g in assign_passive_groups(parts_from_components(placed_comps)):
            members = [placed_comps[r] for r in g.passive_refs if r in placed_comps]
            if len(members) < 2:
                continue
            cx = sum(_bc(m).x for m in members) / len(members)
            cy = sum(_bc(m).y for m in members) / len(members)
            nets = frozenset(
                p.net for m in members for p in m.pads if p.net
            )
            lane_len = min(max_slots, max(len(members) + 2,
                                          int(math.ceil(len(members) * min(overprovision, 4.0)))))
            start = cx - (lane_len - 1) * pitch / 2.0
            for i in range(lane_len):
                _add(Point(_snap(start + i * pitch, grid_snap), _snap(cy, grid_snap)),
                     admitted, g.anchor_ref, [], nets, "lane")

    # Materialize, bounded and deterministic (sorted by position). Enforce a
    # minimum inter-slot separation of the passive courtyard extent so NO two
    # slots overlap -- since a courtyard AABB already includes its clearance
    # margin, touching courtyards are still clearance-legal. This makes any
    # *simultaneous* occupancy overlap-free (two decaps can't be assigned to
    # colliding slots), so the final re-snap can never re-introduce a
    # courtyard-overlap DRC failure. Grid/lane slots are pitch(=extent+gap)
    # apart and survive; only sub-extent cross-pad duplicates are dropped.
    grid = Grid()
    accepted: list[Point] = []
    sep = long_extent - 1e-6  # square courtyards overlap iff |dx|<ext AND |dy|<ext
    for key in sorted(cand.keys()):
        if len(grid.slots) >= max_slots:
            break
        e = cand[key]
        pos = e["pos"]
        if any(abs(pos.x - p.x) < sep and abs(pos.y - p.y) < sep for p in accepted):
            continue
        accepted.append(pos)
        sid = len(grid.slots)
        slot = Slot(
            sid=sid,
            pos=e["pos"],
            admitted_rotations=tuple(sorted(e["admitted"])),
            anchor_ref=e["anchor"],
            near_pins=tuple(sorted(e["pins"])),
            nets=frozenset(e["nets"]),
            side=e["side"],
        )
        grid.slots.append(slot)
        grid.free.add(sid)
        for net in slot.nets:
            grid.by_net.setdefault(net, []).append(sid)
    return grid


# --------------------------------------------------------------------------- #
# Assignment.
# --------------------------------------------------------------------------- #


def _occupy(comps: dict[str, Component], grid: Grid, ref: str, sid: int, rot: float) -> None:
    slot = grid.slots[sid]
    grid.free.discard(sid)
    slot.occupant = ref
    grid.occupied_by_ref[ref] = sid
    grid.rotation_by_ref[ref] = rot
    _place_component(comps[ref], slot.pos, rot)


def assign_initial(comps: dict[str, Component], grid: Grid) -> None:
    """Greedy deterministic seed: each passive -> nearest free slot sharing a
    net (best-first by shared-net count then distance), leftovers -> nearest
    free. Sets each occupant's pose."""
    passives = sorted(
        c.ref for c in _gridable_passives(comps)
    )
    for ref in passives:
        if not grid.free:
            break
        c = comps[ref]
        pnets = frozenset(p.net for p in c.pads if p.net)
        bc = _bc(c)
        best_key = None
        best_sid = None
        for sid in sorted(grid.free):
            slot = grid.slots[sid]
            key = (len(pnets & slot.nets), -_dist(bc, slot.pos))
            if best_key is None or key > best_key:
                best_key = key
                best_sid = sid
        if best_sid is None:
            continue
        rot = _pick_rotation(grid.slots[best_sid].admitted_rotations, c.rotation)
        _occupy(comps, grid, ref, best_sid, rot)


def _swap(comps: dict[str, Component], grid: Grid, a: str, b: str) -> None:
    sa, sb = grid.occupied_by_ref[a], grid.occupied_by_ref[b]
    ra, rb = grid.rotation_by_ref[a], grid.rotation_by_ref[b]
    grid.occupied_by_ref[a], grid.rotation_by_ref[a] = sb, rb
    grid.occupied_by_ref[b], grid.rotation_by_ref[b] = sa, ra
    grid.slots[sb].occupant, grid.slots[sa].occupant = a, b
    _place_component(comps[a], grid.slots[sb].pos, rb)
    _place_component(comps[b], grid.slots[sa].pos, ra)


def _move_to(comps: dict[str, Component], grid: Grid, ref: str, dest: int, rot: float) -> None:
    src = grid.occupied_by_ref[ref]
    grid.slots[src].occupant = None
    grid.free.add(src)
    grid.free.discard(dest)
    grid.slots[dest].occupant = ref
    grid.occupied_by_ref[ref] = dest
    grid.rotation_by_ref[ref] = rot
    _place_component(comps[ref], grid.slots[dest].pos, rot)


def _rerotate(comps: dict[str, Component], grid: Grid, ref: str, rot: float) -> None:
    grid.rotation_by_ref[ref] = rot
    _place_component(comps[ref], grid.slots[grid.occupied_by_ref[ref]].pos, rot)


def _pick_free_slot(comps: dict[str, Component], grid: Grid, ref: str, rng) -> Optional[int]:
    pnets = {p.net for p in comps[ref].pads if p.net}
    matching = sorted(sid for sid in grid.free if grid.slots[sid].nets & pnets)
    pool = matching or sorted(grid.free)
    return rng.choice(pool) if pool else None


def grid_assignment_sa(
    comps: dict[str, Component],
    grid: Grid,
    work_state,
    scorer,
    *,
    rng,
    max_iters: int = 300,
    init_temp: float = 5.0,
    cooling_rate: float = 0.995,
    swap_prob: float = 0.4,
    move_prob: float = 0.4,
    no_improve_break: int = 150,
) -> dict[str, Component]:
    """Metropolis search over slot occupancy. Returns the best components; on
    return ``grid`` is restored to the best assignment (for a final re-snap)."""
    assign_initial(comps, grid)
    if not grid.occupied_by_ref:
        return {r: copy.deepcopy(c) for r, c in comps.items()}

    work_state.components = comps
    current = scorer.score().total
    best = current
    best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}
    best_assign = dict(grid.occupied_by_ref)
    best_rot = dict(grid.rotation_by_ref)

    temp = init_temp
    temp_floor = init_temp * 0.001
    since_improve = 0

    for _ in range(max_iters):
        occupied = sorted(grid.occupied_by_ref.keys())
        roll = rng.random()
        undo = None

        if roll < swap_prob and len(occupied) >= 2:
            a, b = rng.sample(occupied, 2)
            _swap(comps, grid, a, b)
            undo = lambda a=a, b=b: _swap(comps, grid, a, b)  # swap is its own inverse
        elif roll < swap_prob + move_prob and grid.free:
            a = rng.choice(occupied)
            src, old_rot = grid.occupied_by_ref[a], grid.rotation_by_ref[a]
            dest = _pick_free_slot(comps, grid, a, rng)
            if dest is None:
                continue
            new_rot = _pick_rotation(grid.slots[dest].admitted_rotations, old_rot)
            _move_to(comps, grid, a, dest, new_rot)
            undo = lambda a=a, src=src, old_rot=old_rot: _move_to(comps, grid, a, src, old_rot)
        else:
            a = rng.choice(occupied)
            admitted = grid.slots[grid.occupied_by_ref[a]].admitted_rotations
            old_rot = grid.rotation_by_ref[a]
            cands = [r for r in admitted if r != old_rot]
            if not cands:
                continue
            _rerotate(comps, grid, a, rng.choice(cands))
            undo = lambda a=a, old_rot=old_rot: _rerotate(comps, grid, a, old_rot)

        work_state.components = comps
        new_score = scorer.score().total
        delta = new_score - current
        if delta > 0 or rng.random() < math.exp(delta / max(temp, 0.001)):
            current = new_score
            if new_score > best:
                best = new_score
                best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}
                best_assign = dict(grid.occupied_by_ref)
                best_rot = dict(grid.rotation_by_ref)
                since_improve = 0
            else:
                since_improve += 1
        else:
            undo()
            since_improve += 1

        temp *= cooling_rate
        if since_improve >= no_improve_break or temp < temp_floor:
            break

    # Restore grid bookkeeping to the best assignment so a downstream re-snap
    # (after the legality tail) can pin any drifted passive back to its slot.
    for slot in grid.slots:
        slot.occupant = None
    grid.free = set(range(len(grid.slots)))
    grid.occupied_by_ref = {}
    grid.rotation_by_ref = {}
    for ref, sid in best_assign.items():
        grid.free.discard(sid)
        grid.slots[sid].occupant = ref
        grid.occupied_by_ref[ref] = sid
        grid.rotation_by_ref[ref] = best_rot[ref]
    return best_comps


def resnap_to_grid(comps: dict[str, Component], grid: Grid, *, tol_mm: float = 0.05) -> int:
    """Re-place any gridded passive that the legality tail nudged off its slot.
    Idempotent; returns how many were snapped back."""
    n = 0
    for ref, sid in grid.occupied_by_ref.items():
        c = comps.get(ref)
        if c is None:
            continue
        slot = grid.slots[sid]
        rot = grid.rotation_by_ref.get(ref, c.rotation)
        if _dist(_bc(c), slot.pos) > tol_mm or (c.rotation - rot) % 360.0 > 0.5:
            _place_component(c, slot.pos, rot)
            n += 1
    return n


__all__ = [
    "Slot",
    "Grid",
    "build_anchor_grid",
    "assign_initial",
    "grid_assignment_sa",
    "resnap_to_grid",
]
