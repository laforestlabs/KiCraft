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

Two operations:

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
from typing import Any, Optional

from .geometry import rotate_component_in_place
from .leaf_tidiness import (
    DEFAULT_PLANE_NETS,
    assign_passive_groups,
    is_pin_anchor,
    median_pin_distance_mm,
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
    ring: int = 0  # 0 = tight to the pin; higher = one pitch further out
    occupant: Optional[str] = None


@dataclass
class Grid:
    """The slot set plus its live occupancy bookkeeping."""

    slots: list[Slot] = field(default_factory=list)
    by_net: dict[str, list[int]] = field(default_factory=dict)
    free: set[int] = field(default_factory=set)
    occupied_by_ref: dict[str, int] = field(default_factory=dict)
    rotation_by_ref: dict[str, float] = field(default_factory=dict)
    # Build + guard telemetry, surfaced into the leaf round record. Slot
    # starvation and the silent accept-if-better revert were both invisible
    # before this (dense-soc-leaf-unconnected-plan P0.2/P0.4).
    stats: dict[str, Any] = field(default_factory=dict)

    def child_refs(self) -> set[str]:
        """Passives that move only via the grid (SA-assignment owns them)."""
        return set(self.occupied_by_ref)


# --------------------------------------------------------------------------- #
# Small geometry helpers (self-contained).
# --------------------------------------------------------------------------- #


def _bc(c: Component) -> Point:
    return c.body_center if c.body_center is not None else c.pos


def _snap(v: float, grid: float) -> float:
    return round(v / grid) * grid if grid > 0 else v


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _round_mm(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 2)


def _place_component(comp: Component, target_bc: Point, target_rot: float) -> None:
    """Rotate ``comp`` to ``target_rot`` then translate its body_center to
    ``target_bc`` (pads kept in sync). Mutates in place."""
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


def _inside(pos: Point, half: float, tl: Point, br: Point, inset: float,
            half_y: Optional[float] = None) -> bool:
    hy = half if half_y is None else half_y
    return (
        tl.x + inset + half <= pos.x <= br.x - inset - half
        and tl.y + inset + hy <= pos.y <= br.y - inset - hy
    )


def _overlaps_rect(pos: Point, half: float, rtl: Point, rbr: Point,
                   half_y: Optional[float] = None) -> bool:
    hy = half if half_y is None else half_y
    return (
        pos.x - half < rbr.x and pos.x + half > rtl.x
        and pos.y - hy < rbr.y and pos.y + hy > rtl.y
    )


def _slot_extent(
    admitted: tuple[float, ...], long_ext: float, short_ext: float
) -> tuple[float, float]:
    """Footprint (w, h) a slot's occupant will actually have.

    A slot admits one orientation *axis* under the default policy, so its
    occupant's real footprint is known: a decap on a N/S package edge lies
    horizontal (long x short), on an E/W edge vertical. Treating every slot as a
    square of the LONG side -- which the build did -- halved the achievable ring
    density and pushed ring 1 a full half-long-side off the pad, which is a
    third of the reason a decap could not reach its pins. Mixed admitted sets
    (two merged slots) stay square: conservative, always legal.
    """
    axes = {orientation_axis(r) for r in admitted}
    if axes == {"H"}:
        return (long_ext, short_ext)
    if axes == {"V"}:
        return (short_ext, long_ext)
    return (long_ext, long_ext)


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
    min_provision: float = 3.0,
    max_rings: int = 6,
    max_lateral: int = 3,
) -> Grid:
    """Build the anchor-relative slot grid from the *placed* anchors.

    Pin-adjacent slots step outward (``rings``) from each anchor pad along the
    outward cardinal normal, with a lateral spread (``lateral``) so same-side
    pads share a lane. Pitch = widest passive extent + ``pitch_gap_mm``, a
    *courtyard-legal* gap (~0.3-0.6 mm), NOT the leaf placement clearance: the
    courtyard AABB already carries its clearance margin, so touching courtyards
    are legal and a 2.84 mm blanket only pushed the rings 6-12 mm off the
    package, where a decap cannot be. Slots overlapping an anchor courtyard, a
    keep-out, or off the board are culled. An anchor-less passive array (no IC in
    the leaf) instead gets one straight lane per chain-ordered cluster.

    ``min_provision`` is the *honored* over-provisioning target: slots per
    gridable passive. When the base (rings, lateral) under-provisions, the spread
    grows -- laterally first (stays pin-adjacent), then outward -- until the
    target is met, the caps are reached, or the geometry stops yielding new legal
    slots. Starvation at ~1.1x was the assignment's binding constraint
    (dense-soc-leaf-unconnected-plan P0.1/P0.2).
    """
    keepout_rects = keepout_rects or []
    tl, br = board_outline
    passives = _gridable_passives(placed_comps)
    if not passives:
        return Grid()

    long_extent = max(max(c.width_mm, c.height_mm) for c in passives)
    short_extent = max(min(c.width_mm, c.height_mm) for c in passives)
    half = long_extent / 2.0
    gap = max(0.0, pitch_gap_mm)
    pitch = long_extent + gap  # lateral (along-edge) step + lane sep
    step = short_extent + gap  # outward (ring) step: the part's SHORT side
    # ring-1 sits TIGHT to the pins (a decap belongs ~1 mm away, not a full
    # clearance out); the up-front cull drops any ring-1 that would overlap the
    # anchor courtyard, so the closest surviving ring is as pin-adjacent as
    # legality allows. Decoupling from the ring step is what gets pin_mm to ~1-2.
    base = short_extent / 2.0 + min(gap, 0.6)

    anchors = [
        c for c in placed_comps.values()
        if is_pin_anchor(c.kind or "", sum(1 for p in c.pads if p.net))
    ]

    # A slot is illegal if it overlaps ANY fixed part the grid will not move --
    # not just the pin-anchor ICs/regulators/connectors. Inductors, LEDs, diodes,
    # crystals, locked parts and array members are all rigid obstacles here; culling
    # only _ANCHOR_KINDS let a passive slot land on top of e.g. an inductor or LED,
    # producing exactly the 'R1:L1' / 'LED1:C2' courtyard overlaps that are
    # unrepairable downstream (WS1). Obstacles = every placed component that is not
    # a gridable passive (those move via the slots and are kept apart by `sep`).
    gridable_refs = {c.ref for c in passives}
    obstacle_rects = [
        (Point(_bc(c).x - c.width_mm / 2, _bc(c).y - c.height_mm / 2),
         Point(_bc(c).x + c.width_mm / 2, _bc(c).y + c.height_mm / 2))
        for c in placed_comps.values()
        if c.ref not in gridable_refs
    ]

    def _legal(pos: Point, hw: float, hh: float) -> bool:
        if not _inside(pos, hw, tl, br, pad_inset_mm, hh):
            return False
        for rtl, rbr in obstacle_rects:
            if _overlaps_rect(pos, hw, rtl, rbr, hh):
                return False
        for rtl, rbr in keepout_rects:
            if _overlaps_rect(pos, hw, rtl, rbr, hh):
                return False
        return True

    # A decap's *body* sits at the slot; the pins it can bridge are those within
    # roughly one pitch of the slot. So a slot's nets/pins are ALL nearby anchor
    # pads, not just the pad it was generated outward from -- this makes a slot
    # next to a power/ground pin PAIR carry both nets, so a decap that shares
    # both is preferred there and lands straddling the pair (pin-locality), not
    # next to a lone GND pad with its power pad left dangling.
    reach = pitch + short_extent

    def _generate(n_rings: int, n_lateral: int) -> dict[tuple[float, float], dict]:
        """Candidate slots keyed by snapped position, so slots between two pins
        merge -- carrying BOTH nets (and both admitted-rotation sets)."""
        cand: dict[tuple[float, float], dict] = {}

        def _add(pos: Point, admitted: tuple[float, ...], anchor_ref: str,
                 pins: list[tuple[str, str]], nets: frozenset[str], side: str,
                 ring: int) -> None:
            ew, eh = _slot_extent(admitted, long_extent, short_extent)
            if not _legal(pos, ew / 2.0, eh / 2.0):
                return
            key = (round(pos.x, 3), round(pos.y, 3))
            e = cand.get(key)
            if e is None:
                cand[key] = {
                    "pos": pos, "admitted": set(admitted), "anchor": anchor_ref,
                    "pins": set(pins), "nets": set(nets), "side": side,
                    "ring": ring,
                }
            else:
                e["admitted"].update(admitted)
                e["nets"].update(nets)
                e["pins"].update(pins)
                e["ring"] = min(e["ring"], ring)

        for a in anchors:
            a_bc = _bc(a)
            a_pads = [(pad.pos, pad.net, pad.pad_id) for pad in a.pads if pad.net]
            for pad in a.pads:
                if not pad.net:
                    continue
                p_pos = pad.pos
                vx, vy = p_pos.x - a_bc.x, p_pos.y - a_bc.y
                if abs(vx) >= abs(vy):
                    nx, ny, side = (1.0 if vx >= 0 else -1.0), 0.0, ("E" if vx >= 0 else "W")
                    lat = (0.0, 1.0)  # spread vertically along the E/W edge
                else:
                    nx, ny, side = 0.0, (1.0 if vy >= 0 else -1.0), ("N" if vy >= 0 else "S")
                    lat = (1.0, 0.0)  # spread horizontally along the N/S edge
                admitted = _admitted(side, orientation_policy)
                # Outward steps by the occupant's SHORT side (rings of caps
                # stack tight against the package edge); lateral steps by its
                # LONG side (they sit end-to-end along the edge).
                ew, eh = _slot_extent(admitted, long_extent, short_extent)
                out_step = (ew if abs(nx) > abs(ny) else eh) + gap
                lat_step = (ew if lat[0] else eh) + gap
                out_base = (ew if abs(nx) > abs(ny) else eh) / 2.0 + min(gap, 0.6)
                for k in range(n_rings):
                    out = out_base + k * out_step
                    for j in range(-n_lateral, n_lateral + 1):
                        spos = Point(
                            _snap(p_pos.x + nx * out + lat[0] * j * lat_step, grid_snap),
                            _snap(p_pos.y + ny * out + lat[1] * j * lat_step, grid_snap),
                        )
                        near = [(pn, pid) for (pd, pn, pid) in a_pads
                                if _dist(pd, spos) <= reach] or [(pad.net, pad.pad_id)]
                        _add(spos, admitted, a.ref,
                             [(a.ref, pid) for (_pn, pid) in near],
                             frozenset(pn for (pn, _pid) in near), side,
                             k + abs(j))

        # Anchor-less arrays (no IC in the leaf): a straight lane per
        # chain-ordered cluster. Order emerges from the score (net_distance),
        # not an imposed row.
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
                         admitted, g.anchor_ref, [], nets, "lane", 0)
        return cand

    def _materialize(cand: dict[tuple[float, float], dict]) -> Grid:
        """Bounded, deterministic slot list. Enforces an inter-slot separation of
        the two occupants' courtyard extents so NO two slots overlap -- since a
        courtyard AABB already includes its clearance margin, touching courtyards
        are still clearance-legal. This makes any *simultaneous* occupancy
        overlap-free (two decaps can't be assigned to colliding slots), so the
        final re-snap can never re-introduce a courtyard-overlap DRC failure.
        Ordered by ring first, so when ``max_slots`` bites it drops the FARTHEST
        slots -- position-sorted truncation kept one spatial corner of the board
        and starved every other anchor.
        """
        grid = Grid()
        accepted: list[tuple[Point, float, float]] = []
        order = sorted(cand.keys(), key=lambda k: (cand[k]["ring"], k))
        for key in order:
            if len(grid.slots) >= max_slots:
                break
            e = cand[key]
            pos = e["pos"]
            ew, eh = _slot_extent(tuple(sorted(e["admitted"])), long_extent, short_extent)
            if any(
                abs(pos.x - p.x) < (ew + pw) / 2.0 - 1e-6
                and abs(pos.y - p.y) < (eh + ph) / 2.0 - 1e-6
                for p, pw, ph in accepted
            ):
                continue
            accepted.append((pos, ew, eh))
            sid = len(grid.slots)
            slot = Slot(
                sid=sid,
                pos=e["pos"],
                admitted_rotations=tuple(sorted(e["admitted"])),
                anchor_ref=e["anchor"],
                near_pins=tuple(sorted(e["pins"])),
                nets=frozenset(e["nets"]),
                side=e["side"],
                ring=int(e["ring"]),
            )
            grid.slots.append(slot)
            grid.free.add(sid)
            for net in slot.nets:
                grid.by_net.setdefault(net, []).append(sid)
        return grid

    # Grow until the over-provisioning target is met: laterally first (a slot one
    # pitch along the same package edge is still pin-adjacent), then outward.
    target_slots = int(math.ceil(max(1.0, min_provision) * len(passives)))
    n_rings = max(1, rings)
    n_lateral = max(0, lateral)
    grid = _materialize(_generate(n_rings, n_lateral))
    grew: list[str] = []
    while len(grid.slots) < target_slots and (
        n_lateral < max_lateral or n_rings < max_rings
    ):
        before = len(grid.slots)
        if n_lateral < max_lateral:
            n_lateral += 1
        else:
            n_rings += 1
        grid = _materialize(_generate(n_rings, n_lateral))
        grew.append(f"r{n_rings}l{n_lateral}={len(grid.slots)}")
        if len(grid.slots) <= before and n_lateral >= max_lateral:
            break  # geometry is saturated; more rings only add distance

    per_anchor: dict[str, int] = {}
    for slot in grid.slots:
        per_anchor[slot.anchor_ref] = per_anchor.get(slot.anchor_ref, 0) + 1
    passive_nets = {p.net for c in passives for p in c.pads if p.net}
    coverage = sorted(len(grid.by_net.get(n, [])) for n in sorted(passive_nets))
    grid.stats = {
        "gridable_passives": len(passives),
        "anchors": sorted(a.ref for a in anchors),
        "slots_total": len(grid.slots),
        "slots_per_anchor": {k: per_anchor[k] for k in sorted(per_anchor)},
        "provisioning_ratio": round(len(grid.slots) / max(1, len(passives)), 2),
        "target_provisioning": round(float(min_provision), 2),
        "rings": n_rings,
        "lateral": n_lateral,
        "growth": grew,
        # Along-edge step (occupant's long side) vs outward ring step (its short
        # side); ring 1 sits half a short side + the gap off the pad.
        "lateral_pitch_mm": round(pitch, 3),
        "ring_step_mm": round(step, 3),
        "ring1_offset_mm": round(base, 3),
        "slots_capped": len(grid.slots) >= max_slots,
        "net_coverage_min": coverage[0] if coverage else 0,
        "net_coverage_median": coverage[len(coverage) // 2] if coverage else 0,
    }
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


def _match_strength(pnets: frozenset[str], slot: Slot, plane_nets) -> tuple[int, int]:
    """How well a passive's nets fit a slot: ``(point_net_hits, any_net_hits)``.

    A pad on a POURED net reaches its plane through a via wherever the part
    lands, so a slot next to a GND pad is worth far less than one next to the
    specific power/signal pin the part must hug -- the same rule the
    pin-locality kernel scores by. Counting plane hits equally let every decap
    "match" any of the dozens of GND-adjacent slots and be seated by raw
    distance instead of by the pin it exists to bridge.
    """
    point = len((pnets & slot.nets) - plane_nets)
    return (point, len(pnets & slot.nets))


def assign_initial(
    comps: dict[str, Component],
    grid: Grid,
    plane_nets: frozenset[str] = DEFAULT_PLANE_NETS,
) -> None:
    """Greedy deterministic seed: each passive -> nearest free slot sharing a
    net (best-first by point-net match, then any-net match, then distance),
    leftovers -> nearest free. Sets each occupant's pose.

    Passives with the scarcest matching slots are seated FIRST: alphabetical
    order let C1 (a plain VDD/GND decap with dozens of candidate slots) take the
    one slot a crystal load cap or a DEC-pin decap could ever have used.
    """
    passives = sorted(c.ref for c in _gridable_passives(comps))
    nets_by_ref = {
        ref: frozenset(p.net for p in comps[ref].pads if p.net) for ref in passives
    }
    options = {
        ref: sum(
            1 for slot in grid.slots
            if _match_strength(nets_by_ref[ref], slot, plane_nets)[0]
        )
        for ref in passives
    }
    for ref in sorted(passives, key=lambda r: (options[r], r)):
        if not grid.free:
            break
        c = comps[ref]
        pnets = nets_by_ref[ref]
        bc = _bc(c)
        best_key = None
        best_sid = None
        for sid in sorted(grid.free):
            slot = grid.slots[sid]
            key = (*_match_strength(pnets, slot, plane_nets), -_dist(bc, slot.pos))
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


def _pick_free_slot(
    comps: dict[str, Component],
    grid: Grid,
    ref: str,
    rng,
    plane_nets: frozenset[str] = DEFAULT_PLANE_NETS,
) -> Optional[int]:
    """A free slot to try for ``ref``: prefer one carrying the specific pin it
    must hug (point net), then any shared net, then anywhere."""
    pnets = frozenset(p.net for p in comps[ref].pads if p.net)
    point = sorted(
        sid for sid in grid.free
        if _match_strength(pnets, grid.slots[sid], plane_nets)[0]
    )
    pool = point or sorted(
        sid for sid in grid.free if grid.slots[sid].nets & pnets
    ) or sorted(grid.free)
    return rng.choice(pool) if pool else None


def _median_pin_mm(comps: dict[str, Component], plane_nets) -> Optional[float]:
    return median_pin_distance_mm(
        parts_from_components(comps), plane_nets=plane_nets
    )


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
    pin_floor_tol_mm: float = 0.5,
    pin_floor_score_slack: float = 8.0,
) -> dict[str, Component]:
    """Metropolis search over slot occupancy. Returns the best components; on
    return ``grid`` is restored to the best assignment (for a final re-snap).

    Accept-if-better **with a pin-locality floor**. The connectivity-first grid
    can degrade a decent force-loop placement (buck-3a replay: 65.8 -> 43.4 with
    +18 crossovers), so a total-score collapse still discards the assignment.
    But total score is a weighted average in which pin-locality holds ~18% of
    the vote, so an assignment that puts every decap on its pins could *lose* on
    compactness/crossings and be silently reverted -- that revert is what left
    the dense-SoC leaf with 30 mm decap hauls (plan P0.4). So:

    * materially better adjacency (median pad->pin at least ``pin_floor_tol_mm``
      shorter) wins unless the total score collapses by more than
      ``pin_floor_score_slack``;
    * otherwise the classic accept-if-better applies, except a candidate with
      materially WORSE adjacency can no longer win on compactness alone.

    Either way the decision and both metrics land in ``grid.stats`` -- the
    revert used to be entirely silent.
    """
    plane_nets = DEFAULT_PLANE_NETS
    getter = getattr(scorer, "_pin_locality_plane_nets", None)
    if callable(getter):
        try:
            plane_nets = getter()
        except Exception:  # a scorer stub without live pour config
            plane_nets = DEFAULT_PLANE_NETS

    # Snapshot + score the INPUT placement before assign_initial re-grids everyone.
    input_comps = {r: copy.deepcopy(c) for r, c in comps.items()}
    work_state.components = input_comps
    input_score = scorer.score().total
    input_pin_mm = _median_pin_mm(input_comps, plane_nets)

    assign_initial(comps, grid, plane_nets)
    if not grid.occupied_by_ref:
        grid.stats.update({
            "grid_discarded": True,
            "guard": "no_occupants",
            "input_score": round(input_score, 2),
            "input_pin_median_mm": _round_mm(input_pin_mm),
        })
        return input_comps

    work_state.components = comps
    init_pin_mm = _median_pin_mm(comps, plane_nets)
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
            dest = _pick_free_slot(comps, grid, a, rng, plane_nets)
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

    best_pin_mm = _median_pin_mm(best_comps, plane_nets)
    if input_pin_mm is None or best_pin_mm is None:
        pins_better = pins_worse = False
    else:
        pins_better = best_pin_mm <= input_pin_mm - pin_floor_tol_mm
        pins_worse = best_pin_mm >= input_pin_mm + pin_floor_tol_mm

    if pins_better and best >= input_score - pin_floor_score_slack:
        keep, verdict = True, "accept_pin_locality"
    elif best > input_score and not pins_worse:
        keep, verdict = True, "accept_score"
    elif best > input_score:
        keep, verdict = False, "discard_pin_locality_floor"
    else:
        keep, verdict = False, "discard_score"

    grid.stats.update({
        "grid_discarded": not keep,
        "guard": verdict,
        "input_score": round(input_score, 2),
        "grid_score": round(best, 2),
        "input_pin_median_mm": _round_mm(input_pin_mm),
        "init_pin_median_mm": _round_mm(init_pin_mm),
        "grid_pin_median_mm": _round_mm(best_pin_mm),
        "seated": len(best_assign),
    })

    # Guard fired: keep the input placement verbatim and neutralize the grid so
    # the caller's resnap_to_grid finds no occupants and leaves it untouched.
    if not keep:
        for slot in grid.slots:
            slot.occupant = None
        grid.free = set(range(len(grid.slots)))
        grid.occupied_by_ref = {}
        grid.rotation_by_ref = {}
        return input_comps

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


def resnap_to_grid(
    comps: dict[str, Component],
    grid: Grid,
    *,
    tol_mm: float = 0.05,
    exclude: set[str] | None = None,
) -> int:
    """Re-place any gridded passive that the legality tail nudged off its slot.
    Idempotent; returns how many were snapped back.

    ``exclude`` names occupants that must NOT be snapped back -- occupants the
    final courtyard-separation pass (Step 16) deliberately moved to clear an
    overlap. Snapping those back to their slot silently reinstated the exact
    overlap Step 16 had just fixed, which then shipped frozen in the leaf and
    resurfaced as courtyards_overlap at the parent fab gate (2026-07-19 §3.1).
    """
    n = 0
    for ref, sid in grid.occupied_by_ref.items():
        if exclude and ref in exclude:
            continue
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
