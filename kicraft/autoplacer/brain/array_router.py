"""Deterministic router for arrays of identical components (LED matrices, etc.).

A dense, regular array -- e.g. a 5x10 grid of 1515 WS2812 LEDs at 3 mm pitch,
daisy-chained DOUT -> DIN -- is the one case where handing the whole leaf to a
gridless autorouter is the wrong tool. FreeRouting routes the array's data chain
net by net and, in the ~1.5 mm channels between neighbouring LEDs (which also
carry power), abandons a handful of hops -- leaving the leaf with unconnected
signal nets that fail the strict ``no_unconnected`` gate, a *different* few hops
failing on every run (a congestion/nondeterminism signature, not a fixed bug).

But the chain is trivial geometry: ``ArraySpec.refs`` are listed in data-chain
order and ``place_array_leaves`` fills the grid (serpentine) in that order, so
every hop is between two physically adjacent members. This module emits one
short, locked pad-to-pad tie per hop, reusing the proven ``breakout_stubs``
stamping primitive and its foreign-pad / copper guards. A subsequent FreeRouting
pass run with ``freerouting_preserve_existing_copper=True`` keeps those ties and
only has to route what's left (the chain entry through the data resistor, the
header), while the +5V / GND pours deliver power so the inter-LED channels carry
only data.

Returns ``BreakoutSpec`` ties so the caller appends them to the leaf's breakout
specs and stamps them in the existing pass -- no new board write path.
"""
from __future__ import annotations

import math
from typing import Any

import pcbnew

from kicraft.autoplacer.brain.breakout_stubs import (
    BreakoutSpec,
    _board_inner_box_mm,
)
from kicraft.design.models import GND_NET_PATTERNS, is_power_or_ground_name

_FAR = 1 << 30


def _is_ground_name(name: str) -> bool:
    stripped = (name or "").lstrip("/")
    return any(pat.search(stripped) for pat in GND_NET_PATTERNS)


def _array_member_order(
    arrays: list[dict] | None, present: set[str]
) -> dict[str, int]:
    """ref -> data-chain index for every member of a fully-present array.

    Only arrays whose entire ref list is on this leaf are honoured (a partial
    array means the grid placement did not run, so the hop geometry is not
    guaranteed). The index is the position in ``ArraySpec.refs`` -- the order
    ``place_array_leaves`` fills the serpentine grid in, so consecutive indices
    are physical neighbours.
    """
    order: dict[str, int] = {}
    for spec in arrays or []:
        refs = list(spec.get("refs", []))
        if refs and all(r in present for r in refs):
            for i, r in enumerate(refs):
                order.setdefault(r, i)
    return order


def _array_cols_by_ref(
    arrays: list[dict] | None, present: set[str]
) -> dict[str, int]:
    """ref -> column count of its (fully-present) array, for turn detection."""
    cols: dict[str, int] = {}
    for spec in arrays or []:
        refs = list(spec.get("refs", []))
        c = int(spec.get("cols", 0))
        if refs and c > 0 and all(r in present for r in refs):
            for r in refs:
                cols.setdefault(r, c)
    return cols


def _fp_half_width_mm(fp) -> float:
    """Half the span from the footprint centre to its outermost pad, plus a pad
    half-size margin -- how far past the body a turn channel must sit to clear
    this member's foreign pads."""
    cx = pcbnew.ToMM(fp.GetPosition().x)
    spans = [abs(pcbnew.ToMM(p.GetPosition().x) - cx) for p in fp.Pads()]
    return (max(spans) if spans else 0.4) + 0.35


def _turn_hop_spec(
    fp_a, fp_b, ref_a, pad_a, src_mm, tgt_mm, inner_box, layer, width, cfg,
) -> "BreakoutSpec | None":
    """An L/Z edge-channel tie for a serpentine row-turn hop, or ``None`` when
    the board is too tight to route one (caller then leaves it to FreeRouting).

    The turn members sit in the same row-end column against a board edge, so a
    straight DOUT->DIN tie would graze the LEDs' own +5V/GND pads and the
    foreign-pad guard drops it. Instead route OUT to a channel just past that
    column toward the nearer board edge, ALONG the channel to the target row,
    then back IN -- a path that clears every foreign pad.
    """
    if inner_box is None:
        return None  # no outline: off-board locked copper hangs FreeRouting 1.9.0
    bx1, by1, bx2, by2 = inner_box
    sx, sy = src_mm
    tx, ty = tgt_mm
    # row-end column centre (both turn members share it)
    col_x = (pcbnew.ToMM(fp_a.GetPosition().x) + pcbnew.ToMM(fp_b.GetPosition().x)) / 2.0
    # The channel must clear not just the row-end LEDs' bodies but their GND/power
    # pad ESCAPE stubs, which reach ~1 mm into the edge margin -- hence a full-mm
    # default gap, not a hair past the courtyard (a tighter channel collides with
    # those escapes and the hop falls back to FreeRouting).
    gap = float(cfg.get("array_turn_channel_gap_mm", 1.0))
    offset = max(_fp_half_width_mm(fp_a), _fp_half_width_mm(fp_b)) + gap
    mid_x = (bx1 + bx2) / 2.0
    if col_x >= mid_x:  # row end against the RIGHT edge -> channel to the right
        channel_x = min(col_x + offset, bx2)
        if channel_x <= max(sx, tx):
            return None  # no room to clear the pads -> hand to FreeRouting
    else:  # against the LEFT edge -> channel to the left
        channel_x = max(col_x - offset, bx1)
        if channel_x >= min(sx, tx):
            return None
    points = [(channel_x, sy), (channel_x, ty), (tx, ty)]
    if not all(bx1 - 1e-6 <= px <= bx2 + 1e-6 and by1 - 1e-6 <= py <= by2 + 1e-6
               for px, py in points):
        return None
    return BreakoutSpec(
        ref=ref_a, pad=pad_a.GetNumber(), waypoints=points,
        width_mm=float(width) if width else None, layer=layer,
        near_xy=(sx, sy),
    )


def array_daisy_chain_specs(
    board: "pcbnew.BOARD",
    cfg: dict[str, Any] | None = None,
) -> list[BreakoutSpec]:
    """Locked pad-to-pad ties for each daisy-chain hop of every array on a leaf.

    A hop is a *signal* (non power/ground) net with exactly two pads that sit on
    two distinct array members -- which, for an addressable-LED matrix, is
    exactly each ``Dn.DOUT -> D(n+1).DIN`` link. Power/ground nets are left to
    the pours; nets touching a non-member (the chain entry through the data
    resistor, the header) are left to FreeRouting. The stamp-time guards in
    :func:`breakout_stubs.add_breakout_stubs` drop any tie whose straight path
    would cross foreign copper, so a hop the geometry doesn't allow is simply
    handed back to the autorouter -- never shorted.
    """
    cfg = cfg or {}
    if not cfg.get("array_route_enabled", True):
        return []
    arrays = cfg.get("arrays") or []
    present = {fp.GetReferenceAsString() for fp in board.GetFootprints()}
    order = _array_member_order(arrays, present)
    if not order:
        return []
    cols_by_ref = _array_cols_by_ref(arrays, present)
    fp_by_ref = {fp.GetReferenceAsString(): fp for fp in board.GetFootprints()}

    layer = cfg.get("array_data_layer", "F.Cu")
    width = cfg.get("array_data_width_mm")  # None -> add_breakout_stubs default
    turn_routing = cfg.get("array_turn_routing", True)
    inner_box = _board_inner_box_mm(board)

    # net_code -> [(ref, pad), ...] across the whole leaf.
    by_net: dict[int, list[tuple[str, Any]]] = {}
    netname: dict[int, str] = {}
    for fp in board.GetFootprints():
        ref = fp.GetReferenceAsString()
        for pad in fp.Pads():
            nc = pad.GetNetCode()
            if nc == 0:
                continue
            by_net.setdefault(nc, []).append((ref, pad))
            netname.setdefault(nc, pad.GetNetname())

    ties: list[tuple[int, BreakoutSpec]] = []
    skipped_turns = 0
    for nc, pads in by_net.items():
        if is_power_or_ground_name(netname.get(nc, "")):
            continue
        if len(pads) != 2:
            continue
        (ref_a, pad_a), (ref_b, pad_b) = pads
        if ref_a == ref_b or ref_a not in order or ref_b not in order:
            continue
        # Source from the earlier chain member so adjacent hops stamp in chain
        # order (deterministic mutual-clearance handling between neighbours).
        if order[ref_b] < order[ref_a]:
            ref_a, pad_a, ref_b, pad_b = ref_b, pad_b, ref_a, pad_a
        src = pad_a.GetPosition()
        tgt = pad_b.GetPosition()
        src_mm = (pcbnew.ToMM(src.x), pcbnew.ToMM(src.y))
        tgt_mm = (pcbnew.ToMM(tgt.x), pcbnew.ToMM(tgt.y))

        # A row-turn hop joins two members in DIFFERENT grid rows (serpentine
        # reverses the fill, so they share the row-end column against a board
        # edge). Its straight DOUT->DIN tie would graze the LEDs' own power pads
        # and get dropped -> route it round the edge channel instead so the data
        # net is 100% kicraft-stamped (no FreeRouting vias on the chain).
        cols = cols_by_ref.get(ref_a) or cols_by_ref.get(ref_b)
        is_turn = bool(cols) and (order[ref_a] // cols) != (order[ref_b] // cols)
        if is_turn and turn_routing:
            spec = _turn_hop_spec(
                fp_by_ref.get(ref_a), fp_by_ref.get(ref_b), ref_a, pad_a,
                src_mm, tgt_mm, inner_box, layer, width, cfg,
            )
            if spec is None:
                # No edge channel fits -> leave THIS hop to FreeRouting (do NOT
                # stamp a straight tie that the guard would drop anyway). Logged
                # below so the hand-off is never silent.
                skipped_turns += 1
                continue
            ties.append((order[ref_a], spec))
            continue

        # In-row hop: a short straight same-net tie across the inter-component
        # channel. The stamp-time guards in :func:`breakout_stubs.add_breakout_stubs`
        # drop any tie whose straight path would still cross a foreign pad,
        # handing just that one back to FreeRouting -- never a short. Escapes were
        # tried here and measurably HURT (locked clutter raised FreeRouting's
        # abandoned-net count); ties-only is the validated win.
        ties.append((order[ref_a], BreakoutSpec(
            ref=ref_a, pad=pad_a.GetNumber(),
            waypoints=[tgt_mm],
            width_mm=float(width) if width else None, layer=layer,
            near_xy=src_mm,
        )))

    if skipped_turns:
        print(
            f"  array-router: {skipped_turns} row-turn hop(s) left to FreeRouting "
            "(no edge channel fits inside the board outline)"
        )
    ties.sort(key=lambda item: item[0])
    return [spec for _, spec in ties]


def array_ring_power_specs(
    board: "pcbnew.BOARD",
    cfg: dict[str, Any] | None = None,
) -> list[BreakoutSpec]:
    """A deterministic CLOSED-LOOP power bus for every fully-present ring array.

    A ring's canonical power distribution is a bus at the members' power-pad
    radius: n identical short chords, member pad to member pad -- detouring
    through a band decap's power pad when one sits in the gap (see
    ``array_placement._place_ring_band_decaps``, which parks it exactly on the
    chord's sagitta radius) -- CLOSED, so one or two guard-dropped ties cannot
    disconnect the bus. Stamping it makes power deterministic and leaves
    FreeRouting no reason to dip INTO the ring interior, which shaped-compose
    nesting needs clear (docs/plans/shaped-compose-leaf-nesting.md, PR-N5).
    Each gap decap's GND pad also gets a short tangential stub with a via down
    to the B.Cu pour. Member GND ties are deliberately NOT stamped: LED GND
    pads sit on the ring's outer corner, where FreeRouting's shortest paths
    never enter the interior, and the pours own GND.

    Returned SEPARATELY from the daisy-chain data ties so the array stamp
    gate keeps measuring in-row DATA hops only; the ``add_breakout_stubs``
    foreign-pad / copper guards stay the honesty layer (a tie they drop is
    handed to FreeRouting, loudly, via the leaf_routing skipped log).
    """
    cfg = cfg or {}
    if not cfg.get("array_ring_power_bus", True):
        return []
    arrays = cfg.get("arrays") or []
    fp_by_ref = {fp.GetReferenceAsString(): fp for fp in board.GetFootprints()}
    width = float(cfg.get("power_width_mm", 0.5))
    layer = cfg.get("array_data_layer", "F.Cu")
    stub_mm = float(cfg.get("array_ring_gnd_stub_mm", 1.0))

    def _mm_xy(pos) -> tuple[float, float]:
        return (pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y))

    specs: list[BreakoutSpec] = []
    used_decaps: set[str] = set()
    for arr in arrays:
        if str(arr.get("pattern", "grid") or "grid").lower() != "ring":
            continue
        refs = list(arr.get("refs", []))
        if len(refs) < 3 or not all(r in fp_by_ref for r in refs):
            continue

        # The shared rail: a non-ground power net present on EVERY member.
        rails: set[str] | None = None
        for r in refs:
            nets = {
                p.GetNetname() for p in fp_by_ref[r].Pads()
                if p.GetNetCode()
                and is_power_or_ground_name(p.GetNetname())
                and not _is_ground_name(p.GetNetname())
            }
            rails = nets if rails is None else (rails & nets)
        if not rails:
            print(
                f"  ring-power: no common rail across {refs[0]}..{refs[-1]} "
                "-- bus skipped"
            )
            continue
        bus = sorted(rails)[0]
        if len(rails) > 1:
            print(
                f"  ring-power: several common rails {sorted(rails)}; "
                f"stamping {bus}"
            )

        bus_pads = {
            r: sorted(
                (p for p in fp_by_ref[r].Pads() if p.GetNetname() == bus),
                key=lambda p: p.GetNumber(),
            )[0]
            for r in refs
        }
        centers = {r: _mm_xy(fp_by_ref[r].GetPosition()) for r in refs}
        cx = sum(x for x, _ in centers.values()) / len(refs)
        cy = sum(y for _, y in centers.values()) / len(refs)
        r_mean = sum(
            math.hypot(x - cx, y - cy) for x, y in centers.values()
        ) / len(refs)

        def _ang(x: float, y: float) -> float:
            return math.degrees(math.atan2(y - cy, x - cx)) % 360.0

        # Band decaps (2 pads: this rail + ground, inside the band's radial
        # window) available for member->decap->member detours.
        gap_decaps: list[tuple[str, Any, Any, float]] = []
        for ref, fp in sorted(fp_by_ref.items()):
            if ref in refs:
                continue
            pads = list(fp.Pads())
            if len(pads) != 2:
                continue
            d_bus = [p for p in pads if p.GetNetname() == bus]
            d_gnd = [
                p for p in pads
                if p.GetNetCode() and _is_ground_name(p.GetNetname())
            ]
            if len(d_bus) != 1 or len(d_gnd) != 1:
                continue
            fx, fy = _mm_xy(fp.GetPosition())
            if not (0.5 * r_mean <= math.hypot(fx - cx, fy - cy) <= 1.5 * r_mean):
                continue
            gap_decaps.append((ref, d_bus[0], d_gnd[0], _ang(fx, fy)))

        n = len(refs)
        for k in range(n):
            ref_a, ref_b = refs[k], refs[(k + 1) % n]
            pad_a, pad_b = bus_pads[ref_a], bus_pads[ref_b]
            src = _mm_xy(pad_a.GetPosition())
            tgt = _mm_xy(pad_b.GetPosition())
            a_a = _ang(*centers[ref_a])
            step = (_ang(*centers[ref_b]) - a_a) % 360.0
            mid = (a_a + step / 2.0) % 360.0
            in_gap = sorted(
                (
                    d for d in gap_decaps
                    if d[0] not in used_decaps
                    and 0.0 < (d[3] - a_a) % 360.0 < step
                ),
                key=lambda d: (abs((d[3] - mid + 180.0) % 360.0 - 180.0), d[0]),
            )
            if in_gap:
                dref, d_bus, d_gnd, d_ang = in_gap[0]
                used_decaps.add(dref)
                dxy = _mm_xy(d_bus.GetPosition())
                specs.append(BreakoutSpec(
                    ref=ref_a, pad=pad_a.GetNumber(), waypoints=[dxy],
                    width_mm=width, layer=layer, near_xy=src,
                ))
                specs.append(BreakoutSpec(
                    ref=dref, pad=d_bus.GetNumber(), waypoints=[tgt],
                    width_mm=width, layer=layer, near_xy=dxy,
                ))
                # GND: a short tangential stub + via down to the pour, so the
                # decap's ground never pulls FreeRouting into the interior.
                gxy = _mm_xy(d_gnd.GetPosition())
                t = math.radians(d_ang + 90.0)
                specs.append(BreakoutSpec(
                    ref=dref, pad=d_gnd.GetNumber(),
                    waypoints=[(gxy[0] + stub_mm * math.cos(t),
                                gxy[1] + stub_mm * math.sin(t))],
                    width_mm=width, layer=layer, near_xy=gxy,
                    via_at_end=True,
                ))
            else:
                specs.append(BreakoutSpec(
                    ref=ref_a, pad=pad_a.GetNumber(), waypoints=[tgt],
                    width_mm=width, layer=layer, near_xy=src,
                ))
    return specs
