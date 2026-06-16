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

from typing import Any

import pcbnew

from kicraft.autoplacer.brain.breakout_stubs import BreakoutSpec
from kicraft.design.models import is_power_or_ground_name

_FAR = 1 << 30


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

    layer = cfg.get("array_data_layer", "F.Cu")
    width = cfg.get("array_data_width_mm")  # None -> add_breakout_stubs default

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
        tgt = pad_b.GetPosition()
        src = pad_a.GetPosition()
        # Direct same-net tie between the two data pads. When the array members
        # are oriented so each DOUT faces the next DIN (see ``place_array_leaves``
        # chain orientation), the two pads sit across the narrow inter-component
        # channel and this stamps a short, clean, repeating trace. The stamp-time
        # guards in :func:`breakout_stubs.add_breakout_stubs` drop any tie whose
        # straight path would still cross a foreign pad (e.g. the few row-turn
        # hops), handing just those back to FreeRouting -- never a short.
        # Escapes were tried here and measurably HURT (locked clutter raised
        # FreeRouting's abandoned-net count); ties-only is the validated win.
        ties.append((order[ref_a], BreakoutSpec(
            ref=ref_a, pad=pad_a.GetNumber(),
            waypoints=[(pcbnew.ToMM(tgt.x), pcbnew.ToMM(tgt.y))],
            width_mm=float(width) if width else None, layer=layer,
            near_xy=(pcbnew.ToMM(src.x), pcbnew.ToMM(src.y)),
        )))

    ties.sort(key=lambda item: item[0])
    return [spec for _, spec in ties]
