"""Net-code propagation across same-numbered pads.

KiCad treats every pad sharing a number (split thermal pads,
dual-terminal tactile switches, etc.) as one electrical node, but
boards generated through ``FindPadByNumber`` only assign a net to the
first matching instance and leave the duplicates on no net; DRC then
flags them against the copper that legitimately covers the shared
area.

This helper is called by both stamp subprocesses
(``kicraft.autoplacer.hardware._stamp_subcircuit_subprocess`` and
``kicraft.cli._parent_stamp_subprocess``) after they snapshot the
board's footprints. It does NOT touch pads whose number is the empty
string -- those are mounting holes / NPTH and bucketing them all under
the same ``""`` key would silently bridge unrelated standoffs the
moment any one of them carries a net.
"""
from __future__ import annotations

from typing import Iterable


def propagate_pad_nets(footprints: Iterable) -> None:
    """Copy net codes from netted pads to zero-net pads sharing the same
    non-empty number, footprint by footprint.

    Idempotent. Mutates pad net codes in place. ``footprints`` is any
    iterable of pcbnew ``FOOTPRINT`` objects -- typically the snapshot
    list the caller already built via ``list(board.Footprints())``.
    """
    for fp in footprints:
        pads = list(fp.Pads())
        net_for_num: dict[str, int] = {}
        for pad in pads:
            num = pad.GetNumber()
            if not num:
                continue
            nc = pad.GetNetCode()
            if nc:
                net_for_num.setdefault(num, nc)
        if not net_for_num:
            continue
        for pad in pads:
            if pad.GetNetCode() != 0:
                continue
            num = pad.GetNumber()
            if not num:
                continue
            nc = net_for_num.get(num)
            if nc:
                pad.SetNetCode(nc)
