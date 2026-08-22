"""Accept-or-revert wrapper around the C1 signal unconnected repair.

:mod:`unconnected_repair` only stamps guarded copper and reports; the verdict
-- full re-DRC, unconnected must strictly drop, shorts must not rise, no new
illegal geometry, else byte-restore the board -- lives here, so the parent
compose and the leaf solve share ONE containment policy instead of the parent
having the only copy (the leaf had no repair rung at all, so a leaf one tie
short of clean burned a whole new round on a fresh seed --
docs/plans/dense-soc-leaf-unconnected-plan.md P1.6).

Runs the repair itself in a pcbnew subprocess, so it is callable from any
process (the leaf solve has no pcbnew of its own).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

__all__ = ["attempt_signal_unconnected_repair"]

# cfg keys the repair honors; forwarded into the subprocess verbatim.
_REPAIR_CFG_KEYS = (
    "gnd_zone_net",
    "power_plane_nets",
    "signal_repair_extra_nets",
    "signal_repair_max_mm",
    "signal_repair_max_targets",
    "signal_repair_max_attempts",
    "signal_repair_dogleg_offsets_mm",
    "via_size_mm",
    "via_drill_mm",
)


def attempt_signal_unconnected_repair(
    routed_pcb,
    cfg: dict,
    validation: dict,
    *,
    anchor_names: Sequence[str] = (),
    required_anchor_names: Sequence[str] = (),
    carry_keys: Sequence[str] = (
        "post_route_repairs",
        "power_first",
        "illegal_geometry_repair",
        "interface_port_names",
    ),
    label: str = "signal unconnected repair",
) -> dict:
    """Run the C1 repair on ``routed_pcb``; keep it only if re-DRC improves.

    Accept iff unconnected strictly decreased AND shorts did not increase AND no
    new illegal-geometry flag appeared; anything else (including a crashed
    subprocess) restores the pre-repair board byte-for-byte and returns the
    original validation unchanged. ``carry_keys`` name annotations on the
    incoming validation that must survive onto the fresh re-validate dict.
    """
    from kicraft.autoplacer.routing_board import (
        run_pcbnew_script,
        validate_routed_board,
    )

    drc = validation.get("drc") or {}
    unconnected_before = int(drc.get("unconnected", 0) or 0)
    shorts_before = int(drc.get("shorts", 0) or 0)
    backup = Path(str(routed_pcb) + ".pre_signal_repair")
    shutil.copy2(routed_pcb, backup)
    # Recorded on WHICHEVER validation dict is returned (kept, reverted, or
    # crashed): the pass ran invisibly on 1/655 -- its print went to a
    # swallowed subprocess stdout, its sidecar was unlinked, and neither
    # return path carried a trace (KC-ZRAUR7 workstream B).
    record: dict = {"ran": True, "kept": False}
    try:
        _sig_cfg = json.dumps({k: cfg[k] for k in _REPAIR_CFG_KEYS if k in cfg})
        run_pcbnew_script(
            "import json\n"
            "from kicraft.autoplacer.brain.unconnected_repair import "
            "repair_unconnected_signals\n"
            f"cfg = json.loads({_sig_cfg!r})\n"
            f"s = repair_unconnected_signals({str(routed_pcb)!r}, cfg)\n"
            # NB: keys must match repair_unconnected_signals' return contract
            # {edges, tied, skipped, pruned} EXACTLY -- a KeyError here
            # crashes the subprocess AFTER the repair mutated the board, so
            # the except below silently byte-reverts it: the pass ran as a
            # no-op on every rc7 board of the 20260710 batch (N5 sweep).
            "print('signal unconnected repair:', s['edges'], 'edge(s) --',\n"
            "      s['tied'], 'tied,', len(s['skipped']), 'skipped,',\n"
            "      s['pruned'], 'pruned'\n"
            "      + (': ' + '; '.join(s['skipped']) if s['skipped'] else ''))\n"
            # The subprocess's stdout is not echoed into the caller's log, so
            # persist the summary for the parent to surface -- the per-edge
            # skip REASONS are what the next repair-geometry iteration needs
            # (the 20260713 batch had to re-run repairs offline to get them).
            f"open({str(routed_pcb) + '.signal_repair.json'!r}, 'w')"
            ".write(json.dumps(s))\n"
        )
        sidecar = Path(str(routed_pcb) + ".signal_repair.json")
        try:
            s = json.loads(sidecar.read_text(encoding="utf-8"))
            record["summary"] = s
            print(
                f"  {label}: {s.get('edges')} edge(s) -- "
                f"{s.get('tied')} tied, {len(s.get('skipped') or [])} skipped, "
                f"{s.get('pruned')} pruned"
                + (": " + "; ".join(s["skipped"]) if s.get("skipped") else "")
            )
        except Exception:
            pass
        finally:
            sidecar.unlink(missing_ok=True)
        revalidation = validate_routed_board(
            str(routed_pcb),
            cfg=cfg,
            expected_anchor_names=list(anchor_names),
            actual_anchor_names=list(anchor_names),
            required_anchor_names=list(required_anchor_names),
        )
        re_drc = revalidation.get("drc") or {}
        unconnected_after = int(re_drc.get("unconnected", 0) or 0)
        shorts_after = int(re_drc.get("shorts", 0) or 0)
        # A tie that closes an open by stamping ILLEGAL copper trades one
        # fab-gate rejection for another (run_10: the USB_DN tie grazed a
        # foreign pad at 0.05 mm and this gate accepted it on the
        # unconnected drop alone) -- the geometry flags must not appear.
        geometry_worse = (
            (revalidation.get("malformed_board_geometry")
             and not validation.get("malformed_board_geometry"))
            or (revalidation.get("obviously_illegal_routed_geometry")
                and not validation.get("obviously_illegal_routed_geometry"))
        )
        record["unconnected"] = [unconnected_before, unconnected_after]
        record["shorts"] = [shorts_before, shorts_after]
        if (unconnected_after < unconnected_before
                and shorts_after <= shorts_before
                and not geometry_worse):
            print(
                f"  {label} KEPT: unconnected "
                f"{unconnected_before} -> {unconnected_after}, shorts "
                f"{shorts_before} -> {shorts_after}"
            )
            backup.unlink(missing_ok=True)
            record["kept"] = True
            # The fresh re-validate dict must not drop the earlier
            # annotations (post_route_repairs, power_first, prior repairs).
            for key in carry_keys:
                if key in validation:
                    revalidation[key] = validation[key]
            revalidation["signal_unconnected_repair"] = record
            return revalidation
        print(
            f"  {label} reverted (unconnected "
            f"{unconnected_before} -> {unconnected_after}, shorts "
            f"{shorts_before} -> {shorts_after}"
            + (", ties stamped illegal geometry" if geometry_worse else "")
            + "); board restored"
        )
        shutil.copy2(backup, routed_pcb)
        backup.unlink(missing_ok=True)
        record["reverted"] = True
        validation["signal_unconnected_repair"] = record
        return validation
    except Exception as exc:  # noqa: BLE001 -- a repair may never fail a board
        print(f"warning: {label} failed: {exc}", file=sys.stderr)
        if backup.exists():
            shutil.copy2(backup, routed_pcb)
            backup.unlink(missing_ok=True)
        record["failed"] = str(exc)
        validation["signal_unconnected_repair"] = record
        return validation
