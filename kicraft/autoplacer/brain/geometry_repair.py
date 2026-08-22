"""DRC-guided rip of illegal routed copper (self-eval 2026-07-20 N2).

routing sometimes ships geometry the fab gate must reject: copper
escaped past Edge.Cuts (``malformed_board_geometry``), tracks hugging the
outline inside the copper-edge clearance, genuine track-to-track clearance
violations, crossing tracks (all ``illegal_routed_geometry``). Rather than
shipping the rejection, this pass deletes exactly the offending track/via
copper and lets the caller re-run the existing repair machinery
(`unconnected_repair`, `gnd_pour`) to close the opens the rip created --
those stampers re-route through authoritative clearance guards, so they
cannot recreate the violation.

Selection is DRC-grounded, not re-derived: kicad-cli's JSON report names
the offending items by uuid, and only track/via items are ever ripped --
pads, footprints, zones and Edge.Cuts drawings are never touched. For a
track-pair violation the shorter track is ripped (`cleanup_routing`
precedent). Copper outside the outline is added by the same
outline-polygon test `count_copper_outside_outline` uses.

The module only rips and reports. The accept-or-revert verdict (full
re-validate: the geometry flags must clear, shorts/unconnected must not
rise, else byte-restore) is owned by the caller (`cli/_compose_route.py`),
mirroring the C1 signal-repair envelope.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any

import pcbnew

# DRC violation types whose track/via items are safe and useful to rip.
_RIPPABLE_TYPES = (
    "clearance",
    "hole_clearance",
    "copper_edge_clearance",
    "tracks_crossing",
)
# A board needing more rips than this is not a near-miss -- ripping half
# its routing produces an unroutable open pile the repair pass cannot
# close, so bail out and let the honest rejection stand.
_DEFAULT_MAX_RIPS = 50


def _drc_violations(pcb_path: str, timeout_s: int = 120) -> list[dict]:
    """kicad-cli DRC as JSON: error-severity violations with item uuids."""
    report_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            report_path = f.name
        subprocess.run(
            [
                "kicad-cli", "pcb", "drc",
                "-o", report_path,
                "--format", "json",
                "--severity-error",
                pcb_path,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        with open(report_path, encoding="utf-8", errors="replace") as f:
            report = json.load(f)
        return list(report.get("violations") or [])
    except Exception:
        return []
    finally:
        if report_path and os.path.exists(report_path):
            try:
                os.remove(report_path)
            except OSError:
                pass


def rip_illegal_copper(
    pcb_path: str, cfg: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Delete track/via copper implicated in illegal-geometry DRC violations.

    Returns ``{ripped, over_cap, nets, skipped}``; the board is saved (with
    zones refilled) only when at least one item was ripped.
    """
    cfg = cfg or {}
    board = pcbnew.LoadBoard(pcb_path)
    tracks_by_uuid = {t.m_Uuid.AsString(): t for t in board.GetTracks()}

    to_rip: dict[str, Any] = {}  # uuid -> item, insertion-ordered
    skipped: list[str] = []
    for v in _drc_violations(pcb_path):
        vtype = v.get("type")
        if vtype not in _RIPPABLE_TYPES:
            continue
        hits = [
            tracks_by_uuid[u]
            for it in (v.get("items") or [])
            if (u := str(it.get("uuid") or "")) in tracks_by_uuid
        ]
        if not hits:
            # Pad-to-pad clearance, zone involvement: not rippable copper.
            skipped.append(f"{vtype}:no_track_item")
            continue
        if len(hits) > 1:
            # Track pair: rip the shorter one (cleanup_routing precedent).
            hits.sort(key=lambda t: t.GetLength())
            hits = hits[:1]
        to_rip.setdefault(hits[0].m_Uuid.AsString(), hits[0])

    # Copper outside Edge.Cuts (the malformed_board_geometry class) -- same
    # tessellated-outline test as count_copper_outside_outline.
    poly = pcbnew.SHAPE_POLY_SET()
    try:
        outline_ok = bool(board.GetBoardPolygonOutlines(poly))
    except Exception:
        outline_ok = False
    if outline_ok and poly.OutlineCount() > 0:
        tol_nm = int(0.05 * 1e6)
        tol_sq = tol_nm * tol_nm

        def outside(p) -> bool:
            v2 = pcbnew.VECTOR2I(int(p.x), int(p.y))
            if poly.Contains(v2):
                return False
            try:
                return poly.SquaredDistance(v2) > tol_sq
            except Exception:
                return True

        for t in board.GetTracks():
            u = t.m_Uuid.AsString()
            if u in to_rip:
                continue
            if isinstance(t, pcbnew.PCB_VIA):
                if outside(t.GetPosition()):
                    to_rip[u] = t
            elif outside(t.GetStart()) or outside(t.GetEnd()):
                to_rip[u] = t

    max_rips = int(cfg.get("geometry_repair_max_rips", _DEFAULT_MAX_RIPS))
    if len(to_rip) > max_rips:
        return {
            "ripped": 0,
            "over_cap": len(to_rip),
            "nets": [],
            "skipped": skipped,
        }
    if not to_rip:
        return {"ripped": 0, "over_cap": 0, "nets": [], "skipped": skipped}

    nets = sorted({t.GetNetname() for t in to_rip.values()})
    for t in to_rip.values():
        board.Remove(t)
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(pcb_path)
    return {
        "ripped": len(to_rip),
        "over_cap": 0,
        "nets": nets,
        "skipped": skipped,
    }
