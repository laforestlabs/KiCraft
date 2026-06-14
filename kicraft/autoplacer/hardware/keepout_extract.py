"""Extract antenna / RF keep-clear rects from a loaded pcbnew board.

Two sources, both owner-tagged so the placer can exempt the part the keep-out
belongs to:

* **preserve** -- footprint-internal rule-area zones that keep footprints/pads
  out (stock KiCad RF footprints, and KiCraft library footprints carrying an
  on-module antenna strip after Fix 0). pcbnew reports a placed footprint
  zone's outline already in board coordinates, so it is taken as-is.

* **inject** -- a config-driven per-footprint-family near-field rect
  (``cfg["antenna_keepouts"]``, keyed by footprint-name glob). The vendored
  easyeda imports dropped the stock keep-clear, and Fix 0 bakes only a modest
  on-module strip, so the larger near-field clearance is injected here. The
  spec is a rect in the footprint's LOCAL frame; it is transformed to board
  coordinates by the footprint's placed position and orientation.

Both sources are emitted (unioned by the solver's per-rect push), so a footprint
that has both an internal strip and a family-spec match is protected by both.
"""
from __future__ import annotations

from fnmatch import fnmatch

import pcbnew

from ..brain import geometry
from ..brain.types import KeepoutRect, Point


def _footprint_name_candidates(fp) -> list[str]:
    """Strings to match a footprint against an antenna_keepouts glob."""
    names: list[str] = []
    try:
        fpid = fp.GetFPID()
        item = fpid.GetLibItemName()
        names.append(str(item))
        names.append(fp.GetFPIDAsString())
    except Exception:
        pass
    try:
        names.append(fp.GetValue())
    except Exception:
        pass
    return [n for n in names if n]


def _matches_family(name_candidates: list[str], pattern: str) -> bool:
    pat = pattern.lower()
    return any(fnmatch(n.lower(), pat) for n in name_candidates)


def _transform_local_rect(
    spec: dict, origin_x: float, origin_y: float, rotation_deg: float
) -> tuple[Point, Point]:
    """Transform a local-frame rect to a board-coord AABB.

    Uses KiCad's footprint orientation convention via
    :func:`geometry.transform_point` (each local corner -> board coords). For
    90/180/270 the AABB is exact; for arbitrary angles it is the conservative
    bounding box of the rotated rect (over-approximating the keep-out, safe).
    """
    origin = Point(origin_x, origin_y)
    corners = [
        (spec["x_min"], spec["y_min"]),
        (spec["x_max"], spec["y_min"]),
        (spec["x_max"], spec["y_max"]),
        (spec["x_min"], spec["y_max"]),
    ]
    pts = [
        geometry.transform_point(Point(lx, ly), origin, rotation_deg)
        for lx, ly in corners
    ]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return Point(min(xs), min(ys)), Point(max(xs), max(ys))


def extract_keepout_rects(board, cfg: dict) -> list[KeepoutRect]:
    """Return owner-tagged board-coord keep-out rects from ``board``."""
    families: dict[str, dict] = cfg.get("antenna_keepouts", {}) or {}
    rects: list[KeepoutRect] = []

    for fp in board.Footprints():
        ref = fp.GetReferenceAsString()

        # --- preserve: footprint-internal rule-area keep-outs ---
        for zone in list(fp.Zones()):  # ZONES is iterable; no GetCount()
            if not zone.GetIsRuleArea():
                continue
            # Only zones that keep components out matter to the placer.
            if not (zone.GetDoNotAllowFootprints() or zone.GetDoNotAllowPads()):
                continue
            bb = zone.Outline().BBox()  # board coords for a placed footprint
            rects.append(
                KeepoutRect(
                    tl=Point(pcbnew.ToMM(bb.GetLeft()), pcbnew.ToMM(bb.GetTop())),
                    br=Point(pcbnew.ToMM(bb.GetRight()), pcbnew.ToMM(bb.GetBottom())),
                    owner_ref=ref,
                    source="preserve",
                )
            )

        # --- inject: config family-spec near-field rect ---
        if not families:
            continue
        candidates = _footprint_name_candidates(fp)
        if not candidates:
            continue
        pos = fp.GetPosition()
        ox, oy = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
        rot = fp.GetOrientationDegrees()
        for pattern, spec in families.items():
            if not _matches_family(candidates, pattern):
                continue
            tl, br = _transform_local_rect(spec, ox, oy, rot)
            rects.append(KeepoutRect(tl=tl, br=br, owner_ref=ref, source="inject"))

    return rects
