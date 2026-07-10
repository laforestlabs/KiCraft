"""Mechanical-conformance check: does a delivered board match a standard?

A standard form factor is a hard contract, so "did we honor it?" must be a
deterministic, geometry-level check -- not a net-name comparison (the design's
nets carry the user's names, e.g. ``SENSOR_OUT`` wired to ``A0``, not the
canonical ones). The check asks the physical question that decides whether the
board actually mates: **is there a header pad at each position the standard
fixes, and is the board the standard's size?**

This is the read-only core the promote gate (PR3) and the investigate §8.5 audit
consume. It reports; it does not place. Compare board-local positions in the
template's top-left frame -- the board adapter normalizes a delivered board to
its Edge.Cuts top-left before calling :func:`check_conformance`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import FormFactorTemplate


@dataclass(frozen=True)
class ConformanceReport:
    conformant: bool
    outline_ok: bool
    matched_pins: int
    total_pins: int
    missing: list[tuple[str, float, float]] = field(default_factory=list)  # (net, x, y)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        head = "CONFORMANT" if self.conformant else "NON-CONFORMANT"
        return (
            f"{head}: {self.matched_pins}/{self.total_pins} standard header pins "
            f"present at their fixed positions; outline "
            f"{'ok' if self.outline_ok else 'MISMATCH'}"
            + (f"; {self.notes[0]}" if self.notes else "")
        )


def expected_pins(template: FormFactorTemplate) -> list[tuple[str, float, float]]:
    """``(net, x_mm, y_mm)`` for every fixed-connector pin, board-local frame."""
    out: list[tuple[str, float, float]] = []
    for c in template.fixed_connectors:
        out.extend(c.pin_positions())
    return out


def check_conformance(
    template: FormFactorTemplate,
    delivered_xy: list[tuple[float, float]],
    board_wh: tuple[float, float] | None = None,
    *,
    tol_mm: float = 1.5,
) -> ConformanceReport:
    """Check a delivered board's pad geometry against a standard template.

    ``delivered_xy`` is every pad centre on the board, in the template's
    board-local (top-left) frame. ``board_wh`` is the delivered board's
    (width, height); when given, it must match the template within ``tol_mm``.
    A standard header pin "matches" when some delivered pad sits within
    ``tol_mm`` of its fixed position. Net names are deliberately ignored --
    only geometry decides whether the board mates.
    """
    exp = expected_pins(template)
    dpts = list(delivered_xy)
    tol_sq = tol_mm * tol_mm
    matched = 0
    missing: list[tuple[str, float, float]] = []
    for net, ex, ey in exp:
        if any((dx - ex) ** 2 + (dy - ey) ** 2 <= tol_sq for dx, dy in dpts):
            matched += 1
        else:
            missing.append((net, round(ex, 3), round(ey, 3)))

    outline_ok = True
    notes: list[str] = []
    if board_wh is not None:
        w, h = board_wh
        outline_ok = (
            abs(w - template.board_width_mm) <= tol_mm
            and abs(h - template.board_height_mm) <= tol_mm
        )
        if not outline_ok:
            notes.append(
                f"outline {w:.1f}x{h:.1f}mm != standard "
                f"{template.board_width_mm}x{template.board_height_mm}mm"
            )

    conformant = matched == len(exp) and outline_ok
    return ConformanceReport(
        conformant=conformant,
        outline_ok=outline_ok,
        matched_pins=matched,
        total_pins=len(exp),
        missing=missing,
        notes=notes,
    )


def _rotate_cw(x: float, y: float, deg: float) -> tuple[float, float]:
    """Rotate a footprint-local point by ``deg`` in KiCad's convention (the same
    ``x·cos+y·sin, -x·sin+y·cos`` the placer/stamp use, so this reader agrees with
    how the board was actually built)."""
    if deg % 360 == 0.0:
        return x, y
    import math

    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return x * c + y * s, -x * s + y * c


def board_local_pads(pcb_path: str) -> tuple[list[tuple[float, float]], tuple[float, float] | None]:
    """Read a .kicad_pcb and return (pad centres, (width, height)) in the board's
    top-left-local frame (every pad shifted so the Edge.Cuts min corner is 0,0).

    Regex-based and pcbnew-free so it runs anywhere; returns ([], None) on any
    parse trouble rather than raising -- a best-effort input to the read-only
    conformance check.
    """
    import re
    from pathlib import Path

    try:
        txt = Path(pcb_path).read_text()
    except OSError:
        return [], None

    edge_x: list[float] = []
    edge_y: list[float] = []
    for m in re.finditer(
        r"\(gr_line[^)]*\(start ([\-0-9.]+) ([\-0-9.]+)\)[^)]*\(end ([\-0-9.]+) ([\-0-9.]+)\)",
        txt,
    ):
        if "Edge.Cuts" in txt[m.start() : m.start() + 300]:
            edge_x += [float(m.group(1)), float(m.group(3))]
            edge_y += [float(m.group(2)), float(m.group(4))]

    pads: list[tuple[float, float]] = []
    # Footprint at (ox oy [rot]) then each pad's local (at px py); the footprint
    # rotation MUST be applied to the pad offset (a header laid horizontally along
    # a board edge is stamped rotated, so its pads only reach the standard's pin
    # positions after the turn). world = origin + rotate_kicad_cw(local, rot).
    for blk in re.split(r"\n\s*\(footprint ", txt)[1:]:
        mo = re.search(r"\(at ([\-0-9.]+) ([\-0-9.]+)(?: ([\-0-9.]+))?\)", blk)
        if not mo:
            continue
        ox, oy = float(mo.group(1)), float(mo.group(2))
        frot = float(mo.group(3)) if mo.group(3) else 0.0
        for pm in re.finditer(r"\(pad\s+\S+\s+\S+\s+\S+\s*\(at ([\-0-9.]+) ([\-0-9.]+)", blk):
            rx, ry = _rotate_cw(float(pm.group(1)), float(pm.group(2)), frot)
            pads.append((ox + rx, oy + ry))

    if edge_x and edge_y:
        min_x, min_y = min(edge_x), min(edge_y)
        wh = (max(edge_x) - min_x, max(edge_y) - min_y)
        pads = [(px - min_x, py - min_y) for px, py in pads]
        return pads, wh
    return pads, None


__all__ = [
    "ConformanceReport",
    "expected_pins",
    "check_conformance",
    "board_local_pads",
]
