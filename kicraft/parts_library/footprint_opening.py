"""Reviewed wire-entry ("opening") datum for stock horizontal screw terminals.

A 90-degree screw terminal's wire mouth is the face the wires enter from. The
placer must aim that face at the board edge, and the fab gate must verify it,
but a terminal footprint carries no ``PCB Edge`` marker, and its body is all
but symmetric about the pad row -- so the geometry heuristics that read a
connector mouth from ``courtyard + fab`` overhang guess wrong.

Reported on KC-DZQ76R: J2/J3/J4 used the stock
``TerminalBlock_Phoenix_MKDS-1,5-N_1xNN_P5.00mm_Horizontal`` family. Their real
local mouth is +Y (90 deg), but the rear courtyard extends 5.71 mm from the pad
row versus 5.10 mm at the front, and that 0.61 mm difference beat
``detect_opening_direction``'s 0.5 mm body-overhang threshold -- so the placer
and the final ``connector_facings`` gate both read 270 deg and aimed the
screwdriver mouths *into* the board.

The fix is reviewed family metadata, not another tuned threshold: for the
eleven installed 5.00 mm variants the front Fab datum is y=+4.6 mm, the rear
y=-5.2 mm, and the front silk bands y=2.6/4.1 mm. This module is the single
place that records it, keyed on the exact footprint name, and it is the only
thing that knows the family is *horizontal*.

``annotate_connector_opening`` stamps the same datum into a freshly loaded
footprint as a ``PCB Edge`` Dwgs.User marker, so the marker travels with the
part into the synthesized board and every existing marker consumer (placement,
``connector_edge_gap``) reads it unchanged. ``detect_opening_direction`` also
consults the reviewed table directly, so boards synthesized before the
annotation existed still measure correctly.

Also reviewed here: the vendored Kangnex **WJ128V** 5.00 mm screw terminals
(``CONN-TH_*P-P5.00_WJ128V-*``). They are not stock KiCad parts, but the pinned
legacy design backend picks them by MPN, so a legacy-authored BOM can name a
terminal whose footprint the stock lowerer never emits -- and their 0.24 mm
courtyard asymmetry is under every threshold, so they came back unmeasured and
blocked parent composition. See the family block below for the measurement.

Physical family metadata -- never board-code, ref, or project specific. No
other footprint is matched: a terminal whose actual model has not been
reviewed is reported unmeasured rather than guessed.
"""
from __future__ import annotations

from typing import Any

# The family's reviewed local mouth direction, in the footprint frame at zero
# rotation on the FRONT side (KiCad Y-down: +Y == 90 deg). Verified against the
# installed matching STEP model and the footprint's front-face geometry.
_REVIEWED_OPENING_DEG = 90.0

# Front Fab datum (mm, footprint-local Y) of the wire-entry face. The marker
# sits on it, centred between the first and last screw pad, so the existing
# "offset from pad-cluster centre" marker reader resolves it to +Y.
_REVIEWED_MARKER_Y_MM = 4.6

# Screw pads sit at x = 0, 5.00, 10.00, ... so the pad-cluster centre of an
# N-position row is (N-1) * 5.00 / 2 mm.
_REVIEWED_PAD_PITCH_MM = 5.00

_MARKER_TEXT = "PCB Edge"


def _mkds_horizontal_name(count: int) -> str:
    return (
        f"TerminalBlock_Phoenix_MKDS-1,5-{count}_1x{count:02d}_P5.00mm_Horizontal"
    )


#: Exact footprint name -> ``(local opening deg, (marker_x_mm, marker_y_mm))``.
#: n=2..12 is the range the screw-terminal lowerer emits.
_REVIEWED_OPENINGS: dict[str, tuple[float, tuple[float, float]]] = {
    _mkds_horizontal_name(n): (
        _REVIEWED_OPENING_DEG,
        (
            (n - 1) * _REVIEWED_PAD_PITCH_MM / 2.0,
            _REVIEWED_MARKER_Y_MM,
        ),
    )
    for n in range(2, 13)
}

# --- Vendored Kangnex WJ128V 5.00 mm screw terminals -----------------------
# Reviewed 2026-09-23 against each part's vendored easyeda2kicad 3D mesh (WRL,
# 2.54 mm per VRML unit), registered on the footprint's own through-hole pins
# (mesh pin x centres == the pad x row; the pin blade's y -0.43..0.37 == the
# y~0 pad row), with the footprint's own drawing and the vendor customer
# drawing as cross-checks.
#
# 4P (``CONN-TH_4P-P5.00_WJ128V-4P-5.0-14-00A``, LCSC C192769) -- the part
# KC-CG58R4's J2 used -- measured at every pin position:
#   * mesh outline y -5.03..+5.27 mm against the footprint's F.SilkS outline
#     -5.02..+5.29 mm: the same profile, so the mesh frame is registered and
#     the deeper long side is +Y;
#   * the outer +Y wall is absent over z 2.5..6.0 mm of the 14.1 mm-tall body
#     (the wire window), while the -Y wall stops at z 4.75 mm;
#   * the only material along Y at those heights is the clamp cage's back
#     plate, y -3.83..-3.03 mm at z 3..5. A wire enters 8.6..9.1 mm from +Y, so
#     the datasheet's 6~7 mm strip length spans the clamp; from -Y it stops on
#     the plate after 1.2..2.0 mm and never reaches the clamp;
#   * the screws are vertical -- the driver-access hole is in the top face at
#     each pin position -- so the entry is horizontal and the local mouth is
#     +Y = 90 deg. The customer drawing's front view shows the same square
#     openings low on the body, opposite the fence the reference text sits on.
#
# 5P (``CONN-TH_5P-P5.00_WJ128V-5P-5.0-14-00A``, LCSC C192770) corroborates
# the family datum: its pins register the same way (x -10..10, y ~0), its mesh
# carries the same window in the +Y wall (z 3.25..5.0) with the clamp-reaching
# path from +Y (median 6.35 mm vs 4.15 mm from -Y), and its footprint drawing
# repeats the 4P profile (F.SilkS -5.02..+5.29, wire-entry glyphs and the value
# text on +Y, reference text at the -Y fence).
#
# Keyed on the exact footprint name; the two 4P bundles
# (screw-terminal-5mm-4p-128, screw-terminal-5mm-4p-wj128v) ship the same one.
_WJ128V_OPENING_DEG = 90.0

# Wire-entry wall outer face (mm, footprint-local Y): the mesh-measured
# y=+5.27 mm, which the F.SilkS outline draws at +5.29 mm. Both vendored
# footprints centre their pin row on x=0.
_WJ128V_MARKER_MM = (0.0, 5.27)

_WJ128V_FOOTPRINT_NAMES = (
    # 4P (C192769) and 5P (C192770), measured as described above.
    "CONN-TH_4P-P5.00_WJ128V-4P-5.0-14-00A",
    "CONN-TH_5P-P5.00_WJ128V-5P-5.0-14-00A",
)

for _wj128v_name in _WJ128V_FOOTPRINT_NAMES:
    _REVIEWED_OPENINGS[_wj128v_name] = (_WJ128V_OPENING_DEG, _WJ128V_MARKER_MM)


def _bare_name(footprint_name: str) -> str:
    """``"Lib:Name"`` -> ``"Name"``; already-bare names pass through."""
    return (footprint_name or "").rsplit(":", 1)[-1].strip()


def reviewed_connector_opening(
    footprint_name: str,
) -> tuple[float, tuple[float, float]] | None:
    """Reviewed ``(local opening deg, marker point mm)``, or None if unreviewed."""
    return _REVIEWED_OPENINGS.get(_bare_name(footprint_name))


def is_horizontal_terminal(footprint_name: str) -> bool:
    """True when the footprint is a horizontal screw/plug terminal row.

    Recognized either from the reviewed table or from the stock KiCad naming
    (``TerminalBlock_*_Horizontal``). Vertical terminals, bare pin headers,
    switches and battery holders are NOT directional terminals and return
    False: their footprint name cannot answer "which way does the wire go".
    """
    name = _bare_name(footprint_name)
    if not name:
        return False
    if name in _REVIEWED_OPENINGS:
        return True
    return (
        name.startswith("TerminalBlock_")
        and name.endswith("_Horizontal")
        and "_Vertical" not in name
    )


def is_board_normal_header(footprint_name: str) -> bool:
    """True only for header naming contracts that explicitly declare vertical.

    ``PinHeader_*_Vertical`` is KiCad's stock mechanical identity. The
    vendored ``HDR-TH_*`` family uses ``-V-`` for its vertical THT variant;
    the frozen C492422 2x05 footprint is
    ``HDR-TH_10P-P2.54-V-M-R2-C5-S2.54``. Both mate along board normal, not
    through an in-plane opening. This deliberately excludes generic THT
    connectors and all horizontal variants: absence of a direction remains
    unverified rather than becoming evidence of vertical mating.
    """
    name = _bare_name(footprint_name)
    return (
        name.startswith("PinHeader_") and name.endswith("_Vertical")
    ) or (
        name.startswith("HDR-TH_") and "-V-" in name
    )




def _explicit_edge_marker_points(pcbnew_mod: Any, fp) -> list[tuple[float, float]]:
    """Board-coords positions of every ``'PCB Edge'``/``'Board Edge'`` fp text
    on Dwgs.User (the footprint author's declared board-edge line)."""
    points: list[tuple[float, float]] = []
    for item in fp.GraphicalItems():
        try:
            if item.GetLayer() != pcbnew_mod.Dwgs_User:
                continue
            text = item.GetText()
        except Exception:  # noqa: BLE001 -- non-text items expose no GetText
            continue
        if text and "edge" in text.lower():
            pos = item.GetPosition()
            points.append((pcbnew_mod.ToMM(pos.x), pcbnew_mod.ToMM(pos.y)))
    return points


def _direction_from_offset(dx: float, dy: float) -> float:
    """Cardinal direction (deg) of an offset dominating-axis-first."""
    if abs(dx) > abs(dy):
        return 0.0 if dx > 0 else 180.0
    return 90.0 if dy > 0 else 270.0


def explicit_edge_marker_direction(pcbnew_mod: Any, fp) -> float | None:
    """LOCAL opening direction declared by explicit ``PCB Edge`` markers.

    ``None`` when the footprint carries no marker. Raises ``ValueError`` when
    two markers imply different directions: a footprint that contradicts
    itself has no authoritative datum, and silently taking whichever appears
    first is how a coin flip gets promoted to a measured fact.
    """
    points = _explicit_edge_marker_points(pcbnew_mod, fp)
    if not points:
        return None
    pad_xs = [pcbnew_mod.ToMM(p.GetPosition().x) for p in fp.Pads()]
    pad_ys = [pcbnew_mod.ToMM(p.GetPosition().y) for p in fp.Pads()]
    if not pad_xs:
        return None
    pad_cx = (min(pad_xs) + max(pad_xs)) / 2.0
    pad_cy = (min(pad_ys) + max(pad_ys)) / 2.0
    directions = {
        _direction_from_offset(mx - pad_cx, my - pad_cy) for mx, my in points
    }
    if len(directions) > 1:
        raise ValueError(
            "contradictory 'PCB Edge' Dwgs.User markers imply opening "
            f"directions {sorted(directions)}"
        )
    board_deg = directions.pop()
    return (board_deg + fp.GetOrientationDegrees()) % 360.0


def annotate_connector_opening(
    pcbnew_mod: Any, fp, footprint_name: str
) -> bool:
    """Stamp the reviewed ``PCB Edge`` marker into a loaded footprint.

    Operates on the freshly loaded, front-side, zero-rotation footprint the
    synthesis loader hands over. An explicit author marker is authoritative
    and left untouched. Returns True only when this call added the marker;
    False for an unreviewed name, a footprint that already declares its edge,
    or one that is not in the loader's canonical frame. Never writes to a
    system library and never touches pads or the 3D model.
    """
    entry = reviewed_connector_opening(footprint_name)
    if entry is None:
        return False
    if fp.GetLayer() != pcbnew_mod.F_Cu:
        return False
    if (fp.GetOrientationDegrees() % 360.0) != 0.0:
        return False
    if explicit_edge_marker_direction(pcbnew_mod, fp) is not None:
        return False
    _, (marker_x, marker_y) = entry
    text = pcbnew_mod.PCB_TEXT(fp)
    text.SetText(_MARKER_TEXT)
    text.SetLayer(pcbnew_mod.Dwgs_User)
    text.SetPosition(
        pcbnew_mod.VECTOR2I(
            pcbnew_mod.FromMM(marker_x), pcbnew_mod.FromMM(marker_y)
        )
    )
    text.SetTextSize(
        pcbnew_mod.VECTOR2I(pcbnew_mod.FromMM(1.0), pcbnew_mod.FromMM(1.0))
    )
    text.SetTextThickness(pcbnew_mod.FromMM(0.15))
    fp.Add(text)
    return True

__all__ = [
    "annotate_connector_opening",
    "explicit_edge_marker_direction",
    "is_horizontal_terminal",
    "is_board_normal_header",
    "reviewed_connector_opening",
]
