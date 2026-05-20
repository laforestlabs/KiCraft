"""Sheet-level part placement for CircuitChat Stage B.

Simple deterministic placement, deliberately less clever than the prior
spec's 8-orientation scoring + collision iteration:

- Anchor = the highest-pin-count part on the sheet (ties broken by ref).
- Anchor placed at the sheet center.
- Parts in ``bom.ic_groups[anchor.ref]`` tiled around the anchor in a
  fixed 8-position ring at 7.62 mm pitch; second 16-position ring at
  15.24 mm pitch for overflow.
- Peripherals (parts not in any ic_group on this sheet) tiled in a
  right-edge column at 12.7 mm pitch.
- No orientation search. All parts placed at rotation 0.

Trade-off: schematics aren't pretty, but they always succeed and are
deterministic by construction. Stage B's primary success bar is
"connectivity rendered + ERC clean," not visual polish.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import BOM, BomPart, Sheet
from .symbol_pinout import SymbolNotFoundError, lookup_pins


# A4 portrait page (210×297 mm). Center coordinates snapped to the
# 2.54 mm grid that KiCad uses for schematic placement.
SHEET_W_MM = 210.0
SHEET_H_MM = 297.0
ANCHOR_X_MM = 101.6    # 40 * 2.54
ANCHOR_Y_MM = 149.86   # 59 * 2.54
GRID_MM = 2.54

PERIPHERAL_X_MM = 175.26  # 69 * 2.54
PERIPHERAL_START_Y_MM = 30.48
PERIPHERAL_PITCH_MM = 12.7


# 8 first-ring positions at 7.62 mm (3-grid) pitch.
_FIRST_RING_OFFSETS: tuple[tuple[float, float], ...] = (
    (0.0, -7.62),    # N
    (7.62, -7.62),   # NE
    (7.62, 0.0),     # E
    (7.62, 7.62),    # SE
    (0.0, 7.62),     # S
    (-7.62, 7.62),   # SW
    (-7.62, 0.0),    # W
    (-7.62, -7.62),  # NW
)

# 16 second-ring positions at 15.24 mm (6-grid) pitch.
_SECOND_RING_OFFSETS: tuple[tuple[float, float], ...] = (
    (0.0, -15.24), (7.62, -15.24), (15.24, -15.24),
    (15.24, -7.62), (15.24, 0.0), (15.24, 7.62), (15.24, 15.24),
    (7.62, 15.24), (0.0, 15.24), (-7.62, 15.24), (-15.24, 15.24),
    (-15.24, 7.62), (-15.24, 0.0), (-15.24, -7.62), (-15.24, -15.24),
    (-7.62, -15.24),
)


@dataclass(frozen=True)
class PlacedPart:
    ref: str
    x_mm: float
    y_mm: float
    rotation_deg: int   # 0 | 90 | 180 | 270 (always 0 in v1)
    mirror: str | None  # None | "x" | "y" (always None in v1)
    role: str           # anchor | ring1 | ring2 | peripheral


def place_sheet(
    sheet: Sheet,
    sheet_parts: list[BomPart],
    bom: BOM,
) -> list[PlacedPart]:
    """Place every part on a sheet. Returns same order as ``sheet_parts``."""
    if not sheet_parts:
        return []

    # Pin counts; missing symbols become count=0 so they don't anchor.
    pin_counts: dict[str, int] = {}
    for part in sheet_parts:
        try:
            info = lookup_pins(part.symbol)
            pin_counts[part.ref] = len(info["pins"])
        except (SymbolNotFoundError, ValueError):
            pin_counts[part.ref] = 0

    anchor = sorted(
        sheet_parts,
        key=lambda p: (-pin_counts.get(p.ref, 0), p.ref),
    )[0]

    sheet_refs = {p.ref for p in sheet_parts}
    ring_members = [
        ref for ref in bom.ic_groups.get(anchor.ref, [])
        if ref in sheet_refs and ref != anchor.ref
    ]
    ring_member_set = set(ring_members)
    peripherals = [
        p for p in sheet_parts
        if p.ref != anchor.ref and p.ref not in ring_member_set
    ]

    placed_by_ref: dict[str, PlacedPart] = {
        anchor.ref: PlacedPart(
            ref=anchor.ref,
            x_mm=ANCHOR_X_MM,
            y_mm=ANCHOR_Y_MM,
            rotation_deg=0,
            mirror=None,
            role="anchor",
        )
    }

    for i, ref in enumerate(ring_members):
        if i < len(_FIRST_RING_OFFSETS):
            ox, oy = _FIRST_RING_OFFSETS[i]
            role = "ring1"
        elif i - len(_FIRST_RING_OFFSETS) < len(_SECOND_RING_OFFSETS):
            ox, oy = _SECOND_RING_OFFSETS[i - len(_FIRST_RING_OFFSETS)]
            role = "ring2"
        else:
            # Overflow beyond second ring: treat as peripheral.
            for p in sheet_parts:
                if p.ref == ref:
                    peripherals.append(p)
                    break
            continue
        placed_by_ref[ref] = PlacedPart(
            ref=ref,
            x_mm=ANCHOR_X_MM + ox,
            y_mm=ANCHOR_Y_MM + oy,
            rotation_deg=0,
            mirror=None,
            role=role,
        )

    for i, part in enumerate(peripherals):
        placed_by_ref[part.ref] = PlacedPart(
            ref=part.ref,
            x_mm=PERIPHERAL_X_MM,
            y_mm=PERIPHERAL_START_Y_MM + i * PERIPHERAL_PITCH_MM,
            rotation_deg=0,
            mirror=None,
            role="peripheral",
        )

    return [placed_by_ref[p.ref] for p in sheet_parts]
