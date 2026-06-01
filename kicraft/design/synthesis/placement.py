"""Sheet-level part placement for KiCraft Stage B.

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


# A4 portrait page (210 × 297 mm). Parts are laid out in a grid within the
# usable region, each origin snapped to the 2.54 mm grid so pins land on
# the 1.27 mm grid KiCad expects.
SHEET_W_MM = 210.0
SHEET_H_MM = 297.0
GRID_MM = 2.54

USABLE_LEFT_MM = 38.1     # 15 * 2.54
USABLE_RIGHT_MM = 190.0
USABLE_TOP_MM = 38.1
# Padding around each part's pin bounding box so neither the power symbols
# that hang ±5.08 mm off a pin nor the net labels collide with a neighbour.
PAD_X_MM = 11.43          # 9 * 1.27 — label room left/right
PAD_Y_MM = 8.89           # 7 * 1.27 — covers the ±5.08 mm power-symbol stub
COL_GAP_MM = 5.08
ROW_GAP_MM = 7.62


def _snap(value: float, grid: float = GRID_MM) -> float:
    return round(value / grid) * grid


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
    """Place every part on a sheet, bbox-aware so nothing overlaps.

    Connectivity is label-based (see ``router``), so placement only has to
    keep parts — and the power symbols / labels that hang off their pins —
    from overlapping. Parts are laid out left-to-right, top-to-bottom in a
    grid sized by each part's pin bounding box plus padding. Returns the
    same order as ``sheet_parts``.
    """
    if not sheet_parts:
        return []

    # Pin counts (missing symbols → 0 so they don't anchor) and the
    # padded half-extent of each part's pin bounding box.
    pin_counts: dict[str, int] = {}
    half_extent: dict[str, tuple[float, float]] = {}
    for part in sheet_parts:
        try:
            pins = lookup_pins(part.symbol)["pins"]
        except (SymbolNotFoundError, ValueError):
            pins = []
        pin_counts[part.ref] = len(pins)
        if pins:
            xs = [p["position"]["x"] for p in pins]
            ys = [p["position"]["y"] for p in pins]
            hw = (max(xs) - min(xs)) / 2.0 + PAD_X_MM
            hh = (max(ys) - min(ys)) / 2.0 + PAD_Y_MM
        else:
            hw, hh = PAD_X_MM, PAD_Y_MM
        half_extent[part.ref] = (hw, hh)

    # Highest pin-count part first (the IC anchors the top-left), ties by ref.
    order = sorted(sheet_parts, key=lambda p: (-pin_counts[p.ref], p.ref))

    placed_by_ref: dict[str, PlacedPart] = {}
    cursor_x = USABLE_LEFT_MM
    row_top = USABLE_TOP_MM
    row_height = 0.0
    for idx, part in enumerate(order):
        hw, hh = half_extent[part.ref]
        cell_w = 2.0 * hw
        # Wrap to a new row when the cell would overrun the usable width
        # (but never wrap an already-empty row).
        if cursor_x + cell_w > USABLE_RIGHT_MM and cursor_x > USABLE_LEFT_MM:
            row_top += row_height + ROW_GAP_MM
            cursor_x = USABLE_LEFT_MM
            row_height = 0.0
        placed_by_ref[part.ref] = PlacedPart(
            ref=part.ref,
            x_mm=_snap(cursor_x + hw),
            y_mm=_snap(row_top + hh),
            rotation_deg=0,
            mirror=None,
            role="anchor" if idx == 0 else "grid",
        )
        cursor_x += cell_w + COL_GAP_MM
        row_height = max(row_height, 2.0 * hh)

    return [placed_by_ref[p.ref] for p in sheet_parts]
