"""Comb-stub router for CircuitChat Stage B.

For each NetConnection:

- **Power nets** → emit a ``power:<NAME>`` symbol at each endpoint, with a
  short stub from the pin to the symbol when the pin doesn't exit
  vertically. No long wire trunks for power.
- **Signal / inter-sheet nets** → pick a horizontal trunk row at the
  median pin y, each endpoint draws a vertical stub from its pin
  position to the trunk, then a single horizontal trunk segment connects
  the leftmost and rightmost stub joints. Each stub joins at a distinct
  x, so 4-way junctions cannot exist by construction.
- **no_connect_pins** → emit a ``(no_connect …)`` marker at the pin.

This is uglier than a Steiner tree + A* router but always succeeds, is
deterministic, and is easy to verify. v1's "human readable enough" bar is
satisfied by the power symbols + visible signal trunks; the precise tree
shape doesn't matter for ERC or for the downstream pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..models import (
    BOM,
    Architecture,
    InterSheetNet,
    is_power_or_ground_name,
)
from .placement import PlacedPart
from .symbol_pinout import SymbolNotFoundError, lookup_pins


GRID_MM = 2.54

# Hierarchical label column positions (chosen to sit on the left or right
# edge of the A4-portrait usable region, snapped to the 2.54 mm grid).
HIER_LABEL_X_LEFT_MM = 25.4    # 10 * 2.54
HIER_LABEL_X_RIGHT_MM = 185.42  # 73 * 2.54
HIER_LABEL_START_Y_MM = 30.48
HIER_LABEL_PITCH_MM = 10.16  # 4 * 2.54


@dataclass(frozen=True)
class WireSegment:
    x1_mm: float
    y1_mm: float
    x2_mm: float
    y2_mm: float


@dataclass(frozen=True)
class Junction:
    x_mm: float
    y_mm: float


@dataclass(frozen=True)
class NetLabel:
    text: str
    x_mm: float
    y_mm: float
    angle_deg: int = 0


@dataclass(frozen=True)
class PowerSymbol:
    lib_id: str
    x_mm: float
    y_mm: float
    angle_deg: int  # 0 = symbol pointing up (rail); 180 = pointing down (GND)


@dataclass(frozen=True)
class NoConnect:
    x_mm: float
    y_mm: float


@dataclass(frozen=True)
class HierLabelPlacement:
    name: str
    direction: str
    x_mm: float
    y_mm: float


@dataclass
class RoutedSheet:
    wires: list[WireSegment] = field(default_factory=list)
    junctions: list[Junction] = field(default_factory=list)
    labels: list[NetLabel] = field(default_factory=list)
    power_symbols: list[PowerSymbol] = field(default_factory=list)
    no_connects: list[NoConnect] = field(default_factory=list)
    hier_labels: list[HierLabelPlacement] = field(default_factory=list)


_POWER_SYMBOL_MAP: tuple[tuple[str, str], ...] = (
    ("GND", "power:GND"),
    ("PGND", "power:PGND"),
    ("AGND", "power:GNDA"),
    ("DGND", "power:GNDD"),
    ("+3V3", "power:+3V3"),
    ("+3.3V", "power:+3V3"),
    ("3V3", "power:+3V3"),
    ("+5V", "power:+5V"),
    ("5V", "power:+5V"),
    ("+12V", "power:+12V"),
    ("12V", "power:+12V"),
    ("VBUS", "power:VBUS"),
    ("VBAT", "power:VBAT"),
    ("VSYS", "power:VSYS"),
    ("VCC", "power:VCC"),
    ("VDD", "power:VDD"),
)


def power_symbol_for(net_name: str) -> str | None:
    """Return the stock KiCad power-symbol lib_id for a net, or None."""
    n = net_name.lstrip("/").upper()
    for key, lib_id in _POWER_SYMBOL_MAP:
        if n == key.upper():
            return lib_id
    return None


def _snap(value: float, grid: float = GRID_MM) -> float:
    return round(value / grid) * grid


def _pin_position(placed: PlacedPart, pin: dict) -> tuple[float, float]:
    """Absolute (x, y) in schematic coords.

    Symbol-local pin coords use +y up (math convention); KiCad
    schematics use +y down — so we negate y. Rotation is always 0 in v1.
    """
    return (placed.x_mm + pin["position"]["x"], placed.y_mm - pin["position"]["y"])


def _pin_exit_direction(pin: dict) -> str:
    """Direction the wire should exit the pin's connection point in
    schematic coordinates (+x right, +y down).

    In a .kicad_sym, ``(at x y orientation)`` gives the pin's connection
    point and the angle at which the pin BODY extends into the symbol
    body (math convention: +y up). The wire attaches at (x, y) and
    continues in the OPPOSITE direction. After converting to schematic
    coords (y flipped):

      orientation 0   → body +x, wire exits -x  → "left"
      orientation 90  → body +y in sym (up)     → wire exits -y in sym
                                                = +y in schematic → "down"
      orientation 180 → body -x, wire exits +x  → "right"
      orientation 270 → body -y in sym (down)   → wire exits +y in sym
                                                = -y in schematic → "up"
    """
    o = pin.get("orientation", 0) % 360
    if o == 0:
        return "left"
    if o == 90:
        return "down"
    if o == 180:
        return "right"
    if o == 270:
        return "up"
    return "right"


def route_sheet(
    sheet_stem: str,
    sheet_name: str,
    placed_parts: list[PlacedPart],
    bom: BOM,
    architecture: Architecture,
) -> RoutedSheet:
    """Build the wire / junction / power / no-connect set for one leaf."""
    routed = RoutedSheet()
    placed_by_ref: dict[str, PlacedPart] = {p.ref: p for p in placed_parts}
    parts_by_ref = {p.ref: p for p in bom.parts if p.ref in placed_by_ref}
    pin_info_cache: dict[str, dict] = {}

    def _get_pin(ref: str, pin_number: str) -> dict | None:
        info = pin_info_cache.get(ref)
        if info is None:
            try:
                info = lookup_pins(parts_by_ref[ref].symbol)
            except (SymbolNotFoundError, ValueError, KeyError):
                info = {"pins": []}
            pin_info_cache[ref] = info
        for p in info["pins"]:
            if p["number"] == pin_number:
                return p
        return None

    sheet_connections = [c for c in bom.connections if c.sheet == sheet_name]
    inter_by_name: dict[str, InterSheetNet] = {
        n.name: n for n in architecture.inter_sheet_nets
    }

    hier_label_index = 0

    for conn in sheet_connections:
        endpoints: list[tuple[float, float, str]] = []
        for ep in conn.endpoints:
            pin = _get_pin(ep.ref, ep.pin)
            placed = placed_by_ref.get(ep.ref)
            if pin is None or placed is None:
                continue
            x, y = _pin_position(placed, pin)
            endpoints.append((_snap(x), _snap(y), _pin_exit_direction(pin)))

        if not endpoints:
            continue

        # Power-net branch. Only fires when the name maps to a stock
        # power symbol; otherwise we fall through to render as a signal
        # net so the connection isn't silently dropped from the
        # schematic. (The PCB still has the net regardless, since
        # kicad_pcb_stub.py works directly from bom.connections.)
        power_lib_id = (
            power_symbol_for(conn.net_name)
            if is_power_or_ground_name(conn.net_name)
            else None
        )
        if power_lib_id is not None:
            is_gnd = "GND" in conn.net_name.upper()
            for (x, y, exit_dir) in endpoints:
                if is_gnd:
                    sym_x, sym_y = x, _snap(y + 5.08)
                    sym_angle = 0
                else:
                    sym_x, sym_y = x, _snap(y - 5.08)
                    sym_angle = 180

                if exit_dir == "right":
                    turn_x = _snap(x + GRID_MM)
                    routed.wires.append(WireSegment(x, y, turn_x, y))
                    routed.wires.append(WireSegment(turn_x, y, turn_x, sym_y))
                    sym_x = turn_x
                elif exit_dir == "left":
                    turn_x = _snap(x - GRID_MM)
                    routed.wires.append(WireSegment(x, y, turn_x, y))
                    routed.wires.append(WireSegment(turn_x, y, turn_x, sym_y))
                    sym_x = turn_x
                else:
                    routed.wires.append(WireSegment(x, y, sym_x, sym_y))

                routed.power_symbols.append(
                    PowerSymbol(
                        lib_id=power_lib_id,
                        x_mm=sym_x,
                        y_mm=sym_y,
                        angle_deg=sym_angle,
                    )
                )
            continue

        # Signal / inter-sheet branch.
        # Inter-sheet nets get a hier label on the appropriate edge.
        if conn.net_name in inter_by_name:
            inter = inter_by_name[conn.net_name]
            this_pin = next(
                (e for e in inter.endpoints if e.sheet == sheet_name), None
            )
            direction = this_pin.direction if this_pin else "passive"
            label_x = (
                HIER_LABEL_X_LEFT_MM if direction == "input"
                else HIER_LABEL_X_RIGHT_MM
            )
            label_y = _snap(HIER_LABEL_START_Y_MM + hier_label_index * HIER_LABEL_PITCH_MM)
            routed.hier_labels.append(
                HierLabelPlacement(
                    name=conn.net_name,
                    direction=direction,
                    x_mm=label_x,
                    y_mm=label_y,
                )
            )
            hier_label_index += 1
            endpoints.append((label_x, label_y, "right" if direction == "input" else "left"))

        # Trunk row at the median y of all endpoints.
        ys = sorted(y for _, y, _ in endpoints)
        trunk_y = _snap(ys[len(ys) // 2])

        stub_xs: list[float] = []
        for (x, y, _exit) in endpoints:
            stub_xs.append(x)
            if y != trunk_y:
                routed.wires.append(WireSegment(x, y, x, trunk_y))

        x_min = min(stub_xs)
        x_max = max(stub_xs)
        if x_min != x_max:
            routed.wires.append(WireSegment(x_min, trunk_y, x_max, trunk_y))

        # Junctions only at interior stub-trunk joints; endpoints at
        # x_min / x_max terminate the trunk and don't need a marker.
        for sx in sorted({x for x in stub_xs if x_min < x < x_max}):
            routed.junctions.append(Junction(x_mm=sx, y_mm=trunk_y))

        # Net label only for sheet-local nets with ≥3 endpoints (R3).
        if (
            len(endpoints) >= 3
            and conn.net_name not in inter_by_name
            and not is_power_or_ground_name(conn.net_name)
        ):
            routed.labels.append(
                NetLabel(text=conn.net_name, x_mm=x_min, y_mm=trunk_y)
            )

    # no_connect markers.
    for ep in bom.no_connect_pins:
        placed = placed_by_ref.get(ep.ref)
        if placed is None:
            continue
        pin = _get_pin(ep.ref, ep.pin)
        if pin is None:
            continue
        x, y = _pin_position(placed, pin)
        routed.no_connects.append(NoConnect(x_mm=_snap(x), y_mm=_snap(y)))

    return routed
