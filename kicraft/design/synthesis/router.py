"""Comb-stub router for KiCraft Stage B.

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
    angle_deg: int = 0


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


def _stub_end(x: float, y: float, exit_dir: str) -> tuple[float, float, int]:
    """Far end of a one-grid stub out of a pin, plus a label angle.

    Connectivity is by label name, so the stub only needs to carry the pin
    out to a point clear of the symbol body where the label sits and reads
    away from the pin.
    """
    if exit_dir == "left":
        return (x - GRID_MM, y, 180)
    if exit_dir == "up":
        return (x, y - GRID_MM, 90)
    if exit_dir == "down":
        return (x, y + GRID_MM, 270)
    # "right" and any fallback
    return (x + GRID_MM, y, 0)


def route_sheet(
    sheet_stem: str,
    sheet_name: str,
    placed_parts: list[PlacedPart],
    bom: BOM,
    architecture: Architecture,
    flag_nets: frozenset[str] = frozenset(),
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

    for conn in sheet_connections:
        endpoints: list[tuple[float, float, str]] = []
        for ep in conn.endpoints:
            pin = _get_pin(ep.ref, ep.pin)
            placed = placed_by_ref.get(ep.ref)
            if pin is None or placed is None:
                continue
            x, y = _pin_position(placed, pin)
            # Do NOT snap pin coordinates to a coarse grid. Stock KiCad
            # symbol pins sit on a 1.27 mm half-grid; rounding to 2.54 mm
            # would displace every stub up to 1.27 mm off its pin and break
            # all connectivity. The exact _pin_position value is what KiCad
            # renders, so geometry built from it lands on the pins.
            endpoints.append((x, y, _pin_exit_direction(pin)))

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
            # One power symbol per endpoint, on a short straight stub in the
            # pin's exit direction. Straight (not L-shaped) stubs never cross
            # a neighbouring pin's stub, so two power nets on adjacent pins of
            # one IC cannot be shorted together (the failure mode of the old
            # L-stub router). Connectivity is global-by-name via the symbol.
            sym_angle = 0 if "GND" in conn.net_name.upper() else 180
            flag_xy: tuple[float, float] | None = None
            for (x, y, exit_dir) in endpoints:
                ex, ey, _angle = _stub_end(x, y, exit_dir)
                if flag_xy is None:
                    flag_xy = (ex, ey)
                routed.wires.append(WireSegment(x, y, ex, ey))
                routed.power_symbols.append(
                    PowerSymbol(
                        lib_id=power_lib_id,
                        x_mm=ex,
                        y_mm=ey,
                        angle_deg=sym_angle,
                    )
                )
            # One PWR_FLAG per power net (on the first sheet that connects
            # it) marks the net as driven, so ERC doesn't flag the IC
            # power-input pins as undriven. It carries a power-output pin and
            # sits on the same node as the net's first power symbol.
            if conn.net_name in flag_nets and flag_xy is not None:
                routed.power_symbols.append(
                    PowerSymbol(
                        lib_id="power:PWR_FLAG",
                        x_mm=flag_xy[0],
                        y_mm=flag_xy[1],
                        angle_deg=0,
                    )
                )
            continue

        # Signal / inter-sheet branch — label-based connectivity.
        # Each pin gets a short stub in its exit direction plus a label
        # carrying the net name. Same-named labels are one net, so nothing
        # depends on trunk geometry and two nets cannot be shorted by a
        # coincidental wire crossing (the failure mode of a comb router on
        # multi-net IC sheets). Inter-sheet nets use hierarchical labels
        # (which tie to the parent sheet pin); sheet-local nets use plain
        # labels.
        is_inter = conn.net_name in inter_by_name
        hier_direction = "passive"
        if is_inter:
            inter = inter_by_name[conn.net_name]
            this_pin = next(
                (e for e in inter.endpoints if e.sheet == sheet_name), None
            )
            hier_direction = this_pin.direction if this_pin else "passive"

        for (x, y, exit_dir) in endpoints:
            ex, ey, angle = _stub_end(x, y, exit_dir)
            routed.wires.append(WireSegment(x, y, ex, ey))
            if is_inter:
                routed.hier_labels.append(
                    HierLabelPlacement(
                        name=conn.net_name,
                        direction=hier_direction,
                        x_mm=ex,
                        y_mm=ey,
                        angle_deg=angle,
                    )
                )
            else:
                routed.labels.append(
                    NetLabel(text=conn.net_name, x_mm=ex, y_mm=ey, angle_deg=angle)
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
        routed.no_connects.append(NoConnect(x_mm=x, y_mm=y))

    return routed
