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
class GlobalLabel:
    """A hierarchy-wide label: ties same-named nets together across every
    sheet without sheet pins. Used for power/ground nets that have no stock
    KiCad power symbol (e.g. VBAT, VSYS, a +3V coin-cell rail), so the net
    keeps its exact name and stays connected across sheets."""
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
    global_labels: list[GlobalLabel] = field(default_factory=list)


_POWER_SYMBOL_MAP: tuple[tuple[str, str], ...] = (
    # Every entry's target MUST exist in stock KiCad's power library: a name
    # that maps to a non-existent symbol (the old PGND/VBAT/VSYS entries)
    # crashes synthesis with SymbolNotFoundError before ERC runs. A power net
    # with no stock symbol is rendered as a global label + PWR_FLAG instead
    # (see route_sheet); test_power_symbol_map_targets_exist guards this.
    ("GND", "power:GND"),
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

    # Every pin coordinate on the sheet — a signal trunk or riser that passes
    # through a pin NOT on its own net would connect them (a short), so a net
    # with no short-safe trunk falls back to stub+label connectivity.
    all_pin_xy: list[tuple[float, float]] = []
    for ref, part in parts_by_ref.items():
        placed = placed_by_ref.get(ref)
        if placed is None:
            continue
        try:
            info = lookup_pins(part.symbol)
        except (SymbolNotFoundError, ValueError, KeyError):
            info = {"pins": []}
        for p in info["pins"]:
            all_pin_xy.append(_pin_position(placed, p))

    # Geometry already committed this sheet, so each net is routed against the
    # ones before it. Distinct trunk Y per net + these checks guarantee two
    # different nets never overlap or share a junction (only ever cross, which
    # is not a connection in KiCad) — the fix for the shorts that drove the old
    # comb router to abandon trunks for labels.
    placed_trunks: list[tuple[float, float, float]] = []  # (y, x_min, x_max)
    placed_risers: list[tuple[float, float, float]] = []  # (x, y_min, y_max)
    placed_juncs: set[tuple[float, float]] = set()

    def _key(x: float, y: float) -> tuple[float, float]:
        return (round(x, 2), round(y, 2))

    def _col_free(x: float, y_min: float, y_max: float,
                  own: set[tuple[float, float]]) -> bool:
        """A vertical drop at column x from y_min..y_max is short-safe iff it
        hits no FOREIGN pin and overlaps no earlier net's drop. (It may freely
        cross other nets' trunks/drops — a crossing without a junction is not a
        connection in KiCad.)"""
        for (px, py) in all_pin_xy:
            if (abs(px - x) < 0.01 and y_min - 0.01 <= py <= y_max + 0.01
                    and _key(px, py) not in own):
                return False
        return not any(
            abs(rx - x) < 0.01 and not (y_max < a - 0.01 or b < y_min - 0.01)
            for (rx, a, b) in placed_risers
        )

    # One horizontal trunk LANE per net, stacked in the clear band BELOW the
    # single component row. The band has no pins, so every trunk is short-safe
    # by construction; a distinct lane Y per net means no two trunks overlap.
    # Each pin stubs out in its exit direction, then drops straight to its net's
    # lane. A net whose drop would cross a foreign pin falls back to stub+label.
    band_top = max((py for (_px, py) in all_pin_xy), default=150.0) + 7.62
    lane = {"y": band_top}

    def _draw_trunk(net_name: str, endpoints: list[tuple[float, float, str]],
                    is_inter: bool, hier_direction: str) -> bool:
        if len(endpoints) < 2:
            return False  # single-pin (inter-sheet stub) — label path handles it
        ty = lane["y"]
        own = {_key(x, y) for (x, y, _e) in endpoints}
        drops: list[tuple[float, float, float, float, float]] = []
        for (x, y, exit_dir) in endpoints:
            ex, ey, _a = _stub_end(x, y, exit_dir)
            lo, hi = sorted((ey, ty))
            if not _col_free(ex, lo, hi, own):
                return False
            drops.append((ex, ey, x, y, ty))
        for (ex, ey, x, y, ty_) in drops:
            if (abs(ex - x) >= 0.01 or abs(ey - y) >= 0.01):
                routed.wires.append(WireSegment(x, y, ex, ey))  # exit-dir stub
            if abs(ey - ty) >= 0.01:
                routed.wires.append(WireSegment(ex, ey, ex, ty))  # drop to lane
                placed_risers.append((ex, min(ey, ty), max(ey, ty)))
        exs = [d[0] for d in drops]
        x_min, x_max = min(exs), max(exs)
        if abs(x_max - x_min) >= 0.01:
            routed.wires.append(WireSegment(x_min, ty, x_max, ty))  # the lane trunk
            placed_trunks.append((ty, x_min, x_max))
        for ex in sorted({e for e in exs if x_min + 0.01 < e < x_max - 0.01}):
            routed.junctions.append(Junction(x_mm=ex, y_mm=ty))
            placed_juncs.add(_key(ex, ty))
        if is_inter:
            routed.hier_labels.append(HierLabelPlacement(
                name=net_name, direction=hier_direction,
                x_mm=x_min, y_mm=ty, angle_deg=0))
        elif len(endpoints) >= 3:
            routed.labels.append(NetLabel(text=net_name, x_mm=x_min, y_mm=ty))
        lane["y"] = ty + GRID_MM  # next net takes the next lane down
        return True

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

        # Power-net branch: any power/ground net gets global, by-name
        # connectivity (it ties together across the whole hierarchy without
        # sheet pins). Two renderings:
        #   1. a stock KiCad power symbol exists for the name -> one symbol per
        #      endpoint (the nice, conventional power port);
        #   2. it does NOT (e.g. VBAT, VSYS, a +3V coin-cell rail) -> a global
        #      label carrying the exact net name. The old code mapped case 2 to
        #      a power:<name> symbol that does not exist in stock KiCad, which
        #      crashed synthesis (SymbolNotFoundError) before ERC ran. The
        #      global label keeps the name, connects across sheets, and never
        #      references a missing symbol.
        # Straight (not L-shaped) stubs never cross a neighbouring pin's stub,
        # so two power nets on adjacent IC pins cannot be shorted together.
        if is_power_or_ground_name(conn.net_name):
            power_lib_id = power_symbol_for(conn.net_name)
            sym_angle = 0 if "GND" in conn.net_name.upper() else 180
            flag_xy: tuple[float, float] | None = None
            for (x, y, exit_dir) in endpoints:
                ex, ey, lab_angle = _stub_end(x, y, exit_dir)
                if flag_xy is None:
                    flag_xy = (ex, ey)
                routed.wires.append(WireSegment(x, y, ex, ey))
                if power_lib_id is not None:
                    routed.power_symbols.append(
                        PowerSymbol(lib_id=power_lib_id, x_mm=ex, y_mm=ey,
                                    angle_deg=sym_angle)
                    )
                else:
                    routed.global_labels.append(
                        GlobalLabel(text=conn.net_name, x_mm=ex, y_mm=ey,
                                    angle_deg=lab_angle)
                    )
            # One PWR_FLAG per power net (on the first sheet that connects it)
            # marks the net as driven, so ERC doesn't flag IC power-input pins
            # as undriven. It carries a power-output pin and sits on the node of
            # the net's first power symbol / global label.
            if conn.net_name in flag_nets and flag_xy is not None:
                routed.power_symbols.append(
                    PowerSymbol(lib_id="power:PWR_FLAG", x_mm=flag_xy[0],
                                y_mm=flag_xy[1], angle_deg=0)
                )
            continue

        # Signal / inter-sheet branch. Prefer REAL WIRES: a unique-Y horizontal
        # trunk + vertical risers + junctions (the readable, traceable look of
        # the original comb-stub router). Each net gets its own trunk Y and is
        # verified clear of foreign pins/junctions, so two nets can never short
        # (the failure that drove the regression to labels). A net with no
        # short-safe trunk falls back to the stub+label connectivity below, so
        # the sheet is always ERC-clean.
        is_inter = conn.net_name in inter_by_name
        hier_direction = "passive"
        if is_inter:
            inter = inter_by_name[conn.net_name]
            this_pin = next(
                (e for e in inter.endpoints if e.sheet == sheet_name), None
            )
            hier_direction = this_pin.direction if this_pin else "passive"

        if _draw_trunk(conn.net_name, endpoints, is_inter, hier_direction):
            continue

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
