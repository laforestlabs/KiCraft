"""Connectivity renderer for KiCraft Stage B.

Turns the wiring stage's pin-to-net map (``bom.connections``) into the
wires, junctions, power symbols, and labels a leaf .kicad_sch needs.
Works hand-in-hand with the cluster ``placement``: because each passive
is already sitting next to the pin it serves and rotated the right way,
this router can draw the human thing instead of label salad:

- **Power / ground nets** → a short stub out of each pin in its exit
  direction, with a stock power symbol (or, for a rail with no stock
  symbol, a global label) at the stub end, oriented so the symbol points
  away from the wire (rails up, grounds down). One PWR_FLAG per undriven
  rail so ERC sees it as driven.
- **Local signal nets with two pins** → a real wire: a straight segment
  or an L between the two pins, but only when that path is *short-safe*
  (crosses no foreign pin, so it can't short two nets). This is the
  series resistor / pull-up link that used to be a pair of labels.
- **Everything else** (3+ pin signal nets, anything whose direct wire
  isn't short-safe, every inter-sheet net) → a stub + a label per pin
  (hierarchical label for inter-sheet nets, local label otherwise).

Connectivity is verified against KiCad ERC: every emitted geometry lands
on a pin, and a net with no safe wire still connects by label, so the
sheet is always ERC-clean.
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
from .sch_geometry import pin_abs_position, pin_exit_direction, step
from .symbol_pinout import SymbolNotFoundError, lookup_pins

GRID_MM = 2.54
EPS = 0.01
# A direct pin-to-pin wire is only drawn for genuinely local nets; beyond this
# Manhattan span a label is cleaner (and the net is almost certainly
# cross-cluster anyway).
MAX_LINK_MM = 60.0

# Power-symbol rotation per wire exit direction (verified vs kicad-cli):
# a rail symbol (power:+3V3, VBUS, …) points UP at angle 0; a ground symbol
# points DOWN at angle 0. Both: left=90, right=270.
_RAIL_ANGLE = {"up": 0, "left": 90, "down": 180, "right": 270}
_GND_ANGLE = {"down": 0, "left": 90, "up": 180, "right": 270}
# Label angle so the text reads outward along the stub direction.
_LABEL_ANGLE = {"right": 0, "up": 90, "left": 180, "down": 270}


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
    angle_deg: int  # oriented so the symbol points away from the wire


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
    # that maps to a non-existent symbol crashes synthesis with
    # SymbolNotFoundError before ERC runs. A power net with no stock symbol is
    # rendered as a global label + PWR_FLAG instead (see route_sheet);
    # test_power_symbol_map_targets_exist guards this.
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


def _is_ground(net_name: str) -> bool:
    return "GND" in net_name.upper() or net_name.lstrip("/").upper() in {"VSS", "VEE"}


@dataclass(frozen=True)
class _Endpoint:
    x: float
    y: float
    exit: str
    ref: str
    pin: str


def route_sheet(
    sheet_stem: str,
    sheet_name: str,
    placed_parts: list[PlacedPart],
    bom: BOM,
    architecture: Architecture,
    flag_nets: frozenset[str] = frozenset(),
) -> RoutedSheet:
    """Build the wire / junction / power / label set for one leaf."""
    routed = RoutedSheet()
    placed_by_ref: dict[str, PlacedPart] = {p.ref: p for p in placed_parts}
    parts_by_ref = {p.ref: p for p in bom.parts if p.ref in placed_by_ref}
    pin_info_cache: dict[str, dict] = {}

    def _pins(ref: str) -> list[dict]:
        info = pin_info_cache.get(ref)
        if info is None:
            try:
                info = lookup_pins(parts_by_ref[ref].symbol)
            except (SymbolNotFoundError, ValueError, KeyError):
                info = {"pins": []}
            pin_info_cache[ref] = info
        return info["pins"]

    def _get_pin(ref: str, pin_number: str) -> dict | None:
        for p in _pins(ref):
            if p["number"] == pin_number:
                return p
        return None

    # Every pin coordinate on the sheet, with the (ref, pin) that owns it — used
    # to keep a direct wire from passing through (and shorting to) a foreign pin.
    all_pins: list[tuple[float, float, str, str]] = []
    for ref in parts_by_ref:
        placed = placed_by_ref[ref]
        for p in _pins(ref):
            x, y = pin_abs_position(placed.x_mm, placed.y_mm, placed.rotation_deg, p)
            all_pins.append((x, y, ref, p["number"]))

    sheet_connections = [c for c in bom.connections if c.sheet == sheet_name]
    inter_by_name: dict[str, InterSheetNet] = {
        n.name: n for n in architecture.inter_sheet_nets
    }

    for conn in sheet_connections:
        eps: list[_Endpoint] = []
        for ep in conn.endpoints:
            pin = _get_pin(ep.ref, ep.pin)
            placed = placed_by_ref.get(ep.ref)
            if pin is None or placed is None:
                continue
            x, y = pin_abs_position(placed.x_mm, placed.y_mm, placed.rotation_deg, pin)
            eps.append(_Endpoint(
                x, y, pin_exit_direction(placed.rotation_deg, pin), ep.ref, ep.pin))
        if not eps:
            continue

        if is_power_or_ground_name(conn.net_name):
            _route_power(routed, conn.net_name, eps, flag_nets)
            continue

        is_inter = conn.net_name in inter_by_name
        hier_dir = "passive"
        if is_inter:
            this = next(
                (e for e in inter_by_name[conn.net_name].endpoints
                 if e.sheet == sheet_name), None)
            hier_dir = this.direction if this else "passive"

        # Local 2-pin signal net: try a real short wire between the two pins.
        # The wire also carries one net label — a bare (unnamed) wire is
        # flagged dangling by kicad-cli ERC, and the name keeps it legible.
        if not is_inter and len(eps) == 2:
            segs = _safe_link(eps[0], eps[1], all_pins)
            if segs is not None:
                routed.wires.extend(segs)
                lx, ly = _label_anchor(segs)
                routed.labels.append(NetLabel(
                    text=conn.net_name, x_mm=lx, y_mm=ly, angle_deg=0))
                continue

        # Fallback: a stub + a label per pin (hierarchical for inter-sheet).
        for e in eps:
            ex, ey = step(e.x, e.y, e.exit, GRID_MM)
            routed.wires.append(WireSegment(e.x, e.y, ex, ey))
            angle = _LABEL_ANGLE[e.exit]
            if is_inter:
                routed.hier_labels.append(HierLabelPlacement(
                    name=conn.net_name, direction=hier_dir,
                    x_mm=ex, y_mm=ey, angle_deg=angle))
            else:
                routed.labels.append(NetLabel(
                    text=conn.net_name, x_mm=ex, y_mm=ey, angle_deg=angle))

    # no_connect markers.
    for ep in bom.no_connect_pins:
        placed = placed_by_ref.get(ep.ref)
        pin = _get_pin(ep.ref, ep.pin) if placed else None
        if placed is None or pin is None:
            continue
        x, y = pin_abs_position(placed.x_mm, placed.y_mm, placed.rotation_deg, pin)
        routed.no_connects.append(NoConnect(x_mm=x, y_mm=y))

    return routed


def _route_power(
    routed: RoutedSheet,
    net_name: str,
    eps: list[_Endpoint],
    flag_nets: frozenset[str],
) -> None:
    """A short stub + power symbol (or global label) at every pin of a power
    net, oriented to point away from the wire."""
    power_lib = power_symbol_for(net_name)
    angle_table = _GND_ANGLE if _is_ground(net_name) else _RAIL_ANGLE
    flag_xy: tuple[float, float] | None = None
    for e in eps:
        ex, ey = step(e.x, e.y, e.exit, GRID_MM)
        routed.wires.append(WireSegment(e.x, e.y, ex, ey))
        if flag_xy is None:
            flag_xy = (ex, ey)
        if power_lib is not None:
            routed.power_symbols.append(PowerSymbol(
                lib_id=power_lib, x_mm=ex, y_mm=ey, angle_deg=angle_table[e.exit]))
        else:
            routed.global_labels.append(GlobalLabel(
                text=net_name, x_mm=ex, y_mm=ey, angle_deg=_LABEL_ANGLE[e.exit]))
    # One PWR_FLAG per undriven rail marks it as driven for ERC.
    if net_name in flag_nets and flag_xy is not None:
        routed.power_symbols.append(PowerSymbol(
            lib_id="power:PWR_FLAG", x_mm=flag_xy[0], y_mm=flag_xy[1], angle_deg=0))


def _safe_link(
    a: _Endpoint, b: _Endpoint, all_pins: list[tuple[float, float, str, str]]
) -> list[WireSegment] | None:
    """A straight or single-corner wire between pins ``a`` and ``b`` that
    passes through no foreign pin (so it can't short two nets), or None."""
    if abs(a.x - b.x) + abs(a.y - b.y) > MAX_LINK_MM:
        return None
    own = {(a.ref, a.pin), (b.ref, b.pin)}

    def seg_clear(x1: float, y1: float, x2: float, y2: float) -> bool:
        for (px, py, ref, pin) in all_pins:
            if (ref, pin) in own:
                continue
            if abs(y1 - y2) < EPS and abs(py - y1) < EPS:  # horizontal
                if min(x1, x2) - EPS <= px <= max(x1, x2) + EPS:
                    return False
            elif abs(x1 - x2) < EPS and abs(px - x1) < EPS:  # vertical
                if min(y1, y2) - EPS <= py <= max(y1, y2) + EPS:
                    return False
        return True

    if abs(a.x - b.x) < EPS or abs(a.y - b.y) < EPS:
        if seg_clear(a.x, a.y, b.x, b.y):
            return [WireSegment(a.x, a.y, b.x, b.y)]
        return None
    for cx, cy in ((b.x, a.y), (a.x, b.y)):  # two L corners
        if seg_clear(a.x, a.y, cx, cy) and seg_clear(cx, cy, b.x, b.y):
            return [WireSegment(a.x, a.y, cx, cy), WireSegment(cx, cy, b.x, b.y)]
    return None


def _label_anchor(segs: list[WireSegment]) -> tuple[float, float]:
    """A point on the wire to anchor its net label: the corner of an L, or
    the midpoint of a straight run."""
    if len(segs) >= 2:
        return (segs[0].x2_mm, segs[0].y2_mm)
    s = segs[0]
    return ((s.x1_mm + s.x2_mm) / 2.0, (s.y1_mm + s.y2_mm) / 2.0)
