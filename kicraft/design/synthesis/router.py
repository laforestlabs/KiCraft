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

from dataclasses import dataclass, field, replace

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


def _label_position_free(
    x_mm: float, y_mm: float,
    net_name: str,
    existing: list[NetLabel],
    hier_existing: list[HierLabelPlacement],
    tolerance_mm: float = 0.05,
) -> bool:
    """True when no label from a *different* net occupies the same position.

    KiCad merges two net labels at the same coordinates into one shared net,
    silently shorting the two nets (the "label slide" defect: a label from
    one net lands on another net's label and bridges them). This check
    prevents new labels from colliding with existing ones from other nets.
    """
    for lab in existing:
        if lab.text == net_name:
            continue  # same net, safe to share position
        if abs(lab.x_mm - x_mm) < tolerance_mm and abs(lab.y_mm - y_mm) < tolerance_mm:
            return False
    for hlab in hier_existing:
        if hlab.name == net_name:
            continue
        if abs(hlab.x_mm - x_mm) < tolerance_mm and abs(hlab.y_mm - y_mm) < tolerance_mm:
            return False
    return True


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
    # Power stubs placed so far on this sheet, as (net, x1, y1, x2, y2): two
    # rails' stubs meeting head-on (or a stub end landing on a foreign pin)
    # silently merge two GLOBAL nets -- see _route_power.
    power_stubs: list[tuple[str, float, float, float, float]] = []

    # Signal nets route FIRST, power nets after: a power stamp (stub, elbow,
    # merged bus) is the flexible one — it can retreat or fall back to a
    # label — so it is the side that must dodge. With pins one grid apart, a
    # power elbow ("one grid out, one grid down") otherwise lands exactly on
    # the neighboring pin's signal stub: the wires touch, KiCad merges the
    # nets, and the only symptom is a multiple_net_names WARNING while the
    # netlist quietly ties the signal to the rail (the VBUS≡USB_DP /
    # GND≡USB_DN family found on 7/9 self-eval designs).
    def _resolve_eps(conn) -> list[_Endpoint]:
        eps: list[_Endpoint] = []
        for ep in conn.endpoints:
            pin = _get_pin(ep.ref, ep.pin)
            placed = placed_by_ref.get(ep.ref)
            if pin is None or placed is None:
                continue
            x, y = pin_abs_position(placed.x_mm, placed.y_mm, placed.rotation_deg, pin)
            eps.append(_Endpoint(
                x, y, pin_exit_direction(placed.rotation_deg, pin), ep.ref, ep.pin))
        return eps

    signal_conns = [
        c for c in sheet_connections if not is_power_or_ground_name(c.net_name)
    ]
    power_conns = [
        c for c in sheet_connections if is_power_or_ground_name(c.net_name)
    ]

    for conn in signal_conns:
        eps = _resolve_eps(conn)
        if not eps:
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
            segs = _safe_link(eps[0], eps[1], all_pins, routed.wires)
            if segs is not None:
                routed.wires.extend(segs)
                lx, ly = _label_anchor(segs)
                if not _label_position_free(lx, ly, conn.net_name,
                                            routed.labels, routed.hier_labels):
                    # Offset to avoid label-slide net merge
                    lx = lx + 0.2
                    if not _label_position_free(lx, ly, conn.net_name,
                                                routed.labels, routed.hier_labels):
                        lx = lx - 0.4  # try the other direction
                routed.labels.append(NetLabel(
                    text=conn.net_name, x_mm=lx, y_mm=ly, angle_deg=0))
                continue

        # Fallback: a stub + a label per pin (hierarchical for inter-sheet).
        # A stub whose copper would touch an earlier net's wire retreats to
        # half a grid, then to a wireless label at the pin itself — touching
        # wires of different nets merge, and two facing stubs two grids
        # apart otherwise meet head-on at the midpoint (run_06's
        # OUT1A≡SENSE1 short).
        for e in eps:
            ex, ey = step(e.x, e.y, e.exit, GRID_MM)
            if _seg_touches_any_wire(e.x, e.y, ex, ey, routed.wires):
                ex, ey = step(e.x, e.y, e.exit, GRID_MM / 2)
                if _seg_touches_any_wire(e.x, e.y, ex, ey, routed.wires):
                    ex, ey = e.x, e.y
            if (ex, ey) != (e.x, e.y):
                routed.wires.append(WireSegment(e.x, e.y, ex, ey))
            angle = _LABEL_ANGLE[e.exit]
            lx, ly = ex, ey
            # Prevent label-slide net merge: if another net's label already
            # sits at this position, offset to avoid KiCad merging the nets.
            if is_inter:
                if not _label_position_free(lx, ly, conn.net_name,
                                            routed.labels, routed.hier_labels):
                    lx = lx + 0.2
                routed.hier_labels.append(HierLabelPlacement(
                    name=conn.net_name, direction=hier_dir,
                    x_mm=lx, y_mm=ly, angle_deg=angle))
            else:
                if not _label_position_free(lx, ly, conn.net_name,
                                            routed.labels, routed.hier_labels):
                    lx = lx + 0.2
                routed.labels.append(NetLabel(
                    text=conn.net_name, x_mm=lx, y_mm=ly, angle_deg=angle))

    # Everything stamped so far is signal copper the power pass must avoid;
    # power-vs-power conflicts stay the job of the power_stubs ledger.
    signal_wires = list(routed.wires)
    for conn in power_conns:
        eps = _resolve_eps(conn)
        if not eps:
            continue
        _route_power(
            routed, conn.net_name, eps, flag_nets, all_pins, power_stubs,
            signal_wires,
        )

    # no_connect markers.
    for ep in bom.no_connect_pins:
        placed = placed_by_ref.get(ep.ref)
        pin = _get_pin(ep.ref, ep.pin) if placed else None
        if placed is None or pin is None:
            continue
        x, y = pin_abs_position(placed.x_mm, placed.y_mm, placed.rotation_deg, pin)
        routed.no_connects.append(NoConnect(x_mm=x, y_mm=y))

    # Estimated obstacle rects for the label de-collision pass: part bodies
    # (pin extents shrunk by the pin length, so the rect tracks the actual
    # body outline) and power-symbol graphics+text. A collapsed axis (2-pin
    # passive) falls back to a minimum half-extent so labels can't be slid
    # onto a resistor body just because its pins are collinear.
    body_rects: list[tuple[float, float, float, float]] = []
    for ref in parts_by_ref:
        pts = [(x, y) for (x, y, r, _pn) in all_pins if r == ref]
        if not pts:
            continue
        pin_lengths = [p.get("length", 2.54) or 2.54 for p in _pins(ref)]
        shrink = (max(pin_lengths) if pin_lengths else 2.54) + 0.5
        xs2 = [p[0] for p in pts]
        ys2 = [p[1] for p in pts]
        x1, x2 = min(xs2) + shrink, max(xs2) - shrink
        y1, y2 = min(ys2) + shrink, max(ys2) - shrink
        if x1 >= x2:
            cx = (min(xs2) + max(xs2)) / 2.0
            x1, x2 = cx - 1.2, cx + 1.2
        if y1 >= y2:
            cy = (min(ys2) + max(ys2)) / 2.0
            y1, y2 = cy - 1.2, cy + 1.2
        body_rects.append((x1, y1, x2, y2))
    for ps in routed.power_symbols:
        body_rects.append((ps.x_mm - 3.0, ps.y_mm - 4.0, ps.x_mm + 3.0, ps.y_mm + 4.0))

    _resolve_label_collisions(routed, body_rects, all_pins)

    return routed


# Estimated text metrics for the default 1.27 mm schematic font.
_CHAR_W_MM = 1.1
_TEXT_H_MM = 1.9


def _label_rect(
    text: str, x: float, y: float, angle: int
) -> tuple[float, float, float, float]:
    """Estimated bbox (x1, y1, x2, y2) of a label's rendered text."""
    length = len(text) * _CHAR_W_MM + 1.5
    half_h = _TEXT_H_MM / 2.0
    if angle == 0:
        return (x + 0.5, y - half_h, x + 0.5 + length, y + half_h)
    if angle == 180:
        return (x - 0.5 - length, y - half_h, x - 0.5, y + half_h)
    if angle == 90:  # reads bottom-to-top, extends up from the anchor
        return (x - half_h, y - 0.5 - length, x + half_h, y - 0.5)
    return (x - half_h, y + 0.5, x + half_h, y + 0.5 + length)  # 270


def _rects_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


_LABEL_OUTWARD = {0: "right", 90: "up", 180: "left", 270: "down"}


def _wire_component(
    anchor_x: float, anchor_y: float, wires: list[WireSegment]
) -> list[WireSegment]:
    """The wires electrically reachable from (anchor_x, anchor_y).

    Seeded by every wire passing through the anchor (a label binds anywhere
    along a wire), grown via SHARED ENDPOINTS only: the router emits
    connected runs endpoint-to-endpoint, and kicad-cli ERC does not bind a
    wire end teeing into another wire's mid-span even under a junction (see
    _merge_stacked_power_pins) -- so endpoint adjacency is the conservative
    notion of "same net". Under-including only costs a slide candidate;
    over-including would let a label hop onto a foreign net.
    """
    comp = [
        w for w in wires
        if _pt_on_axis_seg(anchor_x, anchor_y, w.x1_mm, w.y1_mm, w.x2_mm, w.y2_mm)
    ]
    seen = {id(w) for w in comp}
    frontier = list(comp)
    while frontier:
        w = frontier.pop()
        w_ends = ((w.x1_mm, w.y1_mm), (w.x2_mm, w.y2_mm))
        for o in wires:
            if id(o) in seen:
                continue
            o_ends = ((o.x1_mm, o.y1_mm), (o.x2_mm, o.y2_mm))
            if any(
                abs(ox - wx) < EPS and abs(oy - wy) < EPS
                for (ox, oy) in o_ends
                for (wx, wy) in w_ends
            ):
                seen.add(id(o))
                comp.append(o)
                frontier.append(o)
    return comp


def _resolve_label_collisions(
    routed: RoutedSheet,
    body_rects: list[tuple[float, float, float, float]],
    all_pins: list[tuple[float, float, str, str]],
) -> None:
    """Nudge net/global labels whose text lands on a symbol body.

    Strategy per colliding label: (1) flip the reading direction -- the
    anchor stays on the wire, the text extends the other way; (2) slide the
    anchor along its OWN net's wires -- the connected component of the
    original anchor, never an arbitrary wire: with pin stubs one grid
    apart, landing on a neighboring stub both abandons this label's stub
    (dangling wire + unconnected pin) and names the neighbor's net (a
    silent net merge -- two labels on one wire is legal KiCad); (3) push
    the anchor one grid outward along its stub and extend the wire so the
    label stays electrically attached -- only when the new segment crosses
    no pin and its new end lands on no existing wire (either would merge
    nets); (4) leave it: a readable overlap beats a silent short.
    """
    wires = routed.wires

    def collides(rect: tuple[float, float, float, float]) -> bool:
        return any(_rects_overlap(rect, body) for body in body_rects)

    def seg_hits_pin(x1: float, y1: float, x2: float, y2: float) -> bool:
        return any(
            _pt_on_axis_seg(px, py, x1, y1, x2, y2) for (px, py, _r, _p) in all_pins
        )

    def on_wire(x: float, y: float) -> bool:
        return any(
            _pt_on_axis_seg(x, y, w.x1_mm, w.y1_mm, w.x2_mm, w.y2_mm) for w in wires
        )

    def other_label_anchors(lab) -> set[tuple[float, float]]:
        return {
            (round(l2.x_mm, 2), round(l2.y_mm, 2))
            for s2 in (routed.labels, routed.global_labels)
            for l2 in s2
            if l2 is not lab
        }

    slide_offsets = (
        (GRID_MM, 0.0), (-GRID_MM, 0.0), (0.0, GRID_MM), (0.0, -GRID_MM),
        (2 * GRID_MM, 0.0), (-2 * GRID_MM, 0.0), (0.0, 2 * GRID_MM), (0.0, -2 * GRID_MM),
    )

    for store in (routed.labels, routed.global_labels):
        for i, lab in enumerate(store):
            if not collides(_label_rect(lab.text, lab.x_mm, lab.y_mm, lab.angle_deg)):
                continue
            flipped = (lab.angle_deg + 180) % 360
            if not collides(_label_rect(lab.text, lab.x_mm, lab.y_mm, flipped)):
                store[i] = replace(lab, angle_deg=flipped)
                continue
            own_wires = _wire_component(lab.x_mm, lab.y_mm, wires)
            taken = other_label_anchors(lab)

            def on_own_wire(x: float, y: float) -> bool:
                return any(
                    _pt_on_axis_seg(x, y, w.x1_mm, w.y1_mm, w.x2_mm, w.y2_mm)
                    for w in own_wires
                )

            slid = False
            for dx, dy in slide_offsets:
                sx, sy = lab.x_mm + dx, lab.y_mm + dy
                if not on_own_wire(sx, sy) or (round(sx, 2), round(sy, 2)) in taken:
                    continue
                for angle in (lab.angle_deg, flipped):
                    if not collides(_label_rect(lab.text, sx, sy, angle)):
                        store[i] = replace(lab, x_mm=sx, y_mm=sy, angle_deg=angle)
                        slid = True
                        break
                if slid:
                    break
            if slid:
                continue
            outward = _LABEL_OUTWARD[lab.angle_deg % 360]
            nx, ny = step(lab.x_mm, lab.y_mm, outward, GRID_MM)
            if (
                not collides(_label_rect(lab.text, nx, ny, lab.angle_deg))
                and not seg_hits_pin(lab.x_mm, lab.y_mm, nx, ny)
                and not on_wire(nx, ny)
            ):
                wires.append(WireSegment(lab.x_mm, lab.y_mm, nx, ny))
                store[i] = replace(lab, x_mm=nx, y_mm=ny)


def _pt_on_axis_seg(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> bool:
    """True when point (px,py) lies on the axis-aligned segment (x1,y1)-(x2,y2)."""
    if abs(y1 - y2) < EPS and abs(py - y1) < EPS:  # horizontal
        return min(x1, x2) - EPS <= px <= max(x1, x2) + EPS
    if abs(x1 - x2) < EPS and abs(px - x1) < EPS:  # vertical
        return min(y1, y2) - EPS <= py <= max(y1, y2) + EPS
    return False


def _segs_touch(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> bool:
    """True when two axis-aligned segments share any point.

    Treats each segment as a degenerate rect, so it covers endpoint
    touches, T-joins, collinear overlap AND pure perpendicular crossings.
    A crossing without a junction does not electrically connect in KiCad,
    so this over-rejects slightly — the conservative direction for copper
    that must never meet."""
    return (
        min(ax1, ax2) <= max(bx1, bx2) + EPS
        and min(bx1, bx2) <= max(ax1, ax2) + EPS
        and min(ay1, ay2) <= max(by1, by2) + EPS
        and min(by1, by2) <= max(ay1, ay2) + EPS
    )


def _seg_touches_any_wire(
    x1: float, y1: float, x2: float, y2: float, wires: list[WireSegment]
) -> bool:
    return any(
        _segs_touch(x1, y1, x2, y2, w.x1_mm, w.y1_mm, w.x2_mm, w.y2_mm)
        for w in wires
    )


def _power_stub_clear(
    net_name: str,
    e: _Endpoint,
    ex: float,
    ey: float,
    own_pins: set[tuple[str, str]],
    all_pins: list[tuple[float, float, str, str]],
    power_stubs: list[tuple[str, float, float, float, float]],
    signal_wires: list[WireSegment] = (),
) -> bool:
    """True when the stub (e.x,e.y)->(ex,ey) can carry *net_name* safely.

    In KiCad a pin end or wire end that merely TOUCHES a wire connects to it,
    so a power stub that runs over a foreign pin -- or whose end meets another
    rail's stub -- silently merges two GLOBAL nets project-wide. The only ERC
    symptom is a baffling "Power output and Power output are connected"
    between the two rails' PWR_FLAGs (the run_05 VBUS+GND short: a GND stub
    stepped one grid right onto the neighbouring resistor's VBUS pin).

    ``signal_wires`` is the copper stamped by the signal pass (which routes
    first); a power segment touching ANY of it ties that signal net to the
    rail with only a multiple_net_names warning to show for it, so any touch
    rejects the stub.
    """
    for (px, py, ref, pin) in all_pins:
        if (ref, pin) in own_pins:
            continue
        if _pt_on_axis_seg(px, py, e.x, e.y, ex, ey):
            return False
    for net2, x1, y1, x2, y2 in power_stubs:
        if net2 == net_name:
            continue
        if (
            _pt_on_axis_seg(ex, ey, x1, y1, x2, y2)
            or _pt_on_axis_seg(x1, y1, e.x, e.y, ex, ey)
            or _pt_on_axis_seg(x2, y2, e.x, e.y, ex, ey)
        ):
            return False
    if _seg_touches_any_wire(e.x, e.y, ex, ey, signal_wires):
        return False
    return True


def _route_power(
    routed: RoutedSheet,
    net_name: str,
    eps: list[_Endpoint],
    flag_nets: frozenset[str],
    all_pins: list[tuple[float, float, str, str]],
    power_stubs: list[tuple[str, float, float, float, float]],
    signal_wires: list[WireSegment] = (),
) -> None:
    """A stub + power symbol (or global label) at every pin of a power net.

    Canonical orientation: rails point UP, grounds point DOWN, never
    sideways -- a sideways GND/VDD reads as a drawing error. A pin that
    exits horizontally gets an elbow (one grid out, one grid up/down) so
    its power symbol still stands upright; only when the elbow is blocked
    does the symbol fall back to lying along the straight stub.

    Each segment is collision-checked (see _power_stub_clear); a blocked
    straight stub retreats to half a grid step, and when even that collides
    the pin gets a global label at its own position instead -- the net stays
    named and connected with no copper stamped onto a foreign pin.
    """
    power_lib = power_symbol_for(net_name)
    is_gnd = _is_ground(net_name)
    angle_table = _GND_ANGLE if is_gnd else _RAIL_ANGLE
    own_pins = {(e.ref, e.pin) for e in eps}
    # (x, y, angle) for the one PWR_FLAG: placed on the wire but rotated
    # away from the rail symbol so the diamond doesn't stack on top of it.
    flag_spot: tuple[float, float, int] | None = None
    eps = _merge_stacked_power_pins(
        routed, net_name, eps, power_lib, own_pins, all_pins, power_stubs,
        signal_wires,
    )
    for e in eps:
        sym_xy: tuple[float, float] | None = None
        sym_angle = 0
        corner: tuple[float, float] | None = None
        if e.exit in ("left", "right") and power_lib is not None:
            ex, ey = step(e.x, e.y, e.exit, GRID_MM)
            vdir = "down" if is_gnd else "up"
            vx, vy = step(ex, ey, vdir, GRID_MM)
            corner_ep = _Endpoint(ex, ey, vdir, e.ref, e.pin)
            if _power_stub_clear(
                net_name, e, ex, ey, own_pins, all_pins, power_stubs,
                signal_wires,
            ) and _power_stub_clear(
                net_name, corner_ep, vx, vy, own_pins, all_pins, power_stubs,
                signal_wires,
            ):
                routed.wires.append(WireSegment(e.x, e.y, ex, ey))
                routed.wires.append(WireSegment(ex, ey, vx, vy))
                power_stubs.append((net_name, e.x, e.y, ex, ey))
                power_stubs.append((net_name, ex, ey, vx, vy))
                sym_xy = (vx, vy)
                sym_angle = angle_table[vdir]
                corner = (ex, ey)
        if sym_xy is None:
            ex, ey = step(e.x, e.y, e.exit, GRID_MM)
            if not _power_stub_clear(
                net_name, e, ex, ey, own_pins, all_pins, power_stubs,
                signal_wires,
            ):
                ex, ey = step(e.x, e.y, e.exit, GRID_MM / 2)
                if not _power_stub_clear(
                    net_name, e, ex, ey, own_pins, all_pins, power_stubs,
                    signal_wires,
                ):
                    routed.global_labels.append(GlobalLabel(
                        text=net_name, x_mm=e.x, y_mm=e.y,
                        angle_deg=_LABEL_ANGLE[e.exit]))
                    continue
            routed.wires.append(WireSegment(e.x, e.y, ex, ey))
            power_stubs.append((net_name, e.x, e.y, ex, ey))
            sym_xy = (ex, ey)
            sym_angle = angle_table[e.exit]
        if power_lib is not None:
            routed.power_symbols.append(PowerSymbol(
                lib_id=power_lib, x_mm=sym_xy[0], y_mm=sym_xy[1],
                angle_deg=sym_angle))
        else:
            routed.global_labels.append(GlobalLabel(
                text=net_name, x_mm=sym_xy[0], y_mm=sym_xy[1],
                angle_deg=_LABEL_ANGLE[e.exit]))
        if flag_spot is None:
            if corner is not None:
                # The elbow corner is on the wire and clear of the symbol;
                # point the diamond away from the vertical run.
                flag_spot = (corner[0], corner[1], 0 if is_gnd else 180)
            else:
                flag_spot = (sym_xy[0], sym_xy[1], 90)
    # One PWR_FLAG per undriven rail marks it as driven for ERC.
    if net_name in flag_nets and flag_spot is not None:
        routed.power_symbols.append(PowerSymbol(
            lib_id="power:PWR_FLAG", x_mm=flag_spot[0], y_mm=flag_spot[1],
            angle_deg=flag_spot[2]))


def _merge_stacked_power_pins(
    routed: RoutedSheet,
    net_name: str,
    eps: list[_Endpoint],
    power_lib: str | None,
    own_pins: set[tuple[str, str]],
    all_pins: list[tuple[float, float, str, str]],
    power_stubs: list[tuple[str, float, float, float, float]],
    signal_wires: list[WireSegment] = (),
) -> list[_Endpoint]:
    """Collapse a vertical stack of same-net pins (e.g. a connector's four
    shield/EH pins) into one rail/ground drop.

    Four stacked pins each growing their own symbol produces four
    overlapping triangles; a human ties the stubs with one vertical wire
    and drops a single symbol at its end. A group is: same horizontal
    exit, same x, consecutive pins at most two grid steps apart. On any
    collision the group falls back to per-pin symbols.
    """
    if power_lib is None or len(eps) < 2:
        return eps
    horiz = [e for e in eps if e.exit in ("left", "right")]
    rest = [e for e in eps if e.exit not in ("left", "right")]
    by_col: dict[tuple[str, float], list[_Endpoint]] = {}
    for e in horiz:
        by_col.setdefault((e.exit, round(e.x, 2)), []).append(e)
    merged: list[_Endpoint] = list(rest)
    for col in by_col.values():
        if len(col) < 2:
            merged.extend(col)
            continue
        col = sorted(col, key=lambda e: e.y)
        if any(b.y - a.y > 2 * GRID_MM + EPS for a, b in zip(col, col[1:])):
            merged.extend(col)
            continue
        stub_ends: list[tuple[_Endpoint, float, float]] = []
        ok = True
        for e in col:
            ex, ey = step(e.x, e.y, e.exit, GRID_MM)
            if not _power_stub_clear(
                net_name, e, ex, ey, own_pins, all_pins, power_stubs,
                signal_wires,
            ):
                ok = False
                break
            stub_ends.append((e, ex, ey))
        bus_x = stub_ends[0][1] if stub_ends else 0.0
        top_y = stub_ends[0][2] if stub_ends else 0.0
        bot_y = stub_ends[-1][2] if stub_ends else 0.0
        if ok:
            run_ep = _Endpoint(bus_x, top_y, "down", col[0].ref, col[0].pin)
            ok = _power_stub_clear(
                net_name, run_ep, bus_x, bot_y, own_pins, all_pins, power_stubs,
                signal_wires,
            )
        if not ok:
            merged.extend(col)
            continue
        for e, ex, ey in stub_ends:
            routed.wires.append(WireSegment(e.x, e.y, ex, ey))
            power_stubs.append((net_name, e.x, e.y, ex, ey))
        # The vertical run is emitted as one segment per gap, meeting the
        # stub ends endpoint-to-endpoint: kicad-cli 9 ERC does not bind a
        # wire end that tees into another wire's mid-span, even under a
        # junction dot (verified empirically -- a single full-height run
        # left every mid-stack pin "not connected").
        for (_ea, _ax, ay), (_eb, _bx, by) in zip(stub_ends, stub_ends[1:]):
            routed.wires.append(WireSegment(bus_x, ay, bus_x, by))
            power_stubs.append((net_name, bus_x, ay, bus_x, by))
        # Mid-stack meeting points carry three wire ends: junction dots.
        for _e, ex, ey in stub_ends[1:-1]:
            routed.junctions.append(Junction(x_mm=ex, y_mm=ey))
        if _is_ground(net_name):
            anchor, ay = col[-1], bot_y
            exit_dir = "down"
        else:
            anchor, ay = col[0], top_y
            exit_dir = "up"
        merged.append(_Endpoint(bus_x, ay, exit_dir, anchor.ref, anchor.pin))
    return merged


def _safe_link(
    a: _Endpoint, b: _Endpoint, all_pins: list[tuple[float, float, str, str]],
    wires: list[WireSegment] = (),
) -> list[WireSegment] | None:
    """A straight or single-corner wire between pins ``a`` and ``b`` that
    passes through no foreign pin and touches no already-stamped wire (so
    it can't short two nets), or None."""
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
        return not _seg_touches_any_wire(x1, y1, x2, y2, wires)

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
