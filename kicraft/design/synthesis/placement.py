"""Sheet-level part placement for KiCraft Stage B — cluster placer.

Lays each leaf out the way a human engineer would, instead of dumping
every part in one row connected by net-label salad:

- **Anchors** (ICs / multi-pin parts) sit on a center line, ordered left
  to right by ``signal_flow_order`` (input → output).
- Each anchor's **2-pin passives** are clustered right next to the pin
  they serve and rotated so the pin facing the anchor connects back to it
  while the *far* pin points into open space — where a power symbol, a
  ground symbol, or a label/wire attaches cleanly (the user's rule:
  "place the resistor near the pin, rotated away from it").
    * decoupling / bulk caps → a tidy row ABOVE the anchor, rail pin up,
      ground pin down (power top, ground bottom — the convention).
    * pull-ups → above (rail up, signal pin down toward the anchor).
    * pull-downs → below (ground down, signal up toward the anchor).
    * series / feedback → pin-aligned just outside the served pin, in the
      signal-flow direction, far pin pointing onward.
- Roles come from ``bom.placement_hints`` when present; otherwise they're
  inferred from ``bom.connections`` (a cap on rail+gnd is decoupling, a
  resistor on rail+signal is a pull-up, …) so the placer works on any
  state, hinted or not.
- Parts that don't cluster (lone connectors, unassignable passives) go in
  a spaced row below everything, connected by labels.

Everything snaps to the 1.27 mm grid and is deterministic. Connectivity
is still owned by ``router`` — placement only decides positions and
rotations; both share ``sch_geometry`` so the router finds the pins of a
rotated part. Where a clean placement can't be found the part falls back
to the free row, so the sheet is always emittable and ERC-clean.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, replace

from ..models import BOM, BomPart, Sheet, is_power_or_ground_name
from .sch_geometry import (
    pin_abs_position,
    pin_exit_direction,
    rotate_vec,
    rotation_for_exit,
)
from .symbol_pinout import SymbolNotFoundError, lookup_pins

GRID_MM = 2.54
HALF_GRID_MM = 1.27

# A4 portrait usable region; the anchor center line sits in the upper third
# so cluster rows above/below and the free row below all fit. Every layout
# constant is a multiple of 1.27 mm so derived pins stay on the connection grid.
CENTER_Y_MM = 95.25        # 75 * 1.27
USABLE_LEFT_MM = 30.48     # 24 * 1.27
ANCHOR_GAP_MM = 20.32      # 16 * 1.27 — clear gap between adjacent clusters

# Distance from an anchor's pin bounding box to the cluster row above/below.
# Generous so the anchor's reference/value text, the member values, and the
# rail/ground symbols (which hang ~5 mm off a pin and carry a net-name label)
# all have clear air around them — the readability the whole rewrite is for.
ROW_GAP_MM = 19.05         # 15 * 1.27
MEMBER_PITCH_MM = 12.7     # 10 * 1.27 — column pitch within a cluster row
SIDE_PITCH_MM = 10.16      # 8 * 1.27 — row pitch for stacked side members
# Gap from the served pin to a pin-aligned member's near pin (room for its
# connecting wire, net label, and the rail/ground symbol beyond it).
PIN_LINK_GAP_MM = 10.16    # 8 * 1.27
FREE_PITCH_MM = 25.4       # spacing in the un-clustered fallback row

# An array (e.g. an LED matrix) is laid out as a 2D grid that mirrors its
# physical shape, so the schematic reads as a matrix instead of a 50-long row.
# Member symbols are small; these pitches leave room for the inter-member wires
# and the per-member power/ground stubs while keeping the grid on one sheet.
ARRAY_TOP_Y_MM = 38.1      # 30 * 1.27 — grid top; clusters drop below it
ARRAY_COL_PITCH_MM = 15.24  # 12 * 1.27
ARRAY_ROW_PITCH_MM = 15.24  # 12 * 1.27


def _snap(value: float, grid: float = HALF_GRID_MM) -> float:
    return round(value / grid) * grid


def _place_array_grid(
    spec, left_x: float, top_y: float, placed: dict[str, PlacedPart]
) -> tuple[float, float]:
    """Lay an array's members on a serpentine ``rows`` x ``cols`` grid (the same
    data-chain order the PCB grid uses, so the schematic shape matches the
    board). A ``ring`` array has no rows/cols — the SCHEMATIC still reads best
    as a near-square serpentine matrix (a circle of symbols wastes sheet and
    tangles the chain wires), so one is synthesized; only the PCB is circular.
    Returns ``(width, bottom)`` of the grid block."""
    n = len(spec.refs)
    if getattr(spec, "pattern", "grid") == "ring" or spec.cols is None:
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = max(1, -(-n // cols))
    else:
        cols = max(1, int(spec.cols))
        rows = max(1, int(spec.rows))
    serpentine = bool(getattr(spec, "serpentine", True))
    for i, ref in enumerate(spec.refs):
        row, col = divmod(i, cols)
        if serpentine and row % 2 == 1:
            col = cols - 1 - col  # boustrophedon: consecutive members adjacent
        placed[ref] = PlacedPart(
            ref=ref,
            x_mm=_snap(left_x + col * ARRAY_COL_PITCH_MM),
            y_mm=_snap(top_y + row * ARRAY_ROW_PITCH_MM),
            rotation_deg=0, mirror=None, role="array_member",
        )
    return cols * ARRAY_COL_PITCH_MM, _snap(top_y + rows * ARRAY_ROW_PITCH_MM)


@dataclass(frozen=True)
class PlacedPart:
    ref: str
    x_mm: float
    y_mm: float
    rotation_deg: int   # 0 | 90 | 180 | 270
    mirror: str | None  # None | "x" | "y" (unused in v1, kept for the emitter)
    role: str           # anchor | decoupling | pullup | ... | free
    unit: int = 1       # symbol unit this placement draws (dual op-amp B = 2)


@dataclass(frozen=True)
class _Pins:
    """Cached pin list + bounding box for one part, at rotation 0."""
    by_number: dict[str, dict]
    count: int


@dataclass
class _Anchor:
    ref: str
    part: BomPart
    pins: _Pins
    x: float = 0.0
    y: float = 0.0
    members: list["_Member"] = field(default_factory=list)


@dataclass
class _Member:
    ref: str
    part: BomPart
    pins: _Pins
    anchor_ref: str
    role: str
    near_pin: str            # passive pin toward the anchor / its served net
    far_pin: str             # the other passive pin
    served_pin: str | None   # anchor pin it sits beside (placement target)
    # For row members (decoupling/pullup/pulldown): which pin points OUT to its
    # rail/ground symbol (up in a rail row, down in a ground row) and which
    # points back toward the anchor.
    outer_pin: str | None = None
    inner_pin: str | None = None


def _load_units(part: BomPart) -> dict[int, _Pins]:
    """Pin inventory per functional unit. ``{1: pins}`` for the common
    single-unit case; a multi-section symbol (dual op-amp, quad gate) gets
    one entry per unit so each section places — and therefore draws and
    wires — as its own entity. Unit-0 (shared power) pins ride with unit 1."""
    try:
        pins = lookup_pins(part.symbol, all_units=True)["pins"]
    except (SymbolNotFoundError, ValueError, KeyError) as exc:
        # Loud, not silent: a part placed with unresolvable pins ends up on
        # the sheet unwired with zero diagnostic pointing at the cause
        # (2026-07-19 review §4.5).
        print(
            f"[placement] WARNING: cannot resolve pins for {part.ref} "
            f"({part.symbol}): {exc} -- part will place but cannot cluster "
            "or wire",
            file=sys.stderr,
        )
        pins = []
    by_unit: dict[int, list[dict]] = {}
    for p in pins:
        by_unit.setdefault(int(p.get("unit", 1) or 1), []).append(p)
    if not by_unit:
        by_unit[1] = []
    return {
        u: _Pins(by_number={p["number"]: p for p in ps}, count=len(ps))
        for u, ps in sorted(by_unit.items())
    }


def _is_gnd(net: str) -> bool:
    n = net.lstrip("/").upper()
    return is_power_or_ground_name(net) and ("GND" in n or n in {"VSS", "VEE"})


def place_sheet(
    sheet: Sheet,
    sheet_parts: list[BomPart],
    bom: BOM,
) -> list[PlacedPart]:
    """Place every part on a sheet. Returns one PlacedPart per input part,
    in the same order as ``sheet_parts``, followed by one entry per EXTRA
    unit of any multi-section symbol (``unit`` >= 2, same ``ref``)."""
    if not sheet_parts:
        return []

    parts_by_ref = {p.ref: p for p in sheet_parts}

    # Multi-unit expansion: each functional unit is its own placeable
    # entity. Unit 1 keeps the part's own ref (so the single-unit path is
    # byte-identical); units >= 2 get a synthetic internal ref that is
    # folded back to (ref, unit) on return.
    entity_parts: list[tuple[str, BomPart, int]] = []
    pins_by_ref: dict[str, _Pins] = {}
    pin_entity: dict[tuple[str, str], str] = {}
    for p in sheet_parts:
        for unit, pins in _load_units(p).items():
            eref = p.ref if unit == 1 else f"{p.ref}#u{unit}"
            entity_parts.append((eref, p, unit))
            pins_by_ref[eref] = pins
            for pn in pins.by_number:
                pin_entity[(p.ref, pn)] = eref

    # pin -> net and net -> endpoints, restricted to this sheet's parts and
    # keyed by the entity that owns the pin (a dual op-amp's unit-B pins
    # belong to the unit-B entity).
    pin_net: dict[tuple[str, str], str] = {}
    net_endpoints: dict[str, list[tuple[str, str]]] = {}
    for conn in bom.connections:
        for ep in conn.endpoints:
            if ep.ref not in parts_by_ref:
                continue
            eref = pin_entity.get((ep.ref, ep.pin), ep.ref)
            pin_net.setdefault((eref, ep.pin), conn.net_name)
            net_endpoints.setdefault(conn.net_name, []).append((eref, ep.pin))

    # Classify each net as gnd / rail / signal. A rail is recognised by name OR
    # by touching a power pin — so a non-standard rail name (e.g. the CH340G's
    # "V3" output) is still treated as a rail and its cap clusters correctly.
    def _classify(net: str) -> str:
        if _is_gnd(net):
            return "gnd"
        if is_power_or_ground_name(net):
            return "rail"
        for (ref, pn) in net_endpoints.get(net, []):
            pin = pins_by_ref[ref].by_number.get(pn)
            if pin and pin["electrical_type"] in ("power_in", "power_out"):
                return "rail"
        return "signal"
    net_kind_map = {net: _classify(net) for net in net_endpoints}

    hints_by_ref = {h.ref: h for h in bom.placement_hints if h.ref in parts_by_ref}
    member_to_group_anchor: dict[str, str] = {}
    for ic, members in bom.ic_groups.items():
        if ic in parts_by_ref:
            for m in members:
                if m in parts_by_ref:
                    member_to_group_anchor[m] = ic

    # Arrays (e.g. an LED matrix) on this sheet: their members are laid out as a
    # 2D grid that mirrors the array shape, NOT strung in the anchor row. Only
    # arrays whose every ref is on this sheet are honoured.
    array_specs = [
        s for s in getattr(bom, "arrays", [])
        if s.refs and all(r in parts_by_ref for r in s.refs)
    ]
    array_member_refs = {r for s in array_specs for r in s.refs}

    # Anchors: 3+ pins, or named as an ic_groups leader. Array members are laid
    # out by the grid, never as anchors. Extra units (>= 2) are always anchors:
    # they are sections of an IC, whatever their pin count (a TL072's power
    # unit has 2), and must not fall into the passive-member machinery.
    anchor_refs = {
        eref for (eref, p, unit) in entity_parts
        if (pins_by_ref[eref].count >= 3 or p.ref in bom.ic_groups or unit > 1)
        and p.ref not in array_member_refs
    }
    anchors: dict[str, _Anchor] = {
        eref: _Anchor(ref=eref, part=p, pins=pins_by_ref[eref])
        for (eref, p, unit) in entity_parts
        if eref in anchor_refs
    }

    placed: dict[str, PlacedPart] = {}
    free_refs: list[str] = []

    # --- array grids first, in the upper region; clusters drop below them ---
    grid_bottom = ARRAY_TOP_Y_MM
    grid_x = USABLE_LEFT_MM
    for spec in array_specs:
        width, bottom = _place_array_grid(spec, grid_x, ARRAY_TOP_Y_MM, placed)
        grid_x += width + ANCHOR_GAP_MM
        grid_bottom = max(grid_bottom, bottom)
    anchor_base_y = (
        _snap(grid_bottom + ROW_GAP_MM) if array_specs else CENTER_Y_MM
    )

    # --- assign each 2-pin passive to an anchor + role ---
    for (eref, part, unit) in entity_parts:
        if eref in anchor_refs or part.ref in array_member_refs:
            continue
        pins = pins_by_ref[eref]
        if pins.count != 2:
            free_refs.append(eref)
            continue
        member = _assign_member(
            part, pins, anchors, hints_by_ref.get(part.ref),
            member_to_group_anchor, pin_net, net_endpoints, anchor_refs,
            net_kind_map,
        )
        if member is None:
            free_refs.append(eref)
        else:
            anchors[member.anchor_ref].members.append(member)

    # --- order + place anchors left to right ---
    # Extra units sort right after their unit-1 sibling (same flow slot,
    # unit as the tiebreak) so a dual op-amp's B section sits beside A.
    entity_meta = {eref: (p.ref, unit) for (eref, p, unit) in entity_parts}
    flow_index = {ref: i for i, ref in enumerate(bom.signal_flow_order)}
    ordered = sorted(
        anchors.values(),
        key=lambda a: (
            flow_index.get(entity_meta[a.ref][0], len(flow_index)),
            entity_meta[a.ref][0],
            entity_meta[a.ref][1],
        ),
    )

    cluster_bottom = anchor_base_y
    cursor_x = USABLE_LEFT_MM
    for anchor in ordered:
        width, bottom = _place_cluster(
            anchor, cursor_x, placed, pin_net, base_y=anchor_base_y
        )
        cursor_x += width + ANCHOR_GAP_MM
        cluster_bottom = max(cluster_bottom, bottom)

    # --- free parts: a spaced row below every cluster (label-connected) ---
    free_y = _snap(cluster_bottom + ROW_GAP_MM + GRID_MM)
    fx = USABLE_LEFT_MM
    for ref in sorted(free_refs):
        placed[ref] = PlacedPart(
            ref=ref, x_mm=_snap(fx), y_mm=free_y,
            rotation_deg=0, mirror=None, role="free",
        )
        fx += FREE_PITCH_MM

    # Unit-1 entries first (one per part, input order — the emitter indexes
    # these positionally), then the extra units folded back to (ref, unit).
    result = [placed[p.ref] for p in sheet_parts]
    for (eref, p, unit) in entity_parts:
        if unit == 1 or eref not in placed:
            continue
        result.append(replace(placed[eref], ref=p.ref, unit=unit))
    return result


def _other_pin(pins: _Pins, pin: str) -> str:
    for n in pins.by_number:
        if n != pin:
            return n
    return pin


def _assign_member(
    part: BomPart,
    pins: _Pins,
    anchors: dict[str, _Anchor],
    hint,
    member_to_group_anchor: dict[str, str],
    pin_net: dict[tuple[str, str], str],
    net_endpoints: dict[str, list[tuple[str, str]]],
    anchor_refs: set[str],
    net_kind_map: dict[str, str],
) -> _Member | None:
    """Decide which anchor this passive clusters with and its role/orientation.
    Returns None if it can't be cleanly clustered (-> free row)."""
    pin_numbers = sorted(pins.by_number)
    if len(pin_numbers) != 2:
        return None
    pa, pb = pin_numbers
    net_a = pin_net.get((part.ref, pa))
    net_b = pin_net.get((part.ref, pb))
    kind_a = net_kind_map.get(net_a, "none") if net_a else "none"
    kind_b = net_kind_map.get(net_b, "none") if net_b else "none"

    def anchor_on(net: str | None) -> tuple[str, str] | None:
        """First (anchor_ref, anchor_pin) on ``net``, ic-group anchor preferred."""
        if net is None:
            return None
        eps = [(r, pn) for (r, pn) in net_endpoints.get(net, []) if r in anchor_refs]
        if not eps:
            return None
        pref = member_to_group_anchor.get(part.ref)
        for r, pn in eps:
            if r == pref:
                return (r, pn)
        return eps[0]

    role = hint.role if hint else None

    # Identify, by net kind, which pin is the "signal" side and which is the
    # rail/ground side. For a cap on rail+gnd both are power; the rail pin is
    # the near (served) pin.
    rail_pin = gnd_pin = sig_pin = None
    for pn, kind in ((pa, kind_a), (pb, kind_b)):
        if kind == "rail" and rail_pin is None:
            rail_pin = pn
        elif kind == "gnd" and gnd_pin is None:
            gnd_pin = pn
        elif kind == "signal" and sig_pin is None:
            sig_pin = pn

    if role is None:
        if kind_a in ("rail", "gnd") and kind_b in ("rail", "gnd"):
            role = "decoupling"
        elif "gnd" in (kind_a, kind_b) and "signal" in (kind_a, kind_b):
            role = "pulldown"
        elif "rail" in (kind_a, kind_b) and "signal" in (kind_a, kind_b):
            role = "pullup"
        elif kind_a == "signal" and kind_b == "signal":
            role = "series"
        else:
            role = "other"

    def net_of(pin: str | None) -> str | None:
        if pin is None:
            return None
        return net_a if pin == pa else net_b

    # Served net / anchor / near + outer/inner pins per role. ``outer`` points
    # away to a rail/ground symbol; ``inner`` points back toward the anchor.
    served: tuple[str, str] | None = None
    near_pin = far_pin = outer_pin = inner_pin = None
    if role in ("decoupling", "bulk"):
        # A bypass cap hangs between a rail (up) and ground (down), hugging the
        # anchor power pin it decouples.
        outer_pin = rail_pin or pa            # rail pin -> points up
        inner_pin = gnd_pin or _other_pin(pins, outer_pin)
        near_pin, far_pin = inner_pin, outer_pin
        served = anchor_on(net_of(rail_pin)) or anchor_on(net_of(gnd_pin))
    elif role == "pullup":
        outer_pin = rail_pin or pa            # rail up
        inner_pin = sig_pin or _other_pin(pins, outer_pin)
        near_pin, far_pin = inner_pin, outer_pin
        served = anchor_on(net_of(sig_pin))
    elif role == "pulldown":
        outer_pin = gnd_pin or pa             # ground down
        inner_pin = sig_pin or _other_pin(pins, outer_pin)
        near_pin, far_pin = inner_pin, outer_pin
        served = anchor_on(net_of(sig_pin))
    else:  # series / feedback / other — pin-aligned to the anchor pin it touches
        if anchor_on(net_a):
            near_pin, served = pa, anchor_on(net_a)
        elif anchor_on(net_b):
            near_pin, served = pb, anchor_on(net_b)
        else:
            near_pin = pa
        far_pin = _other_pin(pins, near_pin)

    anchor_ref = None
    if hint and hint.anchor_ref in anchors:
        anchor_ref = hint.anchor_ref
    elif served is not None:
        anchor_ref = served[0]
    elif part.ref in member_to_group_anchor:
        anchor_ref = member_to_group_anchor[part.ref]
    if anchor_ref is None or anchor_ref not in anchors:
        return None

    served_pin = (hint.anchor_pin if hint and hint.anchor_pin else
                  (served[1] if served and served[0] == anchor_ref else None))

    return _Member(
        ref=part.ref, part=part, pins=pins, anchor_ref=anchor_ref, role=role,
        near_pin=near_pin or pa, far_pin=far_pin or pb, served_pin=served_pin,
        outer_pin=outer_pin, inner_pin=inner_pin,
    )


LEFT_RESERVE_MM = 19.05     # 15 * 1.27 — room left of the anchor for L/R members
CLUSTER_MARGIN_MM = 6.35    # 5 * 1.27 — gap for the symbols/labels off cluster pins


def _place_cluster(
    anchor: _Anchor,
    left_x: float,
    placed: dict[str, PlacedPart],
    pin_net: dict[tuple[str, str], str],
    base_y: float = CENTER_Y_MM,
) -> tuple[float, float]:
    """Place an anchor and its members. Returns (cluster_width, cluster_bottom).

    Decoupling/bulk caps go in a tidy rail row above the IC (rail up, ground
    down) since they connect by power symbol. Pull-ups, pull-downs, and series
    parts are placed pin-aligned at a stub off the exact pin they serve, so the
    wire leaves that pin in its own exit direction and routes cleanly. The
    whole cluster is then shifted so its leftmost element lands at ``left_x``.

    ``base_y`` is the anchor center line; it drops below an array grid when the
    sheet carries one so the grid and the clusters don't overlap.
    """
    # Anchor body top (rotation 0, sheet +y down) — the rail row hangs above it.
    pins = list(anchor.pins.by_number.values())
    body_top = min((-p["position"]["y"] for p in pins), default=0.0)

    # Provisional origin; the cluster is shifted into place afterwards.
    anchor.x = _snap(left_x + LEFT_RESERVE_MM)
    anchor.y = base_y
    placed[anchor.ref] = PlacedPart(
        ref=anchor.ref, x_mm=anchor.x, y_mm=anchor.y,
        rotation_deg=0, mirror=None, role="anchor",
    )

    rail_row = [m for m in anchor.members if m.role in ("decoupling", "bulk")]
    attached = [m for m in anchor.members
                if m.role in ("pullup", "pulldown", "series", "feedback", "other")]

    if rail_row:
        row_y = _snap(anchor.y + body_top - ROW_GAP_MM)
        _place_rail_row(rail_row, anchor, row_y, placed)

    by_side: dict[str, list[_Member]] = {}
    for m in attached:
        by_side.setdefault(_served_dir(m, anchor), []).append(m)
    for side, members in by_side.items():
        _place_side_group(members, anchor, side, placed)

    # Shift the whole cluster so its leftmost pin clears left_x by the margin.
    refs = [anchor.ref] + [m.ref for m in anchor.members if m.ref in placed]
    min_x, max_x, max_y = _cluster_bounds(refs, anchor, placed)
    shift = _snap(left_x + CLUSTER_MARGIN_MM - min_x)
    if abs(shift) > 1e-6:
        for ref in refs:
            pp = placed[ref]
            placed[ref] = PlacedPart(
                ref=pp.ref, x_mm=pp.x_mm + shift, y_mm=pp.y_mm,
                rotation_deg=pp.rotation_deg, mirror=pp.mirror, role=pp.role,
            )
        anchor.x += shift
        max_x += shift

    width = (max_x + CLUSTER_MARGIN_MM) - left_x
    cluster_bottom = max_y + CLUSTER_MARGIN_MM
    return width, cluster_bottom


def _cluster_bounds(
    refs: list[str], anchor: _Anchor, placed: dict[str, PlacedPart]
) -> tuple[float, float, float]:
    """Min x / max x / max y over every pin of every part in the cluster."""
    pins_by_ref = {anchor.ref: anchor.pins}
    for m in anchor.members:
        pins_by_ref[m.ref] = m.pins
    xs: list[float] = []
    ys: list[float] = []
    for ref in refs:
        pp = placed[ref]
        for pin in pins_by_ref[ref].by_number.values():
            x, y = pin_abs_position(pp.x_mm, pp.y_mm, pp.rotation_deg, pin)
            xs.append(x)
            ys.append(y)
    if not xs:
        return (anchor.x, anchor.x, anchor.y)
    return (min(xs), max(xs), max(ys))


def _place_rail_row(
    members: list[_Member],
    anchor: _Anchor,
    row_y: float,
    placed: dict[str, PlacedPart],
) -> None:
    """A row of vertical bypass caps above the IC, each over the power pin it
    serves (input cap over VIN, output cap over VOUT, …), rail pin up / ground
    pin down, nudged right to keep a clear pitch between neighbours."""
    def desired_x(m: _Member) -> float:
        if m.served_pin and m.served_pin in anchor.pins.by_number:
            sx, _ = pin_abs_position(
                anchor.x, anchor.y, 0, anchor.pins.by_number[m.served_pin])
            return sx
        return anchor.x

    prev_x = None
    for m in sorted(members, key=lambda m: (desired_x(m), m.ref)):
        want = desired_x(m)
        ox = want if prev_x is None else max(want, prev_x + MEMBER_PITCH_MM)
        prev_x = ox
        outer = m.pins.by_number.get(m.outer_pin or m.far_pin)
        rot = rotation_for_exit(outer, "up") if outer else 0
        placed[m.ref] = PlacedPart(
            ref=m.ref, x_mm=_snap(ox), y_mm=_snap(row_y),
            rotation_deg=rot, mirror=None, role=m.role,
        )


def _served_dir(m: _Member, anchor: _Anchor) -> str:
    """Exit direction of the anchor pin a member serves (default right)."""
    if m.served_pin and m.served_pin in anchor.pins.by_number:
        return pin_exit_direction(0, anchor.pins.by_number[m.served_pin])
    return "right"


def _place_side_group(
    members: list[_Member],
    anchor: _Anchor,
    side: str,
    placed: dict[str, PlacedPart],
) -> None:
    """Place all attached members on one side of the anchor (the pull-ups,
    pull-downs, and series parts whose served pins exit that way).

    Each member's near pin taps its served pin (a stub ``gap`` out in the
    pin's exit direction) and its far pin points further out — so a rail or
    ground symbol on the far pin never lands back on the IC. Members are
    spread along the side (in y for left/right, in x for top/bottom) with a
    guaranteed pitch, so two parts on adjacent IC pins can't overlap; the
    short offset wire back to each pin is drawn by the router.
    """
    stack_y = side in ("left", "right")

    def served_xy(m: _Member) -> tuple[float, float]:
        if m.served_pin and m.served_pin in anchor.pins.by_number:
            return pin_abs_position(
                anchor.x, anchor.y, 0, anchor.pins.by_number[m.served_pin])
        return (anchor.x, anchor.y)

    ordered = sorted(
        members,
        key=lambda m: (served_xy(m)[1] if stack_y else served_xy(m)[0], m.ref))
    prev = None
    for m in ordered:
        sx, sy = served_xy(m)
        if stack_y:
            nx = sx + (PIN_LINK_GAP_MM if side == "right" else -PIN_LINK_GAP_MM)
            ny = sy if prev is None else max(sy, prev + SIDE_PITCH_MM)
            prev, ntx, nty = ny, nx, ny
        else:
            ny = sy + (PIN_LINK_GAP_MM if side == "down" else -PIN_LINK_GAP_MM)
            nx = sx if prev is None else max(sx, prev + SIDE_PITCH_MM)
            prev, ntx, nty = nx, nx, ny
        near = (m.pins.by_number.get(m.near_pin)
                or next(iter(m.pins.by_number.values())))
        # Near pin faces back toward the anchor; far pin points outward.
        rot = rotation_for_exit(near, _opp(side))
        rx, ry = rotate_vec(near["position"]["x"], near["position"]["y"], rot)
        placed[m.ref] = PlacedPart(
            ref=m.ref, x_mm=_snap(ntx - rx), y_mm=_snap(nty + ry),
            rotation_deg=rot, mirror=None, role=m.role,
        )


def _opp(d: str) -> str:
    return {"left": "right", "right": "left", "up": "down", "down": "up"}[d]
