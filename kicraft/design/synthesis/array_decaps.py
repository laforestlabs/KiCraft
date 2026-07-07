"""Deterministic BOM rule: thin per-LED decoupling caps to a couple of bulk caps
for a LOW-current LED array.

A per-LED decoupling cap (one bypass cap beside every addressable LED) is the
textbook layout, but it costs real board area and parts. Below a modest total
array current a couple of distributed caps decouple the string just as well, so
for a small/low-current array we drop the extras and keep a few.

This is a SYNTHESIS-stage decision, not a placement one: the placer cannot remove
parts (they are in the netlist; deleting one there would orphan its pins). Here we
own the netlist, so we drop each excess cap AND scrub its ref from every BOM
cross-reference, leaving a self-consistent BOM the §9 validation gates accept.

It is LOUD and recorded in ``bom.assumptions`` -- dropping designer/LLM parts is a
visible engineering decision, never a silent degrade (see
kicraft-no-fallbacks-fail-loudly). High-current arrays (a 50-LED+ WS2812 string at
~3 A or more) stay fully per-LED-decoupled.
"""
from __future__ import annotations
from collections import defaultdict

from kicraft.design.models import (
    BOM,
    Architecture,
    InterSheetNet,
    NetConnection,
    Sheet,
    SheetPin,
    is_power_or_ground_name,
)

# WS2812-class addressable LED: ~3x20 mA = 60 mA per device at full white. We use
# this conservative MAX -- overestimating current makes the rule LESS likely to
# drop caps, which is the safe direction.
ARRAY_DECAP_PER_LED_MA = 60.0
# Total array current below which per-LED decoupling is overkill (a couple of bulk
# caps suffice). 3000 mA ~= a 50-LED string. This is the user's deliberate call:
# below 3 A, per-LED bypass caps aren't worth their board area + parts cost, so a
# small/medium array thins to a couple of bulk caps. A 50-LED+ string (>=3 A) keeps
# full per-LED decoupling.
ARRAY_DECAP_BULK_THRESHOLD_MA = 3000.0
# How many caps to keep as distributed bulk decoupling.
ARRAY_DECAP_BULK_KEEP = 2


def _ref_sort_key(ref: str) -> tuple[str, int]:
    digits = "".join(c for c in ref if c.isdigit())
    return ("".join(c for c in ref if not c.isdigit()), int(digits) if digits else 0)


def _scrub_refs(bom: BOM, dropped: set[str]) -> None:
    """Remove every trace of *dropped* refs from the BOM so it re-validates."""
    bom.parts = [p for p in bom.parts if p.ref not in dropped]

    kept_conns = []
    for c in bom.connections:
        eps = [e for e in c.endpoints if e.ref not in dropped]
        if eps:  # a NetConnection must keep >=1 endpoint; an emptied net is dropped
            c.endpoints = eps
            kept_conns.append(c)
    bom.connections = kept_conns

    bom.no_connect_pins = [e for e in bom.no_connect_pins if e.ref not in dropped]
    bom.thermal_refs = [r for r in bom.thermal_refs if r not in dropped]
    bom.signal_flow_order = [r for r in bom.signal_flow_order if r not in dropped]
    bom.component_zones = {
        r: v for r, v in bom.component_zones.items() if r not in dropped
    }
    # A hint whose subject is dropped goes too; a hint that merely ANCHORS to a
    # dropped ref keeps, with its anchor cleared (the placer falls back to the net).
    kept_hints = []
    for h in bom.placement_hints:
        if h.ref in dropped:
            continue
        if h.anchor_ref in dropped:
            h.anchor_ref = None
            h.anchor_pin = None
        kept_hints.append(h)
    bom.placement_hints = kept_hints

    kept_groups = {}
    for leader, members in bom.ic_groups.items():
        if leader in dropped:
            continue
        kept_groups[leader] = [m for m in members if m not in dropped]
    bom.ic_groups = kept_groups
    bom.group_labels = {k: v for k, v in bom.group_labels.items() if k not in dropped}


def _decap_membership(bom: BOM) -> dict[str, bool]:
    """ref -> True when it is a decoupling cap (2-pin, every net power/ground)."""
    nets_by_ref: dict[str, set[str]] = {}
    pins_by_ref: dict[str, set[str]] = {}
    for c in bom.connections:
        for ep in c.endpoints:
            nets_by_ref.setdefault(ep.ref, set()).add(c.net_name)
            pins_by_ref.setdefault(ep.ref, set()).add(ep.pin)
    out: dict[str, bool] = {}
    for p in bom.parts:
        nets = nets_by_ref.get(p.ref, set())
        out[p.ref] = (
            len(pins_by_ref.get(p.ref, set())) == 2
            and bool(nets)
            and all(is_power_or_ground_name(n) for n in nets)
        )
    return out


def drop_decap_only_arrays(bom: BOM) -> list[list[str]]:
    """Remove any ArraySpec whose members are *all* decoupling caps.

    The brief "5x10 array of LEDs" sometimes makes the BOM stage emit a SECOND
    ArraySpec for the per-LED bypass caps -- an identical grid over C1..Cn. That
    is not a placement array: the placer then grids those caps from the same
    origin as the LED grid, landing each cap *on top of* its LED, where its power
    pads block every inter-LED data tie (the array router skips them all and the
    leaf falls through to a doomed FreeRouting run -- KC-NZXXEE).

    A decap-only spec is dropped *only* when a real (non-decap) array shares its
    sheet, so the caps are then picked up as that array's companions and placed
    in the inter-row channel, clear of the data lane (``array_companion_refs`` /
    ``_place_companion_decaps``). A decap-only array with no sibling array is
    left alone (nothing to be a companion of). LOUD + recorded in
    ``bom.assumptions`` -- see kicraft-no-fallbacks-fail-loudly.

    Returns the member-ref lists of the dropped specs (``[]`` when none).
    """
    if len(bom.arrays) < 2:
        return []
    is_decap = _decap_membership(bom)
    sheet_by_ref = {p.ref: p.sheet for p in bom.parts}

    def _all_decap(spec) -> bool:
        return bool(spec.refs) and all(is_decap.get(r, False) for r in spec.refs)

    real_specs = [s for s in bom.arrays if not _all_decap(s)]
    real_sheets: set = set()
    for s in real_specs:
        real_sheets.update(sheet_by_ref.get(r) for r in s.refs)

    kept, dropped = [], []
    for spec in bom.arrays:
        spec_sheets = {sheet_by_ref.get(r) for r in spec.refs}
        if _all_decap(spec) and real_specs and (spec_sheets & real_sheets):
            dropped.append(list(spec.refs))
            note = (
                f"array spec over {len(spec.refs)} decoupling caps "
                f"({spec.refs[0]}..{spec.refs[-1]}) dropped -- they decouple a "
                "sibling array and are placed as its companions, not a grid "
                "co-located on top of it"
            )
            print(f"  [synth] {note}")
            bom.assumptions.append(note)
        else:
            kept.append(spec)
    bom.arrays = kept
    return dropped


def normalize_array_decaps(
    bom: BOM,
    *,
    per_led_ma: float = ARRAY_DECAP_PER_LED_MA,
    threshold_ma: float = ARRAY_DECAP_BULK_THRESHOLD_MA,
    keep: int = ARRAY_DECAP_BULK_KEEP,
) -> list[str]:
    """Thin per-LED decaps to ``keep`` bulk caps for each low-current array.

    Returns the sorted list of dropped refs (``[]`` when the rule does nothing).
    Mutates ``bom`` in place. Idempotent: a second run finds only ``keep``
    companions and drops nothing.
    """
    if not bom.arrays or threshold_ma <= 0 or keep < 0:
        return []

    sheet_by_ref = {p.ref: p.sheet for p in bom.parts}
    nets_by_ref: dict[str, set[str]] = {}
    pins_by_ref: dict[str, set[str]] = {}
    for c in bom.connections:
        for ep in c.endpoints:
            nets_by_ref.setdefault(ep.ref, set()).add(c.net_name)
            pins_by_ref.setdefault(ep.ref, set()).add(ep.pin)

    def _is_decap(ref: str) -> bool:
        # A 2-pin part whose every net is power/ground -- a bypass cap, not a
        # signal part (e.g. a series data resistor keeps a non-power net).
        nets = nets_by_ref.get(ref, set())
        return (
            len(pins_by_ref.get(ref, set())) == 2
            and bool(nets)
            and all(is_power_or_ground_name(n) for n in nets)
        )

    drop: list[str] = []
    notes: list[str] = []
    for spec in bom.arrays:
        sheets = {sheet_by_ref.get(r) for r in spec.refs}
        members = set(spec.refs)
        companions = sorted(
            (
                p.ref
                for p in bom.parts
                if p.sheet in sheets and p.ref not in members and _is_decap(p.ref)
            ),
            key=_ref_sort_key,
        )
        if len(companions) <= keep:
            continue
        est_ma = len(spec.refs) * per_led_ma
        if est_ma >= threshold_ma:
            continue  # high-current array -> per-LED decoupling stays
        victims = companions[keep:]
        drop.extend(victims)
        notes.append(
            f"LED array ({len(spec.refs)} members, ~{est_ma:.0f} mA < "
            f"{threshold_ma:.0f} mA) thinned from {len(companions)} per-LED "
            f"decaps to {keep} bulk caps; dropped {len(victims)}: "
            f"{', '.join(victims)}"
        )

    if not drop:
        return []
    dropped = set(drop)
    _scrub_refs(bom, dropped)
    for note in notes:
        print(f"  [synth] {note}")
        bom.assumptions.append(note)
    return sorted(dropped, key=_ref_sort_key)


# Connector/header ref prefixes -- a displaced array-sheet part with one of these
# names the new sheet "HEADER"; anything else gets the generic "SUPPORT".
_CONNECTOR_PREFIXES = {"J", "CN", "P", "JP", "TB", "CONN"}


def _ref_prefix(ref: str) -> str:
    return "".join(c for c in ref if not c.isdigit())


def _isolation_sheet_names(
    impure_refs: list[str], taken_names: set[str], taken_stems: set[str]
) -> tuple[str, str]:
    """A unique, shape-valid (Sheet.name, Sheet.stem) for the displaced parts."""
    all_conn = bool(impure_refs) and all(
        _ref_prefix(r) in _CONNECTOR_PREFIXES for r in impure_refs
    )
    base = "HEADER" if all_conn else "SUPPORT"
    name, stem, i = base, base, 1
    while name in taken_names or stem in taken_stems:
        i += 1
        name, stem = f"{base} {i}", f"{base}_{i}"
    return name, stem


def _resplit_connections_by_sheet(bom: BOM) -> None:
    """Confine every NetConnection to a single sheet after parts were moved.

    A ``NetConnection`` is per-sheet; a net that now spans sheets (because a part
    moved off the array sheet) must become one connection per sheet, sharing
    ``net_name``. Endpoints are regrouped by their part's current sheet -- the
    cross-sheet join is then carried by power symbols (power nets) or inter-sheet
    nets (signal nets, declared by :func:`_declare_cross_sheet_signal_nets`).
    """
    sheet_by_ref = {p.ref: p.sheet for p in bom.parts}
    out: list[NetConnection] = []
    for c in bom.connections:
        groups: dict[str, list] = {}
        for ep in c.endpoints:
            groups.setdefault(sheet_by_ref.get(ep.ref, c.sheet), []).append(ep)
        if len(groups) == 1:
            c.sheet = next(iter(groups))
            out.append(c)
            continue
        for sh, eps in groups.items():
            out.append(NetConnection(net_name=c.net_name, endpoints=list(eps), sheet=sh))
    bom.connections = out


def _declare_cross_sheet_signal_nets(bom: BOM, architecture: Architecture) -> None:
    """Declare each cross-sheet SIGNAL net as a bidirectional inter-sheet net.

    Power/ground nets join across sheets via global power symbols (the emitter
    skips them for sheet pins -- see ``emitter._emit_sheet_block``), so they are
    exempt, matching §9.14/§9.15 and the wiring stage's own output. Existing
    declarations are left untouched. Every declared endpoint sheet has a
    same-named connection (re-split above), so §9.14 coverage holds.
    """
    valid = {s.name for s in architecture.sheets}
    declared = {n.name for n in architecture.inter_sheet_nets}
    sheets_by_net: dict[str, set[str]] = {}
    for c in bom.connections:
        if c.sheet in valid:
            sheets_by_net.setdefault(c.net_name, set()).add(c.sheet)
    for net_name in sorted(sheets_by_net):
        sheets = sheets_by_net[net_name]
        if net_name in declared or is_power_or_ground_name(net_name) or len(sheets) < 2:
            continue
        architecture.inter_sheet_nets.append(
            InterSheetNet(
                name=net_name,
                endpoints=[
                    SheetPin(sheet=s, direction="bidirectional") for s in sorted(sheets)
                ],
            )
        )
        declared.add(net_name)


def isolate_array_sheets(bom: BOM, architecture: Architecture) -> list[str]:
    """Keep every array sheet a pure grid: array members + their companions only.

    The placer grids an array's members on a locked grid and co-locates its 2-pin
    power/ground companions beside them, but any OTHER part sharing the array's
    sheet (a power/data header, an MCU, ...) is handed to the force/edge solver,
    which pins it against the leaf's loose extraction envelope and strands it far
    from the grid -- the board then bloats to span both (KC-WXN3SN: a 3-pin header
    landed ~60 mm below a 4x8 LED array, 76% of the board empty).

    So a non-member, non-companion part must not live on an array sheet. This
    moves every such part onto its own dedicated sheet -> its own leaf, which the
    parent composer places adjacent to the array via the normal leaf-composition
    path. Connections are re-split per sheet and the now cross-sheet signal nets
    are declared inter-sheet so the schematic still wires them. LOUD + recorded in
    ``bom.assumptions``. Mutates ``bom`` and ``architecture`` in place; returns the
    moved refs (``[]`` when no array sheet carried a stray part).
    """
    if not bom.arrays:
        return []
    is_decap = _decap_membership(bom)
    parts_by_ref = {p.ref: p for p in bom.parts}

    # Per array sheet, the refs allowed to stay: array members + companion decaps.
    allowed_by_sheet: dict[str, set[str]] = {}
    for spec in bom.arrays:
        for r in spec.refs:
            p = parts_by_ref.get(r)
            if p is not None:
                allowed_by_sheet.setdefault(p.sheet, set()).add(r)
    for sh in list(allowed_by_sheet):
        allowed_by_sheet[sh].update(
            p.ref for p in bom.parts if p.sheet == sh and is_decap.get(p.ref, False)
        )

    taken_names = {s.name for s in architecture.sheets}
    taken_stems = {s.stem for s in architecture.sheets}
    moved: list[str] = []
    notes: list[str] = []
    for sh in sorted(allowed_by_sheet):
        impure = sorted(
            (
                p.ref
                for p in bom.parts
                if p.sheet == sh and p.ref not in allowed_by_sheet[sh]
            ),
            key=_ref_sort_key,
        )
        if not impure:
            continue
        name, stem = _isolation_sheet_names(impure, taken_names, taken_stems)
        taken_names.add(name)
        taken_stems.add(stem)
        architecture.sheets.append(Sheet(name=name, stem=stem, function="interface"))
        for ref in impure:
            parts_by_ref[ref].sheet = name
            # A stray part on an array sheet is an INTERNAL part to co-locate with
            # the array (a power/data header, not an off-board edge connector --
            # those get their own sheet at architecture time). Drop any perimeter
            # (edge/corner) zone so the composer places its leaf next to the array
            # instead of pinning it flush to a far board edge (where its power
            # trace then hugs the edge -> copper_edge_clearance). A back-side header
            # rides behind the array via BomPart.side, not an edge zone.
            zone = bom.component_zones.get(ref)
            if zone:
                cleaned = {k: v for k, v in zone.items() if k not in ("edge", "corner")}
                if cleaned:
                    bom.component_zones[ref] = cleaned
                else:
                    bom.component_zones.pop(ref, None)
        moved.extend(impure)
        notes.append(
            f"moved {len(impure)} non-array part(s) ({', '.join(impure)}) off array "
            f"sheet {sh!r} onto a dedicated sheet {name!r} so the array leaf stays a "
            f"pure grid (a stray part otherwise strands far from the array)"
        )

    if not moved:
        return []
    _resplit_connections_by_sheet(bom)
    _declare_cross_sheet_signal_nets(bom, architecture)
    for note in notes:
        print(f"  [synth] {note}")
        bom.assumptions.append(note)
    return sorted(set(moved), key=_ref_sort_key)


# ---------- opposite-edge connector isolation ----------

_OPPOSITE_EDGES = frozenset({frozenset({"top", "bottom"}), frozenset({"left", "right"})})
# When both edges have the same count, prefer moving this edge's connectors.
_DEFAULT_MOVE_EDGE: dict[frozenset[str], str] = {
    frozenset({"top", "bottom"}): "top",
    frozenset({"left", "right"}): "left",
}


def isolate_opposite_edge_connectors(
    bom: BOM, architecture: Architecture, *, verbose: bool = True
) -> list[str]:
    """Split sheets whose edge-zoned connectors demand opposite edges.

    A single rigid leaf can only sit at one edge per axis — connectors
    zoned to opposite edges on one sheet guarantee one will strand inboard
    at compose time (KC-58KPS3: J1 bottom + J2/J3 top on "USB PD INPUT").

    This moves the minority-edge connectors to a dedicated sheet → their
    own leaf, which the parent composer places at the correct edge via the
    normal leaf-composition path.  Connections are re-split per sheet and
    the now cross-sheet signal nets are declared inter-sheet so the
    schematic still wires them.  LOUD + recorded in ``bom.assumptions``.
    Mutates ``bom`` and ``architecture`` in place; returns moved refs
    (``[]`` when every sheet has compatible edge zones).
    """
    if not bom.component_zones:
        return []

    # Map ref -> sheet from BOM parts
    ref_sheet: dict[str, str] = {}
    for p in (bom.parts or []):
        if p.sheet and p.ref:
            ref_sheet[p.ref] = p.sheet

    parts_by_ref = {p.ref: p for p in bom.parts}

    # Collect edges per sheet
    sheet_edge_refs: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for ref, zone in bom.component_zones.items():
        edge = zone.get("edge") if isinstance(zone, dict) else None
        sheet = ref_sheet.get(ref)
        if edge and sheet:
            sheet_edge_refs[sheet][edge].append(ref)

    # Find sheets with opposite-edge conflicts
    taken_names = {s.name for s in architecture.sheets}
    taken_stems = {s.stem for s in architecture.sheets}
    moved: list[str] = []
    notes: list[str] = []

    for sheet, edge_refs in sorted(sheet_edge_refs.items()):
        for pair in _OPPOSITE_EDGES:
            present = [e for e in pair if e in edge_refs]
            if len(present) < 2:
                continue
            # Pick the minority edge to move; on tie, use _DEFAULT_MOVE_EDGE
            e1, e2 = present
            c1, c2 = len(edge_refs[e1]), len(edge_refs[e2])
            if c1 < c2:
                move_edge = e1
            elif c2 < c1:
                move_edge = e2
            else:
                move_edge = _DEFAULT_MOVE_EDGE.get(pair, e1)

            move_refs = sorted(edge_refs[move_edge], key=_ref_sort_key)
            name, stem = _isolation_sheet_names(move_refs, taken_names, taken_stems)
            taken_names.add(name)
            taken_stems.add(stem)
            architecture.sheets.append(
                Sheet(name=name, stem=stem, function="interface")
            )
            for ref in move_refs:
                if ref in parts_by_ref:
                    parts_by_ref[ref].sheet = name
                # Keep the edge zone — the new single-edge leaf can satisfy it.
                # Unlike isolate_array_sheets (where the moved part is a stray
                # internal component), these connectors genuinely belong at an
                # edge, and their new leaf has only same-edge connectors.
            moved.extend(move_refs)
            keep_edge = e1 if move_edge == e2 else e2
            notes.append(
                f"split sheet {sheet!r}: moved {len(move_refs)} "
                f"{move_edge}-edge connector(s) ({', '.join(move_refs)}) "
                f"to sheet {name!r} so the original leaf keeps the "
                f"{keep_edge}-edge connector(s)"
            )

    if not moved:
        return []
    _resplit_connections_by_sheet(bom)
    _declare_cross_sheet_signal_nets(bom, architecture)
    for note in notes:
        if verbose:
            # stage-commit callers must stay quiet: their stdout is the JSON
            # protocol the stage driver parses.
            print(f"  [synth] {note}")
        bom.assumptions.append(note)
    return sorted(set(moved), key=_ref_sort_key)
