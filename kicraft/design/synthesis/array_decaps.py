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
kicraft-no-fallbacks-fail-loudly). High-current arrays (e.g. a 25-LED WS2812
string at ~1.5 A) stay fully per-LED-decoupled.
"""
from __future__ import annotations

from kicraft.design.models import BOM, is_power_or_ground_name

# WS2812-class addressable LED: ~3x20 mA = 60 mA per device at full white. We use
# this conservative MAX -- overestimating current makes the rule LESS likely to
# drop caps, which is the safe direction.
ARRAY_DECAP_PER_LED_MA = 60.0
# Total array current below which per-LED decoupling is overkill (a few caps
# suffice). 500 mA ~= an 8-LED string; a 25-LED string (~1.5 A) keeps per-LED.
ARRAY_DECAP_BULK_THRESHOLD_MA = 500.0
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
