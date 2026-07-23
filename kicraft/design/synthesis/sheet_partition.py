"""Deterministic BOM rule: split DETACHABLE subfunctions off an oversized SoC sheet.

A whole MCU subsystem -- SoC + decoupling farm + crystals + debug header + button
-- routinely lands on one hierarchy sheet, i.e. one leaf. That leaf is the #1
place/route failure class by breadth (docs/plans/dense-soc-leaf-unconnected-plan.md).

The tempting split -- "power/decoupling gets its own sheet" -- is electrically
WRONG and makes it worse: a decoupling cap exists to sit 1-2 mm from the pin it
bridges, and moving it to another leaf guarantees the opposite while exploding
cross-leaf interconnect. So this pass moves only what is genuinely *detachable*:
a subfunction that talks to the SoC over a handful of signals and owns its own
parts (SWD/debug header, user IO button, battery holder, the RF chain). What must
stay is stated positively and enforced: the SoC, every decoupling cap, every
crystal and its load caps.

Same shape as the other deterministic sheet passes in
:mod:`array_decaps` (``isolate_array_sheets`` / ``isolate_opposite_edge_connectors``):
mutates ``bom`` + ``architecture`` in place, re-splits per-sheet connections,
declares the now cross-sheet signal nets, and is LOUD + recorded in
``bom.assumptions``.

DEFAULT OFF (``KICRAFT_SPLIT_DENSE_SHEETS=1`` turns it on). It changes the sheet
structure of every dense design, so it must be measured by a self-eval batch
before it can be trusted as a default -- and turning it on in the same batch that
measures the placement fixes (P0-P2) would make neither attributable. See the
plan's P3.
"""
from __future__ import annotations

import os

from kicraft.design.models import BOM, Architecture, Sheet, is_power_or_ground_name

from kicraft.design.synthesis.array_decaps import (
    _decap_membership,
    _declare_cross_sheet_signal_nets,
    _isolation_sheet_names,
    _nets_pins_by_ref,
    _ref_prefix,
    _ref_sort_key,
    _resplit_connections_by_sheet,
)

__all__ = [
    "split_dense_soc_sheets",
    "split_dense_sheets_enabled",
    "DENSE_SHEET_ROUTABLE_MAX",
]


def split_dense_sheets_enabled() -> bool:
    """Whether the dense-sheet partition runs. OFF unless
    ``KICRAFT_SPLIT_DENSE_SHEETS`` is set to a truthy value."""
    val = os.environ.get("KICRAFT_SPLIT_DENSE_SHEETS")
    return bool(val) and val.strip().lower() not in ("0", "false", "no", "off")

# A sheet with more routable parts than this is a dense-SoC candidate. ~15 is
# where the leaf solver's escape/route budget starts losing nets on real boards
# (nRF52840 beacon: 29 parts on one sheet).
DENSE_SHEET_ROUTABLE_MAX = 15
# A part that anchors a detachable subfunction: an interface or active device
# with its own reason to exist off the SoC. A group with none of these is just
# loose passives -- moving it would strand them from whatever they hug.
_DETACHABLE_PREFIXES = frozenset({"J", "P", "SW", "BT", "ANT", "U", "K", "M"})
# Parts with no netlist presence worth counting toward density.
_NON_ROUTABLE_PREFIXES = frozenset({"H", "MH", "FID", "TP", "LOGO"})
# A crystal/oscillator: its load caps must never leave its side.
_CRYSTAL_PREFIXES = frozenset({"X", "Y"})
# Two-terminal passives -- the only parts that can be "a cap on the IC's pin".
# A button and a battery holder are also 2-pin with power-only nets, so the
# decap predicates must not be applied to every 2-pin part.
_PASSIVE_PREFIXES = frozenset({"C", "R", "L", "FB", "FL"})


def _routable_refs(bom: BOM, sheet: str) -> list[str]:
    return [
        p.ref
        for p in bom.parts
        if p.sheet == sheet and _ref_prefix(p.ref) not in _NON_ROUTABLE_PREFIXES
    ]


def _hub_ref(bom: BOM, refs: list[str], pins_by_ref: dict[str, set[str]]) -> str | None:
    """The sheet's SoC: the part with the most connected pins (>= 8)."""
    best, best_pins = None, 0
    for ref in sorted(refs, key=_ref_sort_key):
        n = len(pins_by_ref.get(ref, set()))
        if n > best_pins:
            best, best_pins = ref, n
    return best if best_pins >= 8 else None


def _must_stay_with_hub(
    bom: BOM,
    sheet_refs: set[str],
    hub: str,
    nets_by_ref: dict[str, set[str]],
    pins_by_ref: dict[str, set[str]],
) -> set[str]:
    """Refs that are electrically bound to the hub and may never be moved.

    * every decoupling cap (2-pin, all nets power/ground);
    * every 2-pin passive whose non-power nets reach ONLY the hub -- that is a
      DEC-pin / feedback cap, named like a signal but electrically a decap;
    * every crystal, and anything sharing a net with one (its load caps).
    """
    is_decap = _decap_membership(bom)
    refs_by_net: dict[str, set[str]] = {}
    for ref in sheet_refs:
        for net in nets_by_ref.get(ref, set()):
            refs_by_net.setdefault(net, set()).add(ref)

    stay = {hub}
    crystals = {r for r in sheet_refs if _ref_prefix(r) in _CRYSTAL_PREFIXES}
    stay |= crystals
    for x in crystals:
        for net in nets_by_ref.get(x, set()):
            if is_power_or_ground_name(net):
                continue  # GND touches every part -- it would pin the whole sheet
            stay |= refs_by_net.get(net, set())
    for ref in sheet_refs:
        if _ref_prefix(ref) not in _PASSIVE_PREFIXES:
            continue  # a button/battery is 2-pin and power-only too
        if is_decap.get(ref, False):
            stay.add(ref)
            continue
        if len(pins_by_ref.get(ref, set())) != 2:
            continue
        signal_nets = {
            n for n in nets_by_ref.get(ref, set()) if not is_power_or_ground_name(n)
        }
        if signal_nets and all(
            refs_by_net.get(n, set()) - {ref} <= {hub} for n in signal_nets
        ):
            stay.add(ref)  # bridges the hub's own pin and nothing else
    return stay


def _detachable_groups(
    sheet_refs: set[str],
    movable: set[str],
    nets_by_ref: dict[str, set[str]],
) -> list[list[str]]:
    """Connected groups of movable parts (via their shared SIGNAL nets), each
    containing at least one interface/active part -- an actual subfunction."""
    adj: dict[str, set[str]] = {r: set() for r in movable}
    refs_by_net: dict[str, set[str]] = {}
    for ref in movable:
        for net in nets_by_ref.get(ref, set()):
            if not is_power_or_ground_name(net):
                refs_by_net.setdefault(net, set()).add(ref)
    for members in refs_by_net.values():
        for a in members:
            adj[a] |= members - {a}

    seen: set[str] = set()
    groups: list[list[str]] = []
    for start in sorted(movable, key=_ref_sort_key):
        if start in seen:
            continue
        stack, comp = [start], []
        while stack:
            r = stack.pop()
            if r in seen:
                continue
            seen.add(r)
            comp.append(r)
            stack.extend(sorted(adj[r] - seen, key=_ref_sort_key))
        if any(_ref_prefix(r) in _DETACHABLE_PREFIXES for r in comp):
            groups.append(sorted(comp, key=_ref_sort_key))
    # Biggest first: fewest moves to get the sheet under the threshold.
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def split_dense_soc_sheets(
    bom: BOM,
    architecture: Architecture,
    *,
    max_routable: int = DENSE_SHEET_ROUTABLE_MAX,
    verbose: bool = True,
) -> list[str]:
    """Move detachable subfunctions off sheets with too many routable parts.

    Returns the moved refs (``[]`` when nothing qualified). Mutates ``bom`` and
    ``architecture`` in place.
    """
    nets_by_ref, pins_by_ref = _nets_pins_by_ref(bom)
    parts_by_ref = {p.ref: p for p in bom.parts}
    taken_names = {s.name for s in architecture.sheets}
    taken_stems = {s.stem for s in architecture.sheets}
    moved: list[str] = []
    notes: list[str] = []

    for sheet in sorted({p.sheet for p in bom.parts}):
        refs = _routable_refs(bom, sheet)
        if len(refs) <= max_routable:
            continue
        hub = _hub_ref(bom, refs, pins_by_ref)
        if hub is None:
            continue  # no SoC to detach FROM: not this rule's case
        sheet_refs = set(refs)
        stay = _must_stay_with_hub(bom, sheet_refs, hub, nets_by_ref, pins_by_ref)
        groups = _detachable_groups(sheet_refs, sheet_refs - stay, nets_by_ref)

        remaining = len(refs)
        for group in groups:
            if remaining <= max_routable:
                break
            name, stem = _isolation_sheet_names(group, taken_names, taken_stems)
            taken_names.add(name)
            taken_stems.add(stem)
            architecture.sheets.append(Sheet(name=name, stem=stem, function="interface"))
            for ref in group:
                parts_by_ref[ref].sheet = name
            moved.extend(group)
            remaining -= len(group)
            notes.append(
                f"moved detachable subfunction ({', '.join(group)}) off dense sheet "
                f"{sheet!r} ({len(refs)} routable parts, > {max_routable}) onto its "
                f"own sheet {name!r}; {hub}'s decoupling caps and crystals stay with "
                f"it -- they must sit at its pins"
            )

    if not moved:
        return []
    _resplit_connections_by_sheet(bom)
    _declare_cross_sheet_signal_nets(bom, architecture)
    for note in notes:
        if verbose:
            print(f"  [synth] {note}")
        bom.assumptions.append(note)
    return sorted(set(moved), key=_ref_sort_key)
