"""Deterministic "replace & rewire" reconcile at the wiring stage-commit.

When a brief named a validated standard form factor (e.g. Arduino shield), this
replaces the LLM's generic stacking connectors with the standard's headers as
REAL BOM parts, binds their power/ground pins to the design rails by net name,
and marks the signal pins no-connect (they mate with the host board below, so
they carry no on-board net). The result is a schematic/BOM/netlist whose edge
interface is exactly the standard's -- the electrical half of replace & rewire.

Env-gated by ``KICRAFT_FORM_FACTOR_ENFORCE`` (see :func:`enforce_enabled`) so it
can never touch a normal build; it only runs when explicitly turned on for a
shield dogfood run. ERC-correctness of the emitted schematic needs validation on
a real build (a lone standard header, power-pin drive treatment) -- unit tests
cover the BOM transformation, not ERC.

Mechanical placement of these real headers at their fixed board positions is the
compose half (``compose_scaffold`` + the ``_compose_artifacts`` fork); wiring
that to consume these real parts (rather than inject synthetic ones) is the
remaining integration step, tracked in the plan.
"""

from __future__ import annotations

import os

from kicraft.design.models import is_power_or_ground_name

from . import get_template
from .synthesis import standard_form_factor_bom_delta

_STANDARD_MARKER = "standard form factor:"


def enforce_enabled() -> bool:
    """Whether standard-form-factor enforcement is on.

    Default ON: a brief that names a validated standard (e.g. Arduino shield) gets
    the replace & rewire reconcile + fixed outline + locked connectors. The
    ``KICRAFT_FORM_FACTOR_ENFORCE`` env var is retained as a production KILL
    SWITCH -- set it to ``0`` / ``false`` / ``no`` / ``off`` to disable the
    feature without a redeploy (e.g. if a fresh-synthesis shield ever misbehaves).
    Only ever engages when :func:`match_standard` matched a *validated* template,
    so non-shield boards are unaffected regardless of this flag.
    """
    val = os.environ.get("KICRAFT_FORM_FACTOR_ENFORCE")
    if val is None or val.strip() == "":
        return True
    return val.strip().lower() not in ("0", "false", "no", "off")


def _is_stacking_header(part) -> bool:
    """A 2.54 mm pin-header/socket connector -- the shield interface class the
    standard replaces. Deliberately narrow on the connector FAMILY (a USB/other
    connector on a functional shield is left alone), but robust to naming: it must
    match both KiCad-stock footprints (``PinHeader_...P2.54mm``) AND the vendored
    library naming (``pin-header-female-2-54-1x40:HDR-TH_40P-P2.54-V-F``). Keying
    only on the stock ``PinHeader_``/``P2.54mm`` substrings missed the vendored
    header, so its ref was never dropped -- the scaffold then added a duplicate
    ref that collided a leaf against its parent-local twin at compose (WS5)."""
    fp = (getattr(part, "footprint", "") or "").lower()
    is_header = any(
        tok in fp
        for tok in ("pinheader", "pinsocket", "pin-header", "pin-socket", "hdr")
    )
    # 2.54 mm pitch: "p2.54mm" (stock), "p2.54"/"2.54" (generic), "2-54"/"2_54"
    # (vendored library slug where dots are hyphenated/underscored).
    is_254 = any(tok in fp for tok in ("2.54", "2-54", "2_54"))
    return is_header and is_254


def _already_standard(part) -> bool:
    return _STANDARD_MARKER in (getattr(part, "sourcing_note", "") or "")


def reconcile_standard_form_factor(state) -> list[str]:
    """Rewire ``state.bom`` to the standard's headers. Returns human notes.

    No-op (returns ``[]``) unless a validated standard was captured and the BOM
    exists. Idempotent: previously-added standard headers are not re-dropped.
    """
    intent = getattr(state, "intent", None)
    ff = getattr(intent, "form_factor", None) if intent is not None else None
    template = get_template(getattr(ff, "standard", None) if ff is not None else None)
    if template is None or not template.validated:
        return []
    bom = getattr(state, "bom", None)
    if bom is None or not getattr(bom, "parts", None):
        return []

    # Idempotent: a BOM that already carries the standard headers is done. A
    # re-commit must not stack a second set.
    if any(_already_standard(p) for p in bom.parts):
        return []

    notes: list[str] = []

    # 1. LLM stacking connectors to replace (never the standard's own headers).
    drop_refs = {
        p.ref for p in bom.parts if _is_stacking_header(p) and not _already_standard(p)
    }
    # 2. Host sheet: reuse a dropped connector's sheet, else the first part's.
    host_sheet = None
    if drop_refs:
        host_sheet = next(p.sheet for p in bom.parts if p.ref in drop_refs)
    elif bom.parts:
        host_sheet = bom.parts[0].sheet
    if host_sheet is None:
        return []

    # 3. Remove dropped parts + prune every reference to them (connections,
    #    no-connects, and the ref-bearing BOM index fields the model validates).
    if drop_refs:
        bom.parts = [p for p in bom.parts if p.ref not in drop_refs]
        pruned = 0
        kept_conns = []
        for c in bom.connections:
            eps = [ep for ep in c.endpoints if ep.ref not in drop_refs]
            pruned += len(c.endpoints) - len(eps)
            if len(eps) >= 1:
                c.endpoints = eps
                kept_conns.append(c)
            # a connection left with no endpoints is dropped entirely
        bom.connections = kept_conns
        bom.no_connect_pins = [
            ep for ep in bom.no_connect_pins if ep.ref not in drop_refs
        ]
        # Ref-index fields (BOM validators reject a stale ref in any of these).
        bom.component_zones = {
            r: z for r, z in bom.component_zones.items() if r not in drop_refs
        }
        bom.thermal_refs = [r for r in bom.thermal_refs if r not in drop_refs]
        bom.signal_flow_order = [
            r for r in bom.signal_flow_order if r not in drop_refs
        ]
        bom.ic_groups = {
            ic: [m for m in members if m not in drop_refs]
            for ic, members in bom.ic_groups.items()
            if ic not in drop_refs
        }
        kept_arrays = []
        for spec in bom.arrays:
            spec.refs = [r for r in spec.refs if r not in drop_refs]
            if spec.refs:
                kept_arrays.append(spec)
        bom.arrays = kept_arrays
        notes.append(
            f"replaced {len(drop_refs)} LLM stacking connector(s) "
            f"{sorted(drop_refs)} (pruned {pruned} endpoint(s))"
        )

    # 4. Add the standard headers as real parts, binding each pin onto a rail
    #    the design ALREADY carries (else no-connect). The design's rails are its
    #    power/ground nets remaining after the drop above (global by name, so the
    #    header binds cross-sheet safely by re-using the design's own net name --
    #    which is what stops KiCad merging a duplicate +3V3/3V3 rail and colliding
    #    the regulator's driver with the emitter's PWR_FLAG).
    design_rails = frozenset(
        c.net_name
        for c in bom.connections
        if c.endpoints and is_power_or_ground_name(c.net_name)
    )
    existing = {p.ref for p in bom.parts}
    parts, rail_conns, noconnects = standard_form_factor_bom_delta(
        template, existing, sheet=host_sheet, design_rails=design_rails
    )
    bom.parts.extend(parts)
    bom.connections.extend(rail_conns)
    bom.no_connect_pins.extend(noconnects)
    bound_nets = sorted({c.net_name for c in rail_conns})
    notes.append(
        f"added {len(parts)} {template.key} header(s) on sheet {host_sheet!r}: "
        f"{[p.ref for p in parts]}; bound pins to {bound_nets or 'no'} rail(s); "
        f"{len(noconnects)} pin(s) no-connect"
    )

    # Loud, not latent: every original stacking header must have been dropped in
    # step 3. If one survived (a footprint naming ``_is_stacking_header`` failed to
    # recognize), the scaffold's headers now coexist with the LLM's -- the exact
    # setup that collides a leaf ref against its parent-local scaffold twin at
    # compose. Fail HERE, naming the offending part+footprint, instead of surfacing
    # a cryptic "Parent-local component ref 'J4' collides with a child" later (WS5).
    survivors = [
        (p.ref, getattr(p, "footprint", ""))
        for p in bom.parts
        if _is_stacking_header(p) and not _already_standard(p)
    ]
    if survivors:
        raise ValueError(
            "form-factor reconcile left un-replaced stacking header(s) alongside "
            f"the standard's scaffold: {survivors}. _is_stacking_header did not "
            "recognize the footprint -- broaden its detection so this ref is dropped "
            "(it would otherwise collide with a scaffold-added ref at compose)."
        )

    # 5. Consolidating the headers onto one host sheet can leave the sheets that
    #    held only LLM connectors with no parts at all. An empty sheet is a
    #    degenerate leaf -- the placement engine aborts on a leaf subcircuit "with
    #    no matching components". Drop those emptied sheets from the architecture
    #    (and any inter-sheet net that referenced them) so the hierarchy the
    #    schematic + compose see matches the rewired BOM.
    notes += _prune_emptied_sheets(state, bom)
    return notes


def _prune_emptied_sheets(state, bom) -> list[str]:
    """Remove architecture sheets left with no BOM parts by the rewire, and any
    inter-sheet net endpoint that pointed at them. Returns human notes."""
    arch = getattr(state, "architecture", None)
    if arch is None or not getattr(arch, "sheets", None):
        return []
    used = {p.sheet for p in bom.parts}
    keep = [s for s in arch.sheets if s.name in used]
    dropped = {s.name for s in arch.sheets if s.name not in used}
    if not dropped:
        return []
    arch.sheets = keep
    if getattr(arch, "inter_sheet_nets", None):
        surviving = []
        for isn in arch.inter_sheet_nets:
            isn.endpoints = [ep for ep in isn.endpoints if ep.sheet not in dropped]
            if len(isn.endpoints) >= 2:  # still spans >=2 sheets -> keep
                surviving.append(isn)
        arch.inter_sheet_nets = surviving
    return [f"pruned emptied sheet(s) {sorted(dropped)}"]


__all__ = ["enforce_enabled", "reconcile_standard_form_factor"]
