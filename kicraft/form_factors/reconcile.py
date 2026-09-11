"""Atomically replace a standard host interface without discarding its circuit.

Only the stacking interface is replaced; other headers (including prototyping
pads) retain their physical and functional ownership. Requested signals are
rebound by the standard's authoritative pin map, never converted to no-connect.
"""

from __future__ import annotations

import os
import re
from itertools import permutations

from kicraft.design.models import (
    BOM, Architecture, BomPart, InterSheetNet, NetConnection, PinEndpoint,
    SheetPin, is_power_or_ground_name,
)

from . import get_template
from .scaffold import standard_header_parts
from .synthesis import _rail_key

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


def _interface_parts(state, bom, arch):
    """Footprint pitch is not ownership: a prototype header is not a host header."""
    requirements = {r.id: r for r in arch.requirements} if arch else {}
    sheets = {s.name: s for s in arch.sheets} if arch else {}
    spec = getattr(state, "functional_spec", None)
    blocks = {b.name: b for b in spec.blocks} if spec else {}
    candidates = []
    for part in bom.parts:
        if not _is_stacking_header(part) or _already_standard(part):
            continue
        requirement = requirements.get(part.lowering_requirement_id)
        if requirements:
            # A typed circuit must establish the header's functional ownership.
            # Never fall back to replacing every connector on its sheet.
            if requirement is None or requirement.role != "connector":
                continue
            if requirement.functional_blocks:
                descriptions = [
                    f"{name} {getattr(blocks.get(name), 'purpose', '')}"
                    for name in requirement.functional_blocks
                ]
            else:
                sheet = sheets.get(part.sheet)
                descriptions = [f"{part.sheet} {getattr(sheet, 'function', '')}"]
            if not any(
                re.search(r"\bstacking\b|\bpass[- ]through\b", text.replace("_", " "), re.I)
                for text in descriptions
            ):
                continue
            if requirement.exact_part:
                raise ValueError(f"cannot replace exact-part standard header {part.ref}")
            if (
                part.resolution_source != "lowerer"
                or part.resolution_id != "pin-header@1"
                or part.lowering_role != "connector"
                or part.lowering_index != 0
            ):
                raise ValueError(f"standard header {part.ref} lacks generic connector ownership")
        if part.recipe_id or part.source_leaf:
            raise ValueError(f"cannot replace physically owned standard header {part.ref}")
        candidates.append(part)
    return candidates


def reconcile_standard_form_factor(state) -> list[str]:
    """Migrate geometry, ports and ownership together, or leave state untouched."""
    intent = getattr(state, "intent", None)
    ff = getattr(intent, "form_factor", None)
    template = get_template(getattr(ff, "standard", None))
    original_bom = getattr(state, "bom", None)
    if template is None or not template.validated or not getattr(original_bom, "parts", None):
        return []
    marked = [p for p in original_bom.parts if _already_standard(p)]
    if marked:
        roles = {p.sourcing_note for p in marked}
        expected = {
            f"{_STANDARD_MARKER} {template.key} {c.role}"
            for c in template.fixed_connectors
        }
        if len(marked) != len(expected) or roles != expected:
            raise ValueError("incomplete standard form-factor header ownership")
        return []

    # Work on copies: an unsupported signal, owner or schema must not leave a
    # half-migrated BOM paired with the original architecture.
    bom = original_bom.model_copy(deep=True)
    original_arch = getattr(state, "architecture", None)
    arch = original_arch.model_copy(deep=True) if original_arch else None
    old_parts = _interface_parts(state, bom, arch)
    requirements = {r.id: r for r in arch.requirements} if arch else {}
    typed = bool(requirements)
    if typed and not old_parts:
        raise ValueError("standard form factor has no explicitly owned stacking interface")
    host_sheet = old_parts[0].sheet if old_parts else bom.parts[0].sheet
    old_refs = {p.ref for p in old_parts}
    retained = [p for p in bom.parts if p.ref not in old_refs]
    existing_j = [int(p.ref[1:]) for p in retained if re.fullmatch(r"J\d+", p.ref)]
    geometry = standard_header_parts(
        template, ref_start=max(existing_j, default=0) + 1, sheet=host_sheet
    )
    canonical_keys = {
        _rail_key(pin["net"])
        for header in geometry for pin in header["pins"] if pin["net"] is not None
    }
    owners = [requirements[p.lowering_requirement_id] for p in old_parts] if typed else []
    if typed:
        if len(owners) != len(geometry) or len({r.id for r in owners}) != len(owners):
            raise ValueError("standard header migration requires one existing owner per fixed connector")
        if len({tuple(sorted(r.functional_blocks)) for r in owners}) != 1:
            raise ValueError("cannot redistribute standard ports across distinct functional owners")
        # Preserve IDs/refs and choose the closest physical role by electrical
        # coverage, not arbitrary BOM ordering or the erroneous old pin count.
        old_parts = list(max(
            permutations(old_parts),
            key=lambda ordering: sum(
                len(
                    {_rail_key(n) for n in requirements[p.lowering_requirement_id].ports.values()}
                    & {_rail_key(pin["net"]) for pin in header["pins"] if pin["net"]}
                )
                for p, header in zip(ordering, geometry)
            ),
        ))

    # Only reviewed spelling aliases from the standard contract are equivalent,
    # never supplies that happen to have the same voltage. Two independently
    # named onboard supplies are a contradiction, not permission to short them.
    net_by_key = {}
    for connection in bom.connections:
        if any(ep.ref not in old_refs for ep in connection.endpoints):
            if is_power_or_ground_name(connection.net_name):
                key = _rail_key(connection.net_name)
                previous = net_by_key.setdefault(key, connection.net_name)
                if previous != connection.net_name:
                    raise ValueError(f"distinct onboard rail bindings: {previous!r}, {connection.net_name!r}")
    owner_ids = {r.id for r in owners}
    for requirement in requirements.values():
        if requirement.id not in owner_ids:
            for net in requirement.ports.values():
                previous = net_by_key.get(_rail_key(net))
                if previous is not None and previous != net:
                    raise ValueError(f"distinct functional rail bindings: {previous!r}, {net!r}")
    for requirement in owners:
        for net in requirement.ports.values():
            key = _rail_key(net)
            if key == "NC":
                continue
            if key not in canonical_keys:
                raise ValueError(f"standard header cannot preserve requested signal {net!r}")
            net_by_key.setdefault(key, net)
    endpoint_keys = {}
    for part in old_parts:
        requirement = requirements.get(part.lowering_requirement_id)
        if requirement:
            for port, net in requirement.ports.items():
                match = re.fullmatch(r"pin(\d+)", port)
                if not match:
                    raise ValueError(f"standard header {part.ref} needs explicit pinN ports")
                endpoint_keys[part.ref, match.group(1)] = _rail_key(net)
    for connection in bom.connections:
        for ep in connection.endpoints:
            if ep.ref not in old_refs:
                continue
            key = endpoint_keys.get((ep.ref, ep.pin), _rail_key(connection.net_name))
            if key == "NC" and connection.net_name == "NC":
                continue  # reserved contact, not a requested electrical signal
            if key not in canonical_keys:
                raise ValueError(f"standard header cannot preserve requested signal {connection.net_name!r}")
            previous = net_by_key.get(key)
            if previous is not None and _rail_key(previous) != _rail_key(connection.net_name):
                raise ValueError(f"conflicting standard pin binding: {previous!r}, {connection.net_name!r}")
            net_by_key.setdefault(key, connection.net_name)
            endpoint_keys[ep.ref, ep.pin] = key

    # An independently owned recipe/fabrication interface is not replaceable.
    for manifest in bom.recipe_ownership:
        if old_refs.intersection(manifest.refs):
            raise ValueError("standard header belongs to a recipe ownership manifest")
    for interface in bom.edge_interfaces:
        if old_refs.intersection(interface.refs):
            raise ValueError("standard header belongs to a fabrication edge interface")

    parts = []
    bound = {}
    noconnects = []
    for index, header in enumerate(geometry):
        old = old_parts[index] if typed else None
        part = BomPart(
            ref=old.ref if old else header["ref"],
            value=header["value"], symbol=header["symbol"],
            footprint=header["footprint"], sheet=old.sheet if old else host_sheet,
            sourcing_note=f"{_STANDARD_MARKER} {template.key} {header['role']}",
        )
        ports = {}
        for pin in header["pins"]:
            net = net_by_key.get(_rail_key(pin["net"])) if pin["net"] else None
            ep = PinEndpoint(ref=part.ref, pin=pin["pin"])
            ports[f"pin{pin['pin']}"] = net or "NC"
            if net is None:
                noconnects.append(ep)
            else:
                bound.setdefault((part.sheet, net), []).append(ep)
        if old:
            requirement = requirements[old.lowering_requirement_id]
            requirement.ports = ports
            requirement.parameters = {"rows": 1, "gender": "female"}
            # Same generic connector implementation and same functional owner;
            # only its mistaken pin geometry has changed.
            part.resolution_source = old.resolution_source
            part.resolution_id = old.resolution_id
            part.lowering_requirement_id = old.lowering_requirement_id
            part.lowering_role = old.lowering_role
            part.lowering_index = old.lowering_index
            part.side = old.side
            part.assembly = old.assembly
        parts.append(part)

    # Keep every non-interface endpoint, including prototype-pad connections.
    kept_connections = []
    for connection in bom.connections:
        connection.endpoints = [ep for ep in connection.endpoints if ep.ref not in old_refs]
        if connection.endpoints:
            key = _rail_key(connection.net_name)
            if key in canonical_keys and key in net_by_key:
                connection.net_name = net_by_key[key]
            kept_connections.append(connection)
    bom.parts = retained + parts
    bom.connections = kept_connections + [
        NetConnection(sheet=sheet, net_name=net, endpoints=endpoints)
        for (sheet, net), endpoints in bound.items()
    ]
    bom.no_connect_pins = [ep for ep in bom.no_connect_pins if ep.ref not in old_refs] + noconnects

    if typed:
        # Ref indexes still point to the same physical owners. Pin-specific hints
        # must follow the electrical signal rather than retain an obsolete pad.
        new_pins = {}
        for connection in bom.connections:
            for ep in connection.endpoints:
                if ep.ref in old_refs:
                    new_pins.setdefault((ep.ref, _rail_key(connection.net_name)), []).append(ep.pin)
        for hint in bom.placement_hints:
            if hint.anchor_ref in old_refs and hint.anchor_pin is not None:
                key = endpoint_keys.get((hint.anchor_ref, hint.anchor_pin))
                choices = new_pins.get((hint.anchor_ref, key), [])
                if len(choices) != 1:
                    raise ValueError("cannot unambiguously migrate standard-header placement anchor")
                hint.anchor_pin = choices[0]
    else:
        # Legacy untyped designs have no durable requirement IDs to retain.
        bom.component_zones = {r: z for r, z in bom.component_zones.items() if r not in old_refs}
        bom.thermal_refs = [r for r in bom.thermal_refs if r not in old_refs]
        bom.signal_flow_order = [r for r in bom.signal_flow_order if r not in old_refs]
        bom.ic_groups = {
            r: [m for m in members if m not in old_refs]
            for r, members in bom.ic_groups.items() if r not in old_refs
        }
        for spec in bom.arrays:
            spec.refs = [r for r in spec.refs if r not in old_refs]
        bom.arrays = [spec for spec in bom.arrays if spec.refs]
        if any(h.ref in old_refs or h.anchor_ref in old_refs for h in bom.placement_hints):
            raise ValueError("cannot discard standard-header placement ownership")
        if arch:
            _migrate_emptied_sheets(arch, bom, {p.sheet for p in old_parts}, host_sheet)

    if arch:
        # Header-only spelling aliases also occur in architecture rail and
        # inter-sheet contracts. Migrate those names in the same transaction.
        def canonical_name(net):
            return net_by_key.get(_rail_key(net), net)

        arch.power_nets = list(dict.fromkeys(canonical_name(net) for net in arch.power_nets))
        rails = {}
        for net, voltage in arch.rail_voltages.items():
            name = canonical_name(net)
            if name in rails and rails[name] != voltage:
                raise ValueError(f"conflicting standard rail voltage for {name!r}")
            rails[name] = voltage
        arch.rail_voltages = rails
        nets = {}
        for net in arch.inter_sheet_nets:
            net.name = canonical_name(net.name)
            if net.name not in nets:
                nets[net.name] = net
                continue
            endpoints = {ep.sheet: ep for ep in nets[net.name].endpoints}
            for ep in net.endpoints:
                previous = endpoints.get(ep.sheet)
                if previous is not None and previous.direction != ep.direction:
                    raise ValueError("conflicting aliased inter-sheet net directions")
                endpoints[ep.sheet] = ep
            nets[net.name].endpoints = list(endpoints.values())
        arch.inter_sheet_nets = list(nets.values())
        # Redistribution can move a rail onto a header sheet that did not
        # previously expose it (R13's +3V3). Derive those sheet endpoints from
        # real parts/ports, retaining existing direction contracts where present.
        if typed:
            net_sheets = {}
            for requirement in arch.requirements:
                for net in requirement.ports.values():
                    if net != "NC":
                        net_sheets.setdefault(net, set()).add(requirement.sheet)
            part_sheets = {p.ref: p.sheet for p in bom.parts}
            for connection in bom.connections:
                net_sheets.setdefault(connection.net_name, set()).update(
                    part_sheets[ep.ref] for ep in connection.endpoints
                )
            nets = {net.name: net for net in arch.inter_sheet_nets}
            for net in set(net_by_key.values()):
                sheets = net_sheets.get(net, set())
                if len(sheets) < 2:
                    nets.pop(net, None)
                    continue
                previous = {ep.sheet: ep for ep in nets[net].endpoints} if net in nets else {}
                nets[net] = InterSheetNet(name=net, endpoints=[
                    previous[sheet] if sheet in previous else SheetPin(sheet=sheet, direction="bidirectional")
                    for sheet in sorted(sheets)
                ])
            arch.inter_sheet_nets = list(nets.values())

    bom = BOM.model_validate(bom.model_dump())
    if arch:
        arch = Architecture.model_validate(arch.model_dump())
    if hasattr(state, "model_dump"):
        payload = state.model_dump()
        payload["bom"] = bom.model_dump()
        payload["architecture"] = arch.model_dump() if arch else None
        type(state).model_validate(payload)
    # Publish only after the complete candidate is valid; preserve callers'
    # references to these models.
    for field in type(bom).model_fields:
        setattr(original_bom, field, getattr(bom, field))
    if arch:
        for field in type(arch).model_fields:
            setattr(original_arch, field, getattr(arch, field))
    return [
        f"replaced stacking interface {sorted(old_refs)} with {template.key} "
        f"{[(p.ref, p.lowering_requirement_id, p.sheet) for p in parts]}; "
        f"preserved requested nets {sorted({net for _, net in bound})}; "
        f"retained non-interface parts {[p.ref for p in retained]}"
    ]


def _migrate_emptied_sheets(arch, bom, replaced_sheets, host_sheet):
    """Move only emptied interface-sheet identities, not unrelated functions."""
    used = {p.sheet for p in bom.parts}
    dropped = replaced_sheets - used
    if not dropped:
        return
    arch.sheets = [s for s in arch.sheets if s.name not in dropped]
    for requirement in arch.requirements:
        if requirement.sheet in dropped:
            requirement.sheet = host_sheet
    for selection in arch.recipe_selections:
        selection.sheets = {
            role: host_sheet if sheet in dropped else sheet
            for role, sheet in selection.sheets.items()
        }
    surviving = []
    for net in arch.inter_sheet_nets:
        endpoints = {}
        for endpoint in net.endpoints:
            if endpoint.sheet in dropped:
                endpoint.sheet = host_sheet
            previous = endpoints.get(endpoint.sheet)
            if previous is not None and previous.direction != endpoint.direction:
                raise ValueError("cannot merge conflicting standard-interface sheet directions")
            endpoints[endpoint.sheet] = endpoint
        net.endpoints = list(endpoints.values())
        if len(net.endpoints) > 1:
            surviving.append(net)
    arch.inter_sheet_nets = surviving


__all__ = ["enforce_enabled", "reconcile_standard_form_factor"]
