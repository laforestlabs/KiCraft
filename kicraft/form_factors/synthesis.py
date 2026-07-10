"""Synthesis side of replace & rewire: standard headers as real BOM parts.

The linchpin of the feature -- the standard's headers must be REAL parts in the
BOM/schematic/netlist (you solder them), never phantom compose injections. This
module is the deterministic core a synthesis reconcile calls: it turns a
validated template into schema-valid :class:`BomPart`s plus the canonical
power/ground :class:`NetConnection`s, with non-colliding refs.

Scope note: power/ground pins (+5V/+3V3/GND/VIN/…) bind here -- they are
unambiguous and merge with the design's rails by net name (KiCraft power nets are
global). **Signal** pins (D0..D13/A0..A5) are emitted as part pins but their net
binding is deliberately NOT decided here: for a functional shield that is the
LLM/wiring stage's job, and getting the ERC treatment right (a lone connector pin
vs. a proto-area passthrough) needs validation on a real build. So this returns
the parts + power nets; wiring the signal pins + the schematic emitter + the ERC
pass is the live step that consumes this.
"""

from __future__ import annotations

import re

from kicraft.design.models import BomPart, NetConnection, PinEndpoint

from . import FormFactorTemplate
from .scaffold import CANONICAL_RAILS, standard_header_parts

_J_RE = re.compile(r"^J(\d+)$")


def _next_j_index(existing_refs: set[str]) -> int:
    """First J-index free above every existing ``J<n>`` ref (so new headers
    never collide with the design's connectors)."""
    used = [int(m.group(1)) for r in existing_refs if (m := _J_RE.match(r))]
    return (max(used) + 1) if used else 1


def standard_form_factor_bom_delta(
    template: FormFactorTemplate,
    existing_refs: set[str],
    *,
    sheet: str = "INTERFACE",
) -> tuple[list[BomPart], list[NetConnection], list[PinEndpoint]]:
    """(new BomParts, canonical power NetConnections, signal-pin no-connects).

    Refs are allocated after the highest existing ``J<n>``. Parts are the single-
    row headers with the template footprints. Power connections cover ONLY the
    canonical rails (:data:`~kicraft.form_factors.scaffold.CANONICAL_RAILS`); a
    rail on several pins (e.g. the two GND pins) collects them into one net. The
    remaining **signal** pins (D0..D13/A0..A5) are returned as no-connect
    endpoints -- on a shield they mate with the host board below, so they carry
    no on-board net and would otherwise trip ERC ``pin_not_connected``.
    """
    parts_data = standard_header_parts(
        template, ref_start=_next_j_index(existing_refs), sheet=sheet
    )
    parts = [
        BomPart(
            ref=p["ref"],
            value=p["value"],
            symbol=p["symbol"],
            footprint=p["footprint"],
            sheet=sheet,
            sourcing_note=f"standard form factor: {template.key} {p['role']}",
        )
        for p in parts_data
    ]

    rail_endpoints: dict[str, list[PinEndpoint]] = {}
    signal_noconnects: list[PinEndpoint] = []
    for p in parts_data:
        for pin in p["pins"]:
            net = pin["net"]
            if net is None:
                continue  # NC / reserved pin: neither bound nor no-connect-marked
            ep = PinEndpoint(ref=p["ref"], pin=pin["pin"])
            if net in CANONICAL_RAILS:
                rail_endpoints.setdefault(net, []).append(ep)
            else:
                signal_noconnects.append(ep)  # D0..D13 / A0..A5
    connections = [
        NetConnection(net_name=net, endpoints=eps, sheet=sheet)
        for net, eps in sorted(rail_endpoints.items())
    ]
    return parts, connections, signal_noconnects


__all__ = ["standard_form_factor_bom_delta"]
