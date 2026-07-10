"""Synthesis side of replace & rewire: standard headers as real BOM parts.

The linchpin of the feature -- the standard's headers must be REAL parts in the
BOM/schematic/netlist (you solder them), never phantom compose injections. This
module is the deterministic core a synthesis reconcile calls: it turns a
validated template into schema-valid :class:`BomPart`s plus the
:class:`NetConnection`s that bind the header pins onto the design's rails, with
non-colliding refs.

**Design-aware binding.** A shield header is the board's edge interface: its pins
mate with the host board below. On-board, a header pin carries a net ONLY when
the design already routes that rail somewhere -- e.g. an onboard 3V3 regulator
gives the board a ``3V3`` rail, so the header's +3V3 pin joins it. A pin whose
rail the design does not use (a bare shield's AREF/RESET/IOREF, or every unused
digital/analog pin) connects to nothing on-board and is emitted no-connect (it
passes through to the host, so it must not trip ERC ``pin_not_connected`` or hang
as a dangling label).

Binding to the design's OWN net name (``3V3``, not the header's canonical
``+3V3``) is what keeps ERC clean: a second name for one rail would make KiCad
merge them and then collide the regulator's power-output driver with the flag the
emitter adds to the seemingly-undriven ``+3V3`` net. So the caller passes the set
of rails the design already carries (its power/ground nets after the LLM's own
connectors are dropped), and each header pin binds to the matching design rail
(alias-normalized) or, failing a match, is no-connect.

Pure over a :class:`~kicraft.form_factors.FormFactorTemplate` + the caller's rail
set; no pydantic-adjacent I/O beyond building the models.
"""

from __future__ import annotations

import re

from kicraft.design.models import BomPart, NetConnection, PinEndpoint

from . import FormFactorTemplate
from .scaffold import standard_header_parts

_J_RE = re.compile(r"^J(\d+)$")


def _rail_key(net: str) -> str:
    """Canonical identity for a power/ground rail name, so the header's
    canonical spelling matches the design's however each writes it.

    Folds the common variants of the same physical rail onto one key: a leading
    ``+`` is cosmetic (``+5V`` == ``5V``), and 3.3 V is written ``3V3`` /
    ``3.3V`` / ``+3.3V`` interchangeably. Deliberately conservative -- it does
    NOT alias ``VCC``/``VDD`` onto a numbered rail (ambiguous) -- so a miss falls
    through to no-connect (safe) rather than a wrong bind.
    """
    u = net.strip().upper().lstrip("+")
    if u in ("3.3V", "3V3", "3.3"):
        return "3V3"
    if u in ("5V", "5.0V", "5.0"):
        return "5V"
    return u


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
    design_rails: frozenset[str] | set[str] = frozenset(),
) -> tuple[list[BomPart], list[NetConnection], list[PinEndpoint]]:
    """(new BomParts, rail NetConnections, no-connect endpoints).

    Refs are allocated after the highest existing ``J<n>``. Parts are the single-
    row headers with the template footprints. Each header pin binds to a design
    rail in ``design_rails`` (matched by :func:`_rail_key`, using the design's own
    net name) when one matches; every other pin -- unused rails, unused
    digital/analog pins, and the reserved/NC header positions -- is returned as a
    no-connect endpoint. A rail landing on several header pins (e.g. the two GND
    pins) collects them into one net.

    ``design_rails`` is the set of net names the design already carries that a
    header pin may join -- in practice the design's power/ground rails (global by
    name, so cross-sheet binding is safe). Passing an empty set makes every pin
    no-connect (a header with nothing to hang onto).
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

    # rail-key -> the design's own net name for that rail (first spelling wins).
    rail_by_key: dict[str, str] = {}
    for name in design_rails:
        rail_by_key.setdefault(_rail_key(name), name)

    bound: dict[str, list[PinEndpoint]] = {}
    noconnects: list[PinEndpoint] = []
    for p in parts_data:
        for pin in p["pins"]:
            net = pin["net"]  # canonical net, or None for a reserved/NC pin
            ep = PinEndpoint(ref=p["ref"], pin=pin["pin"])
            target = rail_by_key.get(_rail_key(net)) if net is not None else None
            if target is None:
                noconnects.append(ep)  # unused rail / signal / reserved pin
            else:
                bound.setdefault(target, []).append(ep)
    connections = [
        NetConnection(net_name=net, endpoints=eps, sheet=sheet)
        for net, eps in sorted(bound.items())
    ]
    return parts, connections, noconnects


__all__ = ["standard_form_factor_bom_delta"]
