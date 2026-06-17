"""Deterministic backstop: size each LED array's decoupling-cap companions to a
package that fits beside the LEDs.

A prompt nudge (server.stage_driver) asks the BOM stage to pick 0603 (or 0402 for
small LEDs); this rule GUARANTEES it deterministically so the array companions
reliably fit in the inter-row channel instead of being pushed into a perimeter
ring. We do NOT measure footprint geometry at runtime -- the package is chosen
from a static LED-package -> cap-package table, keyed on the package token read
straight out of the LED footprint *name*. Simple, file-I/O-free, testable.

It only ever SHRINKS a cap (never enlarges -- respects "go larger only if the
value isn't available in 0603 or smaller") and only touches GENERIC stock
``Capacitor_SMD`` caps with no MPN, so a deliberately-chosen vendored/MPN'd part
keeps its package. Idempotent. LOUD: every change is recorded in
``bom.assumptions`` (a visible engineering decision, never a silent degrade --
see kicraft-no-fallbacks-fail-loudly).

This rule reasons about package GEOMETRY only -- it has no notion of capacitance
(``BomPart`` carries none). A bulk cap that needs a larger body for its value
should be pinned to an MPN/vendored footprint, which the guards then leave alone.
"""
from __future__ import annotations

import re

from kicraft.design.models import BOM
from kicraft.design.synthesis.array_decaps import _decap_membership, _ref_sort_key

# LED package token (read from the LED footprint NAME) -> decoupling-cap package.
# WS2812-class LEDs name their body size in 0.1 mm (5050 = 5.0 mm). LEDs at or
# above 2.5 mm have room for a 0603 cap beside them; smaller ones need 0402.
LED_PACKAGE_TO_CAP = {
    "5050": "0603", "3535": "0603", "3528": "0603", "2727": "0603", "2525": "0603",
    "2020": "0402", "1717": "0402", "1515": "0402", "1313": "0402", "1010": "0402",
}
# LED package not recognised -> the house default (also the global passive default).
DEFAULT_CAP_PACKAGE = "0603"

# cap package -> stock footprint id. Only the two packages this rule ever assigns.
_CAP_FOOTPRINT = {
    "0603": "Capacitor_SMD:C_0603_1608Metric",
    "0402": "Capacitor_SMD:C_0402_1005Metric",
}
# Imperial size-code rank, smallest first, for the downsize-only comparison.
_PKG_RANK = {"0201": 0, "0402": 1, "0603": 2, "0805": 3, "1206": 4, "1210": 5, "1812": 6}
# A canonical stock cap footprint (no suffix -> excludes vendored libs and the
# _HandSolder variants whose courtyard is deliberately oversized).
_STOCK_CAP_RE = re.compile(r"^Capacitor_SMD:C_(\d{4})_\d+Metric$")


def _led_cap_package(led_footprints: list[str]) -> str:
    """Pick the cap package for an array from its LED members' footprint strings.

    Plain substring match on the known package tokens; the SMALLEST mapped package
    wins so a mixed array uses the size its smallest LED needs. Falls back to
    ``DEFAULT_CAP_PACKAGE`` when no token is recognised.
    """
    best: str | None = None
    for fp in led_footprints:
        for token, pkg in LED_PACKAGE_TO_CAP.items():
            if token in fp and (best is None or _PKG_RANK[pkg] < _PKG_RANK[best]):
                best = pkg
    return best or DEFAULT_CAP_PACKAGE


def downsize_array_decap_footprints(bom: BOM) -> list[str]:
    """Resize each LED array's decoupling-cap companions to the LED-matched package.

    0603 by default; 0402 when the array's LEDs are smaller than 2.5 mm. Generic
    stock caps only, no-MPN, downsize-only (never enlarges), idempotent. Mutates
    ``bom`` in place; returns the sorted list of changed refs (``[]`` when nothing
    changed). LOUD -- each change recorded in ``bom.assumptions``.
    """
    if not bom.arrays:
        return []

    is_decap = _decap_membership(bom)
    parts_by_ref = {p.ref: p for p in bom.parts}
    sheet_by_ref = {p.ref: p.sheet for p in bom.parts}
    fp_by_ref = {p.ref: p.footprint for p in bom.parts}

    changed: list[str] = []
    notes: list[str] = []
    for spec in bom.arrays:
        members = set(spec.refs)
        sheets = {sheet_by_ref.get(r) for r in spec.refs}
        target = _led_cap_package([fp_by_ref.get(r, "") for r in spec.refs])
        target_fp = _CAP_FOOTPRINT[target]
        companions = sorted(
            (
                p.ref
                for p in bom.parts
                if p.sheet in sheets and p.ref not in members and is_decap.get(p.ref, False)
            ),
            key=_ref_sort_key,
        )
        for ref in companions:
            part = parts_by_ref[ref]
            if part.mpn:  # a deliberately-chosen real part -> leave its package
                continue
            m = _STOCK_CAP_RE.match(part.footprint)
            if m is None:  # vendored / non-canonical footprint -> leave it
                continue
            cur = m.group(1)
            if _PKG_RANK.get(cur, 99) <= _PKG_RANK[target]:
                continue  # already at/below the target -> never enlarge (idempotent)
            part.footprint = target_fp
            changed.append(ref)
            notes.append(
                f"LED array ({spec.refs[0]}..{spec.refs[-1]}) decoupling cap {ref} "
                f"resized {cur} -> {target} to fit beside the LEDs"
            )

    if not changed:
        return []
    for note in notes:
        print(f"  [synth] {note}")
        bom.assumptions.append(note)
    return sorted(set(changed), key=_ref_sort_key)
