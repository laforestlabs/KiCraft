"""Turn a form-factor template into the placement + netlist scaffold.

"Replace & rewire" means a standard's headers are REAL parts on the board (you
solder them; they must appear in the BOM, schematic, ERC and netlist -- never
phantom parts injected only at compose). This module is the pure data layer that
both sides consume:

* synthesis reconcile -> emits :func:`standard_header_parts` as BOM parts, pins
  them with :func:`standard_placements`, and binds power/ground with
  :func:`canonical_power_bindings`;
* compose -> reads the same placements to lock the headers at their fixed
  positions inside the fixed outline.

Pure functions over a :class:`~kicraft.form_factors.FormFactorTemplate`; no
pydantic, no I/O, so it is trivially testable and has no import cycle with the
design models. Refs are assigned deterministically from ``ref_start``.
"""

from __future__ import annotations

from . import FormFactorTemplate

# Nets that bind deterministically -- the rails an Arduino header always carries,
# regardless of what the shield's function does. Signal pins (D0..D13/A0..A5) are
# design-specific and are NOT auto-bound here (that is the wiring stage's job).
CANONICAL_RAILS = ("+5V", "+3V3", "GND", "VIN", "IOREF", "RESET", "AREF")

# Single-row pin-count -> Connector_Generic symbol.
_CONN_SYMBOL = {
    6: "Connector_Generic:Conn_01x06",
    8: "Connector_Generic:Conn_01x08",
    10: "Connector_Generic:Conn_01x10",
}


def standard_header_parts(
    template: FormFactorTemplate, *, ref_start: int = 1, sheet: str = "INTERFACE"
) -> list[dict]:
    """One dict per fixed connector: a placeable BOM part + its pin map.

    Each entry has ``ref``/``value``/``symbol``/``footprint``/``sheet`` (the BOM
    fields) plus ``role`` and ``pins`` = ``[{pin, net, x_mm, y_mm}]`` (1-indexed
    pin numbers at their board-local positions). ``NC`` pins carry net ``None``.
    """
    parts: list[dict] = []
    ref_n = ref_start
    for conn in template.fixed_connectors:
        symbol = _CONN_SYMBOL.get(conn.pins)
        if symbol is None:  # 2xN / non-standard row -> not modeled here yet
            continue
        pins = []
        for i, (net, x, y) in enumerate(conn.pin_positions(), start=1):
            pins.append(
                {
                    "pin": str(i),
                    "net": None if net == "NC" else net,
                    "x_mm": round(x, 3),
                    "y_mm": round(y, 3),
                }
            )
        parts.append(
            {
                "ref": f"J{ref_n}",
                "role": conn.role,
                "value": f"{template.display_name} {conn.role}",
                "symbol": symbol,
                "footprint": conn.footprint,
                "sheet": sheet,
                "x_mm": round(conn.x_mm, 3),
                "y_mm": round(conn.y_mm, 3),
                "rotation_deg": conn.rotation_deg,
                "pins": pins,
            }
        )
        ref_n += 1
    return parts


def standard_placements(parts: list[dict]) -> dict[str, dict]:
    """``ref -> {x_mm, y_mm, rotation_deg, locked}`` exact-lock placements.

    Fed to compose so each standard header is frozen at its fixed board position
    and the solver auto-places everything else around it.
    """
    return {
        p["ref"]: {
            "x_mm": p["x_mm"],
            "y_mm": p["y_mm"],
            "rotation_deg": p["rotation_deg"],
            "locked": True,
        }
        for p in parts
    }


def canonical_power_bindings(parts: list[dict]) -> dict[str, list[tuple[str, str]]]:
    """``net -> [(ref, pin), ...]`` for the rails that bind deterministically.

    Only the canonical rails (:data:`CANONICAL_RAILS`) -- +5V/+3V3/GND/VIN/…, the
    ones a header always carries. Signal pins are left for the wiring stage. A
    net that lands on several header pins (e.g. the two GND pins) lists them all.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    for p in parts:
        for pin in p["pins"]:
            net = pin["net"]
            if net in CANONICAL_RAILS:
                out.setdefault(net, []).append((p["ref"], pin["pin"]))
    return out


__all__ = [
    "CANONICAL_RAILS",
    "standard_header_parts",
    "standard_placements",
    "canonical_power_bindings",
]
