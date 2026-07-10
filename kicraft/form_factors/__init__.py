"""Standard mechanical form-factor templates.

A brief that names a standard form factor -- "Arduino Uno shield", "Raspberry Pi
HAT", "Feather" -- is stating a *hard* mechanical contract: a fixed board
outline, fixed connector positions, and often a fixed mounting-hole pattern. The
rest of the pipeline sizes the board from whatever the solver places, so that
intent is silently dropped and the board comes out mechanically non-conformant
(see ``docs/plans/standard-form-factor-templates.md`` and KC-99A9M8).

This module is the data + lookup layer (PR1): it holds one
:class:`FormFactorTemplate` per standard and matches a brief to a template by
alias. It changes nothing downstream on its own -- the intent stage records the
matched key in ``FormFactor.standard`` so state.json surfaces it. PR2 emits the
fixed outline + pre-locked connectors into placement; PR3 gates conformance.

**Coordinate frame.** Board-local, origin at the bottom-left corner, X to the
right, Y up (millimetres). A connector's ``x_mm``/``y_mm`` is its pin-1 centre;
pins advance along ``axis`` at ``pitch_mm``.

**``validated``.** A template's *structure* (pin counts, roles, pitch, the
inter-header offset, board size, pin semantics) is authored from the published
spec and is correct. The *absolute datum* (where pin 1 sits relative to the
board corner) still needs a cross-check against the official mechanical drawing;
until then ``validated`` is False and PR2's placement path must refuse to lay a
board out on it (a deliberate interlock so unverified geometry never ships).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FixedConnector:
    """A connector pinned to an exact board-local position by the standard."""

    role: str  # canonical role slug, e.g. "digital_high"
    pins: int
    footprint: str
    x_mm: float  # pin-1 centre (board frame: origin bottom-left, Y up)
    y_mm: float
    axis: str  # "x" or "y": the direction pin numbers advance
    net_by_pin: tuple[str, ...]  # canonical net per pin, in pin order
    rotation_deg: float = 0.0
    pitch_mm: float = 2.54

    def pin_positions(self) -> list[tuple[str, float, float]]:
        """``(net, x_mm, y_mm)`` for every pin, in pin order."""
        out: list[tuple[str, float, float]] = []
        for i, net in enumerate(self.net_by_pin):
            dx = self.pitch_mm * i if self.axis == "x" else 0.0
            dy = self.pitch_mm * i if self.axis == "y" else 0.0
            out.append((net, self.x_mm + dx, self.y_mm + dy))
        return out

    def __post_init__(self) -> None:
        if self.pins != len(self.net_by_pin):
            raise ValueError(
                f"{self.role}: pins={self.pins} != len(net_by_pin)={len(self.net_by_pin)}"
            )
        if self.axis not in ("x", "y"):
            raise ValueError(f"{self.role}: axis must be 'x' or 'y', got {self.axis!r}")


@dataclass(frozen=True)
class MountingHole:
    x_mm: float
    y_mm: float
    diameter_mm: float = 3.2


@dataclass(frozen=True)
class FormFactorTemplate:
    """A named standard mechanical form factor."""

    key: str
    display_name: str
    aliases: tuple[str, ...]
    board_width_mm: float
    board_height_mm: float
    fixed_connectors: tuple[FixedConnector, ...]
    mounting_holes: tuple[MountingHole, ...] = ()
    outline_note: str = ""
    provenance: str = ""
    validated: bool = False

    def canonical_nets(self) -> set[str]:
        """Every canonical net the standard's headers expose (D0.., A0.., rails)."""
        return {net for c in self.fixed_connectors for net in c.net_by_pin}

    def to_autoplacer_dict(self) -> dict:
        """Serialize the template geometry for ``<stem>_autoplacer.json``.

        A self-contained, JSON-safe record the compose pipeline can consume:
        board size, every fixed connector (role/footprint/pin-1/axis/pitch/pin
        nets), and the mounting holes. Carries ``validated`` so a consumer gates
        itself on the datum (PR2's placement path must refuse an unvalidated
        template). Pure data -- emitting it changes no build behavior by itself.
        """
        return {
            "key": self.key,
            "display_name": self.display_name,
            "validated": self.validated,
            "board_width_mm": self.board_width_mm,
            "board_height_mm": self.board_height_mm,
            "outline_note": self.outline_note,
            "provenance": self.provenance,
            "fixed_connectors": [
                {
                    "role": c.role,
                    "pins": c.pins,
                    "footprint": c.footprint,
                    "x_mm": c.x_mm,
                    "y_mm": c.y_mm,
                    "axis": c.axis,
                    "rotation_deg": c.rotation_deg,
                    "pitch_mm": c.pitch_mm,
                    "nets": list(c.net_by_pin),
                }
                for c in self.fixed_connectors
            ],
            "mounting_holes": [
                {"x_mm": h.x_mm, "y_mm": h.y_mm, "diameter_mm": h.diameter_mm}
                for h in self.mounting_holes
            ],
        }


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

# Arduino Uno R3 shield. Board 68.58 x 53.34 mm (2.7" x 2.1"). Four 2.54 mm
# headers; the two digital headers are deliberately offset so the D7<->D8 gap is
# 0.16" (4.064 mm), not 0.1" -- the reason generic protoshields carry that quirk.
# Frame: origin bottom-left, USB/barrel-jack on the LEFT edge, Y up.
#   top edge (Y=50.8):  [SCL SDA AREF GND D13 D12 D11 D10 D9 D8] .. gap .. [D7 D6 D5 D4 D3 D2 D1 D0]
#   bottom edge (Y=2.54): [IOREF RESET +3V3 +5V GND GND VIN NC]  ..  [A0 A1 A2 A3 A4 A5]
# Pin ROLES/COUNTS/pitch/offset are from the R3 spec; the absolute datum is
# provisional pending the official DXF -> validated=False.
_ARDUINO_UNO_SHIELD = FormFactorTemplate(
    key="arduino_uno_shield",
    display_name="Arduino Uno R3 shield",
    aliases=(
        "arduino uno shield",
        "arduino-uno shield",
        "arduino uno-format",
        "arduino-uno-format",
        "uno shield",
        "arduino shield",
        "arduino r3 shield",
        "arduino uno r3 shield",
    ),
    board_width_mm=68.58,
    board_height_mm=53.34,
    fixed_connectors=(
        FixedConnector(
            role="digital_high",
            pins=10,
            footprint="Connector_PinSocket_2.54mm:PinSocket_1x10_P2.54mm_Vertical",
            x_mm=17.526,
            y_mm=50.8,
            axis="x",
            net_by_pin=("SCL", "SDA", "AREF", "GND", "D13", "D12", "D11", "D10", "D9", "D8"),
        ),
        FixedConnector(
            role="digital_low",
            pins=8,
            footprint="Connector_PinSocket_2.54mm:PinSocket_1x08_P2.54mm_Vertical",
            x_mm=44.45,
            y_mm=50.8,
            axis="x",
            net_by_pin=("D7", "D6", "D5", "D4", "D3", "D2", "D1", "D0"),
        ),
        FixedConnector(
            role="power",
            pins=8,
            footprint="Connector_PinSocket_2.54mm:PinSocket_1x08_P2.54mm_Vertical",
            x_mm=15.24,
            y_mm=2.54,
            axis="x",
            net_by_pin=("IOREF", "RESET", "+3V3", "+5V", "GND", "GND", "VIN", "NC"),
        ),
        FixedConnector(
            role="analog",
            pins=6,
            footprint="Connector_PinSocket_2.54mm:PinSocket_1x06_P2.54mm_Vertical",
            x_mm=41.91,
            y_mm=2.54,
            axis="x",
            net_by_pin=("A0", "A1", "A2", "A3", "A4", "A5"),
        ),
    ),
    mounting_holes=(
        MountingHole(15.24, 50.8),
        MountingHole(66.04, 35.56),
        MountingHole(66.04, 7.62),
        MountingHole(13.97, 2.54),
    ),
    outline_note=(
        "68.58 x 53.34 mm bounding rect; the R3 outline also has a small "
        "top-right corner notch not modeled here yet."
    ),
    provenance="Arduino Uno R3 spec (2.7x2.1in, 2.54mm headers, 0.16in D7-D8 offset)",
    validated=False,
)


_TEMPLATES: dict[str, FormFactorTemplate] = {
    t.key: t for t in (_ARDUINO_UNO_SHIELD,)
}

# Alias -> key, longest-alias-first so "arduino uno shield" wins over "arduino
# shield". Matching is whitespace-insensitive and word-boundaried.
_ALIAS_TO_KEY: list[tuple[str, str]] = sorted(
    ((alias, t.key) for t in _TEMPLATES.values() for alias in t.aliases),
    key=lambda kv: -len(kv[0]),
)


def get_template(key: str | None) -> FormFactorTemplate | None:
    """Look a template up by its registry key."""
    if not key:
        return None
    return _TEMPLATES.get(key.strip().lower())


def all_templates() -> list[FormFactorTemplate]:
    return list(_TEMPLATES.values())


def match_standard(text: str) -> FormFactorTemplate | None:
    """Deterministically match a brief to a standard form-factor template.

    Conservative and whitespace-insensitive: an alias matches only on word
    boundaries, and the longest alias wins so a specific standard is preferred
    over a generic one. Returns ``None`` when nothing matches (the common case),
    leaving the default free-outline path untouched.
    """
    if not text or not text.strip():
        return None
    norm = re.sub(r"[\s\-]+", " ", text.lower())
    for alias, key in _ALIAS_TO_KEY:
        alias_norm = re.sub(r"[\s\-]+", " ", alias)
        if re.search(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", norm):
            return _TEMPLATES[key]
    return None


__all__ = [
    "FixedConnector",
    "MountingHole",
    "FormFactorTemplate",
    "get_template",
    "all_templates",
    "match_standard",
]
