"""Deterministic board form-factor (outline shape) extraction.

The intent stage captures the user's requested board shape as a
:class:`~kicraft.design.models.FormFactor`. The model may set it directly, but a
brief like *"round 50 mm coaster"* must be honored even when the LLM leaves it
unset -- so this module provides a deterministic keyword extractor that runs at
the intent stage-commit (mirroring the wiring-stage normalizers
``bridge_duplicate_pins`` / ``reconcile_inter_sheet_nets``).

It is intentionally conservative. Hardware briefs are full of words that look
like shapes but are not (*star* ground/topology, *hex* inverter, *heart* rate
sensor, *ground* plane, *round*-robin), so:

- **Strong** patterns are specific enough to fire on their own
  (``hexagonal``, ``snowman``, ``circular``, ``rounded corners``, ``*-shaped``).
- **Weak** bare words (``round``, ``disc``, ``disk``) fire only when a shape /
  board context word sits nearby, and never on ``round up`` / ``round robin``.

The extractor never invents a rectangle: a no-match returns ``None`` so the
default rectangular path is untouched.
"""

from __future__ import annotations

import re

from kicraft.design.models import FormFactor
from kicraft.form_factors import match_standard

# Strong shape patterns: (compiled regex, canonical shape). Order matters only
# in that the FIRST match wins, so list the more specific shapes first.
_STRONG_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsnowman\b", re.I), "snowman"),
    (re.compile(r"\bhexagonal\b|\bhexagon\b|\bhex[-\s]shaped\b", re.I), "hexagon"),
    (re.compile(r"\boctagonal\b|\boctagon\b", re.I), "octagon"),
    (re.compile(r"\bpentagonal\b|\bpentagon\b", re.I), "pentagon"),
    (re.compile(r"\btriangular\b|\btriangle[-\s]shaped\b", re.I), "triangle"),
    (re.compile(r"\bheart[-\s]shaped\b", re.I), "heart"),
    (re.compile(r"\bstar[-\s]shaped\b", re.I), "star"),
    (re.compile(r"\b(?:gear|cog)[-\s]shaped\b", re.I), "gear"),
    (re.compile(r"\bchamfer(?:ed)?\b|\bbevell?ed[-\s]corners?\b", re.I), "chamfered_rect"),
    (
        re.compile(
            r"\brounded[-\s]?corners?\b|\brounded[-\s]?rect(?:angle)?\b"
            r"|\bfilleted[-\s]corners?\b",
            re.I,
        ),
        "rounded_rect",
    ),
    (
        re.compile(
            r"\bcircular\b|\bcircle\b|\bcoaster\b"
            r"|\bdis[ck][-\s]shaped\b|\bpuck[-\s]shaped\b"
            # Imperative / predicate "round" needs no board-context word.
            r"|\b(?:make\s+it|shaped?|keep\s+it|want\s+it|should\s+be|be)\s+round\b",
            re.I,
        ),
        "circle",
    ),
]

# "Nth round" / "round of testing" idioms that must NOT be read as a circle even
# when a board-context word is present.
_ROUND_IDIOM = re.compile(
    r"\b(?:\d+|first|second|third|fourth|fifth|next|last|final|another|each|every)"
    r"\s+round\b",
    re.I,
)

# Weak (ambiguous) bare words → shape. Only honored when `_CONTEXT` also matches
# somewhere in the text, and never on the `round up`/`round robin` idioms.
_WEAK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bround\b(?!\s+(?:up|down|trip|robin|to|off|out|of)\b)", re.I), "circle"),
    (re.compile(r"\bdis[ck]\b", re.I), "circle"),
]

_CONTEXT = re.compile(
    r"\b(?:board|pcb|outline|shape[ds]?|form[-\s]?factor|enclosure|case|silhouette"
    r"|diameter|dia)\b|[Ø⌀]",
    re.I,
)

# Headline size: a diameter symbol, or a number adjacent to a diameter word.
_SIZE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[Ø⌀]\s*(\d+(?:\.\d+)?)\s*(?:mm)?", re.I),
    re.compile(r"(\d+(?:\.\d+)?)\s*mm\s*(?:diameter|dia\b|round|circular|circle|wide)", re.I),
    re.compile(r"(?:diameter|dia\b)[^\d]{0,8}(\d+(?:\.\d+)?)\s*mm", re.I),
]


def _classify(text: str) -> tuple[str | None, str | None]:
    """Return ``(shape, matched_phrase)`` or ``(None, None)``."""
    for pat, shape in _STRONG_PATTERNS:
        m = pat.search(text)
        if m:
            return shape, m.group(0).strip()
    if _CONTEXT.search(text):
        for pat, shape in _WEAK_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            # Precision bias: drop a bare "round" that is really an "Nth round"
            # idiom rather than a board shape.
            if shape == "circle" and m.group(0).lower() == "round" and _ROUND_IDIOM.search(text):
                continue
            return shape, m.group(0).strip()
    return None, None


def _extract_size_mm(text: str) -> float | None:
    for pat in _SIZE_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except (TypeError, ValueError):
                continue
    return None


def extract_form_factor(text: str) -> FormFactor | None:
    """Deterministically classify a board form factor from free text.

    Two levels, in precedence order:

    1. A **named standard** mechanical form factor ("Arduino Uno shield", a
       future HAT/Feather) -- a HARD outline + fixed-connector contract. Matched
       via :mod:`kicraft.form_factors`; the returned :class:`FormFactor` carries
       ``standard`` (the registry key) plus the template's rectangular bounding
       shape so the default rect path still has a size to work with.
    2. A **shape** ("round 50 mm coaster", hex, rounded corners) -- an advisory
       outline shape only.

    Returns ``None`` when the text requests neither (the common case), leaving
    the default rectangular path untouched.
    """
    if not text or not text.strip():
        return None
    template = match_standard(text)
    if template is not None:
        return FormFactor(
            shape="rect",
            standard=template.key,
            size_mm=max(template.board_width_mm, template.board_height_mm),
            note=template.display_name,
        )
    shape, phrase = _classify(text)
    if shape is None or shape == "rect":
        return None
    return FormFactor(shape=shape, size_mm=_extract_size_mm(text), note=phrase)
