"""Targeted S-expression text edits for refdes renumbering.

This is **not** a general-purpose sexpr parser. It performs three
narrow rewrites against KiCad 9 ``.kicad_sch`` / ``.kicad_pcb`` text:

1. ``(property "Reference" "<ref>" ...)`` -- the canonical refdes in
   both schematics (symbol instances) and PCBs (footprint properties).
   We skip lib_symbols definitions because their value is the letter
   prefix (``"U"``, ``"R"``) without digits and so doesn't match
   ``^[A-Z]+[0-9]+$``.

2. ``(reference "<ref>")`` -- per-instance refs inside
   ``(path "/<uuid>" (reference "<ref>") (unit 1) ...)`` blocks in
   schematics.

3. ``(fp_text reference "<ref>" ...)`` -- legacy silkscreen refs.
   Present in older PCBs; rewritten if found.

Plus a defensive scan for ``(fp_text user "<looks-like-ref>" ...)``
that catches hand-placed silk labels whose value matches the renumber
map -- spec §3 "Defensive scan." We log a warning and rewrite the
string.

All rewrites are byte-level on the captured ref string only.
Surrounding sexpr structure (positions, rotations, layers, font sizes,
hidden flags, UUIDs) is left exactly as-is.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)


# (property "Reference" "U1"  -- only refs matching [A-Z]+[0-9]+ are
# rewritten; the letter-only forms in lib_symbols pass through.
_PROPERTY_REFERENCE_RE = re.compile(
    r'(\(property\s+"Reference"\s+")([A-Z]+[0-9]+)(")'
)

# (reference "U1")  -- inside (path ... ) instance blocks. Whitespace
# inside is permissive.
_REFERENCE_INSTANCE_RE = re.compile(
    r'(\(reference\s+")([A-Z]+[0-9]+)("\s*\))'
)

# (fp_text reference "U1"  -- legacy KiCad <= 8 silkscreen refdes.
_FP_TEXT_REFERENCE_RE = re.compile(
    r'(\(fp_text\s+reference\s+")([A-Z]+[0-9]+)(")'
)

# (fp_text user "..."  -- catches hand-placed silk labels whose value
# happens to look like a refdes that's in the renumber map.
_FP_TEXT_USER_RE = re.compile(
    r'(\(fp_text\s+user\s+")([A-Z]+[0-9]+)(")'
)


def _rewrite_with(
    text: str,
    pattern: re.Pattern[str],
    ref_map: dict[str, str],
    *,
    label: str,
) -> tuple[str, int]:
    """Apply ``ref_map`` to every ``[A-Z]+[0-9]+`` capture from ``pattern``.

    Returns (new_text, replacement_count). Captures whose value is not
    in ``ref_map`` are left alone (this happens when the file already
    contains refs from prior leaves in the same project).
    """
    count = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal count
        old_ref = m.group(2)
        new_ref = ref_map.get(old_ref)
        if new_ref is None:
            return m.group(0)
        count += 1
        log.debug("renumber [%s] %s -> %s", label, old_ref, new_ref)
        return f"{m.group(1)}{new_ref}{m.group(3)}"

    return pattern.sub(_sub, text), count


def renumber_schematic_text(
    text: str, ref_map: dict[str, str]
) -> tuple[str, dict[str, int]]:
    """Apply ``ref_map`` to a ``.kicad_sch`` file's text.

    Returns (new_text, counts) where ``counts`` reports how many
    matches each pattern produced -- useful for telemetry and tests.
    """
    text, c_prop = _rewrite_with(
        text, _PROPERTY_REFERENCE_RE, ref_map, label="sch:property-reference"
    )
    text, c_inst = _rewrite_with(
        text, _REFERENCE_INSTANCE_RE, ref_map, label="sch:reference-instance"
    )
    return text, {
        "property_reference": c_prop,
        "reference_instance": c_inst,
    }


def renumber_pcb_text(
    text: str, ref_map: dict[str, str]
) -> tuple[str, dict[str, int]]:
    """Apply ``ref_map`` to a ``.kicad_pcb`` file's text.

    Performs the three primary rewrites plus the defensive scan. Returns
    (new_text, counts) keyed by pattern label.
    """
    text, c_prop = _rewrite_with(
        text, _PROPERTY_REFERENCE_RE, ref_map, label="pcb:property-reference"
    )
    text, c_fp = _rewrite_with(
        text, _FP_TEXT_REFERENCE_RE, ref_map, label="pcb:fp_text-reference"
    )
    text, c_user = _rewrite_with(
        text, _FP_TEXT_USER_RE, ref_map, label="pcb:fp_text-user"
    )
    if c_user:
        log.warning(
            "renumber_pcb_text rewrote %d (fp_text user) silk label(s) "
            "matching the renumber map -- review for intentional silk text "
            "that happened to look like a refdes",
            c_user,
        )
    return text, {
        "property_reference": c_prop,
        "fp_text_reference": c_fp,
        "fp_text_user": c_user,
    }


__all__ = [
    "renumber_pcb_text",
    "renumber_schematic_text",
]
