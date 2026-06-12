"""Mounting-hole synthesis: screw geometry + stock-footprint resolution.

The manual layout editor lets the user declare N mounting holes; the
schematic usually ships none. ``plan_mounting_holes`` splits the
user's holes into (a) holes mapped onto the parent's existing H-ref
footprints (legacy behaviour, position override only) and (b) surplus
holes for which the composer synthesizes a parent-local component and
the stamp subprocess loads a stock KiCad ``MountingHole.pretty``
footprint onto the board.

Only the plain (non-``_Pad``) NPTH variants are used: no copper, no
net, so the netlist, DSN export, and connectivity are untouched; the
keep-in rule area stamped around each hole is what protects routing.
The stock footprints carry ``exclude_from_pos_files exclude_from_bom``
so fab CPL/BOM exclusion is automatic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from kicraft.layout_editor.model import ManualMountingHole

# Env override for non-standard KiCad installs (also the prod box knob).
MOUNTING_HOLE_LIB_ENV = "KICRAFT_MOUNTING_HOLE_LIB"

# Synthesized refs start here: far above any plausible hand-authored
# H-ref so the alphabetic mapping of existing holes is never disturbed.
_SYNTH_REF_START = 901


@dataclass(frozen=True, slots=True)
class ScrewSpec:
    """Geometry for one fastener size.

    ``drill_mm`` is the hole (clearance) diameter incl. fab tolerance;
    ``courtyard_mm`` approximates the stock footprint's courtyard
    circle diameter, used for the synthesized component's bbox and the
    geometry validator; ``fp_name`` is the stock non-Pad NPTH footprint
    in ``MountingHole.pretty``.
    """

    screw: str
    drill_mm: float
    courtyard_mm: float
    fp_name: str


SCREW_TABLE: dict[str, ScrewSpec] = {
    "M2": ScrewSpec("M2", 2.2, 4.4, "MountingHole_2.2mm_M2"),
    "M2.5": ScrewSpec("M2.5", 2.7, 5.4, "MountingHole_2.7mm_M2.5"),
    "M3": ScrewSpec("M3", 3.2, 6.4, "MountingHole_3.2mm_M3"),
    "M4": ScrewSpec("M4", 4.3, 8.6, "MountingHole_4.3mm_M4"),
}

DEFAULT_SCREW = "M3"


def screw_spec(screw: str | None) -> ScrewSpec:
    """Spec for a screw key; unknown keys fall back to M3."""
    return SCREW_TABLE.get((screw or "").strip(), SCREW_TABLE[DEFAULT_SCREW])


_STOCK_LIB_CANDIDATES = (
    "/usr/share/kicad/footprints/MountingHole.pretty",
    "/usr/local/share/kicad/footprints/MountingHole.pretty",
)


def find_stock_mounting_hole_lib() -> Path | None:
    """Locate KiCad's stock ``MountingHole.pretty`` directory.

    Order: ``KICRAFT_MOUNTING_HOLE_LIB`` env override, then
    ``KICAD9_FOOTPRINT_DIR``/``KICAD_FOOTPRINT_DIR``, then the standard
    install paths. Returns None when nothing exists; callers needing
    synthesis should raise with an actionable message.
    """
    override = os.environ.get(MOUNTING_HOLE_LIB_ENV)
    if override:
        p = Path(override)
        if p.is_dir():
            return p
    for env in ("KICAD9_FOOTPRINT_DIR", "KICAD_FOOTPRINT_DIR"):
        base = os.environ.get(env)
        if base:
            p = Path(base) / "MountingHole.pretty"
            if p.is_dir():
                return p
    for cand in _STOCK_LIB_CANDIDATES:
        p = Path(cand)
        if p.is_dir():
            return p
    return None


def require_stock_mounting_hole_lib() -> Path:
    lib = find_stock_mounting_hole_lib()
    if lib is None:
        raise RuntimeError(
            "mounting-hole synthesis needs KiCad's stock MountingHole.pretty "
            "footprint library and none was found; install the KiCad footprint "
            f"libraries or point {MOUNTING_HOLE_LIB_ENV} at a MountingHole.pretty "
            "directory"
        )
    return lib


def allocate_synth_refs(count: int, taken: set[str]) -> list[str]:
    """``count`` fresh H9xx refs avoiding ``taken`` (case-insensitive)."""
    taken_upper = {r.upper() for r in taken}
    refs: list[str] = []
    n = _SYNTH_REF_START
    while len(refs) < count:
        ref = f"H{n}"
        if ref.upper() not in taken_upper:
            refs.append(ref)
        n += 1
    return refs


def plan_mounting_holes(
    holes: list[ManualMountingHole],
    existing_mh_refs: list[str],
    taken_refs: set[str],
) -> tuple[
    list[tuple[ManualMountingHole, str]],
    list[tuple[ManualMountingHole, str]],
]:
    """Split user holes into (mapped-to-existing, to-synthesize).

    Mapping preserves the legacy contract: holes sorted by index pair
    with the parent's existing mounting-hole refs in alphabetical
    order. Surplus holes get fresh H9xx refs that collide with neither
    the existing refs nor ``taken_refs`` (every component ref known to
    the composition, including child-leaf components).
    """
    ordered = sorted(holes, key=lambda h: h.index)
    existing_sorted = sorted(existing_mh_refs)
    mapped = list(zip(ordered, existing_sorted))
    surplus = ordered[len(existing_sorted):]
    synth_refs = allocate_synth_refs(
        len(surplus), taken_refs | set(existing_sorted)
    )
    return mapped, list(zip(surplus, synth_refs))
