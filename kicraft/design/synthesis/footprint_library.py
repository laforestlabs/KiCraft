"""search_footprints / lookup_footprint: keyword discovery + verification of stock
KiCad footprint ids.

The footprint analog of :mod:`kicraft.design.synthesis.symbol_library`. The BOM
stage's weak server model cannot recall exact stock footprint names and guesses
plausible-but-nonexistent ones (e.g. ``Connector_BarrelJack:BarrelJack_2.1mm_P5.5mm``,
which has no ``.kicad_mod``); these let it find the real ``Library:Name`` by keyword
instead. Footprints live one-per-file as ``<stock>/<Library>.pretty/<Name>.kicad_mod``,
so discovery is a filename glob (no file reads), and the footprint name itself is
verbose enough that substring matching on the ``Library:Name`` id is effective.

Resolution and loadability validation share :func:`load_footprint`, the single
``pcbnew.FootprintLoad`` seam used by BOM commit, lookup tools, and synthesis.
"""
from __future__ import annotations

from pathlib import Path

from .parts_lookup import (
    DEFAULT_KICAD_FOOTPRINT_DIR,
    LibraryNotFoundError,
    resolve_footprint_library_path,
)


class FootprintNotFoundError(LookupError):
    """Raised when a footprint cannot be loaded from the resolver chain."""

# Query terms that describe the *kind* of thing being searched, not the part, and so
# never appear in a footprint id. The model habitually appends "footprint" to queries
# (e.g. "pinheader 2x08 footprint"); since matching ANDs every term, that one token
# would otherwise zero the result. ("smd" is NOT a stopword — it is a real token in
# many footprint library names, e.g. Inductor_SMD / LED_SMD / Capacitor_SMD.)
_STOPWORDS = frozenset({"footprint", "footprints"})


def load_footprint(
    pcbnew_mod,
    library: str,
    name: str,
    *,
    project_root: Path | None = None,
    stock_dir: Path = DEFAULT_KICAD_FOOTPRINT_DIR,
):
    """Resolve and load one footprint through the synthesis resolver."""
    try:
        lib_dir = resolve_footprint_library_path(
            library, project_root=project_root, stock_dir=stock_dir
        )
    except LibraryNotFoundError as exc:
        raise FootprintNotFoundError(str(exc)) from exc
    try:
        fp = pcbnew_mod.FootprintLoad(str(lib_dir), name)
    except Exception as exc:  # noqa: BLE001
        raise FootprintNotFoundError(
            f"could not load {library}:{name} from {lib_dir}: {exc}"
        ) from exc
    if fp is None:
        raise FootprintNotFoundError(
            f"FootprintLoad returned None for {library}:{name} from {lib_dir}"
        )

    from kicraft.parts_library.footprint_courtyard import (
        normalize_pth_pads_for_fab,
        repair_malformed_courtyard,
    )

    repair_malformed_courtyard(fp)
    pth_changes = normalize_pth_pads_for_fab(fp)
    if pth_changes:
        print(
            f"footprint {library}:{name}: normalized {len(pth_changes)} PTH "
            f"pad(s) to fab floors ({'; '.join(pth_changes[:4])}"
            f"{'; ...' if len(pth_changes) > 4 else ''})"
        )
    return fp, lib_dir


def search_footprints(
    query: str,
    *,
    stock_dir: Path = DEFAULT_KICAD_FOOTPRINT_DIR,
    limit: int = 40,
) -> list[str]:
    """Return up to ``limit`` stock KiCad ``Library:Name`` footprint ids whose id
    contains every (non-stopword) whitespace-separated term in ``query``
    (case-insensitive).

    Lets the BOM stage discover a real footprint id by keyword instead of guessing
    one (e.g. ``"pinheader 2x08"`` -> ``Connector_PinHeader_2.54mm:PinHeader_2x08_
    P2.54mm_Vertical``). Each ``<Library>.pretty/<Name>.kicad_mod`` under ``stock_dir``
    contributes the id ``"<Library>:<Name>"``.
    """
    terms = [t for t in (w.lower() for w in (query or "").split())
             if t and t not in _STOPWORDS]
    if not terms or not stock_dir.is_dir():
        return []
    matches: list[str] = []
    seen: set[str] = set()
    for pretty in sorted(stock_dir.glob("*.pretty")):
        libname = pretty.stem
        for mod in sorted(pretty.glob("*.kicad_mod")):
            fp_id = f"{libname}:{mod.stem}"
            key = fp_id.lower()
            if fp_id in seen or not all(t in key for t in terms):
                continue
            seen.add(fp_id)
            matches.append(fp_id)
            if len(matches) >= limit:
                return matches
    return matches


def lookup_footprint(
    footprint: str,
    *,
    project_root: Path | None = None,
    stock_dir: Path = DEFAULT_KICAD_FOOTPRINT_DIR,
) -> dict:
    """Load a ``Library:Name`` footprint and report its actual pad count.

    Uses the same resolver and ``pcbnew.FootprintLoad`` seam as synthesis, so a
    file that exists but KiCad cannot load is rejected before board generation.
    """
    library, _, name = (footprint or "").partition(":")
    if not library or not name:
        raise ValueError(f"footprint {footprint!r} is not 'Library:Name'")
    import pcbnew

    fp, lib_dir = load_footprint(
        pcbnew,
        library,
        name,
        project_root=project_root,
        stock_dir=stock_dir,
    )
    return {
        "footprint": f"{library}:{name}",
        "pad_count": len(list(fp.Pads())),
        "resolved_directory": str(lib_dir),
    }


__all__ = [
    "FootprintNotFoundError",
    "load_footprint",
    "lookup_footprint",
    "search_footprints",
]
