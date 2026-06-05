"""search_footprints / lookup_footprint: keyword discovery + verification of stock
KiCad footprint ids.

The footprint analog of :mod:`kicraft.design.synthesis.symbol_library`. The BOM
stage's weak server model cannot recall exact stock footprint names and guesses
plausible-but-nonexistent ones (e.g. ``Connector_BarrelJack:BarrelJack_2.1mm_P5.5mm``,
which has no ``.kicad_mod``); these let it find the real ``Library:Name`` by keyword
instead. Footprints live one-per-file as ``<stock>/<Library>.pretty/<Name>.kicad_mod``,
so discovery is a filename glob (no file reads), and the footprint name itself is
verbose enough that substring matching on the ``Library:Name`` id is effective.

Resolution/validation already exist in :mod:`parts_lookup` (``resolve_footprint_
library_path``) and ``cli_app._unresolved_footprints``; this module only adds the
missing *discovery* layer.
"""
from __future__ import annotations

import re
from pathlib import Path

from .parts_lookup import (
    DEFAULT_KICAD_FOOTPRINT_DIR,
    LibraryNotFoundError,
    resolve_footprint_library_path,
)


class FootprintNotFoundError(LookupError):
    """Raised when a footprint library is missing or the footprint isn't in it."""


# Query terms that describe the *kind* of thing being searched, not the part, and so
# never appear in a footprint id. The model habitually appends "footprint" to queries
# (e.g. "pinheader 2x08 footprint"); since matching ANDs every term, that one token
# would otherwise zero the result. ("smd" is NOT a stopword — it is a real token in
# many footprint library names, e.g. Inductor_SMD / LED_SMD / Capacitor_SMD.)
_STOPWORDS = frozenset({"footprint", "footprints"})


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
    """Verify a ``Library:Name`` footprint resolves to a real ``.kicad_mod`` and
    report its pad count (the footprint analog of ``lookup_symbol``'s pin list).

    Resolution goes through the same parts-library 4-tier + stock search as the
    BOM-commit check (:func:`parts_lookup.resolve_footprint_library_path`).

    Raises:
        ValueError: ``footprint`` is not in ``Library:Name`` form.
        FootprintNotFoundError: library missing, or the ``.kicad_mod`` not in it.
    """
    library, _, name = (footprint or "").partition(":")
    if not library or not name:
        raise ValueError(f"footprint {footprint!r} is not 'Library:Name'")
    try:
        pretty = resolve_footprint_library_path(
            library, project_root=project_root, stock_dir=stock_dir
        )
    except LibraryNotFoundError as exc:
        raise FootprintNotFoundError(str(exc)) from exc
    mod = pretty / f"{name}.kicad_mod"
    if not mod.is_file():
        raise FootprintNotFoundError(f"no '{name}.kicad_mod' in {pretty}")
    text = mod.read_text(encoding="utf-8", errors="ignore")
    pad_count = len(re.findall(r"\(pad\s", text))
    return {"footprint": f"{library}:{name}", "pad_count": pad_count}


__all__ = [
    "FootprintNotFoundError",
    "lookup_footprint",
    "search_footprints",
]
