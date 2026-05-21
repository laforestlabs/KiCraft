"""Resolve a KiCad library prefix to a concrete on-disk path.

This module bridges :mod:`kicraft.parts_library` and the KiCad stock
libraries. The parts library handles tiers 1-4 (project / home /
vendored / extras); stock KiCad at ``/usr/share/kicad/{symbols,
footprints}`` is the tier-5 last resort.

Search order, highest priority first:

1. ``<project_root>/.kicraft/parts/<library>/<library>.kicad_sym``
2. ``~/.kicraft/parts/<library>/<library>.kicad_sym``
3. ``<kicraft_install>/parts_library/<library>/<library>.kicad_sym``
4. ``$KICRAFT_EXTRA_PARTS_DIRS``-joined dirs, same shape
5. ``/usr/share/kicad/symbols/<library>.kicad_sym``

The footprint resolver is identical with ``.kicad_sym`` → ``.pretty``.
Both raise :class:`LookupError` (subclass of ``LookupError``) when no
tier supplies the requested library; callers should catch this and
present a useful error.
"""
from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

from kicraft.parts_library.loader import resolve_tier_dirs

DEFAULT_KICAD_SYMBOL_DIR: Final = Path("/usr/share/kicad/symbols")
DEFAULT_KICAD_FOOTPRINT_DIR: Final = Path("/usr/share/kicad/footprints")


class LibraryNotFoundError(LookupError):
    """No tier supplied a library file matching the requested prefix."""


def resolve_symbol_library_path(
    library: str,
    *,
    project_root: Path | None = None,
    stock_dir: Path = DEFAULT_KICAD_SYMBOL_DIR,
) -> Path:
    """Return the path to the ``.kicad_sym`` file for library prefix ``library``.

    ``project_root`` defaults to ``Path.cwd()``; pass an explicit value
    when running outside the project (e.g. from tests).
    """
    if project_root is None:
        project_root = Path.cwd()
    for _tier, base in resolve_tier_dirs(project_root):
        candidate = base / library / f"{library}.kicad_sym"
        if candidate.is_file():
            return candidate
    stock = stock_dir / f"{library}.kicad_sym"
    if stock.is_file():
        return stock
    raise LibraryNotFoundError(
        _format_not_found(library, kind="symbol", project_root=project_root, stock_dir=stock_dir)
    )


def resolve_footprint_library_path(
    library: str,
    *,
    project_root: Path | None = None,
    stock_dir: Path = DEFAULT_KICAD_FOOTPRINT_DIR,
) -> Path:
    """Return the path to the ``.pretty`` directory for ``library``."""
    if project_root is None:
        project_root = Path.cwd()
    for _tier, base in resolve_tier_dirs(project_root):
        candidate = base / library / f"{library}.pretty"
        if candidate.is_dir():
            return candidate
    stock = stock_dir / f"{library}.pretty"
    if stock.is_dir():
        return stock
    raise LibraryNotFoundError(
        _format_not_found(library, kind="footprint", project_root=project_root, stock_dir=stock_dir)
    )


def _format_not_found(
    library: str,
    *,
    kind: Literal["symbol", "footprint"],
    project_root: Path,
    stock_dir: Path,
) -> str:
    parts_tiers = [str(base / library) for _t, base in resolve_tier_dirs(project_root)]
    parts_tiers.append(str(stock_dir / library))
    return (
        f"{kind} library {library!r} not found in any tier:\n  "
        + "\n  ".join(parts_tiers)
    )


__all__ = [
    "DEFAULT_KICAD_FOOTPRINT_DIR",
    "DEFAULT_KICAD_SYMBOL_DIR",
    "LibraryNotFoundError",
    "resolve_footprint_library_path",
    "resolve_symbol_library_path",
]
