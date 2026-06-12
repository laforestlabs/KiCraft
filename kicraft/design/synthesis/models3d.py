"""Stage part-bundle 3D models into a generated KiCad project.

Bundle footprints reference their models as
``${KIPRJMOD}/3dmodels/<library>/<file>`` (written by the parts-library
fetch-3d / add-part tooling), and KiCad resolves ``${KIPRJMOD}`` to the
directory containing the open board. Copying each used bundle's ``3d/``
files into ``<project>/3dmodels/<library>/`` therefore makes the generated
project fully self-contained: kicad-cli renders and STEP exports resolve
the models on the build box, and a downloaded project zip opens in desktop
KiCad with part bodies and zero path setup.

Stock-library footprints (passives etc.) reference ``${KICAD9_3DMODEL_DIR}``
and resolve through the system KiCad install; stock ``.pretty`` dirs have no
sibling ``3d/`` dir, so they skip naturally here.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .parts_lookup import LibraryNotFoundError, resolve_footprint_library_path

_MODEL_EXTS = {".step", ".stp", ".wrl"}


def stage_3d_models(
    project_dir: Path,
    bom,
    *,
    project_root: Path | None = None,
) -> list[Path]:
    """Copy 3D models for every bundle-backed footprint library in ``bom``.

    Returns the staged file paths. Best-effort by design: an unresolvable
    library is skipped silently (the PCB stub surfaces that as a real error
    when it loads the footprint), and a library without models just stages
    nothing. ``project_root`` matches the footprint resolver's notion of the
    project tier (defaults to ``Path.cwd()``, same as the PCB stub).
    """
    staged: list[Path] = []
    if bom is None:
        return staged
    libraries = sorted({
        part.footprint.split(":", 1)[0]
        for part in bom.parts
        if ":" in (part.footprint or "")
    })
    for lib in libraries:
        try:
            pretty = resolve_footprint_library_path(lib, project_root=project_root)
        except LibraryNotFoundError:
            continue
        src = pretty.parent / "3d"
        if not src.is_dir():
            continue
        dest = project_dir / "3dmodels" / lib
        dest.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if f.is_file() and f.suffix.lower() in _MODEL_EXTS:
                target = dest / f.name
                shutil.copy2(f, target)
                staged.append(target)
    return staged


__all__ = ["stage_3d_models"]
