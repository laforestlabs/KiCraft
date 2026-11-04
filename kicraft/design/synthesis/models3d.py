"""Stage part-bundle 3D models into a generated KiCad project.

Bundle footprints reference their models as
``${KIPRJMOD}/3dmodels/<library>/<file>`` (written by the parts-library
fetch-3d / add-part tooling), and KiCad resolves ``${KIPRJMOD}`` to the
directory containing the open board. Copying each used bundle's models and
``LICENSE.txt`` attribution into ``<project>/3dmodels/<library>/`` makes the
generated project self-contained and carries the model's license with it.

Stock-library footprints (passives etc.) reference ``${KICAD9_3DMODEL_DIR}``
and resolve through the system KiCad install; stock ``.pretty`` dirs have no
sibling ``3d/`` dir, so they skip naturally here.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Callable
from pathlib import Path

from .parts_lookup import LibraryNotFoundError, resolve_footprint_library_path

_MODEL_EXTS = {".step", ".stp", ".wrl"}

# KiCad's AP214 STEP exports name top-level assembly occurrences with the PCB
# reference. Inspect the exported assembly, not just model path existence:
# unresolved or unsupported models need not become STEP occurrences.
_OCCURRENCE = re.compile(
    r"(?m)^\s*#\d+\s*=\s*NEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\("
    r"\s*'(?:[^']|'')*'\s*,\s*'((?:[^']|'')*)'\s*,"
    r"\s*'(?:[^']|'')*'\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*\$\s*\)\s*;"
)


def collect_step_coverage(step_path: Path, refs: list[str]) -> dict:
    """Report top-level reference occurrences in an actual KiCad STEP export.

    Occurrence presence is not proof of exact geometry, placement or render
    coverage. Unknown means this particular STEP structure was not safely
    interpretable; missing refs are then deliberately not inferred.
    """
    unknown = {"status": "unknown", "observed_refs": None, "missing_refs": None}
    try:
        text = Path(step_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return unknown
    if not text.lstrip().startswith("ISO-10303-21;") or not text.rstrip().endswith("END-ISO-10303-21;"):
        return unknown
    occurrences = _OCCURRENCE.findall(text)
    if not occurrences or len(occurrences) != text.count("NEXT_ASSEMBLY_USAGE_OCCURRENCE"):
        return unknown
    parents = {parent for _, parent, _ in occurrences}
    roots = parents - {child for _, _, child in occurrences}
    if len(roots) != 1:
        return unknown
    root = next(iter(roots))
    expected = set(refs)
    seen = {name.replace("''", "'") for name, parent, _ in occurrences if parent == root}
    return {
        "status": "observed",
        "observed_refs": sorted(expected & seen),
        "missing_refs": sorted(expected - seen),
    }


def stage_3d_models(
    project_dir: Path,
    bom,
    *,
    project_root: Path | None = None,
    owned_files: dict[str, str] | None = None,
    on_written: Callable[[Path], None] | None = None,
) -> list[Path]:
    """Copy bundle-backed 3D models and their ``LICENSE.txt`` attribution.

    Returns staged file paths. An unresolvable library is skipped (the PCB stub
    surfaces that as an error), and a library without models stages no models.
    ``project_root`` matches the footprint resolver's project tier. Existing
    targets, including license files, are replaced only when their current
    hash matches a prior synthesis output manifest.
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
            if f.is_file() and (f.suffix.lower() in _MODEL_EXTS or f.name == "LICENSE.txt"):
                target = dest / f.name
                if not target.resolve().is_relative_to(project_dir.resolve()):
                    raise ValueError(f"3D model target escapes project directory: {target}")
                if target.exists() or target.is_symlink():
                    relative = str(target.relative_to(project_dir))
                    expected = (owned_files or {}).get(relative)
                    if (
                        target.is_symlink() or not target.is_file() or expected is None
                        or hashlib.sha256(target.read_bytes()).hexdigest() != expected
                    ):
                        raise ValueError(f"refusing to overwrite untracked or modified 3D model: {target}")
                shutil.copy2(f, target)
                if on_written is not None:
                    on_written(target)
                staged.append(target)
    return staged


__all__ = ["collect_step_coverage", "stage_3d_models"]
