"""Guard: every shipped vendored bundle's 3D model references must resolve.

Regression context: easyeda2kicad exported every bundled footprint with a
broken bare ``/NAME.wrl`` model path (no 3D file was ever downloaded), so 3D
renders of generated boards showed bare PCBs. After the fetch-3d backfill,
each bundle either ships its model in ``3d/`` and references it as
``${KIPRJMOD}/3dmodels/<name>/<file>`` (synthesis copies it project-local) or
points at a stock ``${KICAD9_3DMODEL_DIR}`` model resolved by the system
KiCad install. This test keeps the real vendored library in that state, so a
re-fetched or hand-edited footprint that regresses to an unresolvable path
fails CI instead of silently rendering an empty board.

Like test_vendored_bundles_load.py, it bypasses the autouse
``mask_vendored_tier`` fixture by deriving the real vendored directory from
the loader module's location.
"""
from __future__ import annotations

from pathlib import Path

import kicraft.parts_library.loader as loader_mod
from kicraft.design.cli_app import _check_3d_model_paths, _model_stanza_paths
from kicraft.parts_library import footprint_file_path, load_manifest
from kicraft.parts_library.loader import _iter_part_dirs


def _real_vendored_dir() -> Path:
    return Path(loader_mod.__file__).resolve().parent


def _vendored_footprints():
    for part_dir in _iter_part_dirs(_real_vendored_dir()):
        manifest = load_manifest(part_dir)
        fp = footprint_file_path(part_dir, manifest.footprint_name)
        yield part_dir, manifest, fp.read_text()


def test_vendored_3d_paths_all_resolve():
    problems: list[str] = []
    checked = 0
    for part_dir, manifest, fp_text in _vendored_footprints():
        problems += [
            f"{part_dir.name}: {p}"
            for p in _check_3d_model_paths(part_dir, manifest.name, fp_text)
        ]
        checked += 1
    assert checked, "expected at least one vendored bundle"
    assert not problems, (
        "vendored footprints with unresolvable 3D model paths (re-run "
        "`kicraft fetch-3d <dir>` or point the stanza at a stock model):\n"
        + "\n".join(f"  {p}" for p in problems)
    )


def test_vendored_bundles_all_reference_a_model():
    """A footprint without any (model ...) stanza renders as a bare board
    spot; every vendored bundle must reference some model (bundle-local
    or stock)."""
    missing = [
        part_dir.name
        for part_dir, _manifest, fp_text in _vendored_footprints()
        if not _model_stanza_paths(fp_text)
    ]
    assert not missing, (
        f"vendored bundles without any (model ...) stanza: {missing}"
    )
