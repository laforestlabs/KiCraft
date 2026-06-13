"""Parts library: curated KiCad symbol+footprint bundles, reusable across projects.

The library is searched in four tiers (highest precedence first):

1. project-local: ``<project_root>/.kicraft/parts/<name>/``
2. vendored:      ``<kicraft_install>/parts_library/<name>/``
3. user-wide:     ``~/.kicraft/parts/<name>/`` (the BOM stage's fetch cache)
4. extras:        directories in ``$KICRAFT_EXTRA_PARTS_DIRS`` (colon-separated)

A part bundle contains a manifest, a ``<name>.kicad_sym`` file whose
KiCad library prefix equals ``<name>``, and a ``<name>.pretty/`` directory
holding the footprint. See ``manifest.py`` for the full on-disk schema.

This package only owns the loader; symbol/footprint extraction for KiCad
synthesis lives in ``kicraft.design.synthesis.parts_lookup``.
"""

from .loader import (
    BrokenPart,
    ENV_VAR,
    LoadedPart,
    PARTS_SUBDIR,
    Tier,
    extra_parts_dirs,
    find_part,
    home_parts_dir,
    load_all_with_overrides,
    project_parts_dir,
    resolve_tier_dirs,
    vendored_parts_dir,
)
from .manifest import (
    PART_NAME_RE,
    SEMVER_RE,
    SOURCING_KEY_RE,
    Maturity,
    PartManifest,
    Provenance,
    compute_content_hash,
    dump_manifest,
    footprint_dir_path,
    footprint_file_path,
    load_manifest,
    manifest_path,
    required_files_present,
    symbol_file_path,
    verify_content_hash,
)

__all__ = [
    "BrokenPart",
    "ENV_VAR",
    "LoadedPart",
    "Maturity",
    "PARTS_SUBDIR",
    "PART_NAME_RE",
    "PartManifest",
    "Provenance",
    "SEMVER_RE",
    "SOURCING_KEY_RE",
    "Tier",
    "compute_content_hash",
    "dump_manifest",
    "extra_parts_dirs",
    "find_part",
    "footprint_dir_path",
    "footprint_file_path",
    "home_parts_dir",
    "load_all_with_overrides",
    "load_manifest",
    "manifest_path",
    "project_parts_dir",
    "required_files_present",
    "resolve_tier_dirs",
    "symbol_file_path",
    "vendored_parts_dir",
    "verify_content_hash",
]
