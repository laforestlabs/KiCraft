"""Leaf-library: promote and reuse pinned KiCad leaves across projects.

Public API is re-exported here for easy import:

    from kicraft.leaf_library import Manifest, load_manifest, LeafLibrary, ...

See ``docs/leaf_library_spec.md`` for the design and on-disk format.
"""

from .extractor import PromoteRequest, extract_leaf
from .installer import (
    DependencyError,
    InstalledLeaf,
    LIBRARY_IMPORT_SNAPSHOT_ID,
    install_leaf,
    verify_dependencies,
)
from .loader import (
    BrokenLeaf,
    DEFAULT_LIBRARY_DIR,
    LIBRARY_ENV_VAR,
    LeafLibrary,
    LoadedLeaf,
    resolve_library_dir,
)
from .manifest import (
    CANONICAL_TRIAD,
    Dependencies,
    HierarchicalLabel,
    Interface,
    LEAF_REQUIRED_FILES,
    Manifest,
    PinDirection,
    Provenance,
    compute_content_hash,
    dump_manifest,
    load_manifest,
    manifest_path,
    required_files_present,
    verify_content_hash,
)
from .renumber import apply_ref_map, parse_ref, renumber_leaf
from .sexpr_edit import renumber_pcb_text, renumber_schematic_text

__all__ = [
    "BrokenLeaf",
    "CANONICAL_TRIAD",
    "DEFAULT_LIBRARY_DIR",
    "Dependencies",
    "DependencyError",
    "HierarchicalLabel",
    "Interface",
    "LEAF_REQUIRED_FILES",
    "LIBRARY_ENV_VAR",
    "LIBRARY_IMPORT_SNAPSHOT_ID",
    "LeafLibrary",
    "LoadedLeaf",
    "Manifest",
    "InstalledLeaf",
    "PinDirection",
    "PromoteRequest",
    "Provenance",
    "apply_ref_map",
    "compute_content_hash",
    "dump_manifest",
    "extract_leaf",
    "install_leaf",
    "load_manifest",
    "manifest_path",
    "parse_ref",
    "renumber_leaf",
    "renumber_pcb_text",
    "renumber_schematic_text",
    "required_files_present",
    "resolve_library_dir",
    "verify_content_hash",
    "verify_dependencies",
]
