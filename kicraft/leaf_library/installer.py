"""Install a library leaf into an in-progress synthesis.

Called once per library-backed sheet by the synthesis emitter. The
installer is the only place that knows how to:

1. Verify the leaf's content_hash at install time (Issue 8).
2. Verify KiCad library deps are present on the target system.
3. Compute the ref_map for this instance.
4. Write the renumbered schematic to the project root.
5. Pre-populate the ``.experiments/subcircuits/<leaf_key>/round_lib0001_*``
   triad so the parent composer treats this sheet as pre-solved.
6. Pin the import via ``pins.pin_leaf``.
7. Merge the renumbered autoplacer fragment into the project's
   in-memory autoplacer dict.

Returns an :class:`InstalledLeaf` record the synthesis stage uses to
populate ``library_leaves[sheet_name]``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from ..autoplacer.brain.pins import pin_leaf
from ..autoplacer.brain.subcircuit_artifacts import derive_leaf_key
from ..design.synthesis.emitter import assert_schematic_parses
from .loader import LoadedLeaf
from .manifest import (
    CANONICAL_TRIAD,
    Manifest,
    compute_content_hash,
)
from .renumber import apply_ref_map, renumber_leaf
from .sexpr_edit import renumber_pcb_text, renumber_schematic_text

log = logging.getLogger(__name__)


LIBRARY_IMPORT_SNAPSHOT_ID = "lib0001"


@dataclass(frozen=True, slots=True)
class InstalledLeaf:
    """The record the synthesis stage writes into ``library_leaves``."""

    sheet_name: str
    sheet_stem: str
    source: str  # e.g. "usb-c-lipo-charger@1.2.0"
    source_hash: str  # leaf manifest content_hash
    instance: int
    leaf_key: str
    ref_map: dict[str, str] = field(default_factory=dict)

    def to_library_leaves_entry(self) -> dict[str, object]:
        """Render the dict that lands under ``library_leaves[sheet_name]``."""
        return {
            "source": self.source,
            "source_hash": self.source_hash,
            "instance": self.instance,
            "ref_map": dict(self.ref_map),
        }


class DependencyError(RuntimeError):
    """Raised when a leaf's KiCad library deps aren't on the target system."""


def verify_content_hash_at_install(leaf: LoadedLeaf) -> None:
    """Re-check the leaf's content_hash. Loader-side cache could be stale."""
    actual = compute_content_hash(leaf.dir)
    if actual != leaf.manifest.content_hash:
        raise RuntimeError(
            f"leaf {leaf.slug} content_hash mismatch at install time: "
            f"manifest says {leaf.manifest.content_hash}, recomputed {actual}"
        )


def _kicad_search_paths() -> list[Path]:
    """Best-effort list of directories where stock KiCad libs may live.

    Honors ``$KICAD9_SYMBOL_DIR`` / ``$KICAD9_FOOTPRINT_DIR`` if set;
    otherwise checks the standard system paths on Linux.
    """
    env_paths: list[Path] = []
    for var in ("KICAD9_SYMBOL_DIR", "KICAD9_FOOTPRINT_DIR"):
        val = os.environ.get(var)
        if val:
            env_paths.append(Path(val))
    system_paths = [
        Path("/usr/share/kicad/symbols"),
        Path("/usr/share/kicad/footprints"),
        Path("/usr/local/share/kicad/symbols"),
        Path("/usr/local/share/kicad/footprints"),
    ]
    return [p for p in env_paths + system_paths if p.exists()]


def verify_dependencies(manifest: Manifest, *, strict: bool = True) -> list[str]:
    """Return the list of missing KiCad library names.

    Empty list means all deps present. When ``strict`` is False, the
    check returns the list without raising even on a totally missing
    KiCad install (useful for tests).
    """
    search_paths = _kicad_search_paths()
    if not search_paths:
        if strict:
            log.warning(
                "no KiCad library paths found on system; skipping dep check"
            )
        return []

    missing: list[str] = []
    sym_dirs = [p for p in search_paths if p.name == "symbols"]
    fp_dirs = [p for p in search_paths if p.name == "footprints"]

    for lib in manifest.dependencies.kicad_symbol_libs:
        if not any((d / f"{lib}.kicad_sym").exists() for d in sym_dirs):
            missing.append(f"symbol:{lib}")

    for lib in manifest.dependencies.kicad_footprint_libs:
        if not any((d / f"{lib}.pretty").is_dir() for d in fp_dirs):
            missing.append(f"footprint:{lib}")

    return missing


def install_leaf(
    leaf: LoadedLeaf,
    *,
    project_dir: Path,
    sheet_name: str,
    sheet_stem: str,
    sheet_uuid: str,
    instance: int,
    project_refs: list[str],
    autoplacer_dict: dict[str, object],
    check_dependencies: bool = True,
) -> InstalledLeaf:
    """Install ``leaf`` into ``project_dir`` as a library-backed sheet.

    ``sheet_uuid`` must match the UUID the emitter writes for this
    sheet's ``(sheet ... (uuid X))`` block in the root schematic, so the
    derived leaf_key agrees with what ``solve_subcircuits`` would later
    compute when discovering the sheet.

    ``autoplacer_dict`` is mutated in place: the leaf's renumbered
    fragment is merged additively (see :func:`_merge_fragment`).
    """
    project_dir = Path(project_dir)
    verify_content_hash_at_install(leaf)

    if check_dependencies:
        missing = verify_dependencies(leaf.manifest)
        if missing:
            raise DependencyError(
                f"leaf {leaf.slug} requires libraries not on target system: "
                f"{', '.join(missing)}"
            )

    # 1. Compute ref_map
    ref_map = renumber_leaf(
        leaf_refs=list(leaf.manifest.refs),
        project_refs=list(project_refs),
    )

    # 2. Derive leaf_key the same way solve_subcircuits will when it
    # later discovers this sheet from the synthesized schematic.
    sheet_file = f"{sheet_stem}.kicad_sch"
    instance_path = f"/{sheet_uuid}"
    leaf_key = derive_leaf_key(
        sheet_name=sheet_name,
        sheet_file=sheet_file,
        instance_path=instance_path,
        parent_instance_path="/",
    )

    # 3. Write renumbered schematic to <project>/<sheet_stem>.kicad_sch
    sch_text = (leaf.dir / "schematic.kicad_sch").read_text(encoding="utf-8")
    new_sch, sch_counts = renumber_schematic_text(sch_text, ref_map)
    log.debug("sheet %s sch rewrites: %s", sheet_name, sch_counts)
    sch_out = project_dir / sheet_file
    sch_out.parent.mkdir(parents=True, exist_ok=True)
    assert_schematic_parses(new_sch, sch_out)
    sch_out.write_text(new_sch, encoding="utf-8")

    # 4. Write renumbered triad to .experiments/subcircuits/<leaf_key>/
    artifact_dir = project_dir / ".experiments" / "subcircuits" / leaf_key
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pcb_text = (leaf.dir / "leaf_routed.kicad_pcb").read_text(encoding="utf-8")
    new_pcb, pcb_counts = renumber_pcb_text(pcb_text, ref_map)
    log.debug("sheet %s pcb rewrites: %s", sheet_name, pcb_counts)
    (artifact_dir / f"round_{LIBRARY_IMPORT_SNAPSHOT_ID}_leaf_routed.kicad_pcb").write_text(
        new_pcb, encoding="utf-8"
    )

    # metadata.json gets a renumbered copy. Some fields embed refs
    # (e.g. component lists); pass it through apply_ref_map for safety.
    metadata = json.loads(
        (leaf.dir / "metadata.json").read_text(encoding="utf-8")
    )
    renumbered_metadata = apply_ref_map(metadata, ref_map)
    # Stamp the new identity so the artifact's metadata matches its host.
    if isinstance(renumbered_metadata, dict):
        sid = renumbered_metadata.get("subcircuit_id")
        if isinstance(sid, dict):
            sid["sheet_name"] = sheet_name
            sid["sheet_file"] = sheet_file
            sid["instance_path"] = instance_path
            sid["parent_instance_path"] = "/"
        renumbered_metadata["sheet_name"] = sheet_name
        renumbered_metadata["sheet_file"] = sheet_file
    (artifact_dir / f"round_{LIBRARY_IMPORT_SNAPSHOT_ID}_metadata.json").write_text(
        json.dumps(renumbered_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    solved = json.loads(
        (leaf.dir / "solved_layout.json").read_text(encoding="utf-8")
    )
    renumbered_solved = apply_ref_map(solved, ref_map)
    (artifact_dir / f"round_{LIBRARY_IMPORT_SNAPSHOT_ID}_solved_layout.json").write_text(
        json.dumps(renumbered_solved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Copy renders (no renumber needed -- they're just PNGs).
    src_renders = leaf.dir / "renders"
    if src_renders.is_dir():
        dst_renders = artifact_dir / "renders"
        dst_renders.mkdir(exist_ok=True)
        for entry in src_renders.iterdir():
            if entry.is_file():
                shutil.copy(entry, dst_renders / entry.name)

    # 5. Pin the import
    experiments_dir = project_dir / ".experiments"
    pin_leaf(
        experiments_dir,
        leaf_key,
        LIBRARY_IMPORT_SNAPSHOT_ID,
        source=f"library:{leaf.slug}",
    )

    # 6. Merge renumbered autoplacer fragment additively.
    fragment_path = leaf.dir / "autoplacer_fragment.json"
    if fragment_path.exists():
        fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
        renumbered_fragment = apply_ref_map(fragment, ref_map)
        if isinstance(renumbered_fragment, dict):
            _merge_fragment(autoplacer_dict, renumbered_fragment)

    return InstalledLeaf(
        sheet_name=sheet_name,
        sheet_stem=sheet_stem,
        source=leaf.slug,
        source_hash=leaf.manifest.content_hash,
        instance=instance,
        leaf_key=leaf_key,
        ref_map=ref_map,
    )


def _merge_fragment(dst: dict[str, object], src: dict[str, object]) -> None:
    """Additive merge of a renumbered fragment into the project autoplacer.

    Semantics (Issue 9):
    - ``ic_groups``, ``group_labels``, ``component_zones``: union of
      keys. Key collisions raise -- they imply a renumber bug because
      every renumbered key is globally unique by construction.
    - ``thermal_refs``, ``signal_flow_order``: concatenate, dedupe.
    - All other keys: ignored (the project owns them).
    """
    for key in ("ic_groups", "group_labels", "component_zones"):
        leaf_val = src.get(key)
        if not isinstance(leaf_val, dict):
            continue
        proj_val = dst.setdefault(key, {})
        if not isinstance(proj_val, dict):
            raise TypeError(
                f"project autoplacer.{key} is not a dict; cannot merge"
            )
        for k, v in leaf_val.items():
            if k in proj_val:
                raise ValueError(
                    f"library fragment collision: {key}[{k!r}] already in "
                    f"project autoplacer.json (renumber map should have "
                    f"prevented this)"
                )
            proj_val[k] = v

    for key in ("thermal_refs", "signal_flow_order"):
        leaf_val = src.get(key)
        if not isinstance(leaf_val, list):
            continue
        proj_val = dst.setdefault(key, [])
        if not isinstance(proj_val, list):
            raise TypeError(
                f"project autoplacer.{key} is not a list; cannot merge"
            )
        for item in leaf_val:
            if item not in proj_val:
                proj_val.append(item)


__all__ = [
    "CANONICAL_TRIAD",
    "DependencyError",
    "InstalledLeaf",
    "LIBRARY_IMPORT_SNAPSHOT_ID",
    "install_leaf",
    "verify_content_hash_at_install",
    "verify_dependencies",
]
