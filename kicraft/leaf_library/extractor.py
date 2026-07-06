"""Build a library-leaf directory from a source project's pinned round.

``extract_leaf`` is the promotion entry point (the desktop-GUI Promote
wizard that wrapped it was removed 2026-06-22): the caller chooses a
(source project, sheet name, round) tuple and fills in the manifest
metadata. It writes a new directory under the library root, atomically.

Inputs the caller must provide because they're not derivable from the
on-disk leaf directly:

- ``name`` / ``version`` / ``description`` / ``tags`` / ``watch_out_for``
- ``bom_rows`` (sliced from the project's BOM; ref/value/footprint/symbol/...)
- ``autoplacer_fragment`` (leaf-scoped slice of the source autoplacer JSON)

Inputs the extractor derives from disk:

- The canonical triad (``leaf_routed.kicad_pcb`` / ``metadata.json`` /
  ``solved_layout.json``) from the chosen round snapshot.
- The leaf schematic (``<sheet_stem>.kicad_sch``) and its hierarchical
  labels (parsed to populate ``manifest.interface``).
- Refdes list, parsed from the schematic and PCB.
- KiCad symbol / footprint library deps, parsed from BOM rows.
- Renders (3 views + thumbnail), produced via ``kicraft.render``.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..autoplacer.brain.pins import _round_snapshot_files
from .manifest import (
    Dependencies,
    HierarchicalLabel,
    Interface,
    Manifest,
    PinDirection,
    Provenance,
    compute_content_hash,
    dump_manifest,
)

log = logging.getLogger(__name__)


_HIER_LABEL_RE = re.compile(
    r'\(hierarchical_label\s+"([A-Z][A-Z0-9_]*)"\s+\(shape\s+(\w+)\)',
    re.MULTILINE,
)

_SYMBOL_REF_PROP_RE = re.compile(
    r'\(property\s+"Reference"\s+"([A-Z]+[0-9]+)"'
)

_FOOTPRINT_REF_PROP_RE = re.compile(
    r'\(property\s+"Reference"\s+"([A-Z]+[0-9]+)"'
)

# The KiCad "shape" attribute maps directly to PinDirection.
_SHAPE_TO_DIRECTION: dict[str, PinDirection] = {
    "input": "input",
    "output": "output",
    "bidirectional": "bidirectional",
    "passive": "passive",
    "tri_state": "bidirectional",
}


@dataclass
class PromoteRequest:
    """All inputs needed to build a leaf directory.

    The wizard fills this in step by step, then hands it to
    :func:`extract_leaf`.
    """

    source_project_dir: Path
    source_project_stem: str
    source_sheet_name: str
    source_sheet_stem: str
    source_leaf_key: str
    source_round: int

    name: str
    version: str
    description: str
    tags: list[str] = field(default_factory=list)
    watch_out_for: str | None = None
    kicad_version: str = ""
    kicad_version_min: str = "9.0.0"

    # Leaf-scoped slice of the project's BOM rows. Each row is a dict
    # with at least: ref, value, symbol, footprint. Plus optional mpn,
    # datasheet, sourcing_note.
    bom_rows: list[dict[str, str]] = field(default_factory=list)

    # Leaf-scoped slice of the project autoplacer JSON (refs that belong
    # to this sheet, plus any leaf-only config keys). Renumbered at
    # install time.
    autoplacer_fragment: dict = field(default_factory=dict)


def _parse_hierarchical_labels(sch_text: str) -> list[HierarchicalLabel]:
    labels: list[HierarchicalLabel] = []
    for m in _HIER_LABEL_RE.finditer(sch_text):
        name = m.group(1)
        shape = m.group(2)
        direction = _SHAPE_TO_DIRECTION.get(shape)
        if direction is None:
            log.warning(
                "hierarchical label %s has unknown shape %r; defaulting "
                "to 'passive' in the manifest", name, shape
            )
            direction = "passive"
        labels.append(HierarchicalLabel(name=name, direction=direction))
    # Dedupe while preserving first-seen order.
    seen: set[tuple[str, str]] = set()
    out: list[HierarchicalLabel] = []
    for lbl in labels:
        key = (lbl.name, lbl.direction)
        if key in seen:
            continue
        seen.add(key)
        out.append(lbl)
    return out


def _parse_refs(text: str, pattern: re.Pattern[str]) -> set[str]:
    return {m.group(1) for m in pattern.finditer(text)}


def _library_names_from_bom(bom_rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    """Return (symbol_libs, footprint_libs) sorted/deduplicated.

    Each BOM row has ``symbol`` and ``footprint`` in ``Library:Name`` form.
    """
    sym: set[str] = set()
    fp: set[str] = set()
    for row in bom_rows:
        s = row.get("symbol", "")
        if ":" in s:
            sym.add(s.split(":", 1)[0])
        f = row.get("footprint", "")
        if ":" in f:
            fp.add(f.split(":", 1)[0])
    return sorted(sym), sorted(fp)


def _render_views(pcb_path: Path, renders_dir: Path) -> None:
    """Render front_all / back_copper / copper_both + thumbnail PNGs.

    Uses ``kicraft.render.pcb_renderer.render_views`` with the standard
    view definitions. Imports lazy so the extractor can be imported in
    contexts without KiCad available (e.g. unit tests that don't render).
    """
    from kicraft.render.pcb_renderer import render_views

    renders_dir.mkdir(parents=True, exist_ok=True)
    render_views(
        pcb_path,
        renders_dir,
        views=["front_all", "back_copper", "copper_both"],
        name_template="{view}.png",
    )

    # Thumbnail: 256x256 PNG downscaled from copper_both.
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        log.warning("Pillow not installed; skipping thumbnail.png generation")
        return

    src = renders_dir / "copper_both.png"
    if not src.exists():
        log.warning("copper_both.png missing; skipping thumbnail")
        return
    with Image.open(src) as img:
        img.thumbnail((256, 256), Image.LANCZOS)
        # PIL's thumbnail keeps aspect ratio; pad/center for a square.
        bg = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        x = (256 - img.width) // 2
        y = (256 - img.height) // 2
        bg.paste(img, (x, y))
        bg.save(renders_dir / "thumbnail.png", "PNG")


def extract_leaf(
    req: PromoteRequest,
    target_library_dir: Path,
    *,
    render: bool = True,
) -> Path:
    """Build a leaf directory under ``target_library_dir``.

    Writes atomically: target_library_dir/.<name>.tmp/ is populated in
    full, then ``os.replace``'d into ``target_library_dir/<name>``.

    Refuses to overwrite an existing leaf directory with the same name
    and version; the caller (GUI) is responsible for bumping the patch
    version when re-promoting.

    Returns the final leaf directory path.
    """
    source_dir = Path(req.source_project_dir)
    leaf_artifact_dir = (
        source_dir / ".experiments" / "subcircuits" / req.source_leaf_key
    )
    snapshot = _round_snapshot_files(leaf_artifact_dir, req.source_round)
    if not snapshot:
        raise FileNotFoundError(
            f"round {req.source_round} for leaf {req.source_leaf_key} has "
            f"no complete snapshot in {leaf_artifact_dir}"
        )

    # Sanity: source schematic must exist at <project>/<sheet_stem>.kicad_sch
    sch_path = source_dir / f"{req.source_sheet_stem}.kicad_sch"
    if not sch_path.exists():
        raise FileNotFoundError(
            f"leaf schematic not found: {sch_path}"
        )

    target_dir = Path(target_library_dir) / req.name
    if target_dir.exists():
        # Match-version refusal: caller should bump version.
        from .manifest import load_manifest as _load
        try:
            existing = _load(target_dir)
        except Exception:
            existing = None
        if existing is not None and existing.version == req.version:
            raise FileExistsError(
                f"leaf {req.name}@{req.version} already exists at {target_dir} "
                f"-- bump the version and retry"
            )

    target_library_dir = Path(target_library_dir)
    target_library_dir.mkdir(parents=True, exist_ok=True)

    # Stage everything into a sibling tmp dir, then atomic-replace into
    # the final name.
    with tempfile.TemporaryDirectory(
        dir=target_library_dir, prefix=f".{req.name}.tmp."
    ) as tmp:
        staging = Path(tmp) / req.name
        staging.mkdir()

        # Canonical triad
        for canonical_name, src_path in snapshot.items():
            shutil.copy(src_path, staging / canonical_name)

        # Schematic
        shutil.copy(sch_path, staging / "schematic.kicad_sch")

        # BOM
        bom_path = staging / "bom.csv"
        _write_bom_csv(req.bom_rows, bom_path)

        # Autoplacer fragment
        (staging / "autoplacer_fragment.json").write_text(
            json.dumps(req.autoplacer_fragment, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

        # Renders
        if render:
            try:
                _render_views(
                    staging / "leaf_routed.kicad_pcb",
                    staging / "renders",
                )
            except Exception as exc:
                log.warning(
                    "render generation failed (continuing without renders): %s",
                    exc,
                )

        # Parse the schematic for refs + interface
        sch_text = (staging / "schematic.kicad_sch").read_text(encoding="utf-8")
        pcb_text = (staging / "leaf_routed.kicad_pcb").read_text(encoding="utf-8")
        sch_refs = _parse_refs(sch_text, _SYMBOL_REF_PROP_RE)
        pcb_refs = _parse_refs(pcb_text, _FOOTPRINT_REF_PROP_RE)
        all_refs = sorted(sch_refs | pcb_refs)

        interface_labels = _parse_hierarchical_labels(sch_text)
        sym_libs, fp_libs = _library_names_from_bom(req.bom_rows)

        manifest = Manifest(
            schema_version="1",
            name=req.name,
            version=req.version,
            content_hash="sha256:" + ("0" * 64),  # placeholder, replaced below
            description=req.description,
            tags=list(req.tags),
            watch_out_for=req.watch_out_for,
            interface=Interface(hierarchical_labels=interface_labels),
            refs=all_refs,
            dependencies=Dependencies(
                kicad_symbol_libs=sym_libs,
                kicad_footprint_libs=fp_libs,
                kicad_version_min=req.kicad_version_min,
            ),
            provenance=Provenance(
                source_project_stem=req.source_project_stem,
                source_sheet_name=req.source_sheet_name,
                source_experiment_round=req.source_round,
                promoted_at=datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                kicad_version=req.kicad_version,
            ),
        )

        # Compute hash with the placeholder in place, then rewrite the
        # manifest with the real hash. content_hash excludes manifest.json
        # itself, so this is stable across the two-write sequence.
        dump_manifest(manifest, staging)
        real_hash = compute_content_hash(staging)
        manifest = manifest.model_copy(update={"content_hash": real_hash})
        dump_manifest(manifest, staging)

        # Final atomic move into place. If target_dir exists (different
        # version), shutil.rmtree it first -- spec says replace-after-confirm.
        if target_dir.exists():
            shutil.rmtree(target_dir)
        # os.replace can't move a non-empty dir across filesystems but
        # within the same fs it works; we're inside target_library_dir
        # so this is safe.
        import os as _os
        _os.replace(staging, target_dir)

    log.info("promoted leaf %s@%s to %s", req.name, req.version, target_dir)
    return target_dir


def _write_bom_csv(rows: list[dict[str, str]], path: Path) -> None:
    """Write BOM rows to ``path`` with a stable column order.

    The canonical column set matches the in-repo BomPart fields so the
    install-side re-import is a 1:1 round-trip.
    """
    fields = [
        "ref",
        "value",
        "symbol",
        "footprint",
        "sheet",
        "mpn",
        "datasheet",
        "sourcing_note",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


__all__ = [
    "PromoteRequest",
    "extract_leaf",
]
