"""Mechanical synthesis: state -> KiCad file set.

Reads a complete ``ConversationState`` and writes the file set described
in ``docs/kicraft_schematic_prompt.md`` (root + leaf schematics,
.kicad_pro, _autoplacer.json). Runs §9.1-§9.6 unconditionally and §9.7
(solve-subcircuits smoke) optionally.

Library reuse: sheets with ``from_library`` set are routed through the
leaf-library installer, which writes the leaf .kicad_sch (with refdes
renumbered against the project), pre-populates
``.experiments/subcircuits/<leaf_key>/round_lib0001_*``, pins it via the
existing pin manager, and merges the leaf's autoplacer fragment into the
in-memory autoplacer dict. The emitter skips ``_emit_leaf`` for those
sheets; it still writes the root sheet block referencing them, using the
same UUID the installer derived the leaf_key from.

This module is intentionally non-LLM. If state validation fails the
function raises ``SynthesisInputError`` BEFORE touching the filesystem.
If §9 validation fails the function raises ``SynthesisValidationError``
AFTER writing files (so the user can inspect what was produced).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .models import ArtifactPaths, ConversationState
from .synthesis.autoplacer import write_autoplacer_json
from .synthesis.emitter import (
    build_sheet_instances,
    emit_schematic,
    ensure_leaf_stems_distinct,
)
from .synthesis.kicad_pcb_stub import write_empty_pcb
from .synthesis.kicad_pro import write_kicad_pro
from .synthesis.models3d import stage_3d_models
from .synthesis.validation import (
    CheckResult,
    SynthesisValidationError,
    collect_validations,
    run_solve_subcircuits_smoke,
)

logger = logging.getLogger(__name__)


class SynthesisInputError(ValueError):
    """The conversation state is missing required slots or self-inconsistent."""


def _require_state(state: ConversationState) -> None:
    missing: list[str] = []
    if state.project_stem is None:
        missing.append("project_stem")
    if state.intent is None:
        missing.append("intent")
    if state.functional_spec is None:
        missing.append("functional_spec")
    if state.architecture is None:
        missing.append("architecture")
    if state.bom is None:
        missing.append("bom")
    if missing:
        raise SynthesisInputError(f"synthesis requires slots: {', '.join(missing)}")

    sheet_names = {s.name for s in state.architecture.sheets}  # type: ignore[union-attr]
    bad = [
        p.ref for p in state.bom.parts if p.sheet not in sheet_names  # type: ignore[union-attr]
    ]
    if bad:
        raise SynthesisInputError(
            f"BOM parts reference sheets not in architecture: {bad}"
        )


def _install_library_sheets(
    state: ConversationState,
    project_dir: Path,
    sheet_instances: list,
) -> tuple[dict, dict[str, dict]]:
    """Install every library-backed sheet. Returns (fragments, library_leaves).

    ``fragments`` is the merged renumbered autoplacer fragment dict
    (in-memory; the caller passes it to ``write_autoplacer_json``).
    ``library_leaves`` is the per-sheet record for the top-level
    ``library_leaves`` map.
    """
    arch = state.architecture
    assert arch is not None

    library_sheets = [s for s in arch.sheets if s.from_library is not None]
    if not library_sheets:
        return {}, {}

    try:
        from kicraft.leaf_library import LeafLibrary, install_leaf
    except ImportError as exc:
        raise SynthesisInputError(
            "kicraft.leaf_library not importable; library-backed sheets "
            "cannot be installed. Either drop the from_library settings "
            "or fix the install."
        ) from exc

    lib = LeafLibrary.from_env()

    library_sheet_names = {s.name for s in library_sheets}
    project_refs: list[str] = [
        p.ref for p in state.bom.parts  # type: ignore[union-attr]
        if p.sheet not in library_sheet_names
    ]

    sheet_uuid_by_name = {si.sheet.name: si.instance_uuid for si in sheet_instances}

    fragments: dict = {}
    library_leaves: dict[str, dict] = {}

    for sheet in library_sheets:
        leaf = lib.find(sheet.from_library) if sheet.from_library else None
        if leaf is None:
            raise SynthesisInputError(
                f"sheet {sheet.name!r} references library leaf "
                f"{sheet.from_library!r} which is not loadable from the "
                f"library at {lib.base_dir}"
            )
        sheet_uuid = sheet_uuid_by_name.get(sheet.name)
        if sheet_uuid is None:
            raise SynthesisInputError(
                f"no pre-allocated UUID for library sheet {sheet.name!r}"
            )
        result = install_leaf(
            leaf,
            project_dir=project_dir,
            sheet_name=sheet.name,
            sheet_stem=sheet.stem,
            sheet_uuid=sheet_uuid,
            instance=sheet.library_instance or 1,
            project_refs=project_refs,
            autoplacer_dict=fragments,
            check_dependencies=True,
        )
        project_refs.extend(result.ref_map.values())
        library_leaves[sheet.name] = result.to_library_leaves_entry()

    return fragments, library_leaves


def run(
    state: ConversationState,
    project_dir: Path,
    *,
    smoke: bool = False,
    smoke_timeout_s: float = 60.0,
) -> tuple[ArtifactPaths, list[CheckResult]]:
    """Synthesize the project. Returns (artifacts, validation results).

    Raises:
        SynthesisInputError: state is incomplete or self-inconsistent.
        SynthesisValidationError: written files fail §9.1-§9.7 checks.
    """
    _require_state(state)
    assert state.project_stem is not None
    assert state.architecture is not None
    assert state.bom is not None

    project_dir = project_dir.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    # Re-synthesis (a resumable-session stage rerun) emits into the same dir; clear
    # the prior generated top-level files first so an orphan leaf from a PREVIOUS
    # architecture cannot linger. A stale `*.kicad_sch` the new root no longer
    # references would surface as a phantom sheet in the web sheet list and leave the
    # hierarchy degenerate (the place/route engine then finds 0 leaf subcircuits and
    # fails). Scoped to this dir's own generated files by extension -- the
    # `.experiments/` and `.kicraft/` trees (subdirectories) are untouched.
    for pat in ("*.kicad_sch", "*.kicad_pcb", "*.kicad_pro", "*.kicad_prl",
                "*_autoplacer.json", "*_erc.rpt"):
        for stale in project_dir.glob(pat):
            if stale.is_file():
                stale.unlink()
    # Staged 3D models follow the BOM, so a re-synthesis with a changed BOM
    # must not leave orphan models behind; stage_3d_models repopulates below.
    shutil.rmtree(project_dir / "3dmodels", ignore_errors=True)

    # Guard the root/leaf filename collision (a leaf stem equal to the project
    # stem would clobber the root, or be clobbered by it) BEFORE building sheet
    # instances or installing library leaves, so the installer, the root sheet
    # pins, and the leaf file all agree on the renamed stem.
    ensure_leaf_stems_distinct(state.project_stem, state.architecture.sheets)

    # Deterministic BOM rule: thin per-LED decoupling to a couple of bulk caps for
    # a LOW-current LED array (loud + recorded in bom.assumptions). Must run BEFORE
    # the schematic / netlist / autoplacer artifacts are emitted from the BOM.
    from kicraft.design.synthesis.array_decap_footprints import (
        downsize_array_decap_footprints,
    )
    from kicraft.design.synthesis.array_decaps import (
        drop_decap_only_arrays,
        isolate_array_sheets,
        isolate_opposite_edge_connectors,
        normalize_array_decaps,
    )
    # A decap-only ArraySpec (per-LED bypass caps mistakenly declared as their
    # own grid) would grid the caps on top of the LED array -> data ties blocked
    # -> doomed route. Drop it first so the caps become array companions.
    drop_decap_only_arrays(state.bom)
    normalize_array_decaps(state.bom)
    # Backstop the prompt's package nudge: size each LED array's surviving decaps
    # to fit beside the LEDs (0603, or 0402 for sub-2.5mm LEDs). Runs after the
    # thinning so only surviving caps are touched.
    downsize_array_decap_footprints(state.bom)
    # Keep array sheets pure: a non-member, non-companion part (a header, an MCU)
    # sharing an array's sheet gets stranded ~60mm from the locked grid by the
    # leaf solver (KC-WXN3SN). Move it to its own sheet -> its own leaf, composed
    # adjacent to the array. Adds a sheet, so re-guard the root-stem collision.
    isolate_array_sheets(state.bom, state.architecture)
    # A sheet with edge-zoned connectors on opposite edges (e.g. J1 bottom +
    # J2/J3 top on the same sheet) can never compose into a single rigid leaf
    # (KC-58KPS3). Move the minority-edge connectors to their own sheet so
    # each leaf has compatible edge constraints.
    isolate_opposite_edge_connectors(state.bom, state.architecture)
    ensure_leaf_stems_distinct(state.project_stem, state.architecture.sheets)

    sheet_instances = build_sheet_instances(state.architecture, state.bom)

    fragments, library_leaves = _install_library_sheets(
        state, project_dir, sheet_instances
    )

    library_sheet_names = {name for name in library_leaves}
    root, leaves = emit_schematic(
        project_dir,
        state.project_stem,
        state.architecture,
        state.bom,
        title=state.intent.goal if state.intent else state.project_stem,
        skip_leaf_sheets=library_sheet_names,
        sheet_instances=sheet_instances,
    )

    if library_sheet_names:
        for sheet in state.architecture.sheets:
            if sheet.name in library_sheet_names:
                lib_path = project_dir / f"{sheet.stem}.kicad_sch"
                if lib_path.exists():
                    leaves.append(lib_path)

    pro = write_kicad_pro(project_dir, state.project_stem, state.architecture)
    ap = write_autoplacer_json(
        project_dir,
        state.project_stem,
        state.architecture,
        state.bom,
        library_fragments=fragments or None,
        library_leaves=library_leaves or None,
        placement=getattr(state, "placement", None),
        form_factor=state.intent.form_factor if state.intent else None,
    )
    write_empty_pcb(project_dir, state.project_stem, state.bom)
    stage_3d_models(project_dir, state.bom)

    # Build the artifact record now — the files exist on disk regardless of
    # whether the §9 checks pass — so a validation failure can still report
    # what was written (with status="failed").
    artifacts = ArtifactPaths(
        project_dir=project_dir,
        project_stem=state.project_stem,
        root_sch=root,
        leaf_schs=leaves,
        kicad_pro=pro,
        autoplacer_json=ap,
    )

    results = collect_validations(project_dir, state.project_stem, bom=state.bom)
    if smoke:
        results.append(
            run_solve_subcircuits_smoke(
                project_dir, state.project_stem, timeout_s=smoke_timeout_s
            )
        )
    failures = [r for r in results if not r.ok]
    if failures:
        raise SynthesisValidationError(
            failures,
            artifacts=artifacts.model_copy(update={"status": "failed"}),
            results=results,
        )
    return artifacts, results
