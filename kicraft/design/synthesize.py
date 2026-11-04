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

import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from pathlib import Path

from kicraft.fsutil import atomic_write_text

from .models import ArtifactPaths, ConversationState, board_copper_layers
from .synthesis.autoplacer import write_autoplacer_json
from .synthesis.emitter import (
    build_sheet_instances,
    emit_schematic,
    ensure_leaf_stems_distinct,
)
from .synthesis.kicad_pcb_stub import DEFAULT_COPPER_LAYERS, PadBindingError, write_empty_pcb
from .synthesis.kicad_pro import write_kicad_pro
from .synthesis.models3d import stage_3d_models
from .synthesis.validation import (
    CheckResult,
    SynthesisValidationError,
    bom_parts_on_unknown_sheets,
    collect_validations,
    check_typed_led_current_paths,
    check_reviewed_analog_input_ranges,
    check_reviewed_device_support_networks,
    check_reviewed_i2c_address_assignments,
    check_reviewed_input_operating_ranges,
    check_reviewed_power_transfer,
    check_typed_passive_crossover_values,
    check_reviewed_constant_current_led_feedback,
    check_requirement_physical_realization,
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

    bad = [ref for ref, _ in bom_parts_on_unknown_sheets(state.architecture, state.bom)]
    if bad:
        raise SynthesisInputError(f"BOM parts reference sheets not in architecture: {bad}")


def _install_library_sheets(
    state: ConversationState,
    project_dir: Path,
    sheet_instances: list,
    on_written: Callable[[Path], None] | None = None,
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
        p.ref
        for p in state.bom.parts  # type: ignore[union-attr]
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
            raise SynthesisInputError(f"no pre-allocated UUID for library sheet {sheet.name!r}")
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
            on_schematic_written=on_written,
        )
        project_refs.extend(result.ref_map.values())
        library_leaves[sheet.name] = result.to_library_leaves_entry()

    return fragments, library_leaves


def _requested_copper_layers(state: ConversationState) -> int:
    """The copper-layer count this design's own stack-up statement asks for.

    Raises ``SynthesisInputError`` for a stated count no printed stack-up can carry: rounding
    it silently would ship a board the brief did not ask for.
    """
    try:
        requested = board_copper_layers(state.intent.obligations if state.intent else None)
    except ValueError as exc:
        raise SynthesisInputError(str(exc)) from exc
    if requested is None:
        return DEFAULT_COPPER_LAYERS
    return requested

def _read_generated_files(project_dir: Path) -> dict[str, str]:
    manifest = project_dir / ".kicraft" / "synthesis-generated.json"
    if not manifest.exists():
        return {}
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("files"), dict):
        raise SynthesisInputError(f"invalid synthesis output manifest: {manifest}")
    return data["files"]


def _check_output_destinations(
    project_dir: Path, paths: list[Path], owned: dict[str, str]
) -> None:
    """Refuse to overwrite files not proven to be unchanged synthesis outputs."""
    for path in paths:
        if not path.resolve().is_relative_to(project_dir):
            raise SynthesisInputError(f"synthesis output escapes project directory: {path}")
        if not path.exists() and not path.is_symlink():
            continue
        expected = owned.get(str(path.relative_to(project_dir)))
        if (
            path.is_symlink() or not path.is_file() or expected is None
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise SynthesisInputError(
                f"refusing to overwrite untracked or modified synthesis output: {path}"
            )


def _record_completed_outputs(
    project_dir: Path, paths: list[Path], *, attempt_id: str
) -> None:
    """Record completed writes without deleting any earlier generated output."""
    files = _read_generated_files(project_dir)
    for path in paths:
        if path.is_file() and not path.is_symlink():
            files[str(path.relative_to(project_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = project_dir / ".kicraft" / "synthesis-generated.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(manifest, json.dumps(
        {"version": 1, "status": "failed", "attempt_id": attempt_id, "files": files},
        indent=2, sort_keys=True,
    ) + "\n")


def _reconcile_generated_files(
    project_dir: Path, generated: list[Path], *, attempt_id: str | None = None
) -> None:
    """Remove only unchanged obsolete files recorded by an earlier synthesis."""
    manifest = project_dir / ".kicraft" / "synthesis-generated.json"
    previous = _read_generated_files(project_dir)
    current = {
        str(path.relative_to(project_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in generated
        if path.is_file()
    }
    for name, digest in previous.items():
        if name in current or not isinstance(name, str) or not isinstance(digest, str):
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            continue
        path = project_dir / relative
        if (
            path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(project_dir)
            and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        ):
            path.unlink()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(manifest, json.dumps(
        {"version": 1, "status": "failed", "attempt_id": attempt_id, "files": current},
        indent=2, sort_keys=True,
    ) + "\n")




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

    # Resolved before anything is written, so an impossible stack-up fails the run cleanly
    # instead of leaving half-emitted files behind.
    copper_layers = _requested_copper_layers(state)

    project_dir = project_dir.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    # Existing files are not assumed to be generated merely from their suffix.
    # Obsolete files are reconciled against our output manifest after emission.

    # Guard the root/leaf filename collision (a leaf stem equal to the project
    # stem would clobber the root, or be clobbered by it) BEFORE building sheet
    # instances or installing library leaves, so the installer, the root sheet
    # pins, and the leaf file all agree on the renamed stem.
    ensure_leaf_stems_distinct(state.project_stem, state.architecture.sheets)

    # Deterministic BOM rule: thin per-LED decoupling to a couple of bulk caps for
    # a LOW-current LED array (loud + recorded in bom.assumptions). Must run BEFORE
    # the schematic / netlist / autoplacer artifacts are emitted from the BOM.
    from kicraft.design.synthesis.array_decaps import (
        drop_decap_only_arrays,
        isolate_array_sheets,
        isolate_opposite_edge_connectors,
        normalize_array_decaps,
    )
    from kicraft.design.synthesis.sheet_partition import (
        split_dense_sheets_enabled,
        split_dense_soc_sheets,
    )

    # A decap-only ArraySpec (per-LED bypass caps mistakenly declared as their
    # own grid) would grid the caps on top of the LED array -> data ties blocked
    # -> doomed route. Drop it first so the caps become array companions.
    drop_decap_only_arrays(state.bom)
    normalize_array_decaps(state.bom)
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
    # A whole MCU subsystem on one sheet is the #1 leaf place/route failure class
    # by breadth. Move only DETACHABLE subfunctions (debug header, button,
    # battery, RF chain) off it -- never a decoupling cap or a crystal, which
    # must sit at the SoC's pins. OFF by default: it restructures every dense
    # design's sheets, so it needs its own self-eval batch to be trusted (see
    # docs/plans/dense-soc-leaf-unconnected-plan.md P3).
    if split_dense_sheets_enabled():
        split_dense_soc_sheets(state.bom, state.architecture)
    ensure_leaf_stems_distinct(state.project_stem, state.architecture.sheets)
    owned_files = _read_generated_files(project_dir)
    output_paths = [
        project_dir / f"{state.project_stem}.kicad_sch",
        project_dir / f"{state.project_stem}.kicad_pcb",
        project_dir / f"{state.project_stem}.kicad_pro",
        project_dir / f"{state.project_stem}_autoplacer.json",
        *(project_dir / f"{sheet.stem}.kicad_sch" for sheet in state.architecture.sheets),
    ]
    _check_output_destinations(project_dir, output_paths, owned_files)


    sheet_instances = build_sheet_instances(state.architecture, state.bom)
    attempt_id = str(uuid.uuid4())
    _record_completed_outputs(project_dir, [], attempt_id=attempt_id)

    def record_written(path: Path) -> None:
        _record_completed_outputs(project_dir, [path], attempt_id=attempt_id)

    fragments, library_leaves = _install_library_sheets(
        state, project_dir, sheet_instances, on_written=record_written
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
        on_written=record_written,
    )

    if library_sheet_names:
        for sheet in state.architecture.sheets:
            if sheet.name in library_sheet_names:
                lib_path = project_dir / f"{sheet.stem}.kicad_sch"
                if lib_path.exists():
                    leaves.append(lib_path)

    pro = write_kicad_pro(project_dir, state.project_stem, state.architecture)
    record_written(pro)
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
    record_written(ap)
    # Build the artifact record BEFORE the PCB stub — the schematic files
    # exist on disk regardless of whether the stub or the §9 checks pass —
    # so any failure below can still report what was written
    # (with status="failed").
    artifacts = ArtifactPaths(
        project_dir=project_dir,
        project_stem=state.project_stem,
        root_sch=root,
        leaf_schs=leaves,
        kicad_pro=pro,
        autoplacer_json=ap,
    )

    try:
        pcb = write_empty_pcb(project_dir, state.project_stem, state.bom, copper_layers=copper_layers)
    except PadBindingError as exc:
        # A wired endpoint matched no footprint pad (invisible dead copper).
        # §9.27 rejects the symbol/footprint pairing at BOM commit, but any
        # residual escape used to propagate as an unhandled ValueError and
        # crash the whole build with a bare traceback (live board 627).
        # Surface it as a structured validation failure instead, so the
        # caller records it and the wiring/BOM stage can be re-driven.
        failure = CheckResult(name="9.27 pcb-stub pad binding", ok=False, message=str(exc))
        raise SynthesisValidationError(
            [failure],
            artifacts=artifacts.model_copy(update={"status": "failed"}),
            results=[failure],
        ) from exc
    record_written(pcb)
    staged_models = stage_3d_models(
        project_dir, state.bom, owned_files=owned_files, on_written=record_written
    )
    _reconcile_generated_files(project_dir, [*output_paths, *staged_models], attempt_id=attempt_id)

    results = collect_validations(
        project_dir, state.project_stem, bom=state.bom, architecture=state.architecture,
    )
    results.append(check_typed_led_current_paths(state.architecture, state.bom))
    results.extend(
        [
            check_reviewed_device_support_networks(state.bom),
            check_reviewed_input_operating_ranges(state.architecture, state.bom),
            check_reviewed_analog_input_ranges(state.architecture, state.bom),
            check_reviewed_i2c_address_assignments(state.architecture, state.bom),
            check_reviewed_power_transfer(state.architecture, state.bom),
            check_typed_passive_crossover_values(state.architecture, state.bom),
            check_reviewed_constant_current_led_feedback(state.architecture, state.bom),
            check_requirement_physical_realization(state.architecture, state.bom),
        ]
    )
    if smoke:
        results.append(
            run_solve_subcircuits_smoke(project_dir, state.project_stem, timeout_s=smoke_timeout_s)
        )
    failures = [r for r in results if not r.ok]
    if failures:
        raise SynthesisValidationError(
            failures,
            artifacts=artifacts.model_copy(update={"status": "failed"}),
            results=results,
        )
    manifest = project_dir / ".kicraft" / "synthesis-generated.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["status"] = "complete"
    atomic_write_text(manifest, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return artifacts, results
