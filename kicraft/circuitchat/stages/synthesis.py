"""Stage 5: mechanical synthesis.

Reads a complete ConversationState and writes the file set described in
`docs/circuitchat_schematic_prompt.md` (root + leaf schematics, .kicad_pro,
_autoplacer.json). Runs §9.1-§9.6 unconditionally and §9.7 (solve-subcircuits
smoke) optionally.

This stage is intentionally non-LLM. If state validation fails the function
raises `SynthesisInputError` BEFORE touching the filesystem. If §9 validation
fails the function raises `SynthesisValidationError` AFTER writing files
(so the user can inspect what was produced).
"""
from __future__ import annotations

from pathlib import Path

from ..models import ArtifactPaths, ConversationState
from ..synthesis.autoplacer import write_autoplacer_json
from ..synthesis.emitter import emit_schematic
from ..synthesis.kicad_pcb_stub import write_empty_pcb
from ..synthesis.kicad_pro import write_kicad_pro
from ..synthesis.validation import (
    CheckResult,
    SynthesisValidationError,
    run_solve_subcircuits_smoke,
    run_validations,
)


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

    # Cross-slot: BOM sheets must all exist in architecture.
    sheet_names = {s.name for s in state.architecture.sheets}  # type: ignore[union-attr]
    bad = [
        p.ref for p in state.bom.parts if p.sheet not in sheet_names  # type: ignore[union-attr]
    ]
    if bad:
        raise SynthesisInputError(
            f"BOM parts reference sheets not in architecture: {bad}"
        )


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
        SynthesisValidationError: written files fail §9.1-§9.6 checks.
    """
    _require_state(state)
    assert state.project_stem is not None
    assert state.architecture is not None
    assert state.bom is not None

    project_dir = project_dir.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    root, leaves = emit_schematic(
        project_dir,
        state.project_stem,
        state.architecture,
        state.bom,
        title=state.intent.goal if state.intent else state.project_stem,
    )
    pro = write_kicad_pro(project_dir, state.project_stem, state.architecture)
    ap = write_autoplacer_json(
        project_dir, state.project_stem, state.architecture, state.bom
    )
    write_empty_pcb(project_dir, state.project_stem)

    results = run_validations(project_dir, state.project_stem)
    if smoke:
        results.append(
            run_solve_subcircuits_smoke(
                project_dir, state.project_stem, timeout_s=smoke_timeout_s
            )
        )
        failures = [r for r in results if not r.ok]
        if failures:
            raise SynthesisValidationError(failures)

    artifacts = ArtifactPaths(
        project_dir=project_dir,
        project_stem=state.project_stem,
        root_sch=root,
        leaf_schs=leaves,
        kicad_pro=pro,
        autoplacer_json=ap,
    )
    return artifacts, results
