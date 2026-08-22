"""Scratch-workspace preparation for headless place+route evaluation.

Factored from ``scripts/ab_compose.py`` and ``scripts/replay_corpus.py``: copy a
synthesized workspace to a private scratch dir, repoint its frozen-leaf
artifacts at the copy, and inject a candidate config overlay. Every evaluation
runs on a fresh copy so concurrent evals never clash and the source corpus is
never mutated.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any

# Frozen leaf artifacts under .experiments commit their absolute paths as this
# token (see replay_corpus.py); live self-eval workspaces instead bake in the
# original absolute path. _detokenize handles both.
PATH_TOKEN = "__KICRAFT_PROJECT_DIR__"

# Env that makes placement reproducible: PYTHONHASHSEED pins set/dict iteration
# + force-state dedup; single-thread BLAS removes FP-reduction jitter that flips
# discrete solver branches. MUST be in the subprocess env (PYTHONHASHSEED only
# takes effect at interpreter startup, so it cannot be set in-process).
PINNED_ENV: dict[str, str] = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def discover_stem(project_dir: Path) -> str:
    """The single ``<stem>`` with a full .kicad_{pro,pcb,sch} triple."""
    stems = [
        pro.stem
        for pro in project_dir.glob("*.kicad_pro")
        if (project_dir / f"{pro.stem}.kicad_pcb").exists()
        and (project_dir / f"{pro.stem}.kicad_sch").exists()
    ]
    if len(stems) != 1:
        raise ValueError(
            f"expected exactly one synthesized stem in {project_dir}, found {stems}"
        )
    return stems[0]


def _detokenize(dest: Path, src: Path) -> None:
    """Repoint frozen-leaf JSON artifacts at this scratch copy."""
    exp = dest / ".experiments"
    if not exp.is_dir():
        return
    src_abs, dest_abs = str(src.resolve()), str(dest.resolve())
    for jf in exp.rglob("*.json"):
        try:
            text = jf.read_text(encoding="utf-8")
        except OSError:
            continue
        new = text.replace(PATH_TOKEN, dest_abs)
        if src_abs != dest_abs:
            new = new.replace(src_abs, dest_abs)
        if new != text:
            jf.write_text(new, encoding="utf-8")


def _drop_stale_parent_artifacts(dest: Path) -> None:
    for name in ("parent_placed.kicad_pcb", "parent_routed.kicad_pcb"):
        for p in glob.glob(
            str(dest / ".experiments" / "subcircuits" / "subcircuit__*" / name)
        ):
            try:
                os.remove(p)
            except OSError:
                pass


def prepare_scratch(
    workspace: Path, dest: Path, *, fresh_leaves: bool
) -> tuple[Path, str]:
    """Copy ``workspace`` to ``dest`` and ready it for evaluation.

    ``fresh_leaves=True`` (replay mode) drops ``.experiments`` so leaf placement
    is regenerated from the schematic under the candidate config — this is what
    makes leaf-placement params observable. ``fresh_leaves=False`` (compose
    mode) keeps the frozen leaves and only clears the stale parent so compose
    re-runs on them.

    Returns ``(dest, stem)``.
    """
    workspace = Path(workspace)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(workspace, dest)
    if fresh_leaves:
        shutil.rmtree(dest / ".experiments", ignore_errors=True)
    else:
        _detokenize(dest, workspace)
        _drop_stale_parent_artifacts(dest)
    return dest, discover_stem(dest)


def _jsonable(value: Any) -> Any:
    """Make a config value JSON-serializable (sets -> sorted lists)."""
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def apply_overlay_autoplacer(dest: Path, overlay: dict) -> Path:
    """Merge ``overlay`` into the scratch project's discovered autoplacer config.

    Used for replay mode, where the engine (solve_hierarchy) auto-discovers the
    config via ``discover_project_config``. We merge over any existing
    per-project config (antenna keepouts, parent_compose_spacing_mm, ...) so
    those board-specific keys are preserved and only the tuned keys change.
    """
    from kicraft.autoplacer.config import discover_project_config, load_project_config

    existing = discover_project_config(dest)
    base: dict = {}
    if existing is not None:
        try:
            base = load_project_config(str(existing))
        except Exception:  # noqa: BLE001 — malformed -> start clean
            base = {}
    merged = {**base, **overlay}
    target = existing if existing is not None else (dest / "autoplacer.json")
    target.write_text(
        json.dumps(_jsonable(merged), indent=2, sort_keys=True), encoding="utf-8"
    )
    return target


def write_overlay_tempfile(dest: Path, overlay: dict) -> Path:
    """Write ``overlay`` as a standalone config file for compose's ``--config``."""
    path = dest / "_tuning_candidate.json"
    path.write_text(
        json.dumps(_jsonable(overlay), indent=2, sort_keys=True), encoding="utf-8"
    )
    return path
