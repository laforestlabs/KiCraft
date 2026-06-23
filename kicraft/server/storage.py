"""Run-storage lifecycle: workspaces (scratch) and the durable project tree.

Extracted from web.py (refactor roadmap Phase 3) so the workspace<->durable
mechanics live in one small module instead of being scattered through the web
monolith. A *workspace* is scratch space under KICRAFT_WORK_DIR that the web and
build-worker processes share; the *durable* copy lives under projects_dir. See
CLAUDE.md "Storage model" and docs/plans/view-from-durable-refactor-v2.md for the
planned collapse of the `.kicraft`/`kicraft` duality.

NOTE: persistence (workspace -> durable, ``_persist_project``) still lives in
web.py for now because it is coupled to the accounts store + notifications; it
moves here when build orchestration is extracted.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from .config import Settings


def _kicraft_dir(root: Path) -> Path:
    """Run-metadata dir for a *workspace* (``.kicraft``, dotted) OR a *durable*
    project (``kicraft``, no dot). Prefer whichever already exists; default to the
    durable name for paths about to be created. This is the one seam that lets a
    reader take either root unchanged — see docs/plans/view-from-durable-refactor-v2.md
    ("The core friction") and CLAUDE.md "Storage model"."""
    for cand in (root / ".kicraft", root / "kicraft"):
        if cand.is_dir():
            return cand
    return root / "kicraft"


def _state_path(root: Path) -> Path:
    """Resolved ``state.json`` for a root: under ``.kicraft``/``kicraft`` if present,
    else a legacy top-level ``state.json`` (durable projects predating the kicraft/
    tree). Lets the readers work against a workspace, a durable project, or a legacy
    project with one call."""
    p = _kicraft_dir(root) / "state.json"
    return p if p.is_file() else (root / "state.json")


def _new_workspace(prefix: str) -> Path:
    """A run workspace under the shared work dir (KICRAFT_WORK_DIR), NOT /tmp:
    the standalone build worker is a separate systemd unit, and PrivateTmp would
    hide a /tmp workspace from it. Also what lets a build survive a web restart."""
    root = Settings.from_env().work_dir
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


def _gc_workspaces(max_age_days: float = 2.0) -> None:
    """Drop abandoned run workspaces. Everything durable was copied into
    projects_dir at finalize time (reopen rehydrates from there, not from the
    workspace), so a workspace only needs to outlive its own live page session.
    Two days bounds the disk held by .experiments trees, which dwarf the
    durable copies."""
    try:
        root = Settings.from_env().work_dir
        if not root.is_dir():
            return
        cutoff = time.time() - max_age_days * 86400
        for d in root.iterdir():
            try:
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                continue
    except Exception:  # housekeeping must never block startup
        pass


def _rehydrate_workspace(project) -> Path:
    """Recreate a working tempdir from a saved project's durable .kicraft/ (state +
    fetched parts) and generated tree, so the session can resume, edit, or rebuild
    against it. Falls back to the top-level state.json for legacy projects that
    predate the saved kicraft/ tree."""
    ws = _new_workspace("kicraft_resume_")
    base = Path(project.dir_path) if project.dir_path else None
    if base and (base / "kicraft").is_dir():
        shutil.copytree(base / "kicraft", ws / ".kicraft")
    elif base and (base / "state.json").is_file():
        (ws / ".kicraft").mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / "state.json", ws / ".kicraft" / "state.json")
    if base and (base / "generated").is_dir():
        shutil.copytree(base / "generated", ws / "generated")
    return ws


def _read_project_stem(ws: Path) -> str | None:
    """The project_stem committed by the intent stage (UPPER_SNAKE_CASE)."""
    try:
        data = json.loads(_state_path(ws).read_text(encoding="utf-8"))
        stem = data.get("project_stem")
        if stem:
            return str(stem)
    except (OSError, json.JSONDecodeError):
        pass
    for pro in (ws / "generated").glob("*/*.kicad_pro"):  # fallback once synth ran
        return pro.stem
    return None


def _discover_generated_dir(ws: Path | None) -> Path | None:
    """The synthesized project dir (``generated/<STEM>/``) in a workspace, found by
    inspection so the schematic stays viewable even when a run FAILS and no
    project_stem was recorded. Prefers the committed stem, then any subdir that
    actually holds schematic sheets. None until synthesis has written a sheet."""
    if ws is None:
        return None
    gen = ws / "generated"
    if not gen.is_dir():
        return None
    stem = _read_project_stem(ws)
    if stem and any((gen / stem).glob("*.kicad_sch")):
        return gen / stem
    for d in sorted(gen.iterdir()):
        if d.is_dir() and any(d.glob("*.kicad_sch")):
            return d
    return None


def _persisted_generated_dir(dir_path, stem) -> Path | None:
    """The generated KiCad dir (`generated/<STEM>/`) inside a persisted project, by
    stem first then by inspection, so it resolves even for legacy/odd-named runs."""
    if not dir_path:
        return None
    base = Path(dir_path)
    if stem and (base / "generated" / stem).is_dir() \
            and any((base / "generated" / stem).glob("*.kicad_sch")):
        return base / "generated" / stem
    return _discover_generated_dir(base)
