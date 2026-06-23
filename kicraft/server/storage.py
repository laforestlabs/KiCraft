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


def _read_root(state: dict) -> Path | None:
    """The project's directory for this session -- where state + artifacts are read AND
    written (build-in-place: the durable project dir). None for a brand-new run before
    it exists."""
    r = state.get("ws")
    return Path(r) if r else None


def _kicraft_dir(root: Path) -> Path:
    """The run-metadata dir for a project: always ``<root>/.kicraft/`` (state.json,
    fetched parts, check files). One name, no fallback -- see CLAUDE.md "Storage model"."""
    return root / ".kicraft"


def _state_path(root: Path) -> Path:
    """The committed ``state.json`` for a root: always ``<root>/.kicraft/state.json``."""
    return root / ".kicraft" / "state.json"


def _new_workspace(prefix: str) -> Path:
    """A run workspace under the shared work dir (KICRAFT_WORK_DIR), NOT /tmp:
    the standalone build worker is a separate systemd unit, and PrivateTmp would
    hide a /tmp workspace from it. Also what lets a build survive a web restart."""
    root = Settings.from_env().work_dir
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


def _gc_workspaces(max_age_days: float = 2.0) -> None:
    """Drop abandoned throwaway tempdirs under KICRAFT_WORK_DIR. Real projects build
    in place under projects_dir (not here), so this only reaps the scratch tempdirs of
    id-less/admin (self-eval) runs. Two days bounds the disk their .experiments trees hold."""
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
