"""Which design pipeline builds a project, and how the legacy one is driven.

The yield-recovery plan's option 3 puts the August pipeline (`bc6a2f8`, 2026-08-25) behind an
admin switch, because it measured 21/34 finished boards against the current tree's 4/34. Two
trees cannot share one interpreter: the legacy package would shadow the current one on
``sys.path``, so a legacy project's *design* is driven by the legacy tree's own headless
driver in a subprocess, and its *build* by the legacy interpreter.

Everything about the switch that both dispatch sites must agree on lives here: the pinned
commit, the legacy tree's paths, the per-process environment the legacy tree needs to reach
the current model at all, and the marker/provenance record that keeps a swap visible
(operator decision D2: nothing silent, including a pipeline swap).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

#: The pinned legacy tree. `bc6a2f8` is the newest commit that keeps the old design pipeline
#: (no typed driver, no recipes/lowerers, no work units) *and* routes with KRT, which is why
#: it can build on this box at all.
LEGACY_COMMIT = "bc6a2f8"
LEGACY_LABEL = f"legacy pipeline ({LEGACY_COMMIT}, 2026-08-25)"
LEGACY_ROOT_ENV = "KICRAFT_LEGACY_ROOT"
DEFAULT_LEGACY_ROOT = Path("/home/kicraft/KiCraft-legacy")

PIPELINE_CURRENT = "current"
PIPELINE_LEGACY = "legacy"
PIPELINES = (PIPELINE_CURRENT, PIPELINE_LEGACY)

#: What the operator is choosing, in the words the admin page shows.
TRADE_OFF = (
    "The legacy tree is a fork that will not receive the fixes from options 1-2, and this host "
    "has one build slot: a legacy build serializes against a production build, so a user's "
    "build can wait up to one build timeout (2400 s)."
)

#: The environment the legacy tree needs. Measured 2026-09-20: its DeepSeek-era defaults cap
#: the price below the current model and pin provider routing to hosts that do not serve it,
#: so *every* call failed in under a second at zero cost until these were set per process.
LEGACY_ENV = {
    "KICRAFT_PROVIDER_ORDER": "openai",
    "KICRAFT_MAX_PRICE_PROMPT": "0.20",
    "KICRAFT_MAX_PRICE_COMPLETION": "1.20",
    "KICRAFT_DESIGN_REASONING_TOKENS": "0",
}

_MARKER_NAME = "pipeline.json"


def legacy_root() -> Path:
    raw = os.environ.get(LEGACY_ROOT_ENV, "").strip()
    return Path(raw) if raw else DEFAULT_LEGACY_ROOT


def legacy_python(root: Path | None = None) -> Path:
    """The legacy tree's interpreter (its own venv, created with --system-site-packages)."""
    return (root or legacy_root()) / ".venv" / "bin" / "python"


def legacy_available(root: Path | None = None) -> bool:
    """Whether the legacy tree and its interpreter are present."""
    base = root or legacy_root()
    return (base / "kicraft" / "server" / "stage_driver.py").is_file() and legacy_python(
        base
    ).is_file()


def normalise(value: object) -> str:
    """A pipeline name, or `current` for anything unset/unknown."""
    name = str(value or "").strip().lower()
    return name if name in PIPELINES else PIPELINE_CURRENT


def selected() -> str:
    """The pipeline the next design run uses.

    The durable routing config wins (it is the admin knob, and the build worker re-reads it
    per job); otherwise the environment (`KICRAFT_PIPELINE`), otherwise the current tree. An
    unavailable legacy tree falls back to `current` rather than failing a user's run.
    """
    from . import routing_config

    configured = normalise(routing_config.load().pipeline)
    if configured == PIPELINE_LEGACY and legacy_available():
        return PIPELINE_LEGACY
    if configured == PIPELINE_CURRENT:
        return PIPELINE_CURRENT
    from .config import Settings

    try:
        env_value = normalise(getattr(Settings.from_env(), "pipeline", None))
    except (SystemExit, ValueError):
        env_value = PIPELINE_CURRENT
    if env_value == PIPELINE_LEGACY and legacy_available():
        return PIPELINE_LEGACY
    return PIPELINE_CURRENT


def _marker_path(ws) -> Path:
    return Path(ws) / ".kicraft" / _MARKER_NAME


def write_marker(ws, pipeline: str) -> Path:
    """Record which pipeline built this workspace, so later readers never have to guess.

    Written at project creation / first dispatch: the build worker and the board page read it
    to pick the right interpreter and to name the pipeline, and the board's own provenance
    carries the same value (a swap that is not recorded must be impossible).
    """
    path = _marker_path(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": normalise(pipeline),
        "legacy_commit": LEGACY_COMMIT if normalise(pipeline) == PIPELINE_LEGACY else None,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_marker(ws) -> str | None:
    """The recorded pipeline for a workspace, or None when no marker exists yet."""
    try:
        payload = json.loads(_marker_path(ws).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("pipeline")
    return name if name in PIPELINES else None


def project_pipeline(ws) -> str:
    """The pipeline to BUILD a workspace with: its marker, else the configured default.

    A legacy design that builds with the current tree would emit a board from a state schema
    the current emitters do not own, so the marker wins once it is written.
    """
    return read_marker(ws) or selected()


def current_tree_owns_tail(ws) -> bool:
    """Whether CURRENT-tree code may write this workspace's ``state.json``.

    False for a legacy workspace, and that is not a style choice: the legacy tree's part model
    has none of the current tree's fields (`assembly`, `recipe_id`, `resolution_*`,
    `lowering_*`), so any current-tree stage that re-serializes the state adds them and the
    legacy build then rejects the file — measured 2026-09-21 on project 878, 171 schema errors,
    after the current tree's post-wiring lifecycle had run on a legacy design. Everything that
    writes a legacy workspace (its own design driver, its own build) must be legacy code; the
    current tree may only read the delivered artifacts.
    """
    return project_pipeline(ws) != PIPELINE_LEGACY


def provenance_fields(pipeline: str) -> dict:
    """The pipeline stamp a project row / promote provenance carries."""
    name = normalise(pipeline)
    return {
        "pipeline": name,
        "pipeline_legacy_commit": LEGACY_COMMIT if name == PIPELINE_LEGACY else None,
        "pipeline_label": LEGACY_LABEL if name == PIPELINE_LEGACY else "current pipeline",
    }


def run_legacy_design(
    ws,
    brief: str,
    stages,
    *,
    budget_usd: float,
    max_tokens: int = 4096,
    max_retries: int = 2,
    timeout_s: float | None = None,
) -> tuple[int, str, str]:
    """Drive the design stages through the legacy tree's own headless driver.

    A subprocess, not an import: two versions of the `kicraft` package cannot coexist in one
    interpreter. The driver commits the same five stage slots into the same workspace (its
    `state.json` schema is its own), so the build tail starts where the current pipeline's
    would. Returns ``(returncode, stdout, stderr)``.
    """
    root = legacy_root()
    command = [
        str(legacy_python(root)),
        "-m",
        "kicraft.server.stage_driver",
        "run",
        "--brief",
        brief,
        "--workspace",
        str(Path(ws)),
        "--stages",
        ",".join(stages),
        "--no-build",
        "--budget",
        f"{float(budget_usd):.4f}",
        "--max-tokens",
        str(int(max_tokens)),
        "--max-retries",
        str(int(max_retries)),
    ]
    # `cwd` is the legacy root and PYTHONPATH pins it: the current checkout must never be the
    # one that resolves, whichever of the two a future caller's cwd happens to be.
    env = {**os.environ, **LEGACY_ENV, "PYTHONPATH": str(root)}
    env["KICRAFT_PROJECTS_DIR"] = os.environ.get(
        "KICRAFT_LEGACY_PROJECTS_DIR", str(Path.home() / ".kicraft" / "projects-legacy")
    )
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(root),
        timeout=timeout_s,
    )
    return completed.returncode, completed.stdout, completed.stderr


def legacy_build_python() -> str:
    """The interpreter a legacy workspace's build runs under."""
    return str(legacy_python())


def build_command(cmd_base: list[str], pipeline: str) -> list[str]:
    """Substitute the legacy interpreter for a legacy workspace's build job.

    Only the interpreter changes: the argv contract (the module, the subcommand and its paths)
    is the same in both trees, and the legacy venv's editable install resolves its own checkout
    from any directory that does not itself hold a `kicraft/` package (measured: a build job's
    workspace, and the legacy root, both resolve to `/home/kicraft/KiCraft-legacy/kicraft`).
    `PYTHONPATH` pins that explicitly (see `legacy_env`) so the current checkout can never
    shadow it, whatever the child's cwd turns out to be.
    """
    if normalise(pipeline) != PIPELINE_LEGACY or not cmd_base:
        return list(cmd_base)
    return [legacy_build_python(), *cmd_base[1:]]


def legacy_env(base: dict | None = None) -> dict:
    """The environment a legacy build subprocess needs (its own package, never this one)."""
    return {**(base or {}), "PYTHONPATH": str(legacy_root())}


def describe() -> dict:
    """The switch's current state, for the admin page and for diagnostics."""
    return {
        "available": legacy_available(),
        "root": str(legacy_root()),
        "commit": LEGACY_COMMIT,
        "label": LEGACY_LABEL,
        "selected": selected(),
        "current_interpreter": sys.executable,
        "legacy_interpreter": str(legacy_python()),
        "trade_off": TRADE_OFF,
    }
