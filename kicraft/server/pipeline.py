"""Design backend selection with one current manufacturing implementation.

Native design runs in the pinned interpreter. Its canonical state remains native
so questions, reconciliation and later edits can resume without schema pollution.
Current manufacturing consumes an isolated snapshot and publishes only build
artifacts back to that state; the obsolete native routing implementation is never
used for new builds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

#: Pinned native design backend; manufacturing always uses the current checkout.
LEGACY_COMMIT = "bc6a2f8"
LEGACY_LABEL = f"native design ({LEGACY_COMMIT}) + current manufacturing"
LEGACY_ROOT_ENV = "KICRAFT_LEGACY_ROOT"
DEFAULT_LEGACY_ROOT = Path("/home/kicraft/KiCraft-legacy")

PIPELINE_CURRENT = "current"
PIPELINE_LEGACY = "legacy"
PIPELINES = (PIPELINE_CURRENT, PIPELINE_LEGACY)

#: What the operator is choosing, in the words the admin page shows.
TRADE_OFF = (
    "Native design uses the pinned August stage engine; manufacturing uses current "
    "routing and fabrication gates. Native projects do not use current typed design "
    "work units or post-wiring authorship. All builds share this host's single build slot."
)

#: The environment the legacy tree needs. Measured 2026-09-20: its DeepSeek-era defaults cap
#: the price below the current model and pin provider routing to hosts that do not serve it,
#: so *every* call failed in under a second at zero cost until these were set per process.
LEGACY_ENV = {
    "KICRAFT_PROVIDER_ORDER": "openai",
    "KICRAFT_MAX_PRICE_PROMPT": "0.10",
    "KICRAFT_MAX_PRICE_COMPLETION": "0.50",
    "KICRAFT_DESIGN_REASONING_TOKENS": "0",
}

_MARKER_NAME = "pipeline.json"
_LEGACY_SESSION_RESULT_PREFIX = "__KICRAFT_LEGACY_SESSION_RESULT__="
_LEGACY_EVENT_PREFIX = "__KICRAFT_LEGACY_EVENT__="


def legacy_session_result(stdout: str) -> dict | None:
    """Extract the legacy session runner's marked JSON response, if it finished."""
    for line in reversed((stdout or "").splitlines()):
        if not line.startswith(_LEGACY_SESSION_RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(_LEGACY_SESSION_RESULT_PREFIX) :])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _legacy_session_runner() -> Path:
    """The tiny protocol endpoint, executed *by* the legacy interpreter."""
    return Path(__file__).with_name("legacy_session_runner.py")


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
    """The workspace's design backend, pinned by its marker once selected."""
    return read_marker(ws) or selected()


def current_tree_owns_design_state(ws) -> bool:
    """Whether current design stages may reserialize the canonical state.

    Native sessions reject current-only part fields. Their manufacturing therefore
    runs through native_build_runner on a separate current-schema snapshot.
    """
    return project_pipeline(ws) != PIPELINE_LEGACY


def provenance_fields(pipeline: str) -> dict:
    """The pipeline stamp a project row / promote provenance carries."""
    name = normalise(pipeline)
    return {
        "pipeline": name,
        "pipeline_legacy_commit": LEGACY_COMMIT if name == PIPELINE_LEGACY else None,
        "manufacturing_pipeline": PIPELINE_CURRENT,
        "pipeline_label": LEGACY_LABEL if name == PIPELINE_LEGACY else "current pipeline",
    }


def run_legacy_design(
    ws,
    brief: str,
    stages,
    *,
    budget_usd: float,
    answers=None,
    instruction: str | None = None,
    run_id: str | None = None,
    timeout_s: float | None = None,
    progress=None,
    auto_default_questions: bool | None = None,
) -> tuple[int, str, str]:
    """Drive a legacy-owned session and its bounded BOM reconciliation.

    The protocol is intentionally only JSON over stdin/stdout. The runner executes
    in the legacy interpreter and imports the legacy session APIs, which keeps its
    models responsible for every state.json write while retaining one capped client
    across the initial stages and every internal reconciliation re-drive.
    """
    root = legacy_root()
    request = {
        "workspace": str(Path(ws)),
        "brief": brief,
        "stages": list(stages),
        "answers": answers,
        "instruction": instruction,
        "run_id": run_id,
        "budget_usd": float(budget_usd),
        "auto_default_questions": auto_default_questions,
    }
    command = [str(legacy_python(root)), "-u", str(_legacy_session_runner())]
    # `cwd` is the legacy root and PYTHONPATH pins it: the current checkout must never be the
    # one that resolves, whichever of the two a future caller's cwd happens to be.
    env = {**os.environ, **LEGACY_ENV, "PYTHONPATH": str(root)}
    env["KICRAFT_PROJECTS_DIR"] = os.environ.get(
        "KICRAFT_LEGACY_PROJECTS_DIR", str(Path.home() / ".kicraft" / "projects-legacy")
    )
    child = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace", bufsize=1, env=env, cwd=str(root),
    )
    output, errors = [], []
    read_errors = []
    finished = threading.Event()
    expired = threading.Event()

    def drain_stderr():
        try:
            errors.append(child.stderr.read())
        except BaseException as exc:
            read_errors.append(exc)
            child.kill()

    def watchdog():
        if not finished.wait(timeout_s):
            expired.set()
            child.kill()

    reader = threading.Thread(target=drain_stderr)
    timer = threading.Thread(target=watchdog) if timeout_s is not None else None
    reader.start()
    if timer:
        timer.start()
    try:
        child.stdin.write(json.dumps(request))
        child.stdin.close()
        for line in child.stdout:
            if line.startswith(_LEGACY_EVENT_PREFIX):
                event = json.loads(line[len(_LEGACY_EVENT_PREFIX):])
                if not isinstance(event, dict) or not isinstance(event.get("kind"), str) or not event["kind"].strip():
                    raise ValueError("Invalid legacy progress event")
                if progress:
                    progress(event)
            else:
                output.append(line)
        child.wait()
    except BaseException:
        if child.poll() is None:
            child.kill()
        child.wait()
        if expired.is_set():
            raise subprocess.TimeoutExpired(command, timeout_s) from None
        raise
    finally:
        finished.set()
        if timer:
            timer.join()
        reader.join()
        try:
            child.stdin.close()
        except OSError:
            pass  # An already-terminated child may have closed the request pipe.
        child.stdout.close()
        child.stderr.close()
    stdout, stderr = "".join(output), "".join(errors)
    if expired.is_set():
        raise subprocess.TimeoutExpired(command, timeout_s, output=stdout, stderr=stderr)
    if read_errors:
        raise read_errors[0]
    return child.returncode, stdout, stderr


def build_command(cmd_base: list[str], pipeline: str) -> list[str]:
    """Use current manufacturing, isolating native canonical design state."""
    if normalise(pipeline) != PIPELINE_LEGACY:
        return list(cmd_base)
    if cmd_base[1:3] != ["-m", "kicraft.design.cli_app"]:
        raise ValueError("Native design builds require the current manufacturing CLI")
    return [cmd_base[0], "-m", "kicraft.server.native_build_runner", *cmd_base[3:]]


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
