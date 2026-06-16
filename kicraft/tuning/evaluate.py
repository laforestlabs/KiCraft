"""Evaluate one (config, board, seed) -> routed reward.

This is the single CPU-bound, $0-LLM primitive the whole tuner is built on. It
runs place+route as a **subprocess** (required: PYTHONHASHSEED only takes effect
at interpreter start, so determinism cannot be pinned in-process), then extracts
the routed reward in-process by reusing ``_verify_routed_board``.

Modes:
* ``replay``  — full place+route from the schematic (``kicraft replay``). Leaf
  placement is regenerated under the candidate config, so leaf-placement params
  are observable. This is the PRIMARY mode for tuning placement.
* ``compose`` — parent compose+route on the workspace's FROZEN routed leaves
  (``kicraft.cli.compose_subcircuits``). Cheaper and isolates parent/route noise,
  but blind to leaf-placement params. Used for the promotion A/B check.

``evaluate_config`` is side-effect-free (no DB writes) and returns a plain
``EvalResult`` so it is safe to fan out across a ``ProcessPoolExecutor``; the
caller owns caching/recording (see ``store.py`` + ``orchestrator.py``).
"""
from __future__ import annotations

import glob
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from kicraft.tuning import workspace as ws

# Penalty applied on the DRC axis when a board fails to produce a routed result
# at all (route crash / timeout / missing board) OR produces a verifiable but
# EMPTY board (no copper routed). Sized to be strictly worse than any real
# board's shorts+unconnected count (real boards run ~0-10 here) so an empty
# board never looks competitive -- but NOT so large that a few unroutable
# boards dominate the whole objective. Lesson from the i8 run: at 999 (~140x a
# real board's DRC), with ~26% of evals empty, empty<->routed flips on a handful
# of unstable boards swamped the fab signal ~6:1 and the optimizer thrashed.
# 100 keeps a ~10x margin over the worst real board while letting fab lead.
MISSING_BOARD_PENALTY = 100


def _effective_drc(*, fab_ready: bool, traces: int, shorts: int, unconnected: int) -> int:
    """DRC-axis value for a verified board, guarding the degenerate optimum.

    A board that routed NO copper (``traces == 0``) and is not fab-ready scores
    ``shorts + unconnected == 0`` and so looks DRC-perfect — but it is an empty
    board, not a clean one. The tuner's reward maximizes ``fab − k·drc − …``, so
    without this guard the optimizer is driven straight into producing empty
    boards (cheap wall-time, zero DRC) instead of routing real ones. Treat such
    a board as a missing board so it can never out-score a board that routes.
    """
    if not fab_ready and traces == 0:
        return MISSING_BOARD_PENALTY
    return shorts + unconnected

REPO_ROOT = Path(__file__).resolve().parents[2]


def _low_priority_prefix() -> list[str]:
    """``nice``/``ionice`` prefix so a foreground user build always preempts the
    tuner. Each binary is included only if present on PATH."""
    prefix: list[str] = []
    if shutil.which("nice"):
        prefix += ["nice", "-n", "19"]
    if shutil.which("ionice"):
        prefix += ["ionice", "-c3"]
    return prefix


@dataclass(frozen=True)
class EvalResult:
    config_hash: str
    board: str
    seed: int
    mode: str
    rc: int
    fab_ready: bool
    shorts: int
    unconnected: int
    drc_total: int  # shorts + unconnected (the DRC-cleanliness axis)
    traces: int
    vias: int
    total_length_mm: float
    wall_s: float
    error: str = ""

    def as_row(self) -> dict:
        return asdict(self)


def _verify_board(pcb: Path) -> dict:
    """Reuse the build's acceptance gate (shorts/unconnected/tracks)."""
    # Importing the engine path also bootstraps pcbnew's sys.path if needed.
    from kicraft.design.cli_app import _verify_routed_board

    return _verify_routed_board(pcb)


def _locate_routed_board(dest: Path, stem: str, mode: str) -> Path | None:
    if mode == "replay":
        promoted = dest / f"{stem}.kicad_pcb"
        return promoted if promoted.exists() else None
    # compose: prefer the routed parent, fall back to the stamped pre-route board
    for name in ("parent_routed.kicad_pcb", "parent_pre_freerouting.kicad_pcb"):
        hits = sorted(
            glob.glob(
                str(dest / ".experiments" / "subcircuits" / "subcircuit__*" / name)
            )
        )
        if hits:
            return Path(hits[-1])
    return None


def _build_argv(dest: Path, stem: str, *, mode: str, seed: int, quality: str,
                spacing_mm: float, cfg_file: Path | None) -> list[str]:
    if mode == "replay":
        return [
            sys.executable, "-m", "kicraft.design.cli_app", "replay",
            "--project", str(dest), "--quality", quality,
            "--seed", str(seed), "--no-fab",
        ]
    if mode == "compose":
        argv = [
            sys.executable, "-m", "kicraft.cli.compose_subcircuits",
            "--project", str(dest), "--parent", stem,
            "--pcb", str(dest / f"{stem}.kicad_pcb"),
            "--spacing-mm", str(spacing_mm), "--stamp", "--route",
            "--seed", str(seed),
        ]
        if cfg_file is not None:
            argv += ["--config", str(cfg_file)]
        return argv
    raise ValueError(f"unknown eval mode: {mode!r}")


def evaluate_config(
    overlay: dict,
    *,
    workspace_path: Path,
    board: str,
    seed: int,
    config_hash: str,
    scratch_dir: Path,
    mode: str = "replay",
    quality: str = "fast",
    spacing_mm: float = 2.0,
    timeout_s: int = 1200,
    use_build_slot: bool = True,
    low_priority: bool = True,
    cleanup: bool = True,
) -> EvalResult:
    """Run place+route for one (config, board, seed) and score the routed board."""
    import os

    from kicraft.build_slots import build_slot

    workspace_path = Path(workspace_path)
    scratch_dir = Path(scratch_dir)

    def _fail(rc: int, err: str, *, shorts: int = 0, unconnected: int = 0,
              traces: int = 0, vias: int = 0, length: float = 0.0,
              wall: float = 0.0, missing: bool = True) -> EvalResult:
        drc = (MISSING_BOARD_PENALTY if missing else shorts + unconnected)
        return EvalResult(
            config_hash=config_hash, board=board, seed=seed, mode=mode, rc=rc,
            fab_ready=False, shorts=shorts, unconnected=unconnected, drc_total=drc,
            traces=traces, vias=vias, total_length_mm=length, wall_s=round(wall, 1),
            error=err,
        )

    try:
        dest, stem = ws.prepare_scratch(
            workspace_path, scratch_dir, fresh_leaves=(mode == "replay")
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(3, f"prepare:{exc}")

    cfg_file: Path | None = None
    try:
        if mode == "replay":
            ws.apply_overlay_autoplacer(dest, overlay)
        else:
            cfg_file = ws.write_overlay_tempfile(dest, overlay)
    except Exception as exc:  # noqa: BLE001
        return _fail(4, f"overlay:{exc}")

    argv = _build_argv(dest, stem, mode=mode, seed=seed, quality=quality,
                       spacing_mm=spacing_mm, cfg_file=cfg_file)
    if low_priority:
        argv = _low_priority_prefix() + argv
    env = {**os.environ, **ws.PINNED_ENV}

    t0 = time.monotonic()
    rc = -1
    err = ""
    try:
        with build_slot(echo=None) if use_build_slot else _nullctx():
            proc = subprocess.run(
                argv, cwd=str(REPO_ROOT), env=env,
                capture_output=True, text=True, timeout=timeout_s,
            )
        rc = proc.returncode
        if rc != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            err = tail[-1][:160] if tail else f"rc={rc}"
    except subprocess.TimeoutExpired:
        wall = time.monotonic() - t0
        if cleanup:
            shutil.rmtree(dest, ignore_errors=True)
        return _fail(124, "timeout", wall=wall)
    except Exception as exc:  # noqa: BLE001
        wall = time.monotonic() - t0
        if cleanup:
            shutil.rmtree(dest, ignore_errors=True)
        return _fail(6, f"subprocess:{exc}", wall=wall)
    wall = time.monotonic() - t0

    board_path = _locate_routed_board(dest, stem, mode)
    if board_path is None:
        result = _fail(rc if rc != 0 else 6, err or "no routed board", wall=wall)
        if cleanup:
            shutil.rmtree(dest, ignore_errors=True)
        return result

    try:
        v = _verify_board(board_path)
    except Exception as exc:  # noqa: BLE001
        result = _fail(rc, f"verify:{exc}", wall=wall)
        if cleanup:
            shutil.rmtree(dest, ignore_errors=True)
        return result

    shorts = int(v.get("shorts", 0) or 0)
    unconnected = int(v.get("unconnected", 0) or 0)
    tracks = v.get("tracks", {}) or {}
    traces = int(tracks.get("traces", 0) or 0)
    fab_ready = bool(v.get("ok", False)) and rc == 0
    result = EvalResult(
        config_hash=config_hash, board=board, seed=seed, mode=mode, rc=rc,
        fab_ready=fab_ready, shorts=shorts, unconnected=unconnected,
        drc_total=_effective_drc(
            fab_ready=fab_ready, traces=traces, shorts=shorts, unconnected=unconnected),
        traces=traces,
        vias=int(tracks.get("vias", 0) or 0),
        total_length_mm=round(float(tracks.get("total_length_mm", 0.0) or 0.0), 2),
        wall_s=round(wall, 1), error=err,
    )
    if cleanup:
        shutil.rmtree(dest, ignore_errors=True)
    return result


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
