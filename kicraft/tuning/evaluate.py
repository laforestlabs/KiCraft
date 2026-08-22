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

# Area-axis sentinel for a missing/empty board (mm^2). Smaller area is better, so
# a degenerate empty board (which has a tiny or even zero outline) would otherwise
# WIN the size objective. Pin it to a crushing value: real corpus boards span
# ~1200-5200 mm^2, so 30000 (~6x the largest) guarantees an empty/failed board
# can never out-score a board that actually routes on the area axis. Mirrors the
# role of MISSING_BOARD_PENALTY on the DRC axis.
MISSING_BOARD_AREA_MM2 = 30000.0


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


def _effective_area(*, fab_ready: bool, traces: int, area_mm2: float) -> float:
    """Area-axis value for a verified board, guarding the degenerate optimum.

    Same spirit as ``_effective_drc``: an empty board (no copper, not fab-ready)
    must not win the smaller-is-better size objective, and a measurement failure
    (``area_mm2 <= 0`` on a real board) must not masquerade as a tiny board.
    Both collapse to the crushing ``MISSING_BOARD_AREA_MM2`` sentinel.
    """
    if (not fab_ready and traces == 0) or area_mm2 <= 0.0:
        return MISSING_BOARD_AREA_MM2
    return area_mm2


def _measure_geometry(board_path: Path) -> tuple[float, float]:
    """(board_area_mm2, orderedness 0-100) for a routed ``.kicad_pcb``.

    Reconstructs a ``BoardState`` from the finished board (the same extraction the
    placer's hardware adapter uses) and reads two layout-quality signals back out:

    * **area** — the Edge.Cuts bounding-box area (``board_width * board_height``),
      the lever for the "minimize board size" objective.
    * **orderedness** — the mean of the placement scorer's geometry-derived
      sub-scores: rotation alignment (parts at 0/90/180/270), board aspect ratio,
      bbox packing tightness, and courtyard non-overlap. These four are computed
      purely from positions/rotations/outline, so they survive the round-trip
      through a routed board that has lost the placer's group/net metadata.

    Scored with ``DEFAULT_CONFIG`` (not the candidate overlay) so the metric is a
    fixed yardstick that fairly compares candidates. Best-effort: any failure
    returns ``(0.0, 0.0)`` and the ``_effective_area`` guard penalizes the zero.
    """
    from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
    from kicraft.autoplacer.config import DEFAULT_CONFIG
    from kicraft.autoplacer.hardware.adapter import KiCadAdapter

    cfg = dict(DEFAULT_CONFIG)
    adapter = KiCadAdapter(str(board_path), config=cfg)
    state = adapter.load()
    area = max(0.0, state.board_width) * max(0.0, state.board_height)
    score = PlacementScorer(state, config=cfg).score()
    ordered = (
        score.rotation_score
        + score.aspect_ratio
        + score.bbox_packing
        + score.courtyard_overlap
    ) / 4.0
    return (round(area, 1), round(ordered, 2))

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
    board_area_mm2: float = 0.0  # Edge.Cuts bbox area (effective; size axis, MINIMIZE)
    orderedness: float = 0.0     # mean layout-quality sub-score 0-100 (MAXIMIZE)
    # Replay quality preset the result was computed at -- part of the cache
    # identity: different presets route differently, and a lookup that
    # ignored quality silently served cross-quality results (2026-07-19 §8.2).
    quality: str = "fast"

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
    for name in ("parent_routed.kicad_pcb", "parent_placed.kicad_pcb"):
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
            # No usable board to measure: crush the size axis, zero the quality axis.
            board_area_mm2=MISSING_BOARD_AREA_MM2, orderedness=0.0,
            quality=quality,
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

    # Geometry axes (size + orderedness). Best-effort: a measurement failure
    # leaves area_raw at 0.0, which _effective_area then crushes to the sentinel
    # so it can never read as a desirably-tiny board.
    area_raw, ordered = 0.0, 0.0
    try:
        area_raw, ordered = _measure_geometry(board_path)
    except Exception:  # noqa: BLE001
        pass

    result = EvalResult(
        config_hash=config_hash, board=board, seed=seed, mode=mode, rc=rc,
        fab_ready=fab_ready, shorts=shorts, unconnected=unconnected,
        drc_total=_effective_drc(
            fab_ready=fab_ready, traces=traces, shorts=shorts, unconnected=unconnected),
        traces=traces,
        vias=int(tracks.get("vias", 0) or 0),
        total_length_mm=round(float(tracks.get("total_length_mm", 0.0) or 0.0), 2),
        wall_s=round(wall, 1), error=err,
        board_area_mm2=_effective_area(
            fab_ready=fab_ready, traces=traces, area_mm2=area_raw),
        orderedness=ordered,
        quality=quality,
    )
    if cleanup:
        shutil.rmtree(dest, ignore_errors=True)
    return result


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
