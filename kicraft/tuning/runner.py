"""Evaluate one candidate overlay across the corpus × seeds, with caching.

Shared by ``screen`` and ``orchestrator``: given an overlay, run every
(board, seed) that isn't already cached through ``evaluate_config`` on a process
pool (each subprocess self-gates on a build slot), record results, and return the
aggregated 3-axis objective. Concurrency leaves one build slot free so a real
user build always has headroom.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Sequence

from kicraft.tuning import store as store_mod
from kicraft.tuning.corpus import Workspace
from kicraft.tuning.evaluate import EvalResult, evaluate_config
from kicraft.tuning.reward import CorpusObjectives, aggregate_results


def default_workers() -> int:
    """Concurrency that leaves a build slot free for real user builds."""
    from kicraft.build_slots import slot_count

    n = slot_count()
    if n <= 0:  # gating disabled (tests / throwaway hosts)
        return max(1, (os.cpu_count() or 2) // 6)
    return max(1, n - 1)


def evaluate_overlay(
    overlay: dict,
    workspaces: Sequence[Workspace],
    seeds: Sequence[int],
    *,
    scratch_root: str | Path,
    mode: str = "replay",
    store: "store_mod.Store | None" = None,
    max_workers: int | None = None,
    quality: str = "fast",
    timeout_s: int = 1200,
    low_priority: bool = True,
    source: str = "",
    on_result: Callable[[EvalResult], None] | None = None,
) -> tuple[CorpusObjectives, list[EvalResult], str]:
    """Return ``(objective, all_results, config_hash)`` for ``overlay``."""
    cfg_hash = store_mod.config_hash(overlay)
    if store is not None:
        store.upsert_config(cfg_hash, overlay, source)
    scratch_root = Path(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)

    results: list[EvalResult] = []
    todo: list[tuple[Workspace, int]] = []
    for w in workspaces:
        for s in seeds:
            cached = (
                store.lookup(cfg_hash, w.name, s, mode, quality)
                if store
                else None
            )
            if cached is not None:
                results.append(cached)
            else:
                todo.append((w, s))

    if todo:
        mw = max_workers or default_workers()
        with ProcessPoolExecutor(max_workers=mw) as ex:
            futs = {}
            for i, (w, s) in enumerate(todo):
                scratch = scratch_root / f"{cfg_hash[:8]}_{w.name}_{s}_{i}"
                fut = ex.submit(
                    evaluate_config, overlay,
                    workspace_path=w.path, board=w.name, seed=s,
                    config_hash=cfg_hash, scratch_dir=scratch, mode=mode,
                    quality=quality, timeout_s=timeout_s, low_priority=low_priority,
                )
                futs[fut] = (w, s)
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                if store is not None:
                    store.record(r)
                if on_result is not None:
                    on_result(r)

    return aggregate_results(results), results, cfg_hash
