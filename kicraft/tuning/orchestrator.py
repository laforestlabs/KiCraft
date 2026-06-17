"""The tuning daemon: ask -> evaluate population -> tell, checkpointed & resumable.

One ``run_tuning`` call optimizes a single scalarization (run several for full
Pareto coverage — the global front is reconstructed from the DB at promote time).
Per generation it asks CMA-ES for a population, evaluates each candidate over the
TRAIN corpus (K seeds, common random numbers), tells CMA the scalarized J,
monitors the gen-best on HOLDOUT (never fed back), appends to the Pareto archive,
and atomically checkpoints. Resuming reloads the CMA state + archive; cached
evals make repeated configs free.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from kicraft.tuning import report_data
from kicraft.tuning import reward as R
from kicraft.tuning import space
from kicraft.tuning.corpus import (
    Workspace, discover_corpus, holdout, split_by_brief, train,
)
from kicraft.tuning.optimizer import load_optimizer, make_optimizer
from kicraft.tuning.runner import evaluate_overlay
from kicraft.tuning.screen import ScreenResult, screen
from kicraft.tuning.store import Store

CHECKPOINT_NAME = "checkpoint.json"
SCREEN_NAME = "screen.json"
REPORT_NAME = "report.json"


@dataclass
class TuneSettings:
    corpus_roots: list[str]
    out_dir: str
    db_path: str = ""           # defaults to <out_dir>/tuning.db
    scratch_root: str = ""      # defaults to <out_dir>/scratch
    mode: str = "replay"
    seeds: tuple[int, ...] = (0, 1, 2)
    scalarization: str = "balanced"
    popsize: int | None = None
    max_gens: int = 30
    max_workers: int | None = None
    quality: str = "fast"
    timeout_s: int = 1200
    holdout_frac: float = 0.3
    split_seed: int = 0
    top_k: int = 12
    n_screen_samples: int = 40
    holdout_every: int = 1
    cma_seed: int = 0
    pin_active: tuple[str, ...] = ()  # always-active knobs (screening fills the rest)

    def resolved(self) -> "TuneSettings":
        out = Path(self.out_dir)
        if not self.db_path:
            self.db_path = str(out / "tuning.db")
        if not self.scratch_root:
            self.scratch_root = str(out / "scratch")
        return self


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _objective_from_record(rec: dict, n_boards: int) -> R.CorpusObjectives:
    return R.CorpusObjectives(
        fab_ready_rate=rec["fab"], mean_drc=rec["drc"], mean_wall_s=rec["wall"],
        worst_board_fab=rec["worst"], n_boards=n_boards,
        mean_area_mm2=rec.get("area", 0.0), mean_orderedness=rec.get("order", 0.0),
    )


def _dedup_archive(archive: list[dict]) -> list[dict]:
    by_hash: dict[str, dict] = {}
    for a in archive:
        by_hash[a["hash"]] = a
    return list(by_hash.values())


def _front(archive: list[dict], n_boards: int) -> list[dict]:
    arch = _dedup_archive(archive)
    objs = [_objective_from_record(a, n_boards) for a in arch]
    idx = R.pareto_front(objs)
    return [arch[i] for i in idx]


def _setup_active(
    settings: TuneSettings, store: Store, tr: Sequence[Workspace], log: Callable
) -> ScreenResult:
    """Load a cached screen.json or run a fresh screening pass."""
    screen_path = Path(settings.out_dir) / SCREEN_NAME
    if screen_path.exists():
        sr = ScreenResult.from_json(screen_path)
        log(f"[tune] loaded {len(sr.active)} active params from {SCREEN_NAME}")
        return sr
    log(f"[tune] screening {len(space.all_param_names())} params "
        f"({settings.n_screen_samples} samples x {len(tr)} train boards) ...")
    if settings.pin_active:
        log(f"[tune] pinned active: {list(settings.pin_active)}")
    sr = screen(
        tr, store=store, scratch_root=settings.scratch_root,
        n_samples=settings.n_screen_samples, seeds=settings.seeds,
        mode=settings.mode, scalarization=settings.scalarization,
        top_k=settings.top_k, pin=settings.pin_active,
        max_workers=settings.max_workers,
        quality=settings.quality, timeout_s=settings.timeout_s,
        progress=lambda i, n, j: log(f"[tune]   screen {i}/{n}  J={j:.3f}"),
    )
    sr.to_json(screen_path)
    log(f"[tune] active params: {sr.active}")
    return sr


def run_tuning(
    settings: TuneSettings, *, run_id: str, log: Callable[[str], None] = print,
    resume: bool = False,
) -> dict:
    settings = settings.resolved()
    out = Path(settings.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = Store(settings.db_path)
    weights = R.SCALARIZATIONS[settings.scalarization]

    workspaces = discover_corpus(settings.corpus_roots)
    split_by_brief(workspaces, holdout_frac=settings.holdout_frac,
                   seed=settings.split_seed)
    tr, ho = train(workspaces), holdout(workspaces)
    if not tr:
        raise RuntimeError(f"no train workspaces under {settings.corpus_roots}")
    log(f"[tune] corpus: {len(tr)} train / {len(ho)} holdout "
        f"board(s); mode={settings.mode} seeds={list(settings.seeds)}")

    sr = _setup_active(settings, store, tr, log)
    active = sr.active

    ckpt_path = out / CHECKPOINT_NAME
    archive: list[dict] = []
    start_gen = 0

    if resume and ckpt_path.exists():
        ck = json.loads(ckpt_path.read_text(encoding="utf-8"))
        opt = load_optimizer(ck["optimizer"])
        archive = ck.get("archive", [])
        start_gen = int(ck.get("gen", 0))
        active = ck.get("active", active)
        log(f"[tune] resumed at gen {start_gen} with {len(archive)} archived configs")
    else:
        x0 = space.initial_vector(active)
        stds = space.initial_stds(active)
        opt = make_optimizer(len(active), x0=x0, stds=stds,
                             popsize=settings.popsize, seed=settings.cma_seed)
        # Baseline: the current default config (empty overlay) as the comparison.
        log("[tune] evaluating baseline (current DEFAULT_CONFIG) ...")
        bobj, _, bhash = evaluate_overlay(
            {}, tr, settings.seeds, scratch_root=settings.scratch_root,
            mode=settings.mode, store=store, max_workers=settings.max_workers,
            quality=settings.quality, timeout_s=settings.timeout_s, source="baseline",
        )
        archive.append({"hash": bhash, "overlay": {}, "fab": bobj.fab_ready_rate,
                        "drc": bobj.mean_drc, "wall": bobj.mean_wall_s,
                        "area": bobj.mean_area_mm2, "order": bobj.mean_orderedness,
                        "worst": bobj.worst_board_fab, "baseline": True})
        log(f"[tune] baseline: fab={bobj.fab_ready_rate:.2f} "
            f"drc={bobj.mean_drc:.2f} wall={bobj.mean_wall_s:.0f}s "
            f"area={bobj.mean_area_mm2:.0f}mm2 order={bobj.mean_orderedness:.1f}")

    for gen in range(start_gen, settings.max_gens):
        if opt.stop():
            log(f"[tune] optimizer converged at gen {gen}")
            break
        t0 = time.monotonic()
        X = opt.ask()
        js: list[float] = []
        best_j, best_overlay = -1e18, {}
        for x in X:
            overlay = space.decode(x, active)
            obj, _, h = evaluate_overlay(
                overlay, tr, settings.seeds, scratch_root=settings.scratch_root,
                mode=settings.mode, store=store, max_workers=settings.max_workers,
                quality=settings.quality, timeout_s=settings.timeout_s,
                source=f"gen{gen}",
            )
            j = R.scalarize(obj, weights)
            js.append(j)
            store.record_generation(
                run_id, gen, h, scalarization=settings.scalarization, j=j,
                is_train=True, fab_ready_rate=obj.fab_ready_rate,
                mean_drc=obj.mean_drc, mean_wall_s=obj.mean_wall_s,
                mean_area_mm2=obj.mean_area_mm2, mean_orderedness=obj.mean_orderedness,
            )
            archive.append({"hash": h, "overlay": overlay,
                            "fab": obj.fab_ready_rate, "drc": obj.mean_drc,
                            "wall": obj.mean_wall_s, "area": obj.mean_area_mm2,
                            "order": obj.mean_orderedness,
                            "worst": obj.worst_board_fab})
            if j > best_j:
                best_j, best_overlay = j, overlay
        opt.tell(X, js)

        # Holdout monitoring (never fed back to the optimizer).
        ho_line = ""
        if ho and settings.holdout_every and gen % settings.holdout_every == 0:
            hobj, _, hh = evaluate_overlay(
                best_overlay, ho, settings.seeds, scratch_root=settings.scratch_root,
                mode=settings.mode, store=store, max_workers=settings.max_workers,
                quality=settings.quality, timeout_s=settings.timeout_s,
                source=f"gen{gen}-holdout",
            )
            store.record_generation(
                run_id, gen, hh, scalarization=settings.scalarization,
                j=R.scalarize(hobj, weights), is_train=False,
                fab_ready_rate=hobj.fab_ready_rate, mean_drc=hobj.mean_drc,
                mean_wall_s=hobj.mean_wall_s,
                mean_area_mm2=hobj.mean_area_mm2, mean_orderedness=hobj.mean_orderedness,
            )
            ho_line = (f" | holdout fab={hobj.fab_ready_rate:.2f} "
                       f"drc={hobj.mean_drc:.2f}")

        _atomic_write_json(ckpt_path, {
            "run_id": run_id, "gen": gen + 1, "active": active,
            "scalarization": settings.scalarization,
            "optimizer": opt.state_dict(), "archive": archive,
            "settings": asdict(settings),
        })
        try:  # publish the chart payload (the one file a remote viewer needs)
            report_data.publish(out)
        except Exception:  # noqa: BLE001 — best-effort, never fail the run
            pass
        dt = time.monotonic() - t0
        log(f"[tune] gen {gen}: pop={len(X)} bestJ={best_j:.3f} "
            f"({dt:.0f}s){ho_line}")

    front = _front(archive, n_boards=len(tr))
    report = {
        "run_id": run_id, "scalarization": settings.scalarization,
        "active_params": active, "n_train": len(tr), "n_holdout": len(ho),
        "n_configs_evaluated": len(_dedup_archive(archive)),
        "pareto_front": sorted(front, key=lambda a: (-a["fab"], a["drc"], a["wall"])),
        "baseline": next((a for a in archive if a.get("baseline")), None),
    }
    _atomic_write_json(out / REPORT_NAME, report)
    try:
        report_data.publish(out)
    except Exception:  # noqa: BLE001
        pass
    store.close()
    log(f"[tune] done: {report['n_configs_evaluated']} configs, "
        f"{len(front)} on the Pareto front -> {out / REPORT_NAME}")
    return report
