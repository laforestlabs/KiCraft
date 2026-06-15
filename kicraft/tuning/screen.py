"""Sensitivity screening: which params actually move the routed objective?

CONFIG_SEARCH_SPACE has ~26 tunable params, but CMA-ES is happiest in ~8-12
dims. We draw random configs (reusing ``autoexperiment._random_sample_config``),
evaluate each against the *routed* objective J over the train corpus, and
correlate each param with J. The top-|corr| params become the CMA active set;
the rest are frozen at their DEFAULT_CONFIG value. This mirrors the
sensitive/insensitive split already documented in config.py, but computed
against the true routed reward rather than the PlacementScore proxy.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from kicraft.tuning import reward as R
from kicraft.tuning.corpus import Workspace
from kicraft.tuning.runner import evaluate_overlay
from kicraft.tuning.space import all_param_names


@dataclass
class ScreenResult:
    active: list[str]
    frozen: list[str]
    correlations: dict[str, float]
    n_samples: int
    scalarization: str
    samples: list[dict] = field(default_factory=list)  # [{overlay, J}]

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps({
            "active": self.active, "frozen": self.frozen,
            "correlations": self.correlations, "n_samples": self.n_samples,
            "scalarization": self.scalarization, "samples": self.samples,
        }, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "ScreenResult":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(active=d["active"], frozen=d["frozen"],
                   correlations=d["correlations"], n_samples=d["n_samples"],
                   scalarization=d.get("scalarization", "balanced"),
                   samples=d.get("samples", []))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx * sy)


def screen(
    workspaces: Sequence[Workspace],
    *,
    store,
    scratch_root: str | Path,
    n_samples: int = 40,
    seeds: Sequence[int] = (0, 1),
    mode: str = "replay",
    scalarization: str = "balanced",
    sample_seed: int = 0,
    top_k: int = 10,
    max_workers: int | None = None,
    quality: str = "fast",
    timeout_s: int = 1200,
    progress: Callable[[int, int, float], None] | None = None,
) -> ScreenResult:
    """Run the screening pass and return the active/frozen param split."""
    from kicraft.autoplacer.config import CONFIG_SEARCH_SPACE
    from kicraft.cli.autoexperiment import _random_sample_config

    params = all_param_names()
    rng = random.Random(sample_seed)
    weights = R.SCALARIZATIONS[scalarization]

    overlays: list[dict] = []
    js: list[float] = []
    for i in range(n_samples):
        full = _random_sample_config(CONFIG_SEARCH_SPACE, rng)
        overlay = {k: full[k] for k in params if k in full}
        obj, _, _ = evaluate_overlay(
            overlay, workspaces, seeds, scratch_root=scratch_root, mode=mode,
            store=store, max_workers=max_workers, quality=quality,
            timeout_s=timeout_s, source="screen",
        )
        j = R.scalarize(obj, weights)
        overlays.append(overlay)
        js.append(j)
        if progress is not None:
            progress(i + 1, n_samples, j)

    corr: dict[str, float] = {}
    for p in params:
        xs = [float(ov[p]) for ov in overlays if p in ov]
        ys = [js[k] for k, ov in enumerate(overlays) if p in ov]
        corr[p] = _pearson(xs, ys)

    ranked = sorted(params, key=lambda p: abs(corr[p]), reverse=True)
    active = ranked[:top_k]
    frozen = [p for p in params if p not in active]
    return ScreenResult(
        active=active, frozen=frozen, correlations=corr, n_samples=n_samples,
        scalarization=scalarization,
        samples=[{"overlay": ov, "J": j} for ov, j in zip(overlays, js)],
    )
