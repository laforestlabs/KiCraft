"""Map between a sparse config overlay and a normalized optimizer vector.

The optimizer works in ``[0, 1]^d`` over a chosen set of ACTIVE parameters
(a subset of ``CONFIG_SEARCH_SPACE`` chosen by sensitivity screening). This
module is the only place that knows the physical bounds/types, so the optimizer
stays domain-agnostic and unit-test-friendly.

Conventions:
* normalize:  value in [min, max]  ->  u in [0, 1]
* denormalize: u in [0, 1]  ->  value (ints rounded; floats rounded to 4 dp,
  matching ``autoexperiment._mutate_config`` so equivalent configs hash equal)
* an overlay is sparse: it contains ONLY the active params, ready to be written
  as an autoplacer.json that merges over DEFAULT_CONFIG.
"""
from __future__ import annotations

from typing import Sequence

from kicraft.autoplacer.config import (
    CONFIG_SEARCH_SPACE,
    DEFAULT_CONFIG,
    enforce_param_constraints,
)

# Parameters that must never be searched even though they may appear in a
# screening pass: rotation-convention coupling silently breaks placement, and
# board dimensions are derived from leaf areas, not tuned.
NEVER_TUNE: frozenset[str] = frozenset({"board_width_mm", "board_height_mm"})


def all_param_names() -> list[str]:
    """Every searchable parameter (deterministic order), minus NEVER_TUNE."""
    return [k for k in CONFIG_SEARCH_SPACE if k not in NEVER_TUNE]


def _spec(name: str) -> dict:
    spec = CONFIG_SEARCH_SPACE.get(name)
    if spec is None:
        raise KeyError(f"{name!r} is not in CONFIG_SEARCH_SPACE")
    return spec


def normalize(name: str, value: float) -> float:
    spec = _spec(name)
    lo, hi = float(spec["min"]), float(spec["max"])
    if hi <= lo:
        return 0.0
    u = (float(value) - lo) / (hi - lo)
    return 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)


def denormalize(name: str, u: float) -> float | int:
    spec = _spec(name)
    lo, hi = float(spec["min"]), float(spec["max"])
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    val = lo + u * (hi - lo)
    if str(spec.get("type")) == "int":
        return int(round(val))
    # force a plain python float (CMA hands back numpy.float64)
    return round(float(val), 4)


def default_value(name: str) -> float:
    """The current DEFAULT_CONFIG value, clamped into the search bounds."""
    spec = _spec(name)
    lo, hi = float(spec["min"]), float(spec["max"])
    cur = float(DEFAULT_CONFIG.get(name, (lo + hi) / 2.0))
    return max(lo, min(hi, cur))


def initial_vector(active: Sequence[str]) -> list[float]:
    """Normalized starting point = the current default for each active param."""
    return [normalize(n, default_value(n)) for n in active]


def initial_stds(active: Sequence[str]) -> list[float]:
    """Per-coordinate initial step = sigma hint mapped to normalized scale.

    A param whose ``sigma`` is a small fraction of its range gets a small step;
    this transfers the hand-tuned exploration scale in CONFIG_SEARCH_SPACE
    straight into the optimizer.
    """
    stds: list[float] = []
    for n in active:
        spec = _spec(n)
        lo, hi = float(spec["min"]), float(spec["max"])
        rng = hi - lo
        s = float(spec.get("sigma", rng * 0.1)) / rng if rng > 0 else 0.1
        stds.append(max(0.02, min(0.5, s)))
    return stds


def decode(vector: Sequence[float], active: Sequence[str]) -> dict[str, float | int]:
    """Normalized vector -> sparse overlay (only active params).

    Applies the same int rounding as the mutator and runs
    ``enforce_param_constraints`` so the overlay is always physically valid.
    """
    overlay: dict[str, float | int] = {}
    for u, name in zip(vector, active):
        overlay[name] = denormalize(name, u)
    enforce_param_constraints(overlay)  # no-op unless coupled keys are present
    return overlay


def encode(overlay: dict, active: Sequence[str]) -> list[float]:
    """Sparse overlay -> normalized vector (default value for missing params)."""
    return [
        normalize(n, overlay[n]) if n in overlay else normalize(n, default_value(n))
        for n in active
    ]
