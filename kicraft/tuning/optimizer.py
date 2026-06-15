"""CMA-ES optimizer for the tuning loop.

ask/tell interface that **MAXIMIZES** the objective, operating in normalized
``[0, 1]^d`` space (see ``space.py``) so every coordinate is commensurate
regardless of the underlying parameter's physical units. Per-coordinate initial
step sizes (``stds``) come from the CONFIG_SEARCH_SPACE ``sigma`` hints via
``space.initial_stds``.

Backed by the ``cma`` package (the reference CMA-ES): robust to noise, handles
correlated dimensions via its covariance matrix, evaluates a whole population
per generation. ``cma`` minimizes, so this adapter negates.

``cma`` is a **hard dependency** (declared under the ``tuning`` extra). There is
deliberately no fallback optimizer: if ``cma`` is missing this module fails
loudly at import rather than silently degrading to a worse search.
"""
from __future__ import annotations

import warnings
from typing import Sequence

with warnings.catch_warnings():
    # cma warns about a missing matplotlib (plotting only); irrelevant here.
    warnings.simplefilter("ignore")
    import cma as _cma  # hard dep — ImportError here is the intended loud failure


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


class CMAES:
    """ask/tell CMA-ES that maximizes fitness over the unit box."""

    def __init__(
        self,
        dim: int,
        *,
        x0: Sequence[float] | None = None,
        sigma0: float = 0.25,
        stds: Sequence[float] | None = None,
        popsize: int | None = None,
        seed: int = 0,
    ) -> None:
        self.dim = int(dim)
        start = list(x0) if x0 is not None else [0.5] * dim
        opts: dict = {"bounds": [0.0, 1.0], "verbose": -9, "seed": int(seed)}
        if stds is not None:
            opts["CMA_stds"] = [float(s) for s in stds]
        if popsize is not None:
            opts["popsize"] = int(popsize)
        self._es = _cma.CMAEvolutionStrategy(start, float(sigma0), opts)

    def ask(self) -> list[list[float]]:
        return [[_clip01(v) for v in x] for x in self._es.ask()]

    def tell(self, xs: Sequence[Sequence[float]], fitnesses: Sequence[float]) -> None:
        # We maximize; cma minimizes -> negate.
        self._es.tell([list(x) for x in xs], [-float(f) for f in fitnesses])

    def stop(self) -> bool:
        return bool(self._es.stop())

    @property
    def popsize(self) -> int:
        return int(self._es.popsize)

    @property
    def generation(self) -> int:
        return int(self._es.countiter)

    def best(self) -> tuple[list[float] | None, float | None]:
        r = self._es.result
        xbest = list(r.xbest) if r.xbest is not None else None
        fbest = (-float(r.fbest)) if r.fbest is not None else None
        return xbest, fbest

    # --- checkpoint / resume: pickle the underlying strategy ---------------
    def state_dict(self) -> dict:
        import base64
        import pickle

        return {
            "kind": "cma",
            "dim": self.dim,
            "pickle": base64.b64encode(pickle.dumps(self._es)).decode("ascii"),
        }

    @classmethod
    def from_state(cls, state: dict) -> "CMAES":
        import base64
        import pickle

        obj = cls.__new__(cls)
        obj._es = pickle.loads(base64.b64decode(state["pickle"]))
        obj.dim = int(state["dim"])
        return obj


def make_optimizer(
    dim: int,
    *,
    x0: Sequence[float] | None = None,
    stds: Sequence[float] | None = None,
    popsize: int | None = None,
    seed: int = 0,
) -> CMAES:
    return CMAES(dim, x0=x0, stds=stds, popsize=popsize, seed=seed)


def load_optimizer(state: dict) -> CMAES:
    """Reconstruct an optimizer from a ``state_dict``."""
    if state.get("kind") != "cma":
        raise ValueError(f"unknown optimizer state kind: {state.get('kind')!r}")
    return CMAES.from_state(state)
