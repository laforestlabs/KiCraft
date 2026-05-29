"""Programmatic grid placement for matrix/array leaves.

Some leaves are regular arrays of identical components — e.g. a 10x20
addressable-LED matrix. Throwing 200 identical, daisy-chained parts at the
force-directed + simulated-annealing solver in :mod:`placement_solver` does not
converge: the sibling-grouping pass and the power-net cliques explode into a
near-complete graph, and the per-iteration crossover scorer becomes the
bottleneck. Such leaves carry an explicit array hint from synthesis
(``autoplacer.json`` -> solver ``cfg["arrays"]``, a list of
``{refs, rows, cols, pitch_mm, serpentine}`` dicts). We lay their members out
deterministically as a serpentine grid and skip the optimizer entirely.

Members are listed in data-chain order, so a serpentine (boustrophedon) fill
keeps consecutive members physically adjacent — the DOUT->DIN routes stay short.
"""
from __future__ import annotations

from .placement_utils import _update_pad_positions
from .types import Component, Point


def _move(comp: Component, x: float, y: float) -> None:
    """Move a component's body center to (x, y), carrying its pads along.

    Rotation is left unchanged, so ``_update_pad_positions`` is a pure
    translation (the canonical move pattern used across the solver).
    """
    old = Point(comp.pos.x, comp.pos.y)
    comp.pos = Point(x, y)
    _update_pad_positions(comp, old, comp.rotation)


def _pitch(members: list[Component], spec: dict, gap: float) -> tuple[float, float]:
    """Grid pitch (x, y). Explicit ``pitch_mm`` wins, else courtyard + gap."""
    p = spec.get("pitch_mm")
    if p:
        return float(p), float(p)
    px = max(c.width_mm for c in members) + gap
    py = max(c.height_mm for c in members) + gap
    return px, py


def leaf_is_fully_array(comps: dict[str, Component], arrays: list[dict]) -> bool:
    """True if this leaf is one or more full array grids plus only simple
    two-terminal passives — i.e. ``place_array_leaves`` would fully handle it.

    Such a leaf is placed deterministically (grid, no force/SA), so its routing
    is identical every round; callers use this to avoid re-routing it. Pure
    predicate — does not mutate ``comps``.
    """
    covered: set[str] = set()
    for spec in arrays or []:
        refs = list(spec.get("refs", []))
        rows = int(spec.get("rows", 0))
        cols = int(spec.get("cols", 0))
        if (refs and rows > 0 and cols > 0 and rows * cols == len(refs)
                and all(r in comps for r in refs)):
            covered.update(refs)
    if not covered:
        return False
    remaining = [r for r in comps if r not in covered]
    return all(len(comps[r].pads) <= 2 for r in remaining)


def place_array_leaves(
    comps: dict[str, Component], arrays: list[dict], cfg: dict
) -> tuple[set[str], bool]:
    """Grid-place every array whose members are all present in ``comps``.

    Returns ``(placed_refs, fully_handled)``:

    - ``placed_refs`` — array members that were grid-placed and ``locked``.
    - ``fully_handled`` — True when the only non-array parts left are simple
      two-terminal passives (decoupling/bulk caps), which are placed in a strip
      below the grid. The caller can then return immediately and skip the whole
      force/SA pipeline. When False, the array members are locked and the caller
      runs the normal pipeline for the remaining (non-trivial) components.
    """
    # The grid pitch must clear the legalizer's placement clearance. If grid
    # cells sit closer than that, the overlap resolver treats every adjacent
    # pair as overlapping and thrashes (O(n^2) escape passes that also never
    # reach a "legal" state). So the derived gap is at least the clearance.
    clearance = float(
        cfg.get("placement_clearance_mm", cfg.get("clearance_mm", 2.5))
    )
    gap = max(float(cfg.get("array_gap_mm", 0.6)), clearance)
    placed: set[str] = set()
    grid_bbox: tuple[float, float, float, float] | None = None

    for spec in arrays or []:
        refs = list(spec.get("refs", []))
        rows = int(spec.get("rows", 0))
        cols = int(spec.get("cols", 0))
        if not refs or rows <= 0 or cols <= 0 or rows * cols != len(refs):
            continue
        if not all(r in comps for r in refs):
            continue  # array belongs to a different leaf
        members = [comps[r] for r in refs]
        serpentine = bool(spec.get("serpentine", True))
        px, py = _pitch(members, spec, gap)
        x0, y0 = px, py  # keep coords positive; board-size search fits the leaf

        for idx, ref in enumerate(refs):
            row, col = divmod(idx, cols)
            if serpentine and row % 2 == 1:
                col = cols - 1 - col  # reverse on odd rows -> boustrophedon
            comp = comps[ref]
            _move(comp, x0 + col * px, y0 + row * py)
            comp.locked = True
            comp.array_member = True  # legalizer skips overlap-resolving the grid
            placed.add(ref)

        b = (x0 - px, y0 - py, x0 + (cols - 1) * px + px, y0 + (rows - 1) * py + py)
        grid_bbox = b if grid_bbox is None else (
            min(grid_bbox[0], b[0]),
            min(grid_bbox[1], b[1]),
            max(grid_bbox[2], b[2]),
            max(grid_bbox[3], b[3]),
        )

    if not placed:
        return placed, False

    remaining = [r for r in comps if r not in placed]
    # Array-dominated leaf: only simple two-terminal passives remain. Place them
    # in a strip below the grid and report the leaf fully handled so the caller
    # skips force/SA. (Two-terminal passives here are decoupling/bulk caps whose
    # nets are power/global, so exact position is not routing-critical.)
    if remaining and all(len(comps[r].pads) <= 2 for r in remaining):
        _place_strip(comps, remaining, grid_bbox, gap)
        return placed, True
    return placed, False


def _place_strip(
    comps: dict[str, Component],
    refs: list[str],
    grid_bbox: tuple[float, float, float, float],
    gap: float,
) -> None:
    """Lay the remaining two-terminal passives in a single row below the grid."""
    min_x, _min_y, _max_x, max_y = grid_bbox
    row_h = max(comps[r].height_mm for r in refs) + gap
    y = max_y + row_h
    x = min_x
    for r in refs:
        c = comps[r]
        _move(c, x + c.width_mm / 2.0, y)
        x += c.width_mm + gap
