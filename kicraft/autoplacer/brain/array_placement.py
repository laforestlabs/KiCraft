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

import math

from .geometry import rotate_component_in_place, rotate_vector
from .placement_utils import _update_pad_positions
from .types import Component, Point


def _orient_chain(comps: dict[str, Component], refs: list[str], cfg: dict) -> None:
    """Rotate each array member so its data-output pad faces the next member in
    the chain (DOUT -> next member's DIN; the last member's DIN faces the prev).

    On the serpentine grid every daisy-chain hop is then a short link straight
    across the narrow inter-component channel, which the deterministic array
    router (:mod:`array_router`) ties with a clean, repeating trace -- instead
    of a long diagonal across the LED bodies that the gridless autorouter
    abandons (a 1515 WS2812's DOUT and DIN sit on opposite corners, so an
    unrotated left-to-right row forces every hop across two parts). Orthogonal
    rotation only, so the member stays on its grid cell. Off via
    ``array_orient_chain=False``.
    """
    if not cfg.get("array_orient_chain", True):
        return
    from kicraft.design.models import is_power_or_ground_name

    def _data_pad(comp: Component, nbr: Component):
        # The pad whose (non-power) net is shared with the neighbour -- the data
        # link between the two parts (DOUT toward next, DIN toward prev).
        nbr_nets = {
            p.net for p in nbr.pads if p.net and not is_power_or_ground_name(p.net)
        }
        for p in comp.pads:
            if p.net and p.net in nbr_nets:
                return p
        return None

    n = len(refs)
    for i, ref in enumerate(refs):
        comp = comps[ref]
        nbr = comps[refs[i + 1]] if i + 1 < n else (comps[refs[i - 1]] if i else None)
        if nbr is None:
            continue
        pad = _data_pad(comp, nbr)
        if pad is None:
            continue
        ox, oy = pad.pos.x - comp.pos.x, pad.pos.y - comp.pos.y
        tx, ty = nbr.pos.x - comp.pos.x, nbr.pos.y - comp.pos.y
        tnorm = math.hypot(tx, ty) or 1.0
        best_delta, best_dot = 0.0, -2.0
        for delta in (0.0, 90.0, 180.0, 270.0):
            r = rotate_vector(Point(ox, oy), delta)
            rnorm = math.hypot(r.x, r.y) or 1.0
            dot = (r.x * tx + r.y * ty) / (rnorm * tnorm)
            if dot > best_dot:
                best_dot, best_delta = dot, delta
        if best_delta:
            rotate_component_in_place(comp, best_delta)


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

        # Orient each member so its DOUT faces the next chain member: turns every
        # daisy-chain hop into a short cross-channel link the array router can
        # tie cleanly (DOUT/DIN sit on opposite corners of the part, so the
        # unrotated grid routes badly). Done after all members of this spec land
        # so neighbour positions are known.
        _orient_chain(comps, refs, cfg)

        b = (x0 - px, y0 - py, x0 + (cols - 1) * px + px, y0 + (rows - 1) * py + py)
        grid_bbox = b if grid_bbox is None else (
            min(grid_bbox[0], b[0]),
            min(grid_bbox[1], b[1]),
            max(grid_bbox[2], b[2]),
            max(grid_bbox[3], b[3]),
        )

    if not placed:
        return placed, False

    # Per-LED decoupling companions: 2-pad passives whose BOTH nets are
    # power/ground (a decap -- not a signal part like a series data resistor).
    # Pack them in a compact LOCKED block directly below the grid so they stay
    # adjacent to the array (the GND/power pour ties them) instead of being
    # scattered by force/SA + the grid-escape pass (Step 9.3) into a wide sprawl
    # that bloats the board and strands GND. On a per-LED-decap matrix this is
    # ~one cap per member, so the eviction would otherwise move all of them.
    if grid_bbox is not None and cfg.get("array_colocate_decaps", True):
        from kicraft.design.models import is_power_or_ground_name

        def _is_decap(c: Component) -> bool:
            if len(c.pads) != 2:
                return False
            nets = {p.net for p in c.pads if p.net}
            return bool(nets) and all(is_power_or_ground_name(n) for n in nets)

        decaps = [
            r for r, c in comps.items()
            if r not in placed and not getattr(c, "locked", False) and _is_decap(c)
        ]
        if decaps:
            decaps.sort(key=_ref_sort_key)
            bx0, bx1, by1 = _place_companion_block(comps, decaps, grid_bbox, gap)
            placed.update(decaps)
            for r in decaps:
                comps[r].locked = True
            # Extend the grid bbox so any later strip (R1 etc.) drops BELOW the
            # cap block, not on top of it.
            grid_bbox = (grid_bbox[0], grid_bbox[1],
                         max(grid_bbox[2], bx1), by1)

    remaining = [r for r in comps if r not in placed]
    # Array-dominated leaf: only simple two-terminal passives remain. Place them
    # in a strip below the grid and report the leaf fully handled so the caller
    # skips force/SA. (Two-terminal passives here are decoupling/bulk caps whose
    # nets are power/global, so exact position is not routing-critical.)
    if remaining and all(len(comps[r].pads) <= 2 for r in remaining):
        _place_strip(comps, remaining, grid_bbox, gap)
        return placed, True
    return placed, False


def _ref_sort_key(ref: str) -> tuple[str, int]:
    """Sort refs by prefix then numeric suffix (C2 before C10)."""
    digits = "".join(ch for ch in ref if ch.isdigit())
    prefix = "".join(ch for ch in ref if not ch.isdigit())
    return (prefix, int(digits) if digits else 0)


def _place_companion_block(
    comps: dict[str, Component],
    refs: list[str],
    grid_bbox: tuple[float, float, float, float],
    gap: float,
) -> tuple[float, float, float]:
    """Pack companion passives (per-LED decaps) in a compact grid block directly
    below the array, as wide as the array. Returns the block's ``(min_x, max_x,
    max_y)``. Members are NOT marked ``array_member`` (they are companions, not
    chain members); the caller locks them."""
    min_x, _min_y, max_x, max_y = grid_bbox
    grid_w = max_x - min_x
    cw = max(comps[r].width_mm for r in refs) + gap
    ch = max(comps[r].height_mm for r in refs) + gap
    cols = max(1, int(grid_w // cw) or 1)  # match the array width
    x0 = min_x + cw / 2.0
    y0 = max_y + ch  # first row just below the grid
    bx1 = min_x
    by1 = y0
    for i, r in enumerate(refs):
        col, row = i % cols, i // cols
        x = x0 + col * cw
        y = y0 + row * ch
        _move(comps[r], x, y)
        bx1 = max(bx1, x + cw / 2.0)
        by1 = max(by1, y + ch / 2.0)
    return (min_x, bx1, by1)


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
