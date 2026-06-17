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


def _orient_array_grid(
    comps: dict[str, Component],
    refs: list[str],
    rows: int,
    cols: int,
    serpentine: bool,
    cfg: dict,
) -> None:
    """Give every array member a UNIFORM per-row rotation so the grid is a clean
    repeating pattern (easy to place + assemble) and the data chain routes
    deterministically.

    A WS2812-class part has DOUT and DIN on opposite diagonal corners, so the
    single rotation that points DOUT along a row's data-flow direction also
    points DIN back toward the previous member. We pick ONE such rotation per
    row, never per member -- so the array is at most two distinct rotations, not
    the four-way scatter a per-member "face the next neighbour" orientation
    produces (which is a placement and assembly nightmare and makes the routing
    non-repeating):

    - serpentine grid: even rows flow +x, odd rows flow -x, so odd rows are the
      even-row rotation + 180 (the standard alternate-row-flipped matrix). Every
      hop -- in-row and at the row turn -- is then short and repeats.
    - non-serpentine grid: every row flows +x, so ALL members share ONE rotation
      (the simplest pick-and-place); the longer end-of-row -> start-of-next-row
      return hop is left to :mod:`array_router` / the autorouter.

    Orthogonal rotation only, so each member stays on its grid cell. Off via
    ``array_orient_chain=False`` (members keep their incoming rotation).
    """
    if not cfg.get("array_orient_chain", True) or cols <= 0:
        return
    from kicraft.design.models import is_power_or_ground_name

    def _shared_data_pad(a: Component, b: Component):
        # The pad of ``a`` whose non-power net is shared with ``b`` -- the data
        # link (DOUT toward the next member, DIN toward the previous one).
        b_nets = {p.net for p in b.pads if p.net and not is_power_or_ground_name(p.net)}
        for p in a.pads:
            if p.net and p.net in b_nets:
                return p
        return None

    n = len(refs)
    # Representative DOUT pad (shared with the next chain member) and the part's
    # rotation when we read it -- so we can recover the INTRINSIC pad offset (at
    # absolute rotation 0) and then choose absolute per-row target rotations. We
    # rotate every member TO a target (never by a per-member delta), so each row
    # is exactly ONE rotation regardless of how members arrive, and the
    # serpentine flip is an exact 180.
    rep = rep_pad = None
    for i in range(n - 1):
        p = _shared_data_pad(comps[refs[i]], comps[refs[i + 1]])
        if p is not None:
            rep, rep_pad = comps[refs[i]], p
            break
    if rep_pad is None:
        return
    cur = Point(rep_pad.pos.x - rep.pos.x, rep_pad.pos.y - rep.pos.y)
    intrinsic = rotate_vector(cur, -rep.rotation)  # DOUT offset at abs rotation 0

    # Even-row target: the absolute rotation pointing DOUT along +x (the row's
    # data-flow direction); the diagonal DOUT/DIN corners leave a 2-fold tie that
    # we break to the smallest rotation so the choice is deterministic.
    r_even, best_key = 0.0, None
    for R in (0.0, 90.0, 180.0, 270.0):
        r = rotate_vector(intrinsic, R)
        rn = math.hypot(r.x, r.y) or 1.0
        key = (-(r.x) / rn, R)
        if best_key is None or key < best_key:
            best_key, r_even = key, R
    r_odd = (r_even + 180.0) % 360.0

    for i, ref in enumerate(refs):
        comp = comps[ref]
        target = r_odd if (serpentine and (i // cols) % 2 == 1) else r_even
        delta = (target - comp.rotation) % 360.0
        if delta:
            rotate_component_in_place(comp, delta)


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


def array_companion_refs(
    comps: dict[str, Component], arrays: list[dict]
) -> list[str]:
    """Refs of the per-array decoupling companions in this leaf.

    A companion is a 2-pad part whose BOTH nets are power/ground (a decap, not a
    signal part like a series data resistor) that rides alongside a *fully
    present* array grid. Returns ``[]`` when no array is present, so a plain
    decap leaf (no grid) is left entirely to the normal pipeline -- we only claim
    companions when there is an array to colocate them with.

    Shared by :func:`place_array_leaves` (which places + tags them) and
    ``leaf_geometry.repair_leaf_placement_legality`` (which must re-tag them
    after a board reload, since the ``array_member`` exemption flag does not
    survive serialize). Deterministic, sorted in ref order.
    """
    present_array_refs: set[str] = set()
    for spec in arrays or []:
        refs = list(spec.get("refs", []))
        rows = int(spec.get("rows", 0))
        cols = int(spec.get("cols", 0))
        if (refs and rows > 0 and cols > 0 and rows * cols == len(refs)
                and all(r in comps for r in refs)):
            present_array_refs.update(refs)
    if not present_array_refs:
        return []
    from kicraft.design.models import is_power_or_ground_name

    out: list[str] = []
    for ref, c in comps.items():
        if ref in present_array_refs or len(c.pads) != 2:
            continue
        nets = {p.net for p in c.pads if p.net}
        if nets and all(is_power_or_ground_name(n) for n in nets):
            out.append(ref)
    out.sort(key=_ref_sort_key)
    return out


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


def _grid_member_bbox(grid: dict) -> tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) of a placed grid's member pad bodies."""
    xs = [c.x for c in grid["centers"]]
    ys = [c.y for c in grid["centers"]]
    hw, hh = grid["led_w"] / 2.0, grid["led_h"] / 2.0
    return (min(xs) - hw, min(ys) - hh, max(xs) + hw, max(ys) + hh)


def _assert_grids_disjoint(grids: list[dict]) -> None:
    """Fail loudly if two array grids were placed over the same coordinates.

    Every grid currently starts at the same origin (``x0, y0 = px, py``), so two
    arrays on one leaf land on top of each other -- each member of grid B sits on
    a member of grid A, and B's pads then block A's inter-member routing (the
    KC-NZXXEE decap-array signature). Layer 1 removes the usual offender (a decap
    array), but any future two-array leaf would silently produce a broken board.
    A grid pair overlapping by more than half the smaller grid's footprint is a
    contradiction, not a layout -- raise rather than route garbage.
    """
    for a in range(len(grids)):
        for b in range(a + 1, len(grids)):
            ax1, ay1, ax2, ay2 = _grid_member_bbox(grids[a])
            bx1, by1, bx2, by2 = _grid_member_bbox(grids[b])
            ox = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            oy = max(0.0, min(ay2, by2) - max(ay1, by1))
            overlap = ox * oy
            if overlap <= 0:
                continue
            area_a = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1e-9, (bx2 - bx1) * (by2 - by1))
            if overlap > 0.5 * min(area_a, area_b):
                ga, gb = grids[a]["refs"], grids[b]["refs"]
                raise ValueError(
                    "array grids overlap on this leaf -- two ArraySpecs were "
                    "placed on the same coordinates "
                    f"({ga[0]}..{ga[-1]} and {gb[0]}..{gb[-1]}, "
                    f"overlap {overlap:.1f}mm^2). Each grid is laid from the same "
                    "origin, so they co-locate and block each other's routing. "
                    "If one is a decoupling-cap array it should be dropped at "
                    "synthesis (drop_decap_only_arrays); a genuine multi-array "
                    "leaf needs distinct grid origins."
                )


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
    grids: list[dict] = []  # per-array geometry for adaptive decap colocation

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

        # Give the grid a uniform per-row rotation (≤2 distinct angles) so it is
        # a clean repeating pattern the array router ties cleanly. Done after all
        # members land so the data-flow direction per row is known.
        _orient_array_grid(comps, refs, rows, cols, serpentine, cfg)

        b = (x0 - px, y0 - py, x0 + (cols - 1) * px + px, y0 + (rows - 1) * py + py)
        grid_bbox = b if grid_bbox is None else (
            min(grid_bbox[0], b[0]),
            min(grid_bbox[1], b[1]),
            max(grid_bbox[2], b[2]),
            max(grid_bbox[3], b[3]),
        )
        grids.append({
            "refs": refs, "px": px, "py": py, "rows": rows, "cols": cols,
            "led_w": max(c.width_mm for c in members),
            "led_h": max(c.height_mm for c in members),
            # member centre per chain index, read AFTER placement (serpentine
            # fill + per-row rotation already applied).
            "centers": [Point(comps[r].pos.x, comps[r].pos.y) for r in refs],
        })

    if not placed:
        return placed, False

    # Belt-and-suspenders: no two grids may occupy the same coordinates.
    _assert_grids_disjoint(grids)

    # Per-LED decoupling companions: 2-pad passives whose BOTH nets are
    # power/ground (a decap -- not a signal part like a series data resistor).
    # Place each ADJACENT to the LED it serves (the GND/power pour + the short
    # neighbour route then tie it) instead of scattering them via force/SA + the
    # grid-escape pass into a wide sprawl, OR packing them in a tall block far
    # below the grid that runs off the leaf outline (the KC-BUCJZ4 rc6 / KC-NESCCB
    # overlap bugs). See _place_companion_decaps for the adaptive beside/edge rule.
    if grid_bbox is not None and cfg.get("array_colocate_decaps", True):
        decaps = [
            r for r in array_companion_refs(comps, arrays)
            if r not in placed and not getattr(comps[r], "locked", False)
        ]
        if decaps:
            grid_bbox = _place_companion_decaps(
                comps, decaps, grids, grid_bbox, cfg
            )
            placed.update(decaps)
            for r in decaps:
                comps[r].locked = True
                # Beside-grid companions share the array's clearance exemption so
                # the legalizer leaves them at the tight pitch (their real
                # spacing is the netclass copper clearance, not the 2.5 mm
                # assembly clearance the grid itself is exempt from).
                comps[r].array_member = True

    # Re-base the framed cluster into the positive quadrant. The perimeter ring
    # frames the grid on ALL four edges, so its top/left caps land ABOVE/LEFT of
    # the grid origin -- at negative coordinates. The leaf board-size search grows
    # the fitted Edge.Cuts from the origin into +x/+y only (the grid deliberately
    # starts at x0,y0=px,py to keep coords positive, see above), so a
    # negative-coordinate pad falls OUTSIDE that outline and the leaf legality gate
    # rejects the placement every round -- 0 leaves, no parent, rc6 (KC-93X3X3:
    # bulk C1 took the bottom edge, pushing decap C2 onto the top edge at y=-0.8
    # above a y=0 outline). A uniform shift keeps the grid rigid and the data chain
    # unchanged, and is a no-op for the common cases (no companions, beside-LED, or
    # a bottom/right-only ring) where the bbox already starts at the origin.
    if grid_bbox is not None:
        shift_x = -min(0.0, grid_bbox[0])
        shift_y = -min(0.0, grid_bbox[1])
        if shift_x or shift_y:
            for r in placed:
                c = comps[r]
                _move(c, c.pos.x + shift_x, c.pos.y + shift_y)
            grid_bbox = (grid_bbox[0] + shift_x, grid_bbox[1] + shift_y,
                         grid_bbox[2] + shift_x, grid_bbox[3] + shift_y)

    remaining = [r for r in comps if r not in placed]
    # Pure array leaf: nothing but the grid. It is fully placed -- report handled
    # so the caller SKIPS force/SA entirely. (Falling through to force/SA here is
    # a latent bug: the members are locked, but SA refine still rotates them and,
    # when the pitch is tighter than the legalizer clearance, the overlap
    # resolver scatters the grid -- which only stayed put at looser pitches by
    # luck. A grid needs no optimizing.)
    if not remaining:
        return placed, True
    # Array-dominated leaf: only simple two-terminal passives remain. Place them
    # in a strip below the grid and report the leaf fully handled so the caller
    # skips force/SA. (Two-terminal passives here are decoupling/bulk caps whose
    # nets are power/global, so exact position is not routing-critical.)
    if all(len(comps[r].pads) <= 2 for r in remaining):
        _place_strip(comps, remaining, grid_bbox, gap)
        return placed, True
    return placed, False


def _ref_sort_key(ref: str) -> tuple[str, int]:
    """Sort refs by prefix then numeric suffix (C2 before C10)."""
    digits = "".join(ch for ch in ref if ch.isdigit())
    prefix = "".join(ch for ch in ref if not ch.isdigit())
    return (prefix, int(digits) if digits else 0)


def _place_companion_decaps(
    comps: dict[str, Component],
    refs: list[str],
    grids: list[dict],
    grid_bbox: tuple[float, float, float, float],
    cfg: dict,
) -> tuple[float, float, float, float]:
    """Adaptively place per-LED decoupling companions and return the (possibly
    extended) grid bbox.

    DEFAULT -- beside each LED: for a single 2-D grid with at most one decap per
    member, drop decap *k* into the inter-row channel directly beside the array
    member at chain index *k* (the channel the *in-row* data hops do NOT use), so
    the cap sits right next to the LED power pads it decouples and stays inside
    the grid bbox (no outline overflow). Companions are tagged ``array_member`` by
    the caller, so they share the grid's clearance exemption and pack at the real
    copper clearance rather than the 2.5 mm placement clearance.

    FALLBACK -- perimeter ring: when the part does not fit beside (gap too tight),
    the grid is 1-D, there are several arrays, or there are more decaps than
    members, lay the decaps as a SINGLE-FILE ring around all four edges of the
    grid (caps on the vertical edges rotated 90 so the ring stays one component
    deep). Still adjacent (pour-tied), still legal -- and a tidy frame around the
    array instead of the old amateurish multi-row block hanging off one edge.
    """
    comp_gap = float(cfg.get("array_companion_gap_mm", 0.3))
    min_x, min_y, max_x, max_y = grid_bbox

    # Beside-each-LED is only well-defined for a single 2-D grid (cols > 1, so the
    # in-row hops run horizontally and the vertical channel is free for caps) with
    # no more decaps than members.
    grid = grids[0] if len(grids) == 1 else None
    if grid is not None and grid["cols"] > 1 and len(refs) <= len(grid["refs"]):
        py, led_h = grid["py"], grid["led_h"]
        cap_h = max(comps[r].height_mm for r in refs)
        cap_w = max(comps[r].width_mm for r in refs)
        # Vertical channel below each LED must hold the cap with copper clearance
        # to the LEDs above and below, and the cap must not be wider than the
        # column pitch (else neighbouring caps collide horizontally).
        fits_y = (py - led_h) >= cap_h + 2.0 * comp_gap
        fits_x = grid["px"] >= cap_w + comp_gap
        if fits_y and fits_x:
            centers = grid["centers"]
            for k, r in enumerate(refs):
                c = centers[k]
                _move(comps[r], c.x, c.y + py / 2.0)  # mid inter-row channel
            print(
                f"  Array decaps: {len(refs)} placed beside-LED in the inter-row "
                f"channel (py={py:.2f} led_h={led_h:.2f} cap_h={cap_h:.2f})"
            )
            # Caps land within the existing grid bbox (last row's caps sit in the
            # bottom margin already covered by grid_bbox); bbox unchanged.
            return grid_bbox

    # FALLBACK: a single-file ring of caps around all four edges of the grid.
    grid_w, grid_h = max_x - min_x, max_y - min_y
    edges = ("bottom", "right", "top", "left")  # clockwise from bottom-left
    edge_len = {"bottom": grid_w, "right": grid_h, "top": grid_w, "left": grid_h}
    active = [e for e in edges if edge_len[e] > 1e-6] or ["bottom"]

    # Spread the caps over the active edges proportional to edge length
    # (largest-remainder, deterministic; refs already ref-sorted). A 1-D grid has
    # one zero-length axis so its caps fall on the two long edges only.
    n = len(refs)
    total_len = sum(edge_len[e] for e in active)
    counts = {e: 0 for e in edges}
    if total_len > 1e-6:
        alloc = {e: n * edge_len[e] / total_len for e in active}
        for e in active:
            counts[e] = int(alloc[e])
        leftover = sorted(active, key=lambda e: alloc[e] - int(alloc[e]), reverse=True)
        for i in range(n - sum(counts.values())):
            counts[leftover[i % len(leftover)]] += 1
    else:
        counts[active[0]] = n

    bx0, by0, bx1, by1 = min_x, min_y, max_x, max_y  # union bbox accumulator
    it = iter(refs)
    overcrowded: list[str] = []
    for edge in edges:
        k = counts[edge]
        if k <= 0:
            continue
        horizontal = edge in ("bottom", "top")
        span = grid_w if horizontal else grid_h
        base = min_x if horizontal else min_y
        demand = 0.0
        for j in range(k):
            r = next(it)
            w0, h0 = comps[r].width_mm, comps[r].height_mm  # capture BEFORE rotate
            demand += w0
            off = h0 / 2.0 + comp_gap
            t = base + ((j + 0.5) * span / k if span > 1e-6 else 0.0)
            if edge == "bottom":
                cx, cy = t, max_y + off
            elif edge == "top":
                cx, cy = t, min_y - off
            elif edge == "right":
                rotate_component_in_place(comps[r], 90.0)
                cx, cy = max_x + off, t
            else:  # left
                rotate_component_in_place(comps[r], 90.0)
                cx, cy = min_x - off, t
            _move(comps[r], cx, cy)
            # occupied AABB: verticals were rotated, so their axes swap.
            ew, eh = (w0, h0) if horizontal else (h0, w0)
            bx0, by0 = min(bx0, cx - ew / 2.0), min(by0, cy - eh / 2.0)
            bx1, by1 = max(bx1, cx + ew / 2.0), max(by1, cy + eh / 2.0)
        if span > 1e-6 and demand > span:
            overcrowded.append(edge)

    print(
        f"  Array decaps: {n} placed in a single-file perimeter ring "
        f"(bottom={counts['bottom']} right={counts['right']} "
        f"top={counts['top']} left={counts['left']}); beside-LED not feasible"
    )
    if overcrowded:
        print(
            f"  WARNING: array decap ring overcrowded on {', '.join(overcrowded)} "
            "-- caps may abut; the array is space-constrained at this pitch"
        )
    return (bx0, by0, bx1, by1)


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
