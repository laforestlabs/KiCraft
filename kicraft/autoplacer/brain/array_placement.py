"""Programmatic pattern placement for matrix/array leaves.

Some leaves are regular arrays of identical components — e.g. a 10x20
addressable-LED matrix or a 12-LED ring. Throwing 200 identical, daisy-chained
parts at the force-directed + simulated-annealing solver in
:mod:`placement_solver` does not converge: the sibling-grouping pass and the
power-net cliques explode into a near-complete graph, and the per-iteration
crossover scorer becomes the bottleneck. Such leaves carry an explicit array
hint from synthesis (``autoplacer.json`` -> solver ``cfg["arrays"]``, a list of
``{refs, pattern, rows, cols, pitch_mm, serpentine, radius_mm,
start_angle_deg}`` dicts). We lay their members out deterministically — a
serpentine grid (``pattern: "grid"``, the default) or an evenly-spaced circle
(``pattern: "ring"``) — and skip the optimizer entirely.

Members are listed in data-chain order, so the fill order (boustrophedon for
grids, circular for rings) keeps consecutive members physically adjacent — the
DOUT->DIN routes stay short.
"""

from __future__ import annotations

import math

from .geometry import rotate_component_in_place, rotate_vector
from .placement_utils import _update_pad_positions
from .types import Component, Point, opening_rotation_for_edge


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


def _orient_ring(comps: dict[str, Component], refs: list[str], cfg: dict) -> None:
    """Rotate every ring member so its DOUT points along the chain direction
    (the chord toward the next member).

    Unlike the grid's deliberate ≤2-rotation uniformity, a ring's canonical
    construction IS one rotation per member — every real LED-ring board turns
    each LED with the circle, which keeps every DOUT->DIN hop an identical
    short chord and the assembly pattern rotationally repeating. Uses the same
    intrinsic-DOUT recovery as :func:`_orient_array_grid`; off via the same
    ``array_orient_chain=False``.
    """
    if not cfg.get("array_orient_chain", True):
        return
    from kicraft.design.models import is_power_or_ground_name

    def _shared_data_pad(a: Component, b: Component):
        b_nets = {p.net for p in b.pads if p.net and not is_power_or_ground_name(p.net)}
        for p in a.pads:
            if p.net and p.net in b_nets:
                return p
        return None

    n = len(refs)
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
    phi_intrinsic = math.degrees(math.atan2(intrinsic.y, intrinsic.x))

    for i, ref in enumerate(refs):
        comp = comps[ref]
        nxt = comps[refs[(i + 1) % n]] if i < n - 1 else comps[refs[i - 1]]
        if i < n - 1:
            d = Point(nxt.pos.x - comp.pos.x, nxt.pos.y - comp.pos.y)
        else:
            # Last member: keep the rotational pattern going (direction FROM
            # the previous member), since its DOUT leaves the chain.
            d = Point(comp.pos.x - nxt.pos.x, comp.pos.y - nxt.pos.y)
        phi_d = math.degrees(math.atan2(d.y, d.x))
        # rotate_vector is KiCad-CW: rotating by R moves a vector's math-angle
        # by -R, so the R that points DOUT (at phi_intrinsic) along phi_d is:
        target = (phi_intrinsic - phi_d) % 360.0
        delta = (target - comp.rotation) % 360.0
        if delta > 1e-6:
            rotate_component_in_place(comp, delta)


def _move(comp: Component, x: float, y: float) -> None:
    """Move a component's body center to (x, y), carrying its pads along.

    Rotation is left unchanged, so ``_update_pad_positions`` is a pure
    translation (the canonical move pattern used across the solver).
    """
    old = Point(comp.pos.x, comp.pos.y)
    comp.pos = Point(x, y)
    _update_pad_positions(comp, old, comp.rotation)


def _pitch(
    members: list[Component],
    spec: dict,
    derived_gap: float,
    explicit_gap: float,
) -> tuple[float, float]:
    """Grid pitch, flooring an explicit request only when bodies overlap."""
    requested = float(spec.get("pitch_mm") or 0.0)
    gap = explicit_gap if requested else derived_gap
    px = max(c.width_mm for c in members) + gap
    py = max(c.height_mm for c in members) + gap
    if requested:
        return max(requested, px), max(requested, py)
    return px, py


_BOARD_EDGES = frozenset({"left", "right", "top", "bottom"})


def _same_edge_connector_bank(
    refs: list[str], comps: dict[str, Component], cfg: dict
) -> str | None:
    """Return the common explicit edge for an all-connector grid, if any."""
    zones = cfg.get("component_zones")
    if not isinstance(zones, dict) or not all(comps[ref].kind == "connector" for ref in refs):
        return None
    edges: list[str] = []
    for ref in refs:
        zone = zones.get(ref)
        if not isinstance(zone, dict) or zone.get("edge") not in _BOARD_EDGES:
            return None
        edges.append(str(zone["edge"]))
    return edges[0] if len(set(edges)) == 1 else None


def _place_edge_connector_bank(
    comps: dict[str, Component],
    refs: list[str],
    edge: str,
    spec: dict,
    derived_gap: float,
    explicit_gap: float,
) -> tuple[dict, tuple[float, float, float, float]]:
    """Orient and physically pack a one-dimensional connector edge bank."""
    members = [comps[ref] for ref in refs]
    for comp in members:
        if comp.opening_direction is None:
            continue
        target = opening_rotation_for_edge(comp.opening_direction, comp.layer, edge)
        delta = (target - comp.rotation) % 360.0
        if delta > 1e-6:
            rotate_component_in_place(comp, delta)

    requested = float(spec.get("pitch_mm") or 0.0)
    gap = explicit_gap if requested else derived_gap
    horizontal = edge in ("top", "bottom")
    extents: list[tuple[float, float, float, float]] = []
    for comp in members:
        tl, br = comp.physical_bbox()
        extents.append(
            (
                tl.x - comp.pos.x,
                tl.y - comp.pos.y,
                br.x - comp.pos.x,
                br.y - comp.pos.y,
            )
        )

    tangent_centers: list[float] = []
    previous_center = previous_max = 0.0
    for index, extent in enumerate(extents):
        tangent_min = extent[0] if horizontal else extent[1]
        tangent_max = extent[2] if horizontal else extent[3]
        if index == 0:
            center = gap - tangent_min
        else:
            center = max(
                previous_center + requested,
                previous_max + gap - tangent_min,
            )
        tangent_centers.append(center)
        previous_center = center
        previous_max = center + tangent_max

    normal_mins = [extent[1] if horizontal else extent[0] for extent in extents]
    normal_maxs = [extent[3] if horizontal else extent[2] for extent in extents]
    if edge in ("top", "left"):
        outward_target = gap
        normal_centers = [outward_target - normal_min for normal_min in normal_mins]
    else:
        outward_target = gap + max(
            normal_max - normal_min for normal_min, normal_max in zip(normal_mins, normal_maxs)
        )
        normal_centers = [outward_target - normal_max for normal_max in normal_maxs]

    for ref, tangent, normal in zip(refs, tangent_centers, normal_centers):
        comp = comps[ref]
        x, y = (tangent, normal) if horizontal else (normal, tangent)
        _move(comp, x, y)
        comp.locked = True
        comp.array_member = True

    physical = [comp.physical_bbox() for comp in members]
    member_bbox = (
        min(tl.x for tl, _ in physical),
        min(tl.y for tl, _ in physical),
        max(br.x for _, br in physical),
        max(br.y for _, br in physical),
    )
    deltas = [
        tangent_centers[index] - tangent_centers[index - 1]
        for index in range(1, len(tangent_centers))
    ]
    tangent_pitch = max(deltas, default=requested)
    grid = {
        "refs": refs,
        "pattern": "grid",
        "edge_connector_bank": edge,
        "px": tangent_pitch if horizontal else 0.0,
        "py": 0.0 if horizontal else tangent_pitch,
        "rows": 1 if horizontal else len(refs),
        "cols": len(refs) if horizontal else 1,
        "led_w": max(comp.width_mm for comp in members),
        "led_h": max(comp.height_mm for comp in members),
        "centers": [Point(comps[ref].pos.x, comps[ref].pos.y) for ref in refs],
        "member_bbox": member_bbox,
    }
    return grid, member_bbox


def _present_member_refs(spec: dict, comps: dict[str, Component]) -> list[str] | None:
    """Validated member refs of a FULLY-present array spec, else None.

    The single spec-shape predicate shared by placement, companion detection
    and the fully-array test, mirroring the ArraySpec model validator:
    ``grid`` needs ``rows*cols == len(refs)``; ``ring`` needs >= 3 refs.
    A spec whose members live on a different leaf returns None (partial
    arrays are never placed here).
    """
    refs = list(spec.get("refs", []))
    if not refs or not all(r in comps for r in refs):
        return None
    pattern = str(spec.get("pattern", "grid") or "grid").lower()
    if pattern == "ring":
        return refs if len(refs) >= 3 else None
    rows = int(spec.get("rows", 0) or 0)
    cols = int(spec.get("cols", 0) or 0)
    return refs if rows > 0 and cols > 0 and rows * cols == len(refs) else None


def array_companion_refs(comps: dict[str, Component], arrays: list[dict]) -> list[str]:
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
        refs = _present_member_refs(spec, comps)
        if refs:
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
        refs = _present_member_refs(spec, comps)
        if refs:
            covered.update(refs)
    if not covered:
        return False
    remaining = [r for r in comps if r not in covered]
    return all(len(comps[r].pads) <= 2 for r in remaining)


def _grid_member_bbox(grid: dict) -> tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) of an array's occupied member bodies."""
    if "member_bbox" in grid:
        return grid["member_bbox"]
    xs = [c.x for c in grid["centers"]]
    ys = [c.y for c in grid["centers"]]
    if grid.get("pattern") == "ring":
        hw = hh = grid["led_diag"] / 2.0
    else:
        hw, hh = grid["led_w"] / 2.0, grid["led_h"] / 2.0
    return (min(xs) - hw, min(ys) - hh, max(xs) + hw, max(ys) + hh)


def _assert_grids_disjoint(grids: list[dict]) -> None:
    """Require every placed array's occupied member bbox to be disjoint."""
    for a in range(len(grids)):
        for b in range(a + 1, len(grids)):
            ax1, ay1, ax2, ay2 = _grid_member_bbox(grids[a])
            bx1, by1, bx2, by2 = _grid_member_bbox(grids[b])
            ox = min(ax2, bx2) - max(ax1, bx1)
            oy = min(ay2, by2) - max(ay1, by1)
            if ox <= 1e-9 or oy <= 1e-9:
                continue
            ga, gb = grids[a]["refs"], grids[b]["refs"]
            raise ValueError(
                "array grids overlap after packing "
                f"({ga[0]}..{ga[-1]} and {gb[0]}..{gb[-1]}, "
                f"overlap {ox * oy:.1f}mm^2)"
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
    clearance = float(cfg.get("placement_clearance_mm", cfg.get("clearance_mm", 2.5)))
    gap = max(float(cfg.get("array_gap_mm", 0.6)), clearance)
    explicit_pitch_gap = float(cfg.get("array_gap_mm", 0.6))
    placed: set[str] = set()
    grid_bbox: tuple[float, float, float, float] | None = None
    grids: list[dict] = []  # per-array geometry for adaptive decap colocation
    packed_member_max_x: float | None = None

    def _pack_grid(
        grid: dict,
        layout_bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Shelf-pack one placed array and merge its layout bbox."""
        nonlocal grid_bbox, packed_member_max_x
        member_bbox = _grid_member_bbox(grid)
        shift_x = 0.0
        if packed_member_max_x is not None:
            shift_x = packed_member_max_x + clearance - member_bbox[0]
            for ref in grid["refs"]:
                comp = comps[ref]
                _move(comp, comp.pos.x + shift_x, comp.pos.y)
            grid["centers"] = [Point(center.x + shift_x, center.y) for center in grid["centers"]]
            if grid.get("center") is not None:
                center = grid["center"]
                grid["center"] = Point(center.x + shift_x, center.y)
            member_bbox = (
                member_bbox[0] + shift_x,
                member_bbox[1],
                member_bbox[2] + shift_x,
                member_bbox[3],
            )
        if "member_bbox" in grid:
            grid["member_bbox"] = member_bbox
        packed_member_max_x = member_bbox[2]
        placed_bbox = (
            layout_bbox[0] + shift_x,
            layout_bbox[1],
            layout_bbox[2] + shift_x,
            layout_bbox[3],
        )
        grid_bbox = (
            placed_bbox
            if grid_bbox is None
            else (
                min(grid_bbox[0], placed_bbox[0]),
                min(grid_bbox[1], placed_bbox[1]),
                max(grid_bbox[2], placed_bbox[2]),
                max(grid_bbox[3], placed_bbox[3]),
            )
        )
        return placed_bbox

    for spec in arrays or []:
        refs = _present_member_refs(spec, comps)
        if refs is None:
            continue  # malformed spec, or array belongs to a different leaf
        members = [comps[r] for r in refs]
        pattern = str(spec.get("pattern", "grid") or "grid").lower()

        if pattern == "ring":
            # Evenly-spaced circle in chain order. The chord between
            # neighbours must clear the member DIAGONAL (members get a
            # per-member tangent rotation, so their worst-case extent along
            # the ring is the diagonal, not one axis).
            n = len(refs)
            diag = max(math.hypot(c.width_mm, c.height_mm) for c in members)
            chord = float(spec.get("pitch_mm") or 0.0) or (diag + gap)
            min_r = chord / (2.0 * math.sin(math.pi / n))
            r_ring = max(float(spec.get("radius_mm") or 0.0), min_r)
            start = float(spec.get("start_angle_deg") or 0.0)
            half = diag / 2.0
            cx = cy = r_ring + half + gap  # positive quadrant, like the grid
            angles = [math.radians(start + 360.0 * i / n) for i in range(n)]
            for idx, ref in enumerate(refs):
                comp = comps[ref]
                _move(
                    comp,
                    cx + r_ring * math.cos(angles[idx]),
                    cy + r_ring * math.sin(angles[idx]),
                )
                comp.locked = True
                comp.array_member = True
                placed.add(ref)
            _orient_ring(comps, refs, cfg)
            b = (cx - r_ring - half, cy - r_ring - half, cx + r_ring + half, cy + r_ring + half)
            grid = {
                "refs": refs,
                "pattern": "ring",
                "px": chord,
                "py": chord,
                "rows": 1,
                "cols": 0,
                "led_w": max(c.width_mm for c in members),
                "led_h": max(c.height_mm for c in members),
                "led_diag": diag,
                "center": Point(cx, cy),
                "radius": r_ring,
                "angles": angles,
                "centers": [Point(comps[r].pos.x, comps[r].pos.y) for r in refs],
            }
            _pack_grid(grid, b)
            grids.append(grid)
            continue

        rows = int(spec.get("rows", 0) or 0)
        cols = int(spec.get("cols", 0) or 0)
        serpentine = bool(spec.get("serpentine", True))
        edge_bank = _same_edge_connector_bank(refs, comps, cfg)
        if edge_bank is not None:
            if rows > 1 and cols > 1:
                raise ValueError(
                    "edge_connector_array_not_one_dimensional:"
                    f"{','.join(refs)}@{edge_bank}({rows}x{cols})"
                )
            grid, b = _place_edge_connector_bank(
                comps,
                refs,
                edge_bank,
                spec,
                gap,
                explicit_pitch_gap,
            )
            placed.update(refs)
            _pack_grid(grid, b)
            grids.append(grid)
            continue
        px, py = _pitch(members, spec, gap, explicit_pitch_gap)
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
        grid = {
            "refs": refs,
            "pattern": "grid",
            "px": px,
            "py": py,
            "rows": rows,
            "cols": cols,
            "led_w": max(c.width_mm for c in members),
            "led_h": max(c.height_mm for c in members),
            # member centre per chain index, read AFTER placement (serpentine
            # fill + per-row rotation already applied).
            "centers": [Point(comps[r].pos.x, comps[r].pos.y) for r in refs],
        }
        _pack_grid(grid, b)
        grids.append(grid)

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
            r
            for r in array_companion_refs(comps, arrays)
            if r not in placed and not getattr(comps[r], "locked", False)
        ]
        if decaps:
            grid_bbox = _place_companion_decaps(comps, decaps, grids, grid_bbox, cfg)
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
            grid_bbox = (
                grid_bbox[0] + shift_x,
                grid_bbox[1] + shift_y,
                grid_bbox[2] + shift_x,
                grid_bbox[3] + shift_y,
            )

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

    grid = grids[0] if len(grids) == 1 else None

    # RING -- into the BAND: decap *k* sits at the gap-midpoint angle between
    # members k and k+1, pad axis radial with its power pad ON the +5V bus
    # chord (see array_router.array_ring_power_specs), GND pad outward. This
    # is canonical ring construction -- a real LED-ring board keeps the middle
    # clear (often physically cut out) -- and it is what keeps the interior
    # hole nestable (docs/plans/shaped-compose-leaf-nesting.md, PR-N5). Falls
    # back to the single-file perimeter ring when a gap is too tight. The
    # legacy radially-inward placement stays behind
    # ``array_ring_band_decaps: False`` (it deliberately parks companions in
    # the interior, which nesting invalidates).
    if grid is not None and grid.get("pattern") == "ring" and len(refs) <= len(grid["refs"]):
        if cfg.get("array_ring_band_decaps", True):
            if _place_ring_band_decaps(comps, refs, grid, comp_gap):
                return grid_bbox
            # too tight / nets unidentifiable -> perimeter fallback below
        else:
            n = len(grid["refs"])
            cap_w = max(comps[r].width_mm for r in refs)
            cap_h = max(comps[r].height_mm for r in refs)
            cap_diag = math.hypot(cap_w, cap_h)
            r_in = grid["radius"] - grid["led_diag"] / 2.0 - cap_diag / 2.0 - comp_gap
            chord_in = 2.0 * r_in * math.sin(math.pi / n) if r_in > 0 else 0.0
            if r_in > cap_diag / 2.0 and chord_in >= cap_diag + comp_gap:
                c0 = grid["center"]
                for k, r in enumerate(refs):
                    a = grid["angles"][k]
                    member = comps[grid["refs"][k]]
                    delta = (member.rotation - comps[r].rotation) % 360.0
                    if delta > 1e-6:
                        rotate_component_in_place(comps[r], delta)
                    _move(
                        comps[r],
                        c0.x + r_in * math.cos(a),
                        c0.y + r_in * math.sin(a),
                    )
                print(
                    f"  Array decaps: {len(refs)} placed radially inside the ring "
                    f"(r_in={r_in:.2f} of r={grid['radius']:.2f})"
                )
                return grid_bbox

    # Beside-each-LED is only well-defined for a single 2-D grid (cols > 1, so the
    # in-row hops run horizontally and the vertical channel is free for caps) with
    # no more decaps than members.
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


def _is_ground_net(net: str | None) -> bool:
    """Ground-ness by the shared synthesis patterns (GND/AGND/DGND/_GND)."""
    from kicraft.design.models import GND_NET_PATTERNS

    stripped = (net or "").lstrip("/")
    return any(pat.search(stripped) for pat in GND_NET_PATTERNS)


def _rotated_aabb(w: float, h: float, rot_deg: float) -> tuple[float, float]:
    """AABB dims of a w x h body at rotation rot_deg."""
    t = math.radians(rot_deg % 360.0)
    return (
        abs(w * math.cos(t)) + abs(h * math.sin(t)),
        abs(w * math.sin(t)) + abs(h * math.cos(t)),
    )


def _place_ring_band_decaps(
    comps: dict[str, Component],
    refs: list[str],
    grid: dict,
    comp_gap: float,
) -> bool:
    """Place ring companions in the BAND: decap *k* at the gap-midpoint angle
    between members k and k+1, pad axis RADIAL with the power pad inward at
    the +5V bus-chord sagitta radius (where the member-to-member power chord
    passes), GND pad outward toward the pour side.

    The whole plan is computed and feasibility-checked (rotated-AABB
    clearance against both adjacent members, and the cap must stay inside
    the ring bbox) BEFORE anything moves; any failure returns False with
    ``comps`` untouched so the caller can fall back to the perimeter ring.
    """
    members = grid["refs"]
    n = len(members)
    c0 = grid["center"]
    angles = grid["angles"]
    bbox_radius = grid["radius"] + grid["led_diag"] / 2.0

    plan: list[tuple[str, float, float, float]] = []
    tap_radii: list[float] = []
    for k, ref in enumerate(refs):
        cap = comps[ref]
        pwr = [p for p in cap.pads if p.net and not _is_ground_net(p.net)]
        gnd = [p for p in cap.pads if p.net and _is_ground_net(p.net)]
        if len(pwr) != 1 or len(gnd) != 1:
            return False  # not a rail+ground pair we can orient
        # Bus radius = where the members' pads on THIS decap's rail sit; the
        # chord between adjacent pads dips to its sagitta radius at the gap
        # midpoint -- put the cap's power pad exactly there so the stamped
        # bus (array_ring_power_specs) runs member pad -> cap pad -> member
        # pad in two straight, foreign-pad-free segments.
        radii = [
            math.hypot(p.pos.x - c0.x, p.pos.y - c0.y)
            for m in members
            for p in comps[m].pads
            if p.net == pwr[0].net
        ]
        if not radii:
            return False  # members don't carry this rail -- not a ring decap
        r_tap = (sum(radii) / len(radii)) * math.cos(math.pi / n)
        a_mid = angles[k] + math.pi / n

        # Absolute rotation pointing the gnd->pwr pad axis INWARD (toward
        # the ring centre). rotate_vector is KiCad-CW: rotating by R moves a
        # vector's math-angle by -R (same recovery as _orient_ring).
        axis = rotate_vector(
            Point(pwr[0].pos.x - gnd[0].pos.x, pwr[0].pos.y - gnd[0].pos.y),
            -cap.rotation,
        )
        phi_axis = math.degrees(math.atan2(axis.y, axis.x))
        target_rot = (phi_axis - (math.degrees(a_mid) + 180.0)) % 360.0

        # Body centre sits outward of the power pad by the pad's offset.
        p_off = rotate_vector(
            Point(pwr[0].pos.x - cap.pos.x, pwr[0].pos.y - cap.pos.y),
            -cap.rotation,
        )
        r_center = r_tap + math.hypot(p_off.x, p_off.y)
        cx = c0.x + r_center * math.cos(a_mid)
        cy = c0.y + r_center * math.sin(a_mid)

        cap_w, cap_h = _rotated_aabb(cap.width_mm, cap.height_mm, target_rot)
        if r_center + math.hypot(cap_w, cap_h) / 2.0 > bbox_radius:
            return False  # cap would poke past the ring bbox
        for m in (members[k], members[(k + 1) % n]):
            mc = comps[m]
            mw, mh = _rotated_aabb(mc.width_mm, mc.height_mm, mc.rotation)
            if (
                abs(cx - mc.pos.x) < (cap_w + mw) / 2.0 + comp_gap
                and abs(cy - mc.pos.y) < (cap_h + mh) / 2.0 + comp_gap
            ):
                return False  # gap too tight at this pitch
        plan.append((ref, target_rot, cx, cy))
        tap_radii.append(r_tap)

    for ref, target_rot, cx, cy in plan:
        cap = comps[ref]
        delta = (target_rot - cap.rotation) % 360.0
        if delta > 1e-6:
            rotate_component_in_place(cap, delta)
        _move(cap, cx, cy)
    print(
        f"  Array decaps: {len(plan)} placed in the ring band at gap "
        f"midpoints (power-pad tap r={min(tap_radii):.2f} "
        f"of r={grid['radius']:.2f}; interior stays clear)"
    )
    return True


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
