"""PlacementSolver antenna keep-out push-out, owner exemption, legality, edges."""
from __future__ import annotations

from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    KeepoutRect,
    Layer,
    Pad,
    Point,
)


def _comp(ref, x, y, *, w=3.0, h=3.0, locked=False):
    return Component(
        ref=ref,
        value="",
        pos=Point(x, y),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=w,
        height_mm=h,
        locked=locked,
    )


def _comp_with_pad(ref, x, y, *, w=2.0, h=2.0, kind="passive", locked=False):
    """A 1-pad component whose pad copper bbox is ``pos ± (w, h)/2``."""
    pad = Pad(ref=ref, pad_id="1", pos=Point(x, y), net="N", layer=Layer.FRONT,
              size_mm=Point(w, h))
    return Component(
        ref=ref, value="", pos=Point(x, y), rotation=0.0, layer=Layer.FRONT,
        width_mm=w, height_mm=h, pads=[pad], kind=kind, locked=locked,
    )


def test_companion_pushed_behind_connector_pads_left():
    # J1 (connector, locked) pad copper x[5,7]; R8 (passive) pad x[3,5] OUTBOARD
    # of J1. R8 must be pushed so its left pad face >= J1 left pad face + clearance;
    # J1 unmoved; a far passive untouched.
    conn = _comp_with_pad("J1", 6.0, 20.0, w=2.0, h=8.0, kind="connector", locked=True)
    victim = _comp_with_pad("R8", 4.0, 20.0, w=2.0, h=1.0)   # pad x[3,5]
    far = _comp_with_pad("R9", 30.0, 20.0, w=1.0, h=1.0)
    state = BoardState(
        components={"J1": conn, "R8": victim, "R9": far},
        board_outline=(Point(0, 0), Point(40, 40)), keepout_rects=[],
    )
    solver = _solver(state)
    solver._edge_pinned_groups = {"left": ["J1"]}
    assert solver._clamp_companions_inboard_of_connectors(state.components, 0.5) == 1
    assert min(p.bbox()[0].x for p in victim.pads) >= 5.5 - 1e-6
    assert (conn.pos.x, conn.pos.y) == (6.0, 20.0)
    assert (far.pos.x, far.pos.y) == (30.0, 20.0)


def test_companion_clamp_noop_when_already_inboard():
    conn = _comp_with_pad("J1", 6.0, 20.0, w=2.0, h=8.0, kind="connector", locked=True)
    inboard = _comp_with_pad("C1", 20.0, 20.0, w=1.0, h=1.0)  # well inboard
    state = BoardState(
        components={"J1": conn, "C1": inboard},
        board_outline=(Point(0, 0), Point(40, 40)), keepout_rects=[],
    )
    solver = _solver(state)
    solver._edge_pinned_groups = {"left": ["J1"]}
    assert solver._clamp_companions_inboard_of_connectors(state.components, 0.5) == 0
    assert (inboard.pos.x, inboard.pos.y) == (20.0, 20.0)


def _overlaps(comp, kr) -> bool:
    c_tl, c_br = comp.bbox(0.0)
    ox = min(c_br.x, kr.br.x) - max(c_tl.x, kr.tl.x)
    oy = min(c_br.y, kr.br.y) - max(c_tl.y, kr.tl.y)
    return ox > 0.0 and oy > 0.0


def _on_board(comp, state) -> bool:
    (tl, br) = state.board_outline
    c_tl, c_br = comp.bbox(0.0)
    return c_tl.x >= tl.x and c_br.x <= br.x and c_tl.y >= tl.y and c_br.y <= br.y


def _solver(state):
    return PlacementSolver(state, {"placement_clearance_mm": 0.0})


def test_resolve_keepout_pushes_unlocked_part_out_owner_exempt():
    kr = KeepoutRect(tl=Point(5, 5), br=Point(23, 15), owner_ref="U1")
    # owner overlaps its own keep-out but is UNLOCKED -> exempt by owner_ref,
    # proving exemption is not merely the locked-skip.
    owner = _comp("U1", 14, 10, w=18, h=10, locked=False)
    victim = _comp("SW1", 8, 7)  # inside the rect
    far = _comp("R1", 45, 45)  # well outside
    state = BoardState(
        components={"U1": owner, "SW1": victim, "R1": far},
        board_outline=(Point(0, 0), Point(60, 60)),
        keepout_rects=[kr],
    )
    assert _overlaps(victim, kr)

    moved = _solver(state)._resolve_keepout_rects(state.components)

    assert moved == 1
    assert not _overlaps(victim, kr), "victim still inside keep-out"
    assert _on_board(victim, state)
    assert (owner.pos.x, owner.pos.y) == (14, 10), "owner must not move"
    assert (far.pos.x, far.pos.y) == (45, 45), "non-overlapping part must not move"


def test_legality_diagnostics_flags_keepout_overlap():
    kr = KeepoutRect(tl=Point(5, 5), br=Point(23, 15), owner_ref="U1", source="inject")
    owner = _comp("U1", 14, 10, w=18, h=10, locked=True)
    victim = _comp("SW1", 8, 7)
    state = BoardState(
        components={"U1": owner, "SW1": victim},
        board_outline=(Point(0, 0), Point(60, 60)),
        keepout_rects=[kr],
    )
    diag = _solver(state).legality_diagnostics(state.components)
    assert diag["keepout_overlap_count"] == 1
    entry = diag["keepout_overlaps"][0]
    assert entry["ref"] == "SW1" and entry["owner"] == "U1"
    assert entry["source"] == "inject"
    # owner is exempt -> not reported
    assert all(e["ref"] != "U1" for e in diag["keepout_overlaps"])


def test_keepout_rect_tracks_moved_owner():
    # The rect was sampled at extraction with U1 at origin (14,10). U1 has since
    # moved +10 in x; because the keep-out is rigidly attached to U1, the
    # effective rect must shift with it. A part clear of the STALE rect (x[5,23])
    # but inside the TRACKED rect (x[15,33]) must be pushed out.
    kr = KeepoutRect(
        tl=Point(5, 5), br=Point(23, 15), owner_ref="U1",
        owner_origin=Point(14, 10),
    )
    owner = _comp("U1", 24, 10, w=18, h=10)  # +10 in x from owner_origin
    victim = _comp("SW1", 28, 10, w=2, h=2)  # bbox x[27,29]: clear of stale, in tracked
    state = BoardState(
        components={"U1": owner, "SW1": victim},
        board_outline=(Point(0, 0), Point(60, 60)),
        keepout_rects=[kr],
    )
    solver = _solver(state)
    assert not _overlaps(victim, kr), "victim should be clear of the STALE rect"
    r_tl, r_br = solver._keepout_rect_now(kr, state.components)
    assert (r_tl.x, r_br.x) == (15, 33), "rect must track the owner's +10 x move"

    moved = solver._resolve_keepout_rects(state.components)
    assert moved == 1
    assert _on_board(victim, state)
    c_tl, c_br = victim.bbox(0.0)
    ox = min(c_br.x, r_br.x) - max(c_tl.x, r_tl.x)
    oy = min(c_br.y, r_br.y) - max(c_tl.y, r_tl.y)
    assert not (ox > 0 and oy > 0), "victim still inside the tracked keep-out"


def test_clear_pinned_connector_slides_along_edge():
    # J2 (USB-C, right-edge pinned, locked) lands in U3's antenna keep-out.
    # _push_out_of_rect skips it (locked) and it can't leave its edge, so the
    # edge-slide pass must move it ALONG the right edge until clear and bake the
    # cleared spot into _pinned_targets so the closing restore keeps it.
    kr = KeepoutRect(tl=Point(28, 10), br=Point(40, 20), owner_ref="U3")
    conn = _comp("J2", 36, 14, w=6, h=8, locked=True)  # bbox x[33,39] y[10,18]
    state = BoardState(
        components={"J2": conn},
        board_outline=(Point(0, 0), Point(40, 60)),
        keepout_rects=[kr],
    )
    assert _overlaps(conn, kr)
    solver = _solver(state)
    solver._edge_pinned_groups = {"right": ["J2"]}
    solver._pinned_targets = {"J2": Point(36, 14)}

    moved = solver._clear_pinned_from_keepouts(state.components)

    assert moved == 1
    assert not _overlaps(conn, kr), "connector still inside the keep-out"
    assert _on_board(conn, state)
    assert abs(conn.pos.x - 36.0) < 1e-6, "must stay flush to its right edge (x fixed)"
    assert solver._pinned_targets["J2"] == Point(conn.pos.x, conn.pos.y)


def test_clear_pinned_connector_noop_when_already_clear():
    kr = KeepoutRect(tl=Point(5, 5), br=Point(15, 15), owner_ref="U3")
    conn = _comp("J2", 36, 40, w=6, h=8, locked=True)  # far from kr
    state = BoardState(
        components={"J2": conn},
        board_outline=(Point(0, 0), Point(40, 60)),
        keepout_rects=[kr],
    )
    solver = _solver(state)
    solver._edge_pinned_groups = {"right": ["J2"]}
    solver._pinned_targets = {"J2": Point(36, 40)}
    assert solver._clear_pinned_from_keepouts(state.components) == 0
    assert (conn.pos.x, conn.pos.y) == (36, 40)


def test_clear_pinned_connector_owner_is_exempt():
    # The connector that OWNS a keep-out is never slid out of its own rect.
    kr = KeepoutRect(tl=Point(33, 10), br=Point(40, 20), owner_ref="J2")
    conn = _comp("J2", 36, 14, w=6, h=8, locked=True)
    state = BoardState(
        components={"J2": conn},
        board_outline=(Point(0, 0), Point(40, 60)),
        keepout_rects=[kr],
    )
    solver = _solver(state)
    solver._edge_pinned_groups = {"right": ["J2"]}
    solver._pinned_targets = {"J2": Point(36, 14)}
    assert solver._clear_pinned_from_keepouts(state.components) == 0
    assert (conn.pos.x, conn.pos.y) == (36, 14)


def test_pin_then_clear_keepout_integration():
    # Full path: the real config-driven _pin_edge_components edge-pins J2 (and
    # populates _edge_pinned_groups), then _clear_pinned_from_keepouts slides it
    # out of a neighbour's antenna keep-out while keeping it flush to its edge.
    j2 = Component(
        ref="J2", value="USB-C", pos=Point(20, 30), rotation=0.0,
        layer=Layer.FRONT, width_mm=6, height_mm=8, kind="connector",
        pads=[Pad(ref="J2", pad_id="1", pos=Point(20, 30), net="N",
                  layer=Layer.FRONT, size_mm=Point(1, 1))],
    )
    u3 = _comp("U3", 10, 30, w=10, h=10)  # the (interior) keep-out owner
    state = BoardState(
        components={"J2": j2, "U3": u3},
        board_outline=(Point(0, 0), Point(40, 60)),
        keepout_rects=[],
    )
    cfg = {
        "component_zones": {"J2": {"edge": "right"}},
        "edge_margin_mm": 2.0,
        "placement_clearance_mm": 0.0,
    }
    solver = PlacementSolver(state, cfg, seed=0)
    solver._pin_edge_components(state.components)
    assert "J2" in solver._edge_pinned_groups.get("right", []), "J2 not edge-pinned"

    # Plant U3's keep-out over where J2 actually pinned, upper half only so a
    # downward slide can clear it. owner_origin == U3's pos -> rect is in place.
    jx, jy = j2.pos.x, j2.pos.y
    kr = KeepoutRect(
        tl=Point(jx - 5, jy - 4), br=Point(jx + 5, jy + 1), owner_ref="U3",
        owner_origin=Point(u3.pos.x, u3.pos.y),
    )
    state.keepout_rects = [kr]
    assert _overlaps(j2, kr), "test setup: J2 should start inside the keep-out"

    moved = solver._clear_pinned_from_keepouts(state.components)
    assert moved == 1
    assert not _overlaps(j2, kr), "J2 still inside the keep-out after the clear pass"
    assert _on_board(j2, state)
    assert abs(j2.pos.x - jx) < 1e-6, "J2 must stay flush to its pinned right edge"


def test_keepout_straddling_board_edge_pushes_inboard():
    # Keep-out crosses the top board edge (antenna faces off-board). The
    # smallest exit (up) would push the part off the board; the solver must
    # instead pick the smallest exit that keeps it on-board (down).
    kr = KeepoutRect(tl=Point(20, -5), br=Point(40, 8), owner_ref="U1")
    victim = _comp("SW1", 30, 1, w=4, h=4)  # bbox (28,-1)-(32,3); up-exit is smallest
    state = BoardState(
        components={"SW1": victim},
        board_outline=(Point(0, 0), Point(60, 60)),
        keepout_rects=[kr],
    )
    assert _overlaps(victim, kr)

    moved = _solver(state)._resolve_keepout_rects(state.components)

    assert moved == 1
    assert not _overlaps(victim, kr)
    assert _on_board(victim, state), "part was pushed off the board edge"
    assert victim.pos.y > 1, "expected an inboard (downward) push, not off-edge"
