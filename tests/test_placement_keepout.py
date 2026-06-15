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
