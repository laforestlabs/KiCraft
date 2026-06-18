"""Regression: a leaf board grows so ALL same-edge connectors fit flush in one line.

When several connectors share one edge (e.g. 4 screw terminals on the right), the
column/row they form can be longer than the leaf board. The old behaviour packed
from the edge and let the overflow run off the board; ``_shift_pads_inside`` then
pulled it back inside, the overlap resolver shoved it inboard into a 2nd
row/column, and that connector read as stranded inboard at compose (run_19 TB2 at
-17.62mm; the whole RELAY_OUTPUT leaf failed to place legally under the fast
engine). ``_pin_edge_components`` now grows the board along the edge's parallel
axis so the group fits in ONE flush line. Grow-only, and only on genuine overflow.
"""
from __future__ import annotations

from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.types import BoardState, Component, Layer, Pad, Point


def _conn(ref: str, w: float = 8.0, h: float = 10.0) -> Component:
    return Component(
        ref=ref, value="x", pos=Point(25.0, 12.0), rotation=0.0,
        layer=Layer.FRONT, width_mm=w, height_mm=h, kind="connector",
        pads=[Pad(ref=ref, pad_id="1", pos=Point(25.0, 12.0), net="", layer=Layer.FRONT)],
    )


def test_board_grows_so_all_right_edge_connectors_are_flush():
    # 4 connectors, 10 mm tall each, all zoned right -> need ~46 mm of column
    # (4*10 + 3*gap) but the board starts only 24 mm tall.
    comps = {f"TB{i}": _conn(f"TB{i}") for i in range(1, 5)}
    state = BoardState(
        components=dict(comps), board_outline=(Point(0.0, 0.0), Point(50.0, 24.0))
    )
    cfg = {
        "component_zones": {f"TB{i}": {"edge": "right"} for i in range(1, 5)},
        "connector_gap_mm": 2.0,
        "edge_margin_mm": 2.0,
    }
    solver = PlacementSolver(state, cfg, seed=0)
    solver._pin_edge_components(comps)

    tl, br = solver.state.board_outline
    # The board must have grown tall enough to fit the whole column in one line.
    assert (br.y - tl.y) >= 4 * 10.0 + 3 * 2.0, (
        f"board height {br.y - tl.y:.1f} did not grow to fit the 4-connector column"
    )
    # All four right edges coincide (one flush column, not a stranded 2nd column).
    right_edges = [c.pos.x + c.width_mm / 2.0 for c in comps.values()]
    assert max(right_edges) - min(right_edges) < 0.5, (
        f"connectors not co-aligned on the right edge: {sorted(right_edges)}"
    )
    # They stack along Y without overlapping (>= one body height apart).
    ys = sorted(c.pos.y for c in comps.values())
    assert all(b - a >= 9.5 for a, b in zip(ys, ys[1:])), f"connectors overlap in Y: {ys}"


def test_board_does_not_grow_when_group_already_fits():
    # A single connector on a roomy board: no overflow, so no growth.
    comps = {"J1": _conn("J1")}
    state = BoardState(
        components=dict(comps), board_outline=(Point(0.0, 0.0), Point(50.0, 40.0))
    )
    cfg = {"component_zones": {"J1": {"edge": "right"}}, "edge_margin_mm": 2.0}
    solver = PlacementSolver(state, cfg, seed=0)
    solver._pin_edge_components(comps)

    tl, br = solver.state.board_outline
    assert (br.x, br.y) == (50.0, 40.0), "board grew when the group already fit"
