"""Tests for programmatic grid placement of matrix/array leaves.

Regression: a 200-LED array fed to the force/SA solver never converged
(pegged a core for hours). Array leaves carry an explicit hint and are now
grid-placed deterministically, skipping the optimizer. See
``kicraft/autoplacer/brain/array_placement.py``.
"""
from __future__ import annotations

from kicraft.autoplacer.brain.array_placement import place_array_leaves
from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.types import BoardState, Component, Layer, Pad, Point


def _comp(ref: str, value: str, w: float, h: float, npads: int) -> Component:
    pads = [
        Pad(ref=ref, pad_id=str(i + 1), pos=Point(-0.5 + i * 0.3, 0.0),
            net="N", layer=Layer.FRONT)
        for i in range(npads)
    ]
    return Component(
        ref=ref, value=value, pos=Point(0.0, 0.0), rotation=0.0,
        layer=Layer.FRONT, width_mm=w, height_mm=h, pads=pads,
    )


def test_grid_geometry_serpentine_lock_and_strip() -> None:
    comps = {f"D{i}": _comp(f"D{i}", "LED", 1.5, 1.5, 4) for i in range(1, 7)}
    comps.update({f"C{i}": _comp(f"C{i}", "100nF", 1.0, 0.5, 2) for i in range(1, 3)})
    d1_pad0 = (comps["D1"].pads[0].pos.x, comps["D1"].pads[0].pos.y)

    arrays = [{"refs": [f"D{i}" for i in range(1, 7)], "rows": 2, "cols": 3,
               "serpentine": True}]
    # clearance 0 so the gap isn't floored to the placement clearance.
    placed, fully = place_array_leaves(
        comps, arrays, {"array_gap_mm": 0.5, "placement_clearance_mm": 0.0}
    )

    assert placed == {f"D{i}" for i in range(1, 7)}
    assert fully is True
    # pitch = max(1.5,1.5)+0.5 = 2.0; origin at (2,2)
    assert (comps["D1"].pos.x, comps["D1"].pos.y) == (2.0, 2.0)
    assert (comps["D3"].pos.x, comps["D3"].pos.y) == (6.0, 2.0)
    # serpentine: row 1 reverses -> D4 sits directly below D3 (chain-adjacent)
    assert (comps["D4"].pos.x, comps["D4"].pos.y) == (6.0, 4.0)
    assert (comps["D6"].pos.x, comps["D6"].pos.y) == (2.0, 4.0)
    assert all(comps[f"D{i}"].locked for i in range(1, 7))
    assert all(comps[f"D{i}"].array_member for i in range(1, 7))
    # pads translate with the body (D1 moved (0,0)->(2,2))
    assert comps["D1"].pads[0].pos.x == d1_pad0[0] + 2.0
    assert comps["D1"].pads[0].pos.y == d1_pad0[1] + 2.0
    # the two simple caps were placed in a strip below the grid
    assert comps["C1"].pos.y > comps["D6"].pos.y
    assert comps["C1"].pos != comps["C2"].pos


def test_explicit_pitch_is_honored() -> None:
    comps = {f"D{i}": _comp(f"D{i}", "LED", 1.5, 1.5, 2) for i in range(1, 5)}
    arrays = [{"refs": ["D1", "D2", "D3", "D4"], "rows": 2, "cols": 2,
               "pitch_mm": 5.0, "serpentine": False}]
    place_array_leaves(comps, arrays, {})
    assert (comps["D2"].pos.x - comps["D1"].pos.x) == 5.0  # column pitch
    assert (comps["D3"].pos.y - comps["D1"].pos.y) == 5.0  # row pitch


def test_mixed_leaf_locks_array_but_not_fully_handled() -> None:
    # Array + a non-trivial (>2-pad) part -> members locked, but the leaf is
    # NOT fully handled, so the caller still runs the normal pipeline.
    comps = {f"D{i}": _comp(f"D{i}", "LED", 1.5, 1.5, 4) for i in range(1, 5)}
    comps["U1"] = _comp("U1", "DRIVER", 5.0, 5.0, 8)
    arrays = [{"refs": ["D1", "D2", "D3", "D4"], "rows": 2, "cols": 2}]
    placed, fully = place_array_leaves(comps, arrays, {})
    assert placed == {"D1", "D2", "D3", "D4"}
    assert fully is False
    assert all(comps[f"D{i}"].locked for i in range(1, 5))
    assert comps["U1"].locked is False


def test_array_for_another_leaf_is_skipped() -> None:
    # Hint references refs not present in this leaf -> nothing placed.
    comps = {"R1": _comp("R1", "10k", 1.0, 0.5, 2)}
    arrays = [{"refs": ["D1", "D2"], "rows": 1, "cols": 2}]
    placed, fully = place_array_leaves(comps, arrays, {})
    assert placed == set()
    assert fully is False


def test_solver_skips_pipeline_for_array_leaf() -> None:
    # The hang guard: a 50-LED array + 2 caps must return a grid via the early
    # path in PlacementSolver.solve() without entering force/SA.
    comps = {f"D{i}": _comp(f"D{i}", "LED", 1.5, 1.5, 4) for i in range(1, 51)}
    comps.update({f"C{i}": _comp(f"C{i}", "100nF", 1.0, 0.5, 2) for i in range(1, 3)})
    state = BoardState(components=comps, nets={})
    cfg = {
        "arrays": [{"refs": [f"D{i}" for i in range(1, 51)], "rows": 5, "cols": 10}],
        "array_gap_mm": 0.5,
    }
    out = PlacementSolver(state, cfg, seed=0).solve()
    assert len(out) == 52
    assert all(out[f"D{i}"].locked for i in range(1, 51))
    # all members on distinct grid positions
    positions = {(round(out[f"D{i}"].pos.x, 3), round(out[f"D{i}"].pos.y, 3))
                 for i in range(1, 51)}
    assert len(positions) == 50


def test_grid_pitch_respects_clearance() -> None:
    # The derived gap is floored to the placement clearance, so the grid is
    # legal by construction (a tighter gap made the legalizer thrash).
    comps = {f"D{i}": _comp(f"D{i}", "LED", 1.5, 1.5, 2) for i in range(1, 5)}
    arrays = [{"refs": ["D1", "D2", "D3", "D4"], "rows": 2, "cols": 2}]
    place_array_leaves(
        comps, arrays, {"array_gap_mm": 0.5, "placement_clearance_mm": 3.0}
    )
    # gap floored to 3.0 -> pitch = courtyard(1.5) + 3.0 = 4.5
    assert (comps["D2"].pos.x - comps["D1"].pos.x) == 4.5


def test_legalizer_skips_array_members() -> None:
    # A tight grid's clearance-zone overlaps must NOT be flagged illegal nor
    # escaped by the legalizer (that thrash was the 200-LED routing hang).
    comps = {f"D{i}": _comp(f"D{i}", "LED", 2.0, 2.0, 2) for i in range(1, 5)}
    place_array_leaves(
        comps, [{"refs": ["D1", "D2", "D3", "D4"], "rows": 2, "cols": 2,
                 "pitch_mm": 2.2}], {},  # 2.2 < courtyard+clearance -> "overlaps"
    )
    state = BoardState(components=comps, nets={})
    solver = PlacementSolver(state, {"placement_clearance_mm": 2.5}, seed=0)
    assert solver.legality_diagnostics(comps)["legal"] is True
    before = {r: (c.pos.x, c.pos.y) for r, c in comps.items()}
    solver._resolve_overlaps(comps)
    assert {r: (c.pos.x, c.pos.y) for r, c in comps.items()} == before


def test_leaf_is_fully_array_gate() -> None:
    from kicraft.autoplacer.brain.array_placement import leaf_is_fully_array
    comps = {f"D{i}": _comp(f"D{i}", "LED", 1.5, 1.5, 4) for i in range(1, 5)}
    comps["C1"] = _comp("C1", "100nF", 1.0, 0.5, 2)
    arrays = [{"refs": ["D1", "D2", "D3", "D4"], "rows": 2, "cols": 2}]
    assert leaf_is_fully_array(comps, arrays) is True          # grid + passive
    comps["U1"] = _comp("U1", "IC", 5.0, 5.0, 8)
    assert leaf_is_fully_array(comps, arrays) is False         # extra non-passive
    assert leaf_is_fully_array({"R1": _comp("R1", "10k", 1.0, 0.5, 2)}, []) is False


def test_deterministic_route_signature() -> None:
    # The route cache reuses a routed board only when placement + routing knobs
    # match; the freerouting *timeout* must not affect the key.
    from kicraft.autoplacer.brain.leaf_routing import _deterministic_route_signature
    comps = {f"D{i}": _comp(f"D{i}", "LED", 1.5, 1.5, 4) for i in range(1, 5)}
    place_array_leaves(
        comps, [{"refs": ["D1", "D2", "D3", "D4"], "rows": 2, "cols": 2,
                 "pitch_mm": 3.0}], {})
    state = BoardState(components=comps, nets={})
    cfg = {"freerouting_max_passes": 4, "freerouting_timeout_s": 60}
    sig = _deterministic_route_signature(state, cfg, "freerouting-1.9.0.jar")
    # timeout-only change -> same key (does not affect copper)
    assert sig == _deterministic_route_signature(
        state, {**cfg, "freerouting_timeout_s": 999}, "freerouting-1.9.0.jar")
    # routing-knob change -> different key
    assert sig != _deterministic_route_signature(
        state, {**cfg, "freerouting_max_passes": 12}, "freerouting-1.9.0.jar")
    # placement change -> different key
    comps["D1"].pos = Point(comps["D1"].pos.x + 1.0, comps["D1"].pos.y)
    assert sig != _deterministic_route_signature(state, cfg, "freerouting-1.9.0.jar")
