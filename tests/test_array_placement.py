"""Tests for programmatic grid placement of matrix/array leaves.

Regression: a 200-LED array fed to the force/SA solver never converged
(pegged a core for hours). Array leaves carry an explicit hint and are now
grid-placed deterministically, skipping the optimizer. See
``kicraft/autoplacer/brain/array_placement.py``.
"""
from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.array_placement import (
    _assert_grids_disjoint,
    place_array_leaves,
)
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
    # array_orient_chain off: this test pins grid geometry + the pure-translation
    # of pads; chain orientation (which rotates members) is covered separately.
    placed, fully = place_array_leaves(
        comps, arrays,
        {"array_gap_mm": 0.5, "placement_clearance_mm": 0.0,
         "array_orient_chain": False},
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


def _led(ref: str, dout_net: str, din_net: str) -> Component:
    """A 4-pad 1515-style LED: DOUT on the bottom-left corner, DIN top-right
    (the WS2812 layout where the chain runs across opposite corners)."""
    pads = [
        Pad(ref=ref, pad_id="1", pos=Point(-0.4, -0.4), net="+5V", layer=Layer.FRONT),
        Pad(ref=ref, pad_id="2", pos=Point(-0.4, +0.4), net=dout_net, layer=Layer.FRONT),
        Pad(ref=ref, pad_id="3", pos=Point(+0.4, +0.4), net="GND", layer=Layer.FRONT),
        Pad(ref=ref, pad_id="4", pos=Point(+0.4, -0.4), net=din_net, layer=Layer.FRONT),
    ]
    return Component(ref=ref, value="LED", pos=Point(0.0, 0.0), rotation=0.0,
                     layer=Layer.FRONT, width_mm=1.3, height_mm=1.3, pads=pads)


def test_chain_orientation_points_dout_at_next() -> None:
    # D1 -> D2 -> D3 in a row. Each LED is rotated so its DOUT pad (the net it
    # shares with the NEXT member) ends up on the side facing that member, so
    # the daisy-chain hop is a short cross-channel link instead of a diagonal
    # across both bodies. Without orientation DOUT sits on the far (left) side.
    comps = {
        "D1": _led("D1", "D1_DOUT", "DATA_IN"),
        "D2": _led("D2", "D2_DOUT", "D1_DOUT"),
        "D3": _led("D3", "D3_DOUT", "D2_DOUT"),
    }
    arrays = [{"refs": ["D1", "D2", "D3"], "rows": 1, "cols": 3,
               "pitch_mm": 3.0, "serpentine": True}]
    place_array_leaves(comps, arrays, {})  # orientation on by default

    def dout_pad(ref, net):
        return next(p for p in comps[ref].pads if p.net == net)

    # D1's DOUT must face D2 (to its right): pad.x on the +x side of the body.
    d1 = comps["D1"]
    assert dout_pad("D1", "D1_DOUT").pos.x > d1.pos.x
    # D2's DOUT must face D3 (also to the right).
    d2 = comps["D2"]
    assert dout_pad("D2", "D2_DOUT").pos.x > d2.pos.x
    # Members stay on their grid cells (rotation is in place).
    assert (d1.pos.x, d1.pos.y) == (3.0, 3.0)
    assert (comps["D3"].pos.x - d1.pos.x) == 6.0


def _grid(n: int) -> dict[str, Component]:
    """A daisy-chain of ``n`` WS2812-style LEDs (Dk_DOUT -> D(k+1)_DIN)."""
    comps: dict[str, Component] = {}
    prev = "DATA_IN"
    for i in range(1, n + 1):
        dout = f"D{i}_DOUT"
        comps[f"D{i}"] = _led(f"D{i}", dout, prev)
        prev = dout
    return comps


def test_serpentine_grid_uses_two_alternating_rotations() -> None:
    # A repeating matrix, not the old per-member 4-way scatter: every row is a
    # single rotation, and serpentine alternates it by 180 so the data flow
    # reverses cleanly row to row.
    comps = _grid(25)
    refs = [f"D{i}" for i in range(1, 26)]
    place_array_leaves(
        comps, [{"refs": refs, "rows": 5, "cols": 5,
                 "pitch_mm": 3.0, "serpentine": True}], {})
    assert len({comps[r].rotation for r in refs}) == 2
    for r in range(5):
        row = {comps[refs[r * 5 + c]].rotation for c in range(5)}
        assert len(row) == 1, f"row {r} not uniform: {row}"
    # adjacent rows differ by 180
    r0 = comps[refs[0]].rotation
    r1 = comps[refs[5]].rotation
    assert (r0 - r1) % 360 == 180


def test_pure_array_leaf_is_fully_handled() -> None:
    # A leaf that is ONLY the grid (no other parts) must report fully_handled so
    # solve() skips force/SA -- otherwise SA refine rotates + the legalizer
    # scatters the locked grid at a tight pitch.
    comps = _grid(25)
    refs = [f"D{i}" for i in range(1, 26)]
    placed, fully = place_array_leaves(
        comps, [{"refs": refs, "rows": 5, "cols": 5,
                 "pitch_mm": 3.0, "serpentine": True}], {})
    assert placed == set(refs)
    assert fully is True


def test_non_serpentine_grid_is_single_uniform_rotation() -> None:
    comps = _grid(25)
    refs = [f"D{i}" for i in range(1, 26)]
    place_array_leaves(
        comps, [{"refs": refs, "rows": 5, "cols": 5,
                 "pitch_mm": 3.0, "serpentine": False}], {})
    assert len({comps[r].rotation for r in refs}) == 1


def test_chain_orientation_can_be_disabled() -> None:
    comps = {"D1": _led("D1", "D1_DOUT", "DATA_IN"),
             "D2": _led("D2", "D2_DOUT", "D1_DOUT")}
    arrays = [{"refs": ["D1", "D2"], "rows": 1, "cols": 2, "pitch_mm": 3.0}]
    place_array_leaves(comps, arrays, {"array_orient_chain": False})
    # rotation untouched -> DOUT stays on its original (-x) side.
    assert comps["D1"].rotation == 0.0
    d1 = comps["D1"]
    assert next(p for p in d1.pads if p.net == "D1_DOUT").pos.x < d1.pos.x


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


def test_non_array_part_pushed_off_grid() -> None:
    # A part wider than the grid pitch dropped onto the locked array must be
    # pushed clear of the WHOLE grid (local overlap resolution can't escape a
    # dense grid -- each nudge lands it on the next cell). Regression for an LED
    # matrix's series resistor overlapping D1/D2 (leaf_pre_stamp_legality_repair).
    from kicraft.autoplacer.brain.types import BoardState, Point

    comps = {f"D{i}": _comp(f"D{i}", "LED", 1.5, 1.5, 4) for i in range(1, 10)}
    place_array_leaves(
        comps, [{"refs": [f"D{i}" for i in range(1, 10)], "rows": 3, "cols": 3,
                 "pitch_mm": 3.0}], {"array_orient_chain": False},
    )
    # grid spans ~ x,y in [1.5..7.5]; drop R1 (5mm wide) right on top of it.
    members = [c for c in comps.values() if getattr(c, "array_member", False)]
    gx = sum(c.pos.x for c in members) / len(members)
    gy = sum(c.pos.y for c in members) / len(members)
    r1 = Component(ref="R1", value="330", pos=Point(gx, gy), rotation=0.0,
                   layer=Layer.FRONT, width_mm=5.0, height_mm=2.0,
                   pads=[Pad(ref="R1", pad_id="1", pos=Point(gx - 2, gy), net="A",
                             layer=Layer.FRONT),
                         Pad(ref="R1", pad_id="2", pos=Point(gx + 2, gy), net="B",
                             layer=Layer.FRONT)])
    r1.body_center = Point(gx, gy)
    comps["R1"] = r1

    state = BoardState(components=comps, nets={})
    state.board_outline = (Point(-30.0, -30.0), Point(60.0, 60.0))  # room to escape
    solver = PlacementSolver(state, {"placement_clearance_mm": 0.5}, seed=0)
    moved = solver._resolve_array_grid(comps)
    assert moved >= 1, "R1 should have been pushed"

    # R1's bbox must no longer overlap any locked array member's bbox.
    r_tl, r_br = comps["R1"].bbox(0.0)
    for c in members:
        m_tl, m_br = c.bbox(0.0)
        ox = min(r_br.x, m_br.x) - max(r_tl.x, m_tl.x)
        oy = min(r_br.y, m_br.y) - max(r_tl.y, m_tl.y)
        assert not (ox > 0.0 and oy > 0.0), f"R1 still overlaps {c.ref}"


def _decap(ref: str, w: float = 1.0, h: float = 0.5) -> Component:
    """A 2-pad 100nF decoupling cap (both pads power/ground)."""
    return Component(
        ref=ref, value="100nF", pos=Point(0.0, 0.0), rotation=0.0,
        layer=Layer.FRONT, width_mm=w, height_mm=h,
        pads=[Pad(ref=ref, pad_id="1", pos=Point(-0.5, 0.0), net="+5V",
                  layer=Layer.FRONT),
              Pad(ref=ref, pad_id="2", pos=Point(0.5, 0.0), net="GND",
                  layer=Layer.FRONT)])


def test_per_led_decaps_beside_led_not_scattered() -> None:
    # Per-LED decoupling caps (2-pad, both nets power/ground) sit BESIDE the LED
    # they serve, in the inter-row channel -- adjacent (pour + short hop ties
    # them), within the grid bbox (no outline overflow), NOT scattered by
    # force/SA + grid-escape into a wide sprawl (the KC-FFFADA regression: 50
    # caps -> 192mm band) nor packed in a tall far-below block (KC-BUCJZ4 rc6).
    comps = {f"D{i}": _led(f"D{i}", f"D{i}_DOUT", "DATA_IN") for i in range(1, 5)}
    for i in range(1, 5):
        comps[f"C{i}"] = _decap(f"C{i}")
    # a series DATA resistor (NOT a decap: signal nets) must NOT be co-located
    comps["R1"] = Component(
        ref="R1", value="330", pos=Point(0.0, 0.0), rotation=0.0, layer=Layer.FRONT,
        width_mm=1.0, height_mm=0.5,
        pads=[Pad(ref="R1", pad_id="1", pos=Point(-0.5, 0.0), net="DATA",
                  layer=Layer.FRONT),
              Pad(ref="R1", pad_id="2", pos=Point(0.5, 0.0), net="DATA_IN",
                  layer=Layer.FRONT)])

    arrays = [{"refs": [f"D{i}" for i in range(1, 5)], "rows": 2, "cols": 2,
               "pitch_mm": 3.0}]
    placed, _ = place_array_leaves(comps, arrays, {})

    # placed, locked, and tagged array_member (so the legalizer's clearance gate
    # exempts them from the 2.5mm placement clearance the tight grid is exempt from)
    assert all(f"C{i}" in placed and comps[f"C{i}"].locked
               and comps[f"C{i}"].array_member for i in range(1, 5))
    assert "R1" not in placed, "signal resistor must not be co-located as a decap"

    led_pos = {i: comps[f"D{i}"].pos for i in range(1, 5)}
    grid_min_x = min(p.x for p in led_pos.values())
    grid_max_x = max(p.x for p in led_pos.values())
    grid_max_y = max(p.y for p in led_pos.values())
    for i in range(1, 5):
        cap = comps[f"C{i}"].pos
        # beside SOME LED: within one pitch of a member centre (not flung away)
        nearest = min(((cap.x - p.x) ** 2 + (cap.y - p.y) ** 2) ** 0.5
                      for p in led_pos.values())
        assert nearest <= 3.0, f"C{i} not beside any LED (nearest {nearest:.2f}mm)"
        # in the vertical channel: aligned to its LED's column, offset in y
        assert any(abs(cap.x - p.x) < 0.01 for p in led_pos.values())
        # compact: stays within the grid's x-span and just below its bottom row
        assert grid_min_x - 0.01 <= cap.x <= grid_max_x + 0.01
        assert cap.y <= grid_max_y + 3.0  # inside the grid bbox bottom margin


def _ring_layout(pitch_mm: float = 1.6):
    """Run the too-tight fallback and return (comps, placed)."""
    comps = {f"D{i}": _led(f"D{i}", f"D{i}_DOUT", "DATA_IN") for i in range(1, 5)}
    for i in range(1, 5):
        comps[f"C{i}"] = _decap(f"C{i}")
    # pitch 1.6 vs 1.3 LED -> only 0.3mm gap, no room for a 0.5mm-tall cap beside
    arrays = [{"refs": [f"D{i}" for i in range(1, 5)], "rows": 2, "cols": 2,
               "pitch_mm": pitch_mm}]
    placed, _ = place_array_leaves(comps, arrays, {})
    return comps, placed


def test_decaps_fall_back_to_perimeter_ring_when_too_tight() -> None:
    # When the cap cannot fit beside the LED (pitch barely clears the LED itself),
    # fall back to a SINGLE-FILE ring around all four edges of the grid -- a tidy
    # frame, NOT the old multi-row block hanging off the bottom edge.
    comps, placed = _ring_layout()

    assert all(f"C{i}" in placed and comps[f"C{i}"].locked for i in range(1, 5))
    leds = [comps[f"D{i}"].pos for i in range(1, 5)]
    lx0, lx1 = min(p.x for p in leds), max(p.x for p in leds)
    ly0, ly1 = min(p.y for p in leds), max(p.y for p in leds)
    caps = [comps[f"C{i}"].pos for i in range(1, 5)]

    # Each cap sits just outside the LED block on exactly one side (a ring, not a
    # block): classify by which side it falls on.
    sides = {"below": 0, "above": 0, "right": 0, "left": 0}
    for c in caps:
        on = [c.y > ly1, c.y < ly0, c.x > lx1, c.x < lx0]
        assert sum(on) == 1, f"cap {c} not cleanly outside one edge"
        sides[["below", "above", "right", "left"][on.index(True)]] += 1
    # 4 caps over a square grid -> one per edge: the ring touches all four sides.
    assert all(v >= 1 for v in sides.values()), f"not a 4-edge ring: {sides}"

    # No two caps land on top of each other.
    pts = [(round(c.x, 3), round(c.y, 3)) for c in caps]
    assert len(set(pts)) == 4, "ring caps must not overlap"


def test_perimeter_ring_is_deterministic() -> None:
    a, _ = _ring_layout()
    b, _ = _ring_layout()
    for i in range(1, 5):
        assert a[f"C{i}"].pos.x == b[f"C{i}"].pos.x
        assert a[f"C{i}"].pos.y == b[f"C{i}"].pos.y


def test_perimeter_ring_stays_in_positive_quadrant() -> None:
    # KC-93X3X3 rc6: the ring's top and left caps were placed ABOVE / LEFT of the
    # grid origin at NEGATIVE coordinates. The leaf board-size search grows the
    # fitted Edge.Cuts from the origin into +x/+y only (array_placement keeps coords
    # positive), so a negative-coordinate pad fell OUTSIDE the outline and the leaf
    # legality gate rejected the placement every round -> 0 leaves -> no parent.
    # The framed cluster (grid + ring) must sit wholly in the positive quadrant.
    # The old ring test only checked caps were outside the LED *block* on each side
    # (relative geometry, blind to the outline), so it never caught this.
    comps, placed = _ring_layout()
    assert placed, "ring fallback did not place the companions"
    # The ring puts a cap on the top edge and one on the left edge -- exactly the
    # ones that used to go negative. Every placed body AND pad must be >= 0.
    for r in placed:
        c = comps[r]
        assert c.pos.x >= 0.0 and c.pos.y >= 0.0, (
            f"{r} body at ({c.pos.x:.3f}, {c.pos.y:.3f}) is outside the positive "
            "quadrant -> would fall outside the fitted leaf outline")
        for p in c.pads:
            assert p.pos.x >= -1e-6 and p.pos.y >= -1e-6, (
                f"{r} pad {p.pad_id} at ({p.pos.x:.3f}, {p.pos.y:.3f}) is negative "
                "-> outside the fitted leaf Edge.Cuts (the KC-93X3X3 legality kill)")
    # The re-base is rigid: the ring is still a clean 4-edge frame around the grid
    # (relative geometry preserved), just translated into positive space.
    leds = [comps[f"D{i}"].pos for i in range(1, 5)]
    lx0, lx1 = min(p.x for p in leds), max(p.x for p in leds)
    ly0, ly1 = min(p.y for p in leds), max(p.y for p in leds)
    sides = {"below": 0, "above": 0, "right": 0, "left": 0}
    for i in range(1, 5):
        c = comps[f"C{i}"].pos
        on = [c.y > ly1, c.y < ly0, c.x > lx1, c.x < lx0]
        assert sum(on) == 1, f"C{i} not cleanly outside one edge after re-base"
        sides[["below", "above", "right", "left"][on.index(True)]] += 1
    assert all(v >= 1 for v in sides.values()), f"frame broke on re-base: {sides}"


def test_companion_refs_reload_retag_helper() -> None:
    # array_companion_refs is the single source of truth for both placement and
    # the post-reload legality re-tag: it must find the decaps (power/ground
    # 2-pad) and exclude signal parts, but ONLY when the array is present.
    from kicraft.autoplacer.brain.array_placement import array_companion_refs

    comps = {f"D{i}": _led(f"D{i}", f"D{i}_DOUT", "DATA_IN") for i in range(1, 5)}
    comps["C1"] = _decap("C1")
    comps["C2"] = _decap("C2")
    comps["R1"] = Component(
        ref="R1", value="330", pos=Point(0.0, 0.0), rotation=0.0, layer=Layer.FRONT,
        width_mm=1.0, height_mm=0.5,
        pads=[Pad(ref="R1", pad_id="1", pos=Point(-0.5, 0.0), net="DATA",
                  layer=Layer.FRONT),
              Pad(ref="R1", pad_id="2", pos=Point(0.5, 0.0), net="DATA_IN",
                  layer=Layer.FRONT)])
    arrays = [{"refs": [f"D{i}" for i in range(1, 5)], "rows": 2, "cols": 2}]
    assert array_companion_refs(comps, arrays) == ["C1", "C2"]
    # no array present in this leaf -> claim nothing (plain decap leaf untouched)
    assert array_companion_refs({"C1": comps["C1"]}, []) == []
    assert array_companion_refs({"C1": comps["C1"]}, arrays) == []


# --- Layer 3: grid-overlap guard --------------------------------------------

def _grid_dict(refs, centers, w=1.5, h=1.5):
    return {"refs": refs, "led_w": w, "led_h": h,
            "centers": [Point(x, y) for x, y in centers]}


def test_assert_grids_disjoint_allows_separated_grids() -> None:
    a = _grid_dict(["D1", "D2"], [(2, 2), (4, 2)])
    b = _grid_dict(["E1", "E2"], [(20, 2), (22, 2)])
    _assert_grids_disjoint([a, b])  # no raise


def test_assert_grids_disjoint_raises_on_colocated_grids() -> None:
    a = _grid_dict(["D1", "D2"], [(2, 2), (4, 2)])
    b = _grid_dict(["C1", "C2"], [(2, 2), (4, 2)])  # same coords as the LED grid
    with pytest.raises(ValueError, match="overlap"):
        _assert_grids_disjoint([a, b])


def test_place_array_leaves_rejects_two_arrays_on_same_origin() -> None:
    # The KC-NZXXEE shape: an LED array and a cap array, both 2x2. Every grid is
    # laid from the same origin, so without Layer 1 they co-locate -- the guard
    # must catch that rather than emit a board where caps sit on the LEDs.
    comps = {f"D{i}": _comp(f"D{i}", "LED", 1.5, 1.5, 4) for i in range(1, 5)}
    comps.update({f"C{i}": _comp(f"C{i}", "100nF", 1.0, 0.5, 2) for i in range(1, 5)})
    arrays = [
        {"refs": [f"D{i}" for i in range(1, 5)], "rows": 2, "cols": 2,
         "serpentine": True},
        {"refs": [f"C{i}" for i in range(1, 5)], "rows": 2, "cols": 2,
         "serpentine": True},
    ]
    with pytest.raises(ValueError, match="overlap"):
        place_array_leaves(
            comps, arrays,
            {"array_gap_mm": 0.5, "placement_clearance_mm": 0.0},
        )
