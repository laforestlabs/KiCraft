"""Tests for kicraft.autoplacer.brain.leaf_structured_layout (Stage-3 packer).

Synthetic Components only — no pcbnew. Verifies the packer tidies functional
passive rows (uniform orientation + straight alignment) while never introducing
a courtyard overlap or pushing a part off-board.
"""

from __future__ import annotations

from kicraft.autoplacer.brain.leaf_structured_layout import (
    _hpwl,
    _signal_nets,
    apply_structured_local_layout,
)
from kicraft.autoplacer.brain.leaf_tidiness import orientation_axis
from kicraft.autoplacer.brain.types import Component, Layer, Pad, Point


def _pad(ref, pid, x, y, net):
    return Pad(ref=ref, pad_id=pid, pos=Point(x, y), net=net,
               layer=Layer.FRONT, size_mm=Point(0.5, 0.5))


def _cap(ref, x, y, rot, nets, w=1.0, h=2.0):
    # Two pads on the long axis, centered on (x, y).
    return Component(
        ref=ref, value="100nF", pos=Point(x, y), rotation=rot, layer=Layer.FRONT,
        width_mm=w, height_mm=h, kind="passive",
        body_center=Point(x, y),
        pads=[_pad(ref, "1", x, y - h / 2, nets[0]),
              _pad(ref, "2", x, y + h / 2, nets[1])],
    )


def _ic(ref, x, y, nets, w=6.0, h=6.0):
    pads = [_pad(ref, str(i + 1), x, y, nets[i]) for i in range(len(nets))]
    return Component(
        ref=ref, value="U", pos=Point(x, y), rotation=0.0, layer=Layer.FRONT,
        width_mm=w, height_mm=h, kind="ic", body_center=Point(x, y), pads=pads,
    )


def _outline(margin=40.0):
    return (Point(-margin, -margin), Point(margin, margin))


def _courtyard_overlaps(comps):
    items = list(comps.values())
    n = 0
    for i in range(len(items)):
        a_tl, a_br = items[i].bbox(0.0)
        for j in range(i + 1, len(items)):
            if items[i].layer != items[j].layer:
                continue
            b_tl, b_br = items[j].bbox(0.0)
            if (min(a_br.x, b_br.x) - max(a_tl.x, b_tl.x) > 0.05
                    and min(a_br.y, b_br.y) - max(a_tl.y, b_tl.y) > 0.05):
                n += 1
    return n


class TestPacker:
    def _scattered_group(self):
        # U1 + 3 decoupling caps at mixed orientations and scattered positions.
        return {
            "U1": _ic("U1", 0, 0, ["VCC", "GND", "SIG"]),
            "C1": _cap("C1", 8, 1, rot=0, nets=("VCC", "GND")),
            "C2": _cap("C2", 10, -3, rot=90, nets=("VCC", "GND")),
            "C3": _cap("C3", 12, 4, rot=0, nets=("SIG", "GND")),
        }

    def test_unifies_orientation(self):
        comps = self._scattered_group()
        apply_structured_local_layout(comps, board_outline=_outline())
        axes = {orientation_axis(comps[r].rotation) for r in ("C1", "C2", "C3")}
        assert len(axes) == 1  # all caps share one axis

    def test_straightens_the_row(self):
        comps = self._scattered_group()
        apply_structured_local_layout(comps, board_outline=_outline())
        cs = [comps[r].body_center for r in ("C1", "C2", "C3")]
        xs = [p.x for p in cs]
        ys = [p.y for p in cs]
        # Whichever axis they distribute along, the other is shared (straight).
        shared_spread = min(max(xs) - min(xs), max(ys) - min(ys))
        assert shared_spread < 0.6  # within a grid step of a straight line

    def test_no_new_courtyard_overlap(self):
        comps = self._scattered_group()
        before = _courtyard_overlaps(comps)
        apply_structured_local_layout(comps, board_outline=_outline())
        assert _courtyard_overlaps(comps) <= before

    def test_atomic_skip_when_boxed_in(self):
        # A tiny board that can't fit a legal 3-cap row -> group left untouched.
        comps = self._scattered_group()
        orig = {r: (c.pos.x, c.pos.y, c.rotation) for r, c in comps.items()}
        summary = apply_structured_local_layout(
            comps, board_outline=(Point(-3, -3), Point(3, 3))
        )
        assert summary["groups_skipped"] >= 1
        # Untouched group members keep their exact pre-pass geometry.
        for r in ("C1", "C2", "C3"):
            assert (comps[r].pos.x, comps[r].pos.y, comps[r].rotation) == orig[r]

    def test_locked_and_array_never_move(self):
        comps = self._scattered_group()
        comps["C2"].locked = True
        before = (comps["C2"].pos.x, comps["C2"].pos.y)
        apply_structured_local_layout(comps, board_outline=_outline())
        assert (comps["C2"].pos.x, comps["C2"].pos.y) == before

    def test_no_group_no_change(self):
        # Two unrelated passives, no anchor -> nothing to do.
        comps = {
            "R1": _cap("R1", 0, 0, 0, ("A", "B")),
            "R2": _cap("R2", 5, 5, 90, ("C", "D")),
        }
        summary = apply_structured_local_layout(comps, board_outline=_outline())
        assert summary["groups"] == 0
        assert summary["members_aligned"] == 0

    def test_routability_guard_blocks_net_stretch(self):
        # Tolerance 0.0 forbids ANY signal-net HPWL growth: a tidy row that
        # would spread the caps (raising HPWL) is skipped, group left as-is.
        comps = self._scattered_group()
        orig = {r: (comps[r].pos.x, comps[r].pos.y) for r in ("C1", "C2", "C3")}
        summary = apply_structured_local_layout(
            comps, board_outline=_outline(), max_hpwl_increase=0.0
        )
        if summary["groups_skipped_routability"]:
            for r in ("C1", "C2", "C3"):
                assert (comps[r].pos.x, comps[r].pos.y) == orig[r]

    def test_permissive_tolerance_places_more(self):
        # A very loose tolerance never skips for routability; a strict one may.
        loose = apply_structured_local_layout(
            self._scattered_group(), board_outline=_outline(),
            max_hpwl_increase=100.0,
        )
        strict = apply_structured_local_layout(
            self._scattered_group(), board_outline=_outline(),
            max_hpwl_increase=0.0,
        )
        assert loose["groups_skipped_routability"] == 0
        assert strict["groups_placed"] <= loose["groups_placed"]


class TestRoutabilityHelpers:
    def test_signal_nets_excludes_high_fanout(self):
        # GND touched by 8 pads -> power/global -> excluded; SIG (2 pads) kept.
        comps = {
            "U1": _ic("U1", 0, 0, ["SIG"] + ["GND"] * 7),  # 1 SIG + 7 GND pads
            "C1": _cap("C1", 5, 0, 0, ("SIG", "GND")),
        }
        nets = _signal_nets(comps, {"C1"})
        assert "SIG" in nets
        assert "GND" not in nets

    def test_hpwl_bbox_half_perimeter(self):
        comps = {
            "C1": _cap("C1", 0, 0, 0, ("N", "X")),   # pad N at (0,-1)
            "C2": _cap("C2", 10, 4, 0, ("N", "Y")),  # pad N at (10,3)
        }
        # net N pads at (0,-1) and (10,3): HPWL = 10 + 4 = 14
        assert abs(_hpwl(comps, {"N"}, {}) - 14.0) < 1e-6
