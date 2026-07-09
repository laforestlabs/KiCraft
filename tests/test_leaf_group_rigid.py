"""Tests for kicraft.autoplacer.brain.leaf_group_rigid.

The rigid-group representation: tidy by construction, and moves/rotates as one
unit under its anchor. Synthetic Components only.
"""

from __future__ import annotations

import math

from kicraft.autoplacer.brain.geometry import rotate_vector
from kicraft.autoplacer.brain.leaf_group_rigid import (
    build_rigid_groups,
    group_child_refs,
    sync_rigid_groups,
)
from kicraft.autoplacer.brain.leaf_tidiness import orientation_axis
from kicraft.autoplacer.brain.types import Component, Layer, Pad, Point


def _pad(ref, pid, x, y, net):
    return Pad(ref=ref, pad_id=pid, pos=Point(x, y), net=net,
               layer=Layer.FRONT, size_mm=Point(0.5, 0.5))


def _cap(ref, x, y, rot, nets, w=1.0, h=2.0):
    return Component(
        ref=ref, value="100nF", pos=Point(x, y), rotation=rot, layer=Layer.FRONT,
        width_mm=w, height_mm=h, kind="passive", body_center=Point(x, y),
        pads=[_pad(ref, "1", x, y - h / 2, nets[0]),
              _pad(ref, "2", x, y + h / 2, nets[1])],
    )


def _ic(ref, x, y, nets):
    return Component(
        ref=ref, value="U", pos=Point(x, y), rotation=0.0, layer=Layer.FRONT,
        width_mm=6.0, height_mm=6.0, kind="ic", body_center=Point(x, y),
        pads=[_pad(ref, str(i + 1), x, y, nets[i]) for i in range(len(nets))],
    )


def _group():
    return {
        "U1": _ic("U1", 0, 0, ["VCC", "GND", "SIG"]),
        "C1": _cap("C1", 8, 1, rot=0, nets=("VCC", "GND")),
        "C2": _cap("C2", 10, -3, rot=90, nets=("VCC", "GND")),
        "C3": _cap("C3", 12, 4, rot=0, nets=("SIG", "GND")),
    }


def _bc(c):
    return c.body_center


class TestBuild:
    def test_builds_one_rigid_group(self):
        rigid = build_rigid_groups(_group())
        assert len(rigid) == 1
        assert rigid[0].anchor_ref == "U1"
        assert set(rigid[0].member_refs) == {"C1", "C2", "C3"}

    def test_tidy_by_construction(self):
        comps = _group()
        rigid = build_rigid_groups(comps)
        sync_rigid_groups(comps, rigid)
        # uniform orientation
        axes = {orientation_axis(comps[r].rotation) for r in ("C1", "C2", "C3")}
        assert len(axes) == 1
        # straight row/column: one axis shared to within a grid step
        cs = [_bc(comps[r]) for r in ("C1", "C2", "C3")]
        xs = [p.x for p in cs]
        ys = [p.y for p in cs]
        assert min(max(xs) - min(xs), max(ys) - min(ys)) < 0.6


class TestRigidFollow:
    def test_translation_is_rigid(self):
        comps = _group()
        rigid = build_rigid_groups(comps)
        sync_rigid_groups(comps, rigid)
        before = {r: (_bc(comps[r]).x, _bc(comps[r]).y) for r in ("C1", "C2", "C3")}
        # move the anchor by (10, 5)
        comps["U1"].pos = Point(10, 5)
        comps["U1"].body_center = Point(10, 5)
        sync_rigid_groups(comps, rigid)
        for r in ("C1", "C2", "C3"):
            bx, by = before[r]
            assert abs(_bc(comps[r]).x - (bx + 10)) < 1e-6
            assert abs(_bc(comps[r]).y - (by + 5)) < 1e-6

    def test_rotation_is_rigid(self):
        comps = _group()
        rigid = build_rigid_groups(comps)
        sync_rigid_groups(comps, rigid)
        a = _bc(comps["U1"])
        rel_before = {
            r: Point(_bc(comps[r]).x - a.x, _bc(comps[r]).y - a.y)
            for r in ("C1", "C2", "C3")
        }
        rot_before = {r: comps[r].rotation for r in ("C1", "C2", "C3")}
        # rotate the anchor 90 deg
        comps["U1"].rotation = 90.0
        sync_rigid_groups(comps, rigid)
        a2 = _bc(comps["U1"])
        for r in ("C1", "C2", "C3"):
            expect = rotate_vector(rel_before[r], 90.0)  # KiCad CW
            got = Point(_bc(comps[r]).x - a2.x, _bc(comps[r]).y - a2.y)
            assert abs(got.x - expect.x) < 1e-6
            assert abs(got.y - expect.y) < 1e-6
            assert abs((comps[r].rotation - (rot_before[r] + 90.0)) % 360.0) < 1e-6

    def test_sync_idempotent(self):
        comps = _group()
        rigid = build_rigid_groups(comps)
        sync_rigid_groups(comps, rigid)
        snap = {r: (_bc(comps[r]).x, _bc(comps[r]).y, comps[r].rotation)
                for r in ("C1", "C2", "C3")}
        sync_rigid_groups(comps, rigid)
        sync_rigid_groups(comps, rigid)
        for r in ("C1", "C2", "C3"):
            x, y, rot = snap[r]
            assert abs(_bc(comps[r]).x - x) < 1e-6
            assert abs(_bc(comps[r]).y - y) < 1e-6
            assert abs(comps[r].rotation - rot) < 1e-6

    def test_child_refs(self):
        rigid = build_rigid_groups(_group())
        assert group_child_refs(rigid) == {"C1", "C2", "C3"}
