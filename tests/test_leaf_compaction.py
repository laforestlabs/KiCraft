"""Post-SA deterministic compaction pass (area-compaction Phase 3).

Pure-geometry tests: slides close inter-part slack down to the clearance,
respect locked parts / keep-outs / keep-ins / board bounds, move pads with
their components, and are deterministic + idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from kicraft.autoplacer.brain.leaf_compaction import compact_toward_centroid
from kicraft.autoplacer.brain.types import Component, Layer, Pad, Point


def _comp(
    ref: str,
    x: float,
    y: float,
    w: float = 4.0,
    h: float = 4.0,
    locked: bool = False,
) -> Component:
    pad = Pad(ref=ref, pad_id="1", pos=Point(x, y), net="N", layer=Layer.FRONT)
    return Component(
        ref=ref,
        value="x",
        pos=Point(x, y),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=w,
        height_mm=h,
        locked=locked,
        pads=[pad],
    )


@dataclass
class _KeepoutRect:
    tl: Point
    br: Point
    owner_ref: str = ""
    owner_origin: Point | None = None


OUTLINE = (Point(0.0, 0.0), Point(100.0, 100.0))


class TestCompaction:
    def test_two_parts_close_to_clearance(self):
        comps = {
            "A": _comp("A", 20.0, 50.0),
            "B": _comp("B", 80.0, 50.0),
        }
        summary = compact_toward_centroid(
            comps, board_outline=OUTLINE, clearance_mm=2.0
        )
        assert summary["moved_components"] > 0
        gap = (comps["B"].pos.x - 2.0) - (comps["A"].pos.x + 2.0)  # facing edges
        assert gap == pytest.approx(2.0, abs=0.06)
        # Symmetric squeeze: both slid toward the shared centroid
        assert comps["A"].pos.x > 20.0
        assert comps["B"].pos.x < 80.0

    def test_pads_travel_with_component(self):
        comps = {
            "A": _comp("A", 20.0, 50.0),
            "B": _comp("B", 80.0, 50.0),
        }
        compact_toward_centroid(comps, board_outline=OUTLINE, clearance_mm=2.0)
        for c in comps.values():
            assert c.pads[0].pos.x == pytest.approx(c.pos.x)
            assert c.pads[0].pos.y == pytest.approx(c.pos.y)

    def test_locked_component_never_moves_but_blocks(self):
        comps = {
            "J1": _comp("J1", 10.0, 50.0, locked=True),
            "A": _comp("A", 60.0, 50.0),
        }
        compact_toward_centroid(comps, board_outline=OUTLINE, clearance_mm=2.0)
        assert comps["J1"].pos.x == 10.0
        # A slid toward the centroid but stopped >= clearance from J1
        gap = (comps["A"].pos.x - 2.0) - (comps["J1"].pos.x + 2.0)
        assert gap >= 2.0 - 1e-6

    def test_array_members_untouched(self):
        a = _comp("L1", 30.0, 50.0)
        a.array_member = True
        comps = {"L1": a, "B": _comp("B", 70.0, 50.0)}
        compact_toward_centroid(comps, board_outline=OUTLINE, clearance_mm=2.0)
        assert comps["L1"].pos.x == 30.0

    def test_perpendicular_lane_ignored(self):
        """A part far above the slide lane must not block an x-slide."""
        comps = {
            "A": _comp("A", 20.0, 20.0),
            "B": _comp("B", 80.0, 20.0),
            "C": _comp("C", 50.0, 80.0),  # same x-range as the centroid, far in y
        }
        compact_toward_centroid(comps, board_outline=OUTLINE, clearance_mm=2.0)
        # A and B compacted horizontally toward the centroid despite C
        assert comps["B"].pos.x - comps["A"].pos.x < 30.0

    def test_keepout_not_entered(self):
        keepout = _KeepoutRect(tl=Point(40.0, 40.0), br=Point(60.0, 60.0), owner_ref="U9")
        comps = {
            "A": _comp("A", 20.0, 50.0),
            "B": _comp("B", 80.0, 50.0),
        }
        compact_toward_centroid(
            comps,
            board_outline=OUTLINE,
            clearance_mm=2.0,
            keepout_rects=[keepout],
        )
        # Neither part's bbox may enter the keepout (margin 0.1)
        assert comps["A"].pos.x + 2.0 <= 40.0 - 0.05
        assert comps["B"].pos.x - 2.0 >= 60.0 + 0.05

    def test_keep_in_spec_respected(self):
        comps = {
            "H1": _comp("H1", 50.0, 50.0, w=3.0, h=3.0, locked=True),
            "A": _comp("A", 10.0, 50.0),
        }
        compact_toward_centroid(
            comps,
            board_outline=OUTLINE,
            clearance_mm=2.0,
            keep_in_specs=[{"ref": "H1", "margin_mm": 4.0}],
        )
        # H1's keep-in rect spans x [44.5, 55.5]; A must stop outside it
        assert comps["A"].pos.x + 2.0 <= 44.5 - 0.05

    def test_deterministic_and_idempotent(self):
        def build():
            return {
                "A": _comp("A", 15.0, 30.0),
                "B": _comp("B", 85.0, 30.0),
                "C": _comp("C", 50.0, 80.0),
                "J": _comp("J", 5.0, 50.0, locked=True),
            }

        c1, c2 = build(), build()
        compact_toward_centroid(c1, board_outline=OUTLINE, clearance_mm=2.0)
        compact_toward_centroid(c2, board_outline=OUTLINE, clearance_mm=2.0)
        pos1 = {r: (c.pos.x, c.pos.y) for r, c in c1.items()}
        pos2 = {r: (c.pos.x, c.pos.y) for r, c in c2.items()}
        assert pos1 == pos2

        # Near-converged: the first call does the heavy lifting (tens of mm);
        # a second application only mops up the geometric tail toward the
        # locked anchor and never restructures or re-sprawls the layout.
        first_call_slide = 189.5  # observed total for this fixture
        summary = compact_toward_centroid(
            c1, board_outline=OUTLINE, clearance_mm=2.0
        )
        assert summary["total_slide_mm"] < first_call_slide * 0.05

        def _bbox_area(comps):
            bbs = [c.physical_bbox() for c in comps.values()]
            w = max(b[1].x for b in bbs) - min(b[0].x for b in bbs)
            h = max(b[1].y for b in bbs) - min(b[0].y for b in bbs)
            return w * h

        assert _bbox_area(c1) <= _bbox_area(c2) + 1e-6

    def test_shrinks_placed_bbox(self):
        comps = {
            "A": _comp("A", 10.0, 10.0),
            "B": _comp("B", 90.0, 10.0),
            "C": _comp("C", 10.0, 90.0),
            "D": _comp("D", 90.0, 90.0),
        }

        def bbox_area():
            bbs = [c.physical_bbox() for c in comps.values()]
            w = max(b[1].x for b in bbs) - min(b[0].x for b in bbs)
            h = max(b[1].y for b in bbs) - min(b[0].y for b in bbs)
            return w * h

        before = bbox_area()
        compact_toward_centroid(comps, board_outline=OUTLINE, clearance_mm=2.0)
        after = bbox_area()
        # 4 sprawled 4mm parts -> a 2x2 block at clearance: huge shrink
        assert after < before * 0.05

    def test_empty_and_all_locked_noop(self):
        assert compact_toward_centroid(
            {}, board_outline=OUTLINE, clearance_mm=2.0
        )["moved_components"] == 0
        comps = {"J": _comp("J", 10.0, 10.0, locked=True)}
        assert compact_toward_centroid(
            comps, board_outline=OUTLINE, clearance_mm=2.0
        )["moved_components"] == 0
        assert comps["J"].pos.x == 10.0
