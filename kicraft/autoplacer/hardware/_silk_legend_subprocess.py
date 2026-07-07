#!/usr/bin/env python3
"""Place the board legend + authored silk labels on a routed board.

Invoked by ``silk_legend.apply_silk_legend`` with a JSON payload path as
``sys.argv[1]`` (own file, not an inline string, for the same lint/IDE
reasons as ``_parent_stamp_subprocess.py``). Runs in a fresh subprocess
because pcbnew's SWIG teardown is not trusted in-process (house rule).

The LLM never chooses geometry: this pass owns placement. Content arrives
pre-linted in the payload; every text lands only where it fits inside the
board outline without colliding with courtyards, pads, or existing silk.
Labels that cannot be placed are DROPPED and reported in the result JSON —
never squeezed onto copper or off the board.

Output: result JSON at ``payload["result_path"]`` with::

    {"placed": [{"id", "x_mm", "y_mm", "height_mm", "layer"}],
     "dropped": [{"id", "reason"}]}

plus the ``__KICRAFT_PCBNEW_OK__`` sentinel on stdout after a successful
board save (must match ``freerouting_runner._PCBNEW_OK_SENTINEL``).
"""

from __future__ import annotations

import json
import sys

import pcbnew

from kicraft.autoplacer.hardware.silk_geometry import (
    bbox_inside_poly,
    boxes_overlap,
    poly_bbox,
)

_THICKNESS_RATIO = 0.15  # KiCad's nominal stroke:height ratio


def _mm_box(bb) -> tuple[float, float, float, float]:
    return (
        pcbnew.ToMM(bb.GetLeft()),
        pcbnew.ToMM(bb.GetTop()),
        pcbnew.ToMM(bb.GetRight()),
        pcbnew.ToMM(bb.GetBottom()),
    )


def _outline_poly(board) -> list[tuple[float, float]]:
    """The board outline as an mm point list (first hole-free outline)."""
    shape = pcbnew.SHAPE_POLY_SET()
    try:
        ok = board.GetBoardPolygonOutlines(shape)
    except Exception:
        ok = False
    if ok and shape.OutlineCount() > 0:
        chain = shape.COutline(0)
        pts = [
            (pcbnew.ToMM(chain.CPoint(i).x), pcbnew.ToMM(chain.CPoint(i).y))
            for i in range(chain.PointCount())
        ]
        if len(pts) >= 3:
            return pts
    bb = board.GetBoardEdgesBoundingBox()
    left, top, right, bottom = _mm_box(bb)
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _collect_obstacles(board) -> dict[str, list]:
    """Per-side (F/B) mm boxes silk text must not touch.

    Courtyards + pads (through-hole pads block both sides) + every existing
    silk item (leaf group boxes/labels, refdes that stayed on silk).
    """
    obstacles: dict[str, list] = {"F": [], "B": []}

    def _add(side: str, bb) -> None:
        box = _mm_box(bb)
        if box[2] > box[0] and box[3] > box[1]:
            obstacles[side].append(box)

    for fp in board.Footprints():
        side = "B" if fp.GetLayer() == pcbnew.B_Cu else "F"
        for layer, court_side in ((pcbnew.F_CrtYd, "F"), (pcbnew.B_CrtYd, "B")):
            try:
                court = fp.GetCourtyard(layer).BBox()
            except Exception:
                continue
            if court.GetWidth() > 0 and court.GetHeight() > 0:
                _add(court_side, court)
        for pad in fp.Pads():
            try:
                through = pad.HasHole()
            except Exception:
                through = True
            if through:
                _add("F", pad.GetBoundingBox())
                _add("B", pad.GetBoundingBox())
            else:
                _add(side, pad.GetBoundingBox())
        for item in list(fp.GraphicalItems()) + [fp.Reference(), fp.Value()]:
            try:
                if not getattr(item, "IsVisible", lambda: True)():
                    continue
                layer = item.GetLayer()
            except Exception:
                continue
            if layer == pcbnew.F_SilkS:
                _add("F", item.GetBoundingBox())
            elif layer == pcbnew.B_SilkS:
                _add("B", item.GetBoundingBox())

    for d in board.GetDrawings():
        layer = d.GetLayer()
        if layer == pcbnew.F_SilkS:
            _add("F", d.GetBoundingBox())
        elif layer == pcbnew.B_SilkS:
            _add("B", d.GetBoundingBox())

    return obstacles


def _make_text(board, text: str, height_mm: float, layer, mirrored: bool):
    txt = pcbnew.PCB_TEXT(board)
    txt.SetText(text)
    txt.SetLayer(layer)
    hi = pcbnew.FromMM(height_mm)
    txt.SetTextSize(pcbnew.VECTOR2I(hi, hi))
    txt.SetTextThickness(pcbnew.FromMM(max(0.08, height_mm * _THICKNESS_RATIO)))
    txt.SetHorizJustify(pcbnew.GR_TEXT_H_ALIGN_LEFT)
    txt.SetVertJustify(pcbnew.GR_TEXT_V_ALIGN_TOP)
    if mirrored:
        txt.SetMirrored(True)
    return txt


class _Placer:
    def __init__(self, board, payload: dict):
        self.board = board
        self.poly = _outline_poly(board)
        self.board_box = poly_bbox(self.poly)
        self.obstacles = _collect_obstacles(board)
        self.clearance = float(payload.get("clearance_mm", 0.25))
        self.edge_margin = float(payload.get("edge_margin_mm", 0.5))
        self.placed: list[dict] = []
        self.dropped: list[dict] = []

    # -- text measurement ---------------------------------------------------
    def _measure(self, txt) -> tuple[float, float, float, float]:
        """(w, h, off_x, off_y): bbox size and anchor->bbox-topleft offset."""
        txt.SetPosition(pcbnew.VECTOR2I(0, 0))
        box = _mm_box(txt.GetBoundingBox())
        return (box[2] - box[0], box[3] - box[1], -box[0], -box[1])

    def _spot_ok(self, side: str, box) -> bool:
        if not bbox_inside_poly(box, self.poly, self.edge_margin):
            return False
        return not any(
            boxes_overlap(box, ob, self.clearance) for ob in self.obstacles[side]
        )

    def _commit(self, txt, side: str, tx: float, ty: float,
                off_x: float, off_y: float, label_id: str, height: float) -> None:
        txt.SetPosition(
            pcbnew.VECTOR2I(pcbnew.FromMM(tx + off_x), pcbnew.FromMM(ty + off_y))
        )
        self.board.Add(txt)
        box = _mm_box(txt.GetBoundingBox())
        self.obstacles[side].append(box)
        self.placed.append({
            "id": label_id,
            "x_mm": round(tx, 3),
            "y_mm": round(ty, 3),
            "height_mm": height,
            "layer": "F.SilkS" if side == "F" else "B.SilkS",
        })

    # -- legend (edge-anchored block, front silk) ---------------------------
    def place_legend(self, lines: list[dict], gap_mm: float) -> bool:
        """Stack of text lines placed as one block along the bottom (then
        top) edge strip, centered candidates first. Returns True if placed."""
        if not lines:
            return False
        left, top, right, bottom = self.board_box
        for scale in (1.0, 0.85, 0.7):
            texts = []
            for ln in lines:
                h = max(0.8, round(float(ln["height_mm"]) * scale, 2))
                texts.append(
                    (_make_text(self.board, ln["text"], h, pcbnew.F_SilkS, False), h)
                )
            metrics = [self._measure(t) for t, _ in texts]
            block_w = max(m[0] for m in metrics)
            block_h = sum(m[1] for m in metrics) + gap_mm * (len(metrics) - 1)

            xs: list[float] = []
            x0 = left + self.edge_margin
            x1 = right - self.edge_margin - block_w
            if x1 >= x0:
                n_steps = int((x1 - x0) / 2.0)
                xs = sorted(
                    (x0 + i * 2.0 for i in range(n_steps + 1)),
                    key=lambda x: abs(x + block_w / 2 - (left + right) / 2),
                )
            candidates = [
                (x, bottom - self.edge_margin - block_h) for x in xs
            ] + [(x, top + self.edge_margin) for x in xs]

            for tx, ty in candidates:
                block_box = (tx, ty, tx + block_w, ty + block_h)
                line_boxes = []
                y = ty
                fits = True
                for (w, h, _ox, _oy) in metrics:
                    lb = (tx, y, tx + w, y + h)
                    if not self._spot_ok("F", lb):
                        fits = False
                        break
                    line_boxes.append(lb)
                    y += h + gap_mm
                if not fits or not bbox_inside_poly(
                    block_box, self.poly, self.edge_margin
                ):
                    continue
                y = ty
                for i, (txt, h) in enumerate(texts):
                    w, lh, ox, oy = metrics[i]
                    self._commit(txt, "F", tx, y, ox, oy, f"legend:{i}", h)
                    y += lh + gap_mm
                return True
        return False

    # -- anchored / free labels ---------------------------------------------
    def place_label(self, label: dict) -> None:
        label_id = label.get("id") or "label"
        ref = label.get("ref")
        fp = self.board.FindFootprintByReference(ref) if ref else None
        if ref and fp is None:
            self.dropped.append({"id": label_id,
                                 "reason": f"anchor {ref} not on board"})
            return

        side = "F"
        mirrored = False
        if fp is not None and fp.GetLayer() == pcbnew.B_Cu:
            side = "B"
            mirrored = True
        layer = pcbnew.B_SilkS if side == "B" else pcbnew.F_SilkS

        heights = [float(h) for h in (label.get("heights_mm") or [1.0, 0.9, 0.8])]
        for h in heights:
            txt = _make_text(self.board, label["text"], h, layer, mirrored)
            w, th, ox, oy = self._measure(txt)
            for tx, ty in self._label_candidates(fp, w, th, label.get("prefer")):
                box = (tx, ty, tx + w, ty + th)
                if self._spot_ok(side, box):
                    self._commit(txt, side, tx, ty, ox, oy, label_id, h)
                    return
        # Priority-1 labels degrade to "anywhere free on the board" before
        # dropping: a DIP table across the board still beats no table. The
        # anchored pass already failed, so sweep candidates (fp=None), front
        # silk, smallest height first.
        if int(label.get("priority", 2)) == 1 and fp is not None:
            for h in sorted(heights):
                txt = _make_text(self.board, label["text"], h, pcbnew.F_SilkS, False)
                w, th, ox, oy = self._measure(txt)
                for tx, ty in self._label_candidates(None, w, th, None):
                    box = (tx, ty, tx + w, ty + th)
                    if self._spot_ok("F", box):
                        self._commit(txt, "F", tx, ty, ox, oy, label_id, h)
                        return
        self.dropped.append({"id": label_id, "reason": "no clear space on silk"})

    def _label_candidates(self, fp, w: float, h: float, prefer: str | None):
        """Candidate bbox top-lefts: rings around the anchor courtyard on the
        preferred side first, else a sweep of the free board area."""
        if fp is None:
            left, top, right, bottom = self.board_box
            for ty in _steps(top + self.edge_margin, bottom - self.edge_margin - h, 2.0):
                for tx in _steps(left + self.edge_margin, right - self.edge_margin - w, 2.0):
                    yield (tx, ty)
            return

        layer = pcbnew.B_CrtYd if fp.GetLayer() == pcbnew.B_Cu else pcbnew.F_CrtYd
        try:
            cb = fp.GetCourtyard(layer).BBox()
            court = _mm_box(cb)
            if court[2] <= court[0]:
                raise ValueError
        except Exception:
            court = _mm_box(fp.GetBoundingBox())

        order = ["right", "below", "above", "left"]
        if prefer in order:
            order.remove(prefer)
            order.insert(0, prefer)

        cx = (court[0] + court[2]) / 2
        cy = (court[1] + court[3]) / 2
        for gap in (0.4, 1.0, 2.0, 3.5):
            for side_name in order:
                if side_name == "right":
                    tx = court[2] + gap
                    for ty in (cy - h / 2, court[1], court[3] - h):
                        yield (tx, ty)
                elif side_name == "left":
                    tx = court[0] - gap - w
                    for ty in (cy - h / 2, court[1], court[3] - h):
                        yield (tx, ty)
                elif side_name == "below":
                    ty = court[3] + gap
                    for tx in (cx - w / 2, court[0], court[2] - w):
                        yield (tx, ty)
                else:  # above
                    ty = court[1] - gap - h
                    for tx in (cx - w / 2, court[0], court[2] - w):
                        yield (tx, ty)


def _steps(lo: float, hi: float, step: float):
    n = int((hi - lo) / step)
    for i in range(max(0, n + 1)):
        yield lo + i * step


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: _silk_legend_subprocess.py <payload.json>", file=sys.stderr)
        return 64
    with open(argv[1]) as f:
        payload = json.load(f)

    board = pcbnew.LoadBoard(payload["pcb_path"])
    placer = _Placer(board, payload)

    legend = payload.get("legend") or {}
    lines = legend.get("lines") or []
    if lines:
        if not placer.place_legend(lines, float(legend.get("gap_mm", 0.3))):
            placer.dropped.append(
                {"id": "legend", "reason": "no clear edge strip for the legend"}
            )

    labels = sorted(
        payload.get("labels") or [],
        key=lambda lb: (int(lb.get("priority", 2)), str(lb.get("id", ""))),
    )
    for label in labels:
        placer.place_label(label)

    with open(payload["result_path"], "w") as f:
        json.dump({"placed": placer.placed, "dropped": placer.dropped}, f, indent=1)

    board.Save(payload.get("output_path") or payload["pcb_path"])
    # Success sentinel (matches freerouting_runner._PCBNEW_OK_SENTINEL) so a
    # pcbnew/wx teardown SIGSEGV after the Save is not read as a failure.
    print("__KICRAFT_PCBNEW_OK__")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
