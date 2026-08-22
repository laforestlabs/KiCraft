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
board save (must match ``routing_board._PCBNEW_OK_SENTINEL``).
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


def _shape_stroke_boxes(shape) -> list[tuple[float, float, float, float]]:
    """Thin per-edge mm boxes for an outline PCB_SHAPE, or [] for non-shapes.

    A silk gr_poly/rect/segment is a STROKE, not a filled region: the leaf
    group boxes span most of a compacted board, and treating their bbox as
    solid walls off nearly all free silk (the replayed KC-7A3VEX dropped
    every label that way). Text may sit INSIDE an outline box — only the
    drawn edges themselves are obstacles.
    """
    try:
        kind = shape.GetShape()
    except AttributeError:
        return []
    half = pcbnew.ToMM(shape.GetWidth()) / 2.0 if hasattr(shape, "GetWidth") else 0.1

    def _edge(x1, y1, x2, y2):
        return (min(x1, x2) - half, min(y1, y2) - half,
                max(x1, x2) + half, max(y1, y2) + half)

    if kind == pcbnew.SHAPE_T_SEGMENT:
        s, e = shape.GetStart(), shape.GetEnd()
        return [_edge(pcbnew.ToMM(s.x), pcbnew.ToMM(s.y),
                      pcbnew.ToMM(e.x), pcbnew.ToMM(e.y))]
    if kind == pcbnew.SHAPE_T_RECT:
        s, e = shape.GetStart(), shape.GetEnd()
        x1, y1 = pcbnew.ToMM(s.x), pcbnew.ToMM(s.y)
        x2, y2 = pcbnew.ToMM(e.x), pcbnew.ToMM(e.y)
        return [_edge(x1, y1, x2, y1), _edge(x2, y1, x2, y2),
                _edge(x2, y2, x1, y2), _edge(x1, y2, x1, y1)]
    if kind == pcbnew.SHAPE_T_POLY:
        try:
            chain = shape.GetPolyShape().COutline(0)
            pts = [(pcbnew.ToMM(chain.CPoint(i).x), pcbnew.ToMM(chain.CPoint(i).y))
                   for i in range(chain.PointCount())]
        except Exception:
            return []
        if len(pts) < 2:
            return []
        return [_edge(*pts[i], *pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    return []


def _collect_obstacles(board) -> dict[str, list]:
    """Per-side (F/B) mm boxes silk text must not touch.

    Courtyards + pads (through-hole pads block both sides) + every existing
    silk item. Outline SHAPES (group boxes) contribute their stroked edges
    only, so text can use the free space inside them; text items block
    their full bbox.
    """
    obstacles: dict[str, list] = {"F": [], "B": []}

    def _add(side: str, bb) -> None:
        box = _mm_box(bb)
        if box[2] > box[0] and box[3] > box[1]:
            obstacles[side].append(box)

    def _add_silk_item(side: str, item) -> None:
        stroke_boxes = _shape_stroke_boxes(item)
        if stroke_boxes:
            obstacles[side].extend(stroke_boxes)
        else:
            _add(side, item.GetBoundingBox())

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
                _add_silk_item("F", item)
            elif layer == pcbnew.B_SilkS:
                _add_silk_item("B", item)

    for d in board.GetDrawings():
        layer = d.GetLayer()
        if layer == pcbnew.F_SilkS:
            _add_silk_item("F", d)
        elif layer == pcbnew.B_SilkS:
            _add_silk_item("B", d)

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

    def _clamped(self, tx: float, ty: float, w: float, h: float):
        # Overhang connectors put their courtyard/pad bbox partly OFF the
        # board, pushing every bbox-relative alignment off with it. Slide the
        # candidate back inside; the side semantics survive (a below-the-pin
        # label stays below, shifted along the edge) and _spot_ok still
        # rejects anything that truly clashes.
        left, top, right, bottom = self.board_box
        m = self.edge_margin
        return (min(max(tx, left + m), right - m - w),
                min(max(ty, top + m), bottom - m - h))

    # -- legend (edge-anchored block) ----------------------------------------
    def place_legend(self, lines: list[dict], gap_mm: float) -> bool:
        """Stack of text lines placed as one block along the bottom (then
        top) edge strip, centered candidates first. Degrades: shrink ladder
        -> re-wrap long lines -> BACK silk (mirrored). Attribution must not
        crowd out functional labels, so the caller places it LAST and this
        block takes whatever honest space remains. Returns True if placed."""
        if not lines:
            return False
        for use_lines in (lines, self._wrap_lines(lines)):
            for scale in (1.0, 0.85, 0.7):
                for side in ("F", "B"):
                    if self._try_legend_block(use_lines, gap_mm, scale, side):
                        return True
        return False

    def _wrap_lines(self, lines: list[dict]) -> list[dict]:
        """Split any line wider than ~60% of the board at the most central
        space, so long attribution lines can fit narrow boards."""
        board_w = self.board_box[2] - self.board_box[0]
        out: list[dict] = []
        for ln in lines:
            text = ln["text"]
            # Rough stroke-font width estimate: ~0.95 * height per char.
            est_w = len(text) * float(ln["height_mm"]) * 0.95
            if est_w > board_w * 0.6 and " " in text.strip():
                mid = len(text) // 2
                spaces = [i for i, c in enumerate(text) if c == " "]
                cut = min(spaces, key=lambda i: abs(i - mid))
                out.append({"text": text[:cut].rstrip(),
                            "height_mm": ln["height_mm"]})
                out.append({"text": text[cut + 1:].lstrip(),
                            "height_mm": ln["height_mm"]})
            else:
                out.append(dict(ln))
        return out

    def _try_legend_block(self, lines: list[dict], gap_mm: float,
                          scale: float, side: str) -> bool:
        left, top, right, bottom = self.board_box
        layer = pcbnew.F_SilkS if side == "F" else pcbnew.B_SilkS
        mirrored = side == "B"
        texts = []
        for ln in lines:
            h = max(0.8, round(float(ln["height_mm"]) * scale, 2))
            texts.append(
                (_make_text(self.board, ln["text"], h, layer, mirrored), h)
            )
        metrics = [self._measure(t) for t, _ in texts]
        block_w = max(m[0] for m in metrics)
        block_h = sum(m[1] for m in metrics) + gap_mm * (len(metrics) - 1)

        xs: list[float] = []
        x0 = left + self.edge_margin
        x1 = right - self.edge_margin - block_w
        if x1 >= x0:
            n_steps = int((x1 - x0) / 1.0)
            xs = sorted(
                (x0 + i * 1.0 for i in range(n_steps + 1)),
                key=lambda x: abs(x + block_w / 2 - (left + right) / 2),
            )
        candidates = [
            (x, bottom - self.edge_margin - block_h) for x in xs
        ] + [(x, top + self.edge_margin) for x in xs]

        for tx, ty in candidates:
            block_box = (tx, ty, tx + block_w, ty + block_h)
            y = ty
            fits = True
            for (w, h, _ox, _oy) in metrics:
                if not self._spot_ok(side, (tx, y, tx + w, y + h)):
                    fits = False
                    break
                y += h + gap_mm
            if not fits or not bbox_inside_poly(
                block_box, self.poly, self.edge_margin
            ):
                continue
            y = ty
            for i, (txt, h) in enumerate(texts):
                w, lh, ox, oy = metrics[i]
                self._commit(txt, side, tx, y, ox, oy, f"legend:{i}", h)
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

    def place_pinout(self, label: dict) -> None:
        """One short text beside each pin of a connector. Each entry is placed
        independently against its own pad bbox (outward from the body
        centroid) and committed as its own obstacle, so later pins and labels
        avoid it. A pin that cannot fit drops alone, never the whole label."""
        label_id = label.get("id") or "label"
        ref = label.get("ref")
        fp = self.board.FindFootprintByReference(ref) if ref else None
        if not ref or fp is None:
            self.dropped.append(
                {"id": label_id, "reason": f"anchor {ref or '(none)'} not on board"}
            )
            return

        side = "B" if fp.GetLayer() == pcbnew.B_Cu else "F"
        layer = pcbnew.B_SilkS if side == "B" else pcbnew.F_SilkS
        mirrored = side == "B"

        pads = {p.GetNumber(): p for p in fp.Pads()}
        if not pads:
            self.dropped.append({"id": label_id, "reason": "no pads"})
            return

        centers = [_mm_box(p.GetBoundingBox()) for p in pads.values()]
        centroid_x = sum((b[0] + b[2]) / 2 for b in centers) / len(centers)
        centroid_y = sum((b[1] + b[3]) / 2 for b in centers) / len(centers)

        heights = [float(h) for h in (label.get("heights_mm") or [0.8])]
        for entry in label.get("pins") or []:
            pin = str(entry.get("pin") or "").strip()
            text = str(entry.get("text") or "").strip()
            pad = pads.get(pin)
            if pad is None:
                self.dropped.append(
                    {"id": f"{label_id}:{pin}",
                     "reason": f"anchor pad {pin} not found"}
                )
                continue

            pb = _mm_box(pad.GetBoundingBox())
            cx = (pb[0] + pb[2]) / 2
            cy = (pb[1] + pb[3]) / 2
            dx = cx - centroid_x
            dy = cy - centroid_y
            if dx >= abs(dy):
                dominant = "right"
            elif -dx > abs(dy):
                dominant = "left"
            elif dy > 0:
                dominant = "below"
            else:
                dominant = "above"
            dirs = [dominant] + [d for d in ("right", "left", "below", "above")
                                 if d != dominant]

            placed = False
            for h in heights:
                txt = _make_text(self.board, text, h, layer, mirrored)
                w, th, ox, oy = self._measure(txt)
                for tx, ty in self._pin_candidates(pb, cx, cy, w, th, dirs):
                    box = (tx, ty, tx + w, ty + th)
                    if self._spot_ok(side, box):
                        self._commit(txt, side, tx, ty, ox, oy,
                                     f"{label_id}:{pin}", h)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                self.dropped.append(
                    {"id": f"{label_id}:{pin}", "reason": "no clear space on silk"}
                )

    def _pin_candidates(self, pb, cx: float, cy: float, w: float, h: float,
                        dirs: list[str]):
        """Candidate bbox top-lefts around one pad bbox, dominant side first —
        the same four-side geometry as ``_label_candidates`` with the pad bbox
        substituted for the courtyard bbox."""
        for gap in (0.4, 1.0, 2.0, 3.5):
            for d in dirs:
                if d == "right":
                    tx = pb[2] + gap
                    for ty in (cy - h / 2, pb[1], pb[3] - h):
                        yield self._clamped(tx, ty, w, h)
                elif d == "left":
                    tx = pb[0] - gap - w
                    for ty in (cy - h / 2, pb[1], pb[3] - h):
                        yield self._clamped(tx, ty, w, h)
                elif d == "below":
                    ty = pb[3] + gap
                    for tx in (cx - w / 2, pb[0], pb[2] - w):
                        yield self._clamped(tx, ty, w, h)
                else:  # above
                    ty = pb[1] - gap - h
                    for tx in (cx - w / 2, pb[0], pb[2] - w):
                        yield self._clamped(tx, ty, w, h)

    def _label_candidates(self, fp, w: float, h: float, prefer: str | None):
        """Candidate bbox top-lefts: rings around the anchor courtyard on the
        preferred side first, else a sweep of the free board area."""
        if fp is None:
            left, top, right, bottom = self.board_box
            for ty in _steps(top + self.edge_margin, bottom - self.edge_margin - h, 1.0):
                for tx in _steps(left + self.edge_margin, right - self.edge_margin - w, 1.0):
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

        for gap in (0.4, 1.0, 2.0, 3.5, 5.0, 7.0):
            for side_name in order:
                if side_name == "right":
                    tx = court[2] + gap
                    for ty in (cy - h / 2, court[1], court[3] - h):
                        yield self._clamped(tx, ty, w, h)
                elif side_name == "left":
                    tx = court[0] - gap - w
                    for ty in (cy - h / 2, court[1], court[3] - h):
                        yield self._clamped(tx, ty, w, h)
                elif side_name == "below":
                    ty = court[3] + gap
                    for tx in (cx - w / 2, court[0], court[2] - w):
                        yield self._clamped(tx, ty, w, h)
                else:  # above
                    ty = court[1] - gap - h
                    for tx in (cx - w / 2, court[0], court[2] - w):
                        yield self._clamped(tx, ty, w, h)


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

    # Functional labels FIRST (largest first within a priority tier — the
    # DIP table is the hardest rectangle to pack and the most valuable),
    # then the legend, which has its own degrade path down to back silk.
    def _size_key(lb: dict) -> float:
        lines = str(lb.get("text", "")).split("\n")
        return -(len(lines) * max((len(ln) for ln in lines), default=0))

    labels = sorted(
        payload.get("labels") or [],
        key=lambda lb: (int(lb.get("priority", 2)), _size_key(lb),
                        str(lb.get("id", ""))),
    )
    for label in labels:
        if label.get("kind") == "pinout":
            placer.place_pinout(label)
        else:
            placer.place_label(label)

    legend = payload.get("legend") or {}
    lines = legend.get("lines") or []
    if lines:
        if not placer.place_legend(lines, float(legend.get("gap_mm", 0.3))):
            placer.dropped.append(
                {"id": "legend", "reason": "no clear silk strip on either side"}
            )

    with open(payload["result_path"], "w") as f:
        json.dump({"placed": placer.placed, "dropped": placer.dropped}, f, indent=1)

    board.Save(payload.get("output_path") or payload["pcb_path"])
    # Success sentinel (matches routing_board._PCBNEW_OK_SENTINEL) so a
    # pcbnew/wx teardown SIGSEGV after the Save is not read as a failure.
    print("__KICRAFT_PCBNEW_OK__")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
