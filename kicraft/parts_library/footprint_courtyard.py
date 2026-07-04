"""Footprint courtyard hygiene: a valid courtyard that clears the pad copper.

A footprint's courtyard is the keep-out rectangle that downstream placement and
board-outline code treats as the part's physical extent. Two failure modes are
repaired here, both at the footprint (the source), never by relaxing a gate:

- ``ensure_courtyard_clears_pads``: pads sitting AT or marginally inside the
  courtyard boundary (some edge-mount connectors/switches put pads right at the
  body front) make the composed board edge cut through the pad's
  copper-to-edge clearance band, tripping ``copper_edge_clearance`` DRC. The
  courtyard is grown to enclose every pad by ``min_clearance_mm``. Grow-only:
  never shrinks, never touches a footprint whose pads already clear.

- ``repair_malformed_courtyard``: imported footprints (easyEDA conversions in
  particular) sometimes draw a courtyard that does not form a closed area --
  e.g. two collinear ``fp_line`` segments drawn back and forth. KiCad flags it
  as ``malformed_courtyard`` and, far worse, ``GetCourtyard(...).BBox()``
  degenerates to a stroke-width sliver, so every placement consumer that reads
  the courtyard as the part's physical extent sees a near-zero part and packs
  neighbours into its body. The malformed layer is rebuilt as a rectangle
  around the part's pads + graphical body.

Applied when vendoring a new part (``add-part``) so library footprints are
well-formed at the source, and at the single point footprints are stamped onto
a board so already-cached parts are repaired in memory too.
"""
from __future__ import annotations

from typing import Any

# Pads must clear the courtyard boundary by at least this much. 0.2 mm matches
# the JLCPCB routed board-edge copper clearance the board outline is sized to.
DEFAULT_COURTYARD_PAD_CLEARANCE_MM = 0.2

# Tolerance so a pad sitting at exactly the clearance doesn't trigger a no-op
# "grow" from floating-point noise (and so an int-nm round-trip stays clear).
_TOL_MM = 1e-4


def _courtyard_layers(pcbnew: Any, footprint: Any) -> list[int]:
    present = []
    for layer in (pcbnew.F_CrtYd, pcbnew.B_CrtYd):
        try:
            poly = footprint.GetCourtyard(layer)
        except Exception:  # noqa: BLE001 -- API shape varies across KiCad
            poly = None
        if poly is not None and poly.OutlineCount() > 0:
            present.append(layer)
    return present


def _pads_union_box(footprint: Any):
    """Union of all pad bounding boxes as ``(left, top, right, bottom)`` in
    integer nm, or ``None`` if the footprint has no pads. Plain ints (no
    ``BOX2I``) to avoid swig object churn across repeated calls."""
    l = t = r = b = None
    for pad in list(footprint.Pads()):
        pb = pad.GetBoundingBox()
        pl, pt, pr, pb_ = pb.GetLeft(), pb.GetTop(), pb.GetRight(), pb.GetBottom()
        l = pl if l is None else min(l, pl)
        t = pt if t is None else min(t, pt)
        r = pr if r is None else max(r, pr)
        b = pb_ if b is None else max(b, pb_)
    return None if l is None else (l, t, r, b)


def _replace_courtyard_rect(
    pcbnew: Any,
    footprint: Any,
    layer: int,
    box: tuple[int, int, int, int],
) -> None:
    """Replace ``layer``'s courtyard graphics with the ``box`` rectangle
    (``(left, top, right, bottom)`` in nm), keeping the existing stroke width."""
    left, top, right, bottom = box
    width = next(
        (it.GetWidth() for it in footprint.GraphicalItems()
         if it.GetLayer() == layer),
        pcbnew.FromMM(0.05),
    )
    for it in [g for g in footprint.GraphicalItems() if g.GetLayer() == layer]:
        footprint.Remove(it)
    corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
    for i in range(4):
        seg = pcbnew.PCB_SHAPE(footprint)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetLayer(layer)
        seg.SetWidth(width)
        seg.SetStart(pcbnew.VECTOR2I(int(corners[i][0]), int(corners[i][1])))
        seg.SetEnd(pcbnew.VECTOR2I(int(corners[(i + 1) % 4][0]),
                                   int(corners[(i + 1) % 4][1])))
        footprint.Add(seg)


def ensure_courtyard_clears_pads(
    footprint: Any,
    *,
    min_clearance_mm: float = DEFAULT_COURTYARD_PAD_CLEARANCE_MM,
) -> bool:
    """Grow ``footprint``'s courtyard so it clears every pad by ``min_clearance_mm``.

    Operates on each courtyard layer present (F.CrtYd / B.CrtYd). Returns True if
    any courtyard was grown. Requires ``pcbnew``; the footprint is mutated in
    place (caller saves). Safe no-op for a footprint with no pads or no courtyard.
    """
    import pcbnew

    pads = _pads_union_box(footprint)
    if pads is None:
        return False
    pl, pt, pr, pb = pads
    clr = pcbnew.FromMM(float(min_clearance_mm))
    tol = pcbnew.FromMM(_TOL_MM)
    req_l, req_t = pl - clr, pt - clr
    req_r, req_b = pr + clr, pb + clr

    grew = False
    for layer in _courtyard_layers(pcbnew, footprint):
        cb = footprint.GetCourtyard(layer).BBox()
        new_l, new_t = min(cb.GetLeft(), req_l), min(cb.GetTop(), req_t)
        new_r, new_b = max(cb.GetRight(), req_r), max(cb.GetBottom(), req_b)
        if (cb.GetLeft() - new_l <= tol and new_r - cb.GetRight() <= tol
                and cb.GetTop() - new_t <= tol and new_b - cb.GetBottom() <= tol):
            continue  # already clears every pad by the margin on this layer
        # Replace this layer's courtyard graphics with the grown rectangle.
        _replace_courtyard_rect(pcbnew, footprint, layer, (new_l, new_t, new_r, new_b))
        grew = True
    return grew


# A courtyard whose bbox is thinner than this on either axis cannot enclose any
# real part body -- it is stroke residue from degenerate geometry (e.g. two
# collinear fp_lines drawn back and forth), not a keep-out area.
_MIN_COURTYARD_SIDE_MM = 0.2


def malformed_courtyard_layers(footprint: Any) -> list[int]:
    """Courtyard layers whose drawn graphics do NOT form a usable keep-out.

    A layer is malformed when it carries courtyard graphics but the polygon
    KiCad builds from them is empty (open / non-closing segments),
    self-intersecting, or degenerate (a sliver thinner than any real part).
    KiCad DRC reports these as ``malformed_courtyard``; worse, every placement
    consumer that reads ``GetCourtyard(...).BBox()`` as the part's physical
    extent sees a near-zero size and packs other parts into the part's body.
    """
    import pcbnew

    bad: list[int] = []
    min_side = pcbnew.FromMM(_MIN_COURTYARD_SIDE_MM)
    for layer in (pcbnew.F_CrtYd, pcbnew.B_CrtYd):
        if not any(g.GetLayer() == layer for g in footprint.GraphicalItems()):
            continue  # no courtyard drawn on this side -- nothing to judge
        try:
            poly = footprint.GetCourtyard(layer)
        except Exception:  # noqa: BLE001 -- API shape varies across KiCad
            poly = None
        if poly is None or poly.OutlineCount() == 0:
            bad.append(layer)
            continue
        try:
            if poly.IsSelfIntersecting():
                bad.append(layer)
                continue
        except AttributeError:
            pass  # older bindings: fall through to the degenerate-bbox check
        bb = poly.BBox()
        if bb.GetWidth() < min_side or bb.GetHeight() < min_side:
            bad.append(layer)
    return bad


def repair_malformed_courtyard(
    footprint: Any,
    *,
    margin_mm: float = DEFAULT_COURTYARD_PAD_CLEARANCE_MM,
) -> bool:
    """Rebuild every malformed courtyard layer as a valid keep-out rectangle.

    The replacement rectangle is the union of the footprint's pad copper and
    its full graphical bounding box (body silk/fab outlines -- the best
    available proxy for the part's physical extent when the drawn courtyard is
    unusable), expanded by ``margin_mm``. Returns True if any layer was
    rebuilt; the footprint is mutated in place (caller saves if persisting).
    """
    import pcbnew

    bad = malformed_courtyard_layers(footprint)
    if not bad:
        return False

    boxes: list[tuple[int, int, int, int]] = []
    pads = _pads_union_box(footprint)
    if pads is not None:
        boxes.append(pads)
    try:
        fb = footprint.GetBoundingBox(False, False)
    except TypeError:  # KiCad >= 9 dropped the second argument
        fb = footprint.GetBoundingBox(False)
    if fb.GetWidth() > 0 and fb.GetHeight() > 0:
        boxes.append((fb.GetLeft(), fb.GetTop(), fb.GetRight(), fb.GetBottom()))
    if not boxes:
        return False  # nothing to derive an extent from; leave it alone

    margin = pcbnew.FromMM(float(margin_mm))
    left = min(b[0] for b in boxes) - margin
    top = min(b[1] for b in boxes) - margin
    right = max(b[2] for b in boxes) + margin
    bottom = max(b[3] for b in boxes) + margin

    for layer in bad:
        _replace_courtyard_rect(pcbnew, footprint, layer, (left, top, right, bottom))
    return True


def courtyard_pad_clearance_mm(footprint: Any) -> float | None:
    """Smallest gap (mm) from any pad edge to the courtyard boundary, over all
    courtyard layers. Negative = a pad pokes outside the courtyard. None if the
    footprint has no courtyard or no pads. Diagnostic / test helper."""
    import pcbnew

    pads = _pads_union_box(footprint)
    layers = _courtyard_layers(pcbnew, footprint)
    if pads is None or not layers:
        return None
    pl, pt, pr, pb = pads
    worst = None
    for layer in layers:
        cb = footprint.GetCourtyard(layer).BBox()
        gaps = (
            pcbnew.ToMM(pl - cb.GetLeft()),
            pcbnew.ToMM(cb.GetRight() - pr),
            pcbnew.ToMM(pt - cb.GetTop()),
            pcbnew.ToMM(cb.GetBottom() - pb),
        )
        m = min(gaps)
        worst = m if worst is None else min(worst, m)
    return worst
