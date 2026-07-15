"""PR-N1 of docs/plans/shaped-compose-leaf-nesting.md: interior-hole
representation + the containment allowance in ``can_overlap_sparse``.

The seam-short regression the same-side veto exists for (RC2) is pinned
explicitly: any PARTIAL overlap between same-side leaves stays forbidden;
only full containment inside a genuinely enclosed interior hole is allowed.
No pcbnew needed -- blocker sets are constructed directly.
"""

from __future__ import annotations

from kicraft.autoplacer.brain.subcircuit_composer import (
    LeafBlockerSet,
    can_overlap_sparse,
    compute_interior_free_rects,
)
from kicraft.autoplacer.brain.types import Point


def _rect(x0, y0, x1, y1):
    return (Point(float(x0), float(y0)), Point(float(x1), float(y1)))


def _annulus_blocker_set(*, with_interior_decap=False, holes=()):
    """A 57x57 ring leaf: 12 pad rects around r~24, empty centre. Optionally
    an interior decap at r~14 -- deep enough inside that the maximal
    inscribed rect must dodge it (pre-N5 ring companions really sat INSIDE
    the annulus; see _place_companion_decaps history)."""
    import math

    pads = []
    for k in range(12):
        ang = 2.0 * math.pi * k / 12.0
        cx, cy = 28.5 + 24.0 * math.cos(ang), 28.5 + 24.0 * math.sin(ang)
        pads.append(_rect(cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5))
    comp_rects = {f"D{k+1}": pads[k] for k in range(12)}
    if with_interior_decap:
        comp_rects["C3"] = _rect(28.5 - 1.0, 13.0, 28.5 + 1.0, 15.5)  # r~14 top
        pads.append(comp_rects["C3"])
    return LeafBlockerSet(
        front_pads=tuple(pads),
        back_pads=(),
        tht_drills=(),
        leaf_outline=_rect(0, 0, 57, 57),
        component_rects=comp_rects,
        interior_free_rects=tuple(holes),
    )


def _small_front_leaf(w=16.0, h=14.0):
    """A guest leaf (MCU-like): front SMT pads spanning w x h."""
    return LeafBlockerSet(
        front_pads=(_rect(1, 1, w - 1, h - 1),),
        back_pads=(),
        tht_drills=(),
        leaf_outline=_rect(0, 0, w, h),
        component_rects={"U1": _rect(1, 1, w - 1, h - 1)},
    )


# --------------------------------------------------------------------------- #
# compute_interior_free_rects
# --------------------------------------------------------------------------- #

def test_annulus_yields_a_centered_interior_hole():
    bs = _annulus_blocker_set()
    holes = compute_interior_free_rects(bs, min_side_mm=8.0)
    assert holes, "the annulus interior must be found"
    (hmin, hmax) = holes[0]
    # Roughly centered and usefully large (ring inner clearance ~ r=21).
    cx, cy = (hmin.x + hmax.x) / 2.0, (hmin.y + hmax.y) / 2.0
    assert abs(cx - 28.5) < 3.0 and abs(cy - 28.5) < 3.0
    assert (hmax.x - hmin.x) >= 20.0 and (hmax.y - hmin.y) >= 20.0
    # The hole is INTERIOR: it must not touch the outline boundary.
    assert hmin.x > 2.0 and hmin.y > 2.0 and hmax.x < 55.0 and hmax.y < 55.0


def test_interior_decap_shrinks_or_shifts_the_hole():
    plain = compute_interior_free_rects(_annulus_blocker_set(), min_side_mm=8.0)
    with_c3 = compute_interior_free_rects(
        _annulus_blocker_set(with_interior_decap=True), min_side_mm=8.0
    )
    assert with_c3, "hole must survive an interior companion"
    area = lambda r: (r[1].x - r[0].x) * (r[1].y - r[0].y)  # noqa: E731
    assert area(with_c3[0]) < area(plain[0])
    # And the decap rect is not inside the reported hole.
    c3 = _rect(27.5, 13.0, 29.5, 15.5)
    (hmin, hmax) = with_c3[0]
    assert not (
        c3[0].x >= hmin.x and c3[1].x <= hmax.x
        and c3[0].y >= hmin.y and c3[1].y <= hmax.y
    )


def test_open_bay_is_not_a_hole():
    # A U-shape: empty space reachable from the boundary is OUTSIDE, not a
    # hole -- nesting there would abut open board edge, not enclosed FR4.
    bs = LeafBlockerSet(
        front_pads=(
            _rect(0, 0, 5, 40),    # left wall
            _rect(35, 0, 40, 40),  # right wall
            _rect(0, 35, 40, 40),  # bottom wall -- top edge (y=0) stays open
        ),
        back_pads=(),
        tht_drills=(),
        leaf_outline=_rect(0, 0, 40, 40),
    )
    assert compute_interior_free_rects(bs, min_side_mm=8.0) == ()


def test_min_side_filters_small_holes():
    bs = _annulus_blocker_set()
    assert compute_interior_free_rects(bs, min_side_mm=60.0) == ()


def test_hole_computation_is_deterministic():
    a = compute_interior_free_rects(_annulus_blocker_set(), min_side_mm=8.0)
    b = compute_interior_free_rects(_annulus_blocker_set(), min_side_mm=8.0)
    assert a == b


# --------------------------------------------------------------------------- #
# can_overlap_sparse containment allowance
# --------------------------------------------------------------------------- #

def _holes_for(bs):
    return compute_interior_free_rects(bs, min_side_mm=8.0)


def test_nested_guest_inside_hole_is_allowed():
    host = _annulus_blocker_set()
    host = _annulus_blocker_set(holes=_holes_for(host))
    guest = _small_front_leaf()
    # Host at world origin (20, 20); guest centered in the hole:
    # hole centre ~ (48.5, 48.5) world; guest is 16x14.
    assert can_overlap_sparse(
        host, Point(20.0, 20.0), 0.0,
        guest, Point(20.0 + 28.5 - 8.0, 20.0 + 28.5 - 7.0), 0.0,
    ) is True


def test_partial_overlap_stays_forbidden_seam_regression():
    host = _annulus_blocker_set(holes=_holes_for(_annulus_blocker_set()))
    guest = _small_front_leaf()
    # Guest straddling the ring copper (half in, half out) -- the exact
    # seam-adjacency RC2 exists to forbid.
    assert can_overlap_sparse(
        host, Point(20.0, 20.0), 0.0,
        guest, Point(20.0 + 50.0, 20.0 + 22.0), 0.0,
    ) is False
    # And fully outside/side-by-side same-side leaves remain incompatible
    # (bbox overlap semantics unchanged for non-nested pairs).
    assert can_overlap_sparse(
        host, Point(20.0, 20.0), 0.0,
        guest, Point(20.0 + 20.0, 20.0 + 45.0), 0.0,
    ) is False


def test_no_holes_means_veto_unchanged():
    host = _annulus_blocker_set()  # interior_free_rects deliberately empty
    guest = _small_front_leaf()
    assert can_overlap_sparse(
        host, Point(20.0, 20.0), 0.0,
        guest, Point(20.0 + 28.5 - 8.0, 20.0 + 28.5 - 7.0), 0.0,
    ) is False


def test_non_cardinal_host_rotation_is_conservative():
    host = _annulus_blocker_set(holes=_holes_for(_annulus_blocker_set()))
    guest = _small_front_leaf()
    assert can_overlap_sparse(
        host, Point(20.0, 20.0), 45.0,
        guest, Point(20.0 + 28.5 - 8.0, 20.0 + 28.5 - 7.0), 0.0,
    ) is False


def test_guest_bigger_than_hole_is_forbidden():
    host = _annulus_blocker_set(holes=_holes_for(_annulus_blocker_set()))
    guest = _small_front_leaf(w=40.0, h=40.0)  # cannot fit the ~26mm hole
    assert can_overlap_sparse(
        host, Point(20.0, 20.0), 0.0,
        guest, Point(20.0 + 28.5 - 20.0, 20.0 + 28.5 - 20.0), 0.0,
    ) is False


# --------------------------------------------------------------------------- #
# PR-N2: the solver nest-proposal pass (Step 8.8)
# --------------------------------------------------------------------------- #

from kicraft.autoplacer.brain.placement_solver import PlacementSolver  # noqa: E402
from kicraft.autoplacer.brain.types import BoardState, Component, Layer  # noqa: E402


def _block(ref, *, pos, width, height, blocker_set, locked=False, rotation=0.0):
    comp = Component(
        ref=ref, value=ref, pos=Point(pos.x, pos.y), rotation=rotation,
        layer=Layer.FRONT, width_mm=width, height_mm=height,
        kind="subcircuit", locked=locked, body_center=Point(pos.x, pos.y),
    )
    comp.block_blocker_set = blocker_set
    comp.block_artifact_origin_offset = Point(width / 2.0, height / 2.0)
    return comp


def _nest_fixture(*, zones=None, cfg_extra=None):
    host_bs = _annulus_blocker_set(holes=_holes_for(_annulus_blocker_set()))
    guest_bs = _small_front_leaf()
    host = _block("BLK_RING", pos=Point(60.0, 60.0), width=57.0, height=57.0,
                  blocker_set=host_bs)
    guest = _block("BLK_MCU", pos=Point(150.0, 60.0), width=16.0, height=14.0,
                   blocker_set=guest_bs)
    comps = {host.ref: host, guest.ref: guest}
    state = BoardState(
        components=comps, nets={}, traces=[], vias=[], silkscreen=[],
        board_outline=(Point(0.0, 0.0), Point(200.0, 200.0)),
    )
    cfg = {
        "leaf_nesting": "auto",
        "board_outline": {"shape": "circle", "size_mm": 60.0},
        "component_zones": zones or {},
    }
    cfg.update(cfg_extra or {})
    return PlacementSolver(state, config=cfg, seed=0), comps, host, guest


def _guest_inside_host_bbox(guest, host):
    g_tl, g_br = guest.bbox()
    h_tl, h_br = host.bbox()
    return (g_tl.x >= h_tl.x and g_tl.y >= h_tl.y
            and g_br.x <= h_br.x and g_br.y <= h_br.y)


def test_nest_pass_nests_guest_into_annulus_hole():
    solver, comps, host, guest = _nest_fixture()
    solver._nest_blocks_in_interior_holes(comps)
    assert guest.block_nested_anchor == "BLK_RING"
    assert guest.locked and host.locked
    assert _guest_inside_host_bbox(guest, host)
    # Landed near the hole centre (host centre for a symmetric annulus).
    assert abs(guest.pos.x - 60.0) < 4.0 and abs(guest.pos.y - 60.0) < 4.0


def test_nest_pass_skips_strict_edge_zoned_guest():
    solver, comps, host, guest = _nest_fixture(
        zones={"BLK_MCU": {"edge": "bottom"}}
    )
    solver._nest_blocks_in_interior_holes(comps)
    assert guest.block_nested_anchor is None
    assert not guest.locked and not host.locked
    assert guest.pos.x == 150.0  # untouched


def test_nest_pass_auto_gate_needs_shaped_outline():
    solver, comps, host, guest = _nest_fixture(
        cfg_extra={"board_outline": None}
    )
    solver._nest_blocks_in_interior_holes(comps)
    assert guest.block_nested_anchor is None and guest.pos.x == 150.0


def test_nest_pass_off_switch():
    solver, comps, host, guest = _nest_fixture(
        cfg_extra={"leaf_nesting": "off"}
    )
    solver._nest_blocks_in_interior_holes(comps)
    assert guest.block_nested_anchor is None and guest.pos.x == 150.0


def test_nest_survives_overlap_and_courtyard_passes():
    solver, comps, host, guest = _nest_fixture()
    solver._nest_blocks_in_interior_holes(comps)
    assert guest.block_nested_anchor == "BLK_RING"
    nested_pos = Point(guest.pos.x, guest.pos.y)

    solver._resolve_overlaps(comps)
    unresolved = solver._resolve_courtyard_overlaps(comps)

    assert (guest.pos.x, guest.pos.y) == (nested_pos.x, nested_pos.y), (
        "later passes must not drift a locked nested guest"
    )
    assert _guest_inside_host_bbox(guest, host)
    assert unresolved == 0, (
        "a nested pair must not be reported as an unresolved courtyard overlap"
    )


# --------------------------------------------------------------------------- #
# PR-N4: edge-pin demotion candidates + stranded-gate consistency
# --------------------------------------------------------------------------- #

def _fake_artifact(name, refs):
    from types import SimpleNamespace

    return SimpleNamespace(
        layout=SimpleNamespace(components={r: object() for r in refs}),
        sheet_name=name,
    )


def test_edge_demotion_candidates_finds_nestable_pinned_leaf(monkeypatch):
    from kicraft.autoplacer.brain import subcircuit_composer as sc
    from kicraft.cli.compose_subcircuits import _edge_demotion_candidates

    ring_art = _fake_artifact("LED RING", ["D1", "D2"])
    mcu_art = _fake_artifact("MCU", ["U1", "J1"])
    ring_bs = _annulus_blocker_set(holes=_holes_for(_annulus_blocker_set()))
    mcu_bs = _small_front_leaf()
    monkeypatch.setattr(
        sc, "extract_leaf_blocker_set",
        lambda art, cfg=None: ring_bs if art is ring_art else mcu_bs,
    )

    cfg = {"component_zones": {"J1": {"edge": "bottom"}}, "nest_margin_mm": 1.5}
    assert _edge_demotion_candidates([ring_art, mcu_art], cfg) == ["J1"]

    # No holes anywhere -> nothing to demote into.
    monkeypatch.setattr(
        sc, "extract_leaf_blocker_set",
        lambda art, cfg=None: _annulus_blocker_set() if art is ring_art else mcu_bs,
    )
    assert _edge_demotion_candidates([ring_art, mcu_art], cfg) == []

    # Pin on a ref whose leaf is TOO BIG for the hole -> not demoted.
    big_bs = _small_front_leaf(w=40.0, h=40.0)
    monkeypatch.setattr(
        sc, "extract_leaf_blocker_set",
        lambda art, cfg=None: ring_bs if art is ring_art else big_bs,
    )
    assert _edge_demotion_candidates([ring_art, mcu_art], cfg) == []

    # No zones -> empty, and never touches geometry.
    assert _edge_demotion_candidates([ring_art, mcu_art], {}) == []


def test_stranded_gate_skips_demoted_refs(tmp_path, monkeypatch):
    import json as _json

    from kicraft.autoplacer.brain import connector_edge_gap as ceg
    from kicraft.design.cli_app import _connector_stranded_refs

    pcb = tmp_path / "BOARD.kicad_pcb"
    pcb.write_text("(kicad_pcb)\n")
    (tmp_path / "BOARD_autoplacer.json").write_text(
        _json.dumps({"component_zones": {"J1": {"edge": "bottom"}}})
    )

    from types import SimpleNamespace
    fake_gap = SimpleNamespace(ref="J1", gap_mm=-5.0, edge="bottom")
    monkeypatch.setattr(ceg, "connector_edge_gaps", lambda *a, **k: [fake_gap])

    # Without the demotion record the inboard connector is flagged.
    assert _connector_stranded_refs(pcb) == [
        "connector_stranded:J1@-5.00mm(bottom)"
    ]

    # With the record (winner came from the demoted wave) it is skipped.
    d = tmp_path / ".experiments" / "subcircuits" / "parent__x"
    d.mkdir(parents=True)
    (d / "edge_pins_demoted.json").write_text(_json.dumps({"refs": ["J1"]}))
    assert _connector_stranded_refs(pcb) == []


# --------------------------------------------------------------------------- #
# PR-N5 (representation legs): standoff decoupled from gap sealing (r2) and
# subdivided trace rects for the hole computation (r1)
# --------------------------------------------------------------------------- #

def test_hole_grows_with_standoff_decoupled_from_sealing():
    # The old rule excluded the whole min_side/2 closing band from the hole;
    # standoff_mm=min_side/2 reproduces it. The decoupled default (1.0) must
    # yield a strictly larger hole whose edge sits ~standoff from copper,
    # while the inter-member gaps stay sealed (a hole is found at all, and it
    # stays enclosed -- nothing leaks past the annulus band).
    bs = _annulus_blocker_set()
    old_rule = compute_interior_free_rects(bs, min_side_mm=8.0, standoff_mm=4.0)
    new_rule = compute_interior_free_rects(bs, min_side_mm=8.0, standoff_mm=1.0)
    assert old_rule and new_rule
    area = lambda r: (r[1].x - r[0].x) * (r[1].y - r[0].y)  # noqa: E731
    assert area(new_rule[0]) > area(old_rule[0])
    (hmin, hmax) = new_rule[0]
    # The maximal rect is corner-limited by the DIAGONAL pads (30/60 deg,
    # inner corners near r~19), so absolute reach stays modest -- but both
    # sides must now clear what the old 4mm band allowed, and the hole must
    # never cross into the band itself (pads inner faces at r=21.5, minus
    # the 1mm standoff).
    assert (hmax.x - hmin.x) >= 28.0 and (hmax.y - hmin.y) >= 28.0
    assert hmin.x >= 6.0 and hmin.y >= 6.0 and hmax.x <= 51.0 and hmax.y <= 51.0


def test_diagonal_gaps_stay_sealed_at_tight_standoff():
    # Sealing is reachability, not clearance: even with the 1mm standoff the
    # ~7mm inter-member gaps (including the DIAGONAL ones, which a true
    # morphological closing fails to bridge) must not leak the interior.
    holes = compute_interior_free_rects(
        _annulus_blocker_set(), min_side_mm=8.0, standoff_mm=1.0
    )
    assert len(holes) == 1
    (hmin, hmax) = holes[0]
    # Enclosed: the hole cannot reach the outline boundary region.
    assert hmin.x > 2.0 and hmin.y > 2.0 and hmax.x < 55.0 and hmax.y < 55.0


def _diag_chord_traces():
    import math

    from kicraft.autoplacer.brain.types import TraceSegment

    # A 45-degree chord between the members at 30 and 60 degrees -- exactly
    # the rotated-LED hop whose one-AABB square bit r=12.8 deep into the
    # real 1/601 interior while its copper stayed in the band.
    a30 = (28.5 + 24.0 * math.cos(math.radians(30.0)),
           28.5 + 24.0 * math.sin(math.radians(30.0)))
    a60 = (28.5 + 24.0 * math.cos(math.radians(60.0)),
           28.5 + 24.0 * math.sin(math.radians(60.0)))
    return [
        TraceSegment(
            start=Point(*a30), end=Point(*a60),
            layer=Layer.FRONT, net="+5V", width_mm=0.25,
        )
    ]


def test_subdivided_trace_rects_hug_the_diagonal():
    from kicraft.autoplacer.brain.subcircuit_composer import (
        _subdivided_trace_rects,
        _trace_local_bbox,
    )

    traces = _diag_chord_traces()
    seg = traces[0]
    coarse = _trace_local_bbox(seg, margin_mm=0.5)
    front, back = _subdivided_trace_rects(traces, margin_mm=0.5)
    assert back == [] and len(front) > 10
    # Union covers the trace endpoints (copper superset at the tips)...
    def covers(rects, x, y):
        return any(
            r[0].x <= x <= r[1].x and r[0].y <= y <= r[1].y for r in rects
        )
    assert covers(front, seg.start.x, seg.start.y)
    assert covers(front, seg.end.x, seg.end.y)
    # ...but the coarse AABB's off-diagonal corner region (the part that
    # reaches into the ring interior) is NOT claimed.
    probe_x, probe_y = coarse[0].x + 1.0, coarse[0].y + 1.0
    assert covers([coarse], probe_x, probe_y)
    assert not covers(front, probe_x, probe_y)


def test_straight_trace_subdivision_is_equivalent():
    from kicraft.autoplacer.brain.types import TraceSegment
    from kicraft.autoplacer.brain.subcircuit_composer import (
        _subdivided_trace_rects,
    )

    seg = TraceSegment(
        start=Point(10.0, 20.0), end=Point(40.0, 20.0),
        layer=Layer.FRONT, net="D", width_mm=0.25,
    )
    front, _ = _subdivided_trace_rects([seg], margin_mm=0.5)
    # The union of the pieces equals the single AABB for an axis-aligned run.
    assert min(r[0].x for r in front) == 10.0 - (0.125 + 0.5)
    assert max(r[1].x for r in front) == 40.0 + (0.125 + 0.5)
    assert all(r[0].y == 20.0 - 0.625 and r[1].y == 20.0 + 0.625 for r in front)


def test_diag_chord_no_longer_eats_the_hole():
    # One diagonal chord in the band, represented coarse vs subdivided: the
    # coarse single-AABB square bites deep into the interior; the subdivided
    # representation must give back (almost all of) the chord-free hole.
    from dataclasses import replace as _replace

    from kicraft.autoplacer.brain.subcircuit_composer import (
        _subdivided_trace_rects,
        _trace_local_bbox,
    )

    bs = _annulus_blocker_set()
    traces = _diag_chord_traces()
    coarse_bs = _replace(
        bs, front_traces=(_trace_local_bbox(traces[0], margin_mm=0.5),)
    )
    tight_front, _ = _subdivided_trace_rects(traces, margin_mm=0.5)
    tight_bs = _replace(bs, front_traces=tuple(tight_front))

    area = lambda r: (r[1].x - r[0].x) * (r[1].y - r[0].y)  # noqa: E731
    plain = compute_interior_free_rects(bs, min_side_mm=8.0)
    coarse = compute_interior_free_rects(coarse_bs, min_side_mm=8.0)
    tight = compute_interior_free_rects(tight_bs, min_side_mm=8.0)
    assert plain and coarse and tight
    assert area(coarse[0]) < area(tight[0]) <= area(plain[0])


def test_nest_pass_centers_occupied_bbox_not_content_pos():
    # PR-N5 landing fix: ``pos`` is the CONTENT centre, but containment
    # tests the OCCUPIED bbox, whose centre can sit off by a couple of mm
    # (traces/pads inflate asymmetrically). At tight hole slack the naive
    # pos-centred landing fails even though the guest fits -- the pass must
    # re-land with the occupied bbox centred (the real 1/601 guest has
    # ~0.3 mm of slack per side).
    guest_bs = LeafBlockerSet(
        front_pads=(_rect(1, 5, 15, 13),),  # occupied centre (8, 9)...
        back_pads=(),
        tht_drills=(),
        leaf_outline=_rect(0, 0, 16, 14),   # ...content centre (8, 7)
        component_rects={"U1": _rect(1, 5, 15, 13)},
    )
    # A hole with 0.2 mm slack per side around occupied + 2x1.0 margin,
    # parked in the annulus' clear interior (leaf-local coords).
    tight_hole = _rect(20.0, 22.0, 20.0 + 14.0 + 2.0 + 0.4, 22.0 + 8.0 + 2.0 + 0.4)
    host_bs = _annulus_blocker_set(holes=(tight_hole,))
    host = _block("BLK_RING", pos=Point(60.0, 60.0), width=57.0, height=57.0,
                  blocker_set=host_bs)
    guest = _block("BLK_MCU", pos=Point(150.0, 60.0), width=16.0, height=14.0,
                   blocker_set=guest_bs)
    comps = {host.ref: host, guest.ref: guest}
    state = BoardState(
        components=comps, nets={}, traces=[], vias=[], silkscreen=[],
        board_outline=(Point(0.0, 0.0), Point(200.0, 200.0)),
    )
    solver = PlacementSolver(state, config={
        "leaf_nesting": "auto",
        "board_outline": {"shape": "circle", "size_mm": 60.0},
    }, seed=0)
    solver._nest_blocks_in_interior_holes(comps)
    assert guest.block_nested_anchor == "BLK_RING", (
        "occupied-bbox-centred re-landing must rescue a tight-slack fit"
    )
    # The landing centred the OCCUPIED bbox (not pos) in the hole: hole
    # world centre = host origin (60-28.5=31.5 each axis) + hole centre.
    hole_cx = 31.5 + (tight_hole[0].x + tight_hole[1].x) / 2.0
    hole_cy = 31.5 + (tight_hole[0].y + tight_hole[1].y) / 2.0
    # guest occupied centre in world = pos + (occupied centre - content centre)
    assert abs((guest.pos.x + 0.0) - hole_cx) < 0.01
    assert abs((guest.pos.y + 2.0) - hole_cy) < 0.01


def test_nest_pass_oversized_guest_left_alone():
    host_bs = _annulus_blocker_set(holes=_holes_for(_annulus_blocker_set()))
    big_bs = _small_front_leaf(w=40.0, h=40.0)
    host = _block("BLK_RING", pos=Point(60.0, 60.0), width=57.0, height=57.0,
                  blocker_set=host_bs)
    guest = _block("BLK_BIG", pos=Point(150.0, 60.0), width=40.0, height=40.0,
                   blocker_set=big_bs)
    comps = {host.ref: host, guest.ref: guest}
    state = BoardState(
        components=comps, nets={}, traces=[], vias=[], silkscreen=[],
        board_outline=(Point(0.0, 0.0), Point(200.0, 200.0)),
    )
    solver = PlacementSolver(state, config={
        "leaf_nesting": "auto",
        "board_outline": {"shape": "circle", "size_mm": 60.0},
    }, seed=0)
    solver._nest_blocks_in_interior_holes(comps)
    assert guest.block_nested_anchor is None
    assert guest.pos.x == 150.0 and not guest.locked and not host.locked
