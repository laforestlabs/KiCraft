from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from kicraft.autoplacer.brain.array_placement import leaf_is_fully_array
from kicraft.autoplacer.brain.leaf_geometry import (
    grow_leaf_outline_to_contain_placement,
    repair_leaf_placement_legality,
)
from kicraft.autoplacer.brain.subcircuit_artifacts import resolve_artifact_paths
from kicraft.autoplacer.brain.subcircuit_extractor import ExtractedSubcircuitBoard
from kicraft.autoplacer.brain.subcircuit_render_diagnostics import (
    LeafStageOpts,
    generate_leaf_diagnostic_artifacts,
    generate_stage_diagnostic_artifacts,
    promote_to_round_snapshot,
)
from kicraft.autoplacer.brain.types import Component, Point
from kicraft.autoplacer.routing_board import (
    import_routed_copper,
    validate_routed_board,
)
from kicraft.autoplacer.hardware.adapter import KiCadAdapter
from kicraft.autoplacer.kicad_routing_tools import route_with_kicad_routing_tools


def _internal_net_verdict(
    validation: dict[str, Any],
    extraction: ExtractedSubcircuitBoard,
) -> tuple[list[str], list[str]]:
    """``(routed_internal_nets, failed_internal_nets)`` from the DRC that just ran.

    The success path used to HARDCODE "everything routed, nothing failed", so
    per-net truth existed only inside the acceptance DRC and never reached the
    round record -- the retry loop could not react per net, and the round log
    said a net was routed while routing had silently added zero copper on it
    (dense-soc-leaf-unconnected-plan P1.5). When the report has unconnected items
    but no parsable net names (format drift), nothing is claimed either way.
    """
    internal = set(extraction.internal_net_names)
    drc = validation.get("drc", {}) or {}
    nets = {str(n) for n in (drc.get("unconnected_nets", []) or [])}
    if not nets and int(drc.get("unconnected", 0) or 0) > 0:
        return [], []
    return sorted(internal - nets), sorted(internal & nets)


def _resolve_breakout_specs(cfg: dict[str, Any]) -> list:
    """Build :class:`BreakoutSpec` objects from ``cfg['breakout_specs']``.

    Each entry is a dict ``{ref, pad, waypoints?, length_mm?, width_mm?,
    layer?, via_at_end?}``. Returns ``[]`` when none are configured. Specs whose
    footprint isn't on the leaf are harmlessly skipped by ``add_breakout_stubs``.
    """
    raw = cfg.get("breakout_specs")
    if not raw:
        return []
    from kicraft.autoplacer.brain.breakout_stubs import BreakoutSpec

    specs = []
    for d in raw:
        if not d.get("ref") or d.get("pad") is None:
            continue
        specs.append(
            BreakoutSpec(
                ref=str(d["ref"]),
                pad=str(d["pad"]),
                waypoints=[tuple(p) for p in d.get("waypoints", [])],
                length_mm=float(d.get("length_mm", 1.5)),
                width_mm=d.get("width_mm"),
                layer=d.get("layer", "F.Cu"),
                via_at_end=bool(d.get("via_at_end", False)),
            )
        )
    return specs


def _silk_for_leaf(
    extraction: ExtractedSubcircuitBoard,
    components: dict[str, Component],
    cfg: dict[str, Any],
    *,
    traces=None,
    vias=None,
) -> list:
    """Compute leaf-local rounded-rect silk + optional label for stamping.

    Lazy-imports the solver helpers so this module stays cheap to import
    in pipelines that don't actually stamp boards. Returns ``[]`` when
    components are empty or no project ``group_labels`` entry matches.

    Pass ``traces`` and ``vias`` post-route so the silk hugs every visible
    piece of copper (courtyards + pad copper + tracks + vias). Pre-route
    callers can omit them; the silk then hugs courtyards + pad copper only.
    """
    if not components:
        return []
    from kicraft.autoplacer.brain.subcircuit_solver import (
        _build_leaf_silkscreen,
        _compute_component_bbox,
    )
    bbox = _compute_component_bbox(components, traces=traces, vias=vias)
    return _build_leaf_silkscreen(components, bbox, extraction, cfg)


def _outline_around_geometry(
    components: dict[str, Component],
    cfg: dict[str, Any],
    *,
    traces=None,
    vias=None,
) -> tuple[Point, Point] | None:
    """Compute a tight Edge.Cuts outline hugging the same bbox as the silk.

    The silk poly is drawn at ``bbox ± silkscreen_margin_mm`` (default 0.5 mm).
    Edge.Cuts defaults to ``edge_margin = silk_margin`` so the yellow board
    outline sits exactly on top of the white silk outline (zero gap).
    Callers can still override via ``leaf_edge_margin_mm`` if they want a
    visible gap. Returns ``None`` for empty leaves so the caller can keep
    the original outline.
    """
    if not components:
        return None
    from kicraft.autoplacer.brain.subcircuit_solver import _compute_component_bbox
    bbox = _compute_component_bbox(components, traces=traces, vias=vias)
    silk_margin = float(cfg.get("silkscreen_margin_mm", 0.5))
    edge_margin = float(cfg.get("leaf_edge_margin_mm", silk_margin))
    return (
        Point(bbox["min_x"] - edge_margin, bbox["min_y"] - edge_margin),
        Point(bbox["max_x"] + edge_margin, bbox["max_y"] + edge_margin),
    )


def _center_on_leaf_page(
    tl: Point, br: Point, cfg: dict[str, Any]
) -> tuple[Point, tuple[Point, Point]]:
    """Translation that centers the ``[tl, br]`` content box on a standard A4 leaf
    page, plus the resulting centered Edge.Cuts outline.

    Mirrors the parent's A4 centering (compose_subcircuits._stamp_parent_board) so a
    standalone leaf opens centered in the title block instead of crammed against the
    top-left origin. Position on the page is free: the parent composer re-bases each
    leaf to its own outline origin on load (subcircuit_instances._layout_from_artifact_payload),
    so a centered leaf composes identically to one anchored at (0, 0). Returns
    ``(delta, (new_tl, new_br))``.
    """
    page_w = float(cfg.get("leaf_page_width_mm", cfg.get("parent_page_width_mm", 297.0)))
    page_h = float(cfg.get("leaf_page_height_mm", cfg.get("parent_page_height_mm", 210.0)))
    w = br.x - tl.x
    h = br.y - tl.y
    dx = (page_w - w) / 2.0 - tl.x
    dy = (page_h - h) / 2.0 - tl.y
    return Point(dx, dy), (Point(tl.x + dx, tl.y + dy), Point(br.x + dx, br.y + dy))


def _deterministic_route_signature(board_state: Any, cfg: dict[str, Any]) -> str:
    """Stable hash of placement and non-timeout KRT routing options."""
    comps = sorted(board_state.components.values(), key=lambda c: c.ref)
    placement = [
        (c.ref, round(c.pos.x, 3), round(c.pos.y, 3),
         round(float(c.rotation), 1), int(c.layer))
        for c in comps
    ]
    tl, br = board_state.board_outline
    route_keys = {
        k: v
        for k, v in cfg.items()
        if k.startswith("kicad_routing_tools_") and "timeout" not in k
    }
    blob = json.dumps(
        [
            placement,
            [round(tl.x, 3), round(tl.y, 3), round(br.x, 3), round(br.y, 3)],
            "kicad-routing-tools:0.20.2@3ceb773722bea67aa3685e7ee430c0c0d17ef38d",
            route_keys,
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _array_inrow_stamp_stats(specs: list, skipped_keys: list) -> tuple[int, int]:
    """(in_row_ties, in_row_skipped) for an array leaf's daisy-chain.

    In-row hops are single straight ties (one waypoint); serpentine row-turn hops
    are L/Z routes (multiple waypoints) OR were never emitted at all (no edge
    channel fit). Either way a turn may legitimately fall back to the autorouter, so
    it must NOT count against the stamp gate -- only the in-row hops, which on a
    healthy oriented array essentially always stamp, are measured. ``skipped_keys``
    are the ``"ref.pad:reason"`` strings the stamp guard dropped.
    """
    in_row = [s for s in specs if len(getattr(s, "waypoints", []) or []) == 1]
    keys = {f"{s.ref}.{s.pad}" for s in in_row}
    n_skipped = sum(1 for k in skipped_keys if str(k).split(":", 1)[0] in keys)
    return len(in_row), n_skipped


def array_stamp_gate_tripped(specs: list, skipped_keys: list, cfg: dict[str, Any]) -> bool:
    """True when an array leaf pre-stamped too few of its IN-ROW data hops.

    A healthy array leaf stamps almost every straight in-row DOUT->DIN hop, so
    the autorouter has next to nothing left. A collapsed in-row stamp rate means the
    channel is obstructed (a foreign pad in the data lane) and the autorouter will be
    handed the whole chain -- which it cannot finish on a dense array before the
    build's wall-clock cap kills it. Row-turn hops are excluded (they may
    legitimately go to the autorouter), and the gate is floored at a minimum in-row
    tie count, so a small array -- e.g. a 2x2 whose only un-stamped hop is its row
    turn -- never false-fails.
    """
    if not cfg.get("array_stamp_gate_enabled", True):
        return False
    n_in_row, n_skipped = _array_inrow_stamp_stats(specs, skipped_keys)
    if not n_in_row:
        return False
    min_ties = int(cfg.get("array_min_data_ties_for_gate", 6))
    min_rate = float(cfg.get("array_min_stamp_rate", 0.5))
    if n_in_row < min_ties:
        return False
    return (n_in_row - n_skipped) / n_in_row < min_rate


def route_local_subcircuit(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
    *,
    generate_diagnostics: bool = True,
    round_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    fast_smoke_mode = bool(cfg.get("subcircuit_fast_smoke_mode", False))
    # Intermediate per-round diagnostics: headless build/worker runs set
    # subcircuit_render_intermediate=False to skip per-round PNG/DRC renders;
    # the winning round's canonical renders are produced downstream and are
    # unaffected by this flag. Defaults True for interactive/debug parity.
    render_intermediate = bool(cfg.get("subcircuit_render_intermediate", True))
    render_pre_route_board_views = bool(
        cfg.get("subcircuit_render_pre_route_board_views", not fast_smoke_mode)
    )
    render_routed_board_views = bool(
        cfg.get("subcircuit_render_routed_board_views", True)
    )
    render_pre_route_drc_overlay = bool(
        cfg.get("subcircuit_render_pre_route_drc_overlay", not fast_smoke_mode)
    )
    render_routed_drc_overlay = bool(
        cfg.get("subcircuit_render_routed_drc_overlay", not fast_smoke_mode)
    )
    write_pre_route_drc_json = bool(
        cfg.get("subcircuit_write_pre_route_drc_json", not fast_smoke_mode)
    )
    write_routed_drc_json = bool(cfg.get("subcircuit_write_routed_drc_json", True))
    write_pre_route_drc_report = bool(
        cfg.get("subcircuit_write_pre_route_drc_report", not fast_smoke_mode)
    )
    write_routed_drc_report = bool(cfg.get("subcircuit_write_routed_drc_report", True))
    build_comparison_contact_sheet = bool(
        cfg.get("subcircuit_build_comparison_contact_sheet", not fast_smoke_mode)
    )
    # A leaf must fully route the on-leaf span of EVERY net that touches it,
    # including nets that also connect to other sheets -- their inter-sheet hop
    # is completed at the parent stage. extraction.local_state.nets already holds
    # those external nets filtered to their on-leaf pads (see _partition_nets /
    # _filter_net_to_components), so routing the board connects them locally.
    # Only a leaf with no net having >=2 pads on it (e.g. a pass-through
    # connector where every net has a single on-leaf pad) is genuinely trivial.
    routable_on_leaf_nets = [
        name
        for name, net in extraction.local_state.nets.items()
        if len(net.pad_refs) >= 2
    ]
    if not routable_on_leaf_nets:
        # Trivial leaf: nothing to route, but we still stamp the placed
        # components onto a real PCB so the leaf flows through the same
        # workflow as every other leaf -- pin_best_leaves can promote it,
        # the GUI snapshot picker shows its rounds, and the composer reads
        # uniformly from leaf_routed.kicad_pcb.
        return _stamp_trivial_leaf(
            extraction=extraction,
            solved_components=solved_components,
            cfg=cfg,
            round_index=round_index,
            generate_diagnostics=generate_diagnostics,
            render_intermediate=render_intermediate,
            fast_smoke_mode=fast_smoke_mode,
        )

    artifact_paths = resolve_artifact_paths(
        Path(extraction.subcircuit.schematic_path).parent,
        extraction.subcircuit.id,
    )
    pre_route_board = Path(artifact_paths.artifact_dir) / "leaf_placed.kicad_pcb"
    routed_board = Path(artifact_paths.artifact_dir) / "leaf_routed.kicad_pcb"
    illegal_board = Path(artifact_paths.artifact_dir) / "leaf_illegal_pre_stamp.kicad_pcb"

    route_timing: dict[str, float] = {}
    route_total_start = time.monotonic()

    legality_start = time.monotonic()
    repaired_components, legality_repair = repair_leaf_placement_legality(
        extraction,
        solved_components,
        cfg,
    )
    route_timing["legality_repair_s"] = round(
        max(0.0, time.monotonic() - legality_start), 3
    )

    # Reframe recovery: a placement that is internally legal (no overlapping
    # copper) but whose pads spill past the content-sized canvas is a
    # canvas-too-small problem, not a bad placement -- e.g. a column of stacked
    # THT headers taller than the roughly-square content canvas (KC-99A9M8, an
    # Arduino-shield header sheet). Grow the leaf outline to bound the placement
    # and re-legalize rather than failing the whole build over it. Only kicks in
    # for a *pure* canvas overflow (zero overlaps, pads outside only): a real
    # overlap still fails, unchanged.
    if not legality_repair.get("resolved", False):
        _diag = legality_repair.get("diagnostics", {}) or {}
        _pure_canvas_overflow = (
            int(_diag.get("overlap_count", 0) or 0) == 0
            and int(_diag.get("pad_outside_count", 0) or 0) > 0
        )
        if _pure_canvas_overflow and grow_leaf_outline_to_contain_placement(
            extraction, repaired_components, cfg
        ):
            repaired_components, legality_repair = repair_leaf_placement_legality(
                extraction, solved_components, cfg
            )
            if legality_repair.get("resolved", False):
                print(
                    "  Leaf canvas reframed to contain a legal-but-oversized "
                    "placement -> outline "
                    f"{extraction.local_state.board_width:.1f}x"
                    f"{extraction.local_state.board_height:.1f}mm "
                    f"(was overflowing by {_diag.get('pad_outside_count')} pad(s))"
                )

    source_pcb = Path(cfg.get("subcircuit_route_source_pcb", cfg.get("pcb_path", "")))
    if not source_pcb.exists():
        source_pcb = Path(extraction.subcircuit.schematic_path).with_suffix(".kicad_pcb")

    if not source_pcb.exists():
        raise RuntimeError(
            "Leaf the autorouter requires a real source PCB to stamp from; "
            f"could not resolve base board for {extraction.subcircuit.id.instance_path}"
        )

    if not legality_repair.get("resolved", False):
        diagnostics = legality_repair.get("diagnostics", {}) or {}
        overlap_count = int(diagnostics.get("overlap_count", 0) or 0)
        pad_outside_count = int(diagnostics.get("pad_outside_count", 0) or 0)
        overlap_pairs = [
            f"{item.get('a', '?')}:{item.get('b', '?')}"
            for item in diagnostics.get("overlaps", [])
        ]
        pad_violations = [
            f"{item.get('ref', '?')}:{item.get('pad_id', '?')}:{','.join(item.get('sides', []))}"
            for item in diagnostics.get("pads_outside_board", [])
        ]

        overlap_details = []
        for item in diagnostics.get("overlaps", []):
            overlap_details.append(
                {
                    "a": item.get("a"),
                    "b": item.get("b"),
                    "overlap_x_mm": item.get("overlap_x_mm"),
                    "overlap_y_mm": item.get("overlap_y_mm"),
                    "overlap_area_mm2": item.get("overlap_area_mm2"),
                }
            )

        component_debug = []
        repaired_by_ref = repaired_components or {}
        for ref in sorted(repaired_by_ref.keys()):
            comp = repaired_by_ref[ref]
            component_debug.append(
                {
                    "ref": ref,
                    "kind": comp.kind,
                    "layer": str(comp.layer),
                    "locked": bool(comp.locked),
                    "x_mm": round(comp.pos.x, 4),
                    "y_mm": round(comp.pos.y, 4),
                    "rotation_deg": round(comp.rotation, 4),
                    "width_mm": round(comp.width_mm, 4),
                    "height_mm": round(comp.height_mm, 4),
                    "pad_count": len(comp.pads),
                }
            )

        print(
            "  Leaf legality repair rejected placement: "
            f"overlaps={overlap_count} "
            f"pads_outside={pad_outside_count} "
            f"overlap_pairs={overlap_pairs} "
            f"pad_violations={pad_violations}"
        )
        if overlap_details:
            print(f"  Leaf legality overlap details: {overlap_details}")
        if component_debug:
            print(f"  Leaf legality component states: {component_debug}")

        illegal_input_board = copy.deepcopy(extraction.local_state)
        illegal_input_board.components = copy.deepcopy(repaired_components)
        illegal_input_board.traces = []
        illegal_input_board.vias = []

        illegal_render_diagnostics: dict[str, Any] = {
            "artifact_dir": artifact_paths.artifact_dir,
            "renders_dir": str(Path(artifact_paths.artifact_dir) / "renders"),
            "illegal_pre_stamp": None,
            "errors": [],
        }

        try:
            route_adapter = KiCadAdapter(str(source_pcb), config=cfg)
            route_adapter.stamp_subcircuit_board(
                illegal_input_board,
                output_path=str(illegal_board),
                clear_existing_tracks=True,
                clear_existing_zones=True,
                remove_unmapped_footprints=True,
            )
            illegal_validation = {
                "accepted": False,
                "rejected": True,
                "rejection_stage": "leaf_pre_stamp_legality_repair",
                "rejection_reasons": ["illegal_unrepaired_leaf_placement"],
                "leaf_legality_repair": copy.deepcopy(legality_repair),
                "drc": {
                    "violations": [],
                    "report_text": (
                        "Leaf placement rejected before routing due to placement legality.\n"
                        f"overlap_count={overlap_count}\n"
                        f"pad_outside_count={pad_outside_count}\n"
                        f"overlap_pairs={overlap_pairs}\n"
                        f"pad_violations={pad_violations}\n"
                    ),
                },
            }
            illegal_render_diagnostics["illegal_pre_stamp"] = (
                generate_stage_diagnostic_artifacts(
                    pcb_path=str(illegal_board),
                    validation=illegal_validation,
                    artifact_dir=artifact_paths.artifact_dir,
                    stage="illegal_pre_stamp",
                    render_board_views=not fast_smoke_mode,
                    write_drc_json=not fast_smoke_mode,
                    write_drc_report=not fast_smoke_mode,
                    render_drc_overlay=not fast_smoke_mode,
                )
            )
        except Exception as exc:
            illegal_render_diagnostics["errors"].append(
                f"illegal_pre_stamp_render_failed:{exc}"
            )

        # Copy the illegal-pre-stamp PNGs (and the .kicad_pcb) to
        # round_XXXX_pre_route_* names so the GUI's _find_round_renders
        # can show *something* for failed rounds. Without this, failed
        # rounds appear as bare "-inf" cards with a placeholder icon and
        # the user has no way to inspect what placement was rejected.
        if round_index is not None:
            _round_prefix = f"round_{int(round_index):04d}"
            _renders_dir = Path(artifact_paths.artifact_dir) / "renders"
            _renders_dir.mkdir(parents=True, exist_ok=True)
            _ip = illegal_render_diagnostics.get("illegal_pre_stamp") or {}
            _view_paths = (_ip.get("board_views", {}) or {}).get("paths", {}) or {}
            for _view, _suffix in (
                ("front_all", "pre_route_front_all"),
                ("back_all", "pre_route_back_all"),
                ("copper_both", "pre_route_copper_both"),
            ):
                _src = _view_paths.get(_view)
                if _src and Path(_src).exists():
                    try:
                        shutil.copy2(_src, _renders_dir / f"{_round_prefix}_{_suffix}.png")
                    except OSError:
                        pass
            if illegal_board.exists():
                try:
                    shutil.copy2(
                        illegal_board,
                        illegal_board.parent / f"{_round_prefix}_leaf_placed{illegal_board.suffix}",
                    )
                except OSError:
                    pass

        route_timing["route_local_subcircuit_total_s"] = round(
            max(0.0, time.monotonic() - route_total_start), 3
        )
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": "illegal_unrepaired_leaf_placement",
                "router": "kicad-routing-tools",
                "traces": 0,
                "vias": 0,
                "total_length_mm": 0.0,
                "routed_internal_nets": [],
                "failed_internal_nets": list(sorted(extraction.internal_net_names)),
                "_trace_segments": [],
                "_via_objects": [],
                "validation": {
                    "accepted": False,
                    "rejected": True,
                    "rejection_stage": "leaf_pre_stamp_legality_repair",
                    "rejection_reasons": ["illegal_unrepaired_leaf_placement"],
                    "leaf_legality_repair": copy.deepcopy(legality_repair),
                    "render_diagnostics": copy.deepcopy(illegal_render_diagnostics)
                    if generate_diagnostics
                    else {"skipped": True, "reason": "size_reduction_fast_path"},
                    "illegal_pre_stamp_board_path": str(illegal_board),
                },
                "leaf_legality_repair": copy.deepcopy(legality_repair),
                "render_diagnostics": copy.deepcopy(illegal_render_diagnostics)
                if generate_diagnostics
                else {"skipped": True, "reason": "size_reduction_fast_path"},
                "illegal_pre_stamp_board_path": str(illegal_board),
                "failed": True,
            },
            route_timing,
        )

    route_input_board = copy.deepcopy(extraction.local_state)
    route_input_board.components = copy.deepcopy(repaired_components)
    route_input_board.traces = []
    route_input_board.vias = []
    # Silk is stamped post-route (after the autorouter) so the rounded
    # outline can hug the routed copper too -- not just courtyards and
    # pad copper. See the silk re-stamp block after import_routed_copper.
    route_input_board.silkscreen = []

    stamp_start = time.monotonic()
    route_adapter = KiCadAdapter(str(source_pcb), config=cfg)
    route_adapter.stamp_subcircuit_board(
        route_input_board,
        output_path=str(pre_route_board),
        clear_existing_tracks=True,
        clear_existing_zones=True,
        remove_unmapped_footprints=True,
    )
    route_timing["stamp_pre_route_board_s"] = round(
        max(0.0, time.monotonic() - stamp_start), 3
    )

    routing_start = time.monotonic()
    leaf_routing_cfg = {**cfg, "pcb_path": str(source_pcb)}
    _breakout_specs = _resolve_breakout_specs(cfg)

    # GND pre-escape (default on): plane-bond stubs for fine-pitch GND pads,
    # early in the spec order so they claim space before the signal escapes
    # and the router do -- the post-route escape pass finds those pads walled
    # in (GND is never routed; signals can route around locked copper).
    if cfg.get("gnd_pre_escape", True):
        try:
            import pcbnew

            from kicraft.autoplacer.brain.gnd_pour import gnd_escape_specs

            _g_board = pcbnew.LoadBoard(str(pre_route_board))
            _breakout_specs = gnd_escape_specs(_g_board, cfg) + _breakout_specs
            del _g_board
        except Exception as exc:  # never fail the leaf on a finishing helper
            print(f"  WARNING: gnd pre-escape spec gen failed: {exc}")
    # Auto power-tie (default on): route a tie around any connector whose spread
    # power pads (e.g. USB-C VBUS) would otherwise fragment the power pour.
    if cfg.get("auto_power_tie", True):
        try:
            import pcbnew

            from kicraft.autoplacer.brain.breakout_stubs import auto_power_tie_specs

            _tie_board = pcbnew.LoadBoard(str(pre_route_board))
            _breakout_specs = _breakout_specs + auto_power_tie_specs(_tie_board, cfg)
            del _tie_board
        except Exception as exc:  # never fail the leaf on a finishing helper
            print(f"  WARNING: auto power-tie spec gen failed: {exc}")
    # Shield tie (default on): a connector's through-hole shield legs sit where
    # neither GND plane can reach them (F.Cu walled out of the pad row by the
    # Power-netclass clearance, B.Cu thermal spokes lost to the slot holes), so
    # the legs facing the pad row survive as unconnected ratlines on an
    # otherwise-routed board. Tie each netted PTH pad to its nearest same-net pad.
    if cfg.get("shield_tie_enabled", True):
        try:
            import pcbnew

            from kicraft.autoplacer.brain.breakout_stubs import shield_tie_specs

            _sh_board = pcbnew.LoadBoard(str(pre_route_board))
            _breakout_specs = _breakout_specs + shield_tie_specs(_sh_board, cfg)
            del _sh_board
        except Exception as exc:  # never fail the leaf on a finishing helper
            print(f"  WARNING: shield tie spec gen failed: {exc}")
    # Array daisy-chain (default on): for an addressable-LED matrix or similar
    # regular array, deterministically stamp the short data hops (DOUT->DIN) as
    # locked ties + pad escapes, so the autorouter -- which abandons a few of these
    # in the dense inter-component channels every run -- only has to finish from
    # open copper. Power is delivered by the +5V/GND pours, so the channels
    # carry only data. Keyed on the array spec, NOT leaf_is_fully_array (a 3-pin
    # header or other non-passive on the leaf must not disable it).
    _arr_specs: list = []
    _arr_stamp_skipped: list = []
    if cfg.get("array_route_enabled", True) and cfg.get("arrays"):
        try:
            import pcbnew

            from kicraft.autoplacer.brain.array_router import array_daisy_chain_specs

            _arr_board = pcbnew.LoadBoard(str(pre_route_board))
            _arr_specs = array_daisy_chain_specs(_arr_board, cfg)
            del _arr_board
            if _arr_specs:
                _breakout_specs = _breakout_specs + _arr_specs
        except Exception as exc:  # never fail the leaf on a finishing helper
            print(f"  WARNING: array daisy-chain spec gen failed: {exc}")
    # Ring +5V bus (default on): for a ring array, deterministically stamp a
    # CLOSED loop of member->member (or member->decap->member) power ties at
    # the pad radius, plus a via stub tying each band decap's GND into the
    # B.Cu pour -- the autorouter then has no reason to dip into the ring
    # interior, which shaped-compose nesting needs clear (PR-N5). Kept as a
    # SEPARATE list from _arr_specs: the array stamp gate measures in-row
    # DATA hops only.
    _ring_pwr_specs: list = []
    if cfg.get("array_ring_power_bus", True) and cfg.get("arrays"):
        try:
            import pcbnew

            from kicraft.autoplacer.brain.array_router import array_ring_power_specs

            _rp_board = pcbnew.LoadBoard(str(pre_route_board))
            _ring_pwr_specs = array_ring_power_specs(_rp_board, cfg)
            del _rp_board
            if _ring_pwr_specs:
                _breakout_specs = _breakout_specs + _ring_pwr_specs
        except Exception as exc:  # never fail the leaf on a finishing helper
            print(f"  WARNING: ring power-bus spec gen failed: {exc}")
    if _breakout_specs:
        try:
            from kicraft.autoplacer.brain.breakout_stubs import add_breakout_stubs

            _bo = add_breakout_stubs(
                str(pre_route_board), _breakout_specs, cfg=cfg
            )
            if _bo["stubs"] > 0:
                leaf_routing_cfg["routing_preserve_existing_copper"] = True
                print(
                    f"  Breakout stubs: {_bo['stubs']} pad(s), "
                    f"{_bo['segments']} segment(s), {_bo['vias']} via(s)"
                )
            # No-silent handoff: a data tie the stamp guards dropped goes to
            # the autorouter -- surface it so an incompletely-stamped chain is
            # visible, not hidden behind a clean-looking "stubs" count.
            if _arr_specs:
                _arr_keys = {f"{s.ref}.{s.pad}" for s in _arr_specs}
                _arr_stamp_skipped = [
                    s for s in _bo.get("skipped", [])
                    if s.split(":", 1)[0] in _arr_keys
                ]
                if _arr_stamp_skipped:
                    print(
                        f"  array-router: {len(_arr_stamp_skipped)}/{len(_arr_specs)} "
                        f"data tie(s) left to the autorouter: {', '.join(_arr_stamp_skipped)}"
                    )
            # Same no-silent-handoff rule for the ring power bus (its ties
            # are excluded from the stamp gate, so this log is the only
            # place a dropped bus segment surfaces).
            if _ring_pwr_specs:
                _rp_keys = {f"{s.ref}.{s.pad}" for s in _ring_pwr_specs}
                _rp_skipped = [
                    s for s in _bo.get("skipped", [])
                    if s.split(":", 1)[0] in _rp_keys
                ]
                if _rp_skipped:
                    print(
                        f"  ring-power: {len(_rp_skipped)}/{len(_ring_pwr_specs)} "
                        f"bus tie(s) left to the autorouter: {', '.join(_rp_skipped)}"
                    )
        except Exception as exc:  # finishing step must never fail the leaf
            print(f"  WARNING: breakout stub step failed: {exc}")

    # Layer-2 guardrail: a healthy array leaf pre-stamps almost its whole data
    # chain so the autorouter has next to nothing left. When the stamp rate collapses
    # (the inter-member channel is obstructed -- e.g. a foreign pad sitting in the
    # data lane, the KC-NZXXEE decaps-on-LEDs signature), the entire chain plus
    # power falls to the autorouter, which cannot finish a dense array before the
    # build's wall-clock cap kills it (30 min -> no board -> raw-component
    # preview). Reject the leaf fast and loud instead, naming the obstruction, so
    # the real cause surfaces in seconds rather than a silent timeout.
    if _arr_specs and array_stamp_gate_tripped(_arr_specs, _arr_stamp_skipped, cfg):
        _n_inrow, _n_inrow_skipped = _array_inrow_stamp_stats(
            _arr_specs, _arr_stamp_skipped
        )
        _stamped = _n_inrow - _n_inrow_skipped
        _rate = _stamped / _n_inrow
        _min_rate = float(cfg.get("array_min_stamp_rate", 0.5))
        print(
            f"  ARRAY GATE: only {_stamped}/{_n_inrow} in-row data ties stamped "
            f"({_rate:.0%} < {_min_rate:.0%}) -- the inter-member data channel "
            "is obstructed (a foreign pad in the data lane?). Rejecting the leaf "
            "BEFORE the autorouter rather than handing it a chain it cannot finish "
            "in time. Skipped: " + ", ".join(_arr_stamp_skipped[:12])
            + (" ..." if len(_arr_stamp_skipped) > 12 else "")
        )
        route_timing["route_local_subcircuit_total_s"] = round(
            max(0.0, time.monotonic() - route_total_start), 3
        )
        _reason = "array_data_channel_obstructed"
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": _reason,
                "router": "kicad-routing-tools",
                "traces": 0,
                "vias": 0,
                "total_length_mm": 0.0,
                "routed_internal_nets": [],
                "failed_internal_nets": list(sorted(extraction.internal_net_names)),
                "_trace_segments": [],
                "_via_objects": [],
                "validation": {
                    "accepted": False,
                    "rejected": True,
                    "rejection_stage": _reason,
                    "rejection_reasons": [_reason],
                    "drc": {
                        "violations": [],
                        "report_text": (
                            "Array leaf rejected before routing: data channel "
                            "obstructed (in-row hops only; row turns excluded).\n"
                            f"in_row_ties_total={_n_inrow}\n"
                            f"in_row_ties_stamped={_stamped}\n"
                            f"in_row_ties_skipped={_n_inrow_skipped}\n"
                            f"all_array_ties={len(_arr_specs)}\n"
                            f"in_row_stamp_rate={_rate:.3f} (min {_min_rate:.3f})\n"
                            "skipped="
                            + ", ".join(_arr_stamp_skipped) + "\n"
                        ),
                    },
                },
                "failed": True,
            },
            route_timing,
        )
    # Route cache: deterministic array leaves reuse identical KRT output.
    _cache_pcb = None
    _cache_meta = None
    if leaf_is_fully_array(route_input_board.components, cfg.get("arrays", [])):
        _sig = _deterministic_route_signature(route_input_board, leaf_routing_cfg)
        _cache_dir = Path(artifact_paths.artifact_dir) / "route_cache"
        _cache_pcb = _cache_dir / f"{_sig}.kicad_pcb"
        _cache_meta = _cache_dir / f"{_sig}.json"
    if _cache_pcb is not None and _cache_pcb.exists() and _cache_meta.exists():
        shutil.copyfile(_cache_pcb, routed_board)
        routing_stats = json.loads(_cache_meta.read_text())
        routing_stats["route_cache_hit"] = True
        print(
            "  [route-cache] deterministic leaf unchanged -> reused routed "
            "board (skipped KRT)"
        )
    else:
        routing_stats = route_with_kicad_routing_tools(
            str(pre_route_board),
            str(routed_board),
            leaf_routing_cfg,
        )
        if _cache_pcb is not None:
            _cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(routed_board, _cache_pcb)
            _cache_meta.write_text(json.dumps(routing_stats))
    route_timing["routing_s"] = round(
        max(0.0, time.monotonic() - routing_start), 3
    )

    pre_route_validation_start = time.monotonic()
    pre_route_validation = validate_routed_board(
        str(pre_route_board),
        cfg=cfg,
        expected_anchor_names=[port.name for port in extraction.interface_ports],
        actual_anchor_names=[port.name for port in extraction.interface_ports],
        required_anchor_names=[
            port.name for port in extraction.interface_ports if port.required
        ],
        timeout_s=int(cfg.get("subcircuit_validation_timeout_s", 30)),
    )
    route_timing["pre_route_validation_s"] = round(
        max(0.0, time.monotonic() - pre_route_validation_start), 3
    )
    # Pre-route DRC is informational only -- we let the autorouter attempt routing
    # regardless of pre-route violations. The post-route DRC gate handles acceptance.
    pre_route_drc = pre_route_validation.get("drc", {})
    if pre_route_drc.get("violations"):
        pre_route_violation_types = {v.get("type") for v in pre_route_drc["violations"]}
        print(
            f"  Pre-route DRC info: {len(pre_route_drc['violations'])} violations ({', '.join(sorted(pre_route_violation_types))})"
        )
    if generate_diagnostics and render_intermediate:
        pre_route_render_start = time.monotonic()
        leaf_diagnostics = generate_leaf_diagnostic_artifacts(
            artifact_dir=artifact_paths.artifact_dir,
            pre_route_board=str(pre_route_board),
            routed_board=str(routed_board) if routed_board.exists() else None,
            pre_route_validation=pre_route_validation,
            pre_route_opts=LeafStageOpts(
                render_board_views=render_pre_route_board_views,
                write_drc_json=write_pre_route_drc_json,
                write_drc_report=write_pre_route_drc_report,
                render_drc_overlay=render_pre_route_drc_overlay,
            ),
            routed_opts=LeafStageOpts.off(),
            build_contact_sheet=False,
            quiet_render=fast_smoke_mode,
        )
        route_timing["pre_route_render_diagnostics_s"] = round(
            max(0.0, time.monotonic() - pre_route_render_start), 3
        )
    else:
        leaf_diagnostics = {
            "skipped": True,
            "reason": "size_reduction_fast_path",
        }
        route_timing["pre_route_render_diagnostics_s"] = 0.0

    round_board_illegal_pre_stamp = ""
    round_board_pre_route = ""
    round_board_routed = ""

    _board_snapshot = promote_to_round_snapshot(pre_route_board, round_index)
    if _board_snapshot is not None:
        round_board_pre_route = str(_board_snapshot)

    if round_index is not None and not leaf_diagnostics.get("skipped", False):
        pre_route_section = leaf_diagnostics.get("pre_route", {})
        if isinstance(pre_route_section, dict):
            pre_route_views = pre_route_section.get("board_views", {})
            if isinstance(pre_route_views, dict):
                pre_route_paths = pre_route_views.get("paths", {})
                if isinstance(pre_route_paths, dict):
                    for _view in ("front_all", "back_all", "copper_both"):
                        snap = promote_to_round_snapshot(
                            pre_route_paths.get(_view), round_index
                        )
                        if snap is not None:
                            pre_route_paths[f"round_{_view}"] = str(snap)

    pre_route_validation["render_diagnostics"] = copy.deepcopy(leaf_diagnostics)
    pre_route_validation["leaf_legality_repair"] = copy.deepcopy(legality_repair)
    if round_board_pre_route:
        pre_route_validation["round_board_pre_route"] = round_board_pre_route

    import_copper_start = time.monotonic()
    imported_copper = import_routed_copper(str(routed_board))
    route_timing["import_routed_copper_s"] = round(
        max(0.0, time.monotonic() - import_copper_start), 3
    )

    # Re-stamp the routed leaf board with silk computed from the FULL
    # post-route content (courtyards + pad copper + traces + vias).
    # Stamping silk here -- rather than on the pre-route board -- means
    # the rounded outline includes routed copper that extends past the
    # components, so the canvas + round-selector PNGs show one tight
    # outline instead of "silk hugs the chip body but the trace stub
    # pokes out past it."
    silk_stamp_start = time.monotonic()
    silk_adapter = KiCadAdapter(str(routed_board), config=cfg)
    routed_state_for_silk = silk_adapter.load()
    # Shrink Edge.Cuts to hug the post-route geometry the silk hugs, and center
    # the whole leaf on a standard A4 page so the standalone board opens centered in
    # the title block instead of crammed against the top-left origin. Centering is
    # safe for composition: the parent composer re-bases each leaf to its own outline
    # origin on load (subcircuit_instances._layout_from_artifact_payload) before it
    # rotates around (0, 0) and translates by the placement origin, so leaf page
    # position is free. Without the shrink, the yellow outline stays at whatever
    # size-reduction accepted (or the raw extractor envelope), and the rounded silk
    # sits inside a much larger sharp rect.
    _new_outline = _outline_around_geometry(
        routed_state_for_silk.components,
        cfg,
        traces=routed_state_for_silk.traces,
        vias=routed_state_for_silk.vias,
    )
    if _new_outline is not None:
        _new_tl, _new_br = _new_outline
        _delta, _centered_outline = _center_on_leaf_page(_new_tl, _new_br, cfg)
        if abs(_delta.x) > 1e-6 or abs(_delta.y) > 1e-6:
            from kicraft.autoplacer.brain.leaf_geometry import (
                copy_components_with_translation,
                copy_traces_with_translation,
                copy_vias_with_translation,
            )
            routed_state_for_silk.components = copy_components_with_translation(
                routed_state_for_silk.components, _delta
            )
            routed_state_for_silk.traces = copy_traces_with_translation(
                routed_state_for_silk.traces, _delta
            )
            routed_state_for_silk.vias = copy_vias_with_translation(
                routed_state_for_silk.vias, _delta
            )
        routed_state_for_silk.board_outline = _centered_outline
    # Silk is computed AFTER the translate so it lands in the centered frame.
    routed_state_for_silk.silkscreen = _silk_for_leaf(
        extraction,
        routed_state_for_silk.components,
        cfg,
        traces=routed_state_for_silk.traces,
        vias=routed_state_for_silk.vias,
    )
    silk_adapter.stamp_subcircuit_board(
        routed_state_for_silk,
        output_path=str(routed_board),
        clear_existing_tracks=True,
        clear_existing_zones=True,
        remove_unmapped_footprints=False,
    )
    route_timing["silk_post_route_stamp_s"] = round(
        max(0.0, time.monotonic() - silk_stamp_start), 3
    )

    # Ground-plane finishing (default on): pour a full B.Cu GND plane -- the
    # ZONE_FILLER keeps it clear of rule-area keepouts like the WROOM antenna --
    # and stitch large GND/thermal pads into it with a dense thermal-via array so
    # the plane connects and the boxed-in center pad escapes to ground. Run after
    # the silk re-stamp (the last write to routed_board) and before acceptance
    # validation, so the now-connected center pad is reflected in shorts/unconnected.
    gnd_pour_summary: dict | None = None
    if cfg.get("gnd_plane_enabled", True):
        gnd_pour_start = time.monotonic()
        try:
            from kicraft.autoplacer.brain.gnd_pour import (
                add_gnd_pour_and_thermal_vias,
            )

            _gnd = add_gnd_pour_and_thermal_vias(str(routed_board), cfg)
            # Persisted into the leaf result (-> debug.json): stdout-only
            # summaries made the GND-strand family untriageable from the
            # experiments tree alone.
            gnd_pour_summary = _gnd
            print(
                f"  GND plane: {_gnd.get('thermal_vias_added', 0)} thermal via(s) "
                f"under {_gnd.get('gnd_pads_stitched', 0)} pad(s), "
                f"{_gnd.get('escape_stitched', 0)} escape(s), "
                f"{_gnd.get('thermal_vias_blocked', 0)} blocked; B.Cu pour filled"
            )
        except Exception as exc:  # finishing step must never fail the leaf
            print(f"  WARNING: GND plane step failed: {exc}")
        route_timing["gnd_pour_s"] = round(
            max(0.0, time.monotonic() - gnd_pour_start), 3
        )

    # Power-plane finishing (default on): pour the primary power rail on the
    # layer opposite GND so paired connector power pads / regulator input / bulk
    # caps connect through copper. Runs before acceptance so the now-connected
    # power pads are reflected in the unconnected count.
    if cfg.get("power_plane_enabled", True):
        power_pour_start = time.monotonic()
        try:
            from kicraft.autoplacer.brain.gnd_pour import (
                pour_power_planes,
                repair_stranded_power,
            )

            _pwr = pour_power_planes(
                str(routed_board),
                cfg,
                layers=(cfg.get("power_plane_layer", "F.Cu"),),
            )
            if _pwr.get("nets"):
                print(
                    f"  Power plane: poured {_pwr['nets']} on "
                    f"{cfg.get('power_plane_layer', 'F.Cu')}"
                )
                # The pour fragments around foreign copper like the GND plane
                # does and can strand a supply pad on its own island
                # (fine-pitch parts especially); tie stranded clusters back
                # so they count as connected in the acceptance below.
                _rep = repair_stranded_power(str(routed_board), _pwr["nets"], cfg)
                if _rep.get("stranded"):
                    print(
                        f"  Power strand repair: {_rep['stranded']} stranded, "
                        f"{_rep['tied']} tied, {len(_rep['skipped'])} skipped"
                    )
        except Exception as exc:  # finishing step must never fail the leaf
            print(f"  WARNING: power plane step failed: {exc}")
        route_timing["power_pour_s"] = round(
            max(0.0, time.monotonic() - power_pour_start), 3
        )

    routed_validation_start = time.monotonic()
    validation = validate_routed_board(
        str(routed_board),
        cfg=cfg,
        expected_anchor_names=[port.name for port in extraction.interface_ports],
        actual_anchor_names=[port.name for port in extraction.interface_ports],
        required_anchor_names=[
            port.name for port in extraction.interface_ports if port.required
        ],
        timeout_s=int(cfg.get("subcircuit_validation_timeout_s", 30)),
    )
    # Always record the leaf's interface (inter-sheet) net names so the
    # unconnected-acceptance gate can exclude them: an interface net is routed
    # across the parent at compose, not within the leaf, so an unconnected item
    # on it must not count against the leaf (mirrors poured power/GND nets).
    # Previously set only on the reject path, leaving normal-path leaves blind.
    validation["interface_port_names"] = [
        port.name for port in extraction.interface_ports
    ]
    route_timing["routed_validation_s"] = round(
        max(0.0, time.monotonic() - routed_validation_start), 3
    )

    # Leaf-level C1 repair: a leaf a couple of guarded ties short of clean used
    # to have no lever at all -- `repair_unconnected_signals` was wired into the
    # parent compose only -- so the sole response to 1-5 residual opens was a
    # whole new round on a fresh seed (dense-soc plan P1.6). Gated on the opens
    # that must actually close IN this leaf (poured and interface nets close at
    # compose), and contained by the same accept-or-revert the parent uses.
    _repair_cap = int(cfg.get("leaf_signal_repair_max_open_nets", 6))
    if cfg.get("leaf_signal_repair_enabled", True) and _repair_cap > 0:
        from kicraft.autoplacer.brain.leaf_acceptance import (
            acceptance_config_from_dict,
            split_unconnected_nets,
        )

        _local_open, _, _ = split_unconnected_nets(
            validation, acceptance_config_from_dict(cfg)
        )
        if _local_open and len(_local_open) <= _repair_cap:
            from kicraft.autoplacer.brain.signal_repair_pass import (
                attempt_signal_unconnected_repair,
            )

            _repair_start = time.monotonic()
            print(
                f"  leaf signal repair: {len(_local_open)} local open net(s) "
                f"({', '.join(_local_open)}) -- attempting guarded ties"
            )
            validation = attempt_signal_unconnected_repair(
                routed_board,
                cfg,
                validation,
                anchor_names=[port.name for port in extraction.interface_ports],
                required_anchor_names=[
                    port.name for port in extraction.interface_ports if port.required
                ],
                label="leaf signal repair",
            )
            route_timing["leaf_signal_repair_s"] = round(
                max(0.0, time.monotonic() - _repair_start), 3
            )
        elif _local_open:
            validation["signal_unconnected_repair"] = {
                "ran": False,
                "reason": f"{len(_local_open)}_open_nets_over_cap_{_repair_cap}",
            }
    if generate_diagnostics and render_intermediate:
        routed_render_start = time.monotonic()
        leaf_diagnostics = generate_leaf_diagnostic_artifacts(
            artifact_dir=artifact_paths.artifact_dir,
            pre_route_board=str(pre_route_board),
            routed_board=str(routed_board),
            pre_route_validation=pre_route_validation,
            routed_validation=validation,
            pre_route_opts=LeafStageOpts(
                render_board_views=render_pre_route_board_views,
                write_drc_json=write_pre_route_drc_json,
                write_drc_report=write_pre_route_drc_report,
                render_drc_overlay=render_pre_route_drc_overlay,
            ),
            routed_opts=LeafStageOpts(
                render_board_views=render_routed_board_views,
                write_drc_json=write_routed_drc_json,
                write_drc_report=write_routed_drc_report,
                render_drc_overlay=render_routed_drc_overlay,
            ),
            build_contact_sheet=build_comparison_contact_sheet,
            quiet_render=fast_smoke_mode,
        )
        route_timing["routed_render_diagnostics_s"] = round(
            max(0.0, time.monotonic() - routed_render_start), 3
        )
    else:
        leaf_diagnostics = {
            "skipped": True,
            "reason": "size_reduction_fast_path",
        }
        route_timing["routed_render_diagnostics_s"] = 0.0

    _routed_board_snapshot = promote_to_round_snapshot(routed_board, round_index)
    if _routed_board_snapshot is not None:
        round_board_routed = str(_routed_board_snapshot)

    # Round-snapshot paths surfaced to downstream consumers (types.py's
    # round_to_layout, the GUI's per-round scrubber via routing dict).
    # Empty string when this stage didn't produce that view, which the
    # consumers already handle as "no per-round preview" (gracefully fall
    # back to canonical render).
    round_preview_pre_route_front = ""
    round_preview_pre_route_back = ""
    round_preview_pre_route_copper = ""
    round_preview_routed_front = ""
    round_preview_routed_back = ""
    round_preview_routed_copper = ""

    if round_index is not None and not leaf_diagnostics.get("skipped", False):
        for _stage_key in ("pre_route", "routed"):
            _section = leaf_diagnostics.get(_stage_key, {})
            if not isinstance(_section, dict):
                continue
            _views = _section.get("board_views", {})
            if not isinstance(_views, dict):
                continue
            _paths = _views.get("paths", {})
            if not isinstance(_paths, dict):
                continue
            for _view in ("front_all", "back_all", "copper_both"):
                snap = promote_to_round_snapshot(_paths.get(_view), round_index)
                if snap is not None:
                    snap_str = str(snap)
                    _paths[f"round_{_view}"] = snap_str
                    if _stage_key == "pre_route":
                        if _view == "front_all":
                            round_preview_pre_route_front = snap_str
                        elif _view == "back_all":
                            round_preview_pre_route_back = snap_str
                        elif _view == "copper_both":
                            round_preview_pre_route_copper = snap_str
                    else:
                        if _view == "front_all":
                            round_preview_routed_front = snap_str
                        elif _view == "back_all":
                            round_preview_routed_back = snap_str
                        elif _view == "copper_both":
                            round_preview_routed_copper = snap_str

    validation["pre_route_validation"] = copy.deepcopy(pre_route_validation)
    validation["render_diagnostics"] = copy.deepcopy(leaf_diagnostics)
    if round_board_pre_route:
        validation["round_board_pre_route"] = round_board_pre_route
    if round_board_routed:
        validation["round_board_routed"] = round_board_routed

    drc = validation.get("drc", {})
    drc_stdout = str(drc.get("stdout", ""))
    drc_stderr = str(drc.get("stderr", ""))
    _ = "\n".join(part for part in (drc_stdout, drc_stderr) if part.strip())

    # Post-route ignorable violation types: cosmetic issues and violations
    # that are inherent to the footprint or subcircuit outline, not caused
    # by the routing itself.
    ignorable_warning_types = {
        "silk_overlap",
        "lib_footprint_mismatch",
        "copper_edge_clearance",  # tight subcircuit outlines
        "silk_edge_clearance",  # cosmetic
        "silk_over_copper",  # cosmetic
        "solder_mask_bridge",  # footprint-internal
        "unconnected_items",  # the autorouter may not route all nets
    }
    significant_violations = [
        violation
        for violation in drc.get("violations", [])
        if violation.get("type") not in ignorable_warning_types
    ]

    # --- Generalized DRC exception: config-driven patterns ---
    # If the config provides ignorable_drc_patterns (list of regex strings),
    # check whether ALL significant violations match at least one pattern.
    ignorable_drc_patterns = cfg.get("ignorable_drc_patterns", [])
    _compiled_drc_patterns = [re.compile(p) for p in ignorable_drc_patterns]
    _all_match_config_patterns = (
        significant_violations
        and _compiled_drc_patterns
        and all(
            any(pat.search(v.get("description", "")) for pat in _compiled_drc_patterns)
            for v in significant_violations
        )
        and not drc.get("shorts", 0)
    )

    # --- Generalized DRC exception: footprint-baseline clearance heuristic ---
    # If ALL significant violations are clearance-type violations whose
    # descriptions reference pads from the SAME single footprint, treat them
    # as footprint-internal baseline clearance issues (e.g. dense USB-C,
    # fine-pitch IC pads closer together than the board clearance rule).
    _footprint_ref_re = re.compile(r"\bof\s+(\S+)")
    _clearance_types = {"clearance", "hole_clearance", "solder_mask_bridge"}
    _all_clearance = (
        significant_violations
        and all(v.get("type") in _clearance_types for v in significant_violations)
        and not drc.get("shorts", 0)
    )
    _single_footprint_baseline = False
    _baseline_footprint_ref = None
    if _all_clearance:
        # Collect all footprint references mentioned across violations
        _violation_footprint_refs: set[str] = set()
        for v in significant_violations:
            desc = v.get("description", "")
            for m in _footprint_ref_re.finditer(desc):
                _violation_footprint_refs.add(m.group(1))
        # If every violation references pads from exactly one footprint,
        # this is a footprint-internal clearance issue.
        if len(_violation_footprint_refs) == 1:
            _single_footprint_baseline = True
            _baseline_footprint_ref = next(iter(_violation_footprint_refs))

    if _all_match_config_patterns or _single_footprint_baseline:
        _ignore_reason = (
            "config_ignorable_drc_patterns"
            if _all_match_config_patterns
            else f"footprint_baseline_clearance:{_baseline_footprint_ref}"
        )
        _ignored_types = {v.get("type") for v in significant_violations}
        validation["obviously_illegal_routed_geometry"] = False
        validation["rejection_reasons"] = [
            reason
            for reason in validation.get("rejection_reasons", [])
            if reason != "illegal_routed_geometry"
        ]
        validation["accepted"] = not validation["rejection_reasons"]
        validation["drc"]["ignored_violation_types"] = sorted(
            ignorable_warning_types | _ignored_types
        )
        validation["drc"]["ignored_violation_count"] = len(drc.get("violations", []))
        validation["drc"]["significant_violation_count"] = 0
        validation["drc"]["ignored_clearance_reason"] = _ignore_reason

    routed_internal, failed_internal = _internal_net_verdict(validation, extraction)

    accepted = bool(validation.get("accepted", False))
    if not accepted:
        validation["accepted"] = False
        validation["rejected"] = True
        validation["rejection_stage"] = "leaf_routed_artifact_validation"
        validation["routed_board_path"] = str(routed_board)
        validation["leaf_placed_board"] = str(pre_route_board)
        validation["router"] = "kicad-routing-tools"
        validation["internal_net_names"] = list(sorted(extraction.internal_net_names))
        validation["interface_port_names"] = [
            port.name for port in extraction.interface_ports
        ]
        validation["imported_copper_summary"] = {
            "trace_count": int(imported_copper.get("trace_count", 0)),
            "via_count": int(imported_copper.get("via_count", 0)),
            "total_length_mm": float(imported_copper.get("total_length_mm", 0.0)),
        }
        validation["routing_stats"] = copy.deepcopy(routing_stats)
        validation["rejection_message"] = "Leaf routed artifact rejected: " + ",".join(
            validation.get("rejection_reasons", [])
        )
        print("  Routed DRC rejected placement: " + validation["rejection_message"])
        route_timing["route_local_subcircuit_total_s"] = round(
            max(0.0, time.monotonic() - route_total_start), 3
        )
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": "routed_drc_rejection",
                "router": "kicad-routing-tools",
                "traces": int(imported_copper.get("trace_count", 0)),
                "vias": int(imported_copper.get("via_count", 0)),
                "total_length_mm": float(imported_copper.get("total_length_mm", 0.0)),
                "round_board_illegal_pre_stamp": round_board_illegal_pre_stamp,
                "round_board_pre_route": round_board_pre_route,
                "round_board_routed": round_board_routed,
                "routed_internal_nets": routed_internal,
                "failed_internal_nets": failed_internal,
                "_trace_segments": [],
                "_via_objects": [],
                "validation": copy.deepcopy(validation),
                "routing_stats": copy.deepcopy(routing_stats),
                "render_diagnostics": copy.deepcopy(leaf_diagnostics),
                "routed_board_path": str(routed_board),
                "leaf_placed_board": str(pre_route_board),
                "round_preview_pre_route_front": round_preview_pre_route_front,
                "round_preview_pre_route_back": round_preview_pre_route_back,
                "round_preview_pre_route_copper": round_preview_pre_route_copper,
                "round_preview_routed_front": round_preview_routed_front,
                "round_preview_routed_back": round_preview_routed_back,
                "round_preview_routed_copper": round_preview_routed_copper,
                "failed": True,
            },
            route_timing,
        )

    route_timing["route_local_subcircuit_total_s"] = round(
        max(0.0, time.monotonic() - route_total_start), 3
    )
    return (
        {
            "enabled": True,
            "skipped": False,
            "reason": "",
            "router": "kicad-routing-tools",
            "traces": int(imported_copper.get("trace_count", 0)),
            "vias": int(imported_copper.get("via_count", 0)),
            "total_length_mm": float(imported_copper.get("total_length_mm", 0.0)),
            "round_board_illegal_pre_stamp": round_board_illegal_pre_stamp,
            "round_board_pre_route": round_board_pre_route,
            "round_board_routed": round_board_routed,
            "routed_internal_nets": routed_internal,
            "failed_internal_nets": failed_internal,
            "_trace_segments": [
                copy.deepcopy(trace) for trace in imported_copper.get("traces", [])
            ],
            "_via_objects": [
                copy.deepcopy(via) for via in imported_copper.get("vias", [])
            ],
            "routing_stats": routing_stats,
            "validation": validation,
            "gnd_pour_summary": copy.deepcopy(gnd_pour_summary),
            "render_diagnostics": copy.deepcopy(leaf_diagnostics),
            "leaf_legality_repair": copy.deepcopy(legality_repair),
            "routed_board_path": str(routed_board),
            "leaf_placed_board": str(pre_route_board),
            "round_preview_pre_route_front": round_preview_pre_route_front,
            "round_preview_pre_route_back": round_preview_pre_route_back,
            "round_preview_pre_route_copper": round_preview_pre_route_copper,
            "round_preview_routed_front": round_preview_routed_front,
            "round_preview_routed_back": round_preview_routed_back,
            "round_preview_routed_copper": round_preview_routed_copper,
            "failed": False,
        },
        route_timing,
    )


def _stamp_trivial_leaf(
    *,
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
    round_index: int | None,
    generate_diagnostics: bool,
    render_intermediate: bool,
    fast_smoke_mode: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Stamp a placed-but-not-routed PCB for a leaf with no internal nets.

    A trivial leaf (e.g. a battery holder with both terminals exposed
    via interface ports) has nothing to route, but we still want a
    real ``leaf_routed.kicad_pcb`` on disk so:

    * ``pin_best_leaves`` can promote a chosen round like every other
      leaf (no more "no-snapshots" status).
    * The GUI snapshot picker can show this leaf's rounds and let the
      user pin keyboard-style.
    * The composer's blocker extraction reads the same way from
      every leaf.

    The resulting PCB is just the placed footprints; no traces, no
    vias, no the autorouter invocation. The returned routing dict
    advertises ``traces=0, vias=0, reason="no_internal_nets"`` and
    ``routed_board_path`` set to the same file as
    ``leaf_placed_board``.
    """
    route_timing: dict[str, float] = {}
    route_total_start = time.monotonic()

    artifact_paths = resolve_artifact_paths(
        Path(extraction.subcircuit.schematic_path).parent,
        extraction.subcircuit.id,
    )
    pre_route_board = Path(artifact_paths.artifact_dir) / "leaf_placed.kicad_pcb"
    routed_board = Path(artifact_paths.artifact_dir) / "leaf_routed.kicad_pcb"

    legality_start = time.monotonic()
    repaired_components, legality_repair = repair_leaf_placement_legality(
        extraction,
        solved_components,
        cfg,
    )
    route_timing["legality_repair_s"] = round(
        max(0.0, time.monotonic() - legality_start), 3
    )

    source_pcb = Path(cfg.get("subcircuit_route_source_pcb", cfg.get("pcb_path", "")))
    if not source_pcb.exists():
        source_pcb = Path(extraction.subcircuit.schematic_path).with_suffix(".kicad_pcb")
    if not source_pcb.exists():
        # Without a source board we can't stamp; degrade to the original
        # behaviour (no PCB on disk, validation accepted because there's
        # nothing to fail). pin_best_leaves will report "no-snapshots"
        # like before, which is honest in this configuration.
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": "no_internal_nets",
                "router": "kicad-routing-tools",
                "traces": 0,
                "vias": 0,
                "total_length_mm": 0.0,
                "routed_internal_nets": [],
                "failed_internal_nets": [],
                "_trace_segments": [],
                "_via_objects": [],
                "validation": {
                    "accepted": True,
                    "reason": "no_internal_nets",
                    "board_exists": False,
                    "shorts": 0,
                    "clearance_violations": 0,
                    "track_summary": {"traces": 0, "vias": 0},
                },
                "failed": False,
            },
            route_timing,
        )

    route_input_board = copy.deepcopy(extraction.local_state)
    route_input_board.components = copy.deepcopy(repaired_components)
    route_input_board.traces = []
    route_input_board.vias = []
    # Trivial leaves skip the autorouter, so the pre-route stamp is also the final
    # stamp. Apply the same shrink-and-center as the main-path silk re-stamp so the
    # rounded silk hugs Edge.Cuts and the standalone leaf opens centered on its A4
    # page (the parent composer re-bases each leaf on load, so this is placement-safe).
    _new_outline = _outline_around_geometry(route_input_board.components, cfg)
    if _new_outline is not None:
        _new_tl, _new_br = _new_outline
        _delta, _centered_outline = _center_on_leaf_page(_new_tl, _new_br, cfg)
        if abs(_delta.x) > 1e-6 or abs(_delta.y) > 1e-6:
            from kicraft.autoplacer.brain.leaf_geometry import (
                copy_components_with_translation,
            )
            route_input_board.components = copy_components_with_translation(
                route_input_board.components, _delta
            )
        route_input_board.board_outline = _centered_outline
    route_input_board.silkscreen = _silk_for_leaf(
        extraction, route_input_board.components, cfg
    )

    stamp_start = time.monotonic()
    route_adapter = KiCadAdapter(str(source_pcb), config=cfg)
    route_adapter.stamp_subcircuit_board(
        route_input_board,
        output_path=str(pre_route_board),
        clear_existing_tracks=True,
        clear_existing_zones=True,
        remove_unmapped_footprints=True,
    )
    route_timing["stamp_pre_route_board_s"] = round(
        max(0.0, time.monotonic() - stamp_start), 3
    )

    # No the autorouter to run; the placed board IS the routed board.
    shutil.copy2(pre_route_board, routed_board)

    round_board_pre_route = ""
    round_board_routed = ""
    if round_index is not None:
        round_prefix = f"round_{int(round_index):04d}"
        for src_path, suffix in (
            (pre_route_board, "leaf_placed"),
            (routed_board, "leaf_routed"),
        ):
            if not src_path.exists():
                continue
            dst = src_path.parent / f"{round_prefix}_{suffix}{src_path.suffix}"
            shutil.copy2(src_path, dst)
            if suffix == "leaf_placed":
                round_board_pre_route = str(dst)
            else:
                round_board_routed = str(dst)

    diagnostics_payload: dict[str, Any]
    if generate_diagnostics and render_intermediate and not fast_smoke_mode:
        try:
            _no_drc_opts = LeafStageOpts(
                render_board_views=True,
                write_drc_json=False,
                write_drc_report=False,
                render_drc_overlay=False,
            )
            diagnostics_payload = generate_leaf_diagnostic_artifacts(
                artifact_dir=artifact_paths.artifact_dir,
                pre_route_board=str(pre_route_board),
                routed_board=str(routed_board),
                pre_route_validation={"accepted": True, "reason": "no_internal_nets"},
                routed_validation={"accepted": True, "reason": "no_internal_nets"},
                pre_route_opts=_no_drc_opts,
                routed_opts=_no_drc_opts,
                build_contact_sheet=False,
                quiet_render=fast_smoke_mode,
            )
        except Exception as exc:
            diagnostics_payload = {"skipped": True, "reason": f"diag_failed:{exc}"}
    else:
        diagnostics_payload = {"skipped": True, "reason": "fast_smoke_or_no_diag"}

    route_timing["route_local_subcircuit_total_s"] = round(
        max(0.0, time.monotonic() - route_total_start), 3
    )

    return (
        {
            "enabled": True,
            "skipped": True,
            "reason": "no_internal_nets",
            "router": "kicad-routing-tools",
            "traces": 0,
            "vias": 0,
            "total_length_mm": 0.0,
            "round_board_illegal_pre_stamp": "",
            "round_board_pre_route": round_board_pre_route,
            "round_board_routed": round_board_routed,
            "routed_internal_nets": [],
            "failed_internal_nets": [],
            "_trace_segments": [],
            "_via_objects": [],
            "validation": {
                "accepted": True,
                "reason": "no_internal_nets",
                "board_exists": True,
                "shorts": 0,
                "clearance_violations": 0,
                "track_summary": {"traces": 0, "vias": 0},
            },
            "render_diagnostics": diagnostics_payload,
            "leaf_legality_repair": copy.deepcopy(legality_repair),
            "routed_board_path": str(routed_board),
            "leaf_placed_board": str(pre_route_board),
            "failed": False,
        },
        route_timing,
    )
