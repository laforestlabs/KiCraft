"""Default configuration — project-agnostic placement/routing engine defaults.

Project-specific overrides (ic_groups, component_zones, etc.) live in a
per-project JSON file (e.g. ``LLUPS_autoplacer.json``).  Use
``discover_project_config()`` to locate it automatically, then
``load_project_config()`` to parse it.
"""

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = {
    # Trace + via floors match OSH Park 2-layer service:
    # https://docs.oshpark.com/services/two-layer/
    # Signal: 0.153mm, just above the 6 mil (0.1524mm) floor -- 0.1524 itself
    # rounds to 152 µm in the DSN, below the 152.4 µm minimum (the same trap
    # freerouting_fine_pitch_track_mm documents below).
    # Power: 0.5mm matches the Power netclass in LLUPS.kicad_pro.
    "signal_width_mm": 0.153,
    "power_width_mm": 0.5,
    # Via: 0.3mm drill + 0.15mm ring = 0.6mm dia (5.91 mil ring, above 5 mil floor)
    "via_drill_mm": 0.3,
    "via_size_mm": 0.6,
    # Placement clearance — minimum gap between component bounding boxes.
    # 2.84mm gives the SA solver enough breathing room to avoid courtyard
    # overlaps while staying within typical 4-layer trace+via budgets.
    # Tuned via overnight parameter sweep (r=-0.59, top-quintile median).
    "placement_clearance_mm": 2.84,
    # Power nets (common names — projects override with their own)
    "power_nets": set(),
    # Placement (spread components — room to route, no courtyard overlaps)
    "placement_grid_mm": 1.20,
    # Gap added to a component's courtyard to derive the grid pitch when an
    # array/matrix leaf is placed programmatically (see
    # brain/array_placement.py). Only used when an ArraySpec omits pitch_mm.
    "array_gap_mm": 0.6,
    "edge_margin_mm": 6.0,
    "force_attract_k": 0.02,
    "force_repel_k": 200.0,
    "cooling_factor": 0.97,
    "sa_refine_enabled": True,
    # Hard ceiling on SA iterations. Adaptive convergence (see
    # `sa_refine_no_improve_break`) typically exits well before this limit
    # once score has plateaued. Lowered from 1000 -> 300 because the parent
    # solve has ~13 components and SA usually plateaus inside ~150 iters;
    # leaf solves rarely benefit from > 300 either, and runaway iterations
    # were the bulk of a 232 s solve on LLUPS.
    "sa_refine_iterations": 300,
    # Adaptive break: exit SA once no new best score has been found for
    # this many consecutive iterations. 150 is wide enough to ride out
    # high-temp Metropolis noise, narrow enough to save wall-clock once
    # SA has truly converged.
    "sa_refine_no_improve_break": 150,
    "sa_refine_initial_temp": 5.0,
    # Faster cooling (0.952 vs 0.995) lets SA spend more iterations near the
    # target temperature instead of crawling through high-temp randomness.
    "sa_refine_cooling_rate": 0.952,
    # Larger SA move radius helps escape local minima found by force solve.
    "sa_refine_move_radius_mm": 5.63,
    "sa_refine_swap_probability": 0.3,
    # Higher rotation probability gives the solver more chances to find a
    # better orientation per component during refinement.
    "sa_refine_rotation_probability": 0.44,
    # Placement solver iterations -- raised from 300 to 3332 because the
    # sweep found measurable score gains for harder leaves at the higher
    # cap. Easy leaves still terminate at placement_convergence_threshold
    # well before this limit.
    "max_placement_iterations": 3332,
    "placement_convergence_threshold": 0.5,
    "placement_score_every_n": 1,
    "intra_cluster_iters": 80,
    # Placement diversity: "cluster" (centroid-based) or "random" (uniform scatter).
    # MINOR mode always uses "cluster"; MAJOR uses "random" 50% of the time;
    # EXPLORE always uses "random".  Set by autoexperiment per mutation mode.
    "scatter_mode": "cluster",
    # Temperature reheat: at 50% of max_iterations, apply a random perturbation
    # kick to escape local minima. 0 = disabled, 0.1 = moderate, 0.3 = aggressive.
    "reheat_strength": 0.1,
    # Randomize IC-group internal layout (radius spread + angular shuffle).
    # True for MAJOR/EXPLORE, False for MINOR.
    "randomize_group_layout": False,
    # Courtyard overlap padding — extra margin (mm) added when scoring
    # courtyard overlaps.  Drives the optimizer to leave breathing room.
    # 1.30mm chosen via parameter sweep (top-quintile median).
    "courtyard_padding_mm": 1.30,
    # Final courtyard-separation legalization (Step 16, the genuine last
    # geometry step in solve()).  The earlier overlap passes run BEFORE
    # pinned-position restore / board clamp / keep-out clear, each of which
    # can nudge a part back into a neighbour's courtyard with nothing
    # re-resolving it -- the root cause of the systematic courtyards_overlap
    # DRC failures.  This pass runs last and pushes only the unlocked partner
    # of each overlapping SAME-SIDE pair apart by a hair, so it cleans up
    # whatever the final steps introduced without re-bloating a tight board.
    "resolve_courtyard_overlaps": True,
    # Target gap (mm) the final courtyard pass separates overlapping
    # courtyards by -- just enough to clear the courtyards_overlap DRC, not
    # the full placement clearance (which would undo tight packing).
    "courtyard_overlap_min_gap_mm": 0.15,
    # Verify-gate severity thresholds for a residual courtyard overlap the
    # placement pass could not remove (e.g. between two pinned parts). A clip
    # shallower than BOTH thresholds, on a board that is otherwise electrically
    # perfect (0 shorts / 0 unconnected), is a WARNING (board still exported +
    # 3D-rendered for inspection), not a hard fab failure -- courtyards carry an
    # assembly margin, so a fraction-of-a-mm clip usually still assembles. A
    # deeper overlap (parts physically colliding) still hard-fails.
    "courtyard_overlap_warn_penetration_mm": 0.5,
    "courtyard_overlap_warn_area_mm2": 0.5,
    # Pad inset margin — minimum distance (mm) all electrical pads must be
    # inside the board Edge.Cuts boundary.  Pads outside are unfabricatable.
    "pad_inset_margin_mm": 0.3,
    # Edge jitter — maximum random displacement (mm) along the assigned edge
    # for edge-pinned components (connectors, mounting holes).  Provides
    # placement diversity across rounds while keeping components on edges.
    "edge_jitter_mm": 5.0,
    # Connector gap — spacing (mm) between connectors grouped on the same edge.
    "connector_gap_mm": 3.58,
    # Connector edge inset — distance (mm) from board edge to the nearest
    # edge of the connector body.  0 = flush, positive = inset, negative =
    # overhang.  Used only by the single-board leaf solver (_pin_edge_components)
    # for intra-leaf positioning; the parent composer uses the signed
    # connector_edge_overhang_mm below for the final board edge.
    "connector_edge_inset_mm": 2.5,
    # Connector edge overhang — how far (mm) an edge-mount connector's mouth
    # sits PROUD of the composed board edge (board stops this far short of the
    # mouth) so a plug clears the FR4.  USB-C / barrel jacks are unusable when
    # the board overhangs the port, so this is a positive flush-or-overhang
    # margin, never an inset.
    "connector_edge_overhang_mm": 0.5,
    # Copper-to-edge clearance (mm) the composed board edge keeps from an
    # edge-mount connector's edge-facing SOLDER PADS. The parent composer places
    # the edge at ``anchor - inward + outward`` with this as ``inward``, so the
    # connector's pads clear the cut line by this much while its mouth still
    # overhangs by ``connector_edge_overhang_mm - this``. 0 reproduces the old
    # pads-flush-with-edge behavior (copper_edge_clearance DRC on BNC GND shields,
    # flush switches, etc.). Matches the board's min_copper_edge_clearance rule.
    "connector_edge_pad_clearance_mm": 0.2,
    # Keep ordinary parts behind an edge-pinned connector's pads (leaf solve).
    # The composed parent board edge is drawn OUTBOARD of the connector's pads,
    # so a companion left outboard strands in the copper-to-edge clearance band
    # (KC-S8PC37 R8). 0 disables. Pairs with connector_edge_owns_side_mm below,
    # which keeps OTHER leaves' blocks from crowding past the connector at the
    # parent level (so the connector stays the flush edge extremity).
    "connector_edge_companion_clearance_mm": 0.5,
    # At parent compose, shift each edge-zoned block outboard so it is the board
    # extremity on its side -> its connector defines the board edge and stays
    # flush, instead of another block edging past it and stranding the connector
    # inboard (KC-S8PC37 J1). Outboard-only, so it can't create overlaps.
    "connector_edge_block_extremity": True,
    # Connector-bank orientation (pcb-area-compaction-plan Phase 6). A row of
    # short single-row pin headers on an edge (16x 1x3 servo headers, an LED /
    # sensor breakout bank) defaults to pins-PARALLEL-to-edge, which strings the
    # board out along the edge (16x1x3 -> ~200mm) AND interleaves each header's
    # signal/power pads on the shared-GND line, fragmenting the GND pour
    # (KC-8A3US3). When enabled, such headers are turned so their pin axis is
    # PERPENDICULAR to the edge: the row packs by body-width (~3x shorter edge)
    # and every same-index pad (all the GNDs) lines up into one uninterrupted
    # strip the pour can follow. Mouthed connectors (USB/barrel), 2-pin screw
    # terminals, multi-row (2xN) and long single headers are left untouched.
    "connector_perp_orientation": True,
    # Max pad-row span (mm, pad-center to pad-center) a header may have and
    # still be turned perpendicular. Above this a lone header would stab too
    # deep into the board, so it keeps the along-edge orientation. 15mm ~ 1x6.
    "connector_perp_max_len_mm": 15.0,
    # Max off-axis pad spread (mm) still considered a single row; above this the
    # connector is multi-row (2xN IDC) and is left along the edge.
    "connector_perp_row_tol_mm": 1.2,
    # Mounting hole geometry. ``count`` is the target number of holes;
    # actual count is whatever the source PCB project ships. ``screw``
    # is a free-form reference (e.g. "M2.5", "M3", "#4-40") used to
    # document the intended fastener.  ``hole_diameter_mm`` is the drill
    # clearance diameter for that screw (M3 ≈ 3.2 mm including fab tolerance).
    #
    # ``pad`` describes the exposed copper / annular ring around the hole
    # and ``keepout`` describes the component-free zone around the hole.
    # ``size_mm`` is the radius from hole center to the outer edge of the
    # shape (for ``circle`` it is literally the radius; for ``hexagon`` and
    # ``square`` it is the half-width across the flats; the inscribed
    # circle has this radius). Only ``keepout.size_mm`` is currently
    # consumed by the placer -- it is the inward distance from each board
    # edge that a corner-anchored mounting hole reserves, equivalent to
    # the legacy flat ``mounting_hole_keep_in_mm`` knob.  ``pad.*`` and
    # ``keepout.shape`` are plumbing for a future footprint generator and
    # do not yet alter geometry.
    "mounting_holes": {
        "count": 2,
        "screw": "M3",
        "hole_diameter_mm": 3.2,
        "pad": {
            "shape": "hexagon",
            "size_mm": 3.0,
        },
        "keepout": {
            "shape": "hexagon",
            "size_mm": 4.0,
        },
    },
    # RF antenna near-field keep-clear, enforced at PLACEMENT time (deliberately
    # NOT baked into the .kicad_mod, where it would bloat the footprint and the
    # copper pour). Keyed by a footprint-name glob (fnmatch, case-insensitive)
    # -> antenna-end rect in the footprint's own LOCAL frame (mm).
    # hardware.keepout_extract synthesizes this rect for any placed footprint
    # whose name matches, transforms it to board coords by the footprint's
    # placed position/rotation, and the placer keeps other parts out of it
    # (the matched footprint itself is exempt). A matched footprint's own
    # internal keep-out (e.g. Fix 0's on-module antenna strip) is also honored;
    # the two are unioned, so both the on-module strip and the near-field hold.
    # Geometry covers the antenna end of the module plus a near-field margin
    # beyond it; tune per module. Frames (measured from the library footprints):
    #   ESP32-S3-WROOM-1 (WIRELM): long axis +y, antenna at -y, body x[-9,9].
    #   ESP32-S3-MINI-1  (BULETM): long axis +y, antenna at -y, body x[-7.7,7.8]
    #     (rect geometry-derived from the footprint; verify vs the datasheet keep-out).
    #   ESP32-WROOM-32x  (WIFI):   long axis +x, antenna at -x, body y[-9,9].
    "antenna_keepouts": {
        "*ESP32-S3-WROOM-1*": {
            "x_min": -12.0,
            "y_min": -24.0,
            "x_max": 12.0,
            "y_max": -9.0,
        },
        "*ESP32-S3-MINI-1*": {
            "x_min": -10.0,
            "y_min": -20.0,
            "x_max": 10.0,
            "y_max": -6.0,
        },
        "*ESP32-WROOM-32*": {
            "x_min": -24.0,
            "y_min": -12.0,
            "x_max": -9.0,
            "y_max": 12.0,
        },
    },
    # Orderedness — how strongly passives are snapped into neat rows/columns.
    # 0.0 = organic/force-directed layout, 1.0 = full grid alignment.
    # Intermediate values blend proportionally.  Searchable by autoexperiment.
    "orderedness": 0.3,
    # PlacementScore sub-score weights (psw_*) — the definition of "good
    # placement" the solver optimizes toward. Defaults MUST equal
    # types.DEFAULT_PLACEMENT_WEIGHTS so an unmodified config scores
    # byte-identically; a tuned config overrides individual weights to bias the
    # placer (e.g. raise psw_bbox_packing / psw_aspect_ratio for smaller boards,
    # psw_topology_structure / psw_group_coherence for orderliness, psw_net_distance
    # / psw_crossover_score for routability). Searchable by the tuner; weights are
    # normalized by their sum in compute_total, so only ratios matter.
    "psw_net_distance": 0.20,
    "psw_crossover_score": 0.17,
    "psw_bbox_packing": 0.15,
    "psw_group_coherence": 0.08,
    "psw_topology_structure": 0.05,
    "psw_aspect_ratio": 0.02,
    # Scorer normalization modes (area-compaction Phase 2). "content" makes
    # net-distance canvas-invariant (normalized by the component-area-derived
    # ideal-canvas diagonal, not the arbitrary solve-canvas diagonal);
    # "strict" zeroes compactness below ~5% fill. None (default) = the legacy
    # board_diag/legacy scoring EVERYWHERE: replay A/B 2026-07-02 showed both
    # a global and a leaf-scoped flip regress routing (parent 530/535; leaf
    # 535 J1) because the psw weights were tuned against the legacy score
    # shapes. These are the knobs for the pending CMA-ES retune campaign
    # (plan Phase 2 item 4); flip the defaults only with retuned weights.
    "placement_score_net_scale": None,
    "placement_compactness_curve": None,
    # Through-hole backside threshold — THT components with bounding-box area
    # above this value (mm²) are placed on B.Cu so SMT parts can use F.Cu.
    # SMT passives always stay on F.Cu — IC group connectivity forces keep
    # them near their THT group leaders, achieving dual-sided board usage.
    # 130mm² (vs former 50) keeps small-medium THT on the front side; only
    # genuinely large THT (battery holders, big connectors) move to B.Cu.
    "tht_backside_min_area_mm2": 130.0,
    # SMT opposite THT — when True, actively attract SMT components on F.Cu
    # toward XY regions occupied by large back-side THT components.  This
    # uses board space efficiently by placing SMT on the opposite side of
    # THT footprints.  Adds an attraction force (0.3× force_attract_k) and
    # a small scoring bonus (~5% weight) for SMT-over-THT overlap.
    "smt_opposite_tht": True,
    # Align large pairs — when True, detect pairs of large non-passive
    # components with similar footprints and force them to be placed
    # side-by-side (aligned on one axis).  Only applies to components
    # with area above tht_backside_min_area_mm2.
    "align_large_pairs": True,
    # Component zone constraints — per-reference placement rules.
    # Each key is a component reference; value is a dict with one of:
    #   {"edge": "left"|"right"|"top"|"bottom"}  — snap to named edge, lock
    #   {"zone": "center-bottom"|"top-left"|...}  — confine to board region
    #   {"corner": "top-left"|"top-right"|"bottom-left"|"bottom-right"} — pin
    # Unassigned connectors fall back to nearest-edge heuristic.
    "component_zones": {},
    # Signal flow order — ordered list of IC group leader references.
    # Biases cluster centroids along the X-axis (left-to-right) during
    # initial placement.  Gives the layout a natural signal-flow direction.
    "signal_flow_order": [],
    # Thermal
    "thermal_refs": [],
    "thermal_radius_mm": 3.0,
    # FreeRouting
    "freerouting_jar": os.path.expanduser("~/.local/lib/freerouting-1.9.0.jar"),
    # Java runtime used to launch the FreeRouting jar. "java" resolves via PATH;
    # the runner additionally searches ~/.local/lib and /usr/lib/jvm so a
    # user-local JRE works even under the minimal PATH a systemd unit runs with.
    # Set to an absolute path to pin a specific JRE.
    "java_bin": "java",
    "freerouting_timeout_s": 60,
    "freerouting_max_passes": 20,
    # Parent inter-leaf routing convergence (C2): a parent with many cross-leaf
    # nets needs more passes + wall-time than the flat defaults, or the densest
    # interconnects stay unrouted (the dominant parent-stage unconnected cause).
    # When the parent's inferred_interconnect_net count reaches the threshold the
    # PARENT route (only) is raised to these budgets. Bounded; only raises.
    "parent_dense_interconnect_threshold": 10,
    "parent_dense_max_passes": 40,
    "parent_dense_timeout_s": 180,
    # Power-first parent routing: freerouting 1.9.0 routes in board item-list
    # order with no net priority, so the wide power-class nets -- which need
    # the fattest clear corridor -- are effectively routed last and end up
    # walled off by thin-net copper (KC-ZRAUR7: VBUS split in two islands on a
    # 55%-empty board). When enabled, the parent route runs an extra
    # freerouting phase FIRST with only the power nets connectable (every
    # other net's DSN pins emptied; pads/wiring stay obstacles), then the
    # normal full route locks that power copper like leaf copper. A failed
    # phase 1 falls through to the single-phase flow. Kill switch below.
    "parent_power_first": True,
    "parent_power_first_timeout_s": 120,
    # Parent freerouting timeout scales with component count, mirroring the
    # leaf budget at a lower rate (the parent starts from pre-routed leaf
    # copper, so per-component cost is lower than routing from scratch).
    # FreeRouting processes the ENTIRE board -- every component, every fixed
    # leaf wire -- even when only a few interconnect nets remain unrouted, so
    # wall-time scales with board size, not unrouted-net count. A 200-LED
    # parent needs ~170 s (72 s routing + 94 s optimization) with only 3
    # unrouted nets; the fixed 60 s default killed it mid-route (rc=-1).
    # Budget is max(freerouting_timeout_s, n_components * s_per_component), capped.
    "parent_freerouting_s_per_component": 1.0,
    "parent_freerouting_timeout_cap_s": 600,
    # Leaf freerouting timeout scales with component count: a large array leaf
    # (e.g. a 200-LED matrix, ~600 nets) routes fine but takes minutes, and the
    # fixed 60s default cut freerouting off mid-route. The leaf budget is
    # max(freerouting_timeout_s, n_components * s_per_component), capped. Small
    # leaves are unaffected (they finish in seconds well under the floor).
    "leaf_freerouting_s_per_component": 4.0,
    "leaf_freerouting_timeout_cap_s": 1200,
    # Separate pass cap for leaf routing. Leaves are smaller and need
    # less optimization than the parent board, so we keep this lower by
    # default. When unset, leaves fall back to freerouting_max_passes.
    "leaf_freerouting_max_passes": 12,
    # Hide the FreeRouting Swing window. For 2.x this is passed as
    # --gui.enabled=false. For 1.x the runner wraps the invocation in
    # xvfb-run when xvfb-run is on PATH (install xorg-x11-server-Xvfb).
    # If neither path is available the window still appears.
    "freerouting_hide_window": True,
    # Fine-pitch clearance handling. The DSN inherits the board's default
    # 0.2 mm clearance, which is wider than a dense connector's pad gaps
    # (USB-C ~0.10 mm), so the autorouter cannot escape its pad field. When
    # freerouting_clearance_mm is None, the router auto-detects the densest
    # different-net pad gap and, if it is below 0.2 mm, lowers the routing
    # clearance to clear it -- floored at freerouting_min_clearance_mm for fab
    # safety -- and reduces the track width to freerouting_fine_pitch_track_mm
    # so a trace can escape. Set freerouting_clearance_mm to force a value.
    "freerouting_clearance_mm": None,
    # Floor for the fine-pitch clearance auto-lower. Must be >= the fab spacing
    # floor (OSH Park 6 mil = 0.1524 mm). 0.1 let the auto-lower route the WHOLE
    # board down to ~0.10 mm (sub-floor, unmanufacturable) just to escape the
    # USB-C. 0.153 mm rounds to 153 µm in the DSN (>= 6 mil). USB-C pads tighter
    # than this need a LOCAL clearance exception, not a global sub-floor drop.
    "freerouting_min_clearance_mm": 0.153,
    # FreeRouting's internal geometry (polygonal pad approximations, integer
    # DSN units) differs from KiCad's exact shapes by up to ~1 µm, so a wire FR
    # places exactly at the clearance rule can measure just UNDER it in KiCad
    # DRC (observed 0.1520-0.1522 vs 0.1530 on rotated pads;
    # KC-9G4YPT/KC-CV4NE3/KC-HE2Q5T). Route with this guard ABOVE the DRC rule
    # (applied to every clearance token in the DSN, after all other rewrites);
    # the board keeps verifying at the real rule, so the guard can never mask a
    # genuine violation. Was 5 µm, but batch 20260716T011056Z run_13 measured a
    # 5.9 µm skew on a rotated aQFN pad (0.1521 vs 0.1530) -- just past it.
    # 10 µm covers the observed tail with margin, still 6.5% of the rule.
    "freerouting_clearance_guard_um": 10,
    # Fine-pitch escape track width. It is written to the DSN as integer microns
    # (int(round(mm*1000))), and KiCad's track-width DRC floor is
    # min_track_width = 0.1524 mm (6 mil, the OSH Park 2-layer minimum). 0.15 mm
    # rounds to 150 µm and 0.1524 mm rounds to 152 µm -- both BELOW the 152.4 µm
    # floor, so every fine-pitch escape became a track_width violation. 0.153 mm
    # rounds to 153 µm (>= floor, fab-legal) while staying narrower than the
    # 0.2 mm default so it still escapes dense pad fields.
    "freerouting_fine_pitch_track_mm": 0.153,
    # Leaf nesting (shaped compose): a leaf whose occupied copper leaves a
    # large interior hole (an LED-ring annulus) may host a smaller leaf fully
    # INSIDE that hole -- see docs/plans/shaped-compose-leaf-nesting.md. The
    # margin is held between the guest's occupied bbox and the hole's edge on
    # top of the per-rect inflation already applied at blocker extraction;
    # holes with an inscribed rect smaller than the minimum side are not
    # worth proposing into. The hole itself stays nest_hole_standoff_mm back
    # from real copper (decoupled from the min_side/2 gap-sealing radius,
    # which is topology, not clearance): effective copper-to-copper spacing
    # for a nested pair is ~0.5 pad-margin each side + standoff + margin,
    # generous vs the 0.2 mm clearance rule.
    "nest_margin_mm": 1.0,
    "nest_min_hole_side_mm": 8.0,
    "nest_hole_standoff_mm": 1.0,
    # Leaf acceptance: a leaf must route all of its *signal* nets. Power/ground
    # nets are excluded automatically (they close on the post-route pour, not on
    # leaf routing). Set to None to disable (historical lenient behaviour). This
    # makes leaf_accepted reflect reality instead of accepting unrouted leaves
    # (e.g. a USB-C connector whose CC pin never escaped its pad field) that then
    # silently fail parent routing.
    "leaf_acceptance_max_unconnected": 0,
    # GND zone pour — automatically created/updated to cover full board.
    # Set gnd_zone_net to "" to disable automatic zone creation.
    "gnd_zone_net": "GND",
    "gnd_zone_layer": "B.Cu",
    "gnd_zone_margin_mm": 0.5,
    "zone_clearance_mm": 0.3,
    "zone_min_thickness_mm": 0.25,
    "zone_thermal_gap_mm": 0.5,
    "zone_thermal_spoke_mm": 0.5,
    # GND pour pad connection: "full" (solid, default) or "thermal". Solid is the
    # plane-correct choice (matches power_plane_pad_connection): a dense pin field
    # -- an ESP32 module's GND ring, a fine-pitch connector -- can't fit the gap a
    # thermal-relief spoke needs, so <2 spokes resolve and KiCad raises
    # starved_thermal. Solid has no spokes to starve; it only adds same-net copper
    # so it can't short or pinch another net. Use "thermal" for hand-assembly.
    "gnd_plane_pad_connection": "full",
    # Full bottom-layer GND plane (default on): after routing, pour a B.Cu GND
    # zone over the whole board (the ZONE_FILLER keeps it clear of rule-area
    # keepouts like the WROOM antenna) and stitch large GND/thermal pads into it
    # with a dense thermal-via array, so the plane connects and boxed-in center
    # pads (e.g. the WROOM exposed pad) escape to ground.
    "gnd_plane_enabled": True,
    # Power-plane pour (default on): after routing, pour the primary power rail
    # (auto-detected as the power net with the most pads, e.g. VBUS on a USB
    # board) as a plane on the layer opposite GND. This ties paired connector
    # power pads, the regulator input, and bulk caps through copper instead of
    # asking the autorouter to thread pad-to-pad traces through a dense pad
    # field. Poured at a higher priority than GND so the two coexist on a layer.
    # Set power_plane_nets to an explicit list to override auto-detection.
    "power_plane_enabled": True,
    "power_plane_layer": "F.Cu",
    "power_plane_nets": None,
    "power_plane_max_nets": 1,
    "power_plane_priority": 1,
    # Power strand repair (default on): the power pour fragments around foreign
    # copper exactly like the GND plane does and can strand a supply pad on its
    # own fill island (fine-pitch parts especially -- KC-Z57JEZ +3V3). Tie each
    # stranded power cluster back with the same guarded-track repair the GND
    # plane gets. Shares gnd_strand_repair_max_mm for the tie-length cap.
    "power_strand_repair_enabled": True,
    # Auto power-tie (default on): before routing, route a locked tie around any
    # connector whose spread power pads (e.g. USB-C VBUS on both sides) would
    # otherwise fragment the power pour into disconnected islands. The tie runs
    # around the footprint bounding box so it never crosses other pads.
    "auto_power_tie": True,
    "power_tie_margin_mm": 1.0,
    "power_tie_exclude_refs": [],
    # Auto signal-escape (default on): before routing, pre-route short locked
    # radial escapes out of the *signal* pads of a dense connector (the same
    # >=2-spread-power-pad signature auto_power_tie uses) so the autorouter can
    # finish nets like a USB-C CC pin -> its pulldown resistor from open copper
    # instead of abandoning them boxed-in. Collision-guarded, so it never shorts.
    "auto_signal_escape": True,
    "signal_escape_exclude_refs": [],
    "signal_escape_length_mm": 1.5,
    "thermal_via_pitch_mm": 1.2,
    "thermal_via_inset_mm": 0.5,
    "thermal_pad_area_mm2": 4.0,
    # Ignorable DRC patterns — list of regex strings.  During post-route
    # DRC validation, if ALL significant violations match at least one
    # pattern (searched against the violation description text), they are
    # treated as ignorable.  This is in addition to the automatic
    # footprint-baseline clearance heuristic.
    "ignorable_drc_patterns": [],
    # Footprint refs whose internal clearance DRCs may be ignored when
    # the report parser cannot reliably extract refs from every violation.
    "ignorable_footprint_refs": [],
    # Explicit IC groups (IC + supporting components that should stay together).
    # Each key is the group leader (typically an IC reference), value is a list
    # of supporting component references.  Optional — groups are auto-discovered
    # from .kicad_sch files when not specified manually.
    "ic_groups": {},
    # Human-readable group labels for silkscreen annotation.
    "group_labels": {},
    # --- Search space flags ---
    # When True, batteries/connectors/mounting holes are NOT auto-locked;
    # edge_compliance scoring still incentivizes edge placement.
    "unlock_all_footprints": True,
    # When True, the autoexperiment loop can vary board_width_mm / board_height_mm.
    "enable_board_size_search": True,
    # Default board dimensions (mm) — overridden per-round when board size search is active.
    "board_width_mm": 90.0,
    "board_height_mm": 58.0,
    # Subcircuit margin — extra space (mm) added around the tight bounding
    # box of component positions when building a local subcircuit board.
    # 10.82mm gives the SA solver enough slack to find good arrangements;
    # the previous 5mm was the strongest leaf-side bottleneck (r=+0.59),
    # and at 1.5mm dense leaves (e.g., BOOST 5V) failed legality repair
    # on ~50% of seeds because D3/C6/L1 got forced into overlapping
    # placements the legalizer couldn't escape. The final leaf Edge.Cuts
    # is shrunk post-route by _outline_around_geometry to hug silk anyway,
    # so a generous extraction margin costs nothing in the final geometry
    # -- it only buys the placement solver more search room.
    "subcircuit_margin_mm": 10.82,
    # Leaf solve-canvas derivation (PCB area-compaction plan, Phase 1).
    # "content": the canvas is sized from component area (see
    # derive_content_canvas) -- Σ physical-bbox area / leaf_canvas_fill_target,
    # near-square, floored for the largest part + clearance packing. This
    # replaces the seed-scatter-bbox canvas that let 11-part leaves inherit a
    # ~195 mm-wide board (RC1) and the flow targets spread across it (RC2).
    # "seed-bbox": the historical envelope (seed component bbox +
    # subcircuit_margin_mm), byte-for-byte, kept for replay A/B comparison.
    "leaf_canvas_mode": "content",
    "leaf_canvas_fill_target": 0.28,
    # Grow-on-failure ladder: when NO round is accepted at the fill target,
    # the leaf is re-extracted at each looser fill in turn; after the ladder
    # is exhausted it falls back to the seed-bbox canvas (today's behavior),
    # so a dense leaf that only routes with generous slack still routes --
    # fab-ready rate can't regress by construction.
    "leaf_canvas_fill_ladder": [0.22, 0.17],
    # Post-SA deterministic compaction squeeze (area-compaction Phase 3):
    # slides each unlocked leaf part toward the placed-bbox centroid as far
    # as legality allows, closing the slack force equilibrium leaves. None =
    # follow the canvas mode (on for "content", off for "seed-bbox" so that
    # mode stays byte-identical to history); True/False forces it.
    "leaf_compaction_pass": None,
    # Parent spacing — gap (mm) between child subcircuit bounding boxes when
    # composing them into the parent board.  1.17mm packs leaves tightly
    # without compromising routability (r=-0.41 in parents-only sweep).
    "parent_spacing_mm": 1.17,
    # Parent seed area overhead -- multiplier on total child area to set
    # the seed board's nominal area before the placer runs.  2.5 is the
    # historical default; 1.8-2.0 forces compaction (placer cannot
    # sprawl); 3.0+ leaves more slack at the cost of bigger final
    # outlines.  Per-candidate aspect ratios are swept independently
    # inside _search_best_layout so K=8 candidates already explore a
    # spread of seed shapes.
    "parent_seed_area_overhead": 2.5,
    # Intermediate per-round diagnostic renders (PNG board views, DRC overlays,
    # contact sheets) are produced inside every leaf routing round. They are
    # useful for interactive/debug runs but are pure overhead in headless
    # build/worker runs, where only the winning round's canonical renders
    # matter. Default True keeps interactive/debug behavior unchanged; the
    # build worker path sets this False to skip per-round diagnostics.
    "subcircuit_render_intermediate": True,
    # Trim losing rounds' heavy diagnostic artifacts after the winner is
    # chosen. When False (default), losing rounds keep their .kicad_pcb
    # (the manual-pin path in pins.py reads them) and small metadata JSON
    # (solved_layout/debug/metadata), but their per-round render PNGs and
    # DRC JSON/report files are deleted to shrink .experiments/. When True,
    # restores today's behavior -- every round keeps its full diagnostics.
    "keep_all_round_artifacts": False,
}


# Only true layout/routing heuristics belong here. Fab/circuit constraints
# (signal_width_mm, power_width_mm, via_drill_mm, via_size_mm, zone_*_mm,
# pad_inset_margin_mm, thermal_radius_mm) are dictated by the fab's minimum
# feature size and the schematic's current/voltage requirements -- they are
# NOT optimization knobs. ``freerouting_timeout_s`` stays a runtime budget (a
# cap, not a quality dial). ``freerouting_max_passes`` / ``leaf_freerouting_max_passes``
# ARE searched: more passes trade routing quality (shorter, fewer vias, more
# complete) against wall-time, and the reward weighs that wall-time axis directly,
# so they sit on the Pareto frontier rather than being a fixed budget. Board
# dimensions (board_width_mm, board_height_mm) are derived from leaf areas
# and enclosure constraints, not searched.
CONFIG_SEARCH_SPACE = {
    # --- Insensitive params: full original ranges retained ---
    # These showed |r| < 0.10 in both stages of the overnight sweep, so
    # narrowing their range would just constrain future searches without
    # gain. Widen if a future change makes them sensitive again.
    "orderedness": {"min": 0.0, "max": 1.0, "sigma": 0.05, "type": "float"},
    "reheat_strength": {"min": 0.0, "max": 0.4, "sigma": 0.05, "type": "float"},
    # force_attract_k floor narrowed 0.001 -> 0.02 = the DEFAULT_CONFIG
    # value. Empirical evidence on LLUPS (3 separate 5-round verifies):
    # mutated values of 0.0098, 0.0148, and 0.01 each produced 0/4
    # accepted candidates due to bh=160-200 mm sprawl, while default
    # 0.02 produced 1+/4 accepted candidates. The intermediate floor
    # of 0.01 was empirically equivalent to no floor at all -- the
    # placement is fragile to even small downward mutations because
    # PlacementScore saturates at >100 on sprawled layouts (nets and
    # crossings hit max), so SA's Metropolis acceptance can't distinguish
    # between sprawled and compact strongly enough to walk back.
    # Sigma stays at 0.01 so the mutator still searches upward (default
    # to ~0.05 at 3-sigma) but cannot drop below the empirically-viable
    # minimum. This is the load-bearing constraint -- the broad
    # mutator search space inherited from earlier sweeps assumed
    # ANY value in [0.001, 0.2] could yield viable placements; in
    # practice, only [default, 3-sigma-up] does for parent compose.
    "force_attract_k": {"min": 0.02, "max": 0.2, "sigma": 0.01, "type": "float"},
    "force_repel_k": {"min": 50.0, "max": 1000.0, "sigma": 50.0, "type": "float"},
    "cooling_factor": {"min": 0.80, "max": 0.999, "sigma": 0.02, "type": "float"},
    "edge_margin_mm": {"min": 0.5, "max": 15.0, "sigma": 0.5, "type": "float"},
    "sa_refine_initial_temp": {"min": 0.5, "max": 30.0, "sigma": 2.0, "type": "float"},
    # sa_refine_iterations floor 100 -> 250 (default 300). At 100 SA
    # has too few moves to escape a sprawled force-loop equilibrium:
    # going from bh=180 mm to bh=90 mm under the bumped bbox_packing
    # weight needs ~30+ sequential accept-toward-compact moves, each
    # bounded by sa_refine_move_radius_mm. 100 iterations after the
    # rotation/swap ratio splits leaves ~50 translations to do that
    # work, and the reheat-at-50% perturbation eats half of those.
    # 250 gives ~150 translations after reheat, enough to cross the
    # space deterministically. Mutator can still go up to 10000;
    # only the lower end is protected.
    "sa_refine_iterations": {"min": 250, "max": 10000, "sigma": 500, "type": "int"},
    "edge_jitter_mm": {"min": 0.0, "max": 15.0, "sigma": 1.0, "type": "float"},
    "intra_cluster_iters": {"min": 10, "max": 500, "sigma": 20, "type": "int"},
    "gnd_zone_margin_mm": {"min": 0.1, "max": 2.0, "sigma": 0.1, "type": "float"},
    "sa_refine_swap_probability": {"min": 0.0, "max": 1.0, "sigma": 0.05, "type": "float"},
    "placement_convergence_threshold": {"min": 0.01, "max": 2.0, "sigma": 0.1, "type": "float"},
    # --- Sensitive params: ranges narrowed to top-quintile [P10, P90] ---
    # min/max replaced from .experiments/param_sweep/proposed_param_ranges.json
    # Sigma roughly = 10% of new range so Gaussian mutation steps at a
    # sensible scale for the narrower interval.
    "placement_clearance_mm": {"min": 1.10, "max": 5.38, "sigma": 0.4, "type": "float"},
    "courtyard_padding_mm": {"min": 0.65, "max": 2.57, "sigma": 0.2, "type": "float"},
    "sa_refine_move_radius_mm": {"min": 1.52, "max": 7.41, "sigma": 0.6, "type": "float"},
    "connector_gap_mm": {"min": 0.46, "max": 7.50, "sigma": 0.7, "type": "float"},
    "max_placement_iterations": {"min": 829, "max": 4551, "sigma": 370, "type": "int"},
    "subcircuit_margin_mm": {"min": 6.97, "max": 13.38, "sigma": 0.6, "type": "float"},
    # Content-canvas fill target (area-compaction Phase 1): higher = tighter
    # boards but harder routing; the grow ladder catches per-leaf failures.
    "leaf_canvas_fill_target": {"min": 0.15, "max": 0.45, "sigma": 0.03, "type": "float"},
    "connector_edge_inset_mm": {"min": 0.47, "max": 4.32, "sigma": 0.4, "type": "float"},
    "sa_refine_cooling_rate": {"min": 0.9076, "max": 0.99, "sigma": 0.008, "type": "float"},
    "sa_refine_rotation_probability": {"min": 0.05, "max": 0.85, "sigma": 0.08, "type": "float"},
    "placement_grid_mm": {"min": 0.34, "max": 2.13, "sigma": 0.18, "type": "float"},
    "tht_backside_min_area_mm2": {"min": 80.0, "max": 176.0, "sigma": 10.0, "type": "float"},
    "parent_spacing_mm": {"min": 0.66, "max": 1.60, "sigma": 0.1, "type": "float"},
    # parent_seed_area_overhead: lower = tighter seed (forces compaction);
    # higher = looser (lets the placer sprawl).  1.5 floor avoids seeds
    # that violate the sum*0.6 / max-child + spacing safety floors;
    # 3.5 ceiling matches the legacy "lots of slack" upper end.
    "parent_seed_area_overhead": {"min": 1.5, "max": 3.5, "sigma": 0.2, "type": "float"},
    # --- Phase 1: routing-effort / routability knobs ---
    # These steer the autorouter and dense-connector escapes, NOT fab geometry.
    # max_passes is the dominant route-quality lever (effort vs wall-time); the
    # fab-readiness bottleneck (most boards don't route clean) lives here, not in
    # the spacing knobs the earlier sweeps tuned. Defaults: parent 20 / leaf 12 /
    # escape 1.5mm. Ranges centered on the defaults with room both ways.
    "freerouting_max_passes": {"min": 8, "max": 40, "sigma": 5, "type": "int"},
    "leaf_freerouting_max_passes": {"min": 6, "max": 30, "sigma": 4, "type": "int"},
    "signal_escape_length_mm": {"min": 0.5, "max": 3.0, "sigma": 0.3, "type": "float"},
    # --- Phase 2: PlacementScore sub-score weights ---
    # The objective the placer optimizes for IS these weights, so tuning them is
    # the highest-leverage lever for orderedness/size/routability. Bounded [0, 0.4]
    # each; compute_total normalizes by the weight sum so only ratios matter.
    "psw_net_distance": {"min": 0.0, "max": 0.4, "sigma": 0.05, "type": "float"},
    "psw_crossover_score": {"min": 0.0, "max": 0.4, "sigma": 0.05, "type": "float"},
    "psw_bbox_packing": {"min": 0.0, "max": 0.4, "sigma": 0.05, "type": "float"},
    "psw_group_coherence": {"min": 0.0, "max": 0.4, "sigma": 0.05, "type": "float"},
    "psw_topology_structure": {"min": 0.0, "max": 0.4, "sigma": 0.05, "type": "float"},
    "psw_aspect_ratio": {"min": 0.0, "max": 0.4, "sigma": 0.05, "type": "float"},
}


def normalize_bounds(
    key: str,
    lo: float,
    hi: float,
    spec: dict[str, Any] | None = None,
) -> tuple[float, float] | None:
    """Clamp and validate a (lo, hi) bound pair against a CONFIG_SEARCH_SPACE spec.

    Returns normalized (lo, hi) or None if the range is invalid (e.g. empty
    integer range after rounding, or non-finite inputs like NaN/Infinity).
    """
    import math as _math

    if spec is None:
        spec = CONFIG_SEARCH_SPACE.get(key)
    if spec is None:
        return None

    # Reject NaN/Infinity before any arithmetic (platform-dependent otherwise)
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return None
    if not (_math.isfinite(lo) and _math.isfinite(hi)):
        return None

    spec_min = float(spec["min"])
    spec_max = float(spec["max"])

    lo = max(spec_min, min(spec_max, lo))
    hi = max(spec_min, min(spec_max, hi))

    if lo > hi:
        lo, hi = hi, lo

    if spec.get("type") == "int":
        lo = _math.ceil(lo)
        hi = _math.floor(hi)
        if lo > hi:
            return None

    return (lo, hi)


PARAM_CONSTRAINTS = [
    ("via_drill_mm", "<", "via_size_mm"),
]


def enforce_param_constraints(config: dict[str, Any]) -> dict[str, Any]:
    """Fix cross-parameter constraint violations in a config dict.

    Modifies values in-place to ensure physical consistency:
    - via_drill_mm must be < via_size_mm (annular ring requirement)

    Returns the (possibly modified) config.
    """
    for key_a, op, key_b in PARAM_CONSTRAINTS:
        if key_a not in config or key_b not in config:
            continue
        a, b = float(config[key_a]), float(config[key_b])
        if op == "<" and a >= b:
            config[key_a] = b * 0.5
        elif op == "<=" and a > b:
            config[key_b] = a
    return config


_MOUNTING_HOLE_SHAPES = ("hexagon", "circle", "square")


def _validate_mounting_holes(section: Any, source: str) -> None:
    """Validate a ``mounting_holes`` config section. Raises on bad input."""
    if not isinstance(section, dict):
        raise ValueError(
            f"{source}: 'mounting_holes' must be a JSON object, got {type(section).__name__}"
        )
    allowed_keys = {"count", "screw", "hole_diameter_mm", "pad", "keepout"}
    extra = set(section.keys()) - allowed_keys
    if extra:
        raise ValueError(
            f"{source}: unknown keys in 'mounting_holes': {sorted(extra)}; "
            f"allowed: {sorted(allowed_keys)}"
        )
    if "count" in section and (
        not isinstance(section["count"], int) or section["count"] < 0
    ):
        raise ValueError(
            f"{source}: 'mounting_holes.count' must be a non-negative integer"
        )
    if "screw" in section and not isinstance(section["screw"], str):
        raise ValueError(f"{source}: 'mounting_holes.screw' must be a string")
    if "hole_diameter_mm" in section:
        d = section["hole_diameter_mm"]
        if not isinstance(d, (int, float)) or d <= 0:
            raise ValueError(
                f"{source}: 'mounting_holes.hole_diameter_mm' must be a positive number"
            )
    for sub in ("pad", "keepout"):
        if sub not in section:
            continue
        block = section[sub]
        if not isinstance(block, dict):
            raise ValueError(
                f"{source}: 'mounting_holes.{sub}' must be a JSON object"
            )
        sub_extra = set(block.keys()) - {"shape", "size_mm"}
        if sub_extra:
            raise ValueError(
                f"{source}: unknown keys in 'mounting_holes.{sub}': "
                f"{sorted(sub_extra)}; allowed: ['shape', 'size_mm']"
            )
        if "shape" in block and block["shape"] not in _MOUNTING_HOLE_SHAPES:
            raise ValueError(
                f"{source}: 'mounting_holes.{sub}.shape' must be one of "
                f"{list(_MOUNTING_HOLE_SHAPES)}, got {block['shape']!r}"
            )
        if "size_mm" in block:
            s = block["size_mm"]
            if not isinstance(s, (int, float)) or s <= 0:
                raise ValueError(
                    f"{source}: 'mounting_holes.{sub}.size_mm' must be a positive number"
                )


def load_project_config(config_path: str | None = None) -> dict[str, Any]:
    """Load a project config from a JSON file.

    If config_path is None, looks for a *_config.json in the autoplacer
    directory. Returns empty dict if no file found.

    JSON values are converted: lists of strings in "power_nets" become sets.
    """
    if config_path is None:
        # Auto-discover config file next to this module
        module_dir = Path(__file__).parent
        candidates = sorted(module_dir.glob("*_config.json"))
        if not candidates:
            return {}
        config_path = str(candidates[0])

    with open(config_path) as f:
        cfg = json.load(f)

    # Convert power_nets list to set for efficient lookup
    if "power_nets" in cfg and isinstance(cfg["power_nets"], list):
        cfg["power_nets"] = set(cfg["power_nets"])

    if "mounting_holes" in cfg:
        _validate_mounting_holes(cfg["mounting_holes"], source=str(config_path))

    return cfg



def discover_project_config(project_dir: str | Path) -> Path | None:
    """Auto-discover a project-specific config file in *project_dir*.

    Search order:
    1. ``autoplacer.json``
    2. <dir_stem>_autoplacer.json  (e.g. LLUPS_autoplacer.json)
    3. the sole ``*_autoplacer.json`` in the dir (handles a renamed/copied
       project dir whose config no longer matches the directory name)
    4. [autoplacer] section in a .kicad_pro file (not yet implemented)

    Returns the :class:`Path` to the first match, or ``None``.

    The placement config (which carries the ``arrays`` grid hints) is named
    after the SYNTHESIS stem, which is not always the directory name -- a
    project copied into a differently-named work dir, or a leaf solved from a
    relocated schematic, breaks the dir-name match. Step 3 keeps the hint alive
    in that common single-config case. When several configs collide and none
    matches the dir name we refuse to guess and warn loudly, because silently
    returning ``None`` makes the array leaf solve with ``arrays=[]`` -> force/SA
    scatters the grid -> the scattered board is best-effort promoted as
    "fab-ready" (a no-fallback violation; see kicraft-no-fallbacks-fail-loudly).
    """
    project_dir = Path(project_dir)

    # 1. Generic name
    generic = project_dir / "autoplacer.json"
    if generic.is_file():
        return generic

    # 2. <stem>_autoplacer.json
    stem_cfg = project_dir / f"{project_dir.name}_autoplacer.json"
    if stem_cfg.is_file():
        return stem_cfg

    # 3. Exactly one *_autoplacer.json (dir was renamed/copied; stem != dir).
    matches = sorted(project_dir.glob("*_autoplacer.json"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        import sys

        print(
            f"warning: {project_dir} has {len(matches)} *_autoplacer.json files, "
            f"none named '{project_dir.name}_autoplacer.json'; placement config "
            "NOT auto-discovered -- pass --config explicitly or array/placement "
            f"hints will be lost (candidates: {', '.join(p.name for p in matches)})",
            file=sys.stderr,
        )

    # 4. .kicad_pro [autoplacer] section -- not yet implemented
    return None
