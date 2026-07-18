"""PlacementSolver -- force-directed placement with edge-first constraints.

Extracted from placement.py for modularity.  Import from
``placement`` (the re-export hub) for backward compatibility,
or directly from this module in new code.
"""

from __future__ import annotations

import copy
import math
import random
import time
from contextlib import contextmanager

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from .graph import (
    AdjacencyGraph,
    build_connectivity_graph,
    count_crossings,
    find_communities,
)


@contextmanager
def _timed_phase(
    timings: dict[str, float],
    key: str,
    capture_comps=None,
):
    """Record perf_counter delta in ms into timings[key].

    Always records, including for gated phases that took ~0 ms — absent
    keys would be ambiguous between "skipped" and "didn't instrument."

    When ``capture_comps`` is supplied (a zero-arg callable returning a
    {ref: Component} dict), also records the AABB of unlocked components
    at the phase's end via _record_placed_extent. The callable is
    evaluated at exit so it sees any reassignment that happened inside
    the with body (e.g. ``comps = best_comps`` in swap_opt).
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = (time.perf_counter() - t0) * 1000.0
        if capture_comps is not None:
            prefix = key[:-3] if key.endswith("_ms") else key
            _record_placed_extent(timings, prefix, capture_comps())


def _record_placed_extent(
    timings: dict[str, float], prefix: str, comps_dict: dict
) -> None:
    """Record AABB of unlocked components under <prefix>_placed_{w,h}_mm.

    Drives the parent-compose sprawl diagnostic. Filters out locked
    components (mounting holes, edge-pinned connectors) since their
    positions are fixed by constraints rather than the solver phase
    being timed -- including them would constant-pad the AABB and hide
    the cluster's actual evolution. Phase keys parallel the existing
    solve_<phase>_ms timing keys.
    """
    bboxes = [c.physical_bbox() for c in comps_dict.values() if not c.locked]
    if bboxes:
        w = max(b[1].x for b in bboxes) - min(b[0].x for b in bboxes)
        h = max(b[1].y for b in bboxes) - min(b[0].y for b in bboxes)
    else:
        w = 0.0
        h = 0.0
    timings[f"{prefix}_placed_w_mm"] = w
    timings[f"{prefix}_placed_h_mm"] = h
from .geometry import rotate_component_in_place, rotate_vector
from .placement_scorer import PlacementScorer
from .placement_utils import (
    _back_courtyard,
    _bbox_overlap_amount,
    _bbox_overlap_xy,
    _blocker_pair_compatible,
    _effective_bbox,
    _pad_half_extents,
    _swap_pad_positions,
    _update_pad_positions,
    _world_artifact_origin,
)
from .types import (
    BoardState,
    Component,
    Layer,
    Point,
    edge_outward_angle,
)


class PlacementSolver:
    """Force-directed placement with edge-first constraints and scoring feedback.

    The solver iterates locally — all geometric computation in Python.
    Placement quality is scored each iteration; the solver converges
    when score improvement plateaus.
    """

    def __init__(self, state: BoardState, config: dict = None, seed: int = 0):
        self.state = state
        self.cfg = config or {}
        self.seed = seed
        self.rng = random.Random(seed)
        # Discrete anchor-relative grid active this solve (SA-as-assignment);
        # None = off. Set in solve() when leaf_grid_assignment is enabled.
        self._grid_assignment_active: bool = False
        self._grid = None
        self.k_attract = max(0.001, min(1.0, self.cfg.get("force_attract_k", 0.08)))
        self.k_repel = max(1.0, min(5000.0, self.cfg.get("force_repel_k", 40.0)))
        self.cooling = max(0.5, min(0.999, self.cfg.get("cooling_factor", 0.97)))
        self.edge_margin = max(0.5, min(30.0, self.cfg.get("edge_margin_mm", 2.0)))
        self.grid_snap = self.cfg.get("placement_grid_mm", 0.5)
        self.max_iterations = max(
            10, min(2000, int(self.cfg.get("max_placement_iterations", 300)))
        )
        self.convergence_threshold = self.cfg.get(
            "placement_convergence_threshold", 0.5
        )
        self.score_every_n = self.cfg.get("placement_score_every_n", 1)
        self.intra_cluster_iters = self.cfg.get("intra_cluster_iters", 80)
        # placement_clearance_mm is the min gap between component bboxes.
        # Falls back to clearance_mm for backwards compatibility, then 2.5mm.
        self.clearance = self.cfg.get(
            "placement_clearance_mm", self.cfg.get("clearance_mm", 2.5)
        )
        self._seen_force_states: set[int] = set()
        # Aligned pairs: list of (ref_a, ref_b, axis) tuples.
        # Populated by _align_large_pairs(); used by _force_step().
        self._aligned_pairs: list[tuple[str, str, str]] = []

    def solve(
        self, max_iterations: int = None, convergence_threshold: float = None
    ) -> dict[str, Component]:
        """Run full placement pipeline. Returns updated components dict."""
        # Per-phase wall-clock timings, populated below. Each key is
        # set unconditionally on every solve() call so callers can sum
        # without worrying about missing entries when a phase is gated
        # off via cfg.
        phase_t: dict[str, float] = {}
        self.last_solve_phase_timings = phase_t

        # Deep copy so we don't mutate the original
        comps = {ref: copy.deepcopy(c) for ref, c in self.state.components.items()}
        # Build a working state for scoring
        work_state = copy.copy(self.state)
        work_state.components = comps

        # Matrix/array leaves: place repeated-component grids (e.g. an LED
        # matrix) deterministically as a serpentine grid and skip the
        # force/SA solver, which does not converge at array scale. The array
        # hint rides in cfg["arrays"] (from autoplacer.json). Members are
        # locked; when only simple passives remain they're placed in a strip
        # and the whole leaf is handled here.
        from kicraft.autoplacer.brain.array_placement import place_array_leaves

        array_refs, array_fully_handled = place_array_leaves(
            comps, self.cfg.get("arrays", []) or [], self.cfg
        )
        if array_refs:
            print(
                f"  Grid-placed {len(array_refs)} array member(s)"
                + (
                    "; only passives remain -> skipping force/SA"
                    if array_fully_handled
                    else " (locked, excluded from force/SA)"
                )
            )
        if array_fully_handled:
            return comps

        # Detect alignment groups from the INITIAL component positions.
        # SA refinement happily scrambles paired components (parallel
        # batteries, header arrays, LED rows) far enough apart that
        # post-SA position-based axis inference can't tell which axis
        # they were meant to share. Detecting up-front captures the
        # user's schematic-time intent; we apply the snap at the end of
        # solve() once the SA-chosen group center is known.
        from kicraft.autoplacer.brain.placement_alignment import (
            apply_alignment_repair,
            detect_alignment_groups,
        )

        alignment_groups = detect_alignment_groups(self.cfg, comps)

        # Build connectivity graph
        conn_graph = build_connectivity_graph(self.state.nets)

        with _timed_phase(phase_t, "solve_pin_edges_ms", capture_comps=lambda: comps):
            # Step 0.5: Assign layers BEFORE edge pinning so pad positions
            # reflect the flip when computing connector placement
            self._assign_layers(comps)

            # Step 1: Pin edge components (connectors, mounting holes)
            self._pin_edge_components(comps)

        with _timed_phase(phase_t, "solve_align_pairs_ms", capture_comps=lambda: comps):
            # Step 1.3: Align large paired components side-by-side
            self._align_large_pairs(comps)

        with _timed_phase(phase_t, "solve_clusters_ms", capture_comps=lambda: comps):
            # Step 1.5: Use explicit IC groups to boost connectivity weights
            ic_groups = self.cfg.get("ic_groups", {})
            if ic_groups:
                # Add extra weight to connections within IC groups
                for ic_ref, supporting in ic_groups.items():
                    for sup_ref in supporting:
                        if sup_ref in comps and ic_ref in comps:
                            conn_graph.add_edge(sup_ref, ic_ref, 2.0)  # Strong bond
                clusters = find_communities(conn_graph, seed=self.seed)
                print(
                    f"  Found {len(clusters)} component clusters (with {len(ic_groups)} IC groups)"
                )
            else:
                # Step 2: Cluster by connectivity (seeded for reproducible variation)
                clusters = find_communities(conn_graph, seed=self.seed)
                print(f"  Found {len(clusters)} component clusters")

            # Step 1.6: Sibling grouping — components with the same kind and
            # similar dimensions should be placed adjacent to conserve space.
            # Detects siblings by kind+value or kind+similar area.
            sibling_pairs = []
            comp_list = list(comps.values())
            for i, a in enumerate(comp_list):
                for b in comp_list[i + 1 :]:
                    if a.locked or b.locked:
                        continue
                    same_kind = a.kind == b.kind and a.kind not in ("", "misc", "passive")
                    similar_size = (
                        a.area > 0
                        and b.area > 0
                        and min(a.area, b.area) / max(a.area, b.area) > 0.7
                    )
                    if same_kind and similar_size:
                        # Weight proportional to component area — larger siblings
                        # benefit more from adjacency (saves more board space)
                        weight = min(3.0, 1.0 + (a.area + b.area) / 200.0)
                        conn_graph.add_edge(a.ref, b.ref, weight)
                        sibling_pairs.append((a.ref, b.ref))
            if sibling_pairs:
                print(
                    f"  Sibling grouping: {len(sibling_pairs)} pair(s) "
                    f"({', '.join(f'{a}+{b}' for a, b in sibling_pairs)})"
                )

        with _timed_phase(phase_t, "solve_place_clusters_ms", capture_comps=lambda: comps):
            # Step 3: Initial cluster placement (with seeded jitter)
            self._place_clusters(comps, clusters, conn_graph)

        with _timed_phase(phase_t, "solve_intra_cluster_ms", capture_comps=lambda: comps):
            # Step 4: Optimize layout within each cluster before global layout
            self._optimize_intra_cluster(comps, clusters, conn_graph)

        with _timed_phase(phase_t, "solve_optimize_rotations_ms", capture_comps=lambda: comps):
            # Step 5: Try 4 rotations per IC/connector, keep best
            self._optimize_rotations(comps, work_state)

        # Step 6: Force-directed refinement with scoring feedback
        scorer = PlacementScorer(work_state, self.cfg)
        best_score = scorer.score()
        best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}

        # Legalize-during-force: periodically call legalize_components()
        # during force iterations to prevent deeply illegal states
        legalize_during = self.cfg.get("legalize_during_force", False)
        legalize_every = max(1, int(self.cfg.get("legalize_every_n", 5)))
        legalize_passes = max(1, int(self.cfg.get("legalize_during_force_passes", 2)))
        prefer_legal = self.cfg.get("prefer_legal_states", False)
        enable_swap = self.cfg.get("enable_swap_optimization", True)

        best_violations = float("inf")  # track legality for prefer_legal_states
        damping = 1.0
        stagnant = 0
        reheat_strength = self.cfg.get("reheat_strength", 0.0)
        reheat_done = False

        print(
            f"  Initial placement score: {best_score.total:.1f} "
            f"(nets={best_score.net_distance:.0f} "
            f"cross={best_score.crossover_score:.0f} "
            f"xovers={best_score.crossover_count})"
        )

        _t_force = time.perf_counter()
        for iteration in range(self.max_iterations):
            # Temperature reheat: at 50% of iterations, apply perturbation kick
            if (
                not reheat_done
                and reheat_strength > 0
                and iteration == self.max_iterations // 2
            ):
                reheat_done = True
                tl_r, br_r = self.state.board_outline
                diag = math.hypot(br_r.x - tl_r.x, br_r.y - tl_r.y)
                kick_mag = diag * reheat_strength
                unlocked_refs = [r for r in comps if not comps[r].locked]
                for ref in unlocked_refs:
                    old_pos = Point(comps[ref].pos.x, comps[ref].pos.y)
                    comps[ref].pos.x += self.rng.gauss(0, kick_mag)
                    comps[ref].pos.y += self.rng.gauss(0, kick_mag)
                    # Clamp to board (pad-aware)
                    hw, hh = _pad_half_extents(comps[ref])
                    comps[ref].pos.x = max(
                        tl_r.x + hw + 1, min(br_r.x - hw - 1, comps[ref].pos.x)
                    )
                    comps[ref].pos.y = max(
                        tl_r.y + hh + 1, min(br_r.y - hh - 1, comps[ref].pos.y)
                    )
                    _update_pad_positions(comps[ref], old_pos, comps[ref].rotation)
                damping = 0.7  # partial reheat of damping
                stagnant = 0
                self._seen_force_states.clear()

            max_disp = self._force_step(comps, conn_graph, damping)
            self._resolve_overlaps(comps)
            self._clamp_pads_to_board(comps)
            # Periodic legalization during force simulation
            if legalize_during and iteration > 0 and iteration % legalize_every == 0:
                self.legalize_components(comps, max_passes=legalize_passes)
            damping *= self.cooling

            # Score more frequently for faster convergence detection
            if iteration % self.score_every_n == 0:
                work_state.components = comps
                s = scorer.score()
                # When prefer_legal_states is on, factor legality into
                # best-state selection: fewer violations wins even if
                # placement score is slightly lower.
                if prefer_legal:
                    diag = self.legality_diagnostics(comps)
                    violations = diag["overlap_count"] + diag["pad_outside_count"]
                    # Accept if: fewer violations, OR same violations + better score
                    if violations < best_violations or (
                        violations == best_violations and s.total > best_score.total
                    ):
                        best_score = s
                        best_violations = violations
                        best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}
                        stagnant = 0
                    else:
                        stagnant += 1
                        if stagnant >= 3 and stagnant % 3 == 0:
                            comps = {r: copy.deepcopy(c) for r, c in best_comps.items()}
                elif s.total > best_score.total:
                    best_score = s
                    best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}
                    stagnant = 0
                else:
                    stagnant += 1
                    if stagnant >= 3 and stagnant % 3 == 0:
                        comps = {r: copy.deepcopy(c) for r, c in best_comps.items()}

                if stagnant >= 20:
                    print(f"  Converged at iteration {iteration + 1}")
                    break

            if max_disp < self.convergence_threshold and iteration > 30:
                print(f"  Displacement converged at iteration {iteration + 1}")
                break

            # Adaptive convergence: early exit when placement is good and
            # stable. The bbox_packing threshold is the load-bearing gate
            # for parent-side composition: PlacementScore.total saturates
            # near 100 on sprawled layouts because nets/crossings hit max
            # whenever connected components find ANY routing-friendly
            # arrangement, regardless of how spread out they are. Without
            # the bbox_packing gate, a sprawled equilibrium triggers
            # early exit at iteration 17 with bh=160-200mm, the
            # candidate-search outline cap rejects it, and the round
            # fails to route. Requiring bbox_packing > 60 keeps the loop
            # iterating until the placement is genuinely compact;
            # max_iterations is the backstop for cases where compaction
            # is unreachable under the current force balance (then the
            # outline cap correctly fails the round, no silent ship).
            if (
                iteration > 15
                and best_score.total > 85.0
                and best_score.bbox_packing > 60.0
                and max_disp < 3.0
                and stagnant >= 3
            ):
                print(
                    f"  Adaptive early exit at iteration {iteration + 1} "
                    f"(score={best_score.total:.1f}, "
                    f"bbox_packing={best_score.bbox_packing:.1f}, "
                    f"disp={max_disp:.2f})"
                )
                break
        phase_t["solve_force_loop_ms"] = (time.perf_counter() - _t_force) * 1000.0
        # Force loop's product is best_comps (the highest-scoring snapshot
        # captured during iteration), not the live comps. Capture extent
        # of best_comps so the diagnostic compares apples-to-apples with
        # the SA phase that consumes best_comps as its input.
        _record_placed_extent(phase_t, "solve_force_loop", best_comps)

        # Discrete anchor-relative grid (connectivity-first) is the leaf placement
        # path: with the anchors placed, derive pin-adjacent slots from their pad
        # geometry so SA becomes *assignment* (which passive -> which slot) rather
        # than continuous positioning -- tidy, legal, and pin-local by
        # construction. Leaf-only; the parent path never sets the flag. Falls back
        # to the classic SA when the leaf has no gridable passives.
        self._grid_assignment_active = False
        self._grid = None
        if self.cfg.get("leaf_grid_assignment", False):
            from kicraft.autoplacer.brain.leaf_compaction import (
                _resolved_keepout_rects,
            )
            from kicraft.autoplacer.brain.leaf_grid_assignment import build_anchor_grid

            keepouts = [
                (tl, br)
                for tl, br, _owner in _resolved_keepout_rects(
                    self.state.keepout_rects, best_comps
                )
            ]
            grid = build_anchor_grid(
                best_comps,
                board_outline=self.state.board_outline,
                pitch_gap_mm=float(
                    self.cfg.get("leaf_grid_pitch_gap_mm", self.clearance)
                ),
                rings=int(self.cfg.get("leaf_grid_rings", 2)),
                lateral=int(self.cfg.get("leaf_grid_lateral", 1)),
                overprovision=float(self.cfg.get("leaf_grid_overprovision", 10.0)),
                max_slots=int(self.cfg.get("leaf_grid_max_slots", 400)),
                orientation_policy=str(
                    self.cfg.get("leaf_grid_orientation_policy", "auto")
                ),
                grid_snap=self.grid_snap,
                keepout_rects=keepouts,
                pad_inset_mm=float(self.cfg.get("pad_inset_margin_mm", 0.3)),
            )
            if grid.slots:
                self._grid = grid
                self._grid_assignment_active = True

        with _timed_phase(phase_t, "solve_sa_refine_ms", capture_comps=lambda: best_comps):
            # SA refinement: escape local minima after FD convergence
            if self._grid_assignment_active:
                work_state.components = best_comps
                best_comps = self._grid_assignment_sa(
                    {r: copy.deepcopy(c) for r, c in best_comps.items()},
                    self._grid,
                    work_state,
                    scorer,
                )
            elif self.cfg.get("sa_refine_enabled", True):
                self._seen_force_states.clear()
                work_state.components = best_comps
                best_comps = self._sa_refine(
                    {r: copy.deepcopy(c) for r, c in best_comps.items()},
                    work_state,
                    scorer,
                    max_iters=int(self.cfg.get("sa_refine_iterations", 300)),
                    init_temp=float(self.cfg.get("sa_refine_initial_temp", 5.0)),
                    cooling_rate=float(self.cfg.get("sa_refine_cooling_rate", 0.995)),
                    move_radius=float(self.cfg.get("sa_refine_move_radius_mm", 2.0)),
                    swap_prob=float(self.cfg.get("sa_refine_swap_probability", 0.3)),
                    rotation_prob=float(self.cfg.get("sa_refine_rotation_probability", 0.2)),
                )

        with _timed_phase(phase_t, "solve_alignment_repair_ms", capture_comps=lambda: best_comps):
            # Alignment repair: apply the alignment_groups detected from the
            # INITIAL positions (before SA could scramble them). Runs after
            # SA so the group's parallel-axis center reflects the solver's
            # chosen position; the repair snaps perpendicular-axis to the
            # current group mean and redistributes at fixed pitch.
            if alignment_groups:
                apply_alignment_repair(best_comps, alignment_groups)

        with _timed_phase(phase_t, "solve_swap_opt_ms", capture_comps=lambda: best_comps):
            # Step 7: Swap optimization — directly minimize crossovers. Disabled
            # under group-rigid mode: it swaps individual parts, which would tear
            # a rigid group apart (the group is the atom now).
            comps = best_comps
            if enable_swap and not self._grid_assignment_active:
                self._seen_force_states.clear()
                work_state.components = comps
                best_cross = count_crossings(work_state)
                print(f"  Starting swap optimization ({best_cross} crossings)")

                # Build set of refs in aligned pairs — exclude from swaps to
                # preserve side-by-side alignment
                aligned_refs = set()
                for ref_a, ref_b, _axis in self._aligned_pairs:
                    aligned_refs.add(ref_a)
                    aligned_refs.add(ref_b)

                improved = True
                swap_round = 0
                while improved and swap_round < 5:
                    improved = False
                    swap_round += 1
                    unlocked = [
                        r for r in comps if not comps[r].locked and r not in aligned_refs
                    ]
                    for i in range(len(unlocked)):
                        for j in range(i + 1, len(unlocked)):
                            a, b = comps[unlocked[i]], comps[unlocked[j]]
                            # Only swap components of similar size
                            size_ratio = max(a.area, b.area) / max(
                                min(a.area, b.area), 0.01
                            )
                            if size_ratio > 4:
                                continue
                            # Swap positions and update pads
                            a.pos, b.pos = Point(b.pos.x, b.pos.y), Point(a.pos.x, a.pos.y)
                            _swap_pad_positions(a, b)
                            cross = count_crossings(work_state)
                            if cross < best_cross:
                                best_cross = cross
                                improved = True
                            else:
                                # Revert
                                a.pos, b.pos = (
                                    Point(b.pos.x, b.pos.y),
                                    Point(a.pos.x, a.pos.y),
                                )
                                _swap_pad_positions(a, b)
                    if improved:
                        print(f"    Swap round {swap_round}: {best_cross} crossings")

                best_comps = comps
            else:
                self._seen_force_states.clear()

            # Re-snap aligned pairs after swap optimization
            self._re_snap_aligned_pairs(best_comps)

            # Step 8: Snap to grid
            self._snap_to_grid(best_comps)

            # Re-snap aligned pairs after grid snap
            self._re_snap_aligned_pairs(best_comps)

        with _timed_phase(phase_t, "solve_orderedness_ms", capture_comps=lambda: best_comps):
            # Step 8.5: Orderedness — align passives into neat rows/columns.
            # Kept for the PARENT compose path; gridded leaves skip it (the grid
            # makes passive rows structural, so orderedness=0 for those leaves).
            orderedness = self.cfg.get("orderedness", 0.0)
            if (
                orderedness > 0.01
                and not self._grid_assignment_active
            ):
                self._apply_orderedness(best_comps, orderedness)
                # Re-snap aligned pairs after orderedness
                self._re_snap_aligned_pairs(best_comps)

        with _timed_phase(phase_t, "solve_stack_blocks_ms", capture_comps=lambda: best_comps):
            # Step 8.7: Block stacking pass -- for parent-side blocks only.
            # Force-directed + SA alone consistently fail to migrate small
            # front-only SMT blocks onto large back-only THT blocks (they
            # converge to a connectivity centroid that's never inside the
            # back-side block). Without active stacking, dual-layer parents
            # like LLUPS waste >50% of board area opposite the battery
            # footprint. This pass deterministically translates each
            # unlocked subcircuit block onto its largest blocker-compatible
            # neighbor whose bbox can accommodate it.
            if self.cfg.get("opposite_side_stacking_pass", True):
                self._stack_compatible_blocks(best_comps)

        with _timed_phase(phase_t, "solve_nest_blocks_ms", capture_comps=lambda: best_comps):
            # Step 8.8: Interior-hole nesting pass -- shaped parents only.
            # A hollow leaf (LED-ring annulus) can host a small companion
            # leaf INSIDE its enclosed empty centre; without this a ⌀60
            # ring + MCU pack side-by-side and the requested circle can
            # never fit (docs/plans/shaped-compose-leaf-nesting.md).
            self._nest_blocks_in_interior_holes(best_comps)

        with _timed_phase(phase_t, "solve_resolve_overlaps_ms", capture_comps=lambda: best_comps):
            # Step 9: Final exhaustive overlap resolution — guarantee no courtyard
            # overlaps before routing. Must run after snap since snapping can
            # re-introduce small overlaps.
            self._resolve_overlaps(best_comps)

            # Re-snap aligned pairs after overlap resolution
            self._re_snap_aligned_pairs(best_comps)

            # Step 9.1: Push any unlocked components out of parent-local
            # keep-in zones (mounting holes etc.). Runs after stacking/SA
            # so we correct any drift those passes introduced; runs before
            # legalize/clamp so the corrections survive into the final
            # output. Iterating up to 3 times handles rare cascades where
            # pushing one component creates a new overlap with another
            # keep-in.
            for _ in range(3):
                if self._resolve_keep_in_rects(best_comps) == 0:
                    break
                self._resolve_overlaps(best_comps)

            # Step 9.2: Push unlocked parts out of antenna keep-out rects (RF
            # near-field, from BoardState.keepout_rects). Same bounded
            # convergence shape as the keep-in pass; the owner footprint is
            # exempt so the ESP32 itself is never pushed off its own antenna.
            for _ in range(3):
                if self._resolve_keepout_rects(best_comps) == 0:
                    break
                self._resolve_overlaps(best_comps)

            # Step 9.3: Push unlocked non-array parts clear of the locked array
            # grid. A part wider than the grid pitch can't escape the dense grid
            # via Step 9's per-pair nudges (each lands it on the next cell), so
            # one push out of the whole grid bbox runs here, then overlaps are
            # re-resolved against the moved part.
            for _ in range(3):
                if self._resolve_array_grid(best_comps) == 0:
                    break
                self._resolve_overlaps(best_comps)

        with _timed_phase(phase_t, "solve_legalize_ms", capture_comps=lambda: best_comps):
            # Step 9.5: Comprehensive legalization repair for subcircuit mode
            if prefer_legal:
                repair_passes = int(self.cfg.get("leaf_legality_repair_passes", 12))
                self.legalize_components(best_comps, max_passes=repair_passes)

        with _timed_phase(phase_t, "solve_clamp_ms", capture_comps=lambda: best_comps):
            # Step 10: Hard clamp — nothing outside the board
            self._clamp_to_board(best_comps)

            # Step 11: Ensure all pads are inside the board boundary
            self._clamp_pads_to_board(best_comps)

            # Step 12: Validate pad containment — re-clamp if any pads still outside
            for clamp_pass in range(3):
                tl_v, br_v = self.state.board_outline
                inset_v = self.cfg.get("pad_inset_margin_mm", 0.3)
                any_outside = False
                for comp in best_comps.values():
                    for pad in comp.pads:
                        if (
                            pad.pos.x < tl_v.x + inset_v
                            or pad.pos.x > br_v.x - inset_v
                            or pad.pos.y < tl_v.y + inset_v
                            or pad.pos.y > br_v.y - inset_v
                        ):
                            any_outside = True
                            break
                    if any_outside:
                        break
                if not any_outside:
                    break
                self._clamp_to_board(best_comps)
                self._clamp_pads_to_board(best_comps)
                if clamp_pass == 2:
                    print("  WARNING: some pads still outside board after 3 clamp passes")

            # Step 13: Re-pin edge/corner components that may have drifted
            # during overlap resolution (both-locked case can push pinned parts)
            self._restore_pinned_positions(best_comps)

            # Step 13.5: Overlap resolution after restoration — restoring pinned
            # components can introduce new overlaps (e.g. a mounting hole restored
            # to its corner now overlaps a component that was pushed there during
            # the force simulation).  Re-resolve, then re-restore to ensure both
            # overlap-free placement AND correct pinned positions.
            self._resolve_overlaps(best_comps)
            self._restore_pinned_positions(best_comps)

            # Step 14: Re-validate pad containment after restoring pinned positions
            self._clamp_pads_to_board(best_comps)

            # Step 15: Slide edge-pinned connectors out of any non-owner antenna
            # keep-out, on final geometry. The push-out pass (Step 9.2) skips
            # locked parts and a connector can't leave its edge, so a USB-C
            # pinned beside an ESP32's antenna near-field would otherwise survive
            # to the stamp as a courtyard overlap + items_not_allowed DRC.
            # Updates _pinned_targets, so no later restore re-introduces it.
            if self._clear_pinned_from_keepouts(best_comps):
                self._clamp_pads_to_board(best_comps)

        with _timed_phase(phase_t, "solve_compaction_ms", capture_comps=lambda: best_comps):
            # Step 15.5: Deterministic compaction squeeze (area-compaction
            # Phase 3). Force equilibrium + SA leave multi-mm slack between
            # parts even on a right-sized canvas; this slides each unlocked
            # part toward the placed-bbox centroid as far as legality allows
            # (clearance, keep-outs, keep-ins, board bounds). Leaf-only:
            # local_solver_config enables it for content-canvas leaf solves;
            # the parent/compose path never sets the flag. Runs before Step
            # 16 so the courtyard pass still has the last word, and re-snaps
            # aligned pairs it may have skewed.
            if self.cfg.get("leaf_compaction_pass", False) and not self._grid_assignment_active:
                from kicraft.autoplacer.brain.leaf_compaction import (
                    compact_toward_centroid,
                )

                compaction = compact_toward_centroid(
                    best_comps,
                    board_outline=self.state.board_outline,
                    clearance_mm=self.clearance,
                    keepout_rects=self.state.keepout_rects,
                    keep_in_specs=self.cfg.get("parent_keep_in_rects", []),
                    pad_inset_mm=float(self.cfg.get("pad_inset_margin_mm", 0.3)),
                )
                if compaction["total_slide_mm"] > 0.0:
                    print(
                        f"  Compaction squeeze: {compaction['moved_components']} "
                        f"slide(s), {compaction['total_slide_mm']:.1f}mm total "
                        f"over {compaction['passes']} pass(es)"
                    )
                    self._re_snap_aligned_pairs(best_comps)
                    self._clamp_pads_to_board(best_comps)

        with _timed_phase(phase_t, "solve_courtyard_ms", capture_comps=lambda: best_comps):
            # Step 16: Final courtyard-separation legalization -- the GENUINE
            # last geometry step. Steps 13-15 (pinned restore, board clamp,
            # keep-out clear) all move parts AFTER the last _resolve_overlaps,
            # with nothing re-resolving courtyards -- so a same-side pair the
            # solver had separated can drift back into overlap and survive to
            # the routed board (the systematic courtyards_overlap DRC failure).
            # Running this last guarantees the placement handed to the router
            # has no same-side courtyard overlap. Only unlocked parts move, so
            # pinned connectors/holes keep the positions Steps 13-15 set.
            if self.cfg.get("resolve_courtyard_overlaps", True):
                unresolved = self._resolve_courtyard_overlaps(best_comps)
                self._clamp_pads_to_board(best_comps)
                if unresolved:
                    print(
                        f"  WARNING: {unresolved} courtyard overlap(s) between two "
                        "locked/pinned parts could not be legalized (left for the "
                        "minor-overlap gate tolerance)"
                    )

        # Grid-assignment: the legality tail moves individual parts, which can
        # nudge a gridded passive off its (legal-by-construction) slot. Re-snap
        # every occupant back to its slot as the genuine last step, so the tidy,
        # pin-local structure the assignment found is what ships.
        if self._grid_assignment_active and self._grid is not None:
            from kicraft.autoplacer.brain.leaf_grid_assignment import resnap_to_grid

            resnap_to_grid(best_comps, self._grid)
            self._clamp_pads_to_board(best_comps)

        # Final score
        work_state.components = best_comps
        final = PlacementScorer(work_state, self.cfg).score()
        print(
            f"  Final placement score: {final.total:.1f} "
            f"(nets={final.net_distance:.0f} "
            f"cross={final.crossover_score:.0f} "
            f"xovers={final.crossover_count})"
        )

        return best_comps

    def _score_rotation_for_routing(
        self, work_state: BoardState, comp: Component
    ) -> float:
        """Score component rotation for routability.

        Considers: crossovers, pad accessibility (pads not blocked by component body),
        and net distance.
        """
        cross = count_crossings(work_state)
        cross_score = 100 / (1 + cross) if cross > 0 else 100

        # Prefer rotations where pads face outward (toward board edge or open space)
        # Check if pads have clear path to edges
        tl, br = work_state.board_outline
        accessible = 0
        for pad in comp.pads:
            px, py = pad.pos.x, pad.pos.y
            # Check each quadrant for openness
            dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dx, dy in dirs:
                dist = 0
                ox, oy = px, py
                while dist < 30:
                    ox += dx * 2
                    oy += dy * 2
                    if tl.x < ox < br.x and tl.y < oy < br.y:
                        dist += 2
                    else:
                        break
                accessible += dist

        # Higher = more accessible area around pads
        access_score = min(100, accessible / 10)

        # Net distance matters for routing
        from .graph import total_ratsnest_length

        net_dist = total_ratsnest_length(work_state)
        dist_score = max(0, 100 - net_dist / 5)

        return cross_score * 0.5 + access_score * 0.3 + dist_score * 0.2

    @staticmethod
    def _connector_wants_perp_axis(comp: Component, cfg: dict | None) -> bool:
        """True when a linear single-row pin header should sit with its pin
        (long) axis PERPENDICULAR to its assigned edge, instead of the default
        long-axis-parallel-to-edge.

        A bank of short headers laid pins-parallel strings the board out along
        the edge (16x 1x3 -> ~200mm) and interleaves each header's signal/power
        pads ON the shared-net line, fragmenting the GND pour (KC-8A3US3).
        Turning each header so its pins point INTO the board packs the row by
        body-width (~3x shorter edge) and lines every same-index pad (all the
        GNDs) into one uninterrupted strip. See pcb-area-compaction-plan Phase 6.

        Scope is deliberately narrow so mouthed connectors (USB/barrel: handled
        by the opening_direction branch), 2-pin screw terminals (want the wire
        cage facing off-board), multi-row (2xN IDC) and long single headers
        (a lone 1x20 GPIO would stab deep into the board) are all left alone.
        """
        if not cfg or not cfg.get("connector_perp_orientation", True):
            return False
        if comp.kind != "connector" or comp.opening_direction is not None:
            return False
        pads = comp.pads or []
        if len(pads) < 3:
            return False  # 2-pin terminals/headers: leave along the edge
        xs = [p.pos.x for p in pads]
        ys = [p.pos.y for p in pads]
        spread_x, spread_y = max(xs) - min(xs), max(ys) - min(ys)
        major, minor = max(spread_x, spread_y), min(spread_x, spread_y)
        if major < 2.0:
            return False
        if minor > cfg.get("connector_perp_row_tol_mm", 1.2):
            return False  # multi-row (2xN) -> keep along the edge
        if major > cfg.get("connector_perp_max_len_mm", 15.0):
            return False  # long header -> would stab too deep; keep parallel
        # width_mm/height_mm are the COURTYARD bbox: a bare 2.54mm header strip
        # measures 3.63mm here (2.64mm fab body + courtyard margin), a 2P screw
        # terminal 7.89mm. The cut must sit between those measured values -- a
        # body-depth-calibrated 3.0 read every real strip as deep, silently
        # disabling this whole heuristic for its target genre (KC-YXQ4EC:
        # 16x 1x3 strung out 193mm, GND pour fragmented). Screw terminals and
        # wire-entry blocks stay excluded (KC-YJ7Q69).
        body_depth = min(comp.width_mm, comp.height_mm)
        if body_depth > cfg.get("connector_perp_max_body_depth_mm", 4.0):
            return False
        return True

    @staticmethod
    def _best_rotation_for_edge(
        comp: Component, edge: str, cfg: dict | None = None
    ) -> float:
        """Find the rotation (0/90/180/270) that orients a connector flush
        against the named edge with its opening facing outward.

        Strategy:
        1. If the component has a known opening_direction (detected from
           body-extension-beyond-pads in local coords), compute the exact
           rotation that points the opening outward from the given edge.
        2. Otherwise fall back to aspect-ratio heuristics. The long axis is
           driven PARALLEL to the edge by default, but a short single-row pin
           header (see ``_connector_wants_perp_axis``) is driven PERPENDICULAR
           so a connector bank packs tight and its shared-net pads line up.
        """
        if comp.opening_direction is not None:
            # Direct computation: we need the opening (local-frame angle)
            # to end up pointing at edge_outward_angle in board-space.
            # KiCad forward: board_angle = local_angle - rotation.
            # So: rotation = opening_direction - outward.
            rot = (comp.opening_direction - edge_outward_angle(comp.layer, edge)) % 360
            return rot

        # -- Fallback: no detectable opening direction --
        if not comp.pads:
            return comp.rotation

        # Which way should the LONG (pin) axis point? Default = parallel to the
        # edge; a bankable single-row header inverts to perpendicular. Framed as
        # an XOR so the default branch stays byte-identical to the legacy code.
        perp = PlacementSolver._connector_wants_perp_axis(comp, cfg)
        long_horizontal = (edge in ("top", "bottom")) != perp

        w, h = comp.width_mm, comp.height_mm
        if long_horizontal:
            # Want width >= height (long axis horizontal).
            if h > w * 1.1:
                return (comp.rotation + 90) % 360
            return comp.rotation
        else:
            # Want height >= width (long axis vertical).
            if w > h * 1.1:
                return (comp.rotation + 90) % 360
            return comp.rotation

    def _pin_edge_components(self, comps: dict[str, Component]):
        """Pin components based on component_zones config, with fallback heuristics.

        Supports three constraint types:
          - edge: snap to named edge (left/right/top/bottom), lock in place
          - corner: pin to named corner (top-left/top-right/bottom-left/bottom-right)
          - zone: confine to a board region (used during _place_clusters, not locked)

        Connectors on the same edge are grouped together in a row/column
        with spacing, preventing them from scattering or falling off the edge.

        Connector orientation is auto-corrected so pads face the board
        center (e.g., USB connector opening faces outward, pads inward).

        Connectors without explicit zone config fall back to nearest-edge heuristic.
        Mounting holes without config fall back to nearest-corner.

        Positions are randomized along the assigned edge/zone each round
        (controlled by self.rng and edge_jitter_mm config) so that placements
        vary across experiment rounds.

        When unlock_all_footprints is True, initial positions are still set for
        edge/corner constraints but components are NOT locked — the force
        simulation can move them, and edge_compliance scoring incentivizes
        keeping them near edges.

        Saves target positions in self._pinned_targets for later restoration
        by _restore_pinned_positions().
        """
        self._pinned_targets: dict[str, Point] = {}
        tl, br = self.state.board_outline
        margin = self.edge_margin
        zones = self.cfg.get("component_zones", {})
        unlock_all = self.cfg.get("unlock_all_footprints", False)
        jitter = self.cfg.get("edge_jitter_mm", 5.0)
        pad_inset = self.cfg.get("pad_inset_margin_mm", 0.3)
        connector_gap = self.cfg.get("connector_gap_mm", 2.0)
        connector_inset = self.cfg.get("connector_edge_inset_mm", 1.0)

        # Validate configured refs against actual components
        missing_refs = [ref for ref in zones if ref not in comps]
        if missing_refs:
            print(
                f"  WARNING: component_zones references not found on board: "
                f"{', '.join(missing_refs)}"
            )

        def _random_in_corner(corner: str, comp: Component) -> Point:
            """Return a position near the named corner with small jitter.

            For ``kind == "subcircuit"`` blocks with an ``anchor_offset_mm``
            zone entry, the returned point is shifted so the named anchor
            -- not the block body center -- lands at the corner target.
            The body bbox of a synthetic block is otherwise meaningless
            for body-flush placement.
            """
            cx = tl.x + margin if "left" in corner else br.x - margin
            cy = tl.y + margin if "top" in corner else br.y - margin
            cx += self.rng.uniform(-jitter, jitter)
            cy += self.rng.uniform(-jitter, jitter)

            if comp.kind == "subcircuit":
                anchor_off = zones.get(comp.ref, {}).get("anchor_offset_mm")
                if anchor_off is not None:
                    # KiCad-CW, matching how the block is actually stamped
                    # (parent_adapter._rotated / _world_artifact_origin).
                    off = rotate_vector(anchor_off, comp.rotation)
                    cx -= off.x
                    cy -= off.y

            # Clamp to board
            hw, hh = comp.width_mm / 2, comp.height_mm / 2
            cx = max(tl.x + hw + 1, min(br.x - hw - 1, cx))
            cy = max(tl.y + hh + 1, min(br.y - hh - 1, cy))
            return Point(cx, cy)

        def _escape_corner_from_locked(
            corner: str, comp: Component, target: Point
        ) -> Point:
            """Slide ``target`` along the corner's edges until comp's bbox no
            longer overlaps any already-locked component's bbox.

            Without this, a corner-pinned mounting hole placed at e.g.
            top-left can land directly on top of an edge-pinned connector
            that also occupies the top-left region. _resolve_overlaps later
            pushes the mount aside, but _restore_pinned_positions then snaps
            it back to the conflicting corner target -- no resolve runs
            after the final restore, so the overlap survives to the stamp.
            By baking the escape into the pinned_target, the restore is a
            no-op and the conflict cannot return.

            Slides along the corner's parallel-to-edge axes (top corners
            slide in x or y, etc.). Picks the smallest move that clears all
            conflicts within board bounds. If neither axis produces a clear
            position, returns the original target (caller falls back to
            letting _resolve_overlaps handle it as best it can).
            """
            half_gap = self.clearance / 2.0
            # Use width_mm/height_mm directly so the bbox is centered on
            # the proposed target, not on comp.pos (which still points at
            # the pre-pin position when this is called).
            hw = comp.width_mm / 2
            hh = comp.height_mm / 2
            comp_tl = Point(target.x - hw - half_gap, target.y - hh - half_gap)
            comp_br = Point(target.x + hw + half_gap, target.y + hh + half_gap)

            conflicts: list[tuple[Point, Point]] = []
            for other_ref, other in comps.items():
                if other is comp or not other.locked:
                    continue
                o_tl, o_br = _effective_bbox(other, half_gap)
                ox = min(comp_br.x, o_br.x) - max(comp_tl.x, o_tl.x)
                oy = min(comp_br.y, o_br.y) - max(comp_tl.y, o_tl.y)
                if ox > 0 and oy > 0:
                    conflicts.append((o_tl, o_br))
            if not conflicts:
                return target

            # Slide along the corner's perpendicular edge so the mount
            # stays in the corner's named region (top-left stays top, etc.)
            min_x = tl.x + hw + 1.0
            max_x = br.x - hw - 1.0
            min_y = tl.y + hh + 1.0
            max_y = br.y - hh - 1.0

            def clear_at(x: float, y: float) -> bool:
                ctl = Point(x - hw - half_gap, y - hh - half_gap)
                cbr = Point(x + hw + half_gap, y + hh + half_gap)
                for o_tl, o_br in conflicts:
                    if (
                        ctl.x < o_br.x
                        and cbr.x > o_tl.x
                        and ctl.y < o_br.y
                        and cbr.y > o_tl.y
                    ):
                        return False
                return True

            candidates: list[tuple[float, float, float]] = []
            for o_tl, o_br in conflicts:
                if "left" in corner:
                    new_x = o_br.x + hw + half_gap + 0.1
                else:
                    new_x = o_tl.x - hw - half_gap - 0.1
                new_x = max(min_x, min(max_x, new_x))
                if clear_at(new_x, target.y):
                    candidates.append((abs(new_x - target.x), new_x, target.y))

                if "top" in corner:
                    new_y = o_br.y + hh + half_gap + 0.1
                else:
                    new_y = o_tl.y - hh - half_gap - 0.1
                new_y = max(min_y, min(max_y, new_y))
                if clear_at(target.x, new_y):
                    candidates.append((abs(new_y - target.y), target.x, new_y))

            if not candidates:
                return target
            candidates.sort()
            _, nx, ny = candidates[0]
            return Point(nx, ny)

        def _shift_pads_inside(comp: Component, assigned_edge: str = None):
            """Shift component so ALL pads are inside the board boundary.

            If assigned_edge is set, skip shifting on the axis perpendicular
            to the edge — don't pull an edge-pinned connector away from its
            assigned edge.  Only enforce containment on the other 3 sides.
            """
            if not comp.pads:
                return
            pad_xs = [p.pos.x for p in comp.pads]
            pad_ys = [p.pos.y for p in comp.pads]
            shift_x = shift_y = 0.0

            # X axis shifts (skip the assigned-edge side)
            if min(pad_xs) < tl.x + pad_inset and assigned_edge != "left":
                shift_x = tl.x + pad_inset - min(pad_xs)
            elif max(pad_xs) > br.x - pad_inset and assigned_edge != "right":
                shift_x = br.x - pad_inset - max(pad_xs)

            # Y axis shifts (skip the assigned-edge side)
            if min(pad_ys) < tl.y + pad_inset and assigned_edge != "top":
                shift_y = tl.y + pad_inset - min(pad_ys)
            elif max(pad_ys) > br.y - pad_inset and assigned_edge != "bottom":
                shift_y = br.y - pad_inset - max(pad_ys)

            if abs(shift_x) > 0.01 or abs(shift_y) > 0.01:
                comp.pos.x += shift_x
                comp.pos.y += shift_y
                for pad in comp.pads:
                    pad.pos.x += shift_x
                    pad.pos.y += shift_y
                if comp.body_center is not None:
                    comp.body_center = Point(
                        comp.body_center.x + shift_x,
                        comp.body_center.y + shift_y,
                    )

        def _connector_edge_x(comp: Component, edge: str) -> float:
            """Compute X position so connector body edge is flush with the
            board edge (plus connector_inset_mm offset).

            For left edge: body left edge at tl.x + connector_inset
            For right edge: body right edge at br.x - connector_inset

            For ``kind == "subcircuit"`` blocks with ``anchor_offset_mm``
            in their zone config, the named anchor -- not the body edge
            -- is what we flush against the board edge. The body half
            offset is replaced by the inverse-rotated anchor offset.
            """
            if comp.kind == "subcircuit":
                anchor_off = zones.get(comp.ref, {}).get("anchor_offset_mm")
                if anchor_off is not None:
                    # KiCad-CW rotation, matching the stamp transform.
                    rotated_x = rotate_vector(anchor_off, comp.rotation).x
                    if edge == "left":
                        return tl.x + connector_inset - rotated_x
                    else:
                        return br.x - connector_inset - rotated_x
            hw = comp.width_mm / 2
            if edge == "left":
                return tl.x + connector_inset + hw
            else:  # right
                return br.x - connector_inset - hw

        def _connector_edge_y(comp: Component, edge: str) -> float:
            """Compute Y position so connector body edge is flush with the
            board edge (plus connector_inset_mm offset).

            For top edge: body top edge at tl.y + connector_inset
            For bottom edge: body bottom edge at br.y - connector_inset

            See ``_connector_edge_x`` for the subcircuit-block override.
            """
            if comp.kind == "subcircuit":
                anchor_off = zones.get(comp.ref, {}).get("anchor_offset_mm")
                if anchor_off is not None:
                    # KiCad-CW rotation, matching the stamp transform.
                    rotated_y = rotate_vector(anchor_off, comp.rotation).y
                    if edge == "top":
                        return tl.y + connector_inset - rotated_y
                    else:
                        return br.y - connector_inset - rotated_y
            hh = comp.height_mm / 2
            if edge == "top":
                return tl.y + connector_inset + hh
            else:  # bottom
                return br.y - connector_inset - hh

        def _orient_for_edge(comp: Component, edge: str):
            """Rotate the connector to face the edge, keeping extents honest.

            ``width_mm``/``height_mm`` describe the part at its CURRENT
            rotation, so a 90-degree turn must swap them. Orientation runs
            BEFORE the group pack measures sizes: without the swap, a
            connector rotated onto a left/right edge is spaced by its short
            side while its long side runs along the edge -- adjacent
            courtyards then overlap by design (KC-FGRSQF J3-J6: 15.35mm-tall
            terminals packed at 8.15mm + gap), and the flush inset uses the
            wrong half-extent.
            """
            zone_cfg = zones.get(comp.ref, {})
            if "rotation" in zone_cfg:
                new_rot = zone_cfg["rotation"]
            else:
                new_rot = self._best_rotation_for_edge(comp, edge, self.cfg)
            old_rot = comp.rotation
            if abs((new_rot - old_rot) % 360.0) < 0.001:
                return
            comp.rotation = new_rot
            # Rotate pads + body_center in place around the current position.
            _update_pad_positions(comp, comp.pos, old_rot)
            if round((new_rot - old_rot) / 90.0) % 2 != 0:
                comp.width_mm, comp.height_mm = comp.height_mm, comp.width_mm

        def _place_at(comp: Component, edge: str, pos: Point):
            """Translate an already-oriented connector to its packed slot."""
            old_pos = Point(comp.pos.x, comp.pos.y)
            comp.pos = pos
            _update_pad_positions(comp, old_pos, comp.rotation)
            _shift_pads_inside(comp, assigned_edge=edge)

        # --- Collect edge-pinned connectors by edge for grouped placement ---
        edge_groups: dict[str, list[str]] = {}  # edge -> [ref, ...]
        for ref, comp in comps.items():
            if comp.locked:
                # Already exact-placed on entry (the form-factor scaffold's fixed
                # connectors arrive pos+rotation frozen at their standard board
                # coordinates). Honor that verbatim -- never edge-repin or
                # re-orient it -- but record its target so the overlap/keepout
                # passes restore it if a neighbour nudges it.
                self._pinned_targets[ref] = Point(comp.pos.x, comp.pos.y)
                continue
            zone_cfg = zones.get(ref, {})
            if "edge" in zone_cfg:
                edge = zone_cfg["edge"]
                edge_groups.setdefault(edge, []).append(ref)
            elif (
                comp.kind == "connector"
                and "corner" not in zone_cfg
                and "zone" not in zone_cfg
            ):
                # Fallback: assign to nearest edge
                x, y = comp.pos.x, comp.pos.y
                distances = {
                    "left": x - tl.x,
                    "right": br.x - x,
                    "top": y - tl.y,
                    "bottom": br.y - y,
                }
                nearest = min(distances, key=distances.get)
                edge_groups.setdefault(nearest, []).append(ref)

        # Edge-pinned discrete parts per side (connectors + explicitly edge-zoned
        # switches), excluding mounting holes and subcircuit blocks. Used by
        # _clamp_companions_inboard_of_connectors to keep ordinary parts behind
        # the connector's pads on its zoned side (KC-S8PC37 R8).
        self._edge_pinned_groups: dict[str, list[str]] = {}
        for edge, refs in edge_groups.items():
            keep = [
                r
                for r in refs
                if r in comps and comps[r].kind not in ("mounting_hole", "subcircuit")
            ]
            if keep:
                self._edge_pinned_groups[edge] = keep

        # Reserve corner space along edge before placing edge groups so
        # edge connectors don't land where a corner-pinned mounting hole
        # will go. Without this, on LLUPS J1 (USB-C, edge=left) could
        # randomly land at the very top of the left edge -- exactly where
        # H4 corner=top-left needs to be -- and either:
        #   - the H4 corner-escape moved H4 down past USB to clear, then
        #     constraint_aware_outline used H4's escaped Y as the top
        #     edge, snapping H4 back to (corner.x, top + keep_in) which
        #     is right inside USB INPUT's pad area = stamp shorts; or
        #   - geometry validation flagged USB pads as outside the
        #     constraint-derived outline.
        # Reserving the corner footprint up front side-steps both.
        mh_cfg = self.cfg.get("mounting_holes") or {}
        corner_keep = float((mh_cfg.get("keepout") or {}).get("size_mm", 4.0))
        zones_cfg_all = self.cfg.get("component_zones", {})
        # Detect corner mounts that landed at this edge's two extremes
        edge_to_corners = {
            "left":   ("top-left", "bottom-left"),
            "right":  ("top-right", "bottom-right"),
            "top":    ("top-left", "top-right"),
            "bottom": ("bottom-left", "bottom-right"),
        }
        # Auto-detect mounting hole assignment for unconfigured holes
        mh_unzoned = [
            r for r, c in comps.items()
            if c.kind == "mounting_hole"
            and zones_cfg_all.get(r, {}).get("corner") is None
            and zones_cfg_all.get(r, {}).get("edge") is None
        ]
        # 2 unzoned mounting holes get diagonal corners (top-left+bottom-
        # right or top-right+bottom-left); both diagonals reserve all
        # four edge endpoints, so reserve every edge endpoint when there
        # are 2 holes regardless of which diagonal the RNG picks.
        auto_diag_all = len(mh_unzoned) == 2

        def _has_corner_mount(corner_name: str) -> bool:
            for r, zc in zones_cfg_all.items():
                if zc.get("corner") == corner_name and r in comps:
                    return True
            if auto_diag_all:
                return True
            return False

        # --- Place each edge group as a compact row/column ---
        for edge, refs in edge_groups.items():
            group_comps = [comps[r] for r in refs]
            # Sort by component area descending (largest first = anchor)
            order = sorted(
                range(len(refs)), key=lambda i: group_comps[i].area, reverse=True
            )

            corner_a, corner_b = edge_to_corners.get(edge, (None, None))
            # Reserve length for any corner mount that shares this edge's ends,
            # so the grow-to-fit step below accounts for the same keep-in the
            # usable-span clamps apply.
            _corner_reserve = 0.0
            if corner_a and _has_corner_mount(corner_a):
                _corner_reserve += corner_keep * 2
            if corner_b and _has_corner_mount(corner_b):
                _corner_reserve += corner_keep * 2

            # Orient every group member BEFORE measuring: the pack pitch,
            # grow-to-fit trigger, and flush inset must all use the extents
            # of the part as it will actually sit on the edge.
            for i in order:
                _orient_for_edge(group_comps[i], edge)

            if edge in ("left", "right"):
                # Column along Y axis — body edge flush with board edge
                sizes = [group_comps[i].height_mm for i in order]
                total_h = sum(sizes) + connector_gap * (len(sizes) - 1)
                # Grow the board height so EVERY same-edge connector fits flush
                # in ONE column. Without this, a column taller than the leaf
                # (e.g. 4 screw terminals on a short board, run_19) overflows:
                # _shift_pads_inside pulls the overrun back inside, the overlap
                # resolver then shoves it inboard into a 2nd column, and that
                # connector reads as stranded. Grow-only, and only when the
                # one-line span genuinely exceeds the board.
                needed_h = total_h + 2 * margin + _corner_reserve
                if needed_h > (br.y - tl.y):
                    br = Point(br.x, tl.y + needed_h)
                    self.state.board_outline = (tl, br)
                usable_top = tl.y + margin + sizes[0] / 2
                usable_bot = br.y - margin - sizes[-1] / 2
                # Subtract corner-mount footprint from each end so the
                # column never starts inside the corner mount's keep-in.
                if corner_a and _has_corner_mount(corner_a):
                    usable_top = max(
                        usable_top, tl.y + corner_keep * 2 + sizes[0] / 2
                    )
                if corner_b and _has_corner_mount(corner_b):
                    usable_bot = min(
                        usable_bot, br.y - corner_keep * 2 - sizes[-1] / 2
                    )
                group_span = total_h
                if group_span < (usable_bot - usable_top):
                    start_y = self.rng.uniform(
                        usable_top, usable_bot - group_span + sizes[0] / 2
                    )
                else:
                    start_y = usable_top  # not enough room, pack from top

                cursor_y = start_y
                for k, idx in enumerate(order):
                    comp = group_comps[idx]
                    fixed_x = _connector_edge_x(comp, edge)
                    pos = Point(fixed_x, cursor_y)
                    _place_at(comp, edge, pos)
                    self._pinned_targets[refs[idx]] = Point(comp.pos.x, comp.pos.y)
                    comp.locked = not unlock_all
                    # The cursor is the part CENTER, so the pitch is
                    # half-this + gap + half-next (keeps the packed span
                    # equal to total_h; full-extent pitch overlaps a
                    # taller-but-smaller-area follower).
                    if k + 1 < len(order):
                        cursor_y += (
                            sizes[k] / 2 + connector_gap + sizes[k + 1] / 2
                        )
            else:
                # Row along X axis — body edge flush with board edge
                sizes = [group_comps[i].width_mm for i in order]
                total_w = sum(sizes) + connector_gap * (len(sizes) - 1)
                # Grow the board width so the same-edge row fits in ONE line
                # (see the column branch above for why overflow strands).
                needed_w = total_w + 2 * margin + _corner_reserve
                if needed_w > (br.x - tl.x):
                    br = Point(tl.x + needed_w, br.y)
                    self.state.board_outline = (tl, br)
                usable_left = tl.x + margin + sizes[0] / 2
                usable_right = br.x - margin - sizes[-1] / 2
                if corner_a and _has_corner_mount(corner_a):
                    usable_left = max(
                        usable_left, tl.x + corner_keep * 2 + sizes[0] / 2
                    )
                if corner_b and _has_corner_mount(corner_b):
                    usable_right = min(
                        usable_right, br.x - corner_keep * 2 - sizes[-1] / 2
                    )
                group_span = total_w
                if group_span < (usable_right - usable_left):
                    start_x = self.rng.uniform(
                        usable_left, usable_right - group_span + sizes[0] / 2
                    )
                else:
                    start_x = usable_left
                cursor_x = start_x
                for k, idx in enumerate(order):
                    comp = group_comps[idx]
                    fixed_y = _connector_edge_y(comp, edge)
                    pos = Point(cursor_x, fixed_y)
                    _place_at(comp, edge, pos)
                    self._pinned_targets[refs[idx]] = Point(comp.pos.x, comp.pos.y)
                    comp.locked = not unlock_all
                    # Half-extent pitch; see the column branch above.
                    if k + 1 < len(order):
                        cursor_x += (
                            sizes[k] / 2 + connector_gap + sizes[k + 1] / 2
                        )

        # --- Non-edge constraints (corners, zones, mounting holes) ---
        for ref, comp in comps.items():
            zone_cfg = zones.get(ref, {})
            # Skip if already handled as edge group
            if ref in self._pinned_targets:
                continue

            if "corner" in zone_cfg:
                corner = zone_cfg["corner"]
                old_pos = Point(comp.pos.x, comp.pos.y)
                target = _random_in_corner(corner, comp)
                target = _escape_corner_from_locked(corner, comp, target)
                comp.pos = target
                _update_pad_positions(comp, old_pos, comp.rotation)
                self._pinned_targets[ref] = Point(comp.pos.x, comp.pos.y)
                comp.locked = not unlock_all

            elif "zone" in zone_cfg:
                zx0, zy0, zx1, zy1 = self._get_zone_bounds(zone_cfg["zone"])
                hw, hh = comp.width_mm / 2, comp.height_mm / 2
                old_pos = Point(comp.pos.x, comp.pos.y)
                # For subcircuit-kind blocks honor anchor_offset_mm so the
                # named anchor (not the block body) lands in the zone, and
                # bias toward the inward edge so larger blocks don't slip
                # outside the zone after edge-of-zone jitter.
                if comp.kind == "subcircuit":
                    anchor_off = zone_cfg.get("anchor_offset_mm")
                    off_x = 0.0
                    off_y = 0.0
                    if anchor_off is not None:
                        # KiCad-CW rotation, matching the stamp transform.
                        off = rotate_vector(anchor_off, comp.rotation)
                        off_x, off_y = off.x, off.y
                    target_x = self.rng.uniform(
                        zx0 + hw, max(zx0 + hw + 1, zx1 - hw)
                    )
                    target_y = self.rng.uniform(
                        zy0 + hh, max(zy0 + hh + 1, zy1 - hh)
                    )
                    comp.pos = Point(target_x - off_x, target_y - off_y)
                else:
                    comp.pos = Point(
                        self.rng.uniform(zx0 + hw, max(zx0 + hw + 1, zx1 - hw)),
                        self.rng.uniform(zy0 + hh, max(zy0 + hh + 1, zy1 - hh)),
                    )
                _update_pad_positions(comp, old_pos, comp.rotation)
                # For subcircuit blocks: lock the zone-placed comp so SA
                # cannot drift it out of the named zone. We do NOT add it
                # to _pinned_targets because the zone is a region, not an
                # exact coordinate -- _resolve_overlaps must be free to
                # push the zone-placed block away from edge-pinned
                # neighbors with conflicting blocker sets, and
                # _restore_pinned_positions would otherwise undo those
                # legalizing pushes on the next pass.
                if comp.kind == "subcircuit":
                    comp.locked = not unlock_all

            elif comp.kind == "mounting_hole":
                pass  # handled in batch below

        # --- Batch mounting hole placement: force diagonal for 2, corners for 4 ---
        mh_refs = [
            ref
            for ref, comp in comps.items()
            if comp.kind == "mounting_hole"
            and ref not in self._pinned_targets
            and zones.get(ref, {}).get("corner") is None
            and zones.get(ref, {}).get("edge") is None
        ]
        if len(mh_refs) == 2:
            # Force to diagonally opposite corners
            diag = self.rng.choice(
                [
                    ("top-left", "bottom-right"),
                    ("top-right", "bottom-left"),
                ]
            )
            for ref, corner in zip(mh_refs, diag):
                comp = comps[ref]
                old_pos = Point(comp.pos.x, comp.pos.y)
                target = _random_in_corner(corner, comp)
                target = _escape_corner_from_locked(corner, comp, target)
                comp.pos = target
                _update_pad_positions(comp, old_pos, comp.rotation)
                self._pinned_targets[ref] = Point(comp.pos.x, comp.pos.y)
                comp.locked = not unlock_all
        elif len(mh_refs) == 4:
            # One per corner
            corners = ["top-left", "top-right", "bottom-left", "bottom-right"]
            self.rng.shuffle(corners)
            for ref, corner in zip(mh_refs, corners):
                comp = comps[ref]
                old_pos = Point(comp.pos.x, comp.pos.y)
                target = _random_in_corner(corner, comp)
                target = _escape_corner_from_locked(corner, comp, target)
                comp.pos = target
                _update_pad_positions(comp, old_pos, comp.rotation)
                self._pinned_targets[ref] = Point(comp.pos.x, comp.pos.y)
                comp.locked = not unlock_all
        else:
            # 1 or 3+ mounting holes: nearest-corner heuristic
            for ref in mh_refs:
                comp = comps[ref]
                corner = ""
                corner += "top" if comp.pos.y < (tl.y + br.y) / 2 else "bottom"
                corner += "-"
                corner += "left" if comp.pos.x < (tl.x + br.x) / 2 else "right"
                old_pos = Point(comp.pos.x, comp.pos.y)
                target = _random_in_corner(corner, comp)
                target = _escape_corner_from_locked(corner, comp, target)
                comp.pos = target
                _update_pad_positions(comp, old_pos, comp.rotation)
                self._pinned_targets[ref] = Point(comp.pos.x, comp.pos.y)
                comp.locked = not unlock_all

    def _restore_pinned_positions(self, comps: dict[str, Component]):
        """Restore edge/corner-pinned components to their target positions.

        Called after overlap resolution as a safety net: the both-locked
        branch can still push pinned components if both are edge/corner
        pinned.  This snaps them back to the positions recorded during
        _pin_edge_components.
        """
        for ref, target in self._pinned_targets.items():
            comp = comps.get(ref)
            if comp is None:
                continue
            dx = target.x - comp.pos.x
            dy = target.y - comp.pos.y
            if abs(dx) < 0.01 and abs(dy) < 0.01:
                continue
            old_pos = Point(comp.pos.x, comp.pos.y)
            comp.pos.x = target.x
            comp.pos.y = target.y
            _update_pad_positions(comp, old_pos, comp.rotation)

    def _get_zone_bounds(self, zone_name: str) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) for a named board zone."""
        tl, br = self.state.board_outline
        margin = self.edge_margin
        mid_x = (tl.x + br.x) / 2
        mid_y = (tl.y + br.y) / 2

        zone_map = {
            "center": (tl.x + margin, tl.y + margin, br.x - margin, br.y - margin),
            "top": (tl.x + margin, tl.y + margin, br.x - margin, mid_y),
            "bottom": (tl.x + margin, mid_y, br.x - margin, br.y - margin),
            "left": (tl.x + margin, tl.y + margin, mid_x, br.y - margin),
            "right": (mid_x, tl.y + margin, br.x - margin, br.y - margin),
            "center-top": (tl.x + margin, tl.y + margin, br.x - margin, mid_y),
            "center-bottom": (tl.x + margin, mid_y, br.x - margin, br.y - margin),
            "center-left": (tl.x + margin, tl.y + margin, mid_x, br.y - margin),
            "center-right": (mid_x, tl.y + margin, br.x - margin, br.y - margin),
            "top-left": (tl.x + margin, tl.y + margin, mid_x, mid_y),
            "top-right": (mid_x, tl.y + margin, br.x - margin, mid_y),
            "bottom-left": (tl.x + margin, mid_y, mid_x, br.y - margin),
            "bottom-right": (mid_x, mid_y, br.x - margin, br.y - margin),
        }
        return zone_map.get(zone_name, zone_map["center"])

    def _place_clusters(
        self,
        comps: dict[str, Component],
        clusters: list[set[str]],
        conn_graph: AdjacencyGraph,
    ):
        """Place each cluster's components near their connectivity centroid.

        Supports three placement strategies controlled by config:
          - scatter_mode="cluster": centroid-based with jitter (default, exploit)
          - scatter_mode="random": uniform random within board bounds (explore)
          - signal_flow_order: biases cluster centroids left-to-right
          - component_zones with "zone": confines components to named regions
          - Decoupling caps (C* in ic_groups) placed at tighter radius to IC leader
        """
        tl, br = self.state.board_outline
        margin = self.edge_margin + 5.0  # keep away from edges
        scatter_mode = self.cfg.get("scatter_mode", "cluster")
        signal_flow = self.cfg.get("signal_flow_order", [])
        ic_groups = self.cfg.get("ic_groups", {})
        zones_cfg = self.cfg.get("component_zones", {})
        randomize_group = self.cfg.get("randomize_group_layout", False)

        # Build reverse map: component ref -> group leader
        ref_to_leader = {}
        for leader, members in ic_groups.items():
            ref_to_leader[leader] = leader
            for m in members:
                ref_to_leader[m] = leader

        # Build signal-flow X targets (evenly spaced across board width)
        flow_x_targets = {}
        if signal_flow:
            usable_left = tl.x + margin
            usable_right = br.x - margin
            for i, leader in enumerate(signal_flow):
                frac = (i + 0.5) / len(signal_flow)
                flow_x_targets[leader] = usable_left + frac * (
                    usable_right - usable_left
                )

        # Find locked component positions for attraction
        locked_positions = {ref: comp.pos for ref, comp in comps.items() if comp.locked}

        # Sort clusters by total connectivity (highest first) so the most
        # connected cluster gets placed first, improving net-topology bias.
        clusters = sorted(
            clusters,
            key=lambda c: sum(conn_graph.degree(r) for r in c),
            reverse=True,
        )

        for cluster in clusters:
            unlocked = [r for r in cluster if not comps[r].locked]
            if not unlocked:
                continue

            if scatter_mode == "random":
                # --- Random scatter: uniform random positions within bounds ---
                # Sort by area descending: large components placed first
                unlocked.sort(key=lambda r: comps[r].area, reverse=True)
                for ref in unlocked:
                    zone_cfg = zones_cfg.get(ref, {})
                    if "zone" in zone_cfg:
                        zx0, zy0, zx1, zy1 = self._get_zone_bounds(zone_cfg["zone"])
                    else:
                        zx0, zy0 = tl.x + margin, tl.y + margin
                        zx1, zy1 = br.x - margin, br.y - margin

                    # Random allowed rotation -- applied FIRST so the position
                    # bounds below use the post-rotation extents and the AABB
                    # (width/height) tracks the new orientation.
                    new_rot = comps[ref].rotation
                    if comps[ref].kind == "ic":
                        new_rot = self.rng.choice([0, 90, 180, 270])
                    elif comps[ref].kind == "passive":
                        new_rot = self.rng.choice([0, 90])
                    old_pos = Point(comps[ref].pos.x, comps[ref].pos.y)
                    rotate_component_in_place(
                        comps[ref], new_rot - comps[ref].rotation
                    )
                    hw, hh = comps[ref].width_mm / 2, comps[ref].height_mm / 2
                    comps[ref].pos = Point(
                        self.rng.uniform(zx0 + hw, max(zx0 + hw + 1, zx1 - hw)),
                        self.rng.uniform(zy0 + hh, max(zy0 + hh + 1, zy1 - hh)),
                    )
                    # Rotation already applied in place; this is now a pure
                    # translation to the drawn position.
                    _update_pad_positions(comps[ref], old_pos, comps[ref].rotation)
                continue

            # --- Cluster mode: centroid-based with signal-flow bias ---
            # Compute centroid from locked neighbors' positions
            cx, cy, weight_sum = 0.0, 0.0, 0.0
            for ref in unlocked:
                for locked_ref, lpos in locked_positions.items():
                    w = conn_graph.weight(ref, locked_ref)
                    if w > 0:
                        cx += lpos.x * w
                        cy += lpos.y * w
                        weight_sum += w

            if weight_sum > 0:
                cx /= weight_sum
                cy /= weight_sum
            else:
                # Default to board center
                cx = (tl.x + br.x) / 2
                cy = (tl.y + br.y) / 2

            # Apply signal-flow X bias: blend centroid toward target X
            # Find the cluster's group leader (if any)
            cluster_leader = None
            for ref in cluster:
                leader = ref_to_leader.get(ref)
                if leader and leader in flow_x_targets:
                    cluster_leader = leader
                    break
            if cluster_leader and cluster_leader in flow_x_targets:
                target_x = flow_x_targets[cluster_leader]
                # 60% bias toward signal-flow target, 40% toward connectivity
                cx = 0.4 * cx + 0.6 * target_x

            # Clamp to board interior
            cx = max(tl.x + margin, min(br.x - margin, cx))
            cy = max(tl.y + margin, min(br.y - margin, cy))

            # Apply zone constraints: override centroid if component has a zone
            # (uses first zone-constrained component in cluster to bias centroid)
            for ref in unlocked:
                zone_cfg = zones_cfg.get(ref, {})
                if "zone" in zone_cfg:
                    zx0, zy0, zx1, zy1 = self._get_zone_bounds(zone_cfg["zone"])
                    cx = max(zx0, min(zx1, cx))
                    cy = max(zy0, min(zy1, cy))
                    break

            # Spread components around centroid (with seeded jitter)
            n = len(unlocked)
            # Sort by area descending: ICs and large components placed first,
            # then passives fill in around them.
            unlocked.sort(key=lambda r: comps[r].area, reverse=True)
            radius = math.sqrt(n) * 3.0  # spread based on count

            # Radius variation: wider for randomize_group_layout mode
            r_lo, r_hi = (0.3, 1.8) if randomize_group else (0.8, 1.2)

            # Track placed components for net-topology bias
            placed_this_cluster: set[str] = set()

            for i, ref in enumerate(unlocked):
                # Net-topology bias: if this component has already-placed
                # connected neighbors, bias position toward their centroid.
                nbr_cx, nbr_cy, nbr_w = 0.0, 0.0, 0.0
                for nbr, w in conn_graph.neighbors(ref).items():
                    if nbr in comps and (
                        comps[nbr].locked or nbr in placed_this_cluster
                    ):
                        nbr_cx += comps[nbr].pos.x * w
                        nbr_cy += comps[nbr].pos.y * w
                        nbr_w += w
                if nbr_w > 0:
                    # Blend 50% toward connected neighbors, 50% toward cluster centroid
                    local_cx = 0.5 * cx + 0.5 * (nbr_cx / nbr_w)
                    local_cy = 0.5 * cy + 0.5 * (nbr_cy / nbr_w)
                else:
                    local_cx, local_cy = cx, cy

                # Decoupling cap proximity: caps in IC groups get tighter radius
                is_decoupling_cap = (
                    ref.startswith("C")
                    and ref in ref_to_leader
                    and ref_to_leader[ref] != ref  # not the leader itself
                )
                if is_decoupling_cap:
                    # Place within 1.5× clearance of centroid (very tight)
                    r = self.clearance * 1.5 * self.rng.uniform(0.6, 1.0)
                else:
                    r = radius * (0.5 + 0.5 * (i % 2)) * self.rng.uniform(r_lo, r_hi)

                angle = 2 * math.pi * i / max(n, 1) + self.rng.gauss(0, 0.3)

                old_pos = Point(comps[ref].pos.x, comps[ref].pos.y)
                old_rot = comps[ref].rotation
                new_x = local_cx + r * math.cos(angle)
                new_y = local_cy + r * math.sin(angle)

                # Enforce zone bounds if component has a zone constraint
                zone_cfg = zones_cfg.get(ref, {})
                if "zone" in zone_cfg:
                    zx0, zy0, zx1, zy1 = self._get_zone_bounds(zone_cfg["zone"])
                    hw, hh = comps[ref].width_mm / 2, comps[ref].height_mm / 2
                    new_x = max(zx0 + hw, min(zx1 - hw, new_x))
                    new_y = max(zy0 + hh, min(zy1 - hh, new_y))

                comps[ref].pos = Point(new_x, new_y)
                _update_pad_positions(comps[ref], old_pos, old_rot)

                # Early rotation: try all 4 orientations for ICs at placement
                # time — prevents suboptimal rotations from locking in.
                if comps[ref].kind == "ic" and len(comps[ref].pads) >= 2:
                    orig_rot = comps[ref].rotation
                    best_rot = orig_rot
                    best_rscore = -1.0
                    temp_state = copy.copy(self.state)
                    temp_state.components = comps
                    rotations = (
                        comps[ref].allowed_rotations
                        if comps[ref].allowed_rotations
                        else [0, 90, 180, 270]
                    )
                    for rot in rotations:
                        # rotate_component_in_place keeps pads, body_center
                        # and the width/height AABB in sync with rotation.
                        rotate_component_in_place(
                            comps[ref], rot - comps[ref].rotation
                        )
                        rscore = self._score_rotation_for_routing(
                            temp_state, comps[ref]
                        )
                        if rscore > best_rscore:
                            best_rscore = rscore
                            best_rot = rot
                    # Apply best rotation (revert from last-tried candidate)
                    rotate_component_in_place(
                        comps[ref], best_rot - comps[ref].rotation
                    )

                placed_this_cluster.add(ref)

    def _optimize_intra_cluster(
        self,
        comps: dict[str, Component],
        clusters: list[set[str]],
        conn_graph: AdjacencyGraph,
    ):
        """Run a short force-directed pass within each cluster independently.

        This arranges components within functional groups (e.g. charger IC
        with its caps and resistors) before the global layout decides
        where groups go relative to each other.
        """
        tl, br = self.state.board_outline
        for cluster in clusters:
            unlocked = [r for r in cluster if not comps[r].locked]
            if len(unlocked) < 2:
                continue

            # Compute cluster centroid
            sum(comps[r].pos.x for r in unlocked) / len(unlocked)
            sum(comps[r].pos.y for r in unlocked) / len(unlocked)

            # Mini force-directed loop: attract connected, repel overlapping
            damping = 1.0
            for _ in range(self.intra_cluster_iters):
                forces = {r: Point(0, 0) for r in unlocked}

                # Attract connected pairs within cluster
                for i, ra in enumerate(unlocked):
                    for rb in unlocked[i + 1 :]:
                        w = conn_graph.weight(ra, rb)
                        if w <= 0:
                            continue
                        a, b = comps[ra], comps[rb]
                        d = max(a.pos.dist(b.pos), 0.1)
                        # Pull together proportional to distance and weight
                        f = self.k_attract * w * d
                        dx = (b.pos.x - a.pos.x) / d * f
                        dy = (b.pos.y - a.pos.y) / d * f
                        forces[ra].x += dx
                        forces[ra].y += dy
                        forces[rb].x -= dx
                        forces[rb].y -= dy

                # Repel overlapping bboxes
                for i, ra in enumerate(unlocked):
                    for rb in unlocked[i + 1 :]:
                        a, b = comps[ra], comps[rb]
                        overlap = _bbox_overlap_amount(a, b)
                        if overlap <= 0:
                            continue
                        d = max(a.pos.dist(b.pos), 0.1)
                        f = 3.0 * math.sqrt(overlap)
                        dx = (a.pos.x - b.pos.x) / d * f
                        dy = (a.pos.y - b.pos.y) / d * f
                        forces[ra].x += dx
                        forces[ra].y += dy
                        forces[rb].x -= dx
                        forces[rb].y -= dy

                # Apply forces
                for r in unlocked:
                    dx = forces[r].x * damping
                    dy = forces[r].y * damping
                    mag = math.hypot(dx, dy)
                    max_step = 1.5 * damping
                    if mag > max_step:
                        dx *= max_step / mag
                        dy *= max_step / mag

                    old_pos = Point(comps[r].pos.x, comps[r].pos.y)
                    comps[r].pos.x += dx
                    comps[r].pos.y += dy
                    # Clamp to board
                    hw, hh = comps[r].width_mm / 2, comps[r].height_mm / 2
                    comps[r].pos.x = max(
                        tl.x + hw + 1.0, min(br.x - hw - 1.0, comps[r].pos.x)
                    )
                    comps[r].pos.y = max(
                        tl.y + hh + 1.0, min(br.y - hh - 1.0, comps[r].pos.y)
                    )
                    _update_pad_positions(comps[r], old_pos, comps[r].rotation)

                damping *= 0.95

        print(f"  Intra-cluster optimization done ({len(clusters)} clusters)")

    def _optimize_rotations(self, comps: dict[str, Component], work_state: BoardState):
        """Try 0/90/180/270 rotations -- two scoring modes:

        - IC/connector with pads: rotate pads via KiCad math, score with
          routing signal (crossings, pad accessibility, ratsnest length).
        - Synthetic leaf block (kind="subcircuit", no pads, populated
          ``block_rotation_geometry``): swap bbox width/height per
          rotation, score with placement signal (opposite-side blocker
          overlap + inverse courtyard overlap). Leaf blocks have no pads
          to rotate, so the IC/connector path doesn't apply -- but they
          still benefit from rotation because their footprint AABB
          changes shape under 90°/270°, opening packing opportunities
          with neighbours that single-rotation placement misses.
        """
        work_state.components = comps

        for ref, comp in comps.items():
            if comp.kind == "mounting_hole":
                continue
            # Skip edge-pinned connectors — rotation set by _best_rotation_for_edge
            # (this orients pads outward toward the board edge, so changing
            # rotation would break the side-orientation contract).
            if ref in self._pinned_targets:
                continue
            # Synthetic leaf blocks: locked means the *position* is committed
            # to a zone, but rotation is still freely searchable. For non-block
            # components, locked still means hands-off.
            is_subcircuit_block = (
                comp.kind == "subcircuit"
                and comp.block_rotation_geometry
            )
            if comp.locked and not is_subcircuit_block:
                continue

            rotations = (
                comp.allowed_rotations
                if comp.allowed_rotations
                else [0, 90, 180, 270]
            )

            # Branch: synthetic leaf block (no pads, but per-rotation geometry)
            if is_subcircuit_block:
                self._optimize_block_rotation(comp, rotations, work_state)
                continue

            if len(comp.pads) < 2:
                continue

            orig_rot = comp.rotation
            best_rot = orig_rot
            best_score = self._score_rotation_for_routing(work_state, comp)

            for rot in rotations:
                if rot == orig_rot:
                    continue
                # rotate_component_in_place keeps pads, body_center and the
                # width/height AABB in sync with the rotation (KiCad CW).
                rotate_component_in_place(comp, rot - comp.rotation)

                rot_score = self._score_rotation_for_routing(work_state, comp)
                if rot_score > best_score:
                    best_score = rot_score
                    best_rot = rot

            # Apply best rotation (revert from the last-tried candidate)
            rotate_component_in_place(comp, best_rot - comp.rotation)

    def _optimize_block_rotation(
        self,
        comp: Component,
        rotations: list[float],
        work_state: BoardState,
    ) -> None:
        """Choose the best rotation for a synthetic leaf block.

        Leaf blocks have no pads; their geometry under rotation is captured
        in ``block_rotation_geometry`` (width/height per rotation; the body
        center is the rotation pivot, so it's invariant). Scoring uses a
        placement signal (opposite-side blocker overlap + inverse courtyard
        overlap) because pad-facing routing scores don't apply to blocks.
        """
        geo_by_rot = comp.block_rotation_geometry or {}
        if not geo_by_rot:
            return

        orig_rot = float(comp.rotation)
        orig_w = comp.width_mm
        orig_h = comp.height_mm

        best_rot = orig_rot
        best_score = self._score_rotation_for_block(work_state)

        for rot in rotations:
            rot = float(rot)
            if rot == orig_rot:
                continue
            geom = geo_by_rot.get(rot)
            if geom is None:
                continue
            comp.rotation = rot
            comp.width_mm = geom.width_mm
            comp.height_mm = geom.height_mm

            score = self._score_rotation_for_block(work_state)
            if score > best_score:
                best_score = score
                best_rot = rot

        # Apply best (revert to original geometry first if no improvement)
        if best_rot == orig_rot:
            comp.rotation = orig_rot
            comp.width_mm = orig_w
            comp.height_mm = orig_h
        else:
            geom = geo_by_rot[best_rot]
            comp.rotation = best_rot
            comp.width_mm = geom.width_mm
            comp.height_mm = geom.height_mm

    def _score_rotation_for_block(self, work_state: BoardState) -> float:
        """Placement-signal score for synthetic leaf block rotation choice.

        Combines opposite-side blocker overlap (rewards F.Cu/B.Cu compatible
        leaves stacking, e.g. front-side regulators on top of a back-side
        battery footprint) with inverse courtyard overlap (penalises
        incompatible bbox overlap). Both come from ``PlacementScorer`` so
        the rotation search uses the same signals as the global scorer.
        """
        scorer = PlacementScorer(work_state, self.cfg)
        opposite = scorer._score_block_opposite_side()
        inv_overlap = scorer._score_courtyard_overlap()
        return 0.6 * opposite + 0.4 * inv_overlap

    def _force_step(
        self, comps: dict[str, Component], conn_graph: AdjacencyGraph, damping: float
    ) -> float:
        """One iteration of force-directed simulation. Returns max displacement.

        Uses numpy-accelerated repulsion when available, otherwise falls back
        to pure Python pairwise computation.
        """
        # State dedup: skip if we've seen this exact layout before
        state_h = hash(
            tuple(
                (r, round(comps[r].pos.x, 2), round(comps[r].pos.y, 2))
                for r in sorted(comps.keys())
            )
        )
        if state_h in self._seen_force_states:
            return 0.01  # signal convergence
        self._seen_force_states.add(state_h)

        tl, br = self.state.board_outline
        forces: dict[str, Point] = {ref: Point(0, 0) for ref in comps}
        refs = [r for r in comps if not comps[r].locked]

        # Accumulate all force contributions
        self._accumulate_attraction(comps, refs, forces, conn_graph)
        if _HAS_NUMPY:
            self._accumulate_repulsion_numpy(comps, forces)
        else:
            self._accumulate_repulsion_python(comps, forces)
        self._accumulate_opposite_side_attraction(comps, forces)
        self._accumulate_smt_opposite_tht_force(comps, refs, forces)
        self._accumulate_boundary_force(comps, refs, forces, tl, br)
        self._accumulate_center_attraction(comps, refs, forces, tl, br)
        self._accumulate_alignment_force(comps, forces)

        # Integrate and clamp
        max_disp = self._apply_forces(comps, refs, forces, damping, tl, br)
        self._post_step_clamp(comps, refs)

        return max_disp

    def _push_out_of_rect(
        self,
        comps: dict[str, Component],
        r_tl: Point,
        r_br: Point,
        owner_ref: str | None,
        min_extent_mm: float = 0.0,
    ) -> int:
        """Push every unlocked, non-owner component out of rect [r_tl, r_br].

        ``min_extent_mm`` (default 0) skips parts whose smaller bbox dimension is
        ``<=`` it — used by the array-grid pass to evict only bulky strays (a
        series resistor) and leave small companions (per-LED decaps) in place, so
        the push never scatters a whole companion set into a sprawl.

        Shared by the keep-in (mounting-hole) and keep-out (antenna near-field)
        passes. Each overlapping component is moved along whichever of the four
        cardinal exits is smallest *while keeping the part inside the board
        outline*. Unlike a radial push-from-center, this behaves correctly for
        a keep-out that straddles the board edge -- e.g. an ESP32 whose antenna
        faces off-board, where a radial push would shove a neighbour toward the
        edge and the board clamp would pin it back inside the rect. If no exit
        keeps the part on-board (the rect spans the board in both axes) the
        smallest exit is taken and the residue is left for legality_diagnostics
        to flag. ``owner_ref`` is exempt. Returns the number of components moved.
        """
        tl, br = self.state.board_outline
        slack = 0.5  # extra so the DRC / clearance margin holds after the push
        corrections = 0
        for ref, comp in comps.items():
            if comp.locked or ref == owner_ref:
                continue
            if min_extent_mm > 0.0 and min(comp.width_mm, comp.height_mm) <= min_extent_mm:
                continue  # small companion (decap) — don't evict it into a sprawl
            c_tl, c_br = comp.bbox(0.0)
            ox = min(c_br.x, r_br.x) - max(c_tl.x, r_tl.x)
            oy = min(c_br.y, r_br.y) - max(c_tl.y, r_tl.y)
            if ox <= 0.0 or oy <= 0.0:
                continue  # no overlap
            # cardinal exits: delta that moves comp fully clear of the rect
            candidates = [
                (r_tl.x - c_br.x - slack, 0.0),  # exit left
                (r_br.x - c_tl.x + slack, 0.0),  # exit right
                (0.0, r_tl.y - c_br.y - slack),  # exit up
                (0.0, r_br.y - c_tl.y + slack),  # exit down
            ]
            on_board = [
                (dx, dy)
                for (dx, dy) in candidates
                if c_tl.x + dx >= tl.x
                and c_br.x + dx <= br.x
                and c_tl.y + dy >= tl.y
                and c_br.y + dy <= br.y
            ]
            dx, dy = min(on_board or candidates, key=lambda d: math.hypot(d[0], d[1]))
            # Translate pos + body_center + pads together so bbox() (centered on
            # body_center) reflects the move and repeated passes converge.
            # (_move_component leaves body_center stale, which would defeat the
            # overlap test on the next iteration.)
            comp.pos = Point(comp.pos.x + dx, comp.pos.y + dy)
            if comp.body_center is not None:
                comp.body_center = Point(comp.body_center.x + dx, comp.body_center.y + dy)
            for pad in comp.pads:
                pad.pos = Point(pad.pos.x + dx, pad.pos.y + dy)
            corrections += 1
        return corrections

    def _resolve_keep_in_rects(self, comps: dict[str, Component]) -> int:
        """Push unlocked components out of parent-local keep-in zones.

        Each entry in cfg["parent_keep_in_rects"] is {ref, margin_mm}: the
        protected component's bbox grown by margin_mm is a rect other unlocked
        components must not overlap (rendered as a KiCad keepout zone). Without
        this pass, SA refine + post-stack reorderings drift unlocked leaves into
        mounting-hole keep-ins, producing stamped-DRC items_not_allowed
        violations. Returns the count of corrections applied.
        """
        specs = self.cfg.get("parent_keep_in_rects", [])
        if not specs:
            return 0
        corrections = 0
        for entry in specs:
            protected_ref = entry.get("ref")
            margin = float(entry.get("margin_mm", 0.0))
            protected = comps.get(protected_ref)
            if protected is None:
                continue
            p_tl, p_br = protected.bbox(margin)
            corrections += self._push_out_of_rect(comps, p_tl, p_br, protected_ref)
        return corrections

    def _keepout_rect_now(
        self, kr, comps: dict[str, Component]
    ) -> tuple[Point, Point]:
        """Owner-tracked board-coord rect for ``kr``.

        The keep-out is rigidly attached to its owner footprint, but the rect
        was sampled once at extraction (adapter.load). Translate it by the
        owner's displacement since then so it follows the owner as the solve
        nudges it -- otherwise parts are pushed out of where the antenna *was*,
        not where it *is*, and the stamped board carries the overlap the solver
        thought it had resolved.
        """
        origin = getattr(kr, "owner_origin", None)
        if origin is not None:
            owner = comps.get(kr.owner_ref)
            if owner is not None:
                dx = owner.pos.x - origin.x
                dy = owner.pos.y - origin.y
                if dx or dy:
                    return (
                        Point(kr.tl.x + dx, kr.tl.y + dy),
                        Point(kr.br.x + dx, kr.br.y + dy),
                    )
        return (kr.tl, kr.br)

    def _resolve_keepout_rects(self, comps: dict[str, Component]) -> int:
        """Push unlocked, non-owner components out of antenna keep-out rects.

        Reads BoardState.keepout_rects (populated by adapter.load via
        hardware.keepout_extract). The owner footprint -- the part whose antenna
        the rect protects -- is exempt. Returns the count of corrections applied.
        """
        rects = getattr(self.state, "keepout_rects", None) or []
        corrections = 0
        for kr in rects:
            r_tl, r_br = self._keepout_rect_now(kr, comps)
            corrections += self._push_out_of_rect(comps, r_tl, r_br, kr.owner_ref)
        return corrections

    def _slide_pinned_clear(
        self,
        comp: Component,
        edge: str,
        boxes: list[tuple[Point, Point]],
        half_gap: float,
    ) -> Point | None:
        """Slide ``comp`` along its pinned edge until its bbox clears every box.

        ``edge`` (left/right slide on Y, top/bottom slide on X) keeps the
        connector flush with the board edge while moving it out of the keep-out.
        Returns the smallest in-bounds move that clears all boxes, or None if
        already clear / no clear position exists (caller leaves it for the
        diagnostic to flag).
        """
        tl, br = self.state.board_outline
        hw = comp.width_mm / 2
        hh = comp.height_mm / 2
        x0, y0 = comp.pos.x, comp.pos.y

        def clear_at(x: float, y: float) -> bool:
            c_tl = Point(x - hw - half_gap, y - hh - half_gap)
            c_br = Point(x + hw + half_gap, y + hh + half_gap)
            for o_tl, o_br in boxes:
                if (
                    c_tl.x < o_br.x
                    and c_br.x > o_tl.x
                    and c_tl.y < o_br.y
                    and c_br.y > o_tl.y
                ):
                    return False
            return True

        if clear_at(x0, y0):
            return None

        candidates: list[tuple[float, float, float]] = []
        if edge in ("left", "right"):
            lo, hi = tl.y + hh + 1.0, br.y - hh - 1.0
            for o_tl, o_br in boxes:
                for ny in (
                    o_br.y + hh + half_gap + 0.1,
                    o_tl.y - hh - half_gap - 0.1,
                ):
                    ny = max(lo, min(hi, ny))
                    if clear_at(x0, ny):
                        candidates.append((abs(ny - y0), x0, ny))
        else:
            lo, hi = tl.x + hw + 1.0, br.x - hw - 1.0
            for o_tl, o_br in boxes:
                for nx in (
                    o_br.x + hw + half_gap + 0.1,
                    o_tl.x - hw - half_gap - 0.1,
                ):
                    nx = max(lo, min(hi, nx))
                    if clear_at(nx, y0):
                        candidates.append((abs(nx - x0), nx, y0))
        if not candidates:
            return None
        candidates.sort()
        _, nx, ny = candidates[0]
        return Point(nx, ny)

    def _clear_pinned_from_keepouts(self, comps: dict[str, Component]) -> int:
        """Slide edge-pinned connectors clear of any non-owner antenna keep-out.

        _push_out_of_rect skips locked parts, and a connector cannot leave its
        edge anyway, so an edge connector that lands in a neighbour's antenna
        near-field (e.g. a USB-C beside an ESP32 module) is never moved -- it
        survives to the stamp as a courtyard overlap + an items_not_allowed DRC
        in the antenna keep-out. Sliding along the pinned edge keeps the
        connector flush while clearing the keep-out. Runs late (final geometry,
        after the owner's last move) and updates _pinned_targets so the closing
        restore keeps the cleared position. Returns the number moved.
        """
        rects = getattr(self.state, "keepout_rects", None) or []
        groups = getattr(self, "_edge_pinned_groups", None) or {}
        if not rects or not groups:
            return 0
        if not hasattr(self, "_pinned_targets"):
            self._pinned_targets = {}
        half_gap = self.clearance / 2.0
        moved = 0
        for edge, refs in groups.items():
            for ref in refs:
                comp = comps.get(ref)
                if comp is None:
                    continue
                boxes = [
                    self._keepout_rect_now(kr, comps)
                    for kr in rects
                    if kr.owner_ref != ref
                ]
                if not boxes:
                    continue
                new_pos = self._slide_pinned_clear(comp, edge, boxes, half_gap)
                if new_pos is None:
                    continue
                dx, dy = new_pos.x - comp.pos.x, new_pos.y - comp.pos.y
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    continue
                comp.pos = Point(comp.pos.x + dx, comp.pos.y + dy)
                if comp.body_center is not None:
                    comp.body_center = Point(
                        comp.body_center.x + dx, comp.body_center.y + dy
                    )
                for pad in comp.pads:
                    pad.pos = Point(pad.pos.x + dx, pad.pos.y + dy)
                self._pinned_targets[ref] = Point(comp.pos.x, comp.pos.y)
                moved += 1
        return moved

    def _resolve_array_grid(self, comps: dict[str, Component]) -> int:
        """Push unlocked NON-array parts out of the locked array grid's bbox.

        A part wider than the grid pitch (e.g. a series resistor on a 3 mm-pitch
        LED matrix) that the force/SA pass drops onto the grid cannot escape via
        normal overlap resolution: every cardinal nudge just lands it on the next
        locked cell, so it stays overlapping and the leaf fails legality
        (``leaf_pre_stamp_legality_repair``). Pushing it clear of the WHOLE grid
        in one move fixes it. Array members are locked, so they are exempt (and
        member-vs-member clearance overlaps are by design). Returns parts moved.
        """
        if not self.cfg.get("array_grid_keepout", True):
            return 0
        members = [c for c in comps.values() if getattr(c, "array_member", False)]
        if not members:
            return 0
        margin = self.clearance / 2.0
        boxes = [c.bbox(margin) for c in members]
        tl = Point(min(b[0].x for b in boxes), min(b[0].y for b in boxes))
        br = Point(max(b[1].x for b in boxes), max(b[1].y for b in boxes))
        # Only evict parts too bulky to coexist beside the grid (a series
        # resistor), NOT small per-LED companions (decaps) -- evicting a whole
        # companion set would scatter it into a board-bloating sprawl. "Bulky" =
        # smaller dimension exceeds a grid member's footprint.
        member_extent = max(max(c.width_mm, c.height_mm) for c in members)
        return self._push_out_of_rect(
            comps, tl, br, owner_ref=None, min_extent_mm=member_extent
        )

    def _clamp_companions_inboard_of_connectors(
        self, comps: dict[str, Component], clearance: float
    ) -> int:
        """Push unlocked, non-connector parts inboard so their copper stays
        ``clearance`` mm behind the edge connector's outboard-most PAD face on
        each zoned side (self._edge_pinned_groups). The composed parent edge is
        drawn outboard of the connector's pads, so keeping companions behind the
        pads keeps them clear of the final edge (KC-S8PC37 R8). Uses pad copper
        faces (Pad.bbox), not component bboxes (a connector's shell/courtyard
        bbox overhangs the board; its pads sit inboard). Returns parts moved."""
        groups = getattr(self, "_edge_pinned_groups", None)
        if not groups:
            return 0
        moved = 0
        for side, conn_refs in groups.items():
            conn_set = set(conn_refs)
            pad_boxes = [
                p.bbox()
                for r in conn_refs
                if (c := comps.get(r)) is not None
                for p in c.pads
            ]
            if not pad_boxes:
                continue
            if side == "left":
                limit = min(b[0].x for b in pad_boxes) + clearance
            elif side == "right":
                limit = max(b[1].x for b in pad_boxes) - clearance
            elif side == "top":
                limit = min(b[0].y for b in pad_boxes) + clearance
            elif side == "bottom":
                limit = max(b[1].y for b in pad_boxes) - clearance
            else:
                continue
            for ref, comp in comps.items():
                if (
                    comp.locked
                    or ref in conn_set
                    or comp.kind in ("mounting_hole", "subcircuit", "connector")
                    or not comp.pads
                ):
                    continue
                boxes = [p.bbox() for p in comp.pads]
                if side == "left":
                    face = min(b[0].x for b in boxes)
                    dx, dy = (limit - face, 0.0) if face < limit else (0.0, 0.0)
                elif side == "right":
                    face = max(b[1].x for b in boxes)
                    dx, dy = (limit - face, 0.0) if face > limit else (0.0, 0.0)
                elif side == "top":
                    face = min(b[0].y for b in boxes)
                    dx, dy = (0.0, limit - face) if face < limit else (0.0, 0.0)
                else:  # bottom
                    face = max(b[1].y for b in boxes)
                    dx, dy = (0.0, limit - face) if face > limit else (0.0, 0.0)
                if dx == 0.0 and dy == 0.0:
                    continue
                comp.pos = Point(comp.pos.x + dx, comp.pos.y + dy)
                if comp.body_center is not None:
                    comp.body_center = Point(
                        comp.body_center.x + dx, comp.body_center.y + dy
                    )
                for pad in comp.pads:
                    pad.pos = Point(pad.pos.x + dx, pad.pos.y + dy)
                moved += 1
        return moved

    def _sa_refine(
        self,
        comps: dict,
        work_state,
        scorer,
        *,
        max_iters: int = 1000,
        init_temp: float = 5.0,
        cooling_rate: float = 0.995,
        move_radius: float = 2.0,
        swap_prob: float = 0.3,
        rotation_prob: float = 0.2,
    ) -> dict:
        """Simulated annealing refinement after force-directed placement.

        Performs single-component moves, pairwise swaps, and rotation
        perturbations with Metropolis acceptance criterion to escape
        local minima found by the force-directed solver.
        """
        import copy
        import math

        # Reuse the solver's primary RNG (self.rng) so SA draws stay on the
        # same deterministic stream as the rest of the solver. Previously
        # this created `random.Random(self.seed + 9999)`, which made SA
        # noise insensitive to upstream RNG state changes -- per-config
        # deltas (THT body blockers, area_factor tweaks, etc.) were drowned
        # out by SA stream variance, plausibly explaining recent
        # "no measurable win" reverts.
        rng = self.rng

        # Score current state
        work_state.components = comps
        current_score = scorer.score().total
        best_score = current_score
        best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}

        # Get unlocked component refs
        unlocked = [r for r, c in comps.items() if not c.locked]
        if not unlocked:
            return best_comps

        # Board bounds for clamping
        tl = work_state.board_outline[0]
        br = work_state.board_outline[1]

        temp = init_temp
        accepted = 0
        improved = 0
        iters_since_improvement = 0
        no_improve_break = int(self.cfg.get("sa_refine_no_improve_break", 150))
        temp_floor = init_temp * 0.001
        iters_run = 0

        for iteration in range(max_iters):
            iters_run = iteration + 1
            prev_improved = improved
            # Choose move type
            roll = rng.random()
            if roll < swap_prob and len(unlocked) >= 2:
                # Pairwise swap
                ref_a, ref_b = rng.sample(unlocked, 2)
                comp_a = comps[ref_a]
                comp_b = comps[ref_b]

                # Save old positions
                old_a = Point(comp_a.pos.x, comp_a.pos.y)
                old_b = Point(comp_b.pos.x, comp_b.pos.y)

                # Swap positions
                comp_a.pos = Point(old_b.x, old_b.y)
                comp_b.pos = Point(old_a.x, old_a.y)
                _update_pad_positions(comp_a, old_a, comp_a.rotation)
                _update_pad_positions(comp_b, old_b, comp_b.rotation)

                # Evaluate
                work_state.components = comps
                new_score = scorer.score().total
                delta = new_score - current_score

                if delta > 0 or rng.random() < math.exp(delta / max(temp, 0.001)):
                    current_score = new_score
                    accepted += 1
                    if new_score > best_score:
                        best_score = new_score
                        best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}
                        improved += 1
                else:
                    # Revert swap
                    comp_a.pos = Point(old_a.x, old_a.y)
                    comp_b.pos = Point(old_b.x, old_b.y)
                    _update_pad_positions(comp_a, old_b, comp_a.rotation)
                    _update_pad_positions(comp_b, old_a, comp_b.rotation)

            elif roll < swap_prob + rotation_prob:
                # Rotation perturbation
                ref = rng.choice(unlocked)
                comp = comps[ref]
                old_rot = comp.rotation
                if comp.allowed_rotations:
                    candidates = [r for r in comp.allowed_rotations if r != old_rot]
                    if not candidates:
                        continue
                    new_rot = float(rng.choice(candidates))
                else:
                    # Try 90-degree rotation increments
                    new_rot = (old_rot + rng.choice([90.0, 180.0, 270.0])) % 360.0
                # Keep the width/height AABB in sync with the rotation so
                # the scorer/legality passes and the eventual stamp see the
                # same extents. Blocks carry per-rotation geometry (not
                # always exact transposes); skip the move if the target
                # rotation has no geometry entry.
                block_geom = None
                if comp.kind == "subcircuit":
                    block_geom = (comp.block_rotation_geometry or {}).get(
                        float(new_rot)
                    )
                    if block_geom is None:
                        continue
                old_w, old_h = comp.width_mm, comp.height_mm
                rotate_component_in_place(comp, new_rot - old_rot)
                if block_geom is not None:
                    comp.width_mm = block_geom.width_mm
                    comp.height_mm = block_geom.height_mm

                work_state.components = comps
                new_score = scorer.score().total
                delta = new_score - current_score

                if delta > 0 or rng.random() < math.exp(delta / max(temp, 0.001)):
                    current_score = new_score
                    accepted += 1
                    if new_score > best_score:
                        best_score = new_score
                        best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}
                        improved += 1
                else:
                    # Revert rotation and restore the exact saved extents
                    rotate_component_in_place(comp, old_rot - new_rot)
                    comp.width_mm, comp.height_mm = old_w, old_h

            else:
                # Single component displacement
                ref = rng.choice(unlocked)
                comp = comps[ref]
                old_pos = Point(comp.pos.x, comp.pos.y)

                # Random displacement within move_radius
                dx = rng.gauss(0, move_radius * 0.5)
                dy = rng.gauss(0, move_radius * 0.5)
                new_x = max(tl.x, min(br.x, comp.pos.x + dx))
                new_y = max(tl.y, min(br.y, comp.pos.y + dy))
                comp.pos = Point(new_x, new_y)
                _update_pad_positions(comp, old_pos, comp.rotation)

                work_state.components = comps
                new_score = scorer.score().total
                delta = new_score - current_score

                if delta > 0 or rng.random() < math.exp(delta / max(temp, 0.001)):
                    current_score = new_score
                    accepted += 1
                    if new_score > best_score:
                        best_score = new_score
                        best_comps = {r: copy.deepcopy(c) for r, c in comps.items()}
                        improved += 1
                else:
                    # Revert displacement
                    comp.pos = Point(old_pos.x, old_pos.y)
                    _update_pad_positions(comp, Point(new_x, new_y), comp.rotation)

            # Cool down
            temp *= cooling_rate

            if improved > prev_improved:
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1

            # Adaptive convergence: exit once SA has stopped finding wins.
            # The 150-iter window is long enough to cover Metropolis stalls
            # at moderate temp (acceptance ~5-10% randomly accepted moves
            # without an actual best-score win), short enough to save
            # ~80% of wall-clock when SA has truly converged.
            if iters_since_improvement >= no_improve_break:
                break
            # Numerical floor: at very low temp the Metropolis exp() under-
            # flows and SA degenerates to greedy; no point continuing.
            if temp < temp_floor:
                break

        if improved > 0:
            print(f"  SA refine: {improved} improvements, {accepted} accepted of "
                  f"{iters_run}/{max_iters} (best {best_score:.1f} vs initial "
                  f"{current_score:.1f})")
        else:
            print(f"  SA refine: no improvement after {iters_run}/{max_iters} iterations")

        return best_comps

    def _grid_assignment_sa(self, comps: dict, grid, work_state, scorer) -> dict:
        """SA-as-assignment over the discrete anchor-relative grid: search which
        passive occupies which pin-adjacent slot (and its rotation), scored by
        the same PlacementScorer (pin-locality + routing). Thin wrapper over the
        pure ``leaf_grid_assignment.grid_assignment_sa`` so the geometry stays out
        of the solver; passes the solver's deterministic RNG."""
        from kicraft.autoplacer.brain.leaf_grid_assignment import grid_assignment_sa

        return grid_assignment_sa(
            comps,
            grid,
            work_state,
            scorer,
            rng=self.rng,
            max_iters=int(self.cfg.get("sa_refine_iterations", 300)),
            init_temp=float(self.cfg.get("sa_refine_initial_temp", 5.0)),
            cooling_rate=float(self.cfg.get("sa_refine_cooling_rate", 0.995)),
            swap_prob=float(self.cfg.get("grid_assignment_swap_prob", 0.4)),
            move_prob=float(self.cfg.get("grid_assignment_move_prob", 0.4)),
            no_improve_break=int(self.cfg.get("sa_refine_no_improve_break", 150)),
        )

    def _accumulate_attraction(
        self,
        comps: dict[str, Component],
        refs: list[str],
        forces: dict[str, Point],
        conn_graph: AdjacencyGraph,
    ) -> None:
        """Attraction: pull connected components together."""
        for ref in refs:
            for nbr, weight in conn_graph.neighbors(ref).items():
                if nbr not in comps:
                    continue
                a = comps[ref]
                b = comps[nbr]
                d = a.pos.dist(b.pos)
                if d < 0.1:
                    continue
                # Target distance based on component sizes
                target = (a.width_mm + b.width_mm) / 2 + self.clearance
                f_mag = self.k_attract * weight * (d - target)
                angle = math.atan2(b.pos.y - a.pos.y, b.pos.x - a.pos.x)
                forces[ref].x += f_mag * math.cos(angle)
                forces[ref].y += f_mag * math.sin(angle)

    def _stack_compatible_blocks(self, comps: dict[str, Component]) -> None:
        """Migrate small unlocked subcircuit blocks onto large
        opposite-side neighbors so dual-layer board real estate is
        actually used (e.g. front-side SMT regulators sit on top of
        the back-side battery footprint).

        Selection uses the dominant-blocker side of each block (front /
        back / dual / none), not the position-dependent
        ``_blocker_pair_compatible`` predicate. The position-dependent
        check returned False whenever a candidate's force-directed
        position happened to land inside a rotated anchor's pads, then
        the candidate fell through to a same-side anchor whose pads
        the candidate's row-pack target would directly overlap. The
        side-only intent rule removes that selection trap: same-side
        cand-anc pairs are never stacked, opposite-side pairs always
        are when geometry fits.

        Within each group, candidates are placed deterministically
        along a row inside the anchor's bbox (centered on the anchor's
        body center, packed left-to-right with spacing) so
        _resolve_overlaps doesn't have to push them apart from a single
        coincident point and watch them scatter outside the anchor.
        """
        from .subcircuit_composer import dominant_blocker_side

        anchors: list[tuple[str, Component]] = []
        candidates: list[tuple[str, Component]] = []
        for ref, comp in comps.items():
            if comp.kind != "subcircuit":
                continue
            if comp.block_blocker_set is None:
                continue
            if comp.locked:
                anchors.append((ref, comp))
            else:
                candidates.append((ref, comp))
        if not anchors or not candidates:
            return
        anchors.sort(key=lambda rc: rc[1].area, reverse=True)
        candidates.sort(key=lambda rc: rc[1].area, reverse=True)

        # Group candidates by their chosen anchor.
        groups: dict[str, list[tuple[str, Component]]] = {}
        anchor_by_ref = {ref: comp for ref, comp in anchors}

        def _effective_side(comp: Component) -> str:
            # block_force_back_only is the project-level override that
            # overrules the heuristic for both can_overlap_sparse and
            # this anchor-selection check. Without honouring it here,
            # a config-flagged anchor still classifies as "front" or
            # "dual" via dominant_blocker_side and the same-side gate
            # below skips it -- the override would be unable to revive
            # stacking on its own.
            if getattr(comp, "block_force_back_only", False):
                return "back"
            return dominant_blocker_side(comp.block_blocker_set)

        for cand_ref, cand in candidates:
            cand_side = _effective_side(cand)
            # Dual cands have copper on both layers and cannot stack on
            # anything without conflict; "none" cands carry no blocker
            # signal at all. Either way, leave at force-directed pos.
            if cand_side in ("dual", "none"):
                continue
            chosen_anchor_ref: str | None = None
            for anc_ref, anc in anchors:
                if cand.width_mm > anc.width_mm + 0.5:
                    continue
                if cand.height_mm > anc.height_mm + 0.5:
                    continue
                anc_side = _effective_side(anc)
                if anc_side == "none" or anc_side == cand_side:
                    continue
                chosen_anchor_ref = anc_ref
                break
            if chosen_anchor_ref is None:
                continue
            groups.setdefault(chosen_anchor_ref, []).append((cand_ref, cand))

        # Pack each group as a row centered on the anchor.
        for anc_ref, group in groups.items():
            anc = anchor_by_ref[anc_ref]
            # Full clearance between same-side candidates: half-clearance
            # left their copper close enough that the post-stack
            # _resolve_overlaps flagged them as conflict and tried to
            # push them apart, undoing the stack.
            spacing = max(0.5, self.clearance)
            row_width = sum(c.width_mm for _, c in group) + spacing * (len(group) - 1)
            col_height = sum(c.height_mm for _, c in group) + spacing * (len(group) - 1)
            anc_left = anc.pos.x - anc.width_mm / 2.0
            anc_top = anc.pos.y - anc.height_mm / 2.0
            anc_right = anc.pos.x + anc.width_mm / 2.0
            anc_bottom = anc.pos.y + anc.height_mm / 2.0
            # Bail if neither orientation fits inside the anchor. The
            # previous fall-through to col-pack clamped each candidate
            # to the anchor bbox, producing piles of overlapping blocks
            # and off-board positions (one observed at x=-1.51). Better
            # to leave the group at its force-directed positions and
            # let opposite-side attraction continue to pull on the
            # next solve.
            if row_width > anc.width_mm and col_height > anc.height_mm:
                continue
            # Body_center accounting: do NOT explicitly assign body_center
            # before _update_pad_positions. The helper translates body_center
            # by (new_pos - old_pos), so passing in a body_center already
            # set to the new pos lands it at (2*target - old_pos) -- a huge
            # offset from pos. _pad_half_extents then returns inflated
            # half-extents (height/2 + abs(body_center.y - pos.y)) and
            # _clamp_to_board snaps cands far up/in, breaking the row-pack
            # alignment. Just move pos and let the helper do its thing.
            if row_width <= anc.width_mm:
                cursor_x = anc.pos.x - row_width / 2.0
                for cand_ref, cand in group:
                    target_x = cursor_x + cand.width_mm / 2.0
                    target_y = anc.pos.y
                    target_x = max(
                        anc_left + cand.width_mm / 2.0,
                        min(anc_right - cand.width_mm / 2.0, target_x),
                    )
                    target_y = max(
                        anc_top + cand.height_mm / 2.0,
                        min(anc_bottom - cand.height_mm / 2.0, target_y),
                    )
                    old_pos = Point(cand.pos.x, cand.pos.y)
                    cand.pos = Point(target_x, target_y)
                    _update_pad_positions(cand, old_pos, cand.rotation)
                    # Mark intent so _resolve_overlaps doesn't escape
                    # this candidate off the anchor on a tiny
                    # position-dependent predicate flip.
                    cand.block_stacked_anchor = anc_ref
                    cursor_x += cand.width_mm + spacing
            else:
                cursor_y = anc.pos.y - col_height / 2.0
                for cand_ref, cand in group:
                    target_x = anc.pos.x
                    target_y = cursor_y + cand.height_mm / 2.0
                    target_x = max(
                        anc_left + cand.width_mm / 2.0,
                        min(anc_right - cand.width_mm / 2.0, target_x),
                    )
                    target_y = max(
                        anc_top + cand.height_mm / 2.0,
                        min(anc_bottom - cand.height_mm / 2.0, target_y),
                    )
                    old_pos = Point(cand.pos.x, cand.pos.y)
                    cand.pos = Point(target_x, target_y)
                    _update_pad_positions(cand, old_pos, cand.rotation)
                    cand.block_stacked_anchor = anc_ref
                    cursor_y += cand.height_mm + spacing

    def _nest_blocks_in_interior_holes(self, comps: dict[str, Component]) -> None:
        """Step 8.8: nest small subcircuit blocks inside another block's
        ENCLOSED interior hole (an LED-ring annulus centre) so a requested
        shaped outline can actually fit -- the K=4 candidate search then
        scores the collapsed bbox and shape fit for free. See
        docs/plans/shaped-compose-leaf-nesting.md (PR-N2).

        Fires only for shaped parents by default (``leaf_nesting: "auto"``:
        a non-rect ``board_outline`` shape was requested); ``"on"`` forces,
        ``"off"`` disables (kill switch). Strict edge/corner-zoned guests
        are never nested (their pin wins until PR-N4's demotion wave).

        The landing is verified with the position-dependent production
        predicate (``_blocker_pair_compatible`` -> the containment allowance
        in ``can_overlap_sparse``), and the PAIR IS LOCKED on success: every
        later pass moves only unlocked comps, so a nest cannot drift into
        the partial overlap the same-side veto exists to prevent.
        """
        mode = str(self.cfg.get("leaf_nesting", "auto") or "auto").lower()
        if mode in ("off", "false", "0", "none"):
            return
        if mode == "auto":
            outline = self.cfg.get("board_outline")
            shape = (str(outline.get("shape", "")).lower()
                     if isinstance(outline, dict) else "")
            if shape in ("", "rect", "rectangle"):
                return
        from .subcircuit_composer import _blocker_occupied_rects, _transform_rect

        zones = self.cfg.get("component_zones", {}) or {}
        hosts: list[tuple[str, Component]] = []
        guests: list[tuple[str, Component]] = []
        for ref in sorted(comps):
            comp = comps[ref]
            if comp.kind != "subcircuit" or comp.block_blocker_set is None:
                continue
            holes = getattr(comp.block_blocker_set, "interior_free_rects", ())
            if holes:
                hosts.append((ref, comp))
            elif not comp.locked:
                zone = zones.get(ref) or {}
                if "edge" in zone or "corner" in zone:
                    continue
                guests.append((ref, comp))
        if not hosts or not guests:
            return

        def _largest_hole_area(comp: Component) -> float:
            return max(
                (h[1].x - h[0].x) * (h[1].y - h[0].y)
                for h in comp.block_blocker_set.interior_free_rects
            )

        hosts.sort(key=lambda rc: (-_largest_hole_area(rc[1]), rc[0]))
        guests.sort(key=lambda rc: (-rc[1].area, rc[0]))

        for host_ref, host in hosts:
            m = abs(host.rotation) % 90.0
            if not (m < 0.01 or m > 89.99):
                continue  # hole rects only transform exactly at cardinals
            origin = _world_artifact_origin(host)
            for hole in host.block_blocker_set.interior_free_rects:
                hole_world = _transform_rect(hole, origin, host.rotation)
                cx = (hole_world[0].x + hole_world[1].x) / 2.0
                cy = (hole_world[0].y + hole_world[1].y) / 2.0
                for guest_ref, guest in guests:
                    if guest is host or guest.locked:
                        continue
                    old_pos = Point(guest.pos.x, guest.pos.y)
                    guest.pos = Point(cx, cy)
                    _update_pad_positions(guest, old_pos, guest.rotation)
                    if not _blocker_pair_compatible(host, guest):
                        # ``pos`` is the CONTENT centre, but the containment
                        # predicate tests the OCCUPIED bbox (content plus
                        # trace/pad inflation), whose centre can sit a few
                        # tenths off -- fatal exactly when the hole slack is
                        # tight (the real 1/601 guest has ~0.3 mm per side).
                        # Re-land with the occupied bbox centred in the hole
                        # and let the same production predicate decide.
                        g_origin = _world_artifact_origin(guest)
                        g_rects = [
                            _transform_rect(r, g_origin, guest.rotation)
                            for r in _blocker_occupied_rects(
                                guest.block_blocker_set
                            )
                        ]
                        if g_rects:
                            obb_cx = (min(r[0].x for r in g_rects)
                                      + max(r[1].x for r in g_rects)) / 2.0
                            obb_cy = (min(r[0].y for r in g_rects)
                                      + max(r[1].y for r in g_rects)) / 2.0
                            before = Point(guest.pos.x, guest.pos.y)
                            guest.pos = Point(
                                guest.pos.x + (cx - obb_cx),
                                guest.pos.y + (cy - obb_cy),
                            )
                            _update_pad_positions(
                                guest, before, guest.rotation
                            )
                    if _blocker_pair_compatible(host, guest):
                        guest.block_nested_anchor = host_ref
                        guest.locked = True
                        host.locked = True
                        print(
                            f"  [nest] block {guest_ref} nested inside "
                            f"{host_ref}'s interior hole at "
                            f"({guest.pos.x:.1f}, {guest.pos.y:.1f})"
                            f" -- pair locked"
                        )
                        break  # one guest per hole
                    # Doesn't fit this hole at its current rotation: revert.
                    reverted = Point(guest.pos.x, guest.pos.y)
                    guest.pos = old_pos
                    _update_pad_positions(guest, reverted, guest.rotation)

    def _accumulate_opposite_side_attraction(
        self,
        comps: dict[str, Component],
        forces: dict[str, Point],
    ) -> None:
        """Pull blocker-compatible block pairs (e.g. front-only x back-only)
        toward each other so opposite-side stacking emerges naturally
        during force-directed refinement.

        Without this term, the blocker-aware repulsion gate only *allows*
        overlap; it never *encourages* it. On dual-layer parents like
        LLUPS that means small SMT blocks scatter to whatever empty
        space exists rather than parking on top of the back-side
        battery footprint -- exactly the wasted-space failure mode the
        user flagged.

        Force is proportional to combined block area (so bigger blocks
        exert stronger pull) and falls off with distance. Skipped for
        non-block leaf components (block_blocker_set is None) so leaf
        placement is unchanged.
        """
        weight = float(self.cfg.get("opposite_side_attraction_k", 0.4))
        if weight <= 0.0:
            return
        ref_list = list(comps.keys())
        for i in range(len(ref_list)):
            a = comps[ref_list[i]]
            if a.block_blocker_set is None:
                continue
            for j in range(i + 1, len(ref_list)):
                b = comps[ref_list[j]]
                if b.block_blocker_set is None:
                    continue
                if a.locked and b.locked:
                    continue
                if not _blocker_pair_compatible(a, b):
                    continue
                d = a.pos.dist(b.pos)
                if d < 0.5:
                    continue
                f_mag = weight * math.sqrt(max(1.0, a.area * b.area)) / d
                angle = math.atan2(b.pos.y - a.pos.y, b.pos.x - a.pos.x)
                fx = f_mag * math.cos(angle)
                fy = f_mag * math.sin(angle)
                if not a.locked:
                    forces[ref_list[i]].x += fx
                    forces[ref_list[i]].y += fy
                if not b.locked:
                    forces[ref_list[j]].x -= fx
                    forces[ref_list[j]].y -= fy

    def _accumulate_repulsion_python(
        self,
        comps: dict[str, Component],
        forces: dict[str, Point],
    ) -> None:
        """Repulsion (pure Python): push overlapping/close components apart.

        Locked components (connectors, holes) act as repellers even though
        they don't move — this keeps unlocked parts from clustering against them.
        """
        ref_list = list(comps.keys())
        for i in range(len(ref_list)):
            a = comps[ref_list[i]]
            for j in range(i + 1, len(ref_list)):
                b = comps[ref_list[j]]
                if a.locked and b.locked:
                    continue  # both fixed, nothing to do
                if _blocker_pair_compatible(a, b):
                    continue
                d = a.pos.dist(b.pos)
                min_dist = (
                    max(a.width_mm, a.height_mm) + max(b.width_mm, b.height_mm)
                ) / 2 + self.clearance
                if d > min_dist * 2:
                    continue  # too far to matter
                if d < 0.1:
                    d = 0.1
                f_mag = self.k_repel * (a.area * b.area) / (d * d)
                angle = math.atan2(a.pos.y - b.pos.y, a.pos.x - b.pos.x)
                fx = f_mag * math.cos(angle)
                fy = f_mag * math.sin(angle)
                if not a.locked:
                    forces[ref_list[i]].x += fx
                    forces[ref_list[i]].y += fy
                if not b.locked:
                    forces[ref_list[j]].x -= fx
                    forces[ref_list[j]].y -= fy

    def _accumulate_repulsion_numpy(
        self,
        comps: dict[str, Component],
        forces: dict[str, Point],
    ) -> None:
        """Repulsion (numpy-accelerated): push overlapping/close components apart.

        Locked components (connectors, holes) act as repellers even though
        they don't move — this keeps unlocked parts from clustering against them.
        """
        ref_list = list(comps.keys())

        pos_x = np.array([comps[r].pos.x for r in ref_list], dtype=np.float64)
        pos_y = np.array([comps[r].pos.y for r in ref_list], dtype=np.float64)
        areas = np.array([comps[r].area for r in ref_list], dtype=np.float64)
        widths = np.array([comps[r].width_mm for r in ref_list], dtype=np.float64)
        heights = np.array([comps[r].height_mm for r in ref_list], dtype=np.float64)
        locked = np.array([comps[r].locked for r in ref_list], dtype=bool)

        max_dims = np.maximum(widths, heights)
        min_dists = (
            max_dims[:, np.newaxis] + max_dims[np.newaxis, :]
        ) / 2 + self.clearance

        dx = pos_x[:, np.newaxis] - pos_x[np.newaxis, :]
        dy = pos_y[:, np.newaxis] - pos_y[np.newaxis, :]
        dists = np.sqrt(dx * dx + dy * dy)

        skip_mask = dists > min_dists * 2

        # Match the pure-Python path: clamp the magnitude distance to 0.1
        # so near/coincident pairs get the STRONGEST repulsion (masking
        # them out left stacked components with zero anti-coincidence
        # force), and use a true unit direction vector.
        clamped = np.maximum(dists, 0.1)
        force_mags = (
            self.k_repel
            * (areas[:, np.newaxis] * areas[np.newaxis, :])
            / (clamped * clamped)
        )
        np.fill_diagonal(force_mags, 0)
        force_mags = np.where(skip_mask, 0, force_mags)

        # Exactly-coincident pairs have no defined direction; the Python
        # path resolves them via atan2(0, 0) == 0, pushing the lower index
        # +x and the higher -x. sign(col - row) reproduces that and keeps
        # the matrix antisymmetric (entry [i, j] is the force ON i FROM j).
        degenerate = dists < 1e-9
        idx = np.arange(len(ref_list))
        fallback_dx = np.sign(idx[np.newaxis, :] - idx[:, np.newaxis]).astype(
            np.float64
        )
        safe_dists = np.where(degenerate, 1.0, dists)
        norm_dx = np.where(degenerate, fallback_dx, dx / safe_dists)
        norm_dy = np.where(degenerate, 0.0, dy / safe_dists)

        fx_matrix = force_mags * norm_dx
        fy_matrix = force_mags * norm_dy

        both_locked = locked[:, np.newaxis] & locked[np.newaxis, :]
        np.fill_diagonal(both_locked, False)

        fx_matrix = np.where(both_locked, 0, fx_matrix)
        fy_matrix = np.where(both_locked, 0, fy_matrix)

        # Blocker-aware compatibility: zero pairwise repulsion when both
        # components carry blocker sets and their copper does not conflict.
        # For pure leaf placement (no blocker_set on any component) this is
        # an O(N) early exit and the matrix is unchanged.
        any_block = any(comps[r].block_blocker_set is not None for r in ref_list)
        if any_block:
            n = len(ref_list)
            compat = np.zeros((n, n), dtype=bool)
            for i in range(n):
                a = comps[ref_list[i]]
                if a.block_blocker_set is None:
                    continue
                for j in range(i + 1, n):
                    b = comps[ref_list[j]]
                    if b.block_blocker_set is None:
                        continue
                    if _blocker_pair_compatible(a, b):
                        compat[i, j] = True
                        compat[j, i] = True
            fx_matrix = np.where(compat, 0, fx_matrix)
            fy_matrix = np.where(compat, 0, fy_matrix)

        fx_totals = fx_matrix.sum(axis=1)
        fy_totals = fy_matrix.sum(axis=1)

        for i, ref in enumerate(ref_list):
            if not comps[ref].locked:
                forces[ref].x += float(fx_totals[i])
                forces[ref].y += float(fy_totals[i])

    def _accumulate_smt_opposite_tht_force(
        self,
        comps: dict[str, Component],
        refs: list[str],
        forces: dict[str, Point],
    ) -> None:
        """SMT-opposite-THT attraction: pull unlocked SMT components toward
        the nearest point on the nearest back-layer THT bounding box.

        This distributes SMT across the available THT courtyard space
        rather than clustering them all at the centroid.
        """
        if not self.cfg.get("smt_opposite_tht", True):
            return
        back_tht = [
            c for c in comps.values() if c.is_through_hole and c.layer == Layer.BACK
        ]
        if not back_tht:
            return
        smt_k = self.k_attract * 0.6
        # Pre-compute back-THT bboxes
        btht_bboxes = [
            (
                t.pos.x - t.width_mm / 2,
                t.pos.y - t.height_mm / 2,
                t.pos.x + t.width_mm / 2,
                t.pos.y + t.height_mm / 2,
            )
            for t in back_tht
        ]
        for ref in refs:
            c = comps[ref]
            if c.is_through_hole or c.layer == Layer.BACK:
                continue
            # Find nearest point on nearest back-THT bbox
            best_dist = float("inf")
            best_tx, best_ty = c.pos.x, c.pos.y
            for bx0, by0, bx1, by1 in btht_bboxes:
                # Clamp SMT center to THT bbox = nearest point on bbox
                nx = max(bx0, min(bx1, c.pos.x))
                ny = max(by0, min(by1, c.pos.y))
                nd = math.hypot(c.pos.x - nx, c.pos.y - ny)
                if nd < best_dist:
                    best_dist = nd
                    best_tx, best_ty = nx, ny
            if best_dist < 0.1:
                continue
            f_mag = smt_k * best_dist
            angle = math.atan2(best_ty - c.pos.y, best_tx - c.pos.x)
            forces[ref].x += f_mag * math.cos(angle)
            forces[ref].y += f_mag * math.sin(angle)

    def _accumulate_boundary_force(
        self,
        comps: dict[str, Component],
        refs: list[str],
        forces: dict[str, Point],
        tl: Point,
        br: Point,
    ) -> None:
        """Boundary: strong spring force at edges (pad-aware extents)."""
        margin = self.edge_margin + 2.0
        k_boundary = 10.0
        for ref in refs:
            c = comps[ref]
            hw, hh = _pad_half_extents(c)
            if c.pos.x - hw < tl.x + margin:
                forces[ref].x += k_boundary * (tl.x + margin - (c.pos.x - hw))
            if c.pos.x + hw > br.x - margin:
                forces[ref].x -= k_boundary * ((c.pos.x + hw) - (br.x - margin))
            if c.pos.y - hh < tl.y + margin:
                forces[ref].y += k_boundary * (tl.y + margin - (c.pos.y - hh))
            if c.pos.y + hh > br.y - margin:
                forces[ref].y -= k_boundary * ((c.pos.y + hh) - (br.y - margin))

    def _accumulate_center_attraction(
        self,
        comps: dict[str, Component],
        refs: list[str],
        forces: dict[str, Point],
        tl: Point,
        br: Point,
    ) -> None:
        """Center attraction: weak force pulling components toward board center
        to prevent edge-clumping bias."""
        cx = (tl.x + br.x) / 2.0
        cy = (tl.y + br.y) / 2.0
        k_center = 0.02  # weak — just enough to break edge-clumping symmetry
        for ref in refs:
            c = comps[ref]
            dx = cx - c.pos.x
            dy = cy - c.pos.y
            dist = max(0.1, (dx * dx + dy * dy) ** 0.5)
            # Scale by distance from center — stronger pull for far-flung components
            strength = k_center * dist / max(1.0, (br.x - tl.x))
            forces[ref].x += strength * dx
            forces[ref].y += strength * dy

    def _accumulate_alignment_force(
        self,
        comps: dict[str, Component],
        forces: dict[str, Point],
    ) -> None:
        """Large-pair alignment: keep paired components sharing an axis."""
        if not self._aligned_pairs:
            return
        for ref_a, ref_b, axis in self._aligned_pairs:
            if ref_a not in comps or ref_b not in comps:
                continue
            a, b = comps[ref_a], comps[ref_b]
            if axis == "y":  # horizontal side-by-side: share Y
                mid_y = (a.pos.y + b.pos.y) / 2
                if ref_a in forces:
                    forces[ref_a].y += 1.5 * (mid_y - a.pos.y)
                if ref_b in forces:
                    forces[ref_b].y += 1.5 * (mid_y - b.pos.y)
            else:  # vertical: share X
                mid_x = (a.pos.x + b.pos.x) / 2
                if ref_a in forces:
                    forces[ref_a].x += 1.5 * (mid_x - a.pos.x)
                if ref_b in forces:
                    forces[ref_b].x += 1.5 * (mid_x - b.pos.x)

    def _apply_forces(
        self,
        comps: dict[str, Component],
        refs: list[str],
        forces: dict[str, Point],
        damping: float,
        tl: Point,
        br: Point,
    ) -> float:
        """Apply accumulated forces with damping and displacement clamping.
        Returns max displacement."""
        max_disp = 0.0
        for ref in refs:
            dx = forces[ref].x * damping
            dy = forces[ref].y * damping
            # Clamp max displacement per step
            mag = math.hypot(dx, dy)
            max_step = 2.0 * damping
            if mag > max_step:
                dx *= max_step / mag
                dy *= max_step / mag
                mag = max_step

            old_pos = Point(comps[ref].pos.x, comps[ref].pos.y)
            old_rot = comps[ref].rotation
            comps[ref].pos.x += dx
            comps[ref].pos.y += dy

            # Hard clamp: pad-aware extents must stay inside board
            c = comps[ref]
            hw, hh = _pad_half_extents(c)
            c.pos.x = max(tl.x + hw + 1.0, min(br.x - hw - 1.0, c.pos.x))
            c.pos.y = max(tl.y + hh + 1.0, min(br.y - hh - 1.0, c.pos.y))

            _update_pad_positions(comps[ref], old_pos, old_rot)

            max_disp = max(max_disp, mag)

        return max_disp

    def _post_step_clamp(
        self,
        comps: dict[str, Component],
        refs: list[str],
    ) -> None:
        """Post-step: zone re-clamping and aligned-pair re-snapping.

        Keep zone-constrained components within their designated zone bounds
        (prevents drift during force simulation).
        """
        zones_cfg = self.cfg.get("component_zones", {})
        for ref in refs:
            zone_cfg = zones_cfg.get(ref, {})
            if "zone" not in zone_cfg:
                continue
            c = comps[ref]
            zx0, zy0, zx1, zy1 = self._get_zone_bounds(zone_cfg["zone"])
            hw, hh = _pad_half_extents(c)
            clamped_x = max(zx0 + hw, min(zx1 - hw, c.pos.x))
            clamped_y = max(zy0 + hh, min(zy1 - hh, c.pos.y))
            if abs(clamped_x - c.pos.x) > 0.01 or abs(clamped_y - c.pos.y) > 0.01:
                old_pos = Point(c.pos.x, c.pos.y)
                c.pos.x = clamped_x
                c.pos.y = clamped_y
                _update_pad_positions(c, old_pos, c.rotation)

        # Post-step: re-snap aligned pairs to shared coordinate
        self._re_snap_aligned_pairs(comps)

    def _resolve_overlaps(self, comps: dict[str, Component]):
        """Push apart components until no bboxes overlap (including clearance gap).

        For each overlapping pair, picks the escape direction that requires the
        least travel distance AND keeps the free component within board bounds.
        This handles edge cases where the shortest-axis push would send a component
        into a board edge (e.g. a small part trapped between a large locked battery
        holder and the board boundary).
        """
        refs = list(comps.keys())
        half_gap = self.clearance / 2.0
        tl, br = self.state.board_outline

        def _clamp_comp_to_board(
            comp: Component, nx: float, ny: float
        ) -> tuple[float, float]:
            hw, hh = _pad_half_extents(comp)
            return (
                max(tl.x + hw + 1.0, min(br.x - hw - 1.0, nx)),
                max(tl.y + hh + 1.0, min(br.y - hh - 1.0, ny)),
            )

        def _total_overlap_area_for(
            comp: Component, others: dict[str, Component]
        ) -> float:
            comp_tl, comp_br = _effective_bbox(comp, half_gap)
            total = 0.0
            for other in others.values():
                if other is comp:
                    continue
                other_tl, other_br = _effective_bbox(other, half_gap)
                _ox = min(comp_br.x, other_br.x) - max(comp_tl.x, other_tl.x)
                _oy = min(comp_br.y, other_br.y) - max(comp_tl.y, other_tl.y)
                if _ox > 0 and _oy > 0:
                    total += _ox * _oy
            return total

        def _escape(free_c: Component, lock_tl: Point, lock_br: Point) -> bool:
            """Push free_c fully out of lock bbox. Returns True if moved."""
            fc_tl, fc_br = _effective_bbox(free_c, half_gap)
            ox, oy = _bbox_overlap_xy(lock_tl, lock_br, fc_tl, fc_br)
            if ox <= 0 or oy <= 0:
                return False

            # Full-clearance distances: move so trailing edge of free_c
            # clears the leading edge of the lock bbox entirely.
            clear_right = lock_br.x - fc_tl.x + 0.1
            clear_left = fc_br.x - lock_tl.x + 0.1
            clear_down = lock_br.y - fc_tl.y + 0.1
            clear_up = fc_br.y - lock_tl.y + 0.1

            moves = [
                (clear_right, free_c.pos.x + clear_right, free_c.pos.y),
                (clear_left, free_c.pos.x - clear_left, free_c.pos.y),
                (clear_down, free_c.pos.x, free_c.pos.y + clear_down),
                (clear_up, free_c.pos.x, free_c.pos.y - clear_up),
            ]

            old = Point(free_c.pos.x, free_c.pos.y)
            old_overlap = _total_overlap_area_for(free_c, comps)
            best_key: tuple[float, int, float] | None = None
            best_move = (free_c.pos.x, free_c.pos.y)

            for travel, nx, ny in moves:
                nx_c, ny_c = _clamp_comp_to_board(free_c, nx, ny)
                clamped = abs(nx_c - nx) > 0.01 or abs(ny_c - ny) > 0.01

                free_c.pos.x, free_c.pos.y = nx_c, ny_c
                _update_pad_positions(free_c, old, free_c.rotation)
                new_overlap = _total_overlap_area_for(free_c, comps)
                improvement = old_overlap - new_overlap

                key = (-improvement, 1 if clamped else 0, travel)
                if best_key is None or key < best_key:
                    best_key = key
                    best_move = (nx_c, ny_c)

                free_c.pos.x, free_c.pos.y = old.x, old.y
                _update_pad_positions(free_c, Point(nx_c, ny_c), free_c.rotation)

            nx, ny = best_move
            free_c.pos.x, free_c.pos.y = nx, ny
            _update_pad_positions(free_c, old, free_c.rotation)
            return abs(nx - old.x) > 0.01 or abs(ny - old.y) > 0.01

        for iteration in range(300):
            moved = False

            # --- Pass 1: resolve free-free overlaps first ---
            for i in range(len(refs)):
                a = comps[refs[i]]
                if a.locked:
                    continue
                a_tl, a_br = _effective_bbox(a, half_gap)
                for j in range(i + 1, len(refs)):
                    b = comps[refs[j]]
                    if b.locked:
                        continue

                    b_tl, b_br = _effective_bbox(b, half_gap)
                    ox, oy = _bbox_overlap_xy(a_tl, a_br, b_tl, b_br)
                    if ox <= 0 or oy <= 0:
                        continue

                    if _blocker_pair_compatible(a, b):
                        continue

                    hw_a, hh_a = _pad_half_extents(a)
                    hw_b, hh_b = _pad_half_extents(b)
                    # Push by overlap + full clearance (not just overlap + 0.1)
                    # so the steady state actually separates by clearance,
                    # not "barely disjoint." With strong inter-net attraction
                    # the previous (overlap + 0.1) / 2 push only kept blocks
                    # 0.1 mm disjoint on the bbox-effective metric, which
                    # left content traces within trace-clearance of each
                    # other and produced stamp shorts at the seam.
                    extra = max(0.5, self.clearance)
                    if ox < oy:
                        push = (ox + extra) / 2
                        sign = 1.0 if a.pos.x >= b.pos.x else -1.0
                        old_a = Point(a.pos.x, a.pos.y)
                        old_b = Point(b.pos.x, b.pos.y)
                        a.pos.x = max(
                            tl.x + hw_a + 1.0,
                            min(br.x - hw_a - 1.0, a.pos.x + sign * push),
                        )
                        b.pos.x = max(
                            tl.x + hw_b + 1.0,
                            min(br.x - hw_b - 1.0, b.pos.x - sign * push),
                        )
                    else:
                        push = (oy + extra) / 2
                        sign = 1.0 if a.pos.y >= b.pos.y else -1.0
                        old_a = Point(a.pos.x, a.pos.y)
                        old_b = Point(b.pos.x, b.pos.y)
                        a.pos.y = max(
                            tl.y + hh_a + 1.0,
                            min(br.y - hh_a - 1.0, a.pos.y + sign * push),
                        )
                        b.pos.y = max(
                            tl.y + hh_b + 1.0,
                            min(br.y - hh_b - 1.0, b.pos.y - sign * push),
                        )
                    _update_pad_positions(a, old_a, a.rotation)
                    _update_pad_positions(b, old_b, b.rotation)
                    a_tl, a_br = _effective_bbox(a, half_gap)
                    moved = True

            # --- Pass 2: resolve locked-involving overlaps (escape) ---
            for i in range(len(refs)):
                a = comps[refs[i]]
                a_tl, a_br = _effective_bbox(a, half_gap)
                for j in range(i + 1, len(refs)):
                    b = comps[refs[j]]
                    if not a.locked and not b.locked:
                        continue  # already handled in pass 1

                    b_tl, b_br = _effective_bbox(b, half_gap)
                    ox, oy = _bbox_overlap_xy(a_tl, a_br, b_tl, b_br)
                    if ox <= 0 or oy <= 0:
                        continue

                    if _blocker_pair_compatible(a, b):
                        continue

                    # Intentional stack: _stack_compatible_blocks
                    # already vetted opposite-side compatibility and
                    # row/col-packed this candidate inside the anchor's
                    # bbox. Pass 1's tiny free-free pushes can drift a
                    # candidate by a few mm so its SMT pad lands on an
                    # anchor's THT corner-ring rect, flipping the
                    # position-dependent _blocker_pair_compatible from
                    # True to False; without this gate _escape() then
                    # moves the candidate ~30-50 mm to clear the
                    # anchor's bbox entirely, undoing the whole stack
                    # pass and producing the sprawled layouts that
                    # routing then fails on. The stacking decision is
                    # the source of truth here -- preserve it.
                    if a.locked and getattr(b, "block_stacked_anchor", None) == refs[i]:
                        continue
                    if b.locked and getattr(a, "block_stacked_anchor", None) == refs[j]:
                        continue
                    # Intentional nest (Step 8.8): same source-of-truth rule
                    # as the stack gate above. Both partners are locked at
                    # nest time so this is defensive parity, not a live path.
                    if getattr(b, "block_nested_anchor", None) == refs[i]:
                        continue
                    if getattr(a, "block_nested_anchor", None) == refs[j]:
                        continue

                    # Array-grid members are intentionally placed on a fixed
                    # grid and are self-legal by construction; never escape one
                    # grid member from another. A dense locked array would
                    # otherwise thrash this O(n^2) escape loop indefinitely.
                    if getattr(a, "array_member", False) and getattr(
                        b, "array_member", False
                    ):
                        continue

                    if a.locked and b.locked:
                        zones = self.cfg.get("component_zones", {})
                        a_pinned = refs[i] in zones and (
                            "edge" in zones[refs[i]] or "corner" in zones[refs[i]]
                        )
                        b_pinned = refs[j] in zones and (
                            "edge" in zones[refs[j]] or "corner" in zones[refs[j]]
                        )
                        if a_pinned and not b_pinned:
                            if _escape(b, a_tl, a_br):
                                b_tl, b_br = _effective_bbox(b, half_gap)
                                moved = True
                        elif b_pinned and not a_pinned:
                            if _escape(a, b_tl, b_br):
                                a_tl, a_br = _effective_bbox(a, half_gap)
                                moved = True
                        else:
                            a_area = a.width_mm * a.height_mm
                            b_area = b.width_mm * b.height_mm
                            if a_area <= b_area:
                                if _escape(a, b_tl, b_br):
                                    a_tl, a_br = _effective_bbox(a, half_gap)
                                    moved = True
                            else:
                                if _escape(b, a_tl, a_br):
                                    b_tl, b_br = _effective_bbox(b, half_gap)
                                    moved = True
                    elif a.locked:
                        if _escape(b, a_tl, a_br):
                            b_tl, b_br = _effective_bbox(b, half_gap)
                            moved = True
                    elif b.locked:
                        if _escape(a, b_tl, b_br):
                            a_tl, a_br = _effective_bbox(a, half_gap)
                            moved = True

            if not moved:
                break  # fully separated

    def _resolve_courtyard_overlaps(self, comps: dict[str, Component]) -> int:
        """Final guarantee: no two SAME-SIDE parts have overlapping courtyards.

        The full overlap resolution (Step 9 / 13.5) runs BEFORE the pinned-
        position restore, board clamp and keep-out-clear passes -- each of
        which can nudge a component back into a neighbour's courtyard with
        nothing re-resolving it. That ordering gap is the root cause of the
        systematic ``courtyards_overlap`` DRC failures: a same-side pair the
        solver correctly separated drifts back together in the final steps and
        survives to the routed board. This pass runs LAST, so it cleans up
        whatever those steps introduced.

        Unlike ``_resolve_overlaps`` (which separates every pair to the full
        placement clearance), this only touches pairs whose courtyards actually
        overlap and pushes them apart by a hair (``courtyard_overlap_min_gap_mm``)
        -- enough to clear the DRC without re-bloating a tight board.

        Pinned/locked parts (edge connectors, mounting holes) are never moved;
        only the unlocked partner is pushed. Opposite-side dual-layer stacks are
        exempt via ``_blocker_pair_compatible`` (their courtyards sit on
        different copper layers and never DRC-overlap), as are array-grid
        members (self-legal by construction).

        Returns the number of overlapping pairs left unresolved (both parts
        locked -- the pass cannot fix those without disturbing a pinned
        position), for telemetry.
        """
        gap = float(self.cfg.get("courtyard_overlap_min_gap_mm", 0.15))
        half_gap = gap / 2.0
        tl, br = self.state.board_outline
        refs = list(comps.keys())

        def _push_clear(free_c: Component, fix_tl: Point, fix_br: Point) -> bool:
            """Push unlocked ``free_c`` minimally out of ``[fix_tl, fix_br]``.

            Prefers the smaller-overlap axis, but if that push is fully blocked
            by the board edge (e.g. an edge-pinned connector already flush
            against the boundary cannot move further out on its pinned axis),
            falls back to the other axis. Without the fallback a same-edge pair
            whose smaller overlap is on the pinned (perpendicular) axis can never
            separate -- the survivor of the run_06 USB-C breakout courtyard
            overlap. Returns True if it moved."""
            f_tl, f_br = _effective_bbox(free_c, half_gap)
            ox, oy = _bbox_overlap_xy(fix_tl, fix_br, f_tl, f_br)
            if ox <= 0 or oy <= 0:
                return False
            hw, hh = _pad_half_extents(free_c)

            def _try_x() -> bool:
                old = Point(free_c.pos.x, free_c.pos.y)
                center = (fix_tl.x + fix_br.x) / 2.0
                sign = 1.0 if free_c.pos.x >= center else -1.0
                nx = free_c.pos.x + sign * (ox + 0.02)
                free_c.pos.x = max(tl.x + hw + 1.0, min(br.x - hw - 1.0, nx))
                if abs(free_c.pos.x - old.x) <= 1e-3:
                    return False
                _update_pad_positions(free_c, old, free_c.rotation)
                return True

            def _try_y() -> bool:
                old = Point(free_c.pos.x, free_c.pos.y)
                center = (fix_tl.y + fix_br.y) / 2.0
                sign = 1.0 if free_c.pos.y >= center else -1.0
                ny = free_c.pos.y + sign * (oy + 0.02)
                free_c.pos.y = max(tl.y + hh + 1.0, min(br.y - hh - 1.0, ny))
                if abs(free_c.pos.y - old.y) <= 1e-3:
                    return False
                _update_pad_positions(free_c, old, free_c.rotation)
                return True

            if ox <= oy:
                return _try_x() or _try_y()
            return _try_y() or _try_x()

        def _slide_locked_pair(ref_a: str, a: Component, ref_b: str, b: Component) -> bool:
            """Separate two LOCKED parts by sliding one ALONG their shared edge.

            Edge-pinned connectors are locked to keep their mouth flush with
            the board edge -- but that only fixes the coordinate PERPENDICULAR
            to the edge; sliding along the edge preserves flushness. Two
            same-edge locked connectors whose courtyards overlap (batch
            20260716T011056Z run_26: servo headers J5/J7 at 3.40 mm pitch vs
            3.63 mm courtyards -- pure-THT leaves are copper-transparent to
            ``can_overlap_sparse``, so nothing upstream held them apart) are
            therefore separable here without breaking either pin. Mounting
            holes stay untouchable (their position is the user's spec, not an
            edge pin). Returns True when a part moved."""
            if a.kind == "mounting_hole" or b.kind == "mounting_hole":
                return False
            a_tl, a_br = _effective_bbox(a, half_gap)
            b_tl, b_br = _effective_bbox(b, half_gap)
            ox, oy = _bbox_overlap_xy(a_tl, a_br, b_tl, b_br)
            if ox <= 0 or oy <= 0:
                return False
            # Same-edge pairs separate mainly along the edge axis = the axis
            # of larger centre separation; the perpendicular (pinned) axis is
            # never touched, so flush contact survives.
            slide_x = abs(a.pos.x - b.pos.x) >= abs(a.pos.y - b.pos.y)
            need = (ox if slide_x else oy) + 0.02
            for mover, other in ((b, a), (a, b)):
                old = Point(mover.pos.x, mover.pos.y)
                hw, hh = _pad_half_extents(mover)
                if slide_x:
                    sign = 1.0 if mover.pos.x >= other.pos.x else -1.0
                    nx = max(tl.x + hw + 1.0, min(br.x - hw - 1.0, mover.pos.x + sign * need))
                    if abs(nx - old.x) <= 1e-3:
                        continue
                    mover.pos.x = nx
                else:
                    sign = 1.0 if mover.pos.y >= other.pos.y else -1.0
                    ny = max(tl.y + hh + 1.0, min(br.y - hh - 1.0, mover.pos.y + sign * need))
                    if abs(ny - old.y) <= 1e-3:
                        continue
                    mover.pos.y = ny
                _update_pad_positions(mover, old, mover.rotation)
                mover_ref = ref_b if mover is b else ref_a
                pinned = getattr(self, "_pinned_targets", None)
                if pinned is not None and mover_ref in pinned:
                    pinned[mover_ref] = Point(mover.pos.x, mover.pos.y)
                return True
            return False

        unresolved = 0
        for _ in range(200):
            moved = False
            unresolved = 0
            for i in range(len(refs)):
                a = comps[refs[i]]
                a_tl, a_br = _effective_bbox(a, half_gap)
                for j in range(i + 1, len(refs)):
                    b = comps[refs[j]]
                    b_tl, b_br = _effective_bbox(b, half_gap)
                    ox, oy = _bbox_overlap_xy(a_tl, a_br, b_tl, b_br)
                    if ox <= 0 or oy <= 0:
                        continue
                    # Opposite-side dual-layer stack: courtyards are on
                    # different copper layers; KiCad never flags them as a
                    # same-side courtyards_overlap. Leave the stack intact.
                    # Copper-compatibility alone is NOT enough -- two SAME-side
                    # leaves whose sparse copper happens not to conflict (e.g.
                    # two THT pin-headers whose annular rings don't touch) still
                    # share one courtyard layer and DO produce a real
                    # courtyards_overlap DRC, so they must be separated here.
                    if _blocker_pair_compatible(a, b) and _back_courtyard(
                        a
                    ) != _back_courtyard(b):
                        continue
                    # Intentional nest (Step 8.8): the BLOCK bboxes overlap
                    # by design, but the guest sits inside the host's
                    # enclosed interior hole, so the real per-footprint
                    # courtyards do not touch (the containment allowance in
                    # can_overlap_sparse guarantees standoff). KiCad's
                    # courtyard DRC on the stamped board stays the
                    # authoritative measurement.
                    if (getattr(a, "block_nested_anchor", None) == refs[j]
                            or getattr(b, "block_nested_anchor", None) == refs[i]):
                        continue
                    if getattr(a, "array_member", False) and getattr(
                        b, "array_member", False
                    ):
                        continue
                    if a.locked and b.locked:
                        # Edge pins fix only the perpendicular coordinate:
                        # slide one part ALONG the shared edge (flushness
                        # survives). Only mounting holes are truly immovable.
                        if _slide_locked_pair(refs[i], a, refs[j], b):
                            a_tl, a_br = _effective_bbox(a, half_gap)
                            moved = True
                        else:
                            unresolved += 1
                        continue
                    if a.locked:
                        if _push_clear(b, a_tl, a_br):
                            b_tl, b_br = _effective_bbox(b, half_gap)
                            moved = True
                    elif b.locked:
                        if _push_clear(a, b_tl, b_br):
                            a_tl, a_br = _effective_bbox(a, half_gap)
                            moved = True
                    else:
                        # Both free: move the smaller part out of the larger,
                        # so a big IC/connector stays put and a passive yields.
                        if a.width_mm * a.height_mm <= b.width_mm * b.height_mm:
                            if _push_clear(a, b_tl, b_br):
                                a_tl, a_br = _effective_bbox(a, half_gap)
                                moved = True
                        else:
                            if _push_clear(b, a_tl, a_br):
                                b_tl, b_br = _effective_bbox(b, half_gap)
                                moved = True
            if not moved:
                break  # fully separated
        return unresolved

    def legality_diagnostics(self, comps: dict[str, Component]) -> dict[str, object]:
        tl, br = self.state.board_outline
        inset = self.cfg.get("pad_inset_margin_mm", 0.3)
        half_gap = self.clearance / 2.0
        pads_outside: list[dict[str, object]] = []
        overlaps: list[dict[str, object]] = []
        refs = list(comps.keys())
        for ref, comp in comps.items():
            for pad in comp.pads:
                violations: list[str] = []
                if pad.pos.x < tl.x + inset:
                    violations.append("left")
                if pad.pos.x > br.x - inset:
                    violations.append("right")
                if pad.pos.y < tl.y + inset:
                    violations.append("top")
                if pad.pos.y > br.y - inset:
                    violations.append("bottom")
                if violations:
                    pads_outside.append(
                        {
                            "ref": ref,
                            "pad_id": pad.pad_id,
                            "sides": violations,
                            "x_mm": round(pad.pos.x, 4),
                            "y_mm": round(pad.pos.y, 4),
                        }
                    )
        locked_overlap_count = 0
        for i in range(len(refs)):
            a = comps[refs[i]]
            a_tl, a_br = _effective_bbox(a, half_gap)
            for j in range(i + 1, len(refs)):
                b = comps[refs[j]]
                b_tl, b_br = _effective_bbox(b, half_gap)
                ox, oy = _bbox_overlap_xy(a_tl, a_br, b_tl, b_br)
                if ox > 0.0 and oy > 0.0:
                    # Intentional array-grid neighbours: their clearance-zone
                    # overlap is by design, not an illegality.
                    if a.array_member and b.array_member:
                        continue
                    involves_locked = a.locked or b.locked
                    if involves_locked:
                        locked_overlap_count += 1
                    overlaps.append(
                        {
                            "a": refs[i],
                            "b": refs[j],
                            "overlap_x_mm": round(ox, 4),
                            "overlap_y_mm": round(oy, 4),
                            "overlap_area_mm2": round(ox * oy, 4),
                            "involves_locked": involves_locked,
                        }
                    )
        # Antenna keep-out overlaps (RF near-field). Reported for visibility;
        # deliberately NOT folded into ``legal`` so this does not change solver
        # acceptance for boards that previously placed without keep-out modeling.
        keepout_overlaps: list[dict[str, object]] = []
        for kr in getattr(self.state, "keepout_rects", None) or []:
            r_tl, r_br = self._keepout_rect_now(kr, comps)
            for ref, comp in comps.items():
                if ref == kr.owner_ref:
                    continue
                c_tl, c_br = comp.bbox(half_gap)
                ox = min(c_br.x, r_br.x) - max(c_tl.x, r_tl.x)
                oy = min(c_br.y, r_br.y) - max(c_tl.y, r_tl.y)
                if ox > 0.0 and oy > 0.0:
                    keepout_overlaps.append(
                        {
                            "ref": ref,
                            "owner": kr.owner_ref,
                            "source": kr.source,
                            "overlap_area_mm2": round(ox * oy, 4),
                        }
                    )
        return {
            "pads_outside_board": pads_outside,
            "overlaps": overlaps,
            "pad_outside_count": len(pads_outside),
            "overlap_count": len(overlaps),
            "locked_overlap_count": locked_overlap_count,
            "keepout_overlaps": keepout_overlaps,
            "keepout_overlap_count": len(keepout_overlaps),
            "legal": not pads_outside and not overlaps,
        }

    def legalize_components(
        self, comps: dict[str, Component], *, max_passes: int = 12
    ) -> dict[str, object]:
        moved_refs: set[str] = set()
        if not hasattr(self, "_pinned_targets"):
            self._pinned_targets = {}
        best_snapshot = {ref: copy.deepcopy(comp) for ref, comp in comps.items()}
        best_diagnostics = self.legality_diagnostics(best_snapshot)

        def _diag_key(diag):
            locked = int(diag.get("locked_overlap_count", 0))
            free = int(diag["overlap_count"]) - locked
            pads = int(diag["pad_outside_count"])
            # Locked overlaps weigh 3x: they require escape pushes that
            # cascade into free-free overlaps, so regressing on them
            # is costlier than having temporary free-free overlaps.
            weighted = locked * 3 + free + pads
            return (weighted, locked, pads)

        def _move_component(comp, nx, ny):
            old_pos = Point(comp.pos.x, comp.pos.y)
            if abs(nx - old_pos.x) <= 0.01 and abs(ny - old_pos.y) <= 0.01:
                return False
            comp.pos.x = nx
            comp.pos.y = ny
            _update_pad_positions(comp, old_pos, comp.rotation)
            return True

        def _clamp_component_to_board(comp, nx, ny):
            tl, br = self.state.board_outline
            hw, hh = _pad_half_extents(comp)
            return (
                max(tl.x + hw + 1.0, min(br.x - hw - 1.0, nx)),
                max(tl.y + hh + 1.0, min(br.y - hh - 1.0, ny)),
            )

        def _keep_out_of_pinned_edge_connectors():
            zones = self.cfg.get("component_zones", {})
            half_gap = self.clearance / 2.0
            pinned_connectors = []
            for ref, comp in comps.items():
                zone_cfg = zones.get(ref, {})
                edge = zone_cfg.get("edge")
                if (
                    edge in {"left", "right", "top", "bottom"}
                    and comp.locked
                    and comp.kind == "connector"
                ):
                    keepout_tl, keepout_br = _effective_bbox(comp, half_gap)
                    pinned_connectors.append((ref, comp, edge, keepout_tl, keepout_br))
            if not pinned_connectors:
                return
            for ref, comp in comps.items():
                if comp.locked:
                    continue
                for _conn_ref, _conn, edge, keepout_tl, keepout_br in pinned_connectors:
                    comp_tl, comp_br = _effective_bbox(comp, half_gap)
                    ox, oy = _bbox_overlap_xy(keepout_tl, keepout_br, comp_tl, comp_br)
                    if ox <= 0.0 or oy <= 0.0:
                        continue
                    old_pos = Point(comp.pos.x, comp.pos.y)
                    candidates = []
                    if edge == "left":
                        candidates.append(
                            (keepout_br.x + (comp_br.x - comp.pos.x) + 0.1, comp.pos.y)
                        )
                    elif edge == "right":
                        candidates.append(
                            (keepout_tl.x - (comp.pos.x - comp_tl.x) - 0.1, comp.pos.y)
                        )
                    elif edge == "top":
                        candidates.append(
                            (comp.pos.x, keepout_br.y + (comp_br.y - comp.pos.y) + 0.1)
                        )
                    else:
                        candidates.append(
                            (comp.pos.x, keepout_tl.y - (comp.pos.y - comp_tl.y) - 0.1)
                        )
                    candidates.extend(
                        [
                            (comp.pos.x + ox + 0.1, comp.pos.y),
                            (comp.pos.x - ox - 0.1, comp.pos.y),
                            (comp.pos.x, comp.pos.y + oy + 0.1),
                            (comp.pos.x, comp.pos.y - oy - 0.1),
                        ]
                    )
                    best_key = None
                    best_move = (comp.pos.x, comp.pos.y)
                    for nx, ny in candidates:
                        nx, ny = _clamp_component_to_board(comp, nx, ny)
                        moved = _move_component(comp, nx, ny)
                        trial_tl, trial_br = _effective_bbox(comp, half_gap)
                        trial_ox, trial_oy = _bbox_overlap_xy(
                            keepout_tl, keepout_br, trial_tl, trial_br
                        )
                        still_overlapping = (
                            1 if trial_ox > 0.0 and trial_oy > 0.0 else 0
                        )
                        travel = old_pos.dist(Point(nx, ny))
                        key = (still_overlapping, travel)
                        if best_key is None or key < best_key:
                            best_key = key
                            best_move = (nx, ny)
                        if moved:
                            _move_component(comp, old_pos.x, old_pos.y)
                    _move_component(comp, best_move[0], best_move[1])

        for _ in range(max_passes):
            before = {ref: (comp.pos.x, comp.pos.y) for ref, comp in comps.items()}
            self._clamp_pads_to_board(comps)
            self._clamp_to_board(comps)
            self._resolve_overlaps(comps)
            self._clamp_to_board(comps)
            self._clamp_pads_to_board(comps)
            self._restore_pinned_positions(comps)
            _keep_out_of_pinned_edge_connectors()
            # Resolve cascading overlaps from connector keepout pushes
            self._resolve_overlaps(comps)
            self._clamp_to_board(comps)
            self._clamp_pads_to_board(comps)
            diagnostics = self.legality_diagnostics(comps)
            if _diag_key(diagnostics) < _diag_key(best_diagnostics):
                best_snapshot = {
                    ref: copy.deepcopy(comp) for ref, comp in comps.items()
                }
                best_diagnostics = diagnostics
            for ref, comp in comps.items():
                old_x, old_y = before[ref]
                if abs(comp.pos.x - old_x) > 0.01 or abs(comp.pos.y - old_y) > 0.01:
                    moved_refs.add(ref)
            if diagnostics["legal"]:
                return {
                    "resolved": True,
                    "passes": _ + 1,
                    "moved_refs": sorted(moved_refs),
                    "diagnostics": diagnostics,
                }
        for ref in list(comps.keys()):
            comps[ref] = copy.deepcopy(best_snapshot[ref])
        return {
            "resolved": best_diagnostics.get("legal", False),
            "passes": max_passes,
            "moved_refs": sorted(moved_refs),
            "diagnostics": best_diagnostics,
        }

    def _re_snap_aligned_pairs(self, comps: dict[str, Component]):
        """Re-snap aligned pairs to shared coordinate after pipeline steps.

        Steps like swap optimization, grid snap, orderedness, and overlap
        resolution can break the alignment set up by _align_large_pairs().
        Call this after any such step to restore side-by-side alignment.
        """
        if not self._aligned_pairs:
            return
        for ref_a, ref_b, axis in self._aligned_pairs:
            if ref_a not in comps or ref_b not in comps:
                continue
            a, b = comps[ref_a], comps[ref_b]
            if axis == "y":
                mid_y = (a.pos.y + b.pos.y) / 2
                old_a = Point(a.pos.x, a.pos.y)
                old_b = Point(b.pos.x, b.pos.y)
                a.pos.y = mid_y
                b.pos.y = mid_y
                _update_pad_positions(a, old_a, a.rotation)
                _update_pad_positions(b, old_b, b.rotation)
            else:
                mid_x = (a.pos.x + b.pos.x) / 2
                old_a = Point(a.pos.x, a.pos.y)
                old_b = Point(b.pos.x, b.pos.y)
                a.pos.x = mid_x
                b.pos.x = mid_x
                _update_pad_positions(a, old_a, a.rotation)
                _update_pad_positions(b, old_b, b.rotation)

    def _clamp_to_board(self, comps: dict[str, Component]):
        """Hard clamp: force every component's bounding box inside the board.

        Uses pad-aware half-extents so that components with pads extending
        beyond the body (e.g. battery holders) are clamped correctly.
        """
        tl, br = self.state.board_outline
        for comp in comps.values():
            if comp.locked:
                continue
            hw, hh = _pad_half_extents(comp)
            old_pos = Point(comp.pos.x, comp.pos.y)
            comp.pos.x = max(tl.x + hw + 1.0, min(br.x - hw - 1.0, comp.pos.x))
            comp.pos.y = max(tl.y + hh + 1.0, min(br.y - hh - 1.0, comp.pos.y))
            if comp.pos.x != old_pos.x or comp.pos.y != old_pos.y:
                _update_pad_positions(comp, old_pos, comp.rotation)

    def _assign_layers(self, comps: dict[str, Component]):
        """Assign large through-hole components to B.Cu (back layer).

        SMT components always stay on F.Cu.  Small THT passives (e.g. axial
        resistors) also stay on F.Cu.  Large THT parts (batteries,
        large connectors) go to back so they don't block SMT placement
        and routing on the front side.

        SMT passives stay on F.Cu even when their IC group contains a
        back-layer THT component — IC group connectivity forces keep them
        nearby in the same XY region, achieving dual-sided board usage.

        Edge-zoned connectors (and any part carrying an ``opening_direction``)
        are EXEMPT: they must define a board edge from the front. Flipping one
        to B.Cu mirrors its pad X *and* swaps left<->right in
        ``edge_outward_angle`` (see types.py), which inverts the opening the
        compose rotation filter solves for and strands the connector inboard of
        the edge it mates at (the run_07 USB-C signature: a >50mm² THT USB-C
        connector got auto-flipped, then composed inboard). A part the BOM
        explicitly placed on the back already arrives as ``Layer.BACK`` (set at
        load from ``component_layers``) and is untouched by the guard below.
        """
        min_area = self.cfg.get("tht_backside_min_area_mm2", 50.0)
        zones = self.cfg.get("component_zones", {}) or {}
        moved = []
        for ref, comp in comps.items():
            if comp.kind == "subcircuit":
                continue  # parent-side blocks own their internal layer assignment
            if not comp.is_through_hole:
                continue
            if comp.area < min_area:
                continue
            # Keep edge-mating connectors on the front (see docstring).
            _zone_edge = (zones.get(ref) or {}).get("edge")
            if (
                comp.kind == "connector"
                or comp.opening_direction is not None
                or _zone_edge in ("left", "right", "top", "bottom")
            ):
                continue
            if comp.layer != Layer.BACK:
                # Mirror to match how the stamp composes KiCad Flip() +
                # SetOrientationDegrees(): the net effect is a mirror of
                # the footprint's LOCAL Y axis (verified against pcbnew),
                # i.e. at flip-time rotation t0 the world reflection
                # R(t0)*M_y*R(-t0). At t0=0 that is a pure Y mirror about
                # pos -- NOT the X mirror formerly applied here, which put
                # 2-pad THT pad identities on the wrong sides. body_center
                # must mirror too, or the modeled courtyard stays on the
                # pre-flip side while the stamped one moves (B24).
                two_theta = math.radians(2.0 * comp.rotation)
                c2, s2 = math.cos(two_theta), math.sin(two_theta)

                def _reflect(p: Point) -> Point:
                    ox, oy = p.x - comp.pos.x, p.y - comp.pos.y
                    return Point(
                        comp.pos.x + ox * c2 - oy * s2,
                        comp.pos.y - ox * s2 - oy * c2,
                    )

                for pad in comp.pads:
                    pad.pos = _reflect(pad.pos)
                if comp.body_center is not None:
                    comp.body_center = _reflect(comp.body_center)
                comp.layer = Layer.BACK
                moved.append(ref)
        if moved:
            print(
                f"  Assigned {len(moved)} large THT component(s) to back layer: "
                f"{', '.join(moved)}"
            )

    def _align_large_pairs(self, comps: dict[str, Component]):
        """Detect and align pairs of large, similarly-sized components side-by-side.

        Finds components with same kind (not passive/misc), similar area
        (ratio > 0.85), and area above tht_backside_min_area_mm2.  Places
        them adjacent on a randomly chosen axis (horizontal or vertical),
        sharing one coordinate.  Respects zone constraints.

        Populates self._aligned_pairs for use by _force_step() to maintain
        alignment during force simulation.
        """
        if not self.cfg.get("align_large_pairs", True):
            return

        min_area = self.cfg.get("tht_backside_min_area_mm2", 50.0)
        zones = self.cfg.get("component_zones", {})
        tl, br = self.state.board_outline

        # Find candidates: large, non-passive, non-misc.
        # Subcircuit blocks are excluded -- alignment is a leaf-level concern;
        # parent-side blocks coordinate via attachment constraints instead.
        candidates = [
            (ref, comp)
            for ref, comp in comps.items()
            if comp.area >= min_area
            and comp.kind
            not in ("", "misc", "passive", "connector", "mounting_hole", "subcircuit")
        ]

        # Detect pairs: same kind, similar area
        paired = set()
        pairs = []
        for i, (ref_a, a) in enumerate(candidates):
            if ref_a in paired:
                continue
            for ref_b, b in candidates[i + 1 :]:
                if ref_b in paired:
                    continue
                if a.kind != b.kind:
                    continue
                ratio = min(a.area, b.area) / max(a.area, b.area)
                if ratio < 0.85:
                    continue
                pairs.append((ref_a, ref_b))
                paired.add(ref_a)
                paired.add(ref_b)
                break  # one partner per component

        if not pairs:
            return

        gap = 1.5  # mm gap between paired components

        for ref_a, ref_b in pairs:
            a, b = comps[ref_a], comps[ref_b]

            # Choose axis based on component shape: place along the longer
            # dimension to minimize total footprint width
            if max(a.width_mm, b.width_mm) >= max(a.height_mm, b.height_mm):
                axis = "y"  # side-by-side horizontally (share Y)
            else:
                axis = "x"  # stacked vertically (share X)

            # Compute zone bounds for clamping (use first component's zone)
            zone_a = zones.get(ref_a, {})
            zone_b = zones.get(ref_b, {})
            zone_name = zone_a.get("zone") or zone_b.get("zone")
            if zone_name:
                zx0, zy0, zx1, zy1 = self._get_zone_bounds(zone_name)
            else:
                margin = self.edge_margin
                zx0, zy0 = tl.x + margin, tl.y + margin
                zx1, zy1 = br.x - margin, br.y - margin

            old_a = Point(a.pos.x, a.pos.y)
            old_b = Point(b.pos.x, b.pos.y)

            if axis == "y":
                # Horizontal side-by-side: same Y, adjacent X
                mid_y = (a.pos.y + b.pos.y) / 2
                total_w = a.width_mm + b.width_mm + gap
                # Center the pair in their zone on X
                pair_cx = self.rng.uniform(
                    zx0 + total_w / 2,
                    max(zx0 + total_w / 2 + 1, zx1 - total_w / 2),
                )
                a.pos.x = pair_cx - (b.width_mm + gap) / 2
                b.pos.x = pair_cx + (a.width_mm + gap) / 2
                # Clamp Y to zone
                mid_y = max(
                    zy0 + max(a.height_mm, b.height_mm) / 2,
                    min(zy1 - max(a.height_mm, b.height_mm) / 2, mid_y),
                )
                a.pos.y = mid_y
                b.pos.y = mid_y
            else:
                # Vertical stack: same X, adjacent Y
                mid_x = (a.pos.x + b.pos.x) / 2
                total_h = a.height_mm + b.height_mm + gap
                pair_cy = self.rng.uniform(
                    zy0 + total_h / 2,
                    max(zy0 + total_h / 2 + 1, zy1 - total_h / 2),
                )
                a.pos.y = pair_cy - (b.height_mm + gap) / 2
                b.pos.y = pair_cy + (a.height_mm + gap) / 2
                mid_x = max(
                    zx0 + max(a.width_mm, b.width_mm) / 2,
                    min(zx1 - max(a.width_mm, b.width_mm) / 2, mid_x),
                )
                a.pos.x = mid_x
                b.pos.x = mid_x

            _update_pad_positions(a, old_a, a.rotation)
            _update_pad_positions(b, old_b, b.rotation)

            self._aligned_pairs.append((ref_a, ref_b, axis))

        if pairs:
            print(
                f"  Aligned {len(pairs)} large pair(s) side-by-side: "
                f"{', '.join(f'{a}+{b}' for a, b in pairs)}"
            )

    def _clamp_pads_to_board(self, comps: dict[str, Component]):
        """Hard clamp: shift components inward so all pads are inside the board."""
        tl, br = self.state.board_outline
        inset = self.cfg.get("pad_inset_margin_mm", 0.3)
        min_x = tl.x + inset
        min_y = tl.y + inset
        max_x = br.x - inset
        max_y = br.y - inset

        for comp in comps.values():
            if not comp.pads:
                continue
            if comp.locked:
                continue

            # Track left/right and top/bottom violations separately
            shift_left = 0.0  # positive = need to move right
            shift_right = 0.0  # negative = need to move left
            shift_up = 0.0  # positive = need to move down
            shift_down = 0.0  # negative = need to move up
            for pad in comp.pads:
                if pad.pos.x < min_x:
                    shift_left = max(shift_left, min_x - pad.pos.x)
                if pad.pos.x > max_x:
                    shift_right = min(shift_right, max_x - pad.pos.x)
                if pad.pos.y < min_y:
                    shift_up = max(shift_up, min_y - pad.pos.y)
                if pad.pos.y > max_y:
                    shift_down = min(shift_down, max_y - pad.pos.y)

            # Use the larger magnitude violation for each axis
            shift_x = shift_left if abs(shift_left) >= abs(shift_right) else shift_right
            shift_y = shift_up if abs(shift_up) >= abs(shift_down) else shift_down

            if abs(shift_x) > 0.001 or abs(shift_y) > 0.001:
                old_pos = Point(comp.pos.x, comp.pos.y)
                comp.pos.x += shift_x
                comp.pos.y += shift_y
                _update_pad_positions(comp, old_pos, comp.rotation)

    def _snap_to_grid(self, comps: dict[str, Component]):
        """Snap all unlocked components to placement grid."""
        g = self.grid_snap
        for comp in comps.values():
            if comp.locked:
                continue
            old_pos = Point(comp.pos.x, comp.pos.y)
            comp.pos.x = round(comp.pos.x / g) * g
            comp.pos.y = round(comp.pos.y / g) * g
            _update_pad_positions(comp, old_pos, comp.rotation)

    def _apply_orderedness(self, comps: dict[str, Component], strength: float):
        """Align passives into neat rows/columns near their IC group leader.

        strength: 0.0 = no effect (organic), 1.0 = full grid alignment.
        Intermediate values blend between organic position and grid position.

        Groups passives by IC group, sorts them by size class, and arranges
        each size class into rows. Components not in any IC group are grouped
        by spatial proximity.
        """
        ic_groups = self.cfg.get("ic_groups", {})
        grid = self.grid_snap

        # Build map: ref -> group leader
        ref_to_leader: dict[str, str] = {}
        for leader, members in ic_groups.items():
            ref_to_leader[leader] = leader
            for m in members:
                ref_to_leader[m] = leader

        # Collect passives by group leader
        grouped: dict[str, list[str]] = {}
        ungrouped: list[str] = []
        for ref, comp in comps.items():
            if comp.locked or comp.kind not in ("passive",):
                continue
            leader = ref_to_leader.get(ref)
            if leader and leader in comps:
                grouped.setdefault(leader, []).append(ref)
            else:
                ungrouped.append(ref)

        # Cluster ungrouped passives by proximity (simple greedy clustering)
        if ungrouped:
            remaining = set(ungrouped)
            cluster_radius = 20.0  # mm
            while remaining:
                seed = remaining.pop()
                cluster = [seed]
                for ref in list(remaining):
                    if comps[ref].pos.dist(comps[seed].pos) < cluster_radius:
                        cluster.append(ref)
                        remaining.discard(ref)
                if len(cluster) >= 2:
                    # Use first component as virtual "leader"
                    grouped[cluster[0]] = cluster

        total_aligned = 0
        for leader, members in grouped.items():
            if len(members) < 2:
                continue

            # Find anchor position: IC leader center or centroid of group
            if leader in comps and leader not in members:
                anchor = comps[leader].pos
            else:
                anchor = Point(
                    sum(comps[r].pos.x for r in members) / len(members),
                    sum(comps[r].pos.y for r in members) / len(members),
                )

            # Bin passives by size class (similar dimensions → same row)
            size_bins: dict[tuple[float, float], list[str]] = {}
            for ref in members:
                c = comps[ref]
                # Round dimensions to nearest 0.5mm for binning
                w_key = round(min(c.width_mm, c.height_mm) * 2) / 2
                h_key = round(max(c.width_mm, c.height_mm) * 2) / 2
                size_bins.setdefault((w_key, h_key), []).append(ref)

            # Arrange each size bin as a row
            row_y_offset = 0.0
            for (w_key, h_key), bin_refs in size_bins.items():
                if not bin_refs:
                    continue
                bin_refs.sort(key=lambda r: comps[r].pos.x)  # left-to-right

                # Determine row direction: horizontal if wider spread, else vertical
                xs = [comps[r].pos.x for r in bin_refs]
                ys = [comps[r].pos.y for r in bin_refs]
                x_spread = max(xs) - min(xs)
                y_spread = max(ys) - min(ys)
                horizontal = x_spread >= y_spread

                # Compute grid-aligned target positions
                sample = comps[bin_refs[0]]
                gap = max(sample.width_mm, sample.height_mm) + self.clearance

                if horizontal:
                    # Row: same Y, evenly spaced X
                    row_cx = sum(xs) / len(xs)
                    row_cy = anchor.y + row_y_offset
                    targets = []
                    start_x = row_cx - (len(bin_refs) - 1) * gap / 2
                    for k, ref in enumerate(bin_refs):
                        tx = round((start_x + k * gap) / grid) * grid
                        ty = round(row_cy / grid) * grid
                        targets.append((ref, tx, ty))
                    row_y_offset += h_key + self.clearance
                else:
                    # Column: same X, evenly spaced Y
                    bin_refs.sort(key=lambda r: comps[r].pos.y)
                    row_cx = anchor.x + row_y_offset
                    row_cy = sum(ys) / len(ys)
                    targets = []
                    start_y = row_cy - (len(bin_refs) - 1) * gap / 2
                    for k, ref in enumerate(bin_refs):
                        tx = round(row_cx / grid) * grid
                        ty = round((start_y + k * gap) / grid) * grid
                        targets.append((ref, tx, ty))
                    row_y_offset += w_key + self.clearance

                # Blend between organic position and grid target
                for ref, tx, ty in targets:
                    comp = comps[ref]
                    old_pos = Point(comp.pos.x, comp.pos.y)
                    comp.pos.x = comp.pos.x + (tx - comp.pos.x) * strength
                    comp.pos.y = comp.pos.y + (ty - comp.pos.y) * strength
                    _update_pad_positions(comp, old_pos, comp.rotation)
                    total_aligned += 1

        if total_aligned > 0:
            print(f"  Orderedness ({strength:.0%}): aligned {total_aligned} passives")
