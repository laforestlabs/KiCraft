"""PlacementScorer -- scores a placement configuration to guide optimization.

Extracted from placement.py for modularity.  Import from
``placement`` (the re-export hub) for backward compatibility,
or directly from this module in new code.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .graph import count_crossings, total_ratsnest_length
from .placement_utils import _blocker_pair_compatible
from .types import BoardState, Layer, PlacementScore, Point

class PlacementScorer:
    """Scores a placement configuration to guide optimization.

    Evaluates: net distance, crossover count, compactness, edge compliance,
    rotation quality. All computation is local.
    """

    def __init__(self, state: BoardState, config: dict = None):
        self.state = state
        self.cfg = config or {}

    def score(self) -> PlacementScore:
        s = PlacementScore()
        s.net_distance = self._score_net_distance()
        s.crossover_count = count_crossings(self.state)
        s.crossover_score = self._crossover_to_score(s.crossover_count)
        s.compactness = self._score_compactness()
        s.edge_compliance = self._score_edge_compliance()
        s.rotation_score = self._score_rotation()
        s.board_containment = self._score_board_containment()
        s.courtyard_overlap = self._score_courtyard_overlap()
        s.smt_opposite_tht = self._score_smt_opposite_tht()
        s.group_coherence = self._score_group_coherence()
        s.topology_structure = self._score_topology_structure()
        s.block_opposite_side = self._score_block_opposite_side()

        # Board aspect ratio scoring
        board_w = self.state.board_width
        board_h = self.state.board_height
        if board_w > 0 and board_h > 0:
            ratio = max(board_w, board_h) / min(board_w, board_h)
            # Score: 100 for 1:1, 80 for 1.5:1, 50 for 2:1, 0 for 3:1+
            s.aspect_ratio = max(0.0, 100.0 * (1.0 - (ratio - 1.0) / 2.0))

        s.compute_total()
        return s

    def _score_net_distance(self) -> float:
        """Score based on total MST ratsnest length.
        Shorter = better. Normalized to 0-100."""
        total_len = total_ratsnest_length(self.state)
        # Heuristic: board diagonal is worst case per net
        diag = math.hypot(self.state.board_width, self.state.board_height)
        n_nets = max(
            1,
            len(
                [
                    n
                    for n in self.state.nets.values()
                    if len(n.pad_refs) >= 2 and n.name not in ("GND", "/GND")
                ]
            ),
        )
        worst_case = diag * n_nets
        if worst_case == 0:
            return 100.0
        ratio = total_len / worst_case
        return max(0, min(100, (1.0 - ratio) * 100))

    def _crossover_to_score(self, crossings: int) -> float:
        """Convert crossing count to 0-100 score. Fewer = better."""
        n_nets = max(1, len(self.state.nets))
        # Max expected crossings ~ n_nets^2 / 4 for random placement
        max_expected = n_nets * n_nets / 4
        if max_expected == 0:
            return 100.0
        ratio = crossings / max_expected
        return max(0, min(100, (1.0 - ratio) * 100))

    def _score_compactness(self) -> float:
        """Ratio of component area to board area. Gentle reward for smaller layouts.
        20% fill = 50, 40% fill = 75, 60%+ = 100. Not heavily penalized."""
        total_area = sum(c.area for c in self.state.components.values())
        board_area = self.state.board_width * self.state.board_height
        if board_area == 0:
            return 0.0
        fill = total_area / board_area
        # Gentle curve: 10% fill ≈ 40, 30% ≈ 65, 50%+ ≈ 90+
        return min(100, fill * 150 + 25)

    def _score_edge_compliance(self) -> float:
        """Check connectors and mounting holes are near board edges.

        Uses the placement edge_margin from config (default 6mm) plus a
        tolerance buffer, so components placed at the edge margin are
        correctly recognised as edge-compliant.
        """
        tl, br = self.state.board_outline
        # Match the placement edge margin so pinned components always score
        margin = self.cfg.get("edge_margin_mm", 6.0) + 2.0
        total = 0
        compliant = 0
        for comp in self.state.components.values():
            if comp.kind not in ("connector", "mounting_hole"):
                continue
            total += 1
            x, y = comp.pos.x, comp.pos.y
            near_edge = (
                x - tl.x <= margin
                or br.x - x <= margin
                or y - tl.y <= margin
                or br.y - y <= margin
            )
            if near_edge:
                compliant += 1
        if total == 0:
            return 100.0
        return (compliant / total) * 100

    def _score_rotation(self) -> float:
        """Score component rotations.
        Passives should be at 0 or 90 degrees.
        ICs should minimize net-crossing angles.
        """
        total = 0
        good = 0
        for comp in self.state.components.values():
            if comp.kind in ("passive",):
                total += 1
                r = comp.rotation % 360
                if r in (0, 90, 180, 270):
                    good += 1
                elif r % 45 == 0:
                    good += 0.5
            elif comp.kind == "ic":
                total += 1
                r = comp.rotation % 360
                if r in (0, 90, 180, 270):
                    good += 1
        if total == 0:
            return 100.0
        return (good / total) * 100

    def _score_board_containment(self) -> float:
        """Score how well components and pads stay within the board outline.

        Uses pad_inset_margin_mm to enforce that pads are inset from the
        board edge, not merely inside it.
        """
        tl, br = self.state.board_outline
        inset = self.cfg.get("pad_inset_margin_mm", 0.3)

        total_pads = 0
        pads_outside = 0
        total_bodies = 0
        bodies_outside = 0

        for comp in self.state.components.values():
            total_bodies += 1
            c_tl, c_br = comp.bbox()
            if c_tl.x < tl.x or c_br.x > br.x or c_tl.y < tl.y or c_br.y > br.y:
                bodies_outside += 1

            for pad in comp.pads:
                total_pads += 1
                # Check the actual pad copper extent, not just the center.
                # A pad whose center is just inside the inset but whose copper
                # crosses the inset is just as fab-illegal as one whose center
                # is outside.
                pad_tl, pad_br = pad.bbox()
                if (
                    pad_tl.x < tl.x + inset
                    or pad_br.x > br.x - inset
                    or pad_tl.y < tl.y + inset
                    or pad_br.y > br.y - inset
                ):
                    pads_outside += 1

        if total_pads == 0 and total_bodies == 0:
            return 100.0

        pad_frac = pads_outside / max(1, total_pads)
        body_frac = bodies_outside / max(1, total_bodies)
        # Weighted: 80% pad containment, 20% body containment
        score = 100.0 * (1.0 - 0.8 * pad_frac - 0.2 * body_frac)
        return max(0.0, min(100.0, score))

    def _score_courtyard_overlap(self) -> float:
        """Penalize overlapping component courtyards using area-proportional scoring.

        Instead of a fixed penalty per overlap pair (which creates a cliff
        at high overlap counts), this measures the total overlap area as a
        fraction of total courtyard area.  Provides a smooth gradient so
        partial improvements are always rewarded."""
        comps = list(self.state.components.values())
        base_clearance = 0.25  # mm courtyard margin
        padding = self.cfg.get("courtyard_padding_mm", 0.0)
        clearance = base_clearance + padding
        n = len(comps)

        total_courtyard_area = 0.0
        total_overlap_area = 0.0

        for i in range(n):
            a = comps[i]
            a_tl, a_br = a.bbox(clearance)
            total_courtyard_area += (a_br.x - a_tl.x) * (a_br.y - a_tl.y)
            for j in range(i + 1, n):
                b = comps[j]
                b_tl, b_br = b.bbox(clearance)
                # Compute overlap rectangle
                ox = max(0.0, min(a_br.x, b_br.x) - max(a_tl.x, b_tl.x))
                oy = max(0.0, min(a_br.y, b_br.y) - max(a_tl.y, b_tl.y))
                if ox > 0.0 and oy > 0.0 and _blocker_pair_compatible(a, b):
                    continue
                total_overlap_area += ox * oy

        if total_courtyard_area <= 0:
            return 100.0
        # Overlap ratio: 0 = no overlaps, 1 = total overlap equals total courtyard
        overlap_ratio = total_overlap_area / total_courtyard_area
        # Smooth penalty: ratio of 0.1 (10% overlap) → score ~70
        #                  ratio of 0.3 (30% overlap) → score ~30
        #                  ratio of 0.0 → score 100
        return max(0.0, min(100.0, 100.0 * (1.0 - overlap_ratio * 3.0)))

    def _score_block_opposite_side(self) -> float:
        """Reward bbox overlap between blocker-compatible block pairs.

        On the parent-side path, synthetic blocks carry block_blocker_set
        and the courtyard scorer waives the overlap penalty for
        compatible pairs. That alone is permissive (overlap doesn't
        hurt) but not active (overlap doesn't help). This term *rewards*
        the overlap so SA refinement actively packs front-side SMT
        blocks on top of back-side THT blocks (e.g. SMT regulators on
        the back-side battery footprint).

        Returns 0 when no compatible pairs exist (leaf placement) so the
        weighted contribution to the total score stays neutral. Returns
        100 when every compatible pair has its smaller bbox fully
        contained in the larger; scales linearly with overlap fraction.
        """
        comps = list(self.state.components.values())
        n = len(comps)
        if n < 2:
            return 0.0
        compatible_total = 0.0
        compatible_overlap = 0.0
        for i in range(n):
            a = comps[i]
            if a.block_blocker_set is None:
                continue
            a_tl, a_br = a.bbox(0.0)
            a_area = max(0.0, (a_br.x - a_tl.x) * (a_br.y - a_tl.y))
            for j in range(i + 1, n):
                b = comps[j]
                if b.block_blocker_set is None:
                    continue
                if not _blocker_pair_compatible(a, b):
                    continue
                b_tl, b_br = b.bbox(0.0)
                b_area = max(0.0, (b_br.x - b_tl.x) * (b_br.y - b_tl.y))
                small_area = min(a_area, b_area)
                if small_area <= 0.0:
                    continue
                ox = max(0.0, min(a_br.x, b_br.x) - max(a_tl.x, b_tl.x))
                oy = max(0.0, min(a_br.y, b_br.y) - max(a_tl.y, b_tl.y))
                overlap = ox * oy
                compatible_total += small_area
                compatible_overlap += min(small_area, overlap)
        if compatible_total <= 0.0:
            return 0.0
        return max(0.0, min(100.0, 100.0 * compatible_overlap / compatible_total))

    def _score_smt_opposite_tht(self) -> float:
        """Bonus for SMT components placed in the XY shadow of back-side THT parts.

        Measures the fraction of front-side SMT component area that overlaps
        (in XY projection) with back-side THT bounding boxes.  Higher overlap
        means better board space utilization.  Returns 100 if all SMT sits
        over backside THT (full shadow utilisation), 0 when no overlap --
        this steep linear curve penalises leaves that ignore the available
        backside shadow even though the sparse blocker model permits it.
        Returns 100 when the feature is disabled, no backside THT exists,
        or there is no front-side SMT (nothing to score).
        """
        if not self.cfg.get("smt_opposite_tht", True):
            return 100.0  # feature disabled — don't penalize

        min_area = self.cfg.get("tht_backside_min_area_mm2", 50.0)
        back_tht = [
            c
            for c in self.state.components.values()
            if c.is_through_hole and c.layer == Layer.BACK and c.area >= min_area
        ]
        if not back_tht:
            return 100.0  # no back-side THT — nothing to optimize

        front_smt = [
            c
            for c in self.state.components.values()
            if not c.is_through_hole and c.layer != Layer.BACK and not c.locked
        ]
        if not front_smt:
            return 100.0

        total_smt_area = sum(c.area for c in front_smt)
        if total_smt_area <= 0:
            return 100.0

        overlap_area = 0.0
        for smt in front_smt:
            s_tl, s_br = smt.bbox()
            for tht in back_tht:
                t_tl, t_br = tht.bbox()
                ox = max(0.0, min(s_br.x, t_br.x) - max(s_tl.x, t_tl.x))
                oy = max(0.0, min(s_br.y, t_br.y) - max(s_tl.y, t_tl.y))
                overlap_area += ox * oy

        overlap_frac = min(1.0, overlap_area / total_smt_area)
        # 0% overlap → 0, 50% → 50, 100% → 100
        return max(0.0, 100.0 * overlap_frac)

    def _score_group_coherence(self) -> float:
        """Score how compact functional groups are.

        For each IC group defined in config, measures the average distance
        of group members from their centroid.  Normalized against the board
        diagonal so the score is resolution-independent.

        Returns 100 if all groups are perfectly compact, 0 if members are
        scattered across the full board diagonal.  Returns 100 if no groups
        are defined (no penalty for projects without groups).
        """
        ic_groups = self.cfg.get("ic_groups", {})
        if not ic_groups:
            return 100.0

        board_diag = math.hypot(self.state.board_width, self.state.board_height)
        if board_diag < 1.0:
            return 100.0

        total_score = 0.0
        n_groups = 0

        for leader, members in ic_groups.items():
            all_refs = [leader] + list(members)
            # Get positions of group members that exist on the board
            positions = []
            for ref in all_refs:
                comp = self.state.components.get(ref)
                if comp:
                    positions.append(comp.pos)
            if len(positions) < 2:
                continue

            # Centroid
            cx = sum(p.x for p in positions) / len(positions)
            cy = sum(p.y for p in positions) / len(positions)
            centroid = Point(cx, cy)

            # Average distance from centroid
            avg_dist = sum(centroid.dist(p) for p in positions) / len(positions)

            # Normalize: 0 distance = 100 score, board_diagonal/4 distance = 0 score
            # Groups should be within ~10-15% of board diagonal for a good score
            group_score = max(0.0, 100.0 * (1.0 - avg_dist / (board_diag * 0.25)))
            total_score += group_score
            n_groups += 1

        if n_groups == 0:
            return 100.0
        return total_score / n_groups

    def _score_topology_structure(self) -> float:
        """Score whether passive components form topology-aware chains around anchors.

        This is intentionally generic and only uses component/net connectivity:
        - anchors are ICs, regulators, and connectors
        - passives are rewarded for staying close to their strongest anchor
        - passive-passive shared-net adjacency is rewarded when arranged as
          ordered local chains instead of scattered clouds

        Returns 100 when no meaningful topology can be inferred, so projects
        without passive-chain structure are not penalized.
        """
        anchors = {
            ref: comp
            for ref, comp in self.state.components.items()
            if comp.kind in ("ic", "regulator", "connector")
        }
        passives = {
            ref: comp
            for ref, comp in self.state.components.items()
            if comp.kind == "passive" and not comp.locked
        }

        if not anchors or len(passives) < 2:
            return 100.0

        nets_by_ref: dict[str, set[str]] = defaultdict(set)
        adjacency: dict[str, dict[str, float]] = defaultdict(dict)

        for net in self.state.nets.values():
            if net.name in ("GND", "/GND"):
                continue
            refs = sorted(
                {ref for ref, _ in net.pad_refs if ref in self.state.components}
            )
            if len(refs) < 2:
                continue
            weight = 3.0 if net.is_power else 1.0
            for ref in refs:
                nets_by_ref[ref].add(net.name)
            for i, ref_a in enumerate(refs):
                for ref_b in refs[i + 1 :]:
                    adjacency[ref_a][ref_b] = adjacency[ref_a].get(ref_b, 0.0) + weight
                    adjacency[ref_b][ref_a] = adjacency[ref_b].get(ref_a, 0.0) + weight

        anchor_assignments: dict[str, str] = {}
        for passive_ref in passives:
            passive_nets = nets_by_ref.get(passive_ref, set())
            best_anchor = None
            best_key = None
            for anchor_ref, anchor_comp in anchors.items():
                shared_nets = len(passive_nets & nets_by_ref.get(anchor_ref, set()))
                edge_weight = adjacency.get(passive_ref, {}).get(anchor_ref, 0.0)
                key = (shared_nets, edge_weight, anchor_comp.area)
                if best_key is None or key > best_key:
                    best_key = key
                    best_anchor = anchor_ref
            if (
                best_anchor is not None
                and best_key is not None
                and (best_key[0] > 0 or best_key[1] > 0)
            ):
                anchor_assignments[passive_ref] = best_anchor

        if not anchor_assignments:
            return 100.0

        board_diag = math.hypot(self.state.board_width, self.state.board_height)
        if board_diag < 1.0:
            return 100.0

        anchor_scores: list[float] = []
        grouped_passives: dict[str, list[str]] = defaultdict(list)
        for passive_ref, anchor_ref in anchor_assignments.items():
            grouped_passives[anchor_ref].append(passive_ref)

        for anchor_ref, passive_refs in grouped_passives.items():
            anchor_comp = anchors.get(anchor_ref)
            if anchor_comp is None or not passive_refs:
                continue

            anchor_pos = anchor_comp.pos
            anchor_distances = [
                anchor_pos.dist(passives[ref].pos)
                for ref in passive_refs
                if ref in passives
            ]
            if not anchor_distances:
                continue

            avg_anchor_dist = sum(anchor_distances) / len(anchor_distances)
            anchor_compactness = max(
                0.0, 100.0 * (1.0 - avg_anchor_dist / (board_diag * 0.22))
            )

            chain_scores: list[float] = []
            for passive_ref in passive_refs:
                neighbors = [
                    other_ref
                    for other_ref in passive_refs
                    if other_ref != passive_ref
                    and adjacency.get(passive_ref, {}).get(other_ref, 0.0) > 0.0
                ]
                if not neighbors:
                    continue

                comp = passives[passive_ref]
                nearest_neighbor_dist = min(
                    comp.pos.dist(passives[other_ref].pos) for other_ref in neighbors
                )
                strongest_neighbor = max(
                    neighbors,
                    key=lambda other_ref: (
                        adjacency.get(passive_ref, {}).get(other_ref, 0.0),
                        -comp.pos.dist(passives[other_ref].pos),
                    ),
                )
                strongest_dist = comp.pos.dist(passives[strongest_neighbor].pos)

                local_chain_score = max(
                    0.0,
                    100.0
                    * (
                        1.0
                        - (0.65 * strongest_dist + 0.35 * nearest_neighbor_dist)
                        / (board_diag * 0.18)
                    ),
                )
                chain_scores.append(local_chain_score)

            if chain_scores:
                anchor_scores.append(
                    0.55 * anchor_compactness
                    + 0.45 * (sum(chain_scores) / len(chain_scores))
                )
            else:
                anchor_scores.append(anchor_compactness)

        if not anchor_scores:
            return 100.0
        return sum(anchor_scores) / len(anchor_scores)
