"""Leaf placement *tidiness* metrics — pure measurement, no placement change.

Phase 0 of the placement-pipeline streamline (see
``docs/plans/placement-pipeline-streamline.md``). These metrics quantify how
"orderly vs random" a placed leaf looks, so later phases have a before/after
number instead of an eyeball verdict. Three sub-metrics, all computed from a
placed leaf's components alone (positions, rotations, and pad nets):

* **orientation consensus %** — do passives that belong together point the same
  way? For each functional group (passives sharing their strongest anchor IC/
  connector) we take the group's dominant axis (horizontal vs vertical) and
  score the fraction of members that match it. Random 0°/90° disagreement — the
  thing that reads as "messy" — drives this down.
* **alignment residual (mm)** — within a group arranged as a row (or column),
  how far off the shared axis do members sit? 0 mm = a perfectly straight row.
* **packing fill %** — total component courtyard area / area of the bounding box
  enclosing the placement. Higher = tighter packing, less wasted board.

The grouping here (``functional_passive_groups``) is an independent, dependency-
light reimplementation of the same anchor-assignment idea used by
``leaf_passive_ordering.build_leaf_passive_topology_groups``. Phase 1 (Stage 1
of the plan) unifies the two into one canonical ``assign_passive_groups``; until
then this stays self-contained so the metric never perturbs the live solve path.

Everything operates on a lightweight :class:`PlacedPart` view so the same code
measures both live ``Component`` objects (during solve) and frozen
``solved_layout.json`` files (for the corpus baseline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# Anchor kinds a passive can be grouped under (a passive's "functional home").
_ANCHOR_KINDS = ("ic", "regulator", "connector")

# Nets that are POURED (a copper plane), not routed point-to-point. A passive
# pad on one of these reaches the plane through a via wherever it lands, so
# pin-locality treats it as via-reachable (distance 0) rather than pulling the
# pad toward the nearest plane *pad*. The scorer overrides this from the live
# pour config (GND + any poured power rail); this default matches the metric's
# standalone use over a frozen ``solved_layout.json``.
DEFAULT_PLANE_NETS = frozenset({"GND", "/GND"})


@dataclass(slots=True)
class PlacedPart:
    """Minimal placed-component view the tidiness metrics need."""

    ref: str
    kind: str
    locked: bool
    rotation: float
    cx: float  # body-center x (mm), world coords
    cy: float  # body-center y (mm), world coords
    w: float  # world-AABB width at current rotation (mm)
    h: float  # world-AABB height at current rotation (mm)
    nets: tuple[str, ...]  # distinct nets this part connects, via its pads
    # Per-pad geometry the *pin-locality* metric needs: each entry is
    # (pad_x, pad_y, net) in world mm. Defaulted so the tidiness metrics and
    # their synthetic tests -- which only need ``nets`` -- construct unchanged.
    pads: tuple[tuple[float, float, str], ...] = ()


@dataclass(slots=True)
class PassiveGroup:
    """Passives that share one anchor (their "functional row")."""

    anchor_ref: str
    passive_refs: tuple[str, ...]


@dataclass(slots=True)
class LeafTidiness:
    """Per-leaf tidiness metrics. ``None`` where undefined (too few parts)."""

    n_components: int
    n_passives: int
    n_groups: int
    grouped_passives: int
    orientation_consensus_grouped_pct: Optional[float]
    orientation_consensus_leaf_pct: Optional[float]
    alignment_residual_mm: Optional[float]
    packing_fill_pct: Optional[float]
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "n_passives": self.n_passives,
            "n_groups": self.n_groups,
            "grouped_passives": self.grouped_passives,
            "orientation_consensus_grouped_pct": _round(
                self.orientation_consensus_grouped_pct
            ),
            "orientation_consensus_leaf_pct": _round(
                self.orientation_consensus_leaf_pct
            ),
            "alignment_residual_mm": _round(self.alignment_residual_mm, 3),
            "packing_fill_pct": _round(self.packing_fill_pct),
            "label": self.label,
        }


@dataclass(slots=True)
class LeafPinLocality:
    """Per-leaf *pin-locality*: do passives hug the anchor pins they connect to?

    The metric the placement redesign optimizes for (and is judged on). A
    decoupling cap exists to sit ~1-2 mm from its IC's power/ground pins; the
    shipped soft-tidiness layouts put them 6-20 mm away (tidy rows, wrong
    place). ``None`` where undefined (no anchors, no scorable passives).
    """

    n_passives: int
    n_scored: int
    pin_locality_pct: Optional[float]  # 0-100, higher = passives hug their pins
    mean_worst_pad_dist_mm: Optional[float]  # honest mm, mean over passives
    worst_pad_dist_mm: Optional[float]  # max over passives (surfaces outliers)
    orientation_span_pct: Optional[float]  # pad-axis aligned to pin-axis, 0-100
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_passives": self.n_passives,
            "n_scored": self.n_scored,
            "pin_locality_pct": _round(self.pin_locality_pct),
            "mean_worst_pad_dist_mm": _round(self.mean_worst_pad_dist_mm, 3),
            "worst_pad_dist_mm": _round(self.worst_pad_dist_mm, 3),
            "orientation_span_pct": _round(self.orientation_span_pct),
            "label": self.label,
        }


def _round(v: Optional[float], ndigits: int = 1) -> Optional[float]:
    return None if v is None else round(v, ndigits)


# --------------------------------------------------------------------------- #
# Adapters — build PlacedPart views from either representation.
# --------------------------------------------------------------------------- #


def parts_from_components(components: dict[str, Any]) -> list[PlacedPart]:
    """Build views from live ``Component`` objects (has ``.pos``/``.pads``)."""
    parts: list[PlacedPart] = []
    for ref, c in components.items():
        center = c.body_center if getattr(c, "body_center", None) is not None else c.pos
        nets = tuple(sorted({p.net for p in getattr(c, "pads", []) if p.net}))
        pads = tuple(
            (float(p.pos.x), float(p.pos.y), str(p.net))
            for p in getattr(c, "pads", [])
            if p.net
        )
        parts.append(
            PlacedPart(
                ref=str(ref),
                kind=str(getattr(c, "kind", "") or ""),
                locked=bool(getattr(c, "locked", False)),
                rotation=float(getattr(c, "rotation", 0.0) or 0.0),
                cx=float(center.x),
                cy=float(center.y),
                w=float(getattr(c, "width_mm", 0.0) or 0.0),
                h=float(getattr(c, "height_mm", 0.0) or 0.0),
                nets=nets,
                pads=pads,
            )
        )
    return parts


def parts_from_layout(layout: dict[str, Any]) -> list[PlacedPart]:
    """Build views from a parsed ``solved_layout.json`` dict."""
    raw = layout.get("components", []) or []
    if isinstance(raw, dict):
        raw = list(raw.values())
    parts: list[PlacedPart] = []
    for c in raw:
        pos = c.get("body_center") or c.get("pos") or {"x": 0.0, "y": 0.0}
        nets = tuple(
            sorted({p.get("net") for p in c.get("pads", []) or [] if p.get("net")})
        )
        pads = tuple(
            (
                float((p.get("pos") or {}).get("x", 0.0)),
                float((p.get("pos") or {}).get("y", 0.0)),
                str(p.get("net")),
            )
            for p in c.get("pads", []) or []
            if p.get("net")
        )
        parts.append(
            PlacedPart(
                ref=str(c.get("ref", "")),
                kind=str(c.get("kind", "") or ""),
                locked=bool(c.get("locked", False)),
                rotation=float(c.get("rotation", 0.0) or 0.0),
                cx=float(pos.get("x", 0.0)),
                cy=float(pos.get("y", 0.0)),
                w=float(c.get("width_mm", 0.0) or 0.0),
                h=float(c.get("height_mm", 0.0) or 0.0),
                nets=nets,
                pads=pads,
            )
        )
    return parts


# --------------------------------------------------------------------------- #
# Functional grouping — passives assigned to their strongest anchor.
# --------------------------------------------------------------------------- #

# A net with more members than this is treated as a bus/rail (GND, VCC, a shared
# enable). Linking through it would merge every array into one blob, so anchor-less
# clustering ignores high-fanout nets and connects passives only through the
# low-fanout signal nets that actually chain a ladder/array together.
_CLUSTER_MAX_NET_FANOUT = 4

# Largest anchor-less array kept as a single "row". A bigger component (a long
# R-2R ladder) is split into chain-ordered sub-rows of this size so each can be
# made a crisp row -- one 15-wide group scores as an impossible single row (SA
# gets orientation but the "row" stays a 2-D scatter). Chain order keeps
# electrically adjacent parts in the same row, which also helps routing.
_ARRAY_ROW_MAX = 6


def _signal_adjacency(
    refs: set[str], net_to_refs: dict[str, set[str]]
) -> dict[str, set[str]]:
    """Neighbor map over ``refs``: two are adjacent when they share a net whose
    total fanout is <= ``_CLUSTER_MAX_NET_FANOUT`` (ladder nodes chain, rails
    don't)."""
    adj: dict[str, set[str]] = {r: set() for r in refs}
    for net_refs in net_to_refs.values():
        if len(net_refs) > _CLUSTER_MAX_NET_FANOUT:
            continue
        local = sorted(net_refs & refs)
        for i, a in enumerate(local):
            for b in local[i + 1:]:
                adj[a].add(b)
                adj[b].add(a)
    return adj


def _order_chain(comp: list[str], adj: dict[str, set[str]]) -> list[str]:
    """DFS pre-order of one connected component, started at a chain endpoint
    (a degree-1 node if any, else the smallest ref). For a ladder/path this is
    the linear chain order, so chunking it yields contiguous rows. Deterministic
    (lowest-ref neighbor first)."""
    comp_set = set(comp)
    deg = {r: len(adj[r] & comp_set) for r in comp}
    endpoints = sorted(r for r in comp if deg[r] == 1)
    start = endpoints[0] if endpoints else min(comp)
    order: list[str] = []
    seen: set[str] = set()
    stack = [start]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        order.append(x)
        for y in sorted(adj[x] & comp_set, reverse=True):
            if y not in seen:
                stack.append(y)
    order.extend(r for r in sorted(comp) if r not in seen)  # any stragglers
    return order


def _cluster_by_signal_nets(
    refs: list[str], net_to_refs: dict[str, set[str]]
) -> list[list[str]]:
    """Anchor-less passive rows: connected components of ``refs`` linked by shared
    low-fanout nets, each chain-ordered and split into sub-rows of at most
    ``_ARRAY_ROW_MAX`` (a trailing singleton folds back into the previous row so no
    row has fewer than 2). Deterministic; returned sorted by first member."""
    members = set(refs)
    if len(members) < 2:
        return []
    adj = _signal_adjacency(members, net_to_refs)

    rows: list[list[str]] = []
    seen: set[str] = set()
    for r in sorted(members):
        if r in seen:
            continue
        stack, comp = [r], []
        seen.add(r)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in sorted(adj[x], reverse=True):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(comp) < 2:
            continue
        chain = _order_chain(comp, adj)
        for i in range(0, len(chain), _ARRAY_ROW_MAX):
            chunk = chain[i:i + _ARRAY_ROW_MAX]
            if len(chunk) == 1 and rows:
                rows[-1] = rows[-1] + chunk  # fold trailing singleton back
            else:
                rows.append(chunk)
    return sorted((r for r in rows if len(r) >= 2), key=lambda c: c[0])


def assign_passive_groups(parts: Iterable[PlacedPart]) -> list[PassiveGroup]:
    """Group unlocked passives by the anchor (IC/regulator/connector) they most
    strongly connect to. Returns one :class:`PassiveGroup` per anchor that owns
    >= 2 passives — the "functional row" unit (e.g. U1's decoupling caps).

    Assignment key per (passive, anchor): (shared-net count, connection weight,
    anchor net-degree, anchor area), tie-broken by ref for determinism. Passives
    that no anchor owns (an anchor-less resistor ladder / DAC network whose ICs
    sit on other sheets) are then clustered among themselves into connected
    components via shared low-fanout signal nets -- one group per array
    (``anchor_ref`` = ``"array:<first-ref>"``). Groups are returned sorted by
    anchor ref so the order is stable.

    This is the single grouping definition shared by the tidiness metric and the
    Stage-3 structured-layout packer, so the two never disagree about what a
    "row" is (the plan's Stage 1 unification, down payment).
    """
    parts = list(parts)
    by_ref = {p.ref: p for p in parts}

    net_to_refs: dict[str, set[str]] = {}
    for p in parts:
        for net in p.nets:
            net_to_refs.setdefault(net, set()).add(p.ref)

    degree: dict[str, int] = {}
    adjacency: dict[str, dict[str, int]] = {}
    for refs in net_to_refs.values():
        if len(refs) < 2:
            continue
        weight = len(refs) - 1
        rl = sorted(refs)
        for r in rl:
            degree[r] = degree.get(r, 0) + weight
        for i, a in enumerate(rl):
            for b in rl[i + 1 :]:
                adjacency.setdefault(a, {})[b] = adjacency.get(a, {}).get(b, 0) + weight
                adjacency.setdefault(b, {})[a] = adjacency.get(b, {}).get(a, 0) + weight

    nets_by_ref = {p.ref: set(p.nets) for p in parts}
    anchors = [p.ref for p in parts if p.kind in _ANCHOR_KINDS]
    passives = [p.ref for p in parts if p.kind == "passive" and not p.locked]
    if len(passives) < 2:
        return []

    # (1) Anchor-owned rows: each unlocked passive joins the anchor (IC/regulator/
    #     connector) it connects most strongly to -- e.g. a chip's decoupling caps.
    anchor_to_passives: dict[str, list[str]] = {}
    assigned: set[str] = set()
    for pref in sorted(passives):
        p_nets = nets_by_ref.get(pref, set())
        best_anchor = None
        best_key: tuple | None = None
        for aref in anchors:
            shared = len(p_nets & nets_by_ref.get(aref, set()))
            edge = adjacency.get(pref, {}).get(aref, 0)
            key = (shared, edge, degree.get(aref, 0), by_ref[aref].w * by_ref[aref].h)
            if best_key is None or key > best_key:
                best_key = key
                best_anchor = aref
        if best_anchor is not None and best_key is not None and (
            best_key[0] > 0 or best_key[1] > 0
        ):
            anchor_to_passives.setdefault(best_anchor, []).append(pref)
            assigned.add(pref)

    groups = [
        PassiveGroup(anchor_ref=aref, passive_refs=tuple(refs))
        for aref, refs in sorted(anchor_to_passives.items())
        if len(refs) >= 2
    ]

    # (2) Anchor-less arrays: passives no anchor owns (a resistor ladder, DAC
    #     network, or termination array whose ICs live on OTHER sheets) still form
    #     a "belongs-together" unit. Cluster them into connected components linked
    #     by shared low-fanout *signal* nets, so soft tidiness aligns them too.
    #     Without this the most regular-grid-like leaves -- exactly the ones an
    #     orderly layout helps most, e.g. an R-2R ladder -- produced zero groups
    #     and were left untidied.
    leftover = [pref for pref in passives if pref not in assigned]
    for cluster in _cluster_by_signal_nets(leftover, net_to_refs):
        groups.append(
            PassiveGroup(anchor_ref=f"array:{cluster[0]}", passive_refs=tuple(cluster))
        )

    return sorted(groups, key=lambda g: g.anchor_ref)


def functional_passive_groups(parts: Iterable[PlacedPart]) -> list[list[str]]:
    """Back-compat view of :func:`assign_passive_groups` — just the ref lists."""
    return [list(g.passive_refs) for g in assign_passive_groups(parts)]


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #


def orientation_axis(rotation: float) -> str:
    """Cardinal orientation of a 2-terminal part: 'H' or 'V'.

    A passive at 0° and 180° reads identically (horizontal); 90° and 270° read
    vertical. So the axis is ``rotation mod 180`` folded to the nearer cardinal.
    """
    r = rotation % 180.0
    return "H" if (r < 45.0 or r >= 135.0) else "V"


def _dominant_axis(rotations: list[float]) -> tuple[str, int]:
    """Return (dominant axis, count matching). Ties resolve to 'H'."""
    h = sum(1 for r in rotations if orientation_axis(r) == "H")
    v = len(rotations) - h
    return ("H", h) if h >= v else ("V", v)


def leaf_tidiness(parts: Iterable[PlacedPart], *, label: str = "") -> LeafTidiness:
    """Compute the three tidiness sub-metrics for one placed leaf."""
    parts = list(parts)
    n_components = len(parts)
    passives = [p for p in parts if p.kind == "passive" and not p.locked]
    n_passives = len(passives)
    groups = functional_passive_groups(parts)
    by_ref = {p.ref: p for p in parts}

    # --- orientation consensus, grouped (member-weighted over groups) ---
    grouped_passives = 0
    matched = 0
    resid_weighted_sum = 0.0
    resid_weight = 0
    for grp in groups:
        members = [by_ref[r] for r in grp if r in by_ref]
        if len(members) < 2:
            continue
        grouped_passives += len(members)
        _, count = _dominant_axis([m.rotation for m in members])
        matched += count

        xs = [m.cx for m in members]
        ys = [m.cy for m in members]
        x_spread = max(xs) - min(xs)
        y_spread = max(ys) - min(ys)
        # Row => members spread along X, share a Y; residual is the off-axis
        # (Y) scatter. Column => the reverse.
        if x_spread >= y_spread:
            mean = sum(ys) / len(ys)
            resid = sum(abs(y - mean) for y in ys) / len(ys)
        else:
            mean = sum(xs) / len(xs)
            resid = sum(abs(x - mean) for x in xs) / len(xs)
        resid_weighted_sum += resid * len(members)
        resid_weight += len(members)

    orient_grouped = (
        100.0 * matched / grouped_passives if grouped_passives else None
    )
    align_resid = resid_weighted_sum / resid_weight if resid_weight else None

    # --- orientation consensus, whole leaf (all passives) ---
    if n_passives >= 1:
        _, leaf_count = _dominant_axis([p.rotation for p in passives])
        orient_leaf = 100.0 * leaf_count / n_passives
    else:
        orient_leaf = None

    # --- packing fill (needs >= 2 parts; a single part fills its own bbox) ---
    packing_fill = None
    if n_components >= 2:
        min_x = min(p.cx - p.w / 2 for p in parts)
        max_x = max(p.cx + p.w / 2 for p in parts)
        min_y = min(p.cy - p.h / 2 for p in parts)
        max_y = max(p.cy + p.h / 2 for p in parts)
        bbox_area = (max_x - min_x) * (max_y - min_y)
        comp_area = sum(p.w * p.h for p in parts)
        if bbox_area > 1e-6:
            packing_fill = 100.0 * min(1.0, comp_area / bbox_area)

    return LeafTidiness(
        n_components=n_components,
        n_passives=n_passives,
        n_groups=len(groups),
        grouped_passives=grouped_passives,
        orientation_consensus_grouped_pct=orient_grouped,
        orientation_consensus_leaf_pct=orient_leaf,
        alignment_residual_mm=align_resid,
        packing_fill_pct=packing_fill,
        label=label,
    )


def aggregate(metrics: Iterable[LeafTidiness]) -> dict[str, Any]:
    """Corpus-level means over per-leaf metrics, skipping ``None`` values.

    Only leaves with >= 2 passives count toward the orientation/alignment means
    (a leaf with no passive group has no "row" to be tidy or messy).
    """
    metrics = list(metrics)

    def _mean(vals: list[float]) -> Optional[float]:
        return sum(vals) / len(vals) if vals else None

    og = [m.orientation_consensus_grouped_pct for m in metrics
          if m.orientation_consensus_grouped_pct is not None]
    ol = [m.orientation_consensus_leaf_pct for m in metrics
          if m.orientation_consensus_leaf_pct is not None]
    ar = [m.alignment_residual_mm for m in metrics
          if m.alignment_residual_mm is not None]
    pf = [m.packing_fill_pct for m in metrics if m.packing_fill_pct is not None]

    return {
        "n_leaves": len(metrics),
        "n_leaves_with_groups": sum(1 for m in metrics if m.n_groups > 0),
        "orientation_consensus_grouped_pct": _round(_mean(og)),
        "orientation_consensus_leaf_pct": _round(_mean(ol)),
        "alignment_residual_mm": _round(_mean(ar), 3),
        "packing_fill_pct": _round(_mean(pf)),
    }


# --------------------------------------------------------------------------- #
# Pin-locality — do passives hug the anchor pins they connect to?
#
# The objective the placement redesign optimizes for. The math here is the
# SINGLE source of truth: `leaf_pin_locality` (below, over the PlacedPart view /
# frozen layouts) and `PlacementScorer._score_pin_locality` (over live Component
# objects during solve) both call `pin_locality_for_passive`, so the reported
# metric and the optimized score never drift.
# --------------------------------------------------------------------------- #


def _nearest_anchor_pad(
    net: str, px: float, py: float, anchor_pads_by_net: dict[str, list[tuple[float, float]]]
) -> tuple[Optional[float], Optional[tuple[float, float]]]:
    """(distance, point) of the nearest anchor pad on ``net`` to (px,py), or
    (None, None) if no anchor has a pad on that net."""
    cand = anchor_pads_by_net.get(net)
    if not cand:
        return None, None
    best_d: Optional[float] = None
    best_pt: Optional[tuple[float, float]] = None
    for ax, ay in cand:
        d = math.hypot(ax - px, ay - py)
        if best_d is None or d < best_d:
            best_d, best_pt = d, (ax, ay)
    return best_d, best_pt


def _nearest_anchor_body(
    center: tuple[float, float],
    nets: set[str] | frozenset[str],
    anchor_bodies: list[tuple[float, float, frozenset[str]]],
) -> Optional[tuple[float, float]]:
    """Body-center of the nearest anchor sharing any of ``nets`` (fallback pull
    for a passive with no reachable same-net anchor *pad*), or None."""
    cx, cy = center
    best_d: Optional[float] = None
    best_pt: Optional[tuple[float, float]] = None
    for bx, by, bnets in anchor_bodies:
        if not (bnets & nets):
            continue
        d = math.hypot(bx - cx, by - cy)
        if best_d is None or d < best_d:
            best_d, best_pt = d, (bx, by)
    return best_pt


def pin_locality_for_passive(
    pads: list[tuple[float, float, str]],
    center: tuple[float, float],
    anchor_pads_by_net: dict[str, list[tuple[float, float]]],
    anchor_bodies: list[tuple[float, float, frozenset[str]]],
    *,
    plane_nets: frozenset[str] = DEFAULT_PLANE_NETS,
    dist_ref_mm: float = 2.0,
    orient_weight: float = 0.3,
) -> Optional[tuple[float, float, float]]:
    """Pin-locality of ONE passive. Returns ``(score, d_worst_mm, orient_score)``
    (all in [0,100] except ``d_worst_mm``) or ``None`` when it can't be scored.

    * ``d_worst_mm`` — the *worst* pad's distance to its nearest same-net anchor
      pad. A plane pad (poured net) is via-reachable → contributes 0, so it never
      dominates the max: the pull comes entirely from the signal/power-point pad
      that must hug a real pin. If the passive has NO reachable non-plane pad,
      fall back to the nearest anchor body sharing any of its nets (keeps an
      all-plane decap from floating); unscored if not even that exists.
    * ``score`` — ``(1-w)·100·exp(-d_worst/ref) + w·orient`` (the smooth reward
      shape the tidiness align term uses), ``w=orient_weight``.
    * ``orient_score`` — for a 2-pad part, |cos| of the angle between its pad-axis
      and the axis joining its two target pins (100 = pads straddle the pin pair).
    """
    pad_infos: list[tuple[str, float, float, Optional[float], Optional[tuple[float, float]]]] = []
    has_real = False
    for px, py, net in pads:
        d, pt = _nearest_anchor_pad(net, px, py, anchor_pads_by_net)
        if net in plane_nets:
            pad_infos.append((net, px, py, 0.0, pt))  # via-reachable distance, real pin for orientation
        else:
            if d is not None:
                has_real = True
            pad_infos.append((net, px, py, d, pt))

    if has_real:
        defined = [d for (_n, _x, _y, d, _pt) in pad_infos if d is not None]
        d_worst = max(defined) if defined else 0.0
    else:
        pass_nets = {n for (n, _x, _y, _d, _pt) in pad_infos}
        near_body = _nearest_anchor_body(center, pass_nets, anchor_bodies)
        if near_body is None:
            return None
        d_worst = math.hypot(near_body[0] - center[0], near_body[1] - center[1])

    dist_score = 100.0 * math.exp(-d_worst / max(dist_ref_mm, 1e-6))

    # orientation-to-span: reward the 2-pad body rotated so its pad-axis lines up
    # with the axis joining its two target pins.
    orient_score = 100.0
    if len(pad_infos) == 2:
        (n0, x0, y0, _d0, t0), (n1, x1, y1, _d1, t1) = pad_infos
        tgt0 = t0 if t0 is not None else _nearest_anchor_body(center, {n0}, anchor_bodies)
        tgt1 = t1 if t1 is not None else _nearest_anchor_body(center, {n1}, anchor_bodies)
        if tgt0 is not None and tgt1 is not None:
            vx, vy = (x1 - x0), (y1 - y0)
            ux, uy = (tgt1[0] - tgt0[0]), (tgt1[1] - tgt0[1])
            vlen, ulen = math.hypot(vx, vy), math.hypot(ux, uy)
            if vlen > 1e-9 and ulen > 1e-9:
                orient_score = 100.0 * min(1.0, abs((vx * ux + vy * uy) / (vlen * ulen)))

    score = (1.0 - orient_weight) * dist_score + orient_weight * orient_score
    return (score, d_worst, orient_score)


def build_anchor_pad_index(
    parts: Iterable[PlacedPart],
) -> tuple[dict[str, list[tuple[float, float]]], list[tuple[float, float, frozenset[str]]]]:
    """(anchor_pads_by_net, anchor_bodies) over the anchor parts in ``parts``."""
    anchor_pads_by_net: dict[str, list[tuple[float, float]]] = {}
    anchor_bodies: list[tuple[float, float, frozenset[str]]] = []
    for p in parts:
        if p.kind not in _ANCHOR_KINDS:
            continue
        for px, py, net in p.pads:
            anchor_pads_by_net.setdefault(net, []).append((px, py))
        anchor_bodies.append((p.cx, p.cy, frozenset(n for (_x, _y, n) in p.pads)))
    return anchor_pads_by_net, anchor_bodies


def leaf_pin_locality(
    parts: Iterable[PlacedPart],
    *,
    plane_nets: frozenset[str] = DEFAULT_PLANE_NETS,
    dist_ref_mm: float = 2.0,
    orient_weight: float = 0.3,
    label: str = "",
) -> LeafPinLocality:
    """Compute pin-locality for one placed leaf (needs per-pad ``pads`` on the
    :class:`PlacedPart` views — build them with the adapters, not by hand)."""
    parts = list(parts)
    anchor_pads_by_net, anchor_bodies = build_anchor_pad_index(parts)
    passives = [p for p in parts if p.kind == "passive" and not p.locked]

    scores: list[float] = []
    worsts: list[float] = []
    orients: list[float] = []
    for p in passives:
        res = pin_locality_for_passive(
            list(p.pads), (p.cx, p.cy), anchor_pads_by_net, anchor_bodies,
            plane_nets=plane_nets, dist_ref_mm=dist_ref_mm, orient_weight=orient_weight,
        )
        if res is None:
            continue
        s, dw, orient = res
        scores.append(s)
        worsts.append(dw)
        orients.append(orient)

    n_scored = len(scores)
    return LeafPinLocality(
        n_passives=len(passives),
        n_scored=n_scored,
        pin_locality_pct=(sum(scores) / n_scored) if n_scored else None,
        mean_worst_pad_dist_mm=(sum(worsts) / n_scored) if n_scored else None,
        worst_pad_dist_mm=(max(worsts)) if n_scored else None,
        orientation_span_pct=(sum(orients) / n_scored) if n_scored else None,
        label=label,
    )


def aggregate_pin_locality(metrics: Iterable[LeafPinLocality]) -> dict[str, Any]:
    """Corpus-level means over per-leaf pin-locality, skipping ``None`` values."""
    metrics = list(metrics)

    def _mean(vals: list[float]) -> Optional[float]:
        return sum(vals) / len(vals) if vals else None

    pl = [m.pin_locality_pct for m in metrics if m.pin_locality_pct is not None]
    mw = [m.mean_worst_pad_dist_mm for m in metrics if m.mean_worst_pad_dist_mm is not None]
    ww = [m.worst_pad_dist_mm for m in metrics if m.worst_pad_dist_mm is not None]
    osp = [m.orientation_span_pct for m in metrics if m.orientation_span_pct is not None]

    return {
        "n_leaves": len(metrics),
        "n_leaves_scored": sum(1 for m in metrics if m.n_scored > 0),
        "pin_locality_pct": _round(_mean(pl)),
        "mean_worst_pad_dist_mm": _round(_mean(mw), 3),
        "worst_pad_dist_mm": _round(_mean(ww), 3),
        "orientation_span_pct": _round(_mean(osp)),
    }


__all__ = [
    "PlacedPart",
    "PassiveGroup",
    "LeafTidiness",
    "LeafPinLocality",
    "DEFAULT_PLANE_NETS",
    "parts_from_components",
    "parts_from_layout",
    "assign_passive_groups",
    "functional_passive_groups",
    "orientation_axis",
    "leaf_tidiness",
    "aggregate",
    "pin_locality_for_passive",
    "build_anchor_pad_index",
    "leaf_pin_locality",
    "aggregate_pin_locality",
]
