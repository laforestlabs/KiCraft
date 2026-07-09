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

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# Anchor kinds a passive can be grouped under (a passive's "functional home").
_ANCHOR_KINDS = ("ic", "regulator", "connector")


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
            )
        )
    return parts


# --------------------------------------------------------------------------- #
# Functional grouping — passives assigned to their strongest anchor.
# --------------------------------------------------------------------------- #


def assign_passive_groups(parts: Iterable[PlacedPart]) -> list[PassiveGroup]:
    """Group unlocked passives by the anchor (IC/regulator/connector) they most
    strongly connect to. Returns one :class:`PassiveGroup` per anchor that owns
    >= 2 passives — the "functional row" unit (e.g. U1's decoupling caps).

    Assignment key per (passive, anchor): (shared-net count, connection weight,
    anchor net-degree, anchor area), tie-broken by ref for determinism. A
    passive with no net in common with any anchor is left ungrouped. Groups are
    returned sorted by anchor ref so the order is stable.

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
    anchors = [p.ref for p in parts if p.kind in _ANCHOR_KINDS and p.kind != "passive"]
    passives = [p.ref for p in parts if p.kind == "passive" and not p.locked]
    if not anchors or len(passives) < 2:
        return []

    anchor_to_passives: dict[str, list[str]] = {}
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

    return [
        PassiveGroup(anchor_ref=aref, passive_refs=tuple(refs))
        for aref, refs in sorted(anchor_to_passives.items())
        if len(refs) >= 2
    ]


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


__all__ = [
    "PlacedPart",
    "PassiveGroup",
    "LeafTidiness",
    "parts_from_components",
    "parts_from_layout",
    "assign_passive_groups",
    "functional_passive_groups",
    "orientation_axis",
    "leaf_tidiness",
    "aggregate",
]
