"""Copper accounting: verify child trace preservation through parent composition.

Provides trace fingerprinting and matching to verify that child subcircuit
copper is preserved through the stamp + route pipeline.  Each child's
contributed traces and vias are fingerprinted before the flat merge, and
then compared against the post-route copper to determine preservation.

Pure Python -- no pcbnew dependency.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

Fingerprint = tuple[Any, ...]

__all__ = [
    "ChildCopperEntry",
    "CopperManifest",
    "build_copper_manifest",
    "verify_copper_preservation",
    "fingerprint_trace",
    "fingerprint_via",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ChildCopperEntry:
    """Copper contribution from one child subcircuit."""

    instance_path: str
    sheet_name: str
    trace_count: int
    via_count: int
    total_length_mm: float
    # Raw absolute coordinates (composition frame) as
    # (x1, y1, x2, y2, layer, width) per trace.  Kept at full precision so
    # verification can solve the exact board translation instead of matching
    # against coordinates rounded to the fingerprint grid (0.01 mm), which is
    # too coarse to recover a non-grid-aligned A4-centering offset.  Vias are
    # (x, y, drill, size).
    trace_coords: list[tuple[float, float, float, float, str, float]] = field(
        default_factory=list
    )
    via_coords: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass(slots=True)
class CopperManifest:
    """Pre-stamp copper ledger recording what each child contributed.

    Built from the composed children *before* the flat merge loses
    provenance information.
    """

    per_child: dict[str, ChildCopperEntry] = field(default_factory=dict)
    total_child_traces: int = 0
    total_child_vias: int = 0
    total_child_length_mm: float = 0.0
    parent_interconnect_traces: int = 0
    parent_interconnect_vias: int = 0
    parent_interconnect_length_mm: float = 0.0
    # Minimum (x, y) across the composed child traces in the pre-stamp frame.
    # Reference-only: verification solves the exact board translation, so this
    # must never be recomputed from the post-route trace set.
    origin: tuple[float, float] = (0.0, 0.0)

    @property
    def total_traces(self) -> int:
        return self.total_child_traces + self.parent_interconnect_traces

    @property
    def total_vias(self) -> int:
        return self.total_child_vias + self.parent_interconnect_vias

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_child": {
                k: _child_entry_to_dict(v) for k, v in self.per_child.items()
            },
            "total_child_traces": self.total_child_traces,
            "total_child_vias": self.total_child_vias,
            "total_child_length_mm": round(self.total_child_length_mm, 3),
            "parent_interconnect_traces": self.parent_interconnect_traces,
            "parent_interconnect_vias": self.parent_interconnect_vias,
            "parent_interconnect_length_mm": round(
                self.parent_interconnect_length_mm, 3
            ),
            "total_traces": self.total_traces,
            "total_vias": self.total_vias,
            "origin": {"x": self.origin[0], "y": self.origin[1]},
        }


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def _trace_set_origin(traces: list[Any]) -> tuple[float, float]:
    """Compute the minimum (x, y) across a set of traces.

    Used only as a reference origin on the manifest; verification solves the
    exact board translation rather than relying on this minimum.
    """
    min_x = min_y = float("inf")
    for t in traces:
        if hasattr(t, "start"):
            sx, sy = t.start.x, t.start.y
            ex, ey = t.end.x, t.end.y
        else:
            sx = float(t.get("start_x", 0))
            sy = float(t.get("start_y", 0))
            ex = float(t.get("end_x", 0))
            ey = float(t.get("end_y", 0))
        min_x = min(min_x, sx, ex)
        min_y = min(min_y, sy, ey)
    return (min_x if min_x != float("inf") else 0.0, min_y if min_y != float("inf") else 0.0)


def _trace_length(trace: Any) -> float:
    """Calculate the length of a trace segment in mm."""
    if hasattr(trace, "length"):
        return trace.length
    if hasattr(trace, "start"):
        dx = trace.end.x - trace.start.x
        dy = trace.end.y - trace.start.y
        return (dx * dx + dy * dy) ** 0.5
    dx = float(trace.get("end_x", 0)) - float(trace.get("start_x", 0))
    dy = float(trace.get("end_y", 0)) - float(trace.get("start_y", 0))
    return (dx * dx + dy * dy) ** 0.5


def _child_entry_to_dict(entry: ChildCopperEntry) -> dict[str, Any]:
    """Serialize a ChildCopperEntry, omitting raw coordinate lists for brevity."""
    return {
        "instance_path": entry.instance_path,
        "sheet_name": entry.sheet_name,
        "trace_count": entry.trace_count,
        "via_count": entry.via_count,
        "total_length_mm": round(entry.total_length_mm, 3),
    }


def _trace_raw(trace: Any) -> tuple[float, float, float, float, str, float]:
    """Extract raw absolute trace coordinates as (x1, y1, x2, y2, layer, width)."""
    if hasattr(trace, "start"):
        return (
            float(trace.start.x),
            float(trace.start.y),
            float(trace.end.x),
            float(trace.end.y),
            str(getattr(trace.layer, "name", trace.layer)),
            float(trace.width_mm),
        )
    return (
        float(trace.get("start_x", 0)),
        float(trace.get("start_y", 0)),
        float(trace.get("end_x", 0)),
        float(trace.get("end_y", 0)),
        str(trace.get("layer", "")),
        float(trace.get("width", trace.get("width_mm", 0))),
    )


def _via_raw(via: Any) -> tuple[float, float, float, float]:
    """Extract raw absolute via coordinates as (x, y, drill, size)."""
    if hasattr(via, "pos"):
        return (
            float(via.pos.x),
            float(via.pos.y),
            float(via.drill_mm),
            float(via.size_mm),
        )
    return (
        float(via.get("x", 0)),
        float(via.get("y", 0)),
        float(via.get("drill", via.get("drill_mm", 0))),
        float(via.get("size", via.get("size_mm", 0))),
    )


def _fingerprint_trace_coords(
    coords: tuple[float, float, float, float, str, float],
    shift: tuple[float, float] = (0.0, 0.0),
) -> Fingerprint:
    """Fingerprint raw trace coordinates, optionally shifted by a translation."""
    sx, sy, ex, ey, layer, width = coords
    tx, ty = shift
    return (
        round(sx + tx, 2),
        round(sy + ty, 2),
        round(ex + tx, 2),
        round(ey + ty, 2),
        layer,
        round(width, 3),
    )


def _fingerprint_via_coords(
    coords: tuple[float, float, float, float],
    shift: tuple[float, float] = (0.0, 0.0),
) -> Fingerprint:
    """Fingerprint raw via coordinates, optionally shifted by a translation."""
    x, y, drill, size = coords
    tx, ty = shift
    return (
        round(x + tx, 2),
        round(y + ty, 2),
        round(drill, 3),
        round(size, 3),
    )


def fingerprint_trace(trace: Any, origin: tuple[float, float] = (0.0, 0.0)) -> Fingerprint:
    """Create a geometric fingerprint for a trace segment.

    Accepts either a ``TraceSegment`` object (with ``.start``, ``.end``,
    ``.layer``, ``.width_mm`` attributes) or a plain dict with equivalent
    keys.

    The fingerprint rounds coordinates to 0.01 mm and widths to 0.001 mm
    so that floating-point jitter from transform round-trips does not break
    matching.  When *origin* is provided, coordinates are made relative to
    it so fingerprints survive uniform translation (A4 centering).
    """
    return _fingerprint_trace_coords(_trace_raw(trace), shift=(-origin[0], -origin[1]))


def fingerprint_via(via: Any, origin: tuple[float, float] = (0.0, 0.0)) -> Fingerprint:
    """Create a geometric fingerprint for a via.

    Accepts either a ``Via`` object (with ``.pos``, ``.drill_mm``,
    ``.size_mm`` attributes) or a plain dict.
    """
    return _fingerprint_via_coords(_via_raw(via), shift=(-origin[0], -origin[1]))


def _trace_shape(coords: tuple[float, float, float, float, str, float]) -> tuple[Any, ...]:
    """Translation-invariant trace shape: layer, width, and segment delta."""
    sx, sy, ex, ey, layer, width = coords
    return (layer, round(width, 3), round(ex - sx, 2), round(ey - sy, 2))


def _via_shape(coords: tuple[float, float, float, float]) -> tuple[float, float]:
    """Translation-invariant via shape: drill and size."""
    _x, _y, drill, size = coords
    return (round(drill, 3), round(size, 3))


def _solve_translation(
    expected_traces: list[tuple[float, float, float, float, str, float]],
    expected_vias: list[tuple[float, float, float, float]],
    post_traces: list[tuple[float, float, float, float, str, float]],
    post_vias: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    """Solve the uniform board translation mapping expected -> post copper.

    Candidate translations come from pairing expected and post items whose
    translation-invariant shape (layer/width/delta for traces, drill/size for
    vias) matches; the chosen translation is the one that maximizes the number
    of child trace + via fingerprints it can match (multiset intersection),
    with a deterministic tie-break.  Returns (0.0, 0.0) when no child copper
    exists to constrain the translation.
    """
    candidates: set[tuple[float, float]] = set()

    post_trace_index: dict[tuple[Any, ...], list[tuple[float, float, float, float, str, float]]] = {}
    for coords in post_traces:
        post_trace_index.setdefault(_trace_shape(coords), []).append(coords)
    for coords in expected_traces:
        for post_coords in post_trace_index.get(_trace_shape(coords), []):
            candidates.add(
                (round(post_coords[0] - coords[0], 6), round(post_coords[1] - coords[1], 6))
            )

    post_via_index: dict[tuple[float, float], list[tuple[float, float, float, float]]] = {}
    for coords in post_vias:
        post_via_index.setdefault(_via_shape(coords), []).append(coords)
    for coords in expected_vias:
        for post_coords in post_via_index.get(_via_shape(coords), []):
            candidates.add(
                (round(post_coords[0] - coords[0], 6), round(post_coords[1] - coords[1], 6))
            )

    if not candidates:
        return (0.0, 0.0)

    post_trace_fps = Counter(
        _fingerprint_trace_coords(coords) for coords in post_traces
    )
    post_via_fps = Counter(
        _fingerprint_via_coords(coords) for coords in post_vias
    )

    best: tuple[float, float] | None = None
    best_matches = -1
    for tx, ty in candidates:
        shifted_trace_fps = Counter(
            _fingerprint_trace_coords(coords, shift=(tx, ty))
            for coords in expected_traces
        )
        shifted_via_fps = Counter(
            _fingerprint_via_coords(coords, shift=(tx, ty))
            for coords in expected_vias
        )
        matches = sum((shifted_trace_fps & post_trace_fps).values())
        matches += sum((shifted_via_fps & post_via_fps).values())
        if matches > best_matches or (matches == best_matches and (tx, ty) < best):
            best_matches = matches
            best = (tx, ty)

    return best if best is not None else (0.0, 0.0)


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def build_copper_manifest(
    composed_children: list[Any],
    parent_traces: list[Any] | None = None,
    parent_vias: list[Any] | None = None,
) -> CopperManifest:
    """Build a manifest recording expected copper from composition.

    Parameters
    ----------
    composed_children:
        List of ``ComposedChild`` objects from
        ``build_parent_composition``.  Each must have a
        ``.transformed.transformed_traces`` and
        ``.transformed.transformed_vias`` attribute.
    parent_traces:
        Optional list of parent interconnect traces (``TraceSegment``
        objects or dicts).
    parent_vias:
        Optional list of parent interconnect vias (``Via`` objects or
        dicts).

    Returns
    -------
    CopperManifest
        Fully populated manifest with per-child absolute copper coordinates.
        The child trace-set minimum is recorded as ``origin`` for reference.
        Verification solves the exact board translation, so parent copper
        added later can never shift the child reference frame.
    """
    manifest = CopperManifest()

    # Record the pre-stamp child trace-set minimum as a reference origin.  It
    # is deliberately NOT used to make fingerprints relative: verification
    # solves the exact composition -> board translation instead, because the
    # A4-centering offset is not grid-aligned to the 0.01 mm fingerprint grid.
    all_child_traces: list[Any] = []
    for child in composed_children:
        all_child_traces.extend(child.transformed.transformed_traces)
    manifest.origin = _trace_set_origin(all_child_traces)

    for child in composed_children:
        transformed = child.transformed
        traces = transformed.transformed_traces
        vias = transformed.transformed_vias

        trace_coords = [_trace_raw(t) for t in traces]
        via_coords = [_via_raw(v) for v in vias]
        total_length = sum(_trace_length(t) for t in traces)

        entry = ChildCopperEntry(
            instance_path=child.instance_path,
            sheet_name=getattr(child.instance, "layout_id", child.instance).sheet_name
            if hasattr(child.instance, "layout_id")
            else str(child.instance_path),
            trace_count=len(traces),
            via_count=len(vias),
            total_length_mm=total_length,
            trace_coords=trace_coords,
            via_coords=via_coords,
        )
        manifest.per_child[entry.instance_path] = entry
        manifest.total_child_traces += len(traces)
        manifest.total_child_vias += len(vias)
        manifest.total_child_length_mm += total_length

    if parent_traces:
        manifest.parent_interconnect_traces = len(parent_traces)
        manifest.parent_interconnect_length_mm = sum(
            _trace_length(t) for t in parent_traces
        )
    if parent_vias:
        manifest.parent_interconnect_vias = len(parent_vias)

    return manifest


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_copper_preservation(
    manifest: CopperManifest,
    post_route_traces: list[Any],
    post_route_vias: list[Any],
) -> dict[str, Any]:
    """Compare expected child copper against post-route copper.

    The composed child copper and the post-route board may live in different
    coordinate frames (the parent stamper centers the assembly on an A4 sheet
    before saving).  A single uniform translation is solved from
    translation-invariant copper shape (layer/width/delta for traces,
    drill/size for vias) and applied once to every expected item, so parent
    interconnect copper added before/after/between child coordinates can never
    shift the child reference frame.

    Parameters
    ----------
    manifest:
        The pre-stamp copper manifest from :func:`build_copper_manifest`.
    post_route_traces:
        List of trace segments from the routed board (``TraceSegment``
        objects or dicts).
    post_route_vias:
        List of vias from the routed board (``Via`` objects or dicts).

    Returns
    -------
    dict
        Structured verification report with ``status``, per-child
        preservation rates, the chosen board translation, and any issues
        found.
    """
    post_trace_coords = [_trace_raw(t) for t in post_route_traces]
    post_via_coords = [_via_raw(v) for v in post_route_vias]

    expected_trace_coords: list[tuple[float, float, float, float, str, float]] = []
    expected_via_coords: list[tuple[float, float, float, float]] = []
    for child in manifest.per_child.values():
        expected_trace_coords.extend(child.trace_coords)
        expected_via_coords.extend(child.via_coords)

    tx, ty = _solve_translation(
        expected_trace_coords,
        expected_via_coords,
        post_trace_coords,
        post_via_coords,
    )

    # Absolute post-route fingerprint multisets.  Expected fingerprints are
    # shifted by the solved translation before matching, so both sides share
    # the final-board frame.
    post_trace_fps: Counter = Counter(
        _fingerprint_trace_coords(c) for c in post_trace_coords
    )
    post_via_fps: Counter = Counter(
        _fingerprint_via_coords(c) for c in post_via_coords
    )
    remaining_trace_fps = dict(post_trace_fps)
    remaining_via_fps = dict(post_via_fps)

    total_matched_traces = 0
    total_matched_vias = 0
    per_child_results: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    unmatched_expected_traces: list[Fingerprint] = []
    unmatched_expected_vias: list[Fingerprint] = []

    for path, child in manifest.per_child.items():
        matched_traces = 0
        for coords in child.trace_coords:
            fp = _fingerprint_trace_coords(coords, shift=(tx, ty))
            if remaining_trace_fps.get(fp, 0) > 0:
                matched_traces += 1
                remaining_trace_fps[fp] -= 1
            elif len(unmatched_expected_traces) < 5:
                unmatched_expected_traces.append(fp)

        matched_vias = 0
        for coords in child.via_coords:
            fp = _fingerprint_via_coords(coords, shift=(tx, ty))
            if remaining_via_fps.get(fp, 0) > 0:
                matched_vias += 1
                remaining_via_fps[fp] -= 1
            elif len(unmatched_expected_vias) < 5:
                unmatched_expected_vias.append(fp)

        trace_preservation = (
            matched_traces / child.trace_count if child.trace_count > 0 else 1.0
        )
        via_preservation = (
            matched_vias / child.via_count if child.via_count > 0 else 1.0
        )

        per_child_results[path] = {
            "sheet_name": child.sheet_name,
            "expected_traces": child.trace_count,
            "matched_traces": matched_traces,
            "expected_vias": child.via_count,
            "matched_vias": matched_vias,
            "trace_preservation": round(trace_preservation, 4),
            "via_preservation": round(via_preservation, 4),
        }

        total_matched_traces += matched_traces
        total_matched_vias += matched_vias

        if matched_traces < child.trace_count:
            lost = child.trace_count - matched_traces
            issues.append(
                f"{child.sheet_name}: lost {lost}/{child.trace_count} traces"
            )
        if matched_vias < child.via_count:
            lost = child.via_count - matched_vias
            issues.append(f"{child.sheet_name}: lost {lost}/{child.via_count} vias")

    overall_trace_preservation = (
        total_matched_traces / manifest.total_child_traces
        if manifest.total_child_traces > 0
        else 1.0
    )
    overall_via_preservation = (
        total_matched_vias / manifest.total_child_vias
        if manifest.total_child_vias > 0
        else 1.0
    )

    # Count new traces/vias added by parent routing (not matching any child)
    new_route_traces = sum(remaining_trace_fps.values())
    new_route_vias = sum(remaining_via_fps.values())

    # Determine overall status
    status = "PASS"
    if issues:
        status = "WARN" if overall_trace_preservation > 0.95 else "FAIL"

    return {
        "status": status,
        "trace_preservation_rate": round(overall_trace_preservation, 4),
        "via_preservation_rate": round(overall_via_preservation, 4),
        "matched_child_traces": total_matched_traces,
        "expected_child_traces": manifest.total_child_traces,
        "matched_child_vias": total_matched_vias,
        "expected_child_vias": manifest.total_child_vias,
        "post_route_total_traces": len(post_route_traces),
        "post_route_total_vias": len(post_route_vias),
        "new_route_traces": new_route_traces,
        "new_route_vias": new_route_vias,
        "per_child": per_child_results,
        "issues": issues,
        "reference_origin": {"x": manifest.origin[0], "y": manifest.origin[1]},
        "chosen_translation_mm": {"x": tx, "y": ty},
        "diagnostics": {
            "unmatched_expected_trace_fps": [list(fp) for fp in unmatched_expected_traces],
            "unmatched_expected_via_fps": [list(fp) for fp in unmatched_expected_vias],
            "unmatched_routed_trace_fps": [
                list(fp) for fp in list(remaining_trace_fps)[:5]
            ],
            "unmatched_routed_via_fps": [
                list(fp) for fp in list(remaining_via_fps)[:5]
            ],
        },
    }
