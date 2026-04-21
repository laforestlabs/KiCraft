"""Copper accounting: verify child trace preservation through parent composition.

Provides trace fingerprinting and matching to verify that child subcircuit
copper is preserved through the stamp + route pipeline.  Each child's
contributed traces and vias are fingerprinted before the flat merge, and
then compared against the post-route copper to determine preservation.

Pure Python -- no pcbnew dependency.
"""

from __future__ import annotations

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
    trace_fingerprints: list[Fingerprint] = field(default_factory=list)
    via_fingerprints: list[Fingerprint] = field(default_factory=list)


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
        }


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


def fingerprint_trace(trace: Any) -> Fingerprint:
    """Create a geometric fingerprint for a trace segment.

    Accepts either a ``TraceSegment`` object (with ``.start``, ``.end``,
    ``.layer``, ``.width_mm`` attributes) or a plain dict with equivalent
    keys.

    The fingerprint rounds coordinates to 0.01 mm and widths to 0.001 mm
    so that floating-point jitter from transform round-trips does not break
    matching.
    """
    if hasattr(trace, "start"):
        # TraceSegment object
        return (
            round(trace.start.x, 2),
            round(trace.start.y, 2),
            round(trace.end.x, 2),
            round(trace.end.y, 2),
            str(getattr(trace.layer, "name", trace.layer)),
            round(trace.width_mm, 3),
        )
    # Dict fallback
    return (
        round(float(trace.get("start_x", 0)), 2),
        round(float(trace.get("start_y", 0)), 2),
        round(float(trace.get("end_x", 0)), 2),
        round(float(trace.get("end_y", 0)), 2),
        str(trace.get("layer", "")),
        round(float(trace.get("width", trace.get("width_mm", 0))), 3),
    )


def fingerprint_via(via: Any) -> Fingerprint:
    """Create a geometric fingerprint for a via.

    Accepts either a ``Via`` object (with ``.pos``, ``.drill_mm``,
    ``.size_mm`` attributes) or a plain dict.
    """
    if hasattr(via, "pos"):
        # Via object
        return (
            round(via.pos.x, 2),
            round(via.pos.y, 2),
            round(via.drill_mm, 3),
            round(via.size_mm, 3),
        )
    # Dict fallback
    return (
        round(float(via.get("x", 0)), 2),
        round(float(via.get("y", 0)), 2),
        round(float(via.get("drill", via.get("drill_mm", 0))), 3),
        round(float(via.get("size", via.get("size_mm", 0))), 3),
    )


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
    """Serialize a ChildCopperEntry, omitting fingerprint lists for brevity."""
    return {
        "instance_path": entry.instance_path,
        "sheet_name": entry.sheet_name,
        "trace_count": entry.trace_count,
        "via_count": entry.via_count,
        "total_length_mm": round(entry.total_length_mm, 3),
    }


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def build_copper_manifest(
    composed_children: list[Any],
    parent_traces: list[Any] | None = None,
    parent_vias: list[Any] | None = None,
    final_child_bboxes: dict[str, tuple[tuple[float, float], tuple[float, float]]] | None = None,
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
        Fully populated manifest with per-child fingerprints.
    """
    manifest = CopperManifest()

    for child in composed_children:
        transformed = child.transformed
        traces = transformed.transformed_traces
        vias = transformed.transformed_vias

        trace_fps = [fingerprint_trace(t) for t in traces]
        via_fps = [fingerprint_via(v) for v in vias]
        total_length = sum(_trace_length(t) for t in traces)

        entry = ChildCopperEntry(
            instance_path=child.instance_path,
            sheet_name=getattr(child.instance, "layout_id", child.instance).sheet_name
            if hasattr(child.instance, "layout_id")
            else str(child.instance_path),
            trace_count=len(traces),
            via_count=len(vias),
            total_length_mm=total_length,
            trace_fingerprints=trace_fps,
            via_fingerprints=via_fps,
        )
        if final_child_bboxes and child.instance_path in final_child_bboxes:
            bbox = final_child_bboxes[child.instance_path]
            entry.total_length_mm = total_length
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

    Uses geometric fingerprint matching to determine which child traces
    survived the stamp + route pipeline.

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
        preservation rates, and any issues found.
    """
    # Build fingerprint multisets from post-route copper.
    # Use a dict to handle duplicate fingerprints (multiple traces with
    # same geometry).
    post_trace_fps: dict[Fingerprint, int] = {}
    for t in post_route_traces:
        fp = fingerprint_trace(t)
        post_trace_fps[fp] = post_trace_fps.get(fp, 0) + 1

    post_via_fps: dict[Fingerprint, int] = {}
    for v in post_route_vias:
        fp = fingerprint_via(v)
        post_via_fps[fp] = post_via_fps.get(fp, 0) + 1

    # Match against each child's expected copper.
    # We consume from the multiset so a single post-route trace is not
    # double-counted across multiple children.
    remaining_trace_fps = dict(post_trace_fps)
    remaining_via_fps = dict(post_via_fps)

    total_matched_traces = 0
    total_matched_vias = 0
    per_child_results: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    for path, child in manifest.per_child.items():
        matched_traces = 0
        for fp in child.trace_fingerprints:
            if remaining_trace_fps.get(fp, 0) > 0:
                matched_traces += 1
                remaining_trace_fps[fp] -= 1

        matched_vias = 0
        for fp in child.via_fingerprints:
            if remaining_via_fps.get(fp, 0) > 0:
                matched_vias += 1
                remaining_via_fps[fp] -= 1

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
    new_route_traces = sum(max(0, v) for v in remaining_trace_fps.values())
    new_route_vias = sum(max(0, v) for v in remaining_via_fps.values())

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
    }
