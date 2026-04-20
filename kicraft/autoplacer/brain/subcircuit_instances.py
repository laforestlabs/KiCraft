"""Rigid solved-artifact loader and transform helpers for subcircuits.

This module is the first composition-side utility for the subcircuits redesign.
It loads solved leaf artifacts from disk, reconstructs rigid subcircuit layout
instances, and applies translation/rotation transforms to the entire solved
artifact as a unit.

Current scope:
- load solved subcircuit metadata/debug artifacts from `.experiments/subcircuits`
- reconstruct solved component geometry from serialized debug payloads
- reconstruct lightweight copper geometry (traces/vias) when present
- build rigid `SubCircuitLayout` and `SubCircuitInstance` objects
- apply rigid transforms (translate + rotate) to solved artifacts
- expose transformed interface anchors and bounding boxes for parent composition

This module intentionally does not yet:
- stamp transformed artifacts back into a parent `BoardState`
- perform parent-level placement optimization
- merge copper between child artifacts
- support whole-subcircuit flipping to the opposite board side

Those capabilities belong to later milestones.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import (
    Component,
    InterfaceAnchor,
    InterfacePort,
    Layer,
    Pad,
    Point,
    SilkscreenElement,
    SubCircuitInstance,
    SubCircuitLayout,
    TraceSegment,
    Via,
)


@dataclass(slots=True)
class LoadedSubcircuitArtifact:
    """Loaded solved artifact bundle from disk."""

    artifact_dir: str
    metadata: dict[str, Any]
    debug: dict[str, Any]
    layout: SubCircuitLayout
    source_files: dict[str, str] = field(default_factory=dict)

    @property
    def sheet_name(self) -> str:
        return self.layout.subcircuit_id.sheet_name

    @property
    def instance_path(self) -> str:
        return self.layout.subcircuit_id.instance_path


@dataclass(slots=True)
class TransformedSubcircuit:
    """Rigid transformed view of a solved subcircuit layout."""

    instance: SubCircuitInstance
    layout: SubCircuitLayout
    transformed_components: dict[str, Component] = field(default_factory=dict)
    transformed_traces: list[TraceSegment] = field(default_factory=list)
    transformed_vias: list[Via] = field(default_factory=list)
    transformed_silkscreen: list[SilkscreenElement] = field(default_factory=list)
    transformed_anchors: list[InterfaceAnchor] = field(default_factory=list)
    bounding_box: tuple[Point, Point] = field(
        default_factory=lambda: (Point(0.0, 0.0), Point(0.0, 0.0))
    )

    @property
    def width_mm(self) -> float:
        tl, br = self.bounding_box
        return max(0.0, br.x - tl.x)

    @property
    def height_mm(self) -> float:
        tl, br = self.bounding_box
        return max(0.0, br.y - tl.y)


def load_solved_artifact(
    artifact_dir: str | Path,
) -> LoadedSubcircuitArtifact:
    """Load a solved subcircuit artifact bundle from disk.

    Required files:
    - `metadata.json`
    - `debug.json`
    - `solved_layout.json`
    """
    artifact_path = Path(artifact_dir)
    metadata_path = artifact_path / "metadata.json"
    debug_path = artifact_path / "debug.json"
    solved_layout_path = artifact_path / "solved_layout.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing artifact metadata: {metadata_path}")
    if not debug_path.exists():
        raise FileNotFoundError(f"Missing artifact debug payload: {debug_path}")
    if not solved_layout_path.exists():
        raise FileNotFoundError(f"Missing solved layout artifact: {solved_layout_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    solved_layout = json.loads(solved_layout_path.read_text(encoding="utf-8"))
    layout = _layout_from_artifact_payload(metadata, debug, solved_layout)

    return LoadedSubcircuitArtifact(
        artifact_dir=str(artifact_path),
        metadata=metadata,
        debug=debug,
        layout=layout,
        source_files={
            "metadata_json": str(metadata_path),
            "debug_json": str(debug_path),
            "mini_pcb": metadata.get("artifact_paths", {}).get("mini_pcb", ""),
            "solved_layout_json": str(solved_layout_path)
            if solved_layout_path.exists()
            else "",
        },
    )


def load_solved_artifacts(
    artifact_dirs: list[str | Path],
) -> list[LoadedSubcircuitArtifact]:
    """Load multiple solved artifacts."""
    return [load_solved_artifact(path) for path in artifact_dirs]


def instantiate_subcircuit(
    layout: SubCircuitLayout,
    origin: Point,
    rotation: float = 0.0,
) -> SubCircuitInstance:
    """Create a rigid subcircuit instance for parent composition."""
    transformed_bbox = _rotated_bbox_size(layout.width, layout.height, rotation)
    return SubCircuitInstance(
        layout_id=layout.subcircuit_id,
        origin=Point(origin.x, origin.y),
        rotation=float(rotation),
        transformed_bbox=transformed_bbox,
    )


def transform_subcircuit_instance(
    layout: SubCircuitLayout,
    instance: SubCircuitInstance,
) -> TransformedSubcircuit:
    """Apply a rigid transform to a solved subcircuit layout.

    The transform is:
    - rotate around the local origin (0, 0)
    - then translate by `instance.origin`

    Internal geometry remains rigid.
    """
    transformed_components = {
        ref: _transform_component(comp, instance.origin, instance.rotation)
        for ref, comp in layout.components.items()
    }
    transformed_traces = [
        _transform_trace(trace, instance.origin, instance.rotation)
        for trace in layout.traces
    ]
    transformed_vias = [
        _transform_via(via, instance.origin, instance.rotation) for via in layout.vias
    ]
    transformed_silkscreen = [
        _transform_silkscreen_element(elem, instance.origin, instance.rotation)
        for elem in layout.silkscreen
    ]
    transformed_anchors = [
        _transform_anchor(anchor, instance.origin, instance.rotation)
        for anchor in layout.interface_anchors
    ]
    bounding_box = _compute_layout_bbox(
        transformed_components,
        transformed_traces,
        transformed_vias,
        transformed_anchors,
    )

    return TransformedSubcircuit(
        instance=instance,
        layout=layout,
        transformed_components=transformed_components,
        transformed_traces=transformed_traces,
        transformed_vias=transformed_vias,
        transformed_silkscreen=transformed_silkscreen,
        transformed_anchors=transformed_anchors,
        bounding_box=bounding_box,
    )


def transform_loaded_artifact(
    artifact: LoadedSubcircuitArtifact,
    origin: Point,
    rotation: float = 0.0,
) -> TransformedSubcircuit:
    """Convenience wrapper: instantiate and transform a loaded artifact."""
    instance = instantiate_subcircuit(artifact.layout, origin, rotation)
    return transform_subcircuit_instance(artifact.layout, instance)


def transformed_anchor_map(
    transformed: TransformedSubcircuit,
) -> dict[str, InterfaceAnchor]:
    """Build a port-name -> transformed anchor map."""
    return {anchor.port_name: anchor for anchor in transformed.transformed_anchors}


def transformed_component_map(
    transformed: TransformedSubcircuit,
) -> dict[str, Component]:
    """Return transformed components keyed by reference."""
    return dict(transformed.transformed_components)


def artifact_summary(artifact: LoadedSubcircuitArtifact) -> str:
    """Human-readable one-line summary for logs/debug output."""
    return (
        f"{artifact.sheet_name} "
        f"[{artifact.instance_path}] "
        f"components={len(artifact.layout.components)} "
        f"traces={len(artifact.layout.traces)} "
        f"vias={len(artifact.layout.vias)} "
        f"anchors={len(artifact.layout.interface_anchors)} "
        f"bbox={artifact.layout.width:.1f}x{artifact.layout.height:.1f}mm "
        f"score={artifact.layout.score:.1f}"
    )


def transformed_summary(transformed: TransformedSubcircuit) -> str:
    """Human-readable one-line summary for transformed rigid instances."""
    return (
        f"{transformed.layout.subcircuit_id.sheet_name} "
        f"[{transformed.instance.layout_id.instance_path}] "
        f"origin=({transformed.instance.origin.x:.1f},{transformed.instance.origin.y:.1f}) "
        f"rot={transformed.instance.rotation:.1f} "
        f"bbox={transformed.width_mm:.1f}x{transformed.height_mm:.1f}mm "
        f"anchors={len(transformed.transformed_anchors)}"
    )


def artifact_debug_dict(artifact: LoadedSubcircuitArtifact) -> dict[str, Any]:
    """Return a JSON-serializable debug view of a loaded artifact."""
    return {
        "artifact_dir": artifact.artifact_dir,
        "source_files": dict(artifact.source_files),
        "subcircuit_id": {
            "sheet_name": artifact.layout.subcircuit_id.sheet_name,
            "sheet_file": artifact.layout.subcircuit_id.sheet_file,
            "instance_path": artifact.layout.subcircuit_id.instance_path,
            "parent_instance_path": artifact.layout.subcircuit_id.parent_instance_path,
        },
        "component_count": len(artifact.layout.components),
        "trace_count": len(artifact.layout.traces),
        "via_count": len(artifact.layout.vias),
        "anchor_count": len(artifact.layout.interface_anchors),
        "bounding_box": {
            "width_mm": artifact.layout.width,
            "height_mm": artifact.layout.height,
        },
        "score": artifact.layout.score,
    }


def transformed_debug_dict(transformed: TransformedSubcircuit) -> dict[str, Any]:
    """Return a JSON-serializable debug view of a transformed instance."""
    tl, br = transformed.bounding_box
    return {
        "instance": {
            "layout_id": {
                "sheet_name": transformed.instance.layout_id.sheet_name,
                "sheet_file": transformed.instance.layout_id.sheet_file,
                "instance_path": transformed.instance.layout_id.instance_path,
                "parent_instance_path": transformed.instance.layout_id.parent_instance_path,
            },
            "origin": {
                "x": transformed.instance.origin.x,
                "y": transformed.instance.origin.y,
            },
            "rotation": transformed.instance.rotation,
            "transformed_bbox": {
                "width_mm": transformed.instance.transformed_bbox[0],
                "height_mm": transformed.instance.transformed_bbox[1],
            },
        },
        "bounding_box": {
            "top_left": {"x": tl.x, "y": tl.y},
            "bottom_right": {"x": br.x, "y": br.y},
            "width_mm": transformed.width_mm,
            "height_mm": transformed.height_mm,
        },
        "component_count": len(transformed.transformed_components),
        "trace_count": len(transformed.transformed_traces),
        "via_count": len(transformed.transformed_vias),
        "anchor_count": len(transformed.transformed_anchors),
    }


def _layout_from_artifact_payload(
    metadata: dict[str, Any],
    debug: dict[str, Any],
    solved_layout: dict[str, Any] | None = None,
) -> SubCircuitLayout:
    """Reconstruct a `SubCircuitLayout` from artifact payloads."""
    subcircuit_id = _subcircuit_id_from_metadata(metadata)
    canonical = _normalize_to_canonical(metadata, debug, solved_layout)

    solved_components = _parse_components(canonical)
    solved_traces = _parse_traces(canonical)
    solved_vias = _parse_vias(canonical)
    solved_silkscreen = _parse_silkscreen(canonical)
    ports = _interface_ports_from_payload(canonical.get("ports", []))
    interface_anchors = _parse_interface_anchors(canonical)
    bbox = _parse_bbox(canonical, solved_components)
    score = _parse_score(canonical.get("score"))

    artifact_paths = dict(metadata.get("artifact_paths", {}))

    return SubCircuitLayout(
        subcircuit_id=subcircuit_id,
        components=solved_components,
        traces=solved_traces,
        vias=solved_vias,
        silkscreen=solved_silkscreen,
        bounding_box=bbox,
        ports=ports,
        interface_anchors=interface_anchors,
        score=score,
        artifact_paths=artifact_paths,
        frozen=True,
    )


def _subcircuit_id_from_metadata(metadata: dict[str, Any]):
    from .types import SubCircuitId

    sid = metadata.get("subcircuit_id", {})
    return SubCircuitId(
        sheet_name=sid.get("sheet_name", metadata.get("sheet_name", "")),
        sheet_file=sid.get("sheet_file", metadata.get("sheet_file", "")),
        instance_path=sid.get("instance_path", metadata.get("instance_path", "")),
        parent_instance_path=sid.get(
            "parent_instance_path",
            metadata.get("parent_instance_path"),
        ),
    )


def _normalize_to_canonical(
    metadata: dict[str, Any],
    debug: dict[str, Any],
    solved_layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize artifact payloads into one canonical dict shape.

    If *solved_layout* is present, it is already canonical -- return as-is.
    Otherwise, remap the debug/metadata fallback dicts into the same
    canonical key layout so downstream parsers never branch.
    """
    if isinstance(solved_layout, dict) and solved_layout:
        return solved_layout

    # --- components ---
    components: Any = debug.get("solved_components")
    if not isinstance(components, dict) or not components:
        components = {}

    # --- traces / vias ---
    traces: list[Any] = []
    vias: list[Any] = []

    # --- ports ---
    ports = metadata.get("interface_ports", [])

    # --- interface anchors (3-level fallback, then normalize flat x/y) ---
    anchors: Any = (
        debug.get("extra", {})
        .get("solve_summary", {})
        .get("placement_result", {})
        .get("interface_anchors")
    )
    if not isinstance(anchors, list):
        anchors = debug.get("extra", {}).get("placement_result", {}).get("interface_anchors")
    if not isinstance(anchors, list):
        anchors = debug.get("extra", {}).get("solve_summary", {}).get("interface_anchors")
    if not isinstance(anchors, list):
        anchors = []

    # Normalize flat x/y anchors to nested {pos: {x, y}} form
    normalized_anchors = []
    for a in anchors:
        if not isinstance(a, dict):
            continue
        if "pos" not in a and ("x" in a or "y" in a):
            a = dict(a)
            a["pos"] = {"x": a.pop("x", 0.0), "y": a.pop("y", 0.0)}
        normalized_anchors.append(a)

    # --- bounding box ---
    outline = metadata.get("local_board_outline", {})
    if not (isinstance(outline, dict) and "width_mm" in outline):
        outline = debug.get("leaf_extraction", {}).get("local_board_outline", {})
    bbox: dict[str, Any] = {}
    if isinstance(outline, dict) and "width_mm" in outline:
        bbox = {
            "width_mm": outline["width_mm"],
            "height_mm": outline["height_mm"],
        }

    # --- score ---
    score: Any = None
    best_round = debug.get("extra", {}).get("best_round", {})
    if isinstance(best_round, dict) and "score" in best_round:
        score = best_round["score"]
    else:
        best_round = debug.get("extra", {}).get("solve_summary", {}).get("best_round", {})
        if isinstance(best_round, dict) and "score" in best_round:
            score = best_round["score"]

    return {
        "components": components if isinstance(components, dict) else {},
        "traces": traces if isinstance(traces, list) else [],
        "vias": vias if isinstance(vias, list) else [],
        "silkscreen": [],
        "ports": ports if isinstance(ports, list) else [],
        "interface_anchors": normalized_anchors,
        "bounding_box": bbox,
        "score": score,
    }


def _interface_ports_from_payload(ports: Any) -> list[InterfacePort]:
    """Build logical interface ports from a serialized payload list."""
    if not isinstance(ports, list):
        return []

    extracted: list[InterfacePort] = []
    for port in ports:
        if not isinstance(port, dict):
            continue
        name = str(port.get("name", "")).strip()
        net_name = str(port.get("net_name", "")).strip()
        if not name or not net_name:
            continue
        extracted.append(
            InterfacePort(
                name=name,
                net_name=net_name,
                cardinality=int(port.get("cardinality", 1) or 1),
                bus_index=port.get("bus_index"),
                required=bool(port.get("required", True)),
                description=str(port.get("description", "")),
                raw_direction=str(port.get("direction", "")),
                source_uuid=port.get("source_uuid"),
                source_kind=str(port.get("source_kind", "sheet_pin")),
            )
        )
    return extracted


def _parse_components(
    canonical: dict[str, Any],
) -> dict[str, Component]:
    """Parse solved component geometry from canonical payload."""
    components = canonical.get("components")
    if not isinstance(components, dict):
        return {}
    return {
        ref: _component_from_dict(comp_dict)
        for ref, comp_dict in components.items()
        if isinstance(comp_dict, dict)
    }


def _parse_traces(canonical: dict[str, Any]) -> list[TraceSegment]:
    """Parse solved traces from canonical payload."""
    traces_payload = canonical.get("traces", [])
    if not isinstance(traces_payload, list):
        return []
    traces: list[TraceSegment] = []
    for item in traces_payload:
        if not isinstance(item, dict):
            continue
        try:
            traces.append(
                TraceSegment(
                    start=_point_from_dict(item.get("start")),
                    end=_point_from_dict(item.get("end")),
                    layer=_layer_from_value(item.get("layer")),
                    net=str(item.get("net", "")),
                    width_mm=float(item.get("width_mm", 0.127)),
                )
            )
        except Exception:
            continue
    return traces


def _parse_vias(canonical: dict[str, Any]) -> list[Via]:
    """Parse solved vias from canonical payload."""
    vias_payload = canonical.get("vias", [])
    if not isinstance(vias_payload, list):
        return []
    vias: list[Via] = []
    for item in vias_payload:
        if not isinstance(item, dict):
            continue
        try:
            vias.append(
                Via(
                    pos=_point_from_dict(item.get("pos")),
                    net=str(item.get("net", "")),
                    drill_mm=float(item.get("drill_mm", 0.3)),
                    size_mm=float(item.get("size_mm", 0.6)),
                )
            )
        except Exception:
            continue
    return vias


def _parse_silkscreen(canonical: dict[str, Any]) -> list[SilkscreenElement]:
    """Parse silkscreen elements from canonical payload."""
    payload = canonical.get("silkscreen", [])
    if not isinstance(payload, list):
        return []
    elements: list[SilkscreenElement] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", ""))
        layer = str(item.get("layer", "F.SilkS"))
        stroke_width = float(item.get("stroke_width", 0.15))
        if kind == "poly":
            points = [
                _point_from_dict(p) for p in item.get("points", [])
                if isinstance(p, dict)
            ]
            elements.append(SilkscreenElement(
                kind="poly",
                layer=layer,
                points=points,
                stroke_width=stroke_width,
            ))
        elif kind == "text":
            elements.append(SilkscreenElement(
                kind="text",
                layer=layer,
                text=str(item.get("text", "")),
                pos=_point_from_dict(item.get("pos")),
                font_height=float(item.get("font_height", 1.0)),
                font_width=float(item.get("font_width", 1.0)),
                font_thickness=float(item.get("font_thickness", 0.15)),
                stroke_width=stroke_width,
            ))
    return elements


def _parse_interface_anchors(
    canonical: dict[str, Any],
) -> list[InterfaceAnchor]:
    """Parse interface anchors from canonical payload."""
    anchors_payload = canonical.get("interface_anchors", [])
    if not isinstance(anchors_payload, list):
        return []
    anchors: list[InterfaceAnchor] = []
    for item in anchors_payload:
        if not isinstance(item, dict):
            continue
        pad_ref = item.get("pad_ref")
        anchors.append(
            InterfaceAnchor(
                port_name=str(item.get("port_name", "")),
                pos=_point_from_dict(item.get("pos")),
                layer=_layer_from_value(item.get("layer")),
                pad_ref=tuple(pad_ref) if isinstance(pad_ref, list) and len(pad_ref) == 2 else None,
            )
        )
    return anchors


def _parse_bbox(
    canonical: dict[str, Any],
    components: dict[str, Component],
) -> tuple[float, float]:
    """Parse layout bounding box from canonical payload."""
    bbox = canonical.get("bounding_box", {})
    width = bbox.get("width_mm") if isinstance(bbox, dict) else None
    height = bbox.get("height_mm") if isinstance(bbox, dict) else None
    if width is not None and height is not None:
        return (float(width), float(height))

    if components:
        tl, br = _compute_component_bbox(components)
        return (
            max(0.0, br.x - tl.x),
            max(0.0, br.y - tl.y),
        )

    return (0.0, 0.0)


def _parse_score(value: Any) -> float:
    """Parse a score value, returning 0.0 on failure."""
    if value is not None:
        try:
            return float(value)
        except Exception:
            pass
    return 0.0


def _component_from_dict(payload: dict[str, Any]) -> Component:
    """Reconstruct a `Component` from serialized artifact geometry."""
    pads = [
        _pad_from_dict(pad_payload)
        for pad_payload in payload.get("pads", [])
        if isinstance(pad_payload, dict)
    ]
    body_center_payload = payload.get("body_center")
    return Component(
        ref=str(payload.get("ref", "")),
        value=str(payload.get("value", "")),
        pos=_point_from_dict(payload.get("pos")),
        rotation=float(payload.get("rotation", 0.0)),
        layer=_layer_from_value(payload.get("layer")),
        width_mm=float(payload.get("width_mm", 0.0)),
        height_mm=float(payload.get("height_mm", 0.0)),
        pads=pads,
        locked=bool(payload.get("locked", False)),
        kind=str(payload.get("kind", "")),
        is_through_hole=bool(payload.get("is_through_hole", False)),
        body_center=(
            _point_from_dict(body_center_payload)
            if isinstance(body_center_payload, dict)
            else None
        ),
        opening_direction=(
            float(payload["opening_direction"])
            if payload.get("opening_direction") is not None
            else None
        ),
    )


def _pad_from_dict(payload: dict[str, Any]) -> Pad:
    """Reconstruct a `Pad` from serialized artifact geometry."""
    return Pad(
        ref=str(payload.get("ref", "")),
        pad_id=str(payload.get("pad_id", "")),
        pos=_point_from_dict(payload.get("pos")),
        net=str(payload.get("net", "")),
        layer=_layer_from_value(payload.get("layer")),
    )


def _point_from_dict(payload: Any) -> Point:
    """Reconstruct a `Point` from a serialized dict."""
    if not isinstance(payload, dict):
        return Point(0.0, 0.0)
    return Point(
        float(payload.get("x", 0.0)),
        float(payload.get("y", 0.0)),
    )


def _layer_from_value(value: Any) -> Layer:
    """Convert serialized layer strings back into `Layer`."""
    if str(value) == "B.Cu":
        return Layer.BACK
    return Layer.FRONT


def _transform_component(
    component: Component,
    origin: Point,
    rotation_deg: float,
) -> Component:
    """Apply rigid transform to a component and all dependent geometry."""
    new_component = copy.deepcopy(component)
    new_component.pos = _transform_point(component.pos, origin, rotation_deg)
    new_component.rotation = (component.rotation + rotation_deg) % 360.0

    if component.body_center is not None:
        new_component.body_center = _transform_point(
            component.body_center, origin, rotation_deg
        )

    new_component.pads = [
        _transform_pad(pad, origin, rotation_deg) for pad in component.pads
    ]
    return new_component


def _transform_pad(
    pad: Pad,
    origin: Point,
    rotation_deg: float,
) -> Pad:
    """Apply rigid transform to a pad."""
    new_pad = copy.deepcopy(pad)
    new_pad.pos = _transform_point(pad.pos, origin, rotation_deg)
    return new_pad


def _transform_trace(
    trace: TraceSegment,
    origin: Point,
    rotation_deg: float,
) -> TraceSegment:
    """Apply rigid transform to a trace segment."""
    new_trace = copy.deepcopy(trace)
    new_trace.start = _transform_point(trace.start, origin, rotation_deg)
    new_trace.end = _transform_point(trace.end, origin, rotation_deg)
    return new_trace


def _transform_via(
    via: Via,
    origin: Point,
    rotation_deg: float,
) -> Via:
    """Apply rigid transform to a via."""
    new_via = copy.deepcopy(via)
    new_via.pos = _transform_point(via.pos, origin, rotation_deg)
    return new_via


def _transform_anchor(
    anchor: InterfaceAnchor,
    origin: Point,
    rotation_deg: float,
) -> InterfaceAnchor:
    """Apply rigid transform to an interface anchor."""
    new_anchor = copy.deepcopy(anchor)
    new_anchor.pos = _transform_point(anchor.pos, origin, rotation_deg)
    return new_anchor


def _transform_silkscreen_element(
    element: SilkscreenElement,
    origin: Point,
    rotation_deg: float,
) -> SilkscreenElement:
    new_elem = copy.deepcopy(element)
    if element.kind == "poly":
        new_elem.points = [
            _transform_point(p, origin, rotation_deg) for p in element.points
        ]
    elif element.kind == "text":
        new_elem.pos = _transform_point(element.pos, origin, rotation_deg)
    return new_elem


def _transform_point(
    point: Point,
    origin: Point,
    rotation_deg: float,
) -> Point:
    """Rotate around local origin, then translate."""
    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x = point.x * cos_t - point.y * sin_t
    y = point.x * sin_t + point.y * cos_t
    return Point(x + origin.x, y + origin.y)


def _rotated_bbox_size(
    width: float,
    height: float,
    rotation_deg: float,
) -> tuple[float, float]:
    """Compute axis-aligned bbox size after rotation."""
    theta = math.radians(rotation_deg % 360.0)
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))
    return (
        width * cos_t + height * sin_t,
        width * sin_t + height * cos_t,
    )


def _compute_layout_bbox(
    components: dict[str, Component],
    traces: list[TraceSegment],
    vias: list[Via],
    anchors: list[InterfaceAnchor],
) -> tuple[Point, Point]:
    """Compute a tight bbox around transformed artifact geometry."""
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for comp in components.values():
        tl, br = comp.bbox()
        min_x = min(min_x, tl.x)
        min_y = min(min_y, tl.y)
        max_x = max(max_x, br.x)
        max_y = max(max_y, br.y)
        for pad in comp.pads:
            min_x = min(min_x, pad.pos.x)
            min_y = min(min_y, pad.pos.y)
            max_x = max(max_x, pad.pos.x)
            max_y = max(max_y, pad.pos.y)

    for trace in traces:
        min_x = min(min_x, trace.start.x, trace.end.x)
        min_y = min(min_y, trace.start.y, trace.end.y)
        max_x = max(max_x, trace.start.x, trace.end.x)
        max_y = max(max_y, trace.start.y, trace.end.y)

    for via in vias:
        min_x = min(min_x, via.pos.x)
        min_y = min(min_y, via.pos.y)
        max_x = max(max_x, via.pos.x)
        max_y = max(max_y, via.pos.y)

    for anchor in anchors:
        min_x = min(min_x, anchor.pos.x)
        min_y = min(min_y, anchor.pos.y)
        max_x = max(max_x, anchor.pos.x)
        max_y = max(max_y, anchor.pos.y)

    if min_x == float("inf"):
        return (Point(0.0, 0.0), Point(0.0, 0.0))

    return (Point(min_x, min_y), Point(max_x, max_y))


def _compute_component_bbox(
    components: dict[str, Component],
) -> tuple[Point, Point]:
    """Compute a tight bbox around components and pads only."""
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for comp in components.values():
        tl, br = comp.bbox()
        min_x = min(min_x, tl.x)
        min_y = min(min_y, tl.y)
        max_x = max(max_x, br.x)
        max_y = max(max_y, br.y)
        for pad in comp.pads:
            min_x = min(min_x, pad.pos.x)
            min_y = min(min_y, pad.pos.y)
            max_x = max(max_x, pad.pos.x)
            max_y = max(max_y, pad.pos.y)

    if min_x == float("inf"):
        return (Point(0.0, 0.0), Point(0.0, 0.0))

    return (Point(min_x, min_y), Point(max_x, max_y))


__all__ = [
    "LoadedSubcircuitArtifact",
    "TransformedSubcircuit",
    "artifact_debug_dict",
    "artifact_summary",
    "instantiate_subcircuit",
    "load_solved_artifact",
    "load_solved_artifacts",
    "transform_loaded_artifact",
    "transform_subcircuit_instance",
    "transformed_anchor_map",
    "transformed_component_map",
    "transformed_debug_dict",
    "transformed_summary",
]
