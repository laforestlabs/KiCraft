"""Leaf extraction and subcircuit artifact metadata helpers.

This module provides the artifact-layer utilities for the subcircuits
pipeline redesign. It is intentionally pure Python and does not depend on
pcbnew.

Current scope:
- derive leaf subcircuit extraction records from parsed hierarchy
- normalize artifact identifiers and output paths
- build JSON-serializable metadata payloads
- serialize solved component and copper geometry for reusable subcircuit artifacts
- save/load artifact metadata files
- save/load canonical solved layout artifact files
- compute stable cache-ish fingerprints from schematic/config inputs

This module does not generate high-fidelity routed `.kicad_pcb` files by
itself. It prepares the canonical machine-readable artifact side of the system
so later pipeline stages can persist both:
1. machine-readable solved layout artifacts
2. inspectable KiCad PCB snapshots
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .types import (
    Component,
    InterfaceAnchor,
    InterfacePort,
    Layer,
    Point,
    SilkscreenElement,
    SubcircuitAccessPolicy,
    SubCircuitDefinition,
    SubCircuitId,
    SubCircuitLayout,
    TraceSegment,
    Via,
)

ARTIFACT_SCHEMA_VERSION = "subcircuits.v1"
SOLVED_LAYOUT_SCHEMA_VERSION = "subcircuits.layout.v1"


@dataclass(slots=True)
class LeafExtraction:
    """Extracted leaf-level subcircuit ready for local solving."""

    subcircuit: SubCircuitDefinition
    project_dir: str
    schematic_path: str
    component_refs: list[str] = field(default_factory=list)
    interface_ports: list[InterfacePort] = field(default_factory=list)
    internal_nets: list[str] = field(default_factory=list)
    external_nets: list[str] = field(default_factory=list)
    local_board_outline: dict[str, float] = field(default_factory=dict)
    local_translation: dict[str, float] = field(default_factory=dict)
    internal_trace_count: int = 0
    internal_via_count: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def artifact_slug(self) -> str:
        return slugify_subcircuit_id(self.subcircuit.id)

    @property
    def is_leaf(self) -> bool:
        return self.subcircuit.is_leaf


@dataclass(slots=True)
class ArtifactPaths:
    """Resolved output paths for one subcircuit artifact bundle."""

    base_dir: str
    artifact_dir: str
    metadata_json: str
    mini_pcb: str
    debug_json: str
    solved_layout_json: str

    def to_dict(self) -> dict[str, str]:
        return {
            "base_dir": self.base_dir,
            "artifact_dir": self.artifact_dir,
            "metadata_json": self.metadata_json,
            "mini_pcb": self.mini_pcb,
            "debug_json": self.debug_json,
            "solved_layout_json": self.solved_layout_json,
        }


@dataclass(slots=True)
class ArtifactMetadata:
    """JSON-serializable metadata for a subcircuit artifact."""

    schema_version: str
    subcircuit_id: dict[str, Any]
    sheet_name: str
    sheet_file: str
    instance_path: str
    parent_instance_path: str | None
    project_dir: str
    schematic_path: str
    component_refs: list[str]
    interface_ports: list[dict[str, Any]]
    internal_nets: list[str]
    external_nets: list[str]
    local_board_outline: dict[str, float]
    local_translation: dict[str, float]
    internal_trace_count: int
    internal_via_count: int
    source_hash: str
    config_hash: str
    solver_version: str
    access_policy: str
    artifact_paths: dict[str, str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def serialize_point(point: Point) -> dict[str, float]:
    """Serialize a Point into a JSON-friendly dict."""
    return {
        "x": float(point.x),
        "y": float(point.y),
    }


def serialize_trace(trace: TraceSegment) -> dict[str, Any]:
    """Serialize a routed trace segment for artifact persistence."""
    return {
        "start": serialize_point(trace.start),
        "end": serialize_point(trace.end),
        "layer": serialize_layer(trace.layer),
        "net": trace.net,
        "width_mm": float(trace.width_mm),
        "length_mm": float(trace.length),
    }


def serialize_via(via: Via) -> dict[str, Any]:
    """Serialize a via for artifact persistence."""
    return {
        "pos": serialize_point(via.pos),
        "net": via.net,
        "drill_mm": float(via.drill_mm),
        "size_mm": float(via.size_mm),
    }


def serialize_interface_anchor(anchor: InterfaceAnchor) -> dict[str, Any]:
    """Serialize an inferred interface anchor."""
    return {
        "port_name": anchor.port_name,
        "pos": serialize_point(anchor.pos),
        "layer": serialize_layer(anchor.layer),
        "pad_ref": list(anchor.pad_ref) if anchor.pad_ref is not None else None,
    }


def serialize_interface_port(port: InterfacePort) -> dict[str, Any]:
    """Serialize a logical interface port for solved-layout persistence."""
    return _port_to_dict(port)


def serialize_layer(layer: Layer) -> str:
    """Serialize a copper layer enum into a stable string."""
    return "B.Cu" if layer == Layer.BACK else "F.Cu"


def serialize_component(component: Component) -> dict[str, Any]:
    """Serialize solved component geometry for artifact/debug persistence."""
    return {
        "ref": component.ref,
        "value": component.value,
        "pos": serialize_point(component.pos),
        "rotation": float(component.rotation),
        "layer": serialize_layer(component.layer),
        "width_mm": float(component.width_mm),
        "height_mm": float(component.height_mm),
        "locked": bool(component.locked),
        "kind": component.kind,
        "is_through_hole": bool(component.is_through_hole),
        "body_center": (
            serialize_point(component.body_center)
            if component.body_center is not None
            else None
        ),
        "opening_direction": (
            float(component.opening_direction)
            if component.opening_direction is not None
            else None
        ),
        "pads": [
            {
                "ref": pad.ref,
                "pad_id": pad.pad_id,
                "pos": serialize_point(pad.pos),
                "net": pad.net,
                "layer": serialize_layer(pad.layer),
                "size_mm": (
                    serialize_point(pad.size_mm)
                    if pad.size_mm is not None
                    else None
                ),
            }
            for pad in component.pads
        ],
    }


def serialize_components(components: dict[str, Component]) -> dict[str, dict[str, Any]]:
    """Serialize a solved component map keyed by reference."""
    return {
        ref: serialize_component(component)
        for ref, component in sorted(components.items())
    }


def serialize_traces(traces: list[TraceSegment]) -> list[dict[str, Any]]:
    """Serialize solved trace geometry."""
    return [serialize_trace(trace) for trace in traces]


def serialize_vias(vias: list[Via]) -> list[dict[str, Any]]:
    """Serialize solved via geometry."""
    return [serialize_via(via) for via in vias]


def serialize_silkscreen_element(element: SilkscreenElement) -> dict[str, Any]:
    """Serialize one silkscreen element for artifact persistence."""
    d: dict[str, Any] = {
        "kind": element.kind,
        "layer": element.layer,
        "stroke_width": float(element.stroke_width),
    }
    if element.kind == "poly":
        d["points"] = [serialize_point(p) for p in element.points]
    elif element.kind == "text":
        d["text"] = element.text
        d["pos"] = serialize_point(element.pos)
        d["font_height"] = float(element.font_height)
        d["font_width"] = float(element.font_width)
        d["font_thickness"] = float(element.font_thickness)
    return d


def serialize_silkscreen(elements: list[SilkscreenElement]) -> list[dict[str, Any]]:
    """Serialize silkscreen elements list."""
    return [serialize_silkscreen_element(e) for e in elements]


def serialize_interface_anchors(
    anchors: list[InterfaceAnchor],
) -> list[dict[str, Any]]:
    """Serialize solved interface anchors."""
    return [serialize_interface_anchor(anchor) for anchor in anchors]


def serialize_interface_ports(
    ports: list[InterfacePort],
) -> list[dict[str, Any]]:
    """Serialize solved logical interface ports."""
    return [serialize_interface_port(port) for port in ports]


def build_anchor_validation(
    ports: list[InterfacePort],
    anchors: list[InterfaceAnchor],
) -> dict[str, Any]:
    """Build validation details for solved interface anchors."""
    required_ports = [port.name for port in ports if port.required]
    optional_ports = [port.name for port in ports if not port.required]
    anchor_port_names = [anchor.port_name for anchor in anchors]

    required_set = set(required_ports)
    anchor_set = set(anchor_port_names)

    missing_required = sorted(required_set - anchor_set)
    anchored_required = sorted(required_set & anchor_set)
    anchored_optional = sorted(set(optional_ports) & anchor_set)
    extra_anchors = sorted(anchor_set - {port.name for port in ports})

    return {
        "port_count": len(ports),
        "required_port_count": len(required_ports),
        "optional_port_count": len(optional_ports),
        "anchor_count": len(anchors),
        "anchored_required_ports": anchored_required,
        "anchored_optional_ports": anchored_optional,
        "missing_required_ports": missing_required,
        "extra_anchor_ports": extra_anchors,
        "all_required_ports_anchored": not missing_required,
    }


def slugify_subcircuit_id(subcircuit_id: SubCircuitId) -> str:
    """Create a filesystem-safe slug for a subcircuit instance."""
    raw = (
        subcircuit_id.path_key
        if hasattr(subcircuit_id, "path_key")
        else subcircuit_id.instance_path
    )
    raw = raw or subcircuit_id.sheet_file or subcircuit_id.sheet_name or "subcircuit"
    cleaned = []
    for ch in raw:
        if ch.isalnum():
            cleaned.append(ch.lower())
        elif ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    slug = "".join(cleaned).strip("_")
    if not slug:
        slug = "subcircuit"
    while "__" in slug:
        slug = slug.replace("__", "_")
    return f"{slug}__{_short_hash(subcircuit_id.instance_path or subcircuit_id.sheet_name)}"


def artifact_root_dir(project_dir: str | Path) -> Path:
    """Return the default artifact root directory for subcircuits."""
    return Path(project_dir) / ".experiments" / "subcircuits"


def resolve_artifact_paths(
    project_dir: str | Path,
    subcircuit_id: SubCircuitId,
) -> ArtifactPaths:
    """Resolve standard artifact output paths for one subcircuit."""
    base_dir = artifact_root_dir(project_dir)
    slug = slugify_subcircuit_id(subcircuit_id)
    artifact_dir = base_dir / slug
    return ArtifactPaths(
        base_dir=str(base_dir),
        artifact_dir=str(artifact_dir),
        metadata_json=str(artifact_dir / "metadata.json"),
        mini_pcb=str(artifact_dir / "layout.kicad_pcb"),
        debug_json=str(artifact_dir / "debug.json"),
        solved_layout_json=str(artifact_dir / "solved_layout.json"),
    )


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def stable_json_hash(value: Any) -> str:
    """Hash arbitrary JSON-serializable data with stable key ordering."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Compute SHA-256 for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_source_hash(
    schematic_path: str | Path,
    component_refs: list[str] | None = None,
    interface_ports: list[InterfacePort] | None = None,
) -> str:
    """Compute a stable source fingerprint for a leaf extraction.

    The hash includes:
    - schematic file contents
    - component membership
    - normalized interface names/roles/directions
    """
    schematic_hash = file_sha256(schematic_path)
    refs = sorted(component_refs or [])
    ports = [
        {
            "name": p.name,
            "net_name": p.net_name,
            "role": getattr(p.role, "value", str(p.role)),
            "direction": getattr(p.direction, "value", str(p.direction)),
            "preferred_side": getattr(p.preferred_side, "value", str(p.preferred_side)),
            "access_policy": getattr(p.access_policy, "value", str(p.access_policy)),
            "cardinality": p.cardinality,
            "bus_index": p.bus_index,
            "required": p.required,
        }
        for p in (interface_ports or [])
    ]
    return stable_json_hash(
        {
            "schematic_sha256": schematic_hash,
            "component_refs": refs,
            "interface_ports": ports,
        }
    )


def compute_config_hash(config: dict[str, Any] | None) -> str:
    """Compute a stable config fingerprint."""
    return stable_json_hash(config or {})


def _port_to_dict(port: InterfacePort) -> dict[str, Any]:
    return {
        "name": port.name,
        "net_name": port.net_name,
        "role": getattr(port.role, "value", str(port.role)),
        "direction": getattr(port.direction, "value", str(port.direction)),
        "preferred_side": getattr(
            port.preferred_side, "value", str(port.preferred_side)
        ),
        "access_policy": getattr(port.access_policy, "value", str(port.access_policy)),
        "cardinality": port.cardinality,
        "bus_index": port.bus_index,
        "required": port.required,
        "description": getattr(port, "description", ""),
    }


def _subcircuit_id_to_dict(subcircuit_id: SubCircuitId) -> dict[str, Any]:
    return {
        "sheet_name": subcircuit_id.sheet_name,
        "sheet_file": subcircuit_id.sheet_file,
        "instance_path": subcircuit_id.instance_path,
        "parent_instance_path": subcircuit_id.parent_instance_path,
    }


def build_leaf_extraction(
    subcircuit: SubCircuitDefinition,
    project_dir: str | Path,
    internal_nets: list[str] | None = None,
    external_nets: list[str] | None = None,
    local_board_outline: dict[str, float] | None = None,
    local_translation: dict[str, float] | None = None,
    internal_trace_count: int = 0,
    internal_via_count: int = 0,
    notes: list[str] | None = None,
) -> LeafExtraction:
    """Build a leaf extraction record from a parsed subcircuit definition."""
    if not subcircuit.is_leaf:
        raise ValueError(
            f"Subcircuit '{subcircuit.id.sheet_name}' is not a leaf and cannot be extracted as a leaf artifact"
        )

    return LeafExtraction(
        subcircuit=subcircuit,
        project_dir=str(project_dir),
        schematic_path=getattr(subcircuit, "schematic_path", ""),
        component_refs=list(subcircuit.component_refs),
        interface_ports=list(subcircuit.ports),
        internal_nets=sorted(set(internal_nets or [])),
        external_nets=sorted(
            set(external_nets or [p.net_name for p in subcircuit.ports])
        ),
        local_board_outline=dict(local_board_outline or {}),
        local_translation=dict(local_translation or {}),
        internal_trace_count=internal_trace_count,
        internal_via_count=internal_via_count,
        notes=list(notes or []),
    )


def build_artifact_metadata(
    extraction: LeafExtraction,
    config: dict[str, Any] | None = None,
    solver_version: str = "subcircuits-m1",
    access_policy: SubcircuitAccessPolicy = SubcircuitAccessPolicy.INTERFACE_ONLY,
) -> ArtifactMetadata:
    """Build artifact metadata for a leaf extraction."""
    paths = resolve_artifact_paths(extraction.project_dir, extraction.subcircuit.id)
    source_hash = compute_source_hash(
        extraction.schematic_path,
        extraction.component_refs,
        extraction.interface_ports,
    )
    config_hash = compute_config_hash(config)

    return ArtifactMetadata(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        subcircuit_id=_subcircuit_id_to_dict(extraction.subcircuit.id),
        sheet_name=extraction.subcircuit.id.sheet_name,
        sheet_file=extraction.subcircuit.id.sheet_file,
        instance_path=extraction.subcircuit.id.instance_path,
        parent_instance_path=extraction.subcircuit.id.parent_instance_path,
        project_dir=extraction.project_dir,
        schematic_path=extraction.schematic_path,
        component_refs=list(extraction.component_refs),
        interface_ports=[_port_to_dict(p) for p in extraction.interface_ports],
        internal_nets=list(extraction.internal_nets),
        external_nets=list(extraction.external_nets),
        local_board_outline=dict(extraction.local_board_outline),
        local_translation=dict(extraction.local_translation),
        internal_trace_count=extraction.internal_trace_count,
        internal_via_count=extraction.internal_via_count,
        source_hash=source_hash,
        config_hash=config_hash,
        solver_version=solver_version,
        access_policy=getattr(access_policy, "value", str(access_policy)),
        artifact_paths=paths.to_dict(),
        notes=list(extraction.notes),
    )


def ensure_artifact_dirs(paths: ArtifactPaths) -> None:
    """Create artifact directories if they do not already exist."""
    Path(paths.artifact_dir).mkdir(parents=True, exist_ok=True)


def build_solved_layout_artifact(
    layout: SubCircuitLayout,
    *,
    project_dir: str | Path,
    source_hash: str = "",
    config_hash: str = "",
    solver_version: str = "",
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Build the canonical solved layout artifact payload."""
    paths = resolve_artifact_paths(project_dir, layout.subcircuit_id)
    anchor_validation = build_anchor_validation(
        layout.ports,
        layout.interface_anchors,
    )
    combined_notes = list(notes or [])
    combined_notes.append(f"interface_port_count={len(layout.ports)}")
    combined_notes.append(f"interface_anchor_count={len(layout.interface_anchors)}")
    combined_notes.append(
        "all_required_ports_anchored="
        f"{anchor_validation['all_required_ports_anchored']}"
    )
    if anchor_validation["missing_required_ports"]:
        combined_notes.append(
            "missing_required_ports="
            + ",".join(anchor_validation["missing_required_ports"])
        )

    return {
        "schema_version": SOLVED_LAYOUT_SCHEMA_VERSION,
        "subcircuit_id": _subcircuit_id_to_dict(layout.subcircuit_id),
        "sheet_name": layout.subcircuit_id.sheet_name,
        "sheet_file": layout.subcircuit_id.sheet_file,
        "instance_path": layout.subcircuit_id.instance_path,
        "parent_instance_path": layout.subcircuit_id.parent_instance_path,
        "project_dir": str(project_dir),
        "source_hash": source_hash,
        "config_hash": config_hash,
        "solver_version": solver_version,
        "frozen": bool(layout.frozen),
        "score": float(layout.score),
        "bounding_box": {
            "width_mm": float(layout.width),
            "height_mm": float(layout.height),
        },
        "components": serialize_components(layout.components),
        "traces": serialize_traces(layout.traces),
        "vias": serialize_vias(layout.vias),
        "silkscreen": serialize_silkscreen(layout.silkscreen),
        "ports": serialize_interface_ports(layout.ports),
        "interface_anchors": serialize_interface_anchors(layout.interface_anchors),
        "anchor_validation": anchor_validation,
        "artifact_paths": paths.to_dict(),
        "notes": combined_notes,
    }


def save_solved_layout_artifact(payload: dict[str, Any]) -> str:
    """Write a canonical solved layout artifact JSON file to disk."""
    artifact_paths = payload.get("artifact_paths", {})
    out_path = Path(artifact_paths["solved_layout_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(out_path)


def load_solved_layout_artifact(path: str | Path) -> dict[str, Any]:
    """Load a canonical solved layout artifact JSON file as a plain dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def solved_layout_is_compatible(payload: dict[str, Any]) -> bool:
    """Check whether a solved layout payload matches the current schema."""
    return payload.get("schema_version") == SOLVED_LAYOUT_SCHEMA_VERSION


def save_artifact_metadata(metadata: ArtifactMetadata) -> str:
    """Write artifact metadata JSON to disk and return the path."""
    out_path = Path(metadata.artifact_paths["metadata_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(out_path)


def save_debug_payload(
    extraction: LeafExtraction,
    metadata: ArtifactMetadata,
    extra: dict[str, Any] | None = None,
) -> str:
    """Write a debug JSON payload alongside metadata."""
    debug_path = Path(metadata.artifact_paths["debug_json"])
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "leaf_extraction": {
            "artifact_slug": extraction.artifact_slug,
            "project_dir": extraction.project_dir,
            "schematic_path": extraction.schematic_path,
            "component_refs": extraction.component_refs,
            "interface_ports": [_port_to_dict(p) for p in extraction.interface_ports],
            "internal_nets": extraction.internal_nets,
            "external_nets": extraction.external_nets,
            "local_board_outline": extraction.local_board_outline,
            "local_translation": extraction.local_translation,
            "internal_trace_count": extraction.internal_trace_count,
            "internal_via_count": extraction.internal_via_count,
            "notes": extraction.notes,
        },
        "metadata": metadata.to_dict(),
    }
    if (
        extra
        and "solved_components" in extra
        and isinstance(extra["solved_components"], dict)
    ):
        payload["solved_components"] = serialize_components(extra["solved_components"])
    if extra:
        payload["extra"] = extra
    debug_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(debug_path)


def load_artifact_metadata(path: str | Path) -> dict[str, Any]:
    """Load artifact metadata JSON as a plain dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metadata_is_compatible(payload: dict[str, Any]) -> bool:
    """Check whether a loaded metadata payload matches the current schema."""
    return payload.get("schema_version") == ARTIFACT_SCHEMA_VERSION


def extraction_summary(extraction: LeafExtraction) -> str:
    """Human-readable one-line summary for logs/debug output."""
    return (
        f"{extraction.subcircuit.id.sheet_name} "
        f"[{extraction.subcircuit.id.instance_path}] "
        f"refs={len(extraction.component_refs)} "
        f"ports={len(extraction.interface_ports)} "
        f"internal_nets={len(extraction.internal_nets)} "
        f"external_nets={len(extraction.external_nets)} "
        f"traces={extraction.internal_trace_count} "
        f"vias={extraction.internal_via_count}"
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "SOLVED_LAYOUT_SCHEMA_VERSION",
    "ArtifactMetadata",
    "ArtifactPaths",
    "LeafExtraction",
    "artifact_root_dir",
    "build_artifact_metadata",
    "build_leaf_extraction",
    "build_solved_layout_artifact",
    "compute_config_hash",
    "compute_source_hash",
    "ensure_artifact_dirs",
    "extraction_summary",
    "file_sha256",
    "load_artifact_metadata",
    "load_solved_layout_artifact",
    "metadata_is_compatible",
    "resolve_artifact_paths",
    "save_artifact_metadata",
    "save_debug_payload",
    "save_solved_layout_artifact",
    "serialize_component",
    "serialize_components",
    "serialize_interface_anchor",
    "serialize_interface_anchors",
    "serialize_layer",
    "serialize_point",
    "serialize_silkscreen",
    "serialize_silkscreen_element",
    "serialize_trace",
    "serialize_traces",
    "serialize_via",
    "serialize_vias",
    "slugify_subcircuit_id",
    "solved_layout_is_compatible",
    "stable_json_hash",
]
