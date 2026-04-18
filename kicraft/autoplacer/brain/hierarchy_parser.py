"""True-sheet schematic hierarchy parser and interface normalization.

This module parses KiCad hierarchical schematics into a strict subcircuit tree.
It is intentionally pure Python and does not depend on pcbnew.

Design goals:
- Use true KiCad schematic sheets as the source of grouping truth
- Identify leaf vs non-leaf sheets
- Extract component membership per sheet
- Normalize sheet pins into typed interface ports
- Reuse shared subcircuit/interface dataclasses from `types.py`
- Provide richer metadata for later artifact generation and composition
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .types import (
    InterfaceDirection,
    InterfacePort,
    InterfaceRole,
    InterfaceSide,
    SubcircuitAccessPolicy,
    SubCircuitDefinition,
    SubCircuitId,
)

# ---------------------------------------------------------------------------
# S-expression mini-parser
# ---------------------------------------------------------------------------


def _tokenize(text: str):
    """Yield tokens: '(', ')', or quoted/unquoted strings."""
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t\n\r":
            i += 1
        elif c == "(":
            yield "("
            i += 1
        elif c == ")":
            yield ")"
            i += 1
        elif c == '"':
            j = i + 1
            buf = []
            while j < n:
                if text[j] == "\\" and j + 1 < n:
                    buf.append(text[j + 1])
                    j += 2
                    continue
                if text[j] == '"':
                    break
                buf.append(text[j])
                j += 1
            yield "".join(buf)
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in " \t\n\r()":
                j += 1
            yield text[i:j]
            i = j


def _parse_sexpr(tokens) -> list:
    """Parse tokenized S-expression into nested lists."""
    result = []
    for tok in tokens:
        if tok == "(":
            result.append(_parse_sexpr(tokens))
        elif tok == ")":
            return result
        else:
            result.append(tok)
    return result


def _parse_file(path: str | Path) -> list:
    """Parse a KiCad schematic file into a nested list tree."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = _tokenize(text)
    for tok in tokens:
        if tok == "(":
            return _parse_sexpr(tokens)
    return []


def _find_nodes(tree: list, tag: str) -> list[list]:
    """Find direct child nodes with the given tag."""
    results = []
    for item in tree:
        if isinstance(item, list) and item and item[0] == tag:
            results.append(item)
    return results


def _find_node(tree: list, tag: str):
    """Find the first direct child node with the given tag."""
    for item in tree:
        if isinstance(item, list) and item and item[0] == tag:
            return item
    return None


def _get_property(tree: list, name: str) -> str | None:
    """Get the value of a `(property "name" "value" ...)` node."""
    for item in tree:
        if isinstance(item, list) and item and item[0] == "property":
            if len(item) >= 3 and item[1] == name:
                return item[2]
    return None


# ---------------------------------------------------------------------------
# Internal metadata structures
# ---------------------------------------------------------------------------


_REF_RE = re.compile(r"^[A-Z]+[0-9]+[A-Z0-9_-]*$", re.IGNORECASE)


@dataclass(slots=True)
class SheetPinMetadata:
    """Raw sheet pin metadata extracted from a `(sheet ...)` node."""

    name: str
    direction: str
    uuid: str | None = None
    angle_deg: float | None = None
    at_x: float | None = None
    at_y: float | None = None


@dataclass(slots=True)
class SheetNodeMetadata:
    """Raw metadata for one `(sheet ...)` node before normalization."""

    sheet_name: str
    sheet_file: str
    sheet_uuid: str | None
    instance_path: str
    parent_instance_path: str | None
    pins: list[SheetPinMetadata] = field(default_factory=list)
    source_schematic_path: str = ""


@dataclass(slots=True)
class HierarchyNode:
    """Tree node for one sheet instance."""

    definition: SubCircuitDefinition
    children: list["HierarchyNode"] = field(default_factory=list)

    @property
    def id(self) -> SubCircuitId:
        return self.definition.id

    @property
    def is_leaf(self) -> bool:
        return self.definition.is_leaf


@dataclass(slots=True)
class HierarchyGraph:
    """Complete parsed hierarchy for a project schematic."""

    project_dir: str
    root_schematic_path: str
    root: HierarchyNode
    nodes_by_path: dict[str, HierarchyNode] = field(default_factory=dict)

    def iter_nodes(self) -> Iterable[HierarchyNode]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def leaf_nodes(self) -> list[HierarchyNode]:
        return [node for node in self.iter_nodes() if node.is_leaf]

    def non_leaf_nodes(self) -> list[HierarchyNode]:
        return [node for node in self.iter_nodes() if not node.is_leaf]

    def summary_lines(self) -> list[str]:
        lines = []
        for node in self.iter_nodes():
            kind = "leaf" if node.is_leaf else "composite"
            lines.append(
                f"{node.id.instance_path or '/'}: {node.id.sheet_name} "
                f"[{kind}] comps={len(node.definition.component_refs)} "
                f"ports={len(node.definition.ports)} children={len(node.children)}"
            )
        return lines


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _sheet_pin_direction(raw: str) -> InterfaceDirection:
    raw_norm = (raw or "").strip().lower()
    if raw_norm == "input":
        return InterfaceDirection.INPUT
    if raw_norm == "output":
        return InterfaceDirection.OUTPUT
    if raw_norm == "bidirectional":
        return InterfaceDirection.BIDIRECTIONAL
    if raw_norm == "passive":
        return InterfaceDirection.PASSIVE
    return InterfaceDirection.UNKNOWN


def _infer_port_role(name: str, direction: InterfaceDirection) -> InterfaceRole:
    n = (name or "").strip().upper()

    if not n:
        return InterfaceRole.UNKNOWN

    if n in {"GND", "PGND", "AGND", "DGND", "EARTH"} or n.endswith("_GND"):
        return InterfaceRole.GROUND

    power_names = {
        "VBUS",
        "VBAT",
        "VSYS",
        "VIN",
        "VCC",
        "VDD",
        "VAA",
        "5V",
        "3V3",
        "3.3V",
        "12V",
        "24V",
        "1V8",
        "1V2",
    }
    if n in power_names or n.startswith("+"):
        if direction == InterfaceDirection.OUTPUT:
            return InterfaceRole.POWER_OUT
        if direction == InterfaceDirection.INPUT:
            return InterfaceRole.POWER_IN
        return InterfaceRole.BIDIR

    if "[" in n and "]" in n:
        return InterfaceRole.BUS

    if n.startswith("TEST") or n.startswith("TP"):
        return InterfaceRole.TEST

    analog_tokens = ("ADC", "DAC", "SENSE", "NTC", "TEMP", "REF", "FB")
    if any(tok in n for tok in analog_tokens):
        return InterfaceRole.ANALOG

    if direction == InterfaceDirection.INPUT:
        return InterfaceRole.SIGNAL_IN
    if direction == InterfaceDirection.OUTPUT:
        return InterfaceRole.SIGNAL_OUT
    if direction in (InterfaceDirection.BIDIRECTIONAL, InterfaceDirection.PASSIVE):
        return InterfaceRole.BIDIR

    return InterfaceRole.UNKNOWN


def _infer_preferred_side(pin_name: str, angle_deg: float | None) -> InterfaceSide:
    if angle_deg is not None:
        angle = int(round(angle_deg)) % 360
        if angle == 0:
            return InterfaceSide.RIGHT
        if angle == 90:
            return InterfaceSide.TOP
        if angle == 180:
            return InterfaceSide.LEFT
        if angle == 270:
            return InterfaceSide.BOTTOM

    n = (pin_name or "").strip().upper()
    if n in {"VBUS", "VIN", "USB_IN", "INPUT"}:
        return InterfaceSide.LEFT
    if n in {"5V", "3V3", "3.3V", "VOUT", "OUTPUT"}:
        return InterfaceSide.RIGHT
    return InterfaceSide.ANY


def _parse_bus_index(name: str) -> int | None:
    if "[" not in name or "]" not in name:
        return None
    try:
        inside = name[name.index("[") + 1 : name.index("]")]
        return int(inside)
    except Exception:
        return None


def _normalize_sheet_pin(pin: SheetPinMetadata) -> InterfacePort:
    direction = _sheet_pin_direction(pin.direction)
    return InterfacePort(
        name=pin.name,
        role=_infer_port_role(pin.name, direction),
        direction=direction,
        net_name=pin.name,
        cardinality=1,
        preferred_side=_infer_preferred_side(pin.name, pin.angle_deg),
        access_policy=SubcircuitAccessPolicy.INTERFACE_ONLY,
        bus_index=_parse_bus_index(pin.name),
        required=True,
        description="normalized from schematic sheet pin",
        raw_direction=pin.direction,
        source_uuid=pin.uuid,
        source_kind="sheet_pin",
    )


def _extract_sheet_pins(sheet_node: list) -> list[SheetPinMetadata]:
    pins = []
    for pin_node in _find_nodes(sheet_node, "pin"):
        if len(pin_node) < 3:
            continue

        name = pin_node[1] if isinstance(pin_node[1], str) else ""
        direction = pin_node[2] if isinstance(pin_node[2], str) else "unknown"

        uuid_node = _find_node(pin_node, "uuid")
        uuid = uuid_node[1] if uuid_node and len(uuid_node) >= 2 else None

        at_node = _find_node(pin_node, "at")
        at_x = at_y = angle_deg = None
        if at_node and len(at_node) >= 4:
            try:
                at_x = float(at_node[1])
                at_y = float(at_node[2])
                angle_deg = float(at_node[3])
            except Exception:
                at_x = at_y = angle_deg = None

        pins.append(
            SheetPinMetadata(
                name=name,
                direction=direction,
                uuid=uuid,
                angle_deg=angle_deg,
                at_x=at_x,
                at_y=at_y,
            )
        )
    return pins


def _extract_symbol_refs(sheet_tree: list) -> list[str]:
    """Extract placed component references from a schematic file.

    Filters out:
    - power symbols (`power:*`)
    - pseudo references beginning with `#`
    - malformed references
    """
    refs: list[str] = []

    for sym in _find_nodes(sheet_tree, "symbol"):
        lib_id_node = _find_node(sym, "lib_id")
        if lib_id_node and len(lib_id_node) >= 2:
            lib_id = lib_id_node[1]
            if isinstance(lib_id, str) and lib_id.startswith("power:"):
                continue

        instances = _find_node(sym, "instances")
        if instances is None:
            continue

        for project_node in _find_nodes(instances, "project"):
            for path_node in _find_nodes(project_node, "path"):
                ref_node = _find_node(path_node, "reference")
                if not ref_node or len(ref_node) < 2:
                    continue
                ref = ref_node[1]
                if not isinstance(ref, str):
                    continue
                if not ref or ref.startswith("#"):
                    continue
                if not _REF_RE.match(ref):
                    continue
                refs.append(ref)

    seen = set()
    ordered = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        ordered.append(ref)
    return ordered


def _sheet_name(sheet_node: list, fallback: str) -> str:
    return _get_property(sheet_node, "Sheetname") or fallback


def _sheet_file(sheet_node: list) -> str | None:
    return _get_property(sheet_node, "Sheetfile")


def _sheet_uuid(sheet_node: list) -> str | None:
    uuid_node = _find_node(sheet_node, "uuid")
    if uuid_node and len(uuid_node) >= 2 and isinstance(uuid_node[1], str):
        return uuid_node[1]
    return None


def _root_project_name(root_tree: list, fallback: str) -> str:
    title_block = _find_node(root_tree, "title_block")
    if title_block:
        title = _get_property(title_block, "title")
        if title:
            return title
    return fallback


def _extract_child_sheet_metadata(
    parent_tree: list,
    parent_instance_path: str,
    source_schematic_path: str,
) -> list[SheetNodeMetadata]:
    children: list[SheetNodeMetadata] = []

    for idx, sheet_node in enumerate(_find_nodes(parent_tree, "sheet"), start=1):
        sheet_name = _sheet_name(sheet_node, f"sheet_{idx}")
        sheet_file = _sheet_file(sheet_node)
        if not sheet_file:
            continue

        sheet_uuid = _sheet_uuid(sheet_node)
        child_instance_path = (
            (
                f"{parent_instance_path}/{sheet_uuid}"
                if parent_instance_path != "/"
                else f"/{sheet_uuid}"
            )
            if sheet_uuid
            else (
                f"{parent_instance_path}/{idx}"
                if parent_instance_path != "/"
                else f"/{idx}"
            )
        )

        children.append(
            SheetNodeMetadata(
                sheet_name=sheet_name,
                sheet_file=sheet_file,
                sheet_uuid=sheet_uuid,
                instance_path=child_instance_path,
                parent_instance_path=parent_instance_path,
                pins=_extract_sheet_pins(sheet_node),
                source_schematic_path=source_schematic_path,
            )
        )

    return children


# ---------------------------------------------------------------------------
# Recursive hierarchy parsing
# ---------------------------------------------------------------------------


def _build_node(
    project_dir: Path,
    sheet_meta: SheetNodeMetadata,
) -> HierarchyNode:
    schematic_path = (project_dir / sheet_meta.sheet_file).resolve()
    tree = _parse_file(schematic_path)

    child_sheet_meta = _extract_child_sheet_metadata(
        tree,
        parent_instance_path=sheet_meta.instance_path,
        source_schematic_path=str(schematic_path),
    )
    child_nodes = [_build_node(project_dir, child) for child in child_sheet_meta]

    definition = SubCircuitDefinition(
        id=SubCircuitId(
            sheet_name=sheet_meta.sheet_name,
            sheet_file=sheet_meta.sheet_file,
            instance_path=sheet_meta.instance_path,
            parent_instance_path=sheet_meta.parent_instance_path,
        ),
        schematic_path=str(schematic_path),
        component_refs=_extract_symbol_refs(tree),
        ports=[_normalize_sheet_pin(pin) for pin in sheet_meta.pins],
        child_ids=[child.definition.id for child in child_nodes],
        parent_id=SubCircuitId(
            sheet_name="",
            sheet_file="",
            instance_path=sheet_meta.parent_instance_path,
            parent_instance_path=None,
        )
        if sheet_meta.parent_instance_path
        else None,
        is_leaf=(len(child_nodes) == 0),
        sheet_uuid=sheet_meta.sheet_uuid or "",
        notes=[
            f"source_schematic={sheet_meta.source_schematic_path}",
            f"resolved_schematic={schematic_path}",
        ],
    )

    return HierarchyNode(definition=definition, children=child_nodes)


def _build_root_node(project_dir: Path, root_schematic_path: Path) -> HierarchyNode:
    root_tree = _parse_file(root_schematic_path)
    root_name = _root_project_name(root_tree, root_schematic_path.stem)

    top_children_meta = _extract_child_sheet_metadata(
        root_tree,
        parent_instance_path="/",
        source_schematic_path=str(root_schematic_path),
    )
    child_nodes = [_build_node(project_dir, child) for child in top_children_meta]

    root_definition = SubCircuitDefinition(
        id=SubCircuitId(
            sheet_name=root_name,
            sheet_file=root_schematic_path.name,
            instance_path="/",
            parent_instance_path=None,
        ),
        schematic_path=str(root_schematic_path),
        component_refs=_extract_symbol_refs(root_tree),
        ports=[],
        child_ids=[child.definition.id for child in child_nodes],
        parent_id=None,
        is_leaf=(len(child_nodes) == 0),
        sheet_uuid="",
        notes=[f"root_schematic={root_schematic_path}"],
    )
    return HierarchyNode(definition=root_definition, children=child_nodes)


def _index_nodes(root: HierarchyNode) -> dict[str, HierarchyNode]:
    indexed: dict[str, HierarchyNode] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        indexed[node.id.instance_path] = node
        stack.extend(reversed(node.children))
    return indexed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_top_level_schematic(project_dir: str | Path) -> Path:
    """Find the top-level schematic file for a project directory.

    Preference order:
    1. A `.kicad_sch` file containing at least one `(sheet ...)` node
    2. A file named after the directory stem
    3. The first `.kicad_sch` file alphabetically
    """
    project_path = Path(project_dir)
    sch_files = sorted(project_path.glob("*.kicad_sch"))
    if not sch_files:
        raise FileNotFoundError(f"No .kicad_sch files found in {project_path}")

    for sch_file in sch_files:
        try:
            tree = _parse_file(sch_file)
            if _find_nodes(tree, "sheet"):
                return sch_file
        except Exception:
            continue

    preferred = project_path / f"{project_path.name}.kicad_sch"
    if preferred.exists():
        return preferred

    return sch_files[0]


def parse_hierarchy(
    project_dir: str | Path, top_schematic: str | Path | None = None
) -> HierarchyGraph:
    """Parse a project into a strict true-sheet hierarchy graph."""
    project_path = Path(project_dir).resolve()
    root_schematic = (
        Path(top_schematic).resolve()
        if top_schematic
        else find_top_level_schematic(project_path)
    )

    root = _build_root_node(project_path, root_schematic)
    return HierarchyGraph(
        project_dir=str(project_path),
        root_schematic_path=str(root_schematic),
        root=root,
        nodes_by_path=_index_nodes(root),
    )


def format_hierarchy_tree(graph: HierarchyGraph) -> str:
    """Render a human-readable hierarchy tree."""
    lines: list[str] = []

    def walk(node: HierarchyNode, prefix: str = "") -> None:
        kind = "leaf" if node.is_leaf else "composite"
        lines.append(
            f"{prefix}{node.id.sheet_name} "
            f"({node.id.instance_path}, {kind}, "
            f"comps={len(node.definition.component_refs)}, "
            f"ports={len(node.definition.ports)})"
        )
        for port in node.definition.ports:
            lines.append(
                f"{prefix}  - port {port.name}: "
                f"role={port.role.value} dir={port.direction.value} "
                f"side={port.preferred_side.value} access={port.access_policy.value}"
            )
        for child in node.children:
            walk(child, prefix + "  ")

    walk(graph.root)
    return "\n".join(lines)


def hierarchy_debug_dict(graph: HierarchyGraph) -> dict:
    """Return a JSON-serializable debug view of the hierarchy."""

    def node_to_dict(node: HierarchyNode) -> dict:
        return {
            "id": {
                "instance_path": node.id.instance_path,
                "sheet_name": node.id.sheet_name,
                "sheet_file": node.id.sheet_file,
                "parent_instance_path": node.id.parent_instance_path,
            },
            "is_leaf": node.definition.is_leaf,
            "sheet_uuid": node.definition.sheet_uuid,
            "component_refs": list(node.definition.component_refs),
            "ports": [
                {
                    "name": p.name,
                    "role": p.role.value,
                    "direction": p.direction.value,
                    "net_name": p.net_name,
                    "cardinality": p.cardinality,
                    "preferred_side": p.preferred_side.value,
                    "access_policy": p.access_policy.value,
                    "bus_index": p.bus_index,
                    "required": p.required,
                    "description": p.description,
                }
                for p in node.definition.ports
            ],
            "notes": list(node.definition.notes),
            "children": [node_to_dict(child) for child in node.children],
        }

    return {
        "project_dir": graph.project_dir,
        "root_schematic_path": graph.root_schematic_path,
        "root": node_to_dict(graph.root),
    }


__all__ = [
    "HierarchyGraph",
    "HierarchyNode",
    "SheetNodeMetadata",
    "SheetPinMetadata",
    "find_top_level_schematic",
    "format_hierarchy_tree",
    "hierarchy_debug_dict",
    "parse_hierarchy",
]
