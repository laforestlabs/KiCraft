#!/usr/bin/env python3
"""Inspect KiCad schematic hierarchy and normalized subcircuit interfaces.

This CLI is the Milestone 1 inspection/debug entrypoint for the subcircuits
redesign. It delegates all parsing and normalization to the shared hierarchy
parser so there is a single source of truth for:

- true KiCad sheet hierarchy
- leaf vs composite sheet detection
- component membership
- normalized interface ports

Usage:
    python3 inspect_subcircuits.py LLUPS.kicad_sch
    python3 inspect_subcircuits.py LLUPS.kicad_sch --json
    python3 inspect_subcircuits.py LLUPS.kicad_sch --show-components
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kicraft.autoplacer.brain.hierarchy_parser import (
    HierarchyGraph,
    HierarchyNode,
    hierarchy_debug_dict,
    parse_hierarchy,
)


def _iter_children(node: HierarchyNode):
    for child in node.children:
        yield child
        yield from _iter_children(child)


def _iter_non_root_nodes(graph: HierarchyGraph):
    for child in graph.root.children:
        yield child
        yield from _iter_children(child)


def _print_tree(node: HierarchyNode, prefix: str = "", is_last: bool = True) -> None:
    branch = "└── " if is_last else "├── "
    kind = "leaf" if node.is_leaf else "composite"
    print(
        f"{prefix}{branch}{node.id.sheet_name} "
        f"[{kind}] refs={len(node.definition.component_refs)} "
        f"ports={len(node.definition.ports)}"
    )

    child_prefix = prefix + ("    " if is_last else "│   ")
    for idx, child in enumerate(node.children):
        _print_tree(child, child_prefix, idx == len(node.children) - 1)


def _print_node_details(node: HierarchyNode, show_components: bool) -> None:
    definition = node.definition
    print(f"- {node.id.sheet_name}")
    print(f"  instance_path : {node.id.instance_path}")
    print(f"  sheet_file    : {node.id.sheet_file}")
    print(f"  schematic     : {definition.schematic_path}")
    print(f"  kind          : {'leaf' if node.is_leaf else 'composite'}")
    print(f"  interfaces    : {len(definition.ports)}")
    print(f"  components    : {len(definition.component_refs)}")
    print(f"  children      : {len(node.children)}")

    if definition.ports:
        print("  ports:")
        for port in definition.ports:
            extra = []
            if port.bus_index is not None:
                extra.append(f"bus_index={port.bus_index}")
            if not port.required:
                extra.append("required=false")
            if port.raw_direction and port.raw_direction != port.direction.value:
                extra.append(f"raw={port.raw_direction}")
            suffix = f" ({', '.join(extra)})" if extra else ""
            print(
                "    - "
                f"{port.name}: "
                f"role={port.role.value} "
                f"dir={port.direction.value} "
                f"side={port.preferred_side.value} "
                f"access={port.access_policy.value}"
                f"{suffix}"
            )

    if show_components and definition.component_refs:
        print("  component_refs:")
        for ref in definition.component_refs:
            print(f"    - {ref}")


def _print_summary(graph: HierarchyGraph, show_components: bool) -> None:
    nodes = list(_iter_non_root_nodes(graph))
    leaves = [node for node in nodes if node.is_leaf]
    composites = [node for node in nodes if not node.is_leaf]

    print("=== Subcircuit Hierarchy ===")
    for idx, child in enumerate(graph.root.children):
        _print_tree(child, "", idx == len(graph.root.children) - 1)
    print()

    print("=== Summary ===")
    print(f"project_dir      : {graph.project_dir}")
    print(f"root_schematic   : {graph.root_schematic_path}")
    print(f"top_level_sheets : {len(graph.root.children)}")
    print(f"total_nodes      : {len(nodes)}")
    print(f"leaf_nodes       : {len(leaves)}")
    print(f"composite_nodes  : {len(composites)}")
    print()

    print("=== Leaf Sheets ===")
    for node in leaves:
        print(
            f"- {node.id.sheet_name}: "
            f"{len(node.definition.component_refs)} components, "
            f"{len(node.definition.ports)} interfaces"
        )
    print()

    print("=== Per-Sheet Details ===")
    for node in nodes:
        _print_node_details(node, show_components=show_components)
        print()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect KiCad schematic hierarchy and normalized subcircuit interfaces"
    )
    parser.add_argument(
        "schematic",
        help="Top-level .kicad_sch file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text",
    )
    parser.add_argument(
        "--show-components",
        action="store_true",
        help="Include component reference lists in text output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    top_schematic = Path(args.schematic)

    if not top_schematic.exists():
        print(f"error: schematic not found: {top_schematic}", file=sys.stderr)
        return 2
    if top_schematic.suffix != ".kicad_sch":
        print(
            f"error: expected a .kicad_sch file, got: {top_schematic}",
            file=sys.stderr,
        )
        return 2

    try:
        graph = parse_hierarchy(
            project_dir=top_schematic.resolve().parent,
            top_schematic=top_schematic.resolve(),
        )
    except Exception as exc:
        print(f"error: failed to inspect hierarchy: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = hierarchy_debug_dict(graph)
        payload["top_level_sheets"] = len(graph.root.children)
        payload["total_nodes"] = len(list(_iter_non_root_nodes(graph)))
        payload["leaf_nodes"] = len(graph.leaf_nodes()) - (
            1 if graph.root.is_leaf else 0
        )
        print(json.dumps(payload, indent=2))
        return 0

    _print_summary(graph, show_components=args.show_components)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
