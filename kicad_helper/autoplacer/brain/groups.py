"""Functional group extraction — discover component groups from schematics or netlist.

Three-tier resolution:
  1. Schematic sheets  (preferred — reads KiCad hierarchical schematics)
  2. Netlist community detection  (fallback — uses connectivity graph)
  3. Manual override   (always wins — from config ic_groups dict)

Pure Python, no pcbnew dependency.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

from .types import FunctionalGroup, GroupSet, Net


# ---------------------------------------------------------------------------
# S-expression mini-parser (enough for KiCad .kicad_sch files)
# ---------------------------------------------------------------------------

def _tokenize(text: str):
    """Yield tokens: '(', ')', or quoted/unquoted strings."""
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in ' \t\n\r':
            i += 1
        elif c == '(':
            yield '('
            i += 1
        elif c == ')':
            yield ')'
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                if text[j] == '\\':
                    j += 1  # skip escaped char
                j += 1
            yield text[i + 1:j]  # strip quotes
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in ' \t\n\r()':
                j += 1
            yield text[i:j]
            i = j


def _parse_sexpr(tokens) -> list:
    """Parse tokenized S-expression into nested lists."""
    result = []
    for tok in tokens:
        if tok == '(':
            result.append(_parse_sexpr(tokens))
        elif tok == ')':
            return result
        else:
            result.append(tok)
    return result


def _parse_file(path: str) -> list:
    """Parse a .kicad_sch file into an S-expression tree."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = _tokenize(text)
    # The file is one top-level S-expression
    for tok in tokens:
        if tok == '(':
            return _parse_sexpr(tokens)
    return []


def _find_nodes(tree: list, tag: str) -> list[list]:
    """Find all child nodes with the given tag (first element)."""
    results = []
    for item in tree:
        if isinstance(item, list) and item and item[0] == tag:
            results.append(item)
    return results


def _find_node(tree: list, tag: str):
    """Find the first child node with the given tag."""
    for item in tree:
        if isinstance(item, list) and item and item[0] == tag:
            return item
    return None


def _get_property(tree: list, name: str) -> str | None:
    """Get the value of a (property "name" "value" ...) node."""
    for item in tree:
        if isinstance(item, list) and item and item[0] == 'property':
            if len(item) >= 3 and item[1] == name:
                return item[2]
    return None


# ---------------------------------------------------------------------------
# Schematic-based group extraction
# ---------------------------------------------------------------------------

def extract_groups_from_schematics(project_dir: str) -> GroupSet | None:
    """Extract functional groups from KiCad hierarchical schematic files.

    Reads the top-level .kicad_sch file, finds sub-sheet definitions,
    then reads each sub-sheet to extract component references.

    Returns None if no hierarchical sheets are found (flat schematic).
    """
    project_path = Path(project_dir)

    # Find the top-level schematic file
    sch_files = list(project_path.glob("*.kicad_sch"))
    if not sch_files:
        return None

    # The top-level schematic is typically the one named after the project,
    # or the one that contains (sheet ...) blocks
    top_sch = None
    for sch_file in sch_files:
        try:
            tree = _parse_file(str(sch_file))
            sheets = _find_nodes(tree, 'sheet')
            if sheets:
                top_sch = sch_file
                break
        except Exception:
            continue

    if top_sch is None:
        return None  # No hierarchical sheets found

    tree = _parse_file(str(top_sch))
    sheets = _find_nodes(tree, 'sheet')

    if not sheets:
        return None

    groups = []
    all_member_refs = set()

    # Build inter-sheet connectivity from hierarchical pins
    sheet_pins: dict[str, list[str]] = {}  # sheet_name -> [pin_names]

    for sheet_node in sheets:
        sheet_name = _get_property(sheet_node, 'Sheetname') or ''
        sheet_file = _get_property(sheet_node, 'Sheetfile') or ''

        if not sheet_file:
            continue

        # Collect hierarchical pin names for inter-group net detection
        pins = _find_nodes(sheet_node, 'pin')
        pin_names = [p[1] for p in pins if len(p) >= 2 and isinstance(p[1], str)]
        sheet_pins[sheet_name] = pin_names

        # Resolve sub-sheet file path (relative to project dir)
        sub_path = project_path / sheet_file
        if not sub_path.exists():
            print(f"  WARNING: Sub-sheet not found: {sub_path}")
            continue

        # Parse sub-sheet and extract component references
        try:
            sub_tree = _parse_file(str(sub_path))
        except Exception as e:
            print(f"  WARNING: Failed to parse {sub_path}: {e}")
            continue

        # Find all (symbol ...) blocks in the sub-sheet
        symbols = _find_nodes(sub_tree, 'symbol')
        refs_in_sheet = []

        for sym in symbols:
            # Get lib_id to filter out power symbols
            lib_id_node = _find_node(sym, 'lib_id')
            if lib_id_node and len(lib_id_node) >= 2:
                lib_id = lib_id_node[1]
                if isinstance(lib_id, str) and lib_id.startswith('power:'):
                    continue

            # Get reference from instances block
            instances = _find_node(sym, 'instances')
            if instances is None:
                continue

            # Navigate: instances -> project -> path -> reference
            for project_node in _find_nodes(instances, 'project'):
                for path_node in _find_nodes(project_node, 'path'):
                    ref_node = _find_node(path_node, 'reference')
                    if ref_node and len(ref_node) >= 2:
                        ref = ref_node[1]
                        if isinstance(ref, str) and not ref.startswith('#'):
                            refs_in_sheet.append(ref)

        if not refs_in_sheet:
            continue

        # Determine the group leader: first IC (U*), or the largest/most
        # connected component. Heuristic: first "U" ref, else first ref
        # alphabetically.
        leader = _pick_group_leader(refs_in_sheet)

        group = FunctionalGroup(
            name=sheet_name,
            leader_ref=leader,
            member_refs=refs_in_sheet,
            inter_group_nets=pin_names,
        )
        groups.append(group)
        all_member_refs.update(refs_in_sheet)

    if not groups:
        return None

    return GroupSet(
        groups=groups,
        ungrouped_refs=[],  # filled in by resolve_groups()
        source="schematic",
    )


def _pick_group_leader(refs: list[str]) -> str:
    """Pick the most likely group leader from a list of component refs.

    Priority: ICs (U*) > active components (Q*) > everything else.
    Within a priority tier, pick the lowest reference number.
    """
    def _sort_key(ref: str) -> tuple[int, str]:
        if ref.startswith('U'):
            return (0, ref)
        if ref.startswith('Q'):
            return (1, ref)
        if ref.startswith('IC'):
            return (0, ref)
        # Connectors, batteries, etc. can also be leaders
        return (2, ref)

    return min(refs, key=_sort_key)


# ---------------------------------------------------------------------------
# Netlist-based group extraction (fallback for flat schematics)
# ---------------------------------------------------------------------------

def extract_groups_from_netlist(
    component_refs: list[str],
    nets: dict[str, Net],
    seed: int = 42,
) -> GroupSet:
    """Extract functional groups via netlist community detection.

    Uses the existing connectivity graph + label propagation algorithm
    from graph.py. Each community becomes a FunctionalGroup.
    """
    from .graph import build_connectivity_graph, find_communities

    conn_graph = build_connectivity_graph(nets)
    communities = find_communities(conn_graph, seed=seed)

    groups = []
    grouped_refs = set()

    for i, community in enumerate(communities):
        refs = [r for r in community if r in set(component_refs)]
        if len(refs) < 2:
            continue

        leader = _pick_group_leader(refs)
        group = FunctionalGroup(
            name=f"Group {i + 1}",
            leader_ref=leader,
            member_refs=refs,
        )
        groups.append(group)
        grouped_refs.update(refs)

    ungrouped = [r for r in component_refs if r not in grouped_refs]

    return GroupSet(
        groups=groups,
        ungrouped_refs=ungrouped,
        source="netlist",
    )


# ---------------------------------------------------------------------------
# Manual group definition (from config ic_groups dict)
# ---------------------------------------------------------------------------

def groups_from_manual_config(
    ic_groups: dict[str, list[str]],
    group_labels: dict[str, str] | None = None,
) -> GroupSet:
    """Build GroupSet from manually-defined ic_groups config dict.

    ic_groups format: {leader_ref: [member_refs]}
    group_labels format: {leader_ref: "Human Name"}
    """
    labels = group_labels or {}
    groups = []
    all_refs = set()

    for leader, members in ic_groups.items():
        all_members = [leader] + list(members)
        group = FunctionalGroup(
            name=labels.get(leader, leader),
            leader_ref=leader,
            member_refs=all_members,
        )
        groups.append(group)
        all_refs.update(all_members)

    return GroupSet(
        groups=groups,
        ungrouped_refs=[],
        source="manual",
    )


# ---------------------------------------------------------------------------
# Three-tier group resolution
# ---------------------------------------------------------------------------

def resolve_groups(
    project_dir: str,
    component_refs: list[str],
    nets: dict[str, Net],
    config: dict | None = None,
    seed: int = 42,
) -> GroupSet:
    """Resolve functional groups using three-tier strategy.

    Priority:
      1. Manual ic_groups from config (if provided and non-empty)
      2. Schematic hierarchical sheets (if project has them)
      3. Netlist community detection (always available)

    The manual override is applied on top of auto-detected groups:
    if config has ic_groups, those override any auto-detected grouping
    for the specified components.
    """
    cfg = config or {}
    group_source = cfg.get("group_source", "auto")
    manual_ic_groups = cfg.get("ic_groups", {})
    manual_labels = cfg.get("group_labels", {})

    result = None

    # --- Tier 1: Try schematic extraction (unless source is "manual" or "netlist") ---
    if group_source in ("auto", "schematic"):
        result = extract_groups_from_schematics(project_dir)
        if result:
            print(f"  Groups: extracted {len(result.groups)} from schematic sheets")

    # --- Tier 2: Fall back to netlist community detection ---
    if result is None and group_source in ("auto", "netlist"):
        result = extract_groups_from_netlist(component_refs, nets, seed=seed)
        print(f"  Groups: discovered {len(result.groups)} from netlist analysis")

    # --- Tier 3: Use manual config if no auto-detection worked ---
    if result is None and manual_ic_groups:
        result = groups_from_manual_config(manual_ic_groups, manual_labels)
        print(f"  Groups: loaded {len(result.groups)} from manual config")

    # --- Apply manual overrides on top of auto-detected groups ---
    if result and manual_ic_groups and result.source != "manual":
        result = _apply_manual_overrides(result, manual_ic_groups, manual_labels)

    # --- Fallback: no groups at all ---
    if result is None:
        result = GroupSet(
            groups=[],
            ungrouped_refs=list(component_refs),
            source="none",
        )
        print("  Groups: none found (all components ungrouped)")

    # --- Fill in ungrouped refs ---
    grouped_refs = set()
    for group in result.groups:
        grouped_refs.update(group.member_refs)

    result.ungrouped_refs = [
        r for r in component_refs
        if r not in grouped_refs
    ]

    if result.ungrouped_refs:
        print(f"  Groups: {len(result.ungrouped_refs)} ungrouped component(s): "
              f"{', '.join(sorted(result.ungrouped_refs)[:10])}")

    # --- Derive signal flow order from inter-group connectivity ---
    _compute_inter_group_nets(result, nets)

    return result


def _apply_manual_overrides(
    group_set: GroupSet,
    ic_groups: dict[str, list[str]],
    group_labels: dict[str, str],
) -> GroupSet:
    """Apply manual ic_groups overrides on top of auto-detected groups.

    Manual groups take precedence: if a component appears in both auto
    and manual groups, the manual assignment wins.
    """
    manual_refs = set()
    for leader, members in ic_groups.items():
        manual_refs.add(leader)
        manual_refs.update(members)

    # Remove manually-assigned refs from auto-detected groups
    new_groups = []
    for group in group_set.groups:
        remaining = [r for r in group.member_refs if r not in manual_refs]
        if len(remaining) >= 2:
            # Recompute leader if original was reassigned
            if group.leader_ref in manual_refs:
                group.leader_ref = _pick_group_leader(remaining)
            group.member_refs = remaining
            new_groups.append(group)
        elif len(remaining) == 1:
            # Single component group — will become ungrouped
            pass

    # Add manual groups
    for leader, members in ic_groups.items():
        all_members = [leader] + list(members)
        name = group_labels.get(leader, leader)
        new_groups.append(FunctionalGroup(
            name=name,
            leader_ref=leader,
            member_refs=all_members,
        ))

    return GroupSet(
        groups=new_groups,
        ungrouped_refs=group_set.ungrouped_refs,
        source=f"{group_set.source}+manual",
    )


def _compute_inter_group_nets(group_set: GroupSet, nets: dict[str, Net]):
    """Populate inter_group_nets for each group based on actual netlist data.

    A net is "inter-group" if it connects components from two or more
    different groups. GND is excluded.
    """
    ref_to_group = group_set.ref_to_group()

    for group in group_set.groups:
        inter_nets = set()
        group_refs = set(group.member_refs)

        for net in nets.values():
            if net.name in ("GND", "/GND"):
                continue
            net_refs = net.component_refs
            # Does this net connect this group to another group?
            has_internal = bool(net_refs & group_refs)
            has_external = bool(net_refs - group_refs)
            if has_internal and has_external:
                inter_nets.add(net.name)

        group.inter_group_nets = sorted(inter_nets)


def derive_signal_flow_order(group_set: GroupSet, nets: dict[str, Net]) -> list[str]:
    """Derive left-to-right signal flow order from inter-group connectivity.

    Builds a DAG of groups based on shared nets and performs topological
    sort. Groups with external connections (connectors) are placed at
    the edges.

    Returns list of group leader refs in signal-flow order.
    """
    if not group_set.groups:
        return []

    ref_to_group = group_set.ref_to_group()

    # Build adjacency: count shared inter-group nets between group pairs
    group_edges: dict[tuple[str, str], int] = defaultdict(int)
    for net in nets.values():
        if net.name in ("GND", "/GND"):
            continue
        connected_groups = set()
        for ref in net.component_refs:
            grp = ref_to_group.get(ref)
            if grp:
                connected_groups.add(grp.leader_ref)
        if len(connected_groups) >= 2:
            leaders = sorted(connected_groups)
            for i in range(len(leaders)):
                for j in range(i + 1, len(leaders)):
                    group_edges[(leaders[i], leaders[j])] += 1

    # Simple heuristic ordering: BFS from the group with the most
    # "input-like" characteristics (connectors on input side).
    # If that fails, use the order they appear in the schematic.
    leaders = [g.leader_ref for g in group_set.groups]

    if not group_edges:
        return leaders

    # Build adjacency list
    adj: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (a, b), w in group_edges.items():
        adj[a].append((b, w))
        adj[b].append((a, w))

    # Start from the group with the fewest inter-group connections
    # (likely an endpoint like USB INPUT or Battery)
    start = min(leaders, key=lambda l: len(adj.get(l, [])))
    visited = set()
    order = []

    # BFS
    queue = [start]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        order.append(current)
        # Sort neighbors by connection weight (strongest first)
        neighbors = sorted(adj.get(current, []), key=lambda x: -x[1])
        for nbr, _ in neighbors:
            if nbr not in visited:
                queue.append(nbr)

    # Add any unvisited groups at the end
    for l in leaders:
        if l not in visited:
            order.append(l)

    return order
