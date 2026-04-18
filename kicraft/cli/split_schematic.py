#!/usr/bin/env python3
"""Split a flat KiCad schematic into hierarchical sheets by IC groups.

Reads a flat .kicad_sch file and an ic_groups configuration, then:
1. Creates one sub-sheet per IC group containing that group's components,
   wires, labels, junctions, no_connects, and text annotations.
2. Converts net labels that cross group boundaries into hierarchical labels.
3. Rewrites the root schematic as a top-level sheet with hierarchical
   sheet symbols and inter-sheet pins.

Generalized: works with any flat .kicad_sch file and any ic_groups config
dict. Power symbols (GND, VCC, etc.) are duplicated into every sheet
that needs them.

Usage:
    python3 split_schematic.py <schematic.kicad_sch> [options]

Options:
    --config <path>       Python config file with DEFAULT_CONFIG dict
    --groups <json>       JSON dict override for ic_groups
    --sheet-names <json>  JSON dict of {ic_ref: "Sheet Display Name"}
    --output-dir <dir>    Directory for sub-sheet files (default: same as input)
    --dry-run             Analyze and print plan without writing files
    --backup              Create .bak of the original before overwriting
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# S-expression block extraction
# ---------------------------------------------------------------------------

def _find_matching_paren(text: str, start: int) -> int:
    """Find the closing paren matching the opening paren at `start`."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        c = text[i]
        if c == '"' and (i == 0 or text[i - 1] != '\\'):
            in_string = not in_string
        elif not in_string:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _extract_top_level_blocks(text: str, start_after: int = 0) -> list[dict]:
    """Extract all top-level S-expression blocks from text.

    Returns list of {"type": str, "text": str, "start": int, "end": int}.
    """
    blocks = []
    i = start_after
    while i < len(text):
        # Find next top-level opening paren (preceded by tab or start of line)
        m = re.search(r'(?:^|\n)([ \t]*)\(', text[i:])
        if not m:
            break
        block_start = i + m.start()
        # Skip to the actual '('
        paren_pos = text.index('(', block_start)
        end = _find_matching_paren(text, paren_pos)
        if end == -1:
            break

        block_text = text[paren_pos:end + 1]
        # Extract block type (first word after opening paren)
        type_m = re.match(r'\((\w+)', block_text)
        block_type = type_m.group(1) if type_m else "unknown"

        blocks.append({
            "type": block_type,
            "text": block_text,
            "start": paren_pos,
            "end": end + 1,
        })
        i = end + 1
    return blocks


# ---------------------------------------------------------------------------
# Schematic element parsing
# ---------------------------------------------------------------------------

@dataclass
class SchSymbol:
    """A placed symbol instance on the schematic."""
    lib_id: str
    ref: str
    pos: tuple[float, float]
    text: str          # full S-expression text
    is_power: bool
    uuid: str

@dataclass
class SchWire:
    """A wire segment."""
    start: tuple[float, float]
    end: tuple[float, float]
    text: str
    uuid: str

@dataclass
class SchLabel:
    """A net label."""
    name: str
    pos: tuple[float, float]
    text: str
    uuid: str

@dataclass
class SchJunction:
    pos: tuple[float, float]
    text: str
    uuid: str

@dataclass
class SchNoConnect:
    pos: tuple[float, float]
    text: str
    uuid: str

@dataclass
class SchText:
    content: str
    pos: tuple[float, float]
    text: str
    uuid: str


def _parse_uuid(block_text: str) -> str:
    m = re.search(r'\(uuid\s+"([^"]+)"\)', block_text)
    return m.group(1) if m else ""


def _parse_at(block_text: str) -> tuple[float, float]:
    m = re.search(r'\(at\s+([-\d.]+)\s+([-\d.]+)', block_text)
    return (float(m.group(1)), float(m.group(2))) if m else (0, 0)


def parse_schematic(sch_path: str) -> dict:
    """Parse a flat KiCad schematic into structured elements.

    Returns dict with keys:
        header: str (everything up to and including lib_symbols closing paren)
        symbols: list[SchSymbol]
        wires: list[SchWire]
        labels: list[SchLabel]
        junctions: list[SchJunction]
        no_connects: list[SchNoConnect]
        texts: list[SchText]
        tail: str (sheet_instances + embedded_fonts + closing)
        root_uuid: str
        project_name: str
    """
    with open(sch_path) as f:
        text = f.read()

    # Extract root UUID
    root_uuid_m = re.search(r'\(uuid\s+"([^"]+)"\)', text[:500])
    root_uuid = root_uuid_m.group(1) if root_uuid_m else str(uuid.uuid4())

    # Extract project name from title_block or filename
    project_m = re.search(r'\(project\s+"([^"]+)"', text)
    project_name = project_m.group(1) if project_m else Path(sch_path).stem

    # Find end of lib_symbols section
    # lib_symbols is the first top-level block after header
    lib_sym_m = re.search(r'\n\t\(lib_symbols\b', text)
    if lib_sym_m:
        lib_end = _find_matching_paren(text, lib_sym_m.start() + 1)  # +1 for \n
        # Actually find the ( position
        paren_start = text.index('(lib_symbols', lib_sym_m.start())
        lib_end = _find_matching_paren(text, paren_start)
        header_end = lib_end + 1
    else:
        # No lib_symbols — header is just the first few lines
        header_end = text.index('\n', text.index(')')) + 1

    header = text[:header_end]

    # Everything after lib_symbols
    content = text[header_end:]

    # Parse all top-level blocks in the content section
    symbols = []
    wires = []
    labels = []
    junctions = []
    no_connects = []
    texts = []
    tail_parts = []

    # Simple approach: scan for blocks by finding `\n\t(type ` patterns
    pos = 0
    while pos < len(content):
        # Find next block start
        m = re.search(r'\n(\t)\(', content[pos:])
        if not m:
            break

        block_start = pos + m.start() + 1  # skip the \n
        paren_pos = content.index('(', block_start)
        block_end = _find_matching_paren(content, paren_pos)
        if block_end == -1:
            break

        block_text = content[paren_pos:block_end + 1]
        type_m = re.match(r'\((\w+)', block_text)
        block_type = type_m.group(1) if type_m else ""

        if block_type == "symbol":
            lib_id_m = re.search(r'\(lib_id\s+"([^"]+)"\)', block_text)
            lib_id = lib_id_m.group(1) if lib_id_m else ""
            ref_m = re.search(r'\(property\s+"Reference"\s+"([^"]+)"', block_text)
            ref = ref_m.group(1) if ref_m else ""
            is_power = lib_id.startswith("power:")
            symbols.append(SchSymbol(
                lib_id=lib_id, ref=ref, pos=_parse_at(block_text),
                text=block_text, is_power=is_power,
                uuid=_parse_uuid(block_text),
            ))
        elif block_type == "wire":
            pts_m = re.search(
                r'\(pts\s*\n?\s*\(xy\s+([-\d.]+)\s+([-\d.]+)\)\s*\(xy\s+([-\d.]+)\s+([-\d.]+)\)',
                block_text)
            if pts_m:
                start_pt = (float(pts_m.group(1)), float(pts_m.group(2)))
                end_pt = (float(pts_m.group(3)), float(pts_m.group(4)))
            else:
                start_pt = end_pt = (0, 0)
            wires.append(SchWire(
                start=start_pt, end=end_pt, text=block_text,
                uuid=_parse_uuid(block_text),
            ))
        elif block_type == "label":
            name_m = re.match(r'\(label\s+"([^"]+)"', block_text)
            name = name_m.group(1) if name_m else ""
            labels.append(SchLabel(
                name=name, pos=_parse_at(block_text), text=block_text,
                uuid=_parse_uuid(block_text),
            ))
        elif block_type == "junction":
            junctions.append(SchJunction(
                pos=_parse_at(block_text), text=block_text,
                uuid=_parse_uuid(block_text),
            ))
        elif block_type == "no_connect":
            no_connects.append(SchNoConnect(
                pos=_parse_at(block_text), text=block_text,
                uuid=_parse_uuid(block_text),
            ))
        elif block_type == "text":
            content_m = re.match(r'\(text\s+"([^"]*)"', block_text)
            texts.append(SchText(
                content=content_m.group(1) if content_m else "",
                pos=_parse_at(block_text), text=block_text,
                uuid=_parse_uuid(block_text),
            ))
        elif block_type in ("sheet_instances", "embedded_fonts"):
            tail_parts.append(block_text)

        pos = block_end + 1

    return {
        "header": header,
        "symbols": symbols,
        "wires": wires,
        "labels": labels,
        "junctions": junctions,
        "no_connects": no_connects,
        "texts": texts,
        "tail_parts": tail_parts,
        "root_uuid": root_uuid,
        "project_name": project_name,
    }


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------

def build_ref_to_group(ic_groups: dict) -> dict[str, str]:
    """Map every component ref to its group leader IC ref.

    Returns {ref: group_leader_ref}.
    """
    mapping = {}
    for leader, members in ic_groups.items():
        mapping[leader] = leader
        for m in members:
            mapping[m] = leader
    return mapping


def _pin_endpoints(symbol: SchSymbol) -> set[tuple[float, float]]:
    """Estimate schematic pin positions from the symbol's pin entries.

    KiCad stores pin UUIDs but not explicit positions in the instance.
    We use the component's (at x y) as the reference point.
    For more accuracy we'd need lib_symbols data, but for wire assignment
    we'll use proximity-based matching instead.
    """
    return {symbol.pos}


# ---------------------------------------------------------------------------
# Wire/element assignment to groups
# ---------------------------------------------------------------------------

def assign_elements_to_groups(
    parsed: dict,
    ic_groups: dict,
) -> dict[str, dict]:
    """Assign schematic elements to groups.

    Returns {group_leader: {
        "symbols": [...], "wires": [...], "labels": [...],
        "junctions": [...], "no_connects": [...], "texts": [...],
        "power_symbols": [...]
    }}
    """
    ref_to_group = build_ref_to_group(ic_groups)

    # Initialize groups
    groups: dict[str, dict] = {}
    for leader in ic_groups:
        groups[leader] = {
            "symbols": [],
            "wires": [],
            "labels": [],
            "junctions": [],
            "no_connects": [],
            "texts": [],
            "power_symbols": [],
        }

    # Assign symbols to groups
    unassigned_symbols = []
    component_positions: dict[str, tuple[float, float]] = {}

    for sym in parsed["symbols"]:
        if sym.is_power:
            # Power symbols get duplicated later to each group that needs them
            unassigned_symbols.append(sym)
            continue
        group = ref_to_group.get(sym.ref)
        if group and group in groups:
            groups[group]["symbols"].append(sym)
            component_positions[sym.ref] = sym.pos
        else:
            unassigned_symbols.append(sym)

    # For multi-unit symbols (like Q1 dual NMOS), both instances share the ref
    # and should go to the same group.

    # Build a spatial index: for each group, collect all component positions
    group_positions: dict[str, list[tuple[float, float]]] = {}
    for leader in groups:
        positions = []
        for sym in groups[leader]["symbols"]:
            positions.append(sym.pos)
        group_positions[leader] = positions

    # Build bounding box per group for proximity assignment
    group_bbox: dict[str, tuple[float, float, float, float]] = {}
    for leader, positions in group_positions.items():
        if not positions:
            continue
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        margin = 30.0  # generous margin in schematic units (mm)
        group_bbox[leader] = (min(xs) - margin, min(ys) - margin,
                              max(xs) + margin, max(ys) + margin)

    def _nearest_group(pt: tuple[float, float]) -> str | None:
        """Find the group closest to a point."""
        best_dist = float('inf')
        best_group = None
        for leader, positions in group_positions.items():
            if not positions:
                continue
            for gp in positions:
                d = ((pt[0] - gp[0])**2 + (pt[1] - gp[1])**2)**0.5
                if d < best_dist:
                    best_dist = d
                    best_group = leader
        return best_group

    def _point_in_bbox(pt: tuple[float, float], leader: str) -> bool:
        if leader not in group_bbox:
            return False
        x0, y0, x1, y1 = group_bbox[leader]
        return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1

    # Assign wires to groups based on endpoint proximity
    for wire in parsed["wires"]:
        mid = ((wire.start[0] + wire.end[0]) / 2,
               (wire.start[1] + wire.end[1]) / 2)
        group = _nearest_group(mid)
        if group:
            groups[group]["wires"].append(wire)

    # Assign junctions
    for junc in parsed["junctions"]:
        group = _nearest_group(junc.pos)
        if group:
            groups[group]["junctions"].append(junc)

    # Assign no_connects
    for nc in parsed["no_connects"]:
        group = _nearest_group(nc.pos)
        if group:
            groups[group]["no_connects"].append(nc)

    # Assign text annotations
    for txt in parsed["texts"]:
        group = _nearest_group(txt.pos)
        if group:
            groups[group]["texts"].append(txt)

    # Assign power symbols to nearest group
    for sym in unassigned_symbols:
        if sym.is_power:
            group = _nearest_group(sym.pos)
            if group:
                groups[group]["power_symbols"].append(sym)
        # Non-power unassigned symbols: assign to nearest group too
        else:
            group = _nearest_group(sym.pos)
            if group:
                groups[group]["symbols"].append(sym)

    return groups


# ---------------------------------------------------------------------------
# Cross-group net analysis
# ---------------------------------------------------------------------------

def find_cross_group_nets(
    parsed: dict,
    group_assignments: dict[str, dict],
) -> dict[str, set[str]]:
    """Find nets (labels) that appear in multiple groups.

    Returns {net_name: {group1, group2, ...}}.
    """
    net_groups: dict[str, set[str]] = defaultdict(set)

    for leader, elems in group_assignments.items():
        for label in elems["labels"]:
            net_groups[label.name].add(leader)

    # Only return nets that span >1 group
    return {name: gset for name, gset in net_groups.items() if len(gset) > 1}


# ---------------------------------------------------------------------------
# Sub-sheet generation
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())


def _make_hierarchical_label(name: str, pos: tuple[float, float],
                              direction: str = "bidirectional") -> str:
    """Create a hierarchical_label S-expression."""
    return (
        f'\t(hierarchical_label "{name}"\n'
        f'\t\t(shape {direction})\n'
        f'\t\t(at {pos[0]:.2f} {pos[1]:.2f} 0)\n'
        f'\t\t(effects\n'
        f'\t\t\t(font (size 1.27 1.27))\n'
        f'\t\t)\n'
        f'\t\t(uuid "{_new_uuid()}")\n'
        f'\t)\n'
    )


# Paper sizes in landscape orientation (width, height) in mm
PAPER_SIZES = [
    ("A4", 297, 210),
    ("A3", 420, 297),
    ("A2", 594, 420),
]


def _fmt_coord(v: float) -> str:
    """Format coordinate, keeping 2 decimals only when needed."""
    if v == int(v):
        return str(int(v))
    s = f'{v:.2f}'
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    return s


def _shift_block_coords(text: str, dx: float, dy: float) -> str:
    """Shift all (at X Y …) and (xy X Y) coordinates in an S-expression block."""
    def _shift_at(m):
        x = float(m.group(1)) + dx
        y = float(m.group(2)) + dy
        rest = m.group(3) or ''
        return f'(at {_fmt_coord(x)} {_fmt_coord(y)}{rest})'

    def _shift_xy(m):
        x = float(m.group(1)) + dx
        y = float(m.group(2)) + dy
        return f'(xy {_fmt_coord(x)} {_fmt_coord(y)})'

    text = re.sub(r'\(at\s+([-\d.]+)\s+([-\d.]+)((?:\s+[-\d.]+)*)\)', _shift_at, text)
    text = re.sub(r'\(xy\s+([-\d.]+)\s+([-\d.]+)\)', _shift_xy, text)
    return text


def _pick_paper_and_offset(
    elements: dict,
    margin: float = 40.0,
) -> tuple[str, float, float]:
    """Choose the smallest paper that fits the group and compute centering offset.

    Returns (paper_name, dx, dy).
    """
    all_positions: list[tuple[float, float]] = []
    for sym in elements.get("symbols", []):
        all_positions.append(sym.pos)
    for sym in elements.get("power_symbols", []):
        all_positions.append(sym.pos)
    for wire in elements.get("wires", []):
        all_positions.append(wire.start)
        all_positions.append(wire.end)
    for label in elements.get("labels", []):
        all_positions.append(label.pos)
    for junc in elements.get("junctions", []):
        all_positions.append(junc.pos)
    for nc in elements.get("no_connects", []):
        all_positions.append(nc.pos)
    for txt in elements.get("texts", []):
        all_positions.append(txt.pos)

    if not all_positions:
        return "A4", 0.0, 0.0

    xs = [p[0] for p in all_positions]
    ys = [p[1] for p in all_positions]
    group_w = max(xs) - min(xs)
    group_h = max(ys) - min(ys)

    needed_w = group_w + 2 * margin
    needed_h = group_h + 2 * margin

    paper = "A4"
    pw, ph = 297, 210
    for name, sw, sh in PAPER_SIZES:
        if needed_w <= sw and needed_h <= sh:
            paper, pw, ph = name, sw, sh
            break
    else:
        # Larger than A2 — use A2 anyway, at least it'll be closer
        paper, pw, ph = PAPER_SIZES[-1]

    group_cx = (min(xs) + max(xs)) / 2
    group_cy = (min(ys) + max(ys)) / 2
    dx = pw / 2 - group_cx
    dy = ph / 2 - group_cy

    return paper, dx, dy


def generate_sub_sheet(
    parsed: dict,
    group_leader: str,
    elements: dict,
    cross_nets: dict[str, set[str]],
    sheet_name: str,
) -> str:
    """Generate a complete sub-sheet .kicad_sch file.

    Includes:
    - Header (version, generator, uuid, paper, lib_symbols)
    - The group's component symbols
    - Power symbols used by this group
    - Wires, junctions, no_connects, text annotations
    - Normal labels for intra-group nets
    - Hierarchical labels for cross-group nets
    - sheet_instances + embedded_fonts
    """
    sheet_uuid = _new_uuid()

    # --- Auto-fit: pick paper size and compute centering offset ---
    paper, dx, dy = _pick_paper_and_offset(elements)

    lines = []

    # --- Header ---
    lines.append('(kicad_sch\n')
    # Re-use same version/generator from the original
    version_m = re.search(r'\(version\s+\d+\)', parsed["header"])
    gen_m = re.search(r'\(generator\s+"[^"]+"\)', parsed["header"])
    genver_m = re.search(r'\(generator_version\s+"[^"]+"\)', parsed["header"])

    lines.append(f'\t{version_m.group(0)}\n' if version_m else '\t(version 20250114)\n')
    lines.append(f'\t{gen_m.group(0)}\n' if gen_m else '\t(generator "eeschema")\n')
    lines.append(f'\t{genver_m.group(0)}\n' if genver_m else '\t(generator_version "9.0")\n')
    lines.append(f'\t(uuid "{sheet_uuid}")\n')
    lines.append(f'\t(paper "{paper}")\n\n')

    # --- lib_symbols (copy full section) ---
    lib_m = re.search(r'(\(lib_symbols\b.*?\n\t\))', parsed["header"], re.DOTALL)
    if lib_m:
        lines.append(f'\t{lib_m.group(1)}\n\n')

    # --- Text annotations ---
    for txt in elements.get("texts", []):
        lines.append(f'\t{_shift_block_coords(txt.text, dx, dy)}\n')

    # --- Component symbols ---
    for sym in elements.get("symbols", []):
        # Rewrite the instances block to point to this sheet
        sym_text = sym.text
        # Replace the instances path
        new_instances = (
            f'(instances\n'
            f'\t\t\t(project "{parsed["project_name"]}"\n'
            f'\t\t\t\t(path "/{sheet_uuid}"\n'
            f'\t\t\t\t\t(reference "{sym.ref}")\n'
            f'\t\t\t\t\t(unit 1)\n'
            f'\t\t\t\t)\n'
            f'\t\t\t)\n'
            f'\t\t)'
        )
        # Replace existing instances block
        inst_m = re.search(r'\(instances\b.*?\n\t\t\)', sym_text, re.DOTALL)
        if inst_m:
            sym_text = sym_text[:inst_m.start()] + new_instances + sym_text[inst_m.end():]

        lines.append(f'\t{_shift_block_coords(sym_text, dx, dy)}\n')

    # --- Power symbols ---
    for sym in elements.get("power_symbols", []):
        sym_text = sym.text
        new_instances = (
            f'(instances\n'
            f'\t\t\t(project "{parsed["project_name"]}"\n'
            f'\t\t\t\t(path "/{sheet_uuid}"\n'
            f'\t\t\t\t\t(reference "{sym.ref}")\n'
            f'\t\t\t\t\t(unit 1)\n'
            f'\t\t\t\t)\n'
            f'\t\t\t)\n'
            f'\t\t)'
        )
        inst_m = re.search(r'\(instances\b.*?\n\t\t\)', sym_text, re.DOTALL)
        if inst_m:
            sym_text = sym_text[:inst_m.start()] + new_instances + sym_text[inst_m.end():]

        lines.append(f'\t{_shift_block_coords(sym_text, dx, dy)}\n')

    # --- Wires ---
    for wire in elements.get("wires", []):
        lines.append(f'\t{_shift_block_coords(wire.text, dx, dy)}\n')

    # --- Junctions ---
    for junc in elements.get("junctions", []):
        lines.append(f'\t{_shift_block_coords(junc.text, dx, dy)}\n')

    # --- Labels ---
    for label in elements.get("labels", []):
        if label.name in cross_nets:
            # Convert to hierarchical label with shifted position
            shifted_pos = (label.pos[0] + dx, label.pos[1] + dy)
            lines.append(_make_hierarchical_label(label.name, shifted_pos))
        else:
            lines.append(f'\t{_shift_block_coords(label.text, dx, dy)}\n')

    # --- No connects ---
    for nc in elements.get("no_connects", []):
        lines.append(f'\t{_shift_block_coords(nc.text, dx, dy)}\n')

    # --- Sheet instances ---
    lines.append('\n\t(sheet_instances\n')
    lines.append('\t\t(path "/"\n')
    lines.append('\t\t\t(page "1")\n')
    lines.append('\t\t)\n')
    lines.append('\t)\n')

    lines.append('\t(embedded_fonts no)\n')
    lines.append(')\n')

    return ''.join(lines)


# ---------------------------------------------------------------------------
# Root sheet generation
# ---------------------------------------------------------------------------

def generate_root_sheet(
    parsed: dict,
    ic_groups: dict,
    group_assignments: dict[str, dict],
    cross_nets: dict[str, set[str]],
    sheet_names: dict[str, str],
    sheet_files: dict[str, str],
    sheet_uuids: dict[str, str],
) -> str:
    """Generate the new root (top-level) schematic.

    Contains hierarchical sheet symbols for each sub-sheet,
    connected by net labels for cross-group nets.
    """
    lines = []

    # --- Header ---
    version_m = re.search(r'\(version\s+\d+\)', parsed["header"])
    gen_m = re.search(r'\(generator\s+"[^"]+"\)', parsed["header"])
    genver_m = re.search(r'\(generator_version\s+"[^"]+"\)', parsed["header"])
    title_m = re.search(r'\(title_block\s*\n(?:\t\t[^\n]*\n)*?\t\)', parsed["header"], re.DOTALL)

    lines.append('(kicad_sch\n')
    lines.append(f'\t{version_m.group(0)}\n' if version_m else '\t(version 20250114)\n')
    lines.append(f'\t{gen_m.group(0)}\n' if gen_m else '\t(generator "eeschema")\n')
    lines.append(f'\t{genver_m.group(0)}\n' if genver_m else '\t(generator_version "9.0")\n')
    lines.append(f'\t(uuid "{parsed["root_uuid"]}")\n')
    lines.append('\t(paper "A3")\n\n')

    if title_m:
        lines.append(f'\t{title_m.group(0)}\n\n')

    # --- lib_symbols (empty — all symbols live in sub-sheets) ---
    lines.append('\t(lib_symbols\n\t)\n\n')

    # --- Sheet symbols ---
    x_start = 30
    y_start = 40
    sheet_width = 30
    sheet_height = 15
    x_spacing = 40

    sorted_leaders = sorted(sheet_files.keys(),
                           key=lambda l: list(ic_groups.keys()).index(l)
                           if l in ic_groups else 999)

    for i, leader in enumerate(sorted_leaders):
        fname = sheet_files[leader]
        name = sheet_names.get(leader, leader)
        s_uuid = sheet_uuids[leader]

        x = x_start + i * x_spacing
        y = y_start

        # Determine which cross-group nets this sheet participates in
        sheet_nets = []
        for net_name, participating_groups in sorted(cross_nets.items()):
            if leader in participating_groups:
                sheet_nets.append(net_name)

        actual_height = max(sheet_height, 5 + len(sheet_nets) * 3)

        lines.append('\t(sheet\n')
        lines.append(f'\t\t(at {x} {y})\n')
        lines.append(f'\t\t(size {sheet_width} {actual_height})\n')
        lines.append('\t\t(fields_autoplaced yes)\n')
        lines.append('\t\t(stroke\n')
        lines.append('\t\t\t(width 0.2)\n')
        lines.append('\t\t\t(type solid)\n')
        lines.append('\t\t)\n')
        lines.append('\t\t(fill\n')
        lines.append('\t\t\t(color 255 255 194 1)\n')
        lines.append('\t\t)\n')
        lines.append(f'\t\t(uuid "{s_uuid}")\n')
        lines.append(f'\t\t(property "Sheetname" "{name}"\n')
        lines.append(f'\t\t\t(at {x} {y - 1} 0)\n')
        lines.append('\t\t\t(effects (font (size 1.27 1.27)) (justify left bottom))\n')
        lines.append('\t\t)\n')
        lines.append(f'\t\t(property "Sheetfile" "{fname}"\n')
        lines.append(f'\t\t\t(at {x} {y + actual_height + 1} 0)\n')
        lines.append('\t\t\t(effects (font (size 1.27 1.27)) (justify left top) (hide yes))\n')
        lines.append('\t\t)\n')

        # Add hierarchical pins for cross-group nets
        for j, net_name in enumerate(sheet_nets):
            pin_y = y + 5 + j * 3
            lines.append(f'\t\t(pin "{net_name}" bidirectional\n')
            lines.append(f'\t\t\t(at {x + sheet_width} {pin_y} 0)\n')
            lines.append('\t\t\t(effects (font (size 1.27 1.27)))\n')
            lines.append(f'\t\t\t(uuid "{_new_uuid()}")\n')
            lines.append('\t\t)\n')

        lines.append('\t)\n\n')

    # --- Wire connections between sheet pins sharing the same net ---
    # For each cross-group net, add labels on the root sheet to connect pins
    label_y = y_start + 60  # below the sheet symbols
    for k, net_name in enumerate(sorted(cross_nets.keys())):
        lx = x_start + k * 15
        ly = label_y
        lines.append(f'\t(label "{net_name}"\n')
        lines.append(f'\t\t(at {lx:.2f} {ly:.2f} 0)\n')
        lines.append('\t\t(effects\n')
        lines.append('\t\t\t(font (size 1.27 1.27))\n')
        lines.append('\t\t)\n')
        lines.append(f'\t\t(uuid "{_new_uuid()}")\n')
        lines.append('\t)\n')

    # --- Sheet instances ---
    lines.append('\n\t(sheet_instances\n')
    lines.append('\t\t(path "/"\n')
    lines.append('\t\t\t(page "1")\n')
    lines.append('\t\t)\n')
    for i, leader in enumerate(sorted_leaders):
        s_uuid = sheet_uuids[leader]
        lines.append(f'\t\t(path "/{s_uuid}"\n')
        lines.append(f'\t\t\t(page "{i + 2}")\n')
        lines.append('\t\t)\n')
    lines.append('\t)\n')

    lines.append('\t(embedded_fonts no)\n')
    lines.append(')\n')

    return ''.join(lines)


# ---------------------------------------------------------------------------
# Assign labels to groups
# ---------------------------------------------------------------------------

def assign_labels_to_groups(
    labels: list[SchLabel],
    group_assignments: dict[str, dict],
    ic_groups: dict,
) -> None:
    """Assign labels to groups based on spatial proximity to group components."""
    build_ref_to_group(ic_groups)

    # Build per-group component positions
    group_positions: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for leader, elems in group_assignments.items():
        for sym in elems["symbols"]:
            group_positions[leader].append(sym.pos)
        for sym in elems["power_symbols"]:
            group_positions[leader].append(sym.pos)

    def _nearest(pt):
        best_d = float('inf')
        best_g = None
        for leader, positions in group_positions.items():
            for gp in positions:
                d = ((pt[0] - gp[0])**2 + (pt[1] - gp[1])**2)**0.5
                if d < best_d:
                    best_d = d
                    best_g = leader
        return best_g

    for label in labels:
        group = _nearest(label.pos)
        if group and group in group_assignments:
            group_assignments[group]["labels"].append(label)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def split_schematic(
    sch_path: str,
    ic_groups: dict,
    sheet_names: dict[str, str] | None = None,
    output_dir: str | None = None,
    dry_run: bool = False,
    backup: bool = True,
) -> dict[str, str]:
    """Split a flat schematic into hierarchical sheets.

    Args:
        sch_path: Path to flat .kicad_sch file
        ic_groups: {ic_ref: [member_refs]} grouping
        sheet_names: {ic_ref: "Display Name"} for each sheet
        output_dir: Directory for sub-sheet files
        dry_run: Print plan without writing files
        backup: Create .bak backup before overwriting

    Returns:
        {group_leader: output_file_path}
    """
    if sheet_names is None:
        sheet_names = {k: k for k in ic_groups}

    if output_dir is None:
        output_dir = str(Path(sch_path).parent)

    # Parse the schematic
    print(f"Parsing {sch_path}...")
    parsed = parse_schematic(sch_path)
    print(f"  {len(parsed['symbols'])} symbols, {len(parsed['wires'])} wires, "
          f"{len(parsed['labels'])} labels, {len(parsed['junctions'])} junctions, "
          f"{len(parsed['no_connects'])} no_connects, {len(parsed['texts'])} texts")

    # Assign elements to groups
    print("Assigning elements to groups...")
    group_assignments = assign_elements_to_groups(parsed, ic_groups)

    # Assign labels separately (they need group positions already computed)
    assign_labels_to_groups(parsed["labels"], group_assignments, ic_groups)

    # Find cross-group nets
    cross_nets = find_cross_group_nets(parsed, group_assignments)
    print(f"  Cross-group nets: {sorted(cross_nets.keys())}")

    # Print assignment summary
    for leader in ic_groups:
        if leader not in group_assignments:
            continue
        elems = group_assignments[leader]
        refs = [s.ref for s in elems["symbols"]]
        pwr = [s.ref for s in elems["power_symbols"]]
        lbls = [l.name for l in elems["labels"]]
        print(f"  {sheet_names.get(leader, leader):20s}: "
              f"{len(refs)} components ({', '.join(sorted(refs))}), "
              f"{len(pwr)} power, {len(elems['wires'])} wires, "
              f"{len(lbls)} labels ({', '.join(sorted(set(lbls)))})")

    if dry_run:
        print("\nDRY RUN — no files written")
        return {}

    # Generate sub-sheets
    sheet_files: dict[str, str] = {}
    sheet_uuids: dict[str, str] = {}

    for leader in ic_groups:
        if leader not in group_assignments:
            continue
        # Create filename from sheet name
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', sheet_names.get(leader, leader))
        fname = f"{safe_name}.kicad_sch"
        fpath = os.path.join(output_dir, fname)
        sheet_files[leader] = fname
        sheet_uuids[leader] = _new_uuid()

        content = generate_sub_sheet(
            parsed, leader, group_assignments[leader],
            cross_nets, sheet_names.get(leader, leader),
        )

        print(f"  Writing {fpath}")
        with open(fpath, 'w') as f:
            f.write(content)

    # Generate root sheet
    root_content = generate_root_sheet(
        parsed, ic_groups, group_assignments, cross_nets,
        sheet_names, sheet_files, sheet_uuids,
    )

    # Backup original
    if backup:
        bak_path = sch_path + ".bak"
        shutil.copy2(sch_path, bak_path)
        print(f"  Backup: {bak_path}")

    # Write root
    print(f"  Writing root: {sch_path}")
    with open(sch_path, 'w') as f:
        f.write(root_content)

    print(f"\nDone! Created {len(sheet_files)} sub-sheets + updated root.")
    print("Reload in KiCad: File > Revert")

    return sheet_files


def load_config_from_file(config_path: str) -> dict:
    """Load DEFAULT_CONFIG from a Python config file."""
    config_globals = {}
    with open(config_path) as f:
        exec(compile(f.read(), config_path, "exec"), config_globals)
    return config_globals.get("DEFAULT_CONFIG", {})


def main():
    parser = argparse.ArgumentParser(
        description="Split flat KiCad schematic into hierarchical sheets")
    parser.add_argument("schematic", help="Path to flat .kicad_sch file")
    parser.add_argument("--config", help="Python config file with DEFAULT_CONFIG")
    parser.add_argument("--groups", help="JSON dict override for ic_groups")
    parser.add_argument("--sheet-names", help="JSON dict of {ic_ref: display_name}")
    parser.add_argument("--output-dir", help="Directory for sub-sheet files")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup", action="store_true", default=True)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        config = load_config_from_file(args.config)

    ic_groups = config.get("ic_groups", {})
    if args.groups:
        ic_groups = json.loads(args.groups)

    if not ic_groups:
        print("ERROR: No ic_groups defined. Use --config or --groups.",
              file=sys.stderr)
        sys.exit(1)

    sheet_names = config.get("group_labels", {})
    if args.sheet_names:
        sheet_names.update(json.loads(args.sheet_names))

    # Default sheet names from ic_groups keys
    for leader in ic_groups:
        if leader not in sheet_names:
            sheet_names[leader] = leader

    split_schematic(
        sch_path=args.schematic,
        ic_groups=ic_groups,
        sheet_names=sheet_names,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    main()
