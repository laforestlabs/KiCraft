#!/usr/bin/env python3
"""Add silkscreen group labels to a KiCad PCB based on ic_groups config.

Computes the bounding box of each IC group's components on the PCB,
then inserts a text label on the appropriate silkscreen layer (F.SilkS
or B.SilkS depending on which side the majority of the group's
components are on).

Generalized: works with any .kicad_pcb and any ic_groups/group_labels
config. No pcbnew dependency — operates on S-expression text directly.

Usage:
    python3 add_group_labels.py <pcb_file> [options]

Options:
    --config <path>      Python config file with DEFAULT_CONFIG dict
    --labels <json>      JSON dict of {ic_ref: "Label Text"} overrides
    --font-height <mm>   Label font height in mm (default: 1.0)
    --font-width <mm>    Label font width in mm (default: 1.0)
    --font-thickness <mm> Stroke thickness in mm (default: 0.15)
    --offset-y <mm>      Vertical offset above group top edge (default: 1.5)
    --in-place           Overwrite the input file (default: saves as _labeled.kicad_pcb)
    --dry-run            Print label positions without modifying the file
"""
import argparse
import json
import math
import re
import sys
import uuid
from collections import defaultdict
from pathlib import Path

# Marker prefix for idempotent label management
GROUP_LABEL_MARKER = "group_label_"
GROUP_BOX_MARKER = "group_box_"


def parse_footprints(pcb_text: str) -> dict:
    """Extract footprint reference, position, and layer from PCB text.

    Returns dict: {ref: {"x": float, "y": float, "layer": str}}
    """
    footprints = {}

    # Match top-level footprint blocks and extract reference + position + layer
    # KiCad footprints start with (footprint "..." and contain property "Reference"
    fp_pattern = re.compile(
        r'\(footprint\s+"[^"]*"'
        r'(?:\s+\(locked\))?'            # optional locked flag
        r'\s+\(layer\s+"([^"]+)"\)'       # layer
        r'\s+\((?:tedit|tstamp|uuid)\s+[^)]+\)'  # tstamp/uuid
        r'\s+\(at\s+([-\d.]+)\s+([-\d.]+)',  # position x y
        re.DOTALL
    )

    ref_pattern = re.compile(
        r'\(property\s+"Reference"\s+"([^"]+)"'
    )

    # Split into footprint blocks for reliable parsing
    # Find all footprint start positions
    fp_starts = [m.start() for m in re.finditer(r'^\t?\(footprint\s+"', pcb_text, re.MULTILINE)]

    for i, start in enumerate(fp_starts):
        # Find the end of this footprint block (next footprint or end of footprints section)
        end = fp_starts[i + 1] if i + 1 < len(fp_starts) else len(pcb_text)
        block = pcb_text[start:end]

        # Extract layer
        layer_m = re.search(r'\(layer\s+"([^"]+)"\)', block)
        if not layer_m:
            continue
        layer = layer_m.group(1)

        # Extract position (the first 'at' after the layer line is the footprint position)
        at_m = re.search(r'\(at\s+([-\d.]+)\s+([-\d.]+)', block)
        if not at_m:
            continue
        x, y = float(at_m.group(1)), float(at_m.group(2))

        # Extract reference
        ref_m = ref_pattern.search(block)
        if not ref_m:
            continue
        ref = ref_m.group(1)

        footprints[ref] = {"x": x, "y": y, "layer": layer}

    return footprints


def load_config_from_file(config_path: str) -> dict:
    """Load DEFAULT_CONFIG from a Python config file."""
    config_globals = {}
    with open(config_path) as f:
        exec(compile(f.read(), config_path, "exec"), config_globals)
    return config_globals.get("DEFAULT_CONFIG", {})


def compute_group_bounds(footprints: dict, ic_groups: dict) -> dict:
    """Compute bounding box and centroid for each IC group.

    Returns dict: {ic_ref: {"min_x", "max_x", "min_y", "max_y",
                             "cx", "cy", "layer"}}
    """
    group_bounds = {}

    for ic_ref, members in ic_groups.items():
        all_refs = [ic_ref] + list(members)
        group_fps = [footprints[r] for r in all_refs if r in footprints]
        if not group_fps:
            continue

        xs = [fp["x"] for fp in group_fps]
        ys = [fp["y"] for fp in group_fps]

        # Determine dominant layer (majority vote)
        layer_counts = defaultdict(int)
        for fp in group_fps:
            layer_counts[fp["layer"]] += 1
        dominant_layer = max(layer_counts, key=layer_counts.get)

        group_bounds[ic_ref] = {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
            "cx": sum(xs) / len(xs),
            "cy": sum(ys) / len(ys),
            "layer": dominant_layer,
        }

    return group_bounds


def silk_layer_for(copper_layer: str) -> str:
    """Map copper layer to corresponding silkscreen layer."""
    if copper_layer == "B.Cu":
        return "B.SilkS"
    return "F.SilkS"


def remove_existing_group_labels(pcb_text: str) -> str:
    """Remove any previously-added group labels and boxes (idempotent)."""
    # Remove gr_text blocks that contain our label marker in their uuid
    text_pattern = re.compile(
        r'\t\(gr_text\s+"[^"]*"\s*\n'
        r'(?:\t\t[^\n]*\n)*?'
        r'\t\t\(uuid\s+"' + GROUP_LABEL_MARKER + r'[^"]*"\)\n'
        r'(?:\t\t[^\n]*\n)*?'
        r'\t\)\n',
        re.MULTILINE
    )
    pcb_text = text_pattern.sub('', pcb_text)
    # Remove gr_poly blocks that contain our box marker in their uuid
    poly_pattern = re.compile(
        r'\t\(gr_poly\s*\n'
        r'(?:\t\t[^\n]*\n)*?'
        r'\t\t\(uuid\s+"' + GROUP_BOX_MARKER + r'[^"]*"\)\n'
        r'(?:\t\t[^\n]*\n)*?'
        r'\t\)\n',
        re.MULTILINE
    )
    pcb_text = poly_pattern.sub('', pcb_text)
    return pcb_text


def make_gr_text(label: str, x: float, y: float, layer: str,
                 font_h: float, font_w: float, thickness: float,
                 uid: str) -> str:
    """Generate a KiCad gr_text S-expression."""
    return (
        f'\t(gr_text "{label}"\n'
        f'\t\t(at {x:.2f} {y:.2f})\n'
        f'\t\t(layer "{layer}")\n'
        f'\t\t(uuid "{uid}")\n'
        f'\t\t(effects\n'
        f'\t\t\t(font\n'
        f'\t\t\t\t(size {font_h} {font_w})\n'
        f'\t\t\t\t(thickness {thickness})\n'
        f'\t\t\t)\n'
        f'\t\t\t(justify left)\n'
        f'\t\t)\n'
        f'\t)\n'
    )


def make_gr_poly_roundrect(x0: float, y0: float, x1: float, y1: float,
                           radius: float, layer: str, stroke_width: float,
                           uid: str) -> str:
    """Generate a KiCad gr_poly S-expression for a rounded rectangle.

    Uses 8 points per corner (32 total) to approximate arc corners.
    """
    r = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
    pts = []
    # Generate corner arcs (clockwise from top-left)
    corners = [
        (x0 + r, y0 + r, math.pi, math.pi / 2),        # top-left
        (x1 - r, y0 + r, math.pi / 2, 0),               # top-right
        (x1 - r, y1 - r, 0, -math.pi / 2),              # bottom-right
        (x0 + r, y1 - r, -math.pi / 2, -math.pi),       # bottom-left
    ]
    n_arc = 8  # points per corner arc
    for cx, cy, a_start, a_end in corners:
        for i in range(n_arc):
            t = a_start + (a_end - a_start) * i / (n_arc - 1)
            px = cx + r * math.cos(t)
            py = cy - r * math.sin(t)  # KiCad Y-down
            pts.append(f'\t\t\t(xy {px:.3f} {py:.3f})')

    pts_str = '\n'.join(pts)
    return (
        f'\t(gr_poly\n'
        f'\t\t(pts\n'
        f'{pts_str}\n'
        f'\t\t)\n'
        f'\t\t(stroke\n'
        f'\t\t\t(width {stroke_width})\n'
        f'\t\t\t(type solid)\n'
        f'\t\t)\n'
        f'\t\t(fill no)\n'
        f'\t\t(layer "{layer}")\n'
        f'\t\t(uuid "{uid}")\n'
        f'\t)\n'
    )


def _label_rect(x: float, y: float, text: str, font_h: float, font_w: float):
    """Return (x_min, y_min, x_max, y_max) bounding box of a label."""
    text_w = len(text) * font_w * 0.7  # approximate glyph width ratio
    return (x - 0.2, y - font_h / 2, x + text_w + 0.2, y + font_h / 2)


def _rects_overlap(r1, r2) -> bool:
    """Check if two (x_min, y_min, x_max, y_max) rectangles overlap."""
    return not (r1[2] < r2[0] or r2[2] < r1[0] or r1[3] < r2[1] or r2[3] < r1[1])


def _component_rects(footprints: dict, ic_groups: dict, ic_ref: str) -> list:
    """Get approximate bounding boxes for all components in a group."""
    rects = []
    members = [ic_ref] + list(ic_groups.get(ic_ref, []))
    for ref in members:
        fp = footprints.get(ref)
        if not fp:
            continue
        # Approximate component as 3mm x 2mm rect centered at its position
        hw, hh = 1.5, 1.0
        rects.append((fp["x"] - hw, fp["y"] - hh, fp["x"] + hw, fp["y"] + hh))
    return rects


def add_group_labels(pcb_path: str, ic_groups: dict, group_labels: dict,
                     font_height: float = 0.0, font_width: float = 0.0,
                     font_thickness: float = 0.15, offset_y: float = 1.5,
                     box_padding: float = 1.0, box_radius: float = 0.8,
                     box_stroke: float = 0.15,
                     in_place: bool = False, dry_run: bool = False) -> str:
    """Add silkscreen group labels with rounded rectangle boxes to a KiCad PCB.

    For each IC group, draws a rounded rectangle around the group's
    bounding box (with padding) and a text label centered on the top
    edge of the box.  Font size auto-scales to group width when
    font_height/font_width are 0 (default).

    Args:
        pcb_path: Path to .kicad_pcb file
        ic_groups: {ic_ref: [member_refs]} — component grouping
        group_labels: {ic_ref: "Label Text"} — display names
        font_height: Text height in mm (0 = auto-scale to group size)
        font_width: Text width in mm (0 = auto-scale to group size)
        font_thickness: Stroke thickness in mm
        offset_y: Vertical offset from group edge in mm (fallback positioning)
        box_padding: Padding around group bounding box for the box (mm)
        box_radius: Corner radius for the rounded rectangle (mm)
        box_stroke: Stroke width for the box outline (mm)
        in_place: Overwrite input file if True
        dry_run: Only print positions, don't modify

    Returns:
        Output file path
    """
    with open(pcb_path) as f:
        pcb_text = f.read()

    # Parse footprint positions
    footprints = parse_footprints(pcb_text)
    if not footprints:
        print("ERROR: No footprints found in PCB file", file=sys.stderr)
        sys.exit(1)

    # Remove old group labels and boxes (idempotent)
    pcb_text = remove_existing_group_labels(pcb_text)

    # Compute group bounding boxes
    bounds = compute_group_bounds(footprints, ic_groups)

    # Generate labels and boxes
    labels_to_add = []
    boxes_to_add = []
    for ic_ref in sorted(bounds.keys()):
        if ic_ref not in group_labels:
            continue
        b = bounds[ic_ref]
        label_text = group_labels[ic_ref]
        silk = silk_layer_for(b["layer"])

        # Auto-scale font to group width
        group_w = b["max_x"] - b["min_x"]
        if font_height <= 0 or font_width <= 0:
            auto_h = max(0.6, min(1.5, group_w / 8.0))
            fh = auto_h
            fw = auto_h
        else:
            fh = font_height
            fw = font_width

        # Box around the group bounding box (with padding)
        bx0 = b["min_x"] - box_padding - 1.5  # extra margin for component size
        by0 = b["min_y"] - box_padding - 1.0
        bx1 = b["max_x"] + box_padding + 1.5
        by1 = b["max_y"] + box_padding + 1.0

        box_uid = f"{GROUP_BOX_MARKER}{ic_ref.lower()}_{uuid.uuid4().hex[:8]}"
        boxes_to_add.append({
            "ic_ref": ic_ref,
            "x0": bx0, "y0": by0, "x1": bx1, "y1": by1,
            "layer": silk,
            "uid": box_uid,
        })

        # Position label centered on the top edge of the box
        text_w = len(label_text) * fw * 0.7
        lx = (bx0 + bx1) / 2 - text_w / 2  # left-justified from center
        ly = by0  # on the top edge line

        label_uid = f"{GROUP_LABEL_MARKER}{ic_ref.lower()}_{uuid.uuid4().hex[:8]}"
        labels_to_add.append({
            "ic_ref": ic_ref,
            "text": label_text,
            "x": lx,
            "y": ly,
            "layer": silk,
            "uid": label_uid,
            "font_h": fh,
            "font_w": fw,
        })

    # Print summary
    print(f"{'DRY RUN: ' if dry_run else ''}Group labels for {pcb_path}:")
    for lbl in labels_to_add:
        print(f"  {lbl['ic_ref']:4s} \"{lbl['text']}\" @ ({lbl['x']:.1f}, {lbl['y']:.1f}) on {lbl['layer']}")

    if dry_run:
        return pcb_path

    # Build S-expressions for boxes and labels
    gr_elements = ""
    for box in boxes_to_add:
        gr_elements += make_gr_poly_roundrect(
            box["x0"], box["y0"], box["x1"], box["y1"],
            box_radius, box["layer"], box_stroke, box["uid"]
        )
    for lbl in labels_to_add:
        gr_elements += make_gr_text(
            lbl["text"], lbl["x"], lbl["y"], lbl["layer"],
            lbl["font_h"], lbl["font_w"], font_thickness, lbl["uid"]
        )

    # Insert before the final closing paren
    insert_pos = pcb_text.rfind("\n)")
    if insert_pos == -1:
        print("ERROR: Could not find insertion point in PCB file", file=sys.stderr)
        sys.exit(1)
    pcb_text = pcb_text[:insert_pos] + "\n" + gr_elements + pcb_text[insert_pos:]

    # Write output
    if in_place:
        out_path = pcb_path
    else:
        p = Path(pcb_path)
        out_path = str(p.with_stem(p.stem + "_labeled"))

    with open(out_path, "w") as f:
        f.write(pcb_text)
    print(f"Saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Add silkscreen group labels to a KiCad PCB")
    parser.add_argument("pcb", help="Path to .kicad_pcb file")
    parser.add_argument("--config", help="Python config file with DEFAULT_CONFIG")
    parser.add_argument("--labels", help="JSON dict of {ic_ref: label_text} overrides")
    parser.add_argument("--font-height", type=float, default=1.0,
                        help="Label font height in mm (default: 1.0)")
    parser.add_argument("--font-width", type=float, default=1.0,
                        help="Label font width in mm (default: 1.0)")
    parser.add_argument("--font-thickness", type=float, default=0.15,
                        help="Stroke thickness in mm (default: 0.15)")
    parser.add_argument("--offset-y", type=float, default=1.5,
                        help="Vertical offset above group top in mm (default: 1.5)")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite input file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print positions without modifying")
    args = parser.parse_args()

    # Load config
    config = {}
    if args.config:
        config = load_config_from_file(args.config)

    ic_groups = config.get("ic_groups", {})
    group_labels = config.get("group_labels", {})

    if not ic_groups:
        print("ERROR: No ic_groups defined. Use --config or ensure DEFAULT_CONFIG has ic_groups.",
              file=sys.stderr)
        sys.exit(1)

    # Override labels from CLI
    if args.labels:
        group_labels.update(json.loads(args.labels))

    if not group_labels:
        # Fall back to using IC references as labels
        group_labels = {ic: ic for ic in ic_groups}

    add_group_labels(
        pcb_path=args.pcb,
        ic_groups=ic_groups,
        group_labels=group_labels,
        font_height=args.font_height,
        font_width=args.font_width,
        font_thickness=args.font_thickness,
        offset_y=args.offset_y,
        in_place=args.in_place,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
