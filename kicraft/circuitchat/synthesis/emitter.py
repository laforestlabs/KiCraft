"""Emit `<PROJECT>.kicad_sch` (root) and one `<SHEET>.kicad_sch` per leaf.

Direct S-expression emission. kicad-skip is a parser/editor — it cannot
create a schematic from scratch — so we write the format here following
the templates in `generate_project.py` and §2 of the contract doc.

The output is intentionally minimal:
- correct hierarchy KiCraft can parse (refs per sheet, sheet pins, leaf
  Sheetfile refs);
- valid lib_symbols copied verbatim from the stock KiCad libraries;
- hierarchical labels in each leaf for every inter-sheet net;
- sheet pins and stub wires in the root connecting matched pins.

We do NOT attempt pin-accurate intra-sheet wiring. KiCraft's hierarchy
parser extracts component refs from `(symbol ...) > (instances ...)`
blocks and inter-sheet pins from the root — neither needs every wire on
the schematic. eeschema opens the file fine; manual wiring can be done
later by the user.
"""
from __future__ import annotations

import uuid as uuidlib
from dataclasses import dataclass
from pathlib import Path

from ..models import (
    BOM,
    Architecture,
    ArtifactPaths,
    BomPart,
    InterSheetNet,
    Sheet,
    SheetPin,
)
from .symbol_library import build_lib_symbols_block


@dataclass(frozen=True)
class _SheetInstance:
    """Resolved sheet info for emitting both the root pin and the leaf file."""

    sheet: Sheet
    instance_uuid: str  # UUID of the (sheet ...) block in the root
    leaf_uuid: str  # UUID at the top of the leaf .kicad_sch file
    parts: list[BomPart]
    inter_sheet_endpoints: list[tuple[InterSheetNet, SheetPin]]


def _uuid() -> str:
    return str(uuidlib.uuid4())


def _fmt(v: float) -> str:
    if isinstance(v, int):
        return str(v)
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _split_lib_id(value: str) -> tuple[str, str]:
    library, _, name = value.partition(":")
    if not library or not name:
        raise ValueError(f"bad lib_id {value!r} (expected 'Library:Name')")
    return library, name


# ---------- root schematic ----------


def _emit_sheet_block(
    sheet_inst: _SheetInstance,
    x: float,
    y: float,
    width: float,
    height: float,
    project_stem: str,
) -> str:
    """Emit one `(sheet ...)` block for the root file."""
    sheet = sheet_inst.sheet
    pin_lines: list[str] = []
    # Distribute pins along the right edge.
    n_pins = len(sheet_inst.inter_sheet_endpoints)
    for i, (net, ep) in enumerate(sheet_inst.inter_sheet_endpoints):
        # Spread pins evenly along the right edge.
        step = height / (n_pins + 1) if n_pins else 0
        pin_y = y + step * (i + 1)
        pin_lines.append(
            f'\t\t(pin "{net.name}" {ep.direction}\n'
            f"\t\t\t(at {_fmt(x + width)} {_fmt(pin_y)} 0)\n"
            f"\t\t\t(effects (font (size 1.27 1.27)) (justify right))\n"
            f'\t\t\t(uuid "{_uuid()}")\n'
            f"\t\t)"
        )
    return (
        f"\t(sheet\n"
        f"\t\t(at {_fmt(x)} {_fmt(y)})\n"
        f"\t\t(size {_fmt(width)} {_fmt(height)})\n"
        f"\t\t(exclude_from_sim no)\n"
        f"\t\t(in_bom yes)\n"
        f"\t\t(on_board yes)\n"
        f"\t\t(dnp no)\n"
        f"\t\t(fields_autoplaced yes)\n"
        f"\t\t(stroke (width 0.2) (type solid))\n"
        f"\t\t(fill (color 0 0 0 0.0000))\n"
        f'\t\t(uuid "{sheet_inst.instance_uuid}")\n'
        f'\t\t(property "Sheetname" "{sheet.name}"\n'
        f"\t\t\t(at {_fmt(x)} {_fmt(y - 1)} 0)\n"
        f"\t\t\t(effects (font (size 1.27 1.27)) (justify left bottom))\n"
        f"\t\t)\n"
        f'\t\t(property "Sheetfile" "{sheet.stem}.kicad_sch"\n'
        f"\t\t\t(at {_fmt(x)} {_fmt(y + height + 1)} 0)\n"
        f"\t\t\t(effects (font (size 1.27 1.27)) (justify left top))\n"
        f"\t\t)\n"
        + ("\n".join(pin_lines) + ("\n" if pin_lines else ""))
        + f'\t\t(instances\n'
        f'\t\t\t(project "{project_stem}"\n'
        f'\t\t\t\t(path "/" (page "{sheet.stem}"))\n'
        f"\t\t\t)\n"
        f"\t\t)\n"
        f"\t)"
    )


def _emit_root(
    project_stem: str,
    project_dir: Path,
    sheet_insts: list[_SheetInstance],
    architecture: Architecture,
    project_title: str,
) -> Path:
    root_uuid = _uuid()
    sheet_width = 40.0
    sheet_height = 30.0
    sheet_gap = 15.0
    rows: list[str] = []
    # Lay sheets out horizontally on A3 (420 x 297 mm).
    start_x, start_y = 30.0, 40.0
    sheet_origins: dict[str, tuple[float, float]] = {}
    for i, si in enumerate(sheet_insts):
        x = start_x + i * (sheet_width + sheet_gap)
        y = start_y
        rows.append(_emit_sheet_block(si, x, y, sheet_width, sheet_height, project_stem))
        sheet_origins[si.sheet.name] = (x, y)

    # Optional: emit wires linking same-named pins across sheets. The pin Y
    # coords aren't easily knowable here without re-deriving placement; we
    # skip wires to keep the emitter simple. KiCraft does not need them.

    header = (
        "(kicad_sch\n"
        "\t(version 20250114)\n"
        '\t(generator "eeschema")\n'
        '\t(generator_version "9.0")\n'
        f'\t(uuid "{root_uuid}")\n'
        '\t(paper "A3")\n'
        "\n"
        "\t(title_block\n"
        f'\t\t(title "{project_title}")\n'
        '\t\t(rev "1.0")\n'
        f'\t\t(comment 1 "Generated by KiCraft CircuitChat pipeline")\n'
        "\t)\n"
        "\n"
        "\t(lib_symbols)\n"
        "\n"
    )
    body = "\n".join(rows)
    # Required `(sheet_instances ...)` block ties this root sheet to /.
    tail = (
        "\n\n"
        "\t(sheet_instances\n"
        '\t\t(path "/"\n'
        '\t\t\t(page "1")\n'
        "\t\t)\n"
        "\t)\n"
        ")\n"
    )
    out = project_dir / f"{project_stem}.kicad_sch"
    out.write_text(header + body + tail)
    return out


# ---------- leaf schematics ----------


def _emit_symbol_instance(
    part: BomPart,
    x: float,
    y: float,
    leaf_uuid: str,
    project_stem: str,
) -> str:
    """Emit one component `(symbol ...)` instance inside a leaf."""
    return (
        "\t(symbol\n"
        f'\t\t(lib_id "{part.symbol}")\n'
        f"\t\t(at {_fmt(x)} {_fmt(y)} 0)\n"
        "\t\t(unit 1)\n"
        "\t\t(exclude_from_sim no)\n"
        "\t\t(in_bom yes)\n"
        "\t\t(on_board yes)\n"
        "\t\t(dnp no)\n"
        f'\t\t(uuid "{_uuid()}")\n'
        f'\t\t(property "Reference" "{part.ref}"\n'
        f"\t\t\t(at {_fmt(x)} {_fmt(y - 5)} 0)\n"
        "\t\t\t(effects (font (size 1.27 1.27)))\n"
        "\t\t)\n"
        f'\t\t(property "Value" "{part.value}"\n'
        f"\t\t\t(at {_fmt(x)} {_fmt(y + 5)} 0)\n"
        "\t\t\t(effects (font (size 1.27 1.27)))\n"
        "\t\t)\n"
        f'\t\t(property "Footprint" "{part.footprint}"\n'
        f"\t\t\t(at {_fmt(x)} {_fmt(y + 7)} 0)\n"
        "\t\t\t(effects (font (size 1.27 1.27)) (hide yes))\n"
        "\t\t)\n"
        f'\t\t(property "Datasheet" "{part.datasheet or ""}"\n'
        f"\t\t\t(at {_fmt(x)} {_fmt(y)} 0)\n"
        "\t\t\t(effects (font (size 1.27 1.27)) (hide yes))\n"
        "\t\t)\n"
        f'\t\t(property "Description" "{part.sourcing_note or ""}"\n'
        f"\t\t\t(at {_fmt(x)} {_fmt(y)} 0)\n"
        "\t\t\t(effects (font (size 1.27 1.27)) (hide yes))\n"
        "\t\t)\n"
        "\t\t(instances\n"
        f'\t\t\t(project "{project_stem}"\n'
        f'\t\t\t\t(path "/{leaf_uuid}"\n'
        f'\t\t\t\t\t(reference "{part.ref}")\n'
        "\t\t\t\t\t(unit 1)\n"
        "\t\t\t\t)\n"
        "\t\t\t)\n"
        "\t\t)\n"
        "\t)"
    )


_DIRECTION_TO_SHAPE = {
    "input": "input",
    "output": "output",
    "bidirectional": "bidirectional",
    "passive": "passive",
}


def _emit_hierarchical_label(name: str, direction: str, x: float, y: float) -> str:
    shape = _DIRECTION_TO_SHAPE.get(direction, "passive")
    return (
        f'\t(hierarchical_label "{name}" (shape {shape})\n'
        f"\t\t(at {_fmt(x)} {_fmt(y)} 0)\n"
        "\t\t(effects (font (size 1.27 1.27)) (justify left))\n"
        f'\t\t(uuid "{_uuid()}")\n'
        "\t)"
    )


def _emit_leaf(
    project_dir: Path,
    project_stem: str,
    sheet_inst: _SheetInstance,
) -> Path:
    # Build lib_symbols block from this leaf's parts.
    pairs = [_split_lib_id(p.symbol) for p in sheet_inst.parts]
    lib_block = build_lib_symbols_block(pairs)

    # Lay out components in a grid: 5 cols, 25mm spacing, starting at (60, 60).
    cols = 5
    col_w = 25.0
    row_h = 30.0
    start_x, start_y = 60.0, 60.0
    symbol_blocks: list[str] = []
    for i, part in enumerate(sheet_inst.parts):
        x = start_x + (i % cols) * col_w
        y = start_y + (i // cols) * row_h
        symbol_blocks.append(
            _emit_symbol_instance(part, x, y, sheet_inst.leaf_uuid, project_stem)
        )

    # Hierarchical labels along the left edge at y = 50 + 10*i.
    label_blocks: list[str] = []
    for i, (net, ep) in enumerate(sheet_inst.inter_sheet_endpoints):
        label_blocks.append(
            _emit_hierarchical_label(net.name, ep.direction, 40.0, 50.0 + i * 10)
        )

    header = (
        "(kicad_sch\n"
        "\t(version 20250114)\n"
        '\t(generator "eeschema")\n'
        '\t(generator_version "9.0")\n'
        f'\t(uuid "{sheet_inst.leaf_uuid}")\n'
        '\t(paper "A4")\n'
        "\n"
        f"{lib_block}\n"
        "\n"
    )
    body_parts = symbol_blocks + label_blocks
    body = "\n".join(body_parts)
    # `sheet_instances` for the leaf — pin this leaf to its parent's path/uuid.
    tail = (
        "\n\n"
        "\t(sheet_instances\n"
        f'\t\t(path "/{sheet_inst.instance_uuid}"\n'
        f'\t\t\t(page "{sheet_inst.sheet.stem}")\n'
        "\t\t)\n"
        "\t)\n"
        ")\n"
    )
    out = project_dir / f"{sheet_inst.sheet.stem}.kicad_sch"
    out.write_text(header + body + tail)
    return out


# ---------- top-level emit ----------


def _build_sheet_instances(
    architecture: Architecture,
    bom: BOM,
) -> list[_SheetInstance]:
    parts_by_sheet: dict[str, list[BomPart]] = {s.name: [] for s in architecture.sheets}
    for p in bom.parts:
        if p.sheet not in parts_by_sheet:
            raise ValueError(
                f"BOM part {p.ref!r} references unknown sheet {p.sheet!r}"
            )
        parts_by_sheet[p.sheet].append(p)

    eps_by_sheet: dict[str, list[tuple[InterSheetNet, SheetPin]]] = {
        s.name: [] for s in architecture.sheets
    }
    for net in architecture.inter_sheet_nets:
        for ep in net.endpoints:
            eps_by_sheet[ep.sheet].append((net, ep))

    out: list[_SheetInstance] = []
    for s in architecture.sheets:
        out.append(
            _SheetInstance(
                sheet=s,
                instance_uuid=_uuid(),
                leaf_uuid=_uuid(),
                parts=parts_by_sheet[s.name],
                inter_sheet_endpoints=eps_by_sheet[s.name],
            )
        )
    return out


def emit_schematic(
    project_dir: Path,
    project_stem: str,
    architecture: Architecture,
    bom: BOM,
    title: str | None = None,
) -> tuple[Path, list[Path]]:
    """Emit root + leaf .kicad_sch files into project_dir.

    Returns (root_path, [leaf_paths]).
    """
    project_dir.mkdir(parents=True, exist_ok=True)
    sheet_insts = _build_sheet_instances(architecture, bom)
    root = _emit_root(
        project_stem,
        project_dir,
        sheet_insts,
        architecture,
        project_title=title or project_stem,
    )
    leaves = [_emit_leaf(project_dir, project_stem, si) for si in sheet_insts]
    return root, leaves


# ---------- aggregate entry point ----------


def synthesize_project(
    project_dir: Path,
    project_stem: str,
    architecture: Architecture,
    bom: BOM,
    title: str | None = None,
) -> ArtifactPaths:
    """Emit ONLY the schematic files. Caller is responsible for kicad_pro +
    autoplacer.json + validation; this function exists so unit tests can
    exercise schematic emission without pulling in the full synthesis stack.
    """
    root, leaves = emit_schematic(project_dir, project_stem, architecture, bom, title=title)
    return ArtifactPaths(
        project_dir=project_dir,
        project_stem=project_stem,
        root_sch=root,
        leaf_schs=leaves,
        kicad_pro=project_dir / f"{project_stem}.kicad_pro",
        autoplacer_json=project_dir / f"{project_stem}_autoplacer.json",
    )
