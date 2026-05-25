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

import hashlib
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
    is_power_or_ground_name,
)
from .placement import place_sheet
from .router import (
    Junction,
    NetLabel,
    NoConnect,
    PowerSymbol,
    RoutedSheet,
    WireSegment,
    power_symbol_for,
    route_sheet,
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


_PROJECT_NAMESPACE_CACHE: dict[str, uuidlib.UUID] = {}


def _project_namespace(project_stem: str) -> uuidlib.UUID:
    if project_stem not in _PROJECT_NAMESPACE_CACHE:
        _PROJECT_NAMESPACE_CACHE[project_stem] = uuidlib.uuid5(
            uuidlib.NAMESPACE_URL, f"kicraft://{project_stem}"
        )
    return _PROJECT_NAMESPACE_CACHE[project_stem]


def _uuid_seeded(salt: str, project_stem: str) -> str:
    """Deterministic UUIDv5 derived from `project_stem` + `salt`.

    Used for Stage B emit helpers so re-running synthesis on the same
    state.json yields byte-identical schematic UUIDs (modulo the legacy
    random root/sheet UUIDs left unchanged for now).
    """
    return str(uuidlib.uuid5(_project_namespace(project_stem), salt))


def _power_ref_for_salt(salt: str) -> str:
    """Stable, very-unlikely-to-collide ref for a power symbol instance."""
    h = int(hashlib.md5(salt.encode()).hexdigest()[:8], 16)
    return f"#PWR{h % 100000:05d}"


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
) -> tuple[str, list[tuple[str, float, float]]]:
    """Emit one `(sheet ...)` block for the root file.

    Returns ``(block_text, signal_pin_records)`` where each record is
    ``(net_name, pin_x, pin_y)`` for a signal sheet pin on the right edge,
    so the caller can wire same-named pins together on the root canvas.
    """
    sheet = sheet_inst.sheet
    pin_lines: list[str] = []
    pin_records: list[tuple[str, float, float]] = []
    # Distribute pins along the right edge. Skip power/global nets: they
    # connect across sheets via global power symbols in the leaves, so a
    # root sheet pin for them would dangle (there is no matching leaf
    # hierarchical label to connect down to).
    signal_endpoints = [
        (net, ep)
        for (net, ep) in sheet_inst.inter_sheet_endpoints
        if not is_power_or_ground_name(net.name)
    ]
    n_pins = len(signal_endpoints)
    for i, (net, ep) in enumerate(signal_endpoints):
        # Spread pins evenly along the right edge.
        step = height / (n_pins + 1) if n_pins else 0
        # Snap to the 1.27 mm grid so the pin (and its root stub) is on-grid.
        pin_y = round((y + step * (i + 1)) / 1.27) * 1.27
        pin_records.append((net.name, x + width, pin_y))
        pin_lines.append(
            f'\t\t(pin "{net.name}" {ep.direction}\n'
            f"\t\t\t(at {_fmt(x + width)} {_fmt(pin_y)} 0)\n"
            f"\t\t\t(effects (font (size 1.27 1.27)) (justify right))\n"
            f'\t\t\t(uuid "{_uuid()}")\n'
            f"\t\t)"
        )
    block = (
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
    return block, pin_records


def _emit_root(
    project_stem: str,
    project_dir: Path,
    sheet_insts: list[_SheetInstance],
    architecture: Architecture,
    project_title: str,
) -> Path:
    root_uuid = _uuid()
    # Grid-aligned layout (1.27 mm multiples) so sheet pins — and the stubs
    # and labels that connect them — land on grid (no endpoint_off_grid).
    sheet_width = 38.1     # 30 * 1.27
    sheet_height = 30.48   # 24 * 1.27
    sheet_gap = 15.24      # 12 * 1.27
    rows: list[str] = []
    # Lay sheets out horizontally on A3 (420 x 297 mm).
    start_x, start_y = 30.48, 38.1
    sheet_origins: dict[str, tuple[float, float]] = {}
    pin_records: list[tuple[str, float, float]] = []
    for i, si in enumerate(sheet_insts):
        x = start_x + i * (sheet_width + sheet_gap)
        y = start_y
        block, recs = _emit_sheet_block(
            si, x, y, sheet_width, sheet_height, project_stem
        )
        rows.append(block)
        pin_records.extend(recs)
        sheet_origins[si.sheet.name] = (x, y)

    # Connect same-named signal sheet pins across sheets: a short stub + a
    # local label off each pin. Same-named local labels merge on the root
    # canvas, so e.g. the SDA pin on MCU and the SDA pin on SENSOR become
    # one net. (Power nets are not emitted as sheet pins — they connect
    # globally via power symbols in the leaves.)
    connect_rows: list[str] = []
    for idx, (net_name, px, py) in enumerate(pin_records):
        stub_end = px + sheet_gap * 0.5
        connect_rows.append(
            _emit_wire(
                WireSegment(px, py, stub_end, py),
                f"rootpin/wire/{idx}", project_stem,
            )
        )
        connect_rows.append(
            _emit_net_label(
                NetLabel(text=net_name, x_mm=stub_end, y_mm=py),
                f"rootpin/label/{idx}", project_stem,
            )
        )

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
    body = "\n".join(rows + connect_rows)
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
    *,
    rotation_deg: int = 0,
    salt: str | None = None,
) -> str:
    """Emit one component `(symbol ...)` instance inside a leaf."""
    uuid_str = _uuid_seeded(salt, project_stem) if salt else _uuid()
    return (
        "\t(symbol\n"
        f'\t\t(lib_id "{part.symbol}")\n'
        f"\t\t(at {_fmt(x)} {_fmt(y)} {rotation_deg})\n"
        "\t\t(unit 1)\n"
        "\t\t(exclude_from_sim no)\n"
        "\t\t(in_bom yes)\n"
        "\t\t(on_board yes)\n"
        "\t\t(dnp no)\n"
        f'\t\t(uuid "{uuid_str}")\n'
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


def _emit_hierarchical_label(
    name: str, direction: str, x: float, y: float,
    *, salt: str | None = None, project_stem: str | None = None,
) -> str:
    shape = _DIRECTION_TO_SHAPE.get(direction, "passive")
    if salt and project_stem:
        uuid_str = _uuid_seeded(salt, project_stem)
    else:
        uuid_str = _uuid()
    return (
        f'\t(hierarchical_label "{name}" (shape {shape})\n'
        f"\t\t(at {_fmt(x)} {_fmt(y)} 0)\n"
        "\t\t(effects (font (size 1.27 1.27)) (justify left))\n"
        f'\t\t(uuid "{uuid_str}")\n'
        "\t)"
    )


# ---------- Stage B emit helpers (wires, junctions, power, labels, NC) ----------


def _emit_wire(w: WireSegment, salt: str, project_stem: str) -> str:
    return (
        "\t(wire\n"
        f"\t\t(pts (xy {_fmt(w.x1_mm)} {_fmt(w.y1_mm)}) "
        f"(xy {_fmt(w.x2_mm)} {_fmt(w.y2_mm)}))\n"
        "\t\t(stroke (width 0) (type default))\n"
        f'\t\t(uuid "{_uuid_seeded(salt, project_stem)}")\n'
        "\t)"
    )


def _emit_junction(j: Junction, salt: str, project_stem: str) -> str:
    return (
        "\t(junction\n"
        f"\t\t(at {_fmt(j.x_mm)} {_fmt(j.y_mm)})\n"
        "\t\t(diameter 0)\n"
        "\t\t(color 0 0 0 0)\n"
        f'\t\t(uuid "{_uuid_seeded(salt, project_stem)}")\n'
        "\t)"
    )


def _emit_net_label(lab: NetLabel, salt: str, project_stem: str) -> str:
    return (
        f'\t(label "{lab.text}"\n'
        f"\t\t(at {_fmt(lab.x_mm)} {_fmt(lab.y_mm)} {lab.angle_deg})\n"
        "\t\t(effects (font (size 1.27 1.27)) (justify left))\n"
        f'\t\t(uuid "{_uuid_seeded(salt, project_stem)}")\n'
        "\t)"
    )


def _emit_no_connect(nc: NoConnect, salt: str, project_stem: str) -> str:
    return (
        "\t(no_connect\n"
        f"\t\t(at {_fmt(nc.x_mm)} {_fmt(nc.y_mm)})\n"
        f'\t\t(uuid "{_uuid_seeded(salt, project_stem)}")\n'
        "\t)"
    )


def _emit_power_symbol(
    ps: PowerSymbol,
    leaf_uuid: str,
    project_stem: str,
    salt: str,
) -> str:
    pwr_ref = _power_ref_for_salt(salt)
    value = ps.lib_id.split(":", 1)[1] if ":" in ps.lib_id else ps.lib_id
    return (
        "\t(symbol\n"
        f'\t\t(lib_id "{ps.lib_id}")\n'
        f"\t\t(at {_fmt(ps.x_mm)} {_fmt(ps.y_mm)} {ps.angle_deg})\n"
        "\t\t(unit 1)\n"
        "\t\t(exclude_from_sim no)\n"
        "\t\t(in_bom no)\n"
        "\t\t(on_board no)\n"
        "\t\t(dnp no)\n"
        f'\t\t(uuid "{_uuid_seeded(salt, project_stem)}")\n'
        f'\t\t(property "Reference" "{pwr_ref}"\n'
        f"\t\t\t(at {_fmt(ps.x_mm)} {_fmt(ps.y_mm - 3)} 0)\n"
        "\t\t\t(effects (font (size 1.27 1.27)) (hide yes))\n"
        "\t\t)\n"
        f'\t\t(property "Value" "{value}"\n'
        f"\t\t\t(at {_fmt(ps.x_mm)} {_fmt(ps.y_mm + 3)} 0)\n"
        "\t\t\t(effects (font (size 1.27 1.27)))\n"
        "\t\t)\n"
        "\t\t(instances\n"
        f'\t\t\t(project "{project_stem}"\n'
        f'\t\t\t\t(path "/{leaf_uuid}"\n'
        f'\t\t\t\t\t(reference "{pwr_ref}")\n'
        "\t\t\t\t\t(unit 1)\n"
        "\t\t\t\t)\n"
        "\t\t\t)\n"
        "\t\t)\n"
        "\t)"
    )


def _emit_leaf(
    project_dir: Path,
    project_stem: str,
    sheet_inst: _SheetInstance,
    architecture: Architecture | None = None,
    bom: BOM | None = None,
    flag_nets: frozenset[str] = frozenset(),
) -> Path:
    # Stage B runs place+route only when the wiring stage has populated
    # bom.connections. Otherwise fall back to the Stage A grid layout
    # (which preserves backwards compatibility for all existing tests
    # and for projects whose wiring stage hasn't run yet).
    stage_b = (
        bom is not None
        and architecture is not None
        and bool(bom.connections)
    )

    sheet_stem = sheet_inst.sheet.stem

    if stage_b:
        placed = place_sheet(sheet_inst.sheet, sheet_inst.parts, bom)
        routed = route_sheet(
            sheet_stem,
            sheet_inst.sheet.name,
            placed,
            bom,
            architecture,
            flag_nets,
        )
    else:
        placed = None
        routed = RoutedSheet()

    # Build lib_symbols: component symbols + every power:* symbol used.
    pairs = [_split_lib_id(p.symbol) for p in sheet_inst.parts]
    power_lib_ids = sorted({p.lib_id for p in routed.power_symbols})
    power_pairs = [_split_lib_id(s) for s in power_lib_ids]
    lib_block = build_lib_symbols_block(pairs + power_pairs)

    # Component symbol instances.
    symbol_blocks: list[str] = []
    for i, part in enumerate(sheet_inst.parts):
        if placed is not None:
            pp = placed[i]
            x = pp.x_mm
            y = pp.y_mm
            rot = pp.rotation_deg
        else:
            # Stage A grid fallback.
            cols = 5
            col_w = 25.0
            row_h = 30.0
            start_x, start_y = 60.0, 60.0
            x = start_x + (i % cols) * col_w
            y = start_y + (i // cols) * row_h
            rot = 0
        symbol_blocks.append(
            _emit_symbol_instance(
                part, x, y, sheet_inst.leaf_uuid, project_stem,
                rotation_deg=rot,
                salt=(f"{sheet_stem}/symbol/{part.ref}" if stage_b else None),
            )
        )

    # Stage B emit blocks.
    wire_blocks = [
        _emit_wire(w, f"{sheet_stem}/wire/{i}", project_stem)
        for i, w in enumerate(routed.wires)
    ]
    junction_blocks = [
        _emit_junction(j, f"{sheet_stem}/junction/{i}", project_stem)
        for i, j in enumerate(routed.junctions)
    ]
    label_blocks = [
        _emit_net_label(lab, f"{sheet_stem}/label/{i}", project_stem)
        for i, lab in enumerate(routed.labels)
    ]
    power_blocks = [
        _emit_power_symbol(
            ps, sheet_inst.leaf_uuid, project_stem,
            f"{sheet_stem}/power/{i}",
        )
        for i, ps in enumerate(routed.power_symbols)
    ]
    noconn_blocks = [
        _emit_no_connect(nc, f"{sheet_stem}/noconnect/{i}", project_stem)
        for i, nc in enumerate(routed.no_connects)
    ]

    # Hierarchical labels: router-computed positions in Stage B, fallback
    # to the prior left-edge column in Stage A.
    hier_label_blocks: list[str] = []
    if stage_b:
        # Stage B emits only the router-placed hier labels (signal
        # inter-sheet nets). Power/global inter-sheet nets connect via
        # power symbols, so routed.hier_labels intentionally omits them —
        # do NOT fall back to the Stage-A label set below, which would emit
        # dangling power-net labels on power-only sheets.
        for hl in routed.hier_labels:
            hier_label_blocks.append(
                _emit_hierarchical_label(
                    hl.name, hl.direction, hl.x_mm, hl.y_mm,
                    salt=f"{sheet_stem}/hier/{hl.name}",
                    project_stem=project_stem,
                )
            )
    else:
        for i, (net, ep) in enumerate(sheet_inst.inter_sheet_endpoints):
            hier_label_blocks.append(
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
    body_parts = (
        symbol_blocks
        + power_blocks
        + wire_blocks
        + junction_blocks
        + label_blocks
        + noconn_blocks
        + hier_label_blocks
    )
    body = "\n".join(body_parts)
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
    *,
    skip_leaf_sheets: set[str] | None = None,
    sheet_instances: list[_SheetInstance] | None = None,
) -> tuple[Path, list[Path]]:
    """Emit root + leaf .kicad_sch files into project_dir.

    Args:
        skip_leaf_sheets: sheet names to skip during leaf emission. Used
            by the synthesis stage to delegate library-backed sheets to
            the leaf-library installer (which writes the leaf
            .kicad_sch with renumbered refs).
        sheet_instances: optional pre-built _SheetInstance list. If
            supplied, the emitter uses these (with their pre-allocated
            UUIDs) instead of building fresh ones. The installer needs
            the same UUIDs to derive the matching leaf_key.

    Returns (root_path, [leaf_paths]).
    """
    project_dir.mkdir(parents=True, exist_ok=True)
    sheet_insts = sheet_instances if sheet_instances is not None else _build_sheet_instances(architecture, bom)
    root = _emit_root(
        project_stem,
        project_dir,
        sheet_insts,
        architecture,
        project_title=title or project_stem,
    )
    skip = skip_leaf_sheets or set()
    # Assign each power net exactly one PWR_FLAG, on the first (non-skipped)
    # sheet that connects it, so ERC sees the global power net as driven.
    flag_by_sheet: dict[str, set[str]] = {}
    seen_power: set[str] = set()
    for si in sheet_insts:
        if si.sheet.name in skip:
            continue
        for c in bom.connections:
            if (
                c.sheet == si.sheet.name
                and is_power_or_ground_name(c.net_name)
                and power_symbol_for(c.net_name) is not None
                and c.net_name not in seen_power
            ):
                flag_by_sheet.setdefault(si.sheet.name, set()).add(c.net_name)
                seen_power.add(c.net_name)
    leaves = [
        _emit_leaf(
            project_dir, project_stem, si,
            architecture=architecture, bom=bom,
            flag_nets=frozenset(flag_by_sheet.get(si.sheet.name, ())),
        )
        for si in sheet_insts
        if si.sheet.name not in skip
    ]
    return root, leaves


def build_sheet_instances(
    architecture: Architecture, bom: BOM
) -> list[_SheetInstance]:
    """Public re-export of the internal _build_sheet_instances helper.

    The synthesis stage uses this to pre-allocate UUIDs so the
    leaf-library installer derives the matching leaf_key.
    """
    return _build_sheet_instances(architecture, bom)


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
