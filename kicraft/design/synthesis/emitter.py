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
from dataclasses import dataclass, replace
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
from .sch_geometry import pin_abs_position
from .router import (
    GlobalLabel,
    Junction,
    NetLabel,
    NoConnect,
    PowerSymbol,
    RoutedSheet,
    WireSegment,
    _label_rect,
    _rects_overlap,
    route_sheet,
)
from .symbol_library import build_lib_symbols_block
from .symbol_pinout import SymbolNotFoundError, lookup_pins


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


def _q(s: str | None) -> str:
    """Escape a string for embedding in a quoted s-expression token.

    Part descriptions routinely contain inch marks (`0.96" OLED`); unescaped,
    the quote terminates the token early and eeschema rejects the whole file —
    silently, when the file is a hierarchy child (the sheet loads as empty).
    Every model- or part-derived string must pass through here.
    """
    return (s or "").replace("\\", "\\\\").replace('"', '\\"')


def assert_schematic_parses(text: str, path: Path) -> None:
    """Refuse to write a .kicad_sch that eeschema cannot parse.

    Checks balanced parens with escape-aware string scanning — the exact
    failure mode of an unescaped quote. A loud error at write time beats
    KiCad loading the child sheet as empty and failing ERC three stages later
    with an unrelated-looking hier_label_mismatch.
    """
    depth = 0
    in_str = False
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                break
        i += 1
    if depth != 0 or in_str:
        raise ValueError(
            f"refusing to write unparseable schematic {path.name}: "
            f"{'unterminated string' if in_str else 'unbalanced parens'} "
            "(an embedded quote in a part field is the usual cause)"
        )


def _split_lib_id(value: str) -> tuple[str, str]:
    library, _, name = value.partition(":")
    if not library or not name:
        raise ValueError(f"bad lib_id {value!r} (expected 'Library:Name')")
    return library, name


def ensure_leaf_stems_distinct(project_stem: str, sheets: list[Sheet]) -> None:
    """Rename any leaf sheet whose stem collides with the project (root) stem.

    The root schematic is emitted to ``<project_stem>.kicad_sch`` and each leaf
    to ``<sheet.stem>.kicad_sch``. A leaf whose stem equals the project stem
    therefore writes to the SAME path as the root, so one write clobbers the
    other and the user is left with a single non-readable block-diagram root
    (the cyan blob on kicraft.io) instead of a component schematic. Architecture
    validation keeps leaf stems unique among themselves but never compares them
    to the project stem, so guard the collision here.

    Mutates the colliding sheet's ``stem`` in place to a unique, shape-valid
    variant (``<STEM>_SHEET``, then ``_SHEET2`` ...). Idempotent: a second call
    finds no collision. Must run BEFORE sheet instances are built and before any
    leaf file is written (incl. the leaf-library installer), so the installer,
    the root sheet pins, and the leaf file all agree on the new stem.
    """
    root = (project_stem or "").upper()
    taken = {root} | {s.stem.upper() for s in sheets}
    for s in sheets:
        if s.stem.upper() != root:
            continue
        i = 1
        while True:
            cand = f"{s.stem}_SHEET" if i == 1 else f"{s.stem}_SHEET{i}"
            if cand.upper() not in taken:
                break
            i += 1
        taken.add(cand.upper())
        s.stem = cand


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
            f'\t\t(pin "{_q(net.name)}" {ep.direction}\n'
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
        f'\t\t(property "Sheetname" "{_q(sheet.name)}"\n'
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


def _fit_title(title: str, max_chars: int = 60) -> str:
    """Trim a long title so it stays inside the title block instead of
    running off the page edge (titles come from the user's brief verbatim)."""
    title = title.strip()
    if len(title) <= max_chars:
        return title
    cut = title[:max_chars].rsplit(" ", 1)[0]
    return f"{cut}…"


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
    # Lay sheets out horizontally, centered on the smallest page that fits:
    # a lone sheet symbol parked in the corner of an A3 reads as a draft,
    # not a finished drawing. The pin stubs + labels extend half a gap past
    # the last sheet block, so include that in the fit check.
    n_sheets = max(1, len(sheet_insts))
    content_w = n_sheets * sheet_width + (n_sheets - 1) * sheet_gap
    content_w += sheet_gap * 0.5 + 12.7  # right-side pin stubs + labels
    content_h = sheet_height
    if content_w <= 297.0 - 2 * 25.4 and content_h <= 210.0 - 2 * 25.4:
        paper, page_w, page_h = "A4", 297.0, 210.0
    else:
        paper, page_w, page_h = "A3", 420.0, 297.0
    grid = 1.27
    start_x = round(((page_w - content_w) / 2) / grid) * grid
    start_y = round(((page_h - content_h) / 2) / grid) * grid
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
        f'\t(paper "{paper}")\n'
        "\n"
        "\t(title_block\n"
        f'\t\t(title "{_q(_fit_title(project_title))}")\n'
        '\t\t(rev "1.0")\n'
        f'\t\t(comment 1 "Generated by KiCraft pipeline")\n'
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
    assert_schematic_parses(header + body + tail, out)
    out.write_text(header + body + tail)
    return out


# ---------- leaf schematics ----------


def _field_angle_justify(rotation_deg: int, justify: str) -> tuple[int, str]:
    """Property angle + justification that render screen-horizontal text.

    KiCad composes a symbol's rotation into its property text: a field
    written at angle 0 on a 90°-rotated symbol draws VERTICAL (and two
    stacked fields collapse into an unreadable jumble). Verified against
    kicad-cli 9.0.9: counter-rotating the field (90°→270°, 270°→90°)
    renders horizontal with justification intact; at 180° the field must
    stay at angle 0 but its justification anchor mirrors (left↔right).
    """
    rot = rotation_deg % 360
    if rot == 90:
        return 270, justify
    if rot == 270:
        return 90, justify
    if rot == 180:
        flipped = {"left": "right", "right": "left"}.get(justify, justify)
        return 0, flipped
    return 0, justify


def _field_rect(
    text: str, x: float, y: float, justify: str
) -> tuple[float, float, float, float]:
    """Estimated bbox of one field's rendered text (1.27 mm font)."""
    length = len(text) * 1.1 + 0.5
    if justify == "left":
        return (x, y - 0.95, x + length, y + 0.95)
    if justify == "right":
        return (x - length, y - 0.95, x, y + 0.95)
    return (x - length / 2, y - 0.95, x + length / 2, y + 0.95)


def _text_anchors(
    part: BomPart,
    x: float,
    y: float,
    rotation_deg: int,
    obstacles: list[tuple[float, float, float, float]] | None = None,
) -> tuple[tuple[float, float, str], tuple[float, float, str]]:
    """Where to put the Reference and Value text so they don't collide with the
    symbol, its wires, or the power/ground symbols hanging off its pins.

    Returns ``((ref_x, ref_y, ref_justify), (val_x, val_y, val_justify))``.
    A 2-pin passive gets both fields stacked beside its body (clear of the
    rail/ground symbols that sit directly above/below it); a multi-pin part
    gets them stacked above the body (clear of any bottom pin like GND).
    When ``obstacles`` is given (estimated rects of net labels, power
    symbols and already-placed fields), candidate spots are tried in
    preference order and the first collision-free one wins.
    """
    try:
        pins = lookup_pins(part.symbol)["pins"]
    except (SymbolNotFoundError, ValueError, KeyError):
        pins = []
    if not pins:
        return ((x, y - 5.0, ""), (x, y + 5.0, ""))
    xs, ys = [], []
    for p in pins:
        ax, ay = pin_abs_position(x, y, rotation_deg, p)
        xs.append(ax)
        ys.append(ay)
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    if len(pins) == 2:
        candidates = [
            ((right + 1.778, y - 1.27, "left"), (right + 1.778, y + 1.27, "left")),
            ((left - 1.778, y - 1.27, "right"), (left - 1.778, y + 1.27, "right")),
            ((right + 4.318, y - 1.27, "left"), (right + 4.318, y + 1.27, "left")),
            ((left - 4.318, y - 1.27, "right"), (left - 4.318, y + 1.27, "right")),
        ]
    else:
        candidates = [
            ((x, top - 2.54, ""), (x, top - 5.08, "")),
            ((x, bottom + 5.08, ""), (x, bottom + 2.54, "")),
            ((right + 1.778, y - 1.27, "left"), (right + 1.778, y + 1.27, "left")),
            ((left - 1.778, y - 1.27, "right"), (left - 1.778, y + 1.27, "right")),
            ((x, top - 5.08, ""), (x, top - 7.62, "")),
        ]
    if not obstacles:
        return candidates[0]

    def _overlap_area(
        a: tuple[float, float, float, float], b: tuple[float, float, float, float]
    ) -> float:
        w = min(a[2], b[2]) - max(a[0], b[0])
        h = min(a[3], b[3]) - max(a[1], b[1])
        return w * h if (w > 0 and h > 0) else 0.0

    best = candidates[0]
    best_overlap = float("inf")
    for ref_spot, val_spot in candidates:
        ref_rect = _field_rect(part.ref, ref_spot[0], ref_spot[1], ref_spot[2])
        val_rect = _field_rect(part.value, val_spot[0], val_spot[1], val_spot[2])
        overlap = sum(
            _overlap_area(rect, ob)
            for rect in (ref_rect, val_rect)
            for ob in obstacles
        )
        if overlap == 0.0:
            return ref_spot, val_spot
        if overlap < best_overlap:
            best_overlap = overlap
            best = (ref_spot, val_spot)
    # Nothing fully clear: take the least-overlapping spot rather than
    # piling onto the (often worst) first candidate.
    return best


def _visible_effects(justify: str) -> str:
    j = f" (justify {justify})" if justify else ""
    return f"(effects (font (size 1.27 1.27)){j})"


def _emit_symbol_instance(
    part: BomPart,
    x: float,
    y: float,
    leaf_uuid: str,
    project_stem: str,
    *,
    rotation_deg: int = 0,
    salt: str | None = None,
    field_anchors: tuple[
        tuple[float, float, str], tuple[float, float, str]
    ] | None = None,
) -> str:
    """Emit one component `(symbol ...)` instance inside a leaf."""
    uuid_str = _uuid_seeded(salt, project_stem) if salt else _uuid()
    if field_anchors is None:
        field_anchors = _text_anchors(part, x, y, rotation_deg)
    (rx, ry, rj), (vx, vy, vj) = field_anchors
    ref_angle, rj = _field_angle_justify(rotation_deg, rj)
    val_angle, vj = _field_angle_justify(rotation_deg, vj)
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
        f'\t\t(property "Reference" "{_q(part.ref)}"\n'
        f"\t\t\t(at {_fmt(rx)} {_fmt(ry)} {ref_angle})\n"
        f"\t\t\t{_visible_effects(rj)}\n"
        "\t\t)\n"
        f'\t\t(property "Value" "{_q(part.value)}"\n'
        f"\t\t\t(at {_fmt(vx)} {_fmt(vy)} {val_angle})\n"
        f"\t\t\t{_visible_effects(vj)}\n"
        "\t\t)\n"
        f'\t\t(property "Footprint" "{_q(part.footprint)}"\n'
        f"\t\t\t(at {_fmt(x)} {_fmt(y + 7)} 0)\n"
        "\t\t\t(effects (font (size 1.27 1.27)) (hide yes))\n"
        "\t\t)\n"
        f'\t\t(property "Datasheet" "{_q(part.datasheet)}"\n'
        f"\t\t\t(at {_fmt(x)} {_fmt(y)} 0)\n"
        "\t\t\t(effects (font (size 1.27 1.27)) (hide yes))\n"
        "\t\t)\n"
        f'\t\t(property "Description" "{_q(part.sourcing_note)}"\n'
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
    name: str, direction: str, x: float, y: float, angle: int = 0,
    *, salt: str | None = None, project_stem: str | None = None,
) -> str:
    shape = _DIRECTION_TO_SHAPE.get(direction, "passive")
    if salt and project_stem:
        uuid_str = _uuid_seeded(salt, project_stem)
    else:
        uuid_str = _uuid()
    return (
        f'\t(hierarchical_label "{_q(name)}" (shape {shape})\n'
        f"\t\t(at {_fmt(x)} {_fmt(y)} {angle})\n"
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


def _label_justify(angle_deg: int) -> str:
    """eeschema's spin-style convention: a label reading leftward/downward
    is stored as angle 180/270 with justify RIGHT; writing justify left
    there makes KiCad render the text mirrored across the anchor (over
    whatever the label was trying to avoid)."""
    return "right" if angle_deg % 360 in (180, 270) else "left"


def _emit_net_label(lab: NetLabel, salt: str, project_stem: str) -> str:
    return (
        f'\t(label "{_q(lab.text)}"\n'
        f"\t\t(at {_fmt(lab.x_mm)} {_fmt(lab.y_mm)} {lab.angle_deg})\n"
        f"\t\t(effects (font (size 1.27 1.27)) (justify {_label_justify(lab.angle_deg)}))\n"
        f'\t\t(uuid "{_uuid_seeded(salt, project_stem)}")\n'
        "\t)"
    )


def _emit_global_label(lab: GlobalLabel, salt: str, project_stem: str) -> str:
    """A global label ties same-named nets together across the whole
    hierarchy (no sheet pins needed). Used for power/ground nets that lack a
    stock KiCad power symbol, so the net keeps its exact name and connects
    across sheets without referencing a symbol that may not exist."""
    return (
        f'\t(global_label "{_q(lab.text)}"\n'
        "\t\t(shape bidirectional)\n"
        f"\t\t(at {_fmt(lab.x_mm)} {_fmt(lab.y_mm)} {lab.angle_deg})\n"
        "\t\t(fields_autoplaced yes)\n"
        f"\t\t(effects (font (size 1.27 1.27)) (justify {_label_justify(lab.angle_deg)}))\n"
        f'\t\t(uuid "{_uuid_seeded(salt, project_stem)}")\n'
        '\t\t(property "Intersheetrefs" "${INTERSHEET_REFS}"\n'
        f"\t\t\t(at {_fmt(lab.x_mm)} {_fmt(lab.y_mm)} 0)\n"
        "\t\t\t(effects (font (size 1.27 1.27)) (hide yes))\n"
        "\t\t)\n"
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
    # PWR_FLAG carries no rail name worth showing — its label just collides with
    # the rail/ground symbol it sits on. Keep the flag graphic, hide the word.
    value_hide = " (hide yes)" if value == "PWR_FLAG" else ""
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
        f"\t\t\t(effects (font (size 1.27 1.27)){value_hide})\n"
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


_LEAF_PAGE_SIZES: tuple[tuple[str, float, float], ...] = (
    ("A4", 297.0, 210.0),
    ("A3", 420.0, 297.0),
)


def _leaf_content_bbox(
    parts: list[BomPart], placed: list, routed: RoutedSheet
) -> tuple[float, float, float, float] | None:
    """Extent of everything drawn on a leaf: symbol pins, wires, labels,
    power symbols. Used to pick the page size and center the drawing."""
    xs: list[float] = []
    ys: list[float] = []
    for part, pp in zip(parts, placed):
        try:
            pins = lookup_pins(part.symbol)["pins"]
        except (SymbolNotFoundError, ValueError, KeyError):
            pins = []
        if pins:
            for p in pins:
                ax, ay = pin_abs_position(pp.x_mm, pp.y_mm, pp.rotation_deg, p)
                xs.append(ax)
                ys.append(ay)
        else:
            xs.append(pp.x_mm)
            ys.append(pp.y_mm)
    for w in routed.wires:
        xs.extend((w.x1_mm, w.x2_mm))
        ys.extend((w.y1_mm, w.y2_mm))
    for items in (
        routed.junctions,
        routed.labels,
        routed.global_labels,
        routed.power_symbols,
        routed.no_connects,
        routed.hier_labels,
    ):
        for item in items:
            xs.append(item.x_mm)
            ys.append(item.y_mm)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _fit_leaf_page(
    parts: list[BomPart], placed: list, routed: RoutedSheet
) -> tuple[list, RoutedSheet, str]:
    """Pick the smallest page that fits the drawing and center it there.

    The placer lays content out from a fixed top-left origin, which leaves
    a small sheet huddled in the corner of its page -- a draft, not a
    finished drawing. The translation is grid-snapped so every pin and
    wire endpoint stays on the 1.27 mm connection grid.
    """
    bbox = _leaf_content_bbox(parts, placed, routed)
    if bbox is None:
        return placed, routed, "A4"
    # Frame + title block + text-overhang allowance around the content.
    pad = 25.4
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    paper, page_w, page_h = _LEAF_PAGE_SIZES[-1]
    for name, pw, ph in _LEAF_PAGE_SIZES:
        if w + 2 * pad <= pw and h + 2 * pad <= ph:
            paper, page_w, page_h = name, pw, ph
            break
    grid = 1.27
    dx = round(((page_w - w) / 2 - bbox[0]) / grid) * grid
    dy = round(((page_h - h) / 2 - bbox[1]) / grid) * grid
    if abs(dx) < grid and abs(dy) < grid:
        return placed, routed, paper
    placed_shifted = [
        replace(pp, x_mm=pp.x_mm + dx, y_mm=pp.y_mm + dy) for pp in placed
    ]
    routed_shifted = RoutedSheet(
        wires=[
            WireSegment(w2.x1_mm + dx, w2.y1_mm + dy, w2.x2_mm + dx, w2.y2_mm + dy)
            for w2 in routed.wires
        ],
        junctions=[
            replace(j, x_mm=j.x_mm + dx, y_mm=j.y_mm + dy) for j in routed.junctions
        ],
        labels=[
            replace(lab, x_mm=lab.x_mm + dx, y_mm=lab.y_mm + dy)
            for lab in routed.labels
        ],
        power_symbols=[
            replace(ps, x_mm=ps.x_mm + dx, y_mm=ps.y_mm + dy)
            for ps in routed.power_symbols
        ],
        no_connects=[
            replace(nc, x_mm=nc.x_mm + dx, y_mm=nc.y_mm + dy)
            for nc in routed.no_connects
        ],
        hier_labels=[
            replace(hl, x_mm=hl.x_mm + dx, y_mm=hl.y_mm + dy)
            for hl in routed.hier_labels
        ],
        global_labels=[
            replace(gl, x_mm=gl.x_mm + dx, y_mm=gl.y_mm + dy)
            for gl in routed.global_labels
        ],
    )
    return placed_shifted, routed_shifted, paper


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
        placed, routed, paper = _fit_leaf_page(sheet_inst.parts, placed, routed)
    else:
        placed = None
        routed = RoutedSheet()
        paper = "A4"

    # Build lib_symbols: component symbols + every power:* symbol used.
    # Map each symbol to its assigned refdes so pin-type normalization can
    # classify the device from KiCraft's own designator (e.g. J2 -> connector),
    # not easyeda's arbitrary intrinsic Reference (a microSD socket is "Card").
    pairs = [_split_lib_id(p.symbol) for p in sheet_inst.parts]
    ref_prefixes = {
        _split_lib_id(p.symbol): p.ref for p in sheet_inst.parts if p.ref
    }
    power_lib_ids = sorted({p.lib_id for p in routed.power_symbols})
    power_pairs = [_split_lib_id(s) for s in power_lib_ids]
    lib_block = build_lib_symbols_block(
        pairs + power_pairs, ref_prefixes=ref_prefixes
    )

    # Obstacles for field placement: estimated rects of net labels and
    # power-symbol graphics+text, grown with each part's chosen fields so
    # later parts dodge earlier ones.
    obstacles: list[tuple[float, float, float, float]] = []
    for lab_list in (routed.labels, routed.global_labels):
        for lab in lab_list:
            obstacles.append(_label_rect(lab.text, lab.x_mm, lab.y_mm, lab.angle_deg))
    for ps in routed.power_symbols:
        obstacles.append((ps.x_mm - 4.0, ps.y_mm - 5.5, ps.x_mm + 4.0, ps.y_mm + 5.5))

    # Component symbol instances.
    symbol_blocks: list[str] = []
    for i, part in enumerate(sheet_inst.parts):
        anchors = None
        if placed is not None:
            pp = placed[i]
            x = pp.x_mm
            y = pp.y_mm
            rot = pp.rotation_deg
            anchors = _text_anchors(part, x, y, rot, obstacles)
            ref_spot, val_spot = anchors
            obstacles.append(_field_rect(part.ref, *ref_spot))
            obstacles.append(_field_rect(part.value, *val_spot))
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
                field_anchors=anchors,
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
    global_label_blocks = [
        _emit_global_label(gl, f"{sheet_stem}/glabel/{i}", project_stem)
        for i, gl in enumerate(routed.global_labels)
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
                    hl.name, hl.direction, hl.x_mm, hl.y_mm, hl.angle_deg,
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
        f'\t(paper "{paper}")\n'
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
        + global_label_blocks
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
    assert_schematic_parses(header + body + tail, out)
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


def _power_nets_with_driver(bom: BOM) -> set[str]:
    """Names of power/ground nets already driven by a real power-output pin.

    A ``PWR_FLAG`` carries its own ``power_out`` pin, so adding one to a net a
    component already drives (a charger's V_BAT output, an LDO/regulator VOUT,
    a boost converter's output rail, …) trips KiCad ERC's *"Pins of type Power
    output and Power output are connected"*. Those nets must therefore be
    EXCLUDED from PWR_FLAG assignment — the flag exists only to mark a power net
    that is real-but-undriven (fed from a connector / battery / passive pin) so
    ERC stops reporting its power-input pins as undriven.

    Power nets are global by name across the whole hierarchy, so a driver on
    ANY sheet protects the entire net: we scan every connection, not just one
    sheet's. Symbols whose pins can't be resolved are treated as non-drivers
    (the conservative choice: we keep the flag rather than risk an undriven net).
    """
    symbol_by_ref = {p.ref: p.symbol for p in bom.parts}
    driven: set[str] = set()
    for c in bom.connections:
        if not is_power_or_ground_name(c.net_name) or c.net_name in driven:
            continue
        for ep in c.endpoints:
            symbol = symbol_by_ref.get(ep.ref)
            if symbol is None:
                continue
            try:
                pins = lookup_pins(symbol)["pins"]
            except (SymbolNotFoundError, ValueError, KeyError):
                continue
            if any(p["number"] == ep.pin and p["electrical_type"] == "power_out"
                   for p in pins):
                driven.add(c.net_name)
                break
    return driven


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
    # A leaf stem equal to the project stem would write to the root's own file;
    # guard it before any path is derived (no-op if run() already deduped).
    ensure_leaf_stems_distinct(project_stem, architecture.sheets)
    sheet_insts = sheet_instances if sheet_instances is not None else _build_sheet_instances(architecture, bom)
    root = _emit_root(
        project_stem,
        project_dir,
        sheet_insts,
        architecture,
        project_title=title or project_stem,
    )
    skip = skip_leaf_sheets or set()
    # Assign each UNDRIVEN power net exactly one PWR_FLAG, on the first
    # (non-skipped) sheet that connects it, so ERC sees the global power net as
    # driven. A net already driven by a real power-output pin must NOT get a
    # flag: PWR_FLAG is itself a power-output, so a second one shorts ERC with
    # "Power output and Power output are connected" (e.g. a charger's V_BAT
    # output rail). See _power_nets_with_driver.
    driven_power_nets = _power_nets_with_driver(bom)
    flag_by_sheet: dict[str, set[str]] = {}
    seen_power: set[str] = set()
    for si in sheet_insts:
        if si.sheet.name in skip:
            continue
        for c in bom.connections:
            # Every undriven power/ground net gets exactly one PWR_FLAG, whether
            # the router renders it as a stock power symbol or as a global label
            # (the no-stock-symbol fallback) -- both can carry the flag.
            if (
                c.sheet == si.sheet.name
                and is_power_or_ground_name(c.net_name)
                and c.net_name not in seen_power
                and c.net_name not in driven_power_nets
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
