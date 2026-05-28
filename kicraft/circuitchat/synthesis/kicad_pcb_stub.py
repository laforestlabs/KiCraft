"""Emit `<PROJECT>.kicad_pcb` populated with footprints + nets from the BOM.

When ``bom.connections`` is populated by the wiring stage, every part's
footprint is loaded from the KiCad 9 stock footprint libraries, every
``NetConnection`` is registered as a ``pcbnew.NETINFO_ITEM``, and each
pad is connected to its net via ``pad.SetNetCode(...)``. The PCB's
ratsnest is then non-empty, which is what ``solve-subcircuits`` and
``autoexperiment`` actually consume.

If ``bom.connections`` is empty (Stage A pre-wiring callers, or tests),
the function falls back to writing an empty board (the prior behavior).
This keeps existing callers working unchanged.

Uses ``pcbnew.NewBoard()`` + ``pcbnew.FootprintLoad()`` — both stable
across the KiCad 6 → 9 transitions in scope for this project.
"""
from __future__ import annotations

from pathlib import Path

from ..models import BOM
from .parts_lookup import (
    DEFAULT_KICAD_FOOTPRINT_DIR,
    LibraryNotFoundError,
    resolve_footprint_library_path,
)


class FootprintNotFoundError(LookupError):
    """Raised when a footprint cannot be loaded from the resolver chain."""


def _split_footprint_id(fid: str) -> tuple[str, str]:
    library, _, name = fid.partition(":")
    if not library or not name:
        raise ValueError(f"bad footprint id {fid!r} (expected 'Library:Name')")
    return library, name


def _load_footprint(
    pcbnew_mod,
    library: str,
    name: str,
    project_root: Path | None = None,
    stock_dir: Path = DEFAULT_KICAD_FOOTPRINT_DIR,
):
    try:
        lib_dir = resolve_footprint_library_path(
            library, project_root=project_root, stock_dir=stock_dir
        )
    except LibraryNotFoundError as exc:
        raise FootprintNotFoundError(str(exc)) from exc
    try:
        fp = pcbnew_mod.FootprintLoad(str(lib_dir), name)
    except Exception as exc:  # noqa: BLE001
        raise FootprintNotFoundError(
            f"could not load {library}:{name} from {lib_dir}: {exc}"
        ) from exc
    if fp is None:
        raise FootprintNotFoundError(
            f"footprint {library}:{name} not found in {lib_dir}"
        )
    return fp


def write_empty_pcb(
    project_dir: Path,
    project_stem: str,
    bom: BOM | None = None,
    *,
    project_root: Path | None = None,
    stock_dir: Path = DEFAULT_KICAD_FOOTPRINT_DIR,
) -> Path:
    """Create `<project_stem>.kicad_pcb`.

    Empty board when ``bom`` is None or ``bom.connections`` is empty
    (Stage A backwards compatibility). Populated board with footprints
    and nets otherwise.
    """
    import pcbnew  # noqa: WPS433 — local import keeps non-pcbnew callers (tests) clean

    out = project_dir / f"{project_stem}.kicad_pcb"

    if bom is None or not bom.connections:
        pcbnew.NewBoard(str(out))
        return out

    board = pcbnew.NewBoard(str(out))

    # Scatter footprints on a 200×150 mm grid; the autoplacer.json
    # carries the real placement plan, this just gets parts on the board.
    fps_by_ref: dict[str, object] = {}
    cols = 10
    pitch_mm = 20.0
    origin_x_mm = 20.0
    origin_y_mm = 20.0

    for idx, part in enumerate(bom.parts):
        lib, name = _split_footprint_id(part.footprint)
        fp = _load_footprint(
            pcbnew, lib, name, project_root=project_root, stock_dir=stock_dir
        )
        fp.SetReference(part.ref)
        fp.SetValue(part.value)
        col = idx % cols
        row = idx // cols
        x_mm = origin_x_mm + col * pitch_mm
        y_mm = origin_y_mm + row * pitch_mm
        fp.SetPosition(
            pcbnew.VECTOR2I(pcbnew.FromMM(x_mm), pcbnew.FromMM(y_mm))
        )
        board.Add(fp)
        fps_by_ref[part.ref] = fp

    for conn in bom.connections:
        net = pcbnew.NETINFO_ITEM(board, conn.net_name)
        board.Add(net)
        net_code = net.GetNetCode()
        for ep in conn.endpoints:
            fp = fps_by_ref.get(ep.ref)
            if fp is None:
                continue
            # KiCad treats every pad sharing a number as electrically one node
            # (split thermal pads, dual-terminal tactile switches, etc.);
            # FindPadByNumber returns only the first match and leaves the
            # remaining instances on no net, which trips DRC against any
            # copper that legitimately occupies the shared area.
            for pad in fp.Pads():
                if pad.GetNumber() == ep.pin:
                    pad.SetNetCode(net_code)

    _draw_board_outline(pcbnew, board, fps_by_ref.values())

    board.Save(str(out))
    return out


def _draw_board_outline(pcbnew_mod, board, footprints, *, margin_mm: float = 5.0) -> None:
    """Draw an Edge.Cuts rectangle enclosing every placed footprint + margin.

    Downstream tools (``compose-subcircuits``, FreeRouting) require a non-zero
    board outline: without one the seed board's edge bbox is ``0×0`` and the
    parent composer under-sizes the board, leaving footprints outside the
    edges so the router produces no SES. This is a *seed* outline — compose
    re-sizes the parent properly; the stub only needs a valid enclosure so the
    geometry is well-formed at every stage.
    """
    bbox = None
    for fp in footprints:
        try:
            fbb = fp.GetBoundingBox()
        except TypeError:
            # Older pcbnew signatures require explicit include-text flags.
            fbb = fp.GetBoundingBox(True, False)
        if bbox is None:
            bbox = fbb
        else:
            bbox.Merge(fbb)

    if bbox is None:
        return

    margin = pcbnew_mod.FromMM(margin_mm)
    left = bbox.GetLeft() - margin
    top = bbox.GetTop() - margin
    right = bbox.GetRight() + margin
    bottom = bbox.GetBottom() + margin

    rect = pcbnew_mod.PCB_SHAPE(board)
    rect.SetShape(pcbnew_mod.SHAPE_T_RECT)
    rect.SetStart(pcbnew_mod.VECTOR2I(left, top))
    rect.SetEnd(pcbnew_mod.VECTOR2I(right, bottom))
    rect.SetLayer(pcbnew_mod.Edge_Cuts)
    rect.SetWidth(pcbnew_mod.FromMM(0.1))
    board.Add(rect)
