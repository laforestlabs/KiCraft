"""KiCad pcbnew adapter — sole interface to .kicad_pcb files.

This is the ONLY module that imports pcbnew. All other modules operate
on pure-Python types from brain.types.
"""

import json
import os
import site
import subprocess
import sys
import tempfile
from pathlib import Path

import pcbnew

from kicraft.autoplacer.brain.types import BoardState, Component, Layer, Net, Pad, Point, TraceSegment, Via
from kicraft.autoplacer.hardware.keepout_extract import extract_keepout_rects

# Generic power net names used as fallback when config doesn't specify power_nets.
# Project-specific power nets should be listed in the project's autoplacer config.
POWER_NETS = {
    "VCC", "VDD", "GND", "VBUS", "5V", "3V3", "3.3V", "+5V", "+3V3", "+3.3V",
}

SIGNAL_WIDTH_MM = 0.127
POWER_WIDTH_MM = 0.127
VIA_DRILL_MM = 0.3
VIA_SIZE_MM = 0.6


def _extract_leaf_outline_polyline_mm(silkscreen) -> list[tuple[float, float]] | None:
    """Pull the leaf-outline polyline (mm) out of a BoardState.silkscreen.

    Returns the F.SilkS poly element's points as a list of (x, y) tuples
    so the same closed contour can be stamped as both the silk outline
    and Edge.Cuts -- single source of truth for the leaf boundary. The
    silk producer (``_build_leaf_silkscreen``) emits exactly one poly
    element per leaf via ``leaf_outline_polyline``, so the first poly
    on F.SilkS is the outline.

    Returns ``None`` for unlabeled leaves (silk producer returned empty,
    parent flow, etc.) so the caller can fall back to a sharp rectangle.
    """
    for elem in silkscreen or []:
        if elem.kind != "poly" or elem.layer != "F.SilkS":
            continue
        pts = list(elem.points or [])
        if len(pts) >= 3:
            return [(p.x, p.y) for p in pts]
    return None


def _is_leaf_outline_silk(dwg) -> bool:
    """Heuristic: a 0.15 mm silk segment qualifies as the leaf outline.

    KiCad silk import doesn't preserve custom tags through save/load,
    so identify the outline lines by their layer + width signature.
    Component silk is typically thicker (default 0.12-0.15 mm but
    drawn as part of footprint silk, not loose graphics) and uses
    different shape kinds; loose 0.15 mm silk segments at exactly
    the Edge.Cuts bbox corners are the ones we want to replace.

    GetDrawings() returns a mixed list -- PCB_SHAPE, PCB_TEXT,
    PCB_DIMENSION, etc. Only PCB_SHAPE has GetShape() / GetWidth();
    calling either on a PCB_TEXT raises AttributeError. Guard with
    hasattr() so the cleanup pass doesn't crash the leaf stamp
    subprocess on boards that have silk text or dimensions.
    """
    try:
        if dwg.GetLayer() != pcbnew.F_SilkS:
            return False
        if not hasattr(dwg, "GetShape") or not hasattr(dwg, "GetWidth"):
            return False
        if dwg.GetShape() != pcbnew.SHAPE_T_SEGMENT:
            return False
        # Width 0.15 mm == 150 nm * 1000. pcbnew.ToMM converts.
        return abs(pcbnew.ToMM(dwg.GetWidth()) - 0.15) < 1e-3
    except Exception:  # noqa: BLE001
        return False


def _atomic_save_board(
    board,
    output_path: str,
    *,
    source_pro_path: str | None = None,
) -> None:
    """Save a pcbnew board to disk durably and sync its sibling .kicad_pro.

    Saves in place, then fsyncs the file and its containing directory so a
    follow-up pcbnew.LoadBoard() in another process cannot observe a partial
    write or unsynced directory entry (which manifests as
    "RuntimeError: Failed to load board: ...").

    We do not write to a temp file and rename: pcbnew.Save() rewrites the
    requested filename based on the .kicad_pcb extension and emits a sidecar
    .kicad_pro keyed off the same basename, which makes a sibling temp +
    os.replace pattern unreliable across KiCad versions.

    The auto-emitted sidecar .kicad_pro carries KiCad's *defaults* (Default
    netclass clearance 0.20 mm, min_clearance 0.0), NOT the project's actual
    netclass values. When ``source_pro_path`` is supplied and exists, we
    overwrite the sibling with that file -- this propagates the project's
    real netclass / rules to anything that DRCs the freshly-saved PCB
    (kicad-cli pcb drc, FreeRouting validators, etc.). Without this sync,
    project edits like ``"clearance": 0.15`` are invisible at validation
    time and cause phantom clearance violations.
    """
    import shutil
    board.Save(output_path)
    if source_pro_path and os.path.exists(source_pro_path):
        sibling_pro = os.path.splitext(output_path)[0] + ".kicad_pro"
        if os.path.abspath(source_pro_path) != os.path.abspath(sibling_pro):
            try:
                shutil.copy2(source_pro_path, sibling_pro)
            except OSError:
                pass
    try:
        with open(output_path, "rb") as tf:
            os.fsync(tf.fileno())
    except OSError:
        pass
    try:
        out_dir = os.path.dirname(output_path) or "."
        dir_fd = os.open(out_dir, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def _pcbnew_subprocess_env() -> dict:
    """Build subprocess env that can import KiCad's pcbnew module.

    In virtualenvs, KiCad's site-packages path may not be visible to child
    Python processes.  This adds common KiCad locations to PYTHONPATH.
    """
    env = os.environ.copy()

    candidates = []
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    candidates.extend(
        [
            f"/usr/lib/python{ver}/site-packages",
            f"/usr/lib64/python{ver}/site-packages",
            "/usr/lib/python3/dist-packages",
            "/usr/lib64/python3/dist-packages",
        ]
    )
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass

    existing = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    merged = list(existing)
    for p in candidates:
        if not p:
            continue
        if (
            os.path.exists(os.path.join(p, "pcbnew.py"))
            or os.path.isdir(os.path.join(p, "pcbnew"))
        ) and p not in merged:
            merged.append(p)

    if merged:
        env["PYTHONPATH"] = os.pathsep.join(merged)

    return env


class StampSubprocessError(RuntimeError):
    """Raised when the leaf-stamp subprocess fails non-recoverably.

    Distinguished from generic routing errors so callers can surface
    "the implementation is broken" loudly instead of letting the round
    quietly degrade to ``routing_exception`` like a recoverable
    FreeRouting timeout. The exception text carries the rc / stderr /
    stdout from ``_run_pcbnew_script_file`` for triage.
    """


def _run_pcbnew_subprocess(script: str) -> str:
    """Run a pcbnew script string in a fresh subprocess.

    Avoids SWIG memory corruption by giving each pcbnew workload its
    own interpreter. Prefer ``_run_pcbnew_script_file`` for any
    nontrivial script -- inline strings sidestep linters / IDEs and
    let pcbnew API misuse hide until runtime.
    """
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=_pcbnew_subprocess_env(),
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pcbnew subprocess failed (rc={result.returncode}):\n{result.stderr}"
        )
    return result.stdout


def _run_pcbnew_script_file(script_path: str, *args: str, timeout: int = 120) -> str:
    """Run a pcbnew script that lives as its own .py file.

    Same isolation as ``_run_pcbnew_subprocess`` but the script is a
    real file -- which means import-time errors fire when the file is
    parsed (catchable by linters and ``python -c "import x"`` smoke
    tests) instead of being concealed inside a runtime string blob.
    """
    cmd = [sys.executable, str(script_path), *map(str, args)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=_pcbnew_subprocess_env(),
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pcbnew script {script_path} failed (rc={result.returncode}):\n"
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )
    return result.stdout


def _classify_component(ref: str, value: str) -> str:
    """Classify component by reference prefix."""
    r = ref.upper()
    if r.startswith("J"):
        return "connector"
    if r.startswith("H"):
        return "mounting_hole"
    if r.startswith("U"):
        return "ic"
    if r.startswith(("R", "C", "L", "D", "F")):
        return "passive"
    if r.startswith("BT"):
        return "battery"
    return "misc"


def _layer_to_enum(kicad_layer: int) -> Layer:
    if kicad_layer == pcbnew.B_Cu:
        return Layer.BACK
    return Layer.FRONT


def _enum_to_layer(layer: Layer) -> int:
    return pcbnew.B_Cu if layer == Layer.BACK else pcbnew.F_Cu


def detect_opening_direction(fp) -> float | None:
    """Detect which direction the connector opening faces, in LOCAL coords.

    Detects the board-space opening direction by comparing pad bbox to body
    bbox (courtyard + fab graphics) — all in board-space coords straight from
    pcbnew, no rotation math.  Then converts to local with one addition:
    local_angle = (board_angle + rotation) % 360.

    Returns 0/90/180/270 in local coords, or None.
    """
    # --- Board-space pad bbox ---
    pad_xs = [pcbnew.ToMM(p.GetPosition().x) for p in fp.Pads()]
    pad_ys = [pcbnew.ToMM(p.GetPosition().y) for p in fp.Pads()]
    if not pad_xs:
        return None

    # --- Board-space body bbox from courtyard + fab ---
    body_xs, body_ys = [], []
    cy_layer = pcbnew.F_CrtYd if fp.GetLayer() == pcbnew.F_Cu else pcbnew.B_CrtYd
    fab_layer = pcbnew.F_Fab if fp.GetLayer() == pcbnew.F_Cu else pcbnew.B_Fab
    for item in fp.GraphicalItems():
        if item.GetLayer() not in (cy_layer, fab_layer):
            continue
        try:
            body_xs.append(pcbnew.ToMM(item.GetStart().x))
            body_xs.append(pcbnew.ToMM(item.GetEnd().x))
            body_ys.append(pcbnew.ToMM(item.GetStart().y))
            body_ys.append(pcbnew.ToMM(item.GetEnd().y))
        except Exception:
            continue
    if not body_xs:
        return None

    pad_cx = (min(pad_xs) + max(pad_xs)) / 2
    pad_cy = (min(pad_ys) + max(pad_ys)) / 2

    opening_board = None

    # (1) Explicit "PCB Edge" / "Board Edge" marker on Dwgs.User -- the
    # footprint author telling us exactly where the board edge belongs.
    # Authoritative when present (most vendor USB footprints carry it).
    for item in fp.GraphicalItems():
        if item.GetLayer() != pcbnew.Dwgs_User:
            continue
        try:
            text = item.GetText()
        except Exception:
            continue
        if not text or "edge" not in text.lower():
            continue
        tp = item.GetPosition()
        off_x = pcbnew.ToMM(tp.x) - pad_cx
        off_y = pcbnew.ToMM(tp.y) - pad_cy
        if abs(off_x) > abs(off_y):
            opening_board = 0 if off_x > 0 else 180
        else:
            opening_board = 90 if off_y > 0 else 270
        break

    # (2) Body (courtyard + fab) extends asymmetrically past the pad cluster:
    # a USB/edge connector's shell overhangs its pins toward the mating mouth.
    if opening_board is None:
        extensions = {
            0: max(body_xs) - max(pad_xs),  # +X (right)
            180: min(pad_xs) - min(body_xs),  # -X (left)
            90: max(body_ys) - max(pad_ys),  # +Y (down)
            270: min(pad_ys) - min(body_ys),  # -Y (up)
        }
        ranked = sorted(extensions.items(), key=lambda kv: kv[1], reverse=True)
        best_dir, best_ext = ranked[0]
        _, second_ext = ranked[1]
        if best_ext >= 1.0 and (best_ext - second_ext) >= 0.5:
            opening_board = best_dir

    # (3) Pad cluster sits off-center inside the body: the pins/tail crowd one
    # end and the mouth is the far side. Catches connectors whose courtyard is
    # near-symmetric but whose pads are clearly biased toward the back.
    if opening_board is None:
        body_cx = (min(body_xs) + max(body_xs)) / 2
        body_cy = (min(body_ys) + max(body_ys)) / 2
        dx = body_cx - pad_cx
        dy = body_cy - pad_cy
        if max(abs(dx), abs(dy)) >= 0.5:
            if abs(dx) > abs(dy):
                opening_board = 0 if dx > 0 else 180
            else:
                opening_board = 90 if dy > 0 else 270

    if opening_board is None:
        return None

    # Convert board-space → local with one addition (no trig)
    rotation = fp.GetOrientationDegrees() % 360
    return (opening_board + rotation) % 360





# ---------------------------------------------------------------------------
# Self-contained pcbnew script executed in a subprocess by
# stamp_subcircuit_board_subprocess(). Lifted out of an inline string into
# its own file so import-time errors fire on adapter import (and so
# linters can actually see it). The script reads its JSON payload path
# from sys.argv[1].
# ---------------------------------------------------------------------------
_STAMP_SUBPROCESS_SCRIPT_PATH = str(
    Path(__file__).parent / "_stamp_subcircuit_subprocess.py"
)

class KiCadAdapter:
    """Reads and writes KiCad board state via pcbnew API."""

    def __init__(self, pcb_path: str, config: dict = None):
        self.pcb_path = pcb_path
        self.board = None
        self.cfg = config or {}

    def _ensure_loaded(self):
        if self.board is None:
            self.board = pcbnew.LoadBoard(self.pcb_path)

    def reload(self):
        """Force fresh board load."""
        self.board = None
        self._ensure_loaded()

    def load(self) -> BoardState:
        """Extract full BoardState from .kicad_pcb."""
        self._ensure_loaded()
        board = self.board

        # --- Board outline ---
        bbox = board.GetBoardEdgesBoundingBox()
        tl = Point(pcbnew.ToMM(bbox.GetLeft()), pcbnew.ToMM(bbox.GetTop()))
        br = Point(pcbnew.ToMM(bbox.GetRight()), pcbnew.ToMM(bbox.GetBottom()))

        # --- Components + Pads ---
        components: dict[str, Component] = {}
        net_pads: dict[str, list[tuple[str, str]]] = {}  # net_name -> [(ref, pad_id)]

        for fp in board.Footprints():
            ref = fp.GetReferenceAsString()
            val = fp.GetFieldText("Value")
            pos = fp.GetPosition()
            # Use courtyard bbox for physical size — it represents the keep-out
            # area on the PCB plane (excludes battery tube space above board).
            # A courtyard is only trusted when it encloses the pad copper: a
            # malformed (non-closing / self-intersecting) courtyard degenerates
            # to a stroke-width sliver that still passes a bare ">0" size check,
            # and a sliver extent lets the solver pack neighbours into the
            # part's real body. Fall back to the copper/graphics bounding box
            # when there is no usable courtyard.
            body_ctr = None
            try:
                cy = fp.GetCourtyard(
                    pcbnew.F_CrtYd if fp.GetLayer() == pcbnew.F_Cu else pcbnew.B_CrtYd
                )
                cbox = cy.BBox()
                if cbox.GetWidth() <= 0 or cbox.GetHeight() <= 0:
                    raise ValueError("empty courtyard")
                pads_box = None
                for _pad in fp.Pads():
                    pb = _pad.GetBoundingBox()
                    if pads_box is None:
                        pads_box = pb
                    else:
                        pads_box.Merge(pb)
                if pads_box is not None:
                    tol = pcbnew.FromMM(0.1)  # vendor hygiene grows real ones
                    if (cbox.GetLeft() > pads_box.GetLeft() + tol
                            or cbox.GetTop() > pads_box.GetTop() + tol
                            or cbox.GetRight() < pads_box.GetRight() - tol
                            or cbox.GetBottom() < pads_box.GetBottom() - tol):
                        raise ValueError("courtyard does not enclose pad copper")
                w_mm = pcbnew.ToMM(cbox.GetWidth())
                h_mm = pcbnew.ToMM(cbox.GetHeight())
                cc = cbox.GetCenter()
                body_ctr = Point(pcbnew.ToMM(cc.x), pcbnew.ToMM(cc.y))
            except Exception:
                fp_bbox = fp.GetBoundingBox(False, False)
                w_mm = pcbnew.ToMM(fp_bbox.GetWidth())
                h_mm = pcbnew.ToMM(fp_bbox.GetHeight())
                fc = fp_bbox.GetCenter()
                body_ctr = Point(pcbnew.ToMM(fc.x), pcbnew.ToMM(fc.y))
            # Sanity cap at board size — prevents degenerate courtyard bboxes
            w_mm = min(w_mm, 150.0)
            h_mm = min(h_mm, 150.0)

            kind = _classify_component(ref, val)
            # Detect through-hole: any pad with PTH attribute means THT footprint
            has_pth = any(p.GetAttribute() == pcbnew.PAD_ATTRIB_PTH for p in fp.Pads())
            # Lock mechanically-fixed parts unless unlock_all_footprints is set.
            # Battery holders have fixed positions by default.
            if self.cfg.get("unlock_all_footprints", False):
                is_locked = fp.IsLocked()
            else:
                is_locked = fp.IsLocked() or kind in ("battery",)
            # Back-side override: synthesis records side-of-board intent in
            # autoplacer.json (component_layers, from BomPart.side); honor it
            # here so the part loads as BACK and the existing stamp path flips
            # the footprint to B.Cu. The seed-PCB footprint is still front, so
            # the pad coords read below are its front positions -- the later
            # Flip() at stamp time owns the actual geometry mirror.
            _layer_override = self.cfg.get("component_layers") or {}
            comp_layer = (
                Layer.BACK
                if _layer_override.get(ref) == "back"
                else _layer_to_enum(fp.GetLayer())
            )
            # Mouth detection runs for connectors AND for anything the BOM
            # zoned to an edge: the facings fab gate checks every edge-zoned
            # ref prefix-blind, so an edge-zoned slide switch (SW1, kind
            # "misc") whose opening stayed None here was oriented by the
            # aspect-ratio fallback -- a coin flip the gate then honestly
            # rejected (self-eval 2026-07-20 run_05 usb-pd-trigger).
            _zone_edge = (
                (self.cfg.get("component_zones") or {}).get(ref) or {}
            ).get("edge")
            _wants_mouth = kind == "connector" or _zone_edge in (
                "left", "right", "top", "bottom"
            )
            comp = Component(
                ref=ref,
                value=val,
                pos=Point(pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)),
                rotation=fp.GetOrientationDegrees(),
                layer=comp_layer,
                width_mm=w_mm,
                height_mm=h_mm,
                pads=[],
                locked=is_locked,
                kind=kind,
                is_through_hole=has_pth,
                body_center=body_ctr,
                opening_direction=(
                    detect_opening_direction(fp) if _wants_mouth else None
                ),
            )

            for pad in fp.Pads():
                net_name = pad.GetNetname()
                if not net_name or net_name.startswith("unconnected-"):
                    continue
                ppos = pad.GetPosition()
                # Pad copper extent, world-axis-aligned. We deliberately use
                # GetBoundingBox() (not GetSize()) because GetSize() returns
                # the pad shape in the pad's LOCAL frame -- before footprint
                # rotation and before pad-local rotation. For rotated
                # footprints (every edge-mounted connector: USB-C, USB-A,
                # pin headers), the local extent's x/y don't match the
                # world axes, and Pad.bbox() -- which treats size_mm as
                # world-aligned half-extents around ``pos`` -- ends up
                # painting an AABB that doesn't enclose the actual pad.
                # That bug propagated through Component.physical_bbox ->
                # _compute_component_bbox -> silk poly -> Edge.Cuts, and
                # produced leaves whose board outline cut through their
                # own solder pads. GetBoundingBox() applies all rotations
                # and returns the world AABB, which is the contract Pad
                # downstream code assumes.
                pbbox = pad.GetBoundingBox()
                pad_size = Point(
                    pcbnew.ToMM(pbbox.GetWidth()),
                    pcbnew.ToMM(pbbox.GetHeight()),
                )
                p = Pad(
                    ref=ref,
                    pad_id=pad.GetNumber(),
                    pos=Point(pcbnew.ToMM(ppos.x), pcbnew.ToMM(ppos.y)),
                    net=net_name,
                    layer=_layer_to_enum(pad.GetLayer()),
                    size_mm=pad_size,
                )
                comp.pads.append(p)
                net_pads.setdefault(net_name, []).append((ref, pad.GetNumber()))

            components[ref] = comp

        # --- Nets ---
        nets: dict[str, Net] = {}
        for net_name, pads in net_pads.items():
            power_nets = self.cfg.get("power_nets", set())
            is_power = net_name in power_nets or net_name.lstrip("/") in power_nets
            pw = self.cfg.get("power_width_mm", POWER_WIDTH_MM)
            sw = self.cfg.get("signal_width_mm", SIGNAL_WIDTH_MM)
            nets[net_name] = Net(
                name=net_name,
                pad_refs=pads,
                width_mm=pw if is_power else sw,
                is_power=is_power,
            )

        # --- Existing traces ---
        traces: list[TraceSegment] = []
        vias: list[Via] = []
        for track in board.GetTracks():
            if isinstance(track, pcbnew.PCB_VIA):
                vpos = track.GetPosition()
                # KiCad 9: GetWidth requires layer arg for vias
                try:
                    via_size = pcbnew.ToMM(track.GetWidth(pcbnew.F_Cu))
                except TypeError:
                    via_size = pcbnew.ToMM(track.GetWidth())
                vias.append(
                    Via(
                        pos=Point(pcbnew.ToMM(vpos.x), pcbnew.ToMM(vpos.y)),
                        net=track.GetNetname(),
                        drill_mm=pcbnew.ToMM(track.GetDrill()),
                        size_mm=via_size,
                    )
                )
            else:
                s = track.GetStart()
                e = track.GetEnd()
                traces.append(
                    TraceSegment(
                        start=Point(pcbnew.ToMM(s.x), pcbnew.ToMM(s.y)),
                        end=Point(pcbnew.ToMM(e.x), pcbnew.ToMM(e.y)),
                        layer=_layer_to_enum(track.GetLayer()),
                        net=track.GetNetname(),
                        width_mm=pcbnew.ToMM(track.GetWidth()),
                    )
                )

        return BoardState(
            components=components,
            nets=nets,
            traces=traces,
            vias=vias,
            board_outline=(tl, br),
            keepout_rects=extract_keepout_rects(board, self.cfg),
        )

    def apply_placement(
        self, components: dict[str, Component], output_path: str = None
    ):
        """Move footprints to new positions/rotations. Preserves existing traces."""
        self._ensure_loaded()
        board = self.board

        # Apply board outline change if config specifies board dimensions
        if self.cfg.get("enable_board_size_search", False):
            w_mm = self.cfg.get("board_width_mm", 90.0)
            h_mm = self.cfg.get("board_height_mm", 58.0)
            self._apply_board_outline(w_mm, h_mm)

        for fp in board.Footprints():
            ref = fp.GetReferenceAsString()
            if ref not in components:
                continue
            comp = components[ref]
            # Only skip components explicitly locked by the user in KiCad.
            # The solver's locked flag (set for connectors/mounting_holes/
            # batteries by _pin_edge_components) is for the force simulation
            # only — their solver-computed positions must still be written.
            if fp.IsLocked():
                continue
            # Flip to correct layer if solver assigned a different side
            current_layer = _layer_to_enum(fp.GetLayer())
            if comp.layer != current_layer:
                fp.Flip(fp.GetPosition(), False)
            fp.SetPosition(
                pcbnew.VECTOR2I(
                    pcbnew.FromMM(comp.pos.x),
                    pcbnew.FromMM(comp.pos.y),
                )
            )
            fp.SetOrientationDegrees(comp.rotation)

        out = output_path or self.pcb_path
        # Pass the source PCB's sibling .kicad_pro so the saved output
        # inherits project netclass/rules instead of pcbnew defaults.
        source_pro = os.path.splitext(self.pcb_path)[0] + ".kicad_pro"
        _atomic_save_board(board, out, source_pro_path=source_pro)
        print(f"Placement saved to {out}")

    def _stamp_subcircuit_board_inprocess(
        self,
        state: BoardState,
        output_path: str | None = None,
        *,
        clear_existing_tracks: bool = True,
        clear_existing_zones: bool = True,
        remove_unmapped_footprints: bool = True,
    ):
        """In-process stamping of a leaf/subcircuit board onto a real KiCad board.

        NOTE: prefer stamp_subcircuit_board() which delegates to a subprocess
        to avoid SWIG memory corruption on repeated calls.

        This helper is intended for routed leaf subcircuits where the exported
        board must be loadable by pcbnew/FreeRouting as a real KiCad board, not
        just a synthetic text snapshot.

        Behavior:
        - rewrites the board outline to match the subcircuit-local board size
        - moves footprints that exist in `state.components`
        - optionally removes footprints not present in the subcircuit state
        - strips non-outline board drawings/text from the source board
        - optionally clears existing tracks/vias and copper zones
        - recreates traces/vias from the provided `BoardState`
        """
        self._ensure_loaded()
        board = self.board

        component_map = state.components or {}

        outline_left_mm = state.board_outline[0].x
        outline_top_mm = state.board_outline[0].y
        outline_right_mm = state.board_outline[1].x
        outline_bottom_mm = state.board_outline[1].y

        # Edge.Cuts traces the same rounded polyline as the F.SilkS leaf
        # outline -- the silk poly's `.points` IS the leaf boundary, so
        # we hand the same list to the outline stamper. Falls back to a
        # sharp rectangle only when no leaf-outline silk exists (unlabeled
        # leaf, parent flow), which is the legacy shape.
        leaf_outline_poly = _extract_leaf_outline_polyline_mm(state.silkscreen)

        self._apply_board_outline(
            max(1.0, outline_right_mm - outline_left_mm),
            max(1.0, outline_bottom_mm - outline_top_mm),
            left_mm=outline_left_mm,
            top_mm=outline_top_mm,
            polyline_mm=leaf_outline_poly,
        )

        component_map = state.components or {}
        footprints = list(board.Footprints())

        for fp in footprints:
            ref = fp.GetReferenceAsString()
            comp = component_map.get(ref)
            if comp is None:
                if remove_unmapped_footprints:
                    board.Remove(fp)
                continue

            if fp.IsLocked():
                continue

            current_layer = _layer_to_enum(fp.GetLayer())
            if comp.layer != current_layer:
                fp.Flip(fp.GetPosition(), False)
            fp.SetPosition(
                pcbnew.VECTOR2I(
                    pcbnew.FromMM(comp.pos.x),
                    pcbnew.FromMM(comp.pos.y),
                )
            )
            fp.SetOrientationDegrees(comp.rotation)

        to_remove = []
        for drawing in board.GetDrawings():
            try:
                if drawing.GetLayer() == pcbnew.Edge_Cuts:
                    continue
            except Exception:
                pass
            to_remove.append(drawing)
        for drawing in to_remove:
            board.Remove(drawing)

        if clear_existing_tracks:
            to_remove = [track for track in board.GetTracks()]
            for track in to_remove:
                board.Remove(track)

        if clear_existing_zones:
            to_remove = [zone for zone in board.Zones() if not zone.GetIsRuleArea()]
            for zone in to_remove:
                board.Remove(zone)

        netinfo = board.GetNetInfo()

        def _resolve_net_code(net_name: str) -> int:
            if not net_name:
                return 0
            net_item = netinfo.GetNetItem(net_name)
            if net_item is None:
                return 0
            try:
                return int(net_item.GetNetCode())
            except Exception:
                return 0

        for trace in state.traces:
            seg = pcbnew.PCB_TRACK(board)
            seg.SetStart(
                pcbnew.VECTOR2I(
                    pcbnew.FromMM(trace.start.x),
                    pcbnew.FromMM(trace.start.y),
                )
            )
            seg.SetEnd(
                pcbnew.VECTOR2I(
                    pcbnew.FromMM(trace.end.x),
                    pcbnew.FromMM(trace.end.y),
                )
            )
            seg.SetLayer(_enum_to_layer(trace.layer))
            seg.SetWidth(pcbnew.FromMM(trace.width_mm))
            net_code = _resolve_net_code(trace.net)
            if net_code > 0:
                seg.SetNetCode(net_code)
            board.Add(seg)

        for via in state.vias:
            track_via = pcbnew.PCB_VIA(board)
            track_via.SetPosition(
                pcbnew.VECTOR2I(
                    pcbnew.FromMM(via.pos.x),
                    pcbnew.FromMM(via.pos.y),
                )
            )
            track_via.SetDrill(pcbnew.FromMM(via.drill_mm))
            try:
                track_via.SetWidth(pcbnew.FromMM(via.size_mm))
            except TypeError:
                track_via.SetWidth(pcbnew.F_Cu, pcbnew.FromMM(via.size_mm))
            net_code = _resolve_net_code(via.net)
            if net_code > 0:
                track_via.SetNetCode(net_code)
            board.Add(track_via)

        board.BuildConnectivity()
        out = output_path or self.pcb_path
        source_pro = os.path.splitext(self.pcb_path)[0] + ".kicad_pro"
        _atomic_save_board(board, out, source_pro_path=source_pro)
        print(f"Subcircuit board stamped to {out}")

    # ------ subprocess-safe stamping (default) ------
    use_subprocess = True  # class-level flag; set False to use in-process path

    def stamp_subcircuit_board(
        self,
        state: BoardState,
        output_path: str | None = None,
        *,
        clear_existing_tracks: bool = True,
        clear_existing_zones: bool = True,
        remove_unmapped_footprints: bool = True,
    ):
        """Stamp a leaf/subcircuit board — delegates to subprocess or in-process.

        By default (use_subprocess=True) runs pcbnew operations in an isolated
        subprocess so that accumulated SWIG C++ objects from repeated calls
        cannot cause memory corruption or segfaults in the parent process.
        """
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if self.use_subprocess:
            return self.stamp_subcircuit_board_subprocess(
                state,
                output_path,
                clear_existing_tracks=clear_existing_tracks,
                clear_existing_zones=clear_existing_zones,
                remove_unmapped_footprints=remove_unmapped_footprints,
            )
        return self._stamp_subcircuit_board_inprocess(
            state,
            output_path,
            clear_existing_tracks=clear_existing_tracks,
            clear_existing_zones=clear_existing_zones,
            remove_unmapped_footprints=remove_unmapped_footprints,
        )

    def stamp_subcircuit_board_subprocess(
        self,
        state: BoardState,
        output_path: str | None = None,
        *,
        clear_existing_tracks: bool = True,
        clear_existing_zones: bool = True,
        remove_unmapped_footprints: bool = True,
    ):
        """Stamp a leaf/subcircuit board using an isolated subprocess.

        Serialises the BoardState to a JSON temp file, writes a self-contained
        pcbnew script, and runs it in a fresh Python process so that SWIG
        objects are discarded when the child exits.
        """
        component_map = state.components or {}
        outline_tl = state.board_outline[0]
        outline_br = state.board_outline[1]

        # -- serialise data to JSON --
        components_json = []
        for ref, comp in component_map.items():
            components_json.append({
                "ref": ref,
                "x": comp.pos.x,
                "y": comp.pos.y,
                "rotation": comp.rotation,
                "layer": 0 if comp.layer == Layer.FRONT else 1,
                "width_mm": comp.width_mm,
                "height_mm": comp.height_mm,
            })

        traces_json = []
        for trace in (state.traces or []):
            traces_json.append({
                "start_x": trace.start.x,
                "start_y": trace.start.y,
                "end_x": trace.end.x,
                "end_y": trace.end.y,
                "width": trace.width_mm,
                "layer": "F.Cu" if trace.layer == Layer.FRONT else "B.Cu",
                "net_name": trace.net or "",
            })

        vias_json = []
        for via in (state.vias or []):
            vias_json.append({
                "x": via.pos.x,
                "y": via.pos.y,
                "size": via.size_mm,
                "drill": via.drill_mm,
                "net_name": via.net or "",
            })

        silkscreen_json = []
        for elem in (state.silkscreen or []):
            if elem.kind == "poly":
                silkscreen_json.append({
                    "kind": "poly",
                    "layer": elem.layer,
                    "points": [{"x": p.x, "y": p.y} for p in elem.points],
                    "stroke_width": elem.stroke_width,
                })
            elif elem.kind == "text":
                silkscreen_json.append({
                    "kind": "text",
                    "layer": elem.layer,
                    "text": elem.text,
                    "pos": {"x": elem.pos.x, "y": elem.pos.y},
                    "font_height": elem.font_height,
                    "font_width": elem.font_width,
                    "font_thickness": elem.font_thickness,
                })

        outline_polyline = _extract_leaf_outline_polyline_mm(state.silkscreen)

        payload = {
            "pcb_path": self.pcb_path,
            "output_path": output_path or self.pcb_path,
            "outline": {
                "tl_x": outline_tl.x,
                "tl_y": outline_tl.y,
                "br_x": outline_br.x,
                "br_y": outline_br.y,
                # When present, Edge.Cuts is stamped as this closed
                # rounded polyline (matching the silk leaf outline);
                # otherwise the subprocess falls back to a sharp
                # rectangle derived from the tl/br bbox.
                "polyline": (
                    [[x, y] for x, y in outline_polyline]
                    if outline_polyline is not None
                    else None
                ),
            },
            "components": components_json,
            "traces": traces_json,
            "vias": vias_json,
            "silkscreen": silkscreen_json,
            "clear_existing_tracks": clear_existing_tracks,
            "clear_existing_zones": clear_existing_zones,
            "remove_unmapped_footprints": remove_unmapped_footprints,
        }

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="stamp_sub_")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(payload, f)

            try:
                _run_pcbnew_script_file(_STAMP_SUBPROCESS_SCRIPT_PATH, tmp_path)
            except RuntimeError as exc:
                # Convert the generic subprocess RuntimeError into a
                # StampSubprocessError so the catch in solve_subcircuits
                # / leaf_routing can distinguish "stamp script crashed"
                # (implementation bug -- bubble up loudly) from
                # "FreeRouting timed out" (recoverable, degrade the
                # round). Without this, a syntax / API regression in
                # _stamp_subcircuit_subprocess.py silently turns into a
                # routing_exception, the round is discarded with
                # parent_route=fail, and autoexperiment still reports
                # leafs=N/N "accepted" from the cached on-disk state.
                raise StampSubprocessError(str(exc)) from exc
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Force reload on next access since the file was written by child
        self.board = None
        out = output_path or self.pcb_path
        print(f"Subcircuit board stamped to {out} (subprocess)")

    def _apply_board_outline(
        self,
        width_mm: float,
        height_mm: float,
        *,
        left_mm: float = 0.0,
        top_mm: float = 0.0,
        polyline_mm: list[tuple[float, float]] | None = None,
    ):
        """Rewrite the Edge.Cuts shape.

        With ``polyline_mm`` supplied (leaf flow), Edge.Cuts is stamped as
        the same closed polyline that the F.SilkS leaf outline traces --
        the silk poly's ``.points`` and Edge.Cuts share one point list
        so the two layers cannot drift (previously Edge.Cuts was a sharp
        4-segment rectangle while silk was a rounded poly, leaving a
        visible ring of bare substrate at each corner). With
        ``polyline_mm=None`` (parent flow / unlabeled leaf), Edge.Cuts
        is stamped as a 4-segment sharp rectangle of the given
        dimensions, the legacy shape.

        Silk text + the leaf-outline silk poly itself are stamped from
        ``BoardState.silkscreen`` later in the stamp pipeline; this
        function only owns Edge.Cuts.
        """
        board = self.board

        # Remove existing Edge.Cuts lines AND any prior leaf-outline
        # silk segments left over from earlier stamps so re-stamps stay
        # clean.
        to_remove = []
        for dwg in board.GetDrawings():
            if dwg.GetLayer() == pcbnew.Edge_Cuts:
                to_remove.append(dwg)
            elif _is_leaf_outline_silk(dwg):
                to_remove.append(dwg)
        for dwg in to_remove:
            board.Remove(dwg)

        if polyline_mm is not None and len(polyline_mm) >= 3:
            n = len(polyline_mm)
            for i in range(n):
                x1, y1 = polyline_mm[i]
                x2, y2 = polyline_mm[(i + 1) % n]
                edge = pcbnew.PCB_SHAPE(board)
                edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
                edge.SetLayer(pcbnew.Edge_Cuts)
                edge.SetWidth(pcbnew.FromMM(0.05))
                edge.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(x1), pcbnew.FromMM(y1)))
                edge.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(x2), pcbnew.FromMM(y2)))
                board.Add(edge)
            return

        new_left = pcbnew.FromMM(left_mm)
        new_top = pcbnew.FromMM(top_mm)
        new_right = pcbnew.FromMM(left_mm + width_mm)
        new_bottom = pcbnew.FromMM(top_mm + height_mm)

        corners = [
            (new_left, new_top),
            (new_right, new_top),
            (new_right, new_bottom),
            (new_left, new_bottom),
        ]
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % 4]
            edge = pcbnew.PCB_SHAPE(board)
            edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
            edge.SetLayer(pcbnew.Edge_Cuts)
            edge.SetWidth(pcbnew.FromMM(0.05))
            edge.SetStart(pcbnew.VECTOR2I(x1, y1))
            edge.SetEnd(pcbnew.VECTOR2I(x2, y2))
            board.Add(edge)

    def strip_zones(self):
        """Remove all non-rule-area copper zones from the board.

        Called before routing to remove pre-existing zones (e.g. F.Cu GND
        zone from the source PCB) that would interfere with the autoplacer's
        zone management.  Rule areas are preserved.

        Runs in a subprocess to avoid pcbnew SWIG corruption.
        """
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import pcbnew\n"
                f"board = pcbnew.LoadBoard({self.pcb_path!r})\n"
                "to_remove = [z for z in board.Zones() if not z.GetIsRuleArea()]\n"
                "for z in to_remove:\n"
                "    board.Remove(z)\n"
                f"board.Save({self.pcb_path!r})\n"
                "print(len(to_remove))\n",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            # Take first line only — SWIG may print memory leak warnings after
            n = result.stdout.strip().split("\n")[0].strip()
            if n and n.isdigit() and int(n) > 0:
                print(f"  Stripped {n} pre-existing copper zone(s)")
        # Force reload on next access since file changed
        self.board = None

    def ensure_gnd_zone(self):
        """Create or update a GND copper pour zone covering the full board.

        Idempotent: if a zone already exists on the target layer with the
        target net, its outline is updated to match the current board
        dimensions. Otherwise a new zone is created.

        Controlled by config keys:
          gnd_zone_net (str): Net name, e.g. "GND". Empty string disables.
          gnd_zone_layer (str): "B.Cu" or "F.Cu".
          gnd_zone_margin_mm (float): Inset from board edge.
        """
        self._ensure_loaded()
        board = self.board

        zone_net_name = self.cfg.get("gnd_zone_net", "GND")
        if not zone_net_name:
            return  # Disabled

        layer_name = self.cfg.get("gnd_zone_layer", "B.Cu")
        target_layer = pcbnew.B_Cu if layer_name == "B.Cu" else pcbnew.F_Cu
        margin = pcbnew.FromMM(self.cfg.get("gnd_zone_margin_mm", 0.5))

        # Find the net
        gnd_net = board.GetNetInfo().GetNetItem(zone_net_name)
        if not gnd_net or gnd_net.GetNetCode() == 0:
            print(
                f"  WARNING: Net '{zone_net_name}' not found — skipping zone creation"
            )
            return

        # Compute board outline rectangle
        rect = board.GetBoardEdgesBoundingBox()
        x1 = rect.GetX() + margin
        y1 = rect.GetY() + margin
        x2 = x1 + rect.GetWidth() - 2 * margin
        y2 = y1 + rect.GetHeight() - 2 * margin

        # Look for existing zone on target layer with matching net
        existing_zone = None
        for zone in board.Zones():
            if (
                zone.GetLayer() == target_layer
                and zone.GetNetname() == zone_net_name
                and not zone.GetIsRuleArea()
            ):
                existing_zone = zone
                break

        if existing_zone:
            # Update outline to match current board size
            outline = existing_zone.Outline()
            outline.RemoveAllContours()
            outline.NewOutline()
            outline.Append(x1, y1)
            outline.Append(x2, y1)
            outline.Append(x2, y2)
            outline.Append(x1, y2)
        else:
            # Create new zone
            zone = pcbnew.ZONE(board)
            zone.SetNet(gnd_net)
            zone.SetLayer(target_layer)
            zone.SetIsRuleArea(False)
            zone.SetDoNotAllowTracks(False)
            zone.SetDoNotAllowVias(False)
            zone.SetDoNotAllowPads(False)
            zone.SetDoNotAllowCopperPour(False)
            zone.SetLocalClearance(pcbnew.FromMM(self.cfg.get("zone_clearance_mm", 0.3)))
            zone.SetMinThickness(pcbnew.FromMM(self.cfg.get("zone_min_thickness_mm", 0.25)))
            zone.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
            zone.SetThermalReliefGap(pcbnew.FromMM(self.cfg.get("zone_thermal_gap_mm", 0.5)))
            zone.SetThermalReliefSpokeWidth(pcbnew.FromMM(self.cfg.get("zone_thermal_spoke_mm", 0.5)))
            zone.SetAssignedPriority(0)
            outline = zone.Outline()
            outline.NewOutline()
            outline.Append(x1, y1)
            outline.Append(x2, y1)
            outline.Append(x2, y2)
            outline.Append(x1, y2)
            board.Add(zone)

        # Fill all zones
        filler = pcbnew.ZONE_FILLER(board)
        filler.Fill(board.Zones())
        print(f"  GND zone on {layer_name}: ensured and filled")
