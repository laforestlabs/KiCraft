"""KC-3WN46Z regression: wide-shell connectors must expose a detectable mouth.

The USB-A receptacle (USB-A-SMD_U-A-24SS-W-2, auto-fetched, no marker) shipped
with its facing verdict stuck at "unknown_mouth": the body-overhang heuristic
ranked its decisive fore/aft mouth overhang (2.41 vs 1.70 mm) against the
part's large SYMMETRIC lateral skirts (1.95 mm each side) and missed the
0.5 mm separation cut by 0.04 mm, so every zoned USB-A warned "connector mouth
unverifiable" on an actually-correct board.

These tests pin the fix layers:
  1. the axis-pair fallback detects the mouth from geometry alone (marker
     disabled), comparing fore-vs-aft instead of best-vs-global-runner-up;
  2. the vendored bundle's 'PCB Edge' Dwgs.User marker is authoritative and
     rotation-invariant;
  3. genuinely mouthless bodies (vertical radial film cap) stay undetected --
     the fallback must not invent a direction, and the shared add-part /
     validate-part lint still trips on them;
  4. the fab-gate facing verdict accepts the marked USB-A aimed off-board and
     flags it when aimed inboard.

NOTE: the marker is disabled by rewriting the .kicad_mod TEXT into a tmp
library, never by FOOTPRINT.Remove on a live footprint -- removing an item
and letting its proxy GC corrupts the SWIG heap for every later pcbnew call
in the process.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

LIB_USB_A = Path("kicraft/parts_library/usb-a-24ss-w-2")
FP_USB_A = "USB-A-SMD_U-A-24SS-W-2"
LIB_FILM_CAP = Path("kicraft/parts_library/mes104j2a-7-50r0")
FP_FILM_CAP = "CAP-TH_L7.2-W4.0-P5.00-D0.5"


def _load(pretty: Path, name: str):
    pcbnew = pytest.importorskip("pcbnew")
    fp = pcbnew.FootprintLoad(str(pretty), name)
    assert fp is not None
    return fp


def _markerless_pretty(tmp_path: Path) -> Path:
    """A tmp .pretty whose USB-A marker text no longer says 'edge', so
    detection rule 1 cannot fire and the geometry fallback is exercised."""
    pretty = tmp_path / "nomarker.pretty"
    pretty.mkdir()
    src = LIB_USB_A / f"{LIB_USB_A.name}.pretty" / f"{FP_USB_A}.kicad_mod"
    text = src.read_text()
    assert '"PCB Edge"' in text
    (pretty / f"{FP_USB_A}.kicad_mod").write_text(
        text.replace('"PCB Edge"', '"stripped"')
    )
    return pretty


# --- Layer 1: the axis-pair fallback (geometry alone, no marker) ---------


def test_wide_shell_mouth_detected_without_marker(tmp_path):
    from kicraft.autoplacer.hardware.adapter import detect_opening_direction

    fp = _load(_markerless_pretty(tmp_path), FP_USB_A)
    assert detect_opening_direction(fp) == 90.0  # mouth +Y at rot 0
    for rot in (90.0, 180.0, 270.0):
        fp.SetOrientationDegrees(rot)
        assert detect_opening_direction(fp) == 90.0


# --- Layer 2: the vendored marker is authoritative ------------------------


def test_vendored_marker_detects_mouth():
    from kicraft.autoplacer.hardware.adapter import detect_opening_direction

    fp = _load(LIB_USB_A / f"{LIB_USB_A.name}.pretty", FP_USB_A)
    assert detect_opening_direction(fp) == 90.0
    for rot in (90.0, 180.0, 270.0):
        fp.SetOrientationDegrees(rot)
        assert detect_opening_direction(fp) == 90.0


# --- Layer 3: no invented mouths on mouthless bodies ----------------------


def test_vertical_film_cap_stays_undetected():
    from kicraft.autoplacer.hardware.adapter import detect_opening_direction

    fp = _load(LIB_FILM_CAP / f"{LIB_FILM_CAP.name}.pretty", FP_FILM_CAP)
    assert detect_opening_direction(fp) is None


def test_mouth_undetectable_lint_predicate(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.design.cli_app import _mouth_undetectable

    # Deep holed body, no detectable mouth -> the add-part/validate-part
    # lint trips (warning-only; vertical parts legitimately have no mouth).
    fp_cap = _load(LIB_FILM_CAP / f"{LIB_FILM_CAP.name}.pretty", FP_FILM_CAP)
    assert _mouth_undetectable(fp_cap, pcbnew)
    # The fixed USB-A no longer trips it, marker or not.
    assert not _mouth_undetectable(
        _load(LIB_USB_A / f"{LIB_USB_A.name}.pretty", FP_USB_A), pcbnew
    )
    assert not _mouth_undetectable(
        _load(_markerless_pretty(tmp_path), FP_USB_A), pcbnew
    )


# --- Layer 4: the fab-gate facing verdict ---------------------------------


def _make_board(tmp_path: Path, *, rotation: float) -> Path:
    pcbnew = pytest.importorskip("pcbnew")
    board = pcbnew.CreateEmptyBoard()
    fp = _load(LIB_USB_A / f"{LIB_USB_A.name}.pretty", FP_USB_A)
    fp.SetReference("J2")
    fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(160), pcbnew.FromMM(100)))
    fp.SetOrientationDegrees(rotation)
    board.Add(fp)
    rect = pcbnew.PCB_SHAPE(board)
    rect.SetShape(pcbnew.SHAPE_T_RECT)
    rect.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(130), pcbnew.FromMM(90)))
    rect.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(170), pcbnew.FromMM(110)))
    rect.SetLayer(pcbnew.Edge_Cuts)
    board.Add(rect)
    out = tmp_path / "usb_a_facing.kicad_pcb"
    pcbnew.SaveBoard(str(out), board)
    return out


ZONES = {"J2": {"edge": "right"}}


def test_facing_accepts_mouth_outward(tmp_path):
    from kicraft.autoplacer.brain.connector_edge_gap import connector_facings

    # board_opening = local(90) - rotation(90) = 0 = right-edge outward.
    (v,) = connector_facings(str(_make_board(tmp_path, rotation=90.0)), ZONES)
    assert v.status == "ok"


def test_facing_flags_mouth_inboard(tmp_path):
    from kicraft.autoplacer.brain.connector_edge_gap import connector_facings

    (v,) = connector_facings(str(_make_board(tmp_path, rotation=270.0)), ZONES)
    assert v.status == "misoriented"
