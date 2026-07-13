"""Tests for fine-pitch connector clearance handling (E2E Finding 3).

Default routing used a 0.2mm clearance (the bare-board default), wider than a
dense connector's pad gaps (USB-C ~0.10mm), so the autorouter could not escape
the pad field and the gaps showed up as clearance DRC violations. The pipeline
now detects fine-pitch parts and lowers the routing clearance + track width for
the FreeRouting pass, and labels intrinsic connector-shield TH-pad DRC items as
waived.
"""
from __future__ import annotations

import os
import re
import tempfile

import pytest

from kicraft.autoplacer import freerouting_runner as fr
from kicraft.autoplacer.config import DEFAULT_CONFIG

# A minimal Specctra DSN rule block in the form KiCad's ExportSpecctraDSN emits.
_DSN = """(pcb board
  (structure
    (rule
      (width 200)
      (clearance 200)
      (clearance 50 (type smd_smd))
    )
  )
  (network
    (class kicad_default
      (rule
        (width 200)
        (clearance 200)
      )
    )
  )
)
"""


def _write_dsn(tmp_path) -> str:
    p = os.path.join(tmp_path, "board.dsn")
    with open(p, "w") as f:
        f.write(_DSN)
    return p


def test_patch_lowers_clearance_and_width_for_fine_pitch(tmp_path):
    dsn = _write_dsn(tmp_path)
    fr._patch_dsn_clearance(dsn, target_clearance_um=100, target_width_um=150)
    out = open(dsn).read()
    # Global + class clearance lowered 200 -> 100.
    assert "(clearance 100)" in out
    assert "(clearance 200)" not in out
    # Track width lowered 200 -> 150.
    assert "(width 150)" in out
    assert "(width 200)" not in out
    # Already-tighter typed clearance left alone (NOT raised to target).
    assert "(clearance 50 (type smd_smd))" in out


def test_patch_legacy_raises_typed_clearance_when_no_target(tmp_path):
    dsn = _write_dsn(tmp_path)
    fr._patch_dsn_clearance(dsn)  # legacy path
    out = open(dsn).read()
    # smd_smd raised up to the global clearance.
    assert "(clearance 200 (type smd_smd))" in out
    # Global + width untouched.
    assert "(clearance 200)" in out
    assert "(width 200)" in out


def test_patch_noop_when_target_not_below_global(tmp_path):
    # target >= global -> falls through to the legacy raise branch, never lowers.
    dsn = _write_dsn(tmp_path)
    fr._patch_dsn_clearance(dsn, target_clearance_um=200, target_width_um=150)
    out = open(dsn).read()
    assert "(clearance 200)" in out
    assert "(width 200)" in out  # width is only lowered on the fine-pitch path


def test_clearance_guard_raises_every_clearance_token(tmp_path):
    # KC-9G4YPT: FreeRouting's pad-shape approximation is ~1 µm off KiCad's
    # exact geometry, so routing at exactly the rule yields sub-rule DRC
    # measurements. The guard raises what FR routes to; widths stay put.
    dsn = _write_dsn(tmp_path)
    fr._apply_dsn_clearance_guard(dsn, 5)
    out = open(dsn).read()
    assert out.count("(clearance 205)") == 2  # structure rule + class rule
    assert "(clearance 55 (type smd_smd))" in out
    assert out.count("(width 200)") == 2  # widths untouched


def test_clearance_guard_noop_when_zero(tmp_path):
    dsn = _write_dsn(tmp_path)
    fr._apply_dsn_clearance_guard(dsn, 0)
    assert open(dsn).read() == _DSN


def test_clearance_guard_runs_last_over_fine_pitch_and_netclass_rules(tmp_path):
    # Full DSN rewrite chain as export_dsn runs it: fine-pitch lowering, then
    # netclass re-injection, then the guard. Every clearance the router will
    # obey -- lowered global, preserved typed, injected per-class -- must gain
    # exactly +5, or the zero-margin hole reopens for that rule class (e.g. a
    # Power-class GND wire, exactly the KC-9G4YPT violation).
    dsn = os.path.join(tmp_path, "board.dsn")
    with open(dsn, "w") as f:
        f.write(
            """(pcb board
  (structure
    (rule
      (width 200)
      (clearance 200)
      (clearance 50 (type smd_smd))
    )
  )
  (network
    (class kicad_default GND SIG
      (circuit (use_via "Via[0-1]_600:300_um"))
      (rule
        (width 200)
        (clearance 200)
      )
    )
  )
)
"""
        )
    import json

    # SIG carries no netclass mapping so it stays in the DSN-default class at
    # the fine-pitch-lowered clearance (a mapped class re-raises to its board
    # netclass value -- _inject_netclass_clearances' documented semantics).
    with open(dsn + ".netclasses.json", "w") as f:
        json.dump(
            {
                "classes": {"Power": 300},
                "net_class": {"GND": "Power"},
            },
            f,
        )

    fr._patch_dsn_clearance(dsn, target_clearance_um=153, target_width_um=153)
    fr._inject_netclass_clearances(dsn)
    fr._apply_dsn_clearance_guard(dsn, 5)
    out = open(dsn).read()

    # Structure rule + Default-class rule: lowered 200 -> 153, guarded -> 158.
    assert out.count("(clearance 158)") == 2
    # Injected Power class: max(153, 300) = 300, guarded -> 305.
    assert "(clearance 305)" in out
    # Typed clearance kept below target by the fine-pitch path, still guarded.
    assert "(clearance 55 (type smd_smd))" in out
    # Widths: lowered to 153 by the fine-pitch path, NOT touched by the guard.
    assert "(width 153)" in out
    assert "(width 158)" not in out and "(width 305)" not in out
    # No unguarded clearance token survives anywhere.
    assert not [
        m for m in re.findall(r"\(clearance\s+(\d+)", out)
        if int(m) in (153, 200, 300, 50)
    ]


def test_resolve_rule_honors_explicit_override(monkeypatch):
    monkeypatch.setattr(fr, "min_intra_footprint_pad_gap_mm",
                        lambda *_a, **_k: pytest.fail("should not detect when override set"))
    cfg = {**DEFAULT_CONFIG, "freerouting_clearance_mm": 0.12}
    clearance_um, width_um = fr._resolve_fine_pitch_rule("board.kicad_pcb", cfg)
    assert clearance_um == 120
    assert width_um == 153  # min(0.2, fine_pitch_track 0.153) * 1000 (fab floor)


def test_resolve_rule_auto_detects_and_floors(monkeypatch):
    # Densest gap 0.08mm is below the fab clearance floor -> clamp up to floor.
    monkeypatch.setattr(fr, "min_intra_footprint_pad_gap_mm", lambda *_a, **_k: 0.08)
    clearance_um, width_um = fr._resolve_fine_pitch_rule("b.kicad_pcb", dict(DEFAULT_CONFIG))
    assert clearance_um == 153  # floored at freerouting_min_clearance_mm 0.153
    assert width_um == 153


def test_resolve_rule_noop_for_coarse_board(monkeypatch):
    monkeypatch.setattr(fr, "min_intra_footprint_pad_gap_mm", lambda *_a, **_k: 0.5)
    assert fr._resolve_fine_pitch_rule("b.kicad_pcb", dict(DEFAULT_CONFIG)) == (None, None)


def test_resolve_rule_noop_when_no_pads(monkeypatch):
    monkeypatch.setattr(fr, "min_intra_footprint_pad_gap_mm", lambda *_a, **_k: None)
    assert fr._resolve_fine_pitch_rule("b.kicad_pcb", dict(DEFAULT_CONFIG)) == (None, None)


def test_connector_shield_annular_padstack_waived(monkeypatch, tmp_path):
    # Residual zero-annular TH shield-tab DRC items on an edge connector are
    # labeled footprint-internal and do not block acceptance.
    board = os.path.join(tmp_path, "routed.kicad_pcb")
    open(board, "w").close()

    report = (
        "[annular_width]: Annular width too small @(50.0 mm, 50.0 mm)\n"
        "    Hole of J1\n"
        "[annular_width]: Annular width too small @(51.0 mm, 50.0 mm)\n"
        "    Hole of J1\n"
        "[padstack]: Pad issue @(50.0 mm, 51.0 mm)\n"
        "    Pad of J1\n"
        "[padstack]: Pad issue @(51.0 mm, 51.0 mm)\n"
        "    Pad of J1\n"
    )
    canned = {
        "shorts": 0, "unconnected": 0, "clearance": 0, "copper_edge_clearance": 0,
        "courtyard": 0, "solder_mask_bridge": 0, "annular_width": 2, "padstack": 2,
        "total": 4, "violations": [], "ran": True, "report_text": report,
        "timed_out": False, "missing_cli": False,
    }
    monkeypatch.setattr(fr, "_run_kicad_cli_drc", lambda *_a, **_k: dict(canned))
    monkeypatch.setattr(fr, "count_board_tracks",
                        lambda *_a, **_k: {"traces": 1, "vias": 0, "total_length_mm": 1.0})

    cfg = {"component_zones": {"J1": {"edge": True}}}
    v = fr.validate_routed_board(board, cfg=cfg)

    assert v["footprint_internal_annular_count"] == 2
    assert v["footprint_internal_padstack_count"] == 2
    assert v["waived_connector_shield_refs"] == ["J1"]
    assert v["accepted"] is True


def test_connector_shield_not_waived_when_ref_unknown(monkeypatch, tmp_path):
    # annular/padstack on a non-edge, non-ignorable ref are not labeled waived
    # (but still do not newly block acceptance -- they were never blockers).
    board = os.path.join(tmp_path, "routed.kicad_pcb")
    open(board, "w").close()
    report = "[annular_width]: too small @(1 mm, 1 mm)\n    Hole of U7\n"
    canned = {
        "shorts": 0, "unconnected": 0, "clearance": 0, "copper_edge_clearance": 0,
        "courtyard": 0, "solder_mask_bridge": 0, "annular_width": 1, "padstack": 0,
        "total": 1, "violations": [], "ran": True, "report_text": report,
        "timed_out": False, "missing_cli": False,
    }
    monkeypatch.setattr(fr, "_run_kicad_cli_drc", lambda *_a, **_k: dict(canned))
    monkeypatch.setattr(fr, "count_board_tracks",
                        lambda *_a, **_k: {"traces": 1, "vias": 0, "total_length_mm": 1.0})
    v = fr.validate_routed_board(board, cfg={"component_zones": {}})
    assert "footprint_internal_annular_count" not in v
    assert v["accepted"] is True


@pytest.mark.parametrize("fp_name", ["USB-C_SMD-TYPE-C-31-M-12_1"])
def test_real_usb_c_detected_as_fine_pitch(fp_name):
    pcbnew = pytest.importorskip("pcbnew")
    fp_dir = "kicraft/parts_library/usb-c-16p/usb-c-16p.pretty"
    if not os.path.isdir(fp_dir):
        pytest.skip("vendored usb-c footprint unavailable")

    d = tempfile.mkdtemp()
    pcb = os.path.join(d, "t.kicad_pcb")
    board = pcbnew.NewBoard(pcb)
    fp = pcbnew.FootprintLoad(fp_dir, fp_name)
    if fp is None:
        pytest.skip("could not load usb-c footprint")
    fp.SetReference("J1")
    fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(50), pcbnew.FromMM(50)))
    board.Add(fp)
    pads = list(fp.Pads())
    n1 = pcbnew.NETINFO_ITEM(board, "CC1"); board.Add(n1)
    n2 = pcbnew.NETINFO_ITEM(board, "CC2"); board.Add(n2)
    pads[0].SetNetCode(n1.GetNetCode())
    pads[1].SetNetCode(n2.GetNetCode())
    board.Save(pcb)

    gap = fr.min_intra_footprint_pad_gap_mm(pcb)
    assert gap is not None and gap < 0.2, f"expected fine pitch, got {gap}"
    clearance_um, width_um = fr._resolve_fine_pitch_rule(pcb, dict(DEFAULT_CONFIG))
    assert clearance_um is not None and clearance_um <= 200
    assert width_um == 153
