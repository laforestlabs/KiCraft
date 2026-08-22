"""Connector stranding is non-fatal: route + flag, never silently fab-ready.

A composed parent whose only defect is an edge-zoned connector stranded inboard
of its board edge used to ``return 1`` *before routing*, surfacing as
``rc=6 "no routed parent"`` -- a misleading route/infra failure with no board at
all. It is now routed and promoted, honestly marked NOT fab-ready, while the
fab-readiness verify gate independently re-checks stranding so such a board can
never reach "fab-ready".

These tests pin both halves of that contract:
  * ``_promotable_strand_only`` -- only a strand-only, electrically complete
    board is promoted (boards with other defects stay hard-rejected).
  * ``_verify_routed_board`` -- a stranded board is ALWAYS not-ok (the safety
    guarantee), a clean board is unaffected (no regression).
  * ``_connector_stranded_refs`` -- discovers project edge zones and never
    invents a failure when zones are absent or unreadable.
"""
from __future__ import annotations

import json

import pytest

from kicraft.autoplacer.brain.connector_edge_gap import EdgeGap
from kicraft.cli.compose_subcircuits import _promotable_strand_only
from kicraft.design import cli_app

STRAND = ["connector_stranded:J2@-3.46mm(right)"]
STRAND2 = [
    "connector_stranded:J2@-3.46mm(right)",
    "connector_stranded:TB2@-1.52mm(bottom)",
]


# --- _promotable_strand_only: only strand-only + electrically complete promotes


def test_promote_strand_only_clean():
    # Sole defect is stranding, no shorts/unconnected -> promote (NOT fab-ready).
    assert _promotable_strand_only(STRAND, STRAND, {"shorts": 0, "unconnected": 0})


def test_promote_two_stranded_connectors():
    assert _promotable_strand_only(STRAND2, STRAND2, {})


def test_reject_when_illegal_routed_geometry_too():
    # Another real defect alongside stranding -> stays hard-rejected (rc=6).
    reasons = ["illegal_routed_geometry", *STRAND]
    assert not _promotable_strand_only(reasons, STRAND, {})


def test_reject_when_unconnected_nets():
    reasons = ["unconnected_nets", *STRAND]
    assert not _promotable_strand_only(reasons, STRAND, {"unconnected": 4})


def test_reject_when_unconnected_in_drc_even_if_reasons_strand_only():
    # Belt-and-suspenders: unconnected reported via DRC, not as a reason string.
    assert not _promotable_strand_only(STRAND, STRAND, {"unconnected": 2})


def test_reject_when_shorts():
    assert not _promotable_strand_only(STRAND, STRAND, {"shorts": 1})


def test_no_strand_never_promotes():
    # Rejected for a non-stranding reason with no recorded stranding -> reject.
    assert not _promotable_strand_only(["illegal_routed_geometry"], [], {})


# --- _verify_routed_board: a stranded board is non-blocking (warns, still fab-acceptable)


def _patch_validate(monkeypatch, *, accepted, drc=None, reasons=None):
    def _fake(_path, cfg=None):
        return {
            "accepted": accepted,
            "drc": drc or {"shorts": 0, "unconnected": 0},
            "rejection_reasons": list(reasons or []),
            "track_summary": {},
        }

    monkeypatch.setattr(
        "kicraft.autoplacer.routing_board.validate_routed_board", _fake
    )

def test_verify_gate_strand_is_warning_not_fail(monkeypatch, tmp_path):
    # A stranded connector on an electrically-clean board is a WARNING
    # (board still fab-acceptable + 3D-rendered), not a hard failure.
    _patch_validate(monkeypatch, accepted=True)
    monkeypatch.setattr(cli_app, "_connector_stranded_refs", lambda _pcb: list(STRAND))
    gate = cli_app._verify_routed_board(tmp_path / "board.kicad_pcb")
    assert gate["ok"] is True              # electrically clean
    assert gate["fab_acceptable"] is True  # strand-only is buildable + exportable
    assert STRAND[0] in gate["reasons"]
    assert gate["warnings"] and "connector stranded" in gate["warnings"][0]


def test_verify_gate_strand_plus_shorts_still_fails(monkeypatch, tmp_path):
    # Strand + real electrical defect -> still hard-fails (regression guard).
    _patch_validate(monkeypatch, accepted=True, drc={"shorts": 1, "unconnected": 0})
    monkeypatch.setattr(cli_app, "_connector_stranded_refs", lambda _pcb: list(STRAND))
    gate = cli_app._verify_routed_board(tmp_path / "board.kicad_pcb")
    assert gate["ok"] is False
    assert gate["fab_acceptable"] is False
    assert STRAND[0] in gate["reasons"]


def test_verify_gate_passes_clean_board(monkeypatch, tmp_path):
    # Accepted, clean DRC, no stranding -> fab-ready (no regression).
    _patch_validate(monkeypatch, accepted=True)
    monkeypatch.setattr(cli_app, "_connector_stranded_refs", lambda _pcb: [])
    gate = cli_app._verify_routed_board(tmp_path / "board.kicad_pcb")
    assert gate["ok"] is True
    assert gate["reasons"] == []


def test_verify_gate_still_fails_on_shorts(monkeypatch, tmp_path):
    # Independent of stranding, shorts still fail the gate.
    _patch_validate(monkeypatch, accepted=True, drc={"shorts": 1, "unconnected": 0})
    monkeypatch.setattr(cli_app, "_connector_stranded_refs", lambda _pcb: [])
    gate = cli_app._verify_routed_board(tmp_path / "board.kicad_pcb")
    assert gate["ok"] is False
    assert gate["shorts"] == 1


def test_verify_gate_fails_on_courtyard_overlap(monkeypatch, tmp_path):
    # Electrically clean (no shorts/unconnected) but two courtyards overlap ->
    # physically un-assemblable -> the verdict backstop must reject it.
    _patch_validate(
        monkeypatch, accepted=True,
        drc={"shorts": 0, "unconnected": 0, "courtyard": 2},
    )
    monkeypatch.setattr(cli_app, "_connector_stranded_refs", lambda _pcb: [])
    gate = cli_app._verify_routed_board(tmp_path / "board.kicad_pcb")
    assert gate["ok"] is False
    assert "courtyards_overlap" in gate["reasons"]
    assert gate["courtyard"] == 2


def test_verify_gate_minor_courtyard_is_warning_not_fail(monkeypatch, tmp_path):
    # A fraction-of-a-mm courtyard clip on an electrically-perfect board is a
    # WARNING (board still fab-acceptable + 3D-rendered), not a hard failure.
    from kicraft.autoplacer.courtyard_overlap import CourtyardOverlap

    _patch_validate(
        monkeypatch, accepted=True,
        drc={"shorts": 0, "unconnected": 0, "courtyard": 1},
    )
    monkeypatch.setattr(cli_app, "_connector_stranded_refs", lambda _pcb: [])
    monkeypatch.setattr(
        "kicraft.autoplacer.courtyard_overlap.measure_courtyard_overlaps",
        lambda _p: [CourtyardOverlap("R7", "SW2", "F", area_mm2=0.23, penetration_mm=0.31)],
    )
    gate = cli_app._verify_routed_board(tmp_path / "board.kicad_pcb")
    assert gate["ok"] is False             # not a pristine board
    assert gate["fab_acceptable"] is True  # ...but still buildable + exportable
    assert gate["courtyard_minor_only"] is True
    assert "courtyards_overlap" not in gate["reasons"]
    assert gate["warnings"] and "R7" in gate["warnings"][0]


def test_verify_gate_gross_courtyard_still_fails(monkeypatch, tmp_path):
    # A deep overlap (parts physically colliding) still hard-fails.
    from kicraft.autoplacer.courtyard_overlap import CourtyardOverlap

    _patch_validate(
        monkeypatch, accepted=True,
        drc={"shorts": 0, "unconnected": 0, "courtyard": 1},
    )
    monkeypatch.setattr(cli_app, "_connector_stranded_refs", lambda _pcb: [])
    monkeypatch.setattr(
        "kicraft.autoplacer.courtyard_overlap.measure_courtyard_overlaps",
        lambda _p: [CourtyardOverlap("U1", "U2", "F", area_mm2=4.0, penetration_mm=1.5)],
    )
    gate = cli_app._verify_routed_board(tmp_path / "board.kicad_pcb")
    assert gate["ok"] is False
    assert gate["fab_acceptable"] is False
    assert gate["courtyard_minor_only"] is False
    assert "courtyards_overlap" in gate["reasons"]


def test_verify_gate_fails_on_keepout_intrusion(monkeypatch, tmp_path):
    # Copper inside an antenna keep-out (items_not_allowed) is electrically
    # invisible but ruins RF / collides -> not fab-ready (KC-8AG6FU backstop).
    _patch_validate(
        monkeypatch, accepted=True,
        drc={"shorts": 0, "unconnected": 0, "items_not_allowed": 5},
    )
    monkeypatch.setattr(cli_app, "_connector_stranded_refs", lambda _pcb: [])
    gate = cli_app._verify_routed_board(tmp_path / "board.kicad_pcb")
    assert gate["ok"] is False
    assert "keepout_intrusion" in gate["reasons"]
    assert gate["keepout"] == 5


def test_verify_gate_clean_board_reports_zero_courtyard_keepout(monkeypatch, tmp_path):
    # Regression: a clean board still passes and surfaces the new counts as 0
    # (so the backstop never false-fires on a healthy board).
    _patch_validate(monkeypatch, accepted=True)
    monkeypatch.setattr(cli_app, "_connector_stranded_refs", lambda _pcb: [])
    gate = cli_app._verify_routed_board(tmp_path / "board.kicad_pcb")
    assert gate["ok"] is True
    assert gate["reasons"] == []
    assert gate["courtyard"] == 0 and gate["keepout"] == 0


# --- _connector_stranded_refs: zone discovery, and never invent a failure


def test_stranded_refs_empty_without_zones(tmp_path):
    # No *_autoplacer.json next to the board -> no zones -> nothing flagged.
    (tmp_path / "board.kicad_pcb").write_text("(kicad_pcb)")
    assert cli_app._connector_stranded_refs(tmp_path / "board.kicad_pcb") == []


def test_stranded_refs_discovers_zones_and_filters_by_tol(monkeypatch, tmp_path):
    (tmp_path / "board.kicad_pcb").write_text("(kicad_pcb)")
    (tmp_path / "PROJ_autoplacer.json").write_text(
        json.dumps({"component_zones": {"J2": {"edge": "right"}, "J1": {"edge": "left"}}})
    )
    captured = {}

    def _fake_gaps(path, zones, *, inboard_tol_mm):
        captured["zones"] = zones
        return [
            EdgeGap("J2", "right", -5.0, False),  # stranded
            EdgeGap("J1", "left", 0.0, True),  # flush -> not stranded
        ]

    monkeypatch.setattr(
        "kicraft.autoplacer.brain.connector_edge_gap.connector_edge_gaps", _fake_gaps
    )
    refs = cli_app._connector_stranded_refs(tmp_path / "board.kicad_pcb")
    assert refs == ["connector_stranded:J2@-5.00mm(right)"]
    # zones were discovered from the sibling *_autoplacer.json
    assert set(captured["zones"]) == {"J2", "J1"}


def test_stranded_refs_swallows_errors(monkeypatch, tmp_path):
    (tmp_path / "board.kicad_pcb").write_text("(kicad_pcb)")
    (tmp_path / "PROJ_autoplacer.json").write_text(
        json.dumps({"component_zones": {"J2": {"edge": "right"}}})
    )

    def _boom(*_a, **_k):
        raise RuntimeError("pcbnew load hiccup")

    monkeypatch.setattr(
        "kicraft.autoplacer.brain.connector_edge_gap.connector_edge_gaps", _boom
    )
    # A pcbnew hiccup must not invent a fab-readiness failure.
    assert cli_app._connector_stranded_refs(tmp_path / "board.kicad_pcb") == []


# --- courtyard defects join stranding as promotable (2026-07-19 review §2.1)


def test_promote_courtyard_only_clean():
    # Sole defect is a gross courtyard overlap on an electrically complete
    # board -> promote for inspection (NOT fab-ready), never rc=6.
    assert _promotable_strand_only(
        ["courtyards_overlap"], [], {"shorts": 0, "unconnected": 0}
    )


def test_promote_courtyard_unmeasured_only():
    assert _promotable_strand_only(["courtyard_unmeasured"], [], {})


def test_promote_courtyard_plus_strand():
    reasons = ["courtyards_overlap", *STRAND]
    assert _promotable_strand_only(reasons, STRAND, {"shorts": 0})


def test_reject_courtyard_with_unconnected():
    reasons = ["courtyards_overlap", "unconnected_nets"]
    assert not _promotable_strand_only(reasons, [], {"unconnected": 2})


def test_reject_courtyard_with_shorts_in_drc():
    assert not _promotable_strand_only(["courtyards_overlap"], [], {"shorts": 1})


def test_empty_reasons_never_promote():
    assert not _promotable_strand_only([], [], {})
