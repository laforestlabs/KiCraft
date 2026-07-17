from __future__ import annotations

from collections import Counter

import pytest

from kicraft.autoplacer import freerouting_runner


def test_run_kicad_cli_drc_counts_tracks_crossing_as_short(monkeypatch, tmp_path):
    """WS7: a tracks_crossing (two DIFFERENT-net tracks physically crossing) is
    a genuine short and must gate fab acceptance -- rounded-c3-devboard shipped
    past shorts=0 with a real GND-over-TXD0 crossing before this."""
    report = (
        "[tracks_crossing]: Tracks crossing @(167.41 mm, 98.84 mm): "
        "[Net 1](GND) [Net 2](TXD0)\n"
    )

    class _FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, *args, **kwargs):
        # kicad-cli writes the DRC report to the path following -o; emulate that
        # so the real parse path (including the new tracks_crossing branch) runs.
        out_path = cmd[cmd.index("-o") + 1]
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        return _FakeResult()

    monkeypatch.setattr(freerouting_runner.subprocess, "run", _fake_run)

    counts = freerouting_runner._run_kicad_cli_drc(str(tmp_path / "board.kicad_pcb"))
    assert counts["tracks_crossing"] == 1
    assert counts["shorts"] == 1


def test_run_kicad_cli_drc_positions_from_continuation_lines(monkeypatch, tmp_path):
    """Real KiCad reports are block-oriented: the [type] header carries the
    rule text while indented continuation lines carry @(x mm, y mm) item
    positions and [Net N](NAME) refs. The parser must complete each
    violation from its block -- positions were always None before, which
    left the manual-layout canvas with no markers to draw."""
    report = (
        "[shorting_items]: Items shorting two nets (nets SIG1 and SIG2)\n"
        "    Rule: board setup constraints clearance; error\n"
        "    @(120.5000 mm, 80.2500 mm): Track [Net 3](SIG1) on F.Cu\n"
        "    @(121.0000 mm, 80.5000 mm): Track [Net 4](SIG2) on F.Cu\n"
        "[courtyards_overlap]: Courtyards overlap\n"
        "    ; error\n"
        "    @(137.4412 mm, 134.0921 mm): Footprint K1 courtyard\n"
    )

    class _FakeResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, *args, **kwargs):
        out_path = cmd[cmd.index("-o") + 1]
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        return _FakeResult()

    monkeypatch.setattr(freerouting_runner.subprocess, "run", _fake_run)

    counts = freerouting_runner._run_kicad_cli_drc(str(tmp_path / "board.kicad_pcb"))
    short, courtyard = counts["violations"]
    # First continuation position wins (the primary offending item).
    assert (short["x_mm"], short["y_mm"]) == (120.5, 80.25)
    assert (short["net1"], short["net2"]) == ("SIG1", "SIG2")
    assert (courtyard["x_mm"], courtyard["y_mm"]) == (137.4412, 134.0921)
    assert counts["courtyard"] == 1


def test_extract_clearance_footprint_refs_counts_refs_within_clearance_blocks():
    report = """
[clearance]: Clearance violation
    @(6.1300 mm, 7.6000 mm): PTH pad A1 [GND] of J1
    @(6.1300 mm, 6.7500 mm): PTH pad A4 [VBUS] of J1
[silk_overlap]: Silkscreen overlap
    @(10.0000 mm, 10.0000 mm): Reference field of R1
[clearance]: Clearance violation
    @(4.7800 mm, 2.5000 mm): PTH pad B4 [VBUS] of J1
    @(4.7800 mm, 3.3500 mm): PTH pad B5 [CC2] of J1
"""

    refs = freerouting_runner._extract_clearance_footprint_refs(report)

    assert refs == Counter({"J1": 4})


def test_extract_violation_footprint_refs_filters_by_violation_type():
    report = """
[copper_edge_clearance]: Board edge clearance violation
    @(1.7700 mm, 0.3000 mm): PTH pad S1 [GND] of J1
[clearance]: Clearance violation
    @(4.0000 mm, 4.0000 mm): Pad 1 [GND] of C1
    @(4.1000 mm, 4.1000 mm): Pad 2 [VBUS] of C1
"""

    refs = freerouting_runner._extract_violation_footprint_refs(
        report,
        {"copper_edge_clearance"},
    )

    assert refs == Counter({"J1": 1})


def test_validate_routed_board_marks_single_footprint_clearance_as_internal(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        freerouting_runner,
        "count_board_tracks",
        lambda _path: {"traces": 0, "vias": 0, "total_length_mm": 0.0},
    )
    monkeypatch.setattr(
        freerouting_runner,
        "_run_kicad_cli_drc",
        lambda _path, timeout_s=30: {
            "report_text": """
[clearance]: Clearance violation
    @(6.1300 mm, 7.6000 mm): PTH pad A1 [GND] of J1
    @(6.1300 mm, 6.7500 mm): PTH pad A4 [VBUS] of J1
[clearance]: Clearance violation
    @(4.7800 mm, 2.5000 mm): PTH pad B4 [VBUS] of J1
    @(4.7800 mm, 3.3500 mm): PTH pad B5 [CC2] of J1
""",
            "violations": [
                {"type": "clearance", "description": "[clearance]: Clearance violation"},
                {"type": "clearance", "description": "[clearance]: Clearance violation"},
            ],
            "clearance": 2,
            "copper_edge_clearance": 0,
            "shorts": 0,
            "timed_out": False,
            "missing_cli": False,
        },
    )

    board_path = tmp_path / "fake_board.kicad_pcb"
    board_path.write_text("stub", encoding="utf-8")

    validation = freerouting_runner.validate_routed_board(str(board_path))

    assert validation["obviously_illegal_routed_geometry"] is False
    assert validation["footprint_internal_clearance_count"] == 2
    assert validation["drc"]["clearance_footprint_refs"] == ["J1"]


def test_validate_routed_board_rejects_trackless_ride_along_clearance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    # One genuinely footprint-internal violation must NOT waive ref-less
    # track-to-track violations that ride along in the same report (the
    # old aggregate-refs waiver let a broken board pass the gate).
    monkeypatch.setattr(
        freerouting_runner,
        "count_board_tracks",
        lambda _path: {"traces": 0, "vias": 0, "total_length_mm": 0.0},
    )
    monkeypatch.setattr(
        freerouting_runner,
        "_run_kicad_cli_drc",
        lambda _path, timeout_s=30: {
            "report_text": """
[clearance]: Clearance violation
    @(6.1300 mm, 7.6000 mm): PTH pad A1 [GND] of J1
    @(6.1300 mm, 6.7500 mm): PTH pad A4 [VBUS] of J1
[clearance]: Clearance violation
    @(9.0000 mm, 4.0000 mm): Track [VSEL0] on F.Cu
    @(9.1000 mm, 4.1000 mm): Track [GND] on F.Cu
""",
            "violations": [
                {"type": "clearance", "description": "[clearance]: Clearance violation"},
                {"type": "clearance", "description": "[clearance]: Clearance violation"},
            ],
            "clearance": 2,
            "copper_edge_clearance": 0,
            "shorts": 0,
            "timed_out": False,
            "missing_cli": False,
        },
    )

    board_path = tmp_path / "fake_board.kicad_pcb"
    board_path.write_text("stub", encoding="utf-8")

    validation = freerouting_runner.validate_routed_board(str(board_path))

    assert validation["obviously_illegal_routed_geometry"] is True
    assert validation["footprint_internal_clearance_count"] == 1


def test_classify_clearance_violations_per_block():
    report = """
[clearance]: Clearance violation
    @(1.0 mm, 1.0 mm): Pad 1 [GND] of C1
    @(1.1 mm, 1.1 mm): Pad 2 [VBUS] of C1
[clearance]: Clearance violation
    @(2.0 mm, 2.0 mm): Pad 1 [GND] of C1
    @(2.1 mm, 2.1 mm): Pad 3 [SIG] of R5
[hole_clearance]: Hole clearance violation
    @(3.0 mm, 3.0 mm): Track [SIG] on F.Cu
    @(3.1 mm, 3.1 mm): PTH pad 2 [SIG] of J1
[silk_overlap]: Silkscreen overlap
    @(4.0 mm, 4.0 mm): Reference field of R1
"""
    # C1-internal waived; C1-vs-R5 (two footprints) genuine; track-vs-pad
    # (ref-less item) genuine; silk block ignored entirely.
    verdict = freerouting_runner._classify_clearance_violations(report)
    assert verdict == {"waived": 1, "genuine": 2}

    # The ignorable escape hatch waives fully-named multi-footprint blocks
    # but never blocks containing ref-less (routed copper) items.
    verdict = freerouting_runner._classify_clearance_violations(
        report, ignorable_refs={"C1", "R5", "J1"}
    )
    assert verdict == {"waived": 2, "genuine": 1}


_EDGE_CONN_DRC = {
    "report_text": """
[copper_edge_clearance]: Board edge clearance violation
    @(0.0000 mm, 0.0000 mm): Segment on Edge.Cuts
    @(1.7700 mm, 0.3000 mm): PTH pad S1 [GND] of J1
""",
    "violations": [
        {
            "type": "copper_edge_clearance",
            "description": "[copper_edge_clearance]: Board edge clearance violation",
        }
    ],
    "clearance": 0,
    "copper_edge_clearance": 1,
    "shorts": 0,
    "timed_out": False,
    "missing_cli": False,
}


def _patch_edge_conn_drc(monkeypatch):
    monkeypatch.setattr(
        freerouting_runner,
        "count_board_tracks",
        lambda _path: {"traces": 0, "vias": 0, "total_length_mm": 0.0},
    )
    monkeypatch.setattr(
        freerouting_runner,
        "_run_kicad_cli_drc",
        lambda _path, timeout_s=30: dict(_EDGE_CONN_DRC),
    )


def test_validate_routed_board_flags_edge_connector_copper_edge_clearance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    # The blanket edge-connector copper_edge waiver was REMOVED: the composer now
    # keeps the board edge a clearance outboard of a flush connector's pads
    # (connector_edge_pad_clearance_mm in _repair_parent_outline), so a genuine
    # pad-to-edge violation is no longer masked at the gate -- it fails loudly.
    _patch_edge_conn_drc(monkeypatch)
    board_path = tmp_path / "fake_board.kicad_pcb"
    board_path.write_text("stub", encoding="utf-8")

    validation = freerouting_runner.validate_routed_board(
        str(board_path),
        cfg={"component_zones": {"J1": {"edge": "left"}}},
    )

    assert validation["obviously_illegal_routed_geometry"] is True
    assert "footprint_internal_copper_edge_count" not in validation
    assert validation["drc"]["copper_edge_footprint_refs"] == ["J1"]


def test_validate_routed_board_waives_ignorable_copper_edge(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    # The explicit per-board ignorable_footprint_refs escape hatch still waives.
    _patch_edge_conn_drc(monkeypatch)
    board_path = tmp_path / "fake_board.kicad_pcb"
    board_path.write_text("stub", encoding="utf-8")

    validation = freerouting_runner.validate_routed_board(
        str(board_path),
        cfg={"ignorable_footprint_refs": ["J1"]},
    )

    assert validation["obviously_illegal_routed_geometry"] is False
    assert validation["footprint_internal_copper_edge_count"] == 1


def _clean_drc(_path, timeout_s=30):
    return {"report_text": "", "violations": [], "clearance": 0,
            "copper_edge_clearance": 0, "shorts": 0, "unconnected": 0,
            "timed_out": False, "missing_cli": False}


def test_validate_routed_board_rejects_empty_board(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    # A board with zero footprints has no shorts and no ratsnest, so a clean
    # DRC would otherwise accept it. The empty_board guard must reject it.
    monkeypatch.setattr(
        freerouting_runner, "count_board_tracks",
        lambda _p: {"traces": 0, "vias": 0, "total_length_mm": 0.0,
                    "footprints": 0, "pads": 0, "footprint_refs": []},
    )
    monkeypatch.setattr(freerouting_runner, "_run_kicad_cli_drc", _clean_drc)
    board_path = tmp_path / "empty.kicad_pcb"
    board_path.write_text("stub", encoding="utf-8")

    v = freerouting_runner.validate_routed_board(str(board_path))
    assert v["accepted"] is False
    assert "empty_board" in v["rejection_reasons"]


def test_validate_routed_board_accepts_populated_clean_board(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    # The empty_board guard must be a no-op for a real, populated board.
    monkeypatch.setattr(
        freerouting_runner, "count_board_tracks",
        lambda _p: {"traces": 12, "vias": 2, "total_length_mm": 80.0,
                    "footprints": 3, "pads": 9, "footprint_refs": ["R1", "C1", "U1"]},
    )
    monkeypatch.setattr(freerouting_runner, "_run_kicad_cli_drc", _clean_drc)
    board_path = tmp_path / "ok.kicad_pcb"
    board_path.write_text("stub", encoding="utf-8")

    v = freerouting_runner.validate_routed_board(str(board_path))
    assert v["accepted"] is True
    assert "empty_board" not in v["rejection_reasons"]


def test_validate_routed_board_unknown_footprint_count_not_empty(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    # A count-subprocess failure reports footprints=-1 (unknown); that must NOT
    # be misread as an empty board.
    monkeypatch.setattr(
        freerouting_runner, "count_board_tracks",
        lambda _p: {"traces": 0, "vias": 0, "total_length_mm": 0.0,
                    "footprints": -1, "pads": -1, "footprint_refs": []},
    )
    monkeypatch.setattr(freerouting_runner, "_run_kicad_cli_drc", _clean_drc)
    board_path = tmp_path / "unknown.kicad_pcb"
    board_path.write_text("stub", encoding="utf-8")

    v = freerouting_runner.validate_routed_board(str(board_path))
    assert "empty_board" not in v["rejection_reasons"]


def test_run_pcbnew_script_retries_transient_failed_to_load_board(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []

    def _fake_run(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            return type(
                "Result",
                (),
                {"returncode": 1, "stderr": "RuntimeError: Failed to load board: /tmp/foo.kicad_pcb\n"},
            )()
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(freerouting_runner.subprocess, "run", _fake_run)
    monkeypatch.setattr(freerouting_runner.time, "sleep", lambda _s: None)

    freerouting_runner._run_pcbnew_script("print('ok')")

    assert len(calls) == 3


def test_run_pcbnew_script_retries_up_to_six_attempts(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []

    def _fake_run(*args, **kwargs):
        calls.append(1)
        if len(calls) < 6:
            return type(
                "Result",
                (),
                {"returncode": 1, "stderr": "RuntimeError: Failed to load board: /tmp/foo.kicad_pcb\n"},
            )()
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(freerouting_runner.subprocess, "run", _fake_run)
    monkeypatch.setattr(freerouting_runner.time, "sleep", lambda _s: None)

    freerouting_runner._run_pcbnew_script("print('ok')")

    assert len(calls) == 6


def test_run_pcbnew_script_gives_up_after_six_failed_load_board(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[int] = []

    def _fake_run(*args, **kwargs):
        calls.append(1)
        return type(
            "Result",
            (),
            {
                "returncode": 1,
                "stderr": "RuntimeError: Failed to load board: /tmp/foo.kicad_pcb\n",
            },
        )()

    monkeypatch.setattr(freerouting_runner.subprocess, "run", _fake_run)
    monkeypatch.setattr(freerouting_runner.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError, match="Failed to load board"):
        freerouting_runner._run_pcbnew_script("print('ok')")

    assert len(calls) == 6


# ---------------------------------------------------------------------------
# DSN non-ANSI sanitization (FreeRouting 1.9.0 hangs forever on non-ANSI
# input -- run_01 rc-lowpass-bnc: 'Ω' in resistor values stalled every round)
# ---------------------------------------------------------------------------

_DSN_WITH_OMEGA = """(pcb board
  (placement
    (component R_0402 (place R1 62400 -30000 front 90 (PN 10kΩ)))
    (component BNC (place J2 65540 -14262 front 90 (PN "BNC 50Ω")))
    (component C_0402 (place C1 50000 -20000 front 0 (PN 10nF)))
    (component POT (place RV1 46795 -22740 front -90 (PN 100kµF)))
  )
  (network (net VOUT (pins R1-2 C1-1)))
)
"""


def test_sanitize_dsn_pn_transliterates_non_ansi(tmp_path):
    p = tmp_path / "board.dsn"
    p.write_text(_DSN_WITH_OMEGA, encoding="utf-8")
    n = freerouting_runner._sanitize_dsn_part_numbers(str(p))
    assert n == 3
    text = p.read_text(encoding="utf-8")
    assert "(PN 10kOhm)" in text                # GREEK CAPITAL OMEGA
    assert '(PN "BNC 50Ohm")' in text           # OHM SIGN, quoted PN
    assert "(PN 100kuF)" in text                # MICRO SIGN
    assert "(PN 10nF)" in text                  # untouched ASCII PN
    assert not any(ord(c) >= 128 for c in text)


def test_sanitize_dsn_pn_noop_on_clean_input(tmp_path):
    p = tmp_path / "board.dsn"
    clean = _DSN_WITH_OMEGA.replace("Ω", "R").replace(
        "Ω", "R").replace("µ", "u")
    p.write_text(clean, encoding="utf-8")
    before = p.stat().st_mtime_ns
    assert freerouting_runner._sanitize_dsn_part_numbers(str(p)) == 0
    assert p.stat().st_mtime_ns == before  # not rewritten


def test_sanitize_dsn_warns_on_non_ansi_net_name(tmp_path, capsys):
    # A non-ASCII NET name round-trips through the SES and cannot be
    # rewritten here -- it must be surfaced loudly, not silently mangled.
    p = tmp_path / "board.dsn"
    p.write_text(
        "(pcb (network (net VOUT_50Ω (pins R1-2))))\n", encoding="utf-8"
    )
    assert freerouting_runner._sanitize_dsn_part_numbers(str(p)) == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and "non-ANSI" in out
