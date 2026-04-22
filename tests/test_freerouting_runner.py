from __future__ import annotations

from collections import Counter

import pytest

from kicraft.autoplacer import freerouting_runner


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


def test_validate_routed_board_ignores_edge_connector_copper_edge_clearance(
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
        },
    )

    board_path = tmp_path / "fake_board.kicad_pcb"
    board_path.write_text("stub", encoding="utf-8")

    validation = freerouting_runner.validate_routed_board(
        str(board_path),
        cfg={
            "component_zones": {"J1": {"edge": "left"}},
        },
    )

    assert validation["obviously_illegal_routed_geometry"] is False
    assert validation["footprint_internal_copper_edge_count"] == 1
    assert validation["drc"]["copper_edge_footprint_refs"] == ["J1"]


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
