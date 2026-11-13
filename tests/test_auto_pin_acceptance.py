"""Auto-pin must respect leaf acceptance, and squeeze rejections must roll back.

Regression cluster from KC-FGRSQF (projects/1/559): the solver's per-round
acceptance gate correctly rejected a high-scoring round whose four same-edge
screw terminals overlapped (its own validation recorded ``drc.courtyard=3``),
and the winner selection respected that -- but ``_auto_pin_best_leaves``
re-selected "best" by a bare ``(routed, score)`` key, pinned the rejected
round's snapshot over the canonical artifacts, and the composer shipped the
never-validated geometry straight into a parent ``courtyards_overlap`` rc7.

Companion hole: a REJECTED size-reduction candidate's reroute leaves its own
board bytes on the winner's canonical + round-snapshot paths; those must be
restored so later consumers read the geometry that was actually validated.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from kicraft.autoplacer.brain import pins
from kicraft.cli.autoexperiment import _auto_pin_best_leaves
from kicraft.cli.triage import collect_leaves


def _validation(courtyard: int, unconnected: int = 0) -> dict:
    """A minimal validation dict that passes every default acceptance gate
    when clean; ``courtyard > 0`` with an unmeasurable board path makes the
    ``no_gross_courtyard_overlap`` gate fail conservatively."""
    return {
        "accepted": True,
        "board_exists": True,
        "board_path": "/nonexistent/board.kicad_pcb",
        "python_exception": False,
        "malformed_board_geometry": False,
        "obviously_illegal_routed_geometry": False,
        "rejection_reasons": [],
        "drc": {
            "ran": True,
            "shorts": 0,
            "unconnected": unconnected,
            "clearance": 0,
            "courtyard": courtyard,
            "annular_width": 0,
            "padstack": 0,
            "copper_edge_clearance": 0,
            "total": courtyard + unconnected,
        },
    }


def _round(
    idx: int, score: float, courtyard: int, unconnected: int = 0
) -> dict:
    accepted = courtyard == 0 and unconnected == 0
    return {
        "round_index": idx,
        "routed": True,
        "score": score,
        "routing": {
            "validation": _validation(courtyard, unconnected),
            "_round_acceptance": {
                "accepted": accepted,
                "rejection_reasons": [] if accepted else ["leaf_gate_failed"],
                "gate_results": {
                    "no_unconnected": {
                        "passed": unconnected == 0,
                        "unconnected_total": unconnected,
                    }
                },
                "drc_summary": {"unconnected": unconnected},
                "notes": [],
            },
        },
    }


def _leaf_fixture(project_dir: Path, rounds: list[dict]) -> Path:
    """Build the minimal on-disk leaf a pin needs: debug.json with the given
    rounds, canonical metadata/solved_layout, and a complete round_NNNN_*
    snapshot triple per round."""
    leaf = project_dir / ".experiments" / "subcircuits" / "leafA__deadbeef01"
    leaf.mkdir(parents=True)
    (leaf / "metadata.json").write_text(json.dumps({"schema_version": "leaf-v1"}))
    (leaf / "solved_layout.json").write_text(
        json.dumps({"config_hash": json.dumps({})})
    )
    (leaf / "debug.json").write_text(json.dumps({"extra": {"all_rounds": rounds}}))
    for r in rounds:
        token = f"{int(r['round_index']):04d}"
        (leaf / f"round_{token}_leaf_routed.kicad_pcb").write_text(
            f"(kicad_pcb round {token})"
        )
        (leaf / f"round_{token}_metadata.json").write_text("{}")
        # These emulate pre-custody snapshots: their raw DRC evidence is
        # present, while the contemporaneous gate verdict remains in
        # all_rounds. pinning must join those same-round records, not use a
        # later canonical debug result.
        (leaf / f"round_{token}_solved_layout.json").write_text(
            json.dumps({"validation": r["routing"]["validation"], "components": {}})
        )
    return leaf


def test_auto_pin_prefers_accepted_round_over_higher_score(tmp_path, capsys):
    # Round 13 scores highest but its validation carries courtyard=3 (the
    # KC-FGRSQF signature); round 35 is clean. The pin must go to 35.
    _leaf_fixture(tmp_path, [_round(13, 88.0, courtyard=3), _round(35, 84.3, 0)])

    _auto_pin_best_leaves(tmp_path)

    pins = json.loads((tmp_path / ".experiments" / "pins.json").read_text())
    entry = pins["pinned_leaves"]["leafA__deadbeef01"]
    assert entry["round"] == 35, entry
    # The pinned snapshot was applied over the canonical board.
    canonical = (
        tmp_path / ".experiments" / "subcircuits" / "leafA__deadbeef01"
        / "leaf_routed.kicad_pcb"
    )
    assert canonical.read_text() == "(kicad_pcb round 0035)"


def test_auto_pin_falls_back_loudly_when_no_round_accepted(tmp_path, capsys):
    _leaf_fixture(tmp_path, [_round(3, 70.0, courtyard=2), _round(5, 90.0, 1)])

    _auto_pin_best_leaves(tmp_path)

    pins = json.loads((tmp_path / ".experiments" / "pins.json").read_text())
    entry = pins["pinned_leaves"]["leafA__deadbeef01"]
    # Falls back to the best routed round by score...
    assert entry["round"] == 5, entry
    # ...but says so loudly instead of pinning silently.
    out = capsys.readouterr().out
    assert "NO round passed leaf acceptance" in out
    assert "UNACCEPTED" in out


def test_pinned_debug_and_triage_follow_older_selected_round(tmp_path):
    # 9102's witness shape: the later invocation leaves R5 in debug, but the
    # all-round selector pins older R2 because its routed score is better.
    rounds = [
        _round(2, 12.43, courtyard=0, unconnected=6),
        _round(5, 0.0, courtyard=0, unconnected=9),
    ]
    leaf = _leaf_fixture(tmp_path, rounds)

    _auto_pin_best_leaves(tmp_path)
    assert (leaf / "leaf_routed.kicad_pcb").read_text() == "(kicad_pcb round 0002)"
    canonical_layout = json.loads((leaf / "solved_layout.json").read_text())
    assert canonical_layout["validation"]["drc"]["unconnected"] == 6


    debug = json.loads((leaf / "debug.json").read_text())
    extra = debug["extra"]
    assert extra["best_round"]["round_index"] == 2
    assert extra["leaf_acceptance_structured"]["accepted"] is False
    assert extra["leaf_acceptance"]["drc"]["unconnected"] == 6
    # Pin reconciliation leaves the all-round search history intact.
    assert [r["round_index"] for r in extra["all_rounds"]] == [2, 5]

    # Retained artifacts from before diagnostic custody still have R5 in
    # debug.json. Triage resolves their pins in memory rather than reporting
    # that stale invocation.
    (leaf / "debug.json").write_text(
        json.dumps(
            {
                "extra": {
                    "all_rounds": rounds,
                    "best_round": rounds[1],
                    "leaf_acceptance": rounds[1]["routing"]["validation"],
                    "leaf_acceptance_structured": rounds[1]["routing"][
                        "_round_acceptance"
                    ],
                }
            }
        )
    )
    leaves = collect_leaves(tmp_path / ".experiments")
    assert len(leaves) == 1
    assert leaves[0]["accepted"] is False
    assert leaves[0]["unconnected"]["total"] == 6


def test_pinned_debug_fails_closed_without_same_round_gate_record(tmp_path):
    round_record = _round(2, 12.43, courtyard=0, unconnected=0)
    del round_record["routing"]["_round_acceptance"]
    leaf = _leaf_fixture(tmp_path, [round_record])

    pins.pin_leaf(tmp_path / ".experiments", leaf.name, 2)

    debug = json.loads((leaf / "debug.json").read_text())
    acceptance = debug["extra"]["leaf_acceptance_structured"]
    assert acceptance["accepted"] is False
    assert acceptance["rejection_reasons"] == [
        "pinned_round_acceptance_unavailable"
    ]


@pytest.mark.parametrize("raw_count", [6, 0, None])
def test_selected_raw_drc_does_not_manufacture_acceptance(tmp_path, raw_count):
    selected = _round(2, 12.43, courtyard=0, unconnected=6)
    del selected["routing"]["_round_acceptance"]
    if raw_count is None:
        del selected["routing"]["validation"]["drc"]
    else:
        selected["routing"]["validation"]["drc"].update({
            "unconnected": raw_count, "unconnected_nets": ["QSPI_CS", "QSPI_SD2", "gpio7"],
        })
    stale = _round(5, 0.0, courtyard=0, unconnected=9)
    leaf = _leaf_fixture(tmp_path, [selected, stale])
    pins.pin_leaf(tmp_path / ".experiments", leaf.name, 2)
    report = collect_leaves(tmp_path / ".experiments")[0]
    assert report["accepted"] is False
    assert report["rejection_reasons"] == ["pinned_round_acceptance_unavailable"]
    assert report["unconnected"] is None
    if raw_count is None:
        assert report["raw_drc"] is None
    else:
        assert report["raw_drc"] == {
            "unconnected": raw_count, "unconnected_nets": ["QSPI_CS", "QSPI_SD2", "gpio7"],
        }


