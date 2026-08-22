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

from kicraft.autoplacer.brain.leaf_size_reduction import (
    _preserve_round_boards,
    _restore_round_boards,
)
from kicraft.cli.autoexperiment import _auto_pin_best_leaves


def _validation(courtyard: int) -> dict:
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
            "unconnected": 0,
            "clearance": 0,
            "courtyard": courtyard,
            "annular_width": 0,
            "padstack": 0,
            "copper_edge_clearance": 0,
            "total": courtyard,
        },
    }


def _round(idx: int, score: float, courtyard: int) -> dict:
    return {
        "round_index": idx,
        "routed": True,
        "score": score,
        "routing": {"validation": _validation(courtyard)},
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
        (leaf / f"round_{token}_solved_layout.json").write_text("{}")
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


def test_squeeze_preserve_restore_round_trips(tmp_path):
    routed = tmp_path / "leaf_routed.kicad_pcb"
    snapshot = tmp_path / "round_0035_leaf_routed.kicad_pcb"
    routed.write_bytes(b"winner canonical")
    snapshot.write_bytes(b"winner snapshot")
    routing = {
        "routed_board_path": str(routed),
        "round_board_routed": str(snapshot),
        "leaf_placed_board": str(tmp_path / "missing.kicad_pcb"),  # tolerated
    }

    saved = _preserve_round_boards(routing)
    assert set(saved) == {str(routed), str(snapshot)}

    # A rejected candidate's reroute clobbers both files...
    routed.write_bytes(b"rejected candidate canonical")
    snapshot.write_bytes(b"rejected candidate snapshot")

    # ...and the restore puts the validated winner back byte-for-byte.
    _restore_round_boards(saved)
    assert routed.read_bytes() == b"winner canonical"
    assert snapshot.read_bytes() == b"winner snapshot"
