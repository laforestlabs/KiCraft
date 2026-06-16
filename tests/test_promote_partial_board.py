"""On a build that places + composes but never routes the parent (rc6), the
promote tail must still surface the richest board the engine produced -- the
composed, placed parent (which carries the leaf-level routing), or failing that
a single placed/routed leaf -- so the project preview shows the real board
instead of the raw, uncomposed scatter board.

Regression for KC-NESCCB (a 5x10 1515 RGB LED matrix): the leaf placed a
correct 3mm grid and routed ~99% of it, but because no round passed the gate
the canonical <stem>.kicad_pcb was left as the raw synth board (0 traces, parts
scattered), so the UI looked like placement never ran. These tests pin the
artifact-precedence the promote tail relies on."""
from __future__ import annotations

from pathlib import Path

from kicraft.design import cli_app


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("(kicad_pcb)\n")
    return p


def test_routed_parent_wins_when_present(tmp_path):
    sub = tmp_path / ".experiments" / "subcircuits" / "subcircuit__abc"
    _touch(sub / "parent_pre_freerouting.kicad_pcb")
    routed = _touch(sub / "parent_routed.kicad_pcb")
    assert cli_app._find_routed_parent(tmp_path) == routed
    # placed-parent prefers the routed board too (it is "more placed").
    assert cli_app._find_placed_parent(tmp_path) == routed


def test_placed_parent_is_the_rc6_fallback(tmp_path):
    """No parent_routed -> the composed pre-freerouting parent is the best board."""
    sub = tmp_path / ".experiments" / "subcircuits" / "subcircuit__abc"
    placed = _touch(sub / "parent_pre_freerouting.kicad_pcb")
    assert cli_app._find_routed_parent(tmp_path) is None
    assert cli_app._find_placed_parent(tmp_path) == placed


def test_best_leaf_board_when_no_parent_at_all(tmp_path):
    """Compose produced no parent board -> a routed leaf beats a placed leaf,
    and either beats the raw scatter board (None means 'leave as-is')."""
    leaf_a = tmp_path / ".experiments" / "subcircuits" / "leafA"
    leaf_b = tmp_path / ".experiments" / "subcircuits" / "leafB"
    _touch(leaf_a / "leaf_pre_freerouting.kicad_pcb")
    routed = _touch(leaf_b / "leaf_routed.kicad_pcb")
    assert cli_app._find_placed_parent(tmp_path) is None
    # routed leaf is preferred over a merely-placed one.
    assert cli_app._find_best_leaf_board(tmp_path) == routed


def test_no_artifacts_yields_none(tmp_path):
    """Nothing produced -> finders return None; the promote tail then leaves the
    raw board untouched (it has nothing better to show)."""
    assert cli_app._find_routed_parent(tmp_path) is None
    assert cli_app._find_placed_parent(tmp_path) is None
    assert cli_app._find_best_leaf_board(tmp_path) is None


# --- completeness gate: expected BOM refs vs footprints on the routed board ---

def test_missing_component_refs_flags_dropped_parts():
    expected = {"U1", "R1", "R2", "C1", "J1"}
    on_board = ["U1", "R1", "C1", "J1"]  # R2 silently dropped
    assert cli_app._missing_component_refs(expected, on_board) == ["R2"]


def test_missing_component_refs_clean_board():
    expected = {"U1", "R1", "C1"}
    # extra footprints on the board (fiducials, logos) are fine
    on_board = ["U1", "R1", "C1", "FID1"]
    assert cli_app._missing_component_refs(expected, on_board) == []


def test_missing_component_refs_unknown_board_refs_never_fires():
    # Empty/unknown board refs (count failure) must not flag every part as
    # missing -- the empty_board gate handles the truly-empty case instead.
    assert cli_app._missing_component_refs({"U1", "R1"}, []) == []
    assert cli_app._missing_component_refs({"U1", "R1"}, None) == []
