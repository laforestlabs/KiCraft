"""PR3 promote gate: `_check_form_factor_conformance` on the routed board.

A validated standard is a hard mechanical contract. When enforcement placed the
board a non-conformant result must fail the fab gate; with enforcement off the
board is free-placed by design, so it is only an advisory. No standard requested
-> nothing to check (None)."""

from __future__ import annotations

from pathlib import Path

from kicraft.design import cli_app
from kicraft.design.models import ConversationState, FormFactor, IntentSlot
from kicraft.form_factors import get_template
from kicraft.form_factors.conformance import expected_pins


def _state(standard: str | None) -> ConversationState:
    ff = FormFactor(shape="rect", standard=standard) if standard else FormFactor()
    return ConversationState(project_stem="X", intent=IntentSlot(goal="g", form_factor=ff))


def _conformant_pcb(tmp_path: Path) -> Path:
    """A minimal board whose Edge.Cuts is the template rect and whose pads sit at
    every template pin position (unrotated -> pin coords are already world)."""
    t = get_template("arduino_uno_shield")
    w, h = t.board_width_mm, t.board_height_mm
    lines = [
        "(kicad_pcb",
        f'  (gr_line (start 0 0) (end {w} 0) (layer "Edge.Cuts"))',
        f'  (gr_line (start 0 0) (end 0 {h}) (layer "Edge.Cuts"))',
        f'  (gr_line (start {w} 0) (end {w} {h}) (layer "Edge.Cuts"))',
        f'  (gr_line (start 0 {h}) (end {w} {h}) (layer "Edge.Cuts"))',
    ]
    for i, (_net, x, y) in enumerate(expected_pins(t)):
        lines.append(
            f'  (footprint "p{i}" (at {x} {y})'
            f' (pad "1" thru_hole circle (at 0 0) (size 1 1)))'
        )
    lines.append(")")
    pcb = tmp_path / "conf.kicad_pcb"
    pcb.write_text("\n".join(lines))
    return pcb


def _free_pcb(tmp_path: Path) -> Path:
    pcb = tmp_path / "free.kicad_pcb"
    pcb.write_text(
        '(kicad_pcb\n'
        '  (gr_line (start 0 0) (end 120 0) (layer "Edge.Cuts"))\n'
        '  (gr_line (start 0 0) (end 0 45) (layer "Edge.Cuts"))\n'
        '  (gr_line (start 120 0) (end 120 45) (layer "Edge.Cuts"))\n'
        '  (gr_line (start 0 45) (end 120 45) (layer "Edge.Cuts"))\n'
        '  (footprint "j" (at 5 5) (pad "1" thru_hole circle (at 0 0) (size 1 1)))\n'
        ')\n'
    )
    return pcb


def test_none_when_no_standard(tmp_path):
    assert cli_app._check_form_factor_conformance(_state(None), _free_pcb(tmp_path)) is None


def test_conformant_board_passes(tmp_path):
    res = cli_app._check_form_factor_conformance(
        _state("arduino_uno_shield"), _conformant_pcb(tmp_path)
    )
    assert res is not None and res["conformant"] is True


def test_nonconformant_fails_gate_when_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_FORM_FACTOR_ENFORCE", "1")
    res = cli_app._check_form_factor_conformance(
        _state("arduino_uno_shield"), _free_pcb(tmp_path)
    )
    assert res == {"conformant": False, "enforced": True, "summary": res["summary"]}
    assert res["enforced"] is True and res["conformant"] is False


def test_nonconformant_is_advisory_when_enforcement_off(tmp_path, monkeypatch):
    monkeypatch.delenv("KICRAFT_FORM_FACTOR_ENFORCE", raising=False)
    res = cli_app._check_form_factor_conformance(
        _state("arduino_uno_shield"), _free_pcb(tmp_path)
    )
    assert res["conformant"] is False and res["enforced"] is False
