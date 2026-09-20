"""Guards for the BLOCK-vs-RECORD bar (design-yield-recovery plan §4.3.1).

A downgrade is only safe while it is *recorded* and while the checks that stay blocking keep
blocking. These tests pin both halves: every RECORD class reaches the artifact and the
rubric's recording gate, and every code the plan keeps on the BLOCK side still raises.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from kicraft.design.advisories import advisory_code, advisory_line, recorded_advisory_codes
from kicraft.design.architecture_intent import ArchitectureIntentError, derive_architecture
from kicraft.eval.rubric import load_rubric
from kicraft.eval.scoring import eval_script_gates

REPO = Path(__file__).resolve().parents[1]
ARCHITECTURE_INTENT = REPO / "kicraft" / "design" / "architecture_intent.py"

# §4.3 A: the architecture codes the plan keeps on the BLOCK side. A silent move across the
# line fails here; moving one deliberately means editing this list in the same commit.
BLOCK_ARCHITECTURE_CODES = {
    "unknown_requirement_sheet",
    "unknown_signal_requirement",
    "malformed_signal_ref",
    "unknown_interface_port",
    "unknown_tie_net",
    "unknown_reference_domain",
    "unknown_supply_rail",
    "unknown_edge_rail",
    "duplicate_requirement_id",
    "duplicate_edge_connector",
    "derived_output_name_collision",
    "conflicting_port_binding",
    "conflicting_supply_binding",
    "conflicting_reference_binding",
    "declared_port_double_bound",
    "declared_signal_port_tied",
    "signal_conflicts_with_rail",
    "signal_names_rail",
    "reference_domain_not_zero_volt",
    "unsupported_standard_stacking_interface",
    "incomplete_standard_stacking_interface",
    "invalid_standard_stacking_owner",
    "invalid_standard_stacking_pinmap",
    "usb_connector_supply_unknown",
    "incomplete_usb_edge",
    "empty_edge_connector",
    "unsupported_lowerer_contract",
    "unbound_required_port",
    # §4.3 A lists this as RECORD; the boundary in §4.1 overrides the table, and it is
    # measured: downgraded, the requirement is dropped from the netlist entirely (no curated
    # recipe, no declared interface, no catalog) and the obligation it owned fails the
    # ownership check one validation later under a message that no longer names the cause.
    "unknown_part_refused",
}

# §4.3 A: the architecture code the plan reclassifies as RECORD, where the requirement IS
# carried (its family is curated, the chosen part is emitted, and the note names it).
RECORD_ARCHITECTURE_CODES = {"unreviewed_exact_part"}


def _code_literals(function: str) -> set[str]:
    """Every literal code the named module calls `function("...")` with."""
    tree = ast.parse(ARCHITECTURE_INTENT.read_text(encoding="utf-8"))
    codes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name == function and isinstance(node.args[0], ast.Constant):
            codes.add(str(node.args[0].value))
    return codes


def test_block_codes_still_raise_and_record_codes_no_longer_do():
    """The snapshot §4.3.1 asks for: neither list may move without this test.

    A code that silently crossed the line would be a downgrade nobody recorded (operator
    decision D2), so the module's own `_fail`/`_advise` literals are the contract.
    """
    failing = _code_literals("_fail")
    advising = _code_literals("_advise")
    assert BLOCK_ARCHITECTURE_CODES <= failing, sorted(BLOCK_ARCHITECTURE_CODES - failing)
    assert RECORD_ARCHITECTURE_CODES <= advising, sorted(RECORD_ARCHITECTURE_CODES - advising)
    assert not (RECORD_ARCHITECTURE_CODES & failing)


def _regulator_intent(
    *, declared: bool, exact_part: str | None, obligation: bool = True
) -> dict:
    """One 3.3 V regulator whose family the library does not curate.

    `declared` gives the part its own interface (so nothing else is missing); `exact_part`
    is what the advisory is about.
    """
    regulator: dict = {
        "id": "reg",
        "sheet": "POWER",
        "role": "regulator",
        "family": "air-quality-sensor",
        "supply": "+5V",
        "functional_blocks": ["POWER_DISTRIBUTION"],
    }
    if obligation:
        # The obligation is what makes the family match a curated recipe: without it the part
        # is genuinely uncurated, which is the case the advisory is about.
        regulator["obligations"] = [
            {
                "kind": "physical",
                "original_obligation_id": "regulator",
                "component_class": "voltage-regulator",
            }
        ]
    if exact_part is not None:
        regulator["exact_part"] = exact_part
    signals = [{"name": "RAIL_3V3", "from": "reg.output", "to": "edge:POWER_OUT"}]
    if declared:
        regulator["declared_ports"] = [
            {"key": "input", "direction": "power", "function": "5 V input", "supply_rail": "+5V"},
            {"key": "output", "direction": "power", "function": "3.3 V output"},
            {"key": "gnd", "direction": "power", "function": "ground", "reference_domain": "GND"},
        ]
        signals.append({"name": "GND", "from": "reg.gnd", "to": "input.pin2"})
    return {
        "topologies": {"POWER": "5 V input header"},
        "comms_protocols": [],
        "mcu_present": False,
        "power": {"rails": {"+5V": {"voltage": 5.0, "from": "input.pin1"}}},
        "sheets": [
            {
                "name": "INPUT",
                "stem": "INPUT",
                "role": "connector",
                "function": "5 V input header.",
            },
            {
                "name": "POWER",
                "stem": "POWER",
                "role": "regulator",
                "function": "Regulate 5 V to 3.3 V.",
            },
        ],
        "requirements": [
            {
                "id": "input",
                "sheet": "INPUT",
                "role": "connector",
                "family": "pin-header",
                "parameters": {"rows": 1, "gender": "male"},
                "ties": {"pin1": "+5V", "pin2": "GND"},
                "functional_blocks": ["POWER_IN"],
            },
            regulator,
        ],
        "signals": signals,
        "assumptions": [],
    }


def test_unreviewed_exact_part_is_recorded_on_the_artifact_not_refused():
    """RECORD class 1: the design compiles and the note rides on the architecture."""
    architecture = derive_architecture(
        _regulator_intent(declared=True, exact_part="AMS1117-3.3")
    )
    codes = [row.code for row in architecture.advisories]
    assert codes == ["unreviewed_exact_part"]
    assert "AMS1117-3.3" in " ".join(architecture.advisories[0].evidence)
    # Mirrored into the assumptions the review reads, and carried into the committed state.
    assert any("advisory [unreviewed_exact_part]" in line for line in architecture.assumptions)
    assert "unreviewed_exact_part" in json.dumps(architecture.model_dump(exclude_none=True))


def test_uncurated_interface_is_refused_because_the_part_would_not_be_carried():
    """§4.1's boundary over the RECORD table: a part nothing carries is not "unproven".

    Downgraded, this case does not ship a board with a note -- it drops the requirement (no
    curated recipe, no declared interface, so it has no catalog at all) and the obligation it
    owned then fails the ownership check one validation later. A part with nothing implementing
    it is a missing feature, so the refusal stands and names the repair.
    """
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(
            _regulator_intent(declared=False, exact_part="SEN5X", obligation=False)
        )
    diagnostics = excinfo.value.diagnostics
    refused = [row for row in diagnostics if row.code == "unknown_part_refused"]
    assert len(refused) == 1
    assert "declared_ports" in refused[0].message


def test_recorded_advisories_reach_the_rubric_gate_and_never_cap():
    """RECORD class 3: the count and codes are recorded, with no effective cap."""
    state = {
        "architecture": {"advisories": [{"code": "unreviewed_exact_part", "evidence": []}]},
        "bom": {"assumptions": [advisory_line("spec_named_mpn_substitution", "walked away from X")]},
    }
    assert recorded_advisory_codes(state["architecture"], state["bom"]) == [
        "unreviewed_exact_part",
        "spec_named_mpn_substitution",
    ]
    rubric = load_rubric()
    gate = next(g for g in rubric["gates"] if g["id"] == "shipped_with_advisories")
    assert gate["detected_by"] == "script"
    fired = eval_script_gates(
        {
            "erc": {"errors": 0},
            "transcript": {"present": True, "synth_attempts": 1},
            "generated": {"synthesized": True},
            "state": {"advisory_codes": ["unreviewed_exact_part", "spec_named_mpn_substitution"]},
        },
        rubric,
    )
    recorded = [row for row in fired if row["id"] == "shipped_with_advisories"]
    assert len(recorded) == 1
    assert recorded[0]["advisories"] == [
        "unreviewed_exact_part",
        "spec_named_mpn_substitution",
    ]
    # No number of advisories can cap a score: the gate's cap cannot bind (scores top out at 100).
    assert recorded[0]["cap"] >= 100


def test_advisory_lines_round_trip():
    line = advisory_line("spec_named_mpn_substitution", "asked for R1", ["R1: 10k -> 9k1"])
    assert line == (
        "advisory [spec_named_mpn_substitution]: asked for R1 -- R1: 10k -> 9k1"
    )
    assert advisory_code(line) == "spec_named_mpn_substitution"
    assert advisory_code("a plain assumption") is None
    assert advisory_code("advisory [unterminated") is None


def test_block_code_still_raises_end_to_end():
    """The BLOCK half, exercised rather than snapshotted: two nets on one pin still refuses."""
    intent = _regulator_intent(declared=True, exact_part=None)
    # The regulator's supply pin already carries its rail, and the draft routes a second net to
    # the same pin: one pin, two nets, which stays a refusal (plan §4.1, second corollary).
    intent["signals"] = [
        *intent["signals"],
        {"name": "SENSE_A", "from": "input.pin3", "to": "reg.input"},
    ]
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(intent)
    assert {row.code for row in excinfo.value.diagnostics} & {"conflicting_port_binding"}
