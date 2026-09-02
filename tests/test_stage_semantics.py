import json
from pathlib import Path

from kicraft.design.stage_semantics import diagnose_stage

FIXTURES = Path(__file__).parent / "fixtures" / "stage_reliability"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


def _codes(stage, candidate, upstream=None):
    brief = _load("rp2040_brief.json")["brief"]
    return {
        d.code
        for d in diagnose_stage(
            stage, brief=brief, upstream_state=upstream or {}, candidate=candidate
        )
    }


def test_live_intent_candidate_reports_classification_defects():
    codes = _codes("intent", _load("rp2040_intent_candidate.json"))
    assert {
        "intent_named_part_omitted",
        "intent_constraints_empty",
        "intent_unclassified_copy",
    } <= codes


def test_vague_intent_may_leave_classification_empty():
    diagnostics = diagnose_stage(
        "intent",
        brief="A small sensor board",
        upstream_state={},
        candidate={"goal": "A small sensor board", "constraints": [], "named_parts": []},
    )
    assert diagnostics == []


def test_live_functional_spec_reports_premature_topology():
    candidate = _load("rp2040_functional_spec_candidate.json")
    codes = _codes("functional_spec", candidate, {"intent": _load("rp2040_intent_candidate.json")})
    assert "functional_spec_premature_topology" in codes


def test_live_architecture_reports_graph_and_domain_defects():
    upstream = {"functional_spec": _load("rp2040_functional_spec_candidate.json")}
    codes = _codes("architecture", _load("rp2040_architecture_candidate.json"), upstream)
    assert "architecture_power_block_as_sheet" in codes
    assert "architecture_fragmented_physical_domain" in codes
    assert "architecture_wrong_signal_direction" in codes


def test_live_bom_and_wiring_report_fabrication_gates():
    bom = _load("rp2040_bom_candidate.json")
    assert "bom_castellation_placeholder" in _codes("bom", bom)
    wiring = _load("rp2040_wiring_candidate.json")
    codes = _codes("wiring", wiring, {"bom": bom})
    assert "wiring_bootsel_unreachable" in codes
    assert "wiring_special_pin_no_connect" in codes
