"""Deterministic lint for LLM-authored silkscreen labels.

The lint is the trust boundary between the LLM's content and the physical
board: anchors must exist in the BOM and every electrical claim must be
corroborated by the design state. These tests pin that contract.
"""

from kicraft.design.models import (
    Architecture,
    BOM,
    BomPart,
    ConversationState,
    IntentSlot,
    NetConnection,
    PinEndpoint,
    Sheet,
)
from kicraft.design.synthesis.silk_plan import (
    build_corroboration_corpus,
    lint_labels,
    normalize_ascii,
    uncovered_connectors,
)


def _state() -> ConversationState:
    return ConversationState(
        project_stem="PD_TEST",
        intent=IntentSlot(goal="USB-C PD trigger, selectable 9 V / 12 V / 20 V"),
        architecture=Architecture(
            rail_voltages={"VOUT": 12.0},
            sheets=[Sheet(name="PD", stem="PD", function="pd trigger")],
            power_nets=["VBUS", "VOUT"],
            inter_sheet_nets=[],
        ),
        bom=BOM(parts=[
            BomPart(ref="U1", value="CH224K", symbol="ch224k:CH224K",
                    footprint="ch224k:ESSOP-10", sheet="PD"),
            BomPart(ref="SW1", value="DIP-3",
                    symbol="dip-switch-3pos:DSHP03TSGER",
                    footprint="dip-switch-3pos:SW-SMD", sheet="PD"),
            BomPart(ref="J2", value="OUT", symbol="Connector:Conn_01x02",
                    footprint="lib:fp", sheet="PD"),
            BomPart(ref="J1", value="IN", symbol="Connector_Generic:Conn_01x03",
                    footprint="Connector_Generic:Conn_01x03", sheet="PD"),
        ]),
    )


def _lint(labels):
    return lint_labels(labels, _state(), project_root=None)


def test_corpus_includes_rails_and_intent_text():
    corpus = build_corroboration_corpus(_state())
    assert ("12", "V") in corpus  # rail_voltages float
    assert ("9", "V") in corpus  # intent text "9 V"
    assert ("20", "V") in corpus


def test_unknown_anchor_ref_dropped():
    kept, dropped = _lint([
        {"id": "x", "text": "OUT 12V", "anchor": {"ref": "Z99"}},
    ])
    assert kept == []
    assert any("Z99" in d for d in dropped)


def test_uncorroborated_claim_dropped_corroborated_kept():
    kept, dropped = _lint([
        {"id": "amps", "text": "OUT 5A MAX", "anchor": {"ref": "J2"}},
        {"id": "volts", "text": "OUT 9/12/20V", "anchor": {"ref": "J2"}},
    ])
    assert [lb.id for lb in kept] == ["volts"]
    assert any("5A" in d for d in dropped)


def test_slash_voltage_list_shares_unit():
    kept, dropped = _lint([
        {"id": "v", "text": "9/12/20V OUT", "anchor": {"ref": "J2"}},
    ])
    assert [lb.id for lb in kept] == ["v"]


def test_table_position_guard_uses_switch_pin_count():
    # dip-switch-3pos is a vendored bundle: 6 pins -> 3 positions. A table
    # header naming position 4 is impossible on this switch.
    kept, dropped = _lint([
        {"id": "tbl", "kind": "table", "anchor": {"ref": "SW1"},
         "text": "VOUT 1 2 3 4\n9V ON - - -"},
    ])
    assert kept == []
    assert any("position 4" in d for d in dropped)

    kept, dropped = _lint([
        {"id": "tbl", "kind": "table", "anchor": {"ref": "SW1"},
         "text": "VOUT 1 2 3\n9V ON - -\n12V - ON -\n20V - - ON"},
    ])
    assert [lb.id for lb in kept] == ["tbl"]


def test_ascii_normalization_and_caps():
    assert normalize_ascii("12 µF · ±5%") == "12 uF - +/-5%"
    long_line = "X" * 60
    kept, _ = _lint([
        {"id": "n", "text": "\n".join([long_line] * 9), "anchor": {"ref": "J2"}},
    ])
    assert len(kept) == 1
    lines = kept[0].text.split("\n")
    assert len(lines) <= 5
    assert all(len(ln) <= 30 for ln in lines)


def test_empty_and_duplicate_labels_dropped():
    kept, dropped = _lint([
        {"id": "a", "text": "   ", "anchor": {"ref": "J2"}},
        {"id": "b", "text": "NOTE", "anchor": {"ref": "J2"}},
        {"id": "b", "text": "NOTE 2", "anchor": {"ref": "J2"}},
    ])
    assert [lb.id for lb in kept] == ["b"]
    assert any("empty" in d for d in dropped)
    assert any("duplicate" in d for d in dropped)


def test_label_cap_keeps_highest_priority():
    # 12 labels: 5 priority-3, 7 priority-1. The cap (10) drops only the two
    # lowest-priority labels; every must-have priority-1 label survives.
    labels = [
        {"id": f"l{i}", "text": "NOTE", "anchor": {"ref": "J2"},
         "priority": 3 if i < 5 else 1}
        for i in range(12)
    ]
    kept, dropped = _lint(labels)
    assert len(kept) == 10
    assert all(any(lb.id == f"l{i}" for lb in kept) for i in range(5, 12))
    assert len([d for d in dropped if "cap" in d]) == 2


def test_pinout_valid_pins_kept():
    kept, dropped = _lint([
        {"id": "j1-pins", "kind": "pinout", "anchor": {"ref": "J1"},
         "pins": [{"pin": "1", "text": "VIN"}, {"pin": "2", "text": "GND"}]},
    ])
    assert dropped == []
    assert [lb.id for lb in kept] == ["j1-pins"]
    assert kept[0].kind == "pinout"
    assert [(p.pin, p.text) for p in kept[0].pins] == [("1", "VIN"), ("2", "GND")]


def test_pinout_unknown_pin_dropped():
    kept, dropped = _lint([
        {"id": "j1-pins", "kind": "pinout", "anchor": {"ref": "J1"},
         "pins": [{"pin": "1", "text": "VIN"}, {"pin": "4", "text": "GND"}]},
    ])
    assert [lb.id for lb in kept] == ["j1-pins"]
    assert [p.pin for p in kept[0].pins] == ["1"]
    assert any("j1-pins:4" in d and "not in symbol" in d for d in dropped)


def test_pinout_uncorroborated_claim_dropped():
    # 48V is not in the _state() corpus (9/12/20 V only); the entry drops.
    kept, dropped = _lint([
        {"id": "p", "kind": "pinout", "anchor": {"ref": "J1"},
         "pins": [{"pin": "1", "text": "48V"}, {"pin": "2", "text": "GND"}]},
    ])
    assert [lb.id for lb in kept] == ["p"]
    assert [e.pin for e in kept[0].pins] == ["2"]
    assert any("p:1" in d and "48V" in d for d in dropped)


def test_pinout_empty_pins_dropped():
    kept, dropped = _lint([
        {"id": "p", "kind": "pinout", "anchor": {"ref": "J1"}, "pins": []},
    ])
    assert kept == []
    assert any("no pins" in d for d in dropped)


def test_pinout_text_truncated_to_max_pin_chars():
    kept, dropped = _lint([
        {"id": "p", "kind": "pinout", "anchor": {"ref": "J1"},
         "pins": [{"pin": "1", "text": "CONNECTOR"}]},
    ])
    assert [lb.id for lb in kept] == ["p"]
    assert kept[0].pins[0].text == "CONNECTO"  # 9 -> 8 chars


def test_uncovered_connectors_reports_unlabeled_wired_connector():
    state = _state()
    state.bom.connections = [
        NetConnection(net_name="VIN", sheet="PD",
                      endpoints=[PinEndpoint(ref="J1", pin="1")]),
    ]
    # No kept label anchors to J1 -> it is reported. J2 is also a connector
    # but is not wired, so it is not reported.
    assert uncovered_connectors([], state) == ["J1"]
