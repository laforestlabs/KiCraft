"""End-to-end synthesis tests.

Builds a frozen ConversationState mirroring (a slimmed-down version of) LLUPS,
synthesizes the project, and verifies:
- §9.1-§9.6 mechanical checks all pass;
- KiCraft's hierarchy parser can parse the result and finds every BOM ref in
  exactly one leaf;
- power/ground net naming is recognized as power.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kicraft.upstream.models import (
    BOM,
    Architecture,
    BomPart,
    ConversationState,
    FunctionalBlock,
    FunctionalSpec,
    IntentSlot,
    InterSheetNet,
    Sheet,
    SheetPin,
)
from kicraft.upstream.stages.synthesis import SynthesisInputError, run
from kicraft.upstream.synthesis.validation import (
    SynthesisValidationError,
    check_named_refs_exist,
    check_pin_directions,
    check_sheetfile_refs_resolve,
    run_validations,
)


@pytest.fixture
def llups_like_state() -> ConversationState:
    """A two-sheet project: USB INPUT + LDO 3V3, with proper inter-sheet net."""
    intent = IntentSlot(goal="USB-powered 3.3V regulator demo", inferred_expertise="expert")
    spec = FunctionalSpec(
        blocks=[
            FunctionalBlock(name="USB_INPUT", category="interface", purpose="USB-C VBUS in"),
            FunctionalBlock(name="LDO_3V3", category="power", purpose="3.3V LDO regulation"),
        ]
    )
    architecture = Architecture(
        sheets=[
            Sheet(name="USB INPUT", stem="USB_INPUT", function="USB-C VBUS input + ESD"),
            Sheet(name="LDO 3V3", stem="LDO_3V3", function="3.3V LDO from VBUS"),
        ],
        power_nets=["VBUS", "+3V3", "GND"],
        inter_sheet_nets=[
            InterSheetNet(
                name="VBUS",
                endpoints=[
                    SheetPin(sheet="USB INPUT", direction="output"),
                    SheetPin(sheet="LDO 3V3", direction="input"),
                ],
            ),
            InterSheetNet(
                name="+3V3",
                endpoints=[
                    SheetPin(sheet="LDO 3V3", direction="output"),
                    SheetPin(sheet="USB INPUT", direction="input"),
                ],
            ),
            InterSheetNet(
                name="GND",
                endpoints=[
                    SheetPin(sheet="USB INPUT", direction="passive"),
                    SheetPin(sheet="LDO 3V3", direction="passive"),
                ],
            ),
        ],
    )
    bom = BOM(
        parts=[
            BomPart(
                ref="J1",
                value="USB-C 16-pin",
                symbol="Connector:USB_C_Receptacle_USB2.0_16P",
                footprint="Connector_USB:USB_C_Receptacle_GCT_USB4105-xx-A_16P_TopMnt_Horizontal",
                sheet="USB INPUT",
            ),
            BomPart(
                ref="C1",
                value="10uF",
                symbol="Device:C",
                footprint="Capacitor_SMD:C_0603_1608Metric",
                sheet="USB INPUT",
            ),
            BomPart(
                ref="U1",
                value="AP2112K-3.3",
                symbol="Regulator_Linear:AP2112K-3.3",
                footprint="Package_TO_SOT_SMD:SOT-23-5",
                sheet="LDO 3V3",
            ),
            BomPart(
                ref="C2",
                value="1uF",
                symbol="Device:C",
                footprint="Capacitor_SMD:C_0402_1005Metric",
                sheet="LDO 3V3",
            ),
            BomPart(
                ref="C3",
                value="1uF",
                symbol="Device:C",
                footprint="Capacitor_SMD:C_0402_1005Metric",
                sheet="LDO 3V3",
            ),
        ],
        ic_groups={"U1": ["C2", "C3"]},
        group_labels={"U1": "LDO 3V3"},
        thermal_refs=["U1"],
        signal_flow_order=["J1", "U1"],
        component_zones={"J1": {"edge": "left"}},
    )
    return ConversationState(
        project_stem="DEMO33",
        intent=intent,
        functional_spec=spec,
        architecture=architecture,
        bom=bom,
    )


def test_synthesis_writes_expected_files(tmp_path, llups_like_state) -> None:
    artifacts, results = run(llups_like_state, tmp_path)
    assert artifacts.root_sch.name == "DEMO33.kicad_sch"
    assert artifacts.root_sch.is_file()
    leaf_stems = {p.stem for p in artifacts.leaf_schs}
    assert leaf_stems == {"USB_INPUT", "LDO_3V3"}
    assert artifacts.kicad_pro.is_file()
    assert artifacts.autoplacer_json.is_file()
    # Every §9.1-§9.6 check passed.
    assert all(r.ok for r in results), [r.message for r in results if not r.ok]


def test_synthesis_passes_all_validations(tmp_path, llups_like_state) -> None:
    run(llups_like_state, tmp_path)
    # Re-run the aggregator on disk; it raises on failure.
    run_validations(tmp_path, "DEMO33")


def test_synthesis_refs_match_autoplacer_named_refs(tmp_path, llups_like_state) -> None:
    run(llups_like_state, tmp_path)
    r = check_named_refs_exist(tmp_path, "DEMO33")
    assert r.ok, r.offenders


def test_synthesis_power_nets_in_autoplacer(tmp_path, llups_like_state) -> None:
    run(llups_like_state, tmp_path)
    cfg = json.loads((tmp_path / "DEMO33_autoplacer.json").read_text())
    assert set(cfg["power_nets"]) == {"VBUS", "+3V3", "GND"}


def test_synthesis_pin_directions_valid(tmp_path, llups_like_state) -> None:
    run(llups_like_state, tmp_path)
    r = check_pin_directions(tmp_path)
    assert r.ok, r.offenders


def test_synthesis_sheetfile_refs_resolve(tmp_path, llups_like_state) -> None:
    run(llups_like_state, tmp_path)
    r = check_sheetfile_refs_resolve(tmp_path)
    assert r.ok, r.offenders


def test_synthesis_kicad_pro_has_default_and_power_netclasses(
    tmp_path, llups_like_state
) -> None:
    run(llups_like_state, tmp_path)
    pro = json.loads((tmp_path / "DEMO33.kicad_pro").read_text())
    names = {c["name"] for c in pro["net_settings"]["classes"]}
    assert names == {"Default", "Power"}


def test_synthesis_input_missing_state_raises(tmp_path) -> None:
    with pytest.raises(SynthesisInputError):
        run(ConversationState(), tmp_path)


def test_synthesis_bom_references_unknown_sheet_raises(tmp_path, llups_like_state) -> None:
    llups_like_state.bom.parts[0] = llups_like_state.bom.parts[0].model_copy(
        update={"sheet": "GHOST"}
    )
    with pytest.raises(SynthesisInputError):
        run(llups_like_state, tmp_path)


# ---------- KiCraft's hierarchy_parser can parse the result ----------


def test_synthesis_output_parses_with_hierarchy_parser(tmp_path, llups_like_state) -> None:
    """KiCraft must be able to extract subcircuits from our synthesis output.

    This is the strongest assertion in the file: it exercises the same code path
    that solve-subcircuits would, minus the PCB routing step.
    """
    from kicraft.autoplacer.brain.hierarchy_parser import parse_hierarchy

    run(llups_like_state, tmp_path)
    graph = parse_hierarchy(str(tmp_path), str(tmp_path / "DEMO33.kicad_sch"))
    leaf_names = {n.definition.id.sheet_name for n in graph.leaf_nodes()}
    assert leaf_names == {"USB INPUT", "LDO 3V3"}, leaf_names


def test_synthesis_every_bom_ref_lives_in_correct_leaf(tmp_path, llups_like_state) -> None:
    from kicraft.autoplacer.brain.hierarchy_parser import parse_hierarchy

    run(llups_like_state, tmp_path)
    graph = parse_hierarchy(str(tmp_path), str(tmp_path / "DEMO33.kicad_sch"))
    ref_to_sheet: dict[str, str] = {}
    for node in graph.leaf_nodes():
        for ref in node.definition.component_refs:
            ref_to_sheet[ref] = node.definition.id.sheet_name
    # Compare against the BOM.
    expected = {p.ref: p.sheet for p in llups_like_state.bom.parts}
    assert ref_to_sheet == expected, f"got={ref_to_sheet}, want={expected}"


def test_validation_failure_raises(tmp_path, llups_like_state) -> None:
    """If we deliberately corrupt the output, validation raises."""
    run(llups_like_state, tmp_path)
    # Corrupt the autoplacer json so §9.5 fails.
    (tmp_path / "DEMO33_autoplacer.json").write_text("{not json")
    with pytest.raises(SynthesisValidationError):
        run_validations(tmp_path, "DEMO33")
