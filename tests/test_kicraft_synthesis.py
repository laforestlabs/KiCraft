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
import re
import sys
from pathlib import Path

import pytest

from kicraft.design.models import (
    BOM,
    Architecture,
    ArraySpec,
    BomPart,
    ConversationState,
    FunctionalBlock,
    FunctionalSpec,
    IntentSlot,
    InterSheetNet,
    Sheet,
    SheetPin,
)
from kicraft.design.synthesize import SynthesisInputError, run
from kicraft.design.synthesis.validation import (
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


def _led_array_arch_bom() -> tuple[Architecture, BOM]:
    arch = Architecture(
        sheets=[Sheet(name="LED MATRIX", stem="LED_MATRIX", function="LED array")],
        power_nets=["VBUS", "GND"],
        inter_sheet_nets=[],
    )
    parts = [
        BomPart(ref=f"D{i}", value="LED", symbol="L:LED", footprint="L:LED",
                sheet="LED MATRIX")
        for i in range(1, 5)
    ]
    return arch, BOM(parts=parts)


def test_autoplacer_json_includes_arrays(tmp_path) -> None:
    from kicraft.design.synthesis.autoplacer import write_autoplacer_json

    arch, bom = _led_array_arch_bom()
    bom.arrays = [ArraySpec(refs=["D1", "D2", "D3", "D4"], rows=2, cols=2, pitch_mm=3.0)]
    out = write_autoplacer_json(tmp_path, "DEMO", arch, bom)
    data = json.loads(out.read_text())
    assert data["arrays"] == [
        {"refs": ["D1", "D2", "D3", "D4"], "pattern": "grid",
         "pitch_mm": 3.0, "serpentine": True, "rows": 2, "cols": 2}
    ]


def test_autoplacer_json_emits_ring_array(tmp_path) -> None:
    from kicraft.design.synthesis.autoplacer import write_autoplacer_json

    arch, bom = _led_array_arch_bom()
    bom.arrays = [ArraySpec(refs=["D1", "D2", "D3", "D4"], pattern="ring",
                            radius_mm=24.0)]
    out = write_autoplacer_json(tmp_path, "DEMO", arch, bom)
    data = json.loads(out.read_text())
    # No rows/cols keys: consumers key ring handling off "pattern" and
    # int(spec.get("rows", 0)) must never see None.
    assert data["arrays"] == [
        {"refs": ["D1", "D2", "D3", "D4"], "pattern": "ring",
         "pitch_mm": None, "serpentine": True,
         "radius_mm": 24.0, "start_angle_deg": 0.0}
    ]


def test_autoplacer_json_omits_arrays_when_empty(tmp_path) -> None:
    from kicraft.design.synthesis.autoplacer import write_autoplacer_json

    arch, bom = _led_array_arch_bom()
    out = write_autoplacer_json(tmp_path, "DEMO", arch, bom)
    assert "arrays" not in json.loads(out.read_text())


def test_synthesis_refs_match_autoplacer_named_refs(tmp_path, llups_like_state) -> None:
    run(llups_like_state, tmp_path)
    r = check_named_refs_exist(tmp_path, "DEMO33")
    assert r.ok, r.offenders


def test_synthesis_power_nets_in_autoplacer(tmp_path, llups_like_state) -> None:
    run(llups_like_state, tmp_path)
    cfg = json.loads((tmp_path / "DEMO33_autoplacer.json").read_text())
    assert set(cfg["power_nets"]) == {"VBUS", "+3V3", "GND"}

def test_synthesis_omits_router_selector_from_autoplacer_config(
    tmp_path, llups_like_state
) -> None:
    run(llups_like_state, tmp_path)
    cfg = json.loads((tmp_path / "DEMO33_autoplacer.json").read_text())
    assert "routing_backend" not in cfg


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


# ---------- root/leaf schematic filename collision (the 'cyan blob') ----------


def _stem_collision_state() -> ConversationState:
    """A single-sheet design whose only sheet stem equals the project stem.

    This is the shape that produced the cyan blob on kicraft.io: the root is
    emitted to ``<project_stem>.kicad_sch`` and the leaf to ``<sheet.stem>``
    ``.kicad_sch`` -- the SAME path when they are equal. One write clobbers the
    other, so instead of a readable component schematic the user is left with a
    lone non-readable block-diagram root (which KiCanvas auto-fits and paints as
    a solid fill).
    """
    intent = IntentSlot(goal="Garden soil-moisture sensor", inferred_expertise="beginner")
    spec = FunctionalSpec(
        blocks=[FunctionalBlock(name="SENSOR", category="sense", purpose="soil moisture")]
    )
    architecture = Architecture(
        sheets=[Sheet(name="GARDEN SENSOR", stem="GARDEN_SENSOR", function="all")],
        power_nets=["VCC", "GND"],
        inter_sheet_nets=[],
    )
    bom = BOM(
        parts=[
            BomPart(ref="U1", value="NE555P", symbol="Timer:NE555P",
                    footprint="Package_DIP:DIP-8_W7.62mm", sheet="GARDEN SENSOR"),
            BomPart(ref="R1", value="10k", symbol="Device:R",
                    footprint="Resistor_SMD:R_0603_1608Metric", sheet="GARDEN SENSOR"),
            BomPart(ref="C1", value="10nF", symbol="Device:C",
                    footprint="Capacitor_SMD:C_0603_1608Metric", sheet="GARDEN SENSOR"),
        ]
    )
    return ConversationState(
        project_stem="GARDEN_SENSOR",
        intent=intent,
        functional_spec=spec,
        architecture=architecture,
        bom=bom,
    )


def test_single_sheet_stem_collision_keeps_readable_leaf(tmp_path) -> None:
    """Regression: a sheet whose stem equals the project stem must not collapse
    the project to a single block-diagram root. The root (``<stem>.kicad_sch``)
    and the leaf must land in DISTINCT files, so the user sees a human-readable
    component schematic and not the cyan-blob block diagram."""
    run(_stem_collision_state(), tmp_path)

    schs = sorted(p.name for p in tmp_path.glob("*.kicad_sch"))
    root = "GARDEN_SENSOR.kicad_sch"
    # Root + exactly one leaf, as separate files (no filename collision).
    assert len(schs) == 2, f"root/leaf filename collision; on disk: {schs}"
    assert root in schs
    leaf_names = [n for n in schs if n != root]
    assert len(leaf_names) == 1, schs

    # The root stays a hierarchical block diagram that references its leaf.
    root_txt = (tmp_path / root).read_text(encoding="utf-8")
    assert "Sheetfile" in root_txt, "root schematic is not a hierarchy (was clobbered by the leaf)"

    # The leaf is the human-readable component schematic.
    leaf_txt = (tmp_path / leaf_names[0]).read_text(encoding="utf-8")
    assert "(symbol" in leaf_txt and "Timer:NE555P" in leaf_txt, \
        "leaf schematic lacks component symbols (was clobbered by the root block diagram)"


def _at_rotations(text: str) -> list[float]:
    """Every explicit rotation from `(at x y ROT)` triples in a schematic."""
    return [float(m.group(1)) for m in
            re.finditer(r"\(at\s+-?[\d.]+\s+-?[\d.]+\s+(-?[\d.]+)\)", text)]


def test_synthesis_emits_only_cardinal_rotations(tmp_path, llups_like_state) -> None:
    """KiCanvas (the in-browser schematic viewer) THROWS on a non-cardinal symbol,
    pin, or label rotation (e.g. ``unexpected rotation 45``), aborts before painting,
    and shows its aqua ``<kicanvas-embed>`` background -- the reported "teal blob".
    Guard that synthesis only ever emits rotations that are multiples of 90 so the
    viewer can render every sheet."""
    run(llups_like_state, tmp_path)
    for sch in tmp_path.glob("*.kicad_sch"):
        bad = [r for r in _at_rotations(sch.read_text(encoding="utf-8")) if r % 90 != 0]
        assert not bad, f"{sch.name} has non-cardinal rotations {bad} (KiCanvas blanks to teal)"


def test_resynthesis_removes_stale_orphan_sheets(tmp_path, llups_like_state) -> None:
    """A resumable-session stage rerun re-synthesizes into the same project dir.
    Synthesis must clear the prior generated sheets first, or an orphan leaf from a
    previous architecture lingers -- showing up as a phantom sheet in the web sheet
    list and leaving the hierarchy degenerate (the parent no longer references it,
    which is what produced ``leafs=0/0`` at place/route)."""
    run(llups_like_state, tmp_path)
    assert (tmp_path / "USB_INPUT.kicad_sch").is_file()
    orphan = tmp_path / "OLD_MOTOR_DRIVER.kicad_sch"
    orphan.write_text("(kicad_sch (version 20250114))\n", encoding="utf-8")

    run(llups_like_state, tmp_path)  # the rerun
    assert not orphan.exists(), "stale orphan leaf survived re-synthesis"
    assert (tmp_path / "USB_INPUT.kicad_sch").is_file()  # real leaf re-emitted
    assert (tmp_path / "DEMO33.kicad_sch").is_file()  # root re-emitted


def _strict_sexpr_parses(text: str) -> bool:
    """Escape-aware balanced-paren check — the same parse contract eeschema
    enforces. An unescaped quote inside a string desynchronizes everything
    after it, which KiCad surfaces as a silently EMPTY hierarchy child."""
    depth, in_str, i, n = 0, False, 0, len(text)
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0 and not in_str


def test_quote_in_part_fields_survives_emission(tmp_path, llups_like_state) -> None:
    """A part description with an inch mark (`0.96" OLED`) used to terminate the
    s-expression string early and corrupt the whole leaf file; KiCad then loaded
    the child sheet as EMPTY (hier_label_mismatch on the parent, part missing
    from netlist AND board). Every model/part-derived string must be escaped."""
    j1 = llups_like_state.bom.parts[0]
    j1.sourcing_note = '4-pin header for SSD1306 0.96" OLED module (VCC, GND, SCL, SDA)'
    j1.datasheet = 'https://example.com/ssd1306-0.96".pdf'
    run(llups_like_state, tmp_path)
    for sch in tmp_path.glob("*.kicad_sch"):
        text = sch.read_text(encoding="utf-8")
        assert _strict_sexpr_parses(text), f"{sch.name} does not parse"
    leaf = (tmp_path / "USB_INPUT.kicad_sch").read_text(encoding="utf-8")
    assert '0.96\\" OLED' in leaf, "description quote was not escaped"


def test_write_guard_rejects_unescaped_quote(tmp_path) -> None:
    """Belt-and-suspenders: if a future writer bypasses escaping, the emitter
    must refuse to write the corrupt file instead of letting KiCad silently
    drop the sheet three stages later."""
    from kicraft.design.synthesis.emitter import assert_schematic_parses

    corrupt = '(kicad_sch (property "Description" "0.96" OLED (VCC)"))\n'
    with pytest.raises(ValueError, match="unparseable schematic"):
        assert_schematic_parses(corrupt, tmp_path / "X.kicad_sch")
    assert_schematic_parses('(kicad_sch (property "D" "0.96\\" OLED"))\n', tmp_path / "X.kicad_sch")
