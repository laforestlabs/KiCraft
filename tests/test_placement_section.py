"""The durable `placement` state section: model, commit, and merge.

- PlacementSection validates anchor-spec SHAPE (catches UI bugs) but
  deliberately not ref existence (parts churn across BOM re-runs).
- `stage-commit placement` is deterministic: it commits without
  touching any LLM stage, invalidates nothing downstream, and surfaces
  stale refs as warnings.
- write_autoplacer_json merges with precedence library fragments <
  BOM (LLM) < placement (user), drops stale placement refs (the §9.6
  named-refs check would otherwise fail synthesis), and emits fixed
  board dimensions.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from kicraft.design.models import (
    BOM,
    Architecture,
    BomPart,
    ConversationState,
    FormFactor,
    PlacementSection,
    Sheet,
)
from kicraft.design.synthesis.autoplacer import write_autoplacer_json
from kicraft.server.session import downstream_stages


# ---- model -------------------------------------------------------------------


def test_placement_section_validates_spec_shape():
    PlacementSection(component_zones={"J1": {"edge": "left", "rotation": 90.0}})
    with pytest.raises(ValidationError, match="unknown keys"):
        PlacementSection(component_zones={"J1": {"side": "left"}})
    with pytest.raises(ValidationError, match="at most one anchor"):
        PlacementSection(component_zones={"J1": {"edge": "left", "corner": "top-left"}})
    with pytest.raises(ValidationError, match="not in"):
        PlacementSection(component_zones={"J1": {"edge": "diagonal"}})
    with pytest.raises(ValidationError, match="0..360"):
        PlacementSection(component_zones={"J1": {"rotation": 720.0}})
    with pytest.raises(ValidationError, match=">= 10"):
        PlacementSection(board={"width_mm": 2.0, "height_mm": 50.0})


def test_placement_refs_not_validated_against_parts():
    """Stale refs must survive the model (they degrade at synthesis)."""
    PlacementSection(component_zones={"GHOST9": {"corner": "top-left"}},
                     thermal_refs=["GHOST9"])


def test_placement_is_not_a_design_stage():
    assert downstream_stages("placement") == []


# ---- stage-commit --------------------------------------------------------------


def _bom() -> BOM:
    return BOM(parts=[
        BomPart(ref="U1", value="ESP32", sheet="MCU",
                symbol="Device:R", footprint="Resistor_SMD:R_0402_1005Metric"),
        BomPart(ref="J1", value="USB-C", sheet="MCU",
                symbol="Device:R", footprint="Resistor_SMD:R_0402_1005Metric"),
    ])


def test_stage_commit_placement_round_trips_and_warns_on_stale_refs(tmp_path, capsys):
    import kicraft.design.cli_app as cli_app

    state = ConversationState(project_stem="WIDGET", bom=_bom())
    state_path = tmp_path / "state.json"
    state_path.write_text(state.model_dump_json(), encoding="utf-8")

    slot = {
        "component_zones": {"J1": {"edge": "left"}, "GHOST9": {"corner": "top-left"}},
        "thermal_refs": ["U1"],
        "backside_through_hole_leaves": ["BATT"],
        "board": {"width_mm": 60.0, "height_mm": 60.0, "size_search": False},
    }
    slot_file = tmp_path / "slot.json"
    slot_file.write_text(json.dumps(slot), encoding="utf-8")

    rc = cli_app.main(["stage-commit", "placement", str(state_path),
                       "--slot-file", str(slot_file), "--no-archive"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert any("GHOST9" in w for w in out.get("warnings", [])), (
        "stale placement refs must surface as commit warnings")

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["placement"]["component_zones"]["J1"] == {"edge": "left"}
    assert persisted["placement"]["board"]["size_search"] is False
    # Nothing upstream was touched.
    assert persisted["bom"]["parts"][0]["ref"] == "U1"


def test_stage_commit_placement_rejects_bad_spec(tmp_path, capsys):
    import kicraft.design.cli_app as cli_app

    state_path = tmp_path / "state.json"
    state_path.write_text(
        ConversationState(project_stem="W").model_dump_json(), encoding="utf-8")
    slot_file = tmp_path / "slot.json"
    slot_file.write_text(
        json.dumps({"component_zones": {"J1": {"edge": "diagonal"}}}),
        encoding="utf-8")
    rc = cli_app.main(["stage-commit", "placement", str(state_path),
                       "--slot-file", str(slot_file), "--no-archive"])
    assert rc == 3
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False


# ---- autoplacer merge -----------------------------------------------------------


def _arch() -> Architecture:
    return Architecture(
        sheets=[Sheet(name="MCU", stem="MCU", function="microcontroller")],
        power_nets=["+3V3", "GND"],
        inter_sheet_nets=[],
    )


def test_write_autoplacer_merge_precedence_and_stale_drop(tmp_path, capsys):
    bom = _bom()
    # LLM zone for J1 says bottom; the user's placement says left -> user wins.
    bom.component_zones["J1"] = {"edge": "bottom"}
    placement = PlacementSection(
        component_zones={"J1": {"edge": "left"},
                         "GHOST9": {"corner": "top-left"}},
        thermal_refs=["U1"],
        backside_through_hole_leaves=["BATT"],
        board={"width_mm": 70.0, "height_mm": 50.0, "size_search": False},
    )
    out = write_autoplacer_json(
        tmp_path, "WIDGET", _arch(), bom,
        library_fragments={"component_zones": {"J1": {"edge": "top"}}},
        placement=placement,
    )
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert cfg["component_zones"]["J1"] == {"edge": "left"}, "user wins"
    assert "GHOST9" not in cfg["component_zones"], "stale ref dropped"
    assert "U1" in cfg["thermal_refs"]
    assert cfg["parent_placement"]["backside_through_hole_leaves"] == ["BATT"]
    assert cfg["board_width_mm"] == 70.0
    assert cfg["board_height_mm"] == 50.0
    assert cfg["enable_board_size_search"] is False
    assert "GHOST9" in capsys.readouterr().out, "drop is warned"


def test_write_autoplacer_without_placement_keeps_size_search():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        out = write_autoplacer_json(Path(d), "WIDGET", _arch(), _bom())
        cfg = json.loads(out.read_text(encoding="utf-8"))
        assert cfg["enable_board_size_search"] is True
        assert "board_width_mm" not in cfg


# ---- form-factor outline (Phase 2: flow into autoplacer.json) -------------------


def test_write_autoplacer_emits_board_outline_for_shape(tmp_path):
    out = write_autoplacer_json(
        tmp_path, "WIDGET", _arch(), _bom(),
        form_factor=FormFactor(shape="circle", size_mm=50.0),
    )
    cfg = json.loads(out.read_text(encoding="utf-8"))
    assert cfg["board_outline"] == {"shape": "circle", "size_mm": 50.0}


def test_write_autoplacer_emits_shape_params(tmp_path):
    out = write_autoplacer_json(
        tmp_path, "WIDGET", _arch(), _bom(),
        form_factor=FormFactor(shape="rounded_rect", corner_radius_mm=3.0),
    )
    cfg = json.loads(out.read_text(encoding="utf-8"))["board_outline"]
    assert cfg["shape"] == "rounded_rect"
    assert cfg["corner_radius_mm"] == 3.0
    assert "size_mm" not in cfg  # not stated -> not emitted


def test_write_autoplacer_omits_board_outline_for_rect(tmp_path):
    # No form factor, or an explicit rectangle, leaves the default path untouched.
    out_none = write_autoplacer_json(tmp_path, "A", _arch(), _bom())
    assert "board_outline" not in json.loads(out_none.read_text())
    out_rect = write_autoplacer_json(
        tmp_path, "B", _arch(), _bom(), form_factor=FormFactor(shape="rect"),
    )
    assert "board_outline" not in json.loads(out_rect.read_text())


def test_write_autoplacer_emits_standard_form_factor_block(tmp_path):
    # A named standard form factor surfaces the template geometry so compose can
    # honor it. Informational (carries `validated`); does not touch board_outline.
    out = write_autoplacer_json(
        tmp_path, "SHIELD", _arch(), _bom(),
        form_factor=FormFactor(shape="rect", standard="arduino_uno_shield",
                               size_mm=68.58),
    )
    cfg = json.loads(out.read_text())
    ffs = cfg.get("form_factor_standard")
    assert ffs is not None
    assert ffs["key"] == "arduino_uno_shield"
    assert ffs["validated"] is False  # dormant until the datum is DXF-verified
    assert ffs["board_width_mm"] == 68.58 and ffs["board_height_mm"] == 53.34
    assert {c["role"] for c in ffs["fixed_connectors"]} == {
        "digital_high", "digital_low", "power", "analog"}
    assert len(ffs["mounting_holes"]) == 4


def test_write_autoplacer_no_standard_block_without_a_standard(tmp_path):
    out = write_autoplacer_json(
        tmp_path, "PLAIN", _arch(), _bom(),
        form_factor=FormFactor(shape="circle", size_mm=50.0),
    )
    assert "form_factor_standard" not in json.loads(out.read_text())


def test_standard_block_survives_project_config_load(tmp_path):
    from kicraft.autoplacer.config import load_project_config

    out = write_autoplacer_json(
        tmp_path, "SHIELD", _arch(), _bom(),
        form_factor=FormFactor(shape="rect", standard="arduino_uno_shield"),
    )
    cfg = load_project_config(str(out))
    assert isinstance(cfg.get("form_factor_standard"), dict)
    assert cfg["form_factor_standard"]["key"] == "arduino_uno_shield"


def test_board_outline_survives_project_config_load(tmp_path):
    # Contract the compose pipeline depends on: the emitted board_outline block
    # is NOT whitelisted away by the project-config loader -- it reaches `cfg`
    # so compose can populate ParentCompositionState.requested_shape from it.
    from kicraft.autoplacer.config import load_project_config

    out = write_autoplacer_json(
        tmp_path, "WIDGET", _arch(), _bom(),
        form_factor=FormFactor(shape="hexagon", size_mm=40.0),
    )
    cfg = load_project_config(str(out))
    assert isinstance(cfg.get("board_outline"), dict)
    assert cfg["board_outline"]["shape"] == "hexagon"
